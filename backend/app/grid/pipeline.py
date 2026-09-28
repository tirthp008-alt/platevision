"""Per-camera processing pipeline worker.

Camera → frames → vehicle detection → local ByteTrack → plate localization →
PlateVision OCR → multi-frame OCR fusion → visual embedding → sighting event.

Each :class:`CameraWorker` owns a single ingestion thread so one failing camera
cannot crash the others. Frames and evidence crops are kept only in the
in-memory :mod:`app.grid.frame_store`; raw video is never written to disk.
"""

import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.grid import repository as repo
from app.grid.embedder import cosine_similarity, dominant_color_name, vehicle_embedder
from app.grid.events import event_bus
from app.grid.frame_store import frame_store
from app.grid.local_tracker import VehicleTrack, VehicleTracker
from app.grid.plate_localizer import find_plate_regions
from app.grid.plate_ocr import (
    PlateReading,
    PlateVisionAdapter,
    fuse_plate_observations,
)
from app.grid.vehicle_detector import (
    VehicleDetection,
    build_vehicle_detector,
    classify_vehicle_type,
)
from app.schemas.detection import BoundingBox
from app.services.detector.factory import get_detector
from app.grid.plate_source import build_composite_plate_detector

CAMERA_STATUS = ("ONLINE", "OFFLINE", "CONNECTING", "PROCESSING", "ERROR")


class CameraWorker:
    def __init__(self, camera_row: dict):
        self.camera_id = camera_row["id"]
        self.name = camera_row["name"]
        self.source_type = camera_row.get("source_type", "device")
        self.source_uri = camera_row.get("source_uri", "")
        self.device_id = camera_row.get("device_id")
        self.is_demo = bool(camera_row.get("is_demo")) or self.source_type == "demo"

        self.status = "OFFLINE"
        self.error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.fps = 0.0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.sightings_created = 0
        self.plates_read = 0
        self.last_frame_at: Optional[float] = None
        self.session_id: Optional[str] = None

        self._current_tracks = 0
        self._recent_events: List[dict] = []

        # Models (per worker so a replacement model does not affect other cameras)
        self._plate_detector = build_composite_plate_detector()
        self._vehicle_detector = build_vehicle_detector(self._plate_detector)
        self._tracker = VehicleTracker()
        self._plate_vision = PlateVisionAdapter()
        self._plate_vision.set_detector(self._plate_detector)
        self._capture = None
        self._frame_index = 0
        self._demo_generator = None
        if self.is_demo:
            from app.grid.demo import DemoSceneGenerator

            self._demo_generator = DemoSceneGenerator(self.name, int(camera_row.get("demo_index", 0)))

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"drishti-cam-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
        with self._lock:
            self.status = "OFFLINE"

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    # ------------------------------------------------------------------
    def _open_capture(self) -> bool:
        if self.is_demo:
            return True
        try:
            if self.source_type == "device":
                index = int((self.device_id or "0").replace("device:", "")) if self.device_id else int(self.source_uri or 0)
                import platform

                backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
                cap = cv2.VideoCapture(index, backend)
            else:
                import os

                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
                cap = cv2.VideoCapture(self.source_uri)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self._capture = None
                return False
            self._capture = cap
            return True
        except Exception as e:
            self.error = f"Failed to open source: {e}"
            self._capture = None
            return False

    def _read_frame(self) -> Optional[np.ndarray]:
        if self.is_demo:
            return self._demo_generator.next_frame()
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    # ------------------------------------------------------------------
    def _run(self) -> None:
        from app.grid.db import session_scope

        with self._lock:
            self.status = "CONNECTING"
        logger.info(f"[{self.name}] starting camera processing ({self.source_type})")

        # Open a processing session record.
        try:
            with session_scope() as session:
                rec = repo.create_session_record(session, self.camera_id, is_demo=self.is_demo)
                self.session_id = rec.id
                repo.set_camera_status(session, self.camera_id, "CONNECTING")
        except Exception as e:
            logger.error(f"[{self.name}] could not create processing session: {e}")

        if not self._open_capture():
            self._mark_error(self.error or "Camera source unavailable.")
            return

        with self._lock:
            self.status = "PROCESSING"
        self._vehicle_detector.reset()
        self._tracker.reset()

        target_dt = 1.0 / max(0.5, settings.PROCESSING_TARGET_FPS)
        last_frame_time = 0.0
        misses = 0
        reconnect_attempts = 0
        fps_window_start = time.time()
        fps_count = 0

        try:
            while not self._stop.is_set():
                frame = self._read_frame()
                now = time.time()
                if frame is None:
                    misses += 1
                    self.frames_dropped += 1
                    if self.is_demo:
                        time.sleep(0.1)
                        continue
                    if misses >= 20:
                        reconnect_attempts += 1
                        with self._lock:
                            self.status = "CONNECTING"
                        if reconnect_attempts > settings.MAX_RECONNECT_ATTEMPTS:
                            self._mark_error("Stream lost; maximum reconnect attempts reached.")
                            return
                        time.sleep(settings.RECONNECT_BACKOFF_SECONDS)
                        self._open_capture()
                        misses = 0
                    else:
                        time.sleep(0.05)
                    continue

                misses = 0
                reconnect_attempts = 0
                self._frame_index += 1

                # Rate limit analysis while still refreshing the live snapshot.
                if (now - last_frame_time) < target_dt:
                    frame_store.put_frame(self.camera_id, frame)
                    continue
                last_frame_time = now

                try:
                    annotated = self._process_frame(frame, now)
                except Exception as e:  # never let one bad frame kill the camera
                    logger.error(f"[{self.name}] frame processing error: {e}")
                    annotated = frame

                frame_store.put_frame(self.camera_id, annotated)
                self.frames_processed += 1
                fps_count += 1
                self.last_frame_at = now
                if now - fps_window_start >= 1.0:
                    self.fps = fps_count / (now - fps_window_start)
                    fps_count = 0
                    fps_window_start = now

                self._persist_progress()
        finally:
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None
            self._close_session()
            with self._lock:
                if self.status != "ERROR":
                    self.status = "OFFLINE"
            logger.info(f"[{self.name}] camera processing stopped")

    def _mark_error(self, message: str) -> None:
        with self._lock:
            self.status = "ERROR"
            self.error = message
        logger.error(f"[{self.name}] {message}")
        try:
            from app.grid.db import session_scope

            with session_scope() as session:
                repo.set_camera_status(session, self.camera_id, "ERROR", error=message)
                if self.session_id:
                    repo.update_session_record(
                        session,
                        self.session_id,
                        stopped_at=time.time(),
                        error=message,
                        frames_processed=self.frames_processed,
                        sightings_created=self.sightings_created,
                        plates_read=self.plates_read,
                    )
        except Exception:
            pass

    def _close_session(self) -> None:
        if not self.session_id:
            return
        try:
            from app.grid.db import session_scope

            with session_scope() as session:
                repo.update_session_record(
                    session,
                    self.session_id,
                    stopped_at=time.time(),
                    frames_processed=self.frames_processed,
                    sightings_created=self.sightings_created,
                    plates_read=self.plates_read,
                    frames_dropped=self.frames_dropped,
                )
                repo.set_camera_status(session, self.camera_id, "OFFLINE")
        except Exception as e:
            logger.debug(f"[{self.name}] session close failed: {e}")

    def _persist_progress(self) -> None:
        if not self.session_id:
            return
        try:
            from app.grid.db import session_scope

            with session_scope() as session:
                repo.update_session_record(
                    session,
                    self.session_id,
                    frames_processed=self.frames_processed,
                    sightings_created=self.sightings_created,
                    plates_read=self.plates_read,
                    frames_dropped=self.frames_dropped,
                )
                repo.set_camera_status(session, self.camera_id, "PROCESSING")
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray, now: float) -> np.ndarray:
        h, w = frame.shape[:2]

        # 1. Plate localisation via the composite source (trained detector if
        #    present + contrast-based localizer).
        plate_boxes: List[Tuple[BoundingBox, float]] = self._plate_vision.detect_plates(frame)
        plate_boxes = _merge_boxes(plate_boxes)

        # 2. Vehicle detection (hybrid uses motion + plate anchoring).
        detections = self._vehicle_detector.detect(frame)

        # 3. Local multi-object tracking.
        tracks = self._tracker.update(detections, now)
        self._current_tracks = len([t for t in self._tracker.tracks.values() if t.time_since_update == 0])

        # 4. Per-track plate OCR + embedding, then sighting emission.
        for track in tracks:
            contained = [
                (b, c)
                for (b, c) in plate_boxes
                if _box_center_inside(b, track.bbox)
            ]
            self._update_track_evidence(frame, track, contained, now)
            self._maybe_emit_sighting(track, now)

        self._draw_overlay(frame, tracks, plate_boxes)
        return frame

    def _update_track_evidence(
        self,
        frame: np.ndarray,
        track: VehicleTrack,
        plate_boxes: List[Tuple[BoundingBox, float]],
        now: float,
    ) -> None:
        h, w = frame.shape[:2]

        # Plate OCR (throttled: every 3rd update or first update, capped).
        if plate_boxes and (track.hits <= 2 or track.hits % 3 == 0):
            for bbox, _conf in plate_boxes[:2]:
                pad = 0.08
                x = max(0, int(bbox.x - bbox.width * pad))
                y = max(0, int(bbox.y - bbox.height * pad))
                x2 = min(w, int(bbox.x + bbox.width * (1 + pad)))
                y2 = min(h, int(bbox.y + bbox.height * (1 + pad)))
                crop = frame[y:y2, x:x2]
                if crop.size == 0:
                    continue
                res = self._plate_vision.read_crop(crop)
                if not (res.raw_text or res.normalized_text):
                    continue
                reading = PlateReading(
                    plate_raw=res.raw_text,
                    plate_normalized=res.normalized_text,
                    ocr_confidence=float(res.confidence),
                    frame_timestamp=now,
                    camera_id=self.camera_id,
                    vehicle_track_id=track.track_id,
                    plate_bbox=[bbox.x, bbox.y, bbox.width, bbox.height],
                    vehicle_bbox=[track.bbox.x, track.bbox.y, track.bbox.width, track.bbox.height],
                    format_status=res.format_status,
                )
                track.plate_readings.append(reading)
                if len(track.plate_readings) > 30:
                    track.plate_readings.pop(0)
                self.plates_read += 1
                # Persist raw observation for audit and multi-frame analysis.
                self._store_plate_observation(reading)

        # Vehicle embedding + colour (throttled, refreshed as the vehicle nears).
        need_embedding = (
            not track.embedding
            or (track.hits % 6 == 0 and track.confidence >= 0.4)
        )
        if need_embedding:
            vx = max(0, track.bbox.x)
            vy = max(0, track.bbox.y)
            vx2 = min(w, track.bbox.x + track.bbox.width)
            vy2 = min(h, track.bbox.y + track.bbox.height)
            crop = frame[vy:vy2, vx:vx2]
            if crop.size > 0 and crop.shape[0] >= 12 and crop.shape[1] >= 12:
                emb = vehicle_embedder.embed(crop)
                if emb:
                    track.embedding = emb
                    track.vehicle_color = dominant_color_name(crop)
                    track.vehicle_type = classify_vehicle_type(
                        track.bbox, float(h * w)
                    ) if track.vehicle_type == "unknown" else track.vehicle_type
                    key = f"{self.camera_id}:{track.track_id}:{int(now)}"
                    frame_store.put_crop(key, crop)
                    track.crop_ref = key
                    track.best_crop = crop

        # Refresh plate hypothesis from accumulated multi-frame readings.
        if track.plate_readings:
            track.plate_hypothesis = fuse_plate_observations(track.plate_readings)

    def _store_plate_observation(self, reading: PlateReading) -> None:
        try:
            from app.grid.db import session_scope

            with session_scope() as session:
                repo.create_plate_observation(
                    session,
                    camera_id=self.camera_id,
                    vehicle_local_track_id=reading.vehicle_track_id,
                    timestamp=reading.frame_timestamp,
                    plate_raw=reading.plate_raw,
                    plate_normalized=reading.plate_normalized,
                    ocr_confidence=reading.ocr_confidence,
                    plate_bbox=reading.plate_bbox,
                    vehicle_bbox=reading.vehicle_bbox,
                )
        except Exception as e:
            logger.debug(f"[{self.name}] plate observation store failed: {e}")

    def _maybe_emit_sighting(self, track: VehicleTrack, now: float) -> None:
        if track.hits < 2 and not track.plate_hypothesis:
            return
        hyp = track.plate_hypothesis
        plate_norm = hyp.plate_normalized if hyp else ""
        plate_conf = hyp.confidence if hyp else 0.0
        plate_candidates = hyp.candidates if hyp else []
        plate_status = hyp.status if hyp else "unknown"

        # Update an existing sighting for this track instead of duplicating.
        if track.sighting_id:
            improved = plate_conf > track.sighting_plate_conf + 0.08
            if not improved:
                return
            try:
                from app.grid.db import session_scope

                with session_scope() as session:
                    s = repo.get_sighting(session, track.sighting_id)
                    if s is None:
                        return
                    s.plate_normalized = plate_norm
                    s.plate_raw = hyp.plate_raw if hyp else ""
                    s.plate_confidence = plate_conf
                    s.plate_candidates = plate_candidates
                    s.plate_format_status = plate_status
                    if track.embedding:
                        s.visual_embedding = track.embedding
                    track.sighting_plate_conf = plate_conf
                fusion_engine.mark_dirty()
            except Exception as e:
                logger.debug(f"[{self.name}] sighting update failed: {e}")
            return

        # Create a new sighting for this local track.
        try:
            from app.grid.db import session_scope

            with session_scope() as session:
                sighting = repo.create_sighting(
                    session,
                    camera_id=self.camera_id,
                    vehicle_local_track_id=track.track_id,
                    timestamp=now,
                    plate_raw=hyp.plate_raw if hyp else "",
                    plate_normalized=plate_norm,
                    plate_candidates=plate_candidates,
                    plate_confidence=plate_conf,
                    plate_format_status=plate_status,
                    visual_embedding=track.embedding,
                    vehicle_type=track.vehicle_type,
                    vehicle_color=track.vehicle_color,
                    vehicle_bbox=[
                        track.bbox.x,
                        track.bbox.y,
                        track.bbox.width,
                        track.bbox.height,
                    ],
                    plate_bbox=(track.plate_readings[-1].plate_bbox if track.plate_readings else None),
                    frame_reference=track.crop_ref,
                    detection_confidence=track.confidence,
                    session_id=self.session_id,
                )
                sighting_id = sighting.id
            track.sighting_id = sighting_id
            track.sighting_plate_conf = plate_conf
            self.sightings_created += 1

            event = {
                "type": "sighting",
                "sighting_id": sighting_id,
                "camera_id": self.camera_id,
                "timestamp": now,
                "track_id": track.track_id,
                "plate": plate_norm,
                "plate_confidence": round(plate_conf, 3),
                "plate_status": plate_status,
                "vehicle_type": track.vehicle_type,
                "vehicle_color": track.vehicle_color,
                "bbox": [track.bbox.x, track.bbox.y, track.bbox.width, track.bbox.height],
                "embedding_dim": len(track.embedding or []),
                "is_demo": self.is_demo,
            }
            event_bus.publish("drishti.sightings", event)
            with self._lock:
                self._recent_events.insert(0, event)
                if len(self._recent_events) > 40:
                    self._recent_events.pop()
            fusion_engine.mark_dirty()
        except Exception as e:
            logger.error(f"[{self.name}] sighting emit failed: {e}")

    # ------------------------------------------------------------------
    def _draw_overlay(
        self,
        frame: np.ndarray,
        tracks: List[VehicleTrack],
        plate_boxes: List[Tuple[BoundingBox, float]],
    ) -> None:
        for bbox, conf in plate_boxes:
            cv2.rectangle(
                frame, (bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height), (255, 200, 0), 1
            )

        for track in tracks:
            hx, hy = track.bbox.x, track.bbox.y
            hw, hh = track.bbox.width, track.bbox.height
            hyp = track.plate_hypothesis
            plate_txt = hyp.plate_normalized if hyp and hyp.plate_normalized else ""
            status = hyp.status if hyp else "unknown"

            if status == "confident":
                color = (0, 220, 120)
            elif plate_txt or (hyp and hyp.candidates):
                color = (0, 200, 255)
            else:
                color = (0, 140, 255)

            cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), color, 2)
            label = f"Vehicle #{track.track_id}"
            if plate_txt:
                label += f"  {plate_txt}"
                if hyp:
                    label += f"  OCR {int(hyp.confidence * 100)}%"
            elif hyp and hyp.candidates:
                label += "  plate: uncertain"
            else:
                label += "  plate: n/a"
            cv2.rectangle(frame, (hx, max(0, hy - 20)), (hx + max(200, hw), hy), (10, 25, 47), -1)
            cv2.putText(
                frame, label, (hx + 4, max(14, hy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

        mode = "DEMO" if self.is_demo else "LIVE"
        cv2.putText(
            frame,
            f"{self.name} | {mode} | FPS {self.fps:.1f} | tracks {self._current_tracks}",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 140),
            2,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    def status_dict(self, db_session=None) -> dict:
        with self._lock:
            status = self.status
            error = self.error
            events = list(self._recent_events[:10])
        recent_plates = [
            {
                "plate": e.get("plate") or "",
                "confidence": e.get("plate_confidence", 0.0),
                "timestamp": e.get("timestamp"),
                "status": e.get("plate_status", "unknown"),
            }
            for e in events
            if e.get("plate")
        ]
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "status": status,
            "is_running": self.is_running(),
            "is_demo": self.is_demo,
            "error": error,
            "fps": round(self.fps, 2),
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "sightings_created": self.sightings_created,
            "plates_read": self.plates_read,
            "tracked_vehicles": self._current_tracks,
            "last_frame_at": self.last_frame_at,
            "last_event_at": events[0]["timestamp"] if events else None,
            "recent_plates": recent_plates,
            "recent_events": events,
        }


def _box_center_inside(inner: BoundingBox, outer: BoundingBox) -> bool:
    cx = inner.x + inner.width / 2.0
    cy = inner.y + inner.height / 2.0
    return (outer.x <= cx <= outer.x + outer.width) and (outer.y <= cy <= outer.y + outer.height)


def _merge_boxes(boxes: List[Tuple[BoundingBox, float]], iou_thr: float = 0.45) -> List[Tuple[BoundingBox, float]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[1], reverse=True)
    kept: List[Tuple[BoundingBox, float]] = []
    for box, conf in boxes:
        dup = False
        for kb, _ in kept:
            x1 = max(box.x, kb.x)
            y1 = max(box.y, kb.y)
            x2 = min(box.x + box.width, kb.x + kb.width)
            y2 = min(box.y + box.height, kb.y + kb.height)
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = box.width * box.height + kb.width * kb.height - inter
            if union > 0 and inter / union > iou_thr:
                dup = True
                break
        if not dup:
            kept.append((box, conf))
    return kept


# Imported at the bottom to avoid a circular import at module load time.
from app.grid.fusion import fusion_engine  # noqa: E402
from app.grid.local_tracker import VehicleTrack  # noqa: E402
