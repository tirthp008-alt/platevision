"""Live CCTV, RTSP, HLS, and Public Traffic Stream Manager with multi-plate detection,
vehicle tracking, stream probing, and MJPEG broadcasting.
"""

import os
import threading
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.core.logging import logger
from app.schemas.detection import BoundingBox
from app.services.detector.factory import get_detector
from app.services.ocr.engine import ocr_engine
from app.utils.image_ops import encode_image_to_base64


class CCTVVehicleEvent:
    def __init__(
        self,
        plate_number: str,
        formatted_number: str,
        confidence: float,
        format_status: str,
        first_seen: float,
        last_seen: float,
        crop_base64: str,
    ):
        self.plate_number = plate_number
        self.formatted_number = formatted_number
        self.confidence = confidence
        self.format_status = format_status
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.crop_base64 = crop_base64

    def to_dict(self):
        return {
            "plate_number": self.plate_number,
            "formatted_number": self.formatted_number,
            "confidence": round(self.confidence, 2),
            "format_status": self.format_status,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "crop_base64": self.crop_base64,
        }


class CCTVStreamManager:
    """Manages an active RTSP/HLS/MJPEG/Public/Synthetic traffic video feed, processes frames with multi-plate AI,
    and exposes an annotated MJPEG stream and real-time ANPR event log.
    """

    def __init__(self):
        self.stream_url: Optional[str] = None
        self.resolved_url: Optional[str] = None
        self.is_running: bool = False
        self.capture_thread: Optional[threading.Thread] = None
        self.current_frame: Optional[np.ndarray] = None
        self.annotated_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.fps: float = 24.0
        self.analysis_fps: int = 3  # Frames per second to analyze
        self.detected_events: List[CCTVVehicleEvent] = []
        self.recent_plates: Dict[str, float] = {}  # Plate -> timestamp
        self.mode: str = "stopped"  # 'synthetic', 'stream', 'stopped'
        self.status_state: str = "OFFLINE"  # 'OFFLINE', 'CONNECTING', 'LIVE STREAMING', 'RECONNECTING', 'ERROR'
        self.reconnect_attempts: int = 0
        self.error_message: Optional[str] = None
        self._last_annotations = []

    def probe_stream(self, url: str) -> Dict:
        """Tests stream reachability, resolves YouTube/HLS/RTSP links, and returns codec/resolution/latency."""
        start_t = time.time()
        test_url = self._resolve_stream_url(url)

        if url.startswith("demo"):
            return {
                "status": "ok",
                "protocol": "Synthetic Highway Simulation",
                "resolution": "960x540",
                "fps": 25.0,
                "latency_ms": 12,
                "message": "Built-in Traffic Stream is ready for immediate connection.",
            }

        try:
            # Set RTSP over TCP for reliable network transmission
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(test_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                return {
                    "status": "error",
                    "protocol": self._detect_protocol(url),
                    "message": "Unable to establish connection to video stream. Verify URL, network access, or authentication credentials.",
                }

            ret, frame = cap.read()
            cap.release()

            latency = int((time.time() - start_t) * 1000)
            if not ret or frame is None:
                return {
                    "status": "error",
                    "protocol": self._detect_protocol(url),
                    "message": "Stream connected but no video frames received.",
                }

            h, w = frame.shape[:2]
            return {
                "status": "ok",
                "protocol": self._detect_protocol(url),
                "resolution": f"{w}x{h}",
                "fps": 25.0,
                "latency_ms": latency,
                "message": f"Successfully probed stream ({w}x{h} @ {latency}ms latency). Ready for live ANPR.",
            }
        except Exception as e:
            return {
                "status": "error",
                "protocol": self._detect_protocol(url),
                "message": f"Connection probe failed: {str(e)}",
            }

    def _detect_protocol(self, url: str) -> str:
        if "youtube.com" in url or "youtu.be" in url:
            return "YouTube Live Stream"
        elif url.startswith("rtsp://"):
            return "RTSP (Real-Time Streaming Protocol)"
        elif ".m3u8" in url:
            return "HLS (HTTP Live Streaming)"
        elif "/mjpg" in url or "mjpeg" in url:
            return "MJPEG Camera Stream"
        elif url.startswith("http://") or url.startswith("https://"):
            return "HTTP Video Stream"
        elif url.startswith("demo"):
            return "Synthetic Highway Simulation"
        return "Custom Video Stream"

    def _resolve_stream_url(self, raw_url: str) -> str:
        """Resolves permitted YouTube Live or third-party links via yt-dlp to direct media streams."""
        if not raw_url:
            return "demo_traffic"

        if "youtube.com" in raw_url or "youtu.be" in raw_url:
            try:
                import yt_dlp  # type: ignore

                ydl_opts = {
                    "format": "best[ext=mp4]/best",
                    "quiet": True,
                    "no_warnings": True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(raw_url, download=False)
                    if "url" in info:
                        return info["url"]
            except Exception as e:
                logger.warning(f"Could not extract YouTube stream via yt-dlp ({e}), attempting direct open.")

        return raw_url

    def start_stream(self, stream_url: str = "demo_traffic", analysis_fps: int = 3) -> Dict[str, str]:
        with self.lock:
            if self.is_running:
                self.is_running = False
                time.sleep(0.1)

            self.stream_url = stream_url
            self.analysis_fps = max(1, min(10, analysis_fps))
            self.is_running = True
            self.status_state = "CONNECTING"
            self.error_message = None
            self.reconnect_attempts = 0
            self.detected_events.clear()
            self.recent_plates.clear()
            self._last_annotations.clear()

            if stream_url in ["demo_traffic", "demo_toll", "demo_blr", "demo"]:
                self.mode = "synthetic"
                self.capture_thread = threading.Thread(target=self._synthetic_stream_worker, daemon=True)
            else:
                self.mode = "stream"
                self.resolved_url = self._resolve_stream_url(stream_url)
                self.capture_thread = threading.Thread(target=self._live_stream_worker, daemon=True)

            self.capture_thread.start()
            logger.info(f"Started Traffic stream in mode '{self.mode}' on '{stream_url}' (Analysis FPS: {self.analysis_fps})")
            return {"status": "started", "mode": self.mode, "stream_url": stream_url}

    def stop_stream(self) -> Dict[str, str]:
        with self.lock:
            self.is_running = False
            self.mode = "stopped"
            self.status_state = "OFFLINE"
            self.current_frame = None
            self.annotated_frame = None
            logger.info("Stopped Traffic stream.")
            return {"status": "stopped"}

    def get_status(self) -> Dict:
        with self.lock:
            return {
                "is_running": self.is_running,
                "status_state": self.status_state,
                "mode": self.mode,
                "stream_url": self.stream_url,
                "protocol": self._detect_protocol(self.stream_url or ""),
                "fps": round(self.fps, 1),
                "analysis_fps": self.analysis_fps,
                "total_events": len(self.detected_events),
                "reconnect_attempts": self.reconnect_attempts,
                "error_message": self.error_message,
                "recent_events": [e.to_dict() for e in self.detected_events[:15]],
            }

    def _run_detection_pass(self, frame: np.ndarray):
        detector = get_detector()
        h, w = frame.shape[:2]
        now = time.time()

        try:
            raw_dets = detector.detect(frame)
        except Exception:
            raw_dets = []

        new_annotations = []

        for det in raw_dets:
            bbox = det.bbox
            pad_x = int(bbox.width * 0.08)
            pad_y = int(bbox.height * 0.08)
            x1 = max(0, bbox.x - pad_x)
            y1 = max(0, bbox.y - pad_y)
            x2 = min(w, bbox.x + bbox.width + pad_x)
            y2 = min(h, bbox.y + bbox.height + pad_y)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 15:
                continue

            ocr_res = ocr_engine.recognize(crop)
            plate_text = ocr_res.formatted_text or ocr_res.normalized_text

            new_annotations.append({
                "bbox": bbox,
                "label": f"{plate_text or 'DETECTING...'} ({int(det.confidence * 100)}%)",
                "conf": det.confidence,
            })

            # Event logging with 3-second deduplication
            target_text = ocr_res.normalized_text or (plate_text.replace(" ", "") if plate_text else "")
            if target_text and len(target_text) >= 3:
                last_seen = self.recent_plates.get(target_text, 0)
                if now - last_seen > 3.0:
                    crop_b64 = encode_image_to_base64(crop, format_type="jpeg", quality=85)
                    event = CCTVVehicleEvent(
                        plate_number=target_text,
                        formatted_number=ocr_res.formatted_text or plate_text,
                        confidence=max(det.confidence, ocr_res.confidence),
                        format_status=ocr_res.format_status,
                        first_seen=now,
                        last_seen=now,
                        crop_base64=crop_b64,
                    )
                    self.detected_events.insert(0, event)
                    if len(self.detected_events) > 100:
                        self.detected_events.pop()

                self.recent_plates[target_text] = now

        with self.lock:
            self._last_annotations = new_annotations

    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame.copy()
        with self.lock:
            current_annotations = list(self._last_annotations)

        for ann in current_annotations:
            bbox = ann["bbox"]
            label = ann["label"]

            # Draw HUD Box
            cv2.rectangle(
                annotated,
                (bbox.x, bbox.y),
                (bbox.x + bbox.width, bbox.y + bbox.height),
                (0, 240, 255),
                2,
            )

            # Draw Label Header
            cv2.rectangle(
                annotated,
                (bbox.x, max(0, bbox.y - 26)),
                (bbox.x + bbox.width, bbox.y),
                (11, 19, 43),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (bbox.x + 4, max(16, bbox.y - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 240, 255),
                2,
                cv2.LINE_AA,
            )

        # Draw Official ANPR HUD overlay
        cv2.putText(
            annotated,
            f"LIVE TRAFFIC ANPR | FPS: {self.fps:.1f} | SENSORS: ACTIVE",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 128),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _live_stream_worker(self):
        """Worker thread for decoding RTSP/HLS/MJPEG live video streams with auto-reconnect."""
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        url = self.resolved_url or self.stream_url

        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        last_time = time.time()
        last_analysis_time = time.time()
        frame_count = 0

        with self.lock:
            self.status_state = "LIVE STREAMING"

        while self.is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                with self.lock:
                    self.status_state = "RECONNECTING"
                    self.reconnect_attempts += 1

                time.sleep(2.0)
                if self.reconnect_attempts > 10:
                    with self.lock:
                        self.status_state = "ERROR"
                        self.error_message = "Maximum reconnect attempts reached. Verify stream availability."
                    break

                cap.open(url)
                continue

            with self.lock:
                self.status_state = "LIVE STREAMING"
                self.reconnect_attempts = 0

            frame_count += 1
            now = time.time()

            # Rate-controlled frame analysis
            if now - last_analysis_time >= (1.0 / self.analysis_fps):
                self._run_detection_pass(frame)
                last_analysis_time = now

            annotated = self._draw_hud(frame)
            with self.lock:
                self.current_frame = frame
                self.annotated_frame = annotated

            if now - last_time >= 1.0:
                self.fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now

            time.sleep(0.03)

        cap.release()

    def _synthetic_stream_worker(self):
        """Generates realistic Indian highway traffic simulation feeds."""
        cars = [
            {"plate": "GJ 01 AB 1234", "color": (216, 78, 29), "x": 100, "speed": 4, "lane": 300},
            {"plate": "DL 01 CA 1234", "color": (30, 41, 59), "x": 650, "speed": -3, "lane": 180},
            {"plate": "MH 12 DE 1433", "color": (180, 50, 50), "x": -200, "speed": 5, "lane": 330},
            {"plate": "22 BH 1234 AA", "color": (40, 140, 60), "x": 1200, "speed": -4, "lane": 160},
            {"plate": "KA 03 MN 4567", "color": (25, 90, 180), "x": -600, "speed": 6, "lane": 290},
        ]

        last_time = time.time()
        last_analysis_time = time.time()
        frame_count = 0
        w, h = 960, 540

        with self.lock:
            self.status_state = "LIVE STREAMING"

        while self.is_running:
            frame = np.full((h, w, 3), 30, dtype=np.uint8)

            # Draw highway lanes
            cv2.rectangle(frame, (0, 120), (w, 480), (45, 55, 72), -1)
            for lane_y in [210, 290, 370]:
                for mark_x in range(0, w, 80):
                    cv2.rectangle(frame, (mark_x, lane_y), (mark_x + 40, lane_y + 4), (240, 240, 240), -1)

            # Draw vehicles
            for car in cars:
                car["x"] += car["speed"]
                if car["speed"] > 0 and car["x"] > w + 200:
                    car["x"] = -400
                elif car["speed"] < 0 and car["x"] < -400:
                    car["x"] = w + 300

                cx = int(car["x"])
                cy = car["lane"]
                cw = 300
                ch = 110

                # Car body
                cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), car["color"], -1)
                cv2.rectangle(frame, (cx + 40, cy - 35), (cx + cw - 40, cy), (15, 23, 42), -1)

                # License Plate
                pw, ph = 200, 42
                px = cx + (cw - pw) // 2
                py = cy + ch - 48
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 255, 255), -1)
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 0), 2)
                # Blue IND badge
                cv2.rectangle(frame, (px, py), (px + 18, py + ph), (180, 50, 20), -1)
                cv2.putText(
                    frame,
                    car["plate"],
                    (px + 22, py + 29),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            frame_count += 1
            now = time.time()

            if now - last_analysis_time >= (1.0 / self.analysis_fps):
                self._run_detection_pass(frame)
                last_analysis_time = now

            annotated = self._draw_hud(frame)
            with self.lock:
                self.current_frame = frame
                self.annotated_frame = annotated

            if now - last_time >= 1.0:
                self.fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now

            time.sleep(0.035)

    def generate_mjpeg_stream(self):
        while self.is_running:
            with self.lock:
                if self.annotated_frame is None:
                    time.sleep(0.03)
                    continue
                frame = self.annotated_frame.copy()

            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.035)


cctv_manager = CCTVStreamManager()
