"""Multi-camera manager.

Owns one :class:`~app.grid.pipeline.CameraWorker` per registered camera. Camera
failures are isolated: each worker runs in its own thread and any exception is
contained to that camera's status. The manager also exposes aggregate helpers
used by the dashboard status panel.
"""

import threading
from typing import Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger
from app.grid.pipeline import CameraWorker


class CameraManager:
    def __init__(self):
        self._workers: Dict[str, CameraWorker] = {}
        self._lock = threading.Lock()

    def _sync_camera(self, cam: dict) -> CameraWorker:
        """Create a worker for a camera record (idempotent)."""
        with self._lock:
            worker = self._workers.get(cam["id"])
            if worker is None:
                worker = CameraWorker(cam)
                self._workers[cam["id"]] = worker
            return worker

    def start(self, cam: dict) -> dict:
        worker = self._sync_camera(cam)
        worker.start()
        return worker.status_dict()

    def stop(self, camera_id: str) -> dict:
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker is None:
            return {"camera_id": camera_id, "status": "OFFLINE", "is_running": False}
        worker.stop()
        return worker.status_dict()

    def stop_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
        for w in workers:
            try:
                w.stop()
            except Exception:
                pass

    def restart(self, cam: dict) -> dict:
        self.stop(cam["id"])
        return self.start(cam)

    def get_worker(self, camera_id: str) -> Optional[CameraWorker]:
        with self._lock:
            return self._workers.get(camera_id)

    def status(self, camera_id: str) -> Optional[dict]:
        w = self.get_worker(camera_id)
        return w.status_dict() if w else None

    def all_status(self) -> Dict[str, dict]:
        with self._lock:
            return {cid: w.status_dict() for cid, w in self._workers.items()}

    def running_count(self) -> int:
        with self._lock:
            return sum(1 for w in self._workers.values() if w.is_running())

    def aggregate(self) -> dict:
        with self._lock:
            workers = list(self._workers.values())
        running = [w for w in workers if w.is_running()]
        return {
            "cameras_registered": len(workers),
            "cameras_processing": len(running),
            "frames_processed": sum(w.frames_processed for w in workers),
            "sightings_created": sum(w.sightings_created for w in workers),
            "plates_read": sum(w.plates_read for w in workers),
            "tracked_vehicles": sum(w._current_tracks for w in workers),
            "average_fps": round(
                sum(w.fps for w in running) / len(running), 2
            ) if running else 0.0,
        }

    def remove(self, camera_id: str) -> None:
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass


camera_manager = CameraManager()
