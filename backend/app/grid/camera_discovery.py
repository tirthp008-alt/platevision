"""Local camera device discovery.

Enumerates USB/built-in cameras via OpenCV and returns human-readable labels.
Enumerating devices reliably is platform dependent (on Windows the DirectShow
backend is queried; elsewhere probe indexes open). Probing is bounded so a
machine without cameras returns quickly instead of hanging.
"""

import platform
import subprocess
from typing import Dict, List

import cv2

from app.core.config import settings
from app.core.logging import logger


def _windows_device_names() -> List[str]:
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-PnpDevice -Class Camera,Image -Status OK | "
                "Select-Object -ExpandProperty FriendlyName",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _v4l2_device_names() -> Dict[int, str]:
    """Map /dev/videoN index to a card name on Linux."""
    names: Dict[int, str] = {}
    try:
        import glob
        import os

        for path in sorted(glob.glob("/sys/class/video4linux/video*")):
            index = int(os.path.basename(path).replace("video", ""))
            name_file = os.path.join(path, "name")
            if os.path.exists(name_file):
                with open(name_file) as fh:
                    names[index] = fh.read().strip()
    except Exception:
        pass
    return names


def discover_cameras(max_devices: int = None) -> List[Dict]:
    """Return discovered local camera devices.

    Each entry: {device_id, index, label, platform, backend}.
    """
    max_devices = max_devices or settings.CAMERA_MAX_DEVICE_PROBE
    os_name = platform.system()
    names = _v4l2_device_names() if os_name == "Linux" else {}
    win_names = _windows_device_names() if os_name == "Windows" else []

    devices: List[Dict] = []
    for index in range(max_devices):
        label = names.get(index) or (win_names[index] if index < len(win_names) else "")
        opened = False
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os_name == "Windows" else cv2.CAP_ANY)
            opened = bool(cap.isOpened())
            if opened:
                ok, _ = cap.read()
                opened = bool(ok)
            cap.release()
        except Exception:
            opened = False
        # Include devices we can identify by name even if the probe failed
        # (e.g. camera already in use by another process).
        available = opened or bool(label)
        if not available:
            continue
        devices.append(
            {
                "device_id": f"device:{index}",
                "index": index,
                "label": label or f"Camera {index}",
                "platform": os_name,
                "backend": "dshow" if os_name == "Windows" else "v4l2",
                "probe_ok": bool(opened),
            }
        )
    logger.info(f"Camera discovery found {len(devices)} local device(s) on {os_name}.")
    return devices
