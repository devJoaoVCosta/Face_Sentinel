"""
camera_manager.py — Multi-camera detection and concurrent feed management

Responsibilities:
  · Scan indices 0–9 and report available cameras with their resolutions
  · Manage N simultaneous VideoCapture instances, each in its own thread
  · Expose a per-camera frame queue consumed by the GUI
"""

import cv2
import threading
import time
import queue
from dataclasses import dataclass, field


# ─── Data types ───────────────────────────────────────────────────────────────

@dataclass
class CameraInfo:
    """Metadata about a physical camera."""
    index:  int
    width:  int
    height: int
    fps:    float
    label:  str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"Camera {self.index}  [{self.width}×{self.height}]"

    def __str__(self):
        return self.label


@dataclass
class _CameraFeed:
    """Runtime state for one active camera."""
    info:       CameraInfo
    cap:        cv2.VideoCapture
    q:          queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=3))
    thread:     threading.Thread | None = None
    running:    bool = False
    error:      str  = ""


# ─── Scanner ──────────────────────────────────────────────────────────────────

def scan_cameras(max_index: int = 9,
                 progress_cb=None) -> list[CameraInfo]:
    """
    Probe camera indices 0 … max_index (inclusive).

    progress_cb(current_index, max_index) — called for each probe attempt.
    Returns a list of CameraInfo for every camera that opened successfully.
    """
    found: list[CameraInfo] = []

    for idx in range(max_index + 1):
        if progress_cb:
            progress_cb(idx, max_index)

        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue

        # try to read one frame to confirm the device is live
        ok, _ = cap.read()
        if not ok:
            cap.release()
            continue

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        found.append(CameraInfo(index=idx, width=w, height=h, fps=fps))

    return found


# ─── Manager ──────────────────────────────────────────────────────────────────

class CameraManager:
    """
    Manages N concurrent VideoCapture streams.

    Usage
    -----
    mgr = CameraManager()
    mgr.open([0, 2])          # open cameras 0 and 2
    frame = mgr.get_frame(0)  # latest frame from camera 0 (or None)
    mgr.close_all()
    """

    # Preferred capture resolution (camera may override)
    PREFERRED_W = 1280
    PREFERRED_H = 720

    def __init__(self):
        self._feeds: dict[int, _CameraFeed] = {}   # index → _CameraFeed
        self._lock  = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def active_indices(self) -> list[int]:
        with self._lock:
            return sorted(self._feeds.keys())

    @property
    def count(self) -> int:
        return len(self._feeds)

    def open(self, indices: list[int]) -> dict[int, str]:
        """
        Open cameras for the given indices.
        Returns {index: error_message} for any that failed; successful ones
        have no entry.
        """
        errors: dict[int, str] = {}
        for idx in indices:
            if idx in self._feeds:
                continue                     # already open
            err = self._open_one(idx)
            if err:
                errors[idx] = err
        return errors

    def close(self, index: int):
        with self._lock:
            feed = self._feeds.pop(index, None)
        if feed:
            feed.running = False
            if feed.thread and feed.thread.is_alive():
                feed.thread.join(timeout=1.5)
            feed.cap.release()

    def close_all(self):
        for idx in list(self._feeds.keys()):
            self.close(idx)

    def get_frame(self, index: int):
        """Return latest frame (ndarray) from camera *index*, or None."""
        with self._lock:
            feed = self._feeds.get(index)
        if feed is None:
            return None
        try:
            return feed.q.get_nowait()
        except queue.Empty:
            return None

    def get_info(self, index: int) -> CameraInfo | None:
        with self._lock:
            feed = self._feeds.get(index)
        return feed.info if feed else None

    def is_open(self, index: int) -> bool:
        with self._lock:
            return index in self._feeds and self._feeds[index].running

    # ── internals ─────────────────────────────────────────────────────────────

    def _open_one(self, index: int) -> str:
        """Try to open camera *index*. Returns error string or '' on success."""
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return f"Cannot open camera {index}"

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.PREFERRED_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.PREFERRED_H)

        ok, _ = cap.read()
        if not ok:
            cap.release()
            return f"Camera {index} opened but returned no frame"

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        info = CameraInfo(index=index, width=w, height=h, fps=fps)

        feed = _CameraFeed(info=info, cap=cap)
        feed.running = True
        feed.thread  = threading.Thread(
            target=self._capture_loop, args=(feed,), daemon=True)

        with self._lock:
            self._feeds[index] = feed
        feed.thread.start()
        return ""

    def _capture_loop(self, feed: _CameraFeed):
        while feed.running and feed.cap.isOpened():
            ok, frame = feed.cap.read()
            if ok:
                # discard oldest if queue full to keep latency low
                if feed.q.full():
                    try:
                        feed.q.get_nowait()
                    except queue.Empty:
                        pass
                feed.q.put(frame)
            else:
                feed.error = "Read error"
                time.sleep(0.05)
