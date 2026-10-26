import sys
import threading
import time

import cv2


class Camera:
    """Reads frames on a background thread and keeps only the newest one,
    so slow inference never works on a stale, queued-up frame."""

    def __init__(self, index=0, width=1280, height=720):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._cond = threading.Condition()
        self._frame = None
        self._t = 0.0
        self._seq = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame = frame
                self._t = time.monotonic()
                self._seq += 1
                self._cond.notify_all()

    def read(self, last_seq=0, timeout=2.0):
        """Return (seq, timestamp, frame) for the newest frame after last_seq."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > last_seq, timeout):
                raise TimeoutError("No frame from camera")
            return self._seq, self._t, self._frame

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()
