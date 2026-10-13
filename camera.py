"""
Camera sources: Pi camera (CSI, via picamera2), USB webcam, or a video file.

Each source runs a capture thread and keeps only the newest frame, so a slow
detection step drops frames instead of building up lag.
"""

import threading
import time

import cv2 as cv


class _ThreadedSource:
    def __init__(self, name):
        self.name = name
        self._cond = threading.Condition()
        self._frame = None
        self._frame_id = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.failed = None

    def start(self):
        self._thread.start()
        return self

    def read(self, last_id=0, timeout=1.0):
        """Wait for a frame newer than last_id. Returns (frame_id, frame) or (last_id, None)."""
        with self._cond:
            self._cond.wait_for(lambda: self._frame_id != last_id or self._stop.is_set(),
                                timeout=timeout)
            if self._frame_id == last_id:
                return last_id, None
            return self._frame_id, self._frame

    def _publish(self, frame):
        with self._cond:
            self._frame = frame
            self._frame_id += 1
            self._cond.notify_all()

    def _loop(self):
        try:
            while not self._stop.is_set():
                frame = self._grab()
                if frame is None:
                    time.sleep(0.01)
                    continue
                self._publish(frame)
        except Exception as e:          # report to the main loop instead of dying silently
            self.failed = str(e)
            with self._cond:
                self._cond.notify_all()

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        self._release()

    def _grab(self):
        raise NotImplementedError

    def _release(self):
        pass


class PiCamera(_ThreadedSource):
    def __init__(self, width, height, fps):
        super().__init__("picamera2")
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise RuntimeError(
                "picamera2 not available. Install it with `sudo apt install python3-picamera2` "
                "and create the venv with --system-site-packages (setup.sh does this).") from e
        self.cam = Picamera2()
        cfg = self.cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},   # RGB888 = BGR byte order
            controls={"FrameRate": float(fps)},
            buffer_count=4)
        self.cam.configure(cfg)
        self.cam.start()

    def _grab(self):
        return self.cam.capture_array("main")

    def _release(self):
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:
            pass


class USBCamera(_ThreadedSource):
    def __init__(self, index, width, height, fps):
        super().__init__(f"usb:{index}")
        self.cap = cv.VideoCapture(index, cv.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open /dev/video{index}")
        # MJPG keeps USB bandwidth (and Pi CPU) down at 640x480+
        self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv.CAP_PROP_FPS, fps)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    def _grab(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def _release(self):
        self.cap.release()


class VideoFile(_ThreadedSource):
    """Loops a video file at its native rate. Handy for testing on a desktop."""

    def __init__(self, path, width, height, fps):
        super().__init__(f"file:{path}")
        self.cap = cv.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file {path}")
        src_fps = self.cap.get(cv.CAP_PROP_FPS)
        self.period = 1.0 / (src_fps if src_fps and src_fps > 0 else fps)
        self.size = (width, height)
        self._next = time.monotonic()

    def _grab(self):
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return None
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv.resize(frame, self.size)
        self._next += self.period
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next = time.monotonic()
        return frame

    def _release(self):
        self.cap.release()


def open_camera(source, width, height, fps):
    """source: 'picam', a device index like '0', or a video file path."""
    s = str(source)
    if s.lower() in ("picam", "csi", "picamera"):
        cam = PiCamera(width, height, fps)
    elif s.isdigit():
        cam = USBCamera(int(s), width, height, fps)
    else:
        cam = VideoFile(s, width, height, fps)
    return cam.start()
