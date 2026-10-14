"""
RemoteDetector: same interface as detectors.Detector, but sends frames to
server.py on a GPU machine instead of running MediaPipe on the Pi.

detect() never waits on the network. It hands the newest frame to a background
thread and returns the most recent result the server sent back, so the camera
loop keeps its frame rate even when the link is slow. Frames that arrive while
a request is in flight are dropped, and only the newest one is sent next. The
thread reconnects on its own if the server goes away.
"""

import socket
import threading
import time

import cv2 as cv

from detectors import Detections
from protocol import detections_from_dict, parse_addr, recv_json, send_json, send_msg, tune

STALE_AFTER = 1.0       # seconds before old results stop being drawn


class RemoteDetector:
    def __init__(self, addr, people=True, quality=80, timeout=5.0):
        self.host, self.port = parse_addr(addr)
        self.quality = quality
        self.timeout = timeout
        self._hello = {"people": people}
        # main.py checks this to decide what can be toggled and drawn
        self.people = True if people else None
        self.status = [f"remote server {self.host}:{self.port}"]
        self.error = "connecting to server"

        self._cond = threading.Condition()
        self._frame = None
        self._result, self._result_t = Detections(), 0.0
        self._sock = None
        self._stop = False
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    @property
    def enabled(self):
        return self.people is not None

    def detect(self, frame_bgr):
        with self._cond:
            self._frame = frame_bgr
            self._cond.notify()
            if time.monotonic() - self._result_t > STALE_AFTER:
                return Detections()
            return self._result

    def close(self):
        with self._cond:
            self._stop = True
            self._cond.notify()
        sock = self._sock
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)

    # --- background thread ---

    def _run(self):
        backoff = 1.0
        while not self._stop:
            try:
                self._session()
                backoff = 1.0
            except (OSError, ConnectionError, ValueError) as e:
                if self._stop:
                    break
                self.error = f"server offline ({e.__class__.__name__})"
            finally:
                if self._sock:
                    self._sock.close()
                    self._sock = None
            with self._cond:
                self._cond.wait_for(lambda: self._stop, timeout=backoff)
            backoff = min(backoff * 2, 10.0)

    def _session(self):
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock = sock
        tune(sock)
        send_json(sock, self._hello)
        reply = recv_json(sock)
        if not reply.get("ok"):
            raise ConnectionError(reply.get("error", "server refused"))
        self.error = None
        print(f"Detection server connected: people on {reply.get('delegate') or 'nothing'}")
        for s in reply.get("status", []):
            print(f"Detection server: {s}")

        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._stop or self._frame is not None)
                if self._stop:
                    return
                frame, self._frame = self._frame, None
            ok, buf = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, self.quality])
            if not ok:
                continue
            t0 = time.perf_counter()
            send_msg(sock, buf.tobytes())
            dets = detections_from_dict(recv_json(sock))
            dets.ms = (time.perf_counter() - t0) * 1000     # round trip, not just GPU time
            with self._cond:
                self._result, self._result_t = dets, time.monotonic()
