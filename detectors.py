"""
Person detection: MediaPipe's object detector (EfficientDet, COCO dataset)
limited to the "person" class. It reports where people are in each frame; it
doesn't tell people apart or track identities. server.py uses the GPU delegate
when it can.

Results are returned in normalised [0, 1] coordinates so they can be drawn on
the full-size frame even though detection ran on a downscaled copy.
"""

import functools
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np


@dataclass
class Person:
    box: tuple               # (x, y, w, h) normalised
    score: float


@dataclass
class Detections:
    people: list = field(default_factory=list)
    ms: float = 0.0          # time spent detecting


@functools.lru_cache(maxsize=None)
def _native_crash(model):
    """MediaPipe's native code can kill the process outright on CPUs it wasn't
    built for (MediaPipe 1.x needs the ARM AES instructions, which the Pi 4
    lacks), and Python can't catch that. So load the model in a child process
    first. Returns a reason string if the child crashed, else None."""
    if os.environ.get("KESTREL_PROBE"):
        return None
    code = "import sys, detectors; detectors.Detector(sys.argv[1])"
    try:
        r = subprocess.run([sys.executable, "-c", code, model],
                           cwd=os.path.dirname(os.path.abspath(__file__)),
                           env={**os.environ, "KESTREL_PROBE": "1"},
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "loading the MediaPipe model timed out"
    if r.returncode >= 0:
        return None
    try:
        sig = signal.Signals(-r.returncode).name
    except ValueError:
        sig = f"signal {-r.returncode}"
    fatal = next((s for s in r.stderr.splitlines() if "FATAL" in s), "")
    hint = " (CPU lacks AES instructions, e.g. Raspberry Pi 4)" if "aes" in fatal else ""
    return f"MediaPipe crashes on this CPU with {sig}{hint}"


class Detector:
    def __init__(self, model=None, gpu=False):
        """gpu: try the GPU delegate. In a long-running process, only pass True
        once probe_gpu() says it works: a model that fails on the GPU leaves an
        object behind whose cleanup can deadlock MediaPipe later."""
        self.people = None
        self.status = []
        self.delegate = None
        self._last_ts = 0

        if not model:
            return
        if not os.path.exists(model):
            self.status.append(f"{model} missing — run download_models.sh")
            return
        crash = _native_crash(model)
        if crash:
            self.status.append(f"{crash} — detection off, use --server <gpu-machine>")
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
        except ImportError:
            self.status.append("mediapipe not installed — detection off")
            return
        self._mp = mp

        def create(delegate):
            return vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
                base_options=mpp.BaseOptions(model_asset_path=model, delegate=delegate),
                category_allowlist=["person"],
                score_threshold=0.5,
                running_mode=vision.RunningMode.VIDEO))

        if gpu:
            # Some models load on the GPU but fail on the first inference,
            # so run one before trusting it
            t = None
            try:
                t = create(mpp.BaseOptions.Delegate.GPU)
                t.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                            data=np.zeros((240, 320, 3), np.uint8)), 0)
                self.people, self.delegate = t, "GPU"
                return
            except Exception as e:
                reason = next((s.strip("└ ") for s in str(e).splitlines() if "Failed" in s),
                              str(e).splitlines()[0] if str(e) else type(e).__name__)
                self.status.append(f"GPU delegate failed ({reason}) — using CPU")
                if t is not None:
                    try:
                        t.close()       # re-raises the failed graph's error
                    except Exception:
                        pass
        self.people, self.delegate = create(mpp.BaseOptions.Delegate.CPU), "CPU"

    @property
    def enabled(self):
        return self.people is not None

    def detect(self, frame_bgr):
        t0 = time.perf_counter()
        out = Detections()
        if not self.enabled:
            return out

        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs strictly increasing timestamps
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts

        res = self.people.detect_for_video(image, ts)
        for det in res.detections:
            b = det.bounding_box
            out.people.append(Person(
                box=(b.origin_x / w, b.origin_y / h, b.width / w, b.height / h),
                score=det.categories[0].score if det.categories else 0.0))

        out.ms = (time.perf_counter() - t0) * 1000
        return out

    def close(self):
        if self.people is not None:
            try:
                self.people.close()
            except Exception:
                pass


def _probe_child(conn, model):
    d = Detector(model, gpu=True)
    conn.send((d.delegate == "GPU", d.status))
    conn.close()
    os._exit(0)     # skip teardown: destroying a GPU task that failed can hang


def probe_gpu(model, timeout=120):
    """Find out whether the model really runs on the GPU, testing in a throwaway
    process so a failure there can't leave anything behind in this one.
    Returns (works on the GPU, status messages)."""
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    recv_end, send_end = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_probe_child, args=(send_end, model), daemon=True)
    proc.start()
    send_end.close()
    try:
        if not recv_end.poll(timeout):
            return False, [f"GPU test timed out after {timeout}s — using CPU"]
        return recv_end.recv()
    except EOFError:
        return False, ["GPU test crashed — using CPU"]
    finally:
        proc.kill()
        proc.join()
