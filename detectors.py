"""
MediaPipe hand landmarks + face detection. On the Pi this runs on the CPU (the
Pi 4 has no usable GPU delegate); server.py uses the GPU delegate when it can.
Face *identity* tracking from the desktop version is removed — dlib/face_recognition
is far too slow on a Pi.

Results are returned in normalised [0, 1] coordinates so they can be drawn on
the full-size frame even though detection ran on a downscaled copy.
"""

import os
import time
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np


@dataclass
class Hand:
    landmarks: list          # 21 x (x, y) normalised
    label: str               # "Left" / "Right"
    score: float


@dataclass
class Face:
    box: tuple               # (x, y, w, h) normalised
    score: float


@dataclass
class Detections:
    hands: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    ms: float = 0.0          # time spent detecting


class Detector:
    def __init__(self, hand_model=None, face_model=None, num_hands=2, gpu=False):
        """gpu: True to try the GPU for every model, or a set of model names
        ("hands", "faces") from probe_gpu(). In a long-running process, pass
        the set: a model that fails on the GPU leaves an object behind whose
        cleanup can deadlock MediaPipe later."""
        self.hands = None
        self.faces = None
        self.status = []
        self._last_ts = 0

        if not (hand_model or face_model):
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
        except ImportError:
            self.status.append("mediapipe not installed — detection off")
            return
        self._mp = mp
        self.delegates = delegates = {}

        def create(name, task, options, model, **kw):
            # Try the GPU per model and fall back to CPU. Some models load on the
            # GPU but fail on the first inference, so run one before trusting it.
            if gpu is True or (not isinstance(gpu, bool) and name in gpu):
                t = None
                try:
                    t = task.create_from_options(options(base_options=mpp.BaseOptions(
                        model_asset_path=model, delegate=mpp.BaseOptions.Delegate.GPU), **kw))
                    blank = mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.zeros((240, 320, 3), np.uint8))
                    t.detect_for_video(blank, 0)
                    delegates[name] = "GPU"
                    return t
                except Exception as e:
                    reason = next((s.strip("└ ") for s in str(e).splitlines() if "Failed" in s),
                                  str(e).splitlines()[0] if str(e) else type(e).__name__)
                    self.status.append(f"{name}: GPU delegate failed ({reason}) — using CPU")
                    if t is not None:
                        try:
                            t.close()       # re-raises the failed graph's error
                        except Exception:
                            pass
            delegates[name] = "CPU"
            return task.create_from_options(options(base_options=mpp.BaseOptions(
                model_asset_path=model, delegate=mpp.BaseOptions.Delegate.CPU), **kw))

        if hand_model:
            if os.path.exists(hand_model):
                self.hands = create(
                    "hands", vision.HandLandmarker, vision.HandLandmarkerOptions, hand_model,
                    num_hands=num_hands,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    running_mode=vision.RunningMode.VIDEO)
            else:
                self.status.append(f"{hand_model} missing — run download_models.sh")

        if face_model:
            if os.path.exists(face_model):
                self.faces = create(
                    "faces", vision.FaceDetector, vision.FaceDetectorOptions, face_model,
                    min_detection_confidence=0.5,
                    running_mode=vision.RunningMode.VIDEO)
            else:
                self.status.append(f"{face_model} missing — run download_models.sh")
        # e.g. "hands GPU, faces CPU"
        self.delegate = ", ".join(f"{k} {v}" for k, v in delegates.items()) or None

    @property
    def enabled(self):
        return self.hands is not None or self.faces is not None

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

        if self.hands is not None:
            res = self.hands.detect_for_video(image, ts)
            for lms, handed in zip(res.hand_landmarks, res.handedness):
                cat = handed[0] if handed else None
                out.hands.append(Hand(
                    landmarks=[(lm.x, lm.y) for lm in lms],
                    label=cat.category_name if cat else "?",
                    score=cat.score if cat else 0.0))

        if self.faces is not None:
            res = self.faces.detect_for_video(image, ts)
            for det in res.detections:
                b = det.bounding_box
                out.faces.append(Face(
                    box=(b.origin_x / w, b.origin_y / h, b.width / w, b.height / h),
                    score=det.categories[0].score if det.categories else 0.0))

        out.ms = (time.perf_counter() - t0) * 1000
        return out

    def close(self):
        for d in (self.hands, self.faces):
            if d is not None:
                try:
                    d.close()
                except Exception:
                    pass


def _probe_child(conn, hand_model, face_model):
    d = Detector(hand_model, face_model, gpu=True)
    conn.send(({k for k, v in d.delegates.items() if v == "GPU"}, d.status))
    conn.close()
    os._exit(0)     # skip teardown: destroying a GPU task that failed can hang


def probe_gpu(hand_model, face_model, timeout=120):
    """Find which models really run on the GPU, testing in a throwaway process
    so a model that fails there can't leave anything behind in this one.
    Returns (set of model names that work on the GPU, status messages)."""
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    recv_end, send_end = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_probe_child, args=(send_end, hand_model, face_model), daemon=True)
    proc.start()
    send_end.close()
    try:
        if not recv_end.poll(timeout):
            return set(), [f"GPU test timed out after {timeout}s — using CPU"]
        return recv_end.recv()
    except EOFError:
        return set(), ["GPU test crashed — using CPU"]
    finally:
        proc.kill()
        proc.join()
