#!/usr/bin/env python3
"""
Detection server: run this on a machine with a GPU and point the Pi at it with
`python main.py --server <host>`. The Pi sends JPEG frames, and this sends back
hand landmarks and face boxes.

    python server.py                    # listen on 0.0.0.0:8765, GPU delegate
    python server.py --port 9000 --cpu  # no GPU (for testing)

The connection is not authenticated or encrypted. Only expose it on a network you trust.
"""

import os
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import socket
import sys
import threading

import cv2 as cv
import numpy as np

import config
from detectors import Detector
from protocol import DEFAULT_PORT, detections_to_dict, recv_json, recv_msg, send_json, tune


def serve_client(conn, addr, args):
    name = f"{addr[0]}:{addr[1]}"
    detector = None
    try:
        tune(conn)
        hello = recv_json(conn)
        # One detector per client: MediaPipe VIDEO mode tracks state per stream
        detector = Detector(
            hand_model=config.HAND_MODEL_PATH if hello.get("hands", True) else None,
            face_model=config.FACE_MODEL_PATH if hello.get("faces", True) else None,
            num_hands=int(hello.get("num_hands", config.NUM_HANDS)),
            gpu=not args.cpu)
        send_json(conn, {"ok": True, "hands": detector.hands is not None,
                         "faces": detector.faces is not None,
                         "delegate": getattr(detector, "delegate", None),
                         "status": detector.status})
        print(f"[{name}] connected — {getattr(detector, 'delegate', None) or 'no detectors'}")
        for s in detector.status:
            print(f"[{name}] {s}")

        while True:
            jpeg = recv_msg(conn)
            frame = cv.imdecode(np.frombuffer(jpeg, np.uint8), cv.IMREAD_COLOR)
            if frame is None:
                send_json(conn, {"hands": [], "faces": [], "ms": 0.0, "error": "bad jpeg"})
                continue
            send_json(conn, detections_to_dict(detector.detect(frame)))
    except (ConnectionError, OSError) as e:
        print(f"[{name}] disconnected ({e})")
    except Exception as e:
        print(f"[{name}] error: {e!r}")
    finally:
        if detector:
            detector.close()
        conn.close()


def main():
    p = argparse.ArgumentParser(description="doc-cam-pi GPU detection server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--cpu", action="store_true", help="don't try the GPU delegate")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)     # show logs promptly under systemd / in files
    os.chdir(os.path.dirname(os.path.abspath(__file__)))    # model paths are relative

    for path in (config.HAND_MODEL_PATH, config.FACE_MODEL_PATH):
        if not os.path.exists(path):
            print(f"Warning: {path} missing — run download_models.sh")

    srv = socket.create_server((args.host, args.port), reuse_port=False)
    print(f"Detection server listening on {args.host}:{args.port}")
    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=serve_client, args=(conn, addr, args), daemon=True).start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
