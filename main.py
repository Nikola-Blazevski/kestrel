#!/usr/bin/env python3
"""
doc-cam-pi — hand & face detection with a GY-271 magnetometer overlay,
stripped down to run on a Raspberry Pi 4.

Examples:
    python main.py                       # USB webcam 0, local window if a desktop is running
    python main.py --source picam        # Pi camera module (CSI)
    python main.py --stream              # also serve http://<pi-ip>:8000
    python main.py --headless --stream   # no window, browser only (SSH / no desktop)
    python main.py --no-hands --no-faces # magnetometer overlay only (lowest CPU)
    python main.py --mag-sim             # test overlay without the sensor wired

Keys (window mode): q / Esc quit, h toggle hands, f toggle faces, m toggle mag panel.
"""

import os
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import signal
import time

import cv2 as cv

import config
from camera import open_camera
from detectors import Detector, Detections
from magnetometer import Calibration, MagReader, SimulatedMag, detect_sensor
from overlay import draw_faces, draw_hands, draw_hud, draw_mag, read_cpu_temp


def parse_args():
    p = argparse.ArgumentParser(description="doc-cam for Raspberry Pi 4 + GY-271 overlay")
    p.add_argument("--source", default="0", help="'picam', USB index (0), or video file")
    p.add_argument("--width", type=int, default=config.FRAME_WIDTH)
    p.add_argument("--height", type=int, default=config.FRAME_HEIGHT)
    p.add_argument("--fps", type=int, default=config.TARGET_FPS)

    g = p.add_argument_group("detection")
    g.add_argument("--no-hands", action="store_true")
    g.add_argument("--no-faces", action="store_true")
    g.add_argument("--detect-width", type=int, default=config.DETECT_WIDTH)
    g.add_argument("--detect-every", type=int, default=config.DETECT_EVERY)
    g.add_argument("--num-hands", type=int, default=config.NUM_HANDS)

    g = p.add_argument_group("magnetometer")
    g.add_argument("--no-mag", action="store_true")
    g.add_argument("--mag-sim", action="store_true", help="simulated sensor for testing")
    g.add_argument("--i2c-bus", type=int, default=config.I2C_BUS)
    g.add_argument("--mag-cal", default=config.MAG_CAL_PATH)
    g.add_argument("--declination", type=float, default=config.DECLINATION_DEG,
                   help="degrees, east positive; non-zero shows true heading")
    g.add_argument("--mag-flip", action="store_true",
                   help="heading turns the wrong way (board mounted upside down)")

    g = p.add_argument_group("output")
    g.add_argument("--headless", action="store_true", help="never open a window")
    g.add_argument("--stream", action="store_true", help="serve MJPEG over HTTP")
    g.add_argument("--port", type=int, default=config.STREAM_PORT)
    g.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0)
    g.add_argument("--flip", action="store_true", help="mirror horizontally")
    return p.parse_args()


ROTATIONS = {90: cv.ROTATE_90_CLOCKWISE, 180: cv.ROTATE_180, 270: cv.ROTATE_90_COUNTERCLOCKWISE}


def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)       # model and calibration paths are relative to the project

    show_window = not args.headless and bool(os.environ.get("DISPLAY") or
                                             os.environ.get("WAYLAND_DISPLAY"))
    if not show_window and not args.stream:
        print("No display found; enabling --stream so you can see the feed.")
        args.stream = True

    # --- Magnetometer ---
    mag, mag_status = None, "disabled"
    if not args.no_mag:
        sensor, err = (SimulatedMag(), None) if args.mag_sim else detect_sensor(args.i2c_bus)
        if sensor:
            cal = Calibration.load(args.mag_cal)
            mag = MagReader(sensor, cal, declination=args.declination,
                            flip=args.mag_flip).start()
            print(f"Magnetometer: {sensor.NAME}"
                  f"{' (calibrated)' if cal.loaded else ' — uncalibrated, run: python magnetometer.py --calibrate'}")
        else:
            mag_status = err
            print(f"Magnetometer: not found ({err})")

    # --- Detection ---
    detector = Detector(
        hand_model=None if args.no_hands else config.HAND_MODEL_PATH,
        face_model=None if args.no_faces else config.FACE_MODEL_PATH,
        num_hands=args.num_hands)
    for s in detector.status:
        print(f"Detection: {s}")
    want_hands, want_faces = detector.hands is not None, detector.faces is not None

    # --- Camera ---
    print(f"Opening camera '{args.source}' at {args.width}x{args.height}@{args.fps}...")
    try:
        cam = open_camera(args.source, args.width, args.height, args.fps)
    except RuntimeError as e:
        if mag:
            mag.stop()
        detector.close()
        raise SystemExit(f"Camera error: {e}")

    # --- Outputs ---
    hub = None
    if args.stream:
        from stream import StreamHub, start_server
        hub = StreamHub(config.JPEG_QUALITY)
        start_server(hub, args.port)
        print(f"Streaming on http://0.0.0.0:{args.port}/  (use the Pi's IP address)")
    title = "doc-cam-pi"
    if show_window:
        cv.namedWindow(title, cv.WINDOW_NORMAL)

    stop = False

    def _sig(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, _sig)

    show_mag = True
    last_id, n = 0, 0
    dets = Detections()
    fps, fps_n, fps_t = 0.0, 0, time.monotonic()
    cpu_temp, temp_t = read_cpu_temp(), 0.0

    try:
        while not stop:
            last_id, frame = cam.read(last_id, timeout=2.0)
            if frame is None:
                if cam.failed:
                    raise SystemExit(f"Camera error: {cam.failed}")
                print("Waiting for camera frames...")
                continue
            if args.rotate:
                frame = cv.rotate(frame, ROTATIONS[args.rotate])
            if args.flip:
                frame = cv.flip(frame, 1)
            else:
                frame = frame.copy()        # never draw into the capture buffer
            n += 1

            # Detection on a downscaled copy, every N frames
            if detector.enabled and n % max(1, args.detect_every) == 0:
                h, w = frame.shape[:2]
                dw = min(args.detect_width, w)
                small = cv.resize(frame, (dw, int(h * dw / w)), interpolation=cv.INTER_AREA) \
                    if dw < w else frame
                dets = detector.detect(small)

            if want_hands:
                draw_hands(frame, dets.hands)
            if want_faces:
                draw_faces(frame, dets.faces)

            reading = mag.latest() if mag else None
            if show_mag and not args.no_mag:
                draw_mag(frame, reading, mag.error if mag else mag_status, args.declination)

            # Stats
            fps_n += 1
            now = time.monotonic()
            if now - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now
            if now - temp_t > 2.0:
                cpu_temp, temp_t = read_cpu_temp(), now
            notes = [] if detector.enabled else ["detection off"]
            draw_hud(frame, fps, dets.ms, len(dets.hands), len(dets.faces), cpu_temp, notes)

            if hub:
                hub.publish(frame, {"fps": fps,
                                    "mag": reading.to_dict() if reading else None,
                                    "hands": len(dets.hands), "faces": len(dets.faces)})
            if show_window:
                cv.imshow(title, frame)
                key = cv.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("h") and detector.hands is not None:
                    want_hands = not want_hands
                elif key == ord("f") and detector.faces is not None:
                    want_faces = not want_faces
                elif key == ord("m"):
                    show_mag = not show_mag
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        cam.stop()
        if mag:
            mag.stop()
        detector.close()
        if show_window:
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
