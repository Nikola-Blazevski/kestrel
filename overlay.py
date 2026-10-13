"""Everything drawn on top of the video: hands, faces, HUD and the GY-271 panel."""

import math

import cv2 as cv
import numpy as np

import config
from magnetometer import cardinal

FONT = cv.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
GREEN = (90, 220, 90)
AMBER = (0, 190, 255)
RED = (60, 60, 255)
CYAN = (230, 200, 0)


def _text(img, s, org, scale=0.5, color=WHITE, thick=1, bg=False):
    """Draw text; bg=True puts a solid box behind it (for labels over the video)."""
    if bg:
        (tw, th), base = cv.getTextSize(s, FONT, scale, thick)
        x, y = org
        cv.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base), (0, 0, 0), -1)
    cv.putText(img, s, org, FONT, scale, color, thick, cv.LINE_AA)


def _panel(img, x0, y0, x1, y1, alpha=0.55):
    """Darken a rectangle so text stays readable on any background."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha)).astype(np.uint8)


def _degree(img, x, y, color, scale=0.5):
    """OpenCV's Hershey fonts have no degree sign, so draw one."""
    r = max(2, int(3 * scale / 0.5))
    cv.circle(img, (x + r + 1, y - int(10 * scale / 0.5)), r, color, 1, cv.LINE_AA)


# ---------------------------------------------------------------------
# Hands & faces
# ---------------------------------------------------------------------

def draw_hands(img, hands):
    h, w = img.shape[:2]
    for hand in hands:
        pts = [(int(x * w), int(y * h)) for x, y in hand.landmarks]
        for a, b in config.HAND_CONNECTIONS:
            cv.line(img, pts[a], pts[b], GREY, 1, cv.LINE_AA)
        for p in pts:
            cv.circle(img, p, 2, GREY, -1)
        for name, tip, color, joints in config.FINGERS:
            for a, b in zip(joints, joints[1:]):
                cv.line(img, pts[a], pts[b], color, 3, cv.LINE_AA)
            for j in joints:
                cv.circle(img, pts[j], 4, color, -1)
            tx, ty = pts[tip]
            _text(img, name, (tx + 4, ty - 6), 0.45, color, bg=True)
        xs, ys = zip(*pts)
        x0, y0, x1, y1 = min(xs) - 15, min(ys) - 15, max(xs) + 15, max(ys) + 15
        cv.rectangle(img, (x0, y0), (x1, y1), GREEN, 2)
        _text(img, f"{hand.label} {hand.score:.0%}", (x0, max(14, y0 - 6)), 0.55, GREEN, bg=True)


def draw_faces(img, faces):
    h, w = img.shape[:2]
    for f in faces:
        x, y, bw, bh = f.box
        p0 = (int(x * w), int(y * h))
        p1 = (int((x + bw) * w), int((y + bh) * h))
        cv.rectangle(img, p0, p1, CYAN, 2)
        _text(img, f"Face {f.score:.0%}", (p0[0], max(14, p0[1] - 6)), 0.5, CYAN, bg=True)


# ---------------------------------------------------------------------
# HUD (top-left)
# ---------------------------------------------------------------------

def read_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def draw_hud(img, fps, detect_ms, n_hands, n_faces, cpu_temp, notes=()):
    lines = [(f"FPS {fps:4.1f}   det {detect_ms:4.0f} ms", WHITE),
             (f"Hands {n_hands}   Faces {n_faces}", WHITE)]
    if cpu_temp is not None:
        col = GREEN if cpu_temp < 70 else AMBER if cpu_temp < 80 else RED
        lines.append((f"CPU {cpu_temp:4.1f} C", col))
    for n in notes:
        lines.append((n, AMBER))
    width = max(cv.getTextSize(s, FONT, 0.5, 1)[0][0] for s, _ in lines) + 16
    _panel(img, 4, 4, 4 + width, 12 + 20 * len(lines))
    for i, (s, col) in enumerate(lines):
        _text(img, s, (12, 22 + 20 * i), 0.5, col)


# ---------------------------------------------------------------------
# GY-271 compass + data panel
# ---------------------------------------------------------------------

def _draw_compass(img, cx, cy, r, heading, color):
    """Rotating compass card: the fixed tick at the top is the camera/sensor +X axis."""
    cv.circle(img, (cx, cy), r + 6, (0, 0, 0), -1, cv.LINE_AA)
    cv.circle(img, (cx, cy), r, GREY, 1, cv.LINE_AA)
    for deg in range(0, 360, 30):
        a = math.radians(deg - heading - 90)     # card rotates opposite to heading
        inner = r - (10 if deg % 90 == 0 else 6)
        cv.line(img, (int(cx + inner * math.cos(a)), int(cy + inner * math.sin(a))),
                (int(cx + r * math.cos(a)), int(cy + r * math.sin(a))), GREY, 1, cv.LINE_AA)
    for label, deg in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
        a = math.radians(deg - heading - 90)
        lx = int(cx + (r - 20) * math.cos(a))
        ly = int(cy + (r - 20) * math.sin(a))
        (tw, th), _ = cv.getTextSize(label, FONT, 0.45, 1)
        col = RED if label == "N" else WHITE
        cv.putText(img, label, (lx - tw // 2, ly + th // 2), FONT, 0.45, col,
                   2 if label == "N" else 1, cv.LINE_AA)
    # North needle
    a = math.radians(-heading - 90)
    tip = (int(cx + (r - 30) * math.cos(a)), int(cy + (r - 30) * math.sin(a)))
    tail = (int(cx - (r - 30) * math.cos(a)), int(cy - (r - 30) * math.sin(a)))
    cv.line(img, (cx, cy), tail, GREY, 2, cv.LINE_AA)
    cv.line(img, (cx, cy), tip, RED, 3, cv.LINE_AA)
    cv.circle(img, (cx, cy), 3, WHITE, -1, cv.LINE_AA)
    # Lubber line (forward)
    cv.fillConvexPoly(img, np.array([(cx, cy - r - 4), (cx - 6, cy - r - 14),
                                     (cx + 6, cy - r - 14)]), color, cv.LINE_AA)


def draw_mag(img, reading, status=None, declination=0.0):
    """Compass rose top-right, numeric readout bottom-right."""
    h, w = img.shape[:2]
    r = max(36, min(60, h // 9))
    cx, cy = w - r - 16, r + 24

    if reading is None:
        msg = f"MAG: {status or 'waiting for data'}"
        (tw, _), _ = cv.getTextSize(msg, FONT, 0.5, 1)
        _panel(img, w - tw - 20, h - 34, w - 4, h - 6)
        _text(img, msg, (w - tw - 12, h - 14), 0.5, AMBER)
        return

    stale = reading.age_s > 1.0
    field_ok = config.EARTH_FIELD_MIN_UT <= reading.magnitude <= config.EARTH_FIELD_MAX_UT
    accent = RED if (stale or reading.overflow) else GREEN if field_ok else AMBER
    _draw_compass(img, cx, cy, r, reading.heading, accent)

    ref = "T" if declination else "M"            # true vs magnetic
    hdg = f"HDG {reading.heading:5.1f}"
    lines = [
        (f"{reading.sensor}{'' if reading.calibrated else '  (uncal)'}", GREY),
        (None, accent),                                           # heading line drawn below
        (f"X {reading.x:7.1f} uT", WHITE),
        (f"Y {reading.y:7.1f} uT", WHITE),
        (f"Z {reading.z:7.1f} uT", WHITE),
        (f"|B| {reading.magnitude:5.1f} uT", accent),
        (f"Dip {reading.inclination:5.1f}", WHITE),
    ]
    if reading.temp_c is not None:
        lines.append((f"Chip T ~{reading.temp_c:4.1f} C", GREY))
    if stale:
        lines.append((f"STALE {reading.age_s:.0f}s {status or ''}".strip(), RED))
    elif reading.overflow:
        lines.append(("OVERFLOW - magnet nearby?", RED))
    elif not field_ok:
        lines.append(("Field disturbed", AMBER))

    line_h = 19
    pw = 168
    x0 = w - pw - 8
    y0 = h - 10 - line_h * len(lines) - 6
    _panel(img, x0, y0, w - 4, h - 4)
    for i, (s, col) in enumerate(lines):
        y = y0 + 18 + i * line_h
        if s is None:
            _text(img, hdg, (x0 + 8, y), 0.55, col, 1)
            tw = cv.getTextSize(hdg, FONT, 0.55, 1)[0][0]
            _degree(img, x0 + 8 + tw, y, col, 0.55)
            _text(img, f" {ref} {cardinal(reading.heading)}", (x0 + 18 + tw, y), 0.55, col)
        else:
            _text(img, s, (x0 + 8, y), 0.48, col)
            if s.startswith("Dip"):
                tw = cv.getTextSize(s, FONT, 0.48, 1)[0][0]
                _degree(img, x0 + 8 + tw, y, col, 0.48)
