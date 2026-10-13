"""Tunable settings for doc-cam-pi. Most of these can also be set from the CLI."""

# --- Camera ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# --- Detection (the expensive part on a Pi 4) ---
# Detection runs on a downscaled copy of the frame; overlays are drawn on the
# full-size frame. 320 px wide is a good balance on a Pi 4.
DETECT_WIDTH = 320
# Run detection every N frames and reuse the last result in between.
DETECT_EVERY = 2
NUM_HANDS = 2
HAND_MODEL_PATH = "models/hand_landmarker.task"
FACE_MODEL_PATH = "models/blaze_face_short_range.tflite"

# --- GY-271 magnetometer ---
I2C_BUS = 1                 # /dev/i2c-1 = GPIO2 (SDA) / GPIO3 (SCL)
MAG_RATE_HZ = 20            # how often the background thread polls the sensor
MAG_SMOOTHING = 0.25        # EMA factor, 0 < a <= 1 (1 = no smoothing)
MAG_CAL_PATH = "mag_cal.json"
# Magnetic declination in degrees (east positive). 0 = show magnetic heading.
DECLINATION_DEG = 0.0
# Earth's field is ~25-65 uT. Outside this the reading is probably disturbed.
EARTH_FIELD_MIN_UT = 25.0
EARTH_FIELD_MAX_UT = 65.0

# --- MJPEG web stream ---
STREAM_PORT = 8000
JPEG_QUALITY = 75

# --- Hand skeleton ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# (name, tip landmark, BGR colour, joints)
FINGERS = [
    ("Thumb",   4,  (0, 200, 255), [1, 2, 3, 4]),
    ("Pointer", 8,  (255, 0, 0),   [5, 6, 7, 8]),
    ("Middle",  12, (0, 0, 255),   [9, 10, 11, 12]),
]
