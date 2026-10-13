#!/bin/bash
# One-shot setup for Raspberry Pi 4 running 64-bit Raspberry Pi OS (Bookworm or newer).
# MediaPipe has no 32-bit ARM wheels, so a 64-bit OS is required.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "WARNING: $(uname -m) detected. MediaPipe needs 64-bit Raspberry Pi OS (aarch64)."
  echo "         Detection will be unavailable; the magnetometer overlay still works."
fi

echo "== System packages =="
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip i2c-tools wget python3-picamera2 python3-smbus2 || \
  sudo apt-get install -y python3-venv python3-pip i2c-tools wget python3-picamera2

echo "== Enabling I2C =="
if command -v raspi-config >/dev/null; then
  sudo raspi-config nonint do_i2c 0
fi
if ! groups "$USER" | grep -qw i2c; then
  sudo usermod -aG i2c "$USER"
  echo "   Added $USER to the i2c group (log out/in or reboot for it to apply)."
fi

echo "== Python venv =="
# --system-site-packages so the apt-installed picamera2/libcamera bindings are visible
python3 -m venv --system-site-packages venv
venv/bin/pip install --upgrade pip wheel

# picamera2 from apt is built against the system numpy. Keep the venv on the same
# numpy major version or importing picamera2 fails with "numpy.dtype size changed".
SYS_NP=$(/usr/bin/python3 -c "import numpy; print(numpy.__version__.split('.')[0])" 2>/dev/null || echo none)
PINS=()
if [ "$SYS_NP" = "1" ]; then
  echo "   System numpy is 1.x -> pinning numpy<2 and an OpenCV that supports it"
  PINS=("numpy<2" "opencv-contrib-python<4.12")
fi
venv/bin/pip install -r requirements.txt "${PINS[@]}"

echo "== Models =="
./download_models.sh

echo "== Checks =="
venv/bin/python - <<'EOF'
def check(name, fn):
    try:
        fn(); print(f"  OK    {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
check("numpy / opencv", lambda: __import__("cv2"))
check("mediapipe", lambda: __import__("mediapipe"))
check("picamera2 (only needed for the CSI camera)", lambda: __import__("picamera2"))
check("smbus2", lambda: __import__("smbus2"))
EOF
echo
echo "I2C scan (GY-271 shows up as 1e = HMC5883L or 0d = QMC5883L):"
sudo i2cdetect -y 1 || true

cat <<'EOF'

Done. Next:
  source venv/bin/activate
  python magnetometer.py --calibrate   # once, rotate the sensor through all orientations
  python main.py --source picam        # or --source 0 for a USB webcam
  python main.py --headless --stream   # view at http://<pi-ip>:8000
EOF
