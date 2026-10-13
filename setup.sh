#!/bin/bash
# One-shot setup.
#
#   ./setup.sh            # same as ./setup.sh client
#   ./setup.sh client     # Raspberry Pi 4, 64-bit Raspberry Pi OS (Bookworm or newer)
#   ./setup.sh server     # GPU detection server (Arch Linux; Debian/Ubuntu also work)
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

ROLE="${1:-client}"
case "$ROLE" in
  client|server) ;;
  *) echo "usage: $0 [client|server]"; exit 1 ;;
esac

check_imports() {
  venv/bin/python - "$@" <<'EOF'
import sys
for spec in sys.argv[1:]:
    mod, _, label = spec.partition("=")
    try:
        __import__(mod); print(f"  OK    {label or mod}")
    except Exception as e:
        print(f"  FAIL  {label or mod}: {e}")
EOF
}

# ---------------------------------------------------------------------
# Server: runs server.py, which does the MediaPipe work for the Pi
# ---------------------------------------------------------------------
setup_server() {
  echo "== System packages =="
  if command -v pacman >/dev/null; then
    # libglvnd provides libEGL/libGLESv2 for MediaPipe's GPU delegate
    PKGS=(python wget libglvnd)
    MISSING=()
    for p in "${PKGS[@]}"; do pacman -Q "$p" >/dev/null 2>&1 || MISSING+=("$p"); done
    if [ ${#MISSING[@]} -gt 0 ]; then
      sudo pacman -S --needed --noconfirm "${MISSING[@]}"
    else
      echo "   ${PKGS[*]} already installed"
    fi
  elif command -v apt-get >/dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3-venv python3-pip wget libegl1 libgles2
  else
    echo "   Unknown package manager. Install Python 3, venv, wget and EGL/GLES libraries yourself."
  fi

  echo "== Python venv =="
  python3 -m venv venv
  venv/bin/pip install --upgrade pip wheel
  venv/bin/pip install -r requirements-server.txt

  echo "== Models =="
  ./download_models.sh

  echo "== Checks =="
  check_imports "cv2=numpy / opencv" mediapipe
  # Same code path as server.py: shows which models get the GPU on this machine.
  # MediaPipe's native logging is noisy, so keep only our own lines.
  venv/bin/python - 2>/dev/null <<'EOF' | grep -E "^  " || echo "  FAIL  detector check crashed (run python server.py to see why)"
import time
import numpy as np
import config
from detectors import Detector
d = Detector(config.HAND_MODEL_PATH, config.FACE_MODEL_PATH, gpu=True)
for s in d.status:
    print(f"  NOTE  {s}")
if d.enabled:
    frame = np.zeros((360, 640, 3), np.uint8)
    d.detect(frame)
    t = time.perf_counter()
    for _ in range(20):
        d.detect(frame)
    print(f"  OK    detection runs ({d.delegate}), "
          f"{(time.perf_counter() - t) / 20 * 1000:.1f} ms per 640x360 frame")
else:
    print("  FAIL  detector did not load")
d.close()
EOF

  PORT=$(venv/bin/python -c "from protocol import DEFAULT_PORT; print(DEFAULT_PORT)")
  IP=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<NF;i++) if($i=="src") print $(i+1)}')
  cat <<EOF

Done. Next:
  source venv/bin/activate
  python server.py                         # listens on 0.0.0.0:$PORT
On the Pi:
  python main.py --source picam --server ${IP:-<this-machine-ip>}

If a firewall is running, allow TCP $PORT from the Pi, e.g.
  sudo ufw allow from <pi-ip> to any port $PORT proto tcp
EOF
}

# ---------------------------------------------------------------------
# Client: the Raspberry Pi with the camera and GY-271
# MediaPipe has no 32-bit ARM wheels, so a 64-bit OS is required for local
# detection. With --server it isn't needed on the Pi at all.
# ---------------------------------------------------------------------
setup_client() {
  if [ "$(uname -m)" != "aarch64" ]; then
    echo "WARNING: $(uname -m) detected. MediaPipe needs 64-bit Raspberry Pi OS (aarch64)."
    echo "         Local detection will be unavailable; --server and the magnetometer overlay still work."
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
  check_imports "cv2=numpy / opencv" "mediapipe=mediapipe (not needed with --server)" \
    "picamera2=picamera2 (only needed for the CSI camera)" smbus2
  echo
  echo "I2C scan (GY-271 shows up as 1e = HMC5883L or 0d = QMC5883L):"
  sudo i2cdetect -y 1 || true

  cat <<'EOF'

Done. Next:
  source venv/bin/activate
  python magnetometer.py --calibrate          # once, rotate the sensor through all orientations
  python main.py --source picam               # or --source 0 for a USB webcam
  python main.py --headless --stream          # view at http://<pi-ip>:8000
  python main.py --source picam --server <gpu-machine-ip>   # detection on the server
EOF
}

"setup_$ROLE"
