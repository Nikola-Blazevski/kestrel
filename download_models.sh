#!/bin/bash
# Download the MediaPipe person detection model doc-cam-pi uses.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p models
BASE=https://storage.googleapis.com/mediapipe-models

fetch() {
  if [ -s "models/$2" ]; then echo "  models/$2 already present"; return; fi
  echo "  downloading $2"
  wget -q --show-progress -O "models/$2.part" "$BASE/$1" && mv "models/$2.part" "models/$2"
}

fetch object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite efficientdet_lite0_int8.tflite
# More accurate for small or distant people, about 2.5x slower. To use it, uncomment
# this and set PERSON_MODEL_PATH in config.py:
# fetch object_detector/efficientdet_lite2/int8/1/efficientdet_lite2.tflite efficientdet_lite2_int8.tflite
ls -lh models
