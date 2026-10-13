#!/bin/bash
# Download the two MediaPipe models doc-cam-pi uses.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p models
BASE=https://storage.googleapis.com/mediapipe-models

fetch() {
  if [ -s "models/$2" ]; then echo "  models/$2 already present"; return; fi
  echo "  downloading $2"
  wget -q --show-progress -O "models/$2.part" "$BASE/$1" && mv "models/$2.part" "models/$2"
}

fetch hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task hand_landmarker.task
fetch face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite blaze_face_short_range.tflite
ls -lh models
