# doc-cam-pi

A stripped-down version of [doc-cam](https://github.com/Nikola-Blazevski/doc-cam) for the Raspberry Pi 4. It draws a box around each person in view and adds a live overlay of the GY-271 magnetometer on the video feed. It finds people but doesn't tell them apart: there are no identities or UUIDs.

## Changes from the desktop version

| Removed | Why |
|---|---|
| Face identity / UUIDs (dlib + face_recognition) | Takes minutes to build on a Pi and runs at about 1 fps |
| Face snapshots, `known_faces.pkl`, stats, dashboard UI | Only existed to serve face identity |
| Face detection, pose landmarker, hand and finger tracking | Replaced by whole-body person detection |
| RGB skin / IR processing and the side-by-side "processed" view | Halves the work per frame |
| CUDA / GPU paths | The Pi 4 has no CUDA, and MediaPipe's GPU delegate isn't usable there |

| Added | |
|---|---|
| GY-271 driver | Auto-detects HMC5883L, QMC5883L or QMC5883P |
| Overlay | Compass rose plus heading, X/Y/Z, field strength, dip angle and chip temperature |
| Calibration | `python magnetometer.py --calibrate` |
| Pi camera support | CSI camera via picamera2 (USB webcams still work) |
| MJPEG web stream | Run headless over SSH and watch in a browser |
| Faster detection | Runs on a downscaled frame every N frames; capture thread always delivers the newest frame |
| CPU temperature in the HUD | The Pi 4 throttles at 80 °C |

## Wiring the GY-271

| GY-271 | Pi 4 pin |
|---|---|
| VCC | Pin 1 (3.3 V) — **not 5 V** |
| GND | Pin 6 |
| SCL | Pin 5 (GPIO3) |
| SDA | Pin 3 (GPIO2) |
| DRDY | not connected |

Mount the sensor flat, with its **X arrow pointing the same way as the camera**. The heading shown is the direction the X arrow points. Keep the sensor a few cm from the Pi, the camera ribbon, and any speakers or motors.

## Setup

You need 64-bit Raspberry Pi OS. MediaPipe has no 32-bit ARM builds.

```bash
chmod +x setup.sh download_models.sh
./setup.sh                # same as ./setup.sh client
source venv/bin/activate
```

The setup script does the following:
- installs `picamera2` and `i2c-tools`
- enables I2C
- creates a venv
- installs MediaPipe and OpenCV
- downloads the person detection model
- runs a quick import check and an `i2cdetect` scan

The GY-271 should appear at **1e** (HMC5883L), **0d** (QMC5883L) or **2c** (QMC5883P).

## Calibrate the magnetometer (once)

```bash
python magnetometer.py --calibrate
```

Over 30 seconds, slowly rotate the sensor through every orientation. This writes `mag_cal.json`. Without it, the heading can be off by tens of degrees, and the overlay shows `(uncal)`. Calibrate with the sensor mounted in its final position, because nearby metal moves with it.

To print raw readings in the terminal, run `python magnetometer.py`.

## Run

```bash
python main.py --source picam              # Pi camera module
python main.py --source 0                  # USB webcam
python main.py --headless --stream         # no desktop: open http://<pi-ip>:8000
python main.py --no-people                 # magnetometer overlay only, lowest CPU
python main.py --mag-sim                   # test the overlay without the sensor
python main.py --declination 15            # show true heading instead of magnetic
```

Window keys: **q/Esc** quit, **p** toggles people, **m** toggles the magnetometer panel.

`http://<pi-ip>:8000/data.json` returns the latest readings as JSON, if you want to log them from another machine.

## Offload detection to a GPU machine

Person detection can run on another computer instead of the Pi. The Pi still captures frames, reads the magnetometer and draws the overlay. It sends each frame to the server as a JPEG, and the server sends back a box for each person.

On the GPU machine (Arch Linux; Debian/Ubuntu also work), clone this repo, then run:

```bash
./setup.sh server
source venv/bin/activate
python server.py                 # listens on 0.0.0.0:8765
```

`setup.sh server` installs the system packages it needs with pacman (or apt), creates a venv from `requirements-server.txt`, and downloads the models. It then runs a short detection test and reports which models use the GPU and how fast they run.

To start the server on boot, run `./setup.sh service`. This installs `server.py` as a systemd user service and turns on lingering, so it starts at boot without anyone logging in. Useful commands:

```bash
journalctl --user-unit kestrel-server -f        # logs
systemctl --user restart kestrel-server         # after pulling new code
systemctl --user disable --now kestrel-server   # stop starting it on boot
```

On the Pi, run `./setup.sh` (see [Setup](#setup)), then:

```bash
python main.py --source picam --server 192.168.1.50              # default port 8765
python main.py --source picam --server 192.168.1.50:9000 --detect-every 1 --detect-width 640
```

- The server tries MediaPipe's GPU delegate first. This uses WebGPU/Vulkan or OpenGL ES, not CUDA, so the machine needs working Vulkan or GL drivers. At startup, the server tests the person detector on the GPU in a separate, short-lived process. If that fails, it runs on CPU and the server logs why. On an RTX 4060 laptop, the detector fails on the GPU, so it runs on CPU, which takes about 15 ms per frame. Pass `--cpu` to skip the GPU attempt.
- Networking never blocks the camera loop. Frames that arrive while a request is in flight are dropped, so overlays may lag the video by one round trip.
- In this mode, `det` in the HUD shows the full round-trip time, not just the GPU time.
- If the server goes away, the HUD shows `server offline`, the boxes disappear after a second, and the Pi keeps reconnecting in the background.
- Because the Pi no longer does detection, you can use `--detect-every 1` and a larger `--detect-width`. The limits then come from your network bandwidth and the Pi's JPEG encoding.
- The connection is **unauthenticated and unencrypted**, so only run it on a network you trust.

### Useful flags

| Flag | Default | Notes |
|---|---|---|
| `--width/--height/--fps` | 640×480 @ 30 | |
| `--detect-width` | 320 | Lower is faster but misses small or distant people |
| `--detect-every` | 2 | Run detection every N frames |
| `--server HOST[:PORT]` | off | Run detection on `server.py` elsewhere (default port 8765) |
| `--declination` | 0 | East positive. Look yours up at NOAA's magnetic field calculator |
| `--mag-flip` | off | Use if the heading turns the wrong way (board upside down) |
| `--rotate 90/180/270`, `--flip` | | Camera orientation |
| `--i2c-bus` | 1 | |

## Reading the overlay

- **HDG**: heading of the sensor's X axis. `M` means magnetic; `T` means true (when a declination is set).
- **X / Y / Z**: field components in µT, after calibration and smoothing.
- **|B|**: total field strength. Earth's field is about 25–65 µT. The value turns amber outside that range, which usually means steel or a magnet is nearby.
- **Dip**: inclination angle. Only meaningful when the board is level.
- **Chip T**: QMC5883L only. Its offset isn't calibrated, so treat it as a relative reading.
- The compass card rotates. The green marker at the top is "forward", and the red needle points north.
- Heading is only accurate when the sensor is level. There is no tilt compensation without an accelerometer.

## Performance notes (Pi 4)

- With `--server`, the Pi only captures frames, draws the overlay and encodes JPEGs. If FPS is low, try `--detect-every 3` or `--detect-width 256` to send less per frame.
- Fit a heatsink or fan. Sustained MediaPipe load will push a bare Pi 4 into throttling, and the HUD turns red at 80 °C.
- Use `--headless --stream` when you can. Drawing a desktop window over VNC costs noticeable CPU.

## Troubleshooting

- **`MAG: no HMC5883L ... on the bus`**: check the wiring, then run `sudo i2cdetect -y 1`. If nothing shows, I2C is disabled or SDA and SCL are swapped.
- **Permission denied on `/dev/i2c-1`**: log out and back in (or reboot) after setup, so the `i2c` group membership applies.
- **`FATAL ERROR: This binary was compiled with aes enabled` / `Illegal instruction`**: MediaPipe 1.x needs the ARM AES instructions, and the Pi 4's CPU doesn't have them. `main.py` now checks for this and turns local detection off instead of crashing. Run detection on a GPU machine with `--server` instead.
- **`No module named 'cv2'`**: setup didn't finish, so there is no `venv/`. Run `./setup.sh` again. `main.py` and `server.py` switch to `venv/` on their own, so you don't need to activate it first.
- **`numpy.dtype size changed` when using `--source picam`**: the venv's numpy doesn't match the system one. Run `venv/bin/pip install "numpy<2" "opencv-contrib-python<4.12"` on Bookworm.

## Run on boot (optional)

On the Pi:

```bash
./setup.sh service client --server 192.168.1.50      # add any other main.py flags at the end
```

This installs `main.py --source picam --headless --stream` plus your flags as a systemd user service (`kestrel-client`). It also turns on lingering, so the service starts at boot without anyone logging in. If the server isn't up yet when the Pi boots, the app keeps retrying until it is.

```bash
journalctl --user-unit kestrel-client -f        # logs
systemctl --user restart kestrel-client         # after pulling new code
systemctl --user disable --now kestrel-client   # stop starting it on boot
```

To change the flags, run `./setup.sh service client ...` again with the new ones.

To start the GPU server on boot, see [Offload detection to a GPU machine](#offload-detection-to-a-gpu-machine).
