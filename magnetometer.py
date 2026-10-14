#!/usr/bin/env python3
"""
GY-271 magnetometer support.

GY-271 boards ship with one of three chips:
  * HMC5883L  (Honeywell)  at I2C address 0x1E
  * QMC5883L  (QST clone)  at I2C address 0x0D
  * QMC5883P  (newer QST)  at I2C address 0x2C
All are auto-detected. Run `i2cdetect -y 1` to see which one you have.

Standalone use:
    python magnetometer.py              # print live readings
    python magnetometer.py --calibrate  # rotate the sensor, saves mag_cal.json
"""

import argparse
import json
import math
import os
import threading
import time
from dataclasses import dataclass, asdict

import config

CARDINALS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _s16(v):
    """Interpret a 16-bit value as signed."""
    return v - 0x10000 if v & 0x8000 else v


def cardinal(heading_deg):
    return CARDINALS[int((heading_deg % 360) / 22.5 + 0.5) % 16]


# =====================================================================
# Chip drivers. Each read() returns (raw_xyz_counts, xyz_uT, temp_c, overflow)
# =====================================================================

class HMC5883L:
    NAME = "HMC5883L"
    ADDRESS = 0x1E
    UT_PER_LSB = 100.0 / 1090.0     # gain 1.3 Ga -> 1090 LSB/gauss, 1 G = 100 uT

    def __init__(self, bus):
        self.bus = bus
        self.init()

    @classmethod
    def probe(cls, bus):
        try:
            return bytes(bus.read_i2c_block_data(cls.ADDRESS, 0x0A, 3)) == b"H43"
        except OSError:
            return False

    def init(self):
        a = self.ADDRESS
        self.bus.write_byte_data(a, 0x00, 0x70)  # CRA: 8-sample avg, 15 Hz, normal
        self.bus.write_byte_data(a, 0x01, 0x20)  # CRB: gain 1.3 Ga
        self.bus.write_byte_data(a, 0x02, 0x00)  # mode: continuous
        time.sleep(0.07)

    def read(self):
        d = self.bus.read_i2c_block_data(self.ADDRESS, 0x03, 6)
        # Register order on this chip is X, Z, Y (big-endian)
        x = _s16(d[0] << 8 | d[1])
        z = _s16(d[2] << 8 | d[3])
        y = _s16(d[4] << 8 | d[5])
        overflow = -4096 in (x, y, z)
        s = self.UT_PER_LSB
        return (x, y, z), (x * s, y * s, z * s), None, overflow


class QMC5883L:
    NAME = "QMC5883L"
    ADDRESS = 0x0D
    UT_PER_LSB = 100.0 / 3000.0     # 8 gauss range -> 3000 LSB/gauss

    def __init__(self, bus):
        self.bus = bus
        self.init()

    @classmethod
    def probe(cls, bus):
        try:
            return bus.read_byte_data(cls.ADDRESS, 0x0D) == 0xFF
        except OSError:
            return False

    def init(self):
        a = self.ADDRESS
        self.bus.write_byte_data(a, 0x0A, 0x80)  # soft reset
        time.sleep(0.01)
        self.bus.write_byte_data(a, 0x0B, 0x01)  # SET/RESET period (datasheet value)
        self.bus.write_byte_data(a, 0x09, 0x1D)  # OSR 512, 8 G, 50 Hz, continuous
        time.sleep(0.03)

    def read(self):
        status = self.bus.read_byte_data(self.ADDRESS, 0x06)
        d = self.bus.read_i2c_block_data(self.ADDRESS, 0x00, 6)
        x = _s16(d[1] << 8 | d[0])
        y = _s16(d[3] << 8 | d[2])
        z = _s16(d[5] << 8 | d[4])
        t = self.bus.read_i2c_block_data(self.ADDRESS, 0x07, 2)
        # Temperature sensor has an uncalibrated offset — only relative changes mean much.
        temp = _s16(t[1] << 8 | t[0]) / 100.0
        overflow = bool(status & 0x02)
        s = self.UT_PER_LSB
        return (x, y, z), (x * s, y * s, z * s), temp, overflow


class QMC5883P:
    """Newer QST chip on recent GY-271 boards. Different registers from the
    QMC5883L, and no temperature sensor."""
    NAME = "QMC5883P"
    ADDRESS = 0x2C
    UT_PER_LSB = 100.0 / 3750.0     # 8 gauss range -> 3750 LSB/gauss

    def __init__(self, bus):
        self.bus = bus
        self.init()

    @classmethod
    def probe(cls, bus):
        try:
            return bus.read_byte_data(cls.ADDRESS, 0x00) == 0x80      # chip ID
        except OSError:
            return False

    def init(self):
        a = self.ADDRESS
        # Configure from suspend mode. No soft reset: the chip drops writes for a
        # while after one, which silently leaves it on the default 30 G range.
        self.bus.write_byte_data(a, 0x0A, 0x00)  # suspend
        time.sleep(0.01)
        self.bus.write_byte_data(a, 0x0B, 0x08)  # set/reset on, 8 G range
        if self.bus.read_byte_data(a, 0x0B) != 0x08:
            raise OSError("range setting didn't stick")
        self.bus.write_byte_data(a, 0x0A, 0xC5)  # OSR 8 + 8, 50 Hz, normal mode
        time.sleep(0.03)
        self.read()                               # clears a stale overflow flag from before init

    def read(self):
        status = self.bus.read_byte_data(self.ADDRESS, 0x09)
        d = self.bus.read_i2c_block_data(self.ADDRESS, 0x01, 6)
        x = _s16(d[1] << 8 | d[0])
        y = _s16(d[3] << 8 | d[2])
        z = _s16(d[5] << 8 | d[4])
        overflow = bool(status & 0x02)
        s = self.UT_PER_LSB
        return (x, y, z), (x * s, y * s, z * s), None, overflow


class SimulatedMag:
    """Fake sensor that slowly rotates, for testing the overlay without hardware."""
    NAME = "SIMULATED"

    def __init__(self, bus=None):
        self.t0 = time.monotonic()

    def read(self):
        t = time.monotonic() - self.t0
        ang = math.radians((t * 15) % 360)          # 15 deg/s
        h, z = 18.0, -48.0                           # ~ northern-hemisphere field
        x, y = h * math.cos(ang), -h * math.sin(ang)
        x += 0.3 * math.sin(t * 7)
        return (0, 0, 0), (x, y, z), 21.0 + math.sin(t / 10), False


DRIVERS = [HMC5883L, QMC5883L, QMC5883P]


def detect_sensor(bus_num=config.I2C_BUS, bus=None):
    """Open the I2C bus and return an initialised driver, or (None, reason)."""
    if bus is None:
        try:
            from smbus2 import SMBus
        except ImportError:
            return None, "smbus2 not installed"
        try:
            bus = SMBus(bus_num)
        except (FileNotFoundError, PermissionError, OSError) as e:
            return None, f"/dev/i2c-{bus_num} unavailable ({e.__class__.__name__}); enable I2C"
    for drv in DRIVERS:
        if drv.probe(bus):
            try:
                return drv(bus), None
            except OSError as e:
                return None, f"{drv.NAME} found but init failed: {e}"
    return None, "no HMC5883L (0x1E), QMC5883L (0x0D) or QMC5883P (0x2C) on the bus"


# =====================================================================
# Calibration
# =====================================================================

class Calibration:
    """Hard-iron offset + simple per-axis soft-iron scale."""

    def __init__(self, offset=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), loaded=False):
        self.offset = list(offset)
        self.scale = list(scale)
        self.loaded = loaded

    def apply(self, xyz):
        return tuple((v - o) * s for v, o, s in zip(xyz, self.offset, self.scale))

    @classmethod
    def load(cls, path):
        try:
            with open(path) as f:
                d = json.load(f)
            return cls(d["offset"], d["scale"], loaded=True)
        except (OSError, KeyError, ValueError):
            return cls()

    @classmethod
    def from_minmax(cls, mins, maxs):
        offset = [(lo + hi) / 2 for lo, hi in zip(mins, maxs)]
        radii = [(hi - lo) / 2 for lo, hi in zip(mins, maxs)]
        avg = sum(radii) / 3
        scale = [avg / r if r > 1e-6 else 1.0 for r in radii]
        return cls(offset, scale, loaded=True)

    def save(self, path, sensor_name):
        with open(path, "w") as f:
            json.dump({"sensor": sensor_name, "offset": self.offset,
                       "scale": self.scale, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
                      f, indent=2)


# =====================================================================
# Background reader
# =====================================================================

@dataclass
class MagReading:
    sensor: str
    x: float               # uT, calibrated + smoothed
    y: float
    z: float
    magnitude: float       # |B| in uT
    heading: float         # degrees, 0 = north (+ declination)
    inclination: float     # dip angle in degrees (only meaningful when level)
    temp_c: float | None
    overflow: bool
    calibrated: bool
    age_s: float = 0.0     # seconds since the last successful read

    def to_dict(self):
        return asdict(self)


class MagReader:
    """Polls the sensor in a thread so a slow I2C read never stalls the video."""

    def __init__(self, sensor, calibration=None, rate_hz=config.MAG_RATE_HZ,
                 smoothing=config.MAG_SMOOTHING, declination=config.DECLINATION_DEG,
                 flip=False):
        self.sensor = sensor
        self.cal = calibration or Calibration()
        self.period = 1.0 / rate_hz
        self.alpha = max(0.01, min(1.0, smoothing))
        self.declination = declination
        self.flip = flip            # board mounted upside down -> heading runs backwards
        self._lock = threading.Lock()
        self._latest = None
        self._last_ok = 0.0
        self._filt = None
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def latest(self):
        with self._lock:
            r = self._latest
            if r is None:
                return None
            r.age_s = time.monotonic() - self._last_ok
            return r

    def _run(self):
        fails = 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                _, xyz, temp, overflow = self.sensor.read()
                fails = 0
                self.error = None
                self._update(self.cal.apply(xyz), temp, overflow)
            except OSError as e:
                fails += 1
                self.error = f"I2C error: {e}"
                if fails % 20 == 0 and hasattr(self.sensor, "init"):
                    try:                       # loose wire / brown-out: re-init
                        self.sensor.init()
                    except OSError:
                        pass
            self._stop.wait(max(0.0, self.period - (time.monotonic() - t0)))

    def _update(self, xyz, temp, overflow):
        if self._filt is None:
            self._filt = list(xyz)
        else:
            a = self.alpha
            self._filt = [a * v + (1 - a) * f for v, f in zip(xyz, self._filt)]
        x, y, z = self._filt
        heading = (math.degrees(math.atan2(-y if self.flip else y, x)) + self.declination) % 360
        horiz = math.hypot(x, y)
        reading = MagReading(
            sensor=self.sensor.NAME, x=x, y=y, z=z,
            magnitude=math.sqrt(x * x + y * y + z * z),
            heading=heading,
            inclination=math.degrees(math.atan2(z, horiz)),
            temp_c=temp, overflow=overflow, calibrated=self.cal.loaded)
        with self._lock:
            self._latest = reading
            self._last_ok = time.monotonic()


# =====================================================================
# CLI: live readout + calibration
# =====================================================================

def _calibrate(sensor, path, seconds):
    print(f"Calibrating {sensor.NAME} for {seconds}s.")
    print("Slowly rotate the sensor through every orientation (figure-8s, flip it over).")
    print("Keep it away from the Pi's speakers, motors and steel.\n")
    mins, maxs = [1e9] * 3, [-1e9] * 3
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            _, xyz, _, _ = sensor.read()
        except OSError:
            continue
        mins = [min(m, v) for m, v in zip(mins, xyz)]
        maxs = [max(m, v) for m, v in zip(maxs, xyz)]
        left = end - time.monotonic()
        print(f"\r  {left:4.0f}s  X[{mins[0]:7.1f},{maxs[0]:7.1f}]  "
              f"Y[{mins[1]:7.1f},{maxs[1]:7.1f}]  Z[{mins[2]:7.1f},{maxs[2]:7.1f}] uT",
              end="", flush=True)
        time.sleep(0.05)
    print()
    spans = [hi - lo for lo, hi in zip(mins, maxs)]
    if min(spans) < 20:
        print(f"Warning: small range on some axis ({min(spans):.1f} uT). "
              "Did you rotate it through all orientations?")
    cal = Calibration.from_minmax(mins, maxs)
    cal.save(path, sensor.NAME)
    print(f"Saved {path}: offset={['%.2f' % v for v in cal.offset]} "
          f"scale={['%.3f' % v for v in cal.scale]}")


def main():
    p = argparse.ArgumentParser(description="GY-271 readout / calibration")
    p.add_argument("--bus", type=int, default=config.I2C_BUS)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--seconds", type=int, default=30, help="calibration duration")
    p.add_argument("--cal", default=config.MAG_CAL_PATH, help="calibration file")
    p.add_argument("--declination", type=float, default=config.DECLINATION_DEG)
    p.add_argument("--flip", action="store_true", help="board mounted upside down")
    p.add_argument("--sim", action="store_true", help="use the simulated sensor")
    args = p.parse_args()

    if args.sim:
        sensor, err = SimulatedMag(), None
    else:
        sensor, err = detect_sensor(args.bus)
    if sensor is None:
        raise SystemExit(f"Magnetometer not found: {err}")
    print(f"Found {sensor.NAME}")

    if args.calibrate:
        _calibrate(sensor, args.cal, args.seconds)
        return

    reader = MagReader(sensor, Calibration.load(args.cal),
                       declination=args.declination, flip=args.flip).start()
    try:
        while True:
            r = reader.latest()
            if r:
                t = f" T~{r.temp_c:5.1f}C" if r.temp_c is not None else ""
                print(f"\rHDG {r.heading:6.1f} {cardinal(r.heading):>3}  "
                      f"X{r.x:7.1f} Y{r.y:7.1f} Z{r.z:7.1f}  |B| {r.magnitude:5.1f} uT"
                      f"  dip {r.inclination:5.1f}{t}"
                      f"{'  OVERFLOW' if r.overflow else ''}"
                      f"{'' if r.calibrated else '  (uncalibrated)'}   ",
                      end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
