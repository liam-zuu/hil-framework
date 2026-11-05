"""
t4_imu_profile.py — IMU values the RPi5 sends for S06 T4 (4) (spec S06 v3.4, Z4, L7-1, L7-6).

(i) static levels, 2 s each: gz in {-5, -0.5, +0.5, +5} rad/s; ax, ay in {-2, +2} m/s^2
    (one axis at a time, the other 0); az = 9.81 m/s^2; yaw in {-2, 0, +2} rad;
(ii) a monotonic 5 s ramp: wz from -5 to +5 rad/s (2 rad/s^2), ax from -2 to +2 m/s^2,
    yaw the integral of wz (from 0, wrapped to [-pi, pi)).
The order of the levels and the neutral values of the other channels (gz 0, ax 0, ay 0,
yaw 0) are a CODE choice [ĐỀ XUẤT], not in the spec. The 27 s cycle repeats, so any run
length is covered by complete cycles plus a part.

    sample(t) -> (gz, ax, ay, az, yaw)    t in s from the first frame
"""
from __future__ import annotations

import math
from typing import Tuple

LEVEL_S = 2.0
RAMP_S = 5.0
AZ = 9.81
LEVELS = [('gz', -5.0), ('gz', -0.5), ('gz', 0.5), ('gz', 5.0),
          ('ax', -2.0), ('ax', 2.0), ('ay', -2.0), ('ay', 2.0),
          ('yaw', -2.0), ('yaw', 0.0), ('yaw', 2.0)]
STATIC_S = LEVEL_S * len(LEVELS)
CYCLE_S = STATIC_S + RAMP_S
WZ0, WZ_RATE = -5.0, 2.0
AX0, AX_RATE = -2.0, 0.8


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def segment(t: float) -> str:
    tc = t % CYCLE_S
    if tc < STATIC_S:
        name, v = LEVELS[int(tc // LEVEL_S)]
        return f'{name}={v:g}'
    return 'ramp'


def sample(t: float) -> Tuple[float, float, float, float, float]:
    tc = t % CYCLE_S
    if tc < STATIC_S:
        name, v = LEVELS[int(tc // LEVEL_S)]
        out = {'gz': 0.0, 'ax': 0.0, 'ay': 0.0, 'yaw': 0.0}
        out[name] = v
        return out['gz'], out['ax'], out['ay'], AZ, out['yaw']
    tr = tc - STATIC_S
    wz = WZ0 + WZ_RATE * tr
    ax = AX0 + AX_RATE * tr
    yaw = wrap_pi(WZ0 * tr + 0.5 * WZ_RATE * tr * tr)
    return wz, ax, 0.0, AZ, yaw
