"""
IMU values for the BNO085 emulator on the H7 (G2, IMU option A).

The RPi5 computes what the IMU on the vehicle would measure and sends it in
the SPI frame (protocol v3: gz, ax, ay, az, yaw, imu_seq); the H7 only packs
it as SH-2 reports. G2 (Liam 27/10, Q3): values without noise or bias, taken
straight from the plant. The noise/bias model comes at G3, after the IMU noise
measurement on the vehicle (K1), and is shared with fwsim.

Frame: body frame x forward, y left, z up. The real BNO085 mounting axes on
the vehicle are not used yet (to be derived from the 10/06 logs at G3).
Accelerometer convention as the BNO085 accelerometer report: specific force
including gravity, so a vehicle at rest on flat ground reads az = +g.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

# MecanumPlant / plant_step state layout (see plant_step.py)
IDX_THETA = 2
IDX_VX_W  = 3      # world frame
IDX_VY_W  = 4      # world frame
IDX_WZ    = 5


def imu_from_plant(state: Sequence[float], prev_state: Optional[Sequence[float]],
                   dt: float, g: float) -> tuple:
    """
    Noise-free IMU sample from two consecutive MecanumPlant states.

    Args:
        state:      current plant state [x, y, θ, vx_w, vy_w, wz, ω1..ω4].
        prev_state: state one step earlier (None on the first step → a = 0).
        dt:         plant step [s].
        g:          gravity [m/s²].

    Returns:
        (gz, ax, ay, az, yaw) [rad/s, m/s², m/s², m/s², rad], the order of
        protocol.encode_state(imu=...).
    """
    theta = state[IDX_THETA]
    if prev_state is None:
        a_wx = a_wy = 0.0
    else:
        a_wx = (state[IDX_VX_W] - prev_state[IDX_VX_W]) / dt
        a_wy = (state[IDX_VY_W] - prev_state[IDX_VY_W]) / dt
    # world → body: rotate by -θ
    c, s = math.cos(theta), math.sin(theta)
    ax = c * a_wx + s * a_wy
    ay = -s * a_wx + c * a_wy
    return (state[IDX_WZ], ax, ay, g, theta)


def parse_imu_static(text: str) -> tuple:
    """'gz,ax,ay,az,yaw' → tuple of 5 floats (run_hil.py --imu-static)."""
    vals = [float(v) for v in text.split(',')]
    if len(vals) != 5:
        raise ValueError(f'--imu-static needs 5 comma-separated values '
                         f'(gz,ax,ay,az,yaw), got {len(vals)}')
    return tuple(vals)
