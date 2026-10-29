"""
HIL Entry Point — RPi5

Spawns two independent OS processes:
  Process 1 — Plant loop (1 kHz, core 0, SCHED_FIFO)
  Process 2 — Trajectory publisher (50 Hz, core 2, normal)

Usage
──────
# 1. Dry run — simulated hardware, 5000 steps, idle trajectory
python run_hil.py --mock --steps 5000

# 2. Jitter measurement only (no SPI/GPIO)
python run_hil.py --jitter-only --jitter-n 2000

# 3. Real hardware, circle trajectory, 60 s
sudo python run_hil.py --traj circle --speed 0.3 --radius 0.5 --duration 60

# 4. Real hardware, line, default duration (forever)
sudo python run_hil.py --traj line --speed 0.3

# 5. SPI loopback test (short MOSI pin 19 to MISO pin 21)
python run_hil.py --loopback-test --steps 1000

# 6. EncoderGen check: send constant wheel speeds (no dynamics), 15 s
sudo python run_hil.py --fixed-omega 10,20,-10,-20 --steps 15000 --no-traj

# 7. BNO085 emulator check (G2): constant IMU values, 60 s, no trajectory
#    (gz,ax,ay,az,yaw in rad/s, m/s^2, m/s^2, m/s^2, rad)
sudo python run_hil.py --imu-static 0.5,0.2,-0.1,9.81,1.0 --steps 60000 --no-traj

Notes
──────
- sudo required for SCHED_FIFO (or grant cap_sys_nice).
- Ensure SPI enabled: sudo raspi-config → Interface → SPI.
- Trajectory setpoints go RPi5 → SPI → H7 → UART → ESP32 (HIL v2).
  The direct RPi5 → ESP32 UART path is legacy (--traj-uart-direct);
  it may need: sudo raspi-config → Interface → Serial.
- multiprocessing start method is 'spawn' (avoids lgpio/spidev fork issues).
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import signal
import sys
import time

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('hil.main')


# ── Process target functions ──────────────────────────────────────────────

def _parse_fixed_omega(text: str) -> list:
    """'10,20,-10,-20' -> [10.0, 20.0, -10.0, -20.0] (wheel order 1-4, rad/s)."""
    vals = [float(v) for v in text.split(',')]
    if len(vals) != 4:
        raise ValueError(f'--fixed-omega needs 4 comma-separated values, got {len(vals)}')
    return vals


def _run_plant(cfg, shared_state: mp.Array, shared_ref: mp.Array, use_mock: bool, seed: int = 0,
               wheels_lifted: bool = False, fixed_omega: list = None) -> None:
    """Process 1 target: plant simulation loop."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    # Pin before importing heavy deps
    os.sched_setaffinity(0, {cfg.cpu_core})

    from hil_node import HILNode
    from mecanum_plant import MecanumPlant
    from plant_interface import MockPlant

    from params import MecanumParams
    params = MecanumParams()
    if wheels_lifted:
        params.update_physical(wheels_lifted=True)     # spec S02 A-K: wheels off the ground
    if fixed_omega is not None:
        # EncoderGen check: constant wheel speeds, no dynamics; SPI/GPIO path stays real
        plant = MockPlant()
        plant.reset(list(fixed_omega) + [0.0] * 6)
        logging.getLogger('hil.main').info('Fixed wheel omega [rad/s] = %s (no plant dynamics)', fixed_omega)
    else:
        plant = MockPlant() if use_mock else MecanumPlant(params=params, dt=cfg.period_us / 1e6, seed=seed)
    logging.getLogger('hil.main').info('Plant RNG seed = %s, wheels_lifted = %s', seed, wheels_lifted)
    node  = HILNode(plant, config=cfg, shared_state=shared_state, shared_ref=shared_ref)
    node.run()


def _run_traj(spec, port: str, simulate: bool, shared_ref: mp.Array, uart_direct: bool) -> None:
    """Process 2 target: trajectory publisher."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    os.sched_setaffinity(0, {2})   # pin to core 2 (away from plant loop)

    from trajectory_pub import TrajectoryPublisher
    pub = TrajectoryPublisher(spec=spec, port=port, simulate=simulate,
                              shared_ref=shared_ref, uart_direct=uart_direct)
    pub.run()


# ── Loopback test ─────────────────────────────────────────────────────────

def _run_loopback_test(n_steps: int) -> int:
    """
    SPI loopback test: short MOSI (pin 19) → MISO (pin 21).
    Sends encoded states, verifies decoded state round-trip.
    MISO = MOSI in hardware loopback → decode_state(encode_state(s)) ≈ s.
    """
    from protocol import encode_state, decode_state, STATE_RANGES
    from spi_master import SPIMaster

    import numpy as np
    rng = np.random.default_rng(42)
    n_pass = n_fail = 0

    print(f"[loopback] Running {n_steps} SPI loopback transactions …", flush=True)
    with SPIMaster(simulate=False) as spi:
        for i in range(n_steps):
            state = [rng.uniform(-r * 0.8, r * 0.8) for r in STATE_RANGES]
            mosi  = encode_state(state)
            miso  = spi.transfer(mosi)   # loopback: MISO = MOSI
            recovered = decode_state(miso)

            max_err = max(abs(a - b) for a, b in zip(state, recovered))
            lsb_max = max(STATE_RANGES) / 32767
            if max_err <= lsb_max * 1.5:
                n_pass += 1
            else:
                n_fail += 1
                print(f"  FAIL step {i}: max_err={max_err:.2e}", flush=True)

    print(f"[loopback] Result: {n_pass}/{n_steps} PASS", flush=True)
    return 0 if n_fail == 0 else 1


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='HIL Node — RPi5 plant + trajectory publisher',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Mode flags
    p.add_argument('--mock',          action='store_true',
                   help='Simulate hardware (no GPIO/SPI/UART)')
    p.add_argument('--jitter-only',   action='store_true',
                   help='Measure timing jitter and exit')
    p.add_argument('--loopback-test', action='store_true',
                   help='SPI loopback test (short MOSI→MISO)')
    # Step/duration
    p.add_argument('--seed',          type=int,   default=0,
                   help='plant RNG seed (wheel-slip noise); record it with the run')
    p.add_argument('--steps',         type=int,   default=0,
                   help='Max plant steps (0 = forever)')
    p.add_argument('--jitter-n',      type=int,   default=2_000,
                   help='Number of ticks for jitter measurement')
    # Trajectory
    p.add_argument('--traj',          default='idle',
                   choices=['idle', 'line', 'circle', 'figure8', 'strafe',
                            'lift_step', 'lift_reverse', 'lift_ramp'],
                   help='Reference trajectory type')
    p.add_argument('--speed',         type=float, default=0.3,
                   help='Nominal linear speed [m/s]')
    p.add_argument('--radius',        type=float, default=0.5,
                   help='Circle/figure-8 radius [m]')
    p.add_argument('--duration',      type=float, default=0.0,
                   help='Run duration [s] (0 = forever)')
    p.add_argument('--test',          default=None,
                   help='Spec S02 group A test ID (A-L1 ... A-K3): sets trajectory, speed, '
                        'duration; A-K* also put the plant in wheels-lifted mode')
    # Hardware
    p.add_argument('--uart-port',     default='/dev/ttyAMA0',
                   help='UART port for ESP32 (legacy --traj-uart-direct only)')
    p.add_argument('--traj-uart-direct', action='store_true',
                   help='Legacy: also send 18-byte packet RPi5→ESP32 over UART '
                        '(default path is SPI → H7 → UART relay)')
    p.add_argument('--spi-speed',     type=int,   default=1_000_000,
                   help='SPI clock [Hz]')
    p.add_argument('--rt-priority',   type=int,   default=90,
                   help='SCHED_FIFO priority for plant loop')
    p.add_argument('--gpio-pin',      type=int,   default=17,
                   help='BCM sync GPIO pin')
    p.add_argument('--no-traj',       action='store_true',
                   help='Skip trajectory publisher process')
    p.add_argument('--imu-static',    default=None, metavar='GZ,AX,AY,AZ,YAW',
                   help='BNO085 emulator check (G2): send these constant IMU values '
                        '[rad/s, m/s^2, m/s^2, m/s^2, rad] in every frame instead of the '
                        'plant values (imu_seq still changes every frame)')
    p.add_argument('--fixed-omega',   default=None, metavar='W1,W2,W3,W4',
                   help='EncoderGen check: send these constant wheel speeds [rad/s] instead of '
                        'running the plant (use with --steps; implies --no-traj)')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    # ── Jitter-only mode ─────────────────────────────────────────────────
    if args.jitter_only:
        from timing import jitter_ok, measure_jitter
        stats = measure_jitter(n_ticks=args.jitter_n, period_us=1_000, rt=True)
        return 0 if jitter_ok(stats) else 1         # a failed restore fails too (B0 round 3 N17)

    # ── Loopback test ────────────────────────────────────────────────────
    if args.loopback_test:
        return _run_loopback_test(n_steps=args.steps or 1_000)

    # ── Normal HIL run ───────────────────────────────────────────────────
    from hil_node import HILConfig
    from imu_model import parse_imu_static
    from trajectory_pub import TrajectorySpec, GROUP_A, LIFTED_TESTS

    hil_cfg = HILConfig(
        period_us    = 1_000,
        rt_priority  = args.rt_priority,
        cpu_core     = 0,
        gpio_pin     = args.gpio_pin,
        spi_speed_hz = args.spi_speed,
        max_steps    = args.steps,
        simulate_hw  = args.mock,
        imu_static   = parse_imu_static(args.imu_static) if args.imu_static else None,
    )
    traj_spec = TrajectorySpec(
        kind     = args.traj,
        v        = args.speed,
        radius   = args.radius,
        duration = args.duration,
    )
    lifted = False
    fixed_omega = _parse_fixed_omega(args.fixed_omega) if args.fixed_omega else None
    if fixed_omega is not None:
        args.no_traj = True
    if args.test:
        if args.test not in GROUP_A:
            raise SystemExit(f'--test must be one of {sorted(GROUP_A)}')
        traj_spec = GROUP_A[args.test]
        lifted = args.test in LIFTED_TESTS
        args.traj = traj_spec.kind
        log.info('Group A test %s: %s', args.test, traj_spec)

    # Shared state array: plant writes, trajectory pub can read (optional)
    shared_state = mp.Array('d', 10)
    # Shared setpoint: trajectory pub writes [vx_ref, vy_ref, wz_ref, counter],
    # plant loop packs it into the SPI frame for the H7 → ESP32 relay
    shared_ref = mp.Array('d', 4)

    # Build process list
    procs = []

    p1 = mp.Process(
        target=_run_plant,
        args=(hil_cfg, shared_state, shared_ref, args.mock, args.seed, lifted, fixed_omega),
        name='HIL-Plant',
        daemon=True,
    )
    procs.append(p1)

    if not args.no_traj:
        p2 = mp.Process(
            target=_run_traj,
            args=(traj_spec, args.uart_port, args.mock, shared_ref, args.traj_uart_direct),
            name='HIL-TrajPub',
            daemon=True,
        )
        procs.append(p2)

    log.info(
        "Starting HIL: traj=%s  mock=%s  steps=%d  spi=%d kHz",
        args.traj, args.mock, args.steps, args.spi_speed // 1_000,
    )

    for proc in procs:
        proc.start()
        log.info("  %s  pid=%d", proc.name, proc.pid)

    # ── Wait + clean shutdown ────────────────────────────────────────────
    def _stop(signum, _frame):
        log.info("Received signal %d — shutting down …", signum)
        for proc in procs:
            if proc.is_alive():
                proc.terminate()

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Primary: wait for plant loop
    p1.join()
    log.info("Plant process exited (code=%s)", p1.exitcode)

    # Cleanup secondaries
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3.0)

    return p1.exitcode or 0


if __name__ == '__main__':
    # 'spawn' avoids fork-related issues with lgpio / spidev file descriptors
    mp.set_start_method('spawn', force=True)
    sys.exit(main())
