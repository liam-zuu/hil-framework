"""
fwsim.py: closed-loop PC simulation that runs the ESP32 firmware's own C code
(fwsim/build/fwsim.dll, built by build_fwsim.py) against the RPi5 plant
(rpi5/plant_step.py, motor model 'dc').

Timing, as on the HIL bench:
  plant         1 kHz  (rpi5/plant_step.py)
  firmware      100 Hz (10 ms tick: encoder_get_rpm over the last 10 ms, then
                vel_ctrl_step, then motor_set)
  setpoint      held at --sp-rate Hz (trajectory publisher on the RPi5: 50 Hz)
  encoder       wheel angle integrated at 1 kHz -> quadrature counts, 1320/rev
                (H7 ENC_PPR 330 x 4), fed to the firmware's PCNT through the shim
  PWM           the duty each GPIO carries after motor_set(), rpwm - lpwm per
                wheel (what H7 PWMCapture_GetDuty() sends to the plant)
  IMU           (simulate(imu=...)) accel, calibrated gyro, game RV reports at
                the report rate from rpi5/imu_model.py (shared with the RPi5:
                noise-free until the IMU noise measurement, K1), written into the
                firmware's imu_data_t at the report time; optional controlled
                gyro injection (method C, imu_model.GyroInjection) and chip resets

True body velocity (v_true) comes from the plant's body state (spec controller-IMU
v2 section 10), so it stays right with the wheels lifted.

Not modelled: RC, ultrasonic interlock, SPI/UART latency (--io-delay-ms adds a
pure delay on the duty), H7 timing jitter, BNO085 report latency.

  python fwsim/fwsim.py --traj circle --speed 0.3 --radius 0.5 --duration 20 --out <dir>
  python fwsim/fwsim.py --traj steps --duration 12 --out <dir>
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rpi5'))
from params import MecanumParams              # noqa: E402
from plant_step import plant_step             # noqa: E402
from trajectory_pub import TrajectorySpec, get_body_velocity   # noqa: E402
from imu_model import imu_from_plant          # noqa: E402

COUNTS_PER_REV = 1320
PCNT_MODES = {'idf': 0, 'wrap16': 1}
FW_DT_US = 10_000
IMU_COLS = ['g_valid', 'r_valid', 'wz_src', 'wz_meas', 'gz', 'yaw', 'age_g_ms', 'age_r_ms',
            'bias_gz', 'r', 'r_th', 'slip_flag', 'slip_on_count', 'src_switch_count',
            'src_switch_dwz', 'wz_cmd', 'head_on', 'psi_ref', 'psi_imu', 'e_psi',
            'psi_reset_count', 'head_toggle_count', 'bias_updates']


class FirmwareSim:
    """ctypes wrapper around fwsim.dll. One instance per process is enough:
    the firmware keeps static state, reset() re-runs its init functions."""

    def __init__(self, dll=os.path.join(HERE, 'build', 'fwsim.dll')):
        if not os.path.isfile(dll):
            sys.exit(f'{dll} not found: run python fwsim/build_fwsim.py first')
        self.lib = ctypes.CDLL(dll)
        L = self.lib
        L.fwsim_abi.restype = ctypes.c_int
        L.fwsim_ctrl_mode.restype = ctypes.c_int
        L.fwsim_reset.argtypes = [ctypes.c_int]
        L.fwsim_set_pid.argtypes = [ctypes.c_int] + [ctypes.c_float] * 5
        L.fwsim_get_pid.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        L.fwsim_set_ctrl_mode.argtypes = [ctypes.c_int]
        L.fwsim_set_adrc.argtypes = [ctypes.c_int] + [ctypes.c_float] * 4
        L.fwsim_get_adrc.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        L.fwsim_tick.restype = ctypes.c_int
        L.fwsim_tick.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
                                 ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_float),
                                 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)]
        L.fwsim_last.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint32)]
        L.fwsim_imu_report.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.POINTER(ctypes.c_float)]
        L.fwsim_imu_reset.argtypes = [ctypes.c_int64]
        L.fwsim_set_run.argtypes = [ctypes.c_int, ctypes.c_int]
        L.fwsim_motor_enable.argtypes = [ctypes.c_int]
        L.fwsim_set_param.restype = ctypes.c_int
        L.fwsim_set_param.argtypes = [ctypes.c_char_p, ctypes.c_double]
        L.fwsim_imu_cols.restype = ctypes.c_int
        L.fwsim_imu_last.argtypes = [ctypes.POINTER(ctypes.c_double)]
        if L.fwsim_abi() != 4:
            sys.exit('fwsim.dll ABI mismatch: rebuild with build_fwsim.py')
        self._counts = (ctypes.c_int32 * 4)()
        self._sp = (ctypes.c_float * 3)()
        self._duty = (ctypes.c_double * 4)()
        self._actual = (ctypes.c_float * 3)()
        self._pwm = (ctypes.c_int * 4)()
        self._rpm = (ctypes.c_float * 4)()
        info = os.path.join(os.path.dirname(dll), 'fwsim_build.json')
        self.build_info = json.load(open(info)) if os.path.isfile(info) else {}

    @property
    def ctrl_mode(self):
        return {0: 'PID', 1: 'ADRC'}.get(self.lib.fwsim_ctrl_mode(), '?')

    def reset(self, pcnt_mode='idf'):
        self.lib.fwsim_reset(PCNT_MODES[pcnt_mode])

    def set_mode(self, mode):
        self.lib.fwsim_set_ctrl_mode({'PID': 0, 'ADRC': 1}[mode])

    def set_adrc(self, axis, omega_c, omega_o, b0, out_limit):
        self.lib.fwsim_set_adrc(axis, omega_c, omega_o, b0, out_limit)

    def get_adrc(self, axis):
        out = (ctypes.c_float * 4)()
        self.lib.fwsim_get_adrc(axis, out)
        return dict(zip(['omega_c', 'omega_o', 'b0', 'out_limit'], [float(v) for v in out]))

    def set_pid(self, axis, kp, ki, kd, int_limit, out_limit):
        self.lib.fwsim_set_pid(axis, kp, ki, kd, int_limit, out_limit)

    def get_pid(self, axis):
        out = (ctypes.c_float * 5)()
        self.lib.fwsim_get_pid(axis, out)
        return dict(zip(['kp', 'ki', 'kd', 'int_limit', 'out_limit'], [float(v) for v in out]))

    def imu_report(self, kind, t_us, v):
        """kind 'accel' (ax, ay, az), 'gyro' (gz) or 'rv' (yaw in degrees), as imu.cpp stores it."""
        k = {'accel': 0, 'gyro': 1, 'rv': 2}[kind]
        v3 = list(v) if k == 0 else [0.0, 0.0, v]
        self.lib.fwsim_imu_report(k, int(t_us), (ctypes.c_float * 3)(*v3))

    def imu_reset(self, t_us):
        self.lib.fwsim_imu_reset(int(t_us))

    def set_run(self, run_active=True, wheels_lifted=False):
        self.lib.fwsim_set_run(int(run_active), int(wheels_lifted))

    def motor_enable(self, on):
        """SWA: driver EN (reset() turns it on, as HIL_BYPASS_MOTOR_ENABLE)."""
        self.lib.fwsim_motor_enable(int(on))

    def set_param(self, name, value):
        if not self.lib.fwsim_set_param(name.encode(), float(value)):
            raise KeyError(f'unknown imu_fusion/heading parameter {name}')

    def imu_last(self):
        """IMU fusion / heading columns of the last tick (IMU_COLS)."""
        out = (ctypes.c_double * len(IMU_COLS))()
        self.lib.fwsim_imu_last(out)
        return np.array(out[:])

    def last(self):
        """encoder_get_count() per wheel and the vel_ctrl counters of the last tick:
        (counts[4], tick, dpwm_sum[4], sat_ticks[4])."""
        c, o = (ctypes.c_int32 * 4)(), (ctypes.c_uint32 * 9)()
        self.lib.fwsim_last(c, o)
        return np.array(c[:]), o[0], np.array(o[1:5]), np.array(o[5:9])

    def tick(self, t_us, counts, sp):
        for i in range(4):
            self._counts[i] = int(counts[i])
        for i in range(3):
            self._sp[i] = sp[i]
        en = self.lib.fwsim_tick(t_us, self._counts, self._sp, self._duty, self._actual, self._pwm, self._rpm)
        return (np.array(self._duty[:]), np.array(self._actual[:]), np.array(self._pwm[:]),
                np.array(self._rpm[:]), en)


def make_setpoint(args):
    """Setpoint function t -> (vx, vy, wz)."""
    if args.traj == 'steps':
        # Separate steps on each axis, 3 s each, with rests: vx, -vx, vy, wz
        v, w = args.speed, args.speed / args.radius
        seq = [(0.5, (0, 0, 0)), (3.0, (v, 0, 0)), (1.5, (0, 0, 0)), (3.0, (-v, 0, 0)), (1.5, (0, 0, 0)),
               (3.0, (0, v, 0)), (1.5, (0, 0, 0)), (3.0, (0, 0, w)), (1.5, (0, 0, 0))]
        edges = np.cumsum([d for d, _ in seq])

        def f(t):
            k = int(np.searchsorted(edges, t, side='right'))
            return seq[k][1] if k < len(seq) else (0.0, 0.0, 0.0)
        f.duration = float(edges[-1])
        return f
    spec = TrajectorySpec(kind=args.traj, v=args.speed, radius=args.radius, period=args.period)
    f = lambda t: get_body_velocity(t, spec)       # noqa: E731
    f.duration = None
    return f


def simulate(fw, setpoint, duration, params=None, seed=0, sp_rate_hz=50.0, pcnt_mode='idf',
             io_delay_ms=0, pid=None, mode=None, adrc=None, imu=None, run_active=True,
             fw_params=None, motor_on=True):
    """Run one closed-loop simulation. Returns a dict of arrays sampled at the
    firmware tick (100 Hz). mode: 'PID' / 'ADRC' (None = firmware CTRL_MODE).
    pid: {axis: (kp, ki, kd, int_limit, out_limit)};
    adrc: {axis: (omega_c, omega_o, b0, out_limit)}; axis 0 vx, 1 vy, 2 wz.
    imu: None = no IMU reports (imu_get() never succeeds: wz from the encoders,
    heading loop off), or a dict:
      rate_hz     report rate of accel, gyro, game RV (default 20, IMU_REPORT_PERIOD_US)
      boot_reset  time of the chip's SH2_RESET at boot [s] (default -1.0: before the run)
      resets      further SH2_RESET times [s]
      injection   imu_model.GyroInjection (method C) or None
      hook        f(t, kind, value) -> value or None (drop the report): faults for tests
    run_active: MODE_AUTO with a VALID trajectory (heading loop may run).
    motor_on: False = SWA off for the whole run (driver disabled, controller idle).
    fw_params: {name: value} thresholds of imu_fusion / heading (fwsim_set_param)."""
    p = params or MecanumParams()
    rng = np.random.default_rng(seed)
    fw.reset(pcnt_mode)
    if mode:
        fw.set_mode(mode)
    for axis, g in (pid or {}).items():
        fw.set_pid(axis, *g)
    for axis, g in (adrc or {}).items():
        fw.set_adrc(axis, *g)
    if not motor_on:
        fw.motor_enable(False)
    fw.set_run(run_active, getattr(p, 'wheels_lifted', False))
    for name, v in (fw_params or {}).items():
        fw.set_param(name, v)
    use_imu = imu is not None
    imu = imu or {}
    imu_period_us = int(round(1e6 / imu.get('rate_hz', 20.0)))
    imu_next_us = imu_period_us
    resets = sorted(imu.get('resets', []))
    inj = imu.get('injection')
    hook = imu.get('hook') or (lambda t, kind, v: v)
    if use_imu:
        fw.imu_reset(int(round(imu.get('boot_reset', -1.0) * 1e6)))
    x_prev = None

    dt = 0.001
    n_plant = int(round(duration / dt))
    x = np.zeros(10)
    theta_w = np.zeros(4)
    duty_q = [np.zeros(4)] * (io_delay_ms + 1)     # duty pipeline, 1 ms per slot
    duty_now = np.zeros(4)
    sp_period_us = int(round(1e6 / sp_rate_hz))
    sp_held, sp_next_us = (0.0, 0.0, 0.0), 0
    log = {k: [] for k in ['t', 'sp', 'v_true', 'v_fw', 'duty', 'pwm', 'rpm', 'omega', 'pose', 'en',
                           'count', 'tick', 'dpwm', 'sat', 'imu', 'r_true', 'inj']}

    for k in range(n_plant + 1):
        t_us = k * 1000
        if t_us >= sp_next_us:                       # trajectory publisher
            sp_held = setpoint(t_us * 1e-6)
            sp_next_us += sp_period_us
        while resets and resets[0] * 1e6 <= t_us:      # chip SH2_RESET
            fw.imu_reset(t_us)
            resets.pop(0)
        if use_imu and t_us >= imu_next_us:              # BNO085 reports (host receive time = t_us)
            imu_next_us += imu_period_us
            gz, ax, ay, az, yaw = imu_from_plant(x, x_prev, dt, p.g)
            t = t_us * 1e-6
            if inj is not None:
                gz = inj.apply(gz, t)
            for kind, v in (('accel', (ax, ay, az)), ('gyro', gz), ('rv', math.degrees(yaw))):
                v = hook(t, kind, v)
                if v is not None:
                    fw.imu_report(kind, t_us, v)
        if k > 0 and t_us % FW_DT_US == 0:           # firmware tick
            counts = np.floor(theta_w * COUNTS_PER_REV / (2 * np.pi))
            duty, v_fw, pwm, rpm, en = fw.tick(t_us, counts, sp_held)
            p.motor_enable = float(en)
            duty_now = duty
            log['t'].append(t_us * 1e-6); log['sp'].append(sp_held); log['v_fw'].append(v_fw)
            log['v_true'].append(body_velocity(x)); log['duty'].append(duty); log['pwm'].append(pwm)
            log['rpm'].append(rpm); log['omega'].append(x[6:10].copy()); log['pose'].append(x[0:3].copy())
            log['en'].append(en)
            cnt, tk, dp, sat = fw.last()
            log['count'].append(cnt); log['tick'].append(tk); log['dpwm'].append(dp); log['sat'].append(sat)
            log['imu'].append(fw.imu_last())
            log['r_true'].append((p.H_fwd @ x[6:10])[2] - x[5])     # wz from wheels - body wz
            log['inj'].append(bool(inj is not None and inj.active(t_us * 1e-6)))
        if k == n_plant:
            break
        duty_q = duty_q[1:] + [duty_now]
        x_prev = x
        x = plant_step(x, duty_q[0], p, dt, rng)
        theta_w += x[6:10] * dt
    return {k: np.asarray(v, dtype=float) for k, v in log.items()}


def body_velocity(x):
    """True body velocity [vx, vy, wz] from the plant state: world velocity rotated by -theta."""
    c, s = math.cos(x[2]), math.sin(x[2])
    return np.array([c * x[3] + s * x[4], -s * x[3] + c * x[4], x[5]])


def metrics(log, t_from=0.0):
    """Body-velocity tracking metrics (true plant velocity vs setpoint)."""
    m = log['t'] >= t_from
    e = log['v_true'][m] - log['sp'][m]
    e_fw = log['v_fw'][m] - log['sp'][m]
    d = log['duty'][m]
    out = {}
    for i, ax in enumerate(['vx', 'vy', 'wz']):
        out[f'rms_{ax}'] = float(np.sqrt(np.mean(e[:, i] ** 2)))
        out[f'rms_{ax}_encoder'] = float(np.sqrt(np.mean(e_fw[:, i] ** 2)))
    out['sat_frac'] = float(np.mean(np.abs(log['pwm'][m]) >= 1000))
    out['tv_duty'] = float(np.sum(np.abs(np.diff(d, axis=0))) / max(log['t'][m][-1] - log['t'][m][0], 1e-9))
    return out


def lib_versions():
    import numpy, scipy  # noqa: E401
    v = dict(python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__)
    try:
        import pymoo
        v['pymoo'] = pymoo.__version__
    except ImportError:
        pass
    return v


def write_run(out, log, met, fw, args_dict, params):
    if os.path.isdir(out) and os.listdir(out):
        sys.exit(f'{out} already holds data: use a new folder.')
    os.makedirs(out, exist_ok=True)
    cols = (['t'] + [f'sp_{a}' for a in 'vx vy wz'.split()] + [f'v_{a}' for a in 'vx vy wz'.split()] +
            [f'vfw_{a}' for a in 'vx vy wz'.split()] + [f'duty_{w}' for w in 'fl fr rl rr'.split()] +
            [f'pwm_{w}' for w in 'fl fr rl rr'.split()] + [f'omega_{w}' for w in 'fl fr rl rr'.split()] +
            ['x', 'y', 'theta', 'en'] + IMU_COLS + ['r_true', 'inj'])
    data = np.column_stack([log['t'], log['sp'], log['v_true'], log['v_fw'], log['duty'], log['pwm'],
                            log['omega'], log['pose'], log['en'], log['imu'], log['r_true'], log['inj']])
    np.savetxt(os.path.join(out, 'log.csv'), data, delimiter=',', header=','.join(cols), comments='', fmt='%.6g')
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(met, f, indent=2)
    b = fw.build_info
    with open(os.path.join(out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"""spec: null
date: {time.strftime('%Y-%m-%d')}
setup: fwsim (PC: firmware C code in a DLL + rpi5/plant_step.py), NOT hardware
firmware:
  esp32: {json.dumps(b.get('esp32'))}
  hil_framework: {json.dumps(b.get('hil_framework'))}
  compiler: "{b.get('compiler')}"
  ctrl_mode: {fw.ctrl_mode}
config: {json.dumps(args_dict)}
plant: {{motor_model: {params.motor_model}, V_batt: {params.V_batt}, slip: {params.slip.enabled}}}
libraries: {json.dumps(lib_versions())}
files:
  - log.csv: "100 Hz firmware ticks: setpoint, true body velocity (plant body state), firmware's encoder velocity, duty (rpwm-lpwm), pwm, wheel speed, pose, EN, IMU fusion / heading columns (imu_fusion.c, heading.c), r_true (wz from wheels - body wz), inj (method C window)"
  - metrics.json: "RMS body-velocity error per axis vs setpoint (true and encoder), saturation fraction, duty total variation"
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--traj', default='steps', choices=['steps', 'line', 'circle', 'figure8', 'idle'])
    ap.add_argument('--speed', type=float, default=0.3)
    ap.add_argument('--radius', type=float, default=0.5)
    ap.add_argument('--period', type=float, default=8.0)
    ap.add_argument('--duration', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sp-rate', type=float, default=50.0)
    ap.add_argument('--pcnt-mode', default='idf', choices=list(PCNT_MODES))
    ap.add_argument('--io-delay-ms', type=int, default=0)
    ap.add_argument('--mode', choices=['PID', 'ADRC'], default=None, help='default: firmware CTRL_MODE')
    ap.add_argument('--imu-rate', type=float, default=20.0, help='BNO085 report rate [Hz] (IMU_REPORT_PERIOD_US)')
    ap.add_argument('--no-imu', action='store_true', help='no IMU reports: only to re-run pre-28/10 runs (QD 23: every firmware has the IMU)')
    ap.add_argument('--out')
    a = ap.parse_args()
    sp = make_setpoint(a)
    duration = a.duration or sp.duration or 20.0
    fw = FirmwareSim()
    params = MecanumParams()
    t0 = time.time()
    imu = None if a.no_imu else dict(rate_hz=a.imu_rate)
    log = simulate(fw, sp, duration, params, a.seed, a.sp_rate, a.pcnt_mode, a.io_delay_ms, mode=a.mode, imu=imu)
    met = metrics(log)
    print(f'{duration:.1f} s simulated in {time.time() - t0:.1f} s, controller {fw.ctrl_mode}')
    for k, v in met.items():
        print(f'  {k:18s} {v:.4g}')
    if a.out:
        write_run(a.out, log, met, fw, dict(vars(a), duration=duration), params)
        print(f'Saved to {a.out}')


if __name__ == '__main__':
    main()
