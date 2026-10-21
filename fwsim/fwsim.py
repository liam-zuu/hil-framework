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

Not modelled: IMU, RC, ultrasonic interlock, SPI/UART latency (--io-delay-ms
adds a pure delay on the duty), H7 timing jitter.

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

COUNTS_PER_REV = 1320
PCNT_MODES = {'idf': 0, 'wrap16': 1}
FW_DT_US = 10_000


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
        if L.fwsim_abi() != 3:
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
             io_delay_ms=0, pid=None, mode=None, adrc=None):
    """Run one closed-loop simulation. Returns a dict of arrays sampled at the
    firmware tick (100 Hz). mode: 'PID' / 'ADRC' (None = firmware CTRL_MODE).
    pid: {axis: (kp, ki, kd, int_limit, out_limit)};
    adrc: {axis: (omega_c, omega_o, b0, out_limit)}; axis 0 vx, 1 vy, 2 wz."""
    p = params or MecanumParams()
    rng = np.random.default_rng(seed)
    fw.reset(pcnt_mode)
    if mode:
        fw.set_mode(mode)
    for axis, g in (pid or {}).items():
        fw.set_pid(axis, *g)
    for axis, g in (adrc or {}).items():
        fw.set_adrc(axis, *g)

    dt = 0.001
    n_plant = int(round(duration / dt))
    x = np.zeros(10)
    theta_w = np.zeros(4)
    duty_q = [np.zeros(4)] * (io_delay_ms + 1)     # duty pipeline, 1 ms per slot
    duty_now = np.zeros(4)
    sp_period_us = int(round(1e6 / sp_rate_hz))
    sp_held, sp_next_us = (0.0, 0.0, 0.0), 0
    log = {k: [] for k in ['t', 'sp', 'v_true', 'v_fw', 'duty', 'pwm', 'rpm', 'omega', 'pose', 'en',
                           'count', 'tick', 'dpwm', 'sat']}

    for k in range(n_plant + 1):
        t_us = k * 1000
        if t_us >= sp_next_us:                       # trajectory publisher
            sp_held = setpoint(t_us * 1e-6)
            sp_next_us += sp_period_us
        if k > 0 and t_us % FW_DT_US == 0:           # firmware tick
            counts = np.floor(theta_w * COUNTS_PER_REV / (2 * np.pi))
            duty, v_fw, pwm, rpm, en = fw.tick(t_us, counts, sp_held)
            p.motor_enable = float(en)
            duty_now = duty
            log['t'].append(t_us * 1e-6); log['sp'].append(sp_held); log['v_fw'].append(v_fw)
            log['v_true'].append(p.H_fwd @ x[6:10]); log['duty'].append(duty); log['pwm'].append(pwm)
            log['rpm'].append(rpm); log['omega'].append(x[6:10].copy()); log['pose'].append(x[0:3].copy())
            log['en'].append(en)
            cnt, tk, dp, sat = fw.last()
            log['count'].append(cnt); log['tick'].append(tk); log['dpwm'].append(dp); log['sat'].append(sat)
        if k == n_plant:
            break
        duty_q = duty_q[1:] + [duty_now]
        x = plant_step(x, duty_q[0], p, dt, rng)
        theta_w += x[6:10] * dt
    return {k: np.asarray(v, dtype=float) for k, v in log.items()}


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
            ['x', 'y', 'theta', 'en'])
    data = np.column_stack([log['t'], log['sp'], log['v_true'], log['v_fw'], log['duty'], log['pwm'],
                            log['omega'], log['pose'], log['en']])
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
  - log.csv: "100 Hz firmware ticks: setpoint, true body velocity (plant), firmware's encoder velocity, duty (rpwm-lpwm), pwm, wheel speed, pose, EN"
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
    ap.add_argument('--out')
    a = ap.parse_args()
    sp = make_setpoint(a)
    duration = a.duration or sp.duration or 20.0
    fw = FirmwareSim()
    params = MecanumParams()
    t0 = time.time()
    log = simulate(fw, sp, duration, params, a.seed, a.sp_rate, a.pcnt_mode, a.io_delay_ms, mode=a.mode)
    met = metrics(log)
    print(f'{duration:.1f} s simulated in {time.time() - t0:.1f} s, controller {fw.ctrl_mode}')
    for k, v in met.items():
        print(f'  {k:18s} {v:.4g}')
    if a.out:
        write_run(a.out, log, met, fw, dict(vars(a), duration=duration), params)
        print(f'Saved to {a.out}')


if __name__ == '__main__':
    main()
