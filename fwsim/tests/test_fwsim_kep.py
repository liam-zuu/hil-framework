"""PID integral clamp indicator "tick cham kep" of fwsim (spec S03 v8.6 section 2.1, section 11 item 6 (c)):
fw_sim.c fwsim_kep_last, fwsim.py KEP_COLS / log['kep'] / kep_indicator.
Run: python -m pytest -q fwsim/tests/test_fwsim_kep.py   (needs fwsim/build/fwsim.dll)
     python fwsim/tests/test_fwsim_kep.py --digest       (old channels of the golden run, as recorded in GOLDEN)
     python fwsim/tests/test_fwsim_kep.py --scan         (cross-check against phan_tich_kep_pid, one JSON line per run)
     python fwsim/tests/test_fwsim_kep.py --pair <dll>   (old / new DLL pair on the bit-exact channels, one JSON
     line per run, with the path and SHA-256 of each DLL as passed and as really loaded)"""
import argparse
import csv
import ctypes
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fwsim  # noqa: E402
import o1_feasibility as o1  # noqa: E402
import build_fwsim  # noqa: E402
import s03_doe as s3  # noqa: E402

_DLL = os.path.join(fwsim.HERE, 'build', 'fwsim.dll')
pytestmark = pytest.mark.skipif(not os.path.isfile(_DLL), reason='fwsim.dll not built')

# channels of simulate() before the kep channel was added
OLD_KEYS = ('t', 'sp', 'v_true', 'v_fw', 'duty', 'pwm', 'rpm', 'omega', 'pose', 'en', 'count', 'tick', 'dpwm',
            'sat', 'imu', 'r_true', 'inj', 'pos', 'ws')
# golden run: integral limits lowered to 0.02 so the clamp is hit on both signs within 4 s
GOLDEN_PID = {0: (1.5, 0.05, 0.0, 0.02, 1.5), 1: (1.5, 0.05, 0.0, 0.02, 1.5), 2: (1.0, 0.05, 0.0, 0.02, 2.0)}
GOLDEN_ILIM = (0.02, 0.02, 0.02)


def sp_golden(t):
    return (0.3, 0.0, 0.3) if t < 2.0 else (0.0, 0.0, 0.0)


def golden_run(fw):
    """4 s: drive, stop, standstill hold; default plant, position loop on, IMU 20 Hz noise-free."""
    return fwsim.simulate(fw, sp_golden, 4.0, seed=1, mode='PID', imu={}, pid=GOLDEN_PID)


def old_digest(log):
    """{key: [dtype, shape, sha256 of the raw bytes]} of the channels that existed before kep."""
    return {k: [str(log[k].dtype), list(log[k].shape),
                hashlib.sha256(np.ascontiguousarray(log[k]).tobytes()).hexdigest()] for k in OLD_KEYS if k in log}


def identity(fw):
    """What the golden digests depend on besides this file: firmware, compiler, plant sources, numpy."""
    b = fw.build_info.get('esp32') or {}
    return dict(esp32_commit=b.get('commit'), esp32_dirty=b.get('dirty'), compiler=fw.build_info.get('compiler'),
                plant_src=fwsim.PLANT_SRC, numpy=np.__version__)


# recorded with --digest on the build above (before the kep channel was added)
GOLDEN_ID = {'esp32_commit': '5bde4c5', 'esp32_dirty': False,
             'compiler': 'gcc (MinGW-W64 x86_64-ucrt-posix-seh, built by Brecht Sanders, r8) 13.2.0',
             'plant_src': {'plant_step_py': 'e56afd1fa6a7e671ccd828c647daac3e2d4790cb',
                           'params_py': '501e5f5e24defc8ec5e8c2dd45d4ee921596e0fa',
                           'method': 'git hash-object'},
             'numpy': '2.4.1'}
GOLDEN = {'t': ['float64', [400], '4383e229850ba37f86ec736bc15dafec3cbf49340dfb7dc0aba903137519eefa'],
          'sp': ['float64', [400, 3], 'd35e325c5020b07a31d7664014782ac90cc8476daae3f5f28ea1702d9c1af069'],
          'v_true': ['float64', [400, 3], '5b09c186a31c1206f3ba8722f1ce2ef1733a80071041f7717f4f31e801db5aca'],
          'v_fw': ['float64', [400, 3], '60d3c4128db758368827c98578b0866505524f02afb87ce735e32114ae181d34'],
          'duty': ['float64', [400, 4], 'bae6a69c0df7373b55ed93a81a997023c40ff29e7ffe13e90184763c3176bfcd'],
          'pwm': ['float64', [400, 4], '0b6255e573bf404751c280f61ee60e432e6bd61263d822029888319bbd129f2e'],
          'rpm': ['float64', [400, 4], '895c7351316c181b7249eb6041f1be0f49a3f32ebb8d17f94534e24b002cfd60'],
          'omega': ['float64', [400, 4], '569bb97ddc2e0a726d4d68c8b5acb6232c4520d510a51664a26f5ce336ba3b70'],
          'pose': ['float64', [400, 3], '410f39203bc3786d0a006dfb8dfd3a4a22b18d6a68d8e0fa83162e5cdf02f4d2'],
          'en': ['float64', [400], '9498dac910b1db181ba67f375928e0a69b5d011418bbc60f185b36b4256fb76c'],
          'count': ['float64', [400, 4], '273be4dd76955aab9134dda29ac5b97fd3fa094938875426fca1302383284175'],
          'tick': ['float64', [400], 'bed067f0db7ebdc88c4418f42a3740987d51aa0f492fdc459de3c92fdfec9689'],
          'dpwm': ['float64', [400, 4], 'c730ef98851b8e71507f38054807666e393e6d52df568caf4420981d97153d80'],
          'sat': ['float64', [400, 4], '59ec91dcb7dc65b5f928091cb0e25c26729a0a4453ebe7d8244fc1ceae7d9712'],
          'imu': ['float64', [400, 24], '86a1c31fa798953a6748bd741e94bdb7618e0d6f334f24548b4f24b99b6c7416'],
          'r_true': ['float64', [400], 'e67433d640c349538204c947c43bd84a9a2fffca4455ff2798b0aacc99fe2591'],
          'inj': ['float64', [400], '5a312281df4bd8dfbb4d4a94ad0bf44d01bb8cfced1206b90e21b4ca0568cdb1'],
          'pos': ['float64', [400, 16], '71069be95af9df62efff6ac4144a3e7e66c4dd7af2e066e258825d071cce4708'],
          'ws': ['float64', [400, 5], 'b175c82d49ebbfee978bbfcf0a297c93e3fb0f5d04ddc47c138b5a34e0a6805d']}

P = {n: i for i, n in enumerate(fwsim.POS_COLS)}
M = {n: i for i, n in enumerate(fwsim.IMU_COLS)}
K = {n: i for i, n in enumerate(fwsim.KEP_COLS)}
AXES = ('vx', 'vy', 'wz')

DT_US = 62_500   # 2**-4 s: (float)62500 * 1e-6f is exactly 0.0625f, so with setpoints that are short
                 # binary fractions e*dt and the sums are exact in float32


def boot(fw, mode='PID', **params):
    """Direct firmware ticks: reset, mode, pos loop off (plus params); tick(sp) -> (kep, still):
    fw.kep_last() (KEP_COLS) and the still column of fw.pos_last() of the tick. Encoder counts stay 0
    and no IMU report is sent: the measured velocity is 0, wz comes from the encoders, the heading loop
    is off, pos_on = 0 gives no position correction, so the PID error is the float32 setpoint on each
    axis. t_us advances by DT_US from 0 (first tick at DT_US: dt_actual = 0.0625f)."""
    fw.reset()
    fw.set_mode(mode)
    for n, v in dict(pos_on=0, **params).items():
        fw.set_param(n, v)
    t_us = [0]

    def tick(sp):
        t_us[0] += DT_US
        fw.tick(t_us[0], [0, 0, 0, 0], sp)
        return fw.kep_last(), fw.pos_last()[P['still']]
    return tick


@pytest.fixture(scope='module')
def fw():
    f = fwsim.FirmwareSim()
    if not f.kep_cols or not f.pos_cols:
        pytest.skip('fwsim.dll without the kep / pos channels: rebuild with build_fwsim.py')
    return f


@pytest.fixture(scope='module')
def glog(fw):
    return golden_run(fw)


def test_no_hit_direct(fw):
    t = boot(fw)
    for k in range(1, 4):
        kep, _ = t((0.25, -0.5, 0.5))
        assert kep[:3] == [0.0, 0.0, 0.0]
        assert kep[3:6] == [0.015625 * k, -0.03125 * k, 0.03125 * k]
        assert kep[K['kep_lech']] == 0.0


def test_hit_and_exact_limit_direct(fw):
    t = boot(fw)
    for k in range(1, 18):
        kep, _ = t((1.0, -1.0, 1.0))
        assert kep[:3] == [float(k >= 8), float(k >= 8), float(k >= 16)]
        assert kep[3:6] == [min(0.0625 * k, 0.5), max(-0.0625 * k, -0.5), min(0.0625 * k, 1.0)]
        assert kep[K['kep_lech']] == 0.0
    kep, _ = t((-1.0, 1.0, -1.0))
    assert kep[:3] == [0.0, 0.0, 0.0]
    assert kep[3:6] == [0.4375, -0.4375, 0.9375]
    assert kep[K['kep_lech']] == 0.0


def test_one_ulp_below_and_float32_rounding_onto_the_limit(fw):
    # (a) one ulp below the limit: the float32 sum lands on 0.5 - 2**-25 exactly (no hit), then the
    # next tick adds exactly that ulp and lands on the limit without being cut
    t = boot(fw)
    for _ in range(7):
        kep, _ = t((1.0, 0.0, 0.0))
    assert kep[K['kep_I_vx']] == 0.4375
    kep, _ = t((1.0 - 2**-21, 0.0, 0.0))
    assert kep[K['kep_I_vx']] == float(np.nextafter(np.float32(0.5), np.float32(0))) == 0.5 - 2**-25
    assert kep[K['kep_vx']] == 0.0
    kep, _ = t((2**-21, 0.0, 0.0))
    assert kep[K['kep_I_vx']] == 0.5
    assert kep[K['kep_vx']] == 1.0
    assert kep[K['kep_lech']] == 0.0
    # (b) a binary64 sum below the limit that float32 rounds onto it
    t = boot(fw)
    for _ in range(7):
        t((1.0, 0.0, 0.0))
    assert 0.4375 + (1.0 - 2**-23) * 0.0625 < 0.5
    kep, _ = t((1.0 - 2**-23, 0.0, 0.0))
    assert kep[K['kep_I_vx']] == 0.5
    assert kep[K['kep_vx']] == 1.0
    assert kep[K['kep_lech']] == 0.0


def test_motor_off_and_edge(fw):
    t = boot(fw)
    for _ in range(9):
        kep, _ = t((1.0, 0.0, 0.0))
    assert kep[K['kep_vx']] == 1.0 and kep[K['kep_I_vx']] == 0.5
    fw.motor_enable(False)
    kep, _ = t((1.0, 0.0, 0.0))
    assert kep == [0.0] * 7
    fw.motor_enable(True)
    kep, _ = t((1.0, 0.0, 0.0))
    assert kep[K['kep_I_vx']] == 0.0625 and kep[K['kep_vx']] == 0.0
    assert kep[K['kep_lech']] == 0.0


def test_standstill_hold_tick_reports_the_state_after_the_clamp(fw):
    t = boot(fw)
    for _ in range(9):
        kep, still = t((1.0, 0.0, 0.0))
    assert kep[K['kep_vx']] == 1.0 and kep[K['kep_I_vx']] == 0.5 and still == 0.0
    for j in range(1, 11):
        kep, still = t((0.0, 0.0, 0.0))
        if j <= 8:
            assert (kep[K['kep_vx']], kep[K['kep_I_vx']], still) == (1.0, 0.5, 0.0)
        elif j == 9:
            # hold starts: the PID is reset after the update, the DLL still reports the state after
            # the clamp (recomputed by the replay of kep_step, held PID already cleared)
            assert (kep[K['kep_vx']], kep[K['kep_I_vx']], still) == (1.0, 0.5, 1.0)
        else:
            assert (kep[K['kep_vx']], kep[K['kep_I_vx']], still) == (0.0, 0.0, 1.0)
        assert kep[K['kep_lech']] == 0.0


def test_adrc_is_zero(fw):
    t = boot(fw, mode='ADRC')
    for _ in range(12):
        kep, _ = t((1.0, -1.0, 1.0))
        assert kep == [0.0] * 7
    log = fwsim.simulate(fw, lambda t_: (0.3, 0.0, 0.0), 1.0, mode='ADRC')
    assert np.all(log['kep'] == 0)
    assert log['kep'].shape == (100, 7)
    assert fwsim.kep_indicator(log, 1.0) == dict(kep_n_vx=0, kep_n_vy=0, kep_n_wz=0, kep_mau=100,
                                               kep_frac_vx=0.0, kep_frac_vy=0.0, kep_frac_wz=0.0)


def test_no_hit_simulate(fw):
    log = fwsim.simulate(fw, lambda t_: (0.3, 0.0, 0.0), 1.0, mode='PID')
    assert np.all(log['kep'][:, :3] == 0)
    assert np.all(np.abs(log['kep'][:, 3:6]) < np.array([0.5, 0.5, 1.0]))
    assert np.all(log['kep'][:, K['kep_lech']] == 0)
    assert fwsim.kep_indicator(log, 1.0) == dict(kep_n_vx=0, kep_n_vy=0, kep_n_wz=0, kep_mau=100,
                                               kep_frac_vx=0.0, kep_frac_vy=0.0, kep_frac_wz=0.0)
    log = fwsim.simulate(fw, lambda t_: (0.3, 0.0, 0.0), 0.5, mode='PID', motor_on=False)
    assert np.all(log['kep'] == 0)
    out = fwsim.kep_indicator(log, 0.5)
    assert (out['kep_n_vx'], out['kep_n_vy'], out['kep_n_wz'], out['kep_mau']) == (0, 0, 0, 0)
    assert math.isnan(out['kep_frac_vx']) and math.isnan(out['kep_frac_vy']) \
        and math.isnan(out['kep_frac_wz'])


def test_golden_run_flags(fw, glog):
    lim = np.asarray(GOLDEN_ILIM, dtype=np.float32)    # the limits the firmware holds (float)
    flag, integ, on = glog['kep'][:, :3], glog['kep'][:, 3:6], glog['en'] == 1
    assert np.array_equal(flag, ((np.abs(integ) == lim) & on[:, None]).astype(float))
    assert np.all(np.abs(integ) <= lim)
    assert np.all(glog['kep'][:, K['kep_lech']] == 0)
    ind = fwsim.kep_indicator(glog, 4.0)
    assert ind['kep_n_vx'] > 0 and ind['kep_n_wz'] > 0
    assert ind['kep_mau'] == 400
    for i, a in enumerate(AXES):
        n = int(flag[:, i].sum())
        assert ind[f'kep_n_{a}'] == n
        assert ind[f'kep_frac_{a}'] == n / 400
    assert np.any(integ[:, 0] == lim[0]) and np.any(integ[:, 0] == -lim[0])
    assert np.any(glog['pos'][:, P['still']] == 1)


def test_old_channels_bit_for_bit_as_before_the_change(fw, glog):
    if identity(fw) != GOLDEN_ID:
        pytest.skip(f'golden recorded on {GOLDEN_ID}, this is {identity(fw)}')
    assert set(glog) == set(GOLDEN) | {'kep'}
    assert old_digest(glog) == GOLDEN


def test_kep_off_leaves_the_old_channels(fw, glog, monkeypatch):
    monkeypatch.setattr(fw, 'kep_cols', 0)
    b = golden_run(fw)
    assert 'kep' not in b
    assert old_digest(b) == old_digest(glog)
    with pytest.raises(ValueError):
        fwsim.kep_indicator(b, 4.0)


def test_indicator_window_and_fields():
    t = np.arange(1, 11) * 0.01
    kep = np.zeros((10, len(K)))
    kep[[2, 3, 4, 7], K['kep_vx']] = 1.0
    kep[7, K['kep_wz']] = 1.0
    log = dict(t=t, en=np.array([1.0] * 8 + [0.0] * 2), kep=kep)
    out = fwsim.kep_indicator(log, 0.05)
    assert (out['kep_n_vx'], out['kep_n_vy'], out['kep_n_wz']) == (3, 0, 0)
    assert out['kep_mau'] == 5 and out['kep_frac_vx'] == 0.6
    assert fwsim.kep_indicator(log, t[4] - 5e-10) == out        # still inside the 1e-9 tolerance
    out = fwsim.kep_indicator(log, t[4] - 2e-9)
    assert out['kep_mau'] == 4 and out['kep_n_vx'] == 2 and out['kep_frac_vx'] == 0.5
    out = fwsim.kep_indicator(log, 1.0)
    assert (out['kep_mau'], out['kep_n_vx'], out['kep_n_wz']) == (8, 4, 1)
    assert out['kep_frac_vx'] == 0.5 and out['kep_frac_wz'] == 0.125
    out = fwsim.kep_indicator(log, 0.0)
    assert out['kep_mau'] == 0 and out['kep_n_vx'] == 0
    assert math.isnan(out['kep_frac_vx']) and math.isnan(out['kep_frac_vy']) \
        and math.isnan(out['kep_frac_wz'])
    assert list(out) == ['kep_n_vx', 'kep_n_vy', 'kep_n_wz', 'kep_mau',
                         'kep_frac_vx', 'kep_frac_vy', 'kep_frac_wz']


def _reject_cases():
    t = np.arange(1, 11) * 0.01
    en = np.ones(10)
    good = np.zeros((10, len(K)))
    cases = [(dict(t=t, en=en), 1.0),                             # no kep channel
             (dict(t=t, en=en, kep=np.zeros((10, 6))), 1.0),      # wrong column count
             (dict(t=t, en=en, kep=np.zeros((9, len(K)))), 1.0)]  # wrong row count
    for bad in (float('nan'), float('inf'), -0.01, None, '1', True):
        cases.append((dict(t=t, en=en, kep=good.copy()), bad))
    for v in (2.0, 0.5, float('nan')):                            # a flag that is not 0 / 1
        k = good.copy()
        k[0, K['kep_vx']] = v
        cases.append((dict(t=t, en=en, kep=k), 1.0))
    k = good.copy()                                               # flag set with the motor off
    k[0, K['kep_vx']] = 1.0
    cases.append((dict(t=t, en=np.array([0.0] + [1.0] * 9), kep=k), 1.0))
    k = good.copy()                                               # kep_lech: the DLL self-check failed
    k[3, K['kep_lech']] = 1.0
    cases.append((dict(t=t, en=en, kep=k), 1.0))
    return cases


@pytest.mark.parametrize('log,t_end', _reject_cases())
def test_indicator_rejects(log, t_end):
    with pytest.raises(ValueError):
        fwsim.kep_indicator(log, t_end)


GOC = 'D:/CLAUDE_THESIS/05_DATA/results/C8/2026-11-20_run02/phan_tich_kep_pid/kiem_kep_pid.py'
MU_GIUA = 0.436      # middle mu of S03 (spec S03 v8.6 section 8 item 12: 0.413 / 0.436 / 0.473)
S03_IMU = dict(rate_hz=20.0, boot_reset=-1.0, noise='replay', noise_seed=1)    # s03_doe.py IMU, SEED = 1
_PATH = []


def s03_defaults(fw):
    """(kp_xy, ki_xy, kp_wz, ki_wz, pos_kp, pos_ki, k_psi) the firmware holds after reset (k_psi: heading.h 1.0)."""
    fw.reset()
    fw.set_mode('PID')
    x, w, pos = fw.get_pid(0), fw.get_pid(2), fwsim.pos_meta(fw, raw=True)
    return (x['kp'], x['ki'], w['kp'], w['ki'], pos['kp'], pos['ki'], 1.0)


def s03_run(fw, v, g):
    """One PID run as s03_doe.run_one: figure 8 of O1 at level v, traction ground, middle mu, IMU 20 Hz with the
    noise replay seed 1; g = (kp_xy, ki_xy, kp_wz, ki_wz, pos_kp, pos_ki, k_psi), the fixed PID fields as the
    firmware holds them. Returns (log, integral limits, summary of evaluate_v7)."""
    if not _PATH:
        _PATH.append(o1.Path8())
    sch = o1.Schedule(v)
    p = fwsim.plant_params(argparse.Namespace(ground='traction', mu=MU_GIUA))
    fw.reset()
    fw.set_mode('PID')
    base = [fw.get_pid(i) for i in range(3)]
    pid = {i: (*((g[2], g[3]) if i == 2 else (g[0], g[1])), 0.0, base[i]['int_limit'], base[i]['out_limit'])
           for i in range(3)}
    log = fwsim.simulate(fw, sch.setpoint, sch.Tcmd + o1.T_LATE, p, seed=1, mode='PID', imu=dict(S03_IMU),
                         pid=pid, fw_params=dict(pos_kp=g[4], pos_ki=g[5], k_psi=g[6]))
    res, _ = o1.evaluate_v7(log, _PATH[0], sch, lsum=p.lx + p.ly)
    return log, tuple(b['int_limit'] for b in base), res


def dung_lai(log, ilim):
    """The rebuild of phan_tich_kep_pid/kiem_kep_pid.py run() on a simulate() log, tick by tick (binary64, dt
    from the time stamps, reset at the motor 0 -> 1 edge and on every still tick, hit = cut by the clamp).
    Columns: sp_* = the float32 setpoint the firmware gets, vx / vy = log['v_fw'], pos_corr_*, still = log['pos'],
    wz_cmd, wz_meas = log['imu'], motor_on = log['en']. Returns (hit, integral before the clamp, after it)."""
    n = len(log['t'])
    ts = np.round(log['t'] * 1e6)
    sp = np.asarray(log['sp'], dtype=np.float32).astype(float)
    pos, imu = log['pos'], log['imu']
    hit, pre, post = np.zeros((n, 3), dtype=bool), np.zeros((n, 3)), np.zeros((n, 3))
    I = [0.0, 0.0, 0.0]
    prev_t, prev_motor = None, 0
    for k in range(n):
        t = float(ts[k])
        dt = (t - prev_t) / 1e6 if prev_t is not None else 0.01
        if dt <= 0 or dt > 0.1:
            dt = 0.01
        prev_t = t
        if not int(log['en'][k]):
            prev_motor = 0
            continue
        if not prev_motor:
            I = [0.0, 0.0, 0.0]
        prev_motor = 1
        e = (sp[k, 0] + pos[k, P['pos_corr_vx']] - log['v_fw'][k, 0],
             sp[k, 1] + pos[k, P['pos_corr_vy']] - log['v_fw'][k, 1],
             imu[k, M['wz_cmd']] - imu[k, M['wz_meas']])
        for a in range(3):
            I[a] += e[a] * dt
            pre[k, a] = I[a]
            if abs(I[a]) > ilim[a]:
                I[a] = math.copysign(ilim[a], I[a])
                hit[k, a] = True
            post[k, a] = I[a]
        if int(pos[k, P['still']]):
            I = [0.0, 0.0, 0.0]
    return hit, pre, post


def truoc_kep32(log):
    """float32 replay of pid.c line 41 (integral += error * dt) of every motor-on tick, from the integral the
    DLL reported on the tick before (0 after a motor-off or a still tick): the sum before the clamp, as the
    firmware computes it (error = (sp + pos_corr) - actual, wz_cmd - wz_meas; dt = (float)10000 * 1e-6f)."""
    n = len(log['t'])
    kep, pos, imu = log['kep'], log['pos'], log['imu']
    sp, act = np.asarray(log['sp'], dtype=np.float32), np.asarray(log['v_fw'], dtype=np.float32)
    e = np.column_stack([(sp[:, 0] + pos[:, P['pos_corr_vx']].astype(np.float32)) - act[:, 0],
                         (sp[:, 1] + pos[:, P['pos_corr_vy']].astype(np.float32)) - act[:, 1],
                         imu[:, M['wz_cmd']].astype(np.float32) - imu[:, M['wz_meas']].astype(np.float32)])
    assert e.dtype == np.float32
    dt = np.float32(10000) * np.float32(1e-6)
    pre = np.zeros((n, 3), dtype=np.float32)
    for k in range(n):
        if log['en'][k] != 1:
            continue
        fresh = k == 0 or log['en'][k - 1] != 1 or pos[k - 1, P['still']] == 1
        i0 = np.zeros(3, dtype=np.float32) if fresh else kep[k - 1, 3:6].astype(np.float32)
        pre[k] = i0 + e[k] * dt
    assert pre.dtype == np.float32
    return pre


def _doan(ks):
    """Runs of consecutive tick indices: [[first, last], ...]."""
    out = []
    for k in ks:
        if out and k == out[-1][1] + 1:
            out[-1][1] = int(k)
        else:
            out.append([int(k), int(k)])
    return out


def so_sanh(log, ilim, t_end=None, n_list=40):
    """DLL flags against the rebuild of phan_tich_kep_pid, tick by tick, whole log and O1 window [0, t_end].
    Kinds of a differing tick: 'bien' DLL 1, rebuild 0, the float32 sum lands exactly on the limit (not cut);
    'cat32' DLL 1, rebuild 0, the firmware's float32 sum is beyond the limit (cut) while the binary64 rebuild
    is not; 'duoi32' DLL 0, rebuild 1; 'khac' anything else."""
    hit, pre64, post64 = dung_lai(log, ilim)
    kep, pos = log['kep'], log['pos']
    flag, on = kep[:, :3] == 1, log['en'] == 1
    lim = np.asarray(ilim, dtype=np.float32)
    pre32 = truoc_kep32(log)
    post32 = np.minimum(np.maximum(pre32, -lim), lim)
    win = np.ones(len(on), dtype=bool) if t_end is None else log['t'] <= t_end + 1e-9
    kind = {'bien': flag & ~hit & (np.abs(pre32) == lim), 'cat32': flag & ~hit & (np.abs(pre32) > lim),
            'duoi32': ~flag & hit}
    kind['khac'] = (flag != hit) & ~kind['bien'] & ~kind['cat32'] & ~kind['duoi32']
    out = dict(n_tick=int(len(on)), n_motor_on=int(on.sum()), n_motor_on_win=int((on & win).sum()),
               n_still=int((pos[:, P['still']] == 1).sum()), n_still_win=int(((pos[:, P['still']] == 1) & win).sum()),
               replay32_khac_dll=int(np.count_nonzero((post32 != kep[:, 3:6].astype(np.float32)) & on[:, None])),
               kep_lech=int(np.count_nonzero(kep[:, K['kep_lech']])))
    for i, a in enumerate(AXES):
        d = dict(dll=int(flag[:, i].sum()), dung_lai=int(hit[:, i].sum()), dll_win=int(flag[win, i].sum()),
                 dung_lai_win=int(hit[win, i].sum()), lech=int((flag[:, i] != hit[:, i]).sum()),
                 lech_win=int((flag[win, i] != hit[win, i]).sum()),
                 I_dll_absmax=float(np.abs(kep[:, 3 + i]).max()), I_dung_lai_absmax=float(np.abs(post64[:, i]).max()))
        d['cham_khi_giu'] = int((flag[:, i] & (pos[:, P['still']] == 1)).sum())    # flag set on a still (hold) tick
        for name, m in kind.items():
            d[name] = int(m[:, i].sum())
            d[name + '_win'] = int(m[win, i].sum())
            d[name + '_doan'] = _doan(np.nonzero(m[:, i])[0])
        out[a] = d
    rows = []
    for k, i in zip(*np.nonzero(flag != hit)):
        rows.append(dict(k=int(k), t=float(log['t'][k]), truc=AXES[i], dll=int(flag[k, i]), dung_lai=int(hit[k, i]),
                         loai=next(name for name, m in kind.items() if m[k, i]), I_dll=float(kep[k, 3 + i]),
                         truoc_kep32=float(pre32[k, i]), truoc_kep64=float(pre64[k, i]),
                         still=int(pos[k, P['still']]), trong_cua_so=bool(win[k])))
    out['n_tick_lech'], out['tick_lech'] = len(rows), rows[:n_list]
    return out


def script_goc(log, tmp, ilim):
    """The ORIGINAL kiem_kep_pid.run() on the simulate() log written as a CSV with its column names (full
    precision). The source is exec'd from the read-only data folder (not imported: no __pycache__ there)."""
    cols = ['timestamp_us', 'motor_on', 'still', 'sp_vx', 'sp_vy', 'pos_corr_vx', 'pos_corr_vy', 'vx', 'vy',
            'wz_cmd', 'wz_meas', 'pwm_fl', 'pwm_fr', 'pwm_rl', 'pwm_rr']
    sp = np.asarray(log['sp'], dtype=np.float32).astype(float)
    pos, imu = log['pos'], log['imu']
    path = os.path.join(str(tmp), 'C8-P-fwsim.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in range(len(log['t'])):
            w.writerow([int(round(log['t'][k] * 1e6)), int(log['en'][k]), int(pos[k, P['still']])] +
                       [repr(float(x)) for x in (sp[k, 0], sp[k, 1], pos[k, P['pos_corr_vx']], pos[k, P['pos_corr_vy']],
                                                 log['v_fw'][k, 0], log['v_fw'][k, 1], imu[k, M['wz_cmd']],
                                                 imu[k, M['wz_meas']])] + [int(x) for x in log['pwm'][k]])
    ns = {'__name__': 'kiem_kep_pid_goc', '__file__': GOC}
    with open(GOC, encoding='utf-8') as f:
        exec(compile(f.read(), GOC, 'exec'), ns)
    ns['ILIM'].update(dict(zip(AXES, ilim)))
    return ns['run'](path)


# Cross-check of the DLL counter against the rebuild of phan_tich_kep_pid (spec S03 v8.6 section 2.1) on S03-like
# PID runs. name: (v, ki_xy or None = firmware default, kep_n in the O1 window (vx, vy, wz), kep_mau); the numbers
# are those of GOLDEN_ID (python fwsim/tests/test_fwsim_kep.py --scan)
KIEM_CHEO = {'v02_mac_dinh': (0.2, None, (1323, 0, 0), 3206),
             'v08_ki_xy_1500': (0.8, 1500.0, (0, 0, 40), 1195),
             'v08_mac_dinh': (0.8, None, (0, 0, 0), 1096)}
_RUNS = {}


def kiem_cheo_run(fw, name):
    """(log, integral limits, t_end) of one cross-check run; 'golden' is the golden run. Simulated once."""
    if name not in _RUNS:
        if name == 'golden':
            _RUNS[name] = (golden_run(fw), GOLDEN_ILIM, 4.0)
        else:
            v, ki_xy = KIEM_CHEO[name][:2]
            d0 = s03_defaults(fw)
            log, ilim, res = s03_run(fw, v, d0 if ki_xy is None else (d0[0], ki_xy) + d0[2:])
            _RUNS[name] = (log, ilim, res['t_end'])
    return _RUNS[name]


@pytest.mark.parametrize('name', ['golden'] + list(KIEM_CHEO))
def test_cross_check_structure(fw, name):
    """Holds for any firmware / plant: the DLL integral is the float32 sum of pid.c line 41 clamped (replayed from
    the DLL's own integral of the tick before), the self-check is clean, no differing tick of an unexplained
    kind, and kep_indicator counts the DLL flags of the window."""
    log, ilim, t_end = kiem_cheo_run(fw, name)
    r = so_sanh(log, ilim, t_end)
    ind = fwsim.kep_indicator(log, t_end)
    assert r['replay32_khac_dll'] == 0 and r['kep_lech'] == 0
    assert ind['kep_mau'] == r['n_motor_on_win']
    for a in AXES:
        assert r[a]['khac'] == 0
        assert ind[f'kep_n_{a}'] == r[a]['dll_win']


@pytest.mark.parametrize('name', ['golden'] + list(KIEM_CHEO))
def test_cross_check_tick_by_tick_as_recorded(fw, name):
    """On the firmware / plant of GOLDEN_ID: the DLL flags equal the rebuild of phan_tich_kep_pid on every tick
    of the log (so the counts are equal, whole log and O1 window), with the recorded counts."""
    if identity(fw) != GOLDEN_ID:
        pytest.skip(f'cross-check recorded on {GOLDEN_ID}, this is {identity(fw)}')
    log, ilim, t_end = kiem_cheo_run(fw, name)
    r = so_sanh(log, ilim, t_end)
    assert r['n_tick_lech'] == 0 and r['tick_lech'] == []
    for a in AXES:
        assert r[a]['dll'] == r[a]['dung_lai'] and r[a]['dll_win'] == r[a]['dung_lai_win']
    ind = fwsim.kep_indicator(log, t_end)
    n_win, mau = ((209, 0, 15), 400) if name == 'golden' else KIEM_CHEO[name][2:]
    assert tuple(ind[f'kep_n_{a}'] for a in AXES) == n_win and ind['kep_mau'] == mau
    if name != 'golden':
        assert tuple(ilim) == (0.5, 0.5, 1.0)


@pytest.mark.parametrize('name', ['golden'] + list(KIEM_CHEO))
def test_port_gives_the_counts_of_the_original_script(fw, tmp_path, name):
    """dung_lai (the per-tick port used above) against the original kiem_kep_pid.run() on the same log."""
    if not os.path.isfile(GOC):
        pytest.skip(f'{GOC} not on this machine')
    log, ilim, _ = kiem_cheo_run(fw, name)
    hit, _, post = dung_lai(log, ilim)
    goc = script_goc(log, tmp_path, ilim)
    assert goc['cham_kep_I'] == {a: int(hit[:, i].sum()) for i, a in enumerate(AXES)}
    assert goc['tick_motor_on'] == int((log['en'] == 1).sum())
    assert [goc[f'I_max_{a}'] for a in AXES] == [round(float(np.abs(post[:, i]).max()), 4) for i in range(3)]


# Codex C3-12 / C3-13: trace DLL, hold-branch mutants of vel_ctrl.c, kep-free pair

TRACE_C = r"""/* TEST only (fwsim/tests/test_fwsim_kep.py): traces the real pid_update_* calls of vel_ctrl_step
 * on the firmware PID (vel_ctrl_pid()): per call the axis count, the integral right after the clamp
 * of pid.c and integral_limit. fwtest_trace_clear() before a tick, fwtest_trace_last() reads the
 * 9 doubles. With TRACE_WRAP the symbols come from -Wl,--wrap=pid_update_*; else vel_ctrl.c alone is
 * compiled with -Dpid_update_*=fwtest_pid_update_* and this file defines them. The replay of fw_sim.c
 * kep_step runs the same functions on a copy: excluded by the pointer test c == vel_ctrl_pid(). */
#include "vel_ctrl.h"
#include <string.h>
#ifdef TRACE_WRAP
#define T_REAL(x) __real_##x
#define T_WRAP(x) __wrap_##x
float __real_pid_update_vx(pid_controller_t *c, float sp, float actual, float dt);
float __real_pid_update_vy(pid_controller_t *c, float sp, float actual, float dt);
float __real_pid_update_wz(pid_controller_t *c, float sp, float actual, float dt);
#else
#define T_REAL(x) x
#define T_WRAP(x) fwtest_##x
#endif
/* calls vx, vy, wz; integral right after the real clamp vx, vy, wz; integral_limit vx, vy, wz */
static double tr[9];
__declspec(dllexport) void fwtest_trace_clear(void) { memset(tr, 0, sizeof(tr)); }
__declspec(dllexport) void fwtest_trace_last(double out[9]) { memcpy(out, tr, sizeof(tr)); }
#define DEF_TRACE(ax, i) \
    float T_WRAP(pid_update_##ax)(pid_controller_t *c, float sp, float actual, float dt) { \
        float u = T_REAL(pid_update_##ax)(c, sp, actual, dt); \
        if (c == vel_ctrl_pid()) { tr[i] += 1; tr[3 + i] = c->ax.integral; tr[6 + i] = c->ax.integral_limit; } \
        return u; \
    }
DEF_TRACE(vx, 0)
DEF_TRACE(vy, 1)
DEF_TRACE(wz, 2)
"""

_BUILD_JSON = os.path.join(fwsim.HERE, 'build', 'fwsim_build.json')
_WRAP_OK = None     # None: --wrap untried; True: it records calls; False: renamed symbols needed


def _fw_tree():
    """The AGV_FIRMWARE tree (read-only): the first candidate with src/controller/vel_ctrl.c."""
    for d in (os.path.join(fwsim.HERE, '..', '..', 'AGV_FIRMWARE'),
              'D:/CLAUDE_THESIS/03_CODE/AGV_FIRMWARE'):
        if os.path.isfile(os.path.join(d, 'src', 'controller', 'vel_ctrl.c')):
            return d
    return None


def _gcc(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'gcc failed ({r.returncode}): {r.stderr[-1500:]}')


def build_variant(out_dll, fw_sim_c=None, defines=(), trace=False, vel_ctrl_c=None):
    """One gcc build of a variant DLL, mirroring build_fwsim.main(): flags and the firmware source
    list from fwsim/build/fwsim_build.json, -I fwsim/shim then the firmware include dirs; sources are
    that list (controller/vel_ctrl.c replaced by vel_ctrl_c when given) + shim/fw_shim.c + fw_sim_c
    (default fwsim/fw_sim.c); a -D per define. trace=True also links the TRACE_C unit, with --wrap of
    the three pid_update symbols while _WRAP_OK allows it, else vel_ctrl.c alone is compiled with
    -Dpid_update_*=fwtest_pid_update_* (the trace unit then defines them). RuntimeError on failure."""
    fw = _fw_tree()
    info = json.load(open(_BUILD_JSON))
    flags = info['flags'].split()
    inc = ['-I', os.path.join(fwsim.HERE, 'shim')]
    for d in build_fwsim.FW_INCLUDES:
        inc += ['-I', os.path.join(fw, 'src', d)]
    vc = vel_ctrl_c or os.path.join(fw, 'src', 'controller/vel_ctrl.c')
    srcs = [vc if s == 'controller/vel_ctrl.c' else os.path.join(fw, 'src', s) for s in info['sources']]
    srcs += [os.path.join(fwsim.HERE, 'shim', 'fw_shim.c'),
             fw_sim_c or os.path.join(fwsim.HERE, 'fw_sim.c')]
    cmd = ['gcc'] + flags + inc + ['-D' + str(x) for x in defines]
    if trace:
        trace_c = os.path.splitext(out_dll)[0] + '_trace.c'
        with open(trace_c, 'w') as f:
            f.write(TRACE_C)
        srcs.append(trace_c)
        if _WRAP_OK is not False:
            cmd += ['-DTRACE_WRAP',
                    '-Wl,--wrap=pid_update_vx,--wrap=pid_update_vy,--wrap=pid_update_wz']
        else:
            i = srcs.index(vc)
            obj = os.path.splitext(out_dll)[0] + '_vel_ctrl.o'
            _gcc(['gcc'] + [x for x in flags if x != '-shared'] + inc +
                 ['-Dpid_update_vx=fwtest_pid_update_vx', '-Dpid_update_vy=fwtest_pid_update_vy',
                  '-Dpid_update_wz=fwtest_pid_update_wz', '-c', srcs[i], '-o', obj])
            srcs[i] = obj
    _gcc(cmd + srcs + ['-o', out_dll])
    return out_dll


class TraceSim(fwsim.FirmwareSim):
    """FirmwareSim on a trace DLL: tick() clears the trace, runs the tick and appends trace_last()
    to self.vet (per tick: calls vx, vy, wz; integral after the real clamp; integral_limit)."""

    def __init__(self, dll):
        super().__init__(dll)
        self.lib.fwtest_trace_clear.argtypes = []
        self.lib.fwtest_trace_last.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.vet = []

    def trace_last(self):
        out = (ctypes.c_double * 9)()
        self.lib.fwtest_trace_last(out)
        return list(out)

    def tick(self, t_us, counts, sp):
        self.lib.fwtest_trace_clear()
        r = super().tick(t_us, counts, sp)
        self.vet.append(self.trace_last())
        return r


def _trace_works(dll):
    """True when one PID tick on the DLL records the three pid_update calls: a trace build that
    records nothing is useless, so the fixture rebuilds it with the renamed-symbol fallback."""
    sim = TraceSim(dll)
    sim.reset()
    sim.set_mode('PID')
    sim.tick(DT_US, [0, 0, 0, 0], (0.0, 0.0, 0.0))
    return sim.vet[-1][:3] == [1.0, 1.0, 1.0]


@pytest.fixture(scope='module')
def variants(tmp_path_factory):
    """Lazy build-and-cache of variant DLLs (trace, FWSIM_NO_KEP, vel_ctrl.c mutants), each get() one
    gcc run of build_variant into a private dir; every variant has a different base name (several are
    loaded in one process). A trace DLL is probed once: when --wrap left no calls the fallback is
    built under its own name <name>_du_phong.dll with the renamed symbols (_WRAP_OK = False):
    the first DLL stays loaded and is not overwritten."""
    if shutil.which('gcc') is None:
        pytest.skip('gcc not on PATH')
    if _fw_tree() is None:
        pytest.skip('AGV_FIRMWARE tree not found')
    if not os.path.isfile(_BUILD_JSON):
        pytest.skip('fwsim/build/fwsim_build.json not found')
    out_dir = str(tmp_path_factory.mktemp('c3dll'))
    cache = {}

    def get(name, **kw):
        global _WRAP_OK
        if name in cache:
            return cache[name]
        dll = os.path.join(out_dir, name + '.dll')
        try:
            build_variant(dll, **kw)
        except RuntimeError:
            if not kw.get('trace') or _WRAP_OK is False:
                raise
            _WRAP_OK = False                         # this ld cannot --wrap into the DLL: fallback
            dll = os.path.join(out_dir, name + '_du_phong.dll')   # own name: the loaded DLL is not overwritten
            build_variant(dll, **kw)
        if kw.get('trace'):
            if not _trace_works(dll):
                if _WRAP_OK is False:
                    raise RuntimeError('the trace records no pid_update calls')
                _WRAP_OK = False                     # --wrap linked but intercepts nothing: fallback
                dll = os.path.join(out_dir, name + '_du_phong.dll')   # own name: the loaded DLL is not overwritten
                build_variant(dll, **kw)
                if not _trace_works(dll):
                    raise RuntimeError('the trace records no pid_update calls (renamed symbols too)')
            elif _WRAP_OK is None:
                _WRAP_OK = True
        cache[name] = dll
        return dll
    return get


def kep_theo_vet(tr):
    """The KEP_COLS row a trace row implies: calls (0, 0, 0) -> seven zeros (the PID did not run);
    calls (1, 1, 1) -> flag |I| == lim per axis, the three integrals after the real clamp, 0.0.
    Anything else (a partial update) is an assertion failure: the DLL cannot model it either."""
    calls, integ, lim = tr[:3], tr[3:6], tr[6:9]
    if calls == [0.0, 0.0, 0.0]:
        return [0.0] * len(fwsim.KEP_COLS)
    assert calls == [1.0, 1.0, 1.0], f'trace calls {calls}'
    return [float(abs(integ[i]) == lim[i]) for i in range(3)] + list(integ) + [0.0]


def boot_trace(sim, mode='PID', **params):
    """boot() on a trace DLL: tick(sp, counts=(0, 0, 0, 0)) -> (kep row, still flag, trace row, v_fw).
    counts: cumulative quadrature counts per wheel (1320 counts/rev, wheel radius 0.0485 m,
    FK vx = r/4 * sum(omega): the same step on all four wheels is pure vx; -54 on each in one
    62.5 ms tick is vx ~ -0.2 m/s, read from v_fw)."""
    sim.reset()
    sim.set_mode(mode)
    for n, v in dict(pos_on=0, **params).items():
        sim.set_param(n, v)
    t_us = [0]

    def tick(sp, counts=(0, 0, 0, 0)):
        t_us[0] += DT_US
        _, v_fw, _, _, _ = sim.tick(t_us[0], list(counts), sp)
        return sim.kep_last(), sim.pos_last()[P['still']], sim.vet[-1], v_fw
    return tick


def codex_ticks(t):
    """The scenario of Codex C3-12 (I_vx = 0.49, then the hold entry while moving), direct ticks at
    dt = 0.0625 s, PID defaults, params still_v=0.5. Returns the (kep, still, trace, v_fw) rows."""
    out = []
    for _ in range(7):
        out.append(t((1.0, 0.0, 0.0)))                     # I_vx = 0.4375 exactly
    out.append(t((0.84, 0.0, 0.0)))                        # I_vx ~ 0.49 (the reviewer's value)
    for _ in range(8):
        out.append(t((0.0, 0.0, 0.0)))                     # integral unchanged, still 0
    out.append(t((0.0, 0.0, 0.0), (-54,) * 4))             # ENTRY: still 1, the sum crosses 0.5
    out.append(t((0.0, 0.0, 0.0), (-54,) * 4))             # hold, wheels stopped
    out.append(t((0.0, 0.0, 0.0), (-108,) * 4))            # hold while moving
    out.append(t((1.0, 0.0, 0.0), (-108,) * 4))            # EXIT
    return out


# hold-branch mutants of vel_ctrl.c for C3-12 (test-only patched copies; each anchor must occur once)
HOLD = "    if (g_still) {\n        pid_reset(&g_pid);\n        adrc_reset(&g_adrc);\n"
CTRL = "    float wz_meas = o.imu.wz_meas;\n"
MODE = "    if (g_mode == CTRL_ADRC) {\n"
_MUTANTS = {
    'reset_truoc_cap_nhat':      # the hold branch resets the PID BEFORE the update, never after
        [(HOLD, "    if (g_still) {\n"),
         (CTRL, "    if (g_still) { pid_reset(&g_pid); adrc_reset(&g_adrc); }\n" + CTRL)],
    'bo_qua_khong_reset':        # the hold branch skips the update and never resets (integral frozen)
        [(HOLD, "    if (g_still) {\n"),
         (MODE, "    if (g_still) {\n    } else if (g_mode == CTRL_ADRC) {\n")],
    'reset_roi_bo_qua':          # the hold branch skips the update; the reset after it is kept
        [(MODE, "    if (g_still) {\n    } else if (g_mode == CTRL_ADRC) {\n")],
}


def mutant_vel_ctrl(name, out_dir):
    """The patched copy of the firmware vel_ctrl.c of one mutant, written into out_dir; skips when an
    anchor does not occur exactly once (the firmware changed under this test)."""
    src = open(os.path.join(_fw_tree(), 'src', 'controller', 'vel_ctrl.c'), encoding='utf-8').read()
    for old, new in _MUTANTS[name]:
        if src.count(old) != 1:
            pytest.skip(f'vel_ctrl.c changed: anchor {old!r} occurs {src.count(old)} times')
        src = src.replace(old, new)
    path = os.path.join(str(out_dir), 'vel_ctrl.c')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(src)
    return path


def _bits_eq(a, b):
    """dtype, shape and raw bytes equal."""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape \
        and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def test_trace_dll_gives_the_log_of_the_production_dll(fw, glog, variants):
    """The trace instrumentation records without disturbing: the golden run through the trace DLL of
    the real firmware is the production log bit for bit (every key, 'kep' too)."""
    log = golden_run(TraceSim(variants('fwsim_trace', trace=True)))
    assert list(log) == list(glog)
    for k in glog:
        assert _bits_eq(log[k], glog[k]), k


def test_trace_enter_hold_exit_still(variants):
    """C3-12 on the real firmware: the hold entry / hold / exit ticks of the scenario, each against
    the trace of the same tick."""
    sim = TraceSim(variants('fwsim_trace', trace=True))
    rows = codex_ticks(boot_trace(sim, still_v=0.5))
    for kep, still, tr, v_fw in rows:
        assert kep == kep_theo_vet(tr)
    assert rows[7][0][K['kep_vx']] == 0.0 and 0.4899 < rows[7][0][K['kep_I_vx']] < 0.4901
    kep, still, tr, v_fw = rows[16]                          # entry: the update ran, then the reset
    assert still == 1.0 and tr[:3] == [1.0, 1.0, 1.0]
    assert -0.21 < v_fw[0] < -0.19                           # vx ~ -0.2 m/s: e crosses the 0.5 limit
    assert tr[3] == tr[6] == 0.5
    assert (kep[K['kep_vx']], kep[K['kep_I_vx']], kep[K['kep_lech']]) == (1.0, 0.5, 0.0)
    kep, still, tr, _ = rows[17]                             # hold, wheels stopped
    assert still == 1.0 and kep[K['kep_I_vx']] == 0.0 and kep[K['kep_vx']] == 0.0
    kep, still, tr, _ = rows[18]                             # hold while moving (update ran, reset after)
    assert still == 1.0 and tr[3] != 0.0 and kep[K['kep_I_vx']] == tr[3]
    kep, still, tr, _ = rows[19]                             # exit
    assert still == 0.0 and kep[K['kep_I_vx']] == 0.0625


def test_trace_motor_off_on(variants):
    """Motor off/on: the edge reset of vel_ctrl_step is what kep_step replays (on_prev), the trace
    agrees on every tick; a hold entry with the integral on the limit and zero error too."""
    sim = TraceSim(variants('fwsim_trace', trace=True))
    t = boot_trace(sim)
    for _ in range(9):
        kep, _, tr, _ = t((1.0, 0.0, 0.0))
        assert kep == kep_theo_vet(tr)
    assert kep[K['kep_vx']] == 1.0 and kep[K['kep_I_vx']] == 0.5
    sim.motor_enable(False)
    for _ in range(2):
        kep, _, tr, _ = t((1.0, 0.0, 0.0))
        assert kep == kep_theo_vet(tr) == [0.0] * len(fwsim.KEP_COLS)
    sim.motor_enable(True)
    kep, _, tr, _ = t((1.0, 0.0, 0.0))
    assert kep == kep_theo_vet(tr) and kep[K['kep_I_vx']] == 0.0625 and kep[K['kep_vx']] == 0.0
    t = boot_trace(sim)
    for _ in range(9):
        kep, still, tr, _ = t((1.0, 0.0, 0.0))
        assert kep == kep_theo_vet(tr)
    for _ in range(20):
        kep, still, tr, _ = t((0.0, 0.0, 0.0))
        assert kep == kep_theo_vet(tr)
        if still == 1.0:
            break
    else:
        pytest.fail('no standstill hold within 20 zero-setpoint ticks')
    assert kep[K['kep_vx']] == 1.0 and kep[K['kep_I_vx']] == 0.5    # the entry tick reports the hit
    kep, still, tr, _ = t((0.0, 0.0, 0.0))
    assert kep == kep_theo_vet(tr) and still == 1.0
    sim.motor_enable(False)
    kep, _, tr, _ = t((0.0, 0.0, 0.0))
    assert kep == kep_theo_vet(tr) == [0.0] * len(fwsim.KEP_COLS)
    sim.motor_enable(True)
    kep, _, tr, _ = t((1.0, 0.0, 0.0))
    assert kep == kep_theo_vet(tr) and kep[K['kep_I_vx']] == 0.0625


def test_trace_pid_adrc_pid(variants):
    """PID -> ADRC -> PID between ticks: no pid_update calls under ADRC, the integrals restart
    from 0 after vel_ctrl_set_mode; the trace agrees on every tick."""
    sim = TraceSim(variants('fwsim_trace', trace=True))
    t = boot_trace(sim)
    for _ in range(9):
        kep, _, tr, _ = t((1.0, -1.0, 1.0))
        assert kep == kep_theo_vet(tr)
    sim.set_mode('ADRC')
    for _ in range(3):
        kep, _, tr, _ = t((1.0, -1.0, 1.0))
        assert kep == kep_theo_vet(tr) == [0.0] * len(fwsim.KEP_COLS)
    sim.set_mode('PID')
    for k in range(1, 4):
        kep, _, tr, _ = t((1.0, -1.0, 1.0))
        assert kep == kep_theo_vet(tr)
        assert kep[3:6] == [0.0625 * k, -0.0625 * k, 0.0625 * k]


def test_trace_gain_change_mid_run(variants):
    """fwsim_set_pid between ticks: the clamp follows integral_limit immediately, on both sides of
    the exact limit; the trace agrees on every tick and kep_lech stays 0."""
    sim = TraceSim(variants('fwsim_trace', trace=True))
    t = boot_trace(sim)
    for _ in range(7):
        kep, _, tr, _ = t((1.0, 0.0, 0.0))
        assert kep == kep_theo_vet(tr)
    g = sim.get_pid(0)
    sim.set_pid(0, g['kp'], g['ki'], g['kd'], 0.25, g['out_limit'])
    kep, _, tr, _ = t((1.0, 0.0, 0.0))
    assert kep == kep_theo_vet(tr) and (kep[K['kep_vx']], kep[K['kep_I_vx']]) == (1.0, 0.25)
    sim.set_pid(0, g['kp'], g['ki'], g['kd'], 1.0, g['out_limit'])
    kep, _, tr, _ = t((1.0, 0.0, 0.0))
    assert kep == kep_theo_vet(tr) and (kep[K['kep_vx']], kep[K['kep_I_vx']]) == (0.0, 0.3125)
    sim.set_pid(0, 3.0, 2.0, g['kd'], 1.0, g['out_limit'])
    kep, _, tr, _ = t((1.0, 0.0, 0.0))
    assert kep == kep_theo_vet(tr) and (kep[K['kep_vx']], kep[K['kep_I_vx']]) == (0.0, 0.375)
    sim.set_pid(0, 3.0, 2.0, g['kd'], 0.375, g['out_limit'])
    kep, _, tr, _ = t((0.0, 0.0, 0.0))
    assert kep == kep_theo_vet(tr) and (kep[K['kep_vx']], kep[K['kep_I_vx']]) == (1.0, 0.375)
    nxt = float(np.nextafter(np.float32(0.375), np.float32(1.0)))
    sim.set_pid(0, 3.0, 2.0, g['kd'], nxt, g['out_limit'])
    kep, _, tr, _ = t((0.0, 0.0, 0.0))
    assert kep == kep_theo_vet(tr) and kep[K['kep_vx']] == 0.0
    assert kep[K['kep_lech']] == 0.0


def test_trace_simulate_runs(variants):
    """kep == kep_theo_vet(trace) on every tick of full simulate() runs: a run with a standstill
    hold (entry, hold, exit ticks) and the S03-like run that flags."""
    sim = TraceSim(variants('fwsim_trace', trace=True))
    sim.vet.clear()

    def sp(t_):
        return (0.3, 0.0, 0.3) if t_ < 1.5 or t_ >= 3.5 else (0.0, 0.0, 0.0)
    log = fwsim.simulate(sim, sp, 5.0, mode='PID', imu={}, pid=GOLDEN_PID)
    assert len(sim.vet) == len(log['t'])
    for k in range(len(log['t'])):
        assert list(log['kep'][k]) == kep_theo_vet(sim.vet[k])
    still, d = log['pos'][:, P['still']], np.diff(log['pos'][:, P['still']])
    assert np.any(d == 1) and np.any(d == -1) and np.count_nonzero(still == 1) > 1   # entry, exit, hold
    sim.vet.clear()
    d0 = s03_defaults(sim)
    log, ilim, res = s03_run(sim, 0.8, (d0[0], 1500.0) + d0[2:])
    assert len(sim.vet) == len(log['t'])
    for k in range(len(log['t'])):
        assert list(log['kep'][k]) == kep_theo_vet(sim.vet[k])
    assert np.any(log['kep'][:, :3] == 1)


@pytest.mark.parametrize('mutant', ['reset_truoc_cap_nhat', 'bo_qua_khong_reset'])
def test_mutant_hold_is_flagged(variants, tmp_path, mutant):
    """Codex C3-12: a hold branch that resets the PID before the update, or never resets it, leaves
    the firmware PID state not in the reset state on the entry tick: kep_lech must be set there."""
    vc = mutant_vel_ctrl(mutant, tmp_path)
    sim = TraceSim(variants('fwsim_m_' + mutant, trace=True, vel_ctrl_c=vc))
    rows = codex_ticks(boot_trace(sim, still_v=0.5))[:17]
    kep, still, tr, _ = rows[16]
    assert still == 1.0
    assert tr[:3] == [0.0, 0.0, 0.0] or abs(tr[3]) != tr[6]        # the real PID made no hit on vx
    # the hit the replay claims in the reviewer's example (the real PID made none):
    assert (kep[K['kep_vx']], kep[K['kep_I_vx']]) == (1.0, 0.5)
    assert kep[K['kep_lech']] == 1.0
    log = dict(t=np.arange(len(rows)) * 0.0625, en=np.ones(len(rows)),
               kep=np.array([r[0] for r in rows], dtype=float))
    with pytest.raises(ValueError, match='kep_lech'):
        fwsim.kep_indicator(log, 1.0)


def test_mutant_reset_then_skip_is_seen_by_the_trace(variants, tmp_path):
    """reset_roi_bo_qua: the update is skipped but the reset kept, so the state after the step is the
    real one's and kep_lech cannot see it; the trace comparison of the other tests is what does."""
    vc = mutant_vel_ctrl('reset_roi_bo_qua', tmp_path)
    sim = TraceSim(variants('fwsim_m_reset_roi_bo_qua', trace=True, vel_ctrl_c=vc))
    rows = codex_ticks(boot_trace(sim, still_v=0.5))
    kep, still, tr, _ = rows[16]
    assert still == 1.0 and tr[:3] == [0.0, 0.0, 0.0]
    assert kep[:6] != kep_theo_vet(tr)[:6]


def test_bit_exact_keys_definition(glog):
    """C3-13: the explicit bit-exact channel list is every channel of simulate() but 'kep'."""
    assert fwsim.BIT_EXACT_KEYS == OLD_KEYS
    assert list(glog) == list(OLD_KEYS) + ['kep']


def _cap_runs(fw_cu, fw_moi):
    """The four run pairs of test_dll_pair_without_and_with_kep and of the --pair CLI:
    (label, old log, new log); old = the DLL without the kep channel, new = the production one.
    Both sides of every pair are simulated directly here (no _RUNS cache)."""
    sp = lambda t: (0.3, 0.0, 0.0)      # noqa: E731
    return [('golden', golden_run(fw_cu), golden_run(fw_moi)),
            ('adrc', fwsim.simulate(fw_cu, sp, 1.0, mode='ADRC'),
                     fwsim.simulate(fw_moi, sp, 1.0, mode='ADRC')),
            ('motor_tat', fwsim.simulate(fw_cu, sp, 0.5, mode='PID', motor_on=False),
                          fwsim.simulate(fw_moi, sp, 0.5, mode='PID', motor_on=False)),
            ('s03', s03_run(fw_cu, 0.8, s03_defaults(fw_cu))[0],
                    s03_run(fw_moi, 0.8, s03_defaults(fw_moi))[0])]


def _dll_tin(path):
    """'<path> sha256=<hex>' of a DLL, for the messages of the pair tests."""
    return f'{os.path.abspath(path)} sha256={build_fwsim.file_sha256(path)}'


def so_cap(old_log, new_log):
    """C3-13 pair check of an old log (no kep channel) and a new one: the key lists, the number of
    BIT_EXACT channels differing in dtype/shape/bytes, whether s03_doe.log_digest of the new log
    restricted to BIT_EXACT_KEYS equals the old digest, whether the full digests differ by exactly
    'kep', and the npz comparison (old keys bit-equal, 'kep' the only extra file)."""
    keys = list(fwsim.BIT_EXACT_KEYS)
    d_cu, d_moi = s3.log_digest(old_log), s3.log_digest(new_log)
    z_cu = np.load(io.BytesIO(s3.npz_bytes(old_log)))
    z_moi = np.load(io.BytesIO(s3.npz_bytes(new_log)))
    return dict(old_keys=list(old_log), new_keys=list(new_log),
                n_lech=int(sum(not _bits_eq(old_log[k], new_log[k]) for k in keys)),
                digest_win_bang=bool(d_cu == {k: d_moi[k] for k in keys}),
                digest_khac_chi_kep=bool(set(d_moi) - set(d_cu) == {'kep'}
                                         and all(d_cu[k] == d_moi[k] for k in d_cu)),
                npz=bool(sorted(z_cu.files) == sorted(keys) and sorted(z_moi.files) == sorted(keys + ['kep'])
                         and all(_bits_eq(z_cu[k], z_moi[k]) for k in keys)))


def test_dll_pair_without_and_with_kep(fw, variants, tmp_path):
    """C3-13: a DLL built with -DFWSIM_NO_KEP is the DLL without the counter (no fwsim_kep_last):
    on the BIT_EXACT channels its logs equal the production DLL's bit for bit, and the whole-log
    outputs differ by the 'kep' entry only."""
    ref_path = variants('fwsim_nokep', defines=('FWSIM_NO_KEP',))
    ref = fwsim.FirmwareSim(ref_path)
    assert ref.kep_cols == 0 and not hasattr(ref.lib, 'fwsim_kep_last')
    msg = f'old {_dll_tin(ref_path)}; new {_dll_tin(_DLL)}'
    runs = _cap_runs(ref, fw)
    for name, old_log, new_log in runs:
        r = so_cap(old_log, new_log)
        assert r['old_keys'] == list(fwsim.BIT_EXACT_KEYS), f'{name}: {msg}'
        assert r['new_keys'] == r['old_keys'] + ['kep'], f'{name}: {msg}'
        assert r['n_lech'] == 0 and r['digest_win_bang'] and r['digest_khac_chi_kep'] and r['npz'], \
            f'{name}: {msg}'
    _, old_log, new_log = runs[0]          # the golden pair, written by save_log_csv and compared as text
    s3.save_log_csv(str(tmp_path), 'cap_cu', old_log)
    s3.save_log_csv(str(tmp_path), 'cap_moi', new_log)
    cu = [r.split(',') for r in open(tmp_path / 'cap_cu.csv', encoding='utf-8').read().splitlines()]
    moi = [r.split(',') for r in open(tmp_path / 'cap_moi.csv', encoding='utf-8').read().splitlines()]
    ik = [i for i, c in enumerate(moi[0]) if c.startswith('kep_')]
    assert [c for i, c in enumerate(moi[0]) if i not in ik] == cu[0]
    assert [[x for i, x in enumerate(r) if i not in ik] for r in moi[1:]] == cu[1:]


# fw_sim.c of commit 62e2fa5 (before the kep channel), a file of the repo now, placed by CODE
NGUON_CU = os.path.join(fwsim.HERE, 'tests', 'data', 'fw_sim_62e2fa5.c')
NGUON_CU_SHA = 'a93b6d0928a83ea3c7301fd98a57ac9c67d2edfa88109f1deaa6310a9e3e4b4d'   # sha256, CR stripped


def _doc_nguon_cu():
    """The raw bytes of NGUON_CU (read at call time so a monkeypatched path is honored): missing
    or wrong CR-stripped sha256 is an assertion failure, never a skip."""
    assert os.path.isfile(NGUON_CU), \
        f'{NGUON_CU} missing: the old wrapper source is a file of the repo (fwsim/tests/data); ' \
        f'the pair cannot run without it'
    with open(NGUON_CU, 'rb') as f:
        raw = f.read()
    sha = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
    assert sha == NGUON_CU_SHA, f'{NGUON_CU}: CR-stripped sha256 {sha} != {NGUON_CU_SHA}'
    return raw


def _dll_nap(lib):
    """(path, sha256) of the DLL the process really loaded (GetModuleFileNameW on the ctypes
    handle; lib._name elsewhere): the file Windows mapped, not the path CDLL was given."""
    if os.name == 'nt':
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
        k32.GetModuleFileNameW.restype = ctypes.c_uint32
        buf = ctypes.create_unicode_buffer(32768)
        n = k32.GetModuleFileNameW(ctypes.c_void_p(lib._handle), buf, ctypes.c_uint32(len(buf)))
        if n == 0:
            raise ctypes.WinError(ctypes.get_last_error())
        path = buf.value
    else:
        path = lib._name
    return path, build_fwsim.file_sha256(path)


def _fw_src_sha():
    """One sha256 for the firmware tree a variant build compiles: sha256 of the concatenation of
    '<rel>\\n<file sha256>\\n' over the 'sources' list of fwsim/build/fwsim_build.json, in order."""
    info = json.load(open(_BUILD_JSON))
    h = hashlib.sha256()
    for rel in info['sources']:
        h.update(f"{rel}\n{build_fwsim.file_sha256(os.path.join(_fw_tree(), 'src', rel))}\n".encode())
    return h.hexdigest()


def _tin_cap(fw, nap_cu, nap_moi):
    """The identification of both sides of the pre-kep pair, for the assert messages: the two
    wrapper sources (old: path + raw and CR-stripped sha256; new: path + sha256 now and the one
    fwsim_build.json recorded at build), the firmware tree of each side (new: commit + dirty of
    the build json; old: the tree path + the combined sha256 of the compiled sources), the gcc
    of each build (old: the gcc on PATH now; new: the recorded compiler + flags) and the two
    DLLs the process really loaded (plus the dll_sha256 the json recorded)."""
    gcc = shutil.which('gcc')
    gcc_ver = subprocess.run([gcc, '--version'], capture_output=True,
                             text=True).stdout.splitlines()[0]
    esp = fw.build_info.get('esp32') or {}
    raw = _doc_nguon_cu()
    sach_cr = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
    return (f'nguon cu {os.path.abspath(NGUON_CU)} sha256={hashlib.sha256(raw).hexdigest()} sha256_sach_cr={sach_cr}; '
            f'nguon moi {os.path.abspath(os.path.join(fwsim.HERE, "fw_sim.c"))} '
            f'sha256={build_fwsim.file_sha256(os.path.join(fwsim.HERE, "fw_sim.c"))} '
            f'luc_build={fw.build_info["wrapper_sha256"]["fw_sim.c"]}; '
            f'firmware cu {os.path.abspath(_fw_tree())} nguon_sha256={_fw_src_sha()}; '
            f'firmware moi commit={esp.get("commit")} dirty={esp.get("dirty")}; '
            f'gcc cu {gcc} | {gcc_ver}; '
            f'gcc moi {fw.build_info.get("compiler")} flags {fw.build_info.get("flags")}; '
            f'dll cu nap {nap_cu[0]} sha256={nap_cu[1]}; '
            f'dll moi nap {nap_moi[0]} sha256={nap_moi[1]} '
            f'ghi_luc_build={fw.build_info.get("dll_sha256")}')


def test_dll_pair_with_the_source_before_kep(fw, variants, tmp_path):
    """The pair against the real pre-kep source: a DLL built of the fw_sim.c of 62e2fa5 has no kep
    channel, and the real --pair CLI reports every run identical on the BIT_EXACT keys. The msg
    identifies both wrapper sources, the firmware tree of each side, the toolchain and the DLLs
    really loaded."""
    _doc_nguon_cu()
    shutil.copytree(os.path.join(fwsim.HERE, 'shim'), str(tmp_path / 'shim'))
    src_cu = str(tmp_path / 'fw_sim_62e2fa5.c')
    shutil.copyfile(os.path.abspath(NGUON_CU), src_cu)
    old = variants('fwsim_62e2fa5', fw_sim_c=src_cu)
    sim = fwsim.FirmwareSim(old)
    assert sim.kep_cols == 0 and not hasattr(sim.lib, 'fwsim_kep_last')
    nap_cu = _dll_nap(sim.lib)
    del sim
    nap_moi = _dll_nap(fw.lib)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), '--pair', old],
                       capture_output=True, text=True, cwd=os.path.dirname(fwsim.HERE), timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    msg = _tin_cap(fw, nap_cu, nap_moi)
    assert build_fwsim.file_sha256(_DLL) == fw.build_info['dll_sha256'], msg
    lines = [json.loads(x) for x in r.stdout.splitlines() if x.strip()]
    assert [x['run'] for x in lines] == ['golden', 'adrc', 'motor_tat', 's03']
    for line in lines:
        m = f"{line['run']}: {msg}"
        assert line['old_keys'] == list(fwsim.BIT_EXACT_KEYS), m
        assert line['new_keys'] == line['old_keys'] + ['kep'], m
        assert line['n_lech'] == 0, m
        assert line['digest_win_bang'], m
        assert line['digest_khac_chi_kep'], m
        assert line['npz'], m
        assert line['dll_cu_sha256'] == build_fwsim.file_sha256(old), m
        assert line['dll_moi_sha256'] == build_fwsim.file_sha256(_DLL), m
        assert os.path.normcase(os.path.abspath(line['dll_cu_nap'])) \
            == os.path.normcase(os.path.abspath(old)), m
        assert os.path.normcase(os.path.abspath(line['dll_moi_nap'])) \
            == os.path.normcase(os.path.abspath(_DLL)), m
        assert line['dll_cu_nap_sha256'] == build_fwsim.file_sha256(old), m
        assert line['dll_moi_nap_sha256'] == build_fwsim.file_sha256(_DLL), m
    print(msg)


def test_old_source_is_in_the_repo():
    """The old wrapper source is a file of this repo (fwsim/tests/data): it exists and passes
    the pinned-sha check of _doc_nguon_cu."""
    assert os.path.normcase(os.path.dirname(os.path.abspath(NGUON_CU))) \
        == os.path.normcase(os.path.abspath(os.path.join(fwsim.HERE, 'tests', 'data')))
    assert isinstance(_doc_nguon_cu(), bytes)


@pytest.mark.parametrize('case', ['thieu', 'sai_sha', 'crlf'])
def test_old_source_check_rejects(tmp_path, monkeypatch, case):
    """_doc_nguon_cu fails (never skips) on a missing file or a wrong CR-stripped sha256; a CRLF
    copy still passes: the pinned sha is computed with CR stripped."""
    mod = sys.modules[__name__]
    if case == 'thieu':
        monkeypatch.setattr(mod, 'NGUON_CU', str(tmp_path / 'khong_co_nguon_cu.c'))
        with pytest.raises(AssertionError, match='missing'):
            _doc_nguon_cu()
        return
    raw = open(NGUON_CU, 'rb').read()
    path = str(tmp_path / 'fw_sim_62e2fa5.c')
    if case == 'sai_sha':
        with open(path, 'wb') as f:
            f.write(raw + b'\n')
        monkeypatch.setattr(mod, 'NGUON_CU', path)
        with pytest.raises(AssertionError, match=NGUON_CU_SHA):
            _doc_nguon_cu()
    else:                                        # crlf: the boundary case of the CR-stripped sha
        with open(path, 'wb') as f:
            f.write(raw.replace(b'\n', b'\r\n'))
        monkeypatch.setattr(mod, 'NGUON_CU', path)
        assert _doc_nguon_cu() == raw.replace(b'\n', b'\r\n')


def test_pair_fails_not_skips_without_the_old_source(fw, variants, tmp_path, monkeypatch):
    """Without the old source the pair test must fail, not skip (a skip would still look green in
    the report): pytest.skip is a BaseException, so it is caught explicitly."""
    mod = sys.modules[__name__]
    monkeypatch.setattr(mod, 'NGUON_CU', str(tmp_path / 'khong_co_nguon_cu.c'))
    try:
        test_dll_pair_with_the_source_before_kep(fw, variants, tmp_path)
    except pytest.skip.Exception as e:
        pytest.fail(f'skipped instead of failing: {e}')
    except AssertionError as e:
        assert 'missing' in str(e)
    else:
        pytest.fail('passed without the old source')


def test_cap_runs_simulates_both_sides(fw, variants, monkeypatch):
    """Codex C3-33: the new side of a pair is simulated on fw_moi here, not taken from the _RUNS
    cache (keyed by the case name only): a stale 'v08_mac_dinh' entry must not leak into 's03'."""
    ref_path = variants('fwsim_nokep', defines=('FWSIM_NO_KEP',))
    ref = fwsim.FirmwareSim(ref_path)
    stale = kiem_cheo_run(fw, 'v08_ki_xy_1500')   # a run another firmware defaults cached under the name
    monkeypatch.setitem(_RUNS, 'v08_mac_dinh', stale)
    runs = dict((n, (o, nw)) for n, o, nw in _cap_runs(ref, fw))
    o, nw = runs['s03']
    assert nw is not stale[0]
    r = so_cap(o, nw)
    msg = f's03: {_dll_tin(ref_path)}; {_dll_tin(_DLL)}'
    assert r['n_lech'] == 0 and r['digest_win_bang'] and r['npz'], msg
    assert _RUNS['v08_mac_dinh'] is stale


def _bien_the(dll):
    """'du_phong' for the renamed-symbol fallback (<name>_du_phong.dll), 'wrap' otherwise."""
    return 'du_phong' if os.path.basename(dll).endswith('_du_phong.dll') else 'wrap'


def _gia_lap_probe(monkeypatch, wrap_ld=True, wrap_bat=True, du_phong_bat=True):
    """Patch the probe chain of the variants fixture (all on the module globals): _WRAP_OK back
    to None, build_variant spied (built = [(out_dll, _WRAP_OK at the call)]), _gcc spied (cmds =
    every gcc command; wrap_ld False makes a --wrap link raise RuntimeError before gcc runs, as
    a rejecting ld would), _trace_works faked per probed variant: the real probe always runs
    first (the DLL is really loaded), then wrap_bat / du_phong_bat decide the reported result.
    Returns (built, cmds, probes); probes = [(_bien_the(dll), dll)]."""
    mod = sys.modules[__name__]
    monkeypatch.setattr(mod, '_WRAP_OK', None)
    built, cmds, probes = [], [], []
    real_build, real_gcc, real_works = mod.build_variant, mod._gcc, mod._trace_works

    def spy_build(out_dll, **kw):
        built.append((out_dll, mod._WRAP_OK))
        return real_build(out_dll, **kw)

    def spy_gcc(cmd):
        cmds.append(cmd)
        if not wrap_ld and any('--wrap' in a for a in cmd):
            raise RuntimeError("gcc failed (1): ld.exe: unrecognized option '--wrap=pid_update_vx'")
        return real_gcc(cmd)

    def fake_works(dll):
        bt = _bien_the(dll)
        probes.append((bt, dll))
        return real_works(dll) and (wrap_bat if bt == 'wrap' else du_phong_bat)

    monkeypatch.setattr(mod, 'build_variant', spy_build)
    monkeypatch.setattr(mod, '_gcc', spy_gcc)
    monkeypatch.setattr(mod, '_trace_works', fake_works)
    return built, cmds, probes


def test_trace_fallback_builds_a_separate_dll(variants, glog, monkeypatch):
    """Codex C3-33: when a --wrap trace DLL records no calls, the renamed-symbol fallback is built
    under its own name <name>_du_phong.dll: the first DLL stays loaded and is not overwritten."""
    mod = sys.modules[__name__]
    built, cmds, probes = _gia_lap_probe(monkeypatch, wrap_bat=False)
    dll = variants('fwsim_trace_thu', trace=True)   # the wrap name must not end in _du_phong
    assert len(built) == 2, built
    msg = f'trace {_dll_tin(built[0][0])}; {_dll_tin(built[1][0])}'
    assert built[0][1] is None and built[1][1] is False, msg
    assert built[1][0] == built[0][0].replace('.dll', '_du_phong.dll'), msg
    assert dll == built[1][0], msg
    assert probes == [('wrap', built[0][0]), ('du_phong', built[1][0])], msg
    assert os.path.isfile(built[0][0]) and os.path.isfile(built[1][0]), msg
    assert mod._WRAP_OK is False, msg
    log = golden_run(TraceSim(dll))
    assert list(log) == list(glog)
    for k in glog:
        assert _bits_eq(log[k], glog[k]), k


def test_trace_fallback_when_the_linker_rejects_wrap(variants, glog, monkeypatch):
    """Codex C3-35: when ld rejects --wrap at build time the first DLL is never created and never
    probed: only the fallback build runs (no --wrap, no -DTRACE_WRAP) and is probed."""
    mod = sys.modules[__name__]
    built, cmds, probes = _gia_lap_probe(monkeypatch, wrap_ld=False)
    dll = variants('fwsim_trace_ld_tu_choi', trace=True)
    assert len(built) == 2, built
    msg = f'trace {built[0][0]} (khong duoc tao); {_dll_tin(built[1][0])}'   # built[0]: no file
    assert built[0][1] is None and built[1][1] is False, msg
    assert built[1][0] == built[0][0].replace('.dll', '_du_phong.dll'), msg
    assert '-DTRACE_WRAP' in cmds[0] and any('--wrap' in a for a in cmds[0]), cmds[0]
    for cmd in cmds[1:]:
        assert '-DTRACE_WRAP' not in cmd and not any('--wrap' in a for a in cmd), cmd
    assert not os.path.exists(built[0][0]), msg            # the rejected link left no DLL
    assert probes == [('du_phong', built[1][0])], msg      # the wrap variant is never probed
    assert dll == built[1][0], msg
    assert mod._WRAP_OK is False, msg
    log = golden_run(TraceSim(dll))
    assert list(log) == list(glog)
    for k in glog:
        assert _bits_eq(log[k], glog[k]), k


def test_trace_wrap_probe_ok(variants, monkeypatch):
    """The normal case: --wrap links and records calls: one build, one probe, no fallback file,
    _WRAP_OK ends True."""
    mod = sys.modules[__name__]
    built, cmds, probes = _gia_lap_probe(monkeypatch)
    dll = variants('fwsim_trace_thu_wrap', trace=True)
    msg = f'trace {_dll_tin(dll)}'
    assert built == [(dll, None)], msg
    assert probes == [('wrap', dll)], msg
    assert not os.path.exists(dll.replace('.dll', '_du_phong.dll')), msg
    assert mod._WRAP_OK is True, msg


@pytest.mark.parametrize('case', ['ld_tu_choi', 'ca_hai'])
def test_trace_fallback_fails_too(variants, monkeypatch, case):
    """When the fallback also records nothing the fixture must raise: ld rejects --wrap ('the
    trace records no pid_update calls'), or both variants probe empty ('renamed symbols too')."""
    if case == 'ld_tu_choi':
        built, cmds, probes = _gia_lap_probe(monkeypatch, wrap_ld=False, du_phong_bat=False)
        with pytest.raises(RuntimeError, match=r'^the trace records no pid_update calls$'):
            variants('fwsim_trace_hong_ld', trace=True)
        assert probes == [('du_phong', built[1][0])]       # the wrap DLL was never probed
    else:
        built, cmds, probes = _gia_lap_probe(monkeypatch, wrap_bat=False, du_phong_bat=False)
        with pytest.raises(RuntimeError, match='renamed symbols too'):
            variants('fwsim_trace_hong_ca_hai', trace=True)
        assert probes == [('wrap', built[0][0]), ('du_phong', built[1][0])]


def _eval_rows(log, v):
    """(res, rows) of o1.evaluate_v7 on an S03-like log at level v, the same call s03_run makes."""
    if not _PATH:
        _PATH.append(o1.Path8())
    sch = o1.Schedule(v)
    p = fwsim.plant_params(argparse.Namespace(ground='traction', mu=MU_GIUA))
    res, rows = o1.evaluate_v7(log, _PATH[0], sch, lsum=p.lx + p.ly)
    return res, rows


def s03_motor_tat(fw, v=0.8):
    """One s03_run at level v and the default gains, with the driver (SWA) off while
    3.0 <= t < 3.5: the log holds en == 0 ticks inside the evaluator window."""
    if not _PATH:
        _PATH.append(o1.Path8())
    sch = o1.Schedule(v)
    p = fwsim.plant_params(argparse.Namespace(ground='traction', mu=MU_GIUA))
    g = s03_defaults(fw)
    fw.reset()
    fw.set_mode('PID')
    base = [fw.get_pid(i) for i in range(3)]
    pid = {i: (*((g[2], g[3]) if i == 2 else (g[0], g[1])), 0.0, base[i]['int_limit'], base[i]['out_limit'])
           for i in range(3)}

    def sp(t):
        fw.motor_enable(not (3.0 <= t < 3.5))
        return sch.setpoint(t)

    log = fwsim.simulate(fw, sp, sch.Tcmd + o1.T_LATE, p, seed=1, mode='PID', imu=dict(S03_IMU),
                         pid=pid, fw_params=dict(pos_kp=g[4], pos_ki=g[5], k_psi=g[6]))
    res, rows = o1.evaluate_v7(log, _PATH[0], sch, lsum=p.lx + p.ly)
    return log, res, rows


@pytest.mark.parametrize('case', ['v08_mac_dinh', 'v08_ki_xy_1500', 'motor_tat'])
def test_denominator_is_the_evaluator_window(fw, case):
    """C3-13: kep_mau is the evaluator's own window [0, t_end] restricted to motor-on ticks. The
    window is the first n = len(rows) frames of the log: evaluate_v7 iterates from the first frame
    and stops at t_end (early) or at the last frame before the deadline."""
    if case == 'motor_tat':
        log, res, rows = s03_motor_tat(fw)
    else:
        log = kiem_cheo_run(fw, case)[0]
        res, rows = _eval_rows(log, 0.8)
    n = len(rows)
    mask = np.arange(len(log['t'])) < n
    assert [r['t'] for r in rows] == list(log['t'][:n])
    assert rows[-1]['t'] == res['t_end']
    assert np.array_equal(log['t'] <= res['t_end'] + 1e-9, mask)
    assert res['sat_frac'] == float(np.mean(np.abs(log['pwm'][mask]) >= 1000))   # mask is m of v7
    ind = fwsim.kep_indicator(log, res['t_end'])
    assert ind['kep_mau'] == int((mask & (log['en'] == 1)).sum())
    for i, a in enumerate(AXES):
        assert ind[f'kep_n_{a}'] == int(log['kep'][mask, i].sum())
    # the tick at t_end is inside the window: masking to t[n-2] drops it and its en from kep_mau
    assert ind['kep_mau'] - fwsim.kep_indicator(log, float(log['t'][n - 2]))['kep_mau'] \
        == int(log['en'][n - 1])
    if res['completed'] == 'true':          # the window closed early: ticks after t_end exist
        assert n < len(log['t'])
    else:                                   # the run reached the deadline without completing
        assert res['t_end'] == log['t'][n - 1]
        if n < len(log['t']):
            assert log['t'][n] >= res['t_deadline'] - 1e-9    # the deadline tick is outside
        else:
            assert log['t'][-1] < res['t_deadline'] - 1e-9    # the log ends before it
    if case == 'motor_tat':
        n_off = int((mask & (log['en'] == 0)).sum())
        assert n_off > 0 and ind['kep_mau'] == n - n_off
        assert np.all(log['kep'][log['en'] == 0] == 0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--digest', action='store_true')
    ap.add_argument('--scan', action='store_true')
    ap.add_argument('--pair', metavar='DLL',
                    help='a DLL without the kep channel: print the so_cap JSON of the four run pairs')
    a = ap.parse_args()
    fw_ = fwsim.FirmwareSim()
    if a.pair:
        ref_ = fwsim.FirmwareSim(a.pair)
        cu_nap_ = _dll_nap(ref_.lib)
        moi_nap_ = _dll_nap(fw_.lib)
        for name_, old_log_, new_log_ in _cap_runs(ref_, fw_):
            print(json.dumps(dict(run=name_, **so_cap(old_log_, new_log_),
                                  dll_cu=os.path.abspath(a.pair),
                                  dll_cu_sha256=build_fwsim.file_sha256(a.pair),
                                  dll_cu_nap=cu_nap_[0],
                                  dll_cu_nap_sha256=cu_nap_[1],
                                  dll_moi=os.path.abspath(_DLL),
                                  dll_moi_sha256=build_fwsim.file_sha256(_DLL),
                                  dll_moi_nap=moi_nap_[0],
                                  dll_moi_nap_sha256=moi_nap_[1])))
        sys.exit(0)
    if a.digest:
        log_ = golden_run(fw_)
        print(json.dumps(dict(identity=identity(fw_), kep_cols=getattr(fw_, 'kep_cols', None),
                              keys=sorted(log_), digest=old_digest(log_))))
    if a.scan:
        import tempfile
        d0 = s03_defaults(fw_)
        t0 = time.time()
        log_ = golden_run(fw_)
        r = so_sanh(log_, GOLDEN_ILIM, 4.0)
        r.update(run='golden', wall_s=round(time.time() - t0, 1), chi_bao=fwsim.kep_indicator(log_, 4.0))
        if os.path.isfile(GOC):
            with tempfile.TemporaryDirectory() as tmp:
                r['script_goc'] = script_goc(log_, tmp, GOLDEN_ILIM)
        print(json.dumps(r))
        sys.stdout.flush()
        scan = [(0.8, d0), (0.8, (d0[0], 0.0) + d0[2:]), (0.8, (0.15,) + d0[1:]), (0.8, (0.15, 0.005) + d0[2:]),
                (0.8, (d0[0], 20.0) + d0[2:]), (0.8, (d0[0], 1500.0) + d0[2:]), (0.5, d0), (0.2, d0)]
        for v_, g_ in scan:
            t0 = time.time()
            try:
                log_, ilim_, res_ = s03_run(fw_, v_, g_)
                r = so_sanh(log_, ilim_, res_['t_end'])
                r.update(chi_bao=fwsim.kep_indicator(log_, res_['t_end']), t_end=res_['t_end'],
                         completed=res_['completed'], status=res_['status'], ilim=list(ilim_))
                if os.path.isfile(GOC):
                    with tempfile.TemporaryDirectory() as tmp:
                        r['script_goc'] = script_goc(log_, tmp, ilim_)
            except Exception as ex:                      # an unstable gain set must not stop the scan
                r = dict(loi=f'{type(ex).__name__}: {ex}')
            r.update(run='s03', v=v_, gains=list(g_), wall_s=round(time.time() - t0, 1))
            print(json.dumps(r))
            sys.stdout.flush()
