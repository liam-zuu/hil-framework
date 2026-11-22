"""PID integral clamp indicator "tick cham kep" of fwsim (spec S03 v8.6 section 2.1, section 11 item 6 (c)):
fw_sim.c fwsim_kep_last, fwsim.py KEP_COLS / log['kep'] / kep_indicator.
Run: python -m pytest -q fwsim/tests/test_fwsim_kep.py   (needs fwsim/build/fwsim.dll)
     python fwsim/tests/test_fwsim_kep.py --digest       (old channels of the golden run, as recorded in GOLDEN)
     python fwsim/tests/test_fwsim_kep.py --scan         (cross-check against phan_tich_kep_pid, one JSON line per run)"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fwsim  # noqa: E402
import o1_feasibility as o1  # noqa: E402

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


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--digest', action='store_true')
    ap.add_argument('--scan', action='store_true')
    a = ap.parse_args()
    fw_ = fwsim.FirmwareSim()
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
