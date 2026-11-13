"""RS-4 (tools/hil_edge_map.py), count side after L98: the ESP32 run_active edge bracketed on made-up runs whose
true edge (Pi time) is known. The counts come from the H7 code itself (tools/tests/encgen_oracle.py: the
unchanged encoder_gen.c in profile mode; Codex Sol G32-12, G32-R2-4), driven by the frame wheel speeds applied
at T_k + d_k (d_k random in 0..750 us plus a constant 0..900 us), read at the ESP32 tick times (L98:
counts read at t_now) mapped to the Pi clock with a rate difference; ticks are 10 ms with some 5 ms late
(ESP_TICK/2026-11-13_scan); the pwm written in a tick reaches the MISO duty of the frame after the one the H7
serviced (capture at the service time T_j + d_j, answer in frame j + 2), with capture glitches. Liam
13/11/2026 CODE local_db99d287 ("Làm rs4 + g4 đi"), local_7f7b0ef4 ("Kẹp hai phía", "Sửa firmware" L98).
Every number is invented."""
import csv
import json
import math
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'fwsim'), os.path.join(HERE, '..', '..', 'rpi5')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import encgen_oracle as O                # noqa: E402
import hil_edge_map as M                 # noqa: E402
import hil_o1_score as H                 # noqa: E402
import o1_gate as G                      # noqa: E402
import protocol as P                     # noqa: E402
import rq3_log as rl                     # noqa: E402
from agv_logger import COLUMNS           # noqa: E402
from test_hil_o1_score import fig8_pose  # noqa: E402

ROOT = os.path.join(HERE, '..', '..')
FW = [dict(sha=M.FW_COUNTS_AT_TICK[0], dirty='0', env='esp32-s3-devkitc-1')]
needs_gcc = pytest.mark.skipif(not O.available(), reason='gcc not on PATH (H7 EncoderGen oracle)')


def wheel_omega(n, k_move, kind='ramp'):
    """Wheel speeds [rad/s] per frame: 0 before k_move (except 'moving'), then by kind."""
    k = np.arange(n, dtype=float)
    u = np.clip((k - k_move) / 400.0, 0.0, 1.0)
    if kind == 'ramp':
        base = 9.0 * u * (1.0 + 0.15 * np.sin(2 * math.pi * np.clip(k - k_move, 0, None) / 1700.0))
    elif kind == 'periodic':                                        # alias test (G32-6)
        base = 6.0 + 3.0 * (np.floor((k - k_move) / 50.0) % 2)
    elif kind == 'slow':                                            # Codex G32-R2-1 (a)
        base = np.where(k < k_move + 17, 1.0, 1.0 + 0.00005 * (k - k_move - 17))
    elif kind == 'stop_start':
        base = 10.0 * u
        base[(k > k_move + 900) & (k < k_move + 1000)] = 0.0
    else:
        raise ValueError(kind)
    om = np.column_stack([base, -base * 1.07, base * 0.94, -base * 1.02])
    om[k < k_move] = 0.0
    return om


def sim(edge_ms=415.3, k_run=400, n=5200, kind='ramp', move_after_ms=3.0, d0_us=400.0, ppm=30.0,
        late=(3, 7, 40, 41, 90, 133), write_ms=0.4, glitch_every=97, n_before=6, seed=1, pwm_period=None):
    """(omega, duty, esp log dict, t_send_ns, true edge step) of a made-up run; the true edge tick is at Pi
    time edge_ms [ms] from frame 0."""
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    T = k * 1000.0 - rng.uniform(0.0, 140.0, n) * (k > 0)         # Pi send times [us] (jitter, as run08)
    T[1:] = np.maximum(T[1:], T[:-1] + 1.0)
    om = wheel_omega(n, int(edge_ms + move_after_ms), kind)
    d = d0_us + rng.uniform(0.0, 750.0, n)                         # H7 applies frame k at T_k + d_k
    app = np.round(T + d + 10_000).astype(np.int64)                # oracle ticks (offset keeps them >= 0)
    for i in range(1, n):
        app[i] = max(app[i], app[i - 1] + 1)
    counts = O.run(app, om, seed)
    t_e = edge_ms * 1000.0
    ts_e = 5_000_000.0

    def pi_time(ts):                                               # ESP32 clock -> Pi time (truth, with drift)
        return t_e + (ts - ts_e) * (1.0 + ppm * 1e-6)
    n_ticks = int((T[-1] - t_e - 60_000) / 10_000)
    ts = [ts_e - i * 100_000.0 for i in range(n_before, 0, -1)]
    ts += [ts_e + j * 10_000.0 + (5_000.0 if j in late else 0.0) for j in range(n_ticks)]
    ts = np.array(ts)
    run = np.r_[np.zeros(n_before, int), np.ones(n_ticks, int)]
    cnt = counts.at(np.round(pi_time(ts) + 10_000).astype(np.int64))   # L98: read at t_now
    j = np.arange(n_ticks)
    if pwm_period:
        p = 60.0 + 30.0 * ((j // (pwm_period // 2)) % 2)
    else:
        p = np.round(60 + 40 * np.sin(0.13 * j) + 20 * np.sin(0.051 * j + 1.0))
    p[0] = 15.0
    pw = np.r_[np.zeros((n_before, 4)), np.round(np.column_stack([p, p * 0.9, p * 1.1, p]))]
    # MISO duty of frame f: the pwm written before the H7 serviced frame f - 2 (capture then, answer in f)
    write_t = pi_time(ts + write_ms * 1000.0)
    serv = np.r_[np.full(2, -1e18), (T + d)[:-2]]
    idx = np.searchsorted(write_t, serv, side='right') - 1
    duty = np.where(idx[:, None] >= 0, pw[np.clip(idx, 0, len(pw) - 1)], 0.0) * (-1.0 / 1000.0)
    if glitch_every:
        duty[k_run + 50::glitch_every, 1] = -1.0                 # capture glitch frames
    esp = dict(timestamp_us=ts.astype(np.int64), log_seq=np.arange(100, 100 + len(ts), dtype=np.int64),
               motor_on=run.copy(), run_active=run.copy(), counts=cnt.astype(float), pwm=pw, fw=list(FW))
    t_send_ns = np.round((T + 7.0e9) * 1000.0).astype(np.int64)
    return om, duty, esp, t_send_ns, float(np.interp(t_e, T, k))


def _check_bracket(em, edge):
    assert em['ok'], em['reason']
    assert em['early_step'] <= edge <= em['late_step'], (em['early_step'], edge, em['late_step'])
    assert abs(em['edge_step'] - edge) <= em['u_t_s'] / em['period_s'] + 1e-9


def test_constants_match_the_h7_source():
    h = open(os.path.join(ROOT, 'stm32h7', 'Core', 'Inc', 'encoder_gen.h'), encoding='utf-8', errors='replace').read()
    c = open(os.path.join(ROOT, 'stm32h7', 'Core', 'Src', 'encoder_gen.c'), encoding='utf-8', errors='replace').read()
    assert int(re.search(r'#define\s+ENC_PPR\s+(\d+)u', h).group(1)) == M.ENC_PPR
    assert float(re.search(r'#define\s+ENC_OMEGA_MIN\s+([0-9.]+)f', c).group(1)) == M.OMEGA_MIN
    assert M.COUNTS_PER_REV == 1320


@needs_gcc
@pytest.mark.parametrize('edge,d0,ppm,write', [(415.3, 400.0, 30.0, 0.4), (437.85, 0.0, -100.0, 0.4),
                                               (408.1, 900.0, 100.0, 7.4), (420.5, 250.0, 0.0, 9.0)])
def test_bracket_holds_the_true_edge(edge, d0, ppm, write):
    om, duty, esp, tsend, k_true = sim(edge_ms=edge, d0_us=d0, ppm=ppm, write_ms=write)
    em = M.map_edge(om, duty, 400, esp, 1e-3, tsend)
    _check_bracket(em, k_true)
    assert em['u_t_s'] < 6e-3
    assert em['u_t_s'] == pytest.approx(sum(em['u_t_parts_s'].values()))
    assert em['frame_times'] == 't_send_ns'


@needs_gcc
def test_moving_at_the_edge_and_stop_start():
    # Codex G32-R2-1 (a): wheels already turning at the edge; a stop / restart inside the fit span
    for kind, after in (('ramp', -40.0), ('stop_start', 3.0)):
        om, duty, esp, tsend, k_true = sim(kind=kind, move_after_ms=after)
        _check_bracket(M.map_edge(om, duty, 400, esp, 1e-3, tsend), k_true)


@needs_gcc
def test_slow_wheels_are_refused_or_bracketed():
    # Codex G32-R2-1 (a): 1 rad/s from before the edge then a very slow ramp: one count is about 4.8 ms
    om, duty, esp, tsend, k_true = sim(kind='slow', move_after_ms=-17.0)
    em = M.map_edge(om, duty, 400, esp, 1e-3, tsend)
    assert (not em['ok']) or (em['early_step'] <= k_true <= em['late_step'])


@needs_gcc
def test_periodic_run_is_refused_or_bracketed():
    # G32-6: speeds and pwm repeating every 100 steps (10 ticks): an alias of the edge must not pass silently
    om, duty, esp, tsend, k_true = sim(kind='periodic', pwm_period=10, glitch_every=0)
    em = M.map_edge(om, duty, 400, esp, 1e-3, tsend)
    assert (not em['ok']) or (em['early_step'] <= k_true <= em['late_step'])


@needs_gcc
def test_stale_counts_of_the_old_firmware_are_caught():
    # the pre-L98 log: counts copied 7 ms before the tick; with the old firmware's #fw line it is refused,
    # and even with the firmware check bypassed the stale counts must not give a wrong narrow bracket silently
    om, duty, esp, tsend, k_true = sim()
    old = dict(esp, fw=[dict(sha='cfc892b0ba28', dirty='0', env='esp32-s3-devkitc-1')])
    em = M.map_edge(om, duty, 400, old, 1e-3, tsend)
    assert not em['ok'] and 'L98' in em['reason']
    for fw, word in ((None, 'no events file'), ([], 'no #fw'), ([dict(FW[0], dirty='1')], 'dirty=1'),
                     (FW + [dict(sha='54b7269abcde', dirty='0')], '54b7269abcde')):
        em = M.map_edge(om, duty, 400, dict(esp, fw=fw), 1e-3, tsend)
        assert not em['ok'] and word in em['reason'], em['reason']


@needs_gcc
def test_lost_edge_lines_refuse():
    # G32-4: the first two run_active lines lost; the next line still looks like an edge
    om, duty, esp, tsend, _ = sim()
    keep = np.r_[0:6, 8:len(esp['log_seq'])]
    esp2 = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in esp.items()}
    em = M.map_edge(om, duty, 400, esp2, 1e-3, tsend)
    assert not em['ok'] and 'log_seq' in em['reason']


@needs_gcc
def test_refusals():
    om, duty, esp, tsend, _ = sim()
    two = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in esp.items()}
    two['run_active'][len(two['run_active']) // 2] = 0
    assert '2 run_active rising edges' in M.map_edge(om, duty, 400, two, 1e-3, tsend)['reason']
    none = dict(esp, run_active=np.zeros_like(esp['run_active']))
    assert 'need exactly 1' in M.map_edge(om, duty, 400, none, 1e-3, tsend)['reason']
    still = dict(esp, counts=np.zeros_like(esp['counts']))
    assert 'never move' in M.map_edge(om, duty, 400, still, 1e-3, tsend)['reason']
    shifted = dict(esp, counts=esp['counts'] + np.where(np.arange(len(esp['counts']))[:, None] > 60, 40.0, 0.0))
    assert 'no Pi time fits' in M.map_edge(om, duty, 400, shifted, 1e-3, tsend)['reason']   # 40 counts jump
    nopwm = dict(esp, pwm=np.zeros_like(esp['pwm']))
    assert 'PWM cross-check failed' in M.map_edge(om, duty, 400, nopwm, 1e-3, tsend)['reason']
    em = M.map_edge(om, np.zeros_like(duty), 400, esp, 1e-3, tsend)                 # no duty in the frames
    assert not em['ok'] and 'PWM' in em['reason']
    early_duty = np.r_[duty[30:], np.repeat(duty[-1:], 30, axis=0)]                   # duty 30 ms before its write
    em = M.map_edge(om, early_duty, 400, esp, 1e-3, tsend)
    assert not em['ok'] and 'cannot precede' in em['reason']
    em = M.map_edge(om, duty, 1000, esp, 1e-3, tsend)                                 # RUN step far after the edge
    assert not em['ok']
    bad_t = tsend.copy()
    bad_t[50] = bad_t[49]
    assert 'increasing' in M.map_edge(om, duty, 400, esp, 1e-3, bad_t)['reason']


def test_plateau_right_end():
    # G32-R2-2: a step-function cost keeps the right end of its flat minimum
    fine = np.arange(0.0, 2.001, 0.01)
    cf = np.where((fine >= 0.6) & (fine < 1.55), 1.0, 5.0)
    lo, hi = M._plateau(fine, cf)
    assert lo == pytest.approx(0.6) and hi == pytest.approx(1.55, abs=0.011)


def test_theta_matches_the_integral():
    T = np.array([0.0, 1000.0, 2100.0, 3000.0])
    om = np.array([[0.2, 1, 2, -3], [1.0, 1, 2, -3], [2.0, 1, 2, -3], [2.0, 0, 0, 0]])
    th, j = M.Theta(T, om).at(np.array([-5.0, 500.0, 1500.0, 3500.0]))
    assert th[0].tolist() == [0, 0, 0, 0] and j.tolist() == [0, 0, 1, 3]
    assert th[1] == pytest.approx([0.0, 0.5e-3, 1.0e-3, -1.5e-3])                   # 0.2 rad/s is below the dead zone
    assert th[3] == pytest.approx([1.1e-3 + 1.8e-3 + 1.0e-3, 3.0e-3, 6.0e-3, -9.0e-3])


def write_esp_csv(path, esp, columns=COLUMNS, extra_rows=(), fw=FW):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(columns)
        rows = []
        for i in range(len(esp['log_seq'])):
            d = dict.fromkeys(columns, '0')
            d.update(timestamp_us=int(esp['timestamp_us'][i]), log_seq=int(esp['log_seq'][i]),
                     motor_on=int(esp['motor_on'][i]), run_active=int(esp['run_active'][i]))
            for name, c in zip(M.WHEELS, esp['counts'][i]):
                d[name] = int(c)
            for name, c in zip(M.PWMS, esp['pwm'][i]):
                d[name] = int(c)
            rows.append([d[c] for c in columns])
        for r in rows + [rows[i] for i in extra_rows]:
            w.writerow(r)
    if fw is not None:
        with open(path + '.events.txt', 'w', encoding='utf-8') as f:
            f.write('2026-11-13T01:45:57.545110 #logger,start\n')
            for d in fw:
                f.write('2026-11-13T01:45:57.545975 #fw,' + ','.join(f'{k}={v}' for k, v in d.items()) + '\n')


@needs_gcc
def test_read_esp_log_order_duplicates_columns_and_fw(tmp_path):
    _, _, esp, _, _ = sim(n=3000)
    p = str(tmp_path / 'r.csv')
    write_esp_csv(p, esp)
    got = M.read_esp_log(p)
    assert np.array_equal(got['log_seq'], esp['log_seq']) and np.array_equal(got['pwm'], esp['pwm'])
    assert got['fw'] == FW and M.fw_check(got['fw']) == ''
    # G32-10: a duplicate line without a segment file refuses the log
    write_esp_csv(p, esp, extra_rows=(40,))
    with pytest.raises(ValueError, match='duplicate line'):
        M.read_esp_log(p)
    # with the segment file marking it, it is dropped; a conflict refuses
    with open(p, newline='') as f:
        body = list(csv.reader(f))[1:]
    seg = p + '.segments.csv'
    ti, si = COLUMNS.index('timestamp_us'), COLUMNS.index('log_seq')

    def write_seg(last):
        with open(seg, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(M.SEG_COLS)
            for i, r in enumerate(body, start=1):
                w.writerow([i, r[si], r[ti], 0, last if i == len(body) else 'ok'])
    write_seg('duplicate')
    assert len(M.read_esp_log(p)['log_seq']) == len(esp['log_seq'])
    write_seg('conflict')
    with pytest.raises(ValueError, match='conflict'):
        M.read_esp_log(p)
    os.remove(seg)
    # any column count that has the needed columns (85 or 86 column logs); no events file: fw None
    p2 = str(tmp_path / 'small.csv')
    write_esp_csv(p2, esp, columns=list(M.NEED) + ['traj_age_ms'], fw=None)
    g2 = M.read_esp_log(p2)
    assert np.array_equal(g2['counts'], got['counts']) and g2['fw'] is None and 'no events file' in M.fw_check(None)
    p3 = str(tmp_path / 'bad.csv')
    write_esp_csv(p3, esp, columns=[c for c in M.NEED if c != 'pwm_fl'])
    with pytest.raises(ValueError, match='pwm_fl'):
        M.read_esp_log(p3)


def write_frame_log(path, task, k_run, om, duty, t_send_ns=None):
    """Figure-8 frame log: plant pose from k where omega starts, wheel speeds in the omega slots, the duty in
    MISO bytes 0-7 (not kinematically tied to the pose: only RS-4 reads them), t_send_ns when given."""
    n = len(om)
    k_move = int(np.argmax(np.any(om != 0, axis=1)))
    t = np.arange(n - k_move) * 1e-3
    pose, _ = fig8_pose(task, t)
    pose = np.vstack([np.zeros((k_move, 3)), pose])
    vel = np.gradient(pose[:, :2], 1e-3, axis=0)
    wz = np.gradient(np.unwrap(pose[:, 2]), 1e-3)
    mosi = np.zeros((n, 40), dtype=np.uint8)
    miso = np.zeros((n, 40), dtype=np.uint8)
    for k in range(n):
        st = list(om[k]) + [pose[k, 0], pose[k, 1], pose[k, 2], vel[k, 0], vel[k, 1], wz[k]]
        on = k >= k_run + 3
        mosi[k] = np.frombuffer(P.encode_state(st, traj_ref=(0.3, 0.0, 0.1) if on else None,
                                               traj_flags=P.TRAJ_VALID if on else 0, seq=k), dtype=np.uint8)
        miso[k, 0:8] = np.frombuffer(np.round(duty[k] * 32767).astype('>i2').tobytes(), dtype=np.uint8)
    manifest = dict(period_us=1000, steps=n, plant='MecanumPlant', miso_check=dict(crc_bad=0, empty=0),
                    lifecycle=dict(outcome='completed', transitions=[dict(step=0, t=0, phase='ARM'),
                                                                     dict(step=k_run, t=0.4, phase='RUN')]))
    extra = {} if t_send_ns is None else dict(t_send_ns=t_send_ns)
    np.savez(path, manifest=np.array(json.dumps(manifest)), mosi=mosi, miso=miso, **extra)
    state, _, _, d = H.decode_frames(mosi, miso)
    return state[:, 0:4], d


@needs_gcc
def test_scorer_with_esp_log_decides_all_five(tmp_path):
    p, c = str(tmp_path / 'run.npz'), str(tmp_path / 'run.csv')
    n = int((H.Schedule(H.FIG8['O1-F05']).Tcmd + 2.5) / 1e-3) + 420
    om, duty, esp, tsend, k_true = sim(edge_ms=400.0, k_run=388, n=n)
    om_q, duty_q = write_frame_log(p, 'O1-F05', 388, om, duty, tsend)   # frames carry omega, duty as int16
    write_esp_csv(c, esp)
    out = H.score(p, 'O1-F05', esp_log=c)
    em, g = out['edge_map'], out['gate_v3']
    _check_bracket(em, k_true)
    assert em['frame_times'] == 't_send_ns' and M.map_run(p, c)['ok']
    assert out['t0_src'] == 'esp32_run_active_edge' and out['k_run'] == 388
    assert g['decide'] == list(G.CONDS) and g['u_t'] == pytest.approx(em['u_t_s']) and g['complete']
    js = H._json_safe(out)
    json.dumps(js)
    assert rl.validate_gate(js['gate_v3'], 'HIL')['complete']
    plain = H.score(p, 'O1-F05')                                         # no ESP32 log: RUN step, G1, G2 decide
    assert plain['t0_src'] == 'lifecycle_RUN' and plain['gate_v3']['decide'] == list(G.DECIDE_HIL)
    assert plain['edge_map'] == dict(ok=False, reason='no --esp-log') and not plain['gate_v3']['complete']
    os.remove(c + '.events.txt')                                         # firmware not known: G1, G2 only
    nofw = H.score(p, 'O1-F05', esp_log=c)
    assert not nofw['edge_map']['ok'] and nofw['gate_v3']['decide'] == list(G.DECIDE_HIL)


@needs_gcc
def test_scorer_cli_esp_log_errors(tmp_path):
    p, c = str(tmp_path / 'run.npz'), str(tmp_path / 'run.csv')
    om, duty, _, tsend, _ = sim(n=3000)
    write_frame_log(p, 'O1-F05', 400, om, duty, tsend)
    with open(c, 'w') as f:
        f.write('timestamp_us,log_seq\n1,1\n')
    assert H.main(['--frame-log', p, '--task', 'O1-F05', '--esp-log', c]) == 2       # bad log: an error
    assert M.main(['--frame-log', p, '--esp-log', c]) == 2
