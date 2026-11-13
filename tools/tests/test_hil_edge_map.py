"""RS-4 (tools/hil_edge_map.py): the ESP32 run_active edge bracketed from the count side (early) and the PWM
side (late), on made-up runs whose true edge step is known. The generator here does NOT use the mapper's
model (G32-12, Codex round 1): encoder counts come late (one count behind the ideal angle, as EncoderGen's
discrete edges) and are sampled on their own 10 ms loop (encoder_task), so the logged count is older than the
tick by `age`; some ticks run 5 ms late; the pwm written in a tick reaches the MISO duty two frames after
it is written, with capture glitches. Liam 13/11/2026 CODE local_db99d287 ("Làm rs4 + g4 đi"),
local_7f7b0ef4 ("Kẹp hai phía", "Duyệt cả, vòng 2"). Every number is invented."""
import csv
import json
import math
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'fwsim'), os.path.join(HERE, '..', '..', 'rpi5')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import hil_edge_map as M                 # noqa: E402
import hil_o1_score as H                 # noqa: E402
import o1_gate as G                      # noqa: E402
import protocol as P                     # noqa: E402
import rq3_log as rl                     # noqa: E402
from agv_logger import COLUMNS           # noqa: E402
from test_hil_o1_score import fig8_pose  # noqa: E402

ROOT = os.path.join(HERE, '..', '..')
K = 1320 / (2 * math.pi)
SUB = 10                                  # sub-steps per plant step for the ideal angle (0.1 ms)


def wheel_omega(n, k_move, period=None):
    """Wheel speeds [rad/s], all forward: 0 before k_move, then a ramp to about 9 rad/s with a slow wobble
    (or a pattern repeating every `period` steps, for the alias test)."""
    k = np.arange(n, dtype=float)
    u = np.clip((k - k_move) / 400.0, 0.0, 1.0)
    if period:
        base = 6.0 + 3.0 * (np.floor((k - k_move) / (period / 2)) % 2)
    else:
        base = 9.0 * u * (1.0 + 0.15 * np.sin(2 * math.pi * np.clip(k - k_move, 0, None) / 1700.0))
    om = np.column_stack([base, base * 1.07, base * 0.94, base * 1.02])
    om[k < k_move] = 0.0
    return om


def ideal_counts(omega):
    """Ideal count at each 0.1 ms sub-step (H7 dead zone 0.3 rad/s), from the step-wise speeds."""
    w = np.repeat(np.where(np.abs(omega) > 0.3, omega, 0.0), SUB, axis=0) * (1e-3 / SUB) * K
    return np.vstack([np.zeros(4), np.cumsum(w, axis=0)])


def emitted(C, t_plant_ms):
    """Counts the ESP32 has seen at plant time t (ms): one count behind the ideal angle (never ahead)."""
    i = np.clip(np.floor(np.asarray(t_plant_ms) * SUB).astype(int), 0, len(C) - 1)
    return np.maximum(np.floor(C[i]) - 1.0, 0.0)


def sim(edge=415.3, k_run=400, n=9000, age_ms=7.0, late=(3, 7, 40, 41, 90, 133), lag_frames=2.0,
        delta_ms=0.4, ppm=30.0, glitch_every=97, period=None, pwm_period=None, n_before=6):
    """(omega, duty, esp log dict) of a made-up run with the true edge at plant step `edge`."""
    om = wheel_omega(n, edge + 3.0, period)
    C = ideal_counts(om)
    ts_e = 5_000_000.0

    def plant(ts):                                           # ESP32 clock -> plant step (truth, with drift)
        return edge + (ts - ts_e) / 1000.0 * (1.0 + ppm * 1e-6)
    n_ticks = int((n - edge - 30) / 10)
    ts = [ts_e - i * 100_000.0 for i in range(n_before, 0, -1)]
    ts += [ts_e + j * 10_000.0 + (5_000.0 if j in late else 0.0) for j in range(n_ticks)]
    ts = np.array(ts)
    run = np.r_[np.zeros(n_before, int), np.ones(n_ticks, int)]
    # encoder_task samples on its own loop, `age_ms` before the control tick's own 10 ms grid point
    grid = ts_e + np.round((ts - ts_e) / 10_000.0) * 10_000.0
    sample = grid - age_ms * 1000.0
    cnt = emitted(C, plant(sample))
    j = np.arange(n_ticks)
    if pwm_period:
        p = 60.0 + 30.0 * ((j // (pwm_period // 2)) % 2)
    else:
        p = np.round(60 + 40 * np.sin(0.13 * j) + 20 * np.sin(0.051 * j + 1.0))
    p[0] = 15.0
    pw = np.r_[np.zeros((n_before, 4)), np.round(np.column_stack([p, p * 0.9, p * 1.1, p]))]   # integer pwm, as logged
    # MISO duty of frame k: the pwm whose write (tick + delta) is at least lag_frames steps before k
    write_k = plant(ts + delta_ms * 1000.0)
    k = np.arange(n)
    idx = np.searchsorted(write_k, k - lag_frames, side='right') - 1
    duty = np.where(idx[:, None] >= 0, pw[np.clip(idx, 0, len(pw) - 1)], 0.0) * (-1.0 / 1000.0)
    if glitch_every:
        duty[k_run + 50::glitch_every, 1] = -1.0                 # capture glitch frames
    esp = dict(timestamp_us=ts.astype(np.int64), log_seq=np.arange(100, 100 + len(ts), dtype=np.int64),
               motor_on=run.copy(), run_active=run.copy(), counts=cnt.astype(float), pwm=pw)
    return om, duty, esp


def _check_bracket(em, edge):
    assert em['ok'], em['reason']
    assert em['early_step'] <= edge <= em['late_step']
    assert abs(em['edge_step'] - edge) <= em['u_t_s'] / em['period_s'] + 1e-9


def test_constants_match_the_h7_source():
    h = open(os.path.join(ROOT, 'stm32h7', 'Core', 'Inc', 'encoder_gen.h'), encoding='utf-8', errors='replace').read()
    c = open(os.path.join(ROOT, 'stm32h7', 'Core', 'Src', 'encoder_gen.c'), encoding='utf-8', errors='replace').read()
    assert int(re.search(r'#define\s+ENC_PPR\s+(\d+)u', h).group(1)) == M.ENC_PPR
    assert float(re.search(r'#define\s+ENC_OMEGA_MIN\s+([0-9.]+)f', c).group(1)) == M.OMEGA_MIN
    assert M.COUNTS_PER_REV == 1320


@pytest.mark.parametrize('edge,age', [(415.3, 7.0), (437.85, 0.0), (408.1, 3.0), (420.5, 9.9)])
def test_bracket_holds_the_true_edge(edge, age):
    om, duty, esp = sim(edge=edge, age_ms=age)
    em = M.map_edge(om, duty, 400, esp, 1e-3)
    _check_bracket(em, edge)
    assert em['count_side']['a'] < edge                       # the count side is early (old, late counts)
    assert em['pwm_side']['a'] > edge                         # the PWM side is late (write + 2 frames)
    assert em['u_t_s'] < 8e-3
    assert em['u_t_s'] == pytest.approx(sum(em['u_t_parts_s'].values()))


def test_count_side_alone_is_wrong_by_the_sample_age():
    # G32-3: with counts 7 ms old the count fit alone sits about 7 ms early, beyond the old bound of 1-2 ms
    om, duty, esp = sim(age_ms=7.0)
    em = M.map_edge(om, duty, 400, esp, 1e-3)
    assert 415.3 - em['count_side']['a'] > 5.0


def test_lost_edge_lines_refuse():
    # G32-4: the first two run_active lines lost; the next line still looks like an edge
    om, duty, esp = sim()
    keep = np.r_[0:6, 8:len(esp['log_seq'])]
    esp2 = {k: v[keep] for k, v in esp.items()}
    em = M.map_edge(om, duty, 400, esp2, 1e-3)
    assert not em['ok'] and 'log_seq' in em['reason']


def test_refusals():
    om, duty, esp = sim()
    two = {k: v.copy() for k, v in esp.items()}
    two['run_active'][len(two['run_active']) // 2] = 0
    assert '2 run_active rising edges' in M.map_edge(om, duty, 400, two, 1e-3)['reason']
    none = dict(esp, run_active=np.zeros_like(esp['run_active']))
    assert 'need exactly 1' in M.map_edge(om, duty, 400, none, 1e-3)['reason']
    still = dict(esp, counts=np.zeros_like(esp['counts']))
    assert 'never move' in M.map_edge(om, duty, 400, still, 1e-3)['reason']
    nopwm = dict(esp, pwm=np.zeros_like(esp['pwm']))
    assert 'pwm never' in M.map_edge(om, duty, 400, nopwm, 1e-3)['reason']
    em = M.map_edge(om, np.zeros_like(duty), 400, esp, 1e-3)              # no duty in the frames
    assert not em['ok'] and 'PWM side' in em['reason']
    em = M.map_edge(om, duty, 1000, esp, 1e-3)                            # RUN step far after the edge
    assert not em['ok']


def test_periodic_run_is_refused_or_bracketed():
    # G32-6: speeds and pwm repeating every 100 steps (10 ticks): an alias of the edge must not pass silently
    edge = 415.3
    om, duty, esp = sim(edge=edge, period=100, pwm_period=10, glitch_every=0)
    em = M.map_edge(om, duty, 400, esp, 1e-3)
    assert (not em['ok']) or (em['early_step'] <= edge <= em['late_step'])


def write_esp_csv(path, esp, columns=COLUMNS, extra_rows=()):
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


def test_read_esp_log_order_duplicates_and_columns(tmp_path):
    _, _, esp = sim(n=3000)
    p = str(tmp_path / 'r.csv')
    write_esp_csv(p, esp)
    got = M.read_esp_log(p)
    assert np.array_equal(got['log_seq'], esp['log_seq']) and np.array_equal(got['pwm'], esp['pwm'])
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
    # any column count that has the needed columns (85 or 86 column logs)
    p2 = str(tmp_path / 'small.csv')
    write_esp_csv(p2, esp, columns=list(M.NEED) + ['traj_age_ms'])
    assert np.array_equal(M.read_esp_log(p2)['counts'], got['counts'])
    p3 = str(tmp_path / 'bad.csv')
    write_esp_csv(p3, esp, columns=[c for c in M.NEED if c != 'pwm_fl'])
    with pytest.raises(ValueError, match='pwm_fl'):
        M.read_esp_log(p3)


def write_frame_log(path, task, k_run, om, duty):
    """Figure-8 frame log: plant pose from the true edge (k where omega starts), wheel speeds in the omega
    slots, the duty in MISO bytes 0-7 (not kinematically tied to the pose: only RS-4 reads them)."""
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
    np.savez(path, manifest=np.array(json.dumps(manifest)), mosi=mosi, miso=miso)
    state, _, _, d = H.decode_frames(mosi, miso)
    return state[:, 0:4], d


def test_scorer_with_esp_log_decides_all_five(tmp_path):
    p, c = str(tmp_path / 'run.npz'), str(tmp_path / 'run.csv')
    n = int((H.Schedule(H.FIG8['O1-F05']).Tcmd + 2.5) / 1e-3) + 420
    om, duty, esp = sim(edge=400.0, k_run=388, n=n)
    write_frame_log(p, 'O1-F05', 388, om, duty)               # frames carry omega, duty as int16
    write_esp_csv(c, esp)
    out = H.score(p, 'O1-F05', esp_log=c)
    em, g = out['edge_map'], out['gate_v3']
    _check_bracket(em, 400.0)
    assert out['t0_src'] == 'esp32_run_active_edge' and out['k_run'] == 388
    assert g['decide'] == list(G.CONDS) and g['u_t'] == pytest.approx(em['u_t_s']) and g['complete']
    js = H._json_safe(out)
    json.dumps(js)
    assert rl.validate_gate(js['gate_v3'], 'HIL')['complete']
    plain = H.score(p, 'O1-F05')                                         # no ESP32 log: RUN step, G1, G2 decide
    assert plain['t0_src'] == 'lifecycle_RUN' and plain['gate_v3']['decide'] == list(G.DECIDE_HIL)
    assert plain['edge_map'] == dict(ok=False, reason='no --esp-log') and not plain['gate_v3']['complete']


def test_scorer_cli_esp_log_errors(tmp_path):
    p, c = str(tmp_path / 'run.npz'), str(tmp_path / 'run.csv')
    om, duty, _ = sim(n=3000)
    write_frame_log(p, 'O1-F05', 400, om, duty)
    with open(c, 'w') as f:
        f.write('timestamp_us,log_seq\n1,1\n')
    assert H.main(['--frame-log', p, '--task', 'O1-F05', '--esp-log', c]) == 2       # bad log: an error
    assert M.main(['--frame-log', p, '--esp-log', c]) == 2
