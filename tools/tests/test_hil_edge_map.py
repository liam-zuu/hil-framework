"""RS-4 (tools/hil_edge_map.py): the ESP32 run_active edge mapped to the plant step by count matching,
on made-up frame logs and ESP32 logs whose true edge step is known; and through hil_o1_score --esp-log.
Liam 13/11/2026 CODE local_db99d287 ("Làm rs4 + g4 đi"). Every number is invented."""
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
GAINS = np.array([1.0, 1.08, 0.93, 1.03])


def wheel_omega(n, k_edge):
    """Wheel speeds [rad/s]: 0 before the edge, then a ramp with a slow reversal on two wheels."""
    k = np.arange(n, dtype=float)
    u = np.clip((k - k_edge) / 400.0, 0.0, None)
    base = 9.0 * np.minimum(u, 1.0)
    rev = np.cos(2 * math.pi * np.clip(k - k_edge, 0, None) / 3000.0)
    om = np.column_stack([base, base * rev, base, base * rev]) * GAINS
    om[k < k_edge] = 0.0
    return om


def esp_lines(omega, k_edge_true, ts_edge=5_000_000, n_before=6, late=(), shift_us=0):
    """ESP32 log lines: 10 Hz motor-off lines before the edge, then one line per 10 ms tick with the counts
    the H7 would have made (dead zone, quadrature 1320) at the plant step of the tick. `late`: tick
    numbers that run 5 ms late (timestamp later, counts at that later moment)."""
    A = M.plant_angle(omega, 1e-3)
    K = M.COUNTS_PER_REV / (2 * math.pi)
    rows, seq = [], 100
    for i in range(n_before, 0, -1):
        rows.append(dict(timestamp_us=ts_edge - i * 100_000, log_seq=seq, motor_on=0, run_active=0, counts=(0, 0, 0, 0)))
        seq += 1
    n_ticks = int((len(omega) - k_edge_true - 20) / 10)
    for j in range(n_ticks):
        ts = ts_edge + j * 10_000 + (5_000 if j in late else 0) + shift_us
        k = k_edge_true + (ts - ts_edge - shift_us) / 1000.0
        ang = np.array([np.interp(k, np.arange(len(A)), A[:, w]) for w in range(4)])
        rows.append(dict(timestamp_us=ts, log_seq=seq, motor_on=1, run_active=1,
                         counts=tuple(int(math.floor(x)) for x in ang * K)))
        seq += 1
    return rows


def write_esp_csv(path, rows, columns=COLUMNS):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            d = dict.fromkeys(columns, '0')
            d.update(timestamp_us=r['timestamp_us'], log_seq=r['log_seq'], motor_on=r['motor_on'],
                     run_active=r['run_active'])
            for name, c in zip(M.WHEELS, r['counts']):
                d[name] = c
            w.writerow([d[c] for c in columns])


def esp_dict(rows):
    out = {c: np.array([r[c] for r in rows], dtype=np.int64) for c in ('timestamp_us', 'log_seq', 'motor_on', 'run_active')}
    out['counts'] = np.array([r['counts'] for r in rows], dtype=float)
    return out


def test_constants_match_the_h7_source():
    h = open(os.path.join(ROOT, 'stm32h7', 'Core', 'Inc', 'encoder_gen.h'), encoding='utf-8', errors='replace').read()
    c = open(os.path.join(ROOT, 'stm32h7', 'Core', 'Src', 'encoder_gen.c'), encoding='utf-8', errors='replace').read()
    assert int(re.search(r'#define\s+ENC_PPR\s+(\d+)u', h).group(1)) == M.ENC_PPR
    assert float(re.search(r'#define\s+ENC_OMEGA_MIN\s+([0-9.]+)f', c).group(1)) == M.OMEGA_MIN
    assert M.COUNTS_PER_REV == 1320


@pytest.mark.parametrize('k_edge', [412.3, 437.85, 401.0])
def test_mapping_finds_the_true_edge(k_edge):
    n, k_run = 9000, 400
    om = wheel_omega(n, k_edge)
    em = M.map_edge(om, k_run, esp_dict(esp_lines(om, k_edge, late={3, 7, 40, 41, 90})), 1e-3)
    assert em['ok'], em['reason']
    assert em['edge_step'] == pytest.approx(k_edge, abs=0.3)
    assert em['offset_ms'] == pytest.approx(k_edge - k_run, abs=0.3)
    assert em['rms_counts'] < 1.0 and em['n_edges'] == 1
    assert 1e-3 <= em['u_t_s'] < 3e-3
    assert em['u_t_s'] == pytest.approx(sum(em['u_t_parts_s'].values()))


def test_mapping_refuses():
    n, k_run, ke = 9000, 400, 415.0
    om = wheel_omega(n, ke)
    rows = esp_lines(om, ke)
    two = [dict(r) for r in rows]
    two[len(two) // 2]['run_active'] = 0                     # run_active drops and rises again
    em = M.map_edge(om, k_run, esp_dict(two), 1e-3)
    assert not em['ok'] and '2 run_active rising edges' in em['reason']
    none = [dict(r, run_active=0) for r in rows]
    assert 'need exactly 1' in M.map_edge(om, k_run, esp_dict(none), 1e-3)['reason']
    still = [dict(r, counts=(0, 0, 0, 0)) for r in rows]
    assert 'never move' in M.map_edge(om, k_run, esp_dict(still), 1e-3)['reason']
    wrong = [dict(r, counts=tuple(-c for c in r['counts'])) for r in rows]   # counts that are not this plant
    em = M.map_edge(om, k_run, esp_dict(wrong), 1e-3)
    assert not em['ok'] and ('RMS' in em['reason'] or 'edge of the search' in em['reason'])
    far = M.map_edge(om, k_run, esp_dict(esp_lines(om, ke)), 1e-3)
    assert far['ok']
    em = M.map_edge(om, 1000, esp_dict(esp_lines(om, ke)), 1e-3)  # true edge 585 steps before "RUN"
    assert not em['ok'] and 'edge of the search' in em['reason']


def test_read_esp_log_duplicates_order_and_columns(tmp_path):
    n, ke = 6000, 410.0
    om = wheel_omega(n, ke)
    rows = esp_lines(om, ke)
    shuffled = rows[:20] + rows[25:30] + rows[20:25] + rows[30:] + [rows[40]]    # reordered + one duplicate
    p = str(tmp_path / 'r.csv')
    write_esp_csv(p, shuffled)
    got = M.read_esp_log(p)
    assert list(got['log_seq']) == [r['log_seq'] for r in rows]
    # segment file: the duplicate is marked and dropped; a conflict refuses the log
    with open(p, newline='') as f:
        body = list(csv.reader(f))[1:]
    seg = p + '.segments.csv'
    ti, si = COLUMNS.index('timestamp_us'), COLUMNS.index('log_seq')

    def write_seg(statuses):
        with open(seg, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(M.SEG_COLS)
            for i, (r, st) in enumerate(zip(body, statuses), start=1):
                w.writerow([i, r[si], r[ti], 0, st])
    write_seg(['ok'] * (len(body) - 1) + ['duplicate'])
    assert len(M.read_esp_log(p)['log_seq']) == len(rows)
    write_seg(['ok'] * (len(body) - 1) + ['conflict'])
    with pytest.raises(ValueError, match='conflict'):
        M.read_esp_log(p)
    os.remove(seg)
    # any column count with the needed columns (old 85 or new 86 column logs)
    p2 = str(tmp_path / 'small.csv')
    write_esp_csv(p2, rows, columns=list(M.NEED) + ['traj_age_ms'])
    assert np.array_equal(M.read_esp_log(p2)['counts'], got['counts'])
    p3 = str(tmp_path / 'bad.csv')
    write_esp_csv(p3, rows, columns=[c for c in M.NEED if c != 'run_active'])
    with pytest.raises(ValueError, match='run_active'):
        M.read_esp_log(p3)


def write_frame_log_omega(path, task, k_run, k_edge):
    """Figure-8 frame log whose plant moves from k_edge (the true edge), RUN step k_run before it, with
    wheel speeds in the omega slots (not kinematically tied to the pose: only RS-4 reads them)."""
    n_run = int((H.Schedule(H.FIG8[task]).Tcmd + 2.5) / 1e-3)
    t = np.arange(n_run) * 1e-3
    pose, _ = fig8_pose(task, t)
    n = k_edge + n_run
    pose = np.vstack([np.zeros((k_edge, 3)), pose])
    vel = np.gradient(pose[:, :2], 1e-3, axis=0)
    wz = np.gradient(np.unwrap(pose[:, 2]), 1e-3)
    om = wheel_omega(n, k_edge)
    mosi = np.zeros((n, 40), dtype=np.uint8)
    for k in range(n):
        st = list(om[k]) + [pose[k, 0], pose[k, 1], pose[k, 2], vel[k, 0], vel[k, 1], wz[k]]
        on = k >= k_run + 3
        mosi[k] = np.frombuffer(P.encode_state(st, traj_ref=(0.3, 0.0, 0.1) if on else None,
                                               traj_flags=P.TRAJ_VALID if on else 0, seq=k), dtype=np.uint8)
    manifest = dict(period_us=1000, steps=n, plant='MecanumPlant', miso_check=dict(crc_bad=0, empty=0),
                    lifecycle=dict(outcome='completed', transitions=[dict(step=0, t=0, phase='ARM'),
                                                                     dict(step=k_run, t=0.4, phase='RUN')]))
    np.savez(path, manifest=np.array(json.dumps(manifest)), mosi=mosi, miso=np.zeros((n, 40), dtype=np.uint8))
    # the omega the scorer decodes (int16): the ESP32 counts are made from that, as the H7 would
    state, _, _, _ = H.decode_frames(mosi, np.zeros((n, 40), dtype=np.uint8))
    return state[:, 0:4]


def test_scorer_with_esp_log_decides_all_five(tmp_path):
    p, c = str(tmp_path / 'run.npz'), str(tmp_path / 'run.csv')
    om = write_frame_log_omega(p, 'O1-F05', k_run=388, k_edge=400)
    write_esp_csv(c, esp_lines(om, 400.0))
    out = H.score(p, 'O1-F05', esp_log=c)
    em, g = out['edge_map'], out['gate_v3']
    assert em['ok'] and em['edge_step'] == pytest.approx(400.0, abs=0.3) and em['offset_ms'] == pytest.approx(12.0, abs=0.3)
    assert out['t0_src'] == 'esp32_run_active_edge' and out['k_run'] == 388 and out['k0'] in (400, 401)
    assert g['decide'] == list(G.CONDS) and g['u_t'] == pytest.approx(em['u_t_s']) and g['complete']
    assert out['completed'] == 'true'
    assert all(g[c]['state'] == 'dat' for c in G.CONDS) and g['outcome'] == 'dat'   # car on the command
    js = H._json_safe(out)
    json.dumps(js)
    assert rl.validate_gate(js['gate_v3'], 'HIL')['complete']
    # same frame log, RUN step taken as t = 0 (no ESP32 log): the 12 ms offset shows in G3
    plain = H.score(p, 'O1-F05')
    assert plain['t0_src'] == 'lifecycle_RUN' and plain['gate_v3']['decide'] == list(G.DECIDE_HIL)
    assert plain['edge_map'] == dict(ok=False, reason='no --esp-log')
    assert plain['gate_v3']['g3']['q'] > g['g3']['q'] + 0.005           # about 0.729 m/s x 12 ms at the peak


def test_scorer_cli_esp_log_errors(tmp_path):
    p, c = str(tmp_path / 'run.npz'), str(tmp_path / 'run.csv')
    write_frame_log_omega(p, 'O1-F05', k_run=388, k_edge=400)
    with open(c, 'w') as f:
        f.write('timestamp_us,log_seq\n1,1\n')
    assert H.main(['--frame-log', p, '--task', 'O1-F05', '--esp-log', c]) == 2       # bad log: an error
    assert M.main(['--frame-log', p, '--esp-log', c]) == 2
