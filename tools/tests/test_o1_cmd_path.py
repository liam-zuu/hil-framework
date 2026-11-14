"""tools/o1_cmd_path.py: command path of an O1 floor run (O1 v11.19 3.5 (1)-(4)) on synthetic ESP32
logs built like the firmware: control tick 10 ms with jitter, schedule packets at 50 Hz stamped at
receive, run_active = motor on and a packet younger than 200 ms, command pose reset at the edge and
integrated in float32 with the mid-tick rotation of pos_ctrl.c (AGV 5bde4c5 67-81)."""
import csv
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import o1_cmd_path as cp  # noqa: E402

COLS = ['timestamp_us', 'tick', 'log_seq', 'motor_on', 'run_active', 'pos_reset_n', 'pos_x_ref', 'pos_y_ref',
        'sp_vx', 'sp_vy', 'sp_wz', 'traj_age_ms']


def us_events(rows, blk=0, cut=0, boot='00000001'):
    """'#us' start and '#us_end' of the motor-on run, firmware format (us_state.c 5bde4c5)."""
    mo = [r['log_seq'] for r in rows if r['motor_on'] == 1]
    return (f'#us,ev=start,boot={boot},on=1,swb=0,first_seq={mo[0]},line_drop=0',
            f'#us_end,boot={boot},on=1,swb=0,changes=0,blk_ticks={blk},cut_ticks={cut},ticks={len(mo)},'
            f'last_data_seq={mo[-1]},line_drop=0')


def make_log(task_id='O1-F05', drop_packets=(), drop_lines=(), motor_on_at=0.0, pkt_start=0.5,
             toggle_gap=None, reset_mid_at=None, big_at=None, pre_packet_at=None, extra_after=1.0,
             jitter=0.002, seed=1, allow_final_set=False, end_early=None):
    task = cp.Task(task_id, allow_final_set)
    rng = np.random.default_rng(seed)
    n_s = task.n_sched
    n_zero = int(math.ceil((task.duration + cp.o1_tasks.O1_TAIL_S) * 50 - 1e-9)) - n_s
    pkts = []                                           # (arrival s, (vx, vy, wz))
    for k in range(n_s + n_zero):
        if k in drop_packets:
            continue
        if toggle_gap and toggle_gap[0] <= k < toggle_gap[1]:
            continue
        val = task.f(k / 50) if k < n_s else (0.0, 0.0, 0.0)
        pkts.append((pkt_start + k / 50 + 0.001, val))
    if pre_packet_at is not None:
        pkts.insert(0, (pre_packet_at, (0.0, 0.0, 0.0)))
    pkts.sort(key=lambda p: p[0])
    t_end = pkt_start + (n_s + n_zero) / 50 + extra_after
    rows, t, i = [], 0.2, 0
    x = y = th = np.float32(0.0)
    running, reset_n, prev_t = False, 0, None
    j = -1
    while t < t_end:
        while j + 1 < len(pkts) and pkts[j + 1][0] <= t:
            j += 1
        age = -1 if j < 0 else int((t - pkts[j][0]) * 1000)
        mo = t >= motor_on_at
        fresh = j >= 0 and (t - pkts[j][0]) < 0.2
        ra = mo and fresh
        sp = pkts[j][1] if (fresh and j >= 0) else (0.0, 0.0, 0.0)
        dt = np.float32(t - prev_t) if prev_t is not None else np.float32(0.01)
        if ra:
            if not running or (reset_mid_at is not None and t >= reset_mid_at and reset_n == 1):
                x = y = th = np.float32(0.0)
                running = True
                reset_n += 1
            vx, vy, wz = (np.float32(v) for v in sp)
            thm = th + np.float32(0.5) * wz * dt
            c, s = np.float32(math.cos(thm)), np.float32(math.sin(thm))
            x = np.float32(x + (vx * c - vy * s) * dt)
            y = np.float32(y + (vx * s + vy * c) * dt)
            th = np.float32(th + wz * dt)
        else:
            running = False
        xr = float(x) + (9.0 if big_at is not None and t >= big_at else 0.0)
        rows.append(dict(timestamp_us=int(round(t * 1e6)), tick=i, log_seq=100 + i, motor_on=int(mo),
                         run_active=int(ra), pos_reset_n=reset_n, pos_x_ref=f'{xr:.4f}', pos_y_ref=f'{float(y):.4f}',
                         sp_vx=f'{sp[0]:.3f}', sp_vy=f'{sp[1]:.3f}', sp_wz=f'{sp[2]:.3f}', traj_age_ms=age))
        prev_t = t
        t += 0.010 + float(rng.uniform(-jitter, jitter))
        i += 1
        if end_early is not None and t > end_early:
            break
    return [r for r in rows if r['log_seq'] not in set(drop_lines)]


def write(tmp_path, rows, name='run.csv', events='auto', cols=COLS):
    if events == 'auto':
        events = us_events(rows)
    p = tmp_path / name
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    if events is not None:
        (tmp_path / (name + '.events.txt')).write_text('\n'.join(events) + '\n', encoding='utf-8')
    return str(p)


def run(tmp_path, rows, task='O1-F05', events='auto', cols=COLS, **kw):
    p = write(tmp_path, rows, events=events, cols=cols)
    t = cp.Task(task, kw.get('allow_final_set', False))
    rr, ev, dup, has_age = cp.read_log(p)
    return cp.analyse(rr, t, ev + cp.read_events(p + '.events.txt'), has_age)


def test_clean_run_has_no_flag_and_small_values(tmp_path):
    res = run(tmp_path, make_log())
    assert res['flags'] == [], res['flags']
    assert res['n_gap_in_window'] == 0 and res['n_hidden_pairs'] == 0
    assert res['cond2']['value_m'] < 0.008                        # on-line deviation only (synthetic jitter)
    assert res['cond4']['applies'] is True and res['cond4']['value_m'] < 0.012
    assert res['cond2']['value_all_m'] > res['cond2']['value_m']  # the every-pair bound is looser
    c3 = res['cond3']
    assert c3['value_ms'] is not None and 20 < c3['value_ms'] < 60  # 50 Hz packets, 10 ms ticks
    assert res['cond2']['value_m'] == res['cond2']['anchor_max_m']


def test_lost_lines_use_the_pair_bound_exactly(tmp_path):
    rows = make_log()
    edge = next(r['log_seq'] for r in rows if r['run_active'] == 1)
    lost = {edge + 200, edge + 201, edge + 202}
    res = run(tmp_path, [r for r in rows if r['log_seq'] not in lost])
    assert res['n_gap_in_window'] == 1 and res['n_hidden_pairs'] == 1
    c2 = res['cond2']
    by = {r['log_seq']: r for r in rows}
    a, b = by[edge + 199], by[edge + 203]
    t = cp.Task('O1-F05')
    da, db = t.lane_dist(float(a['pos_x_ref']), float(a['pos_y_ref'])), t.lane_dist(float(b['pos_x_ref']), float(b['pos_y_ref']))
    dt = (b['timestamp_us'] - a['timestamp_us']) / 1e6
    expect = (da + db + t.v * dt) / 2 + cp.round_term(t.v, dt, 4, False)
    assert c2['pair_hidden_seq'] == [edge + 199, edge + 203]
    assert c2['pair_hidden_max_m'] == pytest.approx(expect, rel=1e-9)
    assert c2['value_m'] >= expect


def test_lost_first_packets_shift_the_schedule_phase(tmp_path):
    clean = run(tmp_path, make_log())
    (tmp_path / 'b').mkdir()
    late = run(tmp_path / 'b', make_log(drop_packets=range(0, 25)))   # 0.5 s of packets lost
    assert late['cond4']['value_m'] > clean['cond4']['value_m'] + 0.01


def test_run_active_toggle_is_no_evidence(tmp_path):
    res = run(tmp_path, make_log(toggle_gap=(200, 215)))          # 300 ms without packets mid run
    assert res['run_active_toggle'] is True
    assert any(f.startswith('run_active_toggle') for f in res['flags'])
    assert res['cond2'].get('no_evidence') and res['cond4'].get('no_evidence')


def test_pos_reset_change_is_cond1(tmp_path):
    res = run(tmp_path, make_log(reset_mid_at=3.0))
    assert any(f.startswith('cond1_pos_reset') for f in res['flags'])


def test_coordinates_over_8_m(tmp_path):
    res = run(tmp_path, make_log(big_at=3.0))
    assert any(f.startswith('coord_over_8m') for f in res['flags'])
    assert res['cond2'].get('no_evidence')


def test_packet_before_the_edge_motor_off(tmp_path):
    # the schedule starts while the motor is off; the motor comes on 1 s later, mid schedule
    res = run(tmp_path, make_log(motor_on_at=1.5))
    assert any(f.startswith('pre_edge_packet') for f in res['flags'])


def test_old_packet_before_the_edge_motor_on(tmp_path):
    # a stray packet 250 ms before the schedule: run_active rises on it, falls, rises at the schedule
    res = run(tmp_path, make_log(pre_packet_at=0.25))
    assert res['edge_seq'] < min(r['log_seq'] for r in make_log() if r['run_active'] == 1)
    assert any(f.startswith('run_active_toggle') for f in res['flags'])


def test_line_before_the_edge_lost_makes_the_edge_unknown(tmp_path):
    rows = make_log()
    edge = next(r['log_seq'] for r in rows if r['run_active'] == 1)
    res = run(tmp_path, [r for r in rows if r['log_seq'] != edge - 1])
    assert res['decided'] is False and any(f.startswith('edge_unknown') for f in res['flags'])


def test_log_ending_before_the_window_end(tmp_path):
    res = run(tmp_path, make_log(end_early=6.0))
    assert any(f.startswith('no_end_anchor') for f in res['flags'])
    assert res['cond2'].get('no_evidence')


def test_ultrasonic_ticks_or_no_us_end(tmp_path):
    rows = make_log()
    res = run(tmp_path, rows, events=us_events(rows, blk=3, cut=2))
    assert res['cond4']['applies'] is False and res['flags'] == []
    (tmp_path / 'b').mkdir()
    res = run(tmp_path / 'b', make_log(), events=None)
    assert any(f.startswith('us_unknown') for f in res['flags'])


def test_duplicate_lines_dropped_conflict_is_an_error(tmp_path):
    rows = make_log()
    p = write(tmp_path, rows + rows[10:12])
    rr, _, dup, _ = cp.read_log(p)
    assert dup == 2 and len(rr) == len(rows)
    bad = dict(rows[10], pos_x_ref='1.2345')
    p = write(tmp_path, rows + [bad], name='bad.csv')
    with pytest.raises(ValueError):
        cp.read_log(p)


def test_K1_final_set(tmp_path):
    res = run(tmp_path, make_log('O1-K1', allow_final_set=True), task='O1-K1', allow_final_set=True)
    assert res['flags'] == [] and res['cond4']['value_m'] < 0.01 and res['cond2']['value_m'] < 0.01


def test_cli_exit_codes(tmp_path):
    p = write(tmp_path, make_log())
    assert cp.main(['--log', p, '--task', 'O1-F05', '--json', str(tmp_path / 'o.json')]) == 0
    assert cp.main(['--log', p, '--task', 'O1-F05', '--json', str(tmp_path / 'o.json')]) == 2   # no overwrite
    p2 = write(tmp_path, make_log(motor_on_at=1.5), name='late.csv')
    assert cp.main(['--log', p2, '--task', 'O1-F05']) == 1


# ---------------------------------------------------------------- Codex Sol round 1 (CP1-1 .. CP1-5)
def test_CP1_1_reset_after_the_window_with_the_off_lines_lost(tmp_path):
    t = cp.Task('O1-F05')
    k0 = t.n_sched + 40                                 # packets lost 0.3 s in the zero tail: timeout
    rows = make_log(toggle_gap=(k0, k0 + 15))
    rows = [r for r in rows if not (r['run_active'] == 0 and r['log_seq'] > rows[0]['log_seq'] + 100)]
    seqs = sorted({r['pos_reset_n'] for r in rows})
    assert len(seqs) > 2                                # 0 before, 1 the run, 2 after the hidden rise
    res = run(tmp_path, rows)
    assert any(f.startswith('cond1_pos_reset') for f in res['flags'])


def test_CP1_2_missing_columns_and_cells(tmp_path):
    rows = make_log()
    res = run(tmp_path, rows, cols=[c for c in COLS if c != 'traj_age_ms'])
    assert any(f.startswith('no_traj_age_column') for f in res['flags'])
    with pytest.raises(ValueError):
        cp.read_log(write(tmp_path, rows, name='nr.csv', cols=[c for c in COLS if c != 'pos_reset_n']))
    edge = next(r['log_seq'] for r in rows if r['run_active'] == 1)
    holed = [dict(r, traj_age_ms='') if r['log_seq'] == edge + 50 else r for r in rows]
    (tmp_path / 'b').mkdir()
    res = run(tmp_path / 'b', holed)
    assert any(f.startswith('cond3_age_missing') for f in res['flags'])
    bad = [dict(r, pos_reset_n='') if r['log_seq'] == edge + 50 else r for r in rows]
    with pytest.raises(ValueError):
        cp.read_log(write(tmp_path, bad, name='empty.csv'))


def test_CP1_3_us_end_of_another_motor_run_is_not_used(tmp_path):
    rows = make_log()
    other = ('#us,ev=start,boot=00000001,on=1,swb=0,first_seq=10,line_drop=0',
             '#us_end,boot=00000001,on=1,swb=0,changes=0,blk_ticks=1,cut_ticks=0,ticks=30,last_data_seq=40,line_drop=0')
    res = run(tmp_path, rows, events=other)
    assert any(f.startswith('us_unknown') for f in res['flags'])
    (tmp_path / 'b').mkdir()
    res = run(tmp_path / 'b', rows, events=other + us_events(rows))
    assert res['cond4']['applies'] is True and res['flags'] == []
    (tmp_path / 'c').mkdir()
    ev = us_events(rows, boot='00000002')[:1] + us_events(rows)[1:]     # end of another boot
    res = run(tmp_path / 'c', rows, events=ev)
    assert any(f.startswith('us_unknown') for f in res['flags'])


def test_CP1_4_manual_command_before_the_edge_is_not_a_packet(tmp_path):
    rows = make_log()
    edge = next(r['log_seq'] for r in rows if r['run_active'] == 1)
    rows = [dict(r, sp_vx='0.200') if edge - 20 <= r['log_seq'] < edge - 5 else r for r in rows]
    assert all(r['traj_age_ms'] == -1 for r in rows if edge - 20 <= r['log_seq'] < edge - 5)
    res = run(tmp_path, rows)
    assert res['flags'] == []


def test_CP1_5_edge_age_above_30_ms_is_a_note(tmp_path):
    rows = make_log()
    edge = next(r['log_seq'] for r in rows if r['run_active'] == 1)
    rows = [dict(r, traj_age_ms=39) if r['log_seq'] == edge else r for r in rows]
    res = run(tmp_path, rows)
    assert res['flags'] == [] and any(n.startswith('edge_age') for n in res['notes'])
