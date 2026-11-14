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


# ---------------------------------------------------------------- Codex Sol round 2 (CP2-1 .. CP2-4)
def gain_line(first_seq, boot='00000001'):
    """'#gain' line of a motor-on run, trailing fields of logger.c 5bde4c5 184 (gain fields shortened)."""
    return f'#gain,v=1,id=1,ctrl=PID,first_seq={first_seq},line_drop=0,boot={boot}'


def us_end_line(last, blk=0, cut=0, boot='00000001'):
    return (f'#us_end,boot={boot},on=1,swb=0,changes=0,blk_ticks={blk},cut_ticks={cut},ticks=10,'
            f'last_data_seq={last},line_drop=0')


def test_CP2_1_us_end_of_a_later_run_is_not_used(tmp_path):
    rows = make_log()
    start, _ = us_events(rows)
    last = rows[-1]['log_seq']
    # the end of THIS run lost; a later motor-on run whose '#us' start is lost: only '#gain' marks it
    res = run(tmp_path, rows, events=(start, gain_line(rows[0]['log_seq']), gain_line(last + 10),
                                      us_end_line(last + 50)))
    assert any(f.startswith('us_unknown') for f in res['flags'])
    # same with no '#gain' at all: the motor-off data line after the run marks the boundary
    off = [dict(rows[-1], log_seq=last + k, tick=rows[-1]['tick'] + k, motor_on=0, run_active=0,
                timestamp_us=rows[-1]['timestamp_us'] + 100000 * k) for k in (1, 2)]
    (tmp_path / 'b').mkdir()
    res = run(tmp_path / 'b', rows + off, events=(start, us_end_line(last + 50)))
    assert any(f.startswith('us_unknown') for f in res['flags'])
    # the right end, before the motor-off line, is still used
    (tmp_path / 'c').mkdir()
    res = run(tmp_path / 'c', rows + off, events=(start, us_end_line(last)))
    assert res['flags'] == [] and res['cond4']['applies'] is True


def run_end_line(last, seq):
    """'#run_end' of logger.c 5bde4c5 202: the 3 copies differ in seq, no boot field."""
    return f'#run_end,seq={seq},last_data_seq={last},log_qdrop=0,log_send_err=0,udp_drop=0,udp_take=460'


def test_CP2_1_run_end_bounds_the_us_end(tmp_path):
    # this run's '#us_end' lost; the later run lost its '#gain' and '#us' start and has no motor-off line
    # in between; only this run's '#run_end' is left (Codex round 3)
    rows = make_log()
    start, _ = us_events(rows)
    last = rows[-1]['log_seq']
    ends = tuple(run_end_line(last, last + 1 + k) for k in range(3))
    res = run(tmp_path, rows, events=(start,) + ends + (us_end_line(last + 13),))
    assert any(f.startswith('us_unknown') for f in res['flags'])
    (tmp_path / 'b').mkdir()
    res = run(tmp_path / 'b', rows, events=(start,) + ends + (us_end_line(last),))
    assert res['flags'] == [] and res['cond4']['applies'] is True


def test_CP3_2_gain_set_line_is_not_a_run_start(tmp_path):
    rows = make_log()
    _, end = us_events(rows)
    res = run(tmp_path, rows, events=('#gain,v=1,id=2,ctrl=PID,first_seq=0,line_drop=0,boot=00000001', end))
    assert any(f.startswith('us_unknown') for f in res['flags'])


def test_CP3_2_new_reset_count_only_on_run_active_0_lines(tmp_path):
    # a hidden rise after the window whose run_active = 1 lines are all lost: only the following
    # run_active = 0 lines carry the new count (pos_ctrl.c 132-135 outputs it on every call)
    rows = make_log()
    last = rows[-1]
    tail = [dict(last, log_seq=last['log_seq'] + 40 + k, tick=last['tick'] + 40 + k, run_active=0,
                 pos_reset_n=last['pos_reset_n'] + 1, timestamp_us=last['timestamp_us'] + 400000 + 10000 * k)
            for k in range(3)]
    res = run(tmp_path, rows + tail)
    assert any(f.startswith('cond1_pos_reset') for f in res['flags'])


def test_CP2_1_us_start_between_the_edge_and_the_end(tmp_path):
    rows = make_log()
    start, _ = us_events(rows)
    last = rows[-1]['log_seq']
    later = f'#us,ev=start,boot=00000001,on=1,swb=0,first_seq={last + 10},line_drop=0'
    res = run(tmp_path, rows, events=(start, later, us_end_line(last + 50)))
    assert any(f.startswith('us_unknown') for f in res['flags'])


def test_CP2_2_gain_line_ties_the_end_when_the_us_start_is_lost(tmp_path):
    full = make_log()
    _, end = us_events(full)                             # metadata of the firmware, not of the lines kept
    rows = full[:-3]                                     # the last lines lost
    ev = (gain_line(full[0]['log_seq']), '#gain,v=1,id=2,ctrl=PID,first_seq=0,line_drop=0,boot=00000001', end)
    res = run(tmp_path, rows, events=ev)
    assert res['flags'] == [] and res['cond4']['applies'] is True
    (tmp_path / 'b').mkdir()                             # '#gain' of another boot does not tie it
    res = run(tmp_path / 'b', rows, events=(gain_line(full[0]['log_seq'], boot='00000002'), end))
    assert any(f.startswith('us_unknown') for f in res['flags'])


def test_CP2_3_negative_age_on_a_run_active_line_is_a_flag(tmp_path):
    rows = make_log()
    edge = next(r['log_seq'] for r in rows if r['run_active'] == 1)
    res = run(tmp_path, [dict(r, traj_age_ms=-1) if r['log_seq'] == edge else r for r in rows])
    assert any(f.startswith('ra_age_negative') for f in res['flags'])
    (tmp_path / 'b').mkdir()
    res = run(tmp_path / 'b', [dict(r, traj_age_ms=-1) if r['log_seq'] == edge + 300 else r for r in rows])
    assert any(f.startswith('ra_age_negative') for f in res['flags'])


def test_CP2_4_empty_age_before_the_edge(tmp_path):
    rows = make_log()
    edge = next(r['log_seq'] for r in rows if r['run_active'] == 1)
    res = run(tmp_path, [dict(r, traj_age_ms='') if r['log_seq'] == edge - 20 else r for r in rows])
    assert any(f.startswith('pre_edge_age_missing') for f in res['flags'])


def test_first_packet_one_line_before_the_edge_is_a_number(tmp_path):
    # traj_task moves the packet every 10 ms: the line before the edge already has a fresh age (real logs)
    rows = make_log()
    edge = next(r['log_seq'] for r in rows if r['run_active'] == 1)
    rows = [dict(r, traj_age_ms=0) if r['log_seq'] == edge - 1 else r for r in rows]
    res = run(tmp_path, rows)
    assert res['flags'] == [] and 0 < res['pre_edge_packet_lag_ms'] < 30
    assert any(n.startswith('pre_edge_packet_lag') for n in res['notes'])


def test_pre_edge_packet_lag_interval_and_200_ms(tmp_path):
    # lag in [D + a - S, D + a + 1) ms (age read after timestamp_us, floored; CP3-1): flag when the lower
    # bound reaches 200 ms, undecided across 200 ms, a note below
    rows = make_log()
    edge = next(r['log_seq'] for r in rows if r['run_active'] == 1)
    t0 = next(r['timestamp_us'] for r in rows if r['log_seq'] == edge)
    # line edge-1 put exactly 10 ms before the edge: for line edge-2 the lower bound is 10 ms + a
    base = [dict(r, timestamp_us=t0 - 10000) if r['log_seq'] == edge - 1 else r for r in rows]
    for age, want, sub in ((190, 'pre_edge_packet:', 'a'),            # lower bound exactly 200 ms
                           (189, 'pre_edge_packet_undecided', 'b'),   # 199 ms .. above 200 ms
                           (150, None, 'c')):
        (tmp_path / sub).mkdir()
        res = run(tmp_path / sub, [dict(r, traj_age_ms=age) if r['log_seq'] == edge - 2 else r for r in base])
        assert res['pre_edge_packet_lag_lo_ms'] == pytest.approx(10.0 + age)
        assert res['pre_edge_packet_lag_lo_ms'] <= res['pre_edge_packet_lag_ms'] < res['pre_edge_packet_lag_hi_ms']
        got = [f for f in res['flags'] if f.startswith('pre_edge_packet')]
        if want is None:
            assert got == [] and any(n.startswith('pre_edge_packet_lag') for n in res['notes'])
        else:
            assert len(got) == 1 and got[0].startswith(want), (age, got)


def test_pre_edge_lag_interval_holds_the_true_receive_time():
    # firmware timing (main.c 532-540): tick start t, age read at t + s, floored; the true lag must be
    # inside the reported interval for any s < next tick and any receive time
    rng = np.random.default_rng(7)
    for _ in range(2000):
        t_line, t_next, t_edge = 0, int(rng.integers(5000, 15000)), int(rng.integers(15000, 400000))
        t_ref = t_line + int(rng.integers(0, t_next))
        t_rx = t_ref - int(rng.integers(0, 199000))
        a = max(0, (t_ref - t_rx) // 1000)
        d, s = (t_edge - t_line) / 1e3, (t_next - t_line) / 1e3
        true = (t_edge - t_rx) / 1e3
        assert d + a - s <= true < d + a + 1.0


def test_events_file_with_the_agv_logger_time_prefix(tmp_path):
    # agv_logger.py writes '<PC time ISO> <payload>' per line (880, 894), each event 3 times
    rows = make_log()
    start, end = us_events(rows, blk=4)
    ev = [f'2026-11-14T16:43:44.40{k}400 {e}' for e in (start, end) for k in range(3)]
    res = run(tmp_path, rows, events=ev)
    assert res['cond4']['applies'] is False and res['flags'] == []
    assert cp.read_events(str(tmp_path / 'run.csv.events.txt')) == [start] * 3 + [end] * 3


def test_CP2_4_cli_prints_the_edge_age_note(tmp_path, capsys):
    rows = make_log()
    edge = next(r['log_seq'] for r in rows if r['run_active'] == 1)
    p = write(tmp_path, [dict(r, traj_age_ms=39) if r['log_seq'] == edge else r for r in rows])
    assert cp.main(['--log', p, '--task', 'O1-F05']) == 0
    out = capsys.readouterr().out
    assert '-- edge_age: traj_age_ms 39' in out and out.rstrip().endswith('#ket_qua,ma=0')
