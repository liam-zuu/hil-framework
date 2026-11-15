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


def nominal_sched(task):
    """(sched_first_s, sched_last_s) of an on-time sender: packet k sent at k/50 s (make_log)."""
    return 0.0, (task.n_sched - 1) / 50


def sender_log(tmp_path, task_id='O1-F05', sched='nominal', name='run_sender.txt'):
    """'#test_end' of pc_sender.test_loop with the two fields agreed with CODE 15/11 at the end."""
    t = cp.Task(task_id)
    a, b = nominal_sched(t) if sched == 'nominal' else sched
    p = tmp_path / name
    p.write_text(f'#test_start,test={task_id},t_epoch=1.000,iso=x,duration_s={t.duration:g},send_hz=50\n'
                 f'#test_end,test={task_id},t_epoch=20.000,zeros=50,tail_planned=50,sent={t.n_sched},'
                 f'planned={t.n_sched},complete=1,sched_first_s={a},sched_last_s={b}\n', encoding='utf-8')
    return str(p)


def run(tmp_path, rows, task='O1-F05', events='auto', cols=COLS, sched='nominal', bao=None, sender_ok=None, **kw):
    p = write(tmp_path, rows, events=events, cols=cols)
    t = cp.Task(task, kw.get('allow_final_set', False))
    rr, ev, dup, has_age = cp.read_log(p)
    return cp.analyse(rr, t, ev + cp.read_events(p + '.events.txt'), has_age,
                      nominal_sched(t) if sched == 'nominal' else sched, bao, sender_ok)


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
    s = sender_log(tmp_path)
    assert cp.main(['--log', p, '--task', 'O1-F05', '--sender', s, '--json', str(tmp_path / 'o.json')]) == 0
    assert cp.main(['--log', p, '--task', 'O1-F05', '--sender', s, '--json', str(tmp_path / 'o.json')]) == 2
    p2 = write(tmp_path, make_log(motor_on_at=1.5), name='late.csv')
    assert cp.main(['--log', p2, '--task', 'O1-F05', '--sender', s]) == 1
    assert cp.main(['--log', p, '--task', 'O1-F05']) == 1                     # no sender log (B81c-1)


def test_B81c_1_window_end_from_the_real_send_times(tmp_path):
    rows = make_log()
    t = cp.Task('O1-F05')
    a, b = nominal_sched(t)
    res = run(tmp_path, rows)
    assert res['flags'] == [] and res['sched_send_span_s'] == pytest.approx(b - a)
    assert res['t_w1_us'] - res['t_w0_us'] == round(((b - a) * (1 + cp.E_MAX) + cp.T_SEND) * 1e6)
    (tmp_path / 'late').mkdir()                        # the sender was late, the last packet 0.3 s later
    late = run(tmp_path / 'late', rows, sched=(a, b + 0.3))
    assert late['t_w1_us'] - res['t_w1_us'] == pytest.approx(0.3 * (1 + cp.E_MAX) * 1e6, abs=1)
    for sub, sched in (('none', None),):
        (tmp_path / sub).mkdir()
        res = run(tmp_path / sub, rows, sched=sched)
        assert any(f.startswith('sched_times_missing') for f in res['flags'])
        assert res['cond2'].get('no_evidence') and res['cond3'].get('no_evidence') and res['cond4'].get('no_evidence')


def test_B81c_1_read_sender(tmp_path):
    assert cp.read_sender(sender_log(tmp_path, sched=(0.000012, 13.640031)), 'O1-F05') == (0.000012, 13.640031)
    assert cp.read_sender(sender_log(tmp_path, name='k.txt'), 'O1-F08') is None             # other task
    assert cp.read_sender(str(tmp_path / 'absent.txt'), 'O1-F05') is None
    p = tmp_path / 'cut.txt'                           # interrupted: last packet not sent, field empty
    p.write_text('#test_interrupted\n#test_end,test=O1-F05,t_epoch=5.0,zeros=0,tail_planned=50,sent=200,'
                 'planned=683,complete=0,sched_first_s=0.000010,sched_last_s=\n', encoding='utf-8')
    assert cp.read_sender(str(p), 'O1-F05') is None
    old = tmp_path / 'old.txt'                         # sender before the fields (C6 10/11)
    old.write_text('#test_end,test=O1-F05,t_epoch=1794322949.749,sent=683,planned=683,complete=1\n', encoding='utf-8')
    assert cp.read_sender(str(old), 'O1-F05') is None
    two = tmp_path / 'two.txt'                         # two different #test_end lines of the task
    two.write_text('#test_end,test=O1-F05,complete=1,sched_first_s=0.0,sched_last_s=13.6\n'
                   '#test_end,test=O1-F05,complete=1,sched_first_s=0.0,sched_last_s=1.0\n', encoding='utf-8')
    with pytest.raises(ValueError):
        cp.read_sender(str(two), 'O1-F05')
    bad = tmp_path / 'bad.txt'
    bad.write_text('#test_end,test=O1-F05,complete=1,sched_first_s=2.0,sched_last_s=1.0\n', encoding='utf-8')
    with pytest.raises(ValueError):
        cp.read_sender(str(bad), 'O1-F05')


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
    # lines edge-1, edge-2 put exactly 10, 20 ms before the edge: for line edge-2 the interval is
    # [10 ms + a, 21 ms + a)
    base = [dict(r, timestamp_us=t0 - 10000) if r['log_seq'] == edge - 1 else
            dict(r, timestamp_us=t0 - 20000) if r['log_seq'] == edge - 2 else r for r in rows]
    for age, want, sub in ((190, 'pre_edge_packet:', 'a'),            # lower bound exactly 200 ms
                           (189, 'pre_edge_packet_undecided', 'b'),   # 199 ms .. above 200 ms
                           (180, 'pre_edge_packet_undecided', 'd'),   # 190 .. 201 ms
                           (179, None, 'e'),                          # 189 .. 200 ms: open bound, below (CP4-1)
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


def test_CP4_3_lag_interval_from_analyse_holds_the_true_receive_time(tmp_path):
    # the line before the edge is 10 ms before it; its age is read at t_ref (main.c 532-540), floored:
    # the two extremes of the firmware timing must stay inside the interval analyse reports
    rows = make_log()
    edge = next(r['log_seq'] for r in rows if r['run_active'] == 1)
    t0 = next(r['timestamp_us'] for r in rows if r['log_seq'] == edge)
    t_line = t0 - 10000
    cases = (('hi', t_line, t_line - 150999),                  # age read at the tick start, true = D + a + 0.999
             ('lo', t_line + 9999, t_line + 9999 - 150000))    # age read just before the next tick, true ~ lower
    for sub, t_ref, t_rx in cases:
        a = (t_ref - t_rx) // 1000
        true = (t0 - t_rx) / 1e3
        lines = [dict(r, timestamp_us=t_line, traj_age_ms=a) if r['log_seq'] == edge - 1 else r for r in rows]
        (tmp_path / sub).mkdir()
        res = run(tmp_path / sub, lines)
        assert res['pre_edge_packet_lag_lo_ms'] <= true < res['pre_edge_packet_lag_hi_ms'], (sub, true, res)


def test_CP4_3_window_uses_the_send_time_difference(tmp_path):
    rows = make_log()
    t = cp.Task('O1-F05')
    a, b = nominal_sched(t)
    r0 = run(tmp_path, rows, sched=(a, b))
    (tmp_path / 's').mkdir()
    r2 = run(tmp_path / 's', rows, sched=(a + 2.0, b + 2.0))          # same span, later start
    assert r2['t_w1_us'] == r0['t_w1_us'] and r2['sched_send_span_s'] == pytest.approx(b - a)


def test_K1_enclosing_window_to_test_end(tmp_path):
    # no send time of the last schedule packet: the window runs to #test_end (O1 v11.24 3.5, K-1)
    rows = make_log()
    t = cp.Task('O1-F05')
    nominal = (t.n_sched - 1) / 50
    n_zero = int(math.ceil((t.duration + cp.o1_tasks.O1_TAIL_S) * 50 - 1e-9)) - t.n_sched
    bao = nominal + n_zero / 50 + 0.05                  # #test_start before packet 0, #test_end after the tail
    res = run(tmp_path, rows, sched=None, bao=bao)
    assert res['window_source'] == 'test_end' and res['flags'] == [], res['flags']
    assert res['t_w1_us'] - res['t_w0_us'] == round((bao * (1 + cp.E_MAX) + cp.T_SEND) * 1e6)
    (tmp_path / 'x').mkdir()
    sch = run(tmp_path / 'x', rows)
    assert res['t_w1_us'] > sch['t_w1_us']              # the enclosing window holds the schedule window
    for sub, sched, b, flag in (('short', None, nominal - 0.5, 'test_end_span_short'),
                                ('mismatch', nominal_sched(t), nominal - 0.5, 'window_mismatch')):
        (tmp_path / sub).mkdir()
        r = run(tmp_path / sub, rows, sched=sched, bao=b)
        assert any(f.startswith(flag) for f in r['flags']), (sub, r['flags'])
        assert r['cond2'].get('no_evidence') and r['cond3'].get('no_evidence') and r['cond4'].get('no_evidence')
    (tmp_path / 'both').mkdir()                          # both present and agreeing: the schedule window
    r = run(tmp_path / 'both', rows, sched=nominal_sched(t), bao=bao)
    assert r['window_source'] == 'sched' and r['flags'] == [] and r['t_w1_us'] == sch['t_w1_us']


def test_K1_read_sender_bao(tmp_path):
    assert cp.read_sender_bao(sender_log(tmp_path), 'O1-F05') == pytest.approx(19.0)
    assert cp.read_sender_bao(sender_log(tmp_path, name='k.txt'), 'O1-F08') is None
    old = tmp_path / 'c6.txt'                           # C6 10/11 sender log, no sched fields
    old.write_text('#test_start,test=O1-F05,t_epoch=1794322935.904,iso=2026-11-10T22:02:15,duration_s=13.6518,'
                   'send_hz=50,dst=192.168.1.69:5010\n'
                   '#test_end,test=O1-F05,t_epoch=1794322949.749,sent=683,planned=683,complete=1\n', encoding='utf-8')
    assert cp.read_sender(str(old), 'O1-F05') is None
    assert cp.read_sender_bao(str(old), 'O1-F05') == pytest.approx(13.845, abs=1e-6)
    no_start = tmp_path / 'ns.txt'
    no_start.write_text('#test_end,test=O1-F05,t_epoch=5.0,complete=1\n', encoding='utf-8')
    assert cp.read_sender_bao(str(no_start), 'O1-F05') is None
    two = tmp_path / 'two.txt'
    two.write_text('#test_start,test=O1-F05,t_epoch=1.0\n#test_end,test=O1-F05,t_epoch=15.0\n'
                   '#test_end,test=O1-F05,t_epoch=16.0\n', encoding='utf-8')
    with pytest.raises(ValueError):
        cp.read_sender_bao(str(two), 'O1-F05')


def test_CP4_2_timestamp_going_back_with_the_tick_increasing(tmp_path):
    rows = make_log()
    edge = next(r['log_seq'] for r in rows if r['run_active'] == 1)
    by = {r['log_seq']: r for r in rows}
    clean = run(tmp_path, rows)
    end = next(r['log_seq'] for r in rows if r['timestamp_us'] >= clean['t_w1_us'])
    for sub, seq, t_new in (('mid', edge + 100, by[edge + 99]['timestamp_us'] - 5000),    # inside the window
                            ('end', end + 1, by[end]['timestamp_us'] - 10000)):           # after the end anchor
        (tmp_path / sub).mkdir()
        res = run(tmp_path / sub, [dict(r, timestamp_us=t_new) if r['log_seq'] == seq else r for r in rows])
        assert any(f.startswith('pair_not_increasing') for f in res['flags']), sub
        assert res['cond2'].get('no_evidence') and res['cond4'].get('no_evidence'), sub


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


def test_B81c_7_tick_or_time_not_increasing_is_no_evidence(tmp_path):
    rows = make_log()
    edge = next(r['log_seq'] for r in rows if r['run_active'] == 1)
    by = {r['log_seq']: r for r in rows}
    for sub, change in (('a', dict(tick=by[edge + 99]['tick'])),                      # tick repeats
                        ('b', dict(tick=0)),                                          # counter reset
                        ('c', dict(timestamp_us=by[edge + 99]['timestamp_us']))):     # time repeats
        (tmp_path / sub).mkdir()
        res = run(tmp_path / sub, [dict(r, **change) if r['log_seq'] == edge + 100 else r for r in rows])
        assert any(f.startswith('pair_not_increasing') for f in res['flags']), sub
        assert res['cond2'].get('no_evidence') and res['cond4'].get('no_evidence'), sub


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
    assert cp.main(['--log', p, '--task', 'O1-F05', '--sender', sender_log(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert '-- edge_age: traj_age_ms 39' in out and out.rstrip().endswith('#ket_qua,ma=0')


# ---------------------------------------------------------------- Codex Sol round 5 (CP5-1 .. CP5-4), sender check
def test_CP5_1_motor_off_line_coming_back_inside_the_window(tmp_path):
    rows = make_log()
    clean = run(tmp_path, rows)
    last = rows[-1]
    back = dict(last, log_seq=last['log_seq'] + 4, tick=last['tick'] + 4, motor_on=0, run_active=0,
                timestamp_us=clean['t_w1_us'] - 5000)          # motor off, time back inside the window
    (tmp_path / 'b').mkdir()
    res = run(tmp_path / 'b', rows + [back])
    assert any(f.startswith('pair_not_increasing') for f in res['flags'])


def test_CP5_2_enclosing_span_must_reach_the_nominal_schedule(tmp_path):
    rows = make_log()
    nominal = (cp.Task('O1-F05').n_sched - 1) / 50
    for sub, bao, short in (('eq', nominal, False), ('lt', nominal - 0.02, True)):
        (tmp_path / sub).mkdir()
        res = run(tmp_path / sub, rows, sched=None, bao=bao)
        assert any(f.startswith('test_end_span_short') for f in res['flags']) is short, (sub, res['flags'])


def test_CP5_3_wall_clock_window_is_noted(tmp_path):
    rows = make_log()
    nominal = (cp.Task('O1-F05').n_sched - 1) / 50
    res = run(tmp_path, rows, sched=None, bao=nominal + 1.0)
    assert any(n.startswith('window_from_wall_clock') for n in res['notes'])
    (tmp_path / 's').mkdir()
    res = run(tmp_path / 's', rows)
    assert not any(n.startswith('window_from_wall_clock') for n in res['notes'])


def test_CP5_4_end_anchor_needs_a_positive_L(tmp_path):
    # the first line at or after t_w1 whose next line has the same timestamp (L = 0) is not the end anchor
    # of (3) (the run is flagged anyway by pair_not_increasing; this checks (3) on its own)
    rows = make_log()
    clean = run(tmp_path, rows)
    fin = next(r for r in rows if r['timestamp_us'] >= clean['t_w1_us'])
    nxt = fin['log_seq'] + 1
    (tmp_path / 'b').mkdir()
    res = run(tmp_path / 'b', [dict(r, timestamp_us=fin['timestamp_us']) if r['log_seq'] == nxt else r for r in rows])
    c3 = res['cond3']
    assert c3.get('end_anchor_seq') != fin['log_seq'], c3


def test_CP5_4_cli_with_a_sender_log_without_send_times(tmp_path):
    p = write(tmp_path, make_log())
    t = cp.Task('O1-F05')
    s = tmp_path / 'old_sender.txt'
    s.write_text(f'#test_start,test=O1-F05,t_epoch=1000.000,iso=x,duration_s={t.duration:g},send_hz=50\n'
                 f'#test_end,test=O1-F05,t_epoch={1000.0 + (t.n_sched - 1) / 50 + 1.0:.3f},sent={t.n_sched},'
                 f'planned={t.n_sched},complete=1\n', encoding='utf-8')
    out = tmp_path / 'o.json'
    assert cp.main(['--log', p, '--task', 'O1-F05', '--sender', str(s), '--json', str(out)]) == 0
    import json
    res = json.loads(out.read_text(encoding='utf-8'))
    assert res['window_source'] == 'test_end' and res['test_end_span_s'] == pytest.approx((t.n_sched - 1) / 50 + 1.0)


def test_sender_interrupted_or_incomplete_is_not_scored(tmp_path):
    t = cp.Task('O1-F05')
    base = (f'#test_start,test=O1-F05,t_epoch=1.000\n#test_end,test=O1-F05,t_epoch=20.000,sent={t.n_sched},'
            f'planned={t.n_sched},complete=1,sched_first_s=0.000010,sched_last_s={(t.n_sched - 1) / 50:.6f}\n')
    cases = (('ok', base, True), ('int', '#test_interrupted\n' + base, False),
             ('c0', base.replace('complete=1', 'complete=0'), False))
    for name, text, ok in cases:
        f = tmp_path / f'{name}.txt'
        f.write_text(text, encoding='utf-8')
        assert cp.read_sender_complete(str(f), 'O1-F05')[0] is ok, name
    assert cp.read_sender_complete(str(tmp_path / 'none.txt'), 'O1-F05')[0] is None
    rows = make_log()
    res = run(tmp_path, rows, sender_ok=(False, '#test_interrupted'))
    assert any(f.startswith('sender_incomplete') for f in res['flags'])
    assert res['cond2'].get('no_evidence') and res['cond3'].get('no_evidence') and res['cond4'].get('no_evidence')
    p = write(tmp_path, rows, name='r2.csv')
    f = tmp_path / 'int.txt'
    assert cp.main(['--log', p, '--task', 'O1-F05', '--sender', str(f)]) == 1


def test_tick_reset_on_motor_off_lines_is_not_a_flag(tmp_path):
    # motor off after the run: the firmware counter restarts (vel_ctrl.c 22-23, 41, 49); a lower tick on a
    # motor-off line is not a pair of the run (real CAR_LIFTED 14/11 K2 logs)
    rows = make_log()
    clean = run(tmp_path, rows)
    t_off = clean['t_w1_us'] - 500000                   # motor off 0.5 s before the window end (inside it)
    cut, k = [], 0
    for r in rows:
        if r['timestamp_us'] >= t_off:
            r = dict(r, motor_on=0, run_active=0, tick=k)
            k += 1
        cut.append(r)
    (tmp_path / 'b').mkdir()
    res = run(tmp_path / 'b', cut)
    assert any(f.startswith('run_active_toggle') for f in res['flags'])      # the run is lost anyway
    assert not any(f.startswith('pair_not_increasing') for f in res['flags']), res['flags']
