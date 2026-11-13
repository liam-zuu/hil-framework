#!/usr/bin/env python3
"""
s05_final_table.py: tv_duty of a final-check run from the ESP32 log, and the fixed comparison table of
the final check (S05 v3.8 section 4.7, G-01; C9 and C10 of section 8.1). THESIS asked 11/11/2026
(session local_082bcf30, Liam approved S05 v3.8 there); Liam 11/11 CODE local_b905f2ba: "viec thesis
nho ban lam di". Design and the check on a real run: CLAUDE_THESIS
03_CODE/docs/2026-11-11_tv_duty_va_bang_so_S05.md.

tv_duty (S03 Y3, S05 4.7 (1)), from the cumulative firmware counters dpwm_fl..dpwm_rr of the ESP32 log
(AGV 54b7269: vel_ctrl_applied() adds |pwm_k - pwm_k-1| after motor_set(), only while the motor is on):
  t = 0     the first data line with run_active = 1 (motor on and a VALID trajectory: the command pose
            starts there, pos_ctrl.h); it must follow a line with run_active = 0 whose log_seq is the
            one before (no lost line at the edge), or come within EDGE_MAX_MS of it
  t_end     t = 0 + t_end of the run (seconds after t = 0, from the scorer of that run)
  a, b      the first and the last data line with t = 0 <= timestamp_us <= t_end, motor_on = 1
  tv_duty = sum over 4 wheels (dpwm(b) - dpwm(a)) / PWM_MAX / ((t_b - t_a) s)        [1/s]
  no number (tv_duty empty, reason given) when:
    - the end gap t_end - t_b is above EDGE_MAX_MS (lines lost at the end);
    - tick(b) - tick(a) != log_seq(b) - log_seq(a): log_seq counts every pushed line (data lines,
      motor-off lines, the three '#run_end' lines of an on -> off edge), tick counts the motor-on
      control ticks, so they differ iff the motor was off at some tick in the window;
    - tick or log_seq not strictly increasing over the window lines (counter reset by
      vel_ctrl_set_mode or a reboot);
    - a line with motor_on = 0 or run_active = 0 inside the window;
    - more than one run_active rising edge in the file (one file = one run, CG7);
    R134 package (Liam 14/11 CODE_R134 local_eb9f3b5c "duyet goi sua code fix/r134-answers"; S05 v3.13-v3.17
    4.7 (iv), RS-5, RS2-15, R3V-9; CLAUDE_THESIS 03_CODE/docs/2026-11-14_Tra_loi_THESIS_R134_R130_S06B_v1.md):
    - the tail (RS-5): the data line after b (log_seq(b) + 1) is missing, has motor_on = 0, or a time below
      t_end (the motor may have stopped between b and t_end);
    - timestamp_us not strictly increasing over the window lines, or the window lines in more than one
      segment of agv_logger.py (<log>.segments.csv, when it is there): a DUT restart (RS2-15);
    - boot (S05 4.7 (iv), R3V-9): no '#gain_running' answer of gain_send.py --query just before AND just
      after the run, or two different boot values: the ESP32 may have restarted in between (the answer is
      paired with its question by the nonce tx, gain_send.py; req_id is 0 for a query, gain_rx.c 114);
    - no '#gain' line whose first_seq is the log_seq of the first motor-on data line of the run (S05 6.3,
      events file <log>.events.txt of agv_logger.py).
    The two last checks are skipped only by the 'tv' command with --no-boot / --no-gain-line (tool tests,
    old runs), never in 'table'.
  The timestamp of a line is t_now at the top of the control loop; the counter of that line already
  holds the step applied in the same loop (vel_ctrl_applied before push_log), so both ends carry the
  same in-loop delay and it cancels in the difference.

Comparison table (4.7 (2)): from m7_tuning_log.csv rows of stage 'final' (car, platform HW) and
'final_hil' (HIL); gain set from the prefix of gain_set_id (RQ2-INIT-<ctrl> = khoi_dau, A-<ctrl> = A,
B-<ctrl> = B); condition from 'trajectory' (O1-F02, O1-F05, O1-F08, O1-K1, O1-K2); valid = run_outcome
dat / khong_dat / chua_ket_luan (hong_he_do out). Per run, from <results_dir>/<--score-name> (the JSON of
hil_o1_score.py, or of the camera scorer with the same keys) and <results_dir>/<--esp32-name>:
  e_perp_max                         e_perp_max
  e_theta_lane_max_deg               figure 8 only (K1, K2 compare the command heading only, O1 3.7)
  e_theta_cmd_max_deg                figure 8: e_theta_cmd_max_deg; K1, K2: e_theta_start_max_deg
                                     [CACH HIEU: the K-lane command heading is the start heading]
  e_s_absmax                         e_s_absmax
  e_trk_max                          e_trk_max
  tv_duty                            tv_duty_esp32() above (NOT the scorer's tv_duty_h7_1khz)
  flag                               HIL: track_err_frames > 0; camera: n_back_frames > 0
Each cell (controller x condition x platform x gain set): n_valid, n_dat / n_khong_dat / n_chua_ket_luan,
n_flag, and per quantity median and max over the valid runs that have a number, n_m, and the reason of
every valid run without a number. The quantity list is fixed here (do not change after seeing numbers).

  python tools/s05_final_table.py tv --log <esp32 csv> --t-end <s> --query-before F --query-after F
      [--no-boot] [--no-gain-line]       (F: stdout of gain_send.py --query, with its '#gain_running' line)
  python tools/s05_final_table.py table [--dataset 05_DATA/M7_dataset] --out <dir>
      [--score-name score.json] [--esp32-name esp32.csv]
      [--query-before-name gain_query_before.txt] [--query-after-name gain_query_after.txt]
  table writes <out>/S05_final_table.csv (one row per cell and quantity), S05_final_table.md (the
  three gain sets side by side) and S05_final_runs.csv (the value of every valid run: appendix).
Last line: #ket_qua,ma=0 done / 2 error.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gain_line_check as glc      # noqa: E402  ('#gain' lines of the events file)
DEFAULT_DATASET = os.path.normpath(os.path.join(HERE, '..', '..', '..', '05_DATA', 'M7_dataset'))
PWM_MAX = 1000                     # AGV motor.h
EDGE_MAX_MS = 20.0                 # [DE XUAT, khoa o C10] two log periods at 100 Hz
WHEELS = ('dpwm_fl', 'dpwm_fr', 'dpwm_rl', 'dpwm_rr')
CONDITIONS = ('O1-F02', 'O1-F05', 'O1-F08', 'O1-K1', 'O1-K2')
PLATFORMS = (('HW', 'xe', 'final'), ('HIL', 'HIL', 'final_hil'))
GAIN_SETS = ('khoi_dau', 'A', 'B')
CONTROLLERS = ('PID', 'LADRC')
QUANTITIES = ('e_perp_max', 'e_theta_lane_max_deg', 'e_theta_cmd_max_deg', 'e_s_absmax', 'e_trk_max', 'tv_duty')
VALID = ('dat', 'khong_dat', 'chua_ket_luan')


# ---------------------------------------------------------------- tv_duty
def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


SEGMENT_STATUS_BAD = ('conflict', 'ambiguous')


def read_esp32_log(path):
    """Data lines of an agv_logger CSV (header line first; '#' lines go to the events file). When
    <path>.segments.csv of agv_logger.py is there (one row per data line: row, log_seq, timestamp_us,
    segment, status), each line gets its segment in '_segment'; lines marked 'duplicate' (identical to an
    earlier line, L92) are dropped; a 'conflict' or 'ambiguous' line, or rows that do not match the CSV,
    raise ValueError (RS2-15, R134 package)."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if _int(r.get('timestamp_us')) is not None]
    seg_path = path + '.segments.csv'
    if not os.path.isfile(seg_path):
        return rows
    with open(seg_path, newline='', encoding='utf-8') as f:
        marks = list(csv.DictReader(f))
    if len(marks) != len(rows):
        raise ValueError(f'{seg_path}: {len(marks)} rows for {len(rows)} data lines')
    out, seen = [], set()
    for i, (r, m) in enumerate(zip(rows, marks), start=1):
        if (m.get('row'), m.get('log_seq'), m.get('timestamp_us')) != (str(i), r.get('log_seq'),
                                                                        r.get('timestamp_us')):
            raise ValueError(f'{seg_path} row {i} does not match data line {i} of {path}')
        status = m.get('status')
        if status in SEGMENT_STATUS_BAD:
            raise ValueError(f'{path}: {status} line {i} (see {seg_path}); not usable as is')
        key = tuple(r.values())
        if status == 'duplicate':
            if key not in seen:
                raise ValueError(f'{seg_path} row {i}: marked duplicate but no earlier equal line')
            continue
        seen.add(key)
        out.append(dict(r, _segment=m.get('segment')))
    return out


def read_boot(path):
    """boot of the '#gain_running' line of a gain_send.py --query output file, or None (no file, no
    line, more than one line, or the query did not end with '#ket_qua,ma=0')."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        lines = [x.strip() for x in f if x.strip()]
    run = [x for x in lines if x.startswith('#gain_running,')]
    if len(run) != 1 or '#ket_qua,ma=0' not in lines:
        return None
    fields = dict(p.partition('=')[::2] for p in run[0].split(',')[1:])
    b = fields.get('boot', '')
    return b if b and all(c in '0123456789abcdefABCDEF' for c in b) else None


def gain_seqs(log_path):
    """first_seq values (not 0) of the distinct '#gain' lines of <log>.events.txt, or None without the file."""
    ev = log_path + '.events.txt'
    if not os.path.isfile(ev):
        return None
    try:
        return {int(g['first_seq']) for g in glc.gain_lines(ev) if g['first_seq'] != '0'}
    except glc.InputError as e:
        raise ValueError(str(e))


def tv_duty_esp32(rows, t_end_s, edge_max_ms=EDGE_MAX_MS, t0_us=None, boots=None, gain_first_seqs=None,
                  check_boot=True, check_gain_line=True):
    """tv_duty of one run from its ESP32 log lines (dicts of strings). Returns a dict with tv_duty
    (None when no number) and every check; 'reason' says why there is no number.
    boots: (boot before, boot after) of the gain_send --query answers around the run (None = missing);
    gain_first_seqs: first_seq values of the '#gain' lines of the run's events file (None = no file).
    check_boot / check_gain_line False skip those two checks (tool tests, old runs; never for the table)."""
    out = dict(tv_duty=None, reason='', edge_max_ms=edge_max_ms)
    need = ('timestamp_us', 'tick', 'log_seq', 'motor_on', 'run_active') + WHEELS
    if not rows or any(k not in rows[0] for k in need):
        out['reason'] = 'log thieu cot ' + ','.join(k for k in need if not rows or k not in rows[0])
        return out
    ts = [_int(r['timestamp_us']) for r in rows]
    ra = [r['run_active'] == '1' for r in rows]
    rises = [i for i in range(len(rows)) if ra[i] and (i == 0 or not ra[i - 1])]
    out['n_run_active_rises'] = len(rises)
    if t0_us is None:
        if len(rises) != 1:
            out['reason'] = f'{len(rises)} lan run_active len trong file (can dung 1)'
            return out
        i0 = rises[0]
        if i0 == 0:
            out['reason'] = 'dong dau file da run_active = 1: khong thay canh len'
            return out
        prev, first = rows[i0 - 1], rows[i0]
        contiguous = _int(first['log_seq']) == _int(prev['log_seq']) + 1
        gap_ms = (ts[i0] - ts[i0 - 1]) / 1000
        out.update(t0_edge_contiguous=contiguous, t0_edge_gap_ms=gap_ms)
        if not contiguous and gap_ms > edge_max_ms:
            out['reason'] = f'mat dong o canh t = 0 ({gap_ms:.1f} ms > {edge_max_ms:g} ms)'
            return out
        t0_us = ts[i0]
        out['t0_src'] = 'run_active_edge'
    else:
        out['t0_src'] = 'given'
    t_end_us = t0_us + t_end_s * 1e6
    out.update(t0_us=t0_us, t_end_s=t_end_s)
    win = [i for i in range(len(rows)) if t0_us <= ts[i] <= t_end_us]
    if len(win) < 2:
        out['reason'] = 'cua so co duoi 2 dong log'
        return out
    off = [i for i in win if rows[i]['motor_on'] != '1' or rows[i]['run_active'] != '1']
    if off:
        out['reason'] = f'{len(off)} dong motor_on = 0 hay run_active = 0 trong cua so'
        return out
    a, b = win[0], win[-1]
    segs = {rows[i].get('_segment') for i in win}
    if len(segs) > 1:
        out['reason'] = f'cua so qua {len(segs)} doan logger (ESP32 khoi dong lai?)'
        return out
    if any(ts[j] <= ts[i] for i, j in zip(win, win[1:])):
        out['reason'] = 'timestamp_us khong tang ngat trong cua so (khoi dong lai hay dong lan)'
        return out
    start_gap = (ts[a] - t0_us) / 1000
    end_gap = (t_end_us - ts[b]) / 1000
    out.update(start_gap_ms=start_gap, end_gap_ms=end_gap, n_lines=len(win))
    if start_gap > edge_max_ms or end_gap > edge_max_ms:
        out['reason'] = f'khoang bien {max(start_gap, end_gap):.1f} ms > {edge_max_ms:g} ms'
        return out
    tk = [_int(rows[i]['tick']) for i in win]
    sq = [_int(rows[i]['log_seq']) for i in win]
    if any(y <= x for x, y in zip(tk, tk[1:])) or any(y <= x for x, y in zip(sq, sq[1:])):
        out['reason'] = 'tick hay log_seq khong tang deu (dat lai bo dem hay khoi dong lai)'
        return out
    seq_b = _int(rows[b]['log_seq'])
    nxt = [x for x in rows if _int(x.get('log_seq')) == seq_b + 1]
    if len(nxt) != 1:
        out['reason'] = (f'duoi cua so: {"khong co" if not nxt else "hon mot"} dong log_seq {seq_b + 1} '
                         'sau mau cuoi (RS-5)')
        return out
    n_ts = _int(nxt[0]['timestamp_us'])
    out.update(tail_seq=seq_b + 1, tail_ms_after_t_end=(n_ts - t_end_us) / 1000)
    if nxt[0].get('motor_on') != '1' or n_ts < t_end_us:
        out['reason'] = ('duoi cua so: dong ke sau mau cuoi co motor_on = 0' if nxt[0].get('motor_on') != '1'
                         else 'duoi cua so: dong ke sau mau cuoi truoc t_end') + ' (RS-5)'
        return out
    d_tick, d_seq = tk[-1] - tk[0], sq[-1] - sq[0]
    out.update(d_tick=d_tick, d_seq=d_seq, lines_lost=d_seq - (len(win) - 1))
    if d_tick != d_seq:
        out['reason'] = f'tick tang {d_tick}, log_seq tang {d_seq}: motor tat o mot tick trong cua so'
        return out
    if check_gain_line:
        on = [x.get('motor_on') == '1' for x in rows]
        firsts = [_int(rows[i]['log_seq']) for i in range(len(rows)) if on[i] and (i == 0 or not on[i - 1])]
        out['run_first_seq'] = firsts[0] if len(firsts) == 1 else None
        if len(firsts) != 1 or on[0]:
            out['reason'] = 'khong xac dinh duoc dong du lieu dau luot (motor_on) de khop #gain'
            return out
        if gain_first_seqs is None or firsts[0] not in gain_first_seqs:
            out['reason'] = ('thieu file events (#gain)' if gain_first_seqs is None else
                             f'khong co dong #gain first_seq = {firsts[0]} (dong dau luot; S05 6.3)')
            return out
    if check_boot:
        before, after = boots if boots else (None, None)
        out.update(boot_before=before, boot_after=after)
        if before is None or after is None:
            out['reason'] = (f'thieu hoi boot {"truoc" if before is None else "sau"} luot '
                             '(gain_send --query; S05 4.7 (iv))')
            return out
        if before.lower() != after.lower():
            out['reason'] = f'boot truoc {before} khac boot sau {after}: ESP32 khoi dong lai giua luot'
            return out
    num = sum(_int(rows[b][w]) - _int(rows[a][w]) for w in WHEELS) / PWM_MAX
    dt = (ts[b] - ts[a]) / 1e6
    out.update(tv_duty=num / dt, sum_abs_dduty=num, t_a_s=(ts[a] - t0_us) / 1e6, t_b_s=(ts[b] - t0_us) / 1e6)
    return out


# ---------------------------------------------------------------- table
def gain_set_of(gain_set_id, controller):
    g = (gain_set_id or '').split('/')[0]
    if g == f'RQ2-INIT-{controller}':
        return 'khoi_dau'
    if g == f'A-{controller}':
        return 'A'
    if g == f'B-{controller}':
        return 'B'
    return None


def run_values(score, task, tv):
    fig8 = task.startswith('O1-F')
    v = dict(e_perp_max=score.get('e_perp_max'),
             e_theta_lane_max_deg=score.get('e_theta_lane_max_deg') if fig8 else None,
             e_theta_cmd_max_deg=score.get('e_theta_cmd_max_deg') if fig8 else score.get('e_theta_start_max_deg'),
             e_s_absmax=score.get('e_s_absmax'), e_trk_max=score.get('e_trk_max'), tv_duty=tv.get('tv_duty'))
    miss = {}
    for q in QUANTITIES:
        if q == 'e_theta_lane_max_deg' and not fig8:
            miss[q] = 'khong ap dung (K1, K2: chi so huong lenh, O1 3.7)'
        elif v[q] is None:
            miss[q] = (tv.get('reason') or 'thieu') if q == 'tv_duty' else 'diem cham khong co so'
    return v, miss


def load_runs(dataset, score_name, esp32_name, query_before_name='gain_query_before.txt',
              query_after_name='gain_query_after.txt'):
    path = os.path.join(dataset, 'm7_tuning_log.csv')
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    runs, problems = [], []
    for r in rows:
        plat = next((p for p in PLATFORMS if r.get('platform') == p[0] and r.get('stage') == p[2]), None)
        if plat is None:
            continue
        ctrl, task = r.get('controller', ''), r.get('trajectory', '')
        gs = gain_set_of(r.get('gain_set_id'), ctrl)
        if ctrl not in CONTROLLERS or task not in CONDITIONS or gs is None:
            problems.append(f"run {r.get('run_uid')}: controller {ctrl!r}, trajectory {task!r}, gain_set_id "
                            f"{r.get('gain_set_id')!r} khong vao bang")
            continue
        run = dict(run_uid=r.get('run_uid'), platform=plat[1], controller=ctrl, condition=task, gain_set=gs,
                   outcome=r.get('run_outcome'), results_dir=r.get('results_dir', ''))
        if run['outcome'] in VALID:
            d = run['results_dir']
            score, tv = {}, dict(reason='')
            sp = os.path.join(d, score_name)
            if os.path.isfile(sp):
                with open(sp, encoding='utf-8') as f:
                    score = json.load(f)
            else:
                tv['reason'] = f'thieu {score_name}'
            ep = os.path.join(d, esp32_name)
            if score.get('t_end') is not None and os.path.isfile(ep):
                tv = tv_duty_esp32(read_esp32_log(ep), float(score['t_end']),
                                   boots=(read_boot(os.path.join(d, query_before_name)),
                                          read_boot(os.path.join(d, query_after_name))),
                                   gain_first_seqs=gain_seqs(ep))
            elif not tv['reason']:
                tv['reason'] = f'thieu {esp32_name}' if not os.path.isfile(ep) else 'diem cham khong co t_end'
            run['values'], run['missing'] = run_values(score, task, tv)
            run['flag'] = (int(score.get('track_err_frames') or 0) > 0 if plat[1] == 'HIL'
                           else int(score.get('n_back_frames') or 0) > 0)
            run['tv_check'] = tv
        runs.append(run)
    return runs, problems


def build_table(runs):
    cells = defaultdict(list)
    for r in runs:
        cells[(r['controller'], r['condition'], r['platform'], r['gain_set'])].append(r)
    out = []
    for ctrl in CONTROLLERS:
        for cond in CONDITIONS:
            for plat in ('xe', 'HIL'):
                for gs in GAIN_SETS:
                    rs = cells.get((ctrl, cond, plat, gs), [])
                    val = [r for r in rs if r['outcome'] in VALID]
                    base = dict(controller=ctrl, condition=cond, platform=plat, gain_set=gs, n_valid=len(val),
                                n_dat=sum(r['outcome'] == 'dat' for r in val),
                                n_khong_dat=sum(r['outcome'] == 'khong_dat' for r in val),
                                n_chua_ket_luan=sum(r['outcome'] == 'chua_ket_luan' for r in val),
                                n_hong_he_do=sum(r['outcome'] == 'hong_he_do' for r in rs),
                                n_flag=sum(bool(r.get('flag')) for r in val))
                    for q in QUANTITIES:
                        xs = [r['values'][q] for r in val if r['values'].get(q) is not None]
                        why = '; '.join(f"{r['run_uid']}: {r['missing'][q]}" for r in val if q in r['missing'])
                        out.append(dict(base, quantity=q, n_m=len(xs),
                                        median=statistics.median(xs) if xs else None,
                                        max=max(xs) if xs else None, missing=why))
    return out


def markdown(table):
    by = {(t['controller'], t['condition'], t['platform'], t['gain_set'], t['quantity']): t for t in table}
    L = ['| controller | condition | platform | quantity | ' + ' | '.join(GAIN_SETS) + ' |',
         '|---|---|---|---|' + '---|' * len(GAIN_SETS)]
    for ctrl in CONTROLLERS:
        for cond in CONDITIONS:
            for plat in ('xe', 'HIL'):
                head = ' | '.join(
                    f"n {by[(ctrl, cond, plat, g, QUANTITIES[0])]['n_valid']}: "
                    f"{by[(ctrl, cond, plat, g, QUANTITIES[0])]['n_dat']}/"
                    f"{by[(ctrl, cond, plat, g, QUANTITIES[0])]['n_khong_dat']}/"
                    f"{by[(ctrl, cond, plat, g, QUANTITIES[0])]['n_chua_ket_luan']}, co {by[(ctrl, cond, plat, g, QUANTITIES[0])]['n_flag']}"
                    for g in GAIN_SETS)
                L.append(f'| {ctrl} | {cond} | {plat} | n_valid: dat/khong/chua, co | {head} |')
                for q in QUANTITIES:
                    cells = []
                    for g in GAIN_SETS:
                        t = by[(ctrl, cond, plat, g, q)]
                        cells.append('–' if t['n_m'] == 0 else f"{t['median']:.4g} / {t['max']:.4g} (n_m {t['n_m']})")
                    L.append(f'| | | | {q} (trung vi / lon nhat) | ' + ' | '.join(cells) + ' |')
    return '\n'.join(L) + '\n'


def write_table(runs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    table = build_table(runs)
    cols = ['controller', 'condition', 'platform', 'gain_set', 'n_valid', 'n_dat', 'n_khong_dat', 'n_chua_ket_luan',
            'n_hong_he_do', 'n_flag', 'quantity', 'n_m', 'median', 'max', 'missing']
    with open(os.path.join(out_dir, 'S05_final_table.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: ('' if t[k] is None else t[k]) for k in cols} for t in table)
    with open(os.path.join(out_dir, 'S05_final_table.md'), 'w', encoding='utf-8') as f:
        f.write(markdown(table))
    rcols = ['run_uid', 'platform', 'controller', 'condition', 'gain_set', 'outcome', 'flag'] + list(QUANTITIES) + \
            ['tv_reason', 'tv_d_tick', 'tv_d_seq', 'tv_end_gap_ms', 'results_dir']
    with open(os.path.join(out_dir, 'S05_final_runs.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rcols)
        w.writeheader()
        for r in runs:
            v = r.get('values', {})
            tv = r.get('tv_check', {})
            w.writerow(dict(run_uid=r['run_uid'], platform=r['platform'], controller=r['controller'],
                            condition=r['condition'], gain_set=r['gain_set'], outcome=r['outcome'],
                            flag=r.get('flag', ''), **{q: ('' if v.get(q) is None else v[q]) for q in QUANTITIES},
                            tv_reason=tv.get('reason', ''), tv_d_tick=tv.get('d_tick', ''), tv_d_seq=tv.get('d_seq', ''),
                            tv_end_gap_ms=tv.get('end_gap_ms', ''), results_dir=r['results_dir']))
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('tv')
    a1.add_argument('--log', required=True)
    a1.add_argument('--t-end', type=float, required=True, help='t_end of the run, s after t = 0')
    a1.add_argument('--edge-max-ms', type=float, default=EDGE_MAX_MS)
    a1.add_argument('--query-before', default=None, help='gain_send.py --query output just before the run')
    a1.add_argument('--query-after', default=None, help='gain_send.py --query output just after the run')
    a1.add_argument('--no-boot', action='store_true', help='skip the boot check (tool test, old run)')
    a1.add_argument('--no-gain-line', action='store_true', help="skip the '#gain' first_seq check")
    a2 = sub.add_parser('table')
    a2.add_argument('--dataset', default=DEFAULT_DATASET)
    a2.add_argument('--out', required=True)
    a2.add_argument('--score-name', default='score.json')
    a2.add_argument('--esp32-name', default='esp32.csv')
    a2.add_argument('--query-before-name', default='gain_query_before.txt')
    a2.add_argument('--query-after-name', default='gain_query_after.txt')
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'tv':
            r = tv_duty_esp32(read_esp32_log(a.log), a.t_end, a.edge_max_ms,
                              boots=(read_boot(a.query_before), read_boot(a.query_after)),
                              gain_first_seqs=gain_seqs(a.log), check_boot=not a.no_boot,
                              check_gain_line=not a.no_gain_line)
            print(json.dumps(r, ensure_ascii=False, indent=1))
        else:
            runs, problems = load_runs(a.dataset, a.score_name, a.esp32_name, a.query_before_name,
                                       a.query_after_name)
            for p in problems:
                print('CANH BAO:', p)
            write_table(runs, a.out)
            print(f'{len(runs)} luot kiem cuoi -> {a.out}')
    except (OSError, ValueError, KeyError) as e:
        print('loi:', e)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
