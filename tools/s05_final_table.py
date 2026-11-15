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
    - the tail (RS-5, R134C-4): the data line right after b in the normalized lines is missing, in
      another logger segment, has log_seq != log_seq(b) + 1, motor_on = 0, tick != tick(b) + 1, or a time
      below t_end (the motor may have stopped between b and t_end);
    - timestamp_us not strictly increasing over the window lines, or the window lines in more than one
      segment of agv_logger.py (<log>.segments.csv, when it is there): a DUT restart (RS2-15);
    - boot (S05 4.7 (iv), R3V-9, R134C-1): no '#gain_running' answer of gain_send.py --query just before
      AND just after the run (file ending with '#ket_qua,ma=0', one answer line), two different boot or
      MAC values, or a boot other than the one of the run's '#us,ev=start' line (first_seq = first
      motor-on data line; AGV 16702ff main.c 354): an old answer, another board, or a restart (the answer
      is paired with its question by the nonce tx, gain_send.py; req_id is 0 for a query, gain_rx.c 114);
    - no '#gain' line of the run: first_seq = the log_seq of the first motor-on data line, next to the
      run's '#us,ev=start', all copies agreeing except line_drop (S05 6.3, R134C-2, -5; events file
      <log>.events.txt of agv_logger.py; gain_line_check.run_start).
    - B8-1 B81-14 (S05 v3.20 4.7 (iv), C9 (d); CODE 14/11 to CODE_R134): the '#gain_ack' answer of the
      gain setting (saved gain_send.py output, --gain-ack / --gain-ack-name) missing, not ok, or with a boot
      or MAC other than the two query answers, or an id other than the run's '#gain' line
      (gain_line_check.ack_problem). Every final run has an ack, the initial gain too (R134C5-3; Liam 14/11
      CODE_R134 "Bat buoc ack moi luot (Recommended)": gain set from a file with --gain-file).
    Table (R134C4-2, R134C5-1, -2, -3): rows outside the campaigns set apart; the campaign rows go through the row
    checks of rq3_log summary; every judged final run must show its evidence at run level (run_evidence_problem:
    two answers that are two questions, of the row's boots, and a readable ack of the question's boot, MAC and
    running id), else the table stops (exit 2). tv_duty may still have no number for its own reasons.
    R134C6-1 (Codex round 6; Liam 15/11 CODE_R134 "Tham so cua lenh bang (Recommended)"): and the run's '#gain'
    line is the gain file of its set, bit for bit, the file id = the running id of the question before: A, B the
    file of S05_freeze.csv (sha256 checked), the initial gain the file of --init-gain <ctrl>=F; no file for the
    set stops the table too. R134C7-1, -2, -3 (round 7; Liam 15/11 "Kiem gain moi doan (Recommended)"): the '#gain'
    line of every motor-on stretch of the log, the running id of the question after = the file id; a frozen file
    that cannot be proved refuses only the judged runs of its set; a relative frozen name takes the first
    candidate (dataset folder, then working folder) whose sha256 matches. The files and their sha256 go to S05_final_gain_files.csv and S05_final_runs.csv.
    The data lines are read by gain_line_check.read_esp32_log (R134C-3, S05 6.3): lines marked
    'duplicate' in <log>.segments.csv dropped (counted, tv_rows_dropped), 'conflict' / 'ambiguous'
    refused; without the sidecar two lines with one log_seq are refused.
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
t_end on the ESP32 clock (S05 4.7 (i), RS-4; Liam 14/11 CODE_RS4 local_c3031475, Q4 "Giữ công thức, ghi sai
lệch (Recommended)"; design 03_CODE/docs/2026-11-14_Thiet_ke_RS4_so_phien_MOSI_v1.md section 10): the window
end stays t = 0 + t_end on the ESP32 log; each HIL run reports the estimated error of that end,
  t_end_err_ms = t_end x (|esp_pi_ppm| + ESP_PPM_NOISE + |pi_period_ppm|) x 1e-3
esp_pi_ppm: the ESP32 / Pi rate estimate of the run (score edge_map, from the logger WiFi packets);
pi_period_ppm: the Pi step period against the nominal 1 ms (score key if present, else PI_PERIOD_PPM_BOUND);
no esp_pi_ppm (or not finite): the whole sum in brackets is ESP_PI_PRIOR_PPM = 2000 ppm, the ESP32 / Pi bound
built from the two assumed ranges of the mapper (ESP32 / H7 and H7 / Pi, 1000 ppm each), i.e. 2 x t_end ms
(24 ms at 12 s), an assumed bound, not a measurement (S05 v3.20 4.7 (i), B81-13; Liam 14/11 THESIS_B8 "duyệt
cả"); a pi_period_ppm that is not finite counts as missing. Report only; the line choice is unchanged. Car runs:
the camera clock goes to the ESP32 through C7b, not here (column empty, 'xe: C7b').

Each cell (controller x condition x platform x gain set): n_valid, n_dat / n_khong_dat / n_chua_ket_luan,
n_flag, n_cat_t1 (runs khong_dat by the exception before t1, O1 3.5 R2V-4; R134C4-6), and per quantity median and max over the valid runs that have a number, n_m, and the reason of
every valid run without a number. The quantity list is fixed here (do not change after seeing numbers).

  python tools/s05_final_table.py tv --log <esp32 csv> --t-end <s> --query-before F --query-after F
      [--gain-ack A] [--no-boot] [--no-gain-line]   (F: stdout of gain_send.py --query, with its '#gain_running'
      line; A: stdout of gain_send.py setting the set, with its '# ack' line)
  python tools/s05_final_table.py table [--dataset 05_DATA/M7_dataset] --out <dir>
      [--score-name score.json] [--esp32-name esp32.csv]
      [--query-before-name gain_query_before.txt] [--query-after-name gain_query_after.txt]
      [--gain-ack-name gain_ack.txt] [--init-gain PID=F] [--init-gain LADRC=F]
  table writes <out>/S05_final_table.csv (one row per cell and quantity), S05_final_table.md (the
  three gain sets side by side), S05_final_runs.csv (the value of every valid run: appendix) and
  S05_final_gain_files.csv (gain file and sha256 of every set; R134C6-1).
Last line: #ket_qua,ma=0 done / 2 error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
# t_end error (S05 4.7 (i), Liam 14/11 CODE_RS4 Q4)
ESP_PPM_NOISE = 50.0               # spread of the per-run esp_pi_ppm estimate (13/11: 12 runs -52..+31 ppm)
PI_PERIOD_PPM_BOUND = 100.0        # [DE XUAT] when the score has no pi_period_ppm (run14: -12 / -12 / -13 ppm)
ESP_PI_PRIOR_PPM = 2000.0          # S05 v3.20 4.7 (i): ESP32/H7 1000 (RHO_MAX_PPM, L100) + H7/Pi 1000, no esp_pi_ppm


def _finite(v):
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError, OverflowError):               # OverflowError: an int too big for a float
        return False


def t_end_err_ms(t_end_s, esp_pi_ppm=None, pi_period_ppm=None):
    """Estimated error (ms) of t = 0 + t_end on the ESP32 log of a HIL run (module docstring), and its source.
    t_end not finite or < 0, or a result that is not finite: no number (Codex B8C-R1-6)."""
    if not _finite(t_end_s) or float(t_end_s) < 0:
        return None, 'khong co t_end'
    t = float(t_end_s)
    if not _finite(esp_pi_ppm):
        v, src = t * (ESP_PI_PRIOR_PPM * 1e-3), f'khong co esp_pi_ppm: can {ESP_PI_PRIOR_PPM:g} ppm (gia dinh)'
    else:
        if not _finite(pi_period_ppm):
            pi_period_ppm = None
        pi = PI_PERIOD_PPM_BOUND if pi_period_ppm is None else abs(float(pi_period_ppm))
        src = 'esp_pi_ppm + nhieu + ' + ('can Pi' if pi_period_ppm is None else 'pi_period_ppm')
        v = t * ((abs(float(esp_pi_ppm)) + ESP_PPM_NOISE + pi) * 1e-3)
    return (v, src) if math.isfinite(v) else (None, 'khong co so huu han')


# ---------------------------------------------------------------- tv_duty
def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def read_esp32_log_info(path):
    """(data lines, lines dropped as marked duplicates): the one reader shared with gain_line_check.py
    (R134C-3; S05 6.3, L92): <path>.segments.csv when there, else two lines with one log_seq refused."""
    return glc.read_esp32_log_info(path)


def read_esp32_log(path):
    return glc.read_esp32_log(path)


def read_query(path):
    """'#gain_running' fields of a gain_send.py --query output file, or None (gain_line_check.read_query:
    exactly one answer line and the last line '#ket_qua,ma=0', R134C-1)."""
    return glc.read_query(path)


def read_boot(path):
    """boot of read_query(path), or None."""
    q = glc.read_query(path)
    return q['boot'] if q else None


def read_events(log_path):
    """'#' lines of <log>.events.txt (gain_line_check.read_events), or None without the file."""
    return glc.read_events(log_path + '.events.txt')


def tv_duty_esp32(rows, t_end_s, edge_max_ms=EDGE_MAX_MS, t0_us=None, queries=None, events=None,
                  check_boot=True, check_gain_line=True, ack=None):
    """tv_duty of one run from its ESP32 log lines (dicts of strings, read_esp32_log). Returns a dict with
    tv_duty (None when no number) and every check; 'reason' says why there is no number.
    queries: (before, after) read_query dicts of the gain_send --query answers around the run (None =
    missing); events: read_events of the run (None = no events file); ack: gain_line_check.read_ack of the
    gain setting (None = missing; B81-14, checked with the boot).
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
    # RS-5 tail (R134C-4): the line right after b in the normalized lines, same logger segment,
    # log_seq + 1 and tick + 1 (no line lost, no counter reset), motor on, time >= t_end
    seq_b, tick_b = _int(rows[b]['log_seq']), _int(rows[b]['tick'])
    if b + 1 >= len(rows):
        out['reason'] = f'duoi cua so: khong co dong log_seq {seq_b + 1} sau mau cuoi (RS-5)'
        return out
    nx = rows[b + 1]
    n_ts = _int(nx['timestamp_us'])
    out.update(tail_seq=_int(nx.get('log_seq')), tail_ms_after_t_end=(n_ts - t_end_us) / 1000)
    if nx.get('_segment') != rows[b].get('_segment'):
        why = 'dong ke sau mau cuoi o doan logger khac'
    elif _int(nx.get('log_seq')) != seq_b + 1:
        why = f'dong ke sau mau cuoi co log_seq {nx.get("log_seq")}, khong co dong log_seq {seq_b + 1}'
    elif nx.get('motor_on') != '1':
        why = 'dong ke sau mau cuoi co motor_on = 0'
    elif _int(nx.get('tick')) != tick_b + 1:
        why = f'dong ke sau mau cuoi co tick {nx.get("tick")}, can {tick_b + 1}'
    elif n_ts < t_end_us:
        why = 'dong ke sau mau cuoi truoc t_end'
    else:
        why = ''
    if why:
        out['reason'] = f'duoi cua so: {why} (RS-5)'
        return out
    d_tick, d_seq = tk[-1] - tk[0], sq[-1] - sq[0]
    out.update(d_tick=d_tick, d_seq=d_seq, lines_lost=d_seq - (len(win) - 1))
    if d_tick != d_seq:
        out['reason'] = f'tick tang {d_tick}, log_seq tang {d_seq}: motor tat o mot tick trong cua so'
        return out
    if check_gain_line or check_boot:
        # the run's start (R134C-1, -2): '#us,ev=start' with first_seq = first motor-on data line gives
        # the boot of the run; the '#gain' lines next to it give the set (gain_line_check.run_start)
        on = [x.get('motor_on') == '1' for x in rows]
        firsts = [_int(rows[i]['log_seq']) for i in range(len(rows)) if on[i] and (i == 0 or not on[i - 1])]
        out['run_first_seq'] = firsts[0] if len(firsts) == 1 else None
        if len(firsts) != 1 or on[0]:
            out['reason'] = 'khong xac dinh duoc dong du lieu dau luot (motor_on) de khop #gain, #us'
            return out
        if events is None:
            out['reason'] = 'thieu file events (#gain, #us,ev=start)'
            return out
        try:
            st = glc.run_start(events, firsts[0])
        except glc.InputError as e:
            out['reason'] = f'file events hong: {e}'
            return out
        out.update(run_boot=st['boot'], gain_line_drop_max=st['line_drop_max'])
        if check_gain_line and st['gain'] is None:
            out['reason'] = f'dong #gain cua luot: {"; ".join(st["why"])} (S05 6.3)'
            return out
    if check_boot:
        before, after = queries if queries else (None, None)
        out.update(boot_before=before and before['boot'], boot_after=after and after['boot'],
                   mac_before=before and before.get('mac'), mac_after=after and after.get('mac'))
        bad = glc.query_problem(before, after, st['boot']) or glc.ack_problem(ack, before, st.get('gain'))
        out.update(boot_ack=ack and ack.get('boot'))                                         # B81-14
        if bad:
            out['reason'] = bad
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


def _a_number(v):
    """B8-1 B81-16 (S05 v3.20 C9 (e)): a finite int or float (not bool); NaN, inf, text are no number."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return v == v and v not in (float('inf'), float('-inf'))


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
        elif not _a_number(v[q]):                                       # B81-16: NaN / inf is no number
            miss[q] = f'gia tri khong huu han hay khong phai so ({v[q]!r})'
            v[q] = None
    return v, miss


def gain_files(dataset, init_gain=None):
    """R134C6-1 (Codex round 6; Liam 15/11 CODE_R134 hop chon "Tham so cua lenh bang (Recommended)"): the gain file
    of every gain set the table may meet, {gain_set_id or 'RQ2-INIT-<ctrl>': dict(file, sha256, source, want)}.
    A, B: the rows of <dataset>/S05_freeze.csv (rq3_log freeze); the file as written there (a relative name is
    looked for in the dataset folder first, then in the working folder) whose sha256 equals gain_sha256: the first
    candidate that matches is taken (R134C7-3). A frozen set whose file is not found, matches no sha256 or does not
    load gives an entry dict(error=..) instead: only a judged run of that set is refused for it (R134C7-2; a set
    with no judged run in the table does not stop the table).
    Initial gain (RQ2-INIT-<ctrl>): init_gain {ctrl: file} of --init-gain, its sha256 written to the output; a bad
    --init-gain stops the table at once (the argument the operator gave).
    want = gain_line_check.load_gain_file of the file; its ctrl must be the controller of the set."""
    import rq3_log as rl
    out = {}
    for r in rl.read_rows(os.path.join(dataset, rl.FREEZE)):
        gs, p = r.get('gain_set_id', ''), r.get('gain_file', '')
        if not gs:
            continue
        cand = [p] if os.path.isabs(p) else [os.path.join(dataset, p), p]
        # R134C8-2 (Codex round 8): one candidate at a time, its read error kept for it alone; the first that
        # matches gain_sha256 is taken and the rest is not read
        f = sha = None
        tried, n_there = [], 0
        for c in cand:
            if not os.path.isfile(c):
                tried.append(f'{c}: khong co')
                continue
            n_there += 1
            try:
                h = rl.sha256_file(c)
            except OSError as e:
                tried.append(f'{c}: khong doc duoc ({e})')
                continue
            if h == r.get('gain_sha256', ''):
                f, sha = c, h
                break
            tried.append(f'{c}: sha256 {h}')
        try:
            if f is None and not n_there:
                raise ValueError(f'S05_freeze.csv: gain_file {p!r} of {gs} not found (tried {cand})')
            if f is None:
                raise ValueError(f'S05_freeze.csv: gain_file {p!r} of {gs}: moi ung vien doc duoc deu khac gain_sha256 '
                                 f'{r.get("gain_sha256")!r} ({"; ".join(tried)})')
            out[gs] = dict(file=f, sha256=sha, source='S05_freeze.csv',
                           want=_want(f, gs.split('/')[0].split('-')[-1]))
        except (ValueError, OSError) as e:
            out[gs] = dict(error=str(e))
    for ctrl, f in (init_gain or {}).items():
        if ctrl not in CONTROLLERS:
            raise ValueError(f'--init-gain {ctrl}=...: controller one of {CONTROLLERS}')
        if not os.path.isfile(f):
            raise ValueError(f'--init-gain {ctrl}={f}: no such file')
        out[f'RQ2-INIT-{ctrl}'] = dict(file=f, sha256=rl.sha256_file(f), source='--init-gain', want=_want(f, ctrl))
    return out


FW_CTRL = {'PID': 'PID', 'LADRC': 'ADRC'}     # controller of the log -> ctrl of the gain file / '#gain' line


def _want(path, ctrl):
    try:
        w = glc.load_gain_file(path)
    except glc.InputError as e:
        raise ValueError(f'gain file: {e}')
    if w['ctrl'] != FW_CTRL.get(ctrl):
        raise ValueError(f'gain file {path}: ctrl {w["ctrl"]}, the set is of {ctrl} (ctrl {FW_CTRL.get(ctrl)})')
    return w


def gain_file_of(gain_set_id, files):
    """The entry of gain_files() for one row: the frozen set by its id (A, B), the initial gain by its prefix."""
    g = (gain_set_id or '').split('/')[0]
    return files.get(g) if g.startswith('RQ2-INIT-') else files.get(gain_set_id or '')


def load_runs(dataset, score_name, esp32_name, query_before_name='gain_query_before.txt',
              query_after_name='gain_query_after.txt', gain_ack_name='gain_ack.txt', init_gain=None):
    path = os.path.join(dataset, 'm7_tuning_log.csv')
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    # R134C4-2 (Codex round 4): the rows go through the same row checks as rq3_log summary (gate columns
    # recomputed, the camera gate of a judged car run, both boots of a judged car / final_hil run, the HIL
    # mapping rules); any error stops the table. A final row outside the campaigns is not tabled (reported).
    # R134C5-2: rows outside the campaigns (historic) are set apart first, so they cannot stop the table.
    import rq3_log as rl
    live = rl.live_rows(rows)
    rows = [r for r in live if r.get('campaign') in rl.CAMPAIGNS]
    problems = [f"run {r.get('run_uid')}: campaign {r.get('campaign')!r} ngoai cac chien dich, khong vao bang "
                '(R134C4-2)' for r in live if r.get('campaign') not in rl.CAMPAIGNS
                and any(r.get('platform') == p[0] and r.get('stage') == p[2] for p in PLATFORMS)]
    try:
        rl.check_gate_rows(rows)
        rl.check_map_rows(rows)
    except rl.LogError as e:
        raise ValueError(f'm7_tuning_log.csv: {e}')
    files = gain_files(dataset, init_gain)                                              # R134C6-1
    runs, bad = [], []
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
        gf = gain_file_of(r.get('gain_set_id'), files)
        if r.get('run_outcome') in VALID:           # R134C5-1, -3, R134C6-1: the run's evidence, whatever tv_duty
            why = run_evidence_problem(r, query_before_name, query_after_name, gain_ack_name, esp32_name, gf)
            if why:
                bad.append(f"run {r.get('run_uid')}: {why}")
        run = dict(run_uid=r.get('run_uid'), platform=plat[1], controller=ctrl, condition=task, gain_set=gs,
                   outcome=r.get('run_outcome'), results_dir=r.get('results_dir', ''),
                   cat_t1=r.get('g1_cat_t1') == 'true',      # R134C4-6: O1 3.5 R2V-4, counted per cell
                   gain_set_id=r.get('gain_set_id', ''), gain_file=(gf or {}).get('file', ''),
                   gain_sha256=(gf or {}).get('sha256', ''), gain_source=(gf or {}).get('source', ''))
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
                rows_ep, n_drop = read_esp32_log_info(ep)
                tv = tv_duty_esp32(rows_ep, float(score['t_end']),
                                   queries=(read_query(os.path.join(d, query_before_name)),
                                            read_query(os.path.join(d, query_after_name))),
                                   events=read_events(ep), ack=glc.read_ack(os.path.join(d, gain_ack_name)))
                tv['rows_dropped'] = n_drop
            elif not tv['reason']:
                tv['reason'] = f'thieu {esp32_name}' if not os.path.isfile(ep) else 'diem cham khong co t_end'
            run['values'], run['missing'] = run_values(score, task, tv)
            run['flag'] = (int(score.get('track_err_frames') or 0) > 0 if plat[1] == 'HIL'
                           else int(score.get('n_back_frames') or 0) > 0)
            run['tv_check'] = tv
            if plat[1] == 'HIL':
                em = score.get('edge_map') or {}
                run['t_end_err_ms'], run['t_end_err_src'] = t_end_err_ms(
                    score.get('t_end'), em.get('esp_pi_ppm'), score.get('pi_period_ppm'))
            else:
                run['t_end_err_ms'], run['t_end_err_src'] = None, 'xe: C7b'
        runs.append(run)
    if bad:
        raise ValueError('luot da phan ma thieu bang chung bat buoc (luot hong do thao tac, sua log hay chay lai): '
                         + '; '.join(bad))
    return runs, problems


def run_gain_problem(esp32_path, want):
    """R134C6-1, R134C7-1 (Codex rounds 6, 7; Liam 15/11 CODE_R134 hop chon "Kiem gain moi doan (Recommended)"):
    None, or why the run does not show the set of the gain file `want` in EVERY motor-on stretch of its ESP32 log:
    each stretch (a data line with motor_on = 1 after one with 0) must have its own '#gain' line, next to its
    '#us,ev=start' (first_seq = the first data line of the stretch; gain_line_check.run_start), equal to the file
    bit for bit. A run whose motor stopped and started again with the same set still counts (tv_duty has no number
    then); a stretch with another set, or without its '#gain' line, refuses the run.
    R134C8-1 (Codex round 8; Liam 15/11 CODE_R134 "Sua 2 muc, goi vong 9 (Recommended)"): the firmware writes a
    motor-off data line only every 10 ticks, so a short stop leaves no 0 -> 1 edge in the CSV. Hence:
      (1) every start the events file shows ('#us,ev=start' or '#gain', by first_seq) must have its '#gain' line
          equal to the file (one file = one run, CG7: no start of another run belongs there); a '#gain' line
          with first_seq = 0 is the line the firmware writes when a set is applied with the motor off
          (logger_push_gain(0), no '#us' with it), not a start (S05 6.3; R134C9-1, Codex round 9);
      (2) every boundary of a stretch in the data must hold a start of (1) with lo < first_seq <= hi: a 0 -> 1 edge
          (lo = log_seq of the off line, hi = of the on line), or two motor-on lines one after the other with
          d log_seq != d tick (the motor was off at some tick between them: log_seq counts every pushed line, tick
          only the motor-on ticks; a lost data line moves both by one). A lost first data line of a stretch is
          therefore no problem when its start is shown. A boundary without a start refuses the run."""
    if not os.path.isfile(esp32_path):
        return f'thieu {os.path.basename(esp32_path)}: khong so duoc dong #gain cua luot voi file gain'
    try:
        rows, _ = read_esp32_log_info(esp32_path)
        events = read_events(esp32_path)
        if events is None:
            return 'thieu file events (#gain, #us,ev=start): khong so duoc dong #gain cua luot voi file gain'
        on = [x.get('motor_on') == '1' for x in rows]
        if not any(on) or on[0]:
            return 'khong xac dinh duoc dong du lieu dau luot (motor_on) de khop #gain voi file gain'
        seqs = set()
        for line in events:
            _, text = glc.split_event(line)
            if text.startswith('#gain,'):
                fs = glc._int(glc.parse_gain_line(text)['first_seq'])
                if fs != 0:       # R134C9-1: first_seq = 0 is the line of a gain setting (S05 6.3: not counted)
                    seqs.add(fs)
            elif text.startswith('#us,') and glc._fields(text).get('ev') == 'start':
                seqs.add(glc._int(glc._fields(text).get('first_seq')))
        if None in seqs:
            return 'dong #gain hay #us,ev=start co first_seq khong phai so'
        for seq in sorted(seqs):                                                         # (1)
            st = glc.run_start(events, seq)
            where = f'moc bat dau first_seq={seq} ({len(seqs)} moc trong events)'
            if st['gain'] is None:
                return f'dong #gain cua {where}: {"; ".join(st["why"])} (S05 6.3)'
            why = glc.gain_mismatch(st['gain'], want)
            if why:
                return f'dong #gain cua {where} khac file gain: {"; ".join(why)}'
        seq_of = [_int(x.get('log_seq')) for x in rows]
        tick_of = [_int(x.get('tick')) for x in rows]
        for i in range(1, len(rows)):                                                    # (2)
            if not on[i]:
                continue
            if on[i - 1]:
                if None in (seq_of[i], seq_of[i - 1], tick_of[i], tick_of[i - 1]):
                    return f'dong du lieu log_seq {seq_of[i]} thieu log_seq hay tick: khong tach duoc doan motor bat'
                if seq_of[i] - seq_of[i - 1] == tick_of[i] - tick_of[i - 1]:
                    continue
                kind = 'motor tat ngan giua hai dong motor bat (d log_seq != d tick)'
            else:
                kind = 'canh motor_on 0 -> 1'
            lo, hi = seq_of[i - 1], seq_of[i]
            if lo is None or hi is None:
                return f'dong du lieu quanh {kind} thieu log_seq: khong tach duoc doan motor bat'
            if not any(lo < s <= hi for s in seqs):
                return (f'{kind} giua log_seq {lo} va {hi}: khong co moc bat dau (#us,ev=start, #gain) trong '
                        'events de chung minh bo gain cua doan nay')
    except (OSError, glc.InputError) as e:
        return f'dong #gain cua luot: {e}'
    return None


def run_evidence_problem(row, query_before_name='gain_query_before.txt', query_after_name='gain_query_after.txt',
                         gain_ack_name='gain_ack.txt', esp32_name='esp32.csv', gain_file=None):
    """R134C5-1, R134C5-3 (Codex round 5; Liam 14/11 CODE_R134 hop chon "Bat buoc ack moi luot (Recommended)"):
    None, or why a judged final run (car or HIL) lacks the evidence S05 4.7 (iv) asks for, checked at the level of
    the run, apart from tv_duty: two answers in <results_dir> that are two questions one after the other
    (gain_line_check.query_problem), their boots equal to the two boot columns of the row, and the '#gain_ack' of
    the gain setting (every final run, the initial gain too, sets its gain from a file with --gain-file): present,
    readable, of the boot, MAC and running id of the question before the run.
    R134C6-1 (Codex round 6): and the set of the row's cell (gain_file: entry of gain_files(), None = no file for
    the gain_set_id): the run's '#gain' line equals that file bit for bit (run_gain_problem), and the id of the file
    is the running id of the question before (so of the ack too).
    R134C7-1, -2 (Codex round 7): the '#gain' line of every motor-on stretch; the running id of the question after
    equals the file id too; a frozen set whose file could not be proved (entry with 'error') refuses its runs."""
    d = row.get('results_dir', '')
    qb, qa = read_query(os.path.join(d, query_before_name)), read_query(os.path.join(d, query_after_name))
    why = glc.query_problem(qb, qa, check_run_boot=False)
    if why:
        return why
    cols = ((row.get('boot_before') or '').lower(), (row.get('boot_after') or '').lower())
    if cols != (qb['boot'].lower(), qa['boot'].lower()):
        return f'boot cua dong log {cols} khac hai file hoi ({qb["boot"]}, {qa["boot"]})'
    ap = os.path.join(d, gain_ack_name)
    ack = glc.read_ack(ap)
    if ack is None:
        return (f'{gain_ack_name} hong (khong mot dong # ack ok=1, hay dong cuoi khong ma=0)' if os.path.exists(ap)
                else f'thieu {gain_ack_name}: moi luot kiem cuoi dat gain bang --gain-file (Liam 14/11)')
    why = glc.ack_problem(ack, qb, {'id': qb.get('id')})
    if why:
        return why
    if gain_file is None:
        g = (row.get('gain_set_id') or '').split('/')[0]
        return (f'khong co file gain cua {row.get("gain_set_id")!r}: '
                + (f'cho --init-gain {g.split("-")[-1]}=<file>' if g.startswith('RQ2-INIT-')
                   else 'bo gain khong co dong trong S05_freeze.csv'))
    if 'error' in gain_file:                                                         # R134C7-2
        return f'file gain cua {row.get("gain_set_id")!r}: {gain_file["error"]}'
    want = gain_file['want']
    if int(qb['id']) != want['id']:
        return f'id dang chay {qb["id"]} cua lan hoi truoc khac id {want["id"]} cua file gain {gain_file["file"]}'
    if not glc._dec(qa.get('id')) or int(qa['id']) != want['id']:                    # R134C7-1
        return (f'id dang chay {qa.get("id")!r} cua lan hoi sau khac id {want["id"]} cua file gain '
                f'{gain_file["file"]}: bo gain doi trong luot')
    return run_gain_problem(os.path.join(d, esp32_name), want)


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
                                n_flag=sum(bool(r.get('flag')) for r in val),
                                n_cat_t1=sum(bool(r.get('cat_t1')) for r in val))      # R134C4-6
                    for q in QUANTITIES:
                        xs = [r['values'][q] for r in val if _a_number(r['values'].get(q))]   # B81-16
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
                    f"{by[(ctrl, cond, plat, g, QUANTITIES[0])]['n_chua_ket_luan']}, co {by[(ctrl, cond, plat, g, QUANTITIES[0])]['n_flag']}, "
                    f"cat_t1 {by[(ctrl, cond, plat, g, QUANTITIES[0])]['n_cat_t1']}"
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
            'n_hong_he_do', 'n_flag', 'n_cat_t1', 'quantity', 'n_m', 'median', 'max', 'missing']
    with open(os.path.join(out_dir, 'S05_final_table.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: ('' if t[k] is None else t[k]) for k in cols} for t in table)
    with open(os.path.join(out_dir, 'S05_final_table.md'), 'w', encoding='utf-8') as f:
        f.write(markdown(table))
    rcols = ['run_uid', 'platform', 'controller', 'condition', 'gain_set', 'outcome', 'flag'] + list(QUANTITIES) + \
            ['tv_reason', 'tv_d_tick', 'tv_d_seq', 'tv_end_gap_ms', 'tv_rows_dropped', 't_end_err_ms', 't_end_err_src',
             'results_dir', 'gain_set_id', 'gain_file', 'gain_sha256']
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
                            tv_end_gap_ms=tv.get('end_gap_ms', ''), tv_rows_dropped=tv.get('rows_dropped', ''),
                            t_end_err_ms='' if r.get('t_end_err_ms') is None else r['t_end_err_ms'],
                            t_end_err_src=r.get('t_end_err_src', ''),
                            results_dir=r['results_dir'], gain_set_id=r.get('gain_set_id', ''),
                            gain_file=r.get('gain_file', ''), gain_sha256=r.get('gain_sha256', '')))
    used = sorted({(r.get('gain_set_id', '').split('/')[0] if r.get('gain_source') == '--init-gain'
                    else r.get('gain_set_id', ''), r.get('gain_file', ''), r.get('gain_sha256', ''),
                    r.get('gain_source', '')) for r in runs if r.get('gain_file')})
    with open(os.path.join(out_dir, 'S05_final_gain_files.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)                        # R134C6-1: the gain file of every set in the table, with sha256
        w.writerow(['gain_set', 'gain_file', 'gain_sha256', 'source'])
        w.writerows(used)
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
    a1.add_argument('--gain-ack', default=None, help='gain_send.py output of the gain setting (#gain_ack; B81-14)')
    a1.add_argument('--no-boot', action='store_true', help='skip the boot check (tool test, old run)')
    a1.add_argument('--no-gain-line', action='store_true', help="skip the '#gain' first_seq check")
    a2 = sub.add_parser('table')
    a2.add_argument('--dataset', default=DEFAULT_DATASET)
    a2.add_argument('--out', required=True)
    a2.add_argument('--score-name', default='score.json')
    a2.add_argument('--esp32-name', default='esp32.csv')
    a2.add_argument('--query-before-name', default='gain_query_before.txt')
    a2.add_argument('--query-after-name', default='gain_query_after.txt')
    a2.add_argument('--gain-ack-name', default='gain_ack.txt')
    a2.add_argument('--init-gain', action='append', default=[], metavar='CTRL=FILE',
                    help='gain file of the initial gain RQ2-INIT-<CTRL> (PID, LADRC; R134C6-1)')
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'tv':
            rows, n_drop = read_esp32_log_info(a.log)
            r = tv_duty_esp32(rows, a.t_end, a.edge_max_ms,
                              queries=(read_query(a.query_before), read_query(a.query_after)),
                              events=read_events(a.log), check_boot=not a.no_boot,
                              check_gain_line=not a.no_gain_line, ack=glc.read_ack(a.gain_ack))
            r['rows_dropped'] = n_drop
            print(json.dumps(r, ensure_ascii=False, indent=1))
        else:
            init = {}
            for x in a.init_gain:
                c, sep, f = x.partition('=')
                if not sep or not f or c in init:
                    raise ValueError(f'--init-gain {x!r}: write CTRL=FILE, once per controller')
                init[c] = f
            runs, problems = load_runs(a.dataset, a.score_name, a.esp32_name, a.query_before_name,
                                       a.query_after_name, a.gain_ack_name, init)
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
