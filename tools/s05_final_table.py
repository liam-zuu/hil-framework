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
    - more than one run_active rising edge in the file (one file = one run, CG7).
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
no esp_pi_ppm: the RHO_PRIOR_PPM bound (12 ms at 12 s). Report only; the line choice is unchanged. Car runs:
the camera clock goes to the ESP32 through C7b, not here (column empty, 'xe: C7b').

Each cell (controller x condition x platform x gain set): n_valid, n_dat / n_khong_dat / n_chua_ket_luan,
n_flag, and per quantity median and max over the valid runs that have a number, n_m, and the reason of
every valid run without a number. The quantity list is fixed here (do not change after seeing numbers).

  python tools/s05_final_table.py tv --log <esp32 csv> --t-end <s>
  python tools/s05_final_table.py table [--dataset 05_DATA/M7_dataset] --out <dir>
      [--score-name score.json] [--esp32-name esp32.csv]
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
RHO_PRIOR_PPM = 1000.0             # ESP32 vs H7 rate bound of RS-4 (Liam 13/11, L100), when esp_pi_ppm is missing


def t_end_err_ms(t_end_s, esp_pi_ppm=None, pi_period_ppm=None):
    """Estimated error (ms) of t = 0 + t_end on the ESP32 log of a HIL run (module docstring), and its source."""
    if t_end_s is None:
        return None, 'khong co t_end'
    t = abs(float(t_end_s))
    if esp_pi_ppm is None:
        return t * RHO_PRIOR_PPM * 1e-3, f'khong co esp_pi_ppm: can {RHO_PRIOR_PPM:g} ppm'
    pi = PI_PERIOD_PPM_BOUND if pi_period_ppm is None else abs(float(pi_period_ppm))
    src = 'esp_pi_ppm + nhieu + ' + ('can Pi' if pi_period_ppm is None else 'pi_period_ppm')
    return t * (abs(float(esp_pi_ppm)) + ESP_PPM_NOISE + pi) * 1e-3, src


# ---------------------------------------------------------------- tv_duty
def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def read_esp32_log(path):
    """Data lines of an agv_logger CSV (header line first; '#' lines go to the events file)."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if _int(r.get('timestamp_us')) is not None]
    return rows


def tv_duty_esp32(rows, t_end_s, edge_max_ms=EDGE_MAX_MS, t0_us=None):
    """tv_duty of one run from its ESP32 log lines (dicts of strings). Returns a dict with tv_duty
    (None when no number) and every check; 'reason' says why there is no number."""
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
    d_tick, d_seq = tk[-1] - tk[0], sq[-1] - sq[0]
    out.update(d_tick=d_tick, d_seq=d_seq, lines_lost=d_seq - (len(win) - 1))
    if d_tick != d_seq:
        out['reason'] = f'tick tang {d_tick}, log_seq tang {d_seq}: motor tat o mot tick trong cua so'
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


def load_runs(dataset, score_name, esp32_name):
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
                tv = tv_duty_esp32(read_esp32_log(ep), float(score['t_end']))
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
            ['tv_reason', 'tv_d_tick', 'tv_d_seq', 'tv_end_gap_ms', 't_end_err_ms', 't_end_err_src', 'results_dir']
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
                            tv_end_gap_ms=tv.get('end_gap_ms', ''),
                            t_end_err_ms='' if r.get('t_end_err_ms') is None else r['t_end_err_ms'],
                            t_end_err_src=r.get('t_end_err_src', ''), results_dir=r['results_dir']))
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('tv')
    a1.add_argument('--log', required=True)
    a1.add_argument('--t-end', type=float, required=True, help='t_end of the run, s after t = 0')
    a1.add_argument('--edge-max-ms', type=float, default=EDGE_MAX_MS)
    a2 = sub.add_parser('table')
    a2.add_argument('--dataset', default=DEFAULT_DATASET)
    a2.add_argument('--out', required=True)
    a2.add_argument('--score-name', default='score.json')
    a2.add_argument('--esp32-name', default='esp32.csv')
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'tv':
            r = tv_duty_esp32(read_esp32_log(a.log), a.t_end, a.edge_max_ms)
            print(json.dumps(r, ensure_ascii=False, indent=1))
        else:
            runs, problems = load_runs(a.dataset, a.score_name, a.esp32_name)
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
