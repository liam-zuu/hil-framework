"""T9 (J v6 section 5 (1), section 7): read ESP32 logs for the heading-loop latch and the tick start ->
latch interval. Analysis only, nothing is written next to the data. Not the scorer: run windows (end
anchor), boot and failure order belong to the scorer; this tool shows the edge evidence per file.

  edges <log.csv ...>   every run_active 0 -> 1 change in agv_logger CSVs. Per file it first checks the
                        boot evidence it can see: log_seq unique and timestamp_us increasing in log_seq
                        order (else 'CANH_BAO' and no bounds from that file: maybe two boots or mixed
                        files). Per edge: edge-row r_valid, rv_n, head_on, psi_ref, pos_reset_n, age_r_ms;
                        the run_active stretch (to the last active row before the next 0 -> 1 change;
                        rows with run_active 0 inside a run, e.g. a trajectory timeout, end the stretch:
                        the next edge then shows pos_reset_n + 1 and 'since_prev_ms', the time from the end
                        of the previous stretch); gap to the next row; and, only when the row before the
                        edge is log_seq - 1 with run_active 0, the edge row shows the full latch (r_valid 1,
                        rv_n > 0, head_on 1, psi_ref 0), the next row is log_seq + 1 and later, and the
                        logged age lies in [-gap - 0.05 ms, 125.05 ms], the age bound
                        age_r_ms + 50 us + gap + 3 us (integer us, as ref_cond1 of
                        fwsim/t9_heading_cases.py); plus all motor-on consecutive-row gaps
  s06 <run_dir ...>     S06 runs (env s06_load, pi/esp/esp32_data.txt + esp32_events.txt): P line
                        t_before (first compare write of the tick, after vel_ctrl_step) minus t_now of
                        the same tick (latest data row timestamp_us <= t_before, next row log_seq + 1)

Why the next row bounds the latch (AGV 5bde4c5 main.c control_task): the last t_now read of tick k (line
404), then vel_ctrl_step() (line 621) which calls heading_step(), then push_log(); the next data row's
timestamp is read later in the same task. So on the integer esp_timer scale latch(k) - t_now(k) <= the
timestamp difference to the next data row of the same boot (+ 1 us for continuous time). With consecutive
log_seq that row is the next pushed data row (#fw/#gain/#us take no log_seq; #run_end does, only on the
tick the motor turns off); it is the next tick while the motor is on.
"""
import bisect
import csv
import math
import os
import sys

T_AGE_US = 125_000      # imu_fusion.h t_age_us at 20 Hz (IMU_REPORT_PERIOD_US 50 000)


def rows_of(path):
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        hdr = None
        for row in csv.reader(f):
            if not row:
                continue
            if 'timestamp_us' in row:
                hdr = row
                continue
            if hdr is None or row[0].startswith('#') or len(row) < len(hdr):
                continue
            yield dict(zip(hdr, row))


def num(r, k):
    try:
        v = float(r.get(k))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def q(v, p):
    return v[min(len(v) - 1, int(p * len(v)))]


def edges(paths):
    bounds, gaps = [], []
    for path in paths:
        rs = [r for r in rows_of(path) if num(r, 'log_seq') is not None and num(r, 'timestamp_us') is not None]
        if not rs or 'run_active' not in rs[0]:
            print(f'{path}: no run_active column')
            continue
        rs.sort(key=lambda r: num(r, 'log_seq'))
        # identical duplicate lines (UDP sent twice) are dropped; a log_seq seen with different content is not
        uniq, conflict = [], 0
        for r in rs:
            if uniq and num(uniq[-1], 'log_seq') == num(r, 'log_seq'):
                conflict += uniq[-1] != r
                continue
            uniq.append(r)
        n_same = len(rs) - len(uniq) - conflict
        rs = uniq
        seqs = [num(r, 'log_seq') for r in rs]
        ts = [num(r, 'timestamp_us') for r in rs]
        nonmono = sum(1 for a, b in zip(ts, ts[1:]) if b <= a)
        trusted = conflict == 0 and nonmono == 0
        if n_same:
            print(f'{path}: bo {n_same} dong trung y het (cung log_seq, cung noi dung)')
        if not trusted:
            print(f'{path}: CANH_BAO log_seq trung khac noi dung={conflict}, gio khong tang={nonmono}: '
                  f'nghi nhieu boot hay file tron; khong tinh can tu file nay')
        for i in range(len(rs) - 1):
            a, b = rs[i], rs[i + 1]
            if a['motor_on'] == '1' and b['motor_on'] == '1' and seqs[i + 1] == seqs[i] + 1 and trusted:
                gaps.append((ts[i + 1] - ts[i]) / 1000)
        prev_end = None
        for i in range(len(rs)):
            if not (num(rs[i], 'run_active') == 1 and (i == 0 or num(rs[i - 1], 'run_active') != 1)):
                continue
            e = rs[i]
            j = i
            while j + 1 < len(rs) and num(rs[j + 1], 'run_active') == 1:
                j += 1
            ch = {k: sorted({r[k] for r in rs[i:j + 1]}) for k in ('psi_reset_count', 'reset_n', 'pos_reset_n')}
            ch = {k: v for k, v in ch.items() if len(v) > 1}
            since_prev = None if prev_end is None else (ts[i] - ts[prev_end]) / 1000
            prev_end = j
            pre_ok = i > 0 and seqs[i - 1] == seqs[i] - 1 and num(rs[i - 1], 'run_active') == 0
            latched = (num(e, 'r_valid') == 1 and (num(e, 'rv_n') or 0) > 0 and num(e, 'head_on') == 1
                       and num(e, 'psi_ref') == 0.0)
            gap_us = (int(ts[i + 1] - ts[i]) if i + 1 < len(rs) and seqs[i + 1] == seqs[i] + 1
                      and ts[i + 1] > ts[i] else None)
            age = num(e, 'age_r_ms')
            age_us = None if age is None else int(round(age * 10)) * 100
            bound = None
            if (trusted and pre_ok and latched and gap_us is not None and age_us is not None
                    and -gap_us - 50 <= age_us <= T_AGE_US + 50):
                bound = (age_us + 50 + gap_us + 3) / 1000
                bounds.append(bound)
            print(f"{path}: seq={e['log_seq']} pre_ok={pre_ok} "
                  f"r_valid={e.get('r_valid')} rv_n={e.get('rv_n')} head_on={e.get('head_on')} "
                  f"psi_ref={e.get('psi_ref')} e_psi={e.get('e_psi')} pos_reset_n={e.get('pos_reset_n')} "
                  f"age_r_ms={e.get('age_r_ms')} gap_ms={None if gap_us is None else gap_us / 1000} "
                  f"bound_ms={bound} rows={j - i + 1} since_prev_ms={since_prev} changed={ch}")
    if bounds:
        b = sorted(bounds)
        print(f'edges with bound: n={len(b)} min={b[0]:.3f} median={q(b, .5):.3f} max={b[-1]:.3f} ms')
    if gaps:
        g = sorted(gaps)
        print(f'motor-on consecutive gaps: n={len(g)} min={g[0]} p50={q(g, .5)} p99={q(g, .99)} '
              f'p999={q(g, .999)} max={g[-1]} ms')


def s06(dirs):
    allv = []
    for d in dirs:
        rows = []
        for line in open(os.path.join(d, 'pi', 'esp', 'esp32_data.txt'), encoding='utf-8', errors='replace'):
            p = line.rstrip('\n').split('\t', 1)
            if len(p) < 2 or not p[1][:1].isdigit():
                continue
            f = p[1].split(',')
            if len(f) < 41:
                continue
            try:
                rows.append((int(f[0]), int(f[37])))      # timestamp_us, log_seq (logger.c column order)
            except ValueError:
                pass
        rows.sort()
        ts = [r[0] for r in rows]
        per_tick, skipped, fw = {}, 0, ''
        for line in open(os.path.join(d, 'pi', 'esp', 'esp32_events.txt'), encoding='utf-8', errors='replace'):
            p = line.rstrip('\n').split('\t', 1)
            if len(p) < 2:
                continue
            if p[1].startswith('#run_start'):
                fw = ','.join(x for x in p[1].split(',') if x.startswith(('fw=', 'env=', 'profile=')))
            if not p[1].startswith('P,'):
                continue
            t = int(p[1].split(',')[2])
            i = bisect.bisect_right(ts, t) - 1
            if i < 0 or i + 1 >= len(rows) or rows[i + 1][1] != rows[i][1] + 1:
                skipped += 1
                continue
            per_tick[rows[i][0]] = min(per_tick.get(rows[i][0], 10 ** 12), t - rows[i][0])
        v = sorted(per_tick.values())
        allv += v
        if v:
            print(f'{d}: {fw} ticks={len(v)} skipped={skipped} min={v[0]} p50={q(v, .5)} max={v[-1]} us')
        else:
            print(f'{d}: {fw} no tick with a P line and both rows')
    if allv:
        allv.sort()
        print(f'all: ticks={len(allv)} min={allv[0]} p50={q(allv, .5)} p99={q(allv, .99)} max={allv[-1]} us')


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ('edges', 's06'):
        sys.exit(__doc__)
    (edges if sys.argv[1] == 'edges' else s06)(sys.argv[2:])
