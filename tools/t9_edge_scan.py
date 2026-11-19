"""T9 (J v6 section 5 (1), section 7): read ESP32 logs for the heading-loop latch and the tick start ->
latch interval. Analysis only, nothing is written next to the data.

  edges <log.csv ...>   every run_active rising edge in agv_logger CSVs: edge-row r_valid, head_on,
                        psi_ref, age_r_ms; counters on the run_active stretch; gap to the next row
                        (rigorous bound of tick start -> latch, see below) and age bound
                        age_r_ms + 0.05 + gap; plus all motor-on consecutive-row gaps
  s06 <run_dir ...>     S06 runs (env s06_load, pi/esp/esp32_data.txt + esp32_events.txt): P line
                        t_before (first compare write of the tick, after vel_ctrl_step) minus t_now of
                        the same tick (latest data row timestamp_us <= t_before, next row log_seq + 1)

Why the next row bounds the latch (AGV 5bde4c5 main.c control_task): t_now (line 404), then
vel_ctrl_step() (line 621) which calls heading_step(), then push_log(); the next iteration reads the next
t_now. One task, in order, so latch(k) < t_now(k + 1) = timestamp_us of the next data row when the two
rows have consecutive log_seq (motor on: one data row per tick; #fw/#gain/#us lines take no log_seq).
"""
import bisect
import csv
import os
import sys


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


def q(v, p):
    return v[min(len(v) - 1, int(p * len(v)))]


def edges(paths):
    bounds, gaps = [], []
    for path in paths:
        rs = sorted(rows_of(path), key=lambda r: int(r['log_seq']))
        if not rs or 'run_active' not in rs[0]:
            print(f'{path}: no run_active column')
            continue
        for i in range(len(rs) - 1):
            a, b = rs[i], rs[i + 1]
            if a['motor_on'] == '1' and b['motor_on'] == '1' and int(b['log_seq']) == int(a['log_seq']) + 1:
                gaps.append((int(b['timestamp_us']) - int(a['timestamp_us'])) / 1000)
        for i in range(1, len(rs)):
            if not (rs[i]['run_active'] == '1' and rs[i - 1]['run_active'] != '1'):
                continue
            e = rs[i]
            j = i
            while j + 1 < len(rs) and rs[j + 1]['run_active'] == '1':
                j += 1
            ch = {k: sorted({r[k] for r in rs[i:j + 1]}) for k in ('psi_reset_count', 'reset_n', 'pos_reset_n')}
            ch = {k: v for k, v in ch.items() if len(v) > 1}
            nxt = rs[i + 1] if i + 1 < len(rs) else None
            gap = ((int(nxt['timestamp_us']) - int(e['timestamp_us'])) / 1000
                   if nxt and int(nxt['log_seq']) == int(e['log_seq']) + 1 else None)
            bound = float(e['age_r_ms']) + 0.05 + gap if gap is not None else None
            if bound is not None:
                bounds.append(bound)
            print(f"{os.path.relpath(path)}: seq={e['log_seq']} dseq_prev={int(e['log_seq']) - int(rs[i - 1]['log_seq'])} "
                  f"r_valid={e['r_valid']} head_on={e['head_on']} psi_ref={e['psi_ref']} e_psi={e['e_psi']} "
                  f"age_r_ms={e['age_r_ms']} gap_ms={gap} bound_ms={None if bound is None else round(bound, 2)} "
                  f"rows={j - i + 1} changed={ch}")
    if bounds:
        b = sorted(bounds)
        print(f'edges with bound: n={len(b)} min={b[0]:.2f} median={q(b, .5):.2f} max={b[-1]:.2f} ms')
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
