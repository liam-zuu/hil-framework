#!/usr/bin/env python3
"""
s06_t3_score.py — chấm S06 T3 (i) trễ HIL và (ii) lượng tử cạnh, theo cửa sổ ghép (spec S06 dòng T3: V2, Z7,
L7-10, V2-4; thiết kế 03_CODE/docs/2026-11-15_Thiet_ke_S06_T3_danh_dau_v1.md mục 4 ý 4, Liam 15/11 "Duyệt,
code luôn"; hồ sơ lượt t3: Liam 17/11 "Hồ sơ t3 riêng (Recommended)").

Lượt: RPi5 `run_hil.py --omega-schedule T3_schedule.csv --marker t3 --frame-log frames.npz` (lịch của
rpi5/t3_schedule.py), ESP32 rời env `s06_load` lượt `START t3 …` (s06_logger.py --profile t3). Mọi giờ trong phép
chấm là giờ ESP32 (`esp_timer` trong ISR GPIO), nên (i) và (ii) không qua đồng hồ Pi:
  * mốc j = cạnh M (chân đánh dấu Pi, nâng ngay trước `transfer` của frame có bậc j);
  * cửa sổ j = [mốc j; mốc j + 2·hp_cũ,j + 1 ms], hp_cũ = nửa chu kỳ H7 ở ω cũ của bánh đổi (t3_schedule.hp_ticks);
  * cạnh D = PF3 đảo (H7 vừa hẹn lại CCR); mỗi cửa sổ phải có đúng một D: 0 → bản ghi hết hạn (vượt cận), ≥ 2 →
    bậc không đạt (xung thừa); D ngoài mọi cửa sổ → H3 không đạt;
  * (i) = t_D − t_M; (ii) = cạnh N đầu tiên của bánh đổi (kênh A hay B, sớm nhất) sau D và trong cửa sổ, trừ t_D.
Ghép bậc j ↔ cạnh M thứ j: ghép cặp cạnh Pi (frames.npz `marker`) với cạnh M bằng s06_clock_sync (khoảng cách
giữa bậc 60–120 ms ngẫu nhiên); mỗi cạnh Pi của lượt phải có cặp, mỗi cạnh M của lượt phải có cặp. Frame có
marker phải theo lịch (đúng bánh, đúng ω mới đã lượng tử, đúng thứ tự).

Độ bất định của cách đo (spec ≤ 0,25 ms; mục 4.1): M, D, N cùng đi đường ISR GPIO lõi 0 của ESP32, nên phần
chung của trễ ngắt triệt tiêu trong hiệu; còn phần dao động. Đo bằng cặp O → L (ESP32 đảo GPIO39 trong vùng tắt
ngắt, ghi giờ ngay trước lệnh ghi; ISR GPIO37 qua dây nối tắt): u = (trễ O→L lớn nhất − nhỏ nhất) + 2 µs (độ phân
giải `esp_timer` ở hai đầu) [ĐỀ XUẤT CODE_S06]. Ít hơn LOOP_MIN cặp → thiếu bằng chứng (mã 3). Khoảng GPIO → SPI
phía Pi (t_send − t_pulse1 của frame có marker) là thành phần của trễ đo được (thiên an toàn), báo kèm.

Kết luận (i), mục 4 spec: trễ lớn nhất + u ≤ 1 ms → đạt; trễ lớn nhất − u > 1 ms → không đạt; còn lại chưa kết luận
(mã 3). u > 0,25 ms → chưa kết luận. (ii) chỉ báo số; một lần đo > hp_mới + u ghi `bug_class` = loi_cong_cu_HIL,
không đổi kết luận (spec: không đặt ngưỡng H3 cho (ii)).
PF3-4 (Codex PF3 vòng 1–2): cạnh D trước `#run_start` của lượt (giờ ESP32 `timestamp_us`) bị bỏ, đếm riêng.

Mất dữ liệu (Z19) → mã 1: luồng qseq của lượt hở (s06_stream.qseq_stream), hở `gseq` giữa các dòng T3 của lượt,
`t3_ovf_run` > 0 ở #run_stats, cạnh Pi hay cạnh M không có cặp, frames.npz thiếu frame (manifest_check).
Mã thoát: 0 đạt; 1 không đạt hoặc mất dữ liệu; 3 chưa kết luận / thiếu bằng chứng / không chọn được lượt;
2 lỗi đầu vào (kể cả frame có marker không theo lịch). Dòng cuối `#ket_qua,…`.

CLI:
  python tools/s06_t3_score.py --schedule T3_schedule.csv --frames frames.npz --esp32-events esp32_events.txt \\
      [--run N] --out <thư mục>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'rpi5'))
import s06_esp32_check as chk          # noqa: E402
import s06_stream as stm               # noqa: E402
import s06_clock_sync as cs            # noqa: E402
import s06_frames_decode as fd         # noqa: E402
import t3_schedule as ts               # noqa: E402  rpi5/

LIMIT_US = 1000.0            # (i) trễ HIL lớn nhất ≤ 1 ms
U_MAX_US = 250.0             # độ bất định ≤ 0,25 ms (¼ mục tiêu)
TIMER_RES_US = 1.0           # esp_timer: 1 µs
LOOP_MIN = 10                # cặp O → L tối thiểu để có u
PIN_MARK, PIN_DBG = 38, 36
# chân encoder ESP32 theo bánh của frame (FL, FR, RL, RR): AGV src/encoder/encoder.h (A, B)
WHEEL_PINS = {0: (4, 5), 1: (2, 42), 2: (6, 7), 3: (20, 19)}

COLUMNS = ['step', 'wheel', 'w_from', 'w_to', 'hp_old_us', 'hp_new_us', 'frame', 't_mark_esp_us',
           'win_end_esp_us', 'n_dbg', 't_dbg_esp_us', 'delay_us', 'pin_n', 't_n_esp_us', 'quant_us',
           'gpio_spi_us', 'verdict_i', 'bug_class', 'note']


def marked_frames(z) -> List[dict]:
    """Frame có marker: chỉ số, bánh đổi, ω cũ/mới (giá trị frame, đã lượng tử) theo MOSI thật gửi."""
    from protocol import decode_state
    m = np.nonzero(np.asarray(z['marker']).astype(bool))[0]
    mosi = np.asarray(z['mosi'], dtype=np.uint8)
    out = []
    for k in m:
        if k == 0:
            raise ValueError('frame 0 có marker (không có frame trước để biết ω cũ)')
        w0 = decode_state(bytes(mosi[k - 1]))[0:4]
        w1 = decode_state(bytes(mosi[k]))[0:4]
        ch = [i for i in range(4) if w0[i] != w1[i]]
        out.append(dict(frame=int(k), changed=ch, w_old=w0, w_new=w1))
    return out


def check_schedule(marks: List[dict], sched) -> List[str]:
    """Frame có marker phải theo lịch: cùng số bậc, đúng bánh, đúng ω mới đã lượng tử, chỉ một bánh đổi."""
    rows = sorted(sched, key=lambda r: r.t0_ms)
    bad = []
    if len(marks) != len(rows):
        bad.append(f'{len(marks)} frame có marker, lịch có {len(rows)} bậc')
    for j, (mk, r) in enumerate(zip(marks, rows)):
        if mk['changed'] != [r.wheel]:
            bad.append(f'bậc {j}: frame {mk["frame"]} đổi bánh {mk["changed"]}, lịch bánh {r.wheel}')
            continue
        q_to = ts.frame_int16(r.w_to) * ts.OMEGA_RANGE / 32767.0
        q_from = ts.frame_int16(r.w_from) * ts.OMEGA_RANGE / 32767.0
        if abs(mk['w_new'][r.wheel] - q_to) > 1e-6 or abs(mk['w_old'][r.wheel] - q_from) > 1e-6:
            bad.append(f'bậc {j}: frame {mk["frame"]} ω {mk["w_old"][r.wheel]:.4f} → {mk["w_new"][r.wheel]:.4f}, '
                       f'lịch {r.w_from} → {r.w_to}')
    return bad


def loop_latency(t3_rows: List[dict]) -> dict:
    """Trễ O → L (µs): mỗi O ghép với L đầu tiên sau nó (theo gseq) cùng mức, trước O kế."""
    ev = sorted((r for r in t3_rows if r['src'] in 'OL'), key=lambda r: r['gseq'])
    lat, pend = [], None
    for r in ev:
        if r['src'] == 'O':
            pend = r
        elif pend is not None and r['lvl'] == pend['lvl']:
            lat.append(r['t_us'] - pend['t_us'])
            pend = None
    if len(lat) < LOOP_MIN:
        return dict(n=len(lat), min_us=None, median_us=None, max_us=None, u_us=None)
    a = np.array(lat, dtype=np.float64)
    return dict(n=len(lat), min_us=float(a.min()), median_us=float(np.median(a)), max_us=float(a.max()),
                u_us=float(a.max() - a.min() + 2 * TIMER_RES_US))


def score(steps: List[dict], t3_rows: List[dict], t_run_start: int, u_us: Optional[float]) -> dict:
    """steps: mỗi bậc có wheel, w_from, w_to, t_mark (giờ ESP32 của cạnh M ghép được, None nếu không có), frame,
    gpio_spi_us. Trả các dòng kết quả và tổng hợp."""
    dbg = sorted(r['t_us'] for r in t3_rows if r['src'] == 'D' and r['t_us'] >= t_run_start)
    pre = sum(1 for r in t3_rows if r['src'] == 'D' and r['t_us'] < t_run_start)
    enc: Dict[int, List[int]] = {}
    for r in t3_rows:
        if r['src'] == 'N':
            enc.setdefault(r['gpio'], []).append(r['t_us'])
    for v in enc.values():
        v.sort()
    used = set()
    rows = []
    for j, s in enumerate(steps):
        hp_old = float(ts.hp_ticks(s['w_from']))
        hp_new = float(ts.hp_ticks(s['w_to']))
        row = dict(step=j, wheel=s['wheel'], w_from=s['w_from'], w_to=s['w_to'], hp_old_us=hp_old,
                   hp_new_us=hp_new, frame=s['frame'], t_mark_esp_us=s['t_mark'], win_end_esp_us=None,
                   n_dbg=None, t_dbg_esp_us=None, delay_us=None, pin_n=None, t_n_esp_us=None, quant_us=None,
                   gpio_spi_us=s['gpio_spi_us'], verdict_i='', bug_class='', note='')
        if s['t_mark'] is None:
            row.update(verdict_i='no_data', note='không có cạnh M ghép với frame có marker')
            rows.append(row)
            continue
        t0 = s['t_mark']
        t1 = t0 + 2.0 * hp_old + 1000.0
        row['win_end_esp_us'] = t1
        i0, i1 = np.searchsorted(dbg, t0, 'left'), np.searchsorted(dbg, t1, 'right')
        inw = dbg[i0:i1]
        used.update(range(i0, i1))
        row['n_dbg'] = len(inw)
        if len(inw) == 0:
            row.update(verdict_i='het_han', note='không có cạnh D trong cửa sổ: bản ghi hết hạn (vượt cận)')
        elif len(inw) > 1:
            row.update(verdict_i='khong_dat', note=f'{len(inw)} cạnh D trong cửa sổ (xung thừa của bộ HIL)')
        else:
            td = inw[0]
            row.update(t_dbg_esp_us=td, delay_us=float(td - t0), verdict_i='do_duoc')
            best = None
            for pin in WHEEL_PINS[s['wheel']]:
                ts_pin = enc.get(pin, [])
                k = int(np.searchsorted(ts_pin, td, 'right'))
                if k < len(ts_pin) and ts_pin[k] <= t1 and (best is None or ts_pin[k] < best[1]):
                    best = (pin, ts_pin[k])
            if best is None:
                row.update(note='(ii): không có cạnh encoder của bánh sau D trong cửa sổ (bản ghi hết hạn)',
                           bug_class='loi_cong_cu_HIL')
            else:
                q = float(best[1] - td)
                row.update(pin_n=best[0], t_n_esp_us=best[1], quant_us=q)
                if u_us is not None and q > hp_new + u_us:
                    row.update(bug_class='loi_cong_cu_HIL', note=f'(ii) {q:.0f} µs > hp mới {hp_new:.0f} + u')
        rows.append(row)
    stray = len(dbg) - len(used)
    delays = [r['delay_us'] for r in rows if r['delay_us'] is not None]
    summ = dict(steps=len(rows), measured=len(delays),
                expired=sum(1 for r in rows if r['verdict_i'] == 'het_han'),
                extra_pulse=sum(1 for r in rows if r['verdict_i'] == 'khong_dat'),
                no_data=sum(1 for r in rows if r['verdict_i'] == 'no_data'),
                stray_dbg=stray, dbg_before_run_start=pre,
                delay_max_us=max(delays) if delays else None,
                delay_median_us=float(np.median(delays)) if delays else None,
                quant_max_us=max((r['quant_us'] for r in rows if r['quant_us'] is not None), default=None),
                quant_bug=sum(1 for r in rows if r['bug_class']),
                per_wheel={w: sum(1 for r in rows if r['wheel'] == w) for w in range(4)})
    return dict(rows=rows, summary=summ)


def conclude(summ: dict, u_us: Optional[float], lost: List[str]) -> tuple:
    """(mã, lý do mã 3, danh sách lý do) theo spec mục 4 và Z19."""
    why = []
    if lost or summ['no_data']:
        return 1, [], lost + ([f"{summ['no_data']} bậc không có cạnh M"] if summ['no_data'] else [])
    fails = []
    if summ['expired']:
        fails.append(f"{summ['expired']} bậc hết hạn (không có D trong cửa sổ)")
    if summ['extra_pulse']:
        fails.append(f"{summ['extra_pulse']} bậc có ≥ 2 cạnh D")
    if summ['stray_dbg']:
        fails.append(f"{summ['stray_dbg']} cạnh D ngoài mọi cửa sổ")
    dmax = summ['delay_max_us']
    if u_us is not None and dmax is not None and dmax - u_us > LIMIT_US:
        fails.append(f'trễ HIL lớn nhất {dmax:.0f} µs − u {u_us:.0f} > {LIMIT_US:.0f} µs')
    if fails:
        return 1, [], fails
    if u_us is None:
        return 3, ['thieu_bang_chung'], ['không đủ cặp O → L (dây GPIO39 → GPIO37) để có độ bất định']
    if dmax is None:
        return 3, ['thieu_bang_chung'], ['không bậc nào đo được']
    if u_us > U_MAX_US:
        why.append(f'độ bất định {u_us:.0f} µs > {U_MAX_US:.0f} µs')
    if dmax + u_us > LIMIT_US:
        why.append(f'trễ lớn nhất {dmax:.0f} µs ± u {u_us:.0f} chạm {LIMIT_US:.0f} µs')
    if why:
        return 3, ['chua_ket_luan'], why
    return 0, [], []


def write_rows(path: str, rows: List[dict]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.3f}' if isinstance(v, float) else '' if v is None else v) for k, v in r.items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Chấm S06 T3 (i), (ii)')
    ap.add_argument('--schedule', required=True, help='T3_schedule.csv của rpi5/t3_schedule.py')
    ap.add_argument('--frames', required=True, help='frames.npz của run_hil.py --marker t3 --frame-log')
    ap.add_argument('--esp32-events', required=True, help='esp32_events.txt của s06_logger.py --profile t3')
    ap.add_argument('--run', type=int, default=None, help='số lượt (run= của #run_start) khi file có nhiều lượt')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    lost: List[str] = []
    notes: List[str] = []
    try:
        sched = ts.read_csv(a.schedule)
        bad = ts.check(sched)
        if bad:
            raise ValueError('lịch không đạt tự kiểm: ' + '; '.join(bad[:3]))
        z = np.load(a.frames)
        manifest = fd.load_manifest(z)
        if manifest is None or manifest.get('marker_mode') != 't3':
            raise ValueError('frames.npz không phải lượt run_hil --marker t3 (manifest marker_mode)')
        lost += fd.manifest_check(manifest, z['t_send_ns'], z['mosi'], z['miso'])
        mk = marked_frames(z)
        bad = check_schedule(mk, sched)
        if bad:
            raise ValueError('frame có marker không theo lịch: ' + '; '.join(bad[:3]))
        marks = stm.run_marks(a.esp32_events, a.run)
        stm.require_run(marks)
        kv = marks['run_start']['kv']
        if kv.get('profile') != 't3':
            raise ValueError(f"lượt ESP32 hồ sơ {kv.get('profile')!r}, cần t3")
        run_id = int(kv['run'])
        t_run_start = int(kv['timestamp_us'])
        ev = chk.parse_events(chk.read_raw(a.esp32_events))
    except stm.RunProblem as ex:
        return stm.report_run_problem(ex)
    except (OSError, ValueError, KeyError) as ex:
        print(f'lỗi đầu vào: {ex}', file=sys.stderr)
        return 2
    # Z19: luồng của lượt
    qs = stm.qseq_stream(a.esp32_events, marks)
    if not qs['ok']:
        lost.append(f"luồng qseq của lượt: thiếu {qs['missing']}, trùng {qs['dups']}, bỏ {qs['ev_qdrop_run']}")
    elif not qs['known']:
        lost.append('không chứng minh được luồng qseq đủ (thiếu mốc #run_start/#run_end)')
    rows3 = [r for r in ev['T3'] if r['run'] == run_id]
    g = sorted(r['gseq'] for r in rows3)
    gaps = sum(b - a_ - 1 for a_, b in zip(g, g[1:]) if b - a_ > 1)
    if gaps:
        lost.append(f'hở gseq {gaps} cạnh trong lượt (vòng đệm T3 tràn hay mất UDP)')
    ovf = marks.get('run_stats', {}).get('kv', {}).get('t3_ovf_run')
    if ovf is None:
        lost.append('#run_stats không có t3_ovf_run (firmware chưa có T3?)')
    elif int(ovf):
        lost.append(f't3_ovf_run = {ovf}')
    dbg_lv = [r['lvl'] for r in sorted(rows3, key=lambda r: r['gseq']) if r['src'] == 'D' and r['t_us'] >= t_run_start]
    rep = sum(1 for x, y in zip(dbg_lv, dbg_lv[1:]) if x == y)
    if rep:
        notes.append(f'{rep} lần hai cạnh D liền nhau cùng mức (lỡ một cạnh PF3?)')
    # ghép bậc ↔ cạnh M
    p0 = np.asarray(z['t_pulse0_ns'], dtype=np.int64)
    p1 = np.asarray(z['t_pulse1_ns'], dtype=np.int64)
    fr = np.array([m['frame'] for m in mk])
    p_us = (p0[fr] + p1[fr]) / 2000.0
    half = (p1[fr] - p0[fr]) / 2000.0
    e, e_rx = cs.esp_marks(ev, run_id)
    sync = cs.sync(p_us, half, e, e_rx)
    pair = dict(sync.get('_pairs', []))
    if sync['n_pairs'] < len(mk):
        lost.append(f"{len(mk) - sync['n_pairs']} frame có marker không có cạnh M (cặp {sync['n_pairs']}/{len(mk)})")
    if sync['esp_unmatched']:
        lost.append(f"{sync['esp_unmatched']} cạnh M của lượt không thuộc frame có marker nào")
    gspi = (np.asarray(z['t_send_ns'], dtype=np.int64)[fr] - p1[fr]) / 1000.0
    rows_sched = sorted(sched, key=lambda r: r.t0_ms)
    steps = [dict(wheel=r.wheel, w_from=r.w_from, w_to=r.w_to, frame=mk[j]['frame'],
                  t_mark=float(e[pair[j]]) if j in pair else None, gpio_spi_us=float(gspi[j]))
             for j, r in enumerate(rows_sched)]
    lat = loop_latency(rows3)
    res = score(steps, rows3, t_run_start, lat['u_us'])
    summ = res['summary']
    code, reasons, why = conclude(summ, lat['u_us'], lost)
    os.makedirs(a.out, exist_ok=True)
    write_rows(os.path.join(a.out, 'T3_result.csv'), res['rows'])
    out = dict(code=code, reasons=why, notes=notes, run=run_id, summary=summ, loop_latency=lat,
               gpio_spi_us=dict(median=float(np.median(gspi)) if len(gspi) else None,
                                max=float(np.max(gspi)) if len(gspi) else None),
               clock={k: v for k, v in sync.items() if not k.startswith('_')},
               qseq_stream=qs, gseq_gaps=gaps, t3_ovf_run=ovf, schedule=a.schedule, frames=a.frames,
               esp32_events=a.esp32_events)
    with open(os.path.join(a.out, 'T3_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print(f"T3: {summ['steps']} bậc, đo được {summ['measured']}, hết hạn {summ['expired']}, xung thừa "
          f"{summ['extra_pulse']}, D ngoài cửa sổ {summ['stray_dbg']}; trễ lớn nhất {summ['delay_max_us']} µs, "
          f"u {lat['u_us']} µs; (ii) lớn nhất {summ['quant_max_us']} µs")
    for x in why + notes:
        print(' -', x)
    return stm.report_result(code, reasons)


if __name__ == '__main__':
    raise SystemExit(main())
