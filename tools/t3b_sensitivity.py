#!/usr/bin/env python3
"""
t3b_sensitivity.py — kiểm độ nhạy T3b có đối chứng độc lập (spec S06 v3.4 T3b, Z23), trên PC.

Cùng lịch T3b (rpi5/t3b_schedule.py, seed), cùng frame (ω lượng tử như frame SPI), cùng thời điểm
gửi/áp, cùng các lần đọc 1 ms trên MỘT cơ sở thời gian; chấm bằng đúng tools/t3b_score.py:
  (0)  bộ phát lý tưởng "như khai báo" (Python): cạnh đúng từ ω_eff theo thời gian thực kể từ lúc
       áp frame; mỗi lần khởi động đặt lại pha, cạnh đầu ra sau hp (= 2 count tích phân), mỗi lần
       dừng bỏ phần lẻ; không có cạnh thừa. PHẢI ĐẠT MỌI CA.
  (0s) như (0) cộng 2 cạnh thừa ngược chiều cũ ở mỗi lần dừng (cận "tới 2 cạnh" của spec): chỉ báo.
  (1)  EncoderGen cũ (git `--old-rev`, mặc định cb91e7a = trước bản sửa retime) trên harness
       stm32h7/pc_tests/encgen/encgen_t3b_main.c: ca có lỗi A11 (bước giải mã ngược theo từng cạnh,
       cột truth_rev) phải bị báo không đạt; ca có lỗi mà T3b không thấy = giới hạn phát hiện;
       4 loại khởi động lệch lớn của mô hình 28/10 (τ 0,05/0,1/0,4 → 10, τ 0,4 → 30) phải bị phát hiện.
  (2)  EncoderGen hiện tại của cây làm việc (đối tượng đo): ca không đạt là lỗi sản phẩm.
Mỗi nhóm loại hồ sơ × bánh: "chưa kết luận" ≤ 10 %.

Giả định của mô hình (ghi vào summary): trễ áp frame d ~ U[d_lo; d_hi] µs (mặc định 50–900, < d_max
1 ms) [ĐỀ XUẤT, chưa đo: T3 đo trễ thật]; Pi gửi đều 1 ms, lệch ngẫu nhiên ±jitter µs; đọc count mỗi
1 ms, khoảng đọc span µs, thời điểm lấy mẫu đều trong khoảng đó; δ_sync = 0 (cùng đồng hồ) và thêm một
lần chấm lại với δ_sync khác 0 (chỉ báo). Harness không mô hình: chân thả nổi khi dừng ở code cũ, độ trễ
ngắt thật, timer chạy trong lúc `_retime` (giới hạn Z24).

Sửa theo Codex Sol vòng 1/2 (S29; Liam duyệt 30/10 "Duyệt, sửa hết"):
  (b) u_t ≤ ¼·(N_tol − phần cố định) của MỌI nhóm vào cổng (trước chỉ báo);
  (c) "lệch lớn quan sát được" gắn nhãn từ sự thật TRƯỚC khi đọc kết quả chấm: e_truth = Δcount của
      harness − Δcount của bộ phát lý tưởng ở cùng hai lần đọc, quan sát được khi |e_truth| > N_tol + u_t
      (`T3b_<tag>_truth.csv`); mọi ca quan sát được của code cũ phải không đạt, 4 loại lệch lớn phải có ca
      quan sát được (trước: chọn theo err_counts > 100 của chính bộ chấm);
  (a) luật ≤ 10 % "chưa kết luận" cho (2) đối tượng đo: chữ spec chưa rõ, hỏi THESIS 30/10; chưa vào cổng,
      vượt thì in dòng ghi chú.

Usage:
  python tools/t3b_sensitivity.py --out <thư mục> [--seed 1] [--old-rev cb91e7a]
Mã thoát: 0 khi cổng Z23 đạt (dòng cuối summary.txt); ngược lại 1. Kết quả (2) không ảnh hưởng mã thoát
(đối tượng đo, báo riêng), trừ u_t.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'rpi5'))
import t3b_schedule as ts  # noqa: E402
import t3b_score as sc  # noqa: E402

H7 = os.path.join(ROOT, 'stm32h7')
SIM = os.path.join(H7, 'pc_tests', 'encgen')
BIG_KINDS = ('exp_tau0.05_to10', 'exp_tau0.1_to10', 'exp_tau0.4_to10', 'exp_tau0.4_to30')
INCONCLUSIVE_MAX = 0.10


# ── input frames ──

def make_frames(sched, seed, jitter_us, d_lo_us, d_hi_us, tail_ms=50):
    rng = random.Random(seed + 1000)
    table = ts.omega_table(sched)
    n = len(table[0]) + tail_ms
    w_d = np.zeros((n, 4))
    for w in range(4):
        for k, v in enumerate(table[w]):
            w_d[k, w] = ts.quantize(v)
    t_send = np.array([k * 1000.0 + rng.uniform(-jitter_us, jitter_us) for k in range(n)])
    t_apply = t_send + np.array([rng.uniform(d_lo_us, d_hi_us) for _ in range(n)])
    return t_send, t_apply, w_d


def write_frames(path, t_send, t_apply, w_d):
    with open(path, 'w', newline='\n') as f:
        f.write(f'{len(t_send)}\n')
        for k in range(len(t_send)):
            f.write(f'{t_send[k] * 1000:.0f} {t_apply[k] * 1000:.0f} '
                    + ' '.join(f'{v:.9g}' for v in w_d[k]) + '\n')


# ── (0) ideal generator ──

def ideal_reads(t_apply, w_d, read_phase_us, read_span_us, seed, stress=False, read_period_us=1000.0):
    """Counts of an ideal generator at 1 ms reads, and its start/stop calls."""
    rng = random.Random(seed + 2000)
    weff = np.where(np.abs(w_d) <= ts.DEADBAND, 0.0, w_d)
    t_end = t_apply[-1] + 2 * read_period_us
    tb = np.arange(read_phase_us, t_end, read_period_us)
    t_s = tb + np.array([rng.random() * read_span_us for _ in tb])
    counts = np.zeros((len(tb), 4), dtype=np.int64)
    calls = {w: [] for w in range(4)}

    def g(phi):                      # declared start: first edge after hp = 2 counts
        return max(0, math.floor(phi + 1e-9) - 1)

    for w in range(4):
        base, phi, d, k, t_last = 0, 0.0, 0, 0, 0.0
        for i, t in enumerate(t_s):
            while k < len(t_apply) and t_apply[k] <= t:
                phi += abs(weff[k - 1, w]) * sc.K * (t_apply[k] - t_last) * 1e-6 if k > 0 and d != 0 else 0.0
                t_last = t_apply[k]
                nd = int(np.sign(weff[k, w]))
                if nd != d:
                    if d != 0:
                        base += d * g(phi)
                        if stress:
                            base -= 2 * d
                        calls[w].append(('P', t_apply[k], 0))
                    if nd != 0:
                        calls[w].append(('S', t_apply[k], 0))
                    phi, d = 0.0, nd
                k += 1
            cur = phi + (abs(weff[k - 1, w]) * sc.K * (t - t_last) * 1e-6 if k > 0 and d != 0 else 0.0)
            counts[i, w] = base + d * g(cur)
    return tb, tb + read_span_us, counts, calls


# ── (1), (2) harness ──

def build(tmp, src_c, tag, defines=()):
    d = os.path.join(tmp, tag)
    os.makedirs(d)
    for f in (os.path.join(H7, 'Core', 'Inc', 'encoder_gen.h'), os.path.join(SIM, 'stm32h7xx_hal.h'),
              os.path.join(SIM, 'encgen_t3b_main.c')):
        shutil.copy(f, d)
    shutil.copy(src_c, os.path.join(d, 'encoder_gen.c'))
    exe = os.path.join(d, 'encgen_t3b.exe')
    r = subprocess.run(['gcc', '-std=gnu11', '-O2', '-Wall', '-Werror', *defines, '-I', d,
                        os.path.join(d, 'encgen_t3b_main.c'), os.path.join(d, 'encoder_gen.c'),
                        '-lm', '-o', exe], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'build {tag}: {r.stderr}')
    return exe


def run_harness(exe, frames_txt, seed, read_phase_us, read_span_us):
    p = subprocess.run([exe, frames_txt, str(seed), f'{read_phase_us * 1000:.0f}', f'{read_span_us * 1000:.0f}'],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f'harness exit {p.returncode}: {p.stderr}')
    tb, ta, c, rv = [], [], [], []
    calls = {w: [] for w in range(4)}
    for line in p.stdout.splitlines():
        f = line.split()
        if f[0] == 'R':
            tb.append(float(f[1]) / 1000)
            ta.append(float(f[2]) / 1000)
            c.append([int(x) for x in f[3:7]])
            rv.append([int(x) for x in f[7:11]])
        elif f[0] == 'C':
            calls[int(f[1])].append((f[2], float(f[3]) / 1000, int(f[4])))
    return np.array(tb), np.array(ta), np.array(c), np.array(rv), calls


def truth_rev(rows, run, rv):
    """Per window: per-edge reverse steps of the harness between the two reads."""
    for x in rows:
        ia = int(np.searchsorted(run.tb, x['t_read_before_us'], side='left'))
        ib = int(np.searchsorted(run.tb, x['t_read_before_end_us'], side='left'))
        x['truth_rev'] = int(rv[ib, x['wheel']] - rv[ia, x['wheel']])


def truth_big(rows, ideal_rows, ideal_run, run):
    """S29 (c): label BEFORE looking at the verdicts which windows of a harness run deviate from the
    declared (ideal) generator by more than the T3b tolerance, from the truth: e_truth = Δcount of the
    harness − Δcount of the ideal generator at the same two read indices (same read grid). Label
    obs_big = |e_truth| > N_tol + u_t of that window (the threshold of the spec rule, not the scorer's
    integral estimate). None when the two runs do not share the read indices of that window."""
    by_id = {x['id']: x for x in ideal_rows}
    for x in rows:
        y = by_id[x['id']]
        ia = int(np.searchsorted(run.tb, x['t_read_before_us'], side='left'))
        ib = int(np.searchsorted(run.tb, x['t_read_before_end_us'], side='left'))
        ja = int(np.searchsorted(ideal_run.tb, y['t_read_before_us'], side='left'))
        jb = int(np.searchsorted(ideal_run.tb, y['t_read_before_end_us'], side='left'))
        if ia != ja or ib != jb:
            x['e_truth'], x['obs_big'] = None, None
            continue
        w = x['wheel']
        e = int((run.c[ib, w] - run.c[ia, w]) - (ideal_run.c[jb, w] - ideal_run.c[ja, w]))
        x['e_truth'], x['obs_big'] = e, abs(e) > x['n_tol'] + x['uncertainty']


def group_stats(rows):
    g = defaultdict(lambda: defaultdict(int))
    for x in rows:
        s = g[(x['transition'], x['wheel'])]
        s['n'] += 1
        s[x['verdict']] += 1
        s['u_t_bad'] += 0 if x['u_t_ok'] else 1
        if 'truth_rev' in x:
            bad = x['truth_rev'] > 0
            s['truth_bad'] += bad
            s['truth_bad_detected'] += bad and x['verdict'] == 'fail'
            big = bool(x.get('obs_big'))             # S29 (c): truth label, not the scorer's err_counts
            s['big'] += big
            s['big_detected'] += big and x['verdict'] == 'fail'
        s['max_abs_err'] = max(s['max_abs_err'], abs(x['err_counts']))
        s['max_err_over_ntol'] = max(s.get('max_err_over_ntol', 0.0), abs(x['err_counts']) / x['n_tol'])
    return g


def git(*a):
    return subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Kiểm độ nhạy T3b (S06 Z23) trên PC')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=1, help='seed lịch T3b (cũng là seed khóa trước của lượt thật)')
    ap.add_argument('--sim-seed', type=int, default=7, help='seed trễ áp frame, jitter, pha đọc, ngắt')
    ap.add_argument('--old-rev', default='cb91e7a')
    ap.add_argument('--jitter-us', type=float, default=20.0)
    ap.add_argument('--d-lo-us', type=float, default=50.0)
    ap.add_argument('--d-hi-us', type=float, default=900.0)
    ap.add_argument('--read-phase-us', type=float, default=310.0)
    ap.add_argument('--read-span-us', type=float, default=30.0)
    ap.add_argument('--delta-sync-alt-us', type=float, default=100.0)
    ap.add_argument('--max-ms', type=int, default=0, help='chỉ chạy các chuyển tiếp kết thúc trước mốc này (test nhanh)')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    sched = ts.make_schedule(a.seed)
    if a.max_ms:
        sched = [r for r in sched if r.slot_end_ms <= a.max_ms]
        for i, r in enumerate(sched):
            r.id = i
    t_send, t_apply, w_d = make_frames(sched, a.sim_seed, a.jitter_us, a.d_lo_us, a.d_hi_us)
    frames_txt = os.path.join(a.out, 'frames_input.txt')
    write_frames(frames_txt, t_send, t_apply, w_d)
    ts.write_csv(os.path.join(a.out, 'T3b_schedule.csv'), sched)

    results, runs = {}, {}
    tb, ta, c, calls = ideal_reads(t_apply, w_d, a.read_phase_us, a.read_span_us, a.sim_seed)
    runs['ideal'] = (sc.Run(t_send, w_d, tb, ta, c, calls=calls), None)
    tb, ta, c, calls = ideal_reads(t_apply, w_d, a.read_phase_us, a.read_span_us, a.sim_seed, stress=True)
    runs['ideal_stress'] = (sc.Run(t_send, w_d, tb, ta, c, calls=calls), None)

    with tempfile.TemporaryDirectory() as tmp:
        old_src = subprocess.run(['git', 'show', f'{a.old_rev}:stm32h7/Core/Src/encoder_gen.c'], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout
        p_old = os.path.join(tmp, 'encoder_gen_old.c')
        with open(p_old, 'w', newline='\n') as f:
            f.write(old_src)
        # old encoder_gen.c has no event hook: its calls are inferred from the dir change (harness)
        for tag, src, dfn in (('old', p_old, ['-DENCGEN_NO_HOOK']),
                              ('new', os.path.join(H7, 'Core', 'Src', 'encoder_gen.c'), [])):
            exe = build(tmp, src, tag, dfn)
            tb, ta, c, rv, calls = run_harness(exe, frames_txt, a.sim_seed, a.read_phase_us, a.read_span_us)
            runs[tag] = (sc.Run(t_send, w_d, tb, ta, c, calls=calls), rv)

    for tag, (run, rv) in runs.items():
        rows = sc.score_run(run, sched)
        if rv is not None:
            truth_rev(rows, run, rv)
        results[tag] = rows
        sc.write_rows(os.path.join(a.out, f'T3b_{tag}.csv'), rows)
    # S29 (c): truth labels of the harness runs against the ideal generator, written before the gate reads
    # any verdict
    for tag in ('old', 'new'):
        truth_big(results[tag], results['ideal'], runs['ideal'][0], runs[tag][0])
        with open(os.path.join(a.out, f'T3b_{tag}_truth.csv'), 'w', newline='') as f:
            f.write('id,truth_rev,e_truth,obs_big\n')
            f.writelines(f"{x['id']},{x['truth_rev']},{'' if x['e_truth'] is None else x['e_truth']},"
                         f"{'' if x['obs_big'] is None else int(x['obs_big'])}\n" for x in results[tag])
    # same reads, delta_sync != 0 (only u_t changes): inconclusive rate sensitivity
    alt = {}
    for tag in ('ideal', 'new'):
        run = runs[tag][0]
        run.delta_sync = a.delta_sync_alt_us
        alt[tag] = sc.score_run(run, sched)
        run.delta_sync = 0.0

    lines, ok = [], True
    sha, dirty = git('rev-parse', '--short', 'HEAD'), bool(git('status', '--porcelain'))
    lines.append(f'hil-framework {sha} dirty={dirty}; schedule seed {a.seed}, sim seed {a.sim_seed}, '
                 f'{len(sched)} transitions; old EncoderGen {a.old_rev}')
    lines.append(f'model: d ~ U[{a.d_lo_us}; {a.d_hi_us}] us, Pi jitter +-{a.jitter_us} us, read every 1 ms '
                 f'phase {a.read_phase_us} us span {a.read_span_us} us, delta_sync 0')
    for tag in ('ideal', 'ideal_stress', 'old', 'new'):
        rows = results[tag]
        tot = defaultdict(int)
        for x in rows:
            tot[x['verdict']] += 1
            tot['u_t_bad'] += 0 if x['u_t_ok'] else 1
        lines.append(f'\n== {tag}: pass {tot["pass"]}, fail {tot["fail"]}, inconclusive {tot["inconclusive"]}, '
                     f'u_t over limit {tot["u_t_bad"]} (of {len(rows)})')
        hdr = f'{"kind":22s} w   n pass fail inc  max|e|  max|e|/Ntol'
        if tag in ('old', 'new'):
            hdr += '  truth_bad det  big det'
        lines.append(hdr)
        for (kind, w), s in sorted(group_stats(rows).items()):
            ln = (f'{kind:22s} {w} {s["n"]:3d} {s["pass"]:4d} {s["fail"]:4d} {s["inconclusive"]:3d} '
                  f'{s["max_abs_err"]:7.1f} {s["max_err_over_ntol"]:12.2f}')
            if tag in ('old', 'new'):
                ln += f'  {s["truth_bad"]:9d} {s["truth_bad_detected"]:3d} {s["big"]:4d} {s["big_detected"]:3d}'
            lines.append(ln)
            if s['inconclusive'] > INCONCLUSIVE_MAX * s['n']:
                if tag != 'new':
                    ok = False
                    lines.append(f'   ^ chưa kết luận > {INCONCLUSIVE_MAX:.0%}')
                else:                                # S29 (a): spec wording unclear for (2), asked THESIS 30/10
                    lines.append(f'   ^ chưa kết luận > {INCONCLUSIVE_MAX:.0%} ở đối tượng đo (2): chữ spec chưa '
                                 f'rõ luật ≤ 10 % có áp cho (2) không, chờ THESIS; chưa tính vào cổng')
        if tot['u_t_bad']:                            # S29 (b): the tool requirement u_t ≤ ¼(N_tol − fixed)
            ok = False
            lines.append(f'   KHÔNG ĐẠT: {tot["u_t_bad"]} cửa sổ u_t vượt yêu cầu công cụ')
        if tag == 'ideal' and (tot['fail'] or tot['inconclusive']):
            ok = False
            lines.append('   KHÔNG ĐẠT: bộ phát lý tưởng phải đạt mọi ca')
    old = results['old']
    lines.append('\n== (1) old code, 4 large-deviation start-up kinds of the 28/10 model')
    for kind in BIG_KINDS:
        rk = [x for x in old if x['transition'] == kind]
        big = [x for x in rk if x['obs_big']]             # S29 (c): labelled from the truth, before the gate
        det = [x for x in big if x['verdict'] == 'fail']
        tb_ = [x for x in rk if x['truth_rev'] > 0]
        lines.append(f'{kind:22s}: {len(rk)} windows, truth reverse {len(tb_)} (detected '
                     f'{sum(1 for x in tb_ if x["verdict"] == "fail")}), truth |e| > N_tol + u_t: {len(big)} '
                     f'(detected {len(det)})')
        if not big or len(det) < len(big):
            ok = False
            lines.append('   ^ KHÔNG ĐẠT: loại lệch lớn phải xuất hiện và bị phát hiện')
    # S29 (c): every old-code window whose truth deviation is observable by the T3b rule must fail
    obs_miss = [x for x in old if x['obs_big'] and x['verdict'] != 'fail']
    unlabelled = sum(1 for x in old if x['obs_big'] is None)
    lines.append(f'(1) observable by the rule (truth |e| > N_tol + u_t): {sum(1 for x in old if x["obs_big"])} '
                 f'windows, not failed {len(obs_miss)}; not labelled (read grid differs) {unlabelled}')
    if obs_miss or unlabelled:
        ok = False
        lines.append('   ^ KHÔNG ĐẠT: ca quan sát được phải báo không đạt; mọi ca phải gắn nhãn được')
    miss = [x for x in old if x['truth_rev'] > 0 and x['verdict'] != 'fail']
    lines.append(f'(1) detection limit: {len(miss)} old-code windows with per-edge reverse steps not failed by T3b'
                 + (': ' + ', '.join(f"{x['id']}({x['transition']},w{x['wheel']},rev {x['truth_rev']},"
                                      f"e {x['err_counts']:.1f}/Ntol {x['n_tol']:.1f})" for x in miss[:20]) if miss else ''))
    for tag, rows in alt.items():
        n_inc = sum(1 for x in rows if x['verdict'] == 'inconclusive')
        n_ut = sum(1 for x in rows if not x['u_t_ok'])
        lines.append(f'(delta_sync {a.delta_sync_alt_us} us, info) {tag}: inconclusive {n_inc}, '
                     f'u_t over limit {n_ut} of {len(rows)}')
    lines.append(f'\nZ23 gate ((0) all pass, (1) every window observable from the truth failed incl. the 4 large '
                 f'kinds, inconclusive <= 10% in (0)/(1), u_t within the tool requirement in every group): '
                 f'{"PASS" if ok else "FAIL"}')
    txt = '\n'.join(lines)
    with open(os.path.join(a.out, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    with open(os.path.join(a.out, 'args.json'), 'w') as f:
        json.dump(vars(a) | {'hil_framework': sha, 'dirty': dirty}, f, indent=1)
    print(txt)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
