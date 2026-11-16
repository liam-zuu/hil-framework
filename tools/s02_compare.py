"""
s02_compare.py: spec S02 P2-6 comparison of platforms: delta, 95 % CI per cell, n per side, delta %, label "below
encoder resolution", from the metrics.csv of several run folders (SIM, HIL, car).

Design: 03_CODE/docs/2026-11-15_Thiet_ke_CODE_S02_v1.md (repo luan van) section 22 (a); Liam 15/11 CODE_S02
local_0eb4085a, "CODE_S02 lam (Recommended)"; spec S02 v1.13 P2-6, Viec CODE 10.

    python tools/s02_compare.py --out <folder> <run_dir_or_metrics.csv> [<run_dir_or_metrics.csv> ...]
    -> <folder>/compare_<YYYY-MM-DD>.csv (never over an existing file); last line '#ket_qua,ma=0 | 2'

Input: lines of metrics.csv (tools/s02_metrics.run_dir_metrics) with status ok only (a run not completed has a
status line and no metric, P2-5); the metric columns are those of <stem>_columns.json with kind 'metric' (family E
or R), else s02_metrics.column_info. Controller names: ADRC (fwsim) is LADRC (the same controller, adrc.c).
Pairs (platform_a = reference, platform_b): SIM -> HIL (delta_HIL = m_HIL - m_SIM), HIL -> REAL, SIM -> REAL, for
each (test, controller, metric) present on both sides.
  n_a, n_b: completed runs with a finite value of that metric (n_metric); n_undef_a, n_undef_b: completed runs
  where it is undefined (e.g. no 0.9 crossing: no rise time). Fewer than MIN_N on a side: label insufficient, no
  number (P2-6 "khong du gia tri").
  Test (P2-6 Kiem dinh): when side a is SIM and every value of it is the same number (SIM without a noise model: the
  runs repeat bit for bit), a one-sample t interval of side b around that number (method one_sample); otherwise
  Welch (method welch, Welch-Satterthwaite degrees of freedom; S2M1-11: a constant HIL side is Welch too).
  Inputs: a file given twice, or one (platform, run_id) in two files, is refused (S2M1-7). CI = delta +- t_{0.975, df} se. The SIM determinism check itself
  (Viec CODE 7) is not done here.
  delta_pct = delta / mean_a x 100, only for metrics that cannot be negative (columns info nonneg), and not when
  mean_a = 0 or, for an encoder metric, mean_a < 2 qrms.
  label below_encoder_resolution: an encoder metric (e_vx, e_vy, rms_w_err; e_wz when every run of both sides used
  the encoder source, wz_src_enc_frac = 1) whose whole CI lies within +-qrms (P2-6, descriptive only).
No conclusion is written: the sign of delta is read only where the CI excludes 0 (P2-6), by the analysis.
"""
import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np
from scipy import stats as sst

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import s02_metrics  # noqa: E402

MIN_N = 3
PAIRS = (('sim', 'hil'), ('hil', 'real'), ('sim', 'real'))
CONTROLLER_ALIAS = {'ADRC': 'LADRC'}
ENCODER_METRICS = ('e_vx', 'e_vy', 'rms_w_err')          # e_wz too when its source was the encoder
OUT_COLUMNS = ['test_id', 'controller', 'metric', 'family', 'unit', 'platform_a', 'platform_b', 'mean_a', 'mean_b',
               'n_a', 'n_b', 'n_undef_a', 'n_undef_b', 'method', 'df', 'delta', 'ci_low', 'ci_high', 'ci_width',
               'delta_pct', 'label']


def _metrics_path(p):
    return os.path.join(p, 'metrics.csv') if os.path.isdir(p) else p


def load(paths):
    """(rows with status ok, column info) of several metrics.csv; rows keep their source file."""
    rows, info = [], {}
    # S2M1-7 (Codex round 1 of v3): one file given twice (as a folder and as its metrics.csv too), or one run in two
    # files, would count its runs twice: refused
    seen_path, seen_run = {}, {}
    for p in paths:
        mp = _metrics_path(p)
        key = os.path.normcase(os.path.realpath(mp))
        if key in seen_path:
            raise ValueError(f'{mp}: given twice (also as {seen_path[key]})')
        seen_path[key] = mp
        with open(mp, newline='') as f:
            got = list(csv.DictReader(f))
        for r in got:
            rk = ((r.get('platform') or '').strip().lower(), (r.get('run_id') or '').strip())
            if rk in seen_run and seen_run[rk] != mp:
                raise ValueError(f'{mp}: run {rk[1]!r} ({rk[0]}) is also in {seen_run[rk]}')
            seen_run.setdefault(rk, mp)
        cj = os.path.splitext(mp)[0] + '_columns.json'
        if os.path.isfile(cj):
            with open(cj, encoding='utf-8') as f:
                info.update(json.load(f).get('columns', {}))
        for r in got:
            if 'status' not in r:
                raise ValueError(f'{mp}: no status column (metrics.csv older than s02_metrics v3)')
            if r['status'].strip() != 'ok':
                continue
            r = dict(r, _src=mp)
            r['platform'] = r['platform'].strip().lower()
            c = r['controller'].strip()
            r['controller'] = CONTROLLER_ALIAS.get(c.upper(), c)
            rows.append(r)
    return rows, info


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def metric_columns(rows, info):
    """Metric columns (kind metric, family E or R) present in the rows, in first-seen order."""
    seen = []
    for r in rows:
        for k in r:
            if k not in seen and not k.startswith('_'):
                seen.append(k)
    out = []
    for k in seen:
        i = info.get(k) or s02_metrics.column_info(k)
        if i.get('kind') == 'metric' and i.get('family') in ('E', 'R'):
            out.append(k)
    return out


def interval(a, b, a_is_sim=True):
    """(method, df, delta, ci_low, ci_high) of side b against side a (module docstring). One-sample only when side a
    is SIM (S2M1-11, Codex round 1 of v3: the method follows the platform, not the data; a constant HIL or car side
    goes through Welch, whose interval is then the same, labelled welch)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_is_sim and np.all(a == a[0]):
        d = float(np.mean(b) - a[0])
        se = float(np.std(b, ddof=1) / math.sqrt(len(b)))
        df = len(b) - 1
        method = 'one_sample'
    else:
        d = float(np.mean(b) - np.mean(a))
        va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
        se = float(math.sqrt(va + vb))
        df = float((va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))) if va + vb > 0 else math.inf
        method = 'welch'
    h = float(sst.t.ppf(0.975, df) * se) if se > 0 else 0.0
    return method, df, d, d - h, d + h


def _qrms(metric, ra, rb):
    """qrms of an encoder metric for this cell, or None (e_wz only when every run used the encoder source)."""
    if metric in ENCODER_METRICS:
        return s02_metrics.QRMS[metric]
    if metric == 'e_wz' and all(_num(r.get('wz_src_enc_frac')) == 1.0 for r in ra + rb):
        return s02_metrics.QRMS['e_wz']
    return None


def compare(rows, info):
    """Lines of compare_<date>.csv (module docstring)."""
    out = []
    metrics = metric_columns(rows, info)
    cells = sorted({(r['test_id'], r['controller']) for r in rows})
    for pa, pb in PAIRS:
        for tid, ctrl in cells:
            ra = [r for r in rows if r['platform'] == pa and r['test_id'] == tid and r['controller'] == ctrl]
            rb = [r for r in rows if r['platform'] == pb and r['test_id'] == tid and r['controller'] == ctrl]
            if not ra or not rb:
                continue
            for k in metrics:
                va = [_num(r.get(k)) for r in ra]
                vb = [_num(r.get(k)) for r in rb]
                if all(r.get(k, '') == '' for r in ra + rb):
                    continue                                   # metric not of this test (e.g. family R on A-K)
                fa, fb = [x for x in va if math.isfinite(x)], [x for x in vb if math.isfinite(x)]
                i = info.get(k) or s02_metrics.column_info(k)
                line = dict(test_id=tid, controller=ctrl, metric=k, family=i.get('family', ''), unit=i.get('unit', ''),
                            platform_a=pa, platform_b=pb, n_a=len(fa), n_b=len(fb),
                            n_undef_a=len(va) - len(fa), n_undef_b=len(vb) - len(fb),
                            mean_a=float(np.mean(fa)) if fa else math.nan,
                            mean_b=float(np.mean(fb)) if fb else math.nan)
                if len(fa) < MIN_N or len(fb) < MIN_N:
                    line.update(label='insufficient')
                    out.append(line)
                    continue
                method, df, d, lo, hi = interval(fa, fb, pa == 'sim')
                line.update(method=method, df=df, delta=d, ci_low=lo, ci_high=hi, ci_width=hi - lo, label='')
                q = _qrms(k, ra, rb)
                ma = line['mean_a']
                if i.get('nonneg') and ma != 0 and not (q is not None and ma < 2 * q):
                    line['delta_pct'] = d / ma * 100.0
                if q is not None and -q <= lo and hi <= q:
                    line['label'] = 'below_encoder_resolution'
                out.append(line)
    return out


def write(lines, path):
    def fmt(x):
        return f'{x:.6g}' if isinstance(x, float) else x
    with open(path, 'x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        w.writerows({k: fmt(o.get(k, '')) for k in OUT_COLUMNS} for o in lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='folder for compare_<date>.csv')
    ap.add_argument('--date', default=datetime.date.today().isoformat())
    ap.add_argument('inputs', nargs='+', help='run folders (metrics.csv inside) or metrics.csv files')
    a = ap.parse_args(argv)
    path = os.path.join(a.out, f'compare_{a.date}.csv')
    if os.path.exists(path):
        print(f'{path} already exists: outputs are not overwritten')
        print('#ket_qua,ma=2')
        return 2
    try:
        rows, info = load(a.inputs)
        lines = compare(rows, info)
    except (OSError, ValueError, KeyError) as exc:
        print(f'error: {exc}')
        print('#ket_qua,ma=2')
        return 2
    os.makedirs(a.out, exist_ok=True)
    write(lines, path)
    n_ok = sum(1 for x in lines if x.get('label') != 'insufficient')
    print(f'{len(lines)} lines ({n_ok} with an interval, {len(lines) - n_ok} insufficient) -> {path}')
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
