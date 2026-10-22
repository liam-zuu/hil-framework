"""
s02_holdout.py: spec S02 part P1, out-of-sample check of the DC plant on S01 run01.

The same extraction code runs on the measurement (S01 run01 raw/) and on the
plant replay of the same plan (S01_PLANT run01 raw/, plant frozen at e18d217):
  H1  brake (PWM 0, EN 1) from 400 / 700 / 1000: tau_brake = time to 36.8 % of
      the speed at the brake command (63 % of the drop to rest), per level
  H2  reversal +600 -> -600 -> +600, per transition: t0 (command to omega
      crossing 0), t63 (63 % of the change to the new steady speed),
      overshoot beyond the new steady speed [%]
  H3  dead-zone extrapolation: |omega_ss| = s * (u - u_dz) fitted on levels
      >= 400 only (per wheel x direction), predicted at 125 / 150 / 200. Two rows
      per level (since 22/10, review R07): 'turns' (turning = |omega| > q_omega,
      same threshold for the prediction and for the measurement, majority of
      reps) and 'omega_ss' (speed, k * s criterion). H3 tests the linear
      extrapolation from PWM >= 400, not the whole DC plant.

Acceptance (spec S02 P1-3, fixed before the results): err = |model - mean(meas)|
passes when err <= k * s_used, k = 3.04, s_used = max(sd of the 5 reps, q / sqrt(12)).
q: 1 ms for times; for speeds the omega resolution 2*pi/1320/0.010 = 0.476 rad/s
(omega_rad_s is a 10 ms difference of counts, s01_openloop_runner.omega_series);
for the overshoot that resolution as % of the new steady speed.

  python tools/s02_holdout.py --out D:\\CLAUDE_THESIS\\05_DATA\\results\\S02\\2026-10-21_run01
"""
import argparse
import csv
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', '..', '..', '05_DATA', 'results')
MEAS = os.path.join(RESULTS, 'S01', '2026-10-20_run01')
MODEL = os.path.join(RESULTS, 'S01_PLANT', '2026-10-21_run01')
K = 3.04
Q_T = 0.001
Q_W = 2 * math.pi / 1320 / 0.010
H3_FIT_MIN = 400
H3_LEVELS = (125, 150, 200)
TURN_THRESH = Q_W             # |omega_ss| above one count per 10 ms window = turning


def load_runs(d):
    runs = []
    with open(os.path.join(d, 'runs.csv'), newline='') as f:
        for r in csv.DictReader(f):
            if r['aborted'] not in ('0', 'False', 'false', ''):
                continue
            path = os.path.join(d, r['file'].replace('\\', os.sep))
            a = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=(0, 2, 3, 5))
            runs.append(dict(wheel=r['wheel'], scenario=r['scenario'], level=int(r['pwm_level']),
                             direction=r['direction'], rep=int(r['rep']), run_id=r['run_id'],
                             t=a[:, 0] * 1e-6, pwm=a[:, 1], w=a[:, 3]))
    return runs


def first(cond):
    k = np.flatnonzero(cond)
    return int(k[0]) if k.size else None


def brake_tau(r):
    t, pwm, w = r['t'], r['pwm'], r['w']
    ib = first((pwm == 0) & (np.arange(len(pwm)) > 0) & (np.roll(pwm, 1) != 0))
    w0 = np.mean(w[ib - 100:ib])
    k = first((np.arange(len(w)) >= ib) & (np.sign(w0) * w <= 0.368 * abs(w0)))
    return dict(tau_brake=t[k] - t[ib] if k is not None else np.nan)


def reversal(r):
    t, pwm, w = r['t'], r['pwm'], r['w']
    idx = np.arange(len(pwm))
    out = {}
    switches = np.flatnonzero((np.sign(pwm[1:]) != np.sign(pwm[:-1])) & (pwm[1:] != 0) & (pwm[:-1] != 0)) + 1
    for n, i_s in enumerate(switches[:2], start=1):
        seg_end = first((idx > i_s) & (pwm != pwm[i_s]))
        seg_end = seg_end if seg_end is not None else len(pwm)
        w_old = np.mean(w[i_s - 100:i_s])
        w_new = np.mean(w[seg_end - 300:seg_end])
        sg = np.sign(w_new)
        rng = (idx >= i_s) & (idx < seg_end)
        k0 = first(rng & (sg * w > 0))
        k63 = first(rng & (sg * (w - w_old) >= 0.632 * abs(w_new - w_old)))
        peak = np.max(sg * w[rng])
        out[n] = dict(t0=t[k0] - t[i_s] if k0 is not None else np.nan,
                      t63=t[k63] - t[i_s] if k63 is not None else np.nan,
                      overshoot_pct=(peak - abs(w_new)) / abs(w_new) * 100, w_new=abs(w_new))
    return out


def steady_omega(r):
    pwm, w = r['pwm'], r['w']
    hold = np.flatnonzero(pwm == r['level'] * (1 if r['direction'] == 'pos' else -1))
    if hold.size == 0:
        hold = np.flatnonzero(pwm != 0)
    end = hold[-1] + 1
    return abs(np.mean(w[end - 500:end]))


def stats(v):
    v = np.asarray([x for x in v if not np.isnan(x)])
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if v.size > 1 else np.nan, int(v.size))


def row(test, wheel, direction, level, transition, quantity, unit, meas, model, q):
    m, sd, n = stats(meas)
    mod = float(np.nanmean(model))
    s_used = max(sd, q / math.sqrt(12)) if not np.isnan(sd) else q / math.sqrt(12)
    err = abs(mod - m)
    return dict(test=test, wheel=wheel, direction=direction, pwm_level=level, transition=transition,
                quantity=quantity, unit=unit, meas_mean=m, meas_sd=sd, n=n, model=mod, err=err, k=K,
                s_used=s_used, err_over_s=err / s_used, pass_=bool(err <= K * s_used))


def git_state(repo):
    def git(*a):
        return subprocess.run(['git', '-C', repo] + list(a), capture_output=True, text=True).stdout.strip()
    changed = [l[3:] for l in git('status', '--porcelain').splitlines() if l[3:] and not l[3:].startswith('.vscode/')]
    return dict(commit=git('rev-parse', '--short', 'HEAD'), dirty=bool(changed), modified=changed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--notes', default='', help='why this run exists (written to meta.yaml)')
    a = ap.parse_args()
    if os.path.isdir(a.out) and os.listdir(a.out):
        sys.exit(f'{a.out} already holds data: use a new folder.')
    os.makedirs(a.out, exist_ok=True)

    meas, model = load_runs(MEAS), load_runs(MODEL)
    wheels = ['FL', 'FR', 'RL', 'RR']
    sel = lambda runs, **kw: [r for r in runs if all(r[k] == v for k, v in kw.items())]   # noqa: E731
    rows = []

    # H1
    for w in wheels:
        for lv in (400, 700, 1000):
            mv = [brake_tau(r)['tau_brake'] for r in sel(meas, wheel=w, scenario='brake', level=lv)]
            pv = [brake_tau(r)['tau_brake'] for r in sel(model, wheel=w, scenario='brake', level=lv)]
            rows.append(row('H1', w, 'pos', lv, '', 'tau_brake', 's', mv, pv, Q_T))
    # H2
    for w in wheels:
        mr = [reversal(r) for r in sel(meas, wheel=w, scenario='rev')]
        pr = [reversal(r) for r in sel(model, wheel=w, scenario='rev')]
        for n, label in ((1, '+600->-600'), (2, '-600->+600')):
            w_new = np.mean([x[n]['w_new'] for x in mr])
            for qty, unit, q in (('t0', 's', Q_T), ('t63', 's', Q_T), ('overshoot_pct', '%', Q_W / w_new * 100)):
                rows.append(row('H2', w, 'pos', 600, label, qty, unit,
                                [x[n][qty] for x in mr], [x[n][qty] for x in pr], q))
    # H3
    fits = []
    for w in wheels:
        for d in ('pos', 'neg'):
            fit_runs = [r for r in sel(meas, wheel=w, scenario='step', direction=d) if r['level'] >= H3_FIT_MIN]
            u = np.array([r['level'] / 1000 for r in fit_runs])
            y = np.array([steady_omega(r) for r in fit_runs])
            slope, icpt = np.polyfit(u, y, 1)
            u_dz = -icpt / slope
            fits.append(dict(wheel=w, direction=d, slope_rad_s_per_duty=slope, u_dz=u_dz, n_runs=len(u)))
            for lv in H3_LEVELS:
                mv = [steady_omega(r) for r in sel(meas, wheel=w, scenario='step', direction=d, level=lv)]
                pred = max(slope * (lv / 1000 - u_dz), 0.0)
                # Same rule for measurement and model (R07, 22/10): turning = |omega| > TURN_THRESH,
                # the measurement in a majority of reps. Two rows per level: state and speed.
                turning = [x > TURN_THRESH for x in mv]
                meas_turns = sum(turning) > len(turning) / 2
                model_turns = pred > TURN_THRESH
                rows.append(dict(test='H3', wheel=w, direction=d, pwm_level=lv, transition='', quantity='turns',
                                 unit='fraction of reps', meas_mean=float(np.mean(turning)), meas_sd=np.nan,
                                 n=len(turning), model=float(model_turns), err=np.nan, k=K, s_used=np.nan,
                                 err_over_s=np.nan, pass_=bool(meas_turns == model_turns),
                                 note=f'predicted omega {pred:.3g} rad/s; mean measured {np.mean(mv):.3g} rad/s; '
                                      f'threshold {TURN_THRESH:.3f} rad/s'))
                rows.append(row('H3', w, d, lv, '', 'omega_ss', 'rad/s', mv, [pred], Q_W))

    cols = ['test', 'wheel', 'direction', 'pwm_level', 'transition', 'quantity', 'unit', 'meas_mean', 'meas_sd',
            'n', 'model', 'err', 'k', 's_used', 'err_over_s', 'pass', 'note']
    fmt = lambda v: f'{v:.6g}' if isinstance(v, float) else v   # noqa: E731
    with open(os.path.join(a.out, 'holdout_metrics.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for r in rows:
            wr.writerow([fmt(r.get('pass_' if c == 'pass' else c, '')) for c in cols])
    with open(os.path.join(a.out, 'h3_fit_ge400.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(fits[0]))
        wr.writeheader()
        wr.writerows({k: fmt(v) for k, v in x.items()} for x in fits)

    lines = [f'S02 P1 hold-out, k = {K}, s_used = max(sd, q/sqrt(12)); q_t = 1 ms, q_w = {Q_W:.3f} rad/s', '']
    for test in ('H1', 'H2', 'H3'):
        rs = [r for r in rows if r['test'] == test]
        lines.append(f'{test}: {sum(r["pass_"] for r in rs)}/{len(rs)} pass')
        for qty in sorted({r['quantity'] for r in rs}):
            rq = [r for r in rs if r['quantity'] == qty]
            rel = [100 * (r['model'] - r['meas_mean']) / r['meas_mean'] for r in rq
                   if qty != 'turns' and not np.isnan(r['meas_mean'])
                   and abs(r['meas_mean']) > (TURN_THRESH if qty == 'omega_ss' else 0)]   # % of ~0 is meaningless
            e_s = [r['err_over_s'] for r in rq if not np.isnan(r['err_over_s'])]
            extra = (f', model - meas {min(rel):+.0f}..{max(rel):+.0f} %, err/s {min(e_s):.1f}..{max(e_s):.1f}'
                     if rel and qty != 'overshoot_pct' else '')
            if qty == 'overshoot_pct':
                d = [r['model'] - r['meas_mean'] for r in rq]
                extra = f', model - meas {min(d):+.1f}..{max(d):+.1f} percentage points, err/s {min(e_s):.1f}..{max(e_s):.1f}'
            lines.append(f'  {qty}: {sum(r["pass_"] for r in rq)}/{len(rq)} pass{extra}')
    lines += ['', 'Pass/fail per quantity x wheel x level: holdout_metrics.csv. Not aggregated into one score (spec P1-3).']
    open(os.path.join(a.out, 'holdout_summary.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    if not a.no_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for w in wheels:
            fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
            for ax, kw, title in ((axs[0], dict(scenario='brake', level=1000), 'H1 brake from 1000'),
                                  (axs[1], dict(scenario='rev'), 'H2 reversal +600/-600/+600')):
                for i, r in enumerate(sel(meas, wheel=w, **kw)):
                    ax.plot(r['t'], r['w'], color='tab:blue', lw=0.6, alpha=0.6, label='measured (5 reps)' if i == 0 else None)
                pr = sel(model, wheel=w, **kw)
                if pr:
                    ax.plot(pr[0]['t'], pr[0]['w'], color='tab:red', lw=1.2, label='plant replay')
                ax.set_title(f'{w}: {title}'); ax.set_xlabel('t (s)'); ax.set_ylabel('omega (rad/s)')
                ax.grid(alpha=0.3); ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(a.out, f'fig_holdout_{w}.png'), dpi=120)
            plt.close(fig)

    hf = git_state(os.path.join(HERE, '..'))
    with open(os.path.join(a.out, 'meta.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"""spec: S02
part: P1
run: {a.run}
date: {time.strftime('%Y-%m-%d')}
setup: analysis_holdout          # re-analysis of existing data, no new measurement
operator: Claude (phiên CODE)
sources:
  measured: 05_DATA/results/S01/2026-10-20_run01 (ESP32 firmware 61fce0f, 5 reps)
  model: 05_DATA/results/S01_PLANT/2026-10-21_run01 (plant rpi5/plant_step.py frozen at hil-framework e18d217)
script: {{repo: hil-framework, path: tools/s02_holdout.py, commit: {hf['commit']}, dirty: {str(hf['dirty']).lower()}, modified: {hf['modified']}}}
acceptance: "err = |model - mean(meas)| <= k * max(sd, q/sqrt(12)), k = {K} (spec S02 P1-3, Liam 21/10)"
resolution: {{q_time_s: {Q_T}, q_omega_rad_s: {Q_W:.4f}, note: "omega = 10 ms count difference; overshoot q = q_omega / new steady speed"}}
definitions:
  H1: "tau_brake: time from brake command to |omega| <= 0.368 |omega at command| (mean of last 100 ms before)"
  H2: "per transition: t0 = to omega sign of new steady speed; t63 = to 63 % of change; overshoot = (peak - |w_new|)/|w_new|; w_new = mean of last 300 ms of the 2 s segment"
  H3: "|omega_ss| (mean last 0.5 s of hold) = s (u - u_dz) fitted on levels >= {H3_FIT_MIN} only; predicted at {list(H3_LEVELS)}; two rows per level: turns (turning = |omega| > {Q_W:.3f} rad/s, same threshold for prediction and measurement, measurement by majority of reps) and omega_ss (k * s). Tests the linear extrapolation, not the whole DC plant"
notes: "{a.notes}"
files:
  - holdout_metrics.csv: "one row per quantity x wheel x level (x transition), columns of spec P1-4 plus err_over_s, note"
  - holdout_summary.txt: "pass counts per test and quantity, range of model - measured"
  - h3_fit_ge400.csv: "slope and dead zone fitted on levels >= 400"
  - fig_holdout_<wheel>.png: "measured (5 reps) vs plant replay, brake from 1000 and reversal"
""")
    print(f'Saved to {a.out}')


if __name__ == '__main__':
    main()
