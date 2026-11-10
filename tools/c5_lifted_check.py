#!/usr/bin/env python3
"""
c5_lifted_check.py: functional check of the controller with the IMU on a lifted-wheel test
(S05 section 8.1 C5; Outline QD 24), one WiFi log of tools/agv_logger.py, the same for the HIL
bench and for the vehicle with raised wheels. Criteria: 03_CODE/docs/
2026-11-10_Chay_thu_tren_giay_Kiem_xe_ke_banh.md section 4 (Liam 10/11 CODE local_69fa8ade
"Dung bo nay (Recommended)"; script approved the same day, "Duyet ca ba"). Functional check,
not a thesis number.

  python tools/c5_lifted_check.py --log <run>.csv --test A-K1 --expect-sha 54b7269 \
      --expect-env esp32-s3-devkitc-1 [--platform car|hil] [--json <out>.json]

Reads <run>.csv, <run>.csv.summary.json and <run>.csv.events.txt (agv_logger outputs). Checks:
  fw       every '#fw' line: sha = --expect-sha (prefix either way), dirty=0, env = --expect-env
  logger   summary closed, one segment, no seq hole / duplicate / conflict / reordered line,
           one '#run_end' group with no footer and no tail line missing, log_qdrop unchanged
  motor    motor_on rises once and falls once, the fall after the last non-zero setpoint
  imu      imu_init = 0 on every line; wz_src = 1 on >= 99 % of motor-on lines from T_settle
           after the rise
  slip     no WRONG slip flag (Liam 10/11 CODE local_796cfeea "cham khong bat nham", replacing
           "slip_flag = 0 on every motor-on line"): the body of a lifted-wheel test does not turn,
           so the true IMU-encoder inconsistency is r_true = wz_enc - 0 = the log column wz;
           each rise of slip_flag during motor-on is RIGHT when mean |wz| over the T_slip before
           it is >= r_th / 2 (spec controller-IMU v2.6 section 6.1 classification), else WRONG.
           A free-spinning lifted car whose left and right wheels turn at different speeds
           raises the flag rightly (CAR_LIFTED/2026-11-10_run01 pid_k1a_r2: wz_enc -0.107 rad/s)
  pos      no pose reset in the middle of the run (tools/run_validity.pos_reset_mid = 0, L37)
  track    each hold phase of the schedule with a non-zero vx setpoint (>= 1.5 s): over its last
           1 s, |mean vx (encoder) - setpoint| <= 0.015 m/s (5 % of 0.3 m/s), the mean rpm of every
           wheel has the sign of the setpoint, no sat_* counter (cumulative ticks at the PWM
           limit) grows over the phase
  stop     recorded, not scored: the most negative / positive wheel rpm within 1.5 s after each
           setpoint step to 0 (HIL 08/11: reverse spin to -23 rpm PID, -29 rpm LADRC)
Safety (SWA off -> motor off, transmitter off -> failsafe) is checked by the operator at the
session (RC_FAILSAFE/2026-11-10_run01), not from the log.
Last line: '#ket_qua,ma=<0 pass | 1 fail | 2 input error>,...'.
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'rpi5'))
from run_validity import pos_reset_mid  # noqa: E402

SCORED_TESTS = ('A-K1', 'A-K2')        # section 4: A-K1 x 2 + A-K2 x 1 per controller and platform
T_SETTLE_S = 0.5                       # spec controller-IMU v2.6 section 3 table, T_settle [DE XUAT]
WZ_SRC_MIN = 0.99
VX_TOL = 0.015                         # 5 % of 0.3 m/s
HOLD_MIN_S = 1.5
LAST_S = 1.0
STOP_WIN_S = 1.5
WHEELS = ('fl', 'fr', 'rl', 'rr')
# Slip flag of firmware AGV 54b7269 (imu_fusion.h IMU_FUSION_PARAMS_DEFAULT): r_th = k_r * sqrt(sigma_gz^2 + q_wz^2)
# with k_r 4, sigma_gz 0 (bias estimation off), q_wz 0.0196 rad/s; T_slip 200 ms. Change with the firmware.
SLIP_R_TH = 4.0 * 0.0196
SLIP_T_S = 0.2
FW_RE = re.compile(r'#fw,sha=([0-9a-fA-F]+|unknown),dirty=(\d),env=(\S+)')


class InputError(Exception):
    pass


def load(path):
    if not os.path.isfile(path):
        raise InputError(f'no log {path}')
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise InputError(f'{path}: no data line')
    def col(c):
        if c not in rows[0]:
            raise InputError(f'{path}: no column {c}')
        return np.array([float(r[c]) if r[c] not in ('', None) else np.nan for r in rows])
    summ = path + '.summary.json'
    events = path + '.events.txt'
    if not os.path.isfile(summ) or not os.path.isfile(events):
        raise InputError(f'{path}: summary or events file missing')
    return col, json.load(open(summ, encoding='utf-8')), open(events, encoding='utf-8').read().splitlines()


def expected_holds(test_id):
    """[(t0, t1, vx)] of the schedule's hold phases with non-zero vx, from trajectory_pub."""
    from trajectory_pub import GROUP_A, get_body_velocity
    spec = GROUP_A[test_id]
    ts = np.arange(0.0, spec.duration, 0.01)
    vx = np.array([round(get_body_velocity(t, spec)[0], 6) for t in ts])
    out, i = [], 0
    while i < len(ts):
        j = i
        while j + 1 < len(ts) and vx[j + 1] == vx[i]:
            j += 1
        if vx[i] != 0.0 and ts[j] - ts[i] + 0.01 >= HOLD_MIN_S:
            out.append((float(ts[i]), float(ts[j] + 0.01), float(vx[i])))
        i = j + 1
    return out


def phases_in_log(t, sp):
    """[(i0, i1, value)] index ranges (i1 exclusive) of constant setpoint in the log."""
    out, i = [], 0
    n = len(sp)
    while i < n:
        j = i
        while j + 1 < n and sp[j + 1] == sp[i]:
            j += 1
        out.append((i, j + 1, float(sp[i])))
        i = j + 1
    return out


def check(col, summ, events, test_id, expect_sha, expect_env, t_settle=T_SETTLE_S):
    res = []          # (name, ok, detail)

    def add(name, ok, detail):
        res.append((name, bool(ok), detail))

    # fw
    fws = [m.groups() for m in (FW_RE.search(e) for e in events) if m]
    e = expect_sha.lower()
    bad = [f for f in fws if f[0] == 'unknown' or not (f[0].lower().startswith(e) or e.startswith(f[0].lower()))
           or f[1] != '0' or f[2] != expect_env]
    add('fw', fws and not bad, f'{len(fws)} #fw lines' + (f', wrong: {bad[:2]}' if bad else f', {fws[0]}' if fws else ''))

    # logger
    segs = summ.get('segments', [])
    probs = []
    if summ.get('status') != 'closed':
        probs.append(f"status {summ.get('status')}")
    if len(segs) != 1:
        probs.append(f'{len(segs)} segments')
    for s in segs:
        for k in ('seq_holes', 'duplicates', 'conflicts', 'reordered'):
            if s.get(k, 0):
                probs.append(f'{k} {s[k]}')
        g = s.get('run_end_groups', [])
        if len(g) != 1:
            probs.append(f'{len(g)} #run_end groups')
        for x in g:
            if x.get('footers_missing') or x.get('tail_data_missing'):
                probs.append(f'#run_end footers_missing {x.get("footers_missing")} tail_data_missing {x.get("tail_data_missing")}')
        qd = s.get('counters', {}).get('log_qdrop')
        if qd and qd[0] != qd[-1]:
            probs.append(f'log_qdrop {qd}')
    add('logger', not probs, '; '.join(probs) or f"1 segment, {segs[0].get('data_lines') if segs else 0} lines, no hole")

    t = (col('timestamp_us') - col('timestamp_us')[0]) / 1e6
    motor = col('motor_on')
    sp = col('sp_vx')
    d = np.diff(np.nan_to_num(motor))
    rises, falls = np.nonzero(d > 0)[0] + 1, np.nonzero(d < 0)[0] + 1
    nz = np.nonzero(np.nan_to_num(sp) != 0)[0]
    ok = len(rises) == 1 and len(falls) == 1 and len(nz) > 0 and falls[0] > nz[-1] and rises[0] <= nz[0]
    add('motor', ok, f'rises {len(rises)}, falls {len(falls)}' +
        (f', on {t[rises[0]]:.2f}-{t[falls[0]]:.2f} s, setpoint {t[nz[0]]:.2f}-{t[nz[-1]]:.2f} s' if ok else ''))
    if not (len(rises) == 1 and len(falls) == 1):
        run = np.nan_to_num(motor) > 0.5
        r0 = int(rises[0]) if len(rises) else 0
    else:
        run = np.zeros(len(t), bool)
        run[rises[0]:falls[0]] = True
        r0 = int(rises[0])

    # imu
    init = col('imu_init')
    wz_src, slip = col('wz_src'), col('slip_flag')
    after = run & (t >= t[r0] + t_settle)
    share = float(np.mean(wz_src[after] == 1)) if after.any() else float('nan')
    ok = bool(np.all(init == 0)) and after.any() and share >= WZ_SRC_MIN
    add('imu', ok, f'imu_init {sorted(set(init[np.isfinite(init)].astype(int).tolist()))}, wz_src=1 on '
        f'{share:.4f} of lines from T_settle {t_settle} s')

    # slip: no wrong flag (lifted body: r_true = wz_enc)
    wz_enc = col('wz')
    sflag = np.nan_to_num(slip) != 0
    n_slip = int(np.sum(sflag & run))
    ups = [i for i in np.nonzero(np.diff(sflag.astype(int)) > 0)[0] + 1 if run[i]]
    if sflag[0] and run[0]:
        ups = [0] + ups
    right, wrong = [], []
    for i in ups:
        w = (t > t[i] - SLIP_T_S - 1e-9) & (t <= t[i])
        m = float(np.nanmean(np.abs(wz_enc[w])))
        (right if m >= SLIP_R_TH / 2 else wrong).append((round(float(t[i]), 2), round(m, 4)))
    wz_run = wz_enc[run] if run.any() else np.array([np.nan])
    add('slip', not wrong, f'slip_flag lines {n_slip}, rises {len(ups)}: right {len(right)}, wrong {len(wrong)}'
        + (f' {wrong[:3]}' if wrong else '') + f' (r_th/2 {SLIP_R_TH / 2:.4f}); '
        f'wz_enc motor-on mean {float(np.nanmean(wz_run)):+.4f}, max |.| {float(np.nanmax(np.abs(wz_run))):.4f} rad/s')

    # pos
    try:
        pr = pos_reset_mid(col('run_active'), col('pos_reset_n'), motor)
        add('pos', pr == 0, f'pos_reset_mid {pr}')
    except ValueError as exc:
        add('pos', False, f'pos_reset_mid refused: {exc}')

    # track
    holds = expected_holds(test_id)
    ph = [p for p in phases_in_log(t, np.nan_to_num(sp)) if p[2] != 0.0 and run[p[0]:p[1]].all()
          and t[p[1] - 1] - t[p[0]] + 0.01 >= HOLD_MIN_S]
    tr = []
    if [round(p[2], 3) for p in ph] != [round(h[2], 3) for h in holds]:
        add('track', False, f'hold phases in the log {[round(p[2], 3) for p in ph]} != schedule {[h[2] for h in holds]}')
    else:
        vx = col('vx')
        all_ok = True
        for (i0, i1, v) in ph:
            w = np.arange(i0, i1)[t[i0:i1] >= t[i1 - 1] - LAST_S + 1e-9]
            err = float(np.mean(vx[w]) - v)
            rpm = {k: float(np.mean(col(f'{k}_rpm')[w])) for k in WHEELS}
            sign_ok = all(np.sign(x) == np.sign(v) for x in rpm.values())
            # sat_* are cumulative tick counters since boot (main.c vc->sat_ticks): saturation in the
            # hold = the counter grows over it (bug 10/11: '!= 0' failed a car log whose counters were
            # already 1848-2130 from before the run; Liam 'Sua roi chay tiep')
            sat = int(sum(np.nanmax(col(f'sat_{k}')[i0:i1]) - col(f'sat_{k}')[max(i0 - 1, 0)] for k in WHEELS))
            ok = abs(err) <= VX_TOL and sign_ok and sat == 0
            all_ok &= ok
            tr.append(dict(t0=round(float(t[i0]), 2), sp=v, vx_err=round(err, 4),
                           rpm={k: round(x, 1) for k, x in rpm.items()}, sat_lines=sat, ok=ok))
        add('track', all_ok, '; '.join(f"sp {x['sp']:+.2f} at {x['t0']} s: vx err {x['vx_err']:+.4f}, "
                                       f"sat {x['sat_lines']}, rpm {x['rpm']}" for x in tr))

    # stop (recorded)
    stops = []
    for (i0, i1, v) in phases_in_log(t, np.nan_to_num(sp)):
        if v == 0.0 and i0 > 0 and sp[i0 - 1] != 0 and run[i0]:
            w = (t >= t[i0]) & (t < t[i0] + STOP_WIN_S)
            r = np.array([col(f'{k}_rpm')[w] for k in WHEELS])
            stops.append(dict(t=round(float(t[i0]), 2), from_sp=float(sp[i0 - 1]),
                              rpm_min=round(float(np.nanmin(r)), 1), rpm_max=round(float(np.nanmax(r)), 1)))
    return res, stops, tr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', required=True)
    ap.add_argument('--test', required=True, choices=SCORED_TESTS)
    ap.add_argument('--expect-sha', required=True)
    ap.add_argument('--expect-env', required=True)
    ap.add_argument('--platform', choices=['car', 'hil'], default=None)
    ap.add_argument('--t-settle', type=float, default=T_SETTLE_S)
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    try:
        col, summ, events = load(a.log)
        res, stops, tr = check(col, summ, events, a.test, a.expect_sha, a.expect_env, a.t_settle)
    except InputError as exc:
        print(f'#ket_qua,ma=2,log={a.log},loi={exc}')
        return 2
    for name, ok, detail in res:
        print(f"{name:7s} {'DAT' if ok else 'KHONG DAT'}  {detail}")
    for s in stops:
        print(f"stop    (ghi lai) setpoint {s['from_sp']:+.2f} -> 0 at {s['t']} s: rpm min {s['rpm_min']}, max {s['rpm_max']}")
    passed = all(ok for _, ok, _ in res)
    if a.json:
        json.dump(dict(log=a.log, test=a.test, platform=a.platform, expect_sha=a.expect_sha,
                       expect_env=a.expect_env, t_settle=a.t_settle, passed=passed,
                       checks=[dict(name=n, ok=o, detail=d) for n, o, d in res], track=tr, stops=stops),
                  open(a.json, 'w', encoding='utf-8'), indent=2)
    fails = ','.join(n for n, o, _ in res if not o) or '-'
    print(f"#ket_qua,ma={0 if passed else 1},test={a.test},platform={a.platform},khong_dat={fails}")
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
