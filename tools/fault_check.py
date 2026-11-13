#!/usr/bin/env python3
"""
fault_check.py: the common check of every run with --faults (L82; design 03_CODE/docs/
2026-11-14_Thiet_ke_bom_loi_dot1_v1.md of the thesis repo, section 8.1). Pi side only: "the Pi sent what the
schedule says", byte by byte. Whether the fault reached the DUT is checked from the ESP32 / H7 logs (S06 T8
type I, later).

  python tools/fault_check.py <frames.npz> --faults <schedule.csv> --step-log <steps.csv> [--out ...]

Checks (all must hold for ok):
  1. the frame log has the true frames (mosi_true) and the manifest has 'faults';
  2. SHA-256 of the schedule file (CRLF -> LF) = manifest; rows of the file = rows of the manifest;
  3. t0 and the stop step are taken from the run phases (manifest 'lifecycle' transitions: first RUN step,
     first step after it out of RUN; without phases t0 = 0, no stop), not from the patcher (Codex FI1-2);
     the patcher's t0_step / stop_step must equal them;
  4. every frame k of the log, inside and outside the windows (ARM, STOPPING, END tail too): every byte
     sent = expected_sent(), built here from the schedule, t0, the stop step and mosi_true only, with its
     own exact decimal arithmetic for gyro_scale (written apart from rpi5/fault_schedule.FramePatcher on
     purpose: no shared state, no ring buffer, no shared rounding, FI1-1);
  5. the events, not_run, cut, complete of the manifest = those expected from the schedule, t0, the stop
     step and the number of frames; events typed strictly and in step order (FI1-7);
  6. MISO recomputed from the miso array (L50, as hil_node: frames from MISO_SKIP_FRAMES on): no bad CRC, no
     empty frame; the manifest counts must agree (FI1-8);
  7. truth self-check (FI1-9): seq byte of every true frame = k mod 256; with the plant IMU and no noise,
     int16 yaw = int16 theta and gz = wz within half a step of each scale;
  8. step log (FI1-10): every step of the frame log present, no missed deadline, nothing lost or torn.
Exit 0 = ok, 1 = a check failed, 2 = the input cannot be checked. Writes fault_check.json.

Assumptions (stated by the author, to be checked): frame k of the log is step k of the loop, and the
lifecycle counts its steps from step 0 of the loop (HILNode calls the lifecycle every step before the END
tail); the step log is the CSV of run_hil.py --step-log with its .json next to it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rpi5'))
from fault_schedule import FAULT_IMU_LOSS, file_sha256, parse, rows_as_json   # noqa: E402
from protocol import (IMU_RANGES, MISO_CRC_BAD, MISO_EMPTY, OFS_FAULT, OFS_IMU_ACC, OFS_IMU_GZ,   # noqa: E402
                      OFS_IMU_YAW, OFS_SEQ, STATE_RANGES, miso_check)
from hil_node import MISO_SKIP_FRAMES                                          # noqa: E402

WHEEL_IDX = {'FL': 0, 'FR': 1, 'RL': 2, 'RR': 3}
WHY_RANK = {'off': 0, 'on': 1, 'cut': 2}
OFS_THETA, OFS_WZ = 12, 18                     # int16 theta, wz of [w1..w4, x, y, theta, vx, vy, wz]


def ref_scale(raw: int, s) -> int:
    """Reference for gyro_scale, apart from fault_schedule.scale_raw: Decimal arithmetic on the exact value
    of s, ROUND_HALF_UP on the magnitude (= half away from zero), int16 clamp."""
    fr = s if isinstance(s, Fraction) else Fraction(repr(float(s)))
    with localcontext() as ctx:
        ctx.prec = 60
        sd = Decimal(fr.numerator) / Decimal(fr.denominator)
        v = (Decimal(1) - sd) * Decimal(raw)
        m = abs(v).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    r = int(m) if v >= 0 else -int(m)
    return max(-32767, min(32767, r))


def window_on(row, k, t0, stop) -> bool:
    """Row injects at absolute frame k: from t0 on, before the stop step, inside [start, end) from t0."""
    if t0 is None or k < t0 or (stop is not None and k >= stop):
        return False
    return row.start_ms <= k - t0 < row.end_ms


def expected_sent(rows, k, t0, stop, true_frames) -> bytes:
    """The frame k must be: true frame k, then enc_loss, delay (IMU bytes of true frame k - d), gyro scale,
    loss bit, in this order."""
    b = bytearray(bytes(true_frames[k]))
    on = [r for r in rows if window_on(r, k, t0, stop)]
    for r in on:
        if r.fault == 'enc_loss':
            for w in r.target.split('+'):
                i = WHEEL_IDX[w]
                b[2 * i:2 * i + 2] = b'\x00\x00'
    for r in on:
        if r.fault == 'imu_delay':
            src = bytes(true_frames[k - r.p('d_ms')])
            b[OFS_IMU_GZ:OFS_IMU_GZ + 2] = src[OFS_IMU_GZ:OFS_IMU_GZ + 2]
            b[OFS_IMU_ACC:OFS_IMU_YAW + 2] = src[OFS_IMU_ACC:OFS_IMU_YAW + 2]
    for r in on:
        if r.fault == 'gyro_scale':
            raw = int.from_bytes(b[OFS_IMU_GZ:OFS_IMU_GZ + 2], 'big', signed=True)
            b[OFS_IMU_GZ:OFS_IMU_GZ + 2] = ref_scale(raw, r.p('s')).to_bytes(2, 'big', signed=True)
    for r in on:
        if r.fault == 'imu_loss':
            b[OFS_FAULT] |= FAULT_IMU_LOSS
    return bytes(b)


def expected_events(rows, t0, stop, n_frames):
    """([[k_rel, id, 1/0, why], ...] sorted by (k, off < on < cut, id), not_run ids). The run ends at the
    stop step, or after the last frame."""
    if t0 is None:
        return [], [r.id for r in rows]
    end = (stop if stop is not None else n_frames) - t0
    ev, not_run = [], []
    for r in rows:
        if r.start_ms >= end:
            not_run.append(r.id)
            continue
        ev.append([r.start_ms, r.id, 1, 'on'])
        ev.append([r.end_ms, r.id, 0, 'off'] if r.end_ms <= end else [end, r.id, 0, 'cut'])
    ev.sort(key=lambda e: (e[0], WHY_RANK[e[3]], e[1]))
    return ev, not_run


def phases_t0_stop(man):
    """(t0, stop) from the run phases: lifecycle transitions, or (0, None) without phases (FI1-2)."""
    lc = man.get('lifecycle')
    if lc is None:
        return 0, None
    t0 = stop = None
    for tr in lc.get('transitions', []):
        if t0 is None:
            if tr['phase'] == 'RUN':
                t0 = int(tr['step'])
        elif tr['phase'] != 'RUN':
            stop = int(tr['step'])
            break
    return t0, stop


def _is_int(x) -> bool:
    return type(x) is int


def events_typed(evs) -> bool:
    """Strict schema of the manifest events (FI1-7): [int k, str id, 0/1, 'on'|'off'|'cut'], k not falling."""
    if not isinstance(evs, list):
        return False
    prev = None
    for e in evs:
        if not (isinstance(e, list) and len(e) == 4 and _is_int(e[0]) and isinstance(e[1], str)
                and _is_int(e[2]) and e[2] in (0, 1) and e[3] in WHY_RANK and (e[2] == 1) == (e[3] == 'on')):
            return False
        if prev is not None and e[0] < prev:
            return False
        prev = e[0]
    return True


def miso_counts(miso):
    """(checked, crc_bad, empty) over the frames from MISO_SKIP_FRAMES on, as hil_node counts them."""
    crc_bad = empty = 0
    for k in range(MISO_SKIP_FRAMES, len(miso)):
        c = miso_check(bytes(miso[k]))
        crc_bad += c == MISO_CRC_BAD
        empty += c == MISO_EMPTY
    return max(0, len(miso) - MISO_SKIP_FRAMES), crc_bad, empty


def _i16(fr, ofs) -> int:
    return int.from_bytes(bytes(fr[ofs:ofs + 2]), 'big', signed=True)


def truth_self_check(man, true):
    """FI1-9 (design 8.1 step 5): mismatches of the invariants of the true frames."""
    bad_seq = [k for k in range(len(true)) if int(true[k][OFS_SEQ]) != k & 0xFF]
    out = dict(seq_bad=len(bad_seq), first_seq_bad=bad_seq[0] if bad_seq else None, imu_checked=False)
    if man.get('imu_source') == 'plant' and man.get('imu_noise') is None:
        g, w = IMU_RANGES[0] / 32767.0, STATE_RANGES[9] / 32767.0
        tol = 0.5 * g + 0.5 * w + 1e-9
        yaw_bad = gz_bad = 0
        for fr in true:
            yaw_bad += _i16(fr, OFS_IMU_YAW) != _i16(fr, OFS_THETA)
            wz = _i16(fr, OFS_WZ) * w
            if abs(_i16(fr, OFS_WZ)) < 32767:                 # wz not clamped at its range
                gz_bad += abs(_i16(fr, OFS_IMU_GZ) * g - wz) > tol
        out.update(imu_checked=True, yaw_theta_bad=yaw_bad, gz_wz_bad=gz_bad)
    return out


def step_log_check(path, n):
    """FI1-10: every step 0..n-1 in the step log, no 'miss', nothing lost or torn (its .json)."""
    steps, misses = set(), 0
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            steps.add(int(row['step']))
            misses += int(row['miss'])
    missing = [k for k in range(n) if k not in steps]
    meta = {}
    if os.path.isfile(path + '.json'):
        with open(path + '.json') as fh:
            meta = json.load(fh)
    return dict(file=path, rows=len(steps), missed_deadline=misses, steps_missing=len(missing),
                first_missing=missing[0] if missing else None, json=bool(meta),
                lost_or_torn=meta.get('lost_or_torn'), overflow=meta.get('overflow'))


def check(npz_path: str, sched_path: str, step_log: str = None) -> dict:
    f = np.load(npz_path)
    man = json.loads(str(f['manifest']))
    out = dict(frames=npz_path, schedule=sched_path, ok=False)
    fm = man.get('faults')
    if fm is None or 'mosi_true' not in f.files:
        out['error'] = 'no fault manifest or no mosi_true in the frame log (run without --faults?)'
        return out
    with open(sched_path, 'rb') as fh:
        raw = fh.read()
    rows = parse(raw.decode('utf-8'))
    mosi, true, miso = f['mosi'], f['mosi_true'], f['miso']
    n = len(mosi)
    fails = []
    sha = file_sha256(raw)
    if sha != fm.get('sha256'):
        fails.append(f'sha256 of the file {sha[:12]} != manifest {str(fm.get("sha256"))[:12]}')
    if rows_as_json(rows) != fm.get('rows'):
        fails.append('rows of the file differ from the manifest rows')
    t0, stop = phases_t0_stop(man)
    for name, want in (('t0_step', t0), ('stop_step', stop)):
        got = fm.get(name)
        if not (got is None or _is_int(got)) or got != want:
            fails.append(f'manifest {name}={got!r}, the run phases give {want!r}')
    bad = [k for k in range(n) if bytes(mosi[k]) != expected_sent(rows, k, t0, stop, true)]
    if bad:
        fails.append(f'{len(bad)} frames differ from the expected frame, first at frame {bad[0]}')
    exp_ev, exp_nr = expected_events(rows, t0, stop, n)
    evs = fm.get('events')
    if not events_typed(evs):
        fails.append('manifest events: wrong types or not in step order')
    elif sorted(evs, key=lambda e: (e[0], WHY_RANK[e[3]], e[1])) != exp_ev:
        fails.append('executed events differ from the events expected from the schedule')
    if sorted(fm.get('not_run') or []) != sorted(exp_nr) or not isinstance(fm.get('not_run'), list):
        fails.append(f'not_run {fm.get("not_run")} != expected {exp_nr}')
    cut = any(e[3] == 'cut' for e in exp_ev)
    complete = not exp_nr and not cut
    if fm.get('cut') is not cut or fm.get('complete') is not complete:
        fails.append(f'manifest cut={fm.get("cut")!r} complete={fm.get("complete")!r}, expected {cut}, {complete}')
    if not complete:
        fails.append('the schedule did not run in full (a window cut or never started)')
    checked, crc_bad, empty = miso_counts(miso)
    mc = man.get('miso_check') or {}
    if crc_bad or empty:
        fails.append(f'MISO: {crc_bad} bad CRC, {empty} empty (L50, recomputed from the frames)')
    if mc.get('crc_bad') != crc_bad or mc.get('empty') != empty:
        fails.append(f'manifest miso_check {mc} differs from the frames ({crc_bad} bad CRC, {empty} empty)')
    sc = truth_self_check(man, true)
    if sc['seq_bad'] or sc.get('yaw_theta_bad') or sc.get('gz_wz_bad'):
        fails.append(f'true frames fail the self-check: {sc}')
    if step_log:
        tl = step_log_check(step_log, n)
        if tl['missed_deadline'] or tl['steps_missing'] or not tl['json'] or tl['lost_or_torn'] or tl['overflow']:
            fails.append(f'step log: {tl}')
    else:
        tl = None
        fails.append('no step log: the run does not show that no step missed its deadline (design 8.1)')
    diff_frames = int(np.count_nonzero(np.any(mosi != true, axis=1)))
    out.update(ok=not fails, fails=fails, n_frames=n, t0_step=t0, stop_step=stop, sha256=sha,
               frames_bad=len(bad), first_bad=bad[0] if bad else None, frames_injected=diff_frames,
               events=exp_ev, not_run=exp_nr, complete=complete, imu_source=man.get('imu_source'),
               miso=dict(checked=checked, crc_bad=crc_bad, empty=empty), self_check=sc, step_log=tl)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames', help='frames.npz of run_hil.py --frame-log ... --faults ...')
    ap.add_argument('--faults', required=True, help='the schedule file of the run')
    ap.add_argument('--step-log', default=None, help='the --step-log CSV of the run (its .json next to it)')
    ap.add_argument('--out', default=None, help='default: fault_check.json next to the frames')
    a = ap.parse_args(argv)
    try:
        res = check(a.frames, a.faults, a.step_log)
    except (OSError, ValueError, KeyError) as exc:
        print(f'fault_check: cannot check: {exc}', file=sys.stderr)
        return 2
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.frames)), 'fault_check.json')
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1)
    if 'error' in res:
        print(f'fault_check: {res["error"]}', file=sys.stderr)
        return 2
    print(f'fault_check: ok={res["ok"]} frames={res["n_frames"]} injected={res["frames_injected"]} '
          f'bad={res["frames_bad"]} complete={res["complete"]} -> {out}')
    for m in res['fails']:
        print(f'  FAIL: {m}')
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
