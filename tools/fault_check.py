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
  8. step log (FI1-10, FI2-6): every step of the frame log present once and in rising order, every miss
     0 or 1 and none 1, its .json rows = CSV rows and lost_or_torn, overflow present as int 0;
  9. the run ended the normal way (FI1-4, FI4-1): id_confirmed True (a MISO carried the run_id back, RS-4);
     stop_reason 'max_steps' with n = max_steps without phases; with phases last phase END, outcome not
     error:..., END tail complete, stop_reason = outcome;
  10. the frame arrays (mosi, mosi_true, miso, t_send_ns) have the same number of rows, the three frame
     arrays are 2-D with frames of protocol.FRAME_SIZE bytes, manifest steps = n, MISO checked =
     n - MISO_SKIP_FRAMES (FI2-5, FI3-3, FI3-4; a log of another frame size cannot be checked, exit 2); the
     lifecycle transitions are typed strictly, follow the phase changes RunLifecycle can make, inside the
     log (FI2-2, FI3-2).
  11. batch 2, plant faults (design 2026-11-15 section 7.1; only for a schedule with a plant row): the plant
     record named in the manifest (faults.plant_record) exists next to the frames with that SHA-256 and the
     expected columns, did not overflow, names the inner plant MecanumPlant; rpi5/params.py of this tree is
     the blob the run imported (manifest plant_src) and the nominal values are its values, mu_hd being four
     times --mu when the argv of the run (step log .json, read by run_hil.py's parser) has --mu; its steps are
     exactly the frames where a plant row is on (same window rule as the frames); at every step the mask and
     the parameters are rebuilt here from the rows, the nominal values and the recorded pose: hubs and zone
     bounds in float64 by the definition, the battery ramp in float64 by its definition and V compared
     exactly, b, tc, ts as exact fractions within a few ULP (Codex FP1-1..FP2-1, FP3-1); x, y, theta agree with
     the true frame of that step within half a step of their int16 scale. The frames of a plant fault are not
     patched: expected_sent ignores plant rows. A schedule without plant rows must have no plant record.
Exit 0 = ok, 1 = a check failed, 2 = the input cannot be checked. Writes fault_check.json.

Assumptions (stated by the author, to be checked): frame k of the log is step k of the loop, and the
lifecycle counts its steps from step 0 of the loop (HILNode calls the lifecycle every step before the END
tail); the step log is the CSV of run_hil.py --step-log with its .json next to it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from decimal import ROUND_HALF_UP, Decimal, Inexact, localcontext
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rpi5'))
from fault_schedule import FAULT_IMU_LOSS, file_sha256, parse, rows_as_json   # noqa: E402
from protocol import (FRAME_SIZE, IMU_RANGES, MISO_CRC_BAD, MISO_EMPTY, OFS_FAULT, OFS_IMU_ACC, OFS_IMU_GZ,   # noqa: E402
                      OFS_IMU_YAW, OFS_SEQ, STATE_RANGES, miso_check)
from hil_node import MISO_SKIP_FRAMES                                          # noqa: E402
from run_lifecycle import _PHASES as PHASES                                    # noqa: E402

# Phase changes RunLifecycle can record (rpi5/run_lifecycle.py: begin() enters ARM, fail_prepare() END from
# PREPARE; ARM -> RUN or STOPPING; RUN -> STOPPING; STOPPING -> END). PREPARE is the start, never recorded;
# None = before the first transition (Codex FI3-2).
NEXT_PHASE = {None: ('ARM', 'END'), 'ARM': ('RUN', 'STOPPING'), 'RUN': ('STOPPING',), 'STOPPING': ('END',),
              'END': (), 'PREPARE': ()}

WHEEL_IDX = {'FL': 0, 'FR': 1, 'RL': 2, 'RR': 3}
WHY_RANK = {'off': 0, 'on': 1, 'cut': 2}
OFS_THETA, OFS_WZ = 12, 18                     # int16 theta, wz of [w1..w4, x, y, theta, vx, vy, wz]


def ref_scale(raw: int, s) -> int:
    """Reference for gyro_scale, apart from fault_schedule.scale_raw: Decimal arithmetic on the exact value
    of s, ROUND_HALF_UP on the magnitude (= half away from zero), int16 clamp. The precision is sized from
    the input (a decimal num/den, den = 2^a 5^b, has at most digits(num) + max(a, b) significant digits) and
    Inexact is trapped, so no step rounds before the final one, whatever the length of s (Codex FI2-1); an s
    that is not a finite decimal raises ValueError (exit 2)."""
    fr = s if isinstance(s, Fraction) else Fraction(repr(float(s)))
    nd = len(str(abs(fr.numerator))) + 4 * len(str(fr.denominator)) + len(str(abs(int(raw)))) + 4
    with localcontext() as ctx:
        ctx.prec = nd
        ctx.traps[Inexact] = True
        try:
            sd = Decimal(fr.numerator) / Decimal(fr.denominator)
            v = (Decimal(1) - sd) * Decimal(raw)
        except Inexact:
            raise ValueError(f'gyro_scale s={fr} is not a finite decimal') from None
        ctx.traps[Inexact] = False                     # the final rounding to a whole number is meant
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
    stop step, or after the last frame, whichever comes first: no event is taken as executed past the
    frames of the log (Codex FI2-2)."""
    if t0 is None:
        return [], [r.id for r in rows]
    end = min(stop if stop is not None else n_frames, n_frames) - t0
    ev, not_run = [], []
    for r in rows:
        if r.start_ms >= end:
            not_run.append(r.id)
            continue
        ev.append([r.start_ms, r.id, 1, 'on'])
        ev.append([r.end_ms, r.id, 0, 'off'] if r.end_ms <= end else [end, r.id, 0, 'cut'])
    ev.sort(key=lambda e: (e[0], WHY_RANK[e[3]], e[1]))
    return ev, not_run


def phases_t0_stop(man, n):
    """(t0, stop, problem) from the run phases: lifecycle transitions, or (0, None, None) without phases
    (FI1-2). The transitions are checked before use (Codex FI2-2, FI3-2): a list of {step, phase}, step a
    strict int (no bool, no float), phase one of run_lifecycle's phases, each phase change one that
    RunLifecycle can make (NEXT_PHASE: the list starts at ARM or END, END only after STOPPING; a list cut
    before END is a start of such a path), steps not falling, every step <= n (no phase change past the
    frames of the log). problem = None, or why the phases cannot be used (then t0 = stop = None)."""
    lc = man.get('lifecycle')
    if lc is None:
        return 0, None, None
    trs = lc.get('transitions') if isinstance(lc, dict) else None
    if not isinstance(trs, list):
        return None, None, 'manifest lifecycle: no transitions list'
    prev_step, prev_phase = None, None
    for tr in trs:
        if not (isinstance(tr, dict) and _is_int(tr.get('step')) and tr.get('phase') in PHASES):
            return None, None, f'manifest lifecycle: bad transition {tr!r}'
        if tr['phase'] not in NEXT_PHASE[prev_phase] or (prev_step is not None and tr['step'] < prev_step):
            return None, None, f'manifest lifecycle: transitions out of order at {tr!r} (after {prev_phase})'
        if not 0 <= tr['step'] <= n:
            return None, None, f'manifest lifecycle: step {tr["step"]} outside the {n} frames of the log'
        prev_step, prev_phase = tr['step'], tr['phase']
    t0 = stop = None
    for tr in trs:
        if t0 is None:
            if tr['phase'] == 'RUN':
                t0 = tr['step']
        elif tr['phase'] != 'RUN':
            stop = tr['step']
            break
    return t0, stop, None


def run_end_problem(man, n):
    """Codex FI1-4: None if the run ended the normal way, else why not. Without phases: stop_reason
    'max_steps' and n = max_steps (an exception in HILNode.run writes stop_reason 'error:<name>'). With
    phases (as HILNode.run_failed): the last phase is END, the outcome is not 'error:...', the END tail ran
    in full, and stop_reason = the outcome (an exception, a signal or a full frame log after END rewrites
    it)."""
    reason = man.get('stop_reason')
    # Codex FI4-1: RS-4 P1 takes a run whose MISO never carried its run_id back as failed (HILNode.run_failed on
    # the real link); here for every run, simulated ones too (their MISO is the MOSI echo or fake_h7, which
    # carry the run_id), so no manifest key on the mode is needed. A log without the key cannot pass.
    if man.get('id_confirmed') is not True:
        return f'run did not end normally: id_confirmed={man.get("id_confirmed")!r} (no MISO carried the run_id)'
    lc = man.get('lifecycle')
    if lc is None:
        ms = man.get('max_steps')
        if reason != 'max_steps' or not _is_int(ms) or ms <= 0 or n != ms:
            return f'run did not end normally: stop_reason={reason!r}, max_steps={ms!r}, frames={n}'
        return None
    trs = lc.get('transitions') if isinstance(lc, dict) else None
    trs = trs if isinstance(trs, list) else []
    last = trs[-1].get('phase') if trs and isinstance(trs[-1], dict) else None
    outcome = lc.get('outcome') if isinstance(lc, dict) else None
    if (last != 'END' or not isinstance(outcome, str) or outcome.startswith('error:')
            or man.get('swa_tail_complete') is not True or reason != outcome):
        return (f'run did not end normally: last phase {last!r}, outcome {outcome!r}, stop_reason {reason!r}, '
                f'END tail complete {man.get("swa_tail_complete")!r}')
    return None


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
    """FI1-10, Codex FI2-6: every step 0..n-1 in the step log, each once and in rising order, no step past
    n-1; every 'miss' is the text 0 or 1 and none is 1; its .json has 'rows' = the CSV rows and the counters
    'lost_or_torn', 'overflow' as int 0 (a missing counter is not taken as 0). problems = why it fails."""
    steps, misses, bad_miss, order_bad, last_k = [], 0, 0, 0, None
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            st = row.get('step') or ''
            k = int(st) if st.isascii() and st.isdigit() else None
            if k is None or (last_k is not None and k <= last_k):
                order_bad += 1
            if k is not None:
                last_k = k
            steps.append(k)
            m = row.get('miss')
            if m not in ('0', '1'):
                bad_miss += 1
            misses += m == '1'
    seen = {k for k in steps if k is not None}
    missing = [k for k in range(n) if k not in seen]
    extra = sum(1 for k in seen if k >= n)
    meta = None
    if os.path.isfile(path + '.json'):
        with open(path + '.json') as fh:
            meta = json.load(fh)
    meta_d = meta if isinstance(meta, dict) else {}
    problems = []
    if misses:
        problems.append(f'{misses} steps missed their deadline')
    if bad_miss:
        problems.append(f'{bad_miss} rows with miss not 0/1')
    if order_bad:
        problems.append(f'{order_bad} rows with a step not a whole number or not rising (repeat or out of order)')
    if missing or extra:
        problems.append(f'{len(missing)} steps of the frame log missing, {extra} steps past the frame log')
    if meta is None:
        problems.append('no .json next to the step log')
    else:
        for key in ('lost_or_torn', 'overflow'):
            if not (_is_int(meta_d.get(key)) and meta_d.get(key) == 0):
                problems.append(f'.json {key}={meta_d.get(key)!r}, needs int 0')
        if not (_is_int(meta_d.get('rows')) and meta_d.get('rows') == len(steps)):
            problems.append(f'.json rows={meta_d.get("rows")!r} != {len(steps)} CSV rows')
    return dict(file=path, rows=len(steps), missed_deadline=misses, bad_miss=bad_miss, order_bad=order_bad,
                steps_missing=len(missing), first_missing=missing[0] if missing else None, steps_extra=extra,
                json=meta is not None, lost_or_torn=meta_d.get('lost_or_torn'), overflow=meta_d.get('overflow'),
                json_rows=meta_d.get('rows'), problems=problems)


PLANT_ROWS = ('wheel_jam', 'batt_sag', 'mu_time', 'mu_zone')      # batch 2 plant faults (written here again)
REC_COLS = ('step', 'mask', 'V_batt', 'mu_fl', 'mu_fr', 'mu_rl', 'mu_rr', 'b_fl', 'b_fr', 'b_rl', 'b_rr',
            'tc_fl', 'tc_fr', 'tc_rl', 'tc_rr', 'ts_fl', 'ts_fr', 'ts_rl', 'ts_rr', 'x', 'y', 'theta')
OFS_X, OFS_Y = 8, 10                           # int16 x, y of [w1..w4, x, y, theta, ...] (theta: OFS_THETA)
ULP_TOL = 8                                    # float sums of the Pi against the exact values here (FP1-4)
HUBS = {0: (1, 1), 1: (1, -1), 2: (-1, 1), 3: (-1, -1)}   # FL, FR, RL, RR: (sign of lx, sign of ly), y left


def _close(got, exact) -> bool:
    """A float64 the Pi built by a few float additions against the exact value: within ULP_TOL units in the
    last place of the larger of the two (Codex FP1-4: no absolute tolerance, the range has no upper bound)."""
    e = float(exact)
    return math.isfinite(got) and abs(got - e) <= ULP_TOL * math.ulp(max(abs(got), abs(e)))


def _row_wheels(r):
    if r.fault == 'mu_zone' or r.target == 'all':        # mu_time may take 'all' (Codex FP1-7)
        return (0, 1, 2, 3)
    return tuple(WHEEL_IDX[w] for w in r.target.split('+'))


def hubs(x, y, th, lx, ly):
    """World hub positions FL, FR, RL, RR by the definition of the design (float64, this order; Codex FP1-1):
    c = cos(theta), s = sin(theta), ax = sx lx, ay = sy ly, hub = ((x + c ax) - s ay, (y + s ax) + c ay)."""
    c, s = math.cos(th), math.sin(th)
    out = {}
    for w, (sx, sy) in HUBS.items():
        ax, ay = sx * lx, sy * ly
        out[w] = ((x + c * ax) - s * ay, (y + s * ax) + c * ay)
    return out


def ref_plant(rows, on_idx, k_rel, pose, nom):
    """Parameters of one plant step rebuilt from the rows on (indices, row order), the relative step, the
    pose at the start of the step and the nominal values: (V, mu[4], b[4], tc[4], ts[4]); b, tc, ts and mu as
    exact Fractions, V the nominal value or the float64 ramp of its definition (FP2-1)."""
    F = lambda v: Fraction(v) if not isinstance(v, Fraction) else v           # noqa: E731
    V = F(nom['V_batt'])
    mu = [F(v) for v in nom['mu_hd']]
    b, tc, ts = ([F(v) for v in nom[k]] for k in ('motor_b', 'motor_Tc', 'motor_Ts'))
    for i in on_idx:
        r = rows[i]
        if r.fault == 'wheel_jam':
            for w in _row_wheels(r):
                b[w] += r.p('b')
                tc[w] += r.p('tc')
                ts[w] += r.p('tc')
        elif r.fault == 'batt_sag':
            # Codex FP2-1: the float64 definition of the ramp (one operation at a time, as the design states),
            # rebuilt here and compared exactly; an exact fraction would differ by the rounding of v1 - v0
            ramp, d = r.p('ramp_ms'), k_rel - r.start_ms
            v0, v1 = float(nom['V_batt']), float(r.p('v1'))
            V = v1 if ramp == 0 or d >= ramp else v0 + (v1 - v0) * (d / ramp)
        elif r.fault == 'mu_time':
            for w in _row_wheels(r):
                mu[w] = r.p('mu')
        elif r.fault == 'mu_zone':
            # bounds as float64 (the decimal rounded once), as the definition says (Codex FP1-1)
            x0, x1, y0, y1 = (float(r.p(k)) for k in ('x0', 'x1', 'y0', 'y1'))
            for w, (hx, hy) in hubs(float(pose[0]), float(pose[1]), float(pose[2]),
                                    float(nom['lx']), float(nom['ly'])).items():
                if x0 <= hx < x1 and y0 <= hy < y1:
                    mu[w] = r.p('mu')
    return V, mu, b, tc, ts


def run_mu(argv):
    """--mu of the run as rpi5/run_hil.py's own parser reads its command line (the argv the step log .json
    keeps; Codex FP1-2 round 2): a float, or None without --mu. ValueError when the parser refuses it."""
    import contextlib
    import io
    import run_hil
    if not (isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv)):
        raise ValueError(f'step log argv {argv!r}')
    old, err = sys.argv, io.StringIO()
    sys.argv = ['run_hil.py'] + argv[1:]
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            ns = run_hil._parse_args()
    except SystemExit:
        raise ValueError(f'run_hil.py refuses the argv of the step log: {err.getvalue().strip()[-200:]}') from None
    finally:
        sys.argv = old
    return ns.mu


def nominal_problems(nom, plant_src=None, argv=None) -> list:
    """The nominal values of the record against rpi5/params.py of this tree and the command line of the run.
    Codex FP1-2: this tree's params.py must be the blob the run imported (manifest plant_src); every nominal
    value finite and > 0 (lx, ly, motor_* >= 0); mu_hd = four times --mu of the run's argv (step log .json),
    or params.py mu_hd without --mu (round 2: four equal values alone are not taken as --mu)."""
    from params import MecanumParams
    from plant_blob import plant_source
    p = MecanumParams()
    out = []
    here = plant_source().get('params_py')
    if not isinstance(plant_src, dict) or not here or plant_src.get('params_py') != here:
        out.append(f'params.py of this tree ({str(here)[:12]}) is not the one of the run '
                   f'({str((plant_src or {}).get("params_py"))[:12]}): the nominal values cannot be checked')
    flat = [nom.get(k) for k in ('V_batt', 'lx', 'ly')] + [v for k in ('mu_hd', 'motor_b', 'motor_Tc', 'motor_Ts')
                                                            for v in (nom.get(k) or [None])]
    if not all(type(v) is float and math.isfinite(v) and v >= 0 for v in flat) or \
            not (nom.get('V_batt', 0) > 0 and all(v > 0 for v in nom.get('mu_hd') or [0])):
        out.append(f'nominal values not finite floats > 0: {nom!r}')
    for k in ('V_batt', 'lx', 'ly'):
        if nom.get(k) != float(getattr(p, k)):
            out.append(f'nominal {k}={nom.get(k)!r} != params.py {float(getattr(p, k))}')
    for k in ('motor_b', 'motor_Tc', 'motor_Ts'):
        if nom.get(k) != [float(v) for v in getattr(p, k)]:
            out.append(f'nominal {k}={nom.get(k)!r} != params.py')
    mu = nom.get('mu_hd')
    try:
        m = run_mu(argv)
    except ValueError as exc:
        out.append(f'--mu of the run unknown ({exc}): the nominal mu_hd cannot be checked')
        return out
    want = [float(v) for v in np.broadcast_to(p.mu_hd, (4,))] if m is None else [float(m)] * 4
    if mu != want:
        out.append(f'nominal mu_hd={mu!r} != {want} ({"params.py" if m is None else "--mu " + repr(m)} of the run)')
    return out


def step_log_argv(step_log):
    """argv of the run kept in the .json of its step log (run_hil.py), or None."""
    if not step_log or not os.path.isfile(step_log + '.json'):
        return None
    with open(step_log + '.json') as fh:
        meta = json.load(fh)
    return meta.get('argv') if isinstance(meta, dict) else None


def plant_record_check(npz_path, rows, fm, t0, stop, n, true, plant_src=None, argv=None) -> tuple:
    """Check 11: (problems, summary)."""
    prow = [i for i, r in enumerate(rows) if r.fault in PLANT_ROWS]
    pr = fm.get('plant_record')
    if not prow:
        return (['a plant record without plant rows'] if pr is not None else []), None
    if not isinstance(pr, dict) or not isinstance(pr.get('file'), str) or not pr.get('sha256'):
        return [f'plant rows but no plant record in the manifest: {pr!r}'], None
    # Codex FP1-9: a bare file name, next to the frames
    if os.path.basename(pr['file']) != pr['file'] or pr['file'] in ('', '.', '..') or os.path.isabs(pr['file']):
        return [f'plant record file {pr["file"]!r} is not a bare file name next to the frames'], None
    if pr.get('inner') != 'MecanumPlant':                                      # Codex FP1-11
        return [f'plant record inner plant {pr.get("inner")!r}, plant faults run on MecanumPlant only'], None
    path = os.path.join(os.path.dirname(os.path.abspath(npz_path)), pr['file'])
    if not os.path.isfile(path):
        return [f'plant record {pr["file"]} not found next to the frames'], None
    with open(path, 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    probs = []
    if sha != pr['sha256']:
        probs.append(f'plant record sha256 {sha[:12]} != manifest {str(pr["sha256"])[:12]}')
    g = np.load(path)
    rec = g['record']
    if json.loads(str(g['cols'])) != list(REC_COLS) or pr.get('cols') != list(REC_COLS) or \
            rec.ndim != 2 or rec.shape[1] != len(REC_COLS):
        return probs + [f'plant record columns {json.loads(str(g["cols"]))} / shape {rec.shape} not as expected'], None
    if pr.get('overflow') is not False or not _is_int(pr.get('rows')) or pr.get('rows') != len(rec):   # FP1-3
        probs.append(f'plant record overflow={pr.get("overflow")!r} rows={pr.get("rows")!r}, file has {len(rec)}')
    nom = pr.get('nominal') if isinstance(pr.get('nominal'), dict) else {}
    nprob = nominal_problems(nom, plant_src, argv)
    if nprob:
        return probs + nprob, dict(file=pr['file'], rows=len(rec), steps_ok=None)
    exp_steps = [k for k in range(n) if any(window_on(rows[i], k, t0, stop) for i in prow)]
    steps = [int(v) for v in rec[:, 0]]
    if steps != exp_steps or not np.array_equal(rec[:, 0], np.array(steps, dtype=float)):
        probs.append(f'plant record steps differ from the windows: {len(steps)} recorded, {len(exp_steps)} expected'
                     f'{", first " + str(steps[0]) if steps else ""}')
        return probs, dict(file=pr['file'], rows=len(rec), steps_ok=False)
    c = {name: j for j, name in enumerate(REC_COLS)}
    sx, sy, sth = STATE_RANGES[4] / 32767.0, STATE_RANGES[5] / 32767.0, STATE_RANGES[6] / 32767.0
    bad_val, bad_pose, first = 0, 0, None
    for row in rec:
        k = int(row[0])
        on = [i for i in prow if window_on(rows[i], k, t0, stop)]
        pose = (row[c['x']], row[c['y']], row[c['theta']])
        V, mu, b, tc, ts = ref_plant(rows, on, k - t0, pose, nom)
        # FP1-3: the mask as recorded equals the expected whole number exactly (no int() cast)
        ok = bool(row[c['mask']] == float(sum(1 << i for i in on))) and bool(row[c['V_batt']] == float(V))   # FP2-1
        for w, suf in enumerate(('fl', 'fr', 'rl', 'rr')):
            ok &= bool(row[c['mu_' + suf]] == float(mu[w]))
            ok &= _close(row[c['b_' + suf]], b[w]) and _close(row[c['tc_' + suf]], tc[w])
            ok &= _close(row[c['ts_' + suf]], ts[w])
        if not ok:
            bad_val += 1
            first = k if first is None else first
        fr = true[k]
        pose_ok = (abs(_i16(fr, OFS_X) * sx - pose[0]) <= 0.5 * sx + 1e-9 and
                   abs(_i16(fr, OFS_Y) * sy - pose[1]) <= 0.5 * sy + 1e-9 and
                   abs(_i16(fr, OFS_THETA) * sth - math.remainder(pose[2], 2 * math.pi)) <= 0.5 * sth + 1e-9)
        bad_pose += not pose_ok
    if bad_val:
        probs.append(f'{bad_val} plant steps with parameters that differ from the rows, first at frame {first}')
    if bad_pose:
        probs.append(f'{bad_pose} plant steps whose recorded pose differs from the true frame')
    return probs, dict(file=pr['file'], sha256=sha, rows=len(rec), steps_ok=True, values_bad=bad_val,
                       pose_bad=bad_pose)


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
    t_send = f['t_send_ns'] if 't_send_ns' in f.files else None
    # Codex FI2-5, FI3-3, FI3-4: the frame arrays are 2-D (checked before len()), cover the same steps, and
    # their frames are protocol.FRAME_SIZE bytes (coordination rule 3.7: the constant, not a fixed 40)
    n = len(mosi) if mosi.ndim == 2 else -1
    # FI4-2: and of dtype uint8 (no cast: a uint16 array of shape (n, 48) would be 96 bytes a frame)
    if (n < 0 or any(a.dtype != np.uint8 for a in (mosi, true, miso))
            or mosi.shape[1] != FRAME_SIZE or true.shape != mosi.shape or miso.shape != mosi.shape
            or t_send is None or t_send.shape != (n,)):
        out['error'] = (f'frame arrays do not match: mosi {mosi.shape} {mosi.dtype}, mosi_true {true.shape} '
                        f'{true.dtype}, miso {miso.shape} {miso.dtype}, t_send_ns '
                        f'{None if t_send is None else t_send.shape}, frame size {FRAME_SIZE} uint8')
        return out
    fails = []
    if man.get('steps') != n or not _is_int(man.get('steps')):
        fails.append(f'manifest steps={man.get("steps")!r} != {n} frames in the log')
    end_problem = run_end_problem(man, n)
    if end_problem:
        fails.append(end_problem)
    sha = file_sha256(raw)
    if sha != fm.get('sha256'):
        fails.append(f'sha256 of the file {sha[:12]} != manifest {str(fm.get("sha256"))[:12]}')
    if rows_as_json(rows) != fm.get('rows'):
        fails.append('rows of the file differ from the manifest rows')
    t0, stop, lc_problem = phases_t0_stop(man, n)
    if lc_problem:
        fails.append(lc_problem)
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
    if mc.get('crc_bad') != crc_bad or mc.get('empty') != empty or mc.get('checked') != checked:
        fails.append(f'manifest miso_check {mc} differs from the frames ({checked} checked, {crc_bad} bad CRC, '
                     f'{empty} empty)')
    if checked != max(0, n - MISO_SKIP_FRAMES):                                  # FI2-5
        fails.append(f'MISO: {checked} frames checked, the run has {max(0, n - MISO_SKIP_FRAMES)}')
    sc = truth_self_check(man, true)
    if sc['seq_bad'] or sc.get('yaw_theta_bad') or sc.get('gz_wz_bad'):
        fails.append(f'true frames fail the self-check: {sc}')
    if step_log:
        tl = step_log_check(step_log, n)
        if tl['problems']:
            fails.append(f'step log: {"; ".join(tl["problems"])}')
    else:
        tl = None
        fails.append('no step log: the run does not show that no step missed its deadline (design 8.1)')
    plant_probs, plant = plant_record_check(npz_path, rows, fm, t0, stop, n, true,
                                             man.get('plant_src'), step_log_argv(step_log))   # check 11 (batch 2)
    fails += plant_probs
    diff_frames = int(np.count_nonzero(np.any(mosi != true, axis=1)))
    out.update(ok=not fails, fails=fails, n_frames=n, t0_step=t0, stop_step=stop, sha256=sha,
               frames_bad=len(bad), first_bad=bad[0] if bad else None, frames_injected=diff_frames,
               events=exp_ev, not_run=exp_nr, complete=complete, imu_source=man.get('imu_source'),
               miso=dict(checked=checked, crc_bad=crc_bad, empty=empty), self_check=sc, step_log=tl,
               plant=plant)
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
    except (OSError, ValueError, KeyError, TypeError) as exc:      # TypeError: Codex FI3-4
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
