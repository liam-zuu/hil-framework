#!/usr/bin/env python3
"""
fault_check.py: the common check of every run with --faults (L82; design 03_CODE/docs/
2026-11-14_Thiet_ke_bom_loi_dot1_v1.md of the thesis repo, section 8.1). Pi side only: "the Pi sent what the
schedule says", byte by byte. Whether the fault reached the DUT is checked from the ESP32 / H7 logs (S06 T8
type I, later).

  python tools/fault_check.py <frames.npz> --faults <schedule.csv> [--out fault_check.json]

Checks:
  1. the frame log has the true frames (mosi_true) and the manifest has 'faults';
  2. SHA-256 of the schedule file (CRLF -> LF) = manifest; rows of the file = rows of the manifest;
  3. every frame k of the log, inside and outside the windows (ARM, STOPPING, END tail too): the 40 bytes
     sent = expected_sent(), built here from the schedule, t0, the stop step and mosi_true only (written
     apart from rpi5/fault_schedule.FramePatcher on purpose: no shared state, no ring buffer);
  4. the on / off / cut events of the manifest = the events expected from the schedule, t0, the stop step
     and the number of frames; 'complete' (no cut, no row that never ran);
  5. the MISO check of the run (L50): no frame with a bad CRC, no empty frame.
Exit 0 = ok, 1 = a check failed, 2 = the input cannot be checked. Writes fault_check.json.

Assumptions (stated by the author, to be checked): the manifest t0_step / stop_step are those the node
used (events of a correct patcher would then match; a wrong t0 shows as byte mismatches, since the bytes are
checked from mosi_true); frame k of the log is step k of the loop.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rpi5'))
from fault_schedule import FAULT_IMU_LOSS, file_sha256, parse, rows_as_json, scale_raw   # noqa: E402
from protocol import OFS_FAULT, OFS_IMU_ACC, OFS_IMU_GZ, OFS_IMU_YAW                    # noqa: E402

WHEEL_IDX = {'FL': 0, 'FR': 1, 'RL': 2, 'RR': 3}


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
            b[OFS_IMU_GZ:OFS_IMU_GZ + 2] = scale_raw(raw, r.p('s')).to_bytes(2, 'big', signed=True)
    for r in on:
        if r.fault == 'imu_loss':
            b[OFS_FAULT] |= FAULT_IMU_LOSS
    return bytes(b)


def expected_events(rows, t0, stop, n_frames):
    """[[k_rel, id, 1/0, why], ...] as a patcher must have executed them, sorted like the manifest (by
    step, off before on, then schedule order). The run ends at the stop step, or after the last frame."""
    if t0 is None:
        return [], [r.id for r in rows]
    end = (stop if stop is not None else n_frames) - t0
    ev, not_run = [], []
    for i, r in enumerate(rows):
        if r.start_ms >= end:
            not_run.append(r.id)
            continue
        ev.append((r.start_ms, 1, i, [r.start_ms, r.id, 1, 'on']))
        if r.end_ms <= end:
            ev.append((r.end_ms, 0, i, [r.end_ms, r.id, 0, 'off']))
        else:
            ev.append((end, 2, i, [end, r.id, 0, 'cut']))
    ev.sort(key=lambda e: (e[0], e[1], e[2]))
    return [e[3] for e in ev], not_run


def _norm_events(evs):
    return sorted([[int(e[0]), str(e[1]), int(e[2]), str(e[3])] for e in evs],
                  key=lambda e: (e[0], {'off': 0, 'on': 1, 'cut': 2}[e[3]], e[1]))


def check(npz_path: str, sched_path: str) -> dict:
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
    mosi, true = f['mosi'], f['mosi_true']
    n = len(mosi)
    t0, stop = fm.get('t0_step'), fm.get('stop_step')
    fails = []
    sha = file_sha256(raw)
    if sha != fm.get('sha256'):
        fails.append(f'sha256 of the file {sha[:12]} != manifest {str(fm.get("sha256"))[:12]}')
    if rows_as_json(rows) != fm.get('rows'):
        fails.append('rows of the file differ from the manifest rows')
    bad = [k for k in range(n) if bytes(mosi[k]) != expected_sent(rows, k, t0, stop, true)]
    if bad:
        fails.append(f'{len(bad)} frames differ from the expected frame, first at frame {bad[0]}')
    exp_ev, exp_nr = expected_events(rows, t0, stop, n)
    if _norm_events(fm.get('events', [])) != _norm_events(exp_ev):
        fails.append('executed events differ from the events expected from the schedule')
    if sorted(fm.get('not_run', [])) != sorted(exp_nr):
        fails.append(f'not_run {fm.get("not_run")} != expected {exp_nr}')
    complete = not exp_nr and not any(e[3] == 'cut' for e in exp_ev)
    if not complete:
        fails.append('the schedule did not run in full (a window cut or never started)')
    mc = man.get('miso_check') or {}
    if mc.get('crc_bad') or mc.get('empty'):
        fails.append(f'MISO: {mc.get("crc_bad")} bad CRC, {mc.get("empty")} empty (L50)')
    diff_frames = int(np.count_nonzero(np.any(mosi != true, axis=1)))
    out.update(ok=not fails, fails=fails, n_frames=n, t0_step=t0, stop_step=stop, sha256=sha,
               frames_bad=len(bad), first_bad=bad[0] if bad else None, frames_injected=diff_frames,
               events=exp_ev, not_run=exp_nr, complete=complete, imu_source=man.get('imu_source'))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames', help='frames.npz of run_hil.py --frame-log ... --faults ...')
    ap.add_argument('--faults', required=True, help='the schedule file of the run')
    ap.add_argument('--out', default=None, help='default: fault_check.json next to the frames')
    a = ap.parse_args(argv)
    try:
        res = check(a.frames, a.faults)
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
