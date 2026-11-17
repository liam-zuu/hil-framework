#!/usr/bin/env python3
"""
s02_video_metrics.py: the car's true pose on the floor for the S02 family R (Viec CODE 5; design
CLAUDE_THESIS 03_CODE/docs/2026-11-15_Thiet_ke_CODE_S02_v1.md section 34; Liam 17/11 CODE_S02 local_bc2d1c16,
selection boxes: "Toc do, gyro kiem cheo (Recommended)", "0,68 s nhu O1 (Recommended)", "Giong file plant
(Recommended)", "Chi ghi so (Recommended)"). Spec S02 P2-3 (CLAUDE_THESIS 06_SPECS/2026-10-21_Spec_S02_...md
lines 135, 148-157), P2-5 (line 170).

Inputs: frames.csv of tools/c7_measure.py track (x, y, psi = the common centre of the four wheels on the floor frame
of moc_san_do.csv, t_s on the video clock, valid), and the ESP32 log of the run (agv_logger CSV; the master clock is
timestamp_us).

1. Clock: o1_time_sync.sync on the speed channel (camera speed of the valid frames against hypot(vx, vy) of the
   log), window from the log as the figure-8 scorer (c7_score.clock_match: o1_time_sync.log_window and its checks)
   but started at the first non-zero command (s02_window: the zero commands of the S02 preparation). Not ok
   (correlation below MIN_CORR, start/end thirds apart by more than SPLIT_TOL_S, a short window with a low
   correlation, or no fit at all) -> no pose, reason clock_sync_failed. The gyro channel (camera wz against
   gz_fused) is fitted too and only reported (cross-check, Liam "gyro kiem cheo").
2. Pose on the log grid (P2-3 line 135): for every data line of the log (unique timestamp_us), the camera pose at
   t_cam = t_log - D, linear between the two valid frames around it, psi unwrapped over the valid frames first;
   never extrapolated (before the first or after the last valid frame: empty, cam_lost empty). Two valid frames
   more than T_MISS apart (c7_score.T_MISS, O1 3.3, 0.68 s, [DE XUAT, chot o C8]): the lines between them have no
   pose and cam_lost = 1 (P2-5 line 170: the measuring system failed, group 1); s02_metrics refuses the family R of
   a run with such a line inside its metric window (camera_lost).
3. Outputs: <run>_plant.csv with the columns of a plant file (PLANT_COLUMNS of s02_plant_ticks; only timestamp_us,
   x, y, theta filled) and cam_lost, so s02_metrics.score_run reads it as for SIM and HIL; <run>_plant_video.json
   (ok, reason, the fit, the gyro cross-check, the largest gap between valid frames, counts, SHA-256 of the inputs).
   Never overwritten.

The rest pose p(0), psi(0) (mean camera pose over the 0.5 s before t = 0, P2-3 line 155, Liam 17/11 "0,5 s truoc
t = 0") is taken by s02_metrics on these rows: only score_run knows t = 0 (the two command anchors). No measurement
uncertainty label (P2-3 line 157: the sign of a family-R difference on the car is read only once the C8b model is
locked; Liam "Chi ghi so").

  python tools/s02_video_metrics.py --frames <frames.csv> --log <esp32 log csv> --out <run>_plant.csv
Last line: #ket_qua,ma=0 pose written / 1 not ok (clock) / 2 error.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c7_measure                                                        # noqa: E402
import o1_time_sync                                                      # noqa: E402
from c7_score import T_MISS                                              # noqa: E402
from s02_plant_ticks import PLANT_COLUMNS                                # noqa: E402

COLUMNS = list(PLANT_COLUMNS) + ['cam_lost']
NAN = float('nan')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_log_rows(path):
    """Data lines of the ESP32 log (o1_time_sync.read_log); lines repeated byte for byte (UDP duplicates, spec P2-5)
    dropped; timestamp_us must then increase strictly."""
    rows, seen = [], set()
    for r in o1_time_sync.read_log(path):
        key = tuple(r.items())
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    t = np.array([float(r['timestamp_us']) for r in rows])
    if np.any(np.diff(t) <= 0):
        raise ValueError(f'{path}: timestamp_us does not increase')
    return rows


def camera_series(frames):
    """(t, x, y, psi) of the valid frames with a finite pose, psi unwrapped over them (P2-3: angles unwrapped before
    interpolation)."""
    t = np.array([f['t_s'] for f in frames], dtype=float)
    x = np.array([f['x'] for f in frames], dtype=float)
    y = np.array([f['y'] for f in frames], dtype=float)
    psi = np.array([f['psi'] for f in frames], dtype=float)
    ok = np.array([bool(f['valid']) for f in frames]) & np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & \
        np.isfinite(psi)
    t, x, y, psi = t[ok], x[ok], y[ok], psi[ok]
    if np.any(np.diff(t) <= 0):
        raise ValueError('frame times t_s do not increase')
    return t, x, y, np.unwrap(psi)


def pose_at(tq, t, x, y, psi, t_miss=T_MISS):
    """Camera pose at the query times tq (camera clock): (x, y, psi, lost). Linear between the two valid frames
    around a query (a query on a frame takes it); none outside [t[0], t[-1]] (lost NaN); none between two frames
    more than t_miss apart (lost 1)."""
    tq = np.asarray(tq, dtype=float)
    n = len(tq)
    X, Y, P, L = (np.full(n, NAN) for _ in range(4))
    if t.size == 0:
        return X, Y, P, L
    inside = (tq >= t[0]) & (tq <= t[-1])
    j = np.clip(np.searchsorted(t, tq, side='right') - 1, 0, max(t.size - 2, 0))
    if t.size == 1:
        on = inside                                        # tq == t[0]
        X[on], Y[on], P[on], L[on] = x[0], y[0], psi[0], 0.0
        return X, Y, P, L
    t0, t1 = t[j], t[j + 1]
    gap = t1 - t0
    lost = inside & (gap > t_miss) & (tq != t0) & (tq != t1)
    good = inside & ~lost
    w = np.where(good, (tq - t0) / np.where(gap > 0, gap, 1.0), 0.0)
    for out, v in ((X, x), (Y, y), (P, psi)):
        out[good] = (v[j] + w * (v[j + 1] - v[j]))[good]
    L[inside] = 0.0
    L[lost] = 1.0
    return X, Y, P, L


def _sync_dict(r):
    if r is None:
        return None
    return dict(offset_s=r.offset_s, corr=r.corr, offset_start_s=r.offset_start_s, offset_end_s=r.offset_end_s,
                start_end_diff_s=r.start_end_diff_s, short=bool(r.short), window_cam_s=list(r.window_cam_s),
                n_cam=int(r.n_cam), window_src=r.window_src, channel=r.channel, ok=bool(r.ok), why=r.why)


def s02_window(rows, margin=o1_time_sync.LOG_MARGIN_S):
    """(t_a, t_b) of the fit on the log clock: o1_time_sync.log_window (its checks: one run_active rising edge, F-5;
    log_seq continuous, F-11), started at the first non-zero command - margin instead of the rising edge: an S02 run
    sends zero commands through its preparation (spec P2-2b) with run_active up, so a window from the edge holds
    seconds of rest and its first third cannot be fitted."""
    w = o1_time_sync.log_window(rows, margin)
    t = [float(r['timestamp_us']) / 1e6 for r in rows]
    cmd = [i for i, r in enumerate(rows) if any(abs(float(r.get(c) or 0.0)) > 0 for c in ('sp_vx', 'sp_vy', 'sp_wz'))]
    if not cmd:
        raise ValueError('no non-zero command in the log')
    t_a, t_b = t[cmd[0]] - margin, t[cmd[-1]] + margin
    return (t_a, t_b) if w is None else (max(w[0], t_a), w[1])


def clock(frames, rows):
    """(speed fit or None, gyro fit or None, reason) on the camera frames and the log rows (module docstring 1)."""
    t_cam = np.array([f['t_s'] for f in frames], dtype=float)
    val = np.array([bool(f['valid']) for f in frames])
    sp = np.where(val, [f['speed'] for f in frames], np.nan)
    coarse = dict(coarse_cam=(t_cam, sp), coarse_log=o1_time_sync.log_speed(rows))
    try:
        win = s02_window(rows)
        tl, vl = o1_time_sync.log_channel(rows, 'speed')
        r = o1_time_sync.sync(t_cam, sp, tl, vl, window_log=win, channel='speed', **coarse)
    except ValueError as exc:
        return None, None, f'clock_sync_failed ({exc})'
    g = None
    if 'gz_fused' in rows[0]:
        try:
            wz = np.where(val, [f['wz'] for f in frames], np.nan)
            tg, gz = o1_time_sync.log_channel(rows, 'gyro')
            g = o1_time_sync.sync(t_cam, wz, tg, gz, window_log=win, channel='gyro', **coarse)
        except ValueError:
            g = None
    return r, g, ('' if r.ok else f'clock_sync_failed ({r.why})')


def map_video(frames, rows, t_miss=T_MISS):
    """(info, plant rows) of one run (module docstring). The rows are lists in COLUMNS order, empty strings for
    the cells without a value; none when the clock is not ok."""
    info = dict(ok=False, reason='', t_miss_s=t_miss, offset_s=None, sync=None, gyro=None, gyro_minus_speed_s=None)
    r, g, why = clock(frames, rows)
    info['sync'], info['gyro'] = _sync_dict(r), _sync_dict(g)
    if r is not None and g is not None and math.isfinite(g.offset_s):
        info['gyro_minus_speed_s'] = g.offset_s - r.offset_s
    if why:
        info['reason'] = why
        return info, []
    D = float(r.offset_s)
    info['offset_s'] = D
    t, x, y, psi = camera_series(frames)
    ts = np.array([int(float(q['timestamp_us'])) for q in rows], dtype=np.int64)
    X, Y, P, L = pose_at(ts * 1e-6 - D, t, x, y, psi, t_miss)
    gaps = np.diff(t)
    info.update(ok=True, n_rows=int(ts.size), n_pose=int(np.sum(np.isfinite(X))), n_lost=int(np.sum(L == 1)),
                n_outside=int(np.sum(np.isnan(L))), n_valid_frames=int(t.size),
                largest_gap_s=float(gaps.max()) if gaps.size else None,
                cam_span_s=[float(t[0]), float(t[-1])] if t.size else None)
    out = []
    for k in range(ts.size):
        row = {c: '' for c in COLUMNS}
        row['timestamp_us'] = int(ts[k])
        if math.isfinite(X[k]):
            row['x'], row['y'], row['theta'] = f'{X[k]:.6f}', f'{Y[k]:.6f}', f'{P[k]:.6f}'
        if math.isfinite(L[k]):
            row['cam_lost'] = int(L[k])
        out.append([row[c] for c in COLUMNS])
    return info, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', required=True, help='frames.csv of tools/c7_measure.py track')
    ap.add_argument('--log', required=True, help='ESP32 log of the run (agv_logger CSV)')
    ap.add_argument('--out', required=True, help='<run>_plant.csv (new file); the info goes to <run>_plant_video.json')
    a = ap.parse_args(argv)
    js = os.path.splitext(a.out)[0] + '_video.json'
    for p in (a.out, js):
        if os.path.exists(p):
            print(f'{p} already exists: outputs are not overwritten')
            print('#ket_qua,ma=2')
            return 2
    try:
        frames = c7_measure.read_frames(a.frames)
        rows = read_log_rows(a.log)
        info, out = map_video(frames, rows)
    except (OSError, ValueError, KeyError) as exc:
        print(f'error: {exc}')
        print('#ket_qua,ma=2')
        return 2
    tj = os.path.join(os.path.dirname(os.path.abspath(a.frames)), 'track.json')
    info['inputs'] = dict(frames=os.path.abspath(a.frames), frames_sha256=sha256(a.frames),
                          track_json_sha256=sha256(tj) if os.path.isfile(tj) else None,
                          log=os.path.abspath(a.log), log_sha256=sha256(a.log))
    with open(a.out, 'x', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(out)
    with open(js, 'x', encoding='utf-8') as f:
        json.dump(info, f, indent=1, default=float)
    print(json.dumps({k: info.get(k) for k in ('ok', 'reason', 'offset_s', 'n_lost', 'largest_gap_s')}, default=float))
    code = 0 if info['ok'] else 1
    print(f'#ket_qua,ma={code}')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
