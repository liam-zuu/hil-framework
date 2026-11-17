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
   but started at the run's first non-zero command (s02_window, run_commands: from the run_active edge, motor on;
   the zero commands of the S02 preparation). Not ok -> no pose, reason clock_sync_failed: the o1_time_sync result
   not ok (correlation below MIN_CORR, |D - mean(start, end)| over SPLIT_TOL_S, no fit); the start and end thirds
   more than SPLIT_TOL_S (50 ms) apart (S2V1-1: spec P2-3 line 137); a gap over t_gap (t_trong_s of the C8
   constants file, --hang-so, read by c7_score.load_hang_so; O1 O-3) between valid frames within [-0.2 s, +1.0 s] of the run's first or last non-zero command
   (S2V1-2, Liam 17/11 "Doi camera phu hai canh (Recommended)": the start and the stop carry the timing; without
   them a short window fits a wrong D with a high correlation). The gyro channel (camera wz against gz_fused) is
   fitted too and only reported (cross-check, Liam "gyro kiem cheo").
2. Pose on the log grid (P2-3 line 135): for every data line of the log (byte-for-byte duplicates dropped, sorted
   by timestamp_us), the camera pose at t_cam = t_log - D, linear between the two valid frames around it, psi
   unwrapped over the valid frames first; never extrapolated. Two valid frames more than t_miss apart
   (t_mat_s of the C8 constants file, O1 3.3; Liam 17/11 "0,68 s nhu O1": the O1 number, locked at C8), or a line
   more than t_miss before the first or after the
   last valid frame (S2V1-7): no pose and cam_lost = 1 (P2-5 line 170: the measuring system failed, group 1);
   s02_metrics refuses the family R of a run with such a line inside its metric window (camera_lost). Lines just
   outside the camera span: no pose, cam_lost empty.
3. Outputs, never overwritten: <run>_plant.csv with the columns of a plant file (PLANT_COLUMNS of s02_plant_ticks;
   only timestamp_us, x, y, theta filled) and cam_lost, so s02_metrics.score_run reads it as for SIM and HIL;
   <run>_plant_cam.csv, the valid frames on the log clock (timestamp_us, x, y, theta unwrapped as in the plant
   rows); <run>_plant_video.json (ok, reason, the fit, the edge gaps, the gyro cross-check, the largest gap between
   valid frames, counts, SHA-256 of the inputs (the constants file too, with its version and trang_thai) and of the
   two files, S2V1-3: s02_metrics checks them).

The rest pose p(0), psi(0) (mean camera pose over the 0.5 s before t = 0, P2-3 line 155, Liam 17/11 "0,5 s truoc
t = 0", "Tinh tren khung camera (Recommended)") is taken by s02_metrics on the frames of <run>_plant_cam.csv: only
score_run knows t = 0 (the two command anchors). No measurement uncertainty label (P2-3 line 157: the sign of a
family-R difference on the car is read only once the C8b model is locked; Liam "Chi ghi so").

  python tools/s02_video_metrics.py --frames <frames.csv> --log <esp32 log csv> --hang-so <hang_so.json>
                                    --out <run>_plant.csv
The thresholds t_mat_s, t_trong_s come only from the C8 constants file (fix/s02-video-hang-so, CODE 17/11: the
c7_score constants T_MISS, T_GAP were removed when the figure-8 scorer moved them into that file).
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
import s02_metrics                                                       # noqa: E402
import o1_time_sync                                                      # noqa: E402
import c7_score                                                          # noqa: E402
from s02_plant_ticks import PLANT_COLUMNS                                # noqa: E402

COLUMNS = list(PLANT_COLUMNS) + ['cam_lost']
CAM_COLUMNS = ['timestamp_us', 'x', 'y', 'theta']    # <run>_plant_cam.csv: the valid frames on the log clock
EDGE_BEFORE_S, EDGE_AFTER_S = 0.2, 1.0               # S2V1-2: span around the first and last non-zero command
NAN = float('nan')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_log_rows(path):
    """Data lines of the ESP32 log as dicts of text, in the order s02_metrics.load_log uses (S2V1-5, S2V2-3, Codex
    rounds 1, 2: the same DUT checks as the scorer): a logger header (s02_metrics.HEADS); a DUT log (log_seq,
    motor_on) through s02_metrics.dut_rows (duplicates dropped, log_seq order, the logger segment or the sample-time
    rule against a restart or conflicting lines); an older log without log_seq: byte-for-byte duplicates dropped,
    sorted by timestamp_us. timestamp_us must then increase strictly. path may be a pathlib.Path (S2V3-2)."""
    path = os.fspath(path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith('#')]
    if not rows:
        raise ValueError(f'{path}: empty')
    head, body = rows[0], rows[1:]
    if head not in s02_metrics.HEADS:
        raise ValueError(f'{path}: header is not a logger format')
    if any(len(r) != len(head) for r in body):
        raise ValueError(f'{path}: a line does not have {len(head)} columns')
    if 'log_seq' in head and 'motor_on' in head:
        s, mo = head.index('log_seq'), head.index('motor_on')
        kinds = {r[s] != '' and r[mo] != '' for r in body}
        if len(kinds) > 1:
            raise ValueError(f'{path}: lines with and without log_seq / motor_on')
        if kinds == {True}:
            body, _ = s02_metrics.dut_rows(body, path)
    if 'log_seq' not in head or not body or body[0][head.index('log_seq')] == '':
        uniq = list(dict.fromkeys(tuple(r) for r in body))
        body = sorted(uniq, key=lambda r: float(r[head.index('timestamp_us')]))
    out = [dict(zip(head, r)) for r in body]
    t = np.array([float(r['timestamp_us']) for r in out])
    if not out or np.any(np.diff(t) <= 0):
        raise ValueError(f'{path}: no data line, or timestamp_us does not increase')
    return out


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


def pose_at(tq, t, x, y, psi, t_miss):
    """Camera pose at the query times tq (camera clock): (x, y, psi, lost). Linear between the two valid frames
    around a query (a query on a frame takes it); none between two frames more than t_miss apart (lost 1); none
    outside [t[0], t[-1]] (never extrapolated): lost 1 when the nearest valid frame is more than t_miss away
    (S2V1-7, Codex round 1: a video that starts late or ends early is a camera loss too), else NaN."""
    tq = np.asarray(tq, dtype=float)
    n = len(tq)
    X, Y, P, L = (np.full(n, NAN) for _ in range(4))
    if t.size == 0:
        return X, Y, P, L
    inside = (tq >= t[0]) & (tq <= t[-1])
    L[~inside & s02_metrics.too_wide(np.maximum(t[0] - tq, tq - t[-1]), t_miss)] = 1.0
    j = np.clip(np.searchsorted(t, tq, side='right') - 1, 0, max(t.size - 2, 0))
    if t.size == 1:
        on = inside                                        # tq == t[0]
        X[on], Y[on], P[on], L[on] = x[0], y[0], psi[0], 0.0
        return X, Y, P, L
    t0, t1 = t[j], t[j + 1]
    gap = t1 - t0
    lost = inside & s02_metrics.too_wide(gap, t_miss) & (tq != t0) & (tq != t1)
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


def run_commands(rows):
    """Times (s, log clock) of the run's lines with a non-zero command (sp_vx, sp_vy, sp_wz): from the run_active
    rising edge on (one edge, else ValueError as o1_time_sync.log_window), motor_on = 1 when logged, as the command
    anchors of s02_cmd_path (S2V1-6, Codex round 1: a stray command before the run is not the run)."""
    i0 = 0
    if 'run_active' in rows[0]:
        ra = [r['run_active'] == '1' for r in rows]
        rises = [i for i in range(1, len(ra)) if ra[i] and not ra[i - 1]]
        if len(rises) != 1:
            raise ValueError(f'{len(rises)} run_active rising edges in the log (one file = one run, F-5)')
        i0 = rises[0]
    mo = 'motor_on' in rows[0]
    t = [float(rows[i]['timestamp_us']) / 1e6 for i in range(i0, len(rows))
         if (not mo or float(rows[i]['motor_on'] or 0.0) == 1.0)
         and any(abs(float(rows[i].get(c) or 0.0)) > 0 for c in ('sp_vx', 'sp_vy', 'sp_wz'))]
    if not t:
        raise ValueError('no non-zero command in the run')
    return t


def s02_window(rows, margin=o1_time_sync.LOG_MARGIN_S):
    """(t_a, t_b) of the fit on the log clock: o1_time_sync.log_window (its checks: one run_active rising edge, F-5;
    log_seq continuous, F-11), started at the run's first non-zero command - margin instead of the rising edge: an
    S02 run sends zero commands through its preparation (spec P2-2b) with run_active up, so a window from the edge
    holds seconds of rest and its first third cannot be fitted."""
    w = o1_time_sync.log_window(rows, margin)
    cmd = run_commands(rows)
    t_a, t_b = cmd[0] - margin, cmd[-1] + margin
    return (t_a, t_b) if w is None else (max(w[0], t_a), w[1])


def edge_gaps(t_valid, D, rows):
    """S2V1-2 (Codex round 1; Liam 17/11 "Doi camera phu hai canh (Recommended)"): the largest gap between valid
    frames (camera clock t_valid) over [edge - EDGE_BEFORE_S, edge + EDGE_AFTER_S] around the run's first and last
    non-zero command (the start and the stop carry the timing of the fit; a gap there lets a high correlation sit at
    a wrong D), between real frames only (s02_metrics.largest_gap; S2V2-2: a gap across the span ends counts whole):
    {'start': s, 'stop': s}."""
    cmd = run_commands(rows)
    return {name: s02_metrics.largest_gap(t_valid, e - D - EDGE_BEFORE_S, e - D + EDGE_AFTER_S)
            for name, e in (('start', cmd[0]), ('stop', cmd[-1]))}


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
    if not r.ok:
        return r, g, f'clock_sync_failed ({r.why})'
    # S2V1-1 (Codex round 1): spec S02 P2-3 line 137 refuses a start/end difference over 50 ms (o1_time_sync checks
    # |D - mean(start, end)|, another number); a short window has none (S2V1-2 covers it)
    if not r.short and math.isfinite(r.start_end_diff_s) and abs(r.start_end_diff_s) > o1_time_sync.SPLIT_TOL_S:
        return r, g, (f'clock_sync_failed (start/end {r.start_end_diff_s * 1000:.0f} ms > '
                      f'{o1_time_sync.SPLIT_TOL_S * 1000:.0f} ms)')
    return r, g, ''


def map_video(frames, rows, t_miss, t_gap):
    """(info, plant rows, camera rows) of one run (module docstring); t_miss, t_gap: t_mat_s, t_trong_s of the C8
    constants file (no number kept here: they are locked at C8 with O1). Plant rows: lists in COLUMNS order, empty
    strings for the cells without a value; camera rows: the valid frames on the log clock (CAM_COLUMNS; S2V1-4: the
    rest pose is their mean); none when the clock is not ok."""
    info = dict(ok=False, reason='', t_miss_s=t_miss, t_gap_s=t_gap, offset_s=None, sync=None, gyro=None,
                gyro_minus_speed_s=None, edge_gaps_s=None)
    r, g, why = clock(frames, rows)
    info['sync'], info['gyro'] = _sync_dict(r), _sync_dict(g)
    if r is not None and g is not None and math.isfinite(g.offset_s):
        info['gyro_minus_speed_s'] = g.offset_s - r.offset_s
    if why:
        info['reason'] = why
        return info, [], []
    D = float(r.offset_s)
    t, x, y, psi = camera_series(frames)
    eg = edge_gaps(t, D, rows)
    info['edge_gaps_s'] = eg
    bad = [f'{k} {v:.2f} s' for k, v in eg.items() if s02_metrics.too_wide(v, t_gap)]
    if bad:
        info['reason'] = f'clock_sync_failed (edges_not_covered: {", ".join(bad)} > {t_gap:.2f} s)'
        return info, [], []
    info['offset_s'] = D
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
    cam = [[f'{(t[k] + D) * 1e6:.1f}', f'{x[k]:.6f}', f'{y[k]:.6f}', f'{psi[k]:.6f}'] for k in range(t.size)]
    return info, out, cam


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', required=True, help='frames.csv of tools/c7_measure.py track')
    ap.add_argument('--log', required=True, help='ESP32 log of the run (agv_logger CSV)')
    ap.add_argument('--out', required=True, help='<run>_plant.csv (new file); the camera frames go to '
                    '<run>_plant_cam.csv, the info to <run>_plant_video.json')
    ap.add_argument('--hang-so', required=True, help='the C8 constants file (c7_hang_so_v1, as tools/c7_score.py): '
                    't_mat_s (camera loss) and t_trong_s (edge coverage)')
    a = ap.parse_args(argv)
    js = os.path.splitext(a.out)[0] + '_video.json'
    cp = os.path.splitext(a.out)[0] + '_cam.csv'
    for p in (a.out, cp, js):
        if os.path.exists(p):
            print(f'{p} already exists: outputs are not overwritten')
            print('#ket_qua,ma=2')
            return 2
    try:
        frames = c7_measure.read_frames(a.frames)
        rows = read_log_rows(a.log)
        hs = c7_score.load_hang_so(a.hang_so)
        info, out, cam = map_video(frames, rows, hs['t_mat_s'], hs['t_trong_s'])
    except (OSError, ValueError, KeyError) as exc:
        print(f'error: {exc}')
        print('#ket_qua,ma=2')
        return 2
    tj = os.path.join(os.path.dirname(os.path.abspath(a.frames)), 'track.json')
    info['inputs'] = dict(frames=os.path.abspath(a.frames), frames_sha256=sha256(a.frames),
                          track_json_sha256=sha256(tj) if os.path.isfile(tj) else None,
                          log=os.path.abspath(a.log), log_sha256=sha256(a.log),
                          hang_so=os.path.abspath(a.hang_so), hang_so_sha256=hs['sha256'],
                          hang_so_version=hs['version'], hang_so_trang_thai=hs['trang_thai'])
    for path, cols, data in ((a.out, COLUMNS, out), (cp, CAM_COLUMNS, cam)):
        with open(path, 'x', newline='') as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(data)
    # S2V1-3 (Codex round 1): the json binds the log and the two files it vouches for; s02_metrics checks them
    info['plant_sha256'], info['cam_sha256'] = sha256(a.out), sha256(cp)
    with open(js, 'x', encoding='utf-8') as f:
        json.dump(info, f, indent=1, default=float)
    print(json.dumps({k: info.get(k) for k in ('ok', 'reason', 'offset_s', 'n_lost', 'largest_gap_s')}, default=float))
    code = 0 if info['ok'] else 1
    print(f'#ket_qua,ma={code}')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
