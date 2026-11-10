#!/usr/bin/env python3
"""
o1_time_sync.py: clock offset between the video (camera) and the vehicle log, from the speed
profiles (C7b method A, L77: Liam 10/11 CODE local_796cfeea "A: khop toc do (Recommended)";
CLAUDE_THESIS 03_CODE/docs/2026-11-10_De_xuat_C7b_ghep_dong_ho.md). No firmware change.

The master clock is the ESP32 (log column timestamp_us): the command and the command pose are
in the log on that clock. The video clock is the frame presentation time (PTS) of the file,
not frame index / nominal fps (a phone may change its frame rate). We look for the offset D with

    t_log = t_cam + D

that makes the speed seen by the camera, v_cam(t_cam), match the vehicle's speed in the log,
v_log(t_log) = hypot(vx, vy) of the encoder columns. The figure 8 changes its path speed all the
time (O1 3.1), so the profile has many features. Matching uses the Pearson correlation of the two
profiles on a common grid, so a scale error (wheel slip makes the encoder speed differ from the
true speed) does not move the optimum much; the peak is refined below one grid step with a
parabola.

O1 section 4 "synchronise at the start and at the end of a run": sync() also fits D on the first
and the last third of the moving window and reports both and their difference (end - start). A
difference above a threshold (fixed at the camera trial C8, not here) makes the run a run spoilt
by the measurement system (O1 3.5). The accuracy of D on real video is measured at C8; the tests
here use synthetic profiles with a known D and a real HIL log.

  python tools/o1_time_sync.py --log <agv_logger csv> --cam <csv: t_s,speed_mps>   [--json out]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass

import numpy as np

GRID_DT = 0.002          # s, common grid of the correlation
MOVE_V = 0.05            # m/s, "moving" threshold for the motion window and the coarse guess
SEARCH_S = 2.0           # s, half width of the search around the coarse guess
MIN_OVERLAP_S = 3.0      # s, a fit needs at least this much overlap


@dataclass
class SyncResult:
    offset_s: float          # D over the whole moving window: t_log = t_cam + D
    corr: float              # Pearson correlation at D
    offset_start_s: float    # D on the first third of the moving window
    corr_start: float
    offset_end_s: float      # D on the last third
    corr_end: float
    start_end_diff_s: float  # offset_end_s - offset_start_s
    coarse_s: float          # guess from the motion onsets
    window_cam_s: tuple      # (t0, t1) of the moving window on the camera clock
    n_cam: int               # camera samples with a finite speed in the window


def log_speed(path):
    """(t [s, ESP32 clock], |v| [m/s]) from an agv_logger CSV: hypot of the encoder vx, vy."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith('#')))
    if not rows:
        raise ValueError(f'{path}: no data line')
    t = np.array([float(r['timestamp_us']) for r in rows]) / 1e6
    v = np.hypot([float(r['vx']) for r in rows], [float(r['vy']) for r in rows])
    return t, v


def _onset(t, v, thr=MOVE_V, hold=0.2):
    """First time the speed stays above thr for hold seconds."""
    above = np.nan_to_num(v, nan=0.0) > thr
    for i in np.flatnonzero(above):
        j = np.searchsorted(t, t[i] + hold)
        if j <= len(t) and above[i:j].all():
            return float(t[i])
    raise ValueError('the speed never stays above the moving threshold')


def _moving_window(t, v, thr=MOVE_V):
    m = np.nan_to_num(v, nan=0.0) > thr
    if not m.any():
        raise ValueError('no moving sample')
    return float(t[np.argmax(m)]), float(t[len(m) - 1 - np.argmax(m[::-1])])


def _corr_at(D, tg, vc_g, t_log, v_log):
    """Pearson correlation of v_cam on the grid tg (camera clock) with v_log at tg + D."""
    x = np.interp(tg + D, t_log, v_log, left=np.nan, right=np.nan)
    ok = np.isfinite(x) & np.isfinite(vc_g)
    if ok.sum() < MIN_OVERLAP_S / GRID_DT:
        return -np.inf
    a, b = vc_g[ok] - vc_g[ok].mean(), x[ok] - x[ok].mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / den if den > 0 else -np.inf


def _fit(t_cam, v_cam, t_log, v_log, t0, t1, guess, half=SEARCH_S):
    """Best D in [guess - half, guess + half]: grid scan, then a parabola on the peak."""
    tg = np.arange(t0, t1, GRID_DT)
    fin = np.isfinite(v_cam)
    vc_g = np.interp(tg, t_cam[fin], v_cam[fin])
    # camera gaps (lost frames) stay gaps: a grid point farther than 2 frame steps from a sample is NaN
    near = np.searchsorted(t_cam[fin], tg)
    lo = np.abs(tg - t_cam[fin][np.clip(near - 1, 0, fin.sum() - 1)])
    hi = np.abs(t_cam[fin][np.clip(near, 0, fin.sum() - 1)] - tg)
    step = np.median(np.diff(t_cam[fin])) if fin.sum() > 1 else GRID_DT
    vc_g[np.minimum(lo, hi) > 2.0 * step] = np.nan
    Ds = guess + np.arange(-half, half + GRID_DT / 2, GRID_DT)
    cs = np.array([_corr_at(D, tg, vc_g, t_log, v_log) for D in Ds])
    k = int(np.argmax(cs))
    if not np.isfinite(cs[k]):
        raise ValueError('no overlap between the camera window and the log')
    D = float(Ds[k])
    if 0 < k < len(cs) - 1 and np.all(np.isfinite(cs[k - 1:k + 2])):
        y0, y1, y2 = cs[k - 1], cs[k], cs[k + 1]
        den = y0 - 2.0 * y1 + y2
        if den < 0:
            D += 0.5 * (y0 - y2) / den * GRID_DT
    return D, float(cs[k])


def sync(t_cam, v_cam, t_log, v_log, half=SEARCH_S):
    """Offset D (t_log = t_cam + D) from the speed profiles, with the start/end check (O1 4)."""
    t_cam, v_cam = np.asarray(t_cam, float), np.asarray(v_cam, float)
    t_log, v_log = np.asarray(t_log, float), np.asarray(v_log, float)
    if np.any(np.diff(t_cam) <= 0) or np.any(np.diff(t_log) <= 0):
        raise ValueError('times must increase')
    coarse = _onset(t_log, v_log) - _onset(t_cam, v_cam)
    w0, w1 = _moving_window(t_cam, v_cam)
    if w1 - w0 < 3 * MIN_OVERLAP_S:
        raise ValueError(f'moving window {w1 - w0:.1f} s too short (needs {3 * MIN_OVERLAP_S:.0f} s)')
    D, c = _fit(t_cam, v_cam, t_log, v_log, w0, w1, coarse, half)
    third = (w1 - w0) / 3.0
    Ds, cs_ = _fit(t_cam, v_cam, t_log, v_log, w0, w0 + third, D, min(half, 0.5))
    De, ce = _fit(t_cam, v_cam, t_log, v_log, w1 - third, w1, D, min(half, 0.5))
    n = int(np.sum(np.isfinite(v_cam) & (t_cam >= w0) & (t_cam <= w1)))
    return SyncResult(D, c, Ds, cs_, De, ce, De - Ds, coarse, (w0, w1), n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', required=True, help='agv_logger CSV (ESP32 clock)')
    ap.add_argument('--cam', required=True, help='CSV with columns t_s (frame PTS) and speed_mps (NaN = lost)')
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    try:
        t_log, v_log = log_speed(a.log)
        with open(a.cam, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        t_cam = np.array([float(r['t_s']) for r in rows])
        v_cam = np.array([float(r['speed_mps']) if r['speed_mps'] not in ('', 'nan') else np.nan for r in rows])
        r = sync(t_cam, v_cam, t_log, v_log)
    except (ValueError, KeyError, OSError) as exc:
        print(f'#ket_qua,ma=2,loi={exc}')
        return 2
    print(f'offset {r.offset_s:.4f} s (corr {r.corr:.3f}); start {r.offset_start_s:.4f} (corr {r.corr_start:.3f}), '
          f'end {r.offset_end_s:.4f} (corr {r.corr_end:.3f}), end - start {1000 * r.start_end_diff_s:+.1f} ms')
    if a.json:
        json.dump(asdict(r), open(a.json, 'w', encoding='utf-8'), indent=2)
    print(f'#ket_qua,ma=0,offset_s={r.offset_s:.4f},start_end_ms={1000 * r.start_end_diff_s:+.1f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
