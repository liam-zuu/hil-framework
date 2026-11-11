#!/usr/bin/env python3
"""
o1_time_sync.py: clock offset between the video (camera) and the vehicle log, from the speed
profiles (C7b method A, L77: Liam 10/11 CODE local_796cfeea "A: khop toc do (Recommended)";
CLAUDE_THESIS 03_CODE/docs/2026-11-10_De_xuat_C7b_ghep_dong_ho.md). No firmware change.
Fixes of L86 Q4 (Liam 11/11 CODE local_821ee4dc "duyet Q1-8 theo khuyen nghi"; review
03_CODE/docs/review/2026-11-11_Tham_dinh_Fable_C9_C7b_C5.md section 2), written 11/11 CODE
local_5e745270 with the C7 chain ("oke viet luon di"): F-1/F-2 gross errors caught, F-3 short runs,
F-4 window from the log, F-5/F-11 log checks, F-6 signed channels, F-12 lost frames, Y-1 gyro
channel, Y-2 --half.

The master clock is the ESP32 (log column timestamp_us): the command and the command pose are
in the log on that clock. The video clock is the frame presentation time (PTS) of the file,
not frame index / nominal fps (a phone may change its frame rate). We look for the offset D with

    t_log = t_cam + D

that makes a camera profile match the same quantity in the log: the path speed hypot(vx, vy)
(figure 8), the SIGNED body velocities (vx, vy) together (K1, K2: the magnitude of K2 repeats every
side, F-6), or the yaw rate (gz_fused, the independent gyro channel of the design, Y-1). Matching
uses the Pearson correlation on a common 2 ms grid (several channels: concatenated), so a scale error
(wheel slip) does not move the optimum much; the peak is refined below one grid step with a parabola.

Gross errors (F-1, F-2): a peak on the edge of the search range is searched again in a range twice as
wide (twice); still on the edge -> error. The result is marked NOT ok (exit code 3) when the
correlation is below MIN_CORR, or when the offset of the whole window differs from the mean of the
start and end thirds by more than SPLIT_TOL_S [DE XUAT, both fixed at C8].
O1 section 4 "synchronise at the start and at the end of a run": the first and last thirds of the
window are fitted again; their difference (end - start) is reported. Below 3 x MIN_OVERLAP_S of
window (K1: 5 s) the run is SHORT (F-3): one fit over the whole window, the start/end check is not
possible (NaN, short = True); how such a run is classed is a spec question for THESIS (O1 3.5).
Window (F-4): from the log (run_active up to the last line with a non-zero command, ±margin) mapped
to the camera clock with the coarse guess, when the log columns are there; else from the camera
motion. The log must hold one run_active rising edge (F-5) and continuous log_seq in the window (F-11).
Conventions (F-7, F-10): the log speed is the mean over the 10 ms control period, about +5 ms late on
the line time; not subtracted until C8 confirms it. The camera speed must be centred (symmetric
derivative), unfiltered beyond that, and taken at the wheel centre (a tag offset of 10 mm sideways
moves D by about 8 ms at F08). Vibration at 3-5 Hz can fake a start/end difference (F-9).

  python tools/o1_time_sync.py --log <agv_logger csv> --cam <csv: t_s,speed_mps[,vx_body,vy_body,wz]>
      [--channel speed|body|gyro] [--half 2.0] [--json out]
Last line: #ket_qua,ma=0 ok / 3 not ok (corr or split) / 2 error.
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
MIN_CORR = 0.90          # [DE XUAT, chot o C8] below: result not ok
SPLIT_TOL_S = 0.050      # [DE XUAT, chot o C8] |D - mean(start, end)| above: result not ok
WIDEN = 2                # times the search range is doubled when the peak sits on its edge
LOG_MARGIN_S = 0.5       # s, window from the log: [run_active up - margin, last command + margin]


@dataclass
class SyncResult:
    offset_s: float          # D over the whole window: t_log = t_cam + D
    corr: float              # Pearson correlation at D
    offset_start_s: float    # D on the first third of the window (NaN when short)
    corr_start: float
    offset_end_s: float      # D on the last third (NaN when short)
    corr_end: float
    start_end_diff_s: float  # offset_end_s - offset_start_s (NaN when short)
    coarse_s: float          # guess from the motion onsets
    window_cam_s: tuple      # (t0, t1) of the window on the camera clock
    n_cam: int               # camera samples with a finite value in the window
    short: bool = False      # window below 3 x MIN_OVERLAP_S: no start/end check (F-3)
    window_src: str = 'camera'
    channel: str = 'speed'
    widened: int = 0         # times the search range was widened (F-1)
    ok: bool = True
    why: str = ''


# ---------------------------------------------------------------- log
def read_log(path):
    """Data lines of an agv_logger CSV as dicts (lines starting with '#' skipped)."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith('#')))
    rows = [r for r in rows if r.get('timestamp_us') not in (None, '')]
    if not rows:
        raise ValueError(f'{path}: no data line')
    return rows


def log_speed(path_or_rows):
    """(t [s, ESP32 clock], |v| [m/s]) from an agv_logger CSV or its rows: hypot of the encoder vx, vy."""
    rows = read_log(path_or_rows) if isinstance(path_or_rows, str) else path_or_rows
    t = np.array([float(r['timestamp_us']) for r in rows]) / 1e6
    v = np.hypot([float(r['vx']) for r in rows], [float(r['vy']) for r in rows])
    return t, v


def log_channel(rows, channel):
    """(t, values (n,) or (n, 2)) of the log for a channel: speed, body (vx, vy), gyro (gz_fused)."""
    t = np.array([float(r['timestamp_us']) for r in rows]) / 1e6
    if channel == 'speed':
        return t, np.hypot([float(r['vx']) for r in rows], [float(r['vy']) for r in rows])
    if channel == 'body':
        return t, np.column_stack([[float(r['vx']) for r in rows], [float(r['vy']) for r in rows]])
    if channel == 'gyro':
        return t, np.array([float(r['gz_fused']) for r in rows])
    raise ValueError(f'channel {channel!r}')


def log_window(rows, margin=LOG_MARGIN_S):
    """(t_a, t_b) on the log clock: run_active rising edge - margin .. last line with a non-zero command
    (sp_vx, sp_vy, sp_wz) + margin. One rising edge only (F-5); log_seq continuous in between (F-11).
    None when the log has no run_active column."""
    if 'run_active' not in rows[0]:
        return None
    ra = [r['run_active'] == '1' for r in rows]
    rises = [i for i in range(1, len(ra)) if ra[i] and not ra[i - 1]]
    if len(rises) != 1:
        raise ValueError(f'{len(rises)} run_active rising edges in the log (one file = one run, F-5)')
    i0 = rises[0]
    cmd = [i for i in range(i0, len(rows))
           if any(abs(float(rows[i].get(c) or 0.0)) > 0 for c in ('sp_vx', 'sp_vy', 'sp_wz'))]
    i1 = cmd[-1] if cmd else len(rows) - 1
    if 'log_seq' in rows[0]:
        seq = [int(float(rows[i]['log_seq'])) for i in range(i0, i1 + 1)]
        jumps = [b - a for a, b in zip(seq[:-1], seq[1:]) if b - a != 1]
        if any(j > 4 for j in jumps) or any(j <= 0 for j in jumps):
            # log_seq also counts the '#run_end' and motor-off lines (s05_final_table); a jump of a few is
            # those, a larger one is lost lines
            raise ValueError(f'log_seq not continuous in the run window (jumps {jumps[:5]}, F-11)')
    t = [float(r['timestamp_us']) / 1e6 for r in rows]
    return t[i0] - margin, t[i1] + margin


# ---------------------------------------------------------------- fit
def _onset(t, v, thr=MOVE_V, hold=0.2):
    """First time the value stays above thr for hold seconds; NaN samples (lost frames) are skipped,
    not taken as standing still (F-12)."""
    fin = np.isfinite(v)
    t, v = t[fin], v[fin]
    above = v > thr
    for i in np.flatnonzero(above):
        j = np.searchsorted(t, t[i] + hold)
        if j >= len(t):
            break
        if above[i:j + 1].all():
            return float(t[i])
    raise ValueError('the profile never stays above the moving threshold')


def _moving_window(t, v, thr=MOVE_V, hold=0.2):
    """From the first to the last time the value stays above thr for hold seconds (F-4: single
    samples at the file edges do not stretch it)."""
    t0 = _onset(t, v, thr, hold)
    tr = -t[::-1]
    t1 = -_onset(tr, v[::-1], thr, hold)
    return t0, t1


def _grid_cam(t_cam, v_cam, t0, t1):
    tg = np.arange(t0, t1, GRID_DT)
    v = v_cam if v_cam.ndim == 2 else v_cam[:, None]
    fin = np.all(np.isfinite(v), axis=1)
    tf = t_cam[fin]
    if len(tf) < 2:
        raise ValueError('no camera sample in the window')
    vg = np.column_stack([np.interp(tg, tf, v[fin, c]) for c in range(v.shape[1])])
    near = np.searchsorted(tf, tg)
    lo = np.abs(tg - tf[np.clip(near - 1, 0, len(tf) - 1)])
    hi = np.abs(tf[np.clip(near, 0, len(tf) - 1)] - tg)
    step = np.median(np.diff(tf))
    vg[np.minimum(lo, hi) > 2.0 * step] = np.nan          # camera gaps stay gaps
    return tg, vg


def _corr_at(D, tg, vg, t_log, v_log):
    """Pearson correlation of the camera grid with the log at tg + D (channels concatenated)."""
    vl = v_log if v_log.ndim == 2 else v_log[:, None]
    x = np.column_stack([np.interp(tg + D, t_log, vl[:, c], left=np.nan, right=np.nan) for c in range(vl.shape[1])])
    ok = np.isfinite(x) & np.isfinite(vg)
    if ok.sum() < MIN_OVERLAP_S / GRID_DT:
        return -np.inf
    a, b = vg[ok], x[ok]
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / den if den > 0 else -np.inf


def _fit(t_cam, v_cam, t_log, v_log, t0, t1, guess, half=SEARCH_S, widen=WIDEN):
    """Best D in [guess - half, guess + half]: grid scan, parabola on the peak. A peak on the edge of
    the range is searched again around that edge with the range doubled, up to `widen` times (F-1).
    Returns (D, corr, times widened)."""
    tg, vg = _grid_cam(t_cam, v_cam, t0, t1)
    for n in range(widen + 1):
        Ds = guess + np.arange(-half, half + GRID_DT / 2, GRID_DT)
        cs = np.array([_corr_at(D, tg, vg, t_log, v_log) for D in Ds])
        k = int(np.argmax(cs))
        if not np.isfinite(cs[k]):
            raise ValueError('no overlap between the camera window and the log')
        if 0 < k < len(cs) - 1:
            break
        guess, half = float(Ds[k]), 2.0 * half
    else:
        raise ValueError(f'correlation peak on the edge of the search range after widening {widen} times')
    D = float(Ds[k])
    if np.all(np.isfinite(cs[k - 1:k + 2])):
        y0, y1, y2 = cs[k - 1], cs[k], cs[k + 1]
        den = y0 - 2.0 * y1 + y2
        if den < 0:
            D += 0.5 * (y0 - y2) / den * GRID_DT
    return D, float(cs[k]), n


def sync(t_cam, v_cam, t_log, v_log, half=SEARCH_S, window_log=None, channel='speed',
         coarse_cam=None, coarse_log=None):
    """Offset D (t_log = t_cam + D) with the start/end check (O1 4).
    v_cam, v_log: (n,) or (n, c) on the same channel(s). window_log: (t_a, t_b) on the log clock (F-4).
    coarse_cam, coarse_log: (t, |v|) for the coarse guess when the channel is not a speed."""
    t_cam, v_cam = np.asarray(t_cam, float), np.asarray(v_cam, float)
    t_log, v_log = np.asarray(t_log, float), np.asarray(v_log, float)
    if np.any(np.diff(t_cam) <= 0) or np.any(np.diff(t_log) <= 0):
        raise ValueError('times must increase')
    mag_c = coarse_cam[1] if coarse_cam is not None else (np.linalg.norm(v_cam, axis=1) if v_cam.ndim == 2 else np.abs(v_cam))
    mag_l = coarse_log[1] if coarse_log is not None else (np.linalg.norm(v_log, axis=1) if v_log.ndim == 2 else np.abs(v_log))
    tc_c = coarse_cam[0] if coarse_cam is not None else t_cam
    tl_c = coarse_log[0] if coarse_log is not None else t_log
    coarse = _onset(tl_c, mag_l) - _onset(tc_c, mag_c)
    if window_log is not None:
        w0, w1 = window_log[0] - coarse, window_log[1] - coarse
        w0, w1 = max(w0, float(t_cam[0])), min(w1, float(t_cam[-1]))
        src = 'log'
    else:
        w0, w1 = _moving_window(tc_c, mag_c)
        src = 'camera'
    if w1 - w0 < MIN_OVERLAP_S:
        raise ValueError(f'window {w1 - w0:.1f} s too short (needs {MIN_OVERLAP_S:.0f} s)')
    D, c, nw = _fit(t_cam, v_cam, t_log, v_log, w0, w1, coarse, half)
    n = int(np.sum(np.all(np.isfinite(v_cam.reshape(len(t_cam), -1)), axis=1) & (t_cam >= w0) & (t_cam <= w1)))
    short = w1 - w0 < 3 * MIN_OVERLAP_S
    if short:
        r = SyncResult(D, c, float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), coarse, (w0, w1), n,
                       True, src, channel, nw)
    else:
        third = (w1 - w0) / 3.0
        try:
            Ds, cs_, _ = _fit(t_cam, v_cam, t_log, v_log, w0, w0 + third, D, min(half, 0.5), widen=0)
            De, ce, _ = _fit(t_cam, v_cam, t_log, v_log, w1 - third, w1, D, min(half, 0.5), widen=0)
        except ValueError as exc:                         # a third that does not fit is a gross error (F-1)
            nan = float('nan')
            r = SyncResult(D, c, nan, nan, nan, nan, nan, coarse, (w0, w1), n, False, src, channel, nw)
            r.ok, r.why = False, f'start/end third: {exc}'
            return r
        r = SyncResult(D, c, Ds, cs_, De, ce, De - Ds, coarse, (w0, w1), n, False, src, channel, nw)
    why = []
    if r.corr < MIN_CORR:
        why.append(f'corr {r.corr:.3f} < {MIN_CORR}')
    if not r.short and abs(r.offset_s - 0.5 * (r.offset_start_s + r.offset_end_s)) > SPLIT_TOL_S:
        why.append(f'|D - mean(start, end)| > {SPLIT_TOL_S * 1000:.0f} ms')
    r.ok, r.why = not why, '; '.join(why)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', required=True, help='agv_logger CSV (ESP32 clock)')
    ap.add_argument('--cam', required=True,
                    help='CSV with t_s (frame PTS) and speed_mps (NaN = lost); body: vx_body, vy_body; gyro: wz')
    ap.add_argument('--channel', choices=('speed', 'body', 'gyro'), default='speed')
    ap.add_argument('--half', type=float, default=SEARCH_S, help='half width of the search [s] (Y-2)')
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    try:
        rows = read_log(a.log)
        t_log, v_log = log_channel(rows, a.channel)
        with open(a.cam, newline='', encoding='utf-8') as f:
            cr = list(csv.DictReader(f))

        def col(name):
            return np.array([float(r[name]) if r.get(name) not in ('', 'nan', None) else np.nan for r in cr])
        t_cam = col('t_s')
        sp = col('speed_mps')
        v_cam = {'speed': sp, 'body': None, 'gyro': None}[a.channel]
        if a.channel == 'body':
            v_cam = np.column_stack([col('vx_body'), col('vy_body')])
        elif a.channel == 'gyro':
            v_cam = col('wz')
        r = sync(t_cam, v_cam, t_log, v_log, a.half, log_window(rows), a.channel,
                 coarse_cam=(t_cam, sp), coarse_log=log_speed(rows))
    except (ValueError, KeyError, OSError) as exc:
        print(f'#ket_qua,ma=2,loi={exc}')
        return 2
    se = 'short run, no start/end check' if r.short else (
        f'start {r.offset_start_s:.4f} (corr {r.corr_start:.3f}), end {r.offset_end_s:.4f} (corr {r.corr_end:.3f}), '
        f'end - start {1000 * r.start_end_diff_s:+.1f} ms')
    print(f'offset {r.offset_s:.4f} s (corr {r.corr:.3f}, {r.channel}, window from {r.window_src}); {se}')
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in asdict(r).items()}, f, indent=2)
    if not r.ok:
        print(f'#ket_qua,ma=3,offset_s={r.offset_s:.4f},loi={r.why}')
        return 3
    print(f'#ket_qua,ma=0,offset_s={r.offset_s:.4f},start_end_ms='
          f'{"nan" if r.short else f"{1000 * r.start_end_diff_s:+.1f}"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
