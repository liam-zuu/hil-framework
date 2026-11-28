"""Post-run evidence for a panel demo. All distances on the panel are millimetres."""
from __future__ import annotations

from bisect import bisect_left
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unicodedata

import panel_proto as pp

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
if str(TOOLS) not in sys.path:
    sys.path.append(str(TOOLS))
from run_validity import pos_reset_mid


def tool_cmd(script, *args):
    argv = [sys.executable, str(TOOLS / script), *map(str, args)]
    if os.name == 'posix':
        argv = ['nice', '-n', '10', *argv]
    return argv


def _number(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def read_dut(path):
    columns = {key: [] for key in ('timestamp_us', 'run_active', 'pos_reset_n', 'motor_on',
                                   'pos_x_ref', 'pos_y_ref', 'pos_x', 'pos_y')}
    with path.open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            for key, values in columns.items():
                values.append(_number(row, key))
    return columns


def dut_complete(dut):
    """True when the DUT log shows its run ending: a row with run_active 0 after the last row
    with run_active 1, in log order and in DUT time (timestamp_us later; Sol PDS-1, PDS-9,
    PDS2-3: a late UDP datagram from before the run is not an end). A log cut during the run
    (logger died, datagrams lost at the end) is not complete."""
    if not dut:
        return False
    active = dut.get('run_active') or []
    stamps = dut.get('timestamp_us') or []
    if len(stamps) != len(active):
        return False
    last = max((i for i, a in enumerate(active) if a == 1), default=None)
    if last is None or not math.isfinite(stamps[last]):
        return False
    return any(a == 0 and math.isfinite(t) and t > stamps[last]
               for a, t in zip(active[last + 1:], stamps[last + 1:]))


def _dut_points(dut):
    ref, estimate = [], []
    start = end = None
    if dut:
        for i, active in enumerate(dut['run_active']):
            us = dut['timestamp_us'][i]
            if active != 1 or not math.isfinite(us):
                continue
            if start is None:
                start = us
            end = (us - start) * 1000  # common relative time, ns
            for points, x_key, y_key in ((ref, 'pos_x_ref', 'pos_y_ref'), (estimate, 'pos_x', 'pos_y')):
                x, y = dut[x_key][i] * 1000, dut[y_key][i] * 1000
                if math.isfinite(x) and math.isfinite(y):
                    points.append((end, x, y))
    return ref, estimate, end


def _plant_points(path, end):
    points = []
    origin = None
    with path.open(encoding='ascii', newline='') as f:
        for row in csv.DictReader(f):
            if origin is None and row.get('phase') != 'RUN':
                continue
            try:
                ns = int(row['t_ns'])
            except (KeyError, TypeError, ValueError):
                continue
            x, y, th = (_number(row, key) for key in ('x', 'y', 'th'))
            if not all(map(math.isfinite, (x, y, th))):
                continue
            if origin is None:
                # No shared clock: approximate the DUT's first active row by the first
                # observed RUN pose. Its world heading is the DUT frame's zero heading.
                origin = ns, x, y
                c, s = math.cos(th), math.sin(th)
            t = ns - origin[0]
            if end is not None and t > end:
                break
            dx, dy = (x - origin[1]) * 1000, (y - origin[2]) * 1000
            points.append((t, c * dx + s * dy, -s * dx + c * dy))
    return points


def _errors(plant, ref):
    ordered = sorted(ref, key=lambda p: p[0])
    times = [p[0] for p in ordered]
    maximum = last = None
    for t, x, y in plant:
        if not times or not times[0] <= t <= times[-1]:
            continue
        i = bisect_left(times, t)
        if i and t - times[i-1] <= times[i] - t:
            i -= 1
        last = math.hypot(x - ordered[i][1], y - ordered[i][2])
        maximum = last if maximum is None else max(maximum, last)
    return maximum, last


def _chunks(points):
    if len(points) > 200:
        points = [points[i * (len(points) - 1) // 199] for i in range(200)]
    xy = tuple((round(x), round(y)) for _, x, y in points)
    return tuple(xy[i:i + pp.TRACK_POINTS_MAX] for i in range(0, len(xy), pp.TRACK_POINTS_MAX))


def trajectory(pose_file, dut):
    ref, estimate, end = _dut_points(dut)
    try:
        plant = _plant_points(pose_file, end)
    except FileNotFoundError:
        plant = []
    maximum, last = _errors(plant, ref)  # before decimation or int16 wire saturation
    if not dut_complete(dut):
        last = None  # no error at the end of a run the DUT log does not reach (Sol PDS-9)
    tracks = {series: _chunks(points) for series, points in
              ((pp.TR_REF, ref), (pp.TR_PLANT, plant), (pp.TR_DUT, estimate)) if points}
    return tracks, maximum, last


def _display_number(value, width):
    if value is None:
        return '-'
    digits = str(round(value))
    if len(digits) <= width:
        return digits
    mantissa, exponent = format(value, '.0e').split('e')
    compact = f'{mantissa}e{int(exponent)}'
    return compact if len(compact) <= width else '>9k'


def error_line(maximum, last):
    # Seven columns remain for both distances. Keep the end error exact first.
    end = _display_number(last, 4)
    return f'e_max {_display_number(maximum, 7 - len(end))} e_end {end}'


def timing_line(p99=None, maximum=None):
    prefix = f'p99 {_display_number(p99, 4)} max '
    return prefix + _display_number(maximum, pp.RESULT_LEN - len(prefix) - 2) + 'us'


def _short(reason):
    plain = unicodedata.normalize('NFKD', reason.replace('đ', 'd').replace('Đ', 'D'))
    return ' '.join(plain.encode('ascii', 'ignore').decode().split())[:22]


def inconclusive(reason, text=None):
    """text: English words for the panel (Liam 28/11: "dùng tiếng anh full nhé"); reason is kept
    whole in meta.json, whatever its language."""
    return dict(code=pp.CHK_INCONCLUSIVE, text=_short(text or reason), reason=reason)


def _decode(folder, cancel, timeout):
    frames = folder / 'frames.npz'
    if not frames.is_file():
        raise FileNotFoundError('missing frames.npz')
    out = folder / 'decode.json'  # s06_frames_decode --out takes a directory
    # No --esp32-events: that option reads the S06 logger's '<t_rx monotonic>	<payload>' lines,
    # agv_logger writes '<ISO time> <payload>' (Sol PDS-2). Without run marks the decoder counts
    # empty MISO frames over the whole file (after the first ANSWER_LAG): stricter, never cleaner.
    cmd = tool_cmd('s06_frames_decode.py', frames, '--out', out)
    if cancel.is_set():
        raise ValueError('check cancelled')
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8',
                          errors='replace', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}) as proc:
        deadline = time.monotonic() + timeout
        try:
            while True:
                if cancel.is_set():
                    raise ValueError('check cancelled')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError('decoder timeout')
                try:
                    stdout, stderr = proc.communicate(timeout=min(remaining, 0.25))
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
    lines = stdout.rstrip().splitlines()
    rc = proc.returncode
    if rc not in (0, 1, 3) or not lines or not re.fullmatch(fr'#ket_qua,ma={rc}(,.*)?', lines[-1]):
        raise ValueError(f'decoder exit {rc}: {stderr.strip()[-240:]}')
    summary = out / 'frames_summary.json'
    if not summary.is_file():
        raise ValueError(f'no frames_summary.json (decoder exit {rc}): {stderr.strip()[-240:]}')
    with summary.open(encoding='utf-8') as f:
        return rc, json.load(f)


def run_check(folder, fault, logger_ok, dut, cancel, timeout=120):
    """logger_ok: the UDP logger was alive until the daemon stopped it and closed cleanly (exit 0).
    A clean verdict needs that and a DUT log that reaches the end of the run (dut_complete);
    a proven bad run (bad frames, a pose reset seen) stays BAD without them."""
    reset = reset_error = None
    complete = bool(logger_ok) and dut_complete(dut)
    try:
        if dut:
            # All logger rows, including the inactive tail that can reveal a lost reset.
            reset = pos_reset_mid(dut['run_active'], dut['pos_reset_n'], dut['motor_on'])
    except ValueError as exc:
        reset_error = str(exc)
    details = dict(pos_reset_mid=reset, pos_reset_error=reset_error, dut_log_complete=complete,
                   spifix='not checked')
    other = []
    try:
        rc, summary = _decode(folder, cancel, timeout)
        problems, conditions = summary['problems'], summary['conditions']
        empty, crc = (summary['frames'][key] for key in ('empty_in_run', 'miso_crc_bad'))
        if (not isinstance(problems, list) or not isinstance(conditions, list)
                or not all(isinstance(s, str) for s in problems + conditions)
                or any(type(v) is not int or v < 0 for v in (empty, crc))
                or rc != (1 if problems else 3 if conditions else 0)):
            raise ValueError('invalid decoder summary')
        details.update(decoder_exit=rc, empty_in_run=empty, miso_crc_bad=crc,
                       problems=problems, conditions=conditions)
        # English words for the panel; the decoder's own (Vietnamese) lines stay in meta.json
        bad, bad_text = '', ''
        if empty:
            bad = bad_text = f'empty_in_run {empty}'
        elif crc:
            bad = bad_text = f'miso_crc_bad {crc}'
        if fault:
            # The decoder excludes the first ANSWER_LAG replies in empty_in_run.
            # Its problems() lists these two frame counters first when nonzero.
            other = problems[int(bool(empty)) + int(bool(crc)):] + conditions
        elif rc == 1 and not bad:
            bad, bad_text = problems[0], f'decoder problems {len(problems)}'
        if reset_error:
            verdict = inconclusive(reset_error, 'pos_reset invalid')
        elif bad or reset:
            reason = bad or f'pos_reset_mid {reset}'
            verdict = dict(code=pp.CHK_BAD, text=_short(bad_text or reason), reason=reason)
        elif reset is None:
            verdict = inconclusive('no pos_reset')
        elif not complete:
            verdict = inconclusive('DUT log incomplete: logger stopped early, failed, or no row after the run')
        elif not fault and rc == 3:
            verdict = inconclusive(conditions[0], f'decoder conditions {len(conditions)}')
        else:
            verdict = dict(code=pp.CHK_CLEAN_NO_SPIFIX, text='SPIfix not checked')
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        # fixed English words on the glass (Sol PDS2-4); the decoder's own message stays in meta.json
        text = ('no frames.npz' if isinstance(exc, FileNotFoundError) else
                'check cancelled' if str(exc) == 'check cancelled' else
                'decoder timeout' if str(exc) == 'decoder timeout' else 'decoder failed')
        verdict = inconclusive(str(exc), text)
    return {**details, **verdict}, other
