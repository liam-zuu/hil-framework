#!/usr/bin/env python3
"""L140 pre-run yaw drift gate for the floor car, approved 20/11 at CODE_XETHAT.

Before sending a scored trajectory, measure IMU yaw drift over the latest
7 s with all four encoders unchanged and zero setpoints, allowing a 0.5 s
start trim. While drift exceeds the threshold, hold the trajectory and
measure again. The proposed 0.015 deg/s threshold comes from b1 and is to
be locked at C8. A hold counts only as campaign waiting time, recorded
separately. Post-run slopes are for recording only. This tool sends no
commands and changes neither scoring nor firmware.

Exit codes: 0 pass or post-run measurement recorded, 2 error,
3 not ready, 4 hold for drift. On timeout, wait returns the last decision.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import NamedTuple


DEFAULT_THRESHOLD = 0.015
DEFAULT_MIN_STILL_S = 7.0
DEFAULT_TRIM_S = 0.5
DEFAULT_POST_MIN_STILL_S = 3.0
US_PER_S = 1_000_000
MAX_GAP_US = 200_000
MIN_ROWS_PER_S = 90
POST_MIN_LINES = 50
COUNTS = ('fl_count', 'fr_count', 'rl_count', 'rr_count')
SETPOINTS = ('sp_vx', 'sp_vy', 'sp_wz')
SCALAR_COLUMNS = ('timestamp_us', 'log_seq', 'yaw_rad')
REQUIRED_COLUMNS = SCALAR_COLUMNS + COUNTS + SETPOINTS
EXIT_CODES = {'pass': 0, 'recorded': 0, 'error': 2, 'not_ready': 3, 'hold': 4}


class Sample(NamedTuple):
    timestamp_us: int
    counts: tuple
    setpoint_zero: bool
    yaw_rad: float


class LogError(ValueError):
    def __init__(self, message, duplicates_ignored=0):
        super().__init__(message)
        self.duplicates_ignored = duplicates_ignored


class LogNotReady(LogError):
    """The CSV is missing or its header is still being written."""


def read_log(path, *, require_header=False):
    """Return (samples in arrival order, duplicates ignored) from complete lines.

    agv_logger terminates every row with a newline. An unterminated tail is
    left for the next read, even if it already has the right field count.
    Duplicate comparison uses the full row text, excluding its line ending.
    With require_header, an incomplete header raises LogNotReady.
    """
    samples, seen = [], {}
    duplicates = 0
    line_number = 0
    try:
        with open(path, 'rb') as source:
            header = source.readline()
            if not header.endswith(b'\n'):
                if require_header:
                    raise LogNotReady(f'{path}: no complete CSV header yet')
                return samples, duplicates
            columns = next(csv.reader([header.decode('utf-8-sig')], strict=True))
            indices = {name: index for index, name in enumerate(columns)}
            if len(indices) != len(columns):
                raise ValueError('repeated column name in CSV header')
            missing = [name for name in REQUIRED_COLUMNS if name not in indices]
            if missing:
                raise ValueError(f'missing CSV columns: {", ".join(missing)}')
            timestamp_col, seq_col, yaw_col = (indices[name] for name in SCALAR_COLUMNS)
            count_cols = [indices[name] for name in COUNTS]
            setpoint_cols = [indices[name] for name in SETPOINTS]

            for line_number, line in enumerate(source, start=2):
                if not line.endswith(b'\n'):
                    break
                raw = line.rstrip(b'\r\n')
                fields = next(csv.reader([raw.decode('utf-8')], strict=True))
                if len(fields) != len(columns):
                    raise ValueError(f'expected {len(columns)} fields, found {len(fields)}')
                seq = int(fields[seq_col])
                if seq in seen:
                    if raw != seen[seq]:
                        raise ValueError(f'conflicting content for log_seq {seq}')
                    duplicates += 1
                    continue

                timestamp_us = int(fields[timestamp_col])
                counts = tuple(int(fields[col]) for col in count_cols)
                setpoints = tuple(float(fields[col]) for col in setpoint_cols)
                yaw_rad = float(fields[yaw_col])
                if not math.isfinite(yaw_rad) or not all(map(math.isfinite, setpoints)):
                    raise ValueError('non-finite yaw or setpoint')
                samples.append(Sample(timestamp_us, counts, not any(setpoints), yaw_rad))
                seen[seq] = raw
    except FileNotFoundError as exc:
        raise LogNotReady(f'{path}: CSV does not exist yet') from exc
    except LogNotReady:
        raise
    except (OSError, ValueError, csv.Error) as exc:
        location = f'{path}, line {line_number}' if line_number else str(path)
        raise LogError(f'{location}: {exc}', duplicates) from exc
    return samples, duplicates


def still_spans(samples):
    """Yield [start, stop) indices of contiguous still samples in log order.

    Like b1, a row that breaks continuity is outside the span on either
    side. The first row can start a span if its setpoints are zero.
    """
    start = None
    previous = None
    for index, sample in enumerate(samples):
        continuous = previous is None or (
            sample.counts == previous.counts
            and 0 <= sample.timestamp_us - previous.timestamp_us <= MAX_GAP_US
        )
        if continuous and sample.setpoint_zero:
            if start is None:
                start = index
        elif start is not None:
            yield start, index
            start = None
        previous = sample
    if start is not None:
        yield start, len(samples)


def yaw_slope_deg_s(samples):
    """Fit unwrapped yaw vs time by least squares with an intercept, as b1."""
    if len(samples) < 2 or samples[0].timestamp_us == samples[-1].timestamp_us:
        raise ValueError('yaw slope needs at least two distinct timestamps')
    origin = samples[0].timestamp_us
    times = [(sample.timestamp_us - origin) / US_PER_S for sample in samples]
    previous = samples[0].yaw_rad
    yaw = []
    offset = 0.0
    for sample in samples:
        phase = sample.yaw_rad
        delta = phase - previous
        if abs(delta) > math.pi:
            wrapped = (delta + math.pi) % math.tau - math.pi
            if wrapped == -math.pi and delta > 0.0:
                wrapped = math.pi
            offset += wrapped - delta
        yaw.append(phase + offset)
        previous = phase

    mean_t = math.fsum(times) / len(times)
    mean_yaw = math.fsum(yaw) / len(yaw)
    centered_t = [t - mean_t for t in times]
    variance = math.fsum(t * t for t in centered_t)
    covariance = math.fsum(t * (y - mean_yaw) for t, y in zip(centered_t, yaw))
    slope = math.degrees(covariance / variance)
    if not math.isfinite(slope):
        raise ValueError('non-finite yaw slope')
    return slope


def _window_record(window):
    return {
        'slope_deg_s': yaw_slope_deg_s(window),
        'window_start_us': window[0].timestamp_us,
        'window_end_us': window[-1].timestamp_us,
        'lines_used': len(window),
    }


def _unmeasured_record(status, reason, duplicates):
    return {
        'status': status,
        'reason': reason,
        'slope_deg_s': None,
        'window_start_us': None,
        'window_end_us': None,
        'lines_used': 0,
        'duplicates_ignored': duplicates,
    }


def _error_record(exc, duplicates=0):
    return _unmeasured_record('error', str(exc), getattr(exc, 'duplicates_ignored', duplicates))


def _check_samples(samples, duplicates, threshold, min_still_s, trim_s):
    result = _unmeasured_record('not_ready', 'no still span ending at the latest row', duplicates)
    result['threshold_deg_s'] = threshold
    latest = None
    for span in still_spans(samples):
        latest = span
    if latest is None or latest[1] != len(samples):
        return result

    start, _ = latest
    end_us = samples[-1].timestamp_us
    start_us = end_us - min_still_s * US_PER_S
    if start_us - samples[start].timestamp_us < trim_s * US_PER_S:
        result['reason'] = 'latest still span is too short after the start trim'
        return result

    while samples[start].timestamp_us < start_us:
        start += 1
    window = samples[start:]
    if len(window) < max(2, MIN_ROWS_PER_S * min_still_s) or window[0].timestamp_us == end_us:
        result['reason'] = 'too few samples in the latest still window'
        return result

    result.update(_window_record(window))
    if abs(result['slope_deg_s']) <= threshold:
        result.update(status='pass', reason='yaw drift is within the threshold')
    else:
        result.update(status='hold', reason='yaw drift exceeds the threshold')
    return result


def check(path, *, threshold=DEFAULT_THRESHOLD, min_still_s=DEFAULT_MIN_STILL_S,
          trim_s=DEFAULT_TRIM_S):
    """Return one gate decision from a snapshot of the CSV."""
    duplicates = 0
    try:
        _validate_number(threshold, 'threshold')
        _validate_number(min_still_s, 'min-still-s', positive=True)
        _validate_number(trim_s, 'trim-s')
        samples, duplicates = read_log(path, require_header=True)
        return _check_samples(samples, duplicates, threshold, min_still_s, trim_s)
    except LogNotReady as exc:
        result = _unmeasured_record('not_ready', str(exc), 0)
        result['threshold_deg_s'] = threshold
        return result
    except (ValueError, ArithmeticError) as exc:
        return _error_record(exc, duplicates)


def wait(path, *, threshold=DEFAULT_THRESHOLD, min_still_s=DEFAULT_MIN_STILL_S,
         trim_s=DEFAULT_TRIM_S, poll_s=0.5, timeout_s=None):
    """Poll the CSV, recording one entry per hold and elapsed monotonic time."""
    try:
        _validate_number(poll_s, 'poll-s', positive=True)
        if timeout_s is not None:
            _validate_number(timeout_s, 'timeout-s')
    except ValueError as exc:
        return _error_record(exc)

    started = time.monotonic()
    holds = []
    timed_out = False
    while True:
        decision = check(path, threshold=threshold, min_still_s=min_still_s, trim_s=trim_s)
        elapsed = time.monotonic() - started
        if decision['status'] == 'hold':
            holds.append({**decision, 'waiting_s': elapsed})
        if decision['status'] in ('pass', 'error'):
            break

        delay = poll_s
        if timeout_s is not None:
            remaining = timeout_s - elapsed
            if remaining <= 0:
                timed_out = True
                break
            delay = min(delay, remaining)
        time.sleep(delay)
        elapsed = time.monotonic() - started
        if timeout_s is not None and elapsed >= timeout_s:
            timed_out = True
            break

    return {**decision, 'waiting_s': elapsed, 'holds': holds, 'timed_out': timed_out}


def measure_still_spans(samples, *, min_still_s=DEFAULT_POST_MIN_STILL_S,
                        trim_s=DEFAULT_TRIM_S, after_us=None, trailing_only=False,
                        min_rows_per_s=MIN_ROWS_PER_S, min_lines=2):
    """Fit still spans with b1's trim at both ends.

    The optional timestamp cutoff is exclusive and applied before trimming.
    Use min_rows_per_s=0 and min_lines=50 for b1's sample-count rule.
    Mean line rate is the inverse of the mean interval between fitted rows.
    """
    _validate_number(min_still_s, 'min-still-s', positive=True)
    _validate_number(trim_s, 'trim-s')
    _validate_number(min_rows_per_s, 'min_rows_per_s')
    _validate_number(min_lines, 'min_lines')
    measurements = []
    trim_us = trim_s * US_PER_S
    for start, stop in still_spans(samples):
        if trailing_only and stop != len(samples):
            continue
        if after_us is not None:
            while start < stop and samples[start].timestamp_us <= after_us:
                start += 1
        if start == stop:
            continue
        first_us = samples[start].timestamp_us + trim_us
        last_us = samples[stop - 1].timestamp_us - trim_us
        duration_s = (last_us - first_us) / US_PER_S
        if duration_s < min_still_s:
            continue
        window = [row for row in samples[start:stop] if first_us <= row.timestamp_us <= last_us]
        if len(window) < max(2, min_lines, min_rows_per_s * duration_s):
            continue
        elapsed_us = window[-1].timestamp_us - window[0].timestamp_us
        if elapsed_us > 0:
            record = _window_record(window)
            record['mean_line_rate_hz'] = (len(window) - 1) * US_PER_S / elapsed_us
            measurements.append(record)
    return measurements


def post(path, *, after_us=None, min_still_s=DEFAULT_POST_MIN_STILL_S, trim_s=DEFAULT_TRIM_S):
    """Record post-run drift with b1's 50-line minimum, also for 10 Hz tails."""
    duplicates = 0
    try:
        samples, duplicates = read_log(path)
        measurements = measure_still_spans(samples, min_still_s=min_still_s, trim_s=trim_s,
                                           after_us=after_us, trailing_only=after_us is None,
                                           min_rows_per_s=0, min_lines=POST_MIN_LINES)
    except (ValueError, ArithmeticError) as exc:
        return _error_record(exc, duplicates)
    return {
        'status': 'recorded' if measurements else 'error',
        'reason': 'post-run yaw measured' if measurements else 'no usable post-run still span',
        'after_us': after_us,
        'duplicates_ignored': duplicates,
        'spans': measurements,
    }


def _validate_number(value, name, *, positive=False):
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        requirement = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{name} must be finite and {requirement}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    check_parser = commands.add_parser('check', help='make one gate decision')
    wait_parser = commands.add_parser('wait', help='poll until pass, timeout or error')
    post_parser = commands.add_parser('post', help='record post-run drift')
    for command in (check_parser, wait_parser, post_parser):
        is_post = command is post_parser
        command.add_argument('--csv', type=Path, required=True, help='agv_logger CSV')
        command.add_argument('--json', type=Path, help='also write the JSON record to this file')
        command.add_argument('--min-still-s', type=float,
                             default=DEFAULT_POST_MIN_STILL_S if is_post else DEFAULT_MIN_STILL_S,
                             help='minimum usable still time in seconds (default %(default)s)')
        command.add_argument('--trim-s', type=float, default=DEFAULT_TRIM_S,
                             help=('trim both ends (post)' if is_post else 'trim the start of the span')
                             + ' in seconds (default %(default)s)')
        if not is_post:
            command.add_argument(
                '--threshold', type=float, default=DEFAULT_THRESHOLD,
                help=('proposed limit in deg/s (default %(default)s, to be locked at C8). '
                      'From CODE_XETHAT b1: near the 95th percentile of 155 still spans '
                      '(0.014 deg/s), about half the 20/11 bias episode'),
            )
    wait_parser.add_argument('--poll-s', type=float, default=0.5, help='seconds between reads')
    wait_parser.add_argument('--timeout-s', type=float, help='timeout in seconds, default unlimited')
    post_parser.add_argument('--after-us', type=int, metavar='T',
                             help='record still spans after ESP32 timestamp T in microseconds')
    args = parser.parse_args(argv)

    options = {'min_still_s': args.min_still_s, 'trim_s': args.trim_s}
    try:
        if args.json is not None:
            if args.json.resolve() == args.csv.resolve() or (
                args.json.exists() and args.csv.exists() and args.json.samefile(args.csv)
            ):
                raise ValueError('JSON output would overwrite the input CSV')

        if args.command == 'post':
            record = post(args.csv, after_us=args.after_us, **options)
        elif args.command == 'wait':
            record = wait(args.csv, threshold=args.threshold, poll_s=args.poll_s,
                          timeout_s=args.timeout_s, **options)
        else:
            record = check(args.csv, threshold=args.threshold, **options)

        if args.json is not None:
            args.json.write_text(json.dumps(record, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError, RuntimeError) as exc:
        record = _error_record(exc)

    summary = f'{record["status"]}: {record["reason"]}'
    if record.get('slope_deg_s') is not None:
        summary += f' ({record["slope_deg_s"]:+.6f} deg/s)'
    print(summary)
    print(json.dumps(record, allow_nan=False))
    return EXIT_CODES[record['status']]


if __name__ == '__main__':
    raise SystemExit(main())
