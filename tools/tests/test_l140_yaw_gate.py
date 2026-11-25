"""L140 gate tests with synthetic logs and read-only replay of C8 session 2."""

import csv
import json
import math
from pathlib import Path

import pytest

from tools import l140_yaw_gate as gate


COLUMNS = ('timestamp_us', 'fl_count', 'fr_count', 'rl_count', 'rr_count',
           'sp_vx', 'sp_vy', 'sp_wz', 'yaw_rad', 'log_seq', 'motor_on')
C8_ROOT = Path('D:/CLAUDE_THESIS/05_DATA/results/C8/2026-11-20_run02/pi')
B1_CSV = Path('D:/CLAUDE_THESIS/05_DATA/results/XE_THAT/2026-11-21_run01/b1_troi_dung_yen.csv')
REAL_RUNS = [
    ('HC_K1_1/log.csv', 'hold'),
    ('HC_K1_2/log.csv', 'hold'),
    ('HC_K2_1/log.csv', 'hold'),
    ('C8-L-F02-1.csv', 'pass'),
    ('C8-P-F02-1.csv', 'pass'),
]
REQUIRES_C8 = pytest.mark.skipif(not C8_ROOT.is_dir(), reason='C8 session 2 logs unavailable')


def still_rows(n=751, *, slope=0.0, start_us=10_000_000, seq=0, yaw0_deg=0.0,
               step_us=10_000, motor_on=1):
    return [dict(zip(COLUMNS, (start_us + step_us * i, 18, -21, 6, 4, 0, 0, 0,
                              math.radians(yaw0_deg + slope * step_us * i / 1e6), seq + i, motor_on)))
            for i in range(n)]


def write_log(path, rows, *, append=False):
    with path.open('a' if append else 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, COLUMNS)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def log(tmp_path):
    return tmp_path / 'log.csv'


def run_cli(capsys, command, log, *options):
    code = gate.main([command, '--csv', str(log), *options])
    summary, record = capsys.readouterr().out.splitlines()
    record = json.loads(record)
    assert summary.startswith(record['status'] + ': ')
    return code, record


@pytest.mark.parametrize(('slope', 'status', 'code'), [
    (0.0, 'pass', 0),
    (0.03, 'hold', 4),
    (-0.014999, 'pass', 0),
    (-0.015001, 'hold', 4),
    (0.014999, 'pass', 0),
    (0.015001, 'hold', 4),
])
def test_check_slope_and_threshold(log, capsys, slope, status, code):
    write_log(log, still_rows(slope=slope, yaw0_deg=15))
    exit_code, decision = run_cli(capsys, 'check', log)
    assert (exit_code, decision['status']) == (code, status)
    assert decision['slope_deg_s'] == pytest.approx(slope, abs=1e-12)
    assert decision['lines_used'] == 701
    assert decision['window_start_us'] == 10_500_000
    assert decision['window_end_us'] == 17_500_000


def test_exact_threshold_passes_and_cli_can_override_it(log, capsys):
    write_log(log, still_rows(slope=-0.03))
    measured = abs(gate.check(log)['slope_deg_s'])
    code, decision = run_cli(capsys, 'check', log, '--threshold', str(measured))
    assert (code, decision['status']) == (0, 'pass')


@pytest.mark.parametrize('n', [651, 750])
def test_still_time_includes_start_trim(log, capsys, n):
    write_log(log, still_rows(n))
    code, decision = run_cli(capsys, 'check', log)
    assert (code, decision['status']) == (3, 'not_ready')


def test_check_adjustable_window_and_trim(log, capsys):
    write_log(log, still_rows(351))
    code, decision = run_cli(capsys, 'check', log, '--min-still-s', '3', '--trim-s', '0.5')
    assert (code, decision['status']) == (0, 'pass')
    assert decision['lines_used'] == 301
    assert decision['window_end_us'] == 13_500_000
    assert gate.check(log, min_still_s=3, trim_s=0.51)['status'] == 'not_ready'


def test_regression_uses_only_the_latest_window_with_no_end_trim(log):
    rows = still_rows(901, slope=0.01, yaw0_deg=20)
    for row in rows[:200]:
        row['yaw_rad'] -= math.radians(15)
    rows[-1]['yaw_rad'] += math.radians(1.0)
    write_log(log, rows)

    # In the 701 equally spaced points, a one-degree impulse at t=7 s
    # adds (7 - mean(t)) / sum((t - mean(t)) ** 2) to the fitted slope.
    variance_sum = 0.01**2 * 701 * (701**2 - 1) / 12
    expected = 0.01 + 3.5 / variance_sum
    decision = gate.check(log)
    assert decision['slope_deg_s'] == pytest.approx(expected, abs=1e-12)
    assert decision['window_start_us'] == 12_000_000
    assert decision['window_end_us'] == 19_000_000


@pytest.mark.parametrize('column', COLUMNS[1:8])
def test_each_encoder_and_setpoint_can_break_stillness(log, column):
    rows = still_rows()
    rows[200][column] = -99
    write_log(log, rows)
    assert gate.check(log)['status'] == 'not_ready'


def test_old_still_window_does_not_count_while_moving(log):
    rows = still_rows(901)
    rows[-1]['sp_vx'] = 0.2
    write_log(log, rows)
    assert gate.check(log)['status'] == 'not_ready'


@pytest.mark.parametrize(('n', 'break_at', 'shift_us', 'status'), [
    (901, 200, 190_000, 'pass'),       # 0.200000 s between adjacent samples
    (901, 200, 190_001, 'not_ready'),  # 0.200001 s between adjacent samples
    (1151, 600, -3_000_000, 'not_ready'),
])
def test_gap_or_boot_breaks_stillness(log, n, break_at, shift_us, status):
    rows = still_rows(n)
    for row in rows[break_at:]:
        row['timestamp_us'] += shift_us
    write_log(log, rows)
    assert gate.check(log)['status'] == status


@pytest.mark.parametrize(('missing', 'status'), [(70, 'pass'), (71, 'pass'), (72, 'not_ready')])
def test_density_is_checked_in_the_latest_window(log, missing, status):
    rows = still_rows(slope=0.01)
    absent = {51 + 9 * i for i in range(missing)}
    kept = [row for i, row in enumerate(rows) if i not in absent]
    # Retransmissions cannot fill the sampling deficit.
    write_log(log, kept + kept[80:160])
    decision = gate.check(log)
    assert decision['status'] == status
    assert decision['duplicates_ignored'] == 80
    if status == 'pass':
        assert decision['lines_used'] == 701 - missing
        assert decision['slope_deg_s'] == pytest.approx(0.01, abs=1e-12)


def test_identical_duplicate_is_ignored_even_when_it_arrives_late(log):
    rows = still_rows(slope=0.01)
    write_log(log, rows[:400] + [rows[399]] + rows[400:] + [rows[10]])
    decision = gate.check(log)
    assert decision['status'] == 'pass'
    assert decision['slope_deg_s'] == pytest.approx(0.01, abs=1e-12)
    assert decision['window_end_us'] == rows[-1]['timestamp_us']
    assert decision['lines_used'] == 701
    assert decision['duplicates_ignored'] == 2


def test_conflicting_duplicate_errors_even_in_an_unused_column(log, capsys):
    rows = still_rows()
    write_log(log, rows + [rows[5], {**rows[10], 'motor_on': 0}])
    code, decision = run_cli(capsys, 'check', log)
    assert (code, decision['status']) == (2, 'error')
    assert 'log_seq 10' in decision['reason']
    assert decision['duplicates_ignored'] == 1


@pytest.mark.parametrize('tail', [b'17510000,18,-21', b'17510000,18,-21,6,4,1,0,0,0,751,1'])
def test_unterminated_last_line_is_ignored(log, tail):
    write_log(log, still_rows())
    with log.open('ab') as stream:
        stream.write(tail)
    decision = gate.check(log)
    assert decision['status'] == 'pass'
    assert decision['window_end_us'] == 17_500_000


def test_wrong_field_count_in_the_middle_is_an_error(log, capsys):
    rows = still_rows()
    write_log(log, rows[:2])
    with log.open('ab') as stream:
        stream.write(b'10015000,18,-21\n')
    write_log(log, rows[2:], append=True)
    code, decision = run_cli(capsys, 'check', log)
    assert (code, decision['status']) == (2, 'error')
    assert 'line 4' in decision['reason']


@pytest.mark.parametrize(('initial_contents', 'reason'), [
    (None, 'does not exist yet'),
    (b'', 'no complete CSV header yet'),
    (b'timestamp_us,fl_count', 'no complete CSV header yet'),
])
def test_check_is_not_ready_before_the_logger_header(log, capsys, initial_contents, reason):
    if initial_contents is not None:
        log.write_bytes(initial_contents)
    code, record = run_cli(capsys, 'check', log)
    assert (code, record['status']) == (3, 'not_ready')
    assert reason in record['reason']


def test_check_reports_other_csv_read_errors(log, capsys):
    log.mkdir()
    code, record = run_cli(capsys, 'check', log)
    assert (code, record['status']) == (2, 'error')


def test_check_rejects_complete_header_with_missing_columns(log, capsys):
    log.write_bytes(b'timestamp_us,fl_count\n')
    code, record = run_cli(capsys, 'check', log)
    assert (code, record['status']) == (2, 'error')
    assert 'missing CSV columns' in record['reason']


@pytest.mark.parametrize('slope', [-0.03, 0.03])
def test_unwrap_across_plus_or_minus_pi(log, slope):
    rows = still_rows(slope=slope, yaw0_deg=math.copysign(179.88, slope))
    for row in rows:
        row['yaw_rad'] = (row['yaw_rad'] + math.pi) % math.tau - math.pi
    assert rows[100]['yaw_rad'] * rows[-1]['yaw_rad'] < 0
    write_log(log, rows)
    decision = gate.check(log)
    assert decision['status'] == 'hold'
    assert decision['slope_deg_s'] == pytest.approx(slope, abs=1e-12)


class PollClock:
    def __init__(self, tick=lambda: None):
        self.elapsed = 0.0
        self.tick = tick

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        self.elapsed += seconds
        self.tick()


@pytest.mark.parametrize('initial_contents', [None, b'timestamp_us,fl_count'])
def test_wait_polls_until_logger_writes_header(log, monkeypatch, capsys, initial_contents):
    if initial_contents is not None:
        log.write_bytes(initial_contents)
    monkeypatch.setattr(gate, 'time', PollClock(lambda: write_log(log, still_rows())))
    code, record = run_cli(capsys, 'wait', log, '--timeout-s', '1')
    assert (code, record['status']) == (0, 'pass')
    assert record['waiting_s'] == 0.5
    assert record['holds'] == []
    assert record['timed_out'] is False


def test_wait_polls_growing_csv_records_holds_and_wait_time(log, monkeypatch, capsys):
    drift = still_rows(slope=0.03)
    flat = still_rows(700, start_us=17_510_000, seq=751, yaw0_deg=0.225)
    write_log(log, drift[:300])
    chunks = iter([drift[300:], flat[:20], flat[20:]])
    monkeypatch.setattr(gate, 'time', PollClock(lambda: write_log(log, next(chunks), append=True)))
    json_path = log.with_name('wait.json')

    code, record = run_cli(capsys, 'wait', log, '--poll-s', '0.5', '--timeout-s', '2',
                           '--json', str(json_path))
    assert (code, record['status']) == (0, 'pass')
    assert record['waiting_s'] == 1.5
    assert record['window_end_us'] == flat[-1]['timestamp_us']
    assert record['slope_deg_s'] == pytest.approx(0.0, abs=1e-12)
    assert record['timed_out'] is False
    assert [hold['status'] for hold in record['holds']] == ['hold', 'hold']
    assert [hold['waiting_s'] for hold in record['holds']] == [0.5, 1.0]
    assert [hold['window_end_us'] for hold in record['holds']] == [17_500_000, 17_700_000]
    assert json.loads(json_path.read_text(encoding='utf-8')) == record
    assert sorted(path.name for path in log.parent.iterdir()) == ['log.csv', 'wait.json']


@pytest.mark.parametrize(('n', 'status', 'exit_code'), [(100, 'not_ready', 3), (751, 'hold', 4)])
def test_wait_timeout_returns_last_decision(log, monkeypatch, capsys, n, status, exit_code):
    write_log(log, still_rows(n, slope=0.03))
    monkeypatch.setattr(gate, 'time', PollClock())
    code, record = run_cli(capsys, 'wait', log, '--timeout-s', '0.75')
    assert (code, record['status']) == (exit_code, status)
    assert record['timed_out'] is True
    assert record['waiting_s'] == 0.75
    assert [hold['waiting_s'] for hold in record['holds']] == ([0.0, 0.5] if n == 751 else [])


def test_post_records_drift_without_gating_the_run(log, capsys):
    rows = still_rows(1801, slope=-0.03)
    rows[450]['sp_vx'] = 0.1
    rows[1100]['sp_wz'] = 0.2
    write_log(log, rows)

    code, record = run_cli(capsys, 'post', log)
    assert (code, record['status']) == (0, 'recorded')
    assert len(record['spans']) == 1
    assert record['spans'][0]['slope_deg_s'] == pytest.approx(-0.03)
    assert record['spans'][0]['window_start_us'] == 21_510_000
    assert record['spans'][0]['window_end_us'] == 27_500_000

    code, record = run_cli(capsys, 'post', log, '--after-us', '15000000')
    assert (code, record['status']) == (0, 'recorded')
    assert [span['window_start_us'] for span in record['spans']] == [15_510_000, 21_510_000]
    assert all(span['slope_deg_s'] == pytest.approx(-0.03) for span in record['spans'])


def test_post_errors_when_only_pre_run_still_time_exists(log, capsys):
    rows = still_rows(901)
    rows[-1]['sp_vy'] = 0.1
    write_log(log, rows)
    code, record = run_cli(capsys, 'post', log)
    assert (code, record['status'], record['spans']) == (2, 'error', [])


@pytest.mark.parametrize('lines_after_trimming', [49, 50])
def test_post_10_hz_tail_requires_at_least_50_trimmed_lines(log, capsys, lines_after_trimming):
    motion = still_rows(1, start_us=9_900_000)
    motion[0]['sp_vx'] = 0.1
    tail = still_rows(lines_after_trimming + 10, slope=-0.03, seq=1, step_us=100_000, motor_on=0)
    write_log(log, motion + tail)

    code, record = run_cli(capsys, 'post', log)
    if lines_after_trimming == 49:
        assert (code, record['status'], record['spans']) == (2, 'error', [])
    else:
        assert (code, record['status']) == (0, 'recorded')
        assert len(record['spans']) == 1
        span = record['spans'][0]
        assert span['lines_used'] == lines_after_trimming
        assert span['mean_line_rate_hz'] == pytest.approx(10.0)
        assert span['slope_deg_s'] == pytest.approx(-0.03)
        assert span['window_start_us'] == 10_500_000
        assert span['window_end_us'] == 15_400_000


def test_json_output_cannot_overwrite_input(log, capsys):
    write_log(log, still_rows())
    before = log.read_bytes()
    code, decision = run_cli(capsys, 'check', log, '--json', str(log))
    assert (code, decision['status']) == (2, 'error')
    assert log.read_bytes() == before


@REQUIRES_C8
@pytest.mark.parametrize(('relative_path', '_status'), REAL_RUNS)
def test_real_slopes_match_b1(relative_path, _status):
    path = C8_ROOT / relative_path
    name = path.parent.name if path.name == 'log.csv' else path.stem
    with B1_CSV.open(newline='', encoding='utf-8') as stream:
        expected = {float(row['t0_up_s']): float(row['doc_yaw_deg_s'])
                    for row in csv.DictReader(stream)
                    if row['bo'] == 'C8_run02' and row['log'] == name}

    samples, _ = gate.read_log(path)
    spans = gate.measure_still_spans(samples, min_still_s=3, trim_s=0.5,
                                    min_rows_per_s=0, min_lines=50)
    actual = {round(span['window_start_us'] / 1e6, 2): span['slope_deg_s'] for span in spans}
    assert expected
    assert len(spans) == len(actual)
    assert actual.keys() == expected.keys()
    for start, slope in actual.items():
        assert round(slope, 5) == expected[start]


@REQUIRES_C8
def test_real_post_records_sparse_tail(capsys):
    code, record = run_cli(capsys, 'post', C8_ROOT / 'C8-L-F02-1.csv')
    assert (code, record['status']) == (0, 'recorded')
    assert len(record['spans']) == 1
    span = record['spans'][0]
    assert span['lines_used'] >= 50
    assert 0 < span['mean_line_rate_hz'] < 90
    assert round(span['slope_deg_s'], 5) == -0.00023


def pre_run_csv(source, target):
    """Copy the prefix up to the first encoder change or nonzero setpoint.

    Every C8 log used here begins with adequate still time before that row.
    The cut uses encoder and command fields only, with no yaw-based selection.
    """
    counts = None
    with source.open('rb') as stream, target.open('wb') as out:
        header = stream.readline()
        out.write(header)
        columns = next(csv.reader([header.decode('utf-8')]))
        for raw in stream:
            row = dict(zip(columns, next(csv.reader([raw.decode('utf-8')]))))
            current = tuple(int(row[column]) for column in COLUMNS[1:5])
            if any(float(row[column]) for column in COLUMNS[5:8]) or (counts is not None and current != counts):
                return
            counts = current
            out.write(raw)
    pytest.fail(f'no motion or nonzero command in {source}')


@REQUIRES_C8
@pytest.mark.parametrize(('relative_path', 'status'), REAL_RUNS)
def test_real_pre_run_gate(relative_path, status, log, capsys):
    pre_run_csv(C8_ROOT / relative_path, log)
    code, decision = run_cli(capsys, 'check', log)
    assert (code, decision['status']) == (0 if status == 'pass' else 4, status)
    assert decision['lines_used'] >= 630
