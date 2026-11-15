"""tools/s06_h7_serial.py (H7 serial log: T1 counters, T10 load table) on a log written here in the print
format of H7 feat/s06-h7 (main.c, s06_load.c). Run: python -m pytest tools/tests"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import s06_h7_serial as ser  # noqa: E402


def t10_block(win_ms=10000, isr_us=1_200_000, idle_pass=900_000, idle_us_est=4_000_000, extra='', spi_n=10000):
    return (f'T10: win_ms={win_ms} isr_us={isr_us} idle_pass={idle_pass} idle_us_est={idle_us_est} '
            f'cal=22500/48000000 ovh_cyc=24\r\n'
            'T10 SysTick n=10000 excl_us=2000 max_us=1.25\r\n'
            f'T10 SPI1    n={spi_n} excl_us=300000 max_us=35.50\r\n'
            f'T10 TIM5    n=400000 excl_us=898000 max_us=4.08{extra}\r\n')


def link(lost=0, err=0, rearm=0):
    return f'Link: up lost={lost}  SPI frames=1000/s err={err} rearm={rearm} last_err=0x0'


def frame(applied, skipped=0, unread=0):
    return (f'Frame: unread={unread}  new seen=5 taken=5 repl=0 drop=0  applied={applied} skipped={skipped} '
            f'age_max=420us')


def make_log(blocks=4, print_drop_end=0, pre_up_block=True, ts_prefix=False, pc=True, drop_block=None,
             drop_irq=None, reconnect=False, first_spi_n=5000):
    """Lines as tools/h7_serial_capture.py writes them (`<t_pc_us>\\t<line>`, pc=False: as PuTTY); T10 every
    10 s, Link/Frame lines around the run so the counters are bounded at both ends. T10U-R3-1: the first in-run
    block started before link up, so it holds fewer SPI1 frames than its window (first_spi_n; the tool does not
    tie n to the PC times; on the bench 15/11 link up came 8.04 s before the first header, SPI1 n=8036)."""
    out = []                                        # (t_s, line)
    out += [(0.0, '--- H753 HIL boot ---'), (0.1, 'Init done, entering main loop'), (0.5, link()), (0.5, frame(0))]
    if pre_up_block:
        out += [(0.9, l) for l in t10_block(idle_pass=1_000_000).splitlines()]
    out += [(1.0, 'SPI link up at t=12345 ms'), (1.0, 'S06log: pushed=1 dropped=0  print_drop=0')]
    for k in range(blocks):
        t = 1.0 + 10.0 * k
        out += [(t + 0.5, link()), (t + 0.5, frame(10000 * (k + 1)))]
        if reconnect and k == 1:
            out += [(t + 1.0, 'SPI link lost (no frame for 50 ms): encoders stopped, SPI1 restart'),
                    (t + 1.2, 'SPI link up at t=99999 ms')]
        if k != drop_block:
            lines = t10_block(spi_n=first_spi_n if k == 0 else 10000).splitlines()
            if drop_irq is not None and k == drop_irq:
                lines = [l for l in lines if 'SPI1' not in l]
            out += [(t + 10.0, l) for l in lines]
    t_end = 1.0 + 10.0 * blocks
    out += [(t_end, f'S06log: pushed=99 dropped=0  print_drop={print_drop_end}'),
            (t_end + 0.05, 'SPI link lost (no frame for 50 ms): encoders stopped, SPI1 restart'),
            (t_end + 0.05, 'Link period: 40001 frames, H7 40010 ms from first to last frame (Pi: 40000 ms at 1 kHz)'),
            (t_end + 0.5, link(lost=1 + int(reconnect), rearm=1 + int(reconnect))),
            (t_end + 0.5, frame(10000 * blocks))]
    out += [(t_end + 9.0, l) for l in t10_block(isr_us=5).splitlines()]      # after the run: not used
    if ts_prefix:
        return [f'[12:00:{i:02d}.123] {l}' for i, (_, l) in enumerate(out)]
    if pc:
        return [f'{int(1e9 + t * 1e6)}\t{l}' for t, l in out]
    return [l for _, l in out]


def test_t10_blocks_inside_run_drop_first():
    ev = ser.parse_lines(make_log())
    blocks, span = ser.t10_blocks(ev)
    assert len(blocks) == 3                                  # 4 in the run, first dropped; pre/post excluded
    win, irq = ser.t10_rows(blocks)
    w = win[0]
    assert w['isr_pct'] == pytest.approx(12.0) and w['idle_pct'] == pytest.approx(40.0)
    assert w['rest_pct'] == pytest.approx(48.0)
    assert w['idle_rel_unc'] == pytest.approx(1 / 22500 + 1 / 900_000)
    assert w['meas_cost_us'] == pytest.approx(420_000 * 24 / 480.0)
    spi = [r for r in irq if r['irq'] == 'SPI1'][0]
    assert spi['max_us'] == 35.5
    assert (spi['max_us_real_lo'], spi['max_us_real_hi']) == pytest.approx((35.5 * 480 / 486, 35.5 * 480 / 478))   # X22
    one = [r for r in ser.t10_rows(blocks, 483.1)[1] if r['irq'] == 'SPI1'][0]          # frequency of the run itself
    assert one['max_us_real_lo'] == one['max_us_real_hi'] == pytest.approx(35.5 * 480 / 483.1)
    assert spi['pct'] == pytest.approx(3.0) and spi['gross_us'] == ''
    s = ser.t10_summary(blocks, win, irq, ev, span)
    assert s['complete'] and s['windows'] == 3 and s['per_irq'][0]['irq'] == 'TIM5'
    assert s['per_irq'][0]['count'] == 1_200_000


def test_t10_incomplete_print_drop_and_short_window():
    ev = ser.parse_lines(make_log(print_drop_end=2))
    b, sp = ser.t10_blocks(ev)
    s = ser.t10_summary(b, *ser.t10_rows(b), ev, sp)
    assert not s['complete'] and 'print_drop' in s['problems'][0]


def _summary(**kw):
    ev = ser.parse_lines(make_log(**kw))
    b, sp = ser.t10_blocks(ev)
    return ser.t10_summary(b, *ser.t10_rows(b), ev, sp)


def test_t10_lost_block_and_irq_line_found():
    # S24: the H7 restarts the window before printing, so a block lost on the PC does not show in win_ms;
    # it shows in the PC times. A missing IRQ line shows in isr_us vs the sum of excl_us.
    s = _summary(drop_block=2)
    assert not s['complete'] and any('mất khối' in p for p in s['problems'])
    assert any('khác 10 s sau khối' in p for p in s['problems'])     # the spacing check itself, not the count
    s = _summary(drop_block=3)                               # the last block of the run
    assert not s['complete'] and any('mất khối' in p for p in s['problems'])
    s = _summary(drop_irq=2)
    assert not s['complete'] and any('thiếu dòng IRQ' in p for p in s['problems'])
    s = _summary(pc=False)                                   # PuTTY log: cannot tell
    assert not s['complete'] and any('không có giờ PC' in p for p in s['problems'])


def test_link_stats_and_timestamp_prefix():
    ev = ser.parse_lines(make_log(ts_prefix=True))
    st = ser.link_stats(ev, ser.run_span(ev))
    # S25: bounded by the lines just before link up and just after the final link lost
    assert st['bounded_start'] and st['bounded_end'] and st['expected_end_loss'] == 1
    assert st['applied'] == 40000 and st['skipped'] == 0 and st['err'] == 0 and st['age_max_us_since_boot'] == 420
    assert st['lost'] == 1 and st['rearm'] == 1 and ser.link_fails(st) == []
    assert st['link_periods'] == [dict(frames=40001, h7_ms=40010, pi_ms=40000)]


def test_link_faults_anywhere_in_the_run():
    # S25: an error in the last 0.5 s before the loss, or a reconnect mid-run, is not left out
    lines = [l.replace('err=0 rearm=1', 'err=1 rearm=1') for l in make_log()]   # err only after the final loss
    ev = ser.parse_lines(lines)
    assert any('err tăng 1' in f for f in ser.link_fails(ser.link_stats(ev, ser.run_span(ev))))
    ev = ser.parse_lines(make_log(reconnect=True))
    st = ser.link_stats(ev, ser.run_span(ev))
    assert st['interruptions'] == 1 and any('mất rồi lên lại' in f for f in ser.link_fails(st))
    no_end = make_log()[:-6]                                 # log cut right after the run: end not bounded
    ev = ser.parse_lines(no_end)
    assert not ser.link_stats(ev, ser.run_span(ev))['bounded_end']


def test_cli(tmp_path):
    log = tmp_path / 'h7.log'
    log.write_text('\n'.join(make_log()) + '\n', encoding='utf-8')
    out = tmp_path / 'out'
    script = os.path.join(ROOT, 'tools', 's06_h7_serial.py')
    p = subprocess.run([sys.executable, script, str(log), '--out', str(out)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    s = json.load(open(out / 'T10_summary.json', encoding='utf-8'))
    assert s['complete'] and s['windows'] == 3 and s['link']['applied'] == 40000
    assert open(out / 'T10.csv').readline().startswith('idx,window,irq,count,excl_us')          # S31
    assert open(out / 'T10_windows.csv').readline().startswith('idx,window,')
    log.write_text('\n'.join(make_log(print_drop_end=1)) + '\n', encoding='utf-8')
    # C1-14: print_drop is lost data (Z19), not "not concluded"
    p = subprocess.run([sys.executable, script, str(log), '--out', str(out)], capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    assert p.returncode == 1 and '#ket_qua,ma=1' in p.stdout
    log.write_text('\n'.join(make_log(pc=False)) + '\n', encoding='utf-8')
    p = subprocess.run([sys.executable, script, str(log), '--out', str(out)], capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    assert p.returncode == 3 and 'ly_do=thieu_bang_chung' in p.stdout
    assert subprocess.run([sys.executable, script, str(tmp_path / 'none.log'), '--out', str(out)],
                          capture_output=True, text=True).returncode == 2


IDLE20 = ('idle_pass=900000 idle_us_est=4000000', 'idle_pass=450000 idle_us_est=2000000')   # 40 % -> 20 % idle


def with_header(lines, k, old, new):
    """Replace `old` by `new` in the k-th 'T10: win_ms' header of the run (k = 0 is the first in-run header,
    the one dropped as partial; the pre-link-up header of make_log is not counted)."""
    out, n = [], -1
    seen_up = False
    for l in lines:
        if 'SPI link up' in l:
            seen_up = True
        if seen_up and 'T10: win_ms' in l:
            n += 1
            if n == k:
                l = l.replace(old, new)
        out.append(l)
    return out


def t10_with_schedule(tmp_path, heavy=(8_000, 52_000), pc=True, blocks=8, edit=None):
    """Link up at 1 s; headers 10, 20, …, 80 s after it (the first dropped). With the T10U-3 margin of
    PRINT_DELAY_MAX_MS on both sides and heavy 8–52 s: windows ending 20–50 s heavy, 60 s mixed, 70, 80 s group A."""
    log = tmp_path / 'h7.log'
    lines = make_log(blocks=blocks, pc=pc)
    if edit:
        lines = edit(lines)
    log.write_text('\n'.join(lines) + '\n')
    sched = tmp_path / 's.csv'
    sched.write_text('idx,kind,test,start_ms,end_ms\n0,test,A-L1,0,50000\n'
                     + (f'1,heavy,,{heavy[0]},{heavy[1]}\n' if heavy else ''))
    out = tmp_path / 'o'
    code = ser.main([str(log), '--out', str(out), '--load-schedule', str(sched)])
    return code, json.load(open(out / 'T10_summary.json', encoding='utf-8')), (out / 'T10_windows.csv').read_text()


def test_t10_segments_from_schedule(tmp_path):
    """Spec 2.2: heavy block and group A part reported apart; a window across a block edge (with the T10U-3
    print-delay margin) goes into neither."""
    code, s, rows = t10_with_schedule(tmp_path)
    assert code == 0
    assert s['segments']['heavy']['windows'] == 4 and s['segments']['nhom_a']['windows'] == 2
    assert 'heavy' in rows and 'nhom_a' in rows and 'mixed' in rows


def test_t10_print_delay_margin(tmp_path):
    """T10U-3: a window whose header comes within PRINT_DELAY_MAX_MS of the block end is mixed: heavy 8–50 s
    leaves the window ending 50 s (true end anywhere in [49; 51] s) out of the heavy part."""
    _, s, _ = t10_with_schedule(tmp_path, heavy=(8_000, 50_000))
    assert s['segments']['heavy']['windows'] == 3


def test_t10_window_across_edge_is_mixed(tmp_path):
    code, s, rows = t10_with_schedule(tmp_path, heavy=(15_000, 25_000))
    assert 'mixed' in rows and 'heavy' not in s['segments']
    assert code == 3 and any('khối tải nặng' in m for m in s['missing']) and s['complete'] is False


def test_t10_no_pc_time_leaves_segment_empty(tmp_path):
    code, s, _ = t10_with_schedule(tmp_path, pc=False)
    assert code == 3 and s['segments'] == {}


def test_t10_schedule_without_heavy(tmp_path):
    code, s, _ = t10_with_schedule(tmp_path, heavy=None)
    assert code == 0 and 'heavy' not in s['segments'] and s['segments'] == {}


def test_t10_heavy_without_group_a_is_missing(tmp_path):
    """T10U-6: with a heavy block in the schedule, a run with no group A window is not complete."""
    code, s, _ = t10_with_schedule(tmp_path, heavy=(8_000, 200_000))
    assert 'nhom_a' not in s['segments'] and s['segments']['heavy']['windows'] == 7
    assert code == 3 and s['complete'] is False and any('nhóm A' in m for m in s['missing'])


# B3 (S06 T10, Liam 12/11 at THESIS local_694a2f03): upper bound of the load L = 100 % - lower bound of the idle
# share from the calibration uncertainty (1/cal_n + 1/idle_pass)
UNC = 1 / 22500 + 1 / 900_000


def test_t10_load_upper_per_window_and_summary():
    ev = ser.parse_lines(make_log())
    blocks, span = ser.t10_blocks(ev)
    win, irq = ser.t10_rows(blocks)
    w = win[0]
    assert w['idle_pct_lo'] == pytest.approx(40.0 * (1 - UNC))
    assert w['load_upper_pct'] == pytest.approx(60.0 + 40.0 * UNC)
    assert w['load_upper_pct'] > 100.0 - w['idle_pct']                   # an upper bound, not the point value
    s = ser.t10_summary(blocks, win, irq, ev, span)
    assert s['load_upper_pct'] == pytest.approx(60.0 + 40.0 * UNC)
    assert s['load_upper_pct_max_window'] == pytest.approx(60.0 + 40.0 * UNC)


def test_load_upper_mean_max_and_missing():
    rows = [dict(idle_pct_lo=30.0, win_ms=10000), dict(idle_pct_lo=40.0, win_ms=10000)]
    assert ser.load_upper(rows) == dict(load_upper_pct=pytest.approx(65.0), load_upper_pct_max_window=70.0)
    assert ser.load_upper(rows + [dict(idle_pct_lo=None, win_ms=10000)]) == dict(load_upper_pct=None,
                                                                                 load_upper_pct_max_window=None)
    assert ser.load_upper([])['load_upper_pct'] is None


def test_load_upper_time_weighted():
    """T10U-4 (Codex counterexample): windows of 10 000 and 10 100 ms, idle bound 40 % and 0 %: the run bound
    is the time mean, 100 − 40·10000/20100, not the plain mean 80 %."""
    rows = [dict(idle_pct_lo=40.0, win_ms=10000), dict(idle_pct_lo=0.0, win_ms=10100)]
    assert ser.load_upper(rows)['load_upper_pct'] == pytest.approx(100.0 - 40.0 * 10000 / 20100, abs=1e-12)


def test_idle_lower_formula_large_uncertainty():
    """T10U-7: idle 40 %, relative uncertainty 0.2 -> idle bound 32 %, load bound 68 % (idle / (1 + unc)
    would give 66.67 %); absolute tolerance so a near formula cannot pass."""
    assert ser.idle_lower_pct(40.0, 0.2, 1) == pytest.approx(32.0, abs=1e-12)
    assert 100.0 - ser.idle_lower_pct(40.0, 0.2, 1) == pytest.approx(68.0, abs=1e-12)


def _run_summary(lines):
    ev = ser.parse_lines(lines)
    blocks, span = ser.t10_blocks(ev)
    win, irq = ser.t10_rows(blocks)
    return ser.t10_summary(blocks, win, irq, ev, span), win


def test_idle_out_of_window_or_inconsistent_is_data_lost():
    """T10U-1: idle_us_est above the window (Codex: 12 000 000 µs in 10 s) or not equal to idle_pass·cal
    (8 000 000 instead of 4 000 000): no bound for that window, data lost (code 1), never a negative bound."""
    for new in ('idle_pass=2700000 idle_us_est=12000000', 'idle_pass=900000 idle_us_est=8000000'):
        s, win = _run_summary(with_header(make_log(), 2, 'idle_pass=900000 idle_us_est=4000000', new))
        bad = [w for w in win if w['idle_bad']]
        assert len(bad) == 1 and bad[0]['load_upper_pct'] is None
        assert all(w['load_upper_pct'] is None or 0.0 <= w['load_upper_pct'] <= 100.0 for w in win)
        assert any('số rảnh' in p for p in s['data_lost']) and s['complete'] is False


def test_lost_header_after_the_dropped_one_is_found():
    """T10U-2 (Codex): the header of the first KEPT window (20 s after link up) is damaged, its IRQ lines merge
    into the dropped first block; the kept headers alone are 10 s apart, all headers of the span are not."""
    s, _ = _run_summary(with_header(make_log(), 1, 'ovh_cyc=24', 'ovh_cyc='))
    assert any('mất khối' in p for p in s['data_lost']) and s['complete'] is False


def test_lost_first_header_is_found():
    """T10U-2: the first header of the span is damaged: the next one comes 20 s after link up."""
    s, _ = _run_summary(with_header(make_log(), 0, 'ovh_cyc=24', 'ovh_cyc='))
    assert any('mất khối đầu' in p for p in s['data_lost']) and s['complete'] is False


def test_no_calibration_is_missing():
    """T10U-5: cal=0/… gives no bound; the run is not complete (code 3), not a pass with an empty bound."""
    lines = [l.replace('idle_us_est=4000000 cal=22500/48000000', 'idle_us_est=0 cal=0/48000000') for l in make_log()]
    s, win = _run_summary(lines)
    assert all(w['load_upper_pct'] is None for w in win)
    assert s['data_lost'] == [] and any('hiệu chuẩn' in m for m in s['missing']) and s['complete'] is False


def test_idle_lower_edge_cases():
    assert ser.idle_lower_pct(None, None, 5) is None                     # no calibration
    assert ser.idle_lower_pct(0.0, None, 0) == 0.0                       # no idle pass: load bound 100 %
    assert ser.idle_lower_pct(1.0, 2.0, 1) == 0.0                        # never below 0
    blocks, _ = ser.t10_blocks(ser.parse_lines(
        [l.replace('idle_pass=900000 idle_us_est=4000000', 'idle_pass=0 idle_us_est=0') for l in make_log()]))
    w = ser.t10_rows(blocks)[0][0]
    assert w['idle_rel_unc'] is None and w['load_upper_pct'] == 100.0


def test_t10_load_upper_heavy_block_and_cli(tmp_path):
    """T10U-8: windows differ by segment: the heavy window ending 50 s (k = 4) has 20 % idle (load bound ~80 %), every
    other window 40 % (~60 %): the heavy mean, heavy max, group A and run values must each come out right."""
    unc20 = 1 / 22500 + 1 / 450_000
    code, s, rows = t10_with_schedule(tmp_path, edit=lambda ls: with_header(ls, 4, *IDLE20))
    assert code == 0
    lo40, lo20 = 40.0 * (1 - UNC), 20.0 * (1 - unc20)
    hv = s['segments']['heavy']
    assert hv['load_upper_pct'] == pytest.approx(100.0 - (3 * lo40 + lo20) / 4, abs=1e-9)
    assert hv['load_upper_pct_max_window'] == pytest.approx(100.0 - lo20, abs=1e-9)
    na = s['segments']['nhom_a']
    assert na['load_upper_pct'] == pytest.approx(100.0 - lo40, abs=1e-9)
    assert na['load_upper_pct_max_window'] == pytest.approx(100.0 - lo40, abs=1e-9)
    assert s['load_upper_pct'] == pytest.approx(100.0 - (6 * lo40 + lo20) / 7, abs=1e-9)   # 7 kept windows
    assert s['load_upper_pct_max_window'] == pytest.approx(100.0 - lo20, abs=1e-9)
    assert 'idle_pct_lo' in rows.splitlines()[0] and 'load_upper_pct' in rows.splitlines()[0]


def test_idle_centre_over_window_within_uncertainty_is_valid():
    """T10U-R2-1 (Codex round 2): a header that matches the H7 integer formula exactly, idle centre + ISR
    4 us over the window but the idle lower bound inside: a valid window with a bound, not data lost."""
    new = 'idle_pass=1980001 idle_us_est=8800004'
    s, win = _run_summary(with_header(make_log(), 2, 'idle_pass=900000 idle_us_est=4000000', new))
    w = [x for x in win if x['idle_pass'] == 1980001][0]
    assert w['idle_bad'] == '' and w['load_upper_pct'] is not None and 0.0 <= w['load_upper_pct'] <= 100.0
    assert not any('số rảnh' in p for p in s['data_lost'])


def test_first_header_late_within_print_delay_is_not_lost():
    """T10U-R2-2 (Codex round 2): every header 0.8 s late (inside PRINT_DELAY_MAX_MS), link-up line on time:
    no block lost."""
    ev = ser.parse_lines(make_log())
    for e in ev:
        if e['kind'] in ('t10', 't10_irq') and e.get('t_pc_us') is not None and e['t_pc_us'] > ev[0]['t_pc_us'] + 5e6:
            e['t_pc_us'] += 800_000
    blocks, span = ser.t10_blocks(ev)
    win, irq = ser.t10_rows(blocks)
    s = ser.t10_summary(blocks, win, irq, ev, span)
    assert not any('mất khối' in p for p in s['data_lost']), s['data_lost']
    assert any('dải trễ in' in m for m in s['missing'])         # T10U-R3-1: 10.8 s is in the ambiguous band


def test_no_calibration_and_no_idle_pass_is_missing():
    """T10U-5 (Codex round 2): cal=0/… and idle_pass=0: still no bound, so 'missing', not complete."""
    lines = [l.replace('idle_pass=900000 idle_us_est=4000000 cal=22500/48000000',
                       'idle_pass=0 idle_us_est=0 cal=0/48000000') for l in make_log()]
    s, win = _run_summary(lines)
    assert all(w['load_upper_pct'] is None for w in win)
    assert any('hiệu chuẩn' in m for m in s['missing']) and s['complete'] is False


def test_cli_with_damaged_header_writes_outputs(tmp_path):
    """T10U-R2-3 (Codex round 2): the data-lost path goes through the CSV writer to T10_summary.json and code 1
    (CSV written as UTF-8, idle_bad text ASCII)."""
    log = tmp_path / 'h7.log'
    log.write_text('\n'.join(with_header(make_log(), 2, 'idle_pass=900000 idle_us_est=4000000',
                                         'idle_pass=2700000 idle_us_est=12000000')) + '\n', encoding='utf-8')
    out = tmp_path / 'o'
    code = ser.main([str(log), '--out', str(out)])
    s = json.load(open(out / 'T10_summary.json', encoding='utf-8'))
    rows = (out / 'T10_windows.csv').read_text(encoding='utf-8')
    assert code == 1 and s['complete'] is False and 'idle_lo + isr_us > window' in rows
    assert s['print_delay_max_ms'] == ser.PRINT_DELAY_MAX_MS


def first_head_log(rel_heads, spi_n, end_rel, first_idle=None, drop_first_spi=False, gaps_ms=5):
    """T10U-R3-1: a run with link up at 1 s and the T10 headers at `rel_heads` s after it (PC times), SPI1 n of
    each block from `spi_n`; S06log lines at link up and at `end_rel`, then link lost and the 'Link period' line
    with `gaps_ms` ms more H7 time than frame intervals (None: no such line)."""
    t_up = 1.0
    out = [(0.0, '--- H753 HIL boot ---'), (0.5, link()), (0.5, frame(0)),
           (t_up, 'SPI link up at t=12345 ms'), (t_up, 'S06log: pushed=1 dropped=0  print_drop=0')]
    for k, (r, n) in enumerate(zip(rel_heads, spi_n)):
        lines = t10_block(spi_n=n).splitlines()
        if k == 0 and first_idle:
            lines[0] = lines[0].replace('idle_pass=900000 idle_us_est=4000000', first_idle)
        if k == 0 and drop_first_spi:
            lines = [l for l in lines if 'SPI1' not in l]
        out += [(t_up + r, l) for l in lines]
    out += [(t_up + end_rel, 'S06log: pushed=99 dropped=0  print_drop=0'),
            (t_up + end_rel + 0.05, 'SPI link lost (no frame for 50 ms): encoders stopped, SPI1 restart')]
    if gaps_ms is not None:
        nfr = int(end_rel * 1000)
        out.append((t_up + end_rel + 0.05, f'Link period: {nfr + 1} frames, H7 {nfr + gaps_ms} ms from first to last '
                                           f'frame (Pi: {nfr} ms at 1 kHz)'))
    out += [(t_up + end_rel + 0.5, link(lost=1, rearm=1)), (t_up + end_rel + 0.5, frame(40000))]
    return [f'{int(1e9 + t * 1e6)}\t{l}' for t, l in out]


def test_codex_r4_pi_gaps_make_the_spi1_check_missing():
    """Codex round 4 R3-1 case 1 (Liam 'Sua 2 + vong 5'): the header 0.3 s after link up lost, the Pi skipped 101
    sends in that window, so the next block has SPI1 n=9899 < 9900 and both checks passed (8f73799/781273c:
    complete). The run's Link period shows those gaps (H7 ms - frame intervals = 101): missing, not complete."""
    s, _ = _run_summary(first_head_log([10.3, 20.3, 30.3, 40.3], [9899, 10000, 10000, 10000], 40.95,
                                       first_idle='idle_pass=0 idle_us_est=0', gaps_ms=101))
    assert s['complete'] is False and s['data_lost'] == []
    assert any('không có căn cứ' in m and 'hở 101 ms' in m for m in s['missing'])
    s, _ = _run_summary(first_head_log([9.0, 19.0, 29.0], [9000, 10000, 10000], 31.0, gaps_ms=None))
    assert any('Link period' in m for m in s['missing'])                 # no Link period line: no ground
    s, _ = _run_summary(first_head_log([9.0, 19.0, 29.0], [9000, 10000, 10000], 31.0, gaps_ms=69))
    assert s['complete'] is True                                         # 69 + 30 < 100
    s, _ = _run_summary(first_head_log([9.0, 19.0, 29.0], [9000, 10000, 10000], 31.0, gaps_ms=70))
    assert s['complete'] is False                                        # 70 + 30 >= 100


def test_codex_r4_float_edge_bound_not_below_isr():
    """Codex round 4 R3-2: isr_us=2000360 with idle 8000000 µs fits the window exactly (idle_lo + isr = 10^7 µs)
    but 100 - idle_pct_lo came out 20.00359999999999 < isr_pct 20.0036. Bound now never below the ISR share."""
    hdr = 'T10: win_ms=10000 isr_us=2000360 idle_pass=1800000 idle_us_est=8000000 cal=22500/48000000 ovh_cyc=24'
    blocks, _ = ser.t10_blocks(ser.parse_lines([hdr]), all_blocks=True)
    w = ser.t10_rows(blocks)[0][0]
    assert w['idle_bad'] == '' and w['load_upper_pct'] >= w['isr_pct']
    assert ser.load_upper([w])['load_upper_pct_max_window'] >= w['isr_pct']


def test_codex_r3_1_lost_first_block_is_not_complete(tmp_path):
    """T10U-R3-1 (Codex round 3, exact case): the header 0.9 s after link up is lost; the next one, 10.9 s after
    it, ends a window with idle_pass=0 (load bound 100 %) and is dropped as 'first'; the kept windows give ~60 %.
    8f73799 said complete, code 0, max window ~60 %: the 100 % window was lost silently. Now: not complete."""
    log = tmp_path / 'h7.log'
    log.write_text('\n'.join(first_head_log([10.9, 20.9, 30.9, 40.9], [10000] * 4, 40.95,
                                            first_idle='idle_pass=0 idle_us_est=0')) + '\n', encoding='utf-8')
    out = tmp_path / 'o'
    code = ser.main([str(log), '--out', str(out)])
    s = json.load(open(out / 'T10_summary.json', encoding='utf-8'))
    assert code == 3 and s['complete'] is False and s['data_lost'] == []
    assert any('dải trễ in' in m for m in s['missing']) and any('SPI1 n=10000' in m for m in s['missing'])


def test_first_header_in_print_delay_band_is_missing():
    """T10U-R3-1, time band alone: first header 10.8 s after link up (inside (10 + tol; 10 + tol + d_max]), SPI1 n
    partial: may be a late first header or the second one: missing, not data lost, not complete."""
    s, _ = _run_summary(first_head_log([10.8, 20.8, 30.8], [5000, 10000, 10000], 31.0))
    assert s['data_lost'] == [] and s['complete'] is False
    assert any('dải trễ in' in m for m in s['missing']) and not any('SPI1 n=' in m for m in s['missing'])


def test_first_block_with_full_spi1_is_missing():
    """T10U-R3-1, SPI1 check alone: the header 0.3 s after link up lost, the next 10.3 s after it (inside
    10 + tol, which PC times cannot flag); its window lies wholly after link up, SPI1 n=10000: missing."""
    s, _ = _run_summary(first_head_log([10.3, 20.3, 30.3], [10000] * 3, 31.0))
    assert s['data_lost'] == [] and s['complete'] is False
    assert any('SPI1 n=10000' in m for m in s['missing']) and not any('dải trễ in' in m for m in s['missing'])
    s, _ = _run_summary(first_head_log([10.3, 20.3, 30.3], [9900, 10000, 10000], 31.0))     # edge: win_ms - 100
    assert any('SPI1 n=9900' in m for m in s['missing'])


def test_first_block_without_spi1_line_is_missing():
    s, _ = _run_summary(first_head_log([9.0, 19.0, 29.0], [9000, 10000, 10000], 31.0, drop_first_spi=True))
    assert s['complete'] is False and any('không có dòng SPI1' in m for m in s['missing'])


def test_first_block_partial_and_on_time_is_complete():
    """Control for the R3-1 checks: link up 1 s into the first window (header 9 s after it, SPI1 n=9000, the
    margin edge 9899 too): complete, code 0 path."""
    for n in (9000, 9899):
        s, _ = _run_summary(first_head_log([9.0, 19.0, 29.0], [n, 10000, 10000], 31.0))
        assert s['complete'] is True and s['problems'] == [], s['problems']


def test_first_header_beyond_band_is_data_lost():
    """T10U-R3-1: later than 10 + tol + d_max (11.6 s) the first header cannot be the first one: data lost."""
    s, _ = _run_summary(first_head_log([11.6, 21.6, 31.6], [5000, 10000, 10000], 32.0))
    assert any('mất khối đầu' in p for p in s['data_lost'])


def test_codex_r3_2_idle_lower_plus_isr_over_window_is_rejected():
    """T10U-R3-2 (Codex round 3, exact case): header matching both H7 integer divisions, idle lower bound + ISR
    = 10 000 000.43 µs > the 10 s window. 8f73799 let it through (1 µs tolerance) with a load bound
    11.99999573 % below the ISR share 12 %. Now: no bound for that window, data lost."""
    new = 'idle_pass=1980089 idle_us_est=8800396 cal=22500/48000003'
    s, win = _run_summary(with_header(make_log(), 2, 'idle_pass=900000 idle_us_est=4000000 cal=22500/48000000', new))
    w = [x for x in win if x['idle_pass'] == 1980089][0]
    assert w['idle_bad'] == 'idle_lo + isr_us > window' and w['load_upper_pct'] is None
    assert any('số rảnh' in p for p in s['data_lost']) and s['complete'] is False
    assert all(x['load_upper_pct'] is None or x['load_upper_pct'] >= x['isr_pct'] for x in win)


# ── rounds 6, 7: H7 prints wn (window number), t0_ms (window start tick), lk (link-up count) at the end of the header
# (Liam 15/11, CODE_S06 'H7 in so cua so + vong 6', 'Sua du + vong 7'; H7 feat/s06-pf3 344f21f) ──

M32 = 1 << 32


def wn_log(t_up=12345, h7_ms=33000, t0s=(5000, 15000, 25000, 35000), drop=(), after_lost=(), pi_ms=None,
           first_idle=None, wn0=1, lk=1, lks=None, wins=None, pre=None, post=(), suffix=None, period=True):
    """H7 ticks: link up at t_up (first frame), last frame at t_up + h7_ms; one header per window start in t0s
    (wn = wn0 + index, win_ms 10000 or wins[wn], lk or lks[wn]; printed at window end), `drop` = wn whose block is
    lost on the PC, `after_lost` = wn printed after the 'Link period' line. `pre` = (wn, t0, lk) of a header printed
    before the link-up line; `post` = (wn, t0, win) headers printed after the run (lk as the run). `suffix[wn]`
    replaces the ' wn=… t0_ms=… lk=…' suffix (damaged line). period=False: no 'Link period' line. PC time = tick."""
    wins = wins or {}
    lks = lks or {}
    suffix = suffix or {}

    def blk(wn, t0, win, lk_):
        lines = t10_block(win_ms=win).splitlines()
        if wn == wn0 + 1 and first_idle:
            lines[0] = lines[0].replace('idle_pass=900000 idle_us_est=4000000', first_idle)
        lines[0] = lines[0] + suffix.get(wn, f' wn={wn} t0_ms={t0} lk={lk_}')
        return [((t0 + win) % M32 + 1, l) for l in lines]
    t_last = t_up + h7_ms
    out = [(0, link()), (0, frame(0))]
    if pre is not None:
        out += blk(pre[0], pre[1], 10000, pre[2])
    out += [(t_up, f'SPI link up at t={t_up % M32} ms'), (t_up, 'S06log: pushed=1 dropped=0  print_drop=0')]
    late = []
    for k, t0 in enumerate(t0s):
        wn = wn0 + k
        if wn in drop:
            continue
        (late if wn in after_lost else out).extend(blk(wn, t0, wins.get(wn, 10000), lks.get(wn, lk)))
    n = h7_ms + 1
    out += [(t_last, 'S06log: pushed=99 dropped=0  print_drop=0'),
            (t_last + 50, 'SPI link lost (no frame for 50 ms): encoders stopped, SPI1 restart')]
    if period:
        out += [(t_last + 50, f'Link period: {n} frames, H7 {h7_ms} ms from first to last frame '
                              f'(Pi: {n - 1 if pi_ms is None else pi_ms} ms at 1 kHz)')]
    out += late
    for wn, t0, win in post:
        out += blk(wn, t0, win, lk)
    out += [(t_last + 500, link(lost=1, rearm=1)), (t_last + 500, frame(n))]
    return [f'{int(1e9 + t * 1000)}\t{l}' for t, l in out]


def test_wn_clean_run_keeps_the_full_windows():
    s, win = _run_summary(wn_log())
    assert s['complete'] is True, s['problems']
    assert [w['line'] for w in win] and s['windows'] == 3                 # 15-25, 25-35, 35-45 s; 5-15 s partial


def test_wn_lost_partial_and_first_full_window():
    """Codex rounds 3-5 R3-1 with H7 numbers: the window holding link up and the first full one (load 100 %) both
    lost on the PC. With the header printed before link up (start anchor) the numbers jump: data lost. Without it
    (round 7) nothing before the first kept window proves the start: missing, not complete."""
    s, _ = _run_summary(wn_log(t_up=22345, t0s=(15000, 25000, 35000, 45000), wn0=2, lk=2, pre=(1, 5000, 1),
                               drop=(2, 3)))
    assert any('số cửa sổ T10 nhảy 1 → 4' in p for p in s['data_lost']), s['problems']
    s, win = _run_summary(wn_log(drop=(1, 2)))
    assert s['data_lost'] == [] and any('không chứng minh được không mất khối đầu' in m for m in s['missing'])
    assert s['complete'] is False and s['windows'] == 2


def test_wn_gap_and_last_window_lost():
    s, _ = _run_summary(wn_log(drop=(3,)))
    assert any('số cửa sổ T10 nhảy 2 → 4' in p for p in s['data_lost'])
    # last full window lost: the next header (window past the last frame, printed after the run) shows the jump
    s, _ = _run_summary(wn_log(drop=(4,), post=((5, 45000, 10000),)))
    assert any('số cửa sổ T10 nhảy 3 → 5' in p for p in s['data_lost']), s['problems']
    # ... and without it (log stopped) a whole window fits after the last kept one: missing
    s, _ = _run_summary(wn_log(drop=(4,)))
    assert s['data_lost'] == [] and any('không chứng minh được không mất khối cuối' in m for m in s['missing'])


def test_wn_header_printed_after_link_lost_still_counts():
    s, _ = _run_summary(wn_log(after_lost=(4,)))
    assert s['complete'] is True and s['windows'] == 3, s['problems']


def test_wn_ambiguous_first_window_and_bad_link_period():
    # window 1 (4999 .. 15050, 10051 ms, header lost) holds link up at 5000: the first full one starts 10050 ms
    # after link up and nothing before it was received: not decided (missing), not data lost
    s, _ = _run_summary(wn_log(t_up=5000, t0s=(4999, 15050, 25050, 35050), wins={1: 10051}, drop=(1,), h7_ms=40100))
    assert s['data_lost'] == [] and any('không chứng minh được không mất khối đầu' in m for m in s['missing'])
    s, _ = _run_summary(wn_log(pi_ms=12))                                # Codex round 5: pi_ms != frames - 1
    assert any('Link period hỏng' in m for m in s['missing'])


def test_codex_r6_1_link_up_on_a_window_start():
    """T10U-R6-1: link up on the very tick a window starts (S = 15000); that full window (load 100 %) lost with its
    header. With the header before link up the numbers jump 1 -> 3 (data lost); without it the first kept window
    starts exactly 10 000 ms after link up: missing (1ec34a8: complete, code 0). Control: nothing lost, complete."""
    kw = dict(t_up=15000, h7_ms=40050, t0s=(15000, 25000, 35000, 45000), wn0=2, lk=2)
    s, _ = _run_summary(wn_log(pre=(1, 5000, 1), drop=(2,), **kw))
    assert any('số cửa sổ T10 nhảy 1 → 3' in p for p in s['data_lost']), s['problems']
    s, _ = _run_summary(wn_log(drop=(2,), **kw))
    assert s['complete'] is False and any('không chứng minh được không mất khối đầu' in m for m in s['missing'])
    s, _ = _run_summary(wn_log(pre=(1, 5000, 1), **kw))
    assert s['complete'] is True and s['windows'] == 4, s['problems']


def test_codex_r6_2_end_of_run():
    """T10U-R6-2: H7 344f21f windows are >= 10 000 ms. Last kept window ends 45000, last frame 55000, no later header
    (capture stopped): a full window may or may not fit, missing (1ec34a8: data lost). The next header (10 051 ms,
    past the last frame) printed after the run: complete. Fewer than 10 000 ms left: complete. A window shorter
    than 10 000 ms cannot come from that build: damaged, data lost."""
    s, _ = _run_summary(wn_log(h7_ms=42655))
    assert s['data_lost'] == [] and any('không chứng minh được không mất khối cuối' in m for m in s['missing'])
    s, _ = _run_summary(wn_log(h7_ms=42655, post=((5, 45000, 10051),)))
    assert s['complete'] is True and s['windows'] == 3, s['problems']
    s, _ = _run_summary(wn_log(h7_ms=42654))
    assert s['complete'] is True, s['problems']
    s, _ = _run_summary(wn_log(t0s=(5000, 15000, 25000, 34999), wins={3: 9999}))
    assert any('ngắn hơn 10000 ms' in p for p in s['data_lost']), s['problems']


def test_codex_r6_3_two_link_periods():
    """T10U-R6-3: link lost and up again inside the run. With the link lines: missing, no window (1ec34a8 kept the
    windows, code 0). With both lines of the break lost, lk changes among the headers: missing."""
    lines = wn_log(t_up=12345, h7_ms=42655, t0s=(5000, 15000, 25000, 35000, 45000))
    i = next(k for k, l in enumerate(lines) if 'T10: ' in l and ' wn=3 ' in l)
    ts = lines[i].split('\t')[0]
    brk = [f'{ts}\tSPI link lost (no frame for 50 ms): encoders stopped, SPI1 restart',
           f'{ts}\tLink period: 7605 frames, H7 7604 ms from first to last frame (Pi: 7604 ms at 1 kHz)',
           f'{ts}\tSPI link up at t=22000 ms']
    s, win = _run_summary(lines[:i] + brk + lines[i:])
    assert s['complete'] is False and s['windows'] == 0 and any('link mất rồi lên lại' in m for m in s['missing'])
    s, win = _run_summary(wn_log(t_up=12345, h7_ms=42655, t0s=(5000, 15000, 25000, 35000, 45000), lks={4: 2, 5: 2}))
    assert s['windows'] == 0 and any('lk đổi' in m for m in s['missing']), s['problems']


def test_codex_r6_4_tick_wrap():
    """T10U-R6-4: link up at 2^32 − 25000, the HAL tick wraps inside the run; windows tile across the wrap: complete,
    3 full windows (1ec34a8: the window starting at 0 dropped, tiling 'broken', data lost)."""
    t0s = (M32 - 30000, M32 - 20000, M32 - 10000, 0, 10000)
    s, win = _run_summary(wn_log(t_up=M32 - 25000, h7_ms=40000, t0s=t0s))
    assert s['complete'] is True and s['windows'] == 3, s['problems']


def test_codex_r6_5_gap_after_the_run_is_not_the_run():
    """T10U-R6-5: windows 1-4 of the run all received; after the run wn 5 received, wn 6 lost, wn 7 received: the
    run is complete (1ec34a8: jump 5 -> 7, data lost)."""
    s, _ = _run_summary(wn_log(post=((5, 45000, 10000), (7, 65000, 10000))))
    assert s['complete'] is True and s['windows'] == 3, s['problems']


def test_codex_r6_6_damaged_suffix_and_late_headers():
    """T10U-R6-6: header 4 (printed after link lost) with a damaged suffix: data lost, no TypeError (1ec34a8 raised).
    Every header of the run printed after link lost: the windows are still found (1ec34a8: none)."""
    s, _ = _run_summary(wn_log(drop=(1,), after_lost=(4,), suffix={4: ' wn=4 t0_ms='}))
    assert any('dòng hỏng' in p for p in s['data_lost']), s['problems']
    s, _ = _run_summary(wn_log(after_lost=(1, 2, 3, 4)))
    assert s['complete'] is True and s['windows'] == 3, s['problems']


def test_link_period_of_a_later_segment_not_used():
    """Codex rounds 5, 6: link lost A, Link period A lost, link up B lost, link lost B, Link period B: B's line must
    not stand for the run (1ec34a8 took it)."""
    lines = wn_log(period=False)
    i = next(k for k, l in enumerate(lines) if 'SPI link lost' in l)
    ts = lines[i].split('\t')[0]
    lines = lines[:i + 1] + [f'{ts}\tSPI link lost (no frame for 50 ms): encoders stopped, SPI1 restart',
                             f'{ts}\tLink period: 1001 frames, H7 1000 ms from first to last frame '
                             '(Pi: 1000 ms at 1 kHz)'] + lines[i + 1:]
    s, _ = _run_summary(lines)
    assert any('không có dòng Link period ngay sau link lost' in m for m in s['missing']), s['problems']


def test_wn_without_lk_is_not_accepted():
    """60c69df (wn, t0_ms, no lk; windows could be 9 999 ms) was never flashed: its headers are not accepted."""
    lines = [l.replace(' lk=1', '') for l in wn_log()]
    s, _ = _run_summary(lines)
    assert s['complete'] is False and any('không có lk' in m for m in s['missing'])


def test_start_anchor_of_another_boot_is_ignored():
    """The header before link up is of an earlier H7 boot (wn not below the run's, or lk not one below): not used
    as start anchor; the partial window of the run still proves the start."""
    s, _ = _run_summary(wn_log(pre=(9, 1000, 0)))
    assert s['complete'] is True and s['windows'] == 3, s['problems']
    s, _ = _run_summary(wn_log(pre=(9, 1000, 0), drop=(1, 2)))
    assert s['data_lost'] == [] and any('không chứng minh được không mất khối đầu' in m for m in s['missing'])
