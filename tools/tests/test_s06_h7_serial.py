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


def t10_block(win_ms=10000, isr_us=1_200_000, idle_pass=900_000, idle_us_est=4_000_000, extra=''):
    return (f'T10: win_ms={win_ms} isr_us={isr_us} idle_pass={idle_pass} idle_us_est={idle_us_est} '
            f'cal=22500/48000000 ovh_cyc=24\r\n'
            'T10 SysTick n=10000 excl_us=2000 max_us=1.25\r\n'
            'T10 SPI1    n=10000 excl_us=300000 max_us=35.50\r\n'
            f'T10 TIM5    n=400000 excl_us=898000 max_us=4.08{extra}\r\n')


def link(lost=0, err=0, rearm=0):
    return f'Link: up lost={lost}  SPI frames=1000/s err={err} rearm={rearm} last_err=0x0'


def frame(applied, skipped=0, unread=0):
    return (f'Frame: unread={unread}  new seen=5 taken=5 repl=0 drop=0  applied={applied} skipped={skipped} '
            f'age_max=420us')


def make_log(blocks=4, print_drop_end=0, pre_up_block=True, ts_prefix=False, pc=True, drop_block=None,
             drop_irq=None, reconnect=False):
    """Lines as tools/h7_serial_capture.py writes them (`<t_pc_us>\\t<line>`, pc=False: as PuTTY); T10 every
    10 s, Link/Frame lines around the run so the counters are bounded at both ends."""
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
            lines = t10_block().splitlines()
            if drop_irq is not None and k == drop_irq:
                lines = [l for l in lines if 'SPI1' not in l]
            out += [(t + 10.0, l) for l in lines]
    t_end = 1.0 + 10.0 * blocks
    out += [(t_end, f'S06log: pushed=99 dropped=0  print_drop={print_drop_end}'),
            (t_end + 0.05, 'SPI link lost (no frame for 50 ms): encoders stopped, SPI1 restart'),
            (t_end + 0.05, 'Link period: 40001 frames, H7 40256 ms from first to last frame (Pi: 40000 ms at 1 kHz)'),
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
    assert st['link_periods'] == [dict(frames=40001, h7_ms=40256, pi_ms=40000)]


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
