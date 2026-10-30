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
    assert spi['max_us'] == 35.5 and spi['max_us_real'] == pytest.approx(35.5 * 480 / 483.1)
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
    assert subprocess.run([sys.executable, script, str(log), '--out', str(out)], capture_output=True,
                          text=True).returncode == 3
    assert subprocess.run([sys.executable, script, str(tmp_path / 'none.log'), '--out', str(out)],
                          capture_output=True, text=True).returncode == 2
