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


def make_log(blocks=4, print_drop_end=0, pre_up_block=True, bad_win=None, ts_prefix=False):
    out = ['--- H753 HIL boot ---', 'Init done, entering main loop']
    if pre_up_block:
        out.append(t10_block(win_ms=10000, isr_us=1000, idle_pass=1_000_000).rstrip())
    out.append('SPI link up at t=12345 ms')
    out.append('S06log: pushed=1 dropped=0  print_drop=0')
    for k in range(blocks):
        out.append('Link: up lost=0  SPI frames=1000/s err=0 rearm=0 last_err=0x0')
        out.append('Frame: unread=0  new seen=5 taken=5 repl=0 drop=0  applied=%d skipped=0 age_max=420us'
                   % (10000 * (k + 1)))
        out.append(t10_block(win_ms=bad_win if (bad_win and k == 2) else 10000).rstrip())
    out.append(f'S06log: pushed=99 dropped=0  print_drop={print_drop_end}')
    out.append('SPI link lost (no frame for 50 ms): encoders stopped, SPI1 restart')
    out.append('Link period: 40001 frames, H7 40256 ms from first to last frame (Pi: 40000 ms at 1 kHz)')
    out.append(t10_block(isr_us=5).rstrip())                 # after the run: not used
    text = '\n'.join(out)
    if ts_prefix:
        text = '\n'.join(f'[12:00:{i:02d}.123] {l}' for i, l in enumerate(text.splitlines()))
    return text.splitlines()


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
    ev = ser.parse_lines(make_log(bad_win=20012))           # one block missing: window twice as long
    b, sp = ser.t10_blocks(ev)
    s = ser.t10_summary(b, *ser.t10_rows(b), ev, sp)
    assert not s['complete'] and '[1]' in s['problems'][0]


def test_link_stats_and_timestamp_prefix():
    ev = ser.parse_lines(make_log(ts_prefix=True))
    st = ser.link_stats(ev, ser.run_span(ev))
    assert st['applied'] == 30000 and st['skipped'] == 0 and st['err'] == 0 and st['age_max_us_since_boot'] == 420
    assert st['link_periods'] == [dict(frames=40001, h7_ms=40256, pi_ms=40000)]


def test_cli(tmp_path):
    log = tmp_path / 'h7.log'
    log.write_text('\n'.join(make_log()) + '\n', encoding='utf-8')
    out = tmp_path / 'out'
    script = os.path.join(ROOT, 'tools', 's06_h7_serial.py')
    p = subprocess.run([sys.executable, script, str(log), '--out', str(out)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    s = json.load(open(out / 'T10_summary.json', encoding='utf-8'))
    assert s['complete'] and s['windows'] == 3 and s['link']['applied'] == 30000
    assert open(out / 'T10.csv').readline().startswith('window,irq,count,excl_us')
    log.write_text('\n'.join(make_log(print_drop_end=1)) + '\n', encoding='utf-8')
    assert subprocess.run([sys.executable, script, str(log), '--out', str(out)], capture_output=True,
                          text=True).returncode == 3
    assert subprocess.run([sys.executable, script, str(tmp_path / 'none.log'), '--out', str(out)],
                          capture_output=True, text=True).returncode == 2
