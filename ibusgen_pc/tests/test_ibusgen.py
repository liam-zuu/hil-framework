"""PC tests of the H7 iBUS generator (stm32h7/Core/Src/ibus_gen.c).

The harness (ibusgen_pc/harness.c) runs the H7 file unchanged against a
model of the UART TX FIFO and the 115200 8N1 line. The wire stream is read
back with a copy of the ESP32 parser (AGV_MECANUM src/ibus/ibus.c: header
state machine, checksum 0xFFFF - sum(bytes 0..29), 10 channels LE).

  python -m pytest ibusgen_pc/tests -q      (needs gcc on PATH)
"""
import json
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PC = os.path.dirname(HERE)
REPO = os.path.dirname(PC)
BYTE_US = 1e6 * 10 / 115200
TICK_US = 100.0


@pytest.fixture(scope='session')
def harness(tmp_path_factory):
    gcc = shutil.which('gcc')
    if gcc is None:
        pytest.skip('gcc not on PATH')
    exe = str(tmp_path_factory.mktemp('ibusgen') / 'ibusgen_harness.exe')
    cmd = [gcc, '-std=gnu11', '-O2', '-Wall', '-Wextra', '-Werror',
           '-I', os.path.join(REPO, 'stm32h7', 'Core', 'Inc'),
           os.path.join(REPO, 'stm32h7', 'Core', 'Src', 'ibus_gen.c'),
           os.path.join(PC, 'harness.c'), '-o', exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


def run(exe, *args):
    r = subprocess.run([exe] + [str(a) for a in args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def esp32_parse(stream):
    """Copy of ibus_task + _parse_packet: returns (good frames, bad frames)."""
    good, bad = [], 0
    state, buf = 'H0', []
    for b in stream:
        if state == 'H0':
            if b == 0x20:
                buf, state = [b], 'H1'
        elif state == 'H1':
            if b == 0x40:
                buf.append(b)
                state = 'BODY'
            elif b == 0x20:
                buf = [b]
            else:
                state = 'H0'
        else:
            buf.append(b)
            if len(buf) == 32:
                chk = (0xFFFF - sum(buf[:30])) & 0xFFFF
                if chk == buf[30] | (buf[31] << 8):
                    good.append([buf[2 + 2 * i] | (buf[3 + 2 * i] << 8) for i in range(10)])
                else:
                    bad += 1
                state = 'H0'
    return good, bad


def frames_on_wire(d):
    """Split the wire into 32-byte frames; check each is one unbroken burst.
    The run may stop while the last frame is still being written: that tail
    is dropped (it is the start of a frame, checked)."""
    t, v = d['t_us'], d['val']
    tail = len(t) % 32
    if tail:
        assert v[len(v) - tail:len(v) - tail + 2] == [0x20, 0x40][:tail]
        t, v = t[:len(t) - tail], v[:len(v) - tail]
    out = []
    for k in range(0, len(t), 32):
        gaps = [t[i + 1] - t[i] for i in range(k, k + 31)]
        assert max(gaps) == pytest.approx(BYTE_US, abs=1e-3), 'gap inside a frame'
        out.append((t[k], v[k:k + 32]))
    return out


def test_frame_is_what_the_esp32_expects(harness):
    f = run(harness, 'frame')['frame']
    assert f[:2] == [0x20, 0x40]
    chans = [f[2 + 2 * i] | (f[3 + 2 * i] << 8) for i in range(14)]
    assert chans[4] == 1000          # SWC, channel 5: AUTO (MANUAL is > 1700)
    assert chans[6] == 2000          # SWA, channel 7: motors on (> 1500)
    assert all(c == 1500 for i, c in enumerate(chans) if i not in (4, 6))
    good, bad = esp32_parse(f)
    assert bad == 0 and len(good) == 1
    assert good[0][6] > 1500 and not good[0][4] > 1700


def test_period_floor(harness):
    assert run(harness, 'frame')['period_ticks'] == 70
    assert run(harness, 'run', 10, 'period=10')['period_ticks'] == 37
    assert run(harness, 'run', 10, 'period=37')['period_ticks'] == 37


def test_steady_20_s(harness):
    d = run(harness, 'run', 200000)
    assert len(d['windows']) == 2
    for w in d['windows']:
        assert (w['p_min'], w['p_max']) == (70, 70)
        assert w['late'] == 0 and w['enabled'] == 1
        assert w['frames'] in (1428, 1429)
    assert d['windows'][1]['total'] == d['windows'][0]['frames'] + d['windows'][1]['frames']
    fr = frames_on_wire(d)
    starts = [s for s, _ in fr]
    assert all(b - a == pytest.approx(7000.0, abs=1e-3) for a, b in zip(starts, starts[1:]))
    burst = 32 * BYTE_US
    assert 7000.0 - burst >= 870.0       # line idle long enough for the ESP32 driver
    good, bad = esp32_parse(d['val'])
    assert bad == 0 and len(good) == len(fr)
    assert d['total_frames'] - len(fr) in (0, 1)


def test_disable_mid_frame_finishes_that_frame_only(harness):
    d = run(harness, 'run', 2000, 'off@5')
    assert d['bytes'] == 32 and d['total_frames'] == 1
    good, bad = esp32_parse(d['val'])
    assert (len(good), bad) == (1, 0)


def test_off_gap_is_not_a_period(harness):
    d = run(harness, 'run', 100000, 'off@1000', 'on@5000')
    w = d['windows'][0]
    assert w['p_max'] == 70 and w['p_min'] == 70
    assert w['enables'] == 2             # initial enable + the one at tick 5000
    fr = frames_on_wire(d)
    gaps = [b[0] - a[0] for a, b in zip(fr, fr[1:])]
    assert max(gaps) >= (5000 - 1000) * TICK_US - 7000.0
    assert esp32_parse(d['val'])[1] == 0


def test_blocked_line_shows_as_late_and_long_p(harness):
    d = run(harness, 'run', 100000, 'block@3500:200')
    w = d['windows'][0]
    assert w['late'] > 0
    assert w['p_max'] > 70 and w['p_min'] == 70
    good, bad = esp32_parse(d['val'])
    assert bad == 0 and len(good) == len(frames_on_wire(d))


def test_no_frames_while_off(harness):
    d = run(harness, 'run', 100000, 'off@1')
    assert d['bytes'] == 0 and d['total_frames'] == 0
    assert d['windows'][0]['enabled'] == 0
