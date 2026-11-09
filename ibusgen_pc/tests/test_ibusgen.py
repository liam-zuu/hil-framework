"""PC tests of the H7 iBUS generator (stm32h7/Core/Src/ibus_gen.c).

The harness (ibusgen_pc/harness.c) runs the H7 file unchanged against an
event model of the UART TX FIFO, shift register and 115200 8N1 line. The
wire stream is read back two ways:
  - esp32_parse(): a copy of the ESP32 parser (AGV_MECANUM src/ibus/ibus.c:
    header state machine, checksum 0xFFFF - sum(bytes 0..29), 10 channels);
  - esp32_deliveries(): when the ESP32 UART driver hands bytes to ibus_task.
    Assumed from the ESP-IDF driver code (not measured): a burst is handed
    over after the line is idle for 10 character times, or when 120 bytes
    have arrived. ibus_task waits 20 ms = 2 FreeRTOS ticks at 100 Hz, which
    is 10..20 ms of real time depending on the tick phase, so a hand-over
    gap under 10 ms never times out, one of 20 ms or more always does.

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
RX_IDLE_US = 10 * BYTE_US
RX_FULL = 120
SAFE_GAP_US = 10000.0       # below: no timeout at any tick phase
SURE_TIMEOUT_US = 20000.0   # at or above: timeout at every tick phase
WINDOW_TICKS = 100000       # 10 s at 100 us


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


def esp32_deliveries(t_us):
    """Times at which the ESP32 driver hands received bytes to ibus_task."""
    out, n = [], 0
    for i, t in enumerate(t_us):
        end = t + BYTE_US
        n += 1
        if n >= RX_FULL:
            out.append(end)
            n = 0
            continue
        nxt = t_us[i + 1] if i + 1 < len(t_us) else None
        if nxt is None or nxt - end >= RX_IDLE_US:
            if n:
                out.append(end + RX_IDLE_US)
            n = 0
    return out


def max_gap(xs):
    return max((b - a for a, b in zip(xs, xs[1:])), default=0.0)


def headers(d):
    """Wire start time of every frame header (0x20 0x40 at a 32-byte boundary)."""
    t, v = d['t_us'], d['val']
    return [t[k] for k in range(0, len(t) - 1, 32) if v[k:k + 2] == [0x20, 0x40]]


def frames_on_wire(d):
    """Split the wire into 32-byte frames; check each is one unbroken burst.
    The run may stop while the generator is still writing a frame: that tail
    is dropped (it is the start of a frame, checked)."""
    t, v = d['t_us'], d['val']
    assert len(t) == len(v), 'bytes left in the FIFO'
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


def p_on_wire(d, ticks=WINDOW_TICKS, tick_us=100.0):
    """Header-to-header gaps on the wire inside the first window, only
    between frames of one on period (the caller passes runs without off)."""
    hs = [h for h in headers(d) if h <= ticks * tick_us]
    return [b - a for a, b in zip(hs, hs[1:])]


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


def test_period_floor_on_the_wire(harness):
    assert run(harness, 'frame')['period_us'] == 7000
    d = run(harness, 'run', 2000, 'period=1000')
    assert d['period_us'] == 3700
    hs = headers(d)
    assert len(hs) > 40
    assert min(b - a for a, b in zip(hs, hs[1:])) >= 3700 - 1e-6
    frames_on_wire(d)
    assert max_gap(esp32_deliveries(d['t_us'])) < SAFE_GAP_US


def test_steady_20_s(harness):
    d = run(harness, 'run', 2 * WINDOW_TICKS)
    assert len(d['windows']) == 2
    for w in d['windows']:
        assert (w['p_min'], w['p_max']) == (7000, 7000)
        assert w['late'] == 0 and w['enabled'] == 1 and w['tick_gap_max'] == 100
        assert w['frames'] in (1428, 1429)
    assert d['windows'][1]['total'] == d['windows'][0]['frames'] + d['windows'][1]['frames']
    fr = frames_on_wire(d)
    starts = [s for s, _ in fr]
    assert starts[0] == 100.0            # header on the wire in the tick that starts the frame
    assert all(b - a == pytest.approx(7000.0, abs=1e-3) for a, b in zip(starts, starts[1:]))
    good, bad = esp32_parse(d['val'])
    assert bad == 0 and len(good) == len(fr)
    assert d['total_frames'] - len(fr) in (0, 1)
    dl = esp32_deliveries(d['t_us'])
    assert len(dl) - len(fr) in (0, 1)   # + the tail frame cut by the end of the run
    assert max_gap(dl) == pytest.approx(7000.0, abs=1e-3)


def test_h7_tick_from_the_fast_clock(harness):
    # TIM6 on a clock 0.64 % fast (the H7 before the HSE/5 fix of 09/11, bug
    # H7-CLK-HSE): 100 us of H7 time is 99.36 us. Kept as a robustness case.
    d = run(harness, 'run', WINDOW_TICKS + 700, 'tickus=99.36')
    w = d['windows'][0]
    assert 7000 <= w['p_min'] <= w['p_max'] <= 7100
    assert max(p_on_wire(d, tick_us=99.36)) == pytest.approx(w['p_max'], abs=1.0)
    assert max_gap(esp32_deliveries(d['t_us'])) < SAFE_GAP_US


def test_disable_mid_frame_finishes_that_frame_only(harness):
    d = run(harness, 'run', 2000, 'off@5')
    assert d['bytes'] == 32 and d['total_frames'] == 1
    frames_on_wire(d)
    assert esp32_parse(d['val']) == ([esp32_parse(d['val'])[0][0]], 0)


def test_quick_off_on_keeps_the_spacing(harness):
    # Codex I2: off then on while the first frame is still being written
    d = run(harness, 'run', 3000, 'off@2', 'on@3')
    hs = headers(d)
    assert len(hs) >= 3
    assert min(b - a for a, b in zip(hs, hs[1:])) >= 7000 - 1e-6
    frames_on_wire(d)
    assert max_gap(esp32_deliveries(d['t_us'])) < SAFE_GAP_US


def test_off_gap_is_not_a_p_and_the_dut_times_out(harness):
    d = run(harness, 'run', WINDOW_TICKS, 'off@1000', 'on@5000')
    w = d['windows'][0]
    assert w['p_max'] == 7000 and w['p_min'] == 7000
    assert w['enables'] == 2             # first tick + the one at tick 5000
    frames_on_wire(d)
    dl = esp32_deliveries(d['t_us'])
    gaps = [b - a for a, b in zip(dl, dl[1:])]
    assert sum(g >= SURE_TIMEOUT_US for g in gaps) == 1      # the off period
    assert all(g < SAFE_GAP_US for g in gaps if g < SURE_TIMEOUT_US)
    assert esp32_parse(d['val'])[1] == 0


def test_line_held_at_the_due_time(harness):
    # Codex I3: the header waits for an idle line; P is what the wire shows
    d = run(harness, 'run', WINDOW_TICKS, 'block@3500:200')
    w = d['windows'][0]
    assert w['late'] > 0 and w['p_min'] == 7000
    assert w['p_max'] == pytest.approx(max(p_on_wire(d)), abs=1.0)
    assert w['p_max'] > 20000
    frames_on_wire(d)
    good, bad = esp32_parse(d['val'])
    assert bad == 0 and len(good) == len(frames_on_wire(d))


def test_line_held_with_bytes_in_the_fifo(harness):
    # Codex I6: queued bytes must not leave while the line is held
    d = run(harness, 'run', 3000, 'block@5:200')
    t = d['t_us']
    assert not any(500.0 <= x < 20500.0 for x in t)
    assert any(x >= 20500.0 for x in t[:32])            # rest of frame 1 after the hold
    good, bad = esp32_parse(d['val'])
    assert bad == 0 and len(good) >= 2
    w_hs = headers(d)
    assert min(b - a for a, b in zip(w_hs, w_hs[1:])) >= 7000 - 1e-6


def test_tim6_not_served_shows_in_p_and_tick_gap(harness):
    # Codex I4: two TIM6 updates merged must not hide the time lost
    d = run(harness, 'run', WINDOW_TICKS, 'stall@3565:20')
    w = d['windows'][0]
    assert w['tick_gap_max'] == 2100
    assert w['p_max'] == pytest.approx(max(p_on_wire(d)), abs=1.0)
    assert w['p_max'] > 7000
    frames_on_wire(d)
    assert max_gap(esp32_deliveries(d['t_us'])) < SAFE_GAP_US


def test_no_frames_while_off(harness):
    d = run(harness, 'run', WINDOW_TICKS, 'off@1')
    assert d['bytes'] == 0 and d['total_frames'] == 0
    assert d['windows'][0]['enabled'] == 0 and d['windows'][0]['enables'] == 0


# ── I8: SWA from the RPi5 (Liam 09/11, "Duyet ca 5") ──

def _swa_seq(d):
    """SWA (channel 7) of every frame the ESP32 parser accepts, in order."""
    good, bad = esp32_parse(d['val'])
    assert bad == 0
    return [g[6] for g in good]


def test_frame_off_differs_only_in_swa(harness):
    r = run(harness, 'frame')
    on, off = r['frame'], r['frame_off']
    ch_on = [on[2 + 2 * i] | (on[3 + 2 * i] << 8) for i in range(14)]
    ch_off = [off[2 + 2 * i] | (off[3 + 2 * i] << 8) for i in range(14)]
    assert ch_off[6] == 1000 and not ch_off[6] > 1500          # ESP32: motors off
    assert [c for i, c in enumerate(ch_off) if i != 6] == [c for i, c in enumerate(ch_on) if i != 6]
    good, bad = esp32_parse(off)
    assert bad == 0 and len(good) == 1                          # checksum right for the second frame too
    assert r['last_swa'] == 0                                   # no frame started yet


def test_swa_change_mid_frame_waits_for_the_next_frame(harness):
    # frame 1 starts at tick 1 and is on the wire for about 2.8 ms (28 ticks): ask SWA off at tick 5
    d = run(harness, 'run', 3000, 'swa0@5')
    frames_on_wire(d)
    seq = _swa_seq(d)
    assert seq[0] == 2000 and len(seq) >= 3
    assert all(s == 1000 for s in seq[1:])
    assert d['last_swa'] == 0


def test_swa_off_still_sends_frames(harness):
    # SWA off is "RC on, switch down": frames keep coming, the DUT does not time out
    d = run(harness, 'run', WINDOW_TICKS, 'swa0@1')
    w = d['windows'][0]
    assert w['frames'] in (1428, 1429) and w['swa_frames'] == 0 and w['swa_changes'] == 0 and w['swa'] == 0
    assert set(_swa_seq(d)) == {1000}
    assert max_gap(esp32_deliveries(d['t_us'])) < SAFE_GAP_US


def test_swa_toggles_counted_once_per_change(harness):
    d = run(harness, 'run', WINDOW_TICKS, 'swa0@1000', 'swa1@5000')
    w = d['windows'][0]
    seq = _swa_seq(d)
    changes = sum(a != b for a, b in zip(seq, seq[1:]))
    assert changes == 2 and w['swa_changes'] == 2
    assert w['swa_frames'] == seq[:w['frames']].count(2000)
    assert w['swa'] == 1
    assert max_gap(esp32_deliveries(d['t_us'])) < SAFE_GAP_US


def test_swa_on_but_link_down_sends_nothing(harness):
    # link down wins over the SWA command: the DUT loses RC (S06 T12 (d))
    d = run(harness, 'run', 2000, 'off@1', 'swa1@1')
    assert d['bytes'] == 0 and d['last_swa'] == 0


# ── I8-3: SWA filter of the main loop (Codex round 1, Liam 09/11 "Duyet sua, vong 2") ──

def _filt(harness, bits, *resets):
    r = run(harness, 'filter', bits, *[f'r@{k}' for k in resets])
    assert r['n'] == 3
    return ''.join(str(x) for x in r['out'])


def test_swa_filter_needs_three_in_a_row(harness):
    assert _filt(harness, '0001111') == '0000011'
    assert _filt(harness, '1110000') == '0011100'      # off also needs 3 frames


def test_swa_filter_one_corrupt_frame_changes_nothing(harness):
    assert _filt(harness, '0001000') == '0000000'      # one flipped 0 -> 1
    assert _filt(harness, '1111011111') == '0011111111'  # one flipped 1 -> 0 while on


def test_swa_filter_two_corrupt_frames_change_nothing(harness):
    assert _filt(harness, '00110000') == '00000000'


def test_swa_filter_reset_restarts_from_off(harness):
    # link lost / gap after the filter was on: output 0 again, 3 new frames needed
    assert _filt(harness, '11111111', 4) == '0011' + '0011'
