"""tools/s06_t1_check.py (S06 T1) on synthetic frames built here from the MISO v4 layout (H7 feat/s06-h7:
answer at N+2, CRC-16/CCITT-FALSE, little-endian, one log record per frame). A clean run passes; each
fault put in on purpose must make T1 fail with the matching count. Not checked against frames of the
real H7. Run: python -m pytest tools/tests"""
import binascii
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'rpi5')]
import s06_t1_check as t1  # noqa: E402
from protocol import encode_state  # noqa: E402

N = 20_000
F_H7 = 483.1e6
FAST = dict(skip_s=1.0, window_s=5.0, min_s=10.0)


def crc(b):
    return binascii.crc_hqx(bytes(b), 0xFFFF)


def build(n=N, faults=None, period_us=1000.0):
    f = faults or {}
    rng = np.random.default_rng(5)
    t_send_ns = (np.arange(n) * period_us * 1e3 + rng.uniform(-12e3, 12e3, n)).astype(np.int64)
    mosi = np.zeros((n, 40), dtype=np.uint8)
    for k in range(n):
        w = 10 * np.sin(k / 300.0)
        st = [w, -w, w, -w, 0.001 * k % 5, 0.0, 0.0, 0.1, 0.0, 0.2]
        imu = (0.1 * np.sin(k / 50), 0.0, 0.0, 9.81, 0.0)
        mosi[k] = np.frombuffer(encode_state(st, seq=k, imu=imu, imu_seq=k // 10), dtype=np.uint8)
    miso = np.zeros((n, 40), dtype=np.uint8)
    idx, seqs, rx_shift = 0, [0, 0, 0], 0
    reps = []                                             # (kind 0..2, seq, Pi send time of the frame, µs)
    for j in range(2, n):
        i = j - 2
        if j in f.get('stale', ()):                       # main loop missed frame i: answer i-1 again
            i = j - 3
        if j >= f.get('h7_lost_from', n):                 # H7 never counted one frame
            rx_shift = 1
        cyc_full = int((t_send_ns[i] / 1e3 + 45.0) * 1e-6 * F_H7)
        cyc = cyc_full & 0xFFFFFFFF
        if j in f.get('empty', ()):                       # S1: the H7 did not answer this frame
            continue
        b = bytearray(40)
        mc = crc(mosi[i]) ^ (1 if j in f.get('mixed', ()) else 0)
        b[10:12] = struct.pack('<H', mc)
        b[12:16] = struct.pack('<I', cyc)
        b[17] = mosi[i][21]
        b[18:20] = struct.pack('<H', (i + 1 - rx_shift) & 0xFFFF)
        b[20:22] = struct.pack('<H', f.get('age', {}).get(j, 300))
        rec = None
        if i > 20 and i % 10 == 0 and i not in f.get('no_log', ()):
            s = (i // 10) % 3
            src = i - (f.get('imu_age', {}).get(i, 1))
            raw = bytes(mosi[src][k] for k in t1.dec.MOSI_IMU_BYTES)
            flags = 1 | (2 if i in f.get('qfull', ()) else 0)
            rec = (0x10 + s, bytes([seqs[s] & 0xFF, flags, 1, mosi[src][39]]) + struct.pack(
                '<HHHI', (src + 1) & 0xFFFF, (i + 1) & 0xFFFF, crc(raw), cyc))
            reps.append((s, seqs[s], t_send_ns[i] / 1e3))
            seqs[s] += 1
        elif i % 500 == 499 and not f.get('no_tick'):      # TICK: CYCCNT, _cyc_total (64 bit), dropped
            rec = (0x40, struct.pack('<IIIH', cyc, cyc_full & 0xFFFFFFFF, cyc_full >> 32, 0))
        if rec is not None:
            b[22] = rec[0]
            b[23:25] = struct.pack('<H', idx & 0xFFFF)
            b[25:25 + len(rec[1])] = rec[1]
            idx += 1 + (1 if i in f.get('log_gap', ()) else 0)
        b[8:10] = struct.pack('<H', crc(b[:8] + b[10:]))
        if j in f.get('miso_bit', ()):
            b[30] ^= 0x10
        miso[j] = np.frombuffer(bytes(b), dtype=np.uint8)
    return t_send_ns, mosi, miso, reps


def esp32_events(data, path=None, drop=()):
    """esp32_events.txt for build(): each H7 report reaches the ESP32 SPI port 300 µs after its frame (J),
    callback 50 µs later (I); ESP32 epoch = H7 epoch + 1; ESP32 clock = Pi clock; #run_start/#run_end
    inside the frames."""
    t = data[0]
    path = path or os.path.join(tempfile.mkdtemp(), 'esp32_events.txt')
    lines = [f'{int(t[0] / 1e3) + 200_000}\t#run_start,first_data_seq=1,timestamp_us=0,qseq=0']
    q = 0
    for s, seq, tp in sorted(data[3], key=lambda x: x[2]):
        q += 1
        if (s, seq) in drop:
            continue
        tj = int(tp) + 300
        lines.append(f'{tj}\tJ,{q},{tj},{(1, 2, 8)[s]},1,{seq % 256},2')
        lines.append(f'{tj + 50}\tI,{q},{tj + 50},{(1, 2, 8)[s]},1,{seq % 256},2,{seq},{seq},0,0,0')
    lines.append(f'{int(t[-1] / 1e3) - 100_000}\t#run_end,last_data_seq=1,timestamp_us=0,qseq={q + 1}')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return path


SERIAL = """--- H753 HIL boot ---
Init done, entering main loop
Link: down lost=0  SPI frames=0/s err={e0} rearm=0 last_err=0x0
Frame: unread=0  new seen=0 taken=0 repl=0 drop=0  applied=0 skipped={s0} age_max=0us
SPI link up at t=1200 ms
Link: up lost=0  SPI frames=1000/s err={e0} rearm=0 last_err=0x0
Frame: unread=0  new seen=10 taken=10 repl=0 drop=0  applied=500 skipped={s0} age_max=310us
S06log: pushed=10 dropped=0  print_drop=0
Link: up lost=0  SPI frames=1000/s err={e1} rearm=0 last_err=0x0
Frame: unread=0  new seen=20 taken=20 repl=0 drop=0  applied=20500 skipped={s1} age_max=320us
S06log: pushed=30 dropped=0  print_drop=0
SPI link lost (no frame for 50 ms): encoders stopped, SPI1 restart
Link period: 20001 frames, H7 20128 ms from first to last frame (Pi: 20000 ms at 1 kHz)
Link: down lost=1  SPI frames=0/s err={e1} rearm=1 last_err=0x0
Frame: unread=0  new seen=20 taken=20 repl=0 drop=0  applied=20501 skipped={s1} age_max=320us
"""


def serial(e0=0, e1=0, s0=0, s1=0, text=SERIAL):
    return t1.ser.parse_lines(text.format(e0=e0, e1=e1, s0=s0, s1=s1).splitlines())


@pytest.fixture(scope='module')
def clean():
    return build()


def run(data, full=True, **kw):
    """Score with the serial log and esp32_events of the run (full=True), as a real T1 run has them."""
    t, mosi, miso = data[:3]
    extra = dict(serial_events=serial(), esp32_events=esp32_events(data)) if full else {}
    return t1.check(t, mosi, miso, **{**FAST, **extra, **kw})


def test_clean_run_passes(clean):
    out = run(clean)
    assert out['verdict'] == 'pass', out['fails']
    h1 = out['h1']
    assert abs(h1['mean_period_us'] - 1000.0) < 1e-3 and h1['ok'] and h1['gaps_over_1p5ms'] == 0
    fr = out['frames']
    assert fr['frames_rx'] == fr['pi_answered'] > 0 and fr['seq_gaps'] == fr['miso_seq_bad'] == fr['mixed'] == 0
    assert out['imu']['n'] > 0 and out['imu']['age_over_1'] == 0
    cov = out['coverage_pct']
    assert cov['seq'] == pytest.approx(100.0) and cov['w1'] > 50 and cov['y'] == 0.0
    w = out['_windows']
    assert len(w) == 4 and all(x['seq_gaps'] == 0 and x['frames_rx'] == x['pi_frames'] for x in w)
    st = out['_steps']
    assert st[0]['crc_h7'] == st[0]['crc_sent'] and st[0]['miso_crc_ok']


def test_period_drift_fails():
    out = run(build(n=12_000, period_us=1000.01))
    assert out['verdict'] == 'fail' and not out['h1']['ok']


@pytest.mark.parametrize('faults, key, n', [
    ({'mixed': {5000}}, ('frames', 'mixed'), 1),
    ({'miso_bit': {6000}}, ('frames', 'miso_crc_bad'), 1),
    ({'age': {7000: 1500}}, ('frames', 'age_max_us'), 1500),
    ({'imu_age': {8000: 2}}, ('imu', 'age_over_1'), 1),
    ({'qfull': {9000}}, ('imu', 'queue_full'), 1),
    ({'log_gap': {10000}}, ('stream', 'log_gaps'), 1),
])
def test_single_fault_fails(faults, key, n):
    out = run(build(faults=faults))
    assert out['verdict'] == 'fail'
    assert out[key[0]][key[1]] == n


def test_stale_answer_is_seq_bad_and_gap():
    out = run(build(faults={'stale': {5000}}))
    fr = out['frames']
    # frame 5000 answers 4997 (lag 3, repeat of the previous answer) and 5001 answers 4999: frame 4998 skipped
    assert out['verdict'] == 'fail' and fr['miso_seq_bad'] == 2 and fr['seq_gaps'] == 1


def test_h7_missing_frame_count():
    out = run(build(faults={'h7_lost_from': 12_000}))
    assert out['verdict'] == 'fail' and out['frames']['pi_answered'] - out['frames']['frames_rx'] == 1
    assert any(m.startswith('H7 nhận') for m in out['fails'])       # the count itself, not a side effect


def test_constant_other_lag_points_at_h7_build(clean):
    out = run(clean, answer_lag=1)                           # as if the old H7 (answer at N+1) were expected
    assert out['verdict'] == 'fail' and out['answer_lag_hist'] == {2: N - 2}
    assert any('kiểm bản H7' in m for m in out['fails'])
    assert not any('kiểm bản H7' in m for m in run(build(faults={'stale': {5000}}))['fails'])


def test_short_run_is_short(clean):
    assert run(clean, min_s=60.0)['verdict'] == 'short'


@pytest.mark.parametrize('e1, s1, ok', [(0, 0, True), (2, 0, False), (0, 3, False)])
def test_serial_link_counters(clean, e1, s1, ok):
    out = run(clean, serial_events=serial(e1=e1, s1=s1))
    assert (out['verdict'] == 'pass') == ok
    sr = out['serial']
    assert sr['applied'] == 20501 and sr['link_periods'] == [dict(frames=20001, h7_ms=20128, pi_ms=20000)]


def test_s1_no_answer_is_not_a_pass():
    # MISO all zero (MISO line held low, H7 never built an answer): no evidence, fails
    t, mosi, miso, reps = build(n=12_000)
    out = run((t, mosi, np.zeros_like(miso), []), full=False)
    assert out['verdict'] == 'fail'
    assert any('toàn 0' in m for m in out['fails']) and any('REPORT' in m for m in out['fails'])
    # one empty answer in the middle of the run
    out = run(build(faults={'empty': {9000}}))
    assert out['verdict'] == 'fail' and out['frames']['miso_empty'] == 1
    # no TICK: the H7 log stream is not shown to run
    assert any('TICK' in m for m in run(build(n=12_000, faults={'no_tick': True}))['fails'])


def test_s6_whole_run_scored_and_length(clean):
    # a fault at second 5 is scored (T1 does not drop the first 10 s), and N frames = N ms
    out = run(build(faults={'mixed': {5000}}), skip_s=0.0)
    assert out['verdict'] == 'fail' and out['frames']['mixed'] == 1
    out = run(clean, skip_s=0.0, min_s=N / 1000.0 - 0.01)       # send jitter ±12 µs at both ends
    assert out['scored_s'] == pytest.approx(N / 1000.0, abs=0.001) and out['verdict'] != 'short'


def test_s12_age_in_true_microseconds():
    # 1004 nominal µs at 480 MHz is 997.6 µs at 483.1 MHz: passes; 1010 nominal fails
    v = t1.age_verdict(1004, 483.1)
    assert v['verdict'] == 'pass' and v['age_true_hi_us'] < 1000
    assert t1.age_verdict(1010, 483.1)['verdict'] == 'fail'
    assert t1.age_verdict(1004, None)['verdict'] == 'inconclusive'      # rate unknown: brackets 1000
    out = run(build(faults={'age': {7000: 1004}}))
    assert out['verdict'] == 'pass' and out['state_age']['verdict'] == 'pass', out['fails']


def test_s25_serial_log_required_and_bounded(clean):
    out = run(clean, serial_events=None)
    assert out['verdict'] == 'incomplete' and t1.exit_code(out) == 3
    cut = '\n'.join(SERIAL.splitlines()[:-2])                 # no line after the final loss
    out = run(clean, serial_events=serial(text=cut))
    assert out['verdict'] == 'incomplete' and any('bao hai đầu' in m for m in out['incomplete'])
    reconnect = SERIAL.replace('S06log: pushed=30', 'SPI link lost (no frame for 50 ms): x\n'
                                                    'SPI link up at t=9000 ms\nS06log: pushed=30')
    out = run(clean, serial_events=serial(text=reconnect))
    assert out['verdict'] == 'fail' and any('lên lại' in m for m in out['fails'])


def test_report_missing_at_esp32_fails(clean):
    out = run(clean, esp32_events=esp32_events(clean, drop={(1, 100)}))
    assert out['verdict'] == 'fail' and out['imu']['not_at_esp32_spi'] == 1


def test_cli(clean, tmp_path):
    t, mosi, miso = clean[:3]
    npz = tmp_path / 'frames.npz'
    np.savez(npz, t_send_ns=t, mosi=mosi, miso=miso)
    out = tmp_path / 'out'
    script = os.path.join(ROOT, 'tools', 's06_t1_check.py')
    log = tmp_path / 'h7.log'
    log.write_text(SERIAL.format(e0=0, e1=0, s0=0, s1=0), encoding='utf-8')
    ev = esp32_events(clean, str(tmp_path / 'esp32_events.txt'))
    p = subprocess.run([sys.executable, script, str(npz), '--out', str(out), '--skip-s', '1', '--window-s', '5',
                        '--min-s', '10', '--h7-serial', str(log), '--esp32-events', ev], capture_output=True,
                       text=True)
    assert p.returncode == 0, p.stdout[-2000:]
    summ = json.load(open(out / 'T1_summary.json', encoding='utf-8'))
    assert summ['verdict'] == 'pass'
    for name, first in (('T1.csv', 'idx'), ('T1_steps.csv', 'step'), ('T1_imu_builds.csv', 'idx')):
        assert open(out / name).readline().split(',')[0].strip() == first
    assert subprocess.run([sys.executable, script, str(npz), '--out', str(out)], capture_output=True,
                          text=True).returncode == 3           # 19 s scored < 600 s
