"""tools/s06_frames_decode.py on synthetic frames built here from the MISO v4 layout of the
design note (H7 feat/s06-h7 3235ec5): answer at N+2, CRC-16/CCITT-FALSE, little-endian fields,
one log record per frame, CYCCNT at 483.1 MHz wrapping every ~8.9 s. Faults are put in on
purpose and must be reported. Not checked against frames of the real H7 (none yet).
Run: python -m pytest tools/tests"""
import binascii
import os
import struct
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'rpi5')]
import s06_frames_decode as dec  # noqa: E402
from protocol import encode_state  # noqa: E402

F_H7 = 483.1e6
D_1W_US = 45.0               # Pi send -> H7 callback
N = 20_000                   # 20 s: CYCCNT wraps twice


def crc(b):
    return binascii.crc_hqx(bytes(b), 0xFFFF)


def build(n=N, faults=(), imu_fn=None):
    """imu_fn(k) -> (gz, ax, ay, az, yaw) of frame k (default: a slow sine on gz and a yaw ramp)."""
    rng = np.random.default_rng(3)
    t_send_ns = (np.arange(n) * 1e6 + rng.uniform(-20e3, 20e3, n)).astype(np.int64)
    mosi = np.zeros((n, 40), dtype=np.uint8)
    for k in range(n):
        imu = imu_fn(k) if imu_fn else (0.1 * np.sin(k / 50), 0.0, 0.0, 9.81, 0.01 * k % 3.0)
        mosi[k] = np.frombuffer(encode_state([0.0] * 10, seq=k, imu=imu, imu_seq=k // 10), dtype=np.uint8)
    cyc0 = 4_000_000_000                                       # wraps after ~0.6 s
    cyc = [(cyc0 + int((t_send_ns[k] / 1e3 + D_1W_US) * 1e-6 * F_H7)) for k in range(n)]
    # log stream: RESET_DONE, then a REPORT per kind every 10 ms (src = frame 3 before), TICK every 500
    produced, idx = [[] for _ in range(n)], 0
    seqs = [0, 0, 0]
    for k in range(n):
        queue = produced[k]
        for kk, w, ev, cnt_n in ENC_EVENTS:
            if k == kk and ('drop_enc', kk) not in faults:
                queue.append((0x35 if ev == 'S' else 0x36, bytes([w]) + struct.pack(
                    '<HIIH', (k * 7) & 0xFFFF, cnt_n, cyc[k] & 0xFFFFFFFF, (k + 1) & 0xFFFF)))
        if k == 5:
            queue.append((0x30, bytes([1]) + struct.pack('<IH', cyc[k] & 0xFFFFFFFF, (k + 1) & 0xFFFF)))
        if k > 20 and k % 10 == 0:
            src = k - 3
            raw = bytes(mosi[src][i] for i in dec.MOSI_IMU_BYTES)
            for s in range(3):
                c = crc(raw)
                if ('imu_crc', k, s) in faults:
                    c ^= 1
                p = bytes([seqs[s] & 0xFF, 1, 1, mosi[src][39]]) + struct.pack(
                    '<HHHI', (src + 1) & 0xFFFF, (k + 1) & 0xFFFF, c, (cyc[k] + 1000 * s) & 0xFFFFFFFF)
                queue.append((0x10 + s, p))
                seqs[s] += 1
        if k % 500 == 499:
            drop = 0
            for f in faults:                    # ('tick_drop', counter before the run, from frame, + after)
                if f[0] == 'tick_drop':
                    drop = (f[1] + (f[3] if k >= f[2] else 0)) & 0xFFFF
            queue.append((0x40, struct.pack('<IIIH', cyc[k] & 0xFFFFFFFF, cyc[k] & 0xFFFFFFFF, cyc[k] >> 32, drop)))
    miso = np.zeros((n, 40), dtype=np.uint8)
    pending, qi = [], 0                                        # H7 ring: records leave one per frame
    for j in range(2, n):
        pending += produced[j - 1]
        i = j - 2                                              # answer of frame i (N+2)
        b = bytearray(40)
        b[10:12] = struct.pack('<H', crc(mosi[i]))
        b[12:16] = struct.pack('<I', cyc[i] & 0xFFFFFFFF)
        b[17] = mosi[i][21]
        b[18:20] = struct.pack('<H', (i + 1) & 0xFFFF)
        if qi < len(pending):
            typ, pl = pending[qi]
            if ('drop_log', qi) not in faults:
                b[22] = typ
                b[23:25] = struct.pack('<H', idx & 0xFFFF)
                b[25:25 + len(pl)] = pl
            idx += 1
            qi += 1
        cr = crc(b[:8] + b[10:])
        b[8:10] = struct.pack('<H', cr)
        if ('miso_bit', j) in faults:
            b[30] ^= 0x10
        miso[j] = np.frombuffer(bytes(b), dtype=np.uint8)
    return t_send_ns, mosi, miso, seqs


# (frame, wheel, start/stop, running count of that kind on that wheel)
ENC_EVENTS = [(1000, 0, 'S', 1), (3000, 0, 'P', 1), (3000, 0, 'S', 2), (9000, 2, 'S', 1), (15000, 0, 'P', 2),
              (18000, 0, 'S', 3)]


def esp32_file(tmp_path, seqs, epoch_off=1, skip=(), marks=(1_500_000, N * 1000 - 1_500_000), extra=(), resets=()):
    """J/I lines of the reports of build(): report n of a kind is built at frame 30 + 10 n and reaches
    the ESP32 SPI port ~300 µs later; ESP32 clock = Pi clock here (t_us = t_rx)."""
    lines, q = [], 0
    if marks:
        lines.append(f'{marks[0]}\t#run_start,first_data_seq=1,timestamp_us={marks[0]},qseq=0')
    for s, kind in enumerate((1, 2, 8)):
        for n in range(seqs[s]):
            q += 1
            if (s, n) in skip:
                continue
            t = (30 + 10 * n) * 1000 + 300 + s
            lines.append(f'{t}\tJ,{q},{t},{kind},1,{n % 256},{1 + epoch_off}')
            lines.append(f'{t + 50}\tI,{q},{t + 50},{kind},1,{n % 256},{1 + epoch_off},{n},{n},0,0,0')
    lines += list(extra)
    for t in resets:
        lines.append(f'{t}\tR,{q},{t},1')
    if marks:
        lines.append(f'{marks[1]}\t#run_end,last_data_seq=9,timestamp_us={marks[1]},qseq={q + 1}')
    p = tmp_path / 'esp32_events.txt'
    p.write_text('\n'.join(lines) + '\n')
    return str(p)


@pytest.fixture(scope='module')
def clean():
    return build()


def test_clean_run(clean, tmp_path):
    t, mosi, miso, seqs = clean
    res = dec.decode(t, mosi, miso, esp32_file(tmp_path, seqs))
    f = res['frames']
    assert f['empty'] == 2 and f['miso_crc_bad'] == 0 and f['mosi_crc_bad'] == 0 and f['unanswered'] == 0
    assert f['answer_lag'] == {2: N - 2}
    assert res['log']['gaps'] == 0 and res['log']['h7_dropped'] == 0
    segs = res['clock']['segments']
    assert len(segs) == 1 and abs(segs[0]['f_h7_mhz'] - F_H7 / 1e6) < 0.01 and segs[0]['res_max_us'] < 25
    assert res['clock']['tick_total_mismatch'] == 0
    rp = res['reports']
    assert rp['n'] == sum(seqs) and rp['imu_crc_bad'] == 0 and rp['imu_seq_bad'] == 0 and rp['no_source_frame'] == 0
    m = res['esp32_match']
    assert m['epoch_offset_esp32_minus_h7'] == 1 and m['esp32_without_h7'] == 0
    assert 0 < m['h7_scored'] < sum(seqs) and m['matched_spi'] == m['h7_scored'] and m['not_at_esp32_spi'] == 0
    assert res['resets_near_run_start']['ok'] and res['clock']['calibrated'] is False
    # report build time on the Pi clock: frame k sent at ~k ms, cyc stamped D_1W_US after
    r = next(x for x in res['_reports'] if x['kind'] == 'gyro' and x['n'] == 100)
    assert dec.problems(res) == [] and dec.conditions(res) == []
    calls = res['_calls']
    assert [c[0] for c in calls[0]] == ['S', 'P', 'S', 'P', 'S'] and [c[0] for c in calls[2]] == ['S']
    assert calls[0][1][2] == (3000 * 7) & 0xFFFF and calls[0][2][3] == 2
    t9000 = calls[2][0][1]
    # CYCCNT -> Pi clock is fitted to the send times, so the constant send -> callback delay is absorbed
    assert abs(t9000 - t[9000] / 1e3) < 1
    assert r['t_pi_us'] is not None


def test_faults_reported(tmp_path):
    faults = {('miso_bit', 7000), ('drop_log', 1234), ('imu_crc', 5000, 1)}
    t, mosi, miso, seqs = build(faults=faults)
    res = dec.decode(t, mosi, miso, esp32_file(tmp_path, seqs, skip={(0, 300)}))
    assert res['frames']['miso_crc_bad'] == 1
    assert res['log']['gaps'] >= 1
    assert res['reports']['imu_crc_bad'] == 1
    assert res['esp32_match']['not_at_esp32_spi'] == 1
    assert len(dec.problems(res)) >= 4


def test_missing_enc_record_reported():
    t, mosi, miso, _ = build(faults={('drop_enc', 3000)})       # stop + start of wheel 0 lost
    res = dec.decode(t, mosi, miso)
    assert res['enc_calls']['count_gaps'] == 2           # a loss after the last call would not show


def test_mosi_crc_mismatch():
    t, mosi, miso, _ = build(n=3000)
    mosi = mosi.copy()
    mosi[1500, 0] ^= 1            # the Pi log says one thing, the H7 checksummed another
    res = dec.decode(t, mosi, miso)
    assert res['frames']['mosi_crc_bad'] == 1


def test_epoch_anchored_not_searched(clean, tmp_path):
    # S7: raw epoch counters 8 apart are a valid run (both at their first epoch of the run)
    t, mosi, miso, seqs = clean
    m = dec.decode(t, mosi, miso, esp32_file(tmp_path, seqs, epoch_off=8))['esp32_match']
    assert m['epoch_offset_esp32_minus_h7'] == 8 and m['not_at_esp32_spi'] == 0 and m['esp32_without_h7'] == 0


def test_epoch_step_on_one_side_only(clean, tmp_path):
    # S7: the ESP32 sees a reset (epoch + 1) that the H7 log never had: no anchor, reported
    t, mosi, miso, seqs = clean
    p = esp32_file(tmp_path, seqs)
    lines = open(p).read().splitlines()
    out = []
    for ln in lines:
        f = ln.split('\t')[1].split(',')
        if f[0] in ('J', 'I') and int(f[2]) > 10_000_000:
            f[6] = str(int(f[6]) + 1)
        out.append(ln.split('\t')[0] + '\t' + ','.join(f))
    open(p, 'w').write('\n'.join(out) + '\n')
    res = dec.decode(t, mosi, miso, p)
    assert res['esp32_match']['epoch_offset_esp32_minus_h7'] is None
    assert any('không neo được' in x for x in dec.problems(res))


def test_epoch_offset_rules():
    # S7: same number of epochs but not stepping together (H7 1 -> 2, ESP32 2 -> 4): not anchored
    h7 = [dict(epoch=1), dict(epoch=2)]
    assert dec.epoch_offset(h7, [dict(epoch=2, t_us=0), dict(epoch=3, t_us=1)]) == (1, '')
    off, note = dec.epoch_offset(h7, [dict(epoch=2, t_us=0), dict(epoch=4, t_us=1)])
    assert off is None and 'không cùng bước' in note
    off, note = dec.epoch_offset(h7, [dict(epoch=2, t_us=0), dict(epoch=3, t_us=1), dict(epoch=2, t_us=2)])
    assert off is None                                   # an epoch coming back: not one-to-one


def test_seq_chain_rules():
    # S8: k ≥ 1, unwrap only below 128 periods, cut from 128 periods
    items = [(s % 256, 10 * i) for i, s in enumerate(range(300))]          # clean, period 10
    res, per = dec.seq_chain(items, lambda x: x[0], lambda x: x[1])
    assert per == 10 and [r[0] for r in res] == list(range(300)) and not any(r[2] for r in res)
    dup = items[:5] + [items[4]] + items[5:10]
    assert [r[2] for r in dec.seq_chain(dup, lambda x: x[0], lambda x: x[1])[0]].count('dup') == 1
    cut = items[:50] + [(c[0], c[1] + 2000) for c in items[50:]]          # 200 periods without report
    r = dec.seq_chain(cut, lambda x: x[0], lambda x: x[1])[0]
    assert r[50][2] == 'cut' and r[50][1] == 1 and len({x[0] for x in r}) == len(r)
    jump = items[:50] + [(c[0] + 150, c[1]) for c in items[50:]]          # 150 reports lost in 1 period
    assert dec.seq_chain(jump, lambda x: x[0], lambda x: x[1])[0][50][2] == 'seq_time_bad'


def test_duplicate_report_seq_reported():
    # S8: the H7 builds the same report number twice: not merged silently
    t, mosi, miso, _ = build(n=3000)
    miso = miso.copy()
    # find two consecutive gyro REPORT records and give the second the seq of the first
    rows, _ = dec.decode_frames(t, mosi, miso)
    js = [r['j'] for r in rows if not r['empty'] and r['crc_ok'] and r['log_type'] == 0x11]
    j0, j1 = js[10], js[11]
    b = bytearray(miso[j1])
    b[25] = miso[j0][25]
    b[8:10] = struct.pack('<H', crc(b[:8] + b[10:]))
    miso[j1] = np.frombuffer(bytes(b), dtype=np.uint8)
    res = dec.decode(t, mosi, miso)
    assert res['reports']['seq_flags'].get('dup') == 1 and any('dup' in x for x in dec.problems(res))


def test_long_link_gap_keeps_clock():
    # S11: 5 s without frames (> half the 32-bit CYCCNT range at 483 MHz): times after the gap stay right
    t, mosi, miso, _ = build()
    keep = np.r_[0:5000, 10000:N]
    res = dec.decode(t[keep], mosi[keep], miso[keep])
    segs = res['clock']['segments']
    assert len(segs) == 2 and all(abs(s['f_h7_mhz'] - F_H7 / 1e6) < 0.01 for s in segs)
    t15000 = next(c for c in res['_calls'][0] if c[0] == 'P' and c[3] == 2)[1]
    assert abs(t15000 - t[15000] / 1e3) < 1
    assert res['clock']['tick_total_mismatch'] == 0


def test_dropped_is_run_difference():
    # S36: the counter of TICK runs since S06Log_Init: 7 before the run and none in it is a clean run
    t, mosi, miso, _ = build(n=3000, faults={('tick_drop', 7, 0, 0)})
    assert dec.decode(t, mosi, miso)['log']['h7_dropped'] == 0
    t, mosi, miso, _ = build(n=3000, faults={('tick_drop', 65534, 2000, 3)})   # +3 in the run, wraps 16 bits
    res = dec.decode(t, mosi, miso)
    assert res['log']['h7_dropped'] == 3 and any('bỏ 3' in x for x in dec.problems(res))


def test_reset_near_run_start_reported(clean, tmp_path):
    t, mosi, miso, seqs = clean
    res = dec.decode(t, mosi, miso, esp32_file(tmp_path, seqs, resets=(1_800_000,)))
    assert res['resets_near_run_start'] == dict(h7=0, esp32=1, ok=False)
    assert any('V4-1' in x for x in dec.conditions(res)) and dec.problems(res) == []


def test_report_outside_window_not_scored(clean, tmp_path):
    # S32: a J received after the run window (H7 built it after the last Pi frame) is not a fault;
    # one inside the window without an H7 report is
    t, mosi, miso, seqs = clean
    tail = [f'19800000\tJ,9999,19800000,2,1,{seqs[1] % 256},2']
    m = dec.decode(t, mosi, miso, esp32_file(tmp_path, seqs, extra=tail))['esp32_match']
    assert m['esp32_without_h7'] == 0 and m['not_at_esp32_spi'] == 0
    # the same J before the start of the window is not scored either, a missing one inside is
    m = dec.decode(t, mosi, miso, esp32_file(tmp_path, seqs, skip={(1, 5), (1, 900)}))['esp32_match']
    assert m['not_at_esp32_spi'] == 1


def test_calibrated_clock_adds_d1w(clean):
    # S10: with the pulse times and d_1w the H7 event lands on its true Pi time; without, relative
    t, mosi, miso, _ = clean
    p0, p1 = t - 25_000, t - 15_000                       # pulse raised 15–25 µs before the send time
    cal = dict(t_pulse0_ns=p0, t_pulse1_ns=p1, d1w_us=D_1W_US + 20.0, d1w_span_us=2.0, delta_h_us=0.5, ts_us=0.04)
    res = dec.decode(t, mosi, miso, cal=cal)
    c = res['clock']
    assert c['calibrated'] and c['reference'] == 'pulse' and c['uncertainty_us'] > 10.0
    t9000 = res['_calls'][2][0][1]
    assert abs(t9000 - (t[9000] / 1e3 + D_1W_US)) < 25          # within the ±20 µs send jitter of build()
    assert dec.decode(t, mosi, miso, cal=dict(t_pulse0_ns=p0, t_pulse1_ns=p1))['clock']['calibrated'] is False


def test_unwrap_and_near():
    assert dec.unwrap([65530, 65535, 3, 10], 16) == [65530, 65535, 65539, 65546]
    assert dec.near(5, (1 << 32) - 10) == (1 << 32) + 5
    assert dec.near((1 << 32) - 3, (1 << 32) + 2) == (1 << 32) - 3


H7_REV = '598a451'           # feat/s06-h7: MISO v4 + S06 log + EncoderGen call records


def _h7_defines(path):
    import re
    import subprocess
    p = subprocess.run(['git', 'show', f'{H7_REV}:{path}'], cwd=ROOT, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if p.returncode != 0:
        pytest.skip(f'{H7_REV} not in this clone')
    return {m.group(1): int(m.group(2), 0) for m in re.finditer(r'#define\s+(\w+)\s+(0x[0-9A-Fa-f]+|\d+)u?\b', p.stdout)}


def test_constants_match_h7_headers():
    h = _h7_defines('stm32h7/Core/Inc/hil_protocol.h')
    assert (h['HIL_MISO_CRC'], h['HIL_MISO_MOSI_CRC'], h['HIL_MISO_RX_CYC'], h['HIL_MISO_STATUS'], h['HIL_MISO_SEQ'],
            h['HIL_MISO_RX_NF'], h['HIL_MISO_AGE'], h['HIL_MISO_LOG']) == \
        (dec.MISO_CRC, dec.MISO_MOSI_CRC, dec.MISO_RX_CYC, dec.MISO_STATUS, dec.MISO_SEQ, dec.MISO_RX_NF,
         dec.MISO_AGE, dec.MISO_LOG)
    assert h['HIL_MISO_LOG_LEN'] == dec.FRAME - dec.MISO_LOG
    s = _h7_defines('stm32h7/Core/Inc/s06_log.h')
    assert (s['S06_REC_REPORT'], s['S06_REC_PKT_READ'], s['S06_REC_RESET_DONE'], s['S06_REC_HOLD_ON'],
            s['S06_REC_HOLD_OFF'], s['S06_REC_RST_LOW'], s['S06_REC_RST_HIGH'], s['S06_REC_TICK'],
            s['S06_REC_REARM'], s['S06_REC_ENC_START'], s['S06_REC_ENC_STOP']) ==         (dec.REC_REPORT, dec.REC_PKT_READ, dec.REC_RESET_DONE, dec.REC_HOLD_ON, dec.REC_HOLD_OFF, dec.REC_RST_LOW,
         dec.REC_RST_HIGH, dec.REC_TICK, dec.REC_REARM, dec.REC_ENC_START, dec.REC_ENC_STOP)
