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


def build(n=N, faults=()):
    rng = np.random.default_rng(3)
    t_send_ns = (np.arange(n) * 1e6 + rng.uniform(-20e3, 20e3, n)).astype(np.int64)
    mosi = np.zeros((n, 40), dtype=np.uint8)
    for k in range(n):
        imu = (0.1 * np.sin(k / 50), 0.0, 0.0, 9.81, 0.01 * k % 3.0)
        mosi[k] = np.frombuffer(encode_state([0.0] * 10, seq=k, imu=imu, imu_seq=k // 10), dtype=np.uint8)
    cyc0 = 4_000_000_000                                       # wraps after ~0.6 s
    cyc = [(cyc0 + int((t_send_ns[k] / 1e3 + D_1W_US) * 1e-6 * F_H7)) for k in range(n)]
    # log stream: RESET_DONE, then a REPORT per kind every 10 ms (src = frame 3 before), TICK every 500
    produced, idx = [[] for _ in range(n)], 0
    seqs = [0, 0, 0]
    for k in range(n):
        queue = produced[k]
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
            queue.append((0x40, struct.pack('<IIIH', cyc[k] & 0xFFFFFFFF, cyc[k] & 0xFFFFFFFF, cyc[k] >> 32, 0)))
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


def esp32_file(tmp_path, seqs, epoch_off=1, skip=()):
    lines, q = [], 0
    for s, kind in enumerate((1, 2, 8)):
        for n in range(seqs[s]):
            q += 1
            if (s, n) in skip:
                continue
            t = 1000 * n + s
            lines.append(f'{t}\tJ,{q},{t},{kind},1,{n % 256},{1 + epoch_off}')
            lines.append(f'{t}\tI,{q},{t},{kind},1,{n % 256},{1 + epoch_off},{n},{n},0,0,0')
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
    assert m['epoch_offset_esp32_minus_h7'] == 1 and m['matched_spi'] == sum(seqs) and m['esp32_without_h7'] == 0
    # report build time on the Pi clock: frame k sent at ~k ms, cyc stamped D_1W_US after
    r = next(x for x in res['_reports'] if x['kind'] == 'gyro' and x['n'] == 100)
    assert dec.problems(res) == []
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


def test_mosi_crc_mismatch():
    t, mosi, miso, _ = build(n=3000)
    mosi = mosi.copy()
    mosi[1500, 0] ^= 1            # the Pi log says one thing, the H7 checksummed another
    res = dec.decode(t, mosi, miso)
    assert res['frames']['mosi_crc_bad'] == 1


def test_unwrap_and_near():
    assert dec.unwrap([65530, 65535, 3, 10], 16) == [65530, 65535, 65539, 65546]
    assert dec.near(5, (1 << 32) - 10) == (1 << 32) + 5
    assert dec.near((1 << 32) - 3, (1 << 32) + 2) == (1 << 32) - 3


H7_REV = '3235ec5'           # feat/s06-h7: MISO v4 + S06 log


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
            s['S06_REC_REARM']) == (dec.REC_REPORT, dec.REC_PKT_READ, dec.REC_RESET_DONE, dec.REC_HOLD_ON,
                                    dec.REC_HOLD_OFF, dec.REC_RST_LOW, dec.REC_RST_HIGH, dec.REC_TICK, dec.REC_REARM)
