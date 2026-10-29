"""tools/s06_t4_values.py (S06 T4 (4)) on the synthetic frames of test_s06_frames_decode.build:
callback values equal to the source-frame IMU (within the report quantisation) pass; the faults the
spec names (deg/rad, sign, 1 % scale, value of the previous report, no source frame) fail."""
import math
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'rpi5')]
import s06_t4_values as t4  # noqa: E402
from protocol import decode_imu  # noqa: E402
from test_s06_frames_decode import build  # noqa: E402

N = 3000
KINDS = ((0, 1, 'accel'), (1, 2, 'gyro'), (2, 8, 'game_rv'))


def report_frame(n):
    """build(): the n-th report of each kind is built at frame 30 + 10 n from frame k - 3."""
    return 30 + 10 * n - 3


def callback_value(kind, imu):
    gz, ax, ay, az, yaw = imu[:5]
    if kind == 'accel':
        return [ax + 1e-3, ay - 1e-3, az + 2e-3]          # within Q8 + int16 (2.55e-3)
    if kind == 'gyro':
        return [0.0, 0.0, gz + 1e-3]
    return [0.0, 0.0, math.degrees(yaw + 1e-4)]          # game RV in degrees


@pytest.fixture(scope='module')
def frames():
    return build(n=N)


def events(tmp_path, mosi, seqs, fault=None):
    lines, q = [], 0
    for s, code, kind in KINDS:
        for n in range(seqs[s]):
            q += 1
            src = report_frame(n)
            v = callback_value(kind, decode_imu(bytes(mosi[src])))
            if fault and fault[0] == kind and fault[1] == n:
                if fault[2] == 'deg':
                    v[2] = math.radians(v[2]) if kind == 'game_rv' else math.degrees(v[2])
                elif fault[2] == 'sign':
                    v[2] = -v[2]
                elif fault[2] == 'scale':
                    v[2] *= 1.01
                elif fault[2] == 'stale':
                    v = callback_value(kind, decode_imu(bytes(mosi[report_frame(n - 1)])))
                elif fault[2] == 'axis':
                    v[0], v[2] = v[2], v[0]
                elif fault[2] == 'yonly':
                    v[1] += 0.05                      # only the y axis wrong
                elif fault[2] == 'seq':
                    n = n + 1000
            t = 1000 * n + s
            lines.append(f'{t}\tJ,{q},{t},{code},1,{n % 256},2')
            lines.append(f'{t}\tI,{q},{t},{code},1,{n % 256},2,{n},{n},' + ','.join(f'{x:.9g}' for x in v))
            if n == 7:                                 # not applied (newer report read): ignored
                lines.append(f'{t}\tI,{q},{t},{code},0,{(n + 50) % 256},2,{n},{n},99,99,99')
    p = tmp_path / 'esp32_events.txt'
    p.write_text('\n'.join(lines) + '\n')
    return str(p)


def test_clean_all_pass(frames, tmp_path):
    t, mosi, miso, seqs = frames
    rows, s = t4.check(t, mosi, miso, events(tmp_path, mosi, seqs))
    assert s['fail'] == 0 and s['callback_reports'] == sum(seqs) and s['epoch_offset_esp32_minus_h7'] == 1
    assert all(r['ok'] for r in rows)
    assert set(s['per_kind']) == {'accel', 'gyro', 'game_rv'}


@pytest.mark.parametrize('kind,how', [('game_rv', 'deg'), ('gyro', 'sign'), ('accel', 'scale'),
                                      ('gyro', 'stale'), ('accel', 'axis'), ('gyro', 'seq'),
                                      ('accel', 'yonly')])
def test_faults_fail(frames, tmp_path, kind, how):
    t, mosi, miso, seqs = frames
    n = 120                                             # gz = 0.1 sin(k/50) is changing there
    rows, s = t4.check(t, mosi, miso, events(tmp_path, mosi, seqs, (kind, n, how)))
    assert s['fail'] >= 1 and s['per_kind'][kind]['fail'] >= 1, (how, s)


def test_cli_exit_codes(frames, tmp_path):
    import numpy as np
    t, mosi, miso, seqs = frames
    np.savez(tmp_path / 'frames.npz', t_send_ns=t, mosi=mosi, miso=miso)
    ok = events(tmp_path, mosi, seqs)
    assert t4.main([str(tmp_path / 'frames.npz'), ok, '--out', str(tmp_path / 'o')]) == 0
    bad = events(tmp_path, mosi, seqs, ('gyro', 50, 'sign'))
    assert t4.main([str(tmp_path / 'frames.npz'), bad, '--out', str(tmp_path / 'o2')]) == 1
