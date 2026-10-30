"""tools/s06_t4_values.py (S06 T4 (4)) on the synthetic frames of test_s06_frames_decode.build driven by
the locked T4 schedule (rpi5/t4_imu_profile.py, one 27 s cycle): callback values equal to the
source-frame IMU (within the report quantisation) pass; the faults the spec names (deg/rad, sign, 1 %
scale, value of the previous report, no source frame) and the ones of Codex Sol round 1/2 (NaN on a later
axis S3, sparse callbacks S9, applied = 0 S33, schedule not covered S18, DATA yaw S28) are found."""
import math
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'rpi5')]
import s06_t4_values as t4  # noqa: E402
import t4_imu_profile as prof  # noqa: E402
from protocol import decode_imu  # noqa: E402
from test_s06_frames_decode import build  # noqa: E402

N = 28_500                                   # one full 27 s cycle of the T4 schedule + margins
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
    return build(n=N, imu_fn=lambda k: prof.sample(k / 1000.0))


def events(tmp_path, mosi, seqs, fault=None, skip=None, name='esp32_events.txt'):
    """J/I lines (Pi clock = ESP32 clock), #run_start/#run_end; returns (events path, data path). The DATA
    file has one tick per game-RV callback with its rv_n and yaw in rad (%.4f, as logger.c)."""
    lines = ['1500000\t#run_start,first_data_seq=1,timestamp_us=1500000,qseq=0']   # > 1 s after the reset
    data = ['1\ttimestamp_us,x,yaw_rad,rv_n']
    q, rv_n = 0, 0
    for s, code, kind in KINDS:
        for n in range(seqs[s]):
            q += 1
            src = report_frame(n)
            v = callback_value(kind, decode_imu(bytes(mosi[src])))
            applied, nn = 1, n
            if fault and fault[0] == kind and fault[1] == n:
                how = fault[2]
                if how == 'deg':
                    v[2] = math.radians(v[2]) if kind == 'game_rv' else math.degrees(v[2])
                elif how == 'sign':
                    v[2] = -v[2]
                elif how == 'scale':
                    v[2] *= 1.01
                elif how == 'stale':
                    v = callback_value(kind, decode_imu(bytes(mosi[report_frame(n - 1)])))
                elif how == 'axis':
                    v[0], v[2] = v[2], v[0]
                elif how == 'yonly':
                    v[1] += 0.05                      # only the y axis wrong
                elif how == 'seq':
                    nn = n + 1000
                elif how == 'nan_z':
                    v[2] = float('nan')               # S3: NaN on the last axis
                elif how == 'applied0_wrong':
                    applied, v[2] = 0, v[2] + 1.0      # S33: value read, mutex missed, value wrong
            t = (30 + 10 * n) * 1000 + 300 + s
            lines.append(f'{t}\tJ,{q},{t},{code},1,{nn % 256},2')
            if skip and skip(kind, n):                 # report at the SPI port, its callback never ran
                continue
            if kind == 'game_rv' and applied:
                rv_n += 1
                y = (v[2] + 180.0) % 360.0 - 180.0
                if fault and fault[2] == 'data_yaw' and n == fault[1]:
                    y += 1.0
                data.append(f'{t + 900}\t{t + 900},0,{math.radians(y):.4f},{rv_n}')
            lines.append(f'{t + 50}\tI,{q},{t + 50},{code},{applied},{nn % 256},2,'
                         f'{rv_n if (kind == "game_rv" and applied) else n},{n},'
                         + ','.join(f'{x:.9g}' for x in v))
    lines.append(f'{N * 1000 - 300_000}\t#run_end,last_data_seq=9,timestamp_us=0,qseq={q + 1}')
    p = tmp_path / name
    p.write_text('\n'.join(lines) + '\n')
    d = tmp_path / ('data_' + name)
    d.write_text('\n'.join(data) + '\n')
    return str(p), str(d)


def check(frames, tmp_path, **kw):
    t, mosi, miso, seqs = frames
    ev, data = events(tmp_path, mosi, seqs, **kw)
    return t4.check(t, mosi, miso, ev, data)


def test_clean_all_pass(frames, tmp_path):
    rows, s = check(frames, tmp_path)
    assert s['fail'] == 0 and s['epoch_offset_esp32_minus_h7'] == 1 and s['callback_reports'] > 0
    assert all(r['ok'] for r in rows)
    assert set(s['per_kind']) == {'accel', 'gyro', 'game_rv'} and s['coverage_missing'] == {}
    assert s['data_yaw']['bad'] == 0 and s['data_yaw']['ok'] > 0
    assert t4.exit_code(s) == 0, s['decode_problems']


@pytest.mark.parametrize('kind,how', [('game_rv', 'deg'), ('gyro', 'sign'), ('accel', 'scale'),
                                      ('gyro', 'stale'), ('accel', 'axis'), ('gyro', 'seq'),
                                      ('accel', 'yonly'), ('gyro', 'nan_z'), ('game_rv', 'nan_z'),
                                      ('gyro', 'applied0_wrong')])
def test_faults_fail(frames, tmp_path, kind, how):
    n = 2397                          # source frame 24 000: ramp, t = 2 s in (wz −1, ax −0,4, yaw 0,28)
    rows, s = check(frames, tmp_path, fault=(kind, n, how))
    assert s['fail'] >= 1 and s['per_kind'][kind]['fail'] >= 1, (how, s['per_kind'])
    assert t4.exit_code(s) == 1


def test_sparse_callbacks_keep_identity(frames, tmp_path):
    # S9: the first 300 gyro callbacks never came (DUT overwrote them); report 300 must still be
    # compared with source 300, not with 300 mod 256
    rows, s = check(frames, tmp_path, skip=lambda kind, n: kind == 'gyro' and n < 300)
    assert s['per_kind']['gyro']['fail'] == 0 and s['per_kind']['gyro']['n'] > 0


def test_schedule_not_covered_is_incomplete(frames, tmp_path):
    # S18: every gyro callback of the gz = +5 level lost: values all right, but not the whole schedule
    lvl = prof.LEVELS.index(('gz', 5.0))
    lo, hi = lvl * prof.LEVEL_S * 1000, (lvl + 1) * prof.LEVEL_S * 1000
    rows, s = check(frames, tmp_path, skip=lambda kind, n: kind == 'gyro' and lo <= report_frame(n) < hi)
    assert s['fail'] == 0 and 'gz=5' in s['coverage_missing']['gyro'] and t4.exit_code(s) == 3


def test_data_yaw_mismatch_fails(frames, tmp_path):
    # S28: the controller's DATA yaw for one rv_n differs from the callback's
    rows, s = check(frames, tmp_path, fault=('game_rv', 1500, 'data_yaw'))
    assert s['data_yaw']['bad'] == 1 and t4.exit_code(s) == 1


def test_cli_exit_codes(frames, tmp_path):
    import numpy as np
    t, mosi, miso, seqs = frames
    np.savez(tmp_path / 'frames.npz', t_send_ns=t, mosi=mosi, miso=miso)
    ok, data = events(tmp_path, mosi, seqs)
    assert t4.main([str(tmp_path / 'frames.npz'), ok, '--esp32-data', data, '--out', str(tmp_path / 'o')]) == 0
    assert t4.main([str(tmp_path / 'frames.npz'), ok, '--out', str(tmp_path / 'o1')]) == 3     # no DATA
    bad, data = events(tmp_path, mosi, seqs, ('gyro', 2397, 'sign'), name='bad.txt')
    assert t4.main([str(tmp_path / 'frames.npz'), bad, '--esp32-data', data, '--out', str(tmp_path / 'o2')]) == 1
