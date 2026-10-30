"""tools/s06_t4_values.py (S06 T4 (4)) on the synthetic frames of test_s06_frames_decode.build driven by
the locked T4 schedule (rpi5/t4_imu_profile.py, one 27 s cycle): callback values equal to the
source-frame IMU (within the report quantisation) pass; the faults the spec names (deg/rad, sign, 1 %
scale, value of the previous report, no source frame) and the ones of Codex Sol round 1/2 (NaN on a later
axis S3, sparse callbacks S9, applied = 0 S33, schedule not covered S18, DATA yaw S28) are found."""
import json
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

N = 30_500                                   # one full 27 s cycle of the T4 schedule + margins
KINDS = ((0, 1, 'accel'), (1, 2, 'gyro'), (2, 8, 'game_rv'))
T_START = 1_500_000                          # > 1 s after the reset and ≥ PAD_US after the first frame
T_END = N * 1000 - 1_200_000                 # ≥ PAD_US before the last frame / last J, I line (S38)


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


def events(tmp_path, mosi, seqs, fault=None, skip=None, name='esp32_events.txt', marks=(T_START, T_END),
           drop_tick=None, start_delay_us=0):
    """J/I lines (Pi clock = ESP32 clock), #run_start/#run_end; returns (events path, data path). The DATA
    file has one tick per game-RV callback with log_seq = its rv_n and yaw in rad (%.4f, as logger.c);
    first_data_seq/last_data_seq of the marks span the ticks inside the run. `drop_tick(rv_n)` True leaves
    that tick out of the file (lost DATA line)."""
    lines = [f'{marks[0] + start_delay_us}\t#run_start,first_data_seq=@FIRST@,timestamp_us={marks[0]},qseq=0']
    data = ['1\ttimestamp_us,x,yaw_rad,log_seq,rv_n']
    in_run = []
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
                tick_rv = rv_n + 5000 if (fault and fault[2] == 'data_no_cb' and n == fault[1]) else rv_n
                if marks[0] <= t + 900 <= marks[1]:
                    in_run.append(rv_n)
                if not (drop_tick and drop_tick(rv_n)):
                    data.append(f'{t + 900}\t{t + 900},0,{math.radians(y):.4f},{rv_n},{tick_rv}')
            lines.append(f'{t + 50}\tI,{q},{t + 50},{code},{applied},{nn % 256},2,'
                         f'{rv_n if (kind == "game_rv" and applied) else n},{n},'
                         + ','.join(f'{x:.9g}' for x in v))
    lines.append(f'{marks[1]}\t#run_end,last_data_seq={max(in_run)},timestamp_us={marks[1]},qseq={q + 1}')
    lines[0] = lines[0].replace('@FIRST@', str(min(in_run)))
    lines.sort(key=lambda x: int(x.split('\t')[0]))
    p = tmp_path / name
    p.write_text('\n'.join(lines) + '\n')
    d = tmp_path / ('data_' + name)
    d.write_text('\n'.join(data) + '\n')
    return str(p), str(d)


def manifest(n=N, profile='t4'):
    """What hil_node._save_frame_log writes for a run with --imu-profile t4."""
    return dict(step_first=0, step_last=n - 1, steps=n, period_us=1000, max_steps=n, stop_reason='max_steps',
                imu_profile=profile, imu_static=None)


def check(frames, tmp_path, **kw):
    t, mosi, miso, seqs = frames
    ev, data = events(tmp_path, mosi, seqs, **kw)
    return t4.check(t, mosi, miso, ev, data, manifest=manifest())


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


def test_fault_at_run_start_is_scored(frames, tmp_path):
    # S38: a wrong gyro callback 0,1 s after #run_start used to fall in the 0,5 s left out at each end
    n = (T_START // 1000 + 100 - 30) // 10
    rows, s = check(frames, tmp_path, fault=('gyro', n, 'sign'))
    assert s['per_kind']['gyro']['fail'] == 1 and t4.exit_code(s) == 1


def test_fault_just_before_run_end_is_scored(frames, tmp_path):
    n = (T_END // 1000 - 100 - 30) // 10
    rows, s = check(frames, tmp_path, fault=('accel', n, 'scale'))
    assert s['per_kind']['accel']['fail'] == 1 and t4.exit_code(s) == 1


def test_recording_not_padded_is_incomplete(frames, tmp_path):
    # S38: the run ends 0,3 s before the last frame: the reports at the end may lack their counterpart
    rows, s = check(frames, tmp_path, marks=(T_START, N * 1000 - 300_000))
    assert any('S38' in c for c in s['run_conditions']) and t4.exit_code(s) == 3


def test_data_tick_lost_fails(frames, tmp_path):
    # S39: one DATA line of the run missing from the file (log_seq hole) = lost data, not "one tick is enough"
    rows, s = check(frames, tmp_path, drop_tick=lambda rv: rv == 1200)
    assert s['data_yaw']['seq_missing'] == 1 and s['data_yaw']['bad'] == 0 and t4.exit_code(s) == 1


def test_data_tick_without_callback_fails(frames, tmp_path):
    # S39: a tick that points at an rv_n no callback line has (Codex counter-example) fails
    rows, s = check(frames, tmp_path, fault=('game_rv', 1500, 'data_no_cb'))
    assert s['data_yaw']['no_callback'] == 1 and t4.exit_code(s) == 1


def test_data_bad_line_fails(frames, tmp_path):
    t, mosi, miso, seqs = frames
    ev, data = events(tmp_path, mosi, seqs, name='badline.txt')
    with open(data, 'a') as fh:
        fh.write('29000000\t29000000,0,abc,1,1\n')
    rows, s = t4.check(t, mosi, miso, ev, data, manifest=manifest())
    assert s['data_yaw']['bad_lines'] == 1 and t4.exit_code(s) == 1


def test_stimulus_not_the_schedule_is_incomplete(tmp_path):
    # S42: the Pi sent one static IMU value all run (profile not set) and every report carried it
    # faithfully: values all right, but the schedule was not sent, so coverage must not be credited
    fr = build(n=N, imu_fn=lambda k: (0.0, 0.0, 0.0, prof.AZ, 0.0))
    t, mosi, miso, seqs = fr
    ev, data = events(tmp_path, mosi, seqs)
    rows, s = t4.check(t, mosi, miso, ev, data, manifest=manifest(profile=None))
    assert s['fail'] == 0 and s['stimulus']['frames_differ'] > 0 and t4.exit_code(s) == 3
    # S64: the manifest says the Pi DID run the t4 profile, yet the MOSI is off it: a fault of the HIL
    rows, s = t4.check(t, mosi, miso, ev, data, manifest=manifest())
    assert t4.exit_code(s) == 1
    # no manifest (log written before it): evidence missing, not a pass
    rows, s = t4.check(*frames_args(tmp_path), manifest=None)
    assert s['fail'] == 0 and t4.exit_code(s) == 3


def frames_args(tmp_path):
    fr = build(n=N, imu_fn=lambda k: prof.sample(k / 1000.0))
    t, mosi, miso, seqs = fr
    ev, data = events(tmp_path, mosi, seqs, name='nomanifest.txt')
    return t, mosi, miso, ev, data


def test_one_stimulus_frame_off_fails(frames, tmp_path):
    # S64 + S65: only ONE frame off the schedule, not the source of any report: the whole-run check sees it
    t, mosi, miso, seqs = frames
    m = mosi.copy()
    m[20_001, 22] ^= 0x10                            # gz of a frame no report used (reports use 27, 37, …)
    ev, data = events(tmp_path, mosi, seqs)
    rows, s = t4.check(t, m, miso, ev, data, manifest=manifest())
    assert s['stimulus']['frames_differ'] == 1 and s['stimulus']['first_differ'] == 20_001
    assert t4.exit_code(s) == 1


def test_frame_log_cut_fails(frames, tmp_path):
    # S57: the manifest says more steps than the arrays hold
    t, mosi, miso, seqs = frames
    ev, data = events(tmp_path, mosi, seqs)
    rows, s = t4.check(t, mosi, miso, ev, data, manifest=manifest(n=N + 500))
    assert s['frame_log_faults'] and t4.exit_code(s) == 1


def test_report_just_before_t_rx_of_start_is_scored_by_h7(frames, tmp_path):
    # S55: #run_start reaches the Pi 80 ms late: the H7 window must still open before the true START
    t, mosi, miso, seqs = frames
    ev, data = events(tmp_path, mosi, seqs, start_delay_us=80_000)
    rows, s = t4.check(t, mosi, miso, ev, data, manifest=manifest())
    assert s['window_pi_us'][0] <= T_START - 20_000 and t4.exit_code(s) == 0, s['run_conditions']


def test_start_packet_later_than_the_bound_is_incomplete(frames, tmp_path):
    # S55: the #run_start packet 300 ms late, above the 100 ms bound: the window cannot be trusted
    t, mosi, miso, seqs = frames
    ev, data = events(tmp_path, mosi, seqs, start_delay_us=300_000)
    rows, s = t4.check(t, mosi, miso, ev, data, manifest=manifest())
    assert any('S55' in c for c in s['run_conditions']) and t4.exit_code(s) == 3


def test_cli_exit_codes(frames, tmp_path):
    import numpy as np
    t, mosi, miso, seqs = frames
    np.savez(tmp_path / 'frames.npz', t_send_ns=t, mosi=mosi, miso=miso, manifest=np.array(json.dumps(manifest())))
    ok, data = events(tmp_path, mosi, seqs)
    assert t4.main([str(tmp_path / 'frames.npz'), ok, '--esp32-data', data, '--out', str(tmp_path / 'o')]) == 0
    assert t4.main([str(tmp_path / 'frames.npz'), ok, '--out', str(tmp_path / 'o1')]) == 3     # no DATA
    bad, data = events(tmp_path, mosi, seqs, ('gyro', 2397, 'sign'), name='bad.txt')
    assert t4.main([str(tmp_path / 'frames.npz'), bad, '--esp32-data', data, '--out', str(tmp_path / 'o2')]) == 1
