"""hil_o1_score + o1_lanes (EA-13, EA-11): K1/K2 lanes in the start-pose frame, evaluate_k on
made-up pose logs, and the frame-log path (decode, state order, t = 0 at the RUN step) checked
against evaluate_v7 run on the same poses directly. Every number is invented.
Run: python -m pytest tools/tests/test_hil_o1_score.py"""
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import o1_lanes as L          # noqa: E402
import hil_o1_score as H      # noqa: E402
import protocol as P          # noqa: E402
from o1_feasibility import Path8, Schedule, evaluate_v7, path_point, HEAD0, T_LATE   # noqa: E402

DT = 0.001


def k_log(task, lateral=0.0, tail=1.5, sign=1.0):
    """Car exactly on the command point of `task` (shifted `lateral` m to the left of the lane),
    then still at the end for `tail` s."""
    lane, sch = L.Lane(L.k_corners(task)), L.KSchedule(task)
    t = np.arange(0.0, sch.Tcmd + tail, DT)
    pts = []
    for tk in t:
        x, y, h = lane.point_at(sch.s_cmd(tk))
        pts.append((sign * (x - lateral * math.sin(h)), y + lateral * math.cos(h)))
    pts = np.asarray(pts)
    v = np.gradient(pts, DT, axis=0)
    pose = np.column_stack([pts, np.zeros(len(t))])
    return dict(t=t, pose=pose, v_true=np.column_stack([v, np.zeros(len(t))]), duty=np.zeros((len(t), 4)),
                pwm=np.zeros((len(t), 4)))


def test_lanes():
    k1, k2 = L.Lane(L.k_corners('O1-K1')), L.Lane(L.k_corners('O1-K2'))
    assert k1.S == pytest.approx(2.0) and k1.end == (0.0, 2.0)
    assert L.k_corners('O1-K2') == [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    assert k2.S == pytest.approx(4.0) and k2.end == pytest.approx((0.0, 0.0))
    sch = L.KSchedule('O1-K2')
    assert sch.s_cmd(sch.Tcmd) == pytest.approx(4.0, abs=1e-3) and sch.Tcmd == pytest.approx(15.0)


@pytest.mark.parametrize('task', ['O1-K1', 'O1-K2'])
def test_evaluate_k_on_the_command_completes(task):
    out, _ = L.evaluate_k(k_log(task), task)
    assert out['completed'] == 'true' and out['status'] == 'ok'
    assert out['e_perp_max'] < 1e-3 and out['e_trk_max'] < 1e-3
    # still (< v_d = 0.01 m/s) from 0.02 s before T_cmd at a_r = 0.5 m/s^2, so t_end = T_cmd + 1 - 0.02
    assert out['t_end'] == pytest.approx(out['T_cmd'] + 1.0 - 0.02, abs=0.005)


def test_evaluate_k_lateral_offset_and_wrong_direction():
    out, _ = L.evaluate_k(k_log('O1-K1', lateral=0.05), 'O1-K1')
    assert out['e_perp_max'] == pytest.approx(0.05, abs=1e-3) and out['completed'] == 'true'
    bad, _ = L.evaluate_k(k_log('O1-K2', sign=-1.0), 'O1-K2')        # mirrored: goes right, not left
    assert bad['e_perp_max'] > 0.5 and bad['completed'] != 'true'


def fig8_pose(task, t):
    """Plant-frame pose of the exact figure 8 (floor frame rotated back by HEAD0)."""
    sch = Schedule(H.FIG8[task])
    c, s = math.cos(HEAD0), math.sin(HEAD0)
    out = []
    for tk in t:
        fx, fy, head = path_point(sch.tau(tk)[0])
        out.append((c * fx + s * fy, -s * fx + c * fy, (head - HEAD0 + math.pi) % (2 * math.pi) - math.pi))
    return np.asarray(out), sch


def write_frame_log(path, task, k0=400, lifecycle=True, tail=T_LATE + 0.5):
    """Frame log as run_hil.py writes it: plant held at 0 during ARM (k < k0), then the figure 8,
    logged to T_cmd + tail."""
    sch = Schedule(H.FIG8[task])
    n_run = int((sch.Tcmd + tail) / DT)
    t = np.arange(n_run) * DT
    pose, _ = fig8_pose(task, t)
    pose = np.vstack([np.zeros((k0, 3)), pose])
    vel = np.gradient(pose[:, :2], DT, axis=0)
    wz = np.gradient(np.unwrap(pose[:, 2]), DT)
    n = len(pose)
    mosi = np.zeros((n, P.FRAME_SIZE), dtype=np.uint8)
    for k in range(n):
        st = [0.0] * 4 + [pose[k, 0], pose[k, 1], pose[k, 2], vel[k, 0], vel[k, 1], wz[k]]
        on = k >= k0 + 3
        mosi[k] = np.frombuffer(P.encode_state(st, traj_ref=(0.3, 0.0, 0.1) if on else None,
                                               traj_flags=P.TRAJ_VALID if on else 0, seq=k), dtype=np.uint8)
    manifest = dict(period_us=1000, steps=n, plant='MecanumPlant', miso_check=dict(crc_bad=0, empty=0),
                    lifecycle=(dict(outcome='completed', transitions=[dict(step=0, t=0, phase='ARM'),
                                                                      dict(step=k0, t=0.4, phase='RUN')])
                               if lifecycle else None))
    np.savez(path, manifest=np.array(json.dumps(manifest)), mosi=mosi, miso=np.zeros((n, P.FRAME_SIZE), dtype=np.uint8))
    return pose[k0:], vel[k0:], wz[k0:], sch


def test_frame_log_path_matches_evaluate_v7(tmp_path):
    p = str(tmp_path / 'run.npz')
    pose, vel, wz, sch = write_frame_log(p, 'O1-F05')
    out = H.score(p, 'O1-F05')
    assert out['k0'] == 400 and out['t0_src'] == 'lifecycle_RUN' and out['completed'] == 'true'
    t = np.arange(len(pose)) * DT
    ref, _ = evaluate_v7(dict(t=t, pose=pose, v_true=np.column_stack([vel, wz]), duty=np.zeros((len(t), 4)),
                              pwm=np.zeros((len(t), 4))), Path8(), sch)
    assert out['e_perp_max'] == pytest.approx(ref['e_perp_max'], abs=5e-4)      # int16 x, y: 0.15 mm step
    assert out['t_end'] == pytest.approx(ref['t_end'], abs=0.01)
    assert out['e_perp_max'] < 2e-3


def test_command_start_without_lifecycle_and_final_set_guard(tmp_path):
    p = str(tmp_path / 'run.npz')
    write_frame_log(p, 'O1-F08', k0=250, lifecycle=False)
    out = H.score(p, 'O1-F08')
    assert out['k0'] == 253 and out['t0_src'] == 'first_valid_ref'
    with pytest.raises(ValueError, match='final test set'):
        H.score(p, 'O1-K1')
    assert H.main(['--frame-log', p, '--task', 'O1-X']) == 2


# L85 Q1 (Liam 11/11, CODE local_821ee4dc; Fable O1/C6 F-1b): a log that ends before the deadline
# T_cmd + t_late cannot give completed 'false'

def test_log_ending_before_the_deadline_is_inconclusive_not_false(tmp_path):
    p = str(tmp_path / 'run.npz')
    write_frame_log(p, 'O1-F05', tail=-0.5)            # cut 0.5 s before T_cmd: condition 5 never holds
    out = H.score(p, 'O1-F05')
    assert out['completed'] == 'inconclusive' and out['completed_raw'] == 'false'
    assert out['log_covers_deadline'] is False and out['log_end_t'] < out['t_deadline']
    assert 'before t_deadline' in out['completed_note']


def test_log_to_the_deadline_keeps_the_result(tmp_path):
    p = str(tmp_path / 'run.npz')
    write_frame_log(p, 'O1-F05')                       # logged to T_cmd + t_late + 0.5
    out = H.score(p, 'O1-F05')
    assert out['completed'] == 'true' and out['log_covers_deadline'] is True and 'completed_raw' not in out


@pytest.mark.parametrize('completed,end,expect', [
    ('false', 9.0, 'inconclusive'),     # ends 1 s before the deadline 10.0
    ('false', 9.999, 'false'),          # last frame t < deadline is there (period 1 ms)
    ('false', 9.997, 'inconclusive'),
    ('true', 9.0, 'true'),              # completed inside the log: kept
    ('inconclusive', 9.0, 'inconclusive'),
])
def test_log_coverage_rule(completed, end, expect):
    out = H.log_coverage(dict(completed=completed, t_deadline=10.0), end, 0.001)
    assert out['completed'] == expect
    assert out['log_covers_deadline'] is (end >= 9.999 - 1e-9)


def test_duty_comes_from_miso_bytes_0_7_not_the_crc(tmp_path):
    """MISO v4: duty in the torque slots (bytes 0-7, as protocol.decode_torques / the Pi plant), CRC in
    bytes 8-9. Fixed 11/11/2026 (the scorer read bytes 8-15)."""
    p = str(tmp_path / 'run.npz')
    write_frame_log(p, 'O1-F05')
    z = dict(np.load(p))
    miso = z['miso'].copy()
    n = len(miso)
    duty = np.zeros((n, 4))
    duty[:, 0] = 1.0                                         # FL saturated all run, others 0
    duty[::2, 1] = 0.25                                      # FR toggles 0 / 0.25 every frame
    for k in range(n):
        miso[k, 0:8] = np.frombuffer(np.round(duty[k] * 32767).astype('>i2').tobytes(), dtype=np.uint8)
        miso[k, 8:10] = (0xA5, 0x5A)                         # a CRC-like pattern that must not be read as duty
    np.savez(p, manifest=z['manifest'], mosi=z['mosi'], miso=miso)
    out = H.score(p, 'O1-F05')
    assert out['sat_frac'] == pytest.approx(0.25, abs=1e-3)            # one wheel of four at |pwm| 1000
    assert 'tv_duty' not in out
    assert out['tv_duty_h7_1khz'] == pytest.approx(0.25 * 1000, rel=0.01)   # 0.25 per frame, 1000 frames/s
