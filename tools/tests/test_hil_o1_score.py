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
                    state_out_of_range=None,                        # B81-15 check ran on the Pi
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
    assert out['b81_15_check'] == 'manifest'
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


def test_tool_error_plant_state_out_of_range_b81_15():
    """O1 v11.15 3.5 (B81-15): a run whose plant state was NaN / inf, or whose x, y reached the +-5 m bound, is a
    HIL tool error, refused (ValueError -> exit 2). With the manifest key the Pi writes, only the key decides
    (Codex B8C-R1-3: x = 4.99999 m codes as 32767 without reaching the bound). Without it (a log from before the
    check) the run is refused as not enough evidence (B8C-R1-4), unless allow_pre_b81_15: then x or y at +-32767
    refuses; theta at +-pi and other fields at their bound do not."""
    def frames(*rows):
        return np.array([np.frombuffer(P.encode_state(r), np.uint8) for r in rows])
    ok = [1.0] * 4 + [0.5, -0.5, 0.0, 0.1, 0.0, 0.2]
    mosi = frames(*[ok] * 5)
    key = {'state_out_of_range': None}
    assert H.tool_error(key, mosi) == 'manifest'
    near = frames(ok, ok[:4] + [4.99999, -4.99999] + ok[6:], ok)           # rounds to +-32767, check said fine
    assert abs(P._to_int16(4.99999, 5.0)) == 32767 and P.state_out_of_range(ok[:4] + [4.99999] + ok[5:]) == -1
    assert H.tool_error(key, near) == 'manifest'
    with pytest.raises(ValueError, match='HIL tool error.*field 4.*step 3'):
        H.tool_error({'state_out_of_range': dict(step=3, field=4, value='nan')}, mosi)
    with pytest.raises(ValueError, match='not enough evidence'):
        H.tool_error({}, mosi)                                            # log from before the Pi check
    assert H.tool_error({}, mosi, allow_pre_b81_15=True).startswith('pre_b81_15')
    for i, v in ((4, float('nan')), (5, 5.0), (4, float('-inf'))):
        s = list(ok)
        s[i] = v
        m = mosi.copy()
        m[2] = frames(s)[0]
        with pytest.raises(ValueError, match=f'field {i} at the int16 bound at frame 2'):
            H.tool_error({}, m, allow_pre_b81_15=True)
    m = mosi.copy()
    m[1] = frames(ok[:6] + [-math.pi] + ok[7:])[0]
    assert H.tool_error({}, m, allow_pre_b81_15=True)                    # theta -pi: not an error
    m[1] = frames([40.0] * 4 + [0.5, -0.5, 0.0, 3.0, 0.0, 10.0])[0]
    assert H.tool_error({}, m, allow_pre_b81_15=True)                    # wheel speed, v, wz at bound: not in the rule


# ── B81-3 (S05 v3.20; gate JSON format of CODE_R134, rq3_log 4f1a0ef; design RS-4 section 18.1) ──

def write_frame_log_tmap(path, task, tmap, k0=400, tail=T_LATE + 0.5):
    """As write_frame_log, but the plant pose at frame time t is the exact figure 8 at tmap(t) (a jump makes a
    tracking error); velocities from the poses."""
    sch = Schedule(H.FIG8[task])
    t = np.arange(int((sch.Tcmd + tail) / DT)) * DT
    pose, _ = fig8_pose(task, np.array([tmap(tk) for tk in t]))
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
                    state_out_of_range=None,
                    lifecycle=dict(outcome='completed', transitions=[dict(step=0, t=0, phase='ARM'),
                                                                     dict(step=k0, t=0.4, phase='RUN')]))
    np.savez(path, manifest=np.array(json.dumps(manifest)), mosi=mosi, miso=np.zeros((n, P.FRAME_SIZE), dtype=np.uint8))
    return sch


def _jump_ahead(t):
    return t + 2.0 if t >= 3.0 else t


def _check_b81_3_common(g):
    for c in ('g2', 'g3', 'g4', 'g5'):
        chung = g[c]['chung']
        assert set(chung) == {'q', 'lo', 'hi', 'n_undef'}
        for k in ('q', 'lo', 'hi', 'n_undef'):
            a, b = chung[k], g[c][k]
            assert a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b)), (c, k)


def _rq3_new():
    """rq3_log with the B81-3 checks (CODE_R134 fix/r134-answers 4f1a0ef), else None (before that merge)."""
    import rq3_log
    return rq3_log if hasattr(rq3_log, 'gate_g1_fields') else None


def test_b81_3_fields_run_completed(tmp_path):
    p = str(tmp_path / 'run.npz')
    write_frame_log(p, 'O1-F05')
    out = H.score(p, 'O1-F05')
    g = out['gate_v3']
    assert out['completed'] == 'true' and g['g1']['state'] == 'dat'
    assert g['g1']['t_done'] == [out['t_end']] and g['g1']['t_deadline'] == out['t_deadline']
    assert g['g1']['t1'] is None and g['g1']['prog_before_t1'] is None and g['g1']['cond25_before_deadline'] is None
    assert g['g1_fields_note'] == ''
    _check_b81_3_common(g)
    js = json.loads(json.dumps(H._json_safe(out)))
    assert js['gate_v3']['g1']['t_done'] == [pytest.approx(out['t_end'])]


def test_b81_3_short_log_writes_no_g1_fields(tmp_path):
    """L85 Q1 short log: 'false' turned 'inconclusive'; the fields cannot say "not known" (Liam 15/11 CODE_RS4)."""
    p = str(tmp_path / 'run.npz')
    write_frame_log(p, 'O1-F05', tail=-0.5)
    out = H.score(p, 'O1-F05')
    g = out['gate_v3']
    assert out['completed'] == 'inconclusive' and out['completed_raw'] == 'false'
    assert g['g1']['state'] == 'chua_ket_luan'
    assert all(g['g1'][k] is None for k in H.G1_FIELDS) and 'short log' in g['g1_fields_note']
    _check_b81_3_common(g)


def test_b81_3_tracking_error_before_progress(tmp_path):
    """A jump 2 s ahead at t = 3 s: tracking error (s* on the upper end) before the progress reached S - D_d;
    the car then completes the figure and stops at the origin, so conditions 2-5 hold before the deadline:
    G1 chua_ket_luan (O1 3.5 last block), t1 the jump frame."""
    p = str(tmp_path / 'run.npz')
    write_frame_log_tmap(p, 'O1-F05', _jump_ahead)
    out = H.score(p, 'O1-F05')
    g1 = out['gate_v3']['g1']
    assert out['status'] == 'track_err' and out['completed'] == 'inconclusive'
    assert g1['t1'] == pytest.approx(3.0, abs=2e-3) and g1['t1'] == out['t_first_track_err']
    assert g1['prog_before_t1'] is False and g1['cond25_before_deadline'] is True
    assert g1['t_done'] == [None] and g1['state'] == 'chua_ket_luan'


def _gate_stub(completed):
    frames = {k: np.array([0.01, 0.02]) for k in ('e_perp', 'e_trk', 'e_bao')}
    frames['e_theta_cmd_deg'] = np.array([1.0, 2.0])
    return H.o1_gate.gate(frames, completed, v_cmd_max=0.7, wz_cmd_max=1.0, u_t=0.008)


NAN = float('nan')


@pytest.mark.parametrize('completed,t_first,t_prog,c25,want', [
    ('true', NAN, 5.0, True, dict(t1=None, prog=None, c25=None, done=True)),
    ('false', NAN, NAN, False, dict(t1=None, prog=None, c25=None, done=False)),
    ('true', 6.0, 5.0, True, dict(t1=6.0, prog=True, c25=True, done=True)),      # progress before t1: trusted
    ('false', 6.0, 5.0, False, dict(t1=6.0, prog=True, c25=False, done=False)),
    ('inconclusive', 6.0, 7.0, True, dict(t1=6.0, prog=False, c25=True, done=False)),
    ('false', 6.0, NAN, False, dict(t1=6.0, prog=False, c25=False, done=False)),
    ('inconclusive', 6.0, 6.0, True, dict(t1=6.0, prog=False, c25=True, done=False)),  # t_prog = t1: not before
])
def test_b81_3_fields_from_the_evaluator_summary(completed, t_first, t_prog, c25, want):
    out = dict(completed=completed, t_end=9.5, t_deadline=12.0, t_first_track_err=t_first, t_progress=t_prog,
               cond2_5_ever=c25, log_covers_deadline=True)
    g = H.add_b81_3(_gate_stub(completed), out)
    g1 = g['g1']
    assert g1['t1'] == want['t1'] and g1['prog_before_t1'] is want['prog']
    assert g1['cond25_before_deadline'] is want['c25'] and g1['t_deadline'] == 12.0
    assert g1['t_done'] == ([9.5] if want['done'] else [None])
    assert H.g1_state_from_fields(g1) == g1['state']
    rl = _rq3_new()
    if rl is not None:
        f = rl.gate_g1_fields(json.loads(json.dumps(H._json_safe(g1))), 'HIL')
        assert rl.gate_g1_state(f) == g1['state']


def test_b81_3_time_just_below_zero_is_the_first_frame():
    """Codex B813-R1-1: edge mapped at step 400.00000000000216 -> k0 = 400, t(k0) = -2.16e-15 s; a tracking error
    on that frame gave t1 < 0, refused by rq3_log (times >= 0). Within G1_T_NEG_TOL it is written 0.0; further
    below 0 is an error."""
    a = 400.00000000000216
    k0 = int(math.ceil(a - 1e-9))
    t_first = (k0 - a) * 1e-3
    assert k0 == 400 and -H.G1_T_NEG_STEPS * 1e-3 <= t_first < 0.0
    out = dict(completed='false', t_end=11.9, t_deadline=12.0, t_first_track_err=t_first, t_progress=NAN,
               cond2_5_ever=False, log_covers_deadline=True)
    g1 = H.add_b81_3(_gate_stub('false'), out)['g1']
    assert g1['t1'] == 0.0 and g1['prog_before_t1'] is False and g1['cond25_before_deadline'] is False
    assert g1['state'] == 'khong_dat'
    rl = _rq3_new()
    if rl is not None:
        f = rl.gate_g1_fields(json.loads(json.dumps(H._json_safe(g1))), 'HIL')
        assert rl.gate_g1_state(f) == 'khong_dat'
    # progress at the same (negative) time as t1 is not before it: compared on the evaluator's own times
    out.update(t_progress=t_first, completed='false')
    assert H.add_b81_3(_gate_stub('false'), out)['g1']['prog_before_t1'] is False
    for bad in (-1e-6, -5e-10, -2.1e-12):            # Codex B813-R2-1: -5e-10 is 500 x the 1 ms rounding bound
        out.update(t_first_track_err=bad)
        with pytest.raises(ValueError, match='B81-3: time .* below 0'):
            H.add_b81_3(_gate_stub('false'), out, 1e-3)
    assert H._g1_time(None, 1e-3) is None and H._g1_time(0.0, 1e-3) == 0.0 and H._g1_time(3.0, 1e-3) == 3.0
    assert H._g1_time(-2e-12, 1e-3) == 0.0 and H._g1_time(-2e-9, 1.0) == 0.0     # the bound scales with the period
    with pytest.raises(ValueError):
        H._g1_time(-2e-9, 1e-3)
    # every k0 the scorer can choose: a just above / below an integer, small and large step indices (Codex R2-1:
    # a = 10000.000000001 gives t(k0) = -1.0004e-12 s, beyond the exact -1e-12 by float rounding)
    for base in (0, 400, 10000, 30000, 10 ** 6):
        for d in (0.0, 2.16e-12, 1e-9, 1e-9 + 1e-15, 0.99e-9, -1e-12, 0.5, 0.999999999):
            a = base + d
            k0 = int(math.ceil(a - 1e-9))
            for per in (1e-3, 5e-4, 2e-3):
                t = (k0 - a) * per
                assert H._g1_time(t, per) >= 0.0, (a, per, t)


def test_b81_3_k1_run(tmp_path):
    """K1 (evaluate_k path, Codex B813-R1-2): the car on the command point completes; G1 fields and chung."""
    from test_gate_v3_tools import write_k_frame_log
    p = str(tmp_path / 'k1.npz')
    write_k_frame_log(p, 'O1-K1', 0.0)
    out = H.score(p, 'O1-K1', allow_final_set=True)
    g = out['gate_v3']
    assert out['completed'] == 'true' and g['g1']['state'] == 'dat'
    assert g['g1']['t_done'] == [out['t_end']] and g['g1']['t1'] is None
    _check_b81_3_common(g)


@pytest.mark.parametrize('completed,t_first,t_prog,c25', [
    ('true', 6.0, 7.0, True),           # completed although progress came after t1: the evaluator never says so
    ('inconclusive', NAN, 5.0, True),
    ('inconclusive', 6.0, 7.0, False),
    ('false', 6.0, 7.0, True),
])
def test_b81_3_fields_that_disagree_with_g1_are_refused(completed, t_first, t_prog, c25):
    out = dict(completed=completed, t_end=9.5, t_deadline=12.0, t_first_track_err=t_first, t_progress=t_prog,
               cond2_5_ever=c25, log_covers_deadline=True)
    with pytest.raises(ValueError, match='B81-3'):
        H.add_b81_3(_gate_stub(completed), out)


def test_b81_3_gate_accepted_by_rq3_log_b81_3(tmp_path):
    """The whole gate_v3 block through rq3_log.validate_gate of CODE_R134 (B81-3 checks): skipped until that
    rq3_log is on the branch (fix/r134-answers 4f1a0ef, merged after this one, coordination 3.3)."""
    rl = _rq3_new()
    if rl is None:
        pytest.skip('rq3_log without gate_g1_fields (before fix/r134-answers is merged)')
    for name, kw in (('run.npz', {}), ('short.npz', dict(tail=-0.5))):
        p = str(tmp_path / name)
        write_frame_log(p, 'O1-F05', **kw)
        js = json.loads(json.dumps(H._json_safe(H.score(p, 'O1-F05'))))
        rl.validate_gate(js['gate_v3'], 'HIL')
    p = str(tmp_path / 'trk.npz')
    write_frame_log_tmap(p, 'O1-F05', _jump_ahead)
    js = json.loads(json.dumps(H._json_safe(H.score(p, 'O1-F05'))))
    rl.validate_gate(js['gate_v3'], 'HIL')
    from test_gate_v3_tools import write_k_frame_log                 # K1, K2 (Codex B813-R1-2)
    for task, yaw in (('O1-K1', 0.0), ('O1-K2', 0.0), ('O1-K1', 0.2)):
        p = str(tmp_path / f'{task}_{yaw}.npz')
        write_k_frame_log(p, task, yaw)
        js = json.loads(json.dumps(H._json_safe(H.score(p, task, allow_final_set=True))))
        rl.validate_gate(js['gate_v3'], 'HIL')


# ── pi_period_ppm on the steps of T (S05 v3.29 4.7 (i), B81b-3, B81bR2-2; design RS-4 section 18.8) ──

def test_pi_period_formula_and_refusals():
    t = (np.arange(20001, dtype=np.int64) * 999988 + 5_000_000_000)          # 999.988 us per step: -12 ppm
    o = H.pi_period(t, 1000, 13000, 1000)
    assert o['pi_period_ppm'] == pytest.approx(-12.0, abs=1e-6) and o['pi_period_n'] == 12000
    assert o['pi_period_steps'] == [1000, 13000] and o['pi_period_note'] == ''
    o = H.pi_period(t, 1000, 13000, 500)                                       # period_us in the formula
    assert o['pi_period_ppm'] == pytest.approx((999988 / 500000 - 1) * 1e6)
    # only the two ends count (no fit): a late frame inside T changes nothing, a late end frame does
    t2 = t.copy()
    t2[5000] += 300_000
    assert H.pi_period(t2, 1000, 13000, 1000)['pi_period_ppm'] == pytest.approx(-12.0, abs=1e-6)
    t2[13000] += 120_000                                                       # end sent 120 us late: +10 ppm over 12 s
    assert H.pi_period(t2, 1000, 13000, 1000)['pi_period_ppm'] == pytest.approx(-12.0 + 10.0, abs=1e-6)
    for args, word in (((None, 0, 10, 1000), 'without t_send_ns'), ((t, 1000, None, 1000), 't_end not finite'),
                       ((t, 1000, 1000, 1000), 'no step interval'), ((t, 1000, 999, 1000), 'no step interval'),
                       ((t, 1000, 20001, 1000), 'outside'), ((t, -1, 10, 1000), 'outside')):
        o = H.pi_period(*args)
        assert o['pi_period_ppm'] is None and word in o['pi_period_note'], (args[1:], o)
    for bad in (0, -5):
        t3 = t.copy()
        t3[1000] = bad
        assert 'not usable' in H.pi_period(t3, 1000, 13000, 1000)['pi_period_note']
    t3 = t.copy()
    t3[13000] = t3[1000]
    assert 'not usable' in H.pi_period(t3, 1000, 13000, 1000)['pi_period_note']


def _add_t_send(path, ns_per_step, start=7_000_000_000):
    z = dict(np.load(path, allow_pickle=False))
    z['t_send_ns'] = np.arange(len(z['mosi']), dtype=np.int64) * ns_per_step + start
    np.savez(path, **z)


def test_pi_period_in_score(tmp_path):
    """score(): k_first = k0 (t = 0), k_last = the frame at t_end; no t_send_ns in the log -> null with the reason."""
    p = str(tmp_path / 'run.npz')
    write_frame_log(p, 'O1-F05')
    out = H.score(p, 'O1-F05')
    assert out['pi_period_ppm'] is None and 'without t_send_ns' in out['pi_period_note']
    _add_t_send(p, 1_000_050)                                                  # +50 ppm
    out = H.score(p, 'O1-F05')
    k_last = out['k0'] + int(round(out['t_end'] / 1e-3))
    assert out['pi_period_steps'] == [out['k0'], k_last] and out['pi_period_n'] == k_last - out['k0']
    assert out['pi_period_ppm'] == pytest.approx(50.0, abs=1e-6) and out['pi_period_note'] == ''
    js = json.loads(json.dumps(H._json_safe(out)))
    assert js['pi_period_ppm'] == pytest.approx(50.0, abs=1e-6)
    # the S05 table takes it as the measured pi_period_ppm
    sys.path.insert(0, os.path.join(HERE, '..'))
    import s05_final_table as T
    v, src = T.t_end_err_ms(js['t_end'], None, js['pi_period_ppm'])
    assert 'pi_period_ppm' in src and 'can Pi' not in src
    # the value (Codex PIP-R1-2): T x [(1 + e_max)(1 + 50e-6) - 1] x 1e3, e_max = 1001/999 - 1, not the 100 ppm bound
    want = js['t_end'] * ((1001.0 / 999.0) * (1.0 + 50e-6) - 1.0) * 1e3
    assert v == pytest.approx(want, rel=1e-12)
    assert v != pytest.approx(js['t_end'] * ((1001.0 / 999.0) * (1.0 + 100e-6) - 1.0) * 1e3, rel=1e-6)


def _last_frame_of(t_end, a, k0, n, period=1e-3):
    """The frame whose t = (k - a) x period is t_end (found by search, independent of hil_o1_score)."""
    k = np.arange(k0, n)
    j = int(k[np.argmin(np.abs((k - a) * period - t_end))])
    assert abs((j - a) * period - t_end) < 1e-9
    return j


def _mark_t_send(path, ns_per_step=1_000_000):
    """t_send_ns as _add_t_send, returned so a test can move one end."""
    _add_t_send(path, ns_per_step)
    return dict(np.load(path, allow_pickle=False))


def test_pi_period_with_a_mapped_edge_between_steps(tmp_path, monkeypatch):
    """Codex PIP-R1-1: t = 0 at a mapped edge 400.25 (k0 = 401): the last step is the frame at t_end found from
    t = (k - a) x period, not k0 + round(t_end / period); a mark on the t_send_ns of the two expected ends shows
    exactly those two frames are used."""
    p = str(tmp_path / 'run.npz')
    write_frame_log(p, 'O1-F05', k0=400)
    z = _mark_t_send(p)
    n = len(z['mosi'])
    monkeypatch.setattr(H.hil_edge_map, 'read_esp_log', lambda path: {})
    monkeypatch.setattr(H.hil_edge_map, 'load_frames', lambda path: {})
    monkeypatch.setattr(H.hil_edge_map, 'map_edge',
                        lambda *a, **k: dict(ok=True, edge_step=400.25, u_t_s=0.008, offset_ms=0.25, reason=''))
    out = H.score(p, 'O1-F05', esp_log='esp.csv')
    assert out['t0_src'] == 'esp32_run_active_edge' and out['k0'] == 401
    j = _last_frame_of(out['t_end'], 400.25, 401, n)
    assert j != 401 + int(round(out['t_end'] / 1e-3))                         # the wrong formula would differ
    assert out['pi_period_steps'] == [401, j] and out['pi_period_n'] == j - 401
    assert out['pi_period_ppm'] == pytest.approx(0.0, abs=1e-6)
    z['t_send_ns'][j] += 2_000 * (j - 401) // 1000                            # end later by 2 ppm of the span
    z['t_send_ns'][j - 1] += 10 ** 9                                          # a frame inside T: no effect
    np.savez(p, **z)
    out = H.score(p, 'O1-F05', esp_log='esp.csv')
    assert out['pi_period_ppm'] == pytest.approx(2.0, abs=1e-3)


def test_pi_period_short_log_and_k1(tmp_path):
    """Short log (completed false -> inconclusive): T ends at the last frame of the log. K1: the K branch."""
    p = str(tmp_path / 'short.npz')
    write_frame_log(p, 'O1-F05', tail=-0.5)
    z = _mark_t_send(p, 1_000_020)
    out = H.score(p, 'O1-F05')
    assert out['completed'] == 'inconclusive' and out['pi_period_steps'] == [out['k0'], len(z['mosi']) - 1]
    assert out['pi_period_ppm'] == pytest.approx(20.0, abs=1e-6)
    from test_gate_v3_tools import write_k_frame_log
    p = str(tmp_path / 'k1.npz')
    write_k_frame_log(p, 'O1-K1', 0.0)
    z = _mark_t_send(p, 999_990)
    out = H.score(p, 'O1-K1', allow_final_set=True)
    j = _last_frame_of(out['t_end'], out['k0'], out['k0'], len(z['mosi']))
    assert out['pi_period_steps'] == [out['k0'], j] and out['pi_period_ppm'] == pytest.approx(-10.0, abs=1e-6)
