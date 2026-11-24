"""c8_hc_a.py (CODE_HIEUCHUAN 19/11, design 03_CODE/docs/2026-11-19_Thiet_ke_CODE_HIEUCHUAN_v4.md 2.6-3.5) on
synthetic attempts (c8_hc_synth: firmware kinematics, real schedule, known a)."""
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
import c8_hc_a as ca  # noqa: E402
import c8_hc_synth as sy  # noqa: E402


def E(e):
    """F9 e_hieu_m for the four long legs (i = 0..3)."""
    return {i: e for i in range(4)}


def make(tmp_path, name='HC_K1_1', **kw):
    d = str(tmp_path / name)
    truth = sy.make(d, **kw)
    return d, truth


def checks(d):
    ck, info = ca.kiem(ca.load_attempt(d))
    return {n: ok for n, ok, _ in ck}, ck, info


def analyse(d):
    A = ca.load_attempt(d)
    ck, info = ca.kiem(A)
    F = ca.frames_basic(os.path.join(d, 'frames.csv'), os.path.join(d, 'track.json'))
    return ca.analyse(A, F, info)


@pytest.mark.parametrize('cho', ['goc', 'K1', 'K2'])
def test_nominal_attempt_passes_and_recovers_a(tmp_path, cho):
    d, truth = make(tmp_path, f'HC_{cho}_1', cho=cho)
    ok, ck, info = checks(d)
    assert all(ok.values()), [c for c in ck if not c[1]]
    an = analyse(d)
    r = ca.place_result(cho, [(os.path.basename(d), an)], os.path.basename(d), e_hieu=E(0.002), q=[0.004, -0.003],
                        du_m=0.001, neo=10.0, psi_cuc=0.0)
    assert abs(math.radians(r['a_k_deg']) - truth['a_true']) <= math.radians(r['u_a_k_deg'])
    assert abs(r['a_k_deg'] - 0.7) < 0.05
    assert all(r['cong'].values())
    assert r['trang_thai'] == 'chan_doan' and 'chuoi khung chua khoa (F1, A6, q)' in r['ly_do']
    assert r['phi_hat_chuan_k_deg'] == pytest.approx(10.0 - r['a_k_deg'])
    assert r['u_phi_hat_chuan_k_deg'] == r['u_a_k_deg']


def test_window_roles():
    import c8_hc_seq as hs
    assert ca.window_roles(hs.schedule('goc')) == ([(0, 'ref_12b')] + [(i, 'dung_12b') for i in range(1, 7)]
                                                   + [(6, 'E_dau'), (8, 'E_cuoi')])
    assert ca.window_roles(hs.schedule('K2')) == [(0, 'ref_12b')] + [(i, 'dung_12b') for i in range(1, 7)]
    assert ca.window_roles(hs.schedule('K1', 'E')) == [(0, 'E_dau'), (2, 'E_cuoi')]


def test_cua_so_writes_f2_and_f5(tmp_path):
    d, _ = make(tmp_path, 'HC_goc_1', cho='goc')
    rc = ca.main(['cua_so', d, '--frames', os.path.join(d, 'frames.csv'), '--track', os.path.join(d, 'track.json')])
    assert rc == 0
    rows = [json.loads(x) for x in open(os.path.join(d, 'cua_so_v1.jsonl'), encoding='utf-8')]
    assert [r['loai'] for r in rows] == ['ref_12b'] + ['dung_12b'] * 6 + ['E_dau', 'E_cuoi']
    assert all(r['schema'] == 'cua_so_v1' and r['cho'] == 'goc' and len(r['khung']) >= 120 for r in rows)
    doan = json.load(open(os.path.join(d, 'doan_12b_v1.json'), encoding='utf-8'))
    assert doan['schema'] == 'c8_hc_doan_12b_v1' and len(doan['doan']) == 6
    assert sum(1 for x in doan['doan'] if x['dai']) == 4
    assert all(ca.f2_problems(r) == [] for r in rows) and ca.f5_problems(doan) == []           # HCC1-11
    assert rows[-1]['t_canh_s'] is None and rows[-1]['bien_cuoi'] == 'duoi_log'
    assert all(r['t_canh_s'] is not None and r['bien_cuoi'] == 'canh_lenh' for r in rows[:-1])
    assert [r['vi_tri_lenh_m'] for r in rows[:7]] == [0.0, 0.5, -0.5, 0.5, -0.5, 0.5, 0.0]
    assert doan['doan'][0]['cua_so_a']['loai'] == 'dung_12b' and doan['doan'][4]['cua_so_a']['loai'] == 'ref_12b'
    body = dict(doan)
    sha = body.pop('sha256')
    assert sha == __import__('hashlib').sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    with pytest.raises(FileExistsError):                                        # never overwritten
        ca.cmd_cua_so(type('a', (), dict(dir=d, frames=os.path.join(d, 'frames.csv'), track=os.path.join(d, 'track.json'),
                                         clip=None, dets=None, session_sha256=None, a6=None, xoay=None,
                                         session=None))())


def test_phan_E_only(tmp_path):
    d, _ = make(tmp_path, 'HC_K1_E', cho='K1', phan='E')
    ok, ck, info = checks(d)
    assert all(ok.values()), [c for c in ck if not c[1]]
    an = analyse(d)
    assert an['legs'] == [] and len(an['windows']) == 3


# ------------------------------------------------------------------ kiem failures
def test_kiem_catches_ultrasonic_cut(tmp_path):
    d, _ = make(tmp_path, cho='K1', cut=(12.0, 12.3))                          # inside the first back leg (10.9-16.3 s)
    ok, ck, _ = checks(d)
    assert not ok['lenh_nhan_khop_lich']


def test_kiem_catches_second_run_edge(tmp_path):
    d, _ = make(tmp_path, cho='K2', second_edge=True)
    ok, _, _ = checks(d)
    assert not ok['mot_canh_run_active']


def test_kiem_catches_rv_repeating_during_E(tmp_path):
    d, _ = make(tmp_path, cho='K1', rv_freeze=(70.0, 90.0))                     # inside the first turn
    ok, ck, _ = checks(d)
    e = [c for c in ck if c[0] == 'imu_e_tuong_quan']
    assert len(e) == 2 and not e[0][1] and e[1][1], e


def test_kiem_catches_reboot_between_questions(tmp_path):
    d, _ = make(tmp_path, cho='K2', boot_after='0badb00f')
    ok, _, _ = checks(d)
    assert not ok['dinh_danh_gain']


def test_kiem_passes_nominal_pid_like_turn_lag(tmp_path):
    d, _ = make(tmp_path, cho='goc', lag=0.4)                                   # HC3-1: no integral-tracking gate
    ok, ck, _ = checks(d)
    assert all(ok.values()), [c for c in ck if not c[1]]


# ------------------------------------------------------------------ HC3-4 counter-examples
def result(d, cho, psi_cuc=0.0, extra=()):
    ans = [(os.path.basename(d), analyse(d))] + [(os.path.basename(x), analyse(x)) for x in extra]
    return ca.place_result(cho, ans, os.path.basename(d), e_hieu=E(0.0), q=[0.0, 0.0], du_m=0.0, neo=0.0, psi_cuc=psi_cuc)


def test_camera_heading_drift_needs_floor_referenced_frames(tmp_path):
    """HCB4: a_l takes its heading from the camera only, and the dynamic check is anchored on the two leg ends, so a
    camera heading that drifts in time (the camera turning on its mount) is NOT seen by the checks: it is excluded
    at the source, the frames must be referenced to the floor marks (camera pose refitted: c7_measure track per
    frame, c8_hc_video every second). Frames without that never give a locked result."""
    d, truth = make(tmp_path, 'HC_K2_1', cho='K2', cam_bias=lambda t: math.radians(0.010) * max(t, 0.0),
                    noise_psi_deg=0.02)
    A = ca.load_attempt(d)
    ck, info = ca.kiem(A)
    F = dict(frames(d), neo_san=False)
    an = ca.analyse(A, F, info)
    assert an['neo_san'] is False and analyse(d)['neo_san'] is True
    r = ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0.0, 0.0], du_m=0.0, neo=0.0, psi_cuc=0.0)
    assert any('khong neo moc san' in w for w in r['ly_do']) and r['trang_thai'] != 'khoa'
    ok, vk = video(d, F)
    assert not ok['neo_moc_san']
    ok, vk = video(d, frames(d))
    assert ok['neo_moc_san']


def test_rv_frozen_with_body_wiggle_inside_index_or_flagged(tmp_path):
    # K1: the second long leg (- to +) runs 20.3-25.7 s after the start; the body turns 0.25 deg and back while
    # the RV keeps reporting the old value
    d, truth = make(tmp_path, cho='K1', wiggle=(20.6, 25.4, math.radians(0.25)), rv_freeze=(20.3, 25.7),
                    noise_psi_deg=0.02)
    r = result(d, 'K1')
    err = abs(math.radians(r['a_k_deg']) - truth['a_true'])
    assert err <= math.radians(r['u_a_k_deg']) or r['trang_thai'] == 'chua_ket_luan'


def test_common_camera_bias_at_both_extremes_needs_c8b_bound(tmp_path):
    # +0.15 deg on the camera heading whenever the car is away from the reference point (Codex HC3-4, K1): the four
    # a_l agree, so only the C8B two-end camera bound (u_psi_cuc) covers it; without it the place is chan_doan
    def bias(t):
        return math.radians(0.15) if 6.0 < t < 60.0 else 0.0
    d, truth = make(tmp_path, cho='K1', cam_bias=bias, noise_psi_deg=0.02)
    r0 = ca.place_result('K1', [(os.path.basename(d), analyse(d))], os.path.basename(d), e_hieu=E(0.0), q=[0, 0],
                         du_m=0.0, neo=0.0, psi_cuc=None)
    assert r0['trang_thai'] in ('chan_doan', 'chua_ket_luan')
    r1 = result(d, 'K1', psi_cuc=math.radians(0.15))
    err = abs(math.radians(r1['a_k_deg']) - truth['a_true'])
    assert err <= math.radians(r1['u_a_k_deg'])


def test_max_over_every_attempt(tmp_path):
    d1, _ = make(tmp_path, 'HC_K1_1', cho='K1', seed=1)
    d2, _ = make(tmp_path, 'HC_K1_2', cho='K1', seed=2, a_true_deg=1.0)       # a failed attempt, other mounting angle
    r_one = result(d1, 'K1')
    r_two = result(d1, 'K1', extra=(d2,))
    assert r_two['u_hang_deg']['lech_lon_nhat'] >= 0.25 > r_one['u_hang_deg']['lech_lon_nhat']
    assert r_two['a_k_deg'] == pytest.approx(r_one['a_k_deg'])                 # a is the issued attempt's


def test_spread_gate(tmp_path):
    # the camera heading of one long leg off by 0.45 deg over its whole motion (K1, second long leg 20.3-25.7 s):
    # that a_l moves by the path mean of the bias, spread > 0.3 deg -> chua_ket_luan
    def bias(t):
        return math.radians(0.45) if 20.0 < t < 26.0 else 0.0
    d, _ = make(tmp_path, cho='K1', cam_bias=bias, noise_psi_deg=0.02)
    r = result(d, 'K1')
    assert not r['cong']['do_tan'] and r['trang_thai'] == 'chua_ket_luan'


def test_tag_offset_refused(tmp_path):
    d, _ = make(tmp_path, cho='K2')
    with open(os.path.join(d, 'track.json'), 'w') as f:
        json.dump(dict(tag_offset=[0.01, 0.0, 0.0]), f)
    with pytest.raises(ca.AError):
        ca.frames_basic(os.path.join(d, 'frames.csv'), os.path.join(d, 'track.json'))


def test_kiem_video_with_injected_sparse_frames(tmp_path, monkeypatch):
    d, _ = make(tmp_path, 'HC_goc_1', cho='goc')
    F = ca.frames_basic(os.path.join(d, 'frames.csv'), os.path.join(d, 'track.json'))
    sub = {k: (v[::3] if isinstance(v, np.ndarray) else v) for k, v in F.items()}
    sub['fps'] = 20.0
    import c8_hc_video as hv
    monkeypatch.setattr(hv, 'sparse_frames', lambda clip, session, stride=3: sub)
    assert ca.main(['kiem_video', d, '--clip', 'x.MOV', '--session', 's.json']) == 0


# ------------------------------------------------------------------ Codex code review round 1 (HCC1-4..6)
def edit_log(d, fn):
    """Rewrite log.csv through fn(header, rows) (rows: lists of str) and refresh the logger summary."""
    import csv
    p = os.path.join(d, 'log.csv')
    with open(p, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    hdr, body = rows[0], rows[1:]
    fn(hdr, body)
    with open(p, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows([hdr] + body)
    sy.write_summary(p)


def col(hdr, name):
    return hdr.index(name)


def run_rows(hdr, body):
    ra = [i for i, r in enumerate(body) if r[col(hdr, 'run_active')] == '1']
    return ra[0], ra[-1]


def test_firmware_timeout_wait_is_not_a_sender_gap(tmp_path):
    """HCC1-4: after the last zero packet the firmware keeps run_active up to 200 ms (real logs 193-197 ms); that
    wait is checked on its own and does not fail the 60 ms sender gate."""
    d, _ = make(tmp_path, cho='K2')
    ok, ck, info = checks(d)
    L = ca.load_attempt(d)['log']
    assert L['traj_age_ms'][info['i1']] >= 180 and info['j_s'] < info['i1']
    assert ok['traj_age'] and ok['het_han_firmware'], ck


def test_new_packet_during_timeout_wait_fails(tmp_path):
    d, _ = make(tmp_path, cho='K2')

    def fn(hdr, body):
        i0, i1 = run_rows(hdr, body)
        body[i1 - 5][col(hdr, 'traj_age_ms')] = '5'                          # a packet inside the wait, not fresh-gated
        body[i1 - 5][col(hdr, 'sp_vx')] = '0.000000'
    edit_log(d, fn)
    ok, ck, _ = checks(d)
    # HCB2: with the 160 ms quality limit the ages of the wait before that packet pass traj_age; the packet still
    # moves the end of the sender domain 0.15 s past the schedule, which the zero-tail span check refuses
    assert not (ok['traj_age'] and ok['het_han_firmware'] and ok['lenh_nhan_khop_lich']), ck
    assert not ok['lenh_nhan_khop_lich'] and 'mien phat' in dict((n, x) for n, _, x in ck)['lenh_nhan_khop_lich']


def test_late_packets_after_run_active_falls_fail(tmp_path):
    """HCC1-5 (Codex): sp_vx 0.10 m/s on 10 rows after run_active fell passed every gate before."""
    d, _ = make(tmp_path, cho='K2')

    def fn(hdr, body):
        i0, i1 = run_rows(hdr, body)
        for r in body[i1 + 3:i1 + 13]:
            r[col(hdr, 'sp_vx')] = '0.100000'
    edit_log(d, fn)
    ok, ck, _ = checks(d)
    assert not ok['khong_goi_muon']


def test_vy_during_turn_fails(tmp_path):
    d, _ = make(tmp_path, cho='K1')

    def fn(hdr, body):
        jw = [i for i, r in enumerate(body) if abs(float(r[col(hdr, 'sp_wz')])) > 0.1]
        for i in jw[50:60]:
            body[i][col(hdr, 'sp_vy')] = '0.020000'
    edit_log(d, fn)
    ok, ck, _ = checks(d)
    assert not ok['lenh_nhan_khop_lich']


def test_cancelling_wz_pulses_in_translation_fail(tmp_path):
    d, _ = make(tmp_path, cho='K2')

    def fn(hdr, body):
        jv = [i for i, r in enumerate(body) if abs(float(r[col(hdr, 'sp_vx')])) > 0.05]
        body[jv[30]][col(hdr, 'sp_wz')] = '0.300000'
        body[jv[31]][col(hdr, 'sp_wz')] = '-0.300000'
    edit_log(d, fn)
    ok, ck, _ = checks(d)
    assert not ok['lenh_nhan_khop_lich']


def test_block_shifted_in_time_fails(tmp_path):
    """HCC1-5: same integral and duration, 0.3 s late (a shorter 4 s hold before it)."""
    d, _ = make(tmp_path, cho='K2')

    def fn(hdr, body):
        c = col(hdr, 'sp_vx')
        nz = [i for i, r in enumerate(body) if abs(float(r[c])) > 1e-6]
        blocks, cur = [], [nz[0]]
        for i in nz[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                blocks.append(cur)
                cur = [i]
        blocks.append(cur)
        b = blocks[2]
        vals = [body[i][c] for i in b]
        for i in b:
            body[i][c] = '0.000000'
        for k, i in enumerate(b):
            body[i + 30][c] = vals[k]
    edit_log(d, fn)
    ok, ck, _ = checks(d)
    assert not ok['lenh_nhan_khop_lich']


def test_missing_or_wrong_summary_fails(tmp_path):
    d, _ = make(tmp_path, cho='K2')
    os.remove(os.path.join(d, 'log.csv.summary.json'))
    assert not checks(d)[0]['logger_summary']
    sy.write_summary(os.path.join(d, 'log.csv'))
    with open(os.path.join(d, 'log.csv'), 'a', encoding='utf-8') as f:
        f.write('\n')
    assert not checks(d)[0]['logger_summary']


def write_events(d, lines):
    with open(os.path.join(d, 'log.csv.events.txt'), 'w', encoding='utf-8') as f:
        for ln in lines:
            for _ in range(3):
                f.write(f'2026-11-19T10:00:00.000000+00:00 {ln}\n')


def test_gain_event_first_seq_outside_log_fails(tmp_path):
    """HCC1-6 (Codex): '#gain' of the right boot with first_seq 999999 outside the log passed before."""
    d, _ = make(tmp_path, cho='K2')
    write_events(d, ['#gain,id=0,applied_n=0,ctrl=PID,first_seq=999999,line_drop=0,boot=e75f708f'])
    assert not checks(d)[0]['dinh_danh_fw_gain_log']
    write_events(d, ['#gain,id=0,applied_n=0,ctrl=PID,first_seq=1005,line_drop=0,boot=e75f708f'])
    assert checks(d)[0]['dinh_danh_fw_gain_log']
    write_events(d, ['#gain,id=0,applied_n=0,ctrl=PID,first_seq=1005,line_drop=0,boot=0badb00f'])
    assert not checks(d)[0]['dinh_danh_fw_gain_log']


def test_us_end_cut_ticks_read_and_covering(tmp_path):
    """HCC1-6: the firmware field is cut_ticks (us_state.c 48); 'co'/'khong' only when the #us_end covers the run."""
    d, _ = make(tmp_path, cho='K2')
    L = ca.load_attempt(d)['log']
    last = int(L['log_seq'][-1])
    first = int(L['log_seq'][0])
    start = f'#us,ev=start,boot=e75f708f,on=0,swb=-1,first_seq={first},line_drop=0'
    end = (f'#us_end,boot=e75f708f,on=0,swb=-1,changes=0,blk_ticks=0,cut_ticks={{c}},ticks={last - first + 1},'
           f'last_data_seq={last},line_drop=0')
    write_events(d, [start, end.format(c=6)])
    _, _, info = checks(d)
    assert info['cat_pwm'] == 'co'
    write_events(d, [start, end.format(c=0)])
    assert checks(d)[2]['cat_pwm'] == 'khong'
    write_events(d, [start.replace('e75f708f', '0badb00f'), end.format(c=0).replace('e75f708f', '0badb00f')])
    assert checks(d)[2]['cat_pwm'] == 'chua_xac_minh'
    # HCC2-3: blk_ticks counts the whole motor run (us_state.h): reported, not a gate of the schedule
    write_events(d, [start, end.format(c=0).replace('blk_ticks=0', 'blk_ticks=3')])
    ok, _, info = checks(d)
    assert ok['khong_cat_sieu_am'] and info['blk_ticks'] == 3
    # HCC2-3 (Codex): two footers of the same run that disagree (cut 5 and 0) are no evidence
    write_events(d, [start, end.format(c=5), end.format(c=0)])
    assert checks(d)[2]['cat_pwm'] == 'chua_xac_minh'


# ------------------------------------------------------------------ Codex code review round 1 (HCC1-7..9)
def frames(d):
    return ca.frames_basic(os.path.join(d, 'frames.csv'), os.path.join(d, 'track.json'))


def drop(F, t_from, t_to, D, t_s=6.0):
    """Mark camera frames invalid between sender times t_from..t_to (camera time = sender + t_s - D)."""
    G = dict(F)
    G['ok'] = F['ok'] & ~((F['t'] + D - t_s >= t_from) & (F['t'] + D - t_s <= t_to))
    return G


def video(d, F):
    A = ca.load_attempt(d)
    ck, info = ca.kiem(A)
    vk, an = ca.video_checks(A, F, info)
    return {n: ok for n, ok, _ in vk}, vk


def test_video_E_only_with_all_frames_passes(tmp_path):
    """HCC1-7 (Codex): an E-only attempt with every frame returned LAM_LAI (0/4 long legs)."""
    d, _ = make(tmp_path, 'HC_K1_E', cho='K1', phan='E')
    ok, vk = video(d, frames(d))
    assert all(ok.values()), vk


def test_video_E_gap_of_3_s_fails(tmp_path):
    """HCC1-7 (Codex): 3 s without the tag inside the first turn direction passed DAT."""
    d, truth = make(tmp_path, 'HC_K1_1', cho='K1')
    ok, vk = video(d, drop(frames(d), 65.0, 68.0, truth['D']))
    assert not ok['phu_E'] and ok['giong_camera_log'], vk


def test_tag_off_turn_axis_does_not_make_translations(tmp_path):
    """HCC1-8 (Codex): tag centre 50 mm off the turning axis: E made 8 camera translation blocks vs 6 in the log."""
    d, _ = make(tmp_path, 'HC_goc_1', cho='goc', q=(0.05, 0.0))
    A = ca.load_attempt(d)
    ck, info = ca.kiem(A)
    D, u_D, _ = ca.align(A['log'], info['blocks'], frames(d))
    kinds = [k for _, _, k in ca.cam_motion_events(frames(d))]
    assert kinds == ['vx'] * 6 + ['wz+', 'wz-'] and u_D <= ca.U_D_MAX_S


def test_short_heading_pulse_fails_dynamic_yaw(tmp_path):
    """HCC1-9 (Codex): a 0.40 deg camera-IMU difference for 0.6 s in a long leg gave 0.24 deg with a 1 s window."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2', noise_psi_deg=0.02,
                cam_bias=lambda t: math.radians(0.40) if 12.0 <= t <= 12.6 else 0.0)
    an = analyse(d)
    leg = [lg for lg in an['legs'] if lg['dai']][0]
    assert math.degrees(leg['eps']) >= 0.35
    r = result(d, 'K2')
    assert not r['cong']['yaw_dong'] and r['trang_thai'] == 'chua_ket_luan'


def test_camera_gap_in_leg_is_no_evidence(tmp_path):
    """HCC1-9, HCL1-9: a 1 s camera gap inside a long leg: C would be summed straight across it -> the leg is
    'thieu' (ho khung > EPS_SMOOTH_S), never hop_le with eps NaN; the place is not locked."""
    d, truth = make(tmp_path, 'HC_K2_1', cho='K2', noise_psi_deg=0.02)
    A = ca.load_attempt(d)
    ck, info = ca.kiem(A)
    an = ca.analyse(A, drop(frames(d), 12.0, 13.0, truth['D']), info)
    leg = [lg for lg in an['legs'] if lg['dai']][0]
    assert leg['trang_thai'] == 'thieu' and leg['ho_khung_s'] > ca.EPS_SMOOTH_S and 'ho khung' in leg['ly_do']
    assert all(lg['trang_thai'] == 'hop_le' for lg in an['legs'][1:])
    r = ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0, 0], du_m=0.0, neo=0.0, psi_cuc=0.0)
    assert r['trang_thai'] != 'khoa'


def test_rv_repeat_015s_during_turn_fails(tmp_path):
    """HCC1-9 (Codex): RV repeating 0.15 s while turning still gave corr 0.945."""
    d, _ = make(tmp_path, cho='K1', rv_freeze=(62.0, 62.15))
    ok, ck, _ = checks(d)
    e = [c for c in ck if c[0] == 'imu_e_tuong_quan']
    assert not e[0][1], e


# ------------------------------------------------------------------ Codex code review round 1 (HCC1-10..12)
def tinh_args(d, cho, *extra):
    return ['tinh', '--cho', cho, '--cap', d, '--frames-of',
            f"{d}={os.path.join(d, 'frames.csv')},{os.path.join(d, 'track.json')}", *extra]


def run_cua_so(d):
    assert ca.main(['cua_so', d, '--frames', os.path.join(d, 'frames.csv'), '--track', os.path.join(d, 'track.json'),
                    '--clip', os.path.join(d, 'track.json')]) == 0        # any file stands for the clip (sha only)
    rows = [json.loads(x) for x in open(os.path.join(d, 'cua_so_v1.jsonl'), encoding='utf-8')]
    return rows[0], json.load(open(os.path.join(d, 'doan_12b_v1.json'), encoding='utf-8'))


def write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)
    return path


def test_tinh_refuses_cap_that_failed_kiem(tmp_path, capsys):
    """HCC1-10 (Codex): a cap attempt failing dinh_danh_gain (boot changed) was issued 'khoa'."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2', boot_after='0badb00f')
    out = str(tmp_path / 'f7.json')
    assert ca.main(tinh_args(d, 'K2', '--out', out)) == 1
    r = json.load(open(out, encoding='utf-8'))
    assert r['trang_thai'] == 'chua_ket_luan' and 'dinh_danh_gain' in r['ly_do'][0] and r['a_k_deg'] is None


def f6(ref, cho='K2', ids=None, n_luot=1, top='dat', win=None, **kw):
    """F6 hieu_chuan_neo_v1 as NHACTAM writes it (feat/c8-lan-lap e7b8428): file trang_thai, chuan_12b keyed
    'cho|luot_id', the place block (the record itself with one 12b run there, else chua_ket_luan), and the ref_12b
    windows with khung_chung = a subset of khung_ma_hop_le (anchor gate & A6), sha256 of the sorted IDs."""
    import hashlib
    ids = sorted(ref['khung_ma_hop_le'][1:-1] if ids is None else ids)
    sha = hashlib.sha256(','.join(str(i) for i in sorted(ids)).encode()).hexdigest()
    chuan, wins = {}, []
    for n in range(n_luot):
        luot = ref['luot_id'] if n == 0 else f"{ref['luot_id']}_{n + 1}"
        clip = ref['clip_sha256'] if n == 0 else f'{n}' * 64
        rec = dict(trang_thai='dat', ly_do='', cho=cho, luot_id=luot, clip_sha256=clip, khung_chung_sha256=sha,
                   psi_chuan_k_deg=12.0, frames_sha256=ref['frames_sha256'])
        if n == 0:
            rec.update(kw)
        chuan[f'{cho}|{luot}'] = rec
        w = dict(meta=dict(luot=luot, cho=cho, loai='ref_12b', clip=ref.get('clip'), clip_sha=clip,
                           frames_sha=ref['frames_sha256']),
                 khung_chung=dict(id=list(ids), pts_s=[]), khung_chung_sha256=sha)
        if n == 0 and win:
            w['meta'].update(win)
        wins.append(w)
    blk = next(iter(chuan.values())) if n_luot == 1 else dict(
        trang_thai='chua_ket_luan', ly_do='nhieu_12b_xem_chuan_12b', khoa=list(chuan))
    return dict(schema='hieu_chuan_neo_v1', trang_thai=top, cho={cho: blk}, chuan_12b=chuan, cua_so=wins)


def test_tinh_checks_f6_and_f9_sources(tmp_path):
    """HCC1-10: F6 must be made on the ref_12b window of the cap (clip, frames), F9 on its F5, by leg i."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    ref, doan = run_cua_so(d)
    good_neo = write_json(str(tmp_path / 'neo.json'), f6(ref))
    bad_neo = write_json(str(tmp_path / 'neo_x.json'), f6(ref, clip_sha256='0' * 64))
    good_e = write_json(str(tmp_path / 'e.json'), dict(doan_12b_sha256=doan['sha256'],
                                                       doan=[dict(i=i, e_hieu_m=0.002) for i in range(6)]))
    bad_e = write_json(str(tmp_path / 'e_x.json'), dict(doan_12b_sha256='1' * 64, doan=[dict(i=0, e_hieu_m=0.002)]))
    only99 = write_json(str(tmp_path / 'e_99.json'), dict(doan_12b_sha256=doan['sha256'],
                                                         doan=[dict(i=99, e_hieu_m=0.002)]))
    assert ca.main(tinh_args(d, 'K2', '--neo', bad_neo)) == 2
    assert ca.main(tinh_args(d, 'K2', '--e-hieu', bad_e)) == 2
    out = str(tmp_path / 'f7_99.json')
    assert ca.main(tinh_args(d, 'K2', '--neo', good_neo, '--e-hieu', only99, '--out', out)) == 1
    assert not json.load(open(out, encoding='utf-8'))['cong']['e_hieu_hop_le']
    out = str(tmp_path / 'f7.json')
    assert ca.main(tinh_args(d, 'K2', '--neo', good_neo, '--e-hieu', good_e, '--out', out)) == 0
    r = json.load(open(out, encoding='utf-8'))
    assert r['phi_hat_chuan_k_deg'] == pytest.approx(12.0 - r['a_k_deg'])
    roles = {n['vai_tro'] for n in r['nguon']}
    assert {'hieu_chuan_neo', 'e_hieu_m', 'doan_12b_v1', 'cua_so_v1'} <= roles and r['commit_hil']
    leg = r['luot_thu'][0]['doan'][0]
    assert 'a_l_deg' in leg and 'theta_a_deg' in leg and 'eps_deg' in leg and 'a_l' not in leg   # HCC1-12


def test_empty_required_window_and_lost_holdout_not_khoa(tmp_path):
    """HCC1-11 (Codex): ref_12b without valid frames still gave 'khoa'; a lost held-out leg was skipped."""
    d, truth = make(tmp_path, 'HC_K2_1', cho='K2')
    A = ca.load_attempt(d)
    ck, info = ca.kiem(A)
    F = drop(frames(d), 0.0, 3.9, truth['D'])                     # the whole ref_12b window (before the first move)
    an = ca.analyse(A, F, info)
    r = ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0, 0], du_m=0.0, neo=0.0, psi_cuc=0.0,
                        need_windows=list(range(7)))
    assert r['trang_thai'] == 'chua_ket_luan' and 'cua so thieu khung' in r['ly_do'][0]
    r2 = ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0, 0], du_m=0.0, neo=0.0, psi_cuc=0.0)
    assert not r2['cong']['giu_ngoai'] and r2['trang_thai'] == 'chua_ket_luan'


def test_missing_long_leg_is_chua_ket_luan_not_keyerror(tmp_path, capsys):
    """HCC1-11 (Codex): no long legs printed 'loi: a_k_deg'."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2', noise_psi_deg=0.02)
    with open(os.path.join(d, 'frames.csv'), encoding='utf-8') as f:
        lines = f.readlines()
    with open(os.path.join(d, 'frames.csv'), 'w', encoding='utf-8') as f:
        f.write(lines[0])
        for ln in lines[1:]:
            c = ln.split(',')
            t = float(c[1])
            if 20.0 < t + 50.0 - 6.0 < 30.0:                          # the stop window between legs 1 and 2
                c[2] = '0'
            f.write(','.join(c))
    rc = ca.main(tinh_args(d, 'K2'))
    txt = capsys.readouterr().out
    assert rc == 1 and 'loi:' not in txt and 'chua_ket_luan' in txt


def test_e_outside_baseline_rejected_and_q_turn_term(tmp_path):
    """HCC1-12: e >= L is refused (not clamped); u_q has the q turn term unless F9 says it is covered."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    an = analyse(d)
    r = ca.place_result('K2', [('x', an)], 'x', e_hieu={0: 0.6, 1: 0.0, 2: 0.0, 3: 0.0}, q=[0.01, 0.0], du_m=0.0,
                        neo=0.0, psi_cuc=0.0)
    assert not r['cong']['e_hieu_hop_le'] and r['trang_thai'] == 'chua_ket_luan'
    a1 = ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0.05, 0.0], du_m=0.0, neo=0.0, psi_cuc=0.0)
    a2 = ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0.05, 0.0], du_m=0.0, neo=0.0, psi_cuc=0.0,
                         bao_q=True)
    assert a1['u_hang_deg']['u_q'] > 0.0 == a2['u_hang_deg']['u_q']


def test_tinh_real_footprint_from_camera(tmp_path):
    """HCC1-13: tinh computes the real footprint from the camera poses when the 32 tags are given."""
    import test_c8_hc_vet as tv
    if not os.path.isfile(tv.MOC):
        pytest.skip('moc_san_do_32.csv not reachable (set C8_MOC_32)')
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    out = str(tmp_path / 'f7.json')
    ca.main(tinh_args(d, 'K2', '--moc', tv.MOC, '--out', out))
    r = json.load(open(out, encoding='utf-8'))
    assert r['vet_thuc'] is not None and r['cong']['vet_thuc'] and r['vet_thuc']['khe_m'] > 0.05


# ------------------------------------------------------------------ Codex code review round 2 (HCC2-2..8)
def test_log_cut_while_run_active_fails(tmp_path):
    """HCC2-2 (Codex): a K2 log cut at a row with run_active=1, traj_age 7 ms and a matching summary passed."""
    d, _ = make(tmp_path, cho='K2')

    def fn(hdr, body):
        i0, i1 = run_rows(hdr, body)
        ages = [int(body[i][col(hdr, 'traj_age_ms')]) for i in range(i1 - 40, i1)]
        cut = i1 - 40 + max(k for k, a in enumerate(ages) if a <= 10)
        del body[cut + 1:]
    edit_log(d, fn)
    ok, ck, _ = checks(d)
    assert not ok['het_han_firmware'], ck


def mirror_second_turn(F, D, t_from, t_s=6.0):
    """Camera heading of the second E direction mirrored about its start (both turns then go the same way)."""
    G = dict(F)
    ps = np.unwrap(F['psi'].copy())
    m = F['t'] + D - t_s >= t_from
    i0 = int(np.flatnonzero(m)[0])
    ps[m] = ps[i0] - (ps[m] - ps[i0])
    G['psi'] = (ps + np.pi) % (2 * np.pi) - np.pi
    return G


def test_video_E_wrong_direction_fails(tmp_path):
    """HCC2-4 (Codex): the second turn mirrored (two turns the same way, ~+5 turns) passed with 2.50 each."""
    d, truth = make(tmp_path, 'HC_K1_1', cho='K1')
    import c8_hc_seq as hs
    t2 = [s for s in hs.schedule('K1') if s['loai'] == 'xoay'][1]['t0'] - 2.0
    ok, vk = video(d, mirror_second_turn(frames(d), truth['D'], t2))
    assert not (ok['giong_camera_log'] and ok['phu_E']), vk


def test_tinh_refuses_other_frames_than_f2_f5(tmp_path):
    """HCC2-5 (Codex): F2/F5 made from frames A, tinh run on frames B (+0.25 deg) still gave khoa."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    ref, doan = run_cua_so(d)
    neo = write_json(str(tmp_path / 'neo.json'), f6(ref))
    fb = os.path.join(d, 'frames_B.csv')
    with open(os.path.join(d, 'frames.csv'), encoding='utf-8') as f, open(fb, 'w', encoding='utf-8') as g:
        g.write(f.readline())
        for ln in f:
            c = ln.rstrip('\n').split(',')
            c[6] = f'{float(c[6]) + math.radians(0.25):.6f}'
            g.write(','.join(c) + '\n')
    args = ['tinh', '--cho', 'K2', '--cap', d, '--frames-of', f"{d}={fb},{os.path.join(d, 'track.json')}", '--neo', neo]
    assert ca.main(args) == 2
    # F5 edited after writing (L_m), the embedded sha256 kept: refused
    doan['doan'][0]['L_m'] = 0.9
    write_json(os.path.join(d, 'doan_12b_v1.json'), doan)
    e = write_json(str(tmp_path / 'e.json'), dict(doan_12b_sha256=doan['sha256'],
                                                  doan=[dict(i=i, e_hieu_m=0.002) for i in range(6)]))
    assert ca.main(tinh_args(d, 'K2', '--e-hieu', e)) == 2


def test_vet_thuc_touch_between_kept_frames():
    """HCC2-6 (Codex): 7 frames at 30 fps, the car goes 6 mm forward and back; the middle frame touches."""
    import c8_hc_vet as cv
    q = cv.rect(0.0, 0.05, -0.05, 0.05)
    tags = {1: q}
    reach = cv.reach('340_doc')
    x_touch = -cv.BODY_LONG / 2 - 0.0005                       # body front edge 0.5 mm past the tag edge
    xs = x_touch - 0.006 + 0.006 * np.array([0, 1 / 3, 2 / 3, 1, 2 / 3, 1 / 3, 0])
    t = np.arange(7) / 30.0
    r = cv.vet_thuc_camera(t, xs, np.zeros(7), np.zeros(7), 0.0, tags, 0.0, hyps=('340_doc',), every=1)
    assert r['khe_m'] <= 0.0
    assert reach > 0


def test_f5_roles_of_shared_window(tmp_path):
    """HCC2-7 (Codex): K1 'all' labelled the end of the last short 12b leg E_dau; and khoa needs every source."""
    d, _ = make(tmp_path, 'HC_K1_1', cho='K1')
    _, doan = run_cua_so(d)
    last_short = [x for x in doan['doan'] if not x['dai']][-1]
    assert last_short['cua_so_b']['loai'] == 'dung_12b' and doan['trang_thai'] == 'chan_doan'


def test_q_turn_term_needs_sourced_two_end_bound(tmp_path):
    """HCC2-8 (Codex): with a stable bias inside each window the half-split spread is 0 and u_q became 0."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2', noise_psi_deg=0.0)
    an = analyse(d)
    r0 = ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0.02, 0.0], du_m=0.0, neo=0.0, psi_cuc=0.0)
    assert any('u_psi_hai_dau' in x for x in r0['ly_do']) and r0['trang_thai'] != 'khoa'
    r1 = ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0.02, 0.0], du_m=0.0, neo=0.0, psi_cuc=0.0,
                         psi_hai_dau=math.radians(0.5))
    assert r1['u_hang_deg']['u_q'] >= 0.019                    # |q| 2 sin(0.25 deg) / 0.50 m ~ 0.020 deg


# ------------------------------------------------------------------ Codex code review round 3 (HCC3-3..6)
def test_falling_edge_with_500ms_hole_fails(tmp_path):
    """HCC3-3 (Codex): 500 ms inserted at the first run_active=0 row (log_seq kept continuous) still passed."""
    d, _ = make(tmp_path, cho='K2')

    def fn(hdr, body):
        i0, i1 = run_rows(hdr, body)
        ct, ca_ = col(hdr, 'timestamp_us'), col(hdr, 'traj_age_ms')
        for r in body[i1 + 1:]:
            r[ct] = str(int(r[ct]) + 500000)
            r[ca_] = str(int(r[ca_]) + 500)
    edit_log(d, fn)
    ok, ck, _ = checks(d)
    assert not ok['het_han_firmware'], ck


def test_diagnosis_not_blocked_by_old_f2_f5(tmp_path):
    """HCC3-4 (Codex): old F2/F5 blocked a diagnostic tinh that uses neither F6 nor F9."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    run_cua_so(d)
    fb = os.path.join(d, 'frames_B.csv')
    import shutil
    shutil.copy(os.path.join(d, 'frames.csv'), fb)
    with open(fb, 'a', encoding='utf-8') as f:
        f.write('')
    with open(fb, 'ab') as f:
        f.write(b'\n')
    args = ['tinh', '--cho', 'K2', '--cap', d, '--frames-of', f"{d}={fb},{os.path.join(d, 'track.json')}"]
    assert ca.main(args) in (0, 1)


def test_c8b_bound_outside_domain_refused(tmp_path):
    """HCC3-6 (Codex): -1, NaN and 2 pi rad as the two-end bound gave khoa with u_q ~ 0."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    an = analyse(d)
    for v in (-1.0, float('nan'), 2 * math.pi):
        with pytest.raises(ca.AError):
            ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0.02, 0.0], du_m=0.0, neo=0.0, psi_cuc=0.0,
                            psi_hai_dau=v)
        p = write_json(str(tmp_path / 'c8b.json'), dict(u_psi_cuc_rad=0.001, u_psi_hai_dau_rad=v))
        assert ca.main(tinh_args(d, 'K2', '--can-c8b', p)) == 2


def test_cua_so_refuses_clip_other_than_track(tmp_path):
    """HCC2-5 (round 3, Codex): the track declares clip A, cua_so --clip B was accepted."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    write_json(os.path.join(d, 'track.json'), dict(tag_offset=[0.0, 0.0, 0.0], video_sha256='a' * 64))
    rc = ca.main(['cua_so', d, '--frames', os.path.join(d, 'frames.csv'), '--track', os.path.join(d, 'track.json'),
                  '--clip', os.path.join(d, 'lich.csv')])
    assert rc == 2


# ------------------------------------------------------------------ Codex code review round 4 (HCC4-4)
@pytest.mark.parametrize('dt_us,ok_expected', [(0, False), (-1, False), (30000, True)])
def test_falling_edge_dt_integer_us(tmp_path, dt_us, ok_expected):
    """HCC4-4 (Codex): dt 0 or -1 us across the edge passed; exactly 30000 us failed by float rounding."""
    d, _ = make(tmp_path, cho='K2')

    def fn(hdr, body):
        i0, i1 = run_rows(hdr, body)
        ct, ca_ = col(hdr, 'timestamp_us'), col(hdr, 'traj_age_ms')
        shift = int(body[i1][ct]) + dt_us - int(body[i1 + 1][ct])
        a_shift = (int(body[i1][ca_]) + dt_us // 1000) - int(body[i1 + 1][ca_]) if dt_us > 0 else 3 - (
            int(body[i1 + 1][ca_]) - int(body[i1][ca_]))
        for r in body[i1 + 1:]:
            r[ct] = str(int(r[ct]) + shift)
            r[ca_] = str(max(200, int(r[ca_]) + a_shift))
    edit_log(d, fn)
    ok, ck, _ = checks(d)
    assert ok['het_han_firmware'] == ok_expected, [c for c in ck if c[0] == 'het_han_firmware']


def _fake_ck(monkeypatch, rows=None, exc=None, a6_exc=None):
    """A stand-in c7_chuoi_khung (CAM) with the F1 row keys of its 19/11 version."""
    import types
    m = types.ModuleType('c7_chuoi_khung')
    seen = {}

    def doc_a6(p):
        if a6_exc:
            raise a6_exc
        return None if p is None else dict(sha256='x')

    def chuoi_khung(frames_csv, a6, xoay, lan_lap=None, track=None, cho=None):
        seen['track'], seen['cho'] = track, cho
        if exc:
            raise exc
        return rows
    m.doc_a6, m.chuoi_khung = doc_a6, chuoi_khung
    monkeypatch.setitem(sys.modules, 'c7_chuoi_khung', m)
    return seen


def _row(i, cong=True, a6=True, q=True):
    ok = cong and a6 and q
    return dict(frame_id=i, t_cam=0.1 * i, p_x=0.01 * i if ok else float('nan'), p_y=0.0,
                psi_ma_p=0.0 if a6 else float('nan'), cong_ok=cong, a6_ok=a6, q_ok=q,
                trang_thai='khoa' if ok else 'chan_doan')


def test_frames_chain_without_a6_or_q_falls_back(monkeypatch):
    # buoi 2 before xoay_v1 / A6: no q (xoay without a6_sha256) -> None, not NaN centres marked ok
    _fake_ck(monkeypatch, rows=[_row(i, q=False) for i in range(5)])
    assert ca.frames_chain('f.csv', 'a6.json', 'xoay.json') is None
    _fake_ck(monkeypatch, rows=[_row(i, a6=False, q=False) for i in range(5)])
    assert ca.frames_chain('f.csv', 'a6.json', 'xoay.json') is None
    _fake_ck(monkeypatch, exc=ValueError('track.json tag_offset'))
    assert ca.frames_chain('f.csv', 'a6.json', 'xoay.json') is None
    _fake_ck(monkeypatch, a6_exc=ImportError('c8b_a6'))
    assert ca.frames_chain('f.csv', 'a6.json', 'xoay.json') is None


def test_frames_chain_uses_only_full_frames_and_passes_track(monkeypatch):
    rows = [_row(0), _row(1, cong=False, a6=False, q=False), _row(2), _row(3, q=False)]
    seen = _fake_ck(monkeypatch, rows=rows)
    F = ca.frames_chain('f.csv', 'a6.json', 'xoay.json', track='track.json')
    assert seen['track'] == 'track.json'
    assert F['ok'].tolist() == [True, False, True, False] and F['chuoi'] == 'chan_doan'
    assert np.all(np.isfinite(F['x'][F['ok']]))
    _fake_ck(monkeypatch, rows=[_row(0), _row(1, cong=False, a6=False, q=False), _row(2)])
    assert ca.frames_chain('f.csv', 'a6.json', 'xoay.json')['chuoi'] == 'khoa'


def test_frames_of_without_xoay_does_not_call_chain(tmp_path, monkeypatch):
    seen = _fake_ck(monkeypatch, exc=AssertionError('chain called'))
    d, _ = make(tmp_path)

    class A:
        frames, track, a6, xoay = os.path.join(d, 'frames.csv'), os.path.join(d, 'track.json'), None, None
    F = ca.frames_of(A, None)
    assert F['chuoi'] == 'chan_doan' and 'track' not in seen


def test_frames_chain_passes_cho_and_checks_pts(monkeypatch):
    """HCC7-6 / CT2-10: the place is passed to CAM (xoay_v1 per place). HCC7-7: a frame with PTS NaN is not used
    and the chain is not 'khoa'; PTS not increasing is not 'khoa' either."""
    seen = _fake_ck(monkeypatch, rows=[_row(0), _row(1), _row(2)])
    F = ca.frames_chain('f.csv', 'a6.json', 'xoay.json', cho='K1')
    assert seen['cho'] == 'K1' and F['chuoi'] == 'khoa'
    rows = [_row(0), _row(1), _row(2)]
    rows[1]['t_cam'] = float('nan')
    _fake_ck(monkeypatch, rows=rows)
    F = ca.frames_chain('f.csv', 'a6.json', 'xoay.json', cho='K1')
    assert F['ok'].tolist() == [True, False, True] and F['chuoi'] == 'chan_doan'
    rows = [_row(0), _row(1), _row(2)]
    rows[2]['t_cam'] = 0.05
    _fake_ck(monkeypatch, rows=rows)
    assert ca.frames_chain('f.csv', 'a6.json', 'xoay.json', cho='K1')['chuoi'] == 'chan_doan'


def test_frames_basic_needs_tag_offset_zero_present(tmp_path):
    """HCC7-8 (Codex): track.json without tag_offset was read as offset 0 (psi may not be the tag heading)."""
    d, _ = make(tmp_path, cho='K2')
    with open(os.path.join(d, 'track.json'), 'w') as f:
        json.dump({}, f)
    with pytest.raises(ca.AError):
        ca.frames_basic(os.path.join(d, 'frames.csv'), os.path.join(d, 'track.json'))


def test_read_e_hieu_refuses_diagnostic_f9(tmp_path):
    """KCJ1-7: F9 'chan_doan' (KIEMCHEO writes e_hieu_m_chan_doan, no e_hieu_m) never unlocks u_map."""
    body = dict(schema='c8_hc_doan_12b_v1', doan=[])
    body['sha256'] = __import__('hashlib').sha256(json.dumps(body, ensure_ascii=False, sort_keys=True)
                                                  .encode()).hexdigest()
    dp = write_json(str(tmp_path / 'doan_12b_v1.json'), body)
    ok = write_json(str(tmp_path / 'e_ok.json'), dict(doan_12b_sha256=body['sha256'], trang_thai='dat',
                                                       doan=[dict(i=0, e_hieu_m=0.002)]))
    assert ca.read_e_hieu(ok, dp)[0] == {0: 0.002}
    for bad in (dict(trang_thai='chan_doan', doan=[dict(i=0, e_hieu_m_chan_doan=0.002)]),
                dict(doan=[dict(i=0, e_hieu_m_chan_doan=0.002)]),
                dict(trang_thai='chan_doan', doan=[dict(i=0, e_hieu_m=0.002)])):
        p = write_json(str(tmp_path / 'e_bad.json'), dict(doan_12b_sha256=body['sha256'], **bad))
        with pytest.raises(ca.AError):
            ca.read_e_hieu(p, dp)


def test_read_neo_khung_chung_as_nhactam(tmp_path):
    """NHACTAM 19/11 (via CODE): khung_chung_sha256 is the sha of the common set (anchor gate & khung_ma_hop_le &
    A6), not khung_sha256 / khung_ma_hop_le_sha256 of the window: the old check always refused (code 2).
    Codex HCG1-1..6: two 12b runs at the place, another run, frames_sha of the window, file trang_thai, ID types,
    malformed files."""
    import hashlib
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    ref, _ = run_cua_so(d)
    w = lambda name, obj: write_json(str(tmp_path / f'{name}.json'), obj)                  # noqa: E731
    assert ca.read_neo(w('ok', f6(ref)), 'K2', ref) == 12.0
    assert ca.read_neo(w('hai_luot', f6(ref, n_luot=2)), 'K2', ref) == 12.0               # HCG1-1
    outside = max(ref['khung_ma_hop_le']) + 100000
    ids = ref['khung_ma_hop_le'][1:-1]
    frac = f6(ref)
    frac['cua_so'][0]['khung_chung']['id'][0] = ids[0] + 0.75                                # HCG1-5
    dup = f6(ref, ids=ids + [ids[0]])
    other = f6(ref)                                                                          # HCG1-2: both ends moved
    rec = other['chuan_12b'].pop(f"K2|{ref['luot_id']}")
    rec['luot_id'] = 'HC_K2_9'
    other['chuan_12b']['K2|HC_K2_9'] = rec
    other['cho']['K2'] = rec
    other['cua_so'][0]['meta']['luot'] = 'HC_K2_9'
    no_luot = f6(ref)
    no_luot['chuan_12b'] = {}
    del no_luot['cho']['K2']['luot_id']
    null_win = f6(ref)
    null_win['cua_so'].insert(0, None)                                                       # HCG1-6
    bad = {
        'chua_ket_luan': f6(ref, trang_thai='chua_ket_luan'),
        'toan_file': f6(ref, top='chua_ket_luan'),                                           # HCG1-4
        'frames': f6(ref, frames_sha256='2' * 64),
        'frames_cua_so': f6(ref, win=dict(frames_sha='2' * 64)),                             # HCG1-3
        'frames_cua_so_thieu': f6(ref, win=dict(frames_sha=None)),
        'ngoai_tap': f6(ref, ids=ref['khung_ma_hop_le'][1:3] + [outside]),
        'sha_sai': f6(ref, khung_chung_sha256='3' * 64),
        'khong_cua_so': dict(f6(ref), cua_so=[]),
        'luot_khac': other,
        'khong_luot': no_luot,
        'id_le': frac,
        'id_lap': dup,
        'psi_tran': f6(ref, psi_chuan_k_deg=10 ** 400),
        'psi_nan': f6(ref, psi_chuan_k_deg=float('nan')),
    }
    for name, obj in bad.items():
        with pytest.raises(ca.AError):
            ca.read_neo(w(name, obj), 'K2', ref)
    mixed = f6(ref)                                                                          # HCG2-1
    mixed['cho']['K2'] = dict(mixed['cho']['K2'], frames_sha256='4' * 64)
    twin = f6(ref)                                                                           # HCG2-2
    twin['cua_so'].append(dict(twin['cua_so'][0], meta=dict(twin['cua_so'][0]['meta'], frames_sha='5' * 64)))
    deep = '[' * 3000 + ']' * 3000
    for name, obj in (('null_win', null_win), ('mixed', mixed), ('twin', twin)):            # HCG2-3 null entry
        with pytest.raises(ca.AError):
            ca.read_neo(w(name, obj), 'K2', ref)
    p = str(tmp_path / 'deep.json')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(deep)
    with pytest.raises(ca.AError):
        ca.read_neo(p, 'K2', ref)


def test_can_hai_dau_function_feeds_u_q_and_u_psi_cuc(tmp_path):
    """Design v4 3.3 with c8b_a6.u_psi_hai_dau_rad (C8B, CODE 20/11): the same two-end bound gives u_psi_cuc
    (start window of each long leg vs the reference window) and the end-difference term of u_q (both ends of each
    leg); a missing bound keeps the result chan_doan with the reasons of C8B, never a bound of 0."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    an = analyse(d)
    calls = []

    def fn(p1, p2):
        calls.append((tuple(p1), tuple(p2)))
        return math.radians(0.05), []
    r = ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0.004, -0.003], du_m=0.0, neo=0.0, can_hai_dau=fn)
    assert r['u_hang_deg']['u_psi_cuc'] == pytest.approx(0.05)
    assert not any('thieu can' in w or 'can C8B' in w for w in r['ly_do'])
    p_ref = tuple(an['windows'][0]['p_tag'])
    assert len(calls) == 8 and sum(c[1] == p_ref for c in calls) >= 4
    r2 = ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0.004, -0.003], du_m=0.0, neo=0.0,
                         can_hai_dau=lambda p1, p2: (None, ['p1: A3 missing in the camera file']))
    assert r2['trang_thai'] != 'khoa' and r2['u_hang_deg']['u_psi_cuc'] is None and r2['u_a_k_deg'] is None
    assert r2['u_hang_deg']['u_q'] is None and set(r2['u_hang_thieu']) == {'u_q', 'u_psi_cuc'}
    assert any('A3 missing' in w for w in r2['ly_do']) and any('thieu can huong camera' in w for w in r2['ly_do'])
    assert any('u_psi_hai_dau_rad' in w for w in r2['ly_do'])
    r3 = ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0.0, 0.0], du_m=0.0, neo=0.0,
                         can_hai_dau=lambda p1, p2: (4.0, []))                  # HCH1-4: missing, with reason
    assert r3['u_a_k_deg'] is None and any('ngoai mien' in w for w in r3['ly_do'])
    seen = []
    r4 = ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0.004, 0.0], du_m=0.0, neo=0.0, bao_q=True,
                         can_hai_dau=lambda p1, p2: (seen.append(p2) or math.radians(0.05), []))   # HCH1-7
    assert len(seen) == 4 and r4['u_a_k_deg'] is not None


def test_can_hai_dau_c8b_wrapper(tmp_path, monkeypatch):
    """Interface of C8B 2539ee8 (CODE 20/11): kc = kiem_cheo.json passed through, A3 checked inside C8B (no gate here),
    luot = dai = None; --kc, --tam-id need --a6; c8b_a6 not merged -> no bound (reason kept); parts checked for both
    ways C8B combines them."""
    import types
    kc = write_json(str(tmp_path / 'kc.json'), dict(version='c7_kiem_cheo_v7', A3=dict(quy_ve_ma_noc=True)))
    a6 = write_json(str(tmp_path / 'a6.json'), dict(schema='c8b_a6_v2'))
    with pytest.raises(ca.AError):
        ca.can_hai_dau_c8b(None, kc, 'T1', [])
    monkeypatch.setitem(sys.modules, 'c8b_a6', None)                       # import fails (not merged)
    assert ca.can_hai_dau_c8b(a6, kc, 'T1', [])((0, 0), (1, 0)) == (None, ['c8b_a6 chua gop'])
    m = types.ModuleType('c8b_a6')
    m.__file__ = a6
    m.load_a6 = lambda p: dict(tam_id='T1')
    got = {}

    def f(a6o, kc_, p1, p2, tam, luot=None, dai=None):
        got.update(kc=kc_, luot=luot, dai=dai)
        if tam != a6o['tam_id']:
            return dict(u_rad=None, trang_thai='thieu', ly_do=['khac tam'])
        return dict(u_rad=0.001, trang_thai='dat', ly_do=[], ghep='cong_rieng',
                    phan=dict(A3_rad=[0.0003, 0.0003], A6du_rad=[0.0002, 0.0002]))
    m.u_psi_hai_dau_rad = f
    monkeypatch.setitem(sys.modules, 'c8b_a6', m)
    nguon = []
    assert ca.can_hai_dau_c8b(a6, kc, 'T1', nguon)((0, 0), (1, 0)) == (0.001, [])
    assert got['kc'] == json.load(open(kc, encoding='utf-8')) and got['luot'] is None and got['dai'] is None
    assert {n['vai_tro'] for n in nguon} == {'kiem_cheo', 'c8b_a6_ham'}
    assert ca.can_hai_dau_c8b(a6, kc, 'T2', [])((0, 0), (1, 0)) == (None, ['khac tam'])
    with pytest.raises(ca.AError):
        ca.can_hai_dau_c8b(a6, '', '', [])                                                # HCH1-8
    m.u_psi_hai_dau_rad = lambda *a, **k: 1 / 0
    assert ca.can_hai_dau_c8b(a6, kc, 'T1', [])((0, 0), (1, 0))[0] is None                 # HCH1-6
    bad = [dict(ghep='cong_rieng', phan=dict(A3_rad=[0.001, 0.001], A6du_rad=[-0.0003, -0.0002])),   # HCH2-5
           dict(ghep='cong_rieng', phan=dict(A3_rad=[0.001, 0.001], A6du_rad=[0.0002, 0.0002])),     # sum off
           dict(ghep='replay_chung_theo_ca', phan=dict(ca_rad=-0.001, A6du_khong_meo_rad=[0.001, 0.0015])),
           dict(ghep='khac', phan={})]
    for b in bad:
        m.u_psi_hai_dau_rad = lambda *a, b=b, **k: dict(u_rad=0.0015, trang_thai='dat', ly_do=[], **b)
        assert ca.can_hai_dau_c8b(a6, kc, 'T1', [])((0, 0), (1, 0))[0] is None
    m.u_psi_hai_dau_rad = lambda *a, **k: dict(u_rad=0.0035, trang_thai='dat', ly_do=[], ghep='replay_chung_theo_ca',
                                               phan=dict(ca_rad=0.001, A6du_khong_meo_rad=[0.001, 0.0015]))
    assert ca.can_hai_dau_c8b(a6, kc, 'T1', [])((0, 0), (1, 0)) == (0.0035, [])


def test_can_hai_dau_round2(tmp_path):
    """HCH2-3: the F9 replay (bao_q) covers the issued run only: legs of another run keep the q turn term.
    HCH2-4: the reference end is the tag centre given for the F6 frame set. HCH2-6: F7 legs keep p_a/p_b as before."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    an = analyse(d)
    d2, _ = make(tmp_path, 'HC_K2_2', cho='K2')
    an2 = analyse(d2)
    seen = []
    u = math.radians(0.05)

    def fn(p1, p2):
        seen.append((tuple(p1), tuple(p2)))
        return u, []
    r = ca.place_result('K2', [('x', an), ('y', an2)], 'x', e_hieu=E(0.0), q=[0.02, 0.0], du_m=0.0, neo=0.0,
                        bao_q=True, can_hai_dau=fn, p_ref_tag=np.array([0.123, 0.456]))
    ends = [c for c in seen if c[1] != (0.123, 0.456)]
    refs = [c for c in seen if c[1] == (0.123, 0.456)]
    assert len(ends) == 4 and len(refs) == 8                         # two-end calls only for the other run
    assert r['u_hang_deg']['u_q'] > 0.0 and r['u_hang_deg']['u_psi_cuc'] == pytest.approx(0.05)
    leg = r['luot_thu'][0]['doan'][0]
    assert 'p_tag_a' not in leg and leg['p_a'] == pytest.approx(an['legs'][0]['p_a'].tolist())


def test_tinh_prints_null_u_a(tmp_path, monkeypatch, capsys):
    """HCH2-1 (Codex): u_a null (bound missing) made the CLI print fail after the JSON was written."""
    import types
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    kc = write_json(str(tmp_path / 'kc.json'), dict(A3=dict(max_deg=0.02, quy_ve_ma_noc=True, du_lieu_du=True)))
    a6 = write_json(str(tmp_path / 'a6.json'), dict(schema='c8b_a6_v2', tam_id='T1'))
    m = types.ModuleType('c8b_a6')
    m.__file__ = a6
    m.load_a6 = lambda p: dict(tam_id='T1')
    m.u_psi_hai_dau_rad = lambda *a, **k: dict(u_rad=None, trang_thai='thieu', ly_do=['A6 thieu'])
    monkeypatch.setitem(sys.modules, 'c8b_a6', m)
    out = str(tmp_path / 'f7.json')
    code = ca.main(tinh_args(d, 'K2', '--a6', a6, '--kc', kc, '--tam-id', 'T1', '--out', out))
    txt = capsys.readouterr().out
    assert code in (0, 1) and 'u_a = null' in txt
    assert json.load(open(out, encoding='utf-8'))['u_a_k_deg'] is None


def test_a6_tam_id(tmp_path):
    """HCV3-1 (CODE 20/11): the plate is c8b_a6_v2.tam_id of the A6 file; nothing else counts as a plate."""
    w = lambda name, obj: write_json(str(tmp_path / f'{name}.json'), obj)                  # noqa: E731
    assert ca.a6_tam_id(w('ok', dict(schema='c8b_a6_v2', tam_id='T1'))) == 'T1'
    for name, obj in (('none', dict(schema='c8b_a6_v2')), ('empty', dict(tam_id='')), ('num', dict(tam_id=1)),
                      ('list', [dict(tam_id='T1')])):
        assert ca.a6_tam_id(w(name, obj)) is None, name
    assert ca.a6_tam_id(None) is None and ca.a6_tam_id(str(tmp_path / 'missing.json')) is None
    bad = tmp_path / 'bad.json'
    bad.write_text('{not json', encoding='utf-8')
    assert ca.a6_tam_id(str(bad)) is None


def test_frames_chain_passes_a6_plate_when_no_lan_lap(tmp_path, monkeypatch):
    """HCV3-1: without lan_lap, frames_chain passes tam_id = c8b_a6_v2.tam_id to CAM chuoi_khung once CAM has the
    parameter; with lan_lap it does not (CAM takes the plate from lan_lap); an older CAM without the parameter gets the
    old call (no tam_id keyword, so no TypeError)."""
    import types
    a6 = write_json(str(tmp_path / 'a6.json'), dict(schema='c8b_a6_v2', tam_id='T1'))
    m = types.ModuleType('c7_chuoi_khung')
    seen = {}

    def chuoi_khung(frames_csv, a6_, xoay, lan_lap=None, track=None, cho=None, tam_id=None):
        seen.update(tam_id=tam_id, lan_lap=lan_lap, cho=cho)
        return [_row(0), _row(1), _row(2)]
    m.doc_a6, m.chuoi_khung = (lambda p: dict(sha256='x')), chuoi_khung
    monkeypatch.setitem(sys.modules, 'c7_chuoi_khung', m)
    F = ca.frames_chain('f.csv', a6, 'xoay.json', cho='K2')
    assert seen == dict(tam_id='T1', lan_lap=None, cho='K2') and F['chuoi'] == 'khoa'
    seen.clear()
    ca.frames_chain('f.csv', a6, 'xoay.json', lan_lap=dict(_sha256='x', tam_id='T2'), cho='K2')
    assert seen['tam_id'] is None and seen['lan_lap']['tam_id'] == 'T2'
    seen.clear()
    ca.frames_chain('f.csv', str(tmp_path / 'missing.json'), 'xoay.json')
    assert 'tam_id' in seen and seen['tam_id'] is None                     # no plate: CAM's A6 check refuses
    old = _fake_ck(monkeypatch, rows=[_row(0), _row(1)])                    # CAM 3625211: no tam_id parameter
    assert ca.frames_chain('f.csv', a6, 'xoay.json', cho='K1')['chuoi'] == 'khoa' and old['cho'] == 'K1'


def test_tinh_plate_from_a6(tmp_path, monkeypatch, capsys):
    """HCV3-1: tinh takes the plate for c8b_a6.u_psi_hai_dau_rad from c8b_a6_v2.tam_id; --tam-id is optional and must
    equal it; an A6 file without tam_id cannot give the two-end bound (code 2)."""
    import types
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    kc = write_json(str(tmp_path / 'kc.json'), dict(A3=dict(quy_ve_ma_noc=True)))
    a6 = write_json(str(tmp_path / 'a6.json'), dict(schema='c8b_a6_v2', tam_id='T1'))
    a6_no = write_json(str(tmp_path / 'a6_no.json'), dict(schema='c8b_a6_v2'))
    m = types.ModuleType('c8b_a6')
    m.__file__ = a6
    m.load_a6 = lambda p: json.load(open(p, encoding='utf-8'))
    tams = []
    m.u_psi_hai_dau_rad = lambda a6o, kc_, p1, p2, tam, **k: (tams.append(tam) or
                                                               dict(u_rad=None, trang_thai='thieu', ly_do=['x']))
    monkeypatch.setitem(sys.modules, 'c8b_a6', m)
    out = lambda n: str(tmp_path / f'f7_{n}.json')                                           # noqa: E731
    assert ca.main(tinh_args(d, 'K2', '--a6', a6, '--kc', kc, '--out', out(0))) in (0, 1)
    assert tams and set(tams) == {'T1'}
    assert ca.main(tinh_args(d, 'K2', '--a6', a6, '--kc', kc, '--tam-id', 'T1', '--out', out(1))) in (0, 1)
    tams.clear()
    assert ca.main(tinh_args(d, 'K2', '--a6', a6, '--kc', kc, '--tam-id', 'T2', '--out', out(2))) == 2
    assert 'khac tam_id cua A6' in capsys.readouterr().out and not tams
    assert ca.main(tinh_args(d, 'K2', '--a6', a6_no, '--kc', kc, '--out', out(3))) == 2
    assert ca.main(tinh_args(d, 'K2', '--a6', a6_no, '--kc', kc, '--tam-id', 'T1', '--out', out(4))) == 2
    capsys.readouterr()


def test_round_v3_small_fixes(tmp_path):
    """HCV3-3: a place block without luot_id / psi but other frames, or a window with frames_sha and frames_sha256
    different, is refused. HCV3-4: F6 frames not all rebuilt -> the reference end is missing (no window-0 fallback).
    HCV3-5: the q turn flag says whether some leg carries it."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    ref, _ = run_cua_so(d)
    w = lambda name, obj: write_json(str(tmp_path / f'{name}.json'), obj)                  # noqa: E731
    stripped = f6(ref)
    blk = dict(stripped['cho']['K2'])
    blk.pop('luot_id')
    blk.pop('psi_chuan_k_deg')
    blk['frames_sha256'] = '6' * 64
    stripped['cho']['K2'] = blk
    both = f6(ref, win=dict(frames_sha256='7' * 64))
    for name, obj in (('stripped', stripped), ('both', both)):
        with pytest.raises(ca.AError):
            ca.read_neo(w(name, obj), 'K2', ref)
    an = analyse(d)
    fn = lambda p1, p2: (math.radians(0.05), [])                                            # noqa: E731
    r = ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0.0, 0.0], du_m=0.0, neo=0.0, can_hai_dau=fn,
                        p_ref_tag='thieu')
    assert r['u_hang_deg']['u_psi_cuc'] is None and any('dau chuan' in x for x in r['ly_do'])
    d2, _ = make(tmp_path, 'HC_K2_2', cho='K2')
    r2 = ca.place_result('K2', [('x', an), ('y', analyse(d2))], 'x', e_hieu=E(0.0), q=[0.02, 0.0], du_m=0.0, neo=0.0,
                         bao_q=True, can_hai_dau=fn)
    assert r2['u_q_gom_xoay_q'] is True and r2['u_q_bao_boi_f9'] == ['x']
    r3 = ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0.02, 0.0], du_m=0.0, neo=0.0, bao_q=True,
                         can_hai_dau=fn)
    assert r3['u_q_gom_xoay_q'] is False


# ------------------------------------------------------------------ F5 lan_lap_id from F8 (CODE 20/11, T9 / T9b)
def _f8_setup(tmp_path, monkeypatch, cho='K2'):
    """Run 1 of cua_so (no --a6), then the files of run 2: A6 (tam T1), xoay, session, dets; the chain is made
    'khoa' and the dets fingerprint present, so F5 'khoa' depends only on the F8 link."""
    d, _ = make(tmp_path, f'HC_{cho}_1', cho=cho)
    run_cua_so(d)
    f2 = os.path.join(d, 'cua_so_v1.jsonl')
    w = lambda name, obj: write_json(str(tmp_path / f'{name}.json'), obj)                  # noqa: E731
    a6 = w('a6', dict(schema='c8b_a6_v2', tam_id='T1'))
    xoay = w('xoay', _x_v1(ca.sha256(a6), **{cho: {}}))                       # HCL1-2: F5 khoa needs q of the F4
    ses = w('session', dict(fit_ids=[1, 2], car_side_m=0.3, tag_h_m=0.2))
    dets = w('dets', dict(x=1))
    monkeypatch.setattr(ca, 'dets_fingerprint', lambda p: 'fp' if p else None)
    monkeypatch.setattr(ca, 'frames_chain', lambda fr, a6_, xo, track=None, cho=None, **k:
                        dict(ca.frames_basic(fr, track), chuoi='khoa'))

    def f8(name, luot, **top):
        o = dict(schema='lan_lap_v1', tam_id='T1', a6=dict(sha256=ca.sha256(a6)), cua_so_v1_file=f2,
                 cua_so_v1_sha256=ca.sha256(f2), luot=luot)
        o.update(top)
        return w(name, o)

    def run2(ll, ra, extra=()):
        args = ['cua_so', d, '--frames', os.path.join(d, 'frames.csv'), '--track', os.path.join(d, 'track.json'),
                '--clip', os.path.join(d, 'track.json'), '--a6', a6, '--xoay', xoay, '--session', ses, '--dets', dets,
                '--ra', str(tmp_path / ra), *extra] + (['--lan-lap', ll] if ll else [])
        rc = ca.main(args)
        p = tmp_path / ra / 'doan_12b_v1.json'
        return rc, (json.load(open(p, encoding='utf-8')) if p.exists() else None)
    return d, f2, a6, f8, run2


def _rec(luot_id, cho, lan_lap_id, chuan_12b, tam='T1'):
    return dict(schema='lan_lap_v1', luot_id=luot_id, cho=cho, tam_id=tam, lan_lap_id=lan_lap_id, chuan_12b=chuan_12b)


def test_cua_so_run2_lan_lap_from_f8(tmp_path, monkeypatch):
    """T9b: run 2 of cua_so (--a6 --xoay --lan-lap, new folder) takes lan_lap_id of this 12b from the F8 records of
    the same place verified against it (chuan_12b); F5 'khoa' only then; run 1 files untouched."""
    d, f2, a6, f8, run2 = _f8_setup(tmp_path, monkeypatch)
    sha_before = (ca.sha256(f2), ca.sha256(os.path.join(d, 'doan_12b_v1.json')))
    ll = f8('ll', [_rec('C8_1', 'K2', 'L0', 'HC_K2_1'), _rec('C8_2', 'K2', 'L0', 'HC_K2_1'),
                   _rec('C8_3', 'K1', 'L1', 'HC_K1_9'), _rec('C8_4', 'K2', None, None)])
    rc, f5 = run2(ll, 'r2')
    assert rc == 0 and f5['lan_lap_id'] == 'L0' and f5['trang_thai'] == 'khoa' and f5['lan_lap_id_ly_do'] is None
    assert f5['lan_lap_v1_sha256'] == ca.sha256(ll) and ca.f5_problems(f5) == []
    assert any(n['vai_tro'] == 'lan_lap' and n['sha256'] == ca.sha256(ll) for n in f5['nguon'])
    assert (tmp_path / 'r2' / 'cua_so_v1.jsonl').exists()
    assert (ca.sha256(f2), ca.sha256(os.path.join(d, 'doan_12b_v1.json'))) == sha_before
    rc, f5 = run2(f8('ll_luot', [_rec('HC_K2_1', 'K2', 'L3', None)]), 'r2b')           # a record of the 12b itself
    assert rc == 0 and f5['lan_lap_id'] == 'L3' and f5['trang_thai'] == 'khoa'
    assert run2(ll, 'r2')[0] == 2                                                          # never overwritten (x)


def test_cua_so_run2_no_lan_lap_stays_chan_doan(tmp_path, monkeypatch):
    """F8 with two ids for this 12b, or no --lan-lap: F5 chan_doan with the reason. F8 read (identity checks passed)
    with no id for this 12b (F8 rong, CODE 23/11 muc 1b): the gate passes, lan_lap_id null, lan_lap_nhan labelled."""
    d, f2, a6, f8, run2 = _f8_setup(tmp_path, monkeypatch)
    cases = [('none', [_rec('C8_1', 'K2', None, None), _rec('C8_2', 'K1', 'L0', 'HC_K2_1')], 'khong co luot', True),
             ('two', [_rec('C8_1', 'K2', 'L0', 'HC_K2_1'), _rec('C8_2', 'K2', 'L1', 'HC_K2_1')], 'nhieu lan_lap_id',
              False),
             ('empty', [_rec('C8_1', 'K2', '', 'HC_K2_1')], 'khong co luot', True)]
    for name, luot, why, rong in cases:
        ll = f8(name, luot)
        rc, f5 = run2(ll, f'r_{name}')
        assert rc == 0 and f5['lan_lap_id'] is None, name
        assert why in f5['lan_lap_id_ly_do'], (name, f5['lan_lap_id_ly_do'])
        assert f5['lan_lap_v1_sha256'] == ca.sha256(ll) and ca.f5_problems(f5) == []
        if rong:
            assert f5['trang_thai'] == f5['chuoi'] == 'khoa' and f5['lan_lap_nhan'] == ca.NGUON_LECH_TAM, name
        else:
            assert f5['trang_thai'] == 'chan_doan' and f5['lan_lap_nhan'] is None, name
    rc, f5 = run2(None, 'r_no')
    assert rc == 0 and f5['trang_thai'] == 'chan_doan' and '--lan-lap' in f5['lan_lap_id_ly_do']
    assert f5['lan_lap_nhan'] is None and f5['lan_lap_v1_sha256'] is None


def test_cua_so_run2_refuses_other_f8(tmp_path, monkeypatch):
    """An F8 that is not this session's: schema, plate (file or record), A6 sha, its F2 file (missing, sha, no ref_12b
    of this attempt with the same clip and frames), no --a6 plate -> code 2 and no file written."""
    d, f2, a6, f8, run2 = _f8_setup(tmp_path, monkeypatch)
    good = [_rec('C8_1', 'K2', 'L0', 'HC_K2_1')]
    other_f2 = str(tmp_path / 'f2_other.jsonl')
    rows = [json.loads(x) for x in open(f2, encoding='utf-8')]
    with open(other_f2, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(dict(r, luot_id='HC_K2_9')) + '\n')
    other_frames = str(tmp_path / 'f2_frames.jsonl')
    with open(other_frames, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(dict(r, frames_sha256='0' * 64)) + '\n')
    bad = [('schema', f8('b1', good, schema='lan_lap_v0')),
           ('tam', f8('b2', good, tam_id='T2')),
           ('tam_rec', f8('b3', good + [_rec('C8_9', 'K1', 'L0', 'x', tam='T2')])),
           ('a6', f8('b4', good, a6=dict(sha256='1' * 64))),
           ('f2_sha', f8('b5', good, cua_so_v1_sha256='2' * 64)),
           ('f2_missing', f8('b6', good, cua_so_v1_file=str(tmp_path / 'nope.jsonl'))),
           ('f2_other', f8('b7', good, cua_so_v1_file=other_f2, cua_so_v1_sha256=ca.sha256(other_f2))),
           ('f2_frames', f8('b8', good, cua_so_v1_file=other_frames, cua_so_v1_sha256=ca.sha256(other_frames))),
           ('luot', f8('b9', None))]
    for name, p in bad:
        rc, f5 = run2(p, f'rb_{name}')
        assert rc == 2 and f5 is None and not (tmp_path / f'rb_{name}' / 'cua_so_v1.jsonl').exists(), name
    a6_no = write_json(str(tmp_path / 'a6_no.json'), dict(schema='c8b_a6_v2'))
    rc = ca.main(['cua_so', d, '--frames', os.path.join(d, 'frames.csv'), '--track', os.path.join(d, 'track.json'),
                  '--a6', a6_no, '--lan-lap', f8('b10', good), '--ra', str(tmp_path / 'rb_a6no')])
    assert rc == 2
    # A6 and F8 both without a plate: None == None must not pass (the A6 plate is required on its own)
    no_tam = f8('b11', [_rec('C8_1', 'K2', 'L0', 'HC_K2_1', tam=None)], tam_id=None, a6=dict(sha256=ca.sha256(a6_no)))
    rc = ca.main(['cua_so', d, '--frames', os.path.join(d, 'frames.csv'), '--track', os.path.join(d, 'track.json'),
                  '--clip', os.path.join(d, 'track.json'), '--a6', a6_no, '--lan-lap', no_tam,
                  '--ra', str(tmp_path / 'rb_notam')])
    assert rc == 2 and not (tmp_path / 'rb_notam' / 'doan_12b_v1.json').exists()


def test_tinh_doan_of_run2_carries_lan_lap_id(tmp_path, monkeypatch, capsys):
    """T9b: tinh --e-hieu --doan reads the F5 of run 2 (its folder), checks it is the F5 of the issued attempt and
    writes its lan_lap_id in F7 mien_hieu_luc; another attempt's F5 is refused; tinh without the A6, xoay files that
    F5 was made from is refused (cap_f2_f5: the F5 of --doan is the one checked)."""
    d, f2, a6, f8, run2 = _f8_setup(tmp_path, monkeypatch)
    rc, f5 = run2(f8('ll', [_rec('C8_1', 'K2', 'L0', 'HC_K2_1')]), 'r2')
    assert rc == 0 and f5['trang_thai'] == 'khoa'
    dp = str(tmp_path / 'r2' / 'doan_12b_v1.json')
    f9 = write_json(str(tmp_path / 'f9.json'), dict(doan_12b_sha256=f5['sha256'], trang_thai='dat',
                                                    doan=[dict(i=x['i'], e_hieu_m=0.0) for x in f5['doan']]))
    out = str(tmp_path / 'f7.json')
    src = ('--a6', a6, '--xoay', str(tmp_path / 'xoay.json'))          # the sources run 2 was made from
    assert ca.main(tinh_args(d, 'K2', *src, '--e-hieu', f9, '--doan', dp, '--out', out)) in (0, 1)
    assert json.load(open(out, encoding='utf-8'))['mien_hieu_luc']['lan_lap_id'] == 'L0'
    assert ca.main(tinh_args(d, 'K2', '--e-hieu', f9, '--doan', dp, '--out', str(tmp_path / 'f7x.json'))) == 2
    assert ca.main(tinh_args(d, 'K2', '--e-hieu', f9, '--out', str(tmp_path / 'f7b.json'))) == 2   # run-1 F5: sha
    d2, _ = make(tmp_path, 'HC_K2_2', cho='K2')
    run_cua_so(d2)                                     # d2 has its own files: only the F5 attempt check refuses
    capsys.readouterr()
    assert ca.main(tinh_args(d2, 'K2', *src, '--e-hieu', f9, '--doan', dp, '--out', str(tmp_path / 'f7c.json'))) == 2
    assert 'khong phai luot cap HC_K2_2' in capsys.readouterr().out


# ------------------------------------------------------------------ session 2 (20/11) real-data fixes HCB1, HCB2
def test_lost_packets_on_turn_ramp_are_bounded(tmp_path):
    """HCB2 (HC_goc_2, 20/11): Wi-Fi lost 140 ms of packets on the down ramp of a turn; the held value added
    a age^2/2 ~ 0.0098 rad (> 0.5 deg) and traj_age 140 ms (> 60). Explained by the staleness seen in traj_age:
    accepted; the sender-domain boundary stays at 60 ms (the zero-tail span check still passes)."""
    d, _ = make(tmp_path, 'HC_K1_1', cho='K1', lost=[(86.95, 87.09)])          # turn 1 ramp down 86.82-87.32 s
    ok, ck, info = checks(d)
    det = dict((n, x) for n, _, x in ck)
    assert all(ok.values()), [c for c in ck if not c[1]]
    assert 'can cong them: xoay' in det['lenh_nhan_khop_lich'] and float(det['traj_age'].split()[1]) >= 130
    L = ca.load_attempt(d)['log']
    blk = [b for b in ca.sp_blocks(L, info['i0'], info['j_s']) if b['truc'] == 'wz'][0]
    assert abs(blk['tp_wz'] - 5 * math.pi) > ca.SP_INT_TOL_RAD                  # the old tolerance alone fails


def test_lost_packets_on_plateau_change_nothing(tmp_path):
    """HCB2: 140 ms lost while the command is constant: the held value is the planned one; only traj_age shows it."""
    d, _ = make(tmp_path, 'HC_K1_1', cho='K1', lost=[(70.0, 70.14)])
    ok, ck, _ = checks(d)
    assert all(ok.values()), [c for c in ck if not c[1]]
    assert 'can cong them' not in dict((n, x) for n, _, x in ck)['lenh_nhan_khop_lich']


def test_packets_lost_near_firmware_stop_fail(tmp_path):
    """HCB2: traj_age within two packets of the 200 ms firmware stop (lost 165 ms; the firmware did not stop) is
    refused."""
    d, _ = make(tmp_path, 'HC_K1_1', cho='K1', lost=[(70.0, 70.165)])
    ok, ck, _ = checks(d)
    assert ok['mot_canh_run_active'] and not ok['traj_age'], ck


def test_ramp_change_not_explained_by_age_fails(tmp_path):
    """HCB2: a ramp value held 200 ms while traj_age stays small (not a late packet: e.g. a wrong value) is not
    covered by the bound."""
    d, _ = make(tmp_path, 'HC_K1_1', cho='K1')

    def fn(hdr, body):
        ct, cw = col(hdr, 'timestamp_us'), col(hdr, 'sp_wz')
        t = np.array([int(r[ct]) for r in body]) / 1e6 - 100.0 - 6.0
        rows = np.flatnonzero((t >= 86.90) & (t < 87.10))                      # 0.20 s: +0.02 rad
        for i in rows:
            body[i][cw] = body[rows[0]][cw]
    edit_log(d, fn)
    ok, ck, _ = checks(d)
    assert not ok['lenh_nhan_khop_lich'], ck


def _edit_yaw(d, g):
    """yaw_rad := wrap(yaw0 + g(unwrapped yaw - yaw0, t))."""
    def fn(hdr, body):
        cy, ct = col(hdr, 'yaw_rad'), col(hdr, 'timestamp_us')
        y = np.unwrap([float(r[cy]) for r in body])
        t = np.array([int(r[ct]) for r in body]) / 1e6 - 100.0 - 6.0
        z = y[0] + g(y - y[0], t)
        for r, v in zip(body, z):
            r[cy] = f'{math.atan2(math.sin(v), math.cos(v)):.6f}'
    edit_log(d, fn)


def test_imu_turn_fit_catches_scale_and_jump(tmp_path):
    """HCB1 (session 2): the rate correlation was 0.40-0.58 on a good IMU at constant rate; the check is now the fit
    of IMU yaw on encoder heading (k, residual). A 25 % scale error or a 0.08 rad jump inside a turn fails; the
    synthetic IMU passes with k ~ 1."""
    d, _ = make(tmp_path, 'HC_K1_1', cho='K1')
    ok, ck, _ = checks(d)
    e = [x for n, _, x in ck if n == 'imu_e_tuong_quan']
    assert ok['imu_e_tuong_quan'] and len(e) == 2 and all(' k 0.9' in x or ' k 1.0' in x for x in e), e
    d2, _ = make(tmp_path, 'HC_K1_2', cho='K1')
    _edit_yaw(d2, lambda dy, t: 1.25 * dy)
    _, ck2, _ = checks(d2)
    assert [ok_ for n, ok_, _ in ck2 if n == 'imu_e_tuong_quan'] == [False, False], ck2
    d3, _ = make(tmp_path, 'HC_K1_3', cho='K1')
    _edit_yaw(d3, lambda dy, t: dy + np.where(t > 70.0, 0.08, 0.0))
    _, ck3, _ = checks(d3)
    assert [ok_ for n, ok_, _ in ck3 if n == 'imu_e_tuong_quan'][0] is False, ck3


def test_imu_turn_fit_tolerates_rv_noise_at_constant_rate(tmp_path):
    """HCB1: RV noise of 4 mrad per report (the session 2 residual was 3-4 mrad RMS) passes the fit, where the rate
    correlation of a constant-rate turn falls under 0.9."""
    rng = np.random.default_rng(7)
    d, _ = make(tmp_path, 'HC_K1_1', cho='K1')
    n_rep = {}

    def g(dy, t):
        k = np.floor(t / 0.05).astype(int)
        for x in np.unique(k):
            n_rep.setdefault(x, rng.normal(0, 0.004))
        return dy + np.array([n_rep[x] for x in k])
    _edit_yaw(d, g)
    ok, ck, _ = checks(d)
    assert ok['imu_e_tuong_quan'], [c for c in ck if c[0] == 'imu_e_tuong_quan']


def test_lost_zero_packet_at_block_end_is_bounded(tmp_path):
    """HCB2: the packets around the end of a turn lost (the zero packet late): the block ends ~0.1 s late; the
    duration check widens by the staleness seen at the last row, the integral by the ramp bound."""
    d, _ = make(tmp_path, 'HC_K1_1', cho='K1', lost=[(87.30, 87.44)])        # turn 1 ends 87.32 s
    ok, ck, info = checks(d)
    assert all(ok.values()), [c for c in ck if not c[1]]
    L = ca.load_attempt(d)['log']
    b = [b for b in ca.sp_blocks(L, info['i0'], info['j_s']) if b['truc'] == 'wz'][0]
    assert (b['t1'] - b['t0']) - (87.31592653589793 - 55.4) > ca.DUR_TOL_S          # the old tolerance alone fails


def test_plateau_staleness_does_not_cover_a_ramp_error(tmp_path):
    """HCB2: staleness counts only on the ramps: 140 ms lost on the constant part does not pay for a ramp value
    held 0.15 s with a small age."""
    d, _ = make(tmp_path, 'HC_K1_1', cho='K1', lost=[(70.0, 70.14)])

    def fn(hdr, body):
        ct, cw = col(hdr, 'timestamp_us'), col(hdr, 'sp_wz')
        t = np.array([int(r[ct]) for r in body]) / 1e6 - 100.0 - 6.0
        rows = np.flatnonzero((t >= 86.90) & (t < 87.05))
        for i in rows:
            body[i][cw] = body[rows[0]][cw]
    edit_log(d, fn)
    ok, ck, _ = checks(d)
    assert not ok['lenh_nhan_khop_lich'], ck


def test_imu_turn_fit_uses_rv_updates_only(tmp_path):
    """HCB1: the fit uses the rows where a new RV report came; between reports the yaw is held (20 Hz staircase,
    up to 25 mrad at 0.5 rad/s), which is not an IMU error: a clean synthetic IMU leaves a residual far below it."""
    d, _ = make(tmp_path, 'HC_K1_1', cho='K1')
    _, ck, _ = checks(d)
    for x in [x for n, _, x in ck if n == 'imu_e_tuong_quan']:
        assert float(x.split('du max ')[1].split()[0]) < 5.0, x


def test_one_repeated_rv_report_during_turn_fails(tmp_path):
    """HCC1-9 kept with HCB1: one RV report repeating the old yaw while the wheels turn (25 mrad, under the fit
    residual limit) still fails the check."""
    d, _ = make(tmp_path, 'HC_K1_1', cho='K1', rv_freeze=(62.0, 62.07))          # one report (62.05 s) repeats
    _, ck, _ = checks(d)
    e = [c for c in ck if c[0] == 'imu_e_tuong_quan']
    assert not e[0][1] and 'RV lap luc quay 1' in e[0][2] and float(e[0][2].split('du max ')[1].split()[0]) < 30, e


def test_ramp_bound_pays_only_its_own_staleness(tmp_path):
    """HCB2: a turn with the 140 ms ramp loss (bound ~0.008 rad) plus one wrong plateau row (+0.009 rad) is refused:
    the bound covers what the staleness explains, not more."""
    d, _ = make(tmp_path, 'HC_K1_1', cho='K1', lost=[(86.95, 87.09)])

    def fn(hdr, body):
        ct, cw = col(hdr, 'timestamp_us'), col(hdr, 'sp_wz')
        t = np.array([int(r[ct]) for r in body]) / 1e6 - 100.0 - 6.0
        i = int(np.flatnonzero(t >= 70.0)[0])
        body[i][cw] = f'{float(body[i][cw]) + 0.9:.6f}'
    edit_log(d, fn)
    ok, ck, _ = checks(d)
    assert not ok['lenh_nhan_khop_lich'], ck


# ------------------------------------------------------------------ HCB3 (CODE 20/11): IMU drift in yaw_dong
def test_hcb4_imu_drift_does_not_touch_a(tmp_path):
    """Session 2 K1/K2: the IMU yaw drifted 0.03-0.05 deg/s against the camera. HCB4: a_l uses the camera heading
    and the dynamic check takes off the straight line through the two end windows of the leg: a drift of 0.08 deg/s
    changes neither a_l nor the gate; the camera - IMU change between the leg ends is reported (diagnostic)."""
    d, truth = make(tmp_path, 'HC_K2_1', cho='K2', yaw_bias=lambda t: math.radians(0.08) * t, noise_psi_deg=0.02)
    d0, _ = make(tmp_path, 'HC_K2_0', cho='K2', noise_psi_deg=0.02)
    an, an0 = analyse(d), analyse(d0)
    longs, longs0 = [lg for lg in an['legs'] if lg['dai']], [lg for lg in an0['legs'] if lg['dai']]
    assert all(x['a_l'] == pytest.approx(y['a_l'], abs=1e-12) for x, y in zip(longs, longs0))
    assert max(lg['eps'] for lg in longs) < math.radians(0.1)
    assert all(abs(lg['cam_imu']) > math.radians(0.5) for lg in longs)          # ~0.08 deg/s x ~9.4 s
    assert max(abs(lg['cam_imu']) for lg in longs0) < math.radians(0.05)
    assert abs(an['troi_imu']['toc_do_deg_s'] - 0.08) < 0.005
    r = result(d, 'K2')
    assert r['cong']['yaw_dong'] and all(r['cong'].values())
    assert len(r['troi_imu']['cam_imu_hai_dau_deg']) == 4
    assert abs(math.radians(r['a_k_deg']) - truth['a_true']) <= math.radians(r['u_a_k_deg'])


def test_hcb4_diagonal_creep_is_compensated(tmp_path):
    """HCB4 (session 2 20/11; Liam 20/11 "Duyet cach sua (Recommended)"): the real car creeps sigma*|dx|*(-1, +1)
    in the body frame (the FL-RR diagonal), forwards and backwards alike; the encoders do not see it. Raw: the
    backward legs read a + sigma, the forward ones a - sigma (spread 2 sigma > 0.30 deg), the length ratio differs
    by 2 sigma and the stops walk sideways. Compensated: sigma is recovered from the length ratio alone, the gates
    pass, a (mean of the RAW legs) does not move, and the largest deviation of u_a still carries the raw split."""
    sig = 0.003
    d, truth = make(tmp_path, 'HC_K2_1', cho='K2', sigma=sig, noise_psi_deg=0.02)
    an = analyse(d)
    longs = [lg for lg in an['legs'] if lg['dai']]
    assert an['truot_cheo']['sigma'] == pytest.approx(sig, abs=2e-4)
    assert an['thang_hang_m']['tho'] > 2 * an['thang_hang_m']['bu']
    for lg in longs:
        want = truth['a_true'] + (sig if lg['chieu'] == 'lui' else -sig)
        assert lg['a_l_tho'] == pytest.approx(want, abs=math.radians(0.03))
        assert lg['a_l'] == pytest.approx(truth['a_true'], abs=math.radians(0.03))
        assert lg['ti_le_tho'] == pytest.approx(1 + (sig if lg['chieu'] == 'lui' else -sig), abs=3e-4)
        assert lg['ti_le'] == pytest.approx(1.0, abs=3e-4)
    shorts = [lg for lg in an['legs'] if not lg['dai']]      # the held-out legs get the same ratio (creep ~ distance)
    assert all(lg['a_l'] == pytest.approx(truth['a_true'], abs=math.radians(0.12)) for lg in shorts)
    r = result(d, 'K2')
    assert all(r['cong'].values()), r['cong']
    assert r['do_tan_tho_deg'] > 0.30 > r['do_tan_deg'] and r['truot_cheo']['sigma'] == pytest.approx(sig, abs=2e-4)
    assert r['thang_hang_mm']['bu'] < r['thang_hang_mm']['tho']
    assert abs(math.radians(r['a_k_deg']) - truth['a_true']) < math.radians(0.02)
    assert r['u_hang_deg']['lech_lon_nhat'] >= math.degrees(sig) - 0.03      # the compensation is counted as uncertain
    leg = [x for x in r['luot_thu'][0]['doan'] if x['dai']][0]
    assert {'a_l_deg', 'a_l_tho_deg', 'ti_le', 'ti_le_tho', 'eps_deg', 'eps_tb_deg', 'cam_imu_deg'} <= set(leg)
    longs[0]['ti_le_tho'] = 1.03                              # the range of ti_le still covers the RAW ratio
    assert not ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0.0, 0.0], du_m=0.0, neo=0.0,
                               psi_cuc=0.0)['cong']['ti_le']
    # goc (1 m legs): the creep walks the seven stops 7.5 mm off their line; with the predicted drift taken off
    # they are straight again (session 2 goc: 5.6 / 4.2 mm measured, 1.7 / 2.1 mm after)
    dg_, truth_g = make(tmp_path, 'HC_goc_1', cho='goc', sigma=sig, noise_psi_deg=0.02)
    rg = result(dg_, 'goc')
    assert rg['thang_hang_mm']['tho'] > 5.0 > 1.0 > rg['thang_hang_mm']['bu'] and all(rg['cong'].values()), rg['cong']
    d0, _ = make(tmp_path, 'HC_K2_2', cho='K2', noise_psi_deg=0.02)
    an0 = analyse(d0)
    assert abs(an0['truot_cheo']['sigma']) < 2e-4            # nothing to compensate: nothing changes
    assert all(abs(lg['a_l'] - lg['a_l_tho']) < math.radians(0.015) for lg in an0['legs'] if lg['dai'])


def test_hcb4_lateral_asymmetry_off_the_diagonal_still_fails(tmp_path):
    """The compensation is not a forward-backward equaliser: an even drift that is LATERAL only (not on the FL-RR
    diagonal, so the length ratio does not show it) keeps its spread and the gate fails."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2', lech_ngang=0.004, noise_psi_deg=0.02)
    an = analyse(d)
    assert abs(an['truot_cheo']['sigma']) < 3e-4
    r = result(d, 'K2')
    assert r['do_tan_deg'] > 0.40 and not r['cong']['do_tan'] and r['trang_thai'] == 'chua_ket_luan'


def test_hcb4_one_direction_missing_is_not_compensated(tmp_path):
    """sigma needs a long leg in each direction; without one the legs stay raw and the reason is written."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2', sigma=0.003, noise_psi_deg=0.02)
    an = analyse(d)
    legs = [dict(lg) for lg in an['legs']]
    for lg in legs:
        lg['a_l'], lg['ti_le'] = lg['a_l_tho'], lg['ti_le_tho']
        if lg['dai'] and lg['chieu'] == 'toi':
            lg['trang_thai'] = 'thieu'
    tc = ca.slip_compensate(legs)
    assert tc['sigma'] is None and '0 toi' in tc['ly_do']
    assert all(lg['a_l'] == lg['a_l_tho'] for lg in legs)
    th = ca.stops_straightness(an['windows'], legs, None)
    assert th['bu'] == th['tho']


def test_imu_drift_fit_needs_windows_and_leaves_out_leg_ends():
    """HCB3: 2-parameter line on >= 4 stop windows, the two ends of the leg left out and reported apart."""
    class _L(dict):
        pass
    t = np.arange(0.0, 70.0, 0.01)
    L = _L(t=t)
    W = [dict(psi=0.0, yaw=0.001 * 10 * k + (0.01 if k == 3 else 0.0), th_enc=0.0005 * k * (k % 2), jc=int(1000 * k))
         for k in range(7)]                                          # encoder noise only, no camera drift
    tr = ca.imu_drift_fit(W, L, skip=(2, 3))
    assert tr['n'] == 5 and set(tr['du_ngoai']) == {2, 3}
    assert tr['pf'][0] == pytest.approx(0.001, abs=1e-9) and tr['du_ngoai'][3] == pytest.approx(0.01, abs=1e-9)
    assert tr['du_max'] < 1e-9                                       # the step at window 3 is not fitted away
    W2 = [dict(w) for w in W]
    for k in (0, 1, 4):
        W2[k].pop('psi')                                             # no camera mean at three windows
    assert ca.imu_drift_fit(W2, L, skip=(2, 3)) is None              # 2 left < 4
    assert ca.imu_drift_fit(W2, L)['n'] == 4


def test_hcb4_u_yaw_is_the_path_mean_of_camera_minus_imu(tmp_path):
    """HCB4: a camera heading error in motion turns C by its path-weighted mean. A camera - IMU difference of
    0.2 deg over the moving part of the first long leg (K2, 9.9-15.3 s): eps (largest) ~0.2 deg < 0.30 passes the
    gate, u_yaw (path mean) is most of it, and a stays inside u_a; without the bias u_yaw is ~0."""
    d, truth = make(tmp_path, 'HC_K2_1', cho='K2', noise_psi_deg=0.02,
                    cam_bias=lambda t: math.radians(0.20) if 10.3 <= t <= 15.2 else 0.0)
    r = result(d, 'K2')
    assert r['cong']['yaw_dong'] and 0.12 <= r['u_hang_deg']['u_yaw'] <= 0.21, r['u_hang_deg']
    eps_max = max(x['eps_deg'] for x in r['luot_thu'][0]['doan'] if x['dai'])
    assert r['u_hang_deg']['u_yaw'] < eps_max - 0.005 < 0.30         # the path mean, not the largest value
    assert abs(math.radians(r['a_k_deg']) - truth['a_true']) <= math.radians(r['u_a_k_deg'])
    d0, _ = make(tmp_path, 'HC_K2_0', cho='K2', noise_psi_deg=0.02)
    assert result(d0, 'K2')['u_hang_deg']['u_yaw'] < 0.02


# ------------------------------------------------------------------ da session (CODE 20/11 khuya, contract A)
def _set_track_session(d, sha):
    p = os.path.join(d, 'track.json')
    tj = json.load(open(p, encoding='utf-8'))
    if sha is None:
        tj.pop('session_sha256', None)
    else:
        tj['session_sha256'] = sha
    write_json(p, tj)


def _tong_hang(r):
    """Sum of the terms of u_a present in F7 (deg; u_map is null without F9, HCL1-8)."""
    return sum(v for v in r['u_hang_deg'].values() if v is not None)


def _khung(tmp_path, name, pair, du=0.002, n=24, **kw):
    o = dict(schema='c7_khung_giua_session_v1', session_sha256=list(pair), du_m=du, n_moc=n,
             nguon=[dict(vai_tro='session', sha256=s) for s in pair])
    o.update(kw)
    return write_json(str(tmp_path / f'{name}.json'), o)


def _two(tmp_path, s1, s2):
    d1, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    d2, _ = make(tmp_path, 'HC_K2_2', cho='K2')
    _set_track_session(d1, s1)
    _set_track_session(d2, s2)

    def run(name, *extra):
        out = str(tmp_path / f'{name}.json')
        rc = ca.main(['tinh', '--cho', 'K2', '--cap', d1, '--them', d2, '--frames-of',
                      f"{d1}={os.path.join(d1, 'frames.csv')},{os.path.join(d1, 'track.json')}",
                      f"{d2}={os.path.join(d2, 'frames.csv')},{os.path.join(d2, 'track.json')}", '--out', out, *extra])
        return rc, (json.load(open(out, encoding='utf-8')) if os.path.isfile(out) else None)
    return d1, d2, run


def test_tinh_same_session_adds_nothing_and_records_sessions(tmp_path):
    A_, B_ = 'a' * 64, 'b' * 64
    d1, d2, run = _two(tmp_path, A_, A_)
    rc, r = run('same')
    assert rc in (0, 1) and r['session_sha256'] == A_
    assert r['session_sha256_luot'] == {'HC_K2_1': A_, 'HC_K2_2': A_}
    assert [x['session_sha256'] for x in r['luot_thu']] == [A_, A_]
    k = r['khung_giua_session']
    assert k['luot_khac_session'] == [] and k['u_deg'] == 0.0 and k['L_m'] is None and k['khong_ro'] == []
    assert r['u_hang_deg']['u_khung_giua_session'] == 0.0 and 'khung_giua_session' not in r['cong']
    # a pair file for an unrelated pair changes nothing
    rc2, r2 = run('same_k', '--khung-giua-session', _khung(tmp_path, 'kab', (A_, B_)))
    assert r2['u_a_k_deg'] == r['u_a_k_deg'] and r2['khung_giua_session']['cap'] == []


def test_tinh_other_session_needs_pair_file_and_adds_u(tmp_path):
    A_, B_, C_ = 'a' * 64, 'b' * 64, 'c' * 64
    d1, d2, run = _two(tmp_path, A_, A_)
    _, same = run('same')
    _set_track_session(d2, B_)
    rc, r = run('miss')                                   # no pair file: chua_ket_luan, term and sum null
    assert rc == 1 and r['trang_thai'] == 'chua_ket_luan' and r['cong']['khung_giua_session'] is False
    assert r['u_a_k_deg'] is None and r['u_hang_deg']['u_khung_giua_session'] is None
    assert 'u_khung_giua_session' in r['u_hang_thieu']
    assert any('thieu c7_khung_giua_session_v1' in w and 'HC_K2_2' in w for w in r['ly_do'])
    assert r['khung_giua_session']['luot_khac_session'] == ['HC_K2_2'] and r['khung_giua_session']['cap_thieu'] == [B_]
    rc, r = run('wrong', '--khung-giua-session', _khung(tmp_path, 'kac', (A_, C_)))   # file of another pair
    assert r['cong']['khung_giua_session'] is False and r['u_a_k_deg'] is None
    kf = _khung(tmp_path, 'kba', (B_, A_), du=0.002)     # pair order does not matter
    rc, r = run('ok', '--khung-giua-session', kf)
    k = r['khung_giua_session']
    assert r['cong']['khung_giua_session'] is True and k['du_m'] == 0.002 and k['cap'][0]['sha256'] == ca.sha256(kf)
    u = math.degrees(math.asin(2 * 0.002 / k['L_m']))
    assert k['u_deg'] == pytest.approx(u) and r['u_hang_deg']['u_khung_giua_session'] == pytest.approx(u)
    assert 0.3 < k['L_m'] < 2.0 and u > 0.05
    assert _tong_hang(r) == pytest.approx(_tong_hang(same) + u)      # HCL1-8: no F9 -> u_a null, compare terms
    # the held-out legs of the issued attempt (one session) keep the same-session threshold
    assert [g['nguong_deg'] for g in r['giu_ngoai']] == pytest.approx([g['nguong_deg'] for g in same['giu_ngoai']])
    assert any(n['vai_tro'] == 'khung_giua_session' and n['sha256'] == ca.sha256(kf) for n in r['nguon'])
    big = _khung(tmp_path, 'kbig', (A_, B_), du=0.004)
    _, r4 = run('big', '--khung-giua-session', big)
    assert r4['u_hang_deg']['u_khung_giua_session'] == pytest.approx(math.degrees(math.asin(2 * 0.004 / k['L_m'])))


def test_tinh_session_files_and_unknown_session(tmp_path, capsys):
    s1 = write_json(str(tmp_path / 'ses1.json'), dict(fit_ids=[1], car_side_m=0.3, tag_h_m=0.2))
    s2 = write_json(str(tmp_path / 'ses2.json'), dict(fit_ids=[2], car_side_m=0.3, tag_h_m=0.2))
    d1, d2, run = _two(tmp_path, ca.sha256(s1), ca.sha256(s2))
    kf = _khung(tmp_path, 'k12', (ca.sha256(s1), ca.sha256(s2)))
    rc, r = run('both', '--session', s1, s2, '--khung-giua-session', kf)
    assert rc in (0, 1) and r['session_sha256'] == ca.sha256(s1)
    assert sorted(n['sha256'] for n in r['nguon'] if n['vai_tro'] == 'session') == sorted([ca.sha256(s1),
                                                                                            ca.sha256(s2)])
    capsys.readouterr()
    rc, r = run('one', '--session', s1, '--khung-giua-session', kf)    # HC_K2_2 filmed in a session not given
    assert rc == 2 and r is None and 'khong nam trong cac --session' in capsys.readouterr().out
    _set_track_session(d2, None)                                        # session unknown: never khoa
    rc, r = run('unk')
    assert any('khong ro session cua luot HC_K2_2' in w for w in r['ly_do']) and r['trang_thai'] != 'khoa'
    assert r['khung_giua_session']['khong_ro'] == ['HC_K2_2'] and r['session_sha256_luot']['HC_K2_2'] is None


def test_held_out_legs_ignore_the_cross_session_term(tmp_path):
    """The held-out short legs belong to the issued attempt (one session): a short leg off by more than the
    same-session threshold fails even when u_khung_giua_session makes u_a larger."""
    d1, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    d2, _ = make(tmp_path, 'HC_K2_2', cho='K2')
    kw = dict(e_hieu=E(0.002), q=[0.0, 0.0], du_m=0.0, neo=0.0, psi_cuc=0.0)
    r0 = ca.place_result('K2', [('c', analyse(d1)), ('o', analyse(d2))], 'c', **kw)
    an = analyse(d1)
    k = dict(khac=['o'], du_m=0.02)
    r1 = ca.place_result('K2', [('c', analyse(d1)), ('o', analyse(d2))], 'c', khung=k, **kw)
    u_k = math.radians(r1['u_hang_deg']['u_khung_giua_session'])
    assert u_k > math.radians(1.0)
    g0 = r0['giu_ngoai'][0]
    short = [lg for lg in an['legs'] if not lg['dai']][0]
    short['a_l'] = math.radians(r0['a_k_deg'] + g0['nguong_deg']) + 0.5 * u_k     # between the two thresholds
    r2 = ca.place_result('K2', [('c', an), ('o', analyse(d2))], 'c', khung=k, **kw)
    g = [x for x in r2['giu_ngoai'] if x['i'] == short['i']][0]
    assert g['nguong_deg'] == pytest.approx(g0['nguong_deg']) and g['dat'] is False
    assert r2['cong']['giu_ngoai'] is False


def _phys(**theo_cho):
    return dict(mo_hinh='phys_v1', k_se=3.0, rho_m=0.40, truoc_le=True,
                theo_cho={c: dict(xy_m=[0, 0], hanh_lang_12b=dict(du_xoay_rad=1e-5, du_thang=1e-4,
                                                                  du_bien_dang_rad=v)) for c, v in theo_cho.items()})


def test_tinh_phys_file_uses_corridor_deformation(tmp_path):
    """CODE_KHUNG phys_v1 (Liam 20/11 at CODE_KHUNG: "Truong tron phys (Recommended)"): the HIEUCHUAN term is
    theo_cho[cho].hanh_lang_12b.du_bien_dang_rad of the place, not asin(2 du_m / L); a phys file without the place is
    missing (chua_ket_luan)."""
    A_, B_ = 'a' * 64, 'b' * 64
    d1, d2, run = _two(tmp_path, A_, A_)
    _, same = run('same')
    _set_track_session(d2, B_)
    kf = _khung(tmp_path, 'kp', (A_, B_), du=0.0005, **_phys(K2=1.0e-4, K1=3.0e-4))
    rc, r = run('phys', '--khung-giua-session', kf)
    k = r['khung_giua_session']
    assert r['cong']['khung_giua_session'] is True and k['u_phys_rad'] == 1.0e-4 and k['du_m'] is None
    assert r['u_hang_deg']['u_khung_giua_session'] == pytest.approx(math.degrees(1.0e-4))
    assert r['u_hang_deg']['u_khung_giua_session'] < math.degrees(math.asin(2 * 0.0005 / k['L_m'])) / 10
    assert _tong_hang(r) == pytest.approx(_tong_hang(same) + math.degrees(1.0e-4))
    assert 'phys' in k['cap'][0]['cach'] and k['cap'][0]['mo_hinh'] == 'phys_v1'
    kn = _khung(tmp_path, 'kp_no_k2', (A_, B_), du=0.0005, **_phys(K1=3.0e-4))
    rc, r = run('phys_no_k2', '--khung-giua-session', kn)
    assert rc == 1 and r['trang_thai'] == 'chua_ket_luan' and r['cong']['khung_giua_session'] is False
    assert r['u_a_k_deg'] is None and r['khung_giua_session']['cap_thieu'] == [B_]
    assert 'thieu theo_cho.K2' in r['khung_giua_session']['cap'][0]['cach']


def test_place_result_cross_session_term_is_the_largest_pair():
    """Old and phys pair files together: the larger term; a bad value or a missing pair: no term (gate fails)."""
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        d1, _ = make(__import__('pathlib').Path(t), 'HC_K2_1', cho='K2')
        an = analyse(d1)
    kw = dict(e_hieu=E(0.002), q=[0.0, 0.0], du_m=0.0, neo=0.0, psi_cuc=0.0)
    att = [('c', an), ('o', an)]
    L = min(lg['L_m'] for lg in an['legs'] if lg['dai'] and lg['trang_thai'] == 'hop_le')
    u_tho = math.asin(2 * 0.002 / L)
    for u_p, want in ((1e-5, u_tho), (0.05, 0.05)):
        r = ca.place_result('K2', att, 'c', khung=dict(khac=['o'], du_m=0.002, u_phys_rad=u_p), **kw)
        assert math.radians(r['u_hang_deg']['u_khung_giua_session']) == pytest.approx(want)
    r = ca.place_result('K2', att, 'c', khung=dict(khac=['o'], u_phys_rad=2e-4), **kw)
    assert math.radians(r['u_hang_deg']['u_khung_giua_session']) == pytest.approx(2e-4)
    for bad in (dict(u_phys_rad=-1e-4), dict(u_phys_rad=float('nan')), dict(du_m=-0.001), dict(),
                dict(u_phys_rad=1e-4, cap_thieu=['x'])):
        r = ca.place_result('K2', att, 'c', khung=dict(khac=['o'], **bad), **kw)
        assert r['cong']['khung_giua_session'] is False and r['u_a_k_deg'] is None, bad


def test_read_khung_giua_session_phys_values(tmp_path):
    A_, B_ = 'a' * 64, 'b' * 64
    f = ca.read_khung_giua_session([_khung(tmp_path, 'p', (A_, B_), **_phys(goc=1e-4, K1=2e-4))])[frozenset((A_, B_))]
    assert f['hanh_lang_12b_du_bien_dang_rad'] == {'goc': 1e-4, 'K1': 2e-4} and f['mo_hinh'] == 'phys_v1'
    assert ca.read_khung_giua_session([_khung(tmp_path, 'o', (A_, B_))])[frozenset((A_, B_))][
        'hanh_lang_12b_du_bien_dang_rad'] is None
    p = _khung(tmp_path, 'p2', (A_, B_), theo_cho=dict(K2=dict(xy_m=[0, 0])))        # place without the value
    assert ca.read_khung_giua_session([p])[frozenset((A_, B_))]['hanh_lang_12b_du_bien_dang_rad'] == {}
    for name, v in (('neg', -1e-4), ('nan', float('nan')), ('bool', True), ('str', '1e-4'), ('big', 4.0)):
        with pytest.raises(ca.AError):
            ca.read_khung_giua_session([_khung(tmp_path, f'b_{name}', (A_, B_), **_phys(K2=v))])
    with pytest.raises(ca.AError):
        ca.read_khung_giua_session([_khung(tmp_path, 'b_list', (A_, B_), theo_cho=[1, 2])])


def test_read_khung_giua_session_refuses_bad_files(tmp_path):
    A_, B_ = 'a' * 64, 'b' * 64
    good = _khung(tmp_path, 'g', (A_, B_))
    assert ca.read_khung_giua_session([good])[frozenset((A_, B_))]['du_m'] == 0.002
    assert ca.read_khung_giua_session(None) == {}
    bad = [_khung(tmp_path, 'b1', (A_, B_), schema='c7_khung_v0'), _khung(tmp_path, 'b2', (A_, A_)),
           _khung(tmp_path, 'b3', (A_,)), _khung(tmp_path, 'b4', (A_, B_), du=-0.001),
           _khung(tmp_path, 'b5', (A_, B_), du=float('nan')), _khung(tmp_path, 'b6', (A_, B_), du=True),
           _khung(tmp_path, 'b7', (A_, B_), n=0), _khung(tmp_path, 'b8', (A_, B_), n=2.0),
           _khung(tmp_path, 'b9', (A_, ''))]
    for p in bad:
        with pytest.raises(ca.AError):
            ca.read_khung_giua_session([p])
    with pytest.raises(ca.AError):                                   # two files for one pair
        ca.read_khung_giua_session([good, _khung(tmp_path, 'g2', (B_, A_), du=0.001)])


def test_cua_so_session_from_track_and_checked(tmp_path):
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    _set_track_session(d, 'a' * 64)
    base = ['cua_so', d, '--frames', os.path.join(d, 'frames.csv'), '--track', os.path.join(d, 'track.json')]
    assert ca.main(base + ['--session-sha256', 'b' * 64, '--ra', str(tmp_path / 'x')]) == 2     # other session
    assert not (tmp_path / 'x' / 'doan_12b_v1.json').exists()
    assert ca.main(base) == 0                                          # none given: the track's session
    f5 = json.load(open(os.path.join(d, 'doan_12b_v1.json'), encoding='utf-8'))
    rows = [json.loads(x) for x in open(os.path.join(d, 'cua_so_v1.jsonl'), encoding='utf-8')]
    assert f5['session_sha256'] == 'a' * 64 and {r['session_sha256'] for r in rows} == {'a' * 64}


def test_cua_so_run2_f8_row_of_other_session(tmp_path, monkeypatch):
    """The F2 of F8 is the joined file of every segment: its ref_12b row of this attempt must name this session."""
    d, f2, a6, f8, run2 = _f8_setup(tmp_path, monkeypatch)
    ses = str(tmp_path / 'session.json')
    rows = [json.loads(x) for x in open(f2, encoding='utf-8')]

    def f2_with(name, s):
        p = str(tmp_path / f'{name}.jsonl')
        with open(p, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(dict(r, session_sha256=s)) + '\n')
        return p
    good = [_rec('C8_1', 'K2', 'L0', 'HC_K2_1')]
    for name, s, rc_want in (('other', 'f' * 64, 2), ('same', ca.sha256(ses), 0), ('none', None, 0)):
        p = f2_with(f'f2_{name}', s)
        rc, f5 = run2(f8(f'll_{name}', good, cua_so_v1_file=p, cua_so_v1_sha256=ca.sha256(p)), f'r_{name}')
        assert rc == rc_want, name
        if rc_want == 0:
            assert f5['lan_lap_id'] == 'L0' and f5['session_sha256'] == ca.sha256(ses)


def test_tinh_refuses_f5_of_other_session(tmp_path, capsys):
    """The F5 (and so the F9) of the issued attempt must be of the session of its track."""
    import hashlib
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    _set_track_session(d, 'a' * 64)
    run_cua_so(d)
    dp = os.path.join(d, 'doan_12b_v1.json')

    def f5_f9(name, s):
        f5 = json.load(open(dp, encoding='utf-8'))
        f5['session_sha256'] = s
        f5.pop('sha256')
        f5['sha256'] = hashlib.sha256(json.dumps(f5, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        p5 = write_json(str(tmp_path / f'{name}_f5.json'), f5)
        p9 = write_json(str(tmp_path / f'{name}_f9.json'), dict(doan_12b_sha256=f5['sha256'], trang_thai='dat',
                                                                doan=[dict(i=x['i'], e_hieu_m=0.0) for x in f5['doan']]))
        return p5, p9
    p5, p9 = f5_f9('ok', 'a' * 64)
    assert ca.main(tinh_args(d, 'K2', '--e-hieu', p9, '--doan', p5, '--out', str(tmp_path / 'ok.json'))) in (0, 1)
    p5, p9 = f5_f9('bad', 'e' * 64)
    capsys.readouterr()
    assert ca.main(tinh_args(d, 'K2', '--e-hieu', p9, '--doan', p5, '--out', str(tmp_path / 'bad.json'))) == 2
    assert 'F5 session_sha256 khac session' in capsys.readouterr().out


# ------------------------------------------------------------------ F4 read strictly (CODE 22/11, session 2 chain)
def _x_v1(a6_sha, **cho):
    """F4 xoay_v1 per place (contract C8 line 46): q after A6, u_q_m, a6_sha256, trang_thai."""
    blk = lambda **kw: dict(dict(q_tag_m=[-0.0004, 0.0002], u_q_m=0.0003, a6_sha256=a6_sha, trang_thai='dat'), **kw)  # noqa
    return dict(version='xoay_v1', cho={c: blk(**v) for c, v in cho.items()})


def _x_c8b_v1():
    """The diagnostic file c8b_xoay_v1 C8B wrote on session 2 (q on the tag heading without A6, top level only)."""
    return dict(version='c8b_xoay_v1', q_tag_m=[-0.00045, 0.00024], du_m=0.0015,
                cho=dict(goc=dict(doan=[]), K1=dict(doan=[])))


def test_xoay_q_only_place_block_of_xoay_v1(tmp_path):
    a6 = write_json(str(tmp_path / 'a6.json'), dict(schema='c8b_a6_v2', tam_id='T1'))
    s6 = ca.sha256(a6)
    # (1) c8b_xoay_v1 refused at every place, its top-level q never used
    for c in ('goc', 'K1', 'K2'):
        q, u, why = ca.xoay_q(_x_c8b_v1(), c, a6)
        assert q is None and u is None and 'c8b_xoay_v1' in why, c
    # (2) xoay_v1 without the place (K2: no E turn), even with a top-level q: None
    x = dict(_x_v1(s6, goc={}, K1={}), q_tag_m=[0.1, 0.1], u_q_m=0.1)
    q, u, why = ca.xoay_q(x, 'K2', a6)
    assert q is None and 'khong co cho' in why
    # (3) xoay_v1 with the place, dat, made on this A6: q and u_q_m
    assert ca.xoay_q(x, 'goc', a6) == ([-0.0004, 0.0002], 0.0003, None)
    bad = [(_x_v1(s6, goc=dict(trang_thai='chua_ket_luan')), 'trang_thai'),
           (_x_v1('0' * 64, goc={}), 'a6_sha256'),
           (_x_v1(s6, goc=dict(q_tag_m=[0.1])), 'q_tag_m'),
           (_x_v1(s6, goc=dict(q_tag_m=[0.1, float('nan')])), 'q_tag_m'),
           (_x_v1(s6, goc=dict(u_q_m=-1e-4)), 'u_q_m'),
           (_x_v1(s6, goc=dict(u_q_m=True)), 'u_q_m'),
           (dict(version='xoay_v1', cho='x'), 'khong co cho'),
           ('not a dict', 'xoay_v1')]
    for xx, want in bad:
        q, u, why = ca.xoay_q(xx, 'goc', a6)
        assert q is None and u is None and want in why, (xx, why)
    q, u, why = ca.xoay_q(_x_v1(s6, goc={}), 'goc', None)
    assert q is None and '--a6' in why


def test_tinh_cua_so_use_only_xoay_v1_of_the_place(tmp_path):
    """tinh and cua_so with the diagnostic file: no q (reason in F7 ly_do, F5 car_pose_cfg.q_ly_do); with xoay_v1 of
    the place made on the A6 of the run: q from that block."""
    d, _ = make(tmp_path, 'HC_goc_1', cho='goc')
    a6 = write_json(str(tmp_path / 'a6.json'), dict(schema='c8b_a6_v2', tam_id='T1'))
    xd = write_json(str(tmp_path / 'xd.json'), _x_c8b_v1())
    xv = write_json(str(tmp_path / 'xv.json'), _x_v1(ca.sha256(a6), goc={}, K1={}))
    out = str(tmp_path / 'd.json')
    assert ca.main(tinh_args(d, 'goc', '--xoay', xd, '--a6', a6, '--out', out)) in (0, 1)
    r = json.load(open(out, encoding='utf-8'))
    assert 'thieu q, du_m (xoay_v1)' in r['ly_do'] and any('c8b_xoay_v1' in w for w in r['ly_do'])
    assert r['trang_thai'] != 'khoa'
    out = str(tmp_path / 'v.json')
    assert ca.main(tinh_args(d, 'goc', '--xoay', xv, '--a6', a6, '--out', out)) in (0, 1)
    r = json.load(open(out, encoding='utf-8'))
    assert 'thieu q, du_m (xoay_v1)' not in r['ly_do'] and not any(w.startswith('xoay:') for w in r['ly_do'])
    base = ['cua_so', d, '--frames', os.path.join(d, 'frames.csv'), '--track', os.path.join(d, 'track.json'),
            '--a6', a6]
    assert ca.main(base + ['--xoay', xd, '--ra', str(tmp_path / 'r_d')]) == 0
    cfg = json.load(open(tmp_path / 'r_d' / 'doan_12b_v1.json', encoding='utf-8'))['car_pose_cfg']
    assert cfg['q_m'] is None and cfg['du_m'] is None and 'c8b_xoay_v1' in cfg['q_ly_do']
    assert ca.main(base + ['--xoay', xv, '--ra', str(tmp_path / 'r_v')]) == 0
    cfg = json.load(open(tmp_path / 'r_v' / 'doan_12b_v1.json', encoding='utf-8'))['car_pose_cfg']
    assert cfg['q_m'] == [-0.0004, 0.0002] and cfg['du_m'] == 0.0003 and cfg['q_ly_do'] is None
    xc = write_json(str(tmp_path / 'xc.json'), _x_v1(ca.sha256(a6), goc=dict(trang_thai='chua_ket_luan')))
    assert ca.main(base + ['--xoay', xc, '--ra', str(tmp_path / 'r_c')]) == 0      # a place block not dat: no q
    cfg = json.load(open(tmp_path / 'r_c' / 'doan_12b_v1.json', encoding='utf-8'))['car_pose_cfg']
    assert cfg['q_m'] is None and 'trang_thai' in cfg['q_ly_do']


# ------------------------------------------------------------------ K2 q from a place with E, same mounting (option A)
def _f5_file(tmp_path, name, cho, thu, ll, xoay, a6, **kw):
    import hashlib
    d = dict(schema='c8_hc_doan_12b_v1', thu_id=thu, cho=cho, lan_lap_id=ll, xoay_sha256=ca.sha256(xoay),
             a6_sha256=ca.sha256(a6), doan=[])
    d.update(kw)
    d['sha256'] = hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return write_json(str(tmp_path / f'{name}.json'), d)


def _k2_setup(tmp_path):
    a6 = write_json(str(tmp_path / 'a6.json'), dict(schema='c8b_a6_v2', tam_id='T1'))
    xv = write_json(str(tmp_path / 'xv.json'), _x_v1(ca.sha256(a6), goc=dict(q_tag_m=[0.0011, 0.0]),
                                                     K1=dict(q_tag_m=[-0.0004, 0.0002], u_q_m=0.0003)))
    return a6, xv, json.load(open(xv, encoding='utf-8'))


def test_q_of_k2_from_place_with_e_same_mounting(tmp_path):
    """CODE 22/11 option A: K2 takes q, u_q of K1 (or goc) when lan_lap_id of F5 K2 = lan_lap_id of F5 K1 (F8 run 1);
    another mounting, an unverified one (no F8: null) or no --q-tu: no q."""
    a6, xv, x = _k2_setup(tmp_path)
    k1 = _f5_file(tmp_path, 'f5_k1', 'K1', 'HC_K1_2', 'L2', xv, a6)
    q, u, why, ng = ca.q_cua_cho(x, 'K2', a6, xv, 'L2', k1)                       # same mounting: q of K1
    assert (q, u, why) == ([-0.0004, 0.0002], 0.0003, None)
    assert ng['cho'] == 'K1' and ng['thu_id'] == 'HC_K1_2' and ng['lan_lap_id'] == 'L2'
    assert ng['xoay_sha256'] == ca.sha256(xv) and ng['doan_12b_file_sha256'] == ca.sha256(k1)
    assert ng['doan_12b_sha256'] == json.load(open(k1, encoding='utf-8'))['sha256']
    g = _f5_file(tmp_path, 'f5_goc', 'goc', 'HC_goc_2', 'L2', xv, a6)             # goc works the same way
    assert ca.q_cua_cho(x, 'K2', a6, xv, 'L2', g)[0] == [0.0011, 0.0]
    q, u, why, ng = ca.q_cua_cho(x, 'K2', a6, xv, 'L3', k1)                       # another mounting
    assert q is None and ng is None and 'khac lan lap' in why
    for ll_k2, ll_k1 in ((None, 'L2'), ('L2', None), ('', 'L2')):                 # not verified (no F8 / null)
        k1n = _f5_file(tmp_path, f'f5_k1_{ll_k1}', 'K1', 'HC_K1_2', ll_k1, xv, a6)
        q, u, why, ng = ca.q_cua_cho(x, 'K2', a6, xv, ll_k2, k1n)
        assert q is None and 'chua xac minh' in why, (ll_k2, ll_k1)
    q, u, why, ng = ca.q_cua_cho(x, 'K2', a6, xv, 'L2', None)                     # no --q-tu
    assert q is None and '--q-tu' in why
    other_x = write_json(str(tmp_path / 'xo.json'), dict(x, note=1))
    k1x = _f5_file(tmp_path, 'f5_k1_x', 'K1', 'HC_K1_2', 'L2', other_x, a6)       # F5 K1 made from another xoay
    assert 'xoay / A6 khac' in ca.q_cua_cho(x, 'K2', a6, xv, 'L2', k1x)[2]
    xbad = dict(x, cho=dict(x['cho'], K1=dict(x['cho']['K1'], trang_thai='chua_ket_luan')))
    assert 'cung lan lap' in ca.q_cua_cho(xbad, 'K2', a6, xv, 'L2', k1)[2]       # the E place itself not dat


def test_q_of_place_with_own_block_never_borrows(tmp_path):
    a6, xv, x = _k2_setup(tmp_path)
    k1 = _f5_file(tmp_path, 'f5_k1', 'K1', 'HC_K1_2', 'L2', xv, a6)
    g = _f5_file(tmp_path, 'f5_goc', 'goc', 'HC_goc_2', 'L2', xv, a6)
    q, u, why, ng = ca.q_cua_cho(x, 'K1', a6, xv, 'L2', g)                        # own block wins
    assert q == [-0.0004, 0.0002] and ng['cho'] == 'K1' and ng['cach'] == 'khoi F4 cua chinh cho'
    xbad = dict(x, cho=dict(x['cho'], K1=dict(x['cho']['K1'], trang_thai='chua_ket_luan')))
    q, u, why, ng = ca.q_cua_cho(xbad, 'K1', a6, xv, 'L2', g)                     # a bad own block is not replaced
    assert q is None and 'trang_thai' in why
    assert ca.q_cua_cho(_x_c8b_v1(), 'K2', a6, xv, 'L2', k1)[0] is None           # c8b_xoay_v1: never
    with pytest.raises(ca.AError):
        ca.q_cua_cho(x, 'K2', a6, xv, 'L2', _f5_file(tmp_path, 'f5_k2', 'K2', 'HC_K2_1', 'L2', xv, a6))
    edited = json.load(open(k1, encoding='utf-8'))
    edited['lan_lap_id'] = 'L9'
    with pytest.raises(ca.AError):                                                # an edited F5 (content sha)
        ca.q_cua_cho(x, 'K2', a6, xv, 'L9', write_json(str(tmp_path / 'ed.json'), edited))


def test_tinh_and_cua_so_k2_q_from_k1(tmp_path):
    """tinh --cho K2 --q-tu F5_K1: q from K1 when the F5 of the issued K2 attempt has the same lan_lap_id, recorded in
    F7 q_nguon; cua_so of K2 without F8 (lan_lap_id null): no q, reason in F5 car_pose_cfg.q_ly_do."""
    a6, xv, x = _k2_setup(tmp_path)
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    k1 = _f5_file(tmp_path, 'f5_k1', 'K1', 'HC_K1_2', 'L2', xv, a6)
    own = _f5_file(tmp_path, 'f5_k2_own', 'K2', 'HC_K2_1', 'L2', xv, a6)
    out = str(tmp_path / 'f7.json')
    assert ca.main(tinh_args(d, 'K2', '--xoay', xv, '--a6', a6, '--q-tu', k1, '--doan', own, '--out', out)) in (0, 1)
    r = json.load(open(out, encoding='utf-8'))
    assert 'thieu q, du_m (xoay_v1)' not in r['ly_do'] and r['q_nguon']['cho'] == 'K1'
    assert r['q_nguon']['lan_lap_id'] == 'L2' and any(n['vai_tro'] == 'q_tu_doan_12b_v1' for n in r['nguon'])
    own3 = _f5_file(tmp_path, 'f5_k2_L3', 'K2', 'HC_K2_1', 'L3', xv, a6)
    out = str(tmp_path / 'f7b.json')
    assert ca.main(tinh_args(d, 'K2', '--xoay', xv, '--a6', a6, '--q-tu', k1, '--doan', own3, '--out', out)) in (0, 1)
    r = json.load(open(out, encoding='utf-8'))
    assert 'thieu q, du_m (xoay_v1)' in r['ly_do'] and r['q_nguon'] is None
    assert any('khac lan lap' in w for w in r['ly_do']) and r['trang_thai'] != 'khoa'
    assert ca.main(['cua_so', d, '--frames', os.path.join(d, 'frames.csv'), '--track', os.path.join(d, 'track.json'),
                    '--a6', a6, '--xoay', xv, '--q-tu', k1, '--ra', str(tmp_path / 'r_k2')]) == 0
    cfg = json.load(open(tmp_path / 'r_k2' / 'doan_12b_v1.json', encoding='utf-8'))['car_pose_cfg']
    assert cfg['q_m'] is None and cfg['q_nguon'] is None and 'chua xac minh' in cfg['q_ly_do']


# ------------------------------------------------------------------ option B (CODE 22/11): align off-A6 when frames lost
def _far_cut(F, cut_m=0.04):
    """A chain-like F whose frames near the far end of the 12b legs are lost (session 2 K1: 'a6_khung:ngoai_mien'):
    the frames with the tag centre within cut_m of the largest x of the attempt."""
    G = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in F.items()}
    far = np.asarray(G['x']) > np.nanmax(G['x'][G['ok']]) - cut_m
    G['ok'] = G['ok'] & ~far
    for k in ('x', 'y', 'psi', 'tx', 'ty'):
        if k in G:
            G[k] = np.where(far, np.nan, G[k])
    G['chuoi'] = 'khoa'
    return G


def test_align_off_a6_when_far_end_lost(tmp_path):
    """K1 with the far-end frames lost: before, align raised (code 2, no F2/F5); now align runs on the frames before
    the chain, the windows without chain frames make their legs 'thieu', and tinh gives chua_ket_luan."""
    d, _ = make(tmp_path, 'HC_K1_1', cho='K1')
    A = ca.load_attempt(d)
    _, info = ca.kiem(A)
    G = ca.frames_basic(os.path.join(d, 'frames.csv'), os.path.join(d, 'track.json'))
    F = _far_cut(G)
    assert int(F['ok'].sum()) < int(G['ok'].sum())
    with pytest.raises(ca.AError, match='camera intervals'):
        ca.analyse(A, F, info)                                           # the old behaviour on such frames
    an = ca.analyse(A, F, info, F_gio=G)
    assert an['can_gio'] == dict(khung='chua_A6', n_khung_chuoi_mat=int(G['ok'].sum() - F['ok'].sum()))
    assert an['D'] == ca.analyse(A, G, info)['D']                        # the same alignment as off-A6 frames
    st = [lg['trang_thai'] for lg in an['legs']]
    assert 'thieu' in st and any('frames' in (lg.get('ly_do') or '') for lg in an['legs'])
    r = ca.place_result('K1', [('HC_K1_1', an)], 'HC_K1_1', e_hieu=E(0.0), q=[0, 0], du_m=0.0, neo=0.0, psi_cuc=0.0)
    assert r['trang_thai'] == 'chua_ket_luan' and r['a_k_deg'] is None


def test_cua_so_tinh_write_files_when_far_end_lost(tmp_path, monkeypatch):
    """cua_so writes F2 and F5 (legs thieu, can_gio recorded) instead of code 2, so K2 can take --q-tu from it; tinh
    gives the place chua_ket_luan with code 1."""
    d, _ = make(tmp_path, 'HC_K1_1', cho='K1')
    monkeypatch.setattr(ca, 'frames_chain', lambda fr, a6_, xo, track=None, cho=None, **k:
                        _far_cut(ca.frames_basic(fr, track)))
    a6 = write_json(str(tmp_path / 'a6.json'), dict(schema='c8b_a6_v2', tam_id='T1'))
    xo = write_json(str(tmp_path / 'xo.json'), dict(version='xoay_v1', cho={}))
    assert ca.main(['cua_so', d, '--frames', os.path.join(d, 'frames.csv'), '--track', os.path.join(d, 'track.json'),
                    '--a6', a6, '--xoay', xo, '--ra', str(tmp_path / 'r2')]) == 0
    f5 = json.load(open(tmp_path / 'r2' / 'doan_12b_v1.json', encoding='utf-8'))
    assert f5['can_gio']['khung'] == 'chua_A6' and f5['can_gio']['n_khung_chuoi_mat'] > 0
    assert any(x['trang_thai'] == 'thieu' for x in f5['doan']) and ca.f5_problems(f5) == []
    out = str(tmp_path / 'f7.json')
    assert ca.main(tinh_args(d, 'K1', '--a6', a6, '--xoay', xo, '--out', out)) == 1
    r = json.load(open(out, encoding='utf-8'))
    assert r['trang_thai'] == 'chua_ket_luan' and r['luot_thu'][0]['can_gio']['khung'] == 'chua_A6'


def test_full_chain_unchanged_bit_for_bit(tmp_path):
    """No frame lost by the chain: analyse with F_gio gives exactly the old result (align on the chain frames)."""
    d, _ = make(tmp_path, 'HC_goc_1', cho='goc')
    A = ca.load_attempt(d)
    _, info = ca.kiem(A)
    G = ca.frames_basic(os.path.join(d, 'frames.csv'), os.path.join(d, 'track.json'))
    F = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in G.items()}
    F['psi'] = F['psi'] + 1e-4                       # a chain that changes the values but loses no frame
    F['x'] = F['x'] * 1.002                          # (speeds change, so aligning on G would move D)
    old, new = ca.analyse(A, F, info), ca.analyse(A, F, info, F_gio=G)
    assert new['can_gio'] == dict(khung='chuoi', n_khung_chuoi_mat=None)
    assert old['D'] != ca.analyse(A, G, info)['D']   # the check below can see an align on the wrong frames
    assert (old['D'], old['u_D'], old['diffs']) == (new['D'], new['u_D'], new['diffs'])
    for a_, b_ in zip(old['legs'], new['legs']):
        assert a_.keys() == b_.keys()
        for k in a_:
            va, vb = a_[k], b_[k]
            assert (np.array_equal(va, vb) if isinstance(va, np.ndarray) else va == vb), k
    assert old['truot_cheo'] == new['truot_cheo'] and old['thang_hang_m'] == new['thang_hang_m']


# ------------------------------------------------------------------ option A in the CAM chain (CAM 63dd13d)
def _fake_ck_q(monkeypatch, seen):
    import types
    m = types.ModuleType('c7_chuoi_khung')

    def chuoi_khung(frames_csv, a6_, xoay, lan_lap=None, track=None, cho=None, tam_id=None, cho_q=None, ll_id=None,
                    ll_id_q=None):
        seen.append(dict(cho=cho, cho_q=cho_q, ll_id=ll_id, ll_id_q=ll_id_q))
        return [_row(0), _row(1), _row(2)]
    m.doc_a6, m.chuoi_khung = (lambda p: dict(sha256='x')), chuoi_khung
    monkeypatch.setitem(sys.modules, 'c7_chuoi_khung', m)


def test_frames_chain_passes_option_a_to_cam(tmp_path, monkeypatch):
    a6 = write_json(str(tmp_path / 'a6.json'), dict(schema='c8b_a6_v2', tam_id='T1'))
    seen = []
    _fake_ck_q(monkeypatch, seen)
    ca.frames_chain('f.csv', a6, 'xoay.json', cho='K2', cho_q='K1', ll_id='L2', ll_id_q='L2')
    ca.frames_chain('f.csv', a6, 'xoay.json', cho='K1')
    assert seen == [dict(cho='K2', cho_q='K1', ll_id='L2', ll_id_q='L2'),
                    dict(cho='K1', cho_q=None, ll_id=None, ll_id_q=None)]
    old = _fake_ck(monkeypatch, rows=[_row(0), _row(1)])                  # a CAM without cho_q: not passed
    assert ca.frames_chain('f.csv', a6, 'xoay.json', cho='K2', cho_q='K1', ll_id='L2', ll_id_q='L2') is not None
    assert old['cho'] == 'K2'


def test_q_cung_lan_lap_inputs(tmp_path):
    a6, xv, x = _k2_setup(tmp_path)
    k1 = _f5_file(tmp_path, 'f5_k1', 'K1', 'HC_K1_2', 'L2', xv, a6)
    assert ca.q_cung_lan_lap(x, 'K2', k1, 'L2') == dict(cho_q='K1', ll_id='L2', ll_id_q='L2')
    assert ca.q_cung_lan_lap(x, 'K2', k1, None) == dict(cho_q='K1', ll_id=None, ll_id_q='L2')   # CAM refuses
    assert ca.q_cung_lan_lap(x, 'K1', k1, 'L2') is None                  # own block: CAM reads it, no option A
    assert ca.q_cung_lan_lap(x, 'K2', None, 'L2') is None
    with pytest.raises(ca.AError):
        ca.q_cung_lan_lap(x, 'K2', _f5_file(tmp_path, 'f5_k2', 'K2', 'HC_K2_1', 'L2', xv, a6), 'L2')


def test_cua_so_tinh_give_cam_the_mounting_ids(tmp_path, monkeypatch):
    """cua_so (K2, --lan-lap absent: ll_id None) and tinh (ll_id from the F5 of the issued attempt) pass cho_q and the
    two lan_lap_id to the CAM chain."""
    a6, xv, x = _k2_setup(tmp_path)
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    k1 = _f5_file(tmp_path, 'f5_k1', 'K1', 'HC_K1_2', 'L2', xv, a6)
    own = _f5_file(tmp_path, 'f5_k2_own', 'K2', 'HC_K2_1', 'L2', xv, a6)
    seen = []

    def fake(fr, a6_, xo, track=None, cho=None, **k):
        seen.append(dict(cho=cho, **k))
        return None                                                       # chain refused: basic frames
    monkeypatch.setattr(ca, 'frames_chain', fake)
    assert ca.main(['cua_so', d, '--frames', os.path.join(d, 'frames.csv'), '--track', os.path.join(d, 'track.json'),
                    '--a6', a6, '--xoay', xv, '--q-tu', k1, '--ra', str(tmp_path / 'r')]) == 0
    assert seen[-1] == dict(cho='K2', cho_q='K1', ll_id=None, ll_id_q='L2')
    assert ca.main(tinh_args(d, 'K2', '--xoay', xv, '--a6', a6, '--q-tu', k1, '--doan', own,
                             '--out', str(tmp_path / 'f7.json'))) in (0, 1)
    assert seen[-1] == dict(cho='K2', cho_q='K1', ll_id='L2', ll_id_q='L2')
    n = len(seen)
    assert ca.main(tinh_args(d, 'K2', '--xoay', xv, '--a6', a6, '--out', str(tmp_path / 'f7b.json'))) in (0, 1)
    assert seen[n] == dict(cho='K2')                                      # no --q-tu: nothing extra


# ------------------------------------------------------------------ F8 rong (CODE 23/11, Chot_F8_rong_buoi2_v1 muc 1)
def test_q_of_k2_f8_rong_borrows_with_plate_bound(tmp_path):
    """F8 read on both sides (same lan_lap_v1_sha256) but an id null: K2 borrows q of K1, u_q + 0.57 mm, labelled;
    two different non-null ids still refused; no F8 on one side, or two F8 files: not verified."""
    a6, xv, x = _k2_setup(tmp_path)
    k1 = _f5_file(tmp_path, 'f5_k1r', 'K1', 'HC_K1_1', None, xv, a6, lan_lap_v1_sha256='S8',
                   lan_lap_nhan=ca.NGUON_LECH_TAM)
    q, u, why, ng = ca.q_cua_cho(x, 'K2', a6, xv, None, k1, ll_sha='S8')
    assert q == [-0.0004, 0.0002] and why is None
    assert u == pytest.approx(0.0003 + 0.00057) and ca.U_LECH_TAM_BUOI1_M == 0.00057
    assert ng['nguon_u'] == 'can_lech_tam_buoi1_chua_xac_minh_lan_lap' and ng['u_q_f4_m'] == 0.0003
    assert ng['u_lech_tam_m'] == 0.00057 and ng['lan_lap_v1_sha256'] == 'S8' and 'F8 rong' in ng['cach']
    assert ng['cho'] == 'K1' and ng['doan_12b_file_sha256'] == ca.sha256(k1)
    q, u, why, ng = ca.q_cua_cho(x, 'K2', a6, xv, 'L2', k1, ll_sha='S8')          # one side null: same rule
    assert u == pytest.approx(0.00087) and ng['nguon_u'] == ca.NGUON_LECH_TAM
    for ll_sha in (None, '', 'S9'):                                               # no F8 here / another F8 file
        q, u, why, ng = ca.q_cua_cho(x, 'K2', a6, xv, None, k1, ll_sha=ll_sha)
        assert q is None and ng is None and 'chua xac minh' in why, ll_sha
    k1n = _f5_file(tmp_path, 'f5_k1n', 'K1', 'HC_K1_1', None, xv, a6)            # F5 K1 not read against an F8
    assert ca.q_cua_cho(x, 'K2', a6, xv, None, k1n, ll_sha='S8')[0] is None
    k1t = _f5_file(tmp_path, 'f5_k1t', 'K1', 'HC_K1_1', None, xv, a6, lan_lap_v1_sha256='S8')   # no label (2 ids)
    assert ca.q_cua_cho(x, 'K2', a6, xv, None, k1t, ll_sha='S8')[0] is None
    k1d = _f5_file(tmp_path, 'f5_k1d', 'K1', 'HC_K1_1', 'L3', xv, a6, lan_lap_v1_sha256='S8')
    q, u, why, ng = ca.q_cua_cho(x, 'K2', a6, xv, 'L2', k1d, ll_sha='S8')         # two ids, different: refused
    assert q is None and 'khac lan lap' in why
    k1s = _f5_file(tmp_path, 'f5_k1s', 'K1', 'HC_K1_1', 'L2', xv, a6, lan_lap_v1_sha256='S8')
    q, u, why, ng = ca.q_cua_cho(x, 'K2', a6, xv, 'L2', k1s, ll_sha='S8')         # verified: no bound added
    assert u == 0.0003 and 'nguon_u' not in ng
    k1x = _f5_file(tmp_path, 'f5_k1rx', 'K1', 'HC_K1_1', None, write_json(str(tmp_path / 'xo2.json'), dict(x, n=1)),
                   a6, lan_lap_v1_sha256='S8', lan_lap_nhan=ca.NGUON_LECH_TAM)
    assert 'xoay / A6 khac' in ca.q_cua_cho(x, 'K2', a6, xv, None, k1x, ll_sha='S8')[2]
    q, u, why, ng = ca.q_cua_cho(x, 'K1', a6, xv, None, k1, ll_sha='S8')          # own block: untouched
    assert u == 0.0003 and ng['cach'] == 'khoi F4 cua chinh cho'
    assert ca.cach_lan_lap('L2', None, dict(lan_lap_id='L2')) == 'cung'
    assert ca.cach_lan_lap(None, None, dict(lan_lap_id=None, lan_lap_v1_sha256=None)) == 'chua'


def test_q_cung_lan_lap_f8_rong_adds_bound(tmp_path):
    a6, xv, x = _k2_setup(tmp_path)
    k1 = _f5_file(tmp_path, 'f5_k1r', 'K1', 'HC_K1_1', None, xv, a6, lan_lap_v1_sha256='S8',
                   lan_lap_nhan=ca.NGUON_LECH_TAM)
    base = dict(cho_q='K1', ll_id=None, ll_id_q=None)
    assert ca.q_cung_lan_lap(x, 'K2', k1, None, ll_sha='S8', a6_path=a6, xoay_path=xv) == dict(base, u_them_m=0.00057)
    assert ca.q_cung_lan_lap(x, 'K2', k1, None, ll_sha='S9', a6_path=a6, xoay_path=xv) == base
    assert ca.q_cung_lan_lap(x, 'K2', k1, None, ll_sha='S8') == base                # sha of xoay / A6 not checkable
    other = write_json(str(tmp_path / 'a6b.json'), dict(schema='c8b_a6_v2', tam_id='T2'))
    assert ca.q_cung_lan_lap(x, 'K2', k1, None, ll_sha='S8', a6_path=other, xoay_path=xv) is None   # HCL1-2
    k1s = _f5_file(tmp_path, 'f5_k1s', 'K1', 'HC_K1_1', 'L2', xv, a6, lan_lap_v1_sha256='S8')
    assert 'u_them_m' not in ca.q_cung_lan_lap(x, 'K2', k1s, 'L2', ll_sha='S8', a6_path=a6, xoay_path=xv)
    assert ca.q_cung_lan_lap(x, 'K1', k1, None, ll_sha='S8', a6_path=a6, xoay_path=xv) is None


def test_frames_chain_f8_rong_hands_cam_the_borrowed_block(tmp_path, monkeypatch):
    """u_them_m: the block of cho_q is read with CAM's doc_xoay, u_q raised, and passed as a read xoay (no ids)."""
    import c7_chuoi_khung as ck
    a6, xv, x = _k2_setup(tmp_path)
    seen = []

    def fake(frames_csv, a6_, xoay, lan_lap=None, track=None, cho=None, tam_id=None, cho_q=None, ll_id=None,
             ll_id_q=None):
        seen.append(dict(xoay=xoay, cho=cho, cho_q=cho_q))
        return [_row(0), _row(1), _row(2)]
    monkeypatch.setattr(ck, 'chuoi_khung', fake)
    monkeypatch.setattr(ck, 'doc_a6', lambda p: dict(sha256='x'))
    F = ca.frames_chain('f.csv', a6, xv, cho='K2', cho_q='K1', ll_id=None, ll_id_q=None, u_them_m=0.00057)
    assert F is not None and F['chuoi'] == 'khoa'
    xo = seen[-1]['xoay']
    assert seen[-1]['cho_q'] is None and isinstance(xo, dict) and xo['ok'] and xo['q'] == [-0.0004, 0.0002]
    assert xo['u_q'] == pytest.approx(0.00087) and xo['nguon_q']['nguon_u'] == ca.NGUON_LECH_TAM
    assert xo['nguon_q']['cho'] == 'K1'
    xbad = write_json(str(tmp_path / 'xb.json'), dict(x, cho=dict(x['cho'], K1=dict(x['cho']['K1'],
                                                                                     trang_thai='chua_ket_luan'))))
    ca.frames_chain('f.csv', a6, xbad, cho='K2', cho_q='K1', u_them_m=0.00057)   # the E place not dat: no q
    assert seen[-1]['xoay']['q'] is None or not seen[-1]['xoay']['ok']
    assert 'nguon_q' not in seen[-1]['xoay']
    ca.frames_chain('f.csv', a6, xv, cho='K2', cho_q='K1', ll_id='L2', ll_id_q='L2')   # verified: ids to CAM as before
    assert seen[-1]['xoay'] == xv and seen[-1]['cho_q'] == 'K1'


def test_tinh_k2_f8_rong_q_from_k1_with_bound(tmp_path):
    """tinh --cho K2 --q-tu F5_K1, both F5 read against the same F8 (lan_lap_v1_sha256) with null ids: q of K1,
    du_m = u_q + 0.57 mm, F7 q_nguon labelled; another F8 file on the K2 F5: no q."""
    a6, xv, x = _k2_setup(tmp_path)
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    k1 = _f5_file(tmp_path, 'f5_k1', 'K1', 'HC_K1_1', None, xv, a6, lan_lap_v1_sha256='S8',
                   lan_lap_nhan=ca.NGUON_LECH_TAM)
    own = _f5_file(tmp_path, 'f5_k2_own', 'K2', 'HC_K2_1', None, xv, a6, lan_lap_v1_sha256='S8',
                   lan_lap_nhan=ca.NGUON_LECH_TAM)
    out = str(tmp_path / 'f7.json')
    assert ca.main(tinh_args(d, 'K2', '--xoay', xv, '--a6', a6, '--q-tu', k1, '--doan', own, '--out', out)) in (0, 1)
    r = json.load(open(out, encoding='utf-8'))
    assert 'thieu q, du_m (xoay_v1)' not in r['ly_do'] and r['q_nguon']['cho'] == 'K1'
    assert r['q_nguon']['nguon_u'] == ca.NGUON_LECH_TAM and r['q_nguon']['u_q_f4_m'] == 0.0003
    own9 = _f5_file(tmp_path, 'f5_k2_S9', 'K2', 'HC_K2_1', None, xv, a6, lan_lap_v1_sha256='S9',
                   lan_lap_nhan=ca.NGUON_LECH_TAM)
    out = str(tmp_path / 'f7b.json')
    assert ca.main(tinh_args(d, 'K2', '--xoay', xv, '--a6', a6, '--q-tu', k1, '--doan', own9, '--out', out)) in (0, 1)
    r = json.load(open(out, encoding='utf-8'))
    assert 'thieu q, du_m (xoay_v1)' in r['ly_do'] and r['q_nguon'] is None


def test_cua_so_k2_f8_rong_end_to_end(tmp_path, monkeypatch):
    """cua_so run 2 of K2 with --q-tu F5 K1 (both read against the same F8): F8 rong -> q of K1, du_m + 0.57 mm;
    F8 with two ids for the K2 12b (no label) -> no q even though F5 K1 names that F8."""
    d, f2, a6, f8, run2 = _f8_setup(tmp_path, monkeypatch)
    xv = write_json(str(tmp_path / 'xv.json'), _x_v1(ca.sha256(a6), K1=dict(q_tag_m=[-0.0004, 0.0002], u_q_m=0.0003)))
    for name, luot, rong in (('rong', [_rec('C8_1', 'K2', None, None)], True),
                             ('two', [_rec('C8_1', 'K2', 'L0', 'HC_K2_1'), _rec('C8_2', 'K2', 'L1', 'HC_K2_1')], False)):
        ll = f8(name, luot)
        k1 = _f5_file(tmp_path, f'f5_k1_{name}', 'K1', 'HC_K1_1', None, xv, a6, lan_lap_v1_sha256=ca.sha256(ll),
                      lan_lap_nhan=ca.NGUON_LECH_TAM)
        rc, f5 = run2(ll, f'r_{name}', extra=('--xoay', xv, '--q-tu', k1))
        cfg = f5['car_pose_cfg']
        assert rc == 0, name
        if rong:
            assert cfg['q_m'] == [-0.0004, 0.0002] and cfg['du_m'] == pytest.approx(0.00087)
            assert cfg['q_nguon']['nguon_u'] == ca.NGUON_LECH_TAM and f5['trang_thai'] == 'khoa'
        else:
            assert cfg['q_m'] is None and 'chua xac minh' in cfg['q_ly_do'] and f5['trang_thai'] == 'chan_doan'


def test_q_f8_rong_needs_label_on_this_side_in_tinh(tmp_path):
    """tinh: the F5 of the issued K2 attempt read against F8 but without the F8 rong label (F8 gave two ids): no q,
    even when the F5 K1 of --q-tu is labelled and names the same F8; q_cung_lan_lap: F5 K1 of another xoay: no bound."""
    a6, xv, x = _k2_setup(tmp_path)
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    k1 = _f5_file(tmp_path, 'f5_k1', 'K1', 'HC_K1_1', None, xv, a6, lan_lap_v1_sha256='S8',
                  lan_lap_nhan=ca.NGUON_LECH_TAM)
    own = _f5_file(tmp_path, 'f5_k2_two', 'K2', 'HC_K2_1', None, xv, a6, lan_lap_v1_sha256='S8')
    out = str(tmp_path / 'f7.json')
    assert ca.main(tinh_args(d, 'K2', '--xoay', xv, '--a6', a6, '--q-tu', k1, '--doan', own, '--out', out)) in (0, 1)
    r = json.load(open(out, encoding='utf-8'))
    assert 'thieu q, du_m (xoay_v1)' in r['ly_do'] and r['q_nguon'] is None
    other = write_json(str(tmp_path / 'xo3.json'), dict(x, n=2))
    k1x = _f5_file(tmp_path, 'f5_k1_ox', 'K1', 'HC_K1_1', None, other, a6, lan_lap_v1_sha256='S8',
                   lan_lap_nhan=ca.NGUON_LECH_TAM)
    assert ca.q_cung_lan_lap(x, 'K2', k1x, None, ll_sha='S8', a6_path=a6, xoay_path=xv) is None   # HCL1-2


def test_tinh_reads_f2_f5_of_run2_folder(tmp_path, monkeypatch):
    """T9b: run 2 of cua_so writes F2 + F5 to its --ra folder; tinh --e-hieu --doan checks THAT F2 and F5 against the
    sources (A6, xoay of run 2), not the run-1 files of the attempt folder (no A6: refused before), and records that F2;
    without --doan the attempt folder's own files are used (run 1: other A6 -> refused, code 2)."""
    d, f2, a6, f8, run2 = _f8_setup(tmp_path, monkeypatch)
    rc, f5 = run2(f8('ll', [_rec('C8_1', 'K2', 'L0', 'HC_K2_1')]), 'r2')
    assert rc == 0 and f5['trang_thai'] == 'khoa'
    dp = str(tmp_path / 'r2' / 'doan_12b_v1.json')
    f9 = write_json(str(tmp_path / 'f9.json'), dict(doan_12b_sha256=f5['sha256'], trang_thai='dat',
                                                    doan=[dict(i=x['i'], e_hieu_m=0.0) for x in f5['doan']]))
    xoay = str(tmp_path / 'xoay.json')
    out = str(tmp_path / 'f7.json')
    assert ca.main(tinh_args(d, 'K2', '--a6', a6, '--xoay', xoay, '--e-hieu', f9, '--doan', dp, '--out', out)) in (0, 1)
    r = json.load(open(out, encoding='utf-8'))
    f2_used = [n['duong_dan'] for n in r['nguon'] if n['vai_tro'] == 'cua_so_v1']
    assert f2_used == [os.path.abspath(str(tmp_path / 'r2' / 'cua_so_v1.jsonl'))]
    assert ca.cap_f2_f5(d) == (os.path.join(d, 'cua_so_v1.jsonl'), os.path.join(d, 'doan_12b_v1.json'))
    assert ca.cap_f2_f5(d, dp)[0] == os.path.join(str(tmp_path / 'r2'), 'cua_so_v1.jsonl')
    f9_1 = write_json(str(tmp_path / 'f9_1.json'), dict(doan_12b_sha256=json.load(open(
        os.path.join(d, 'doan_12b_v1.json'), encoding='utf-8'))['sha256'], trang_thai='dat',
        doan=[dict(i=x['i'], e_hieu_m=0.0) for x in f5['doan']]))
    assert ca.main(tinh_args(d, 'K2', '--a6', a6, '--xoay', xoay, '--e-hieu', f9_1,
                             '--out', str(tmp_path / 'f7b.json'))) == 2      # run-1 F5 has no A6
    os.remove(str(tmp_path / 'r2' / 'cua_so_v1.jsonl'))
    assert ca.main(tinh_args(d, 'K2', '--a6', a6, '--xoay', xoay, '--e-hieu', f9, '--doan', dp,
                             '--out', str(tmp_path / 'f7c.json'))) == 2      # run-2 F2 missing: refused


def test_tinh_psi_chuan_k_from_ref_12b_without_f6(tmp_path, monkeypatch):
    """CODE 23/11 (chot T12 muc 4): no F6 'dat' -> psi_chuan_k = circular mean of psi' over the ref_12b window of
    the issued attempt on a locked chain; phi_hat = wrap(psi_chuan_k - a), u_phi_hat = u_a_k, labelled; turning
    every camera heading by +0.3 rad turns psi_chuan_k by +0.3 rad and leaves phi_hat; chain not locked: none."""
    d, f2, a6, f8, run2 = _f8_setup(tmp_path, monkeypatch)
    rc, f5 = run2(f8('ll', [_rec('C8_1', 'K2', 'L0', 'HC_K2_1')]), 'r2')
    dp = str(tmp_path / 'r2' / 'doan_12b_v1.json')
    f9 = write_json(str(tmp_path / 'f9.json'), dict(doan_12b_sha256=f5['sha256'], trang_thai='dat',
                                                    doan=[dict(i=x['i'], e_hieu_m=0.0) for x in f5['doan']]))
    src = ('--a6', a6, '--xoay', str(tmp_path / 'xoay.json'), '--e-hieu', f9, '--doan', dp)

    def f7(name, turn=0.0, chuoi='khoa'):
        def chain(fr, a6_, xo, track=None, cho=None, **k):
            F = dict(ca.frames_basic(fr, track), chuoi=chuoi)
            F['psi'] = np.asarray(F['psi']) + turn
            return F
        monkeypatch.setattr(ca, 'frames_chain', chain)
        out = str(tmp_path / f'{name}.json')
        assert ca.main(tinh_args(d, 'K2', *src, '--out', out)) in (0, 1)
        return json.load(open(out, encoding='utf-8'))
    r0, r1 = f7('r0'), f7('r1', turn=0.3)
    for r in (r0, r1):
        assert r['psi_chuan_k_nguon'] == ca.PSI_K_NGUON_CAM == 'ref_12b_camera_khong_F6'
        assert 'thieu psi_chuan_k (F6)' not in r['ly_do']
        assert r['phi_hat_chuan_k_deg'] == pytest.approx(math.degrees(ca.wrap(math.radians(
            r['psi_chuan_k_deg'] - r['a_k_deg']))))
        assert r['u_phi_hat_chuan_k_deg'] == r['u_a_noi_bo_deg']          # CODE 23/11 F7 thu 2 (option A)
    assert r1['psi_chuan_k_deg'] - r0['psi_chuan_k_deg'] == pytest.approx(math.degrees(0.3), abs=1e-6)
    assert r1['phi_hat_chuan_k_deg'] == pytest.approx(r0['phi_hat_chuan_k_deg'], abs=1e-6)
    rb = f7('rb', chuoi='chan_doan')                                         # chain not locked: no camera psi
    assert rb.get('psi_chuan_k_deg') is None and rb.get('phi_hat_chuan_k_deg') is None
    assert 'thieu psi_chuan_k (F6)' in rb['ly_do'] and rb['trang_thai'] != 'khoa'


def test_cap_f2_picked_by_f5_sha(tmp_path, monkeypatch):
    """cap_f2_f5 with --doan: of the F2 beside that F5 and the attempt folder's, the one whose sha256 is the
    cua_so_v1_sha256 of the F5 (an F5 copied next to another attempt's F2 still finds its own)."""
    import shutil
    d, f2, a6, f8, run2 = _f8_setup(tmp_path, monkeypatch)
    run2(f8('ll', [_rec('C8_1', 'K2', 'L0', 'HC_K2_1')]), 'r2')
    r3 = tmp_path / 'r3'
    r3.mkdir()
    shutil.copy(os.path.join(d, 'doan_12b_v1.json'), str(r3 / 'doan_12b_v1.json'))          # run-1 F5
    shutil.copy(str(tmp_path / 'r2' / 'cua_so_v1.jsonl'), str(r3 / 'cua_so_v1.jsonl'))        # another F2
    dp = str(r3 / 'doan_12b_v1.json')
    assert ca.cap_f2_f5(d, dp) == (os.path.join(d, 'cua_so_v1.jsonl'), dp)
    f5 = json.load(open(dp, encoding='utf-8'))
    f9 = write_json(str(tmp_path / 'f9r3.json'), dict(doan_12b_sha256=f5['sha256'], trang_thai='dat',
                                                      doan=[dict(i=x['i'], e_hieu_m=0.0) for x in f5['doan']]))
    assert ca.main(tinh_args(d, 'K2', '--e-hieu', f9, '--doan', dp, '--out', str(tmp_path / 'f7r3.json'))) in (0, 1)


# ------------------------------------------------------------------ Codex L138 round 1 (HCL1-n), Liam 23/11: Sua 10 muc
def test_hcl1_1_f7_never_locks_on_f5_not_locked(tmp_path, monkeypatch):
    """HCL1-1: an F5 chan_doan (no --lan-lap) with an F9 'dat' on its sha: F7 is not khoa, reason names the F5."""
    d, f2, a6, f8, run2 = _f8_setup(tmp_path, monkeypatch)
    rc, f5 = run2(None, 'r_no')
    assert rc == 0 and f5['trang_thai'] == 'chan_doan' and f5['chuoi'] == 'khoa'
    dp = str(tmp_path / 'r_no' / 'doan_12b_v1.json')
    f9 = write_json(str(tmp_path / 'f9.json'), dict(doan_12b_sha256=f5['sha256'], trang_thai='dat',
                                                    doan=[dict(i=x['i'], e_hieu_m=0.0) for x in f5['doan']]))
    out = str(tmp_path / 'f7.json')
    assert ca.main(tinh_args(d, 'K2', '--a6', a6, '--xoay', str(tmp_path / 'xoay.json'), '--e-hieu', f9,
                             '--doan', dp, '--out', out)) in (0, 1)
    r = json.load(open(out, encoding='utf-8'))
    assert r['trang_thai'] != 'khoa' and any('(khong khoa)' in w and 'chan_doan' in w for w in r['ly_do'])


def test_hcl1_2_no_q_no_lock_and_no_ids_to_cam(tmp_path, monkeypatch):
    """HCL1-2: a place whose q is refused (no own block, F5 of --q-tu made from another xoay) gives an F5 that is
    not khoa (q_m null) even with a locked chain, and CAM gets no mounting ids."""
    d, f2, a6, f8, run2 = _f8_setup(tmp_path, monkeypatch)
    xv = write_json(str(tmp_path / 'xv.json'), _x_v1(ca.sha256(a6), K1=dict(q_tag_m=[-0.0004, 0.0002], u_q_m=0.0003)))
    other = write_json(str(tmp_path / 'xold.json'), dict(json.load(open(xv, encoding='utf-8')), n=1))
    ll = f8('ll', [_rec('C8_1', 'K2', 'L7', 'HC_K2_1')])
    k1 = _f5_file(tmp_path, 'f5_k1_old', 'K1', 'HC_K1_1', 'L7', other, a6)
    x = json.load(open(xv, encoding='utf-8'))
    assert ca.q_cung_lan_lap(x, 'K2', k1, 'L7', a6_path=a6, xoay_path=xv) is None
    rc, f5 = run2(ll, 'r_q', extra=('--xoay', xv, '--q-tu', k1))
    assert rc == 0 and f5['chuoi'] == 'khoa' and f5['car_pose_cfg']['q_m'] is None
    assert f5['trang_thai'] == 'chan_doan'


def test_hcl1_4_f8_xoay_sha_tam_and_id_type(tmp_path, monkeypatch):
    """HCL1-4: F8 naming another xoay file, a non-string lan_lap_id for this 12b: code 2; F4 of another plate:
    no q (reason tam_id)."""
    d, f2, a6, f8, run2 = _f8_setup(tmp_path, monkeypatch)
    xoay = str(tmp_path / 'xoay.json')
    assert run2(f8('xs', [_rec('C8_1', 'K2', 'L0', 'HC_K2_1')], xoay_sha256='0' * 64), 'r_xs')[0] == 2
    assert run2(f8('xs_ok', [_rec('C8_1', 'K2', 'L0', 'HC_K2_1')], xoay_sha256=ca.sha256(xoay)), 'r_xo')[0] == 0
    assert run2(f8('num', [_rec('C8_1', 'K2', 123, 'HC_K2_1')]), 'r_num')[0] == 2
    a6_, xv, x = _k2_setup(tmp_path)
    q, u, why = ca.xoay_q(dict(x, tam_id='T9'), 'K1', a6_)
    assert q is None and 'tam_id' in why
    assert ca.xoay_q(dict(x, tam_id='T1'), 'K1', a6_)[0] is not None


def test_hcl1_5_dets_of_another_attempt_refused(tmp_path, monkeypatch):
    """HCL1-5: the track names its dets file; --dets of another file: code 2; its own dets: runs."""
    monkeypatch.setattr(ca, 'dets_fingerprint', lambda p: 'fp' if p else None)
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    clip = write_json(str(tmp_path / 'clip.json'), dict(x=1))
    da = write_json(str(tmp_path / 'dets_a.json'), dict(a=1))
    db = write_json(str(tmp_path / 'dets_b.json'), dict(b=1))
    tp = os.path.join(d, 'track.json')
    tj = json.load(open(tp, encoding='utf-8'))
    tj['dets_sha256'] = ca.sha256(da)
    write_json(tp, tj)
    base = ['cua_so', d, '--frames', os.path.join(d, 'frames.csv'), '--track', tp, '--clip', clip]
    assert ca.main(base + ['--dets', db, '--ra', str(tmp_path / 'r_b')]) == 2
    assert ca.main(base + ['--dets', da, '--ra', str(tmp_path / 'r_a')]) == 0


def test_hcl1_6_tinh_q_must_match_f5(tmp_path, monkeypatch):
    """HCL1-6: same F2/F5/F9 of K2, tinh with --q-tu of another place (another q): code 2; the same --q-tu: runs."""
    d, f2, a6, f8, run2 = _f8_setup(tmp_path, monkeypatch)
    xv = write_json(str(tmp_path / 'xv.json'), _x_v1(ca.sha256(a6), goc=dict(q_tag_m=[0.0011, 0.0]),
                                                     K1=dict(q_tag_m=[-0.0004, 0.0002], u_q_m=0.0003)))
    ll = f8('ll', [_rec('C8_1', 'K2', None, None)])
    lab = dict(lan_lap_v1_sha256=ca.sha256(ll), lan_lap_nhan=ca.NGUON_LECH_TAM)
    k1 = _f5_file(tmp_path, 'f5_k1', 'K1', 'HC_K1_1', None, xv, a6, **lab)
    g = _f5_file(tmp_path, 'f5_goc', 'goc', 'HC_goc_2', None, xv, a6, **lab)
    rc, f5 = run2(ll, 'r6', extra=('--xoay', xv, '--q-tu', k1))
    assert rc == 0 and f5['car_pose_cfg']['q_m'] == [-0.0004, 0.0002]
    dp = str(tmp_path / 'r6' / 'doan_12b_v1.json')
    f9 = write_json(str(tmp_path / 'f9.json'), dict(doan_12b_sha256=f5['sha256'], trang_thai='dat',
                                                    doan=[dict(i=x['i'], e_hieu_m=0.0) for x in f5['doan']]))
    src = ('--a6', a6, '--xoay', xv, '--e-hieu', f9, '--doan', dp)
    assert ca.main(tinh_args(d, 'K2', *src, '--q-tu', g, '--out', str(tmp_path / 'f7g.json'))) == 2
    assert ca.main(tinh_args(d, 'K2', *src, '--q-tu', k1, '--out', str(tmp_path / 'f7k.json'))) in (0, 1)


def test_hcl1_7_frames_track_pair_checked_in_tinh(tmp_path):
    """HCL1-7: an attempt whose track names other frames: code 2 in tinh."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    tp = os.path.join(d, 'track.json')
    tj = json.load(open(tp, encoding='utf-8'))
    tj['frames_sha256'] = '2' * 64
    write_json(tp, tj)
    assert ca.main(tinh_args(d, 'K2', '--out', str(tmp_path / 'f7.json'))) == 2
    tj['frames_sha256'] = ca.sha256(os.path.join(d, 'frames.csv'))
    write_json(tp, tj)
    assert ca.main(tinh_args(d, 'K2', '--out', str(tmp_path / 'f7b.json'))) in (0, 1)


def test_hcl1_8_no_f9_u_map_null(tmp_path):
    """HCL1-8: tinh without F9: u_map null and listed, u_a null (never 0 / a finite sum)."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    out = str(tmp_path / 'f7.json')
    assert ca.main(tinh_args(d, 'K2', '--out', out)) in (0, 1)
    r = json.load(open(out, encoding='utf-8'))
    assert r['u_hang_deg']['u_map'] is None and 'u_map' in r['u_hang_thieu'] and r['u_a_k_deg'] is None


def test_hcl1_11_phys_file_without_map_refused(tmp_path):
    """HCL1-11: mo_hinh phys_v1 without theo_cho, or a corridor of the wrong type: AError (code 2), never old."""
    pair = ('a' * 64, 'b' * 64)
    with pytest.raises(ca.AError):
        ca.read_khung_giua_session([_khung(tmp_path, 'p1', pair, du=0.0, mo_hinh='phys_v1')])
    with pytest.raises(ca.AError):
        ca.read_khung_giua_session([_khung(tmp_path, 'p2', pair, mo_hinh='phys_v1',
                                           theo_cho=dict(K1=dict(hanh_lang_12b=[1])))])
    with pytest.raises(ca.AError):
        ca.read_khung_giua_session([_khung(tmp_path, 'p3', pair, mo_hinh='phys_v1', theo_cho=dict(K1=[1]))])
    ok = ca.read_khung_giua_session([_khung(tmp_path, 'p4', pair, mo_hinh='phys_v1',
                                            theo_cho=dict(K1=dict(hanh_lang_12b=dict(du_bien_dang_rad=0.001))))])
    assert list(ok.values())[0]['hanh_lang_12b_du_bien_dang_rad'] == {'K1': 0.001}
    old = ca.read_khung_giua_session([_khung(tmp_path, 'p5', pair)])     # an old file (no mo_hinh): as before
    assert list(old.values())[0]['hanh_lang_12b_du_bien_dang_rad'] is None


def test_hcl1_13_rv_repeat_across_block_edge():
    """HCL1-13: the first new RV report of a turn block repeating the value held at the block start (row 0, the
    last report before the block) while turning is a repeat; a mid-block repeat still is; a changed value is not."""
    rate = np.full(12, 0.5)
    rv_n = np.array([7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10])
    yaw = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.02, 1.02, 1.02, 1.04, 1.04, 1.04])
    assert ca.rv_lap_khi_quay(rate, rv_n, yaw) == [3]                    # edge: report 8 = value held at start
    yaw2 = np.array([1.0] * 3 + [1.01] * 3 + [1.01] * 3 + [1.03] * 3)
    assert ca.rv_lap_khi_quay(rate, rv_n, yaw2) == [6]                   # mid-block repeat (as before)
    assert ca.rv_lap_khi_quay(np.full(12, 0.05), rv_n, yaw) == []        # wheels not turning: not a repeat
    yaw3 = np.array([1.0] * 3 + [1.01] * 3 + [1.02] * 3 + [1.03] * 3)
    assert ca.rv_lap_khi_quay(rate, rv_n, yaw3) == []


# ------------------------------------------------------------------ F7 thu 2 (CODE 23/11): u_a_noi_bo_deg, dau_doan
def test_f7_thu2_noi_bo_and_dau_doan(tmp_path, monkeypatch):
    """u_a_noi_bo_deg = u_a without u_psi_cuc (null when another term is missing); option A: u_phi_hat =
    u_a_noi_bo; dau_doan: one record per valid long leg of the issued attempt (no C8B bound function), its frame ids
    = the frames psi' of the leg-start window is averaged on, frames path + sha."""
    d, f2, a6, f8, run2 = _f8_setup(tmp_path, monkeypatch)
    rc, f5 = run2(f8('ll', [_rec('C8_1', 'K2', 'L0', 'HC_K2_1')]), 'r2')
    dp = str(tmp_path / 'r2' / 'doan_12b_v1.json')
    f9 = write_json(str(tmp_path / 'f9.json'), dict(doan_12b_sha256=f5['sha256'], trang_thai='dat',
                                                    doan=[dict(i=x['i'], e_hieu_m=0.0) for x in f5['doan']]))
    out = str(tmp_path / 'f7.json')
    c8b = write_json(str(tmp_path / 'c8b.json'), dict(u_psi_cuc_rad=0.004, u_psi_hai_dau_rad=0.001))
    assert ca.main(tinh_args(d, 'K2', '--a6', a6, '--xoay', str(tmp_path / 'xoay.json'), '--e-hieu', f9,
                             '--doan', dp, '--can-c8b', c8b, '--out', out)) in (0, 1)
    r = json.load(open(out, encoding='utf-8'))
    h = r['u_hang_deg']
    assert h['u_psi_cuc'] == pytest.approx(math.degrees(0.004)) and not r['u_hang_thieu']
    assert r['u_a_noi_bo_deg'] == pytest.approx(r['u_a_k_deg'] - h['u_psi_cuc'])
    assert r['u_a_noi_bo_deg'] == pytest.approx(sum(v for k, v in h.items() if k != 'u_psi_cuc' and v is not None))
    assert r['u_phi_hat_chuan_k_deg'] == r['u_a_noi_bo_deg']
    dd = r['dau_doan']
    fr = os.path.join(d, 'frames.csv')
    assert [x['doan_i'] for x in dd] == [0, 1, 2, 3] and all(x['thu_id'] == 'HC_K2_1' for x in dd)
    assert all(x['frames_duong_dan'] == os.path.abspath(fr) and x['frames_sha256'] == ca.sha256(fr) for x in dd)
    F = ca.frames_basic(fr, os.path.join(d, 'track.json'))
    pos = {int(v): k for k, v in enumerate(F['idx'])}
    F['tx'], F['ty'] = F.get('tx', F['x']), F.get('ty', F['y'])          # frames_basic: x, y are the tag centre
    for x in dd:
        assert x['ids'] and len(set(x['ids'])) == len(x['ids'])
        ps = np.array([F['psi'][pos[i]] for i in x['ids']])
        assert x['psi_deg'] == pytest.approx(math.degrees(ca.mean_unwrapped(ps)), abs=1e-9)
        assert x['p_tag_m'] == pytest.approx([float(np.mean([F['tx'][pos[i]] for i in x['ids']])),
                                              float(np.mean([F['ty'][pos[i]] for i in x['ids']]))], abs=1e-12)


def test_f7_noi_bo_null_when_other_term_missing(tmp_path):
    """u_a_noi_bo_deg is null when a term other than u_psi_cuc is missing (here u_map: no F9); an attempt without
    analysis gives no dau_doan record."""
    d, _ = make(tmp_path, 'HC_K2_1', cho='K2')
    c8b = write_json(str(tmp_path / 'c8b.json'), dict(u_psi_cuc_rad=0.004, u_psi_hai_dau_rad=0.001))
    out = str(tmp_path / 'f7.json')
    assert ca.main(tinh_args(d, 'K2', '--can-c8b', c8b, '--out', out)) in (0, 1)
    r = json.load(open(out, encoding='utf-8'))
    assert 'u_map' in r['u_hang_thieu'] and r['u_a_noi_bo_deg'] is None
    assert ca.dau_doan([('x', None, [])], {}, {}, 'x') == []


def test_dau_doan_every_attempt_or_issued_only(tmp_path):
    """dau_doan: valid long legs only; every=True (C8B bound function) -> every attempt, else the issued one; ids =
    F idx under the window mask."""
    fr = write_json(str(tmp_path / 'fr.json'), dict(x=1))
    F = dict(idx=np.array([10, 11, 12, 13]))

    def an(tag):
        legs = [dict(i=0, dai=True, trang_thai='hop_le', a=0, p_tag_a=np.array([0.1, 0.2]), R_a=0.5),
                dict(i=1, dai=True, trang_thai='thieu', a=1, p_tag_a=np.array([0.0, 0.0]), R_a=0.0),
                dict(i=4, dai=False, trang_thai='hop_le', a=1, p_tag_a=np.array([0.0, 0.0]), R_a=0.0)]
        return dict(legs=legs, windows=[dict(mask=np.array([False, True, True, False])), dict(mask=np.ones(4, bool))])
    att = [('cap', an('c'), []), ('them', an('t'), [])]
    src = {'cap': (fr, F), 'them': (fr, F)}
    allr = ca.dau_doan(att, src, {'cap': 's1', 'them': 's2'}, 'cap', every=True)
    assert [(x['thu_id'], x['doan_i'], x['ids']) for x in allr] == [('cap', 0, [11, 12]), ('them', 0, [11, 12])]
    assert allr[1]['session_sha256'] == 's2' and allr[0]['psi_deg'] == pytest.approx(math.degrees(0.5))
    one = ca.dau_doan(att, src, {}, 'cap', every=False)
    assert [x['thu_id'] for x in one] == ['cap']


# ------------------------------------------------------------------ repeated UDP datagram (CAM, CODE 24/11; L92)
def write_segments(d, status=None):
    """<log>.segments.csv as agv_logger writes it: one row per data line; status[i] (1-based) overrides 'ok'."""
    import csv
    p = os.path.join(d, 'log.csv')
    with open(p, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    hdr, body = rows[0], rows[1:]
    with open(p + '.segments.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['row', 'log_seq', 'timestamp_us', 'segment', 'status'])
        for i, r in enumerate(body, start=1):
            w.writerow([i, r[col(hdr, 'log_seq')], r[col(hdr, 'timestamp_us')], 0, (status or {}).get(i, 'ok')])


def insert_after(d, k_from_run0, copy_of=None, change=None):
    """Insert a copy of data row (run start + k_from_run0) [or of row copy_of] right after that row; change(hdr, row)
    may edit the copy. Returns the 1-based data line number of the inserted line."""
    pos = {}

    def fn(hdr, body):
        i0, _ = run_rows(hdr, body)
        k = i0 + k_from_run0
        src = list(body[k if copy_of is None else i0 + copy_of])
        if change:
            change(hdr, src)
        body.insert(k + 1, src)
        pos['line'] = k + 2
    edit_log(d, fn)
    return pos['line']


def test_dup_line_marked_and_byte_equal_is_dropped(tmp_path):
    """run06 (Pi on LAN + WiFi): the logger receives a datagram twice and marks the copy 'duplicate'; the reader drops
    it and counts it, and every check and the F5 equal those of the log without the copy."""
    d, _ = make(tmp_path, 'HC_goc_1', cho='goc')
    write_segments(d)
    L0 = ca.load_log(os.path.join(d, 'log.csv'))
    line = insert_after(d, 400)
    write_segments(d, {line: 'duplicate'})
    L = ca.load_log(os.path.join(d, 'log.csv'))
    assert L0['n_dong_trung_bo'] == 0 and L['n_dong_trung_bo'] == 1 and L['n'] == L0['n']
    assert L['seq_trung_bo'] == [int(L0['log_seq'][list(L0['run_active']).index(1.0) + 400])]
    assert all(np.array_equal(L[c], L0[c]) for c in ca.LOG_COLS)
    ok, ck, _ = checks(d)
    assert all(ok.values()), [c for c in ck if not c[1]]
    assert 'bo 1 dong trung byte' in dict((n, x) for n, _, x in ck)['log_seq_lien']
    _, doan = run_cua_so(d)
    assert doan['n_dong_trung_bo'] == 1 and doan['seq_trung_bo'] == L['seq_trung_bo']
    out = str(tmp_path / 'f7.json')
    ca.main(tinh_args(d, 'goc', '--out', out))
    r = json.load(open(out, encoding='utf-8'))
    assert [x['n_dong_trung_bo'] for x in r['luot_thu']] == [1]


def test_dup_line_without_segments_file_is_kept_and_fails(tmp_path):
    """No logger segments file: nothing is dropped (L92: only a line the logger marked); log_seq_lien fails as before."""
    d, _ = make(tmp_path, 'HC_goc_1', cho='goc')
    insert_after(d, 400)
    L = ca.load_log(os.path.join(d, 'log.csv'))
    assert L['n_dong_trung_bo'] == 0
    ok, _, _ = checks(d)
    assert ok['log_seq_lien'] is False


@pytest.mark.parametrize('status', ['ok', 'conflict'])
def test_same_seq_other_content_is_kept_and_fails(tmp_path, status):
    """A repeated log_seq whose content differs (logger 'conflict', or not marked) is kept: log_seq_lien fails."""
    d, _ = make(tmp_path, 'HC_goc_1', cho='goc')

    def change(hdr, row):
        row[col(hdr, 'yaw_rad')] = str(float(row[col(hdr, 'yaw_rad')]) + 0.01)
    line = insert_after(d, 400, change=change)
    write_segments(d, {line: status})
    L = ca.load_log(os.path.join(d, 'log.csv'))
    assert L['n_dong_trung_bo'] == 0
    ok, _, _ = checks(d)
    assert ok['log_seq_lien'] is False


def test_marked_duplicate_not_equal_to_line_before_is_refused(tmp_path):
    """Marked 'duplicate' but its bytes differ from the line kept just before (other content, or a copy of an older
    line): AError (code 2), nothing dropped silently."""
    d, _ = make(tmp_path, 'HC_goc_1', cho='goc')

    def change(hdr, row):
        row[col(hdr, 'yaw_rad')] = str(float(row[col(hdr, 'yaw_rad')]) + 0.01)
    line = insert_after(d, 400, change=change)
    write_segments(d, {line: 'duplicate'})
    with pytest.raises(ca.AError, match='not byte-equal'):
        ca.load_log(os.path.join(d, 'log.csv'))
    d2, _ = make(tmp_path, 'HC_goc_2', cho='goc')
    line = insert_after(d2, 400, copy_of=398)                                # an older line, not the one before
    write_segments(d2, {line: 'duplicate'})
    with pytest.raises(ca.AError, match='not byte-equal'):
        ca.load_log(os.path.join(d2, 'log.csv'))


def test_segments_file_not_matching_the_log_is_refused(tmp_path):
    """A segments file with another row count, or a row whose log_seq is not the CSV line's: AError."""
    d, _ = make(tmp_path, 'HC_goc_1', cho='goc')
    write_segments(d)
    insert_after(d, 400)                                                     # log grew, segments did not
    with pytest.raises(ca.AError, match='rows for'):
        ca.load_log(os.path.join(d, 'log.csv'))
    d2, _ = make(tmp_path, 'HC_goc_2', cho='goc')
    write_segments(d2)
    p = os.path.join(d2, 'log.csv.segments.csv')
    lines = open(p, encoding='utf-8').read().splitlines()
    f = lines[5].split(',')
    f[1] = str(int(f[1]) + 1)
    lines[5] = ','.join(f)
    open(p, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    with pytest.raises(ca.AError, match='does not match'):
        ca.load_log(os.path.join(d2, 'log.csv'))
