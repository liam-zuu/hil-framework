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


def test_camera_heading_drift_inside_index_or_flagged(tmp_path):
    d, truth = make(tmp_path, 'HC_K2_1', cho='K2', cam_bias=lambda t: math.radians(0.010) * max(t, 0.0),
                    noise_psi_deg=0.02)
    r = result(d, 'K2')
    err = abs(math.radians(r['a_k_deg']) - truth['a_true'])
    assert err <= math.radians(r['u_a_k_deg']) or r['trang_thai'] == 'chua_ket_luan'


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
    # one long leg with a camera heading step of 0.4 deg at its start window: spread > 0.3 deg -> chua_ket_luan
    def bias(t):
        return math.radians(0.4) if 16.0 < t < 21.0 else 0.0
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
    assert not (ok['traj_age'] and ok['het_han_firmware'])


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
    """HCC1-9: a 1 s camera gap inside a long leg: eps cannot be computed -> the gate fails (not counted as 0)."""
    d, truth = make(tmp_path, 'HC_K2_1', cho='K2', noise_psi_deg=0.02)
    A = ca.load_attempt(d)
    ck, info = ca.kiem(A)
    an = ca.analyse(A, drop(frames(d), 12.0, 13.0, truth['D']), info)
    leg = [lg for lg in an['legs'] if lg['dai']][0]
    assert not math.isfinite(leg['eps'])
    r = ca.place_result('K2', [('x', an)], 'x', e_hieu=E(0.0), q=[0, 0], du_m=0.0, neo=0.0, psi_cuc=0.0)
    assert not r['cong']['yaw_dong']


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
    a6 = write_json(str(tmp_path / 'a6.json'), dict(schema='c8b_a6_v2'))
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
