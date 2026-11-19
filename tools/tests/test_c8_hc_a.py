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
    r = ca.place_result(cho, [(os.path.basename(d), an)], os.path.basename(d), e_hieu=[0.002], q=[0.004, -0.003],
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
    body = dict(doan)
    sha = body.pop('sha256')
    assert sha == __import__('hashlib').sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    with pytest.raises(FileExistsError):                                        # never overwritten
        ca.cmd_cua_so(type('a', (), dict(dir=d, frames=os.path.join(d, 'frames.csv'), track=os.path.join(d, 'track.json'),
                                         clip=None, dets=None, session_sha256=None, a6=None, xoay=None))())


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
    return ca.place_result(cho, ans, os.path.basename(d), e_hieu=[0.0], q=[0.0, 0.0], du_m=0.0, neo=0.0, psi_cuc=psi_cuc)


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
    r0 = ca.place_result('K1', [(os.path.basename(d), analyse(d))], os.path.basename(d), e_hieu=[0.0], q=[0, 0],
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
    write_events(d, [start, end.format(c=0).replace('blk_ticks=0', 'blk_ticks=3')])
    assert not checks(d)[0]['khong_cat_sieu_am']


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
    assert kinds == ['vx'] * 6 + ['wz'] * 2 and u_D <= ca.U_D_MAX_S


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
    r = ca.place_result('K2', [('x', an)], 'x', e_hieu=[0.0], q=[0, 0], du_m=0.0, neo=0.0, psi_cuc=0.0)
    assert not r['cong']['yaw_dong']


def test_rv_repeat_015s_during_turn_fails(tmp_path):
    """HCC1-9 (Codex): RV repeating 0.15 s while turning still gave corr 0.945."""
    d, _ = make(tmp_path, cho='K1', rv_freeze=(62.0, 62.15))
    ok, ck, _ = checks(d)
    e = [c for c in ck if c[0] == 'imu_e_tuong_quan']
    assert not e[0][1], e
