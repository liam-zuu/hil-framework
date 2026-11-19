"""Tests of the relative frame (tools/c7_tuong_doi.py and c7_score --tuong-doi; design CODE_CAM v5, rules J v6 2, 5)
on SYNTHETIC runs (not a measurement). A run of test_c7_score (car on the lane, lane frame) is carried to the floor by
a rigid motion: start centre P_TC, axis angle PHI0, the tag turned A_TAG from the body axis. The relative scorer must
give the numbers of the absolute scorer on the lane-frame run (J v6 2: results do not change under a rigid motion of
the floor; the tag-to-axis angle cancels in changes of the tag heading)."""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import c7_chuoi_khung as ck  # noqa: E402
import c7_score as cs  # noqa: E402
import c7_tuong_doi as ctd  # noqa: E402
import rq3_log  # noqa: E402
import test_c7_score as B  # noqa: E402

P_TC, PHI0, A_TAG, BETA = (0.3, -0.2), 0.4, 0.2, 0.05


@pytest.fixture(autouse=True)
def cmd_ok(monkeypatch):
    def fake(rows, task, ev, has_age, sched, bao, ok):
        return dict(flags=[], notes=[], cond2={}, cond3={}, cond4={})
    monkeypatch.setattr(cs.ocp, 'analyse', fake)


def to_floor(task, rows, p_tc=P_TC, phi=PHI0, a_tag=A_TAG, psi_extra=None):
    """(floor frame rows, chain rows) of lane-frame rows (make_frames with k_start 0)."""
    psi_dd = cs.HEAD0 if task.startswith('O1-F') else 0.0
    out, chain = [], []
    for r in rows:
        bx = math.cos(-psi_dd) * r['x'] - math.sin(-psi_dd) * r['y']
        by = math.sin(-psi_dd) * r['x'] + math.cos(-psi_dd) * r['y']
        px = p_tc[0] + math.cos(phi) * bx - math.sin(phi) * by
        py = p_tc[1] + math.sin(phi) * bx + math.cos(phi) * by
        lp = r['psi'] + (psi_extra(r) if psi_extra else 0.0)
        tag = float(ck.wrap(phi + (lp - psi_dd) + a_tag))
        f = dict(r, x=px, y=py, psi=tag, tag_x=px, tag_y=py)
        out.append(f)
        ok = bool(r['valid'])
        chain.append(dict(frame_id=int(r['idx']), t_cam=float(r['t_s']), tag_x=px, tag_y=py, tag_psi=tag,
                          dpsi_a6=0.0, psi_ma_p=tag, p_x=px, p_y=py, u_q_m=0.0005, cong_ok=ok, a6_ok=ok, q_ok=ok,
                          ly_do='', trang_thai='khoa' if ok else 'chan_doan'))
    return out, chain


def ref_of(frames, chain, lo=-3.2, hi=-0.2):
    """A reference over the still frames at command time [lo, hi) (t_log = t_cam + OFFSET, t0 = T0)."""
    ids = [c['frame_id'] for f, c in zip(frames, chain)
           if c['cong_ok'] and lo <= f['t_s'] + B.OFFSET - B.T0 < hi]
    pre = dict(t_ref=B.T0 - B.OFFSET + hi, khung_ma_hop_le=ids,
               ket_luan=dict(dat=True, ly_do=[]))
    return pre, ids


def rel_for(task, frames, chain, truc=True, beta_ok=True, tam=False, neo=None, u_beta_cam=0.02, kt=None):
    pre, ids = ref_of(frames, chain)
    ll = dict(schema='lan_lap_v1', beta_deg=math.degrees(BETA), u_beta_deg=0.01, cung_lan_lap=True,
              trang_thai={'beta': 'dat' if beta_ok else 'chua_ket_luan', 'delta_c_B': 'dat'},
              tam_doi_trong_luot=tam, khung_chung={'pre': ids if neo is None else neo})
    kt = kt or dict(ctd.KHUNG_TC_DE_XUAT, nguon='test')
    tc = ctd.tham_chieu(pre, chain, ll, kt, 0.10, 1 / B.FPS)
    tx = None if not truc else dict(schema='truc_xe_v1', cho=ctd.cho_of(task), trang_thai='khoa',
                                    phi_hat_chuan_k_deg=math.degrees(PHI0 - BETA), u_phi_hat_chuan_k_deg=0.02)
    return cs.build_rel(task, chain, tc, tx, ll, u_beta_cam)


def score(tmp_path, task, frames, rel=None, log_kw=None, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    hs = B.hs_file(tmp_path)
    log = B.make_log(tmp_path / 'esp32.csv', task, (0.0, 0.0, 0.0), **(log_kw or {}))
    return cs.score(frames, log, task, hs, offset_s=B.OFFSET, offset_reason='test', offset_anchor_s=0.0,
                    offset_err_s=0.0, rel=rel, **kw)


def gate_ok(out):
    g = out['gate_v3']
    if out.get('gate_rq3', 'dat') != 'dat':                    # t_som case: rq3_log has no field for it yet
        assert g['g1'].get('ca_t_som') is True
        return g
    rq3_log.validate_gate(g, 'HW')
    rq3_log.check_outcome_vs_gate(out['outcome'], g, 'HW', out['fail_cause'])
    return g


@pytest.mark.parametrize('task,lateral', [('O1-F05', 0.03), ('O1-F05', 0.12), ('O1-K1', 0.04), ('O1-K2', 0.02)])
def test_relative_equals_absolute_under_rigid_motion(tmp_path, task, lateral):
    # the lateral offset grows after the start, so the start pose is on the lane (a constant offset along the lane
    # normal is not a rigid motion: relative and absolute differ there by design)
    lane_rows = B.make_frames(task, lat_fn=lambda t: lateral * min(1.0, max(0.0, t)))
    out_a, _ = score(tmp_path / 'a', task, lane_rows, k_start=None if task.startswith('O1-F') else (0.0, 0.0, 0.0),
                     allow_final_set=not task.startswith('O1-F'))
    fl, chain = to_floor(task, lane_rows)
    out_r, rows_r = score(tmp_path / 'r', task, fl, rel=rel_for(task, fl, chain),
                          allow_final_set=not task.startswith('O1-F'))
    ga, gr = gate_ok(out_a), gate_ok(out_r)
    assert out_r['outcome'] == out_a['outcome']
    for c in ('g2', 'g3', 'g4', 'g5'):
        assert abs(gr[c]['q'] - ga[c]['q']) < 1e-6, c
    assert gr['g1']['state'] == ga['g1']['state']
    k = out_r['khung_tuong_doi']
    assert k['can_cu_vi_tri'] and k['phi_ok'] and abs(k['phi_rad'] - PHI0) < 1e-12
    assert all({'x_fw', 'dpsi_ma', 'frame_id', 'q_ok', 'nghi_ngo_vi_tri', 'U_p_noi_m', 'nghi_ngo_yaw'} <= set(r)
               for r in rows_r)
    assert k['esp32_sha256'] and k['gio_sha256'] is None


def test_tag_angle_and_floor_motion_cancel(tmp_path):
    lane_rows = B.make_frames('O1-F05', lateral=0.05)
    res = []
    for i, (p, phi, a) in enumerate((((0.3, -0.2), 0.4, 0.2), ((-1.0, 2.0), -2.5, -1.3))):
        fl, chain = to_floor('O1-F05', lane_rows, p, phi, a)
        global PHI0
        old, PHI0 = PHI0, phi
        try:
            out, _ = score(tmp_path / str(i), 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain))
        finally:
            PHI0 = old
        res.append(gate_ok(out))
    for c in ('g2', 'g3', 'g4', 'g5'):
        assert abs(res[0][c]['q'] - res[1][c]['q']) < 1e-6


def test_g3_zero_at_start_off_the_floor_origin(tmp_path):
    """J v6 3 G3 example: the car still exactly at p_tc, 0.20 m from the floor origin, gives e_trk = 0 at t = 0."""
    lane_rows = B.make_frames('O1-K1')
    fl, chain = to_floor('O1-K1', lane_rows, p_tc=(0.2, 0.0))
    out, rows = score(tmp_path, 'O1-K1', fl, rel=rel_for('O1-K1', fl, chain), allow_final_set=True)
    r0 = min(rows, key=lambda r: abs(r['t']))
    assert r0['e_trk'] < 1e-4 and abs(r0['x_san'] - 0.2) < 1e-4       # first frame after the edge: 1/60 s of motion


def test_phi_lever_arm_in_u_p(tmp_path):
    lane_rows = B.make_frames('O1-K1')
    fl, chain = to_floor('O1-K1', lane_rows)
    rel = rel_for('O1-K1', fl, chain, u_beta_cam=0.02)
    out, rows = score(tmp_path, 'O1-K1', fl, rel=rel, allow_final_set=True)
    u = rel['u_phi']
    assert abs(u - math.radians(1.25 * (0.02 + 0.01) + 0.02)) < 1e-15
    r = max(rows, key=lambda r: r['r_k'])
    assert abs(r['U_phi_k'] - 2 * r['r_k'] * math.sin(u / 2)) < 1e-12
    assert abs(r['U_p_k'] - (0.005 + r['U_phi_k'])) < 1e-12


def test_without_u_beta_cam_positions_not_concluded(tmp_path):
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    out, _ = score(tmp_path, 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain, u_beta_cam=None))
    g = gate_ok(out)
    assert g['g2']['state'] == 'chua_ket_luan' and g['g5']['state'] == 'dat'
    assert out['outcome'] == 'chua_ket_luan'


@pytest.mark.parametrize('why', ['truc', 'beta', 'tam_doi_null'])
def test_t_som_case_g1_g4_not_concluded_g5_scored(tmp_path, why):
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    rel = rel_for('O1-F05', fl, chain, truc=why != 'truc', beta_ok=why != 'beta',
                  tam=None if why == 'tam_doi_null' else False)
    assert not rel['can_cu']
    out, _ = score(tmp_path, 'O1-F05', fl, rel=rel)
    g = gate_ok(out)
    assert all(g[c]['state'] == 'chua_ket_luan' for c in ('g2', 'g3', 'g4')) and g['g1']['state'] == 'chua_ket_luan'
    assert g['g5']['state'] == 'dat' and out['outcome'] == 'chua_ket_luan'
    assert g['g1']['t_som_s'] >= out['T_cmd'] - 1e-9 and g['g1']['t_done'] == [None, None]
    assert 't_som_s' not in out and out['khung_tuong_doi']['t_som_s'] == g['g1']['t_som_s']
    assert all(g[c]['n_undef'] > 0 for c in ('g2', 'g3', 'g4'))


def test_t_som_case_g5_violation_before_t_som_is_khong_dat(tmp_path):
    lane_rows = B.make_frames('O1-F05')
    # the body really turns 15 deg more for 1 s: camera and IMU agree (else the frames are yaw-suspect)
    bump = lambda r: math.radians(15.0) if 2.0 <= r['t_s'] + B.OFFSET - B.T0 <= 3.0 else 0.0
    fl, chain = to_floor('O1-F05', lane_rows, psi_extra=bump)
    out, _ = score(tmp_path, 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain, truc=False),
                   log_kw=dict(yaw_err=lambda t: math.radians(15.0) if 2.0 <= t <= 3.0 else 0.0))
    g = gate_ok(out)
    assert g['g5']['state'] == 'khong_dat' and out['outcome'] == 'khong_dat'


def test_t_som_case_g5_violation_after_t_som_only_chua_ket_luan(tmp_path):
    lane_rows = B.make_frames('O1-F05', post=4.0)
    late = lambda r: math.radians(10.0) if r['t_s'] + B.OFFSET - B.T0 >= 13.65 + 1.8 else 0.0
    fl, chain = to_floor('O1-F05', lane_rows, psi_extra=late)
    out, _ = score(tmp_path, 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain, truc=False))
    g = gate_ok(out)
    assert g['g1']['t_som_s'] < 13.65 + 1.8
    assert g['g5']['state'] == 'chua_ket_luan' and out['outcome'] == 'chua_ket_luan'


def test_plate_moved_g5_only_from_imu(tmp_path):
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    out, _ = score(tmp_path, 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain, tam=True))
    g = gate_ok(out)
    assert g['g5']['state'] == 'chua_ket_luan'                  # IMU on the lane: no certain violation, never dat


def test_plate_moved_imu_certain_violation_is_khong_dat(tmp_path):
    lane_rows = B.make_frames('O1-K1')
    fl, chain = to_floor('O1-K1', lane_rows)
    hs = B.hs_file(tmp_path)
    log = B.make_log(tmp_path / 'esp32.csv', 'O1-K1', (0.0, 0.0, 0.0),
                     yaw_err=lambda t: math.radians(10.0) if 2.0 <= t <= 3.0 else 0.0)
    out, _ = cs.score(fl, log, 'O1-K1', hs, offset_s=B.OFFSET, offset_reason='test', offset_anchor_s=0.0,
                      offset_err_s=0.0, allow_final_set=True, rel=rel_for('O1-K1', fl, chain, tam=True))
    g = gate_ok(out)
    assert g['g5']['state'] == 'khong_dat'                      # J5-6: IMU 10 deg, bound about 1 deg


def test_f08_g5_report_only(tmp_path):
    lane_rows = B.make_frames('O1-F08')
    fl, chain = to_floor('O1-F08', lane_rows)
    out, _ = score(tmp_path, 'O1-F08', fl, rel=rel_for('O1-F08', fl, chain))
    assert 'g5' in out['report_only']


def test_reference_window_lost(tmp_path):
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    pre, ids = ref_of(fl, chain)
    kt = dict(ctd.KHUNG_TC_DE_XUAT, nguon='test')
    ll = dict(beta_deg=0.0, u_beta_deg=0.0, trang_thai={'beta': 'dat'}, tam_doi_trong_luot=False,
              khung_chung={'pre': ids[:20]})                        # an intersection with samples, too few
    tc = ctd.tham_chieu(pre, chain, ll, kt, 0.10, 1 / B.FPS)
    assert tc['trang_thai'] == 'hong_cua_so_tc' and tc['tap'] == 'giao'
    out, _ = score(tmp_path, 'O1-F05', fl, rel=cs.build_rel('O1-F05', chain, tc, None, ll))
    assert out['outcome'] == 'hong_he_do' and out['hong_cua_so_tc'] and out['gate_v3'] is None


def test_empty_intersection_is_mat_neo_not_lost():
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    pre, ids = ref_of(fl, chain)
    tc = ctd.tham_chieu(pre, chain, dict(khung_chung={'pre': [10 ** 9]}), dict(ctd.KHUNG_TC_DE_XUAT, nguon='t'),
                        0.10, 1 / B.FPS)
    assert tc['tap'] == 'ma_mat_neo' and tc['mat_neo'] and tc['trang_thai'] == 'dat' and tc['n'] == len(ids)
    assert tc['phep_gop'] == 'trung_binh_go_vong' and tc['psi_tan_rad'] >= 0.0 and tc['p_tc_m'] is not None
    assert abs(tc['t_tc_s'] - np.mean([c['t_cam'] for c in chain if c['frame_id'] in set(ids)])) < 1e-9
    phi, u, ok, why = ctd.phi_beta('O1-F05', dict(schema='truc_xe_v1', cho='goc', trang_thai='khoa',
                                                  phi_hat_chuan_k_deg=1.0, u_phi_hat_chuan_k_deg=0.1),
                                   dict(beta_deg=0.0, u_beta_deg=0.0, trang_thai={'beta': 'dat'}), tc, 0.02)
    assert not ok and any('mat_neo' in w for w in why)                 # J 2: beta needs the anchors


def test_phi_beta_place_and_parts():
    tc = dict(tap='giao')
    tx = dict(schema='truc_xe_v1', cho='K1', trang_thai='khoa', phi_hat_chuan_k_deg=10.0, u_phi_hat_chuan_k_deg=0.1)
    ll = dict(beta_deg=2.0, u_beta_deg=0.05, trang_thai={'beta': 'dat'})
    with pytest.raises(ValueError, match='place'):
        ctd.phi_beta('O1-K2', tx, ll, tc)
    phi, u, ok, why = ctd.phi_beta('O1-K1', tx, ll, tc, None)
    assert ok and abs(phi - math.radians(12.0)) < 1e-15 and math.isinf(u)
    phi, u, ok, why = ctd.phi_beta('O1-K1', tx, ll, tc, 0.03)
    assert abs(u - math.radians(1.25 * 0.15 + 0.03)) < 1e-15


def test_t_som_wz_rules():
    """J v6 5 example K1: a frame at 4.8 s with |wz| >= w_d rules out completion at 5.0 s; J5-4: not when one yaw
    sample of its stencil failed the yaw check."""
    t = np.arange(0.0, 7.0, 1 / 60)
    wz = np.where(np.abs(t - 4.8) < 1e-9 + 0.5 / 60, 1.0, 0.0)
    nan = np.full(len(t), np.nan)
    no = np.zeros(len(t), bool)
    ts = ctd.t_som(t, no, nan, np.ones(len(t), bool), wz, 0.01, 0.01 / 0.24, 1.0, 0.1, 5.0, 7.0)
    assert ts > 5.79
    yaw_ok = np.ones(len(t), bool)
    yaw_ok[int(round(4.75 * 60))] = False
    usable = ctd.wz_usable(yaw_ok, 6)
    ts2 = ctd.t_som(t, no, nan, usable, wz, 0.01, 0.01 / 0.24, 1.0, 0.1, 5.0, 7.0)
    assert abs(ts2 - 5.0) < 1 / 60 + 1e-9
    assert ctd.t_som(t, no, nan, no, wz, 0.01, 1.0, 1.0, 0.1, 5.0, 7.0) <= 5.0 + 1 / 60


def _still_chain(n=400, creep=0.0, dt=1 / 60, gap=None):
    t = np.arange(n) * dt
    frames, chain = [], []
    for k, tk in enumerate(t):
        if gap and gap[0] <= tk < gap[1]:
            continue
        x = 0.5 + creep * tk / t[-1]
        frames.append(dict(idx=k, t_s=float(tk)))
        chain.append(dict(frame_id=k, cong_ok=True, tag_x=x, tag_y=0.1, tag_psi=0.3))
    return frames, chain


def _log(t0, a, b):
    return [dict(_t=(t0 + x) * 1e6, _seq=i) for i, x in enumerate(np.arange(a, b, 0.1))]


@pytest.mark.parametrize('creep,gap,ok', [(0.0, None, True), (0.0125, None, False), (0.0, (4.0, 4.3), False)])
def test_cua_so_gates(creep, gap, ok):
    frames, chain = _still_chain(creep=creep, gap=gap)
    kt = dict(ctd.KHUNG_TC_DE_XUAT, nguon='t')
    rows = ctd.cua_so(frames, chain, _log(0.0, -1.0, 12.0), 6.2e6, [0.0], [0.05], kt, 0.10, 2.0, 'O1-F05')
    pre = rows[0]
    assert pre['loai'] == 'pre' and pre['ket_luan']['dat'] is ok
    assert pre['khung_ma_hop_le_sha256'] == ctd.ids_sha(pre['khung_ma_hop_le'])
    assert all(f['pts_s'] + 1 / 60 < pre['t_ref'] for f in pre['khung'])


def test_cua_so_log_must_cover():
    frames, chain = _still_chain()
    kt = dict(ctd.KHUNG_TC_DE_XUAT, nguon='t')
    rows = ctd.cua_so(frames, chain, _log(0.0, 4.0, 12.0), 6.2e6, [0.0], [0.05], kt, 0.10, 2.0, 'O1-F05')
    assert not rows[0]['ket_luan']['log_phu'] and not rows[0]['ket_luan']['dat']


def test_rel_heading_branch():
    p = np.array([3.1, -3.1, -3.0, np.nan, -2.9])
    psi, d = ctd.rel_heading(p, 3.1, 0.0)
    assert abs(d[1] - (2 * math.pi - 6.2)) < 1e-12 and math.isnan(d[3]) and abs(d[4] - (2 * math.pi - 6.0)) < 1e-12


def test_cli_relative_inputs_need_the_flag(tmp_path):
    lane_rows = B.make_frames('O1-F05')
    p = tmp_path / 'frames.csv'
    cs.cm.write_frames(str(p), lane_rows) if hasattr(cs.cm, 'write_frames') else None
    assert cs.main(['--frames', str(p), '--esp32', 'x', '--task', 'O1-F05', '--hang-so', 'x', '--a6', 'y']) == 2
