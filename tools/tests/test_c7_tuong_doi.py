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
CLIP12 = 'c' * 64                     # sha of the 12b clip named on both sides (truc_xe_v1, lan_lap_v1 dau.chuan)


def ll_of(task, ids, beta_ok=True, tam=False, **over):
    """lan_lap_v1 of one run (NHACTAM F8), bound to the 12b run '12b_1' of placement L0 (J v6 2)."""
    d = dict(schema='lan_lap_v1', luot_id='L_1', cho=ctd.cho_of(task), lan_lap_id='L0', chuan_12b='12b_1',
             dau={'chuan': {'clip_sha256': CLIP12}}, beta_deg=math.degrees(BETA), u_beta_deg=0.01, cung_lan_lap=True,
             trang_thai={'beta': 'dat' if beta_ok else 'chua_ket_luan', 'delta_c_B': 'dat'},
             tam_doi_trong_luot=tam, khung_chung={'pre': ids})
    d.update(over)
    return d


def tx_of(task, phi_hat_deg, u=0.02, **over):
    """truc_xe_v1 of the 12b run '12b_1' (HIEUCHUAN F7)."""
    d = dict(schema='truc_xe_v1', mien_hieu_luc=dict(cho=ctd.cho_of(task), thu_id_cap='12b_1', lan_lap_id=None),
             clip_sha256=CLIP12, trang_thai='khoa', phi_hat_chuan_k_deg=phi_hat_deg, u_phi_hat_chuan_k_deg=u)
    d.update(over)
    return d


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


def rel_for(task, frames, chain, truc=True, beta_ok=True, tam=False, neo=None, u_beta_cam=0.02, kt=None, ll_over=None,
            tx_over=None):
    pre, ids = ref_of(frames, chain)
    ll = ll_of(task, ids if neo is None else neo, beta_ok, tam, **(ll_over or {}))
    kt = kt or dict(ctd.KHUNG_TC_DE_XUAT, nguon='test')
    tc = ctd.tham_chieu(pre, chain, ll, kt, 0.10, 1 / B.FPS)
    tx = None if not truc else tx_of(task, math.degrees(PHI0 - BETA), **(tx_over or {}))
    return cs.build_rel(task, chain, tc, tx, ll, u_beta_cam)


def latch_cols(path, edit=None):
    """Add the heading-latch columns of the firmware log (head_on 1, psi_ref 0) to a make_log CSV; edit(rows, i_edge)
    may change rows (i_edge: first run_active = 1 row)."""
    import csv
    with open(path, encoding='utf-8') as f:
        head = f.readline()
        rows = list(csv.DictReader(f))
    for r in rows:
        r.update(head_on='1', psi_ref='0.0000')
    if edit:
        edit(rows, next(i for i, r in enumerate(rows) if r['run_active'] == '1'))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        f.write(head)
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ev = str(path) + '.events.txt'
    if not os.path.exists(ev):           # a run not blocked (O1 'Luot khong bi chan'): no operator cause (CT1-13)
        us_events(ev, int(rows[0]['log_seq']), int(rows[-1]['log_seq']))
    return path


def us_events(ev, first, last, boot='00000001', on='0', changes='0', blk='0', extra=''):
    """'#us,ev=start' / '#us_end' of one run as the firmware writes them (AGV 5bde4c5 us_state.c), events file."""
    with open(ev, 'w', encoding='utf-8') as f:
        f.write(f'2026-11-19T10:00:00 #gain,id=0,applied_n=1,ctrl=pid,first_seq={first},line_drop=0,boot={boot}\n'
                f'2026-11-19T10:00:00 #us,ev=start,boot={boot},on={on},swb=0,first_seq={first},line_drop=0\n'
                f'2026-11-19T10:00:30 #us_end,boot={boot},on={on},swb=0,changes={changes},blk_ticks={blk},'
                f'cut_ticks=0,ticks={last - first + 1},last_data_seq={last},line_drop=0\n' + extra)


def score(tmp_path, task, frames, rel=None, log_kw=None, latch=None, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    hs = B.hs_file(tmp_path)
    log = latch_cols(B.make_log(tmp_path / 'esp32.csv', task, (0.0, 0.0, 0.0), **(log_kw or {})), latch)
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
    k = out_r['khung_tuong_doi_v1']
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
    assert 't_som_s' not in out and out['khung_tuong_doi_v1']['t_som_s'] == g['g1']['t_som_s']
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
    log = latch_cols(B.make_log(tmp_path / 'esp32.csv', 'O1-K1', (0.0, 0.0, 0.0),
                                yaw_err=lambda t: math.radians(10.0) if 2.0 <= t <= 3.0 else 0.0))
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
    phi, u, ok, why = ctd.phi_beta('O1-F05', tx_of('O1-F05', 1.0, 0.1),
                                   ll_of('O1-F05', [], beta_deg=0.0, u_beta_deg=0.0), tc, 0.02)
    assert not ok and any('mat_neo' in w for w in why)                 # J 2: beta needs the anchors


def test_phi_beta_place_and_parts():
    tc = dict(tap='giao')
    tx = tx_of('O1-K1', 10.0, 0.1)
    ll = ll_of('O1-K1', [], beta_deg=2.0, u_beta_deg=0.05)
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


# ---------------------------------------------------------------- Codex round 1 (CT1-*): each test fails on 921e839
def test_ct1_1_f08_g5_report_only_does_not_decide(tmp_path):
    """CT1-1, CT1-20: F08 with G1-G4 passing and a CERTAIN G5 violation (camera and IMU turn together 15 deg, so the
    frames are not yaw-suspect): G5 keeps its numbers and lo > 7.5 deg, but the run is not 'khong_dat'; the gate
    leaves G5 out of its decide set and passes rq3_log with that required set."""
    lane_rows = B.make_frames('O1-F08')
    bump = lambda r: math.radians(15.0) if 2.0 <= r['t_s'] + B.OFFSET - B.T0 <= 3.0 else 0.0
    fl, chain = to_floor('O1-F08', lane_rows, psi_extra=bump)
    out, _ = score(tmp_path, 'O1-F08', fl, rel=rel_for('O1-F08', fl, chain),
                   log_kw=dict(yaw_err=lambda t: math.radians(15.0) if 2.0 <= t <= 3.0 else 0.0))
    g = out['gate_v3']
    assert g['g5']['lo'] > 7.5 and g['g5']['state'] == 'khong_dat'
    assert 'g5' not in g['decide'] and g['report_only'] == ['g5'] and g['gate_file_sha256'] == cs.J_V6_SHA256
    rq3_log.validate_gate(g, 'HW', g['required'])
    assert out['outcome'] != 'khong_dat' and all(g[c]['state'] != 'khong_dat' for c in ('g2', 'g3', 'g4'))


@pytest.mark.parametrize('case', ['other_12b', 'other_clip', 'no_common_key', 'no_lan_lap_id'])
def test_ct1_2_phi_needs_the_12b_of_this_placement(case):
    """CT1-2 (CODE 19/11): phi only from the truc_xe_v1 of the 12b that lan_lap_v1 chuan_12b names, same session
    (an identity key on both sides, equal), with a placement id."""
    tc = dict(tap='giao')
    tx, ll = tx_of('O1-K1', 10.0, 0.1), ll_of('O1-K1', [], beta_deg=2.0, u_beta_deg=0.05)
    assert ctd.phi_beta('O1-K1', tx, ll, tc, 0.03)[2]
    if case == 'other_12b':
        tx['mien_hieu_luc']['thu_id_cap'] = '12b_0'
    elif case == 'other_clip':
        tx['clip_sha256'] = 'd' * 64
    elif case == 'no_common_key':
        del tx['clip_sha256']
    else:
        ll['lan_lap_id'] = None
    phi, u, ok, why = ctd.phi_beta('O1-K1', tx, ll, tc, 0.03)
    assert not ok and phi is None and math.isinf(u) and why


def test_ct1_2_frames_sha_of_the_12b_run_binds():
    """CT1-2: the frames sha of the 12b run in truc_xe_v1 nguon ('<thu_id>:frames') binds to dau.chuan.frames_sha256."""
    tx = tx_of('O1-K1', 10.0, 0.1, nguon=[dict(vai_tro='12b_1:frames', sha256='e' * 64)])
    del tx['clip_sha256']
    ll = ll_of('O1-K1', [], dau={'chuan': {'frames_sha256': 'e' * 64}})
    assert ctd.phi_beta('O1-K1', tx, ll, dict(tap='giao'), 0.03)[2]
    ll['dau']['chuan']['frames_sha256'] = 'f' * 64
    assert not ctd.phi_beta('O1-K1', tx, ll, dict(tap='giao'), 0.03)[2]


def test_ct1_3_lan_lap_bound_to_run_window_and_clock():
    """CT1-3: lan_lap_v1 of another run / place / window file, anchors outside the window, or another time source
    of the window than the scoring: refused when read."""
    pre = dict(luot_id='L_1', cho='K1', khung=[dict(id=i, pts_s=0.0) for i in range(10)], clip_sha256='a' * 64,
               offset_cfg=dict(offset_s=0.25, offset_err_s=0.0, offset_anchor_s='0'), gio_sha256=None,
               nguong={k: ctd.KHUNG_TC_DE_XUAT[k] for k in ctd.KHUNG_TC_DE_XUAT},
               hang_so_dung=dict(u_t_ghep_s=0.0, troi_ppm=436.0, t_trong_s=0.1))
    ll = ll_of('O1-K1', [1, 2], cua_so_v1_sha256='9' * 64)
    cs.bind_lan_lap(ll, pre, '9' * 64, 'O1-K1')
    for over, msg in ((dict(luot_id='L_2'), 'luot_id'), (dict(cho='K2'), 'cho'), (dict(cua_so_v1_sha256='8' * 64),
                      'cua_so_v1_sha256'), (dict(khung_chung={'pre': [99]}), 'outside'),
                      (dict(dau={'pre': {'clip_sha256': 'b' * 64}}), 'clip_sha256')):
        with pytest.raises(ValueError, match=msg):
            cs.bind_lan_lap(dict(ll, **over), pre, '9' * 64, 'O1-K1')

    class A:
        gio_c8b, offset_s, offset_err_s, offset_anchor_s = None, 0.25, 0.0, '0'
    hs = dict(khung_tc=dict(ctd.KHUNG_TC_DE_XUAT, nguon='t'), trang_thai='de_xuat', u_t_ghep_s=0.0, troi_ppm=436.0,
              t_trong_s=0.1)
    cs.bind_clock(pre, A, hs)
    A.offset_s = 0.30
    with pytest.raises(ValueError, match='time source'):
        cs.bind_clock(pre, A, hs)


def test_ct1_3_missing_anchor_list_is_not_mat_neo():
    """CT1-3: lan_lap_v1 without khung_chung.pre (or no lan_lap_v1) is not a confirmed loss of every anchor: tap
    thieu_neo, the reference not concluded; an EMPTY list stays mat_neo."""
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    pre, ids = ref_of(fl, chain)
    kt = dict(ctd.KHUNG_TC_DE_XUAT, nguon='t')
    for ll in (None, dict(khung_chung={})):
        tc = ctd.tham_chieu(pre, chain, ll, kt, 0.10, 1 / B.FPS)
        assert tc['tap'] == 'thieu_neo' and not tc['mat_neo'] and tc['trang_thai'] == 'chua_ket_luan'
    tc = ctd.tham_chieu(pre, chain, dict(khung_chung={'pre': []}), kt, 0.10, 1 / B.FPS)
    assert tc['tap'] == 'ma_mat_neo' and tc['mat_neo']


def test_ct1_4_imu_aligned_on_the_reference_set(tmp_path):
    """CT1-4: the camera-IMU alignment uses the frames of the reference set and the circular-unwrapped means: an IMU
    offset only before that set (inside the older still window) does not move it."""
    task = 'O1-K1'
    lane_rows = B.make_frames(task)
    hs = B.hs_file(tmp_path)
    log = B.make_log(tmp_path / 'esp32.csv', task, (0.0, 0.0, 0.0),
                     yaw_err=lambda t: math.radians(2.0) if t < -1.5 else 0.0)
    t_log = np.array([r['t_s'] for r in lane_rows]) + B.OFFSET
    psi = np.array([r['psi'] for r in lane_rows])
    tv = t_log - B.T0
    mask = (tv >= -1.2) & (tv < -0.2)
    z = np.zeros(len(psi))
    _, rep_old = cs.yaw_bounds(t_log, psi, z, z, B.T0, hs, task, log, B.T0 + 7.0)
    _, rep = cs.yaw_bounds(t_log, psi, z, z, B.T0, hs, task, log, B.T0 + 7.0, align_mask=mask)
    assert rep['align_set'] == 'tham_chieu' and rep['compared']
    assert abs(rep['align_c_deg'] - math.degrees(B.YAW0)) < 1e-6 and rep['align_spread_deg'] < 1e-6
    assert abs(rep_old['align_c_deg'] - math.degrees(B.YAW0)) > 0.5          # the old window sees the offset


def test_ct1_5_t_som_speed_without_phi(tmp_path):
    """CT1-5 (J v6 5 example): K1 without phi, the car moving until about T_cmd: its speed (centre after A6, q) is
    known without phi, so t_som is not T_cmd but after a still t_d."""
    lane_rows = B.make_frames('O1-K1')
    fl, chain = to_floor('O1-K1', lane_rows)
    out, _ = score(tmp_path, 'O1-K1', fl, rel=rel_for('O1-K1', fl, chain, truc=False), allow_final_set=True)
    assert out['gate_v3']['g1']['t_som_s'] > 5.9


def test_ct1_5_moved_plate_gives_no_speed():
    """CT1-5: a plate that moved in the run: no frame speed (the tag shift is no car motion)."""
    lane_rows = B.make_frames('O1-K1')
    fl, chain = to_floor('O1-K1', lane_rows)
    rel = rel_for('O1-K1', fl, chain, tam=True)
    vi = [k for k, r in enumerate(fl) if r['valid']]
    R = cs.rel_pose(fl, vi, rel, cs.KTask('O1-K1', (0.0, 0.0, 0.0), end_incoming=True))
    assert R['tam_ok'].all() and not R['speed_def'].any()


def test_ct1_6_yaw_suspect_flag_from_the_comparison(tmp_path):
    """CT1-6 (Codex example): U_psi 5 deg, suspect threshold 1 deg, camera 2 deg off the IMU: the frames are
    yaw-suspect (bound 3 deg <= U_psi, so the old reading from the bound missed them)."""
    lane_rows = B.make_frames('O1-F05')
    off = lambda r: math.radians(2.0) if 2.0 <= r['t_s'] + B.OFFSET - B.T0 <= 3.0 else 0.0
    fl, chain = to_floor('O1-F05', lane_rows, psi_extra=off)
    tmp_path.mkdir(parents=True, exist_ok=True)
    hs = B.hs_file(tmp_path, U_psi_deg=5.0, yaw_nghi_deg=1.0)
    log = latch_cols(B.make_log(tmp_path / 'esp32.csv', 'O1-F05', (0.0, 0.0, 0.0)))
    out, rows = cs.score(fl, log, 'O1-F05', hs, offset_s=B.OFFSET, offset_reason='test', offset_anchor_s=0.0,
                         offset_err_s=0.0, rel=rel_for('O1-F05', fl, chain))
    mid = [r for r in rows if 2.1 <= r['t'] <= 2.9]
    assert mid and all(r['nghi_ngo_yaw'] == 1 and r['so_duoc_yaw'] == 1 for r in mid)


def test_ct1_7_frames_without_a6_are_undefined_not_raw(tmp_path):
    """CT1-7: frames that lose A6 inside the run give no raw tag pose to the tracker or G1: they come back as
    undefined frames (n_undef), the run cannot be 'dat'."""
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    rel = rel_for('O1-F05', fl, chain)
    lost = set()
    for c, f in zip(chain, fl):
        if 2.0 <= f['t_s'] + B.OFFSET - B.T0 < 2.05:
            c.update(a6_ok=False, psi_ma_p=float('nan'), p_x=float('nan'), p_y=float('nan'))
            lost.add(c['frame_id'])
    out, rows = score(tmp_path, 'O1-F05', fl, rel=rel)
    g = gate_ok(out)
    assert lost and all(g[c]['n_undef'] > 0 for c in ('g2', 'g3', 'g4', 'g5'))
    assert out['outcome'] == 'chua_ket_luan'
    lr = [r for r in rows if r['frame_id'] in lost]
    assert len(lr) == len(lost) and all(r['khung_thieu_tam'] == 1 and r['vi_tri_xac_dinh'] == 0 for r in lr)
    assert all(not math.isfinite(r['lo_g2']) for r in lr)


def test_ct1_8_k2_closing_corner_takes_the_incoming_side():
    """CT1-8 (Codex numbers): K2 at s = S, (x, y) = (0.05656854, -0.05656854) m, psi 0.127 rad, window [3.8, 4.0]:
    G4 96.996 mm by the incoming side (relative mode), 100.364 mm by side 0 (absolute mode, O1 3.7)."""
    x, y, psi = 0.05656854, -0.05656854, 0.127
    vals = []
    for inc in (True, False):
        lane = cs.KTask('O1-K2', (0.0, 0.0, 0.0), end_incoming=inc)
        s2, e, hd = lane.project(x, y, 3.8, 4.0)
        psi0, al, bl = lane.nominal(hd)
        vals.append(cs.o1_gate_eval.g4_value(e, psi, hd, psi0, al, bl))
        heads = {round(h, 9) for _, _, h in cs.k_jk(lane, x, y, 3.8, 4.0, 0.001)}
        if inc:
            assert heads == {round(float(lane.poly.head[-1]), 9)}
    assert abs(vals[0] - 0.096996) < 2e-6 and abs(vals[1] - 0.100364) < 2e-6


def test_ct1_9_g4_without_yaw_comparison_is_undefined(tmp_path):
    """CT1-9: no IMU yaw (frames not compared): G4 keeps q but no finite lo in the relative mode, so a lane offset
    above D gives G2 'khong_dat' but G4 only 'chua_ket_luan'."""
    lane_rows = B.make_frames('O1-F05', lat_fn=lambda t: 0.14 * min(1.0, max(0.0, t)))
    fl, chain = to_floor('O1-F05', lane_rows)
    out, _ = score(tmp_path, 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain), log_kw=dict(no_imu=True))
    g = out['gate_v3']
    assert not out['yaw_check']['compared']
    assert g['g2']['state'] == 'khong_dat' and g['g4']['state'] == 'chua_ket_luan' and g['g4']['q'] > 0.1


def test_ct1_11_shutter_branch_table(tmp_path):
    """CT1-11: a branch table of every level, all 'mot_khung', needs no alpha; a level missing: refused; the relative
    mode with --offset-s takes the shutter term of its branch (one frame) and names the branch."""
    for d in ('a', 'b'):
        (tmp_path / d).mkdir()
    hs = B.hs_file(tmp_path / 'a', man_trap=dict(nguon='t', nhanh={'O1-F02': 'mot_khung', 'O1-F05': 'mot_khung',
                                                                     'O1-F08': 'mot_khung', 'K': 'mot_khung'}))
    assert hs['man_trap']['alpha_r_s_px'] is None and cs.man_trap_of(hs, 'O1-K1', rel=True) is None
    with pytest.raises(ValueError, match='missing'):
        B.hs_file(tmp_path / 'b', man_trap=dict(nguon='t', alpha_r_s_px=0.0, alpha_c_s_px=0.0,
                                                 nhanh={'O1-F05': 'alpha'}))
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    out, _ = score(tmp_path / 'c', 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain))
    p = out['U_t_parts']
    assert p['man_trap_nhanh'] == 'mot_khung' and abs(p['man_trap_s'] - 1 / B.FPS) < 1e-6


def test_ct1_12_joint_hole_and_waiting_time():
    """CT1-12: a hole above t_trong between t_ref and t_max, or less than 6 s from the first log line to the edge:
    the window fails (both pass the other gates)."""
    kt = dict(ctd.KHUNG_TC_DE_XUAT, nguon='t')
    frames, chain = _still_chain(n=420, gap=(6.05, 6.3))
    rows = ctd.cua_so(frames, chain, _log(0.0, -1.0, 12.0), 6.2e6, [0.0], [0.2], kt, 0.10, 2.0, 'O1-F05')
    k = rows[0]['ket_luan']
    assert k['du_mau'] and k['dung_yen'] and not k['noi_phu'] and not k['dat']
    frames, chain = _still_chain(n=420)
    rows = ctd.cua_so(frames, chain, _log(0.0, 1.0, 12.0), 6.2e6, [0.0], [0.05], kt, 0.10, 2.0, 'O1-F05')
    k = rows[0]['ket_luan']
    assert k['du_mau'] and k['log_phu'] and not k['cho_du'] and not k['dat'] and abs(k['cho_log_s'] - 5.2) < 1e-6


def test_ct1_12_heading_spread_on_the_reference_set():
    """CT1-12: the heading spread gate on the set actually used (psi' after A6): 0.5 deg > 0.3 deg -> run lost to
    the reference window."""
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    pre, ids = ref_of(fl, chain)
    s = set(ids)
    for j, c in enumerate(x for x in chain if x['frame_id'] in s):
        c['psi_ma_p'] += math.radians(0.5) * (j % 2)
    tc = ctd.tham_chieu(pre, chain, ll_of('O1-F05', ids), dict(ctd.KHUNG_TC_DE_XUAT, nguon='t'), 0.10, 1 / B.FPS)
    assert tc['trang_thai'] == 'hong_cua_so_tc' and any('tan huong' in w for w in tc['ly_do'])


def test_ct1_13_window_reasons_and_order(tmp_path, monkeypatch):
    """CT1-13: the reference window lost keeps the command-path reasons of the run; a clip with no valid frame and a
    lost window is a window loss; a log without the tick bound is a window reason, not a tool error."""
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    pre, ids = ref_of(fl, chain)
    ll = ll_of('O1-F05', ids[:20])
    tc = ctd.tham_chieu(pre, chain, ll, dict(ctd.KHUNG_TC_DE_XUAT, nguon='t'), 0.10, 1 / B.FPS)
    rel = cs.build_rel('O1-F05', chain, tc, None, ll)
    monkeypatch.setattr(cs.ocp, 'analyse', lambda *a: dict(flags=['boot giua luot'], notes=[], cond2={}, cond3={},
                                                          cond4={}))
    out, _ = score(tmp_path / 'a', 'O1-F05', fl, rel=rel)
    assert out['hong_cua_so_tc'] and any('boot' in w for w in out['ly_do_khac'])
    bad = [dict(r, valid=False) for r in fl]
    out, _ = score(tmp_path / 'b', 'O1-F05', bad, rel=rel)
    assert out.get('hong_cua_so_tc') is True

    class A:
        gio_c8b, offset_s, offset_err_s, offset_anchor_s = None, B.OFFSET, 0.0, '0'
    log = B.make_log(tmp_path / 'esp32.csv', 'O1-F05', (0.0, 0.0, 0.0))
    rows, _, _, _ = cs.read_esp32(log)
    t0_us = float(rows[cs.ocp.find_edge(rows)[0]]['_t'])
    hs = B.hs_file(tmp_path)
    _, _, _, why = cs._clock_nodes(A, rows[-5:], t0_us, fl, hs, 3.0)
    assert why and 'tick' in why[0]


def test_ct1_14_one_sided_gate_and_full_pts():
    """CT1-14: khung holds every PTS of the window (also one whose capture ends after t_ref); a frame that failed the
    TWO-sided scale gate only because of later frames passes the one-sided gate; a scale jump after t_ref does not
    change the pre set."""
    n, dt = 420, 1 / 60
    frames, chain = [], []
    for k in range(n):
        tk = k * dt
        sc = 1.10 if tk > 6.05 else 1.0                                    # the frames after t_ref differ
        why = ['ti_le'] if 5.9 < tk < 6.05 else []                          # two-sided verdict spoilt by them
        frames.append(dict(idx=k, t_s=tk, car_seen=True, why=why, valid=not why, scale=sc))
        chain.append(dict(frame_id=k, cong_ok=not why, tag_x=0.5, tag_y=0.1, tag_psi=0.3))
    kt = dict(ctd.KHUNG_TC_DE_XUAT, nguon='t')
    gates = dict(scale_min=6, scale_rel=0.02, scale_win_s=0.5)
    rows = ctd.cua_so(frames, chain, _log(0.0, -1.0, 12.0), 6.2e6, [0.0], [0.055], kt, 0.10, 2.0, 'O1-F05',
                      gates=gates)
    pre = rows[0]
    t_ref = pre['t_ref']
    assert any(f['pts_s'] + dt >= t_ref for f in pre['khung'])
    used = set(pre['khung_ma_hop_le'])
    assert all(k in used for k in range(n) if 5.9 < k * dt < 6.05)
    assert pre['cong_ti_le'] == 'mot_phia'


def test_ct1_15_absolute_locked_file_without_khung_tc(tmp_path):
    """CT1-15: a locked constants file of the absolute mode needs no khung_tc; in the relative mode the four F14 keys
    are required and w_s is 3 s."""
    full = dict(trang_thai='khoa_c8', duong_lenh=dict(cond2_m=0.02, cond3_ms=50.0, cond4_m=0.03),
                cong_khung=dict(cs.cm.DEFAULT_GATES), gia_dinh_nghi_vi_tri='dung', nhanh_tick='giu',
                imu=dict(mien_dong={m: dict(goc_quay_deg=720.0, wz_dps=200.0, alpha_dps2=2000.0)
                                    for m in ('O1-F02', 'O1-F05', 'O1-F08', 'O1-K1', 'O1-K2')}))
    for d in ('a', 'b', 'c'):
        (tmp_path / d).mkdir()
    hs = B.hs_file(tmp_path / 'a', **json.loads(json.dumps(full)))
    assert hs['khung_tc'] is None
    with pytest.raises(ValueError, match='khung_tc'):
        cs.khung_tc_of(hs)
    kt4 = dict(n_min=120, frac_min=0.6, p_dung_m=0.003, psi_dung_deg=0.3)
    hs = B.hs_file(tmp_path / 'b', **json.loads(json.dumps(dict(full, khung_tc=kt4))))
    assert cs.khung_tc_of(hs)['w_s'] == 3.0
    with pytest.raises(ValueError, match='w_s'):
        B.hs_file(tmp_path / 'c', **json.loads(json.dumps(dict(full, khung_tc=dict(kt4, w_s=4.0)))))


def test_ct1_16_json_without_infinity(tmp_path):
    """CT1-16: without u_beta_cam the block has u_phi_rad null with its state, no Infinity in the JSON; bien_doi
    names psi_dd (F13)."""
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    out, rows = score(tmp_path, 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain, u_beta_cam=None))
    k = out['khung_tuong_doi_v1']
    assert k['u_phi_rad'] is None and k['u_phi_trang_thai'] == 'thieu_u_beta_cam' and 'psi_dd' in k['bien_doi']
    json.dumps(cs.cm._json(k), allow_nan=False)
    assert all(r['U_phi_k'] is None for r in rows)


def test_ct1_19_heading_reset_mid_run_is_command_path_loss(tmp_path):
    """CT1-19 (part): reset_n changing inside the run window: run lost to the command path (relative mode)."""
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    out, _ = score(tmp_path, 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain), log_kw=dict(reset_at=4.0))
    assert out['outcome'] == 'hong_he_do' and out['fail_cause'] == cs.FAIL_CMD
    assert any('reset_n' in w for w in out['outcome_reasons'])


def test_ct1_20_g5_after_t_som_real_violation_only_chua_ket_luan(tmp_path):
    """CT1-20: a REAL G5 violation (camera and IMU turn 10 deg together, lo > 7.5 deg at known frames) after t_som
    gives chua_ket_luan; the same evidence before t_som gives khong_dat (test above)."""
    lane_rows = B.make_frames('O1-F05', post=4.0)
    t_late = 13.65 + 1.3                                  # t_som is about 14.67 s here, t_deadline 15.65 s
    late = lambda r: math.radians(12.0) if r['t_s'] + B.OFFSET - B.T0 >= t_late else 0.0
    fl, chain = to_floor('O1-F05', lane_rows, psi_extra=late)
    out, rows = score(tmp_path, 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain, truc=False),
                      log_kw=dict(post=5.0, yaw_err=lambda t: math.radians(12.0) if t >= t_late else 0.0))
    g = gate_ok(out)
    assert g['g1']['t_som_s'] < t_late
    assert any(r['t'] > t_late + 0.2 and r['lo_g5'] > 7.5 for r in rows)
    assert not any(r['t'] <= g['g1']['t_som_s'] and r['lo_g5'] > 7.5 for r in rows)
    assert g['g5']['state'] == 'chua_ket_luan' and out['outcome'] == 'chua_ket_luan'


def test_ct1_20_cli_relative_inputs_without_the_flag(tmp_path, capsys):
    """CT1-20: valid frames, log and constants; a relative input without --tuong-doi is refused for that reason;
    --tuong-doi without --rows is refused for its own reason."""
    lane_rows = B.make_frames('O1-F05')
    p = tmp_path / 'frames.csv'
    cs.cm.write_frames(str(p), lane_rows)
    B.hs_file(tmp_path)
    log = B.make_log(tmp_path / 'esp32.csv', 'O1-F05', (0.0, 0.0, 0.0))
    base = ['--frames', str(p), '--esp32', str(log), '--task', 'O1-F05', '--hang-so', str(tmp_path / 'hang_so.json'),
            '--offset-s', str(B.OFFSET), '--offset-reason', 't', '--offset-anchor-s', '0', '--offset-err-s', '0']
    assert cs.main(base + ['--a6', 'y']) == 2
    assert 'without --tuong-doi' in capsys.readouterr().err
    assert cs.main(base + ['--tuong-doi', '--cua-so', 'a', '--tham-chieu', 'b', '--out', str(tmp_path / 'o.json')]) == 2
    assert '--rows' in capsys.readouterr().err


def test_mien_phai_phu_lists_the_union_domain_frames(tmp_path):
    """CODE 19/11: khung_tuong_doi_v1.mien_phai_phu = the frames of the union domain at this node (GATE compares it
    with mien_da_tinh of u_moc_manifest_v1): ids, their hash (ids_sha), node (null without the C8b grid), D."""
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    out, rows = score(tmp_path, 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain))
    m = out['khung_tuong_doi_v1']['mien_phai_phu']
    ids = [r['frame_id'] for r in rows if r['mien_hop']]
    assert m['id_khung'] == ids and m['id_khung_sha256'] == ctd.ids_sha(ids) and m['nut'] is None
    assert m['D_s'] == B.OFFSET and len(ids) > 100


@pytest.mark.parametrize('case', ['ok', 'not_latched', 'age_over', 'no_next_row_lien', 'no_column'])
def test_ct1_19_yaw_latch_at_the_edge(tmp_path, case):
    """CT1-19 (T9, CODE 19/11): the edge row latched (r_valid, rv_n, head_on, psi_ref 0) and the age bound
    age + 0.05 ms + gap + 0.003 ms <= tuoi_yaw_max_ms; the next row not log_seq + 1: a later row, dong_ke_lien false;
    a latch column missing: not determinable, a command-path loss."""
    def edit(rows, i):
        if case == 'not_latched':
            rows[i]['psi_ref'] = '0.0100'
        elif case == 'age_over':
            rows[i]['age_r_ms'] = '124.9'
        elif case == 'no_next_row_lien':
            rows[i + 1]['timestamp_us'] = rows[i]['timestamp_us']       # log_seq + 1 not later: not usable
        elif case == 'no_column':
            for r in rows:
                del r['head_on']
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    out, _ = score(tmp_path, 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain), latch=edit)
    c = out['khung_tuong_doi_v1']['chot_yaw'] if out.get('khung_tuong_doi_v1') else None
    if case in ('ok', 'no_next_row_lien'):
        assert out['outcome'] != 'hong_he_do' and c['chot_dung'] and c['can_tuoi_ms'] <= 125.0
        assert c['dong_ke_lien'] is (case == 'ok')
        if case == 'no_next_row_lien':
            assert c['gap_ms'] > 15.0                                    # two log periods: a looser bound
    else:
        assert out['outcome'] == 'hong_he_do' and out['fail_cause'] == cs.FAIL_CMD
        assert any('J v6 5 (1)' in w for w in out['outcome_reasons'])


def test_ct1_19_age_bound_example():
    """J v6 5 example: true age 8.04 ms printed 8.0, latch gap 2.00 ms -> bound 10.053 ms, above a 10 ms threshold."""
    rows = [dict(log_seq='10', timestamp_us='1000000', run_active='0'),
            dict(log_seq='11', timestamp_us='1010000', run_active='1', r_valid='1', rv_n='3', head_on='1',
                 psi_ref='0.0000', age_r_ms='8.0'),
            dict(log_seq='12', timestamp_us='1012000', run_active='1')]
    fl, info = cs.yaw_latch_flags(rows, 1, 10.0, seq_max=12)
    assert abs(info['can_tuoi_ms'] - 10.053) < 1e-9 and fl and info['dong_ke_lien'] is True
    fl, info = cs.yaw_latch_flags(rows, 1, 10.06, seq_max=12)
    assert not fl


# ---------------------------------------------------------------- Codex round 2 (CT2-*, CT1-* residual): fail on fe7f802
FIX_LL = os.path.join(os.path.dirname(__file__), 'data', 'lan_lap_v1_nhactam_e7b8428.json')


def _real_ll():
    """Run L01 of a lan_lap.json written by NHACTAM c7_nhac_tam.run_lan_lap (hil e7b8428; fixture data/...json) as
    c7_chuoi_khung.doc_lan_lap gives it (file keys copied in), and the pre / post rows of the window file NHACTAM
    read."""
    with open(FIX_LL, encoding='utf-8') as f:
        fx = json.load(f)
    ll = dict(fx['luot'])
    for k, v in fx['lan_lap_file'].items():
        ll.setdefault(k, v)
    return ll, dict(fx['cua_so_pre']), dict(fx['cua_so_post'])


def _chain_of(pre):
    return [dict(frame_id=int(f['id']), t_cam=float(f['pts_s']), q_ok=True, a6_ok=True, p_x=0.5, p_y=0.1,
                 psi_ma_p=0.3) for f in pre['khung']]


def test_ct2_1_reads_the_sets_as_nhactam_writes_them():
    """CT2-1, CT3-1: the anchor-tag intersection dau.pre.khung_giao_ma_neo {id, pts_s} + khung_giao_sha256 (NHACTAM
    output) and khung_chung bound to the pre and post windows (ids, PTS, sha of the one form '1,2'); the reference uses
    the intersection; a PTS or sha changed, on either end: refused."""
    ll, pre, post = _real_ll()
    cs.bind_lan_lap(ll, pre, ll['cua_so_v1_sha256'], 'O1-F05', post)
    for end, row in (('pre', pre), ('post', post)):
        bad = json.loads(json.dumps(ll))
        bad['dau'][end]['khung_giao_ma_neo']['pts_s'][3] += 0.01
        with pytest.raises(ValueError, match='pts_s'):
            cs.bind_lan_lap(bad, pre, ll['cua_so_v1_sha256'], 'O1-F05', post)
        bad = json.loads(json.dumps(ll))
        bad['khung_chung'][end]['pts_s'][3] += 0.01
        with pytest.raises(ValueError, match='pts_s'):
            cs.bind_lan_lap(bad, pre, ll['cua_so_v1_sha256'], 'O1-F05', post)
    bad = json.loads(json.dumps(ll))
    bad['dau']['pre']['khung_giao_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='ids_sha'):
        cs.bind_lan_lap(bad, pre, ll['cua_so_v1_sha256'], 'O1-F05')
    old = json.loads(json.dumps(ll))                 # the earlier hash form '[1,2]' is no longer accepted
    ids = old['dau']['pre']['khung_giao_ma_neo']['id']
    import hashlib
    old['dau']['pre']['khung_giao_sha256'] = hashlib.sha256(json.dumps(sorted(ids), separators=(',', ':'))
                                                            .encode()).hexdigest()
    with pytest.raises(ValueError, match='ids_sha'):
        ctd.neo_of(old, 'pre')
    pre2 = dict(pre, t_ref=pre['khung'][-1]['pts_s'] + 0.05, ket_luan=dict(dat=True, ly_do=[]))
    tc = ctd.tham_chieu(pre2, _chain_of(pre), ll, dict(ctd.KHUNG_TC_DE_XUAT, nguon='t'), 0.10, 1 / 60)
    giao = set(ll['dau']['pre']['khung_giao_ma_neo']['id'])
    assert tc['tap'] == 'giao' and tc['ids'] == [i for i in pre['khung_ma_hop_le'] if i in giao]


def test_ct3_1_intersection_with_few_frames_is_not_mat_neo():
    """CT3-1 (Codex example): NHACTAM leaves khung_chung.pre empty with a null sha when its own set gate fails, while
    the anchor-tag intersection has frames (mat_neo false): the reference is on that intersection (window lost for too
    few frames), never the tag-gate frames with mat_neo; mat_neo only for an EMPTY intersection flagged so."""
    ll, pre, _ = _real_ll()
    pre2 = dict(pre, t_ref=pre['khung'][-1]['pts_s'] + 0.05, ket_luan=dict(dat=True, ly_do=[]))
    kt = dict(ctd.KHUNG_TC_DE_XUAT, nguon='t')
    few = json.loads(json.dumps(ll))
    few['khung_chung']['pre'] = dict(id=[], pts_s=[])
    few['khung_chung_sha256']['pre'] = few['dau']['pre']['khung_chung_sha256'] = None
    few['dau']['pre']['khung_chung'] = dict(id=[], pts_s=[])
    tc = ctd.tham_chieu(pre2, _chain_of(pre), few, kt, 0.10, 1 / 60)
    assert tc['tap'] == 'giao' and not tc['mat_neo'] and tc['trang_thai'] == 'hong_cua_so_tc' and tc['n'] == 12
    lost = json.loads(json.dumps(few))
    lost['dau']['pre'].update(khung_giao_ma_neo=dict(id=[], pts_s=[]), khung_giao_sha256=ctd.ids_sha([]),
                              mat_neo=True)
    tc = ctd.tham_chieu(pre2, _chain_of(pre), lost, kt, 0.10, 1 / 60)
    assert tc['tap'] == 'ma_mat_neo' and tc['mat_neo']
    lost['dau']['pre']['mat_neo'] = False
    with pytest.raises(ValueError, match='mat_neo'):
        ctd.neo_of(lost, 'pre')


def test_ct2_2_missing_q_keeps_t_som_at_the_still_time(tmp_path):
    """CT2-2: K1 without q on every frame (video continuous, speed unknown, yaw usable), car still from T_cmd: t_som is
    about T_cmd (completion cannot be ruled out), not t_deadline."""
    lane_rows = B.make_frames('O1-K1', post=4.0)
    fl, chain = to_floor('O1-K1', lane_rows)
    for c in chain:
        c.update(q_ok=False, p_x=float('nan'), p_y=float('nan'))
    out, _ = score(tmp_path, 'O1-K1', fl, rel=rel_for('O1-K1', fl, chain), allow_final_set=True,
                   log_kw=dict(post=5.0))
    g = out['gate_v3']['g1']
    assert g['ca_t_som'] and g['t_som_s'] < 5.0 + 0.1 and g['t_som_s'] < g['t_deadline'] - 1.0


def test_ct2_3_imu_alignment_on_every_f12_frame(tmp_path):
    """CT2-3: F12 frames that are not 'valid' in frames.csv (one-sided gate of the window) still align the IMU: every
    id of tham_chieu_v1 is used (n_align = n), not the valid subset."""
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    rel = rel_for('O1-F05', fl, chain)
    drop = set(rel['tc']['ids'][::5])
    fl = [dict(r, valid=False) if r['idx'] in drop else r for r in fl]
    out, _ = score(tmp_path, 'O1-F05', fl, rel=rel)
    y = out['yaw_check']
    assert y['compared'] and y['align_set'] == 'tham_chieu_ids' and y['n_align'] == rel['tc']['n']


def test_ct2_4_short_a6_loss_is_undefined_frames_not_a_gap(tmp_path):
    """CT2-4 (Codex example): video continuous, A6 undefined 4.0-4.2 s: undefined frames (n_undef), no gap of the
    measurement system, so the run is not lost."""
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    rel = rel_for('O1-F05', fl, chain)
    for c, f in zip(chain, fl):
        if 4.0 <= f['t_s'] + B.OFFSET - B.T0 < 4.2:
            c.update(a6_ok=False, psi_ma_p=float('nan'), p_x=float('nan'), p_y=float('nan'))
    out, _ = score(tmp_path, 'O1-F05', fl, rel=rel)
    assert out['outcome'] != 'hong_he_do' and out['t1_kind'] != 'khoang_trong'
    assert out['gate_v3']['g2']['n_undef'] > 0


def test_ct2_5_f08_mask_does_not_depend_on_the_call(tmp_path):
    """CT2-5: F08 relative with a certain G5 violation: report_only={'g5'} passed by the caller gives the same gate
    (gate_v3.3, J v6 sha) and outcome as the default call."""
    lane_rows = B.make_frames('O1-F08')
    bump = lambda r: math.radians(15.0) if 2.0 <= r['t_s'] + B.OFFSET - B.T0 <= 3.0 else 0.0
    fl, chain = to_floor('O1-F08', lane_rows, psi_extra=bump)
    kw = dict(log_kw=dict(yaw_err=lambda t: math.radians(15.0) if 2.0 <= t <= 3.0 else 0.0))
    a, _ = score(tmp_path / 'a', 'O1-F08', fl, rel=rel_for('O1-F08', fl, chain), **kw)
    b, _ = score(tmp_path / 'b', 'O1-F08', fl, rel=rel_for('O1-F08', fl, chain), report_only={'g5'}, **kw)
    assert b['gate_v3']['version'] == a['gate_v3']['version'] and b['gate_v3']['gate_file_sha256'] == cs.J_V6_SHA256
    assert b['outcome'] == a['outcome'] != 'khong_dat'


def test_ct2_8_absolute_no_frame_before_the_sender(tmp_path):
    """CT2-8: absolute mode, no valid frame and a sender log with two different '#test_end': hong_he_do as at 921e839
    (the sender is not read), not an error."""
    lane_rows = [dict(r, valid=False) for r in B.make_frames('O1-F05')]
    hs = B.hs_file(tmp_path)
    log = B.make_log(tmp_path / 'esp32.csv', 'O1-F05', (0.0, 0.0, 0.0))
    snd = tmp_path / 'sender.txt'
    snd.write_text('#test_end,test=O1-F05,t_epoch=2.000,complete=1\n#test_end,test=O1-F05,t_epoch=3.000,complete=1\n',
                   encoding='utf-8')
    out, _ = cs.score(lane_rows, log, 'O1-F05', hs, offset_s=B.OFFSET, offset_reason='t', offset_anchor_s=0.0,
                      offset_err_s=0.0, sender=str(snd))
    assert out['outcome'] == 'hong_he_do' and out['fail_cause'] == cs.FAIL_MEAS


def test_ct2_9_next_row_same_boot_and_counter_cells():
    """CT2-9: the next row after the run's last data line (another boot) gives no age bound; an empty counter cell in
    the window is missing evidence."""
    rows = [dict(log_seq='10', timestamp_us='1000000', run_active='0', _t=1000000.0, _seq=10, psi_reset_count='0'),
            dict(log_seq='11', timestamp_us='1010000', run_active='1', r_valid='1', rv_n='3', head_on='1',
                 psi_ref='0.0000', age_r_ms='8.0', _t=1010000.0, _seq=11, psi_reset_count='0'),
            dict(log_seq='12', timestamp_us='1012000', run_active='1', _t=1012000.0, _seq=12, psi_reset_count='')]
    fl, info = cs.yaw_latch_flags(rows, 1, 125.0, seq_max=11)
    assert info['can_tuoi_ms'] is None and fl
    assert cs.psi_reset_flags(rows, 1, 1010000.0, 1.0)


def test_ct1_12_still_six_seconds_not_only_log():
    """CT1-12 residual (Codex example): the car moves in the first 2 s, then still 4.2 s to the edge; the log covers
    6.2 s: the window fails (still < 6 s)."""
    frames, chain = [], []
    for k in range(420):
        tk = k / 60
        x = 0.5 + 0.05 * min(tk, 2.0)
        frames.append(dict(idx=k, t_s=tk))
        chain.append(dict(frame_id=k, cong_ok=True, tag_x=x, tag_y=0.1, tag_psi=0.3))
    kt = dict(ctd.KHUNG_TC_DE_XUAT, nguon='t')
    rows = ctd.cua_so(frames, chain, _log(0.0, 0.0, 12.0), 6.2e6, [0.0], [0.05], kt, 0.10, 2.0, 'O1-F05')
    k = rows[0]['ket_luan']
    assert k['dung_yen'] and k['cho_du'] and not k['dung_yen_6s'] and not k['dat']


def test_ct1_14_one_median_on_the_window():
    """CT1-14 (Codex example): scale rising 1.00 -> 1.08 over 4 s, 2 %, 0.5 s neighbourhood: one median (about 1.04)
    rejects the end of the window (a sliding median would pass it)."""
    frames, chain = [], []
    for k in range(420):
        tk = k / 60
        sc = 1.0 + 0.08 * min(max(tk - 2.2, 0.0), 4.0) / 4.0
        frames.append(dict(idx=k, t_s=tk, car_seen=True, why=[], valid=True, scale=sc))
        chain.append(dict(frame_id=k, cong_ok=True, tag_x=0.5, tag_y=0.1, tag_psi=0.3))
    gates = dict(scale_min=6, scale_rel=0.02, scale_win_s=0.5)
    ok = ctd.mot_phia(frames, chain, gates, 6.2)
    t = np.array([f['t_s'] for f in frames])
    assert not ok[(t > 5.9) & (t < 6.2)].any() and ok[(t > 4.1) & (t < 4.3)].all()


def test_ct1_14_cua_so_checks_track_json(tmp_path, capsys):
    """CT1-14: cua-so takes the gates of track.json only for the track.json of this frames.csv (sha256)."""
    lane_rows = B.make_frames('O1-F05')
    p = tmp_path / 'frames.csv'
    cs.cm.write_frames(str(p), lane_rows)
    (tmp_path / 'track.json').write_text(json.dumps(dict(frames_sha256='0' * 64, gates=dict(cs.cm.DEFAULT_GATES),
                                                         tag_offset=[0.0, 0.0, 0.0])), encoding='utf-8')
    B.hs_file(tmp_path)
    log = B.make_log(tmp_path / 'esp32.csv', 'O1-F05', (0.0, 0.0, 0.0))
    rc = cs.main(['cua-so', '--frames', str(p), '--esp32', str(log), '--task', 'O1-F05', '--hang-so',
                  str(tmp_path / 'hang_so.json'), '--out', str(tmp_path / 'cs.jsonl'), '--offset-s', str(B.OFFSET),
                  '--offset-err-s', '0', '--offset-anchor-s', '0'])
    assert rc == 2 and 'frames_sha256 differs' in capsys.readouterr().err


def test_ct1_11_locked_relative_needs_the_branch_table():
    """CT1-11 residual: a locked constants file of the relative mode without man_trap.nhanh is refused (J v6 6, 7)."""
    with pytest.raises(ValueError, match='nhanh'):
        cs.man_trap_of(dict(trang_thai='khoa_c8', man_trap=None), 'O1-F05', rel=True)
    assert cs.man_trap_of(dict(trang_thai='de_xuat', man_trap=None), 'O1-F05', rel=True) is None


@pytest.mark.parametrize('case', ['boot', 'blk', 'window_and_boot'])
def test_ct1_13_operator_cause_first(tmp_path, case):
    """CT1-13 (CODE 19/11): two boot ids in the events, or an ultrasonic block tick: fail_cause thao_tac with its kind;
    with a lost reference window too, the run is the operator loss (not counted as a window loss)."""
    import csv as _csv
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    tmp_path.mkdir(parents=True, exist_ok=True)
    log = B.make_log(tmp_path / 'esp32.csv', 'O1-F05', (0.0, 0.0, 0.0))
    with open(log, encoding='utf-8') as f:
        f.readline()
        seqs = [int(r['log_seq']) for r in _csv.DictReader(f)]
    extra = ('2026-11-19T10:00:40 #gain_ack,tx=1,req_id=1,ok=1,err=none,sha=x,env=e,mac=m,boot=00000002\n'
             if case != 'blk' else '')
    us_events(str(log) + '.events.txt', seqs[0], seqs[-1], blk='3' if case == 'blk' else '0', extra=extra)
    latch_cols(log)
    hs = B.hs_file(tmp_path)
    if case == 'window_and_boot':
        pre, ids = ref_of(fl, chain)
        ll = ll_of('O1-F05', ids[:20])
        tc = ctd.tham_chieu(pre, chain, ll, dict(ctd.KHUNG_TC_DE_XUAT, nguon='t'), 0.10, 1 / B.FPS)
        rel = cs.build_rel('O1-F05', chain, tc, None, ll)
    else:
        rel = rel_for('O1-F05', fl, chain)
    out, _ = cs.score(fl, log, 'O1-F05', hs, offset_s=B.OFFSET, offset_reason='t', offset_anchor_s=0.0,
                      offset_err_s=0.0, rel=rel)
    assert out['outcome'] == 'hong_he_do' and out['fail_cause'] == 'thao_tac'
    assert out['thao_tac'] == (['khoa_sieu_am'] if case == 'blk' else ['boot'])
    if case == 'window_and_boot':
        assert out['hong_cua_so_tc'] is False and any('cua so' in w for w in out['ly_do_khac'])


def test_ct1_3_clock_constants_and_the_preliminary_grid(tmp_path):
    """CT1-3 residual, CT2-7: the window binds the constants it used (u_t_ghep_s changed: refused); a window built
    from the preliminary grid binds to the final grid that names it (luoi_so_bo_sha256, cua_so_v1_sha256)."""
    so_bo, fin, cua = tmp_path / 'so_bo.json', tmp_path / 'gio.json', tmp_path / 'cs.jsonl'
    so_bo.write_text('{"schema": "c8b_gio_v1", "n": 1}', encoding='utf-8')
    cua.write_text('{}\n', encoding='utf-8')
    fin.write_text(json.dumps(dict(schema='c8b_gio_v1', luoi_so_bo_sha256=cs.cm.sha256(str(so_bo)),
                                   cua_so_v1_sha256=cs.cm.sha256(str(cua)))), encoding='utf-8')
    hs = dict(khung_tc=dict(ctd.KHUNG_TC_DE_XUAT, nguon='t'), trang_thai='de_xuat', u_t_ghep_s=0.0, troi_ppm=436.0,
              t_trong_s=0.1)
    pre = dict(gio_sha256=cs.cm.sha256(str(so_bo)), nguong={k: ctd.KHUNG_TC_DE_XUAT[k] for k in ctd.KHUNG_TC_DE_XUAT},
               hang_so_dung=dict(u_t_ghep_s=0.0, troi_ppm=436.0, t_trong_s=0.1))

    class A:
        gio_c8b, cua_so = str(fin), str(cua)
    cs.bind_clock(pre, A, hs)
    with pytest.raises(ValueError, match='hang_so_dung'):
        cs.bind_clock(pre, A, dict(hs, u_t_ghep_s=0.2))
    A.cua_so = str(so_bo)
    with pytest.raises(ValueError, match='time grid'):
        cs.bind_clock(pre, A, hs)


def test_ct2_7_relative_without_the_grid_is_diagnostic(tmp_path):
    """CT2-7 (CODE 19/11): relative scoring with --offset-s is a diagnostic: cham_c8 false with its reason."""
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    out, _ = score(tmp_path, 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain))
    assert out['cham_c8'] is False and out['cham_c8_ly_do']


def test_ct1_16_lost_u_t_parts_without_infinity(tmp_path):
    """CT1-16 residual: a run lost because the log ends before the union domain: no Infinity in the output."""
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    out, _ = score(tmp_path, 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain), log_kw=dict(post=-3.0))
    assert out['outcome'] == 'hong_he_do'
    json.dumps(cs.cm._json(out), allow_nan=False)


def test_ct1_20_g5_after_t_som_before_a_gap_is_lost(tmp_path):
    """CT1-20 (J5-3 example): t_som case, the only G5 violation after t_som and before t1 of a gap: no exception, the
    run is lost to the measurement system (rerun), not khong_dat."""
    lane_rows = B.make_frames('O1-F05', post=4.0, drop=[(13.65 + 1.6, 13.65 + 1.9)])
    t_late = 13.65 + 1.3
    late = lambda r: math.radians(12.0) if r['t_s'] + B.OFFSET - B.T0 >= t_late else 0.0
    fl, chain = to_floor('O1-F05', lane_rows, psi_extra=late)
    out, rows = score(tmp_path, 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain, truc=False),
                      log_kw=dict(post=5.0, yaw_err=lambda t: math.radians(12.0) if t >= t_late else 0.0))
    assert not out['cat_t1'] and out['outcome'] == 'hong_he_do' and out['fail_cause'] == cs.FAIL_MEAS


# ---------------------------------------------------------------- Codex round 3 (CT3-*, residuals): fail on 7e8b291
def test_ct3_2_only_g5_of_f08_is_exempt(tmp_path):
    """CT3-2: in the relative mode a caller mask beyond G5 of F08 (here G3) is refused, it cannot borrow the J sha."""
    lane_rows = B.make_frames('O1-F08')
    fl, chain = to_floor('O1-F08', lane_rows)
    with pytest.raises(ValueError, match='only g5'):
        score(tmp_path, 'O1-F08', fl, rel=rel_for('O1-F08', fl, chain), report_only={'g3', 'g5'})


def _gio(log, fsha, final=True):
    return dict(file='gio.json', sha256='a' * 64, nut=0, D=B.OFFSET, win_log_s=[B.T0 - 3.0, B.T0 - 1.0], step_s=0.0,
                n_nodes=1, vao=dict(esp32=cs.cm.sha256(str(log)), frames=fsha),
                luoi_so_bo_sha256='b' * 64 if final else None, cua_so_v1_sha256='c' * 64 if final else None)


@pytest.mark.parametrize('final', [False, True])
def test_ct3_3_c8_result_needs_the_final_grid_and_locked_constants(tmp_path, final):
    """CT3-3: cham_c8 needs a FINAL grid (luoi_so_bo_sha256 and cua_so_v1_sha256) and constants locked for C8."""
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    tmp_path.mkdir(parents=True, exist_ok=True)
    hs = B.hs_file(tmp_path)
    log = latch_cols(B.make_log(tmp_path / 'esp32.csv', 'O1-F05', (0.0, 0.0, 0.0)))
    out, _ = cs.score(fl, log, 'O1-F05', hs, rel=rel_for('O1-F05', fl, chain), frames_sha256='f' * 64,
                      gio=_gio(log, 'f' * 64, final))
    assert out['cham_c8'] is False and 'chua khoa' in out['cham_c8_ly_do']
    assert ('khong phai file gio cuoi' in out['cham_c8_ly_do']) is (not final)


def test_ct3_4_operator_evidence_without_an_edge(tmp_path):
    """CT3-4: a '#gain_ack ok=0' with no run_active edge: the run is lost to the operator (loi_nap_gain), not a tool
    error; the same from the log alone when the relative inputs cannot be built (operator_loss)."""
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    tmp_path.mkdir(parents=True, exist_ok=True)
    log = B.make_log(tmp_path / 'esp32.csv', 'O1-F05', (0.0, 0.0, 0.0))
    us_events(str(log) + '.events.txt', 0, 10,
              extra='2026-11-19T10:00:40 #gain_ack,tx=1,req_id=1,ok=0,err=crc,sha=x,env=e,mac=m,boot=00000001\n')

    def no_edge(rows, i):
        for r in rows:
            r['run_active'] = '0'
    latch_cols(log, no_edge)
    out, _ = cs.score(fl, log, 'O1-F05', B.hs_file(tmp_path), offset_s=B.OFFSET, offset_reason='t',
                      offset_anchor_s=0.0, offset_err_s=0.0, rel=rel_for('O1-F05', fl, chain))
    assert out['fail_cause'] == 'thao_tac' and out['thao_tac'] == ['loi_nap_gain']
    op = cs.operator_loss(str(log), None, 'O1-F05', 'x')
    assert op['fail_cause'] == 'thao_tac' and 'loi_nap_gain' in op['thao_tac']


@pytest.mark.parametrize('case', ['past_motor_off', 'ticks'])
def test_ct3_5_ultrasonic_summary_tied_to_the_run(case):
    """CT3-5 (Codex examples): first_seq 10, edge 11, motor off at 12, '#us_end' last_data_seq 20: not this run's
    summary (khoa_sieu_am, no T9 bound from it); first_seq 10, last 20, ticks 1: contradicting (khoa_sieu_am)."""
    mo = (lambda q: q != 12) if case == 'past_motor_off' else (lambda q: True)
    rows = [dict(_seq=q, _mo=mo(q)) for q in range(10, 21)]
    ticks = 1 if case == 'ticks' else 11
    ev = ['#us,ev=start,boot=00000001,on=0,swb=0,first_seq=10,line_drop=0',
          f'#us_end,boot=00000001,on=0,swb=0,changes=0,blk_ticks=0,cut_ticks=0,ticks={ticks},last_data_seq=20,'
          'line_drop=0']
    kinds, why = cs.thao_tac_of(rows, ev, 1)
    assert kinds == ['khoa_sieu_am']
    assert not cs.run_us(rows, ev, 1)['ok']


def test_ct3_7_absolute_no_frame_before_the_events(tmp_path):
    """CT3-7: absolute mode, no valid frame and an events file with undecodable bytes: hong_he_do (921e839 order)."""
    lane_rows = [dict(r, valid=False) for r in B.make_frames('O1-F05')]
    hs = B.hs_file(tmp_path)
    log = B.make_log(tmp_path / 'esp32.csv', 'O1-F05', (0.0, 0.0, 0.0))
    with open(str(log) + '.events.txt', 'wb') as f:
        f.write(b'\xff\xfe\x00#us bad\n')
    out, _ = cs.score(lane_rows, log, 'O1-F05', hs, offset_s=B.OFFSET, offset_reason='t', offset_anchor_s=0.0,
                      offset_err_s=0.0)
    assert out['outcome'] == 'hong_he_do' and out['fail_cause'] == cs.FAIL_MEAS


def test_ct3_8_align_set_named_on_early_exit(tmp_path):
    """CT3-8: align_ref given, no IMU sample: not compared, the source is still tham_chieu_ids."""
    hs = B.hs_file(tmp_path)
    log = B.make_log(tmp_path / 'esp32.csv', 'O1-K1', (0.0, 0.0, 0.0), no_imu=True)
    t = np.array([B.T0 - 1.0, B.T0 - 0.5])
    _, rep = cs.yaw_bounds(t, np.zeros(2), np.zeros(2), np.zeros(2), B.T0, hs, 'O1-K1', log, B.T0 + 7.0,
                           align_ref=(t, np.zeros(2)))
    assert not rep['compared'] and rep['align_set'] == 'tham_chieu_ids'


def test_ct2_4_long_a6_loss_with_video_is_not_a_tracking_error(tmp_path):
    """CT2-4 residual (Codex example): K1, A6 undefined 4.0-5.0 s, video continuous: undefined frames, no tracking
    error at the reconnect, the run not lost."""
    lane_rows = B.make_frames('O1-K1')
    fl, chain = to_floor('O1-K1', lane_rows)
    rel = rel_for('O1-K1', fl, chain)
    for c, f in zip(chain, fl):
        if 4.0 <= f['t_s'] + B.OFFSET - B.T0 < 5.0:
            c.update(a6_ok=False, psi_ma_p=float('nan'), p_x=float('nan'), p_y=float('nan'))
    out, _ = score(tmp_path, 'O1-K1', fl, rel=rel, allow_final_set=True)
    assert out['outcome'] != 'hong_he_do' and out['t1_kind'] not in ('loi_theo_doi', 'khoang_trong')


def test_ct1_3_window_gates_bound_at_scoring():
    """CT1-3 residual: a window gated with scale_rel 0.035 is refused when the scoring gates (track.json) say 0.011."""
    g = dict(cs.cm.DEFAULT_GATES, scale_rel=0.035)
    pre = dict(cong_khung=g)
    hs = dict(trang_thai='de_xuat')
    cs.bind_gates(pre, hs, dict(frames_sha256='f' * 64, gates=g), 'f' * 64)
    with pytest.raises(ValueError, match='cong_khung'):
        cs.bind_gates(pre, hs, dict(frames_sha256='f' * 64, gates=dict(g, scale_rel=0.011)), 'f' * 64)


def test_ct1_14_track_gates_need_the_frames_sha():
    """CT1-14 residual: track.json gates without frames_sha256 are not used, also before C8; defaults named mac_dinh."""
    hs = dict(trang_thai='de_xuat')
    with pytest.raises(ValueError, match='frames_sha256'):
        cs.window_gates(hs, dict(gates=dict(cs.cm.DEFAULT_GATES)), 'f' * 64)
    assert cs.window_gates(hs, None, 'f' * 64)[1] == 'mac_dinh'


def test_ct1_20_t_som_wz_uses_the_stencil(tmp_path, monkeypatch):
    """CT1-20 residual: the t_som case takes the usability of wz from the yaw flags over the derivative stencil
    (ctd.wz_usable with the half width of c7_measure.sg_derivative), not from the central frame alone."""
    seen = []
    real = ctd.wz_usable

    def spy(yaw_ok, half):
        seen.append((len(yaw_ok), half))
        return real(yaw_ok, half)
    monkeypatch.setattr(ctd, 'wz_usable', spy)
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    score(tmp_path, 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain, truc=False))
    assert seen and all(h == cs.cm.SG_HALF and n > 100 for n, h in seen)


# ---------------------------------------------------------------- Codex round 4 (CT3-* residual, CT4-*): fail on ed1d020
def test_ct2_4_progress_undefined_after_a_long_centre_less_stretch(tmp_path):
    """CT2-4 (CODE 19/11: J wins in the relative mode): K1, A6 undefined 4.0-5.8 s with continuous video: no tracking
    error, the progress is not determined from the reconnect (G2, G4 undefined there), G1 not concluded."""
    lane_rows = B.make_frames('O1-K1', post=4.0)
    fl, chain = to_floor('O1-K1', lane_rows)
    rel = rel_for('O1-K1', fl, chain)
    for c, f in zip(chain, fl):
        if 4.0 <= f['t_s'] + B.OFFSET - B.T0 < 5.8:
            c.update(a6_ok=False, psi_ma_p=float('nan'), p_x=float('nan'), p_y=float('nan'))
    out, rows = score(tmp_path, 'O1-K1', fl, rel=rel, allow_final_set=True, log_kw=dict(post=5.0))
    g = out['gate_v3']
    assert out['outcome'] == 'chua_ket_luan' and out['t1_kind'] not in ('loi_theo_doi', 'khoang_trong')
    assert g['g1']['state'] == 'chua_ket_luan' and any(r.get('tien_do_chua_xd') == 1 for r in rows)
    assert all(not math.isfinite(r['hi_g2']) for r in rows if r.get('tien_do_chua_xd') == 1)


def test_ct2_4_centre_lost_over_the_whole_domain(tmp_path):
    """CT2-4 (CODE 19/11): no frame with a centre in the run window, video there: G1-G4 not concluded, never lost
    'no valid camera frame'."""
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    rel = rel_for('O1-F05', fl, chain)
    for c, f in zip(chain, fl):
        if f['t_s'] + B.OFFSET - B.T0 >= 0.0:
            c.update(q_ok=False, p_x=float('nan'), p_y=float('nan'))
    out, _ = score(tmp_path, 'O1-F05', fl, rel=rel)
    g = out['gate_v3']
    assert out['outcome'] != 'hong_he_do' and all(g[c]['state'] == 'chua_ket_luan' for c in ('g1', 'g2', 'g3', 'g4'))


def test_ct4_3_post_window_missing_is_an_empty_list():
    """CT4-3 (CODE 19/11): a clip without w_s of frames after the command: post khung [] with the reason, never null."""
    frames, chain = _still_chain(n=420)
    kt = dict(ctd.KHUNG_TC_DE_XUAT, nguon='t')
    rows = ctd.cua_so(frames, chain, _log(0.0, -1.0, 12.0), 6.2e6, [0.0], [0.05], kt, 0.10, 2.0, 'O1-F05')
    post = rows[1]
    assert post['loai'] == 'post' and post['khung'] == [] and post['khung_ma_hop_le'] == [] and post['ly_do']


def _hs_locked():
    return dict(trang_thai='khoa_c8', man_trap=dict(nhanh={'O1-F02': 'mot_khung', 'O1-F05': 'mot_khung',
                                                            'O1-F08': 'mot_khung', 'K': 'mot_khung'}),
                khung_tc=dict(n_min=120, frac_min=0.6, p_dung_m=0.003, psi_dung_deg=0.3), tuoi_yaw_max_ms=125.0)


def test_ct3_3_c8_flag_needs_the_chain_and_the_locks():
    """CT3-3 (round 4): a final grid AND the window of its chain AND locked constants: a C8 result; a window built
    straight from the final grid (no chain), or a locked file without the branch table: not."""
    fin = dict(luoi_so_bo_sha256='b' * 64, cua_so_v1_sha256='c' * 64)
    hs = _hs_locked()
    assert cs.cham_c8_why(dict(chuoi_gio_c8=True), fin, hs, 'O1-F05') == []
    assert any('chuoi' in w for w in cs.cham_c8_why(dict(chuoi_gio_c8=False), fin, hs, 'O1-F05'))
    assert any('nhanh' in w for w in cs.cham_c8_why(dict(chuoi_gio_c8=True), fin, dict(hs, man_trap=None), 'O1-F05'))
    assert any('cuoi' in w for w in cs.cham_c8_why(dict(chuoi_gio_c8=True), dict(luoi_so_bo_sha256=None), hs,
                                                    'O1-F05'))


def test_ct3_3_bind_clock_same_file_is_no_chain(tmp_path):
    """CT3-3 (round 4): a window built straight from the final grid binds (diagnostic) but is not the C8 chain."""
    fin = tmp_path / 'gio.json'
    fin.write_text(json.dumps(dict(schema='c8b_gio_v1', luoi_so_bo_sha256='b' * 64, cua_so_v1_sha256='c' * 64)),
                   encoding='utf-8')
    hs = dict(khung_tc=dict(ctd.KHUNG_TC_DE_XUAT, nguon='t'), trang_thai='de_xuat', u_t_ghep_s=0.0, troi_ppm=436.0,
              t_trong_s=0.1)
    pre = dict(gio_sha256=cs.cm.sha256(str(fin)), nguong={k: ctd.KHUNG_TC_DE_XUAT[k] for k in ctd.KHUNG_TC_DE_XUAT},
               hang_so_dung=dict(u_t_ghep_s=0.0, troi_ppm=436.0, t_trong_s=0.1))

    class A:
        gio_c8b, cua_so = str(fin), str(fin)
    assert cs.bind_clock(pre, A, hs) is False


def test_ct3_4_cli_operator_loss_without_the_window_file(tmp_path, capsys):
    """CT3-4 (round 4): --cua-so names a file never written (no edge) and the events hold '#gain_ack ok=0': exit 0 with
    the operator loss in score.json, not a tool error."""
    lane_rows = B.make_frames('O1-F05')
    p = tmp_path / 'frames.csv'
    cs.cm.write_frames(str(p), lane_rows)
    B.hs_file(tmp_path)
    log = B.make_log(tmp_path / 'esp32.csv', 'O1-F05', (0.0, 0.0, 0.0))
    us_events(str(log) + '.events.txt', 0, 10,
              extra='2026-11-19T10:00:40 #gain_ack,tx=1,req_id=1,ok=0,err=crc,sha=x,env=e,mac=m,boot=00000001\n')
    out = tmp_path / 'score.json'
    rc = cs.main(['--frames', str(p), '--esp32', str(log), '--task', 'O1-F05', '--hang-so',
                  str(tmp_path / 'hang_so.json'), '--offset-s', str(B.OFFSET), '--offset-reason', 't',
                  '--offset-anchor-s', '0', '--offset-err-s', '0', '--tuong-doi', '--cua-so', str(tmp_path / 'no.jsonl'),
                  '--tham-chieu', str(tmp_path / 'no.json'), '--out', str(out), '--rows', str(tmp_path / 'rows.csv')])
    assert rc == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    assert d['fail_cause'] == 'thao_tac' and 'loi_nap_gain' in d['thao_tac']


def test_ct3_5_contradicting_copies_and_missing_fields():
    """CT3-5 (round 4, Codex example): two '#us_end' of the same boot and last_data_seq, blk_ticks 0 and 1: a
    contradiction (khoa_sieu_am, no T9 bound); a start without 'on': missing evidence."""
    rows = [dict(_seq=q, _mo=True) for q in range(10, 21)]
    gain = '#gain,id=0,applied_n=1,ctrl=pid,first_seq=10,line_drop=0,boot=00000001'
    st = '#us,ev=start,boot=00000001,on=0,swb=0,first_seq=10,line_drop=0'
    e0 = '#us_end,boot=00000001,on=0,swb=0,changes=0,blk_ticks=0,cut_ticks=0,ticks=11,last_data_seq=20,line_drop=0'
    assert cs.run_us(rows, [gain, st, e0], 1)['ok']
    k, _ = cs.thao_tac_of(rows, [gain, st, e0, e0.replace('blk_ticks=0', 'blk_ticks=1')], 1)
    assert k == ['khoa_sieu_am']
    assert not cs.run_us(rows, [gain, st.replace('on=0,', ''), e0], 1)['ok']
    assert not cs.run_us(rows, [st, e0], 1)['ok']                       # no '#gain' of the run


def test_ct3_5_score_gives_no_t9_bound_from_an_untied_summary(tmp_path, monkeypatch):
    """CT1-20 / CT3-5: score itself does not take seq_max from a summary run_us did not confirm."""
    monkeypatch.setattr(cs, 'run_us', lambda rows, ev, i: dict(ok=False, why=['x'], boot=None, start=[], end=None,
                                                               last=10 ** 9, blk=None))
    monkeypatch.setattr(cs, 'thao_tac_of', lambda rows, ev, i: ([], []))
    lane_rows = B.make_frames('O1-F05')
    fl, chain = to_floor('O1-F05', lane_rows)
    out, _ = score(tmp_path, 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain))
    c = out['khung_tuong_doi_v1']['chot_yaw']
    assert c['seq_max_cung_boot'] is None and c['can_tuoi_ms'] is None


def test_ct2_1_post_end_identity_and_presence():
    """CT2-1 (round 4): a non-empty F8 post set without the post window is refused; dau.post frames / clip of another
    source than the F2 post row: refused."""
    ll, pre, post = _real_ll()
    with pytest.raises(ValueError, match='post'):
        cs.bind_lan_lap(ll, pre, ll['cua_so_v1_sha256'], 'O1-F05', None)
    bad = json.loads(json.dumps(ll))
    bad['dau']['post']['frames_sha256'] = 'z' * 64
    with pytest.raises(ValueError, match='frames_sha256'):
        cs.bind_lan_lap(bad, pre, ll['cua_so_v1_sha256'], 'O1-F05', post)


def test_ct4_1_both_khung_chung_sha_copies():
    """CT4-1: dau.pre.khung_chung_sha256 changed alone (top copy and intersection kept): refused."""
    ll, pre, post = _real_ll()
    bad = json.loads(json.dumps(ll))
    bad['dau']['pre']['khung_chung_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='differs'):
        cs.bind_lan_lap(bad, pre, ll['cua_so_v1_sha256'], 'O1-F05', post)


def test_ct1_20_t_som_follows_the_stencil_result(tmp_path, monkeypatch):
    """CT1-20 (round 4): t_som uses the RESULT of ctd.wz_usable: no usable wz at all moves t_som no later than the
    real flags (a mutation that calls it and keeps the central flag would give the same t_som)."""
    lane_rows = B.make_frames('O1-F05', post=4.0)
    fl, chain = to_floor('O1-F05', lane_rows)
    for c in chain:                                  # no q: the speed is undefined, only wz can rule completion out
        c.update(q_ok=False, p_x=float('nan'), p_y=float('nan'))
    a, _ = score(tmp_path / 'a', 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain), log_kw=dict(post=5.0))
    monkeypatch.setattr(ctd, 'wz_usable', lambda yaw_ok, half: np.zeros(len(yaw_ok), bool))
    b, _ = score(tmp_path / 'b', 'O1-F05', fl, rel=rel_for('O1-F05', fl, chain), log_kw=dict(post=5.0))
    assert b['gate_v3']['g1']['t_som_s'] < a['gate_v3']['g1']['t_som_s']
