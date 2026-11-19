"""c8b_edo (J v6 section 3 e_do, section 4 level rule, section 5 (i)-(ii); design C8B v7 section 9 item 6)."""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c8b_edo as E  # noqa: E402

TASK = 'O1-K1'                     # lane along +y in the lane frame, S = 2 m


def _log(t0=5.0, n=800, dt=0.01, x_off=0.0, reset_at=None, edge_pos=(0.0, 0.0)):
    """Firmware log: run_active rises at t0; pos follows y = 0.4 (t - t0) on x = x_off (edge row as given); reset_at:
    the IMU chip reset count reset_n goes 3 -> 4 there (not a firmware restart)."""
    t = 4.0 + np.arange(n) * dt
    run = t >= t0
    y = np.where(run, 0.4 * (t - t0), 0.0) + edge_pos[1]
    x = np.full(n, x_off + edge_pos[0])
    reset = ['3' if reset_at is None or ti < reset_at else '4' for ti in t]
    return dict(t=t, x=x, y=y, run=run, reset=reset, restart=[])


def _node(off, keys=None, up=0.005, D=0.0, phi_ok=True, **kt):
    """Score dict + rows of one node: camera centre at x = off beside the lane, y = 0.4 t_cam (t_cam from 5.1 s)."""
    rows = []
    for j, tc in enumerate(np.arange(5.1, 8.0, 1 / 60)):
        k = str(j) if keys is None else keys(j)
        y = 0.4 * (tc - 5.0)
        rows.append(dict(frame_id=k, t_cam_s=tc, x=off, y=y, s_perp=y, w_lo=y - 0.05, w_hi=y + 0.05, U_p_k=up,
                         scored=1))
    blk = dict(schema='khung_tuong_doi_v1', phi_ok=phi_ok, t_som_s=None, ca_t_som=None, tam_doi='khong_doi',
               mat_neo=False, bien_doi=dict(p_dd=[0.0, 0.0], psi_dd=0.0))
    blk.update(kt)
    return dict(khung_tuong_doi_v1=blk, gio_c8b=dict(D=D)), rows


GIO = dict(format='c8b_gio_v1', task=TASK, nodes_s=[0.0, 0.002], step_s=0.002)


def test_offset_beyond_d_at_every_node():
    r = E.analyse_run(GIO, [_node(0.12), _node(0.12)], _log(), TASK)
    assert r['trang_thai'] == 'dat' and r['vuot_moi_nut'] and r['can_xac_dinh_moi_nut']
    assert r['theo_nut'][0]['lo_max_m'] == pytest.approx(0.12 - 0.005, abs=1e-3)


def test_offset_inside_d_not_over():
    r = E.analyse_run(GIO, [_node(0.09), _node(0.12)], _log(), TASK)
    assert r['trang_thai'] == 'dat' and not r['vuot_moi_nut']


def test_cell_rule_needs_one_frame_at_both_nodes_of_a_cell():
    # node 0 over D only on even frames, node 1 only on odd frames: no frame over at both nodes of the cell
    a, b = {'0': (0.12, 0.13), '1': (0.05, 0.06)}, {'0': (0.05, 0.06), '1': (0.12, 0.13)}
    assert not E.cell_rule([a, b]) and E.cell_rule([a]) and E.cell_rule([a, a])
    # three nodes: cell 1 by frame 0, cell 2 by frame 1 (frames may differ between cells)
    c = {'0': (0.12, 0.13), '1': (0.12, 0.13)}
    assert E.cell_rule([a, c, b])


def test_edge_row_pose_used_as_logged():
    # the firmware edge row is 0.11 m off (not set to 0): e_do sees it
    r = E.analyse_run(GIO, [_node(0.0), _node(0.0)], _log(edge_pos=(0.11, 0.0)), TASK)
    assert r['vuot_moi_nut'] and r['theo_nut'][1]['lo_max_m'] == pytest.approx(0.105, abs=1e-3)


def test_sign_change_gives_zero_lower_bound():
    lane = E.lane_of(TASK)
    lo, hi = E.frame_bounds(lane, 0.0, 1.0, np.array([-0.01, 0.01]), np.array([1.0, 1.0]), 1.0, 0.95, 1.05, 0.002)
    assert lo == 0.0 and hi == pytest.approx(0.012)


@pytest.mark.parametrize('kw,txt', [(dict(phi_ok=False), 'phi_ok'), (dict(ca_t_som='thieu q'), 't_som'),
                                    (dict(tam_doi='doi'), 'tam_doi'), (dict(mat_neo=True), 'mat_neo')])
def test_run_not_concluded(kw, txt):
    r = E.analyse_run(GIO, [_node(0.12, **kw), _node(0.12, **kw)], _log(), TASK)
    assert r['trang_thai'] == 'chua_ket_luan' and not r['vuot_moi_nut'] and any(txt in w for w in r['ly_do'])


def test_second_reset_not_concluded(tmp_path):
    # J v6 section 3: a second reset is a firmware restart: timestamp_us falls back in the log (CODE 19/11)
    L = _log()
    p = tmp_path / 'log.csv'
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['timestamp_us', 'pos_x', 'pos_y', 'run_active', 'reset_n'])
        for i in range(len(L['t'])):
            ts = L['t'][i] if L['t'][i] < 6.0 else L['t'][i] - 5.9       # boot at 6.0 s: the clock starts again
            w.writerow([int(round(ts * 1e6)), L['x'][i], L['y'][i], int(L['run'][i]), '3'])
    Lr = E.read_log(str(p))
    assert Lr['restart'] == [200]
    r = E.analyse_run(GIO, [_node(0.12), _node(0.12)], Lr, TASK)
    assert r['trang_thai'] == 'chua_ket_luan' and any('second reset' in w for w in r['ly_do'])
    assert not r['vuot_moi_nut']


def test_imu_chip_reset_is_not_a_second_reset():
    # CODE 19/11 (CAM checked AGV logger.c 147, main.c 299): reset_n counts IMU chip resets, not firmware boots; it
    # is not a reason for e_do (the scorer loses the run to the command path, J v6 section 5 (1))
    r = E.analyse_run(GIO, [_node(0.12), _node(0.12)], _log(reset_at=6.0), TASK)
    assert r['trang_thai'] == 'dat' and r['vuot_moi_nut'] and not any('reset' in w for w in r['ly_do'])


def test_node_not_scored_means_not_bounded_everywhere():
    r = E.analyse_run(GIO, [_node(0.12), None], _log(), TASK)
    assert r['trang_thai'] == 'dat' and not r['can_xac_dinh_moi_nut'] and not r['vuot_moi_nut']


def _run(lid, ctrl, vuot, du=True, ban='cuoi', two=None, hong=False, danh=True):
    one = dict(trang_thai='dat', vuot_moi_nut=vuot, can_xac_dinh_moi_nut=du)
    lc = [one] if two is None else [one, dict(one, vuot_moi_nut=two)]
    return dict(luot_id=lid, muc='O1-F05', controller=ctrl, danh_sach=danh, hong=hong, ban_cham=ban, lan_cham=lc)


def test_level_one_run_is_enough_and_ladrc_missing_does_not_block():
    # J v6 section 4 example: PID one run over at every node, one not; LADRC without bound -> level out
    m = E.muc([_run('a', 'PID', True), _run('b', 'PID', False), _run('c', 'LADRC', False, du=False)])
    lv = m['muc'][0]
    assert lv['loai_muc'] and lv['trang_thai'] == 'loai_muc' and lv['controller']['PID']['luot_chung_cu'] == ['a']


def test_level_two_scorings_need_both_and_final_only():
    # J3-1: over by H first, not by H last -> not evidence; a provisional scoring (thieu_H1) is not evidence (J5-7)
    m = E.muc([_run('a', 'PID', True, two=False), _run('b', 'PID', True, ban='tam'), _run('c', 'LADRC', False)])
    lv = m['muc'][0]
    assert not lv['loai_muc'] and lv['controller']['PID']['luot_chung_cu'] == []


def test_level_kept_with_risk_when_a_controller_has_no_bound():
    m = E.muc([_run('a', 'PID', False), _run('b', 'LADRC', False, du=False), _run('c', 'LADRC', True, hong=True)])
    lv = m['muc'][0]
    assert lv['trang_thai'] == 'giu_muc_rui_ro' and 'LADRC' in lv['c8_chua_xong']
    m2 = E.muc([_run('a', 'PID', False), _run('b', 'LADRC', False)])
    assert m2['muc'][0]['trang_thai'] == 'giu_muc'


def test_cli_luot(tmp_path):
    d = tmp_path / 'luoi'
    d.mkdir()
    for k in range(2):
        sc, rows = _node(0.12)
        (d / f'nut_{k:03d}.json').write_text(json.dumps(sc), encoding='utf-8')
        with open(d / f'nut_{k:03d}_rows.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    L = _log()
    lp = tmp_path / 'log.csv'
    with open(lp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['timestamp_us', 'pos_x', 'pos_y', 'run_active', 'reset_n'])
        for i in range(len(L['t'])):
            w.writerow([int(round(L['t'][i] * 1e6)), L['x'][i], L['y'][i], int(L['run'][i]), L['reset'][i]])
    gp = tmp_path / 'gio.json'
    gp.write_text(json.dumps(GIO), encoding='utf-8')
    out = tmp_path / 'edo.json'
    assert E.main(['luot', '--gio', str(gp), '--luoi-dir', str(d), '--esp32', str(lp), '--out', str(out)]) == 0
    r = json.loads(out.read_text(encoding='utf-8'))
    assert r['vuot_moi_nut'] and r['gio_sha256'] and r['esp32_sha256']


def _luoi(ok=True, hi=0.08, nu=0, st='cham'):
    return dict(trang_thai_luot=st, nodes_ok=[ok, True], states={c: 'dat' for c in ('g1', 'g2', 'g3', 'g4', 'g5')},
                nums=dict(g4=dict(hi=hi, n_undef=nu)))


def test_chung_cu_one_and_two_scorings():
    e1 = dict(trang_thai='dat', vuot_moi_nut=True, can_xac_dinh_moi_nut=True)
    c = E.chung_cu([_luoi()], [e1], True)
    assert c == dict(phan_loai_du=True, ban_cham_cuoi=True, bo_so=[dict(
        e_do_lo_vuot_D_moi_nut=True, e_do_xac_dinh_moi_nut=True, g4_can_huu_han_mien_hop_moi_nut=True)])
    c2 = E.chung_cu([_luoi(), _luoi(hi=None)], [e1, dict(e1, trang_thai='chua_ket_luan')], False)
    assert c2['bo_so'][1] == dict(e_do_lo_vuot_D_moi_nut=False, e_do_xac_dinh_moi_nut=False,
                                  g4_can_huu_han_mien_hop_moi_nut=False) and c2['ban_cham_cuoi'] is False
    assert not E.g4_finite_every_node(_luoi(ok=False)) and not E.g4_finite_every_node(_luoi(nu=1))
    assert not E.chung_cu([_luoi(st='chua_ket_luan_khong_nut')], [e1], True)['phan_loai_du']
    with pytest.raises(E.EdoError):
        E.chung_cu([_luoi()], [e1, e1], True)


def test_ct5_3_f13_key_of_the_scorer():
    # Codex CAM CT5-3: the scorer writes the F13 block under khung_tuong_doi_v1; the old key is not taken
    sc, rows = _node(0.12)
    r = E.analyse_run(GIO, [(sc, rows), (sc, rows)], _log(), TASK)
    assert r['trang_thai'] == 'dat' and r['vuot_moi_nut']
    old = dict(sc)
    old['khung_tuong_doi'] = old.pop('khung_tuong_doi_v1')
    r2 = E.analyse_run(GIO, [(old, rows), (old, rows)], _log(), TASK)
    assert r2['trang_thai'] == 'chua_ket_luan' and any('khung_tuong_doi_v1' in w for w in r2['ly_do'])
