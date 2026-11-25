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


# ── the D2 cell mask (design C8B v7 section 7c line 166; CODE 25/11 after merge 6) ───────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
import c8_u_imu  # noqa: E402
import test_c8b_mat_na_d2 as TM  # noqa: E402   (the node, mask-file and grid helpers of the mask tests)


def _mn(mo, **kw):
    return dict(dict(mo=set(mo), nut_khong_cham={}, ly_do_luot=[]), **kw)


def _node_off(off_of):
    """A node whose camera centre is off_of(j) beside the lane at frame j (frame_id j)."""
    sc, rows = _node(0.0)
    for j, r in enumerate(rows):
        r['x'] = off_of(j)
    return sc, rows


def test_mask_none_and_mask_open_everywhere_change_nothing():
    nd = [_node(0.12), _node(0.12)]
    r0 = E.analyse_run(GIO, nd, _log(), TASK)
    r1 = E.analyse_run(GIO, nd, _log(), TASK, mat_na=_mn(range(1000)))
    assert all(n.pop('n_che') == 0 for n in r1['theo_nut']) and r1 == r0 and r0['vuot_moi_nut']


def test_masked_frame_is_undefined_and_makes_no_conclusion():
    # only frame 7 is over D (0.12 m), at both nodes: without the mask the run is over D at every node
    nd = [_node_off(lambda j: 0.12 if j == 7 else 0.0)] * 2
    r0 = E.analyse_run(GIO, nd, _log(), TASK)
    assert r0['vuot_moi_nut'] and r0['can_xac_dinh_moi_nut'] and r0['theo_nut'][0]['khung_vuot'] == ['7']
    n = r0['theo_nut'][0]['n_khung']
    # frame 7 masked: not over D any more (the masked frame was the only evidence), and the node is not bounded
    r = E.analyse_run(GIO, nd, _log(), TASK, mat_na=_mn(set(range(n)) - {7}))
    assert r['trang_thai'] == 'dat' and not r['vuot_moi_nut'] and not r['can_xac_dinh_moi_nut']
    assert [(x['n_che'], x['n_undef'], x['khung_vuot']) for x in r['theo_nut']] == [(1, 1, [])] * 2
    assert r['theo_nut'][0]['lo_max_m'] < 0.01
    # another frame masked: the finite lower bound of frame 7 stays (design 7c), the node still not bounded
    r = E.analyse_run(GIO, nd, _log(), TASK, mat_na=_mn(set(range(n)) - {3}))
    assert r['vuot_moi_nut'] and not r['can_xac_dinh_moi_nut'] and r['theo_nut'][1]['khung_vuot'] == ['7']


def test_mask_cell_rule_needs_an_open_frame_over_d():
    # frames 7 and 9 over D at both nodes: the cell has a frame over D as long as one of them is open
    nd = [_node_off(lambda j: 0.12 if j in (7, 9) else 0.0)] * 2
    n = len(nd[0][1])
    assert E.analyse_run(GIO, nd, _log(), TASK, mat_na=_mn(set(range(n)) - {7}))['vuot_moi_nut']
    assert not E.analyse_run(GIO, nd, _log(), TASK, mat_na=_mn(set(range(n)) - {7, 9}))['vuot_moi_nut']
    # every frame masked: nothing concluded from the frames
    r = E.analyse_run(GIO, nd, _log(), TASK, mat_na=_mn(()))
    assert not r['vuot_moi_nut'] and not r['can_xac_dinh_moi_nut'] and r['theo_nut'][0]['lo_max_m'] is None


def test_mask_node_turned_not_scored_and_run_reason():
    nd = [_node(0.12), _node(0.12)]
    r = E.analyse_run(GIO, nd, _log(), TASK, mat_na=_mn(range(1000), nut_khong_cham={1: 'cat_t1 on masked frames'}))
    assert r['trang_thai'] == 'dat' and not r['vuot_moi_nut'] and not r['can_xac_dinh_moi_nut']
    assert r['theo_nut'][1] == dict(i=1, cham=False, ly_do=['mat_na_d2: cat_t1 on masked frames'])
    r = E.analyse_run(GIO, nd, _log(), TASK, mat_na=_mn(range(1000), ly_do_luot=['mat_na_d2: no node used']))
    assert r['trang_thai'] == 'chua_ket_luan' and not r['vuot_moi_nut'] and 'mat_na_d2: no node used' in r['ly_do']


def test_mask_needs_a_frame_id_once_on_every_scored_row():
    for keys in (lambda j: '', lambda j: str(j // 2), lambda j: f'{j}.5'):
        with pytest.raises(E.EdoError):
            E.analyse_run(GIO, [_node(0.12, keys=keys)] * 2, _log(), TASK, mat_na=_mn(range(1000)))
    r = E.analyse_run(GIO, [_node(0.12, keys=lambda j: '')] * 2, _log(), TASK)      # without the mask: as before
    assert r['trang_thai'] == 'dat'


def _write_log(p, L):
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['timestamp_us', 'pos_x', 'pos_y', 'run_active', 'reset_n'])
        for i in range(len(L['t'])):
            w.writerow([int(round(L['t'][i] * 1e6)), L['x'][i], L['y'][i], int(L['run'][i]), L['reset'][i]])


KTD = dict(schema='khung_tuong_doi_v1', phi_ok=True, t_som_s=None, ca_t_som=None, tam_doi='khong_doi', mat_neo=False,
           bien_doi=dict(p_dd=[0.0, 0.0], psi_dd=0.0))


def _luoi_that(tmp_path, over=(102,), n_nodes=2, viol=None, **kw):
    """A grid folder as c8b_luoi reads it (the node helpers of the mask tests) whose rows also carry the e_do columns:
    6 frames 100..105 per node, the camera centre 0.12 m beside the lane on the frames in `over`. Returns
    (dir, clock file, log file)."""
    d = str(tmp_path / 'luoi')
    os.makedirs(d)
    gp, lp = str(tmp_path / 'gio.json'), str(tmp_path / 'log.csv')
    nodes_s = [0.001 * k for k in range(n_nodes)]
    json.dump(dict(format='c8b_gio_v1', task=TASK, ok=True, nodes_s=nodes_s, step_s=0.001),
              open(gp, 'w', encoding='utf-8'))
    _write_log(lp, _log())
    fr = [(100 + i, 5.5 + i / 60.0, 1, 1, dict({'g2': 0.03, 'g3': 0.04, 'g4': 0.05, 'g5': -2.0},
                                               **(viol or {}).get(100 + i, {}))) for i in range(6)]
    for k in range(n_nodes):
        json.dump(dict(scorer=dict(commit='abc1234', dirty=False), gio_sha256=c8_u_imu.sha256(gp), D_s=nodes_s[k]),
                  open(os.path.join(d, f'nut_{k:03d}_nguon.json'), 'w', encoding='utf-8'))
        TM.write_node(d, k, fr, D=nodes_s[k], ktd=KTD, **kw)
        rp = os.path.join(d, f'nut_{k:03d}_rows.csv')
        rows = list(csv.DictReader(open(rp, newline='', encoding='utf-8')))
        for r in rows:
            y = 0.4 * (float(r['t_cam_s']) - 5.0)
            r.update(x=0.12 if int(r['frame_id']) in over else 0.0, y=y, s_perp=y, w_lo=y - 0.05, w_hi=y + 0.05,
                     U_p_k=0.005, scored=1)
        with open(rp, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        TM.resha(d, k)
    return d, gp, lp


def _edo(tmp_path, d, gp, lp, mp=None, name='edo.json', luot='r1'):
    out = str(tmp_path / name)
    rc = E.main(['luot', '--gio', gp, '--luoi-dir', d, '--esp32', lp, '--out', out]
                + (['--mat-na-d2', mp, '--luot', luot] if mp else []))
    return rc, (json.load(open(out, encoding='utf-8')) if rc == 0 else None)


def test_cli_luot_reads_the_mask_through_the_masked_grid(tmp_path, capsys):
    d, gp, lp = _luoi_that(tmp_path, over=(102,))
    assert TM.run(d, gp)[0] == 0
    rc, r0 = _edo(tmp_path, d, gp, lp, name='edo0.json')
    assert rc == 0 and r0['vuot_moi_nut'] and r0['can_xac_dinh_moi_nut']
    assert r0['mat_na_d2'] is None and r0['luoi_d2_sha256'] is None
    mp = TM.mask_file(tmp_path, [100, 101, 103, 104, 105])
    # the mask before its masked grid: refused
    assert _edo(tmp_path, d, gp, lp, mp, name='e_som.json')[0] == 2
    assert 'masked grid must be there first' in capsys.readouterr().err
    rc, g = TM.run(d, gp, mp)
    assert rc == 0 and g['mat_na_d2']['nut'][0]['n_che'] == 1
    rc, r = _edo(tmp_path, d, gp, lp, mp)
    assert rc == 0 and r['trang_thai'] == 'dat' and not r['vuot_moi_nut'] and not r['can_xac_dinh_moi_nut']
    assert [(x['n_khung'], x['n_che'], x['n_undef'], x['khung_vuot']) for x in r['theo_nut']] == [(6, 1, 1, [])] * 2
    assert r['mat_na_d2'] == dict(sha256=c8_u_imu.sha256(mp), sha_noi_dung=g['mat_na_d2']['sha_noi_dung'], luot='r1',
                                  lan_cham=g['mat_na_d2']['lan_cham'], n_khung_mo=5, nut_khong_cham=[])
    assert r['luoi_d2_sha256'] == c8_u_imu.sha256(os.path.join(d, 'luoi_d2.json'))
    assert r['gio_sha256'] == r0['gio_sha256'] and r['esp32_sha256'] == r0['esp32_sha256']
    # --mat-na-d2 without --luot, and a run that is not in the target set of the mask
    assert E.main(['luot', '--gio', gp, '--luoi-dir', d, '--esp32', lp, '--out', str(tmp_path / 'x.json'),
                   '--mat-na-d2', mp]) == 2
    assert E.main(['luot', '--gio', gp, '--luoi-dir', d, '--esp32', lp, '--out', str(tmp_path / 'x1.json'),
                   '--luot', 'r1']) == 2                 # --luot alone would run without the mask, silently
    assert not os.path.exists(tmp_path / 'x1.json')
    assert _edo(tmp_path, d, gp, lp, mp, name='x2.json', luot='r7')[0] == 2
    # chung-cu: the masked grid with the e_do file made with its mask; mixed pairs refused
    p0, p2 = os.path.join(d, 'luoi.json'), os.path.join(d, 'luoi_d2.json')
    e0, e2 = str(tmp_path / 'edo0.json'), str(tmp_path / 'edo.json')

    def cc(lu, ed, name):
        return E.main(['chung-cu', '--luoi', lu, '--edo', ed, '--ban-cham-cuoi', 'true', '--out', str(tmp_path / name)])
    assert cc(p2, e2, 'cc.json') == 0 and cc(p0, e0, 'cc0.json') == 0
    c = json.load(open(tmp_path / 'cc.json', encoding='utf-8'))
    assert c['chung_cu_c8']['bo_so'] == [dict(e_do_lo_vuot_D_moi_nut=False, e_do_xac_dinh_moi_nut=False,
                                              g4_can_huu_han_mien_hop_moi_nut=False)]
    assert [x['sha256'] for x in c['nguon']] == [c8_u_imu.sha256(p2), c8_u_imu.sha256(e2)]
    c0 = json.load(open(tmp_path / 'cc0.json', encoding='utf-8'))
    assert c0['chung_cu_c8']['bo_so'] == [dict(e_do_lo_vuot_D_moi_nut=True, e_do_xac_dinh_moi_nut=True,
                                               g4_can_huu_han_mien_hop_moi_nut=True)]
    capsys.readouterr()
    assert cc(p2, e0, 'cc_x1.json') == 2 and cc(p0, e2, 'cc_x2.json') == 2
    assert capsys.readouterr().err.count('both from the same masked grid') == 2


def test_cli_luot_open_frame_over_d_stays_under_the_mask(tmp_path):
    d, gp, lp = _luoi_that(tmp_path, over=(102, 104))
    assert TM.run(d, gp)[0] == 0
    mp = TM.mask_file(tmp_path, [100, 101, 103, 104, 105])
    assert TM.run(d, gp, mp)[0] == 0
    rc, r = _edo(tmp_path, d, gp, lp, mp)
    assert rc == 0 and r['vuot_moi_nut'] and not r['can_xac_dinh_moi_nut']
    assert r['theo_nut'][0]['khung_vuot'] == ['104'] and r['theo_nut'][0]['n_che'] == 1


def test_cli_luot_refuses_what_is_not_of_the_masked_grid(tmp_path, capsys):
    d, gp, lp = _luoi_that(tmp_path)
    assert TM.run(d, gp)[0] == 0
    mp = TM.mask_file(tmp_path, [100, 101, 103, 104, 105])
    assert TM.run(d, gp, mp)[0] == 0
    assert _edo(tmp_path, d, gp, lp, mp, name='ok.json')[0] == 0
    capsys.readouterr()

    def tu_choi(name, txt, mp_=mp, gp_=gp):
        assert _edo(tmp_path, d, gp_, lp, mp_, name=name)[0] == 2
        assert txt in capsys.readouterr().err
    # another mask file of the same run (other open frames): not the mask of the grid
    tu_choi('a.json', 'its mask is not', mp_=TM.mask_file(tmp_path, [100, 101, 102, 103], name='o2.json'))
    # another clock file
    gp2 = str(tmp_path / 'gio2.json')
    json.dump(dict(json.load(open(gp, encoding='utf-8')), step_s=0.002), open(gp2, 'w', encoding='utf-8'))
    tu_choi('b.json', 'another clock file', gp_=gp2)
    # a rows file edited after the masked grid (one frame id changed: another frame would be masked)
    rp = os.path.join(d, 'nut_001_rows.csv')
    old = open(rp, encoding='utf-8', newline='').read()
    assert '\n105,' in old
    open(rp, 'w', encoding='utf-8', newline='').write(old.replace('\n105,', '\n205,'))
    tu_choi('c.json', 'not the rows file the mask was applied on')
    open(rp, 'w', encoding='utf-8', newline='').write(old)
    assert _edo(tmp_path, d, gp, lp, mp, name='ok2.json')[0] == 0
    # a node score file edited after the masked grid
    sp = os.path.join(d, 'nut_000.json')
    old = open(sp, encoding='utf-8', newline='').read()
    json.dump(dict(json.loads(old), U_t_s=0.04), open(sp, 'w', encoding='utf-8'))
    tu_choi('d.json', '(nguon_nut)')
    open(sp, 'w', encoding='utf-8', newline='').write(old)
    assert _edo(tmp_path, d, gp, lp, mp, name='ok3.json')[0] == 0
    capsys.readouterr()
    # the block of the masked grid says other frame counts than the rows give under this mask
    lp2 = os.path.join(d, 'luoi_d2.json')
    g = json.load(open(lp2, encoding='utf-8'))
    g['mat_na_d2']['nut'][0]['n_che'] = 0
    json.dump(g, open(lp2, 'w', encoding='utf-8'))
    tu_choi('e.json', 'frames masked here')


def test_cli_luot_node_the_mask_turned_not_scored(tmp_path):
    # cat_t1 nodes whose certain violation (G2 at frame 101) stands only on a masked frame: not scored in the masked
    # grid (c8b_mat_na_d2 step 5), so not scored for e_do either, though frame 104 (open) is over D
    d, gp, lp = _luoi_that(tmp_path, over=(104,), viol={101: {'g2': 0.14}}, cat=True, t1=0.2)
    assert TM.run(d, gp)[0] == 0
    mp = TM.mask_file(tmp_path, [100, 102, 103, 104, 105])
    rc, g = TM.run(d, gp, mp)
    assert rc == 0 and g['nodes_ok'] == [False, False]
    rc, r = _edo(tmp_path, d, gp, lp, mp)
    assert rc == 0 and not r['vuot_moi_nut'] and not r['can_xac_dinh_moi_nut']
    assert all(x['cham'] is False and 'mat_na_d2' in x['ly_do'][0] for x in r['theo_nut'])
    assert r['mat_na_d2']['nut_khong_cham'] == [0, 1]
    assert _edo(tmp_path, d, gp, lp, name='e0.json')[1]['vuot_moi_nut']           # without the mask: over D


def test_cli_luot_masked_grid_without_a_node(tmp_path):
    # clock file not bounded: the masked grid has no node used (mat_na_d2 null) -> e_do not concluded under the mask
    d, gp, lp = _luoi_that(tmp_path)
    mp = TM.mask_file(tmp_path, [100, 101, 103, 104, 105])
    json.dump(dict(mat_na_d2=None, gio_sha256=c8_u_imu.sha256(gp), states={}),
              open(os.path.join(d, 'luoi_d2.json'), 'w', encoding='utf-8'))
    rc, r = _edo(tmp_path, d, gp, lp, mp)
    assert rc == 0 and r['trang_thai'] == 'chua_ket_luan' and not r['vuot_moi_nut']
    assert any('no node used' in w for w in r['ly_do']) and r['mat_na_d2']['luot'] == 'r1'


def test_chung_cu_mask_pairs():
    e1 = dict(trang_thai='dat', vuot_moi_nut=False, can_xac_dinh_moi_nut=False)
    blk = dict(sha256='a' * 64, sha_noi_dung='b' * 64, luot='r1', lan_cham='c' * 16)
    em = dict(e1, mat_na_d2=dict(blk, n_khung_mo=5, nut_khong_cham=[]), luoi_d2_sha256='d' * 64)
    lm = dict(_luoi(nu=1), mat_na_d2=dict(blk, nut=[]))
    assert E.chung_cu([lm], [em], True)['bo_so'][0]['g4_can_huu_han_mien_hop_moi_nut'] is False
    assert E.chung_cu([lm], [em], True, luoi_sha256=['d' * 64])
    for lu, ed, sha in ((lm, e1, None), (lm, dict(e1, mat_na_d2=None), None), (_luoi(), em, None),
                        (lm, em, ['e' * 64]),
                        (lm, dict(em, mat_na_d2=dict(em['mat_na_d2'], lan_cham='0' * 16)), None),
                        (lm, dict(em, mat_na_d2=dict(em['mat_na_d2'], sha256='0' * 64)), None),
                        (lm, dict(em, mat_na_d2=dict(em['mat_na_d2'], luot='r2')), None)):
        with pytest.raises(E.EdoError):
            E.chung_cu([lu], [ed], True, luoi_sha256=sha)
    # a masked grid without a node used (block null): the pair goes by the SHA of the grid file
    l0 = dict(_luoi(st='chua_ket_luan_khong_nut'), mat_na_d2=None)
    assert E.chung_cu([l0], [em], True, luoi_sha256=['d' * 64])['phan_loai_du'] is False
    with pytest.raises(E.EdoError):
        E.chung_cu([l0], [em], True, luoi_sha256=['e' * 64])
    # two scorings: the first with the mask, the second (H1, no mask of its own) without
    assert len(E.chung_cu([lm, _luoi()], [em, e1], True)['bo_so']) == 2
