"""Tests of c7_u_moc khung (CAM design v4 section 5, can moc dich v3 section 2). SYNTHETIC detections projected from a
known camera (test_c7_33_moc helpers), NOT a measurement; D and the car pose are test numbers."""
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import c7_measure as cm  # noqa: E402
import c7_u_moc as um  # noqa: E402
from test_c7_33_moc import make_session, run_frames, CAR_SIDE  # noqa: E402

D_TEST = 0.0017
GATES = dict(cm.DEFAULT_GATES)


@pytest.fixture(scope='module')
def sess():
    return make_session()


@pytest.fixture(scope='module')
def run(sess):
    frames = run_frames(20, fps=60.0, noise=0.0, car=(0.2, -0.1, math.radians(20.0)))   # >= scale_min neighbours
    rows, _, _ = cm.track_chains(frames, 'syn', 'V', sess, tag_offset=(0.0, 0.0, 0.0))
    return frames, rows


def _cfg(sess, frames, tmp_path):
    p = tmp_path / 'dets.npz'
    cm.save_dets(str(p), frames, 'V', (10, 10))
    return dict(session_obj=sess, dets=str(p), a6=None, D=D_TEST, gates=GATES, dets_sha='DS', lich_sha='LS', run='r1')


def _task(rows, k):
    r = rows[k]
    return (k, dict(tag_x=r['tag_x'], tag_y=r['tag_y'], tag_psi=r['psi'], moc_neo_xe=r['moc_neo_xe'],
                    scale_ref=r['scale_ref']))


def test_seed_is_fixed_by_the_inputs_and_differs_per_frame():
    a = um.seed_of('d', 'l', 'r', 5)
    assert a == um.seed_of('d', 'l', 'r', 5)
    assert len({a, um.seed_of('d', 'l', 'r', 6), um.seed_of('d', 'l', 'r2', 5), um.seed_of('d', 'l2', 'r', 5),
                um.seed_of('d2', 'l', 'r', 5)}) == 5


def test_combinations_lie_in_the_disc_half_on_its_edge():
    s = um.combos(np.random.default_rng(1), 30, D_TEST)
    assert s.shape == (um.N_COMB, 30, 2)
    r = np.hypot(s[..., 0], s[..., 1])
    assert r.max() <= D_TEST * (1 + 1e-12)
    edge = np.isclose(r, D_TEST, rtol=0, atol=1e-15).mean()
    assert 0.45 < edge < 0.55


def test_frame_bound_matches_the_linear_sensitivity(sess, run, tmp_path):
    """The 8-direction sum S_t lies between cos(22.5 deg) and 1 times the linear bound of c7_measure.car_sensitivity
    (sum of spectral norms x D, tag offset 0 so the wheel centre is the tag centre); the problem is near linear at D."""
    frames, rows = run
    um._init(_cfg(sess, frames, tmp_path))
    k = 10
    out = um.khung_mot(_task(rows, k))
    assert out['loi'] == '' and out['n_neo'] == len(str(rows[k]['moc_neo_xe']).split(';'))
    ctx = cm.ChainCtx(sess)
    _, _, d = frames[k]
    anchors = sorted(int(x) for x in str(rows[k]['moc_neo_xe']).split(';'))
    cam = ctx.pose(cm.CHAIN_XE, anchors, d)[0]
    lin = cm.car_sensitivity(cam, anchors, d, ctx.marks, d[cm.CAR_ID], sess['car_side_m'], sess['kappa'],
                             sess['plumb_xy_m'], (0.0, 0.0, 0.0), D_TEST)
    assert math.cos(math.radians(22.5)) * 0.97 * lin['U_xy'] <= out['S_t_m'] <= 1.03 * lin['U_xy']
    assert math.cos(math.radians(22.5)) * 0.97 * lin['U_psi'] <= out['S_psi_rad'] <= 1.03 * lin['U_psi']
    assert out['C_t_m'] <= out['S_t_m'] and out['C_psi_rad'] <= out['S_psi_rad']
    assert out['S_sin'] <= out['S_psi_rad'] / 2 + 1e-15
    assert out['cong_goc'] and out['doi_cong_cuc_dai'] == []


def test_same_seed_same_numbers_and_a_mismatch_blocks_the_frame(sess, run, tmp_path):
    frames, rows = run
    um._init(_cfg(sess, frames, tmp_path))
    a, b = um.khung_mot(_task(rows, 2)), um.khung_mot(_task(rows, 2))
    assert a == b
    k, r = _task(rows, 2)
    r = dict(r, tag_x=r['tag_x'] + 1e-4)                      # frames.csv not reproduced by the refit
    assert um.khung_mot((k, r))['loi'] == 'khop_goc_khac_frames'
    k, r = _task(rows, 2)
    r = dict(r, moc_neo_xe=r['moc_neo_xe'] + ';99')           # an anchor not detected in the frame
    assert um.khung_mot((k, r))['loi'] == 'thieu_phat_hien'


def test_cli_refuses_a_schedule_not_converged(sess, run, tmp_path, monkeypatch):
    frames, rows = run
    sp = tmp_path / 's.json'
    sp.write_text(json.dumps(cm._json(sess)), encoding='utf-8')
    dp = tmp_path / 'dets.npz'
    cm.save_dets(str(dp), frames, 'V', (10, 10))
    fp = tmp_path / 'frames.csv'
    cm.write_frames(str(fp), rows)
    (tmp_path / 'track.json').write_text(json.dumps(dict(tag_offset=[0.0, 0.0, 0.0])), encoding='utf-8')
    for n, o in (('u.json', dict(x=1)), ('lich.json', dict(lich_sha256='L'))):
        (tmp_path / n).write_text(json.dumps(o), encoding='utf-8')
    hs = tmp_path / 'hs.json'
    hs.write_text(json.dumps(dict(cong_khung=GATES)), encoding='utf-8')
    args = ['khung', '--frames', str(fp), '--dets', str(dp), '--session', str(sp), '--u-so', str(tmp_path / 'u.json'),
            '--lich', str(tmp_path / 'lich.json'), '--run', 'r1', '--hang-so', str(hs), '--out', str(tmp_path / 'o')]
    monkeypatch.setattr(cm, 'verify_final', lambda *x, **k: ['not converged'])
    assert um.main(args) == 2
    monkeypatch.setattr(cm, 'verify_final', lambda *x, **k: [])
    monkeypatch.setattr(cm, 'undetected_bound', lambda u: D_TEST)
    assert um.main(args + ['--tu-s', '0.03', '--den-s', '0.05']) == 0
    lines = (tmp_path / 'o' / 'u_moc_khung.jsonl').read_text(encoding='utf-8').splitlines()
    info = json.loads(lines[0])
    assert info['format'] == um.FORMAT_KHUNG and info['bo_lay_mau'] == um.SAMPLER and info['n_loi'] == 0
    assert info['n_khung'] == len(lines) - 1 == 2 and info['huong_ma'] == 'tag_psi'     # frames.csv of c7_measure since 10043dc
    assert info['car_pose']['side_m'] == CAR_SIDE


def test_only_gate_2a_is_a_dependence(sess, run, tmp_path):
    """The floor-fit rms under an alternative map is not a dependence (the observed rms does not move): it is
    reported only. Gate (2a) res_sim is: with car_res_sim_m set just above the base value, the worst poses flip it."""
    frames, rows = run
    cfg = _cfg(sess, frames, tmp_path)
    cfg['gates'] = dict(GATES, rms_extra_px=0.0)
    um._init(cfg)
    out = um.khung_mot(_task(rows, 10))
    assert out['loi'] == '' and out['rms_max_px'] > out['rms_goc_px'] and out['doi_cong_cuc_dai'] == []
    base_res = min(out['xau_xy']['res_sim_m'], out['xau_psi']['res_sim_m'])
    cfg['gates'] = dict(GATES, car_res_sim_m=base_res * 0.999999)
    um._init(cfg)
    out2 = um.khung_mot(_task(rows, 10))
    assert out2['doi_cong_bat_ky'] > 0


# ---------------------------------------------------------------- step 'manifest' (synthetic files, test numbers)
import c7_chuoi_khung as ck  # noqa: E402
import c7_tuong_doi as ctd  # noqa: E402

Q_TEST = (0.1, 0.05)
MANIFEST_NGUON = {'bo_lay_mau', 'car_pose', 'tap_moc', 'lich'}      # gate_c8 d4b2909 MANIFEST_NGUON (CODE 22/11)


class _FakeA6:
    """c8b_a6 stand-in: dpsi_a6 = obj['c'] + obj['k'] * x per frame, all ok."""
    @staticmethod
    def delta_psi_a6(obj, xy, psi, plumb, zc):
        xy = np.asarray(xy, float)
        return obj['c'] + obj.get('k', 0.0) * xy[:, 0], np.ones(len(xy), bool), [''] * len(xy)


def _w(p, o):
    p = str(p)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(o, f)
    return dict(duong_dan=p, sha256=ck.sha256(p))


def _rd(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _sub(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return d


def _world(tmp_path, rows, dong=(8, 15), W=range(0, 5), a6k=None, nodes_s=(0.0, 0.01)):
    """Every file of one scoring: frames, khung, lich, session, A6, tham_chieu, cua_so, two node scores, luoi.json,
    and the two rescorings on moved frames (same results as the scoring)."""
    fp = tmp_path / 'frames.csv'
    cm.write_frames(str(fp), rows)
    (tmp_path / 'track.json').write_text(json.dumps(dict(tag_offset=[0.0, 0.0, 0.0])), encoding='utf-8')
    fr = dict(duong_dan=str(fp), sha256=ck.sha256(str(fp)))
    ses = _w(tmp_path / 'session.json', dict(car_side_m=CAR_SIDE, kappa=0.047, tag_h_m=0.124175))
    lich = _w(tmp_path / 'lich.json', dict(version='c7_lich_v3', hoi_tu=True, tap_moc_file_sha256='e' * 64))
    a6c = _w(tmp_path / 'a6.json', dict(c=0.001))
    a6k = a6c if a6k is None else _w(tmp_path / 'a6_khung.json', a6k)
    other = {k: _w(tmp_path / f'{k}.json', dict(k=k)) for k in ('u_so', 'hang_so', 'dets')}
    head = dict(format=um.FORMAT_KHUNG, bo_lay_mau=um.SAMPLER, run='r1', luot_id='L1',
                car_pose=dict(tag_offset=[0.0, 0.0, 0.0], side_m=CAR_SIDE, kappa=0.047, plumb_xy_m=[-0.48, -0.03],
                              phien_ban='car_pose_v2'),
                nguon=dict(frames=fr, session=ses, lich=lich, a6=a6k, **other))
    recs = []
    for r in rows:
        i = int(r['idx'])
        z = dict(dx=1e-4, dy=-1e-4, dpsi_p=2e-5, res_sim_m=1e-4, scale=1.0, rms_px=1.0)
        recs.append(dict(k=i, idx=i, t_s=r['t_s'], loi='', psi_p=float(ck._f(r['tag_psi'])) - 0.001, dpsi_a6=0.001,
                         S_t_m=0.001 + 1e-5 * i, S_sin=1e-4, C_t_m=5e-4, C_psi_rad=2e-4, S_psi_rad=5e-4 + 1e-6 * i,
                         xau_xy=z, xau_psi=z, doi_cong_cuc_dai=[], doi_cong_bat_ky=0))
    kp = tmp_path / 'u_moc_khung.jsonl'
    kp.write_text('\n'.join(json.dumps(x) for x in [head] + recs) + '\n', encoding='utf-8')
    ids = [int(rows[i]['idx']) for i in W]
    tc = _w(tmp_path / 'tc.json', dict(ids=ids, ids_sha256=ctd.ids_sha(ids)))
    cs = tmp_path / 'cua_so.jsonl'
    cs.write_text(json.dumps(dict(loai='pre', luot_id='L1', frames_sha256=fr['sha256'])) + '\n', encoding='utf-8')
    src = dict(cua_so_v1=dict(duong_dan=str(cs), sha256=ck.sha256(str(cs))), tham_chieu_v1=tc, c8b_a6_v2=a6c)
    a, b = dong

    def node(k, lo, hi, out, extra=None):
        idk = [int(rows[i]['idx']) for i in range(lo, hi + 1)]
        sc = dict(task='O1-F02', frames_sha256=fr['sha256'], hong_cua_so_tc=False,
                  gate_v3=dict(g1=dict(t_done=[0.2, 0.21], t1=None)),
                  khung_tuong_doi_v1=dict(nguon=src, tap='neo', tham_chieu_trang_thai='dat',
                                          mien_phai_phu=dict(nut=k, D_s=0.01 * k, id_khung=idk,
                                                             id_khung_sha256=ctd.ids_sha(idk))))
        sc.update(extra or {})
        return dict(nut=k, **_w(out, sc))

    luoi = dict(nodes_s=list(nodes_s), hong_nut=[False, False], nodes_ok=[True, True], gio_sha256='g' * 64,
                nguon_nut=[node(0, a, b - 1, tmp_path / 'nut_000.json'), node(1, a + 1, b, tmp_path / 'nut_001.json')])
    lp = tmp_path / 'luoi.json'
    _w(lp, luoi)
    nh = {}
    for loai in ('xy', 'psi'):
        d = _sub(tmp_path, f'nhieu_{loai}')
        kn = dict(khung_nhieu=dict(sha256=ck.sha256(str(kp)), loai=loai))
        L = dict(luoi, nguon_nut=[node(0, a, b - 1, d / 'nut_000.json', kn), node(1, a + 1, b, d / 'nut_001.json', kn)])
        nh[loai] = _w(d / 'luoi.json', L)['duong_dan']
    return dict(khung=str(kp), luoi=str(lp), nhieu=nh)


def _edit_khung(w, fn):
    """Edit the khung file (fn(lines) in place), then re-point the moved-frame scorings to its new sha256."""
    with open(w['khung'], encoding='utf-8') as f:
        lines = f.read().splitlines()
    fn(lines)
    with open(w['khung'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    for p in w['nhieu'].values():
        L = _rd(p)
        for e in L['nguon_nut']:
            sc = _rd(e['duong_dan'])
            sc['khung_nhieu']['sha256'] = ck.sha256(w['khung'])
            e.update(_w(e['duong_dan'], sc))
        _w(p, L)


def _set_rec(i, **kv):
    def fn(lines):
        r = json.loads(lines[1 + i])
        r.update(kv)
        lines[1 + i] = json.dumps(r)
    return fn


def _fake_chain(monkeypatch, q=Q_TEST):
    def chain(frames_csv, a6, xoay, lan_lap=None, **kw):
        out = []
        for r in cm.read_frames(frames_csv):
            tx, ty, ps = float(r['tag_x']), float(r['tag_y']), float(ck._f(r['tag_psi'])) - 0.001
            c, s = math.cos(ps), math.sin(ps)
            qq = q or (0.0, 0.0)
            out.append(dict(frame_id=int(r['idx']), tag_x=tx, tag_y=ty, psi_ma_p=ps, p_x=tx - (c * qq[0] - s * qq[1]),
                            p_y=ty - (s * qq[0] + c * qq[1]), q_ok=q is not None,
                            ly_do='' if q else 'thieu_delta_c_B'))
        return out
    monkeypatch.setattr(um.ck, 'chuoi_khung', chain)
    monkeypatch.setattr(um.ck, 'doc_a6', lambda p: dict(obj=_rd(p), mod=_FakeA6, path=p, sha256=ck.sha256(p)))


def _manifest(w, d, name='m.json', nhieu=True):
    out = d / name
    args = ['manifest', '--khung', w['khung'], '--luoi', w['luoi'], '--lan-cham', 'H0', '--out', str(out)]
    if nhieu:
        args += ['--nhieu', f'xy={w["nhieu"]["xy"]}', '--nhieu', f'psi={w["nhieu"]["psi"]}']
    code = um.main(args)
    return code, (_rd(out) if out.exists() else None)


def test_manifest_numbers_domain_and_source_follow_the_design(run, tmp_path, monkeypatch):
    _fake_chain(monkeypatch)
    rows = run[1]
    w = _world(tmp_path, rows)
    code, m = _manifest(w, tmp_path)
    assert code == 0 and m['ok'] is True and m['ly_do'] == []
    assert m['format'] == um.FORMAT_MANIFEST and m['file_cham_sha256'] == ck.sha256(w['luoi'])
    want = [_rd(e['duong_dan'])['khung_tuong_doi_v1']['mien_phai_phu'] for e in _rd(w['luoi'])['nguon_nut']]
    assert m['mien_da_tinh'] == want                                  # gate_c8: equal to mien_phai_phu of every node
    assert set(m['nguon']) == MANIFEST_NGUON and m['nguon']['tap_moc'] == 'e' * 64
    assert set(m['nguon']['car_pose']) == {'tag_offset', 'side_m', 'h_m', 'kappa', 'phien_ban'}
    assert m['nguon']['car_pose']['h_m'] == 0.124175
    aq = math.hypot(*Q_TEST)
    assert m['q']['q_m'] == pytest.approx(list(Q_TEST), abs=1e-12)

    def uxy(i):
        return max(0.001 + 1e-5 * i + 2 * aq * 1e-4, 5e-4 + 2 * aq * math.sin(1e-4))
    idx = [int(r['idx']) for r in rows]
    assert m['U_moc_dong_xy_m'] == pytest.approx(uxy(idx[15]), rel=1e-12)          # domain 8..15 (+0.01 s < 1 frame)
    assert m['U_moc_dong_psi_rad'] == pytest.approx(5e-4 + 1e-6 * idx[15], rel=1e-12)
    assert m['U_moc_tham_xy_m'] == pytest.approx(np.mean([uxy(idx[i]) for i in range(5)]), rel=1e-12)
    assert m['mien']['n_khung'] == 8 and m['mien']['n_W'] == 5
    psi = [float(ck._f(rows[i]['tag_psi'])) - 0.001 for i in range(5)]
    R = abs(np.mean(np.exp(1j * np.array(psi))))
    S = np.mean([2 * math.sin((5e-4 + 1e-6 * idx[i]) / 2) for i in range(5)])
    assert m['U_moc_tham_psi_rad'] == pytest.approx(math.asin(S / R), rel=1e-12)
    assert m['U_moc_hieu_xy_m'] == pytest.approx(m['U_moc_dong_xy_m'] + m['U_moc_tham_xy_m'])
    assert m['a6_sha256'] == ck.sha256(str(tmp_path / 'a6.json')) and m['kiem_a6']['cung_file'] is True


def test_manifest_domain_widened_by_the_largest_grid_step(run, tmp_path, monkeypatch):
    """Grid step 0.025 s > one frame (1/60 s): the domain gains the frame before and after the node union (7, 16),
    not two (0.025 < 2/60)."""
    _fake_chain(monkeypatch)
    rows = run[1]
    w = _world(tmp_path, rows, nodes_s=(0.0, 0.025))
    code, m = _manifest(w, tmp_path)
    assert m['ok'] is True and m['mien']['n_khung'] == 10 and m['mien']['buoc_luoi_s'] == pytest.approx(0.025)
    aq = math.hypot(*Q_TEST)
    i16 = int(rows[16]['idx'])
    assert m['U_moc_dong_xy_m'] == pytest.approx(max(0.001 + 1e-5 * i16 + 2 * aq * 1e-4, 5e-4 + 2 * aq * math.sin(1e-4)))


def test_manifest_not_ok_without_the_dependence_check_or_with_a_changed_t_done(run, tmp_path, monkeypatch):
    _fake_chain(monkeypatch)
    w = _world(tmp_path, run[1])
    code, m = _manifest(w, tmp_path, nhieu=False)
    assert code == 3 and m['ok'] is False and 'chua_kiem_phu_thuoc_xy' in m['ly_do']
    dt = 1.0 / 60
    for n, (shift, ok) in enumerate(((0.9 * dt, True), (1.5 * dt, False))):
        L = _rd(w['nhieu']['psi'])
        e = L['nguon_nut'][1]
        sc = _rd(e['duong_dan'])
        sc['gate_v3']['g1']['t_done'] = [0.2, 0.21 + shift]
        e.update(_w(e['duong_dan'], sc))
        _w(w['nhieu']['psi'], L)
        code, m = _manifest(w, tmp_path, name=f'm{n}.json')
        assert m['ok'] is ok, m['ly_do']


def test_manifest_not_ok_on_a_gate_flip_a_missing_frame_or_no_q(run, tmp_path, monkeypatch):
    _fake_chain(monkeypatch)
    w = _world(tmp_path, run[1])
    _edit_khung(w, _set_rec(10, doi_cong_cuc_dai=[3]))
    code, m = _manifest(w, tmp_path)
    assert code == 3 and m['ly_do'] == ['doi_cong_2a: 1 khung (dau [%d])' % int(run[1][10]['idx'])]
    _edit_khung(w, lambda lines: lines.__delitem__(1 + 12))            # a domain frame missing from the khung file
    code, m = _manifest(w, tmp_path, name='m2.json')
    assert any(x.startswith('thieu_khung: 1') for x in m['ly_do']) and m['U_moc_dong_xy_m'] is None
    w2 = _world(_sub(tmp_path, 'y'), run[1])
    _fake_chain(monkeypatch, q=None)
    code, m = _manifest(w2, tmp_path / 'y')
    assert code == 3 and any(x.startswith('thieu_q') for x in m['ly_do']) and m['U_moc_dong_xy_m'] is None


def test_manifest_a6_of_the_khung_file_against_the_a6_of_the_scoring(run, tmp_path, monkeypatch):
    _fake_chain(monkeypatch)
    w = _world(tmp_path, run[1], a6k=dict(c=0.001, ten='ban khac, cung so'))
    code, m = _manifest(w, tmp_path)
    assert m['ok'] is True and m['kiem_a6']['n_khac'] == 0 and m['kiem_a6']['n_khung'] == len(run[1])
    w = _world(_sub(tmp_path, 'z'), run[1], a6k=dict(c=0.001, k=1e-6))
    code, m = _manifest(w, tmp_path / 'z')
    assert m['ok'] is False and any(x.startswith('a6_khung_khac_a6_cham') for x in m['ly_do'])


def test_manifest_domain_not_stable_between_adjacent_nodes(run, tmp_path, monkeypatch):
    _fake_chain(monkeypatch)
    rows = run[1]
    w = _world(tmp_path, rows, dong=(4, 15))
    L = _rd(w['luoi'])
    e = L['nguon_nut'][0]
    sc = _rd(e['duong_dan'])
    ids = [int(rows[i]['idx']) for i in range(4, 11)]                 # node 0 ends 4 frames before node 1
    sc['khung_tuong_doi_v1']['mien_phai_phu'].update(id_khung=ids, id_khung_sha256=ctd.ids_sha(ids))
    e.update(_w(e['duong_dan'], sc))
    _w(w['luoi'], L)
    code, m = _manifest(w, tmp_path)
    assert any(x.startswith('mien_chua_on_dinh') for x in m['ly_do'])
    # the rescorings name the node scores by sha256 too, so they no longer match the edited node 0
    assert m['ok'] is False


def test_tham_psi_bound_and_frame_bound():
    ut, R, S = um.u_tham_psi([0.0, 0.0], [1e-3, 1e-3])
    assert ut == pytest.approx(math.asin(2 * math.sin(5e-4)), rel=1e-12) and R == pytest.approx(1.0)
    ut, R, S = um.u_tham_psi([0.0, math.pi], [1e-3, 1e-3])             # R = 0: no bound
    assert ut is None
    rec = dict(S_t_m=1e-3, S_sin=1e-4, C_t_m=2e-3, C_psi_rad=1e-3, S_psi_rad=4e-4)
    assert um.u_khung(rec, 0.1) == (pytest.approx(2e-3 + 0.2 * math.sin(5e-4)), 1e-3)


def test_khung_nhieu_moves_the_frames_of_the_khung_file_only(run, tmp_path, monkeypatch):
    _fake_chain(monkeypatch)
    w = _world(tmp_path, run[1])
    _edit_khung(w, _set_rec(3, loi='khop_goc_khong_tin'))
    fp = str(tmp_path / 'frames.csv')
    rows = cm.read_frames(fp)
    out, info = um.khung_nhieu(rows, w['khung'], 'xy', ck.sha256(fp))
    assert info['n_khung_doi'] == len(rows) - 1 and info['loai'] == 'xy' and info['sha256'] == ck.sha256(w['khung'])
    assert out[3] is rows[3]
    assert out[5]['tag_x'] == pytest.approx(rows[5]['tag_x'] + 1e-4)
    assert out[5]['y'] == pytest.approx(rows[5]['y'] - 1e-4)
    assert out[5]['tag_psi'] == pytest.approx(float(rows[5]['tag_psi']) + 2e-5)
    with pytest.raises(ValueError):
        um.khung_nhieu(rows, w['khung'], 'xy', 'f' * 64)


def test_c7_score_khung_nhieu_only_with_tuong_doi(capsys):
    import c7_score
    base = ['--frames', 'f', '--esp32', 'e', '--task', 'O1-F02', '--hang-so', 'h']
    for extra in (['--khung-nhieu', 'k', '--khung-nhieu-loai', 'xy'], ['--tuong-doi', '--khung-nhieu', 'k']):
        assert c7_score.main(base + extra) == 2
        assert '--khung-nhieu and --khung-nhieu-loai go together' in capsys.readouterr().err
