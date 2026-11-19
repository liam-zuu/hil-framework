"""Tests of tools/c8_doc_nhoe.py (design 03_CODE/docs/2026-11-19_Thiet_ke_CODE_DOCNHOE_v5.md; tests of v3 6 and v4 6)."""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path.insert(0, TOOLS)
import c7_measure as cm                                                  # noqa: E402
import c8_doc_nhoe as dn                                                 # noqa: E402

REAL = r'D:\CLAUDE_THESIS\05_DATA\results\C8\2026-11-18_run01\C1'
SIDE = 0.1445


# ── v_t ──────────────────────────────────────────────────────────────────────────────────────────

def test_omega_loo_uniform_and_gaps():
    t = np.arange(200) / 60.0
    ang = 3.0 * t + 0.2 * t ** 2
    seen = np.ones(200, bool)
    om = dn.omega_loo(t, ang, seen)
    k = np.flatnonzero(np.isfinite(om))
    assert np.allclose(om[k], 3.0 + 0.4 * t[k], atol=1e-9)
    assert np.isnan(om[0]) and np.isnan(om[-1])          # one side short
    seen[100:104] = False                                # a 5-frame gap: 0.083 s > 0.05 s
    om2 = dn.omega_loo(t, ang, seen)
    assert np.isnan(om2[99]) and np.isnan(om2[104]) and np.isfinite(om2[90]) and np.isfinite(om2[112])
    ang2 = ang.copy()
    ang2[50] += 0.5                                      # a wrong frame does not move its own rate
    om3 = dn.omega_loo(t, ang2, np.ones(200, bool))
    assert om3[50] == pytest.approx(om[50], abs=1e-9)


# ── gates ────────────────────────────────────────────────────────────────────────────────────────

def _rows(n=120, res=1e-3, scale=0.98):
    rows = []
    for k in range(n):
        rows.append(dict(idx=k, t_s=k / 60.0, valid=False, why=[], n_marks=24, rms_px=1.0, shift_px=1.0, cam_z=2.6,
                         car_seen=True, x=0.1 * math.cos(k / 20), y=0.1 * math.sin(k / 20), psi=k / 20, tag_x=0.0,
                         tag_y=0.0, res_m=2e-3, res_sim_m=res, scale=scale, scale_ref=float('nan'), row_px=900.0,
                         vx=float('nan'), vy=float('nan'), wz=float('nan'), speed=float('nan'), shift_mag_px=1.0,
                         col_px=2000.0, _dec={}))
    return rows


def test_regate_take_and_neighbours():
    rows = _rows(res=2.0e-3)
    take = np.zeros(len(rows), bool)
    out = dn.regate_with(rows, take, 1.0, dict(cm.DEFAULT_GATES))
    assert not any(r['valid'] for r in out) and all('xe_meo' in r['why'] for r in out)
    take[:] = True
    out = dn.regate_with(rows, take, 1.0, dict(cm.DEFAULT_GATES))
    assert all(r['valid'] for r in out[10:-10])
    assert all(r['res_sim_m'] == 2.0e-3 for r in out)    # the measured residual is written, not 0
    # a cluster of wrong scale among taken frames fails (2b) against its neighbours
    rows2 = _rows(res=2.0e-3)
    for r in rows2[60:63]:
        r['scale'] = 0.95
    out = dn.regate_with(rows2, take, 1.0, dict(cm.DEFAULT_GATES))
    assert all('ti_le' in r['why'] for r in out[60:63])


@pytest.mark.skipif(not os.path.isdir(REAL), reason='real track not on this machine')
def test_regate_nothing_taken_equals_track():
    with open(os.path.join(REAL, 'calib_tagB', 'session.json'), encoding='utf-8') as f:
        s = json.load(f)
    C = dn.load_clip(os.path.join(REAL, 'B_c1_track', 'frames.csv'))
    out = dn.regate_with(C['rows'], np.zeros(len(C['t']), bool), s['rms_px_H'], dict(cm.DEFAULT_GATES))
    assert all(a['valid'] == b['valid'] and a['why'] == b['why'] for a, b in zip(out, C['rows']))


# ── files ────────────────────────────────────────────────────────────────────────────────────────

def _write(path, rows):
    cm.write_frames(str(path), rows)


def test_load_clip_checks(tmp_path):
    rows = _rows()
    p = tmp_path / 'f.csv'
    _write(p, rows)
    C = dn.load_clip(str(p))
    assert C['seen'].all()
    rows[5]['t_s'] = rows[4]['t_s']
    p2 = tmp_path / 'g.csv'
    _write(p2, rows)
    with pytest.raises(dn.DnError):
        dn.load_clip(str(p2))
    p3 = tmp_path / 'h.csv'
    with open(p3, 'w', newline='') as f:
        csv.writer(f).writerow(['idx', 't_s'])
    with pytest.raises(dn.DnError):
        dn.load_clip(str(p3))


def test_check_join(tmp_path):
    rows = _rows()
    p1, p2 = tmp_path / 'a.csv', tmp_path / 'b.csv'
    _write(p1, rows)
    _write(p2, rows)
    with open(p1, newline='') as f:
        a = list(csv.reader(f))
    with open(p2, newline='') as f:
        b = list(csv.reader(f))
    g = [a[0]] + a[1:]
    shift = float(a[-1][1]) + dn.JOIN_GAP_S - float(b[1][1])
    for r in b[1:]:
        r = list(r)
        r[0] = str(int(r[0]) + len(a) - 1)
        r[1] = '%.6f' % (float(r[1]) + shift)
        g.append(r)
    pg = tmp_path / 'g.csv'
    with open(pg, 'w', newline='') as f:
        csv.writer(f).writerows(g)
    dn._check_join(str(p1), str(p2), str(pg))
    g2 = [list(r) for r in g]                                # DNC1-15: a gap other than 10 s is refused
    for r in g2[len(a):]:
        r[1] = '%.6f' % (float(r[1]) + 1.0)
    pg2 = tmp_path / 'g2.csv'
    with open(pg2, 'w', newline='') as f:
        csv.writer(f).writerows(g2)
    with pytest.raises(dn.DnError):
        dn._check_join(str(p1), str(p2), str(pg2))
    g[5][5] = '9'
    with open(pg, 'w', newline='') as f:
        csv.writer(f).writerows(g)
    with pytest.raises(dn.DnError):
        dn._check_join(str(p1), str(p2), str(pg))


def test_cong_refuses_without_sensitive_khoa(tmp_path):
    k = tmp_path / 'khoa.json'
    k.write_text(json.dumps(dict(version=dn.VERSION, du_nhay=False)), encoding='utf-8')
    rc = dn.main(['cong', '--khoa', str(k), '--c1', 'a', 'b', '--c2', 'c', 'd', '--session', 's', '--cu', 'u',
                  '--ghep', 'g', '--frames-a', 'f', '--out', str(tmp_path / 'o')])
    assert rc == 2 and not (tmp_path / 'o').exists()


def test_cong_refuses_existing_out(tmp_path, monkeypatch):
    k = tmp_path / 'khoa.json'
    k.write_text(json.dumps(dict(version=dn.VERSION, du_nhay=True, cu_sha256='x', session_sha256='y', gate_2a_m=0.0015, clips={}, gates=dn._clean(dict(cm.DEFAULT_GATES)))), encoding='utf-8')
    monkeypatch.setattr(dn, 'read_cu', lambda p: dict(sha256='x', R0_m=0.25))
    monkeypatch.setattr(dn, 'load_session', lambda p: (dict(car_side_m=SIDE, rms_px_H=1.0), 'y'))
    (tmp_path / 'o').mkdir()
    rc = dn.main(['cong', '--khoa', str(k), '--c1', 'a', 'b', '--c2', 'c', 'd', '--session', 's', '--cu', 'u',
                  '--ghep', 'g', '--frames-a', 'f', '--out', str(tmp_path / 'o')])
    assert rc == 2


def test_criteria_cells():
    def cell(band, grp, n, sd_dr, sd_hp, med_dr=0.0, med_hp=0.0):
        st = lambda m, s: dict(n=n, trung_vi=m, sd=s, p95_abs=2 * s, max_abs=3 * s)   # noqa: E731
        return dict(dai=band, nhom=grp, n=n, vong=5.0, dr_m=st(med_dr, sd_dr), hp_rad=st(med_hp, sd_hp),
                    tiep_loo_m=st(0.0, 1e-4))
    cells = [cell('0.8-1.0', 'cu', 100, 5e-5, 0.002), cell('0.8-1.0', 'moi', 100, 5e-5, 0.002, med_dr=0.2e-3),
             cell('1.0-1.166', 'cu', 0, 0, 0), cell('1.0-1.166', 'moi', 100, 0.3e-3, 0.002)]
    cells[2]['dr_m'] = cells[2]['hp_rad'] = None
    cr = {c['dai']: c for c in dn._criteria(cells)}
    assert cr['0.8-1.0']['moi'] == 'dat' and cr['0.8-1.0']['chong'] == 'truot'
    assert cr['1.0-1.166']['moi'] == 'truot' and cr['1.0-1.166']['chong'] == 'thieu'


# ── v3: v_t details (DN2-04) ─────────────────────────────────────────────────────────────────────

def test_omega_loo_target_left_out_before_unwrap():
    """DN2-04: a wild target angle (pi off) must not create a 2 pi step in its own window."""
    t = np.arange(60) / 60.0
    ang = np.mod(0.6 * t, 2 * math.pi)
    ang[30] = ang[30] + math.pi
    om = dn.omega_loo(t, ang, np.ones(60, bool))
    assert om[30] == pytest.approx(0.6, abs=1e-6)


def test_omega_loo_wrapped_input_and_bad_neighbour():
    t = np.arange(120) / 60.0
    ang = np.mod(5.0 * t, 2 * math.pi) - math.pi            # wrapped every turn
    om = dn.omega_loo(t, ang, np.ones(120, bool))
    k = np.flatnonzero(np.isfinite(om))
    assert len(k) > 100 and np.allclose(om[k], 5.0, atol=1e-6)
    ang2 = 5.0 * t
    ang2[60] += 0.3                                          # a wrong neighbour: left and right rates disagree
    om2 = dn.omega_loo(t, ang2, np.ones(120, bool))
    assert np.isnan(om2[58]) or abs(om2[58] - 5.0) < 0.5     # either refused or not pulled far


# ── v3: shapes ───────────────────────────────────────────────────────────────────────────────────

def _quad(cx=500.0, cy=400.0, side=92.0, ang=0.3, skew=None):
    h = side / 2
    q = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    if skew is not None:
        q = q + skew
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    return q @ R.T + [cx, cy]


def test_shape_dev_similarity_invariant():
    ref = dn.norm_shape(_quad())
    q = _quad(cx=900.0, cy=100.0, side=80.0, ang=-1.0)
    assert np.max(np.abs(dn.shape_dev(dn.norm_shape(q), ref, SIDE))) < 1e-12


def test_moved_corner_shows_in_e_hinh():
    """One corner moved d mm: e_hinh about d / 2 (the similarity fit takes part of it)."""
    ref = dn.norm_shape(_quad())
    B = np.zeros((2, 8))
    ppm = 92.0 / SIDE
    for mm in (3.0, 5.0, 8.0):
        sk = np.zeros((4, 2))
        sk[2] = [mm * 1e-3 * ppm, 0.0]
        q = _quad(skew=sk)
        e = dn.e_hinh(dn.shape_dev(dn.norm_shape(q), ref, SIDE), 1.2, B)[0]
        assert 0.45 * mm * 1e-3 < e < 0.8 * mm * 1e-3


def test_huber_pattern_robust():
    rng = np.random.default_rng(3)
    P = rng.normal(0, 1e-3, (2, 8))
    v = rng.uniform(0, 1.6, 3000)
    dev = np.column_stack([v, v ** 2]) @ P + rng.normal(0, 1e-4, (3000, 8))
    bad = rng.choice(3000, 300, replace=False)
    dev[bad] += rng.normal(0, 5e-3, (300, 8))                # 10 % bad frames
    B = dn.huber_pattern(v, dev)
    assert np.max(np.abs(B - P)) < 2e-4
    e = dn.e_hinh(dev, v, B)
    good = np.setdiff1d(np.arange(3000), bad)
    assert np.percentile(e[good], 99) < 0.6e-3


# ── v3: old frames keep the track's verdict (DN2-06) ─────────────────────────────────────────────

def _track(rows):
    """The rows as the track writes them (gated once, nothing taken)."""
    return dn.regate_with(rows, np.zeros(len(rows), bool), 1.0, dict(cm.DEFAULT_GATES))


def test_combine_old_frames_keep_run_a():
    rows = _rows(res=1.0e-3, scale=1.000)
    take = np.zeros(len(rows), bool)
    for k in range(40, 80):                                  # taken frames with another scale outnumber the old ones
        rows[k]['res_sim_m'] = 2.5e-3
        rows[k]['scale'] = 1.008
        take[k] = True
    rows = _track(rows)
    out, ra, rb, flips = dn.combine(rows, take, 1.0, dict(cm.DEFAULT_GATES))
    for k in range(len(rows)):
        if not take[k]:
            assert out[k]['valid'] == ra[k]['valid'] and out[k]['why'] == ra[k]['why']
    assert all(out[k]['res_sim_m'] == rows[k]['res_sim_m'] for k in range(len(rows)))


def _same(a, b):
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def test_combine_old_rows_unchanged_taken_rows_from_run_b():
    """DNC1-01: a frame not taken keeps every field of the track row (speeds included); a taken frame gets valid,
    why, scale_ref and the speeds of run B."""
    rows = _rows(res=1.0e-3)
    take = np.zeros(len(rows), bool)
    for k in range(50, 70):
        rows[k]['res_sim_m'] = 2.5e-3
        take[k] = True
    rows = _track(rows)
    out, ra, rb, flips = dn.combine(rows, take, 1.0, dict(cm.DEFAULT_GATES))
    for k in range(len(rows)):
        if take[k]:
            for c in ('valid', 'why', 'scale_ref', 'vx', 'vy', 'wz', 'speed'):
                assert _same(out[k][c], rb[k][c])
            assert out[k]['res_sim_m'] == 2.5e-3
        else:
            assert all(_same(out[k][c], rows[k][c]) for c in rows[k])


def test_combine_nothing_taken_is_run_a():
    rows = _track(_rows(res=1.0e-3))
    out, ra, rb, flips = dn.combine(rows, np.zeros(len(rows), bool), 1.0, dict(cm.DEFAULT_GATES))
    assert [r['valid'] for r in out] == [r['valid'] for r in ra] and not flips


def test_cong_refuses_wrong_version(tmp_path):
    k = tmp_path / 'khoa.json'
    k.write_text(json.dumps(dict(version='c8_doc_nhoe_v1', du_nhay=True)), encoding='utf-8')
    rc = dn.main(['cong', '--khoa', str(k), '--c1', 'a', 'b', '--c2', 'c', 'd', '--session', 's', '--cu', 'u',
                  '--ghep', 'g', '--frames-a', 'f', '--out', str(tmp_path / 'o')])
    assert rc == 2 and not (tmp_path / 'o').exists()


# ── v5: synthetic tag image ──────────────────────────────────────────────────────────────────────

import cv2                                                               # noqa: E402

CACHE = os.environ.get('C8_DOC_NHOE_CACHE', '')                         # prefix of a real crop cache (optional)


def _synth(ang=0.2, side=92.0, c=(110.0, 110.0), blur=1.0, bg=200):
    canvas, can = dn.template()
    h = side / 2
    q = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    q = q @ R.T + np.asarray(c)
    H = cv2.getPerspectiveTransform(can, q.astype(np.float32))
    img = cv2.warpPerspective(canvas, H, (220, 220), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=float(bg))
    img = cv2.GaussianBlur(img, (0, 0), blur)
    return np.clip(img, 0, 255).astype(np.uint8), q


def test_refine_edges_diag_same_corners_synthetic():
    det = cm.make_detector()
    for ang in (0.0, 0.3, 1.1):
        img, q = _synth(ang=ang)
        ref = cm.detect_refined(det, img)
        qd, dg = dn.detect_diag(det, img)
        assert cm.CAR_ID in ref and qd is not None
        assert np.array_equal(ref[cm.CAR_ID], qd)
        assert len(dg) == 4 and all(d['kept'] >= dn.KEPT_MIN and d['rms_px'] < 0.2 for d in dg)
        assert np.max(np.abs(qd - q)) < 0.3


@pytest.mark.skipif(not (CACHE and os.path.exists(CACHE + '.npz')), reason='no real crop cache (C8_DOC_NHOE_CACHE)')
def test_refine_edges_diag_same_corners_real_crops():
    crops = np.load(CACHE + '.npz')['crops']                 # decompressed once
    det = cm.make_detector()
    n = 0
    for j in range(0, len(crops), max(1, len(crops) // 300)):
        g = crops[j]
        ref = cm.detect_refined(det, g)
        qd, _ = dn.detect_diag(det, g)
        if cm.CAR_ID in ref:
            assert qd is not None and np.array_equal(ref[cm.CAR_ID], qd)
            n += 1
        else:
            assert qd is None
    assert n > 100


def _det_q(img):
    q, dg = dn.detect_diag(cm.make_detector(), img)
    return q


def test_injection_families_synthetic():
    img, _ = _synth(ang=0.25)
    q0 = _det_q(img)
    ppm = dn.side_px(q0) / SIDE
    base = dict(goc=1, huong=0.7, giua=False, o=(2, 3), huong2=2.0)
    ref = dn.norm_shape(q0)
    B0 = np.zeros((2, 8))
    # P: one corner moved 5 mm shows in the shape
    q = _det_q(dn.inject(img, q0, ppm, dict(base, fam='P', mag=5.0)))
    assert q is not None and dn.e_hinh(dn.shape_dev(dn.norm_shape(q), ref, SIDE), 0.0, B0)[0] > 2e-3
    # S: the whole tag shifted 3 mm, shape unchanged
    q = _det_q(dn.inject(img, q0, ppm, dict(base, fam='S', mag=3.0)))
    sh = (q - q0).mean(axis=0)
    assert np.linalg.norm(sh) / ppm == pytest.approx(3e-3, abs=0.1e-3)
    assert dn.e_hinh(dn.shape_dev(dn.norm_shape(q), ref, SIDE), 0.0, B0)[0] < 0.3e-3
    # PS: a corner and a shift
    q = _det_q(dn.inject(img, q0, ppm, dict(base, fam='PS', mag=(3.0, 3.0))))
    assert q is not None and np.linalg.norm((q - q0).mean(axis=0)) > 2.0
    # B: one inner cell flipped: the car tag is lost
    assert _det_q(dn.inject(img, q0, ppm, dict(base, fam='B', mag=1))) is None
    # Od, Ob, E: the reading changes (or the tag is lost)
    for fam, mag in (('Od', 15), ('Ob', 15), ('E', 3)):
        q = _det_q(dn.inject(img, q0, ppm, dict(base, fam=fam, mag=mag)))
        assert q is None or np.max(np.abs(q - q0)) > 0.05


def test_draw_params_paired_and_deterministic():
    a = [dn.draw_params(f, m, np.random.default_rng(1)) for f, m in dn.FAMILIES]
    b = [dn.draw_params(f, m, np.random.default_rng(1)) for f, m in dn.FAMILIES]
    assert a == b


# ── v5: phase references from other passes (DN3-04) ─────────────────────────────────────────────

def test_refset_other_passes_only():
    sq = dn.norm_shape(_quad())
    sk = np.zeros((4, 2))
    sk[0] = [0.02 * 92, 0.0]
    odd = dn.norm_shape(_quad(skew=sk))
    angs = np.r_[np.full(6, 0.01), np.full(6, 0.01 + 2 * math.pi)]   # 6 frames on each of two passes, cell 0
    shapes = np.r_[np.tile(odd, (6, 1, 1)), np.tile(sq, (6, 1, 1))]
    rs = dn.RefSet(np.arange(12), np.zeros(12, int), angs, shapes)
    assert np.allclose(rs.ref(0, 0.01), sq)                   # its own pass left out
    assert np.allclose(rs.ref(0, 0.01 + 2 * math.pi), odd)
    rs2 = dn.RefSet(np.arange(8), np.zeros(8, int), np.r_[np.full(4, 0.0), np.full(4, 2 * math.pi)],
                    np.tile(sq, (8, 1, 1)))
    assert rs2.ref(0, 0.0) is None                             # 4 < PHASE_MIN on the other pass
    assert rs.ref(-1, 0.0) is None and rs.ref(5, 0.0) is None


# ── v5: the main gate (b), (f), (g), (h) ─────────────────────────────────────────────────────────

def _gate(noise=0.0, vmax=10.0, n=240):
    rng = np.random.default_rng(5)
    t = np.arange(n) / 60.0
    ang = 4.0 * t + 0.3 * t ** 2 + rng.normal(0, noise, n)
    seen = np.ones(n, bool)
    om, pred, win, ts, an = dn.loo_all(t, ang, seen)
    sq = dn.norm_shape(_quad())
    A = dict(vt=np.abs(om) * 0.25, cell=dn.phase_cell(pred), ang=ang, win=win, ts=ts, an=an, R0=0.25)
    ra = np.array([c * 2 * math.pi / 72 + 0.01 + p * 2 * math.pi for p in range(-3, 6) for c in range(72)
                   for _ in range(6)])
    rs = dn.RefSet(np.arange(len(ra)), dn.phase_cell(ra), ra, np.tile(sq, (len(ra), 1, 1)))
    G = dn.Gate(A, rs, np.zeros((2, 8)), np.zeros((3, 8)), vmax, 1e-3, 0.3, SIDE)
    return G, A, sq


def _F4(kept=24, rms=0.1):
    return np.array([[kept, rms, 50.0, 1.5]] * 4, float)


def test_gate_passes_clean_and_res_sim_quant():
    G, A, sq = _gate()
    k = 120
    assert G.judge(k, sq, np.zeros(8), _F4(), 1e-3)[0] == ''
    assert G.judge(k, sq, np.zeros(8), _F4(), dn.RES_SIM_CAP_M - 4e-11)[0] == 'res_sim_vuot_tran'
    assert G.judge(k, sq, np.zeros(8), _F4(), dn.RES_SIM_CAP_M - 6e-11)[0] == ''
    assert G.judge(k, sq, np.zeros(8), _F4(kept=19), 1e-3)[0] == 'canh_co_ban'
    assert G.judge(k, sq, np.zeros(8), _F4(rms=0.6), 1e-3)[0] == 'canh_co_ban'
    assert G.judge(k, sq, np.full(8, 0.5), _F4(), 1e-3)[0] == 'r_canh_vuot'


def test_gate_vt_stable_leave_each_neighbour_out():
    G, A, sq = _gate(noise=0.004)
    k = 120
    vt = A['vt'][k]
    dr = G.drops(k)
    assert len(dr) == A['win'][k][2] - A['win'][k][1] - 1
    hi = max(d[0] for d in dr if d is not None)
    assert hi > vt                                         # some neighbour left out raises v_t
    G2, _, _ = _gate(noise=0.004, vmax=0.5 * (vt + hi))    # the frame is in range, one leave-out is not
    assert G2.judge(k, sq, np.zeros(8), _F4(), 1e-3)[0] == 'vt_khong_ben'
    G3, _, _ = _gate(noise=0.004, vmax=hi + 1e-9)
    assert G3.judge(k, sq, np.zeros(8), _F4(), 1e-3)[0] == ''


def test_gate_moved_corner_fails_shape():
    G, A, sq = _gate()
    sk = np.zeros((4, 2))
    sk[2] = [3e-3 * 92 / SIDE, 0.0]
    assert G.judge(120, dn.norm_shape(_quad(skew=sk)), np.zeros(8), _F4(), 1e-3)[0] == 'e_hinh_vuot'


# ── v5: temporal check (i) ───────────────────────────────────────────────────────────────────────

def _circle_clip(n=300, R=0.25, w=4.0):
    t = np.arange(n) / 60.0
    th = w * t + 0.4 * t ** 2
    return dict(t=t, x=R * np.cos(th), y=R * np.sin(th), psi=th + 0.3, seen=np.ones(n, bool)), R


def test_temporal_one_smooth_and_shift():
    C, R = _circle_clip()
    nidx = np.arange(len(C['t']))
    r = dn.temporal_one(C['t'], C['x'], C['y'], C['psi'], nidx, 150, (0.0, 0.0), R)
    assert max(abs(r[0]), abs(r[1]), abs(r[2])) < 1e-5
    assert r[3] == pytest.approx(R * (4.0 + 0.8 * C['t'][150]), rel=1e-3)
    x = C['x'].copy()
    x[150] += 2e-3
    r = dn.temporal_one(C['t'], x, C['y'], C['psi'], nidx, 150, (0.0, 0.0), R)
    assert math.hypot(r[0], r[1]) == pytest.approx(2e-3, rel=0.05)
    # DN5-04: the target pi off on the heading does not change the neighbours' unwrap
    psi = C['psi'].copy()
    psi[150] += math.pi
    r = dn.temporal_one(C['t'], C['x'], C['y'], psi, nidx, 150, (0.0, 0.0), R)
    assert abs(abs(r[2]) - math.pi) < 1e-4
    assert dn.temporal_one(C['t'], C['x'], C['y'], C['psi'], nidx, 0, (0.0, 0.0), R) is None


def test_temporal_two_passes_bad_neighbour():
    C, R = _circle_clip()
    C['x'] = C['x'].copy()
    C['x'][150] += 3e-3                                     # one bad frame of N
    old = np.ones(len(C['t']), bool)
    TH = dict(a_m=0.2e-3, b_s=0.5e-3, T_phap_m=0.3e-3, T_huong_rad=math.radians(0.3))
    res, qua, ok, _, N2 = dn.temporal_check(C, old, np.zeros(len(C['t']), bool), (0.0, 0.0), R, TH)
    assert ok[150] and not qua[150] and not N2[150]
    assert all(qua[k] for k in range(140, 161) if k != 150 and ok[k])


# ── v5: injection table ──────────────────────────────────────────────────────────────────────────

def test_inj_table_rules():
    def r(fam, mag, cu, moi):
        return dict(ho=fam, do_lon=mag, cu=cu, moi=moi)

    def acc(c, p, **kw):
        return dict(tang='nhan', loi_tam_m=c, loi_huong_deg=p, **kw)
    rej = dict(tang='truot_hinh', loi_tam_m=3e-3, loi_huong_deg=1.0)
    rows = [r('P', 2.0, acc(0.6e-3, 0.1), acc(0.52e-3, 0.1)), r('P', 2.0, rej, rej),
            r('P', 3.0, acc(0.1e-3, 0.1), acc(0.7e-3, 0.1)),
            r('S', 1.0, acc(1.0e-3, 0.01), acc(1.04e-3, 0.02, qua_i=False))]
    tab = dn.inj_table(rows)
    assert tab['P_2.0']['dat'] and tab['P_2.0']['cu']['nhan_sai'] == 1 and tab['P_2.0']['moi']['sai_that'] == 2
    assert not tab['P_3.0']['dat'] and 'nhan_sai' in tab['P_3.0']['ly_do']
    assert tab['S_1.0']['dat'] and tab['S_1.0']['bao_cao_max']['tam_vuot']    # the max is reported only (v5 8)
    assert tab['S_1.0']['moi_co_i']['nhan'] == 0
    assert tab['P_4.0']['n'] == 0 and tab['P_4.0']['dat']
    assert tab['P_2.0']['dieu_kien'] and not tab['S_1.0']['dieu_kien'] and not tab['PS_2.0_1.0']['dieu_kien']
    assert tab['P_2.0']['theo_cap']['ca_hai_nhan'] == 1


def test_gate_vt_lost_when_leaving_out_is_its_own_reason():
    """design v5 7: a leave-one-out that makes v_t undefined is 'vt_mat_khi_bo' (not taken; health: cannot score)."""
    G, A, sq = _gate()
    k = 120
    j, a, b = A['win'][k]
    G._drops[k] = [None] + G.drops(k)[1:]
    assert G.judge(k, sq, np.zeros(8), _F4(), 1e-3)[0] == 'vt_mat_khi_bo'



# ── round 1 fixes ────────────────────────────────────────────────────────────────────────────────

def test_loo_gap_bridged_by_the_target_is_refused():
    """DNC1-02: the frame's own PTS must not split a gap of the neighbours (a, b) into two halves under 50 ms."""
    t = np.arange(120) / 60.0
    ang = 4.0 * t
    seen = np.ones(120, bool)
    seen[[59, 61]] = False                                   # neighbours of 60 at -/+ 33.3 ms: a 66.7 ms hole
    om = dn.omega_loo(t, ang, seen)
    assert np.isnan(om[60]) and np.isfinite(om[50])
    C = dict(t=t, x=0.25 * np.cos(ang), y=0.25 * np.sin(ang), psi=ang, seen=seen)
    assert dn.temporal_one(t, C['x'], C['y'], C['psi'], np.flatnonzero(seen), 60, (0.0, 0.0), 0.25) is None


def test_gate_flip_wins_over_lost_vt():
    """DNC1-08: a leave-out with a defined v_t that flips the verdict is reported even when another leave-out has no
    v_t."""
    G, A, sq = _gate(vmax=10.0)
    k = 120
    d = list(G.drops(k))
    d[0] = None
    d[1] = (20.0, d[1][1])                                   # out of the pattern range: the verdict flips
    G._drops[k] = d
    assert G.judge(k, sq, np.zeros(8), _F4(), 1e-3)[0] == 'vt_khong_ben'


def test_scale_check_matches_apply_gates():
    """(2b) used to score the injected frames equals c7_measure.apply_gates on the same pool."""
    rows = _rows(res=1.0e-3)
    for k in (30, 31):
        rows[k]['scale'] = 0.96
    tr = _track(rows)
    C = dict(t=np.array([r['t_s'] for r in tr]), scale=np.array([r['scale'] for r in tr]))
    pool = dn.p12_mask(tr)
    g = dict(cm.DEFAULT_GATES)
    for k in range(len(tr)):
        ok, why = dn.scale_check(C, pool, k, C['scale'][k], g)
        assert ok == tr[k]['valid'], k


def test_long_runs_by_idx():
    """DNC1-11: a lost frame ends a run."""
    take = np.zeros(40, bool)
    take[5:12] = True
    take[13:20] = True                                       # 12 is not taken (lost or old): two runs of 7
    C = dict(seen=np.ones(40, bool))
    r = dn.long_runs(C, take)
    assert r['so_cum'] == 2 and r['dai_nhat'] == 7


def test_cong_refuses_flag_against_tables(tmp_path):
    """DNC1-09: du_nhay true with a clip verdict not met is refused."""
    k = tmp_path / 'khoa.json'
    k.write_text(json.dumps(dict(version=dn.VERSION, du_nhay=True, clips=dict(c1=dict(du_nhay=False, ly_do=['x']),
                                                                                c2=dict(du_nhay=True, ly_do=[])))),
                 encoding='utf-8')
    rc = dn.main(['cong', '--khoa', str(k), '--c1', 'a', 'b', '--c2', 'c', 'd', '--session', 's', '--cu', 'u',
                  '--ghep', 'g', '--frames-a', 'f', '--out', str(tmp_path / 'o')])
    assert rc == 2 and not (tmp_path / 'o').exists()


def test_inj_table_paired_only_new_wrong():
    """design v5 8: a wrong fault only the new gate accepts, with no such fault for the old gate, fails the family."""
    acc = dict(tang='nhan', loi_tam_m=0.8e-3, loi_huong_deg=0.1)
    rej = dict(tang='truot_2a', loi_tam_m=0.8e-3, loi_huong_deg=0.1)
    ok = dict(tang='nhan', loi_tam_m=0.1e-3, loi_huong_deg=0.01)
    rows = [dict(ho='P', do_lon=3.0, cu=rej, moi=acc), dict(ho='P', do_lon=3.0, cu=acc, moi=acc),
            dict(ho='P', do_lon=3.0, cu=acc, moi=ok)]
    d = dn.inj_table(rows)['P_3.0']
    assert d['cu']['nhan_sai'] == 2 and d['moi']['nhan_sai'] == 2
    assert d['theo_cap']['chi_moi_nhan_sai'] == 1 and d['theo_cap']['chi_cu_nhan_sai'] == 0
    assert not d['dat'] and d['ly_do'] == ['chi_moi_nhan_sai']


# ── round 2 fixes ────────────────────────────────────────────────────────────────────────────────

def test_gate_verdict_equals_combine():
    """DNC2-01: the verdict of one frame from the slice run equals cong's combine on the whole clip."""
    rows = _rows(n=200, res=1.0e-3)
    take = np.zeros(200, bool)
    for k in range(80, 120):
        rows[k]['res_sim_m'] = 2.5e-3
        take[k] = k % 3 != 0
    for k in (100, 101):
        rows[k]['scale'] = 0.99
    tr = _track(rows)
    out, ra, rb, _ = dn.combine(tr, take, 1.0, dict(cm.DEFAULT_GATES))
    C = dict(t=np.array([r['t_s'] for r in tr]), rows=tr)
    for k in range(200):
        ok, why = dn.gate_verdict(C, k, tr[k], bool(take[k]), take, 1.0, dict(cm.DEFAULT_GATES))
        assert ok == out[k]['valid'] and why == out[k]['why'], k


def test_temporal_local_equals_two_passes():
    """DNC2-02: (i) of one frame from the local two passes equals temporal_check on the whole chain."""
    C, R = _circle_clip()
    C['x'] = C['x'].copy()
    C['x'][148] += 3e-3                                     # a bad neighbour of 150 (dropped in pass 1)
    C['x'][150] += 0.25e-3                                  # the frame itself, slightly off
    TH = dict(a_m=0.2e-3, b_s=0.5e-3, T_phap_m=0.3e-3, T_huong_rad=math.radians(0.3))
    old = np.ones(len(C['t']), bool)
    res, qua, ok, _, _ = dn.temporal_check(C, old, np.zeros(len(C['t']), bool), (0.0, 0.0), R, TH)
    for k in (140, 147, 149, 150, 151, 160):
        q = dn.temporal_local(C, C['x'], C['y'], C['psi'], old, k, (0.0, 0.0), R, TH)
        assert q == (bool(qua[k]) if ok[k] else None), k


def test_clip_verdict_from_tables():
    """DNC2-04: the clip verdict is recomputed from the stored detail."""
    env = dict(tam_m=[1.0, 1.1, 1.2], huong_deg=[0.4, 0.5, 0.6])
    env_old = dict(tam_m=[1.3, 1.4, 1.5], huong_deg=[0.7, 0.8, 0.9])
    acc = dict(tang='nhan', loi_tam_m=0.1e-3, loi_huong_deg=0.01)
    det = [dict(ho=f, do_lon=m, cham=i, nhanh=100 + i, cu=acc, moi=acc) for f, m in dn.FAMILIES
           for i in range(dn.INJ_PAIRS)]
    K = dict(e_max_m=1.3e-3, r_max=0.35, bao_cu_cham=env_old, bao_moi_nhanh=env,
             bom_loi=dict(cap=[[i, 100 + i] for i in range(dn.INJ_PAIRS)], khong_cham_cap=[], chi_tiet=det),
             suc_khoe=dict(cham_duoc=5000, truot=2))
    assert dn.clip_verdict(K) == []
    bad = dict(tang='nhan', loi_tam_m=0.8e-3, loi_huong_deg=0.01)
    K2 = dict(K, bom_loi=dict(K['bom_loi'], chi_tiet=det[:1] + [dict(det[1], moi=bad)] + det[2:]))
    assert any('P_2.0' in w for w in dn.clip_verdict(K2))
    K3 = dict(K, bao_moi_nhanh=dict(env, tam_m=[1.0, 1.1, 1.6]))
    assert any('bao tam_m' in w for w in dn.clip_verdict(K3))
