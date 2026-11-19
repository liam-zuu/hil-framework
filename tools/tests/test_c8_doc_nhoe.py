"""Tests of tools/c8_doc_nhoe.py (design 03_CODE/docs/2026-11-19_Thiet_ke_CODE_DOCNHOE_v2.md section 7)."""
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
    for r in b[1:]:
        r = list(r)
        r[0] = str(int(r[0]) + len(a) - 1)
        r[1] = '%.6f' % (float(r[1]) + 12.0)
        g.append(r)
    pg = tmp_path / 'g.csv'
    with open(pg, 'w', newline='') as f:
        csv.writer(f).writerows(g)
    dn._check_join(str(p1), str(p2), str(pg))
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
    k.write_text(json.dumps(dict(version=dn.VERSION, du_nhay=True, cu_sha256='x', session_sha256='y', gate_2a_m=0.0015, clips={})), encoding='utf-8')
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


def test_phase_refs_min_count():
    sh = np.tile(dn.norm_shape(_quad()), (20, 1, 1))
    cell = np.array([0] * 4 + [1] * 16)
    ref, n = dn.phase_refs(sh, cell, np.ones(20, bool))
    assert np.isnan(ref[0]).all() and np.isfinite(ref[1]).all() and n[1] == 16


# ── v3: old frames keep the track's verdict (DN2-06) ─────────────────────────────────────────────

def test_combine_old_frames_keep_run_a():
    rows = _rows(res=1.0e-3, scale=1.000)
    take = np.zeros(len(rows), bool)
    for k in range(40, 80):                                  # taken frames with another scale outnumber the old ones
        rows[k]['res_sim_m'] = 2.5e-3
        rows[k]['scale'] = 1.008
        take[k] = True
    out, ra, rb, flips = dn.combine(rows, take, 1.0, dict(cm.DEFAULT_GATES))
    for k in range(len(rows)):
        if not take[k]:
            assert out[k]['valid'] == ra[k]['valid'] and out[k]['why'] == ra[k]['why']
    assert all(out[k]['res_sim_m'] == rows[k]['res_sim_m'] for k in range(len(rows)))


def test_combine_nothing_taken_is_run_a():
    rows = _rows(res=1.0e-3)
    out, ra, rb, flips = dn.combine(rows, np.zeros(len(rows), bool), 1.0, dict(cm.DEFAULT_GATES))
    assert [r['valid'] for r in out] == [r['valid'] for r in ra] and not flips


def test_cong_refuses_wrong_version(tmp_path):
    k = tmp_path / 'khoa.json'
    k.write_text(json.dumps(dict(version='c8_doc_nhoe_v1', du_nhay=True)), encoding='utf-8')
    rc = dn.main(['cong', '--khoa', str(k), '--c1', 'a', 'b', '--c2', 'c', 'd', '--session', 's', '--cu', 'u',
                  '--ghep', 'g', '--frames-a', 'f', '--out', str(tmp_path / 'o')])
    assert rc == 2 and not (tmp_path / 'o').exists()
