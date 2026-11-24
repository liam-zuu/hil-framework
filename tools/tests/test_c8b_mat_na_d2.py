"""c8b_mat_na_d2 + c8b_luoi --gop-only --mat-na-d2 (design C8B v7 7c line 166; CODE 24/11 khuya). Synthetic node files
whose gate numbers are made from their rows the way c7_score.build_gate makes them (o1_gate.aggregate on the common
and union domains, cond_state2)."""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'fwsim'))
import c8_u_imu  # noqa: E402
import c8b_a6 as A  # noqa: E402
import c8b_luoi as LU  # noqa: E402
import c8b_mat_na_d2 as M  # noqa: E402
import o1_gate  # noqa: E402

QC = {'g2': 'e_perp', 'g3': 'e_trk', 'g4': 'e_bao', 'g5': 'e_theta_cmd_deg'}
U = {'g2': 0.01, 'g3': 0.01, 'g4': 0.01, 'g5': 0.5}


def frames(n=6, viol=None):
    """[(frame_id, t_cam_s, mien_chung, mien_hop, {c: q})]; viol {frame_id: {c: q}}."""
    out = []
    for i in range(n):
        q = {'g2': 0.03, 'g3': 0.04, 'g4': 0.05, 'g5': -2.0}
        q.update((viol or {}).get(100 + i, {}))
        out.append((100 + i, 10.0 + i / 60.0, 1, 1, q))
    return out


def write_node(d, k, fr, D=1.0, cat=False, t1=None, edit=None, undef=(), ktd=None, g1=None):
    """undef: frame ids without a bound (lo -inf, hi +inf for G2-G5, q kept); ktd: more fields of
    khung_tuong_doi_v1; g1: more fields of the gate g1 block."""
    per = {c: ([], [], []) for c in LU.MEASURED}
    for fid, _, _, _, q in fr:
        for c in LU.MEASURED:
            v = abs(q[c]) if c == 'g5' else q[c]
            per[c][0].append(v)
            per[c][1].append(-math.inf if fid in undef else v - U[c])
            per[c][2].append(math.inf if fid in undef else v + U[c])
    mc = np.array([f[2] == 1 for f in fr])
    mu = np.array([f[3] == 1 for f in fr])
    g = {'g1': dict(dict(state='dat', t1=t1), **(g1 or {}))}   # no t_done: it would move between nodes (C8BS-6)
    for c in LU.MEASURED:                                # c7_score.build_gate
        q, lo, hi = (np.array(x) for x in per[c])
        thr = o1_gate.THRESHOLDS[c]
        ac = o1_gate.aggregate(q[mc], lo[mc], hi[mc], thr)
        au = o1_gate.aggregate(q[mu], lo[mu], hi[mu], thr)
        g[c] = dict({x: au[x] for x in ('q', 'lo', 'hi', 'n_undef', 'n_frames')},
                    state=LU._state_from_numbers(ac['lo'], au['lo'], au['hi'], au['n_undef'], thr),
                    chung={x: ac[x] for x in ('q', 'lo', 'hi', 'n_undef')})
    if edit:
        edit(g)
    rp = os.path.join(d, f'nut_{k:03d}_rows.csv')
    with open(rp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['frame_id', 't_cam_s', 'mien_chung', 'mien_hop'] + [QC[c] for c in LU.MEASURED]
                   + [f'{b}_{c}' for c in LU.MEASURED for b in ('lo', 'hi')])
        for j, (fid, t, a, b, q) in enumerate(fr):
            w.writerow([fid, t, a, b] + [q[c] for c in LU.MEASURED]
                       + [per[c][x][j] for c in LU.MEASURED for x in (1, 2)])
    with open(os.path.join(d, f'nut_{k:03d}.json'), 'w', encoding='utf-8') as f:     # the scorer names its rows file
        json.dump(dict(outcome='dat', gate_v3=g, U_t_s=0.03, t0_log_us=5e6, gio_c8b=dict(D=D), cat_t1=cat,
                       khung_tuong_doi_v1=dict(dict(rows_sha256=c8_u_imu.sha256(rp)), **(ktd or {}))),
                  f, allow_nan=True)


def resha(d, k):
    """The node score names the rows file as it is now (after a test edited the rows)."""
    sp = os.path.join(d, f'nut_{k:03d}.json')
    sc = json.load(open(sp, encoding='utf-8'))
    sc['khung_tuong_doi_v1']['rows_sha256'] = c8_u_imu.sha256(os.path.join(d, f'nut_{k:03d}_rows.csv'))
    json.dump(sc, open(sp, 'w', encoding='utf-8'))


def grid(tmp_path, fr_by_node, **kw):
    d = str(tmp_path / 'luoi')
    os.makedirs(d)
    nodes_s = [1.0 + 0.001 * k for k in range(len(fr_by_node))]
    gp = str(tmp_path / 'gio.json')
    json.dump(dict(format='c8b_gio_v1', task='O1-F05', ok=True, nodes_s=nodes_s), open(gp, 'w', encoding='utf-8'))
    for k, fr in enumerate(fr_by_node):
        with open(os.path.join(d, f'nut_{k:03d}_nguon.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(scorer=dict(commit='abc1234', dirty=False), gio_sha256=c8_u_imu.sha256(gp),
                           D_s=nodes_s[k]), f)
        write_node(d, k, fr, D=nodes_s[k], **kw)
    return d, gp


def mask_file(tmp_path, open_ids, luot='r1', name='o.json', drop_khung_mo=False):
    mn = dict(o_dat=['A'], o_thieu=[dict(o_id='B', ly_do=['vuot'])], ngoai_dai_u=[],
              khung_mo=[[luot, i] for i in open_ids] + [['r9', 5]])
    if drop_khung_mo:
        mn.pop('khung_mo')
    d = A.with_sha(dict(schema=A.SCHEMA_O, version=A.SCHEMA_O, muc_dich='d2', trang_thai='dat',
                        tap=dict(khop=['c1'], kiem=['c2'], dich=['r1', 'r9']), mat_na=mn))
    p = str(tmp_path / name)
    json.dump(d, open(p, 'w', encoding='utf-8'))
    return p


def run(d, gp, mp=None, luot='r1', them=()):
    base = ['--gio', gp, '--out-dir', d, '--gop-only'] + list(them)
    if mp:
        base += ['--mat-na-d2', mp, '--luot', luot]
    rc = LU.main(base)
    out = os.path.join(d, 'luoi_d2.json' if mp else 'luoi.json')
    return rc, (json.load(open(out, encoding='utf-8')) if rc == 0 else None)


def test_mask_open_everywhere_changes_nothing(tmp_path):
    fr = frames(viol={102: {'g3': 0.15}})
    d, gp = grid(tmp_path, [fr, fr])
    rc0, r0 = run(d, gp)
    rc1, r1 = run(d, gp, mask_file(tmp_path, range(100, 106)))
    assert rc0 == 0 and rc1 == 0
    assert r1['states'] == r0['states'] and r1['nums'] == r0['nums'] and r1['outcome'] == r0['outcome'] == 'khong_dat'
    assert r1['mat_na_d2']['nut'][0]['n_che'] == 0 and 'mat_na_d2' not in r0


def test_masked_violation_becomes_undefined_other_stays(tmp_path):
    # frame 102: G3 violation in a masked cell -> undefined (n_undef); frame 104: G2 violation in an open cell stays
    fr = frames(viol={102: {'g3': 0.15}, 104: {'g2': 0.14}})
    d, gp = grid(tmp_path, [fr, fr])
    _, r0 = run(d, gp)
    assert r0['states']['g3'] == 'khong_dat' and r0['states']['g2'] == 'khong_dat'
    rc, r = run(d, gp, mask_file(tmp_path, [100, 101, 103, 104, 105]))
    assert rc == 0
    assert r['states']['g3'] == 'chua_ket_luan' and r['nums']['g3']['n_undef'] == 1
    assert r['states']['g2'] == 'khong_dat'                         # a finite violation elsewhere stays (design 7c)
    assert r['states']['g1'] == 'chua_ket_luan'                     # step 4 (L156, Liam 25/11)
    assert r['nums']['g3']['q'] == pytest.approx(0.15)               # q kept as a diagnostic (the scorer's rule)
    e = r['mat_na_d2']
    assert e['luot'] == 'r1' and e['n_khung_mo'] == 5 and e['nut'][0]['n_che'] == 1 and e['nut'][0]['g1_che']
    assert e['sha256'] == c8_u_imu.sha256(str(tmp_path / 'o.json'))
    assert os.path.exists(os.path.join(d, 'luoi.json'))             # the plain grid file stays beside it


def test_mask_never_adds_dat(tmp_path):
    rng = np.random.default_rng(7)
    for t in range(15):
        viol = {100 + int(i): {'g3': float(rng.uniform(0.0, 0.2))} for i in rng.integers(0, 6, 3)}
        fr = frames(viol=viol)
        sub = tmp_path / f'c{t}'
        os.makedirs(sub)
        d, gp = grid(sub, [fr, fr])
        _, r0 = run(d, gp)
        _, r = run(d, gp, mask_file(sub, [int(i) for i in rng.choice(range(100, 106), 3, replace=False)]))
        for c in LU.CONDS:
            assert not (r['states'][c] == 'dat' and r0['states'][c] != 'dat')
            assert not (r['states'][c] == 'khong_dat' and r0['states'][c] != 'khong_dat')


def test_self_check_refuses_numbers_not_from_the_rows(tmp_path):
    fr = frames()
    d, gp = grid(tmp_path, [fr, fr])

    def edit(g):
        g['g4']['chung']['q'] = 0.051
    write_node(d, 1, fr, D=1.001, edit=edit)
    rc, _ = run(d, gp, mask_file(tmp_path, range(100, 106)))
    assert rc == 2
    assert run(d, gp)[0] == 0                                        # the plain grid does not check this


def test_frame_not_in_d2_rows_is_masked(tmp_path):
    fr = frames(n=4)
    d, gp = grid(tmp_path, [fr])
    assert run(d, gp)[0] == 0
    rc, r = run(d, gp, mask_file(tmp_path, [100, 101, 102]))       # 103 not listed (no D2 row): masked
    assert rc == 0 and r['mat_na_d2']['nut'][0]['n_che'] == 1 and r['nums']['g2']['n_undef'] == 1
    rw = r['mat_na_d2']['nut'][0]['rows']                               # GATE: the rows file named with its sha
    assert rw['sha256'] == c8_u_imu.sha256(os.path.join(d, 'nut_000_rows.csv')) and os.path.isabs(rw['duong_dan'])
    p = os.path.join(d, 'nut_000_rows.csv')                             # a repeated frame_id: refused
    lines = open(p, encoding='utf-8').read().split('\n')
    lines = ['102,' + x[4:] if x.startswith('103,') else x for x in lines]
    open(p, 'w', encoding='utf-8', newline='').write('\n'.join(lines))
    resha(d, 0)                                                         # refused for the repeated id, not the sha
    os.remove(os.path.join(d, 'luoi_d2.json'))
    assert run(d, gp, str(tmp_path / 'o.json'))[0] == 2


def test_mask_file_checks(tmp_path):
    fr = frames()
    d, gp = grid(tmp_path, [fr])
    assert run(d, gp, mask_file(tmp_path, range(100, 106)))[0] == 2      # D2-1: luoi.json must be there first
    assert run(d, gp)[0] == 0
    assert run(d, gp, mask_file(tmp_path, range(100, 106)), luot='r7')[0] == 2          # run not in tap.dich
    assert run(d, gp, mask_file(tmp_path, range(100, 106), name='n.json', drop_khung_mo=True))[0] == 2
    p = mask_file(tmp_path, range(100, 106), name='e.json')
    x = json.load(open(p, encoding='utf-8'))
    x['mat_na']['khung_mo'].append(['r1', 999])                       # edited after the sha
    json.dump(x, open(p, 'w', encoding='utf-8'))
    assert run(d, gp, p)[0] == 2
    assert LU.main(['--gio', gp, '--out-dir', d, '--mat-na-d2', p, '--luot', 'r1']) == 2   # not --gop-only


def test_cat_node_loses_its_violation_before_t1(tmp_path):
    fr = frames(viol={101: {'g2': 0.14}})
    d, gp = grid(tmp_path, [fr], cat=True, t1=0.2)
    nodes = LU.load_nodes(json.load(open(gp, encoding='utf-8')), d)
    assert nodes[0]['ok'] and nodes[0]['cat']
    mn = M.doc_mat_na(mask_file(tmp_path, [100, 102, 103, 104, 105]), 'r1')
    out, tom = M.ap_dung(nodes, d, mn, LU._state_from_numbers)
    assert not out[0]['ok'] and out[0]['kind'] == 'loi' and 'cat_t1' in out[0]['why']
    nodes = LU.load_nodes(json.load(open(gp, encoding='utf-8')), d)
    mn = M.doc_mat_na(mask_file(tmp_path, [100, 101, 103, 104, 105], name='o2.json'), 'r1')
    out, _ = M.ap_dung(nodes, d, mn, LU._state_from_numbers)
    assert out[0]['ok'] and out[0]['states']['g2'] == 'khong_dat'      # the violation is on an open frame


def test_q_sd_recomputed_on_defined_frames_when_the_node_has_it():
    # SD1-1 (GATE 50025f8): q_so_do of a node is the largest q on DEFINED frames; masked frames leave it
    H = dict(fid=[1, 2], t_cam=np.array([1.0, 2.0]), mc=np.array([True, True]), mu=np.array([True, True]),
             q={c: np.array([8.0, 3.0]) for c in LU.MEASURED},
             lo={c: np.array([7.0, 2.0]) for c in LU.MEASURED}, hi={c: np.array([9.0, 4.0]) for c in LU.MEASURED})
    assert M.so_lieu(H)['g5']['q_sd'] == 8.0
    H['lo']['g5'][0], H['hi']['g5'][0] = -math.inf, math.inf
    s = M.so_lieu(H)['g5']
    assert s['q_sd'] == 3.0 and s['q'] == 8.0 and s['n_undef'] == 1


# ── Codex D2 round 1 (24/11 khuya) ───────────────────────────────────────────────────────────────────────────────

def test_rows_file_must_be_the_scorers(tmp_path):
    """D2-6: the rows file is the one the node score names (rows_sha256); two rows with their ids swapped keep every
    aggregate and would mask the wrong frame."""
    fr = frames(viol={102: {'g3': 0.15}})
    d, gp = grid(tmp_path, [fr])
    assert run(d, gp)[0] == 0
    p = os.path.join(d, 'nut_000_rows.csv')
    lines = open(p, encoding='utf-8').read().split('\n')
    lines = [('102,' if x.startswith('103,') else '103,' if x.startswith('102,') else x[:4]) + x[4:] for x in lines]
    open(p, 'w', encoding='utf-8', newline='').write('\n'.join(lines))
    mp = mask_file(tmp_path, range(100, 106))
    assert run(d, gp, mp)[0] == 2
    resha(d, 0)
    assert run(d, gp, mp)[0] == 0                                       # only the sha stood in the way
    sp = os.path.join(d, 'nut_000.json')
    sc = json.load(open(sp, encoding='utf-8'))
    del sc['khung_tuong_doi_v1']['rows_sha256']
    json.dump(sc, open(sp, 'w', encoding='utf-8'))
    os.remove(os.path.join(d, 'luoi_d2.json'))
    assert run(d, gp, mp)[0] == 2                                       # a score that names no rows file


def test_self_check_refuses_numbers_on_the_other_side_of_the_threshold(tmp_path):
    """D2-6: the rows rebuild lo = 0.0999999998, the gate block says 0.1000000002 and khong_dat: equal within the
    tolerance, two states; an open mask would have changed the state."""
    fr = frames(viol={102: {'g2': 0.1099999998}})
    d, gp = grid(tmp_path, [fr])

    def edit(g):
        g['g2']['lo'] = g['g2']['chung']['lo'] = 0.1000000002
        g['g2']['state'] = 'khong_dat'
    write_node(d, 0, fr, D=1.0, edit=edit)
    assert run(d, gp)[0] == 0
    nodes = LU.load_nodes(json.load(open(gp, encoding='utf-8')), d)
    assert nodes[0]['ok'] and nodes[0]['states']['g2'] == 'khong_dat'
    mn = M.doc_mat_na(mask_file(tmp_path, range(100, 106)), 'r1')
    with pytest.raises(M.MatNaError, match='rebuild g2'):
        M.ap_dung(nodes, d, mn, LU._state_from_numbers)


def test_g1_not_concluded_for_any_masked_frame_in_the_union(tmp_path):
    """D2-4: frame 103 had no finite bound already (so the mask makes no bound newly undefined), but its centre fed
    the progress of G1: masked, G1 is not concluded."""
    fr = frames()
    d, gp = grid(tmp_path, [fr, fr], undef=(103,))
    _, r0 = run(d, gp)
    assert r0['states']['g1'] == 'dat'
    rc, r = run(d, gp, mask_file(tmp_path, [100, 101, 102, 104, 105]))
    e = r['mat_na_d2']['nut'][0]
    assert rc == 0 and e['n_che'] == 1 and e['n_che_moi'] == 0 and e['n_che_mien_hop'] == 1 and e['g1_che']
    assert r['states']['g1'] == 'chua_ket_luan'
    # a masked frame outside the union domain of the node leaves G1
    sub = tmp_path / 'ngoai'
    os.makedirs(sub)
    fr2 = [(f[0], f[1], 0 if f[0] == 105 else f[2], 0 if f[0] == 105 else f[3], f[4]) for f in frames()]
    d2, gp2 = grid(sub, [fr2, fr2])
    assert run(d2, gp2)[0] == 0
    rc, r = run(d2, gp2, mask_file(sub, [100, 101, 102, 103, 104]))
    assert rc == 0 and r['states']['g1'] == 'dat' and r['mat_na_d2']['nut'][0]['n_che'] == 1


def test_cat_checked_for_any_masked_frame_before_t1(tmp_path):
    """D2-3: a so_do condition; the only frame above the threshold before t1 has no finite bound (the scorer before
    SD1-1 took its q); masked, no certain violation is left on the open frames: the node is not scored."""
    fr = frames(viol={101: {'g5': 8.0}})
    d, gp = grid(tmp_path, [fr], cat=True, t1=0.2, undef=(101,))
    nodes = LU.load_nodes(json.load(open(gp, encoding='utf-8')), d)
    nodes[0]['so_do'] = ['g5']
    mn = M.doc_mat_na(mask_file(tmp_path, [100, 102, 103, 104, 105]), 'r1')
    out, tom = M.ap_dung(nodes, d, mn, LU._state_from_numbers)
    assert not out[0]['ok'] and out[0]['kind'] == 'loi' and tom['nut'][0]['n_che_moi'] == 0


def test_branches_the_rows_cannot_rebuild_are_refused(tmp_path):
    """D2-7: plate-changed case (G5 on the IMU heading). D2-5: t_som case with a masked frame in the union domain
    (open everywhere: applied)."""
    fr = frames()
    d, gp = grid(tmp_path, [fr], ktd=dict(tam_doi='doi'))
    nodes = LU.load_nodes(json.load(open(gp, encoding='utf-8')), d)
    with pytest.raises(M.MatNaError, match='tam_doi'):
        M.ap_dung(nodes, d, M.doc_mat_na(mask_file(tmp_path, range(100, 106)), 'r1'), LU._state_from_numbers)
    sub = tmp_path / 'tsom'
    os.makedirs(sub)
    d2, gp2 = grid(sub, [fr], g1=dict(ca_t_som=True, t_som_s=1.0), ktd=dict(tam_doi='khong_doi'))
    nodes = LU.load_nodes(json.load(open(gp2, encoding='utf-8')), d2)
    assert nodes[0]['ok']
    with pytest.raises(M.MatNaError, match='t_som'):
        M.ap_dung(nodes, d2, M.doc_mat_na(mask_file(sub, [100, 101, 102, 103, 104]), 'r1'), LU._state_from_numbers)
    nodes = LU.load_nodes(json.load(open(gp2, encoding='utf-8')), d2)
    out, _ = M.ap_dung(nodes, d2, M.doc_mat_na(mask_file(sub, range(100, 106), name='mo.json'), 'r1'),
                       LU._state_from_numbers)
    assert out[0]['ok'] and out[0]['states']['g1'] == 'dat'


def test_masked_grid_keeps_the_chain_fields_of_luoi_json(tmp_path):
    """D2-1: the flags of the grid without the mask must be passed again; luoi.json is named with its sha."""
    fr = frames()
    d, gp = grid(tmp_path, [fr, fr])
    clip = ['--clip', 'C8-P-F05-1.MOV', '--clip-sha256', 'ab' * 32]
    assert run(d, gp, them=clip)[0] == 0
    mp = mask_file(tmp_path, range(100, 106))
    assert run(d, gp, mp)[0] == 2                                       # --clip not passed again: clip would be null
    rc, r = run(d, gp, mp, them=clip)
    assert rc == 0 and r['clip'] == dict(ten='C8-P-F05-1.MOV', clip_sha256='ab' * 32)
    assert r['luoi_goc']['sha256'] == c8_u_imu.sha256(os.path.join(d, 'luoi.json'))


def test_mask_never_makes_a_conclusion(tmp_path):
    """D2-2: a condition or the run dat / khong_dat with the mask and not so without it -> refused."""
    p = str(tmp_path / 'luoi.json')
    st = dict(g1='dat', g2='dat', g3='chua_ket_luan', g4='dat', g5='khong_dat')
    base = dict(states=st, outcome='khong_dat', task='O1-F08', gio_sha256='g' * 64, gate_file_sha256=None,
                clip=None, file_so_h=None, file_cham_h1=None, file_gio_h1=None, scorer=dict(commit='abc'))
    json.dump(base, open(p, 'w', encoding='utf-8'))
    ok = dict(base, states=dict(st, g1='chua_ket_luan', g5='chua_ket_luan'), outcome='chua_ket_luan')
    assert M.goc_luoi(p, ok)['sha256'] == c8_u_imu.sha256(p)
    for bad in (dict(base, states=dict(st, g3='khong_dat')), dict(base, states=dict(st, g3='dat')),
                dict(base, states=dict(st, g5='chua_ket_luan'), outcome='dat'),
                dict(base, scorer=dict(commit='def')), dict(base, gio_sha256='h' * 64)):
        with pytest.raises(M.MatNaError):
            M.goc_luoi(p, bad)
    with pytest.raises(M.MatNaError, match='made a conclusion'):
        M.goc_luoi(p, dict(base, states=dict(st, g3='khong_dat')))
    json.dump(dict(base, mat_na_d2={}), open(p, 'w', encoding='utf-8'))
    with pytest.raises(M.MatNaError):                                  # a masked file is not the origin
        M.goc_luoi(p, ok)


def _nodes_sd(tmp_path, n_nodes=1):
    """Nodes as c8b_luoi.node_from_files of GATE 2ae87ad gives them (SD1-1): q_sd, q_sd_chung on every condition, NaN
    where the gate block has no q_so_do; here G5 is the so_do condition (two frames: 8 deg and 3 deg)."""
    fr = frames(n=2, viol={100: {'g5': 8.0}, 101: {'g5': 3.0}})

    def edit(g):
        g['g5']['q_so_do'] = dict(nut=None, q=8.0, q_chung=8.0)
    d, gp = grid(tmp_path, [fr] * n_nodes, edit=edit)
    gio = json.load(open(gp, encoding='utf-8'))
    nodes = LU.load_nodes(gio, d)
    for x in nodes:
        for c in LU.MEASURED:
            x['nums'][c].update(q_sd=8.0 if c == 'g5' else math.nan, q_sd_chung=8.0 if c == 'g5' else math.nan)
        x['so_do'] = ['g5']
    return d, gio, nodes


def test_q_sd_of_the_merged_grid_is_checked_and_recomputed_for_so_do_only(tmp_path):
    """After the merge with GATE 2ae87ad every condition has q_sd (NaN without a q_so_do block): only the so_do
    condition is compared in the self-check and recomputed under the mask (D2-2)."""
    d, gio, nodes = _nodes_sd(tmp_path)
    mn = M.doc_mat_na(mask_file(tmp_path, [101]), 'r1')                  # frame 100 (8 deg) masked
    out, _ = M.ap_dung(nodes, d, mn, LU._state_from_numbers)
    n5 = out[0]['nums']['g5']
    assert n5['q_sd'] == 3.0 and n5['q_sd_chung'] == 3.0 and n5['q'] == 8.0 and n5['n_undef'] == 1
    assert math.isnan(out[0]['nums']['g2']['q_sd'])                     # not a so_do condition: left as it was
    d2, gio2, nodes2 = _nodes_sd(tmp_path / 'b')
    nodes2[0]['nums']['g5']['q_sd'] = 7.0                               # not the rows' number: refused
    with pytest.raises(M.MatNaError):
        M.ap_dung(nodes2, d2, M.doc_mat_na(mask_file(tmp_path, [101], name='o2.json'), 'r1'), LU._state_from_numbers)


@pytest.mark.skipif('q_sd' not in __import__('inspect').getsource(LU.combine),
                    reason='c8b_luoi.combine before SD1-1 (GATE 2ae87ad): runs after the merge')
def test_so_do_decides_on_the_open_defined_frames_through_combine(tmp_path):
    """D2-2 end to end: the masked frame carried the 8 deg; through ap_dung and combine G5 is not khong_dat."""
    d, gio, nodes = _nodes_sd(tmp_path, n_nodes=3)
    assert LU.combine(nodes, {'g5'}, gio['nodes_s'])['states']['g5'] == 'khong_dat'
    d, gio, nodes = _nodes_sd(tmp_path / 'b', n_nodes=3)
    out, _ = M.ap_dung(nodes, d, M.doc_mat_na(mask_file(tmp_path, [101]), 'r1'), LU._state_from_numbers)
    assert LU.combine(out, {'g5'}, gio['nodes_s'])['states']['g5'] == 'chua_ket_luan'


def test_t_som_node_refused_for_a_masked_frame_outside_its_domain(tmp_path):
    """Codex round 2 D2v2-3: a frame outside the union domain of a t_som node can still be a sample of the turn rate
    that places t_som: any masked frame of such a node is refused."""
    fr = frames()
    fr[5] = (fr[5][0], fr[5][1], 0, 0, fr[5][4])                        # frame 105 outside both domains
    d, gp = grid(tmp_path, [fr], g1=dict(ca_t_som=True, t_som_s=1.0), ktd=dict(tam_doi='khong_doi'))
    nodes = LU.load_nodes(json.load(open(gp, encoding='utf-8')), d)
    assert nodes[0]['ok']
    with pytest.raises(M.MatNaError, match='t_som'):
        M.ap_dung(nodes, d, M.doc_mat_na(mask_file(tmp_path, [100, 101, 102, 103, 104]), 'r1'), LU._state_from_numbers)


def test_luoi_json_must_be_the_grid_of_these_node_files(tmp_path, capsys):
    """Codex round 2 D2v2-1: a luoi.json left from other node files (their SHA, or the states of a fresh combine
    without the mask) is refused; combined again from the node files of the folder, the mask is applied."""
    fr = frames(viol={102: {'g3': 0.15}})
    d, gp = grid(tmp_path, [fr, fr])
    assert run(d, gp)[0] == 0
    mp = mask_file(tmp_path, range(100, 106))
    assert run(d, gp, mp)[0] == 0
    os.remove(os.path.join(d, 'luoi_d2.json'))
    gio = json.load(open(gp, encoding='utf-8'))
    write_node(d, 1, frames(), D=gio['nodes_s'][1])                    # node 1 scored again: no violation there
    capsys.readouterr()
    assert run(d, gp, mp)[0] == 2 and 'not the grid of the node files' in capsys.readouterr().err
    os.remove(os.path.join(d, 'luoi.json'))
    assert run(d, gp)[0] == 0
    rc, r = run(d, gp, mp)
    assert rc == 0 and r['luoi_goc']['sha256'] == c8_u_imu.sha256(os.path.join(d, 'luoi.json'))
    # the check itself: the node SHA in order, the states and the scored nodes of the combine without the mask
    g = json.load(open(os.path.join(d, 'luoi.json'), encoding='utf-8'))
    p = os.path.join(d, 'luoi.json')
    kc = dict(states=g['states'], nodes_ok=g['nodes_ok'])
    assert M.goc_luoi(p, g, kc)['sha256'] == c8_u_imu.sha256(p)
    for bad_res, bad_kc in ((dict(g, nguon_nut=g['nguon_nut'][::-1]), kc),
                            (g, dict(kc, states=dict(g['states'], g3='dat'))),
                            (g, dict(kc, nodes_ok=[True, False]))):
        with pytest.raises(M.MatNaError, match='not the grid'):
            M.goc_luoi(p, bad_res, bad_kc)


@pytest.mark.skipif('q_sd' in __import__('inspect').getsource(LU.combine),
                    reason='c8b_luoi.combine of SD1-1 (after the merge with the gate branch): the mask is applied')
def test_mask_on_a_so_do_condition_refused_before_sd1_1(tmp_path, capsys):
    """Codex round 2 D2v2-2: before SD1-1 the combine decides a so_do condition on the q of the middle node, masked
    frames included: the mask is refused for a run with a so_do condition (O1-F08: G5)."""
    fr = frames()
    d, gp = grid(tmp_path, [fr, fr])
    gio = json.load(open(gp, encoding='utf-8'))
    json.dump(dict(gio, task='O1-F08'), open(gp, 'w', encoding='utf-8'))
    capsys.readouterr()
    rc = LU.main(['--gio', gp, '--out-dir', d, '--gop-only', '--mat-na-d2', mask_file(tmp_path, range(100, 106)),
                  '--luot', 'r1'])
    assert rc == 2 and 'SD1-1' in capsys.readouterr().err
