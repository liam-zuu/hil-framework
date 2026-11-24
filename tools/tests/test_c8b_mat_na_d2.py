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


def write_node(d, k, fr, D=1.0, cat=False, t1=None, edit=None):
    per = {c: ([], [], []) for c in LU.MEASURED}
    for _, _, _, _, q in fr:
        for c in LU.MEASURED:
            v = abs(q[c]) if c == 'g5' else q[c]
            per[c][0].append(v)
            per[c][1].append(v - U[c])
            per[c][2].append(v + U[c])
    mc = np.array([f[2] == 1 for f in fr])
    mu = np.array([f[3] == 1 for f in fr])
    g = {'g1': dict(state='dat', t1=t1)}         # no t_done: a command-time end would move between nodes (C8BS-6)
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
    with open(os.path.join(d, f'nut_{k:03d}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(outcome='dat', gate_v3=g, U_t_s=0.03, t0_log_us=5e6, gio_c8b=dict(D=D), cat_t1=cat), f)
    with open(os.path.join(d, f'nut_{k:03d}_rows.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['frame_id', 't_cam_s', 'mien_chung', 'mien_hop'] + [QC[c] for c in LU.MEASURED]
                   + [f'{b}_{c}' for c in LU.MEASURED for b in ('lo', 'hi')])
        for j, (fid, t, a, b, q) in enumerate(fr):
            w.writerow([fid, t, a, b] + [q[c] for c in LU.MEASURED]
                       + [per[c][x][j] for c in LU.MEASURED for x in (1, 2)])


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


def run(d, gp, mp=None, luot='r1'):
    base = ['--gio', gp, '--out-dir', d, '--gop-only']
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
    assert r['states']['g1'] == 'chua_ket_luan'                     # step 4 [CACH HIEU]
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


def test_frame_without_id_or_not_in_d2_rows_is_masked(tmp_path):
    fr = frames(n=4)
    d, gp = grid(tmp_path, [fr])
    rc, r = run(d, gp, mask_file(tmp_path, [100, 101, 102]))       # 103 not listed (no D2 row): masked
    assert rc == 0 and r['mat_na_d2']['nut'][0]['n_che'] == 1 and r['nums']['g2']['n_undef'] == 1


def test_mask_file_checks(tmp_path):
    fr = frames()
    d, gp = grid(tmp_path, [fr])
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
