"""Tests of fwsim/s04_ga.py (spec S04 v5). Synthetic points and S03 folders with a known answer;
the last test needs fwsim/build/fwsim.dll and runs the three phases end to end with a tiny
budget (--tool-check). Run: python -m pytest fwsim/tests -q"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import re
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import fwsim  # noqa: E402
import s03_doe as s3  # noqa: E402
import s04_ga as s4  # noqa: E402

_DLL = os.path.join(fwsim.HERE, 'build', 'fwsim.dll')
needs_dll = pytest.mark.skipif(not os.path.isfile(_DLL), reason='fwsim.dll not built')

# gain sets inside the range table of tools/gain_table.py (S04 v6.8 section 2: Evaluator.candidate checks them)
G_PID = dict(kp_xy=1.5, ki_xy=0.05, kp_wz=1.0, ki_wz=0.05, pos_kp=6.0, pos_ki=3.0, k_psi=1.0)
G_ADRC = dict(omega_c_xy=8.0, omega_o_xy=30.0, omega_c_wz=8.0, omega_o_wz=30.0, pos_kp=6.0, pos_ki=3.0, k_psi=1.0)
DOM_PID7 = [dict(name=n, L=lo, H=hi, scale='lin') for n, (lo, hi) in zip(s3.FACTORS['PID'], [
    (1.0, 2.0), (0.01, 0.1), (0.5, 2.0), (0.01, 0.1), (2.0, 8.0), (1.0, 4.0), (0.5, 2.0)])]
DOM_ADRC7 = [dict(name=n, L=lo, H=hi, scale='lin') for n, (lo, hi) in zip(s3.FACTORS['ADRC'], [
    (100.0, 190.0), (20.0, 48.0), (5.0, 10.0), (20.0, 30.0), (2.0, 8.0), (1.0, 4.0), (0.5, 2.0)])]


def P(f1, f2, x=(1.0,), seed=1, feasible=True):
    return dict(f1=f1, f2=f2, x=list(x), xga=list(x), ga_seed=seed, feasible=feasible, gen=1, cand_id=0, g1=0, g2=0,
                g3=0, g4=0, g5=0)


# ── selection, Pareto, HV (sections 4.2, 5) ─────────────────────────────────────────

def test_nondominated_keeps_equal_points():
    pts = [P(0.02, 5.0), P(0.02, 5.0), P(0.03, 4.0), P(0.03, 6.0)]
    nd = s4.nondominated(pts)
    assert [(p['f1'], p['f2']) for p in nd] == [(0.02, 5.0), (0.02, 5.0), (0.03, 4.0)]


def test_select_tolerance_then_f2_then_f1_then_gains_then_seed():
    # f1* = 0.020; 0.0299 is within 0.01 and has the smallest f2 -> chosen over f1*
    assert s4.select([P(0.020, 9.0), P(0.0299, 4.0), P(0.031, 1.0)])['f1'] == 0.0299
    # 0.0301 is outside the tolerance even with a smaller f2
    assert s4.select([P(0.020, 9.0), P(0.0301, 4.0)])['f1'] == 0.020
    # equal f2: smaller f1; equal f1 and f2: lexicographic gains, then the smaller seed
    assert s4.select([P(0.025, 4.0), P(0.022, 4.0)])['f1'] == 0.022
    assert s4.select([P(0.02, 4.0, x=(2.0, 1.0)), P(0.02, 4.0, x=(1.0, 9.0))])['x'] == [1.0, 9.0]
    assert s4.select([P(0.02, 4.0, seed=3), P(0.02, 4.0, seed=2)])['ga_seed'] == 2
    assert s4.select([]) is None


def test_select_filters_dominated_first():
    # (0.025, 5.0) is dominated by (0.021, 4.0); the rule never returns a dominated point
    ch = s4.select([P(0.020, 9.0), P(0.025, 5.0), P(0.021, 4.0)])
    assert (ch['f1'], ch['f2']) == (0.021, 4.0)


def test_hypervolume_staircase_and_reference():
    ref = (0.10, 10.0)
    hv, out = s4.hypervolume([P(0.02, 6.0), P(0.06, 2.0)], ref)
    # (0.10 - 0.02)(10 - 6) + (0.10 - 0.06)(6 - 2)
    assert hv == pytest.approx(0.08 * 4 + 0.04 * 4) and out == 0
    hv, out = s4.hypervolume([P(0.10, 1.0), P(0.05, 12.0)], ref)      # on f1 = D, beyond f2 ref
    assert hv == 0.0 and out == 2
    assert s4.hypervolume([], ref) == (0.0, 0)
    # a dominated point adds nothing
    assert s4.hypervolume([P(0.02, 6.0), P(0.03, 7.0)], ref)[0] == pytest.approx(0.08 * 4)


# ── scoring (section 3) ──────────────────────────────────────────────────────────────

def R(completed='true', status='ok', e=0.02, tv=5.0, trk=0.03, bao=0.04, th=2.0):
    return dict(completed=completed, status=status, e_perp_max=e, tv_duty=tv, e_trk_max=trk, e_bao_max=bao,
                e_theta_cmd_max_deg=th)


def test_score_objectives_and_constraints():
    rows = [(0.4, 0.2, R(e=0.03, tv=4.0)), (0.4, 0.5, R(e=0.05, tv=6.0)),
            (0.6, 0.2, R(e=0.02, tv=10.0)), (0.6, 0.5, R(e=0.04, tv=2.0))]
    s = s4.score(rows, False)
    assert s['f1'] == 0.05                                # max over every run
    assert s['f2'] == pytest.approx(6.0)                  # max over mu of the mean over v: max(5, 6)
    assert s['g1'] == 0 and s['g2'] == pytest.approx((0.05 - s4.D) / s4.D)


def test_score_track_error_counts_in_g1_even_when_completed():
    # S04 v5 (B8b-R-1): a track_err run with y1 'true' violates g1; inconclusive and false too
    rows = [(0.5, 0.2, R(status='track_err')), (0.5, 0.5, R('inconclusive', 'track_err')), (0.5, 0.8, R('false'))]
    s = s4.score(rows, False)
    assert s['g1'] == 3 and s['n_track_err'] == 2 and s['n_inconclusive'] == 1 and s['n_false'] == 1


def test_score_run_error_penalty():
    rows = [(0.5, 0.2, R()), (0.5, 0.5, dict(completed='', status='run_error', error='x'))]
    s = s4.score(rows, True)
    assert (s['f1'], s['f2'], s['g1'], s['status']) == (s4.PENALTY, s4.PENALTY, 2.0, 'run_error')


# ── domain, mu rule (sections 2, 3) ──────────────────────────────────────────────────

def _head():
    import subprocess
    return subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()


# S03 v8 design: the 257 points of the mu rule (S04 v6 section 9 item 4) and, in design.csv, the
# axial and test points the rule must ignore
_POINTS = [(0, 'centre')] + [(i, 'outer') for i in range(1, 129)] + [(i, 'inner') for i in range(129, 257)]
_V8_EXTRA = [(i, 'axial') for i in range(257, 271)] + [(i, 'test') for i in range(271, 371)]


def _s03_dir(tmp_path, runs=None, trial=False, mu=(0.4, 0.5, 0.6), levels=None, plant_src=None, o1c=None,
             script=None, imu=None, gate=None, nguon_mien='sang_loc', ly_do='', sealed_levels=True,
             sealed_design=True, sangloc_files=True, screen_u=True, number=b'so goc\n'):
    """S03 v8.5 layout: public meta.yaml + levels.csv (old column set: read_domain / mu_rule never open
    them), sealed/levels_sangloc.csv (levels + L_nguon, H_nguon, b, b_n, nguon_mien, ly_do),
    sealed/design.csv, sealed/sangloc/ket_qua.json, sealed/sangloc_thu/so_dem.json, and a root number
    file recorded in meta.yaml under 'screen_u' (S04 v6.7 section 8). levels: {controller:
    {factor: (L, H)}}; default a synthetic domain. plant_src, o1c, imu, script: the S03 source record
    (S4C-2); default the source of this checkout (HEAD, clean). gate: the gate speeds (default
    s03_doe.SPEEDS); the real GA trials use 0.2 only, where the default gains meet g3-g5 on fwsim (at
    0.8 the PID default heading error is above 7.5 deg; R134 package)."""
    d = tmp_path / 's03'
    (d / 'sealed' / 'sangloc').mkdir(parents=True)
    (d / 'sealed' / 'sangloc_thu').mkdir(parents=True)
    with open(d / 'levels.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['controller', 'factor', 'L', 'H', 'centre', 'scale', 'default', 'rule_note'])
        for c in s3.CONTROLLERS:
            for k, n in enumerate(s3.FACTORS[c]):
                L, H = levels[c][n] if levels else ((0.5, 8.0) if k % 2 == 0 else (1.0, 3.0))
                w.writerow([c, n, L, H, '', 'log' if L > 0 and H / L >= 4 else 'lin', '', ''])
    if sealed_levels:
        with open(d / 'sealed' / 'levels_sangloc.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['controller', 'factor', 'L', 'H', 'centre', 'scale', 'default', 'rule_note',
                        'L_nguon', 'H_nguon', 'b', 'b_n', 'nguon_mien', 'ly_do'])
            for c in s3.CONTROLLERS:
                for k, n in enumerate(s3.FACTORS[c]):
                    L, H = levels[c][n] if levels else ((0.5, 8.0) if k % 2 == 0 else (1.0, 3.0))
                    w.writerow([c, n, L, H, '', 'log' if L > 0 and H / L >= 4 else 'lin', '', '',
                                '', '', '', '', nguon_mien, ly_do])
    if sealed_design:
        with open(d / 'sealed' / 'design.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['controller', 'point_id', 'box'])
            for c in s3.CONTROLLERS:
                for pid, box in _POINTS + _V8_EXTRA:
                    w.writerow([c, pid, box])
    if sangloc_files:
        (d / 'sealed' / 'sangloc' / 'ket_qua.json').write_text('{}')
        (d / 'sealed' / 'sangloc_thu' / 'so_dem.json').write_text('{}')
    num = tmp_path / 'so_goc.txt'
    num.write_bytes(number)
    y3 = {f'{c} v={v:g}': 5.0 for c in s3.CONTROLLERS for v in s3.SPEEDS}
    dll_sha = s3.sha256_file(_DLL) if os.path.isfile(_DLL) else 'no dll'
    su = ('sang_loc: ' + json.dumps(dict(file_so=dict(file=str(num), sha256=hashlib.sha256(number).hexdigest())))
          + '\n'
          if screen_u else '')
    (d / 'meta.yaml').write_text(
        f'trial: {str(trial).lower()}     # x\nstatus: "ran to the end"\nmu_levels: {json.dumps(list(mu))}\n'
        f'{su}'
        f'script: {json.dumps(script if script is not None else dict(commit=_head(), dirty=False))}\n'
        f'firmware:\n  dll_sha256: {dll_sha}\n'
        f'plant: {json.dumps(dict(motor_model="dc", src=plant_src or fwsim.PLANT_SRC))}\n'
        f'o1: {json.dumps(o1c or s4.o1_constants())}\nimu: {json.dumps(imu or s3.IMU)}\n'
        f'gate_speeds: {json.dumps(list(gate or s3.SPEEDS))}\ny3_reference: {json.dumps(y3)}\n', encoding='utf-8')
    if runs is not None:
        s3.write_csv(str(d / 'sealed' / 'runs.csv'), runs)
    return d


@pytest.fixture(autouse=True)
def _self_blobs(monkeypatch):
    """fwsim/s04_ga.py is uncommitted (it is the file under test) and fwsim/s03_sangloc.py is written in
    parallel: their working-tree and S03-commit git blobs are one fixed fake value here (other paths
    real); tests of the new comparison override this."""
    real_w, real_b = s4.work_blob, s4.blob_at
    fake = 'f' * 40
    monkeypatch.setattr(s4, 'work_blob',
                        lambda p: fake if p in ('fwsim/s04_ga.py', s4.S03_SCREEN_SCRIPT) else real_w(p))
    monkeypatch.setattr(s4, 'blob_at',
                        lambda c, p: fake if p in ('fwsim/s04_ga.py', s4.S03_SCREEN_SCRIPT) else real_b(c, p))


def _runs(mod=None):
    out = []
    for pid, box in _POINTS:
        for mu in (0.4, 0.5, 0.6):
            for v in s3.SPEEDS:
                r = dict(controller='PID', point_id=pid, box=box, mu=mu, v=v, status='ok', completed='true',
                         e_perp_max=0.03, tv_duty=5.0,
                         g3_state='dat', g4_state='dat', g5_state='dat')   # v6.7 9.7: gate state columns
                if mod:
                    mod(r)
                out.append(r)
    return out


def test_domain_and_log_coding(tmp_path):
    d = _s03_dir(tmp_path)
    dom = s4.read_domain(str(d), 'PID')
    lo, hi = s4.ga_bounds(dom)
    assert dom[0]['scale'] == 'log' and lo[0] == pytest.approx(math.log(0.5)) and hi[1] == 3.0
    g = s4.decode(dom, hi)
    assert g[dom[0]['name']] == 8.0 and g[dom[1]['name']] == 3.0          # exp(log H) clipped to H
    mid = (lo + hi) / 2
    assert s4.decode(dom, mid)[dom[0]['name']] == pytest.approx(2.0)      # geometric centre of [0.5, 8]
    assert s4.edge_flags(dom, lo + 0.04 * (hi - lo)) == [x['name'] for x in dom]
    assert s4.edge_flags(dom, mid) == []
    assert all(x['L_src'] == '' and x['H_src'] == '' for x in dom)       # an S03 levels.csv before v8.3


# ── S04 v6.4 (C14-2): edge side and level source (design 03_CODE/docs/2026-11-15_Thiet_ke_S03_v83_luoi_noi_v1.md 6)

def _edit_levels(d, mod=None, **upd):
    """Edit every row of sealed/levels_sangloc.csv (mod(row) per row first, then row.update(upd))."""
    import csv as _csv
    p = d / 'sealed' / 'levels_sangloc.csv'
    rows = list(_csv.DictReader(open(p, encoding='utf-8')))
    for r in rows:
        if mod:
            mod(r)
        r.update(upd)
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_v64_read_domain_reads_the_level_sources(tmp_path):
    d = _s03_dir(tmp_path)
    _edit_levels(d, L_nguon='0', H_nguon='hết lưới', b='8.0')
    dom = s4.read_domain(str(d), 'PID')
    assert all(x['L_src'] == '0' and x['H_src'] == 'hết lưới' for x in dom)
    assert all(x['L_kiem'] == '' and x['H_kiem'] == '' for x in dom)           # no S03 v8.4 check file


@pytest.mark.parametrize('upd', [dict(H_nguon='hết lưới nối'), dict(b='2.0')])
def test_v66_read_domain_refuses_levels_on_the_extension(tmp_path, upd):
    """S04 v6.6 + v6.7 9.6: the v6.6 original-grid refusal applies to nguon_mien 'du_phong' levels only;
    S03 v8.3 levels (H on the extension) are refused."""
    d = _s03_dir(tmp_path, nguon_mien='du_phong')
    _edit_levels(d, L_nguon='0', **upd)
    with pytest.raises(SystemExit, match='extension'):
        s4.read_domain(str(d), 'PID')


@pytest.mark.parametrize('upd', [dict(H_nguon='hết lưới nối'), dict(b='2.0')])
def test_v67_sang_loc_levels_on_the_extension_are_accepted(tmp_path, upd):
    """S04 v6.7 section 9 item 6: nguon_mien 'sang_loc' drops the v6.6 refusal (H may exceed b, up to b_n)."""
    d = _s03_dir(tmp_path)
    _edit_levels(d, L_nguon='0', **upd)
    dom = s4.read_domain(str(d), 'PID')
    assert dom and all(x['L_src'] == '0' for x in dom)


def test_v67_du_phong_h_equal_b_is_accepted(tmp_path):
    """du_phong levels whose H is still on the original grid (H == b) are accepted."""
    d = _s03_dir(tmp_path, nguon_mien='du_phong')
    _edit_levels(d, H_nguon='hết lưới', mod=lambda r: r.update(b=r['H']))
    assert s4.read_domain(str(d), 'PID')


def test_v66_read_domain_reads_the_out_of_range_check(tmp_path):
    import csv as _csv
    d = _s03_dir(tmp_path)
    cols = ['dong', 'controller', 'factor', 'value', 'can', 'ket_qua']
    rows = [dict(dong='diem', controller='PID', factor='kp_xy', value=str(9 + i), can='H', ket_qua=k)
            for i, k in enumerate(('không cờ', 'không cờ', 'chưa kết luận'))]
    rows += [dict(dong='diem', controller='ADRC', factor='kp_xy', value='9', can='H', ket_qua='cờ'),
             dict(dong='tong', controller='PID', ket_qua='đã kiểm đủ'), dict(dong='tong', controller='ADRC',
                                                                            ket_qua='kiểm dở')]
    with open(d / 'kiem_ngoai_mien.csv', 'w', newline='', encoding='utf-8') as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    by = {x['name']: x for x in s4.read_domain(str(d), 'PID')}
    assert by['kp_xy']['H_kiem'] == '3 điểm: không cờ 2, chưa kết luận 1, cờ 0'
    assert by['kp_xy']['L_kiem'] == '0 điểm' and by['ki_xy']['H_kiem'] == '0 điểm'
    assert s4.read_out_of_range(str(d), 'ADRC')[('pos_kp', 'H')] == '0 điểm (lượt kiểm: kiểm dở)'


def test_v66_fingerprint_locks_the_check_file(tmp_path):
    """S84-2: read_domain reads kiem_ngoai_mien.csv, so the fingerprint of trial / main / post holds its SHA-256
    (None for an S03 run without it); any byte changed after the trial changes the fingerprint."""
    d = _s03_dir(tmp_path)
    dll = str(d / 'levels.csv')                                   # any file: only its hash is taken
    f0 = s4.fingerprint(dll, str(d))
    assert f0['s03_files']['kiem_ngoai_mien.csv'] is None
    k = d / 'kiem_ngoai_mien.csv'
    k.write_bytes('dong,controller,factor,value,can,ket_qua\n'.encode('utf-8'))
    f1 = s4.fingerprint(dll, str(d))
    assert f1['s03_files']['kiem_ngoai_mien.csv'] == s3.sha256_file(str(k)) and f1 != f0
    k.write_bytes('dong,controller,factor,value,can,ket_qua\n\n'.encode('utf-8'))   # a blank line only
    assert s4.fingerprint(dll, str(d)) != f1


def test_v66_check_file_alias_refused_before_it_is_opened(tmp_path, monkeypatch):
    """S84-3: a hard link (or a symbolic link, when the system allows one) at kiem_ngoai_mien.csv to a sealed file is
    refused by read_domain and fingerprint before the file is opened or hashed."""
    import builtins
    d = _s03_dir(tmp_path)
    (d / 'sealed_test').mkdir()
    target = d / 'sealed_test' / 'runs_test.csv'
    target.write_text('controller,point_id\n', encoding='utf-8')
    k = d / 'kiem_ngoai_mien.csv'
    os.link(str(target), str(k))
    opened = []
    real_open = builtins.open

    def spy(f, *a, **kw):
        opened.append(os.path.basename(str(f)))
        return real_open(f, *a, **kw)
    monkeypatch.setattr(builtins, 'open', spy)
    with pytest.raises(SystemExit, match='one name'):
        s4.read_domain(str(d), 'PID')
    with pytest.raises(SystemExit, match='one name'):
        s4.fingerprint(str(d / 'levels.csv'), str(d))
    monkeypatch.setattr(builtins, 'open', real_open)
    assert 'kiem_ngoai_mien.csv' not in opened and 'runs_test.csv' not in opened
    os.remove(str(k))
    try:
        os.symlink(str(target), str(k))
    except (OSError, NotImplementedError):
        return                                                    # no symlink right on this system
    with pytest.raises(SystemExit, match='link'):
        s4.read_domain(str(d), 'PID')


DOM64 = [dict(name='a', L=1.0, H=2.0, scale='lin', L_src='hết lưới dưới', H_src='hết lưới', L_kiem='0 điểm',
              H_kiem='2 điểm: không cờ 2, chưa kết luận 0, cờ 0'),
         dict(name='b', L=1.0, H=3.0, scale='lin', L_src='0', H_src='điểm kế tiếp không hoàn thành'),
         dict(name='c', L=1.0, H=2.0, scale='lin', L_src='', H_src='')]


def _limit(dom, limit):
    """S04 v6.6 edge_limit text: 'vùng ngoài biên chưa khảo sát bằng GA: ' + factor:side [kiểm ngoài miền: ...]."""
    if not limit:
        return ''
    by = {d['name']: d for d in dom}
    return 'vùng ngoài biên chưa khảo sát bằng GA: ' + ' '.join(
        f'{n}:{s} [kiểm ngoài miền: {by[n].get(s + "_kiem") or "?"}]' for n, s in (p.split(':') for p in limit.split()))


@pytest.mark.parametrize('x, side, limit', [
    ((1.02, 2.0, 1.5), 'a:L', 'a:L'),                 # near L, a grid end below: limit
    ((1.98, 2.0, 1.5), 'a:H', 'a:H'),                 # near H, the extension end: limit
    ((1.5, 2.95, 1.5), 'b:H', ''),                    # near H, the next point did not complete: no limit
    ((1.5, 1.02, 1.5), 'b:L', ''),                    # near L = 0 source: no limit
    ((1.5, 2.0, 1.99), 'c:H', ''),                    # no source (S03 before v8.3): '?', no limit
    ((1.5, 2.0, 1.5), '', ''),                        # no flag: no limit, whatever the sources
])
def test_v64_edge_report_side_source_and_limit(x, side, limit):
    rep = s4.edge_report(DOM64, np.array(x))
    assert rep['edge'] == ' '.join(s4.edge_flags(DOM64, np.array(x)))       # the old flag, unchanged
    assert rep['edge_side'] == side and rep['edge'] == ' '.join(p.split(':')[0] for p in side.split())
    assert rep['edge_limit'] == _limit(DOM64, limit)
    if side:
        n, s = side.split(':')
        src = {('a', 'L'): 'hết lưới dưới', ('a', 'H'): 'hết lưới', ('b', 'H'): 'điểm kế tiếp không hoàn thành',
               ('b', 'L'): '0', ('c', 'H'): '?'}[(n, s)]
        assert rep['edge_src'] == f'{side}:{src}'


def test_v64_summarise_rows_carry_the_edge_columns():
    finals = {1: [P(0.020, 9.0, x=(1.98, 2.0, 1.5), seed=1)], 2: [P(0.022, 4.0, x=(1.98, 2.0, 1.5), seed=2)]}
    rows, _ = s4.summarise('PID', DOM64, finals)
    for r in rows:
        assert r['edge_side'] == 'a:H' and r['edge_limit'] == _limit(DOM64, 'a:H'), r['kind']
        assert r['edge_limit'].endswith('a:H [kiểm ngoài miền: 2 điểm: không cờ 2, chưa kết luận 0, cờ 0]')


DOMLOG = [dict(name='a', L=0.5, H=8.0, scale='log', L_src='hết lưới dưới', H_src='điểm kế tiếp không hoàn thành'),
          dict(name='b', L=0.5, H=8.0, scale='log', L_src='0', H_src='hết lưới')]


@pytest.mark.parametrize('gains, side, limit', [
    ((2.0, 2.0), '', ''),                           # geometric centre: no flag (the gain 2 is near H on a linear axis)
    ((0.51, 2.0), 'a:L', 'a:L'),                    # near L, a grid end below: limit
    ((7.9, 2.0), 'a:H', ''),                        # near H, the next point did not complete: no limit
    ((2.0, 7.9), 'b:H', 'b:H'),                     # near H, the extension end: limit
    ((2.0, 0.51), 'b:L', ''),                       # near L = 0 source: no limit
])
def test_v64_summarise_uses_the_ga_coordinate_on_a_log_axis(gains, side, limit):
    """S83-6: on a log factor the flag and its side come from the GA coordinate xga = log(gain), not from the
    gain x; the two sides of each factor have different sources."""
    p = [P(0.020, 9.0, x=gains, seed=1), P(0.022, 4.0, x=gains, seed=2)]
    for q in p:
        q['xga'] = [math.log(g) for g in gains]
    rows, _ = s4.summarise('PID', DOMLOG, {1: [p[0]], 2: [p[1]]})
    for r in rows:
        assert r['edge_side'] == side, r['kind']
        assert r['edge_limit'] == _limit(DOMLOG, limit), r['kind']


_TABLE_LIMIT = 'vùng ngoài bảng chặn không thuộc thí nghiệm (DUT từ chối gain đó)'   # S04 v6.8 section 2, the spec text


@pytest.mark.parametrize('x, limit, src', [
    # Codex C3-7: flag + source 'trần bảng chặn' -> the table-ceiling limit, now with the message
    ((4.55, 10.0, 1.5), 'TABLE:kp_wz:H', 'kp_wz:H:trần bảng chặn'),
    ((2.0, 10.0, 1.5), '', ''),                                  # no flag, no limit whatever the source
    ((0.65, 10.0, 1.5), 'vùng ngoài biên chưa khảo sát bằng GA: kp_wz:L [kiểm ngoài miền: 0 điểm]',
     'kp_wz:L:hết lưới dưới'),                                   # the L side has another source: text as before
    ((2.0, 19.0, 1.5), 'TABLE:b:H', 'b:H:trần bảng chặn'),       # exactly on the 5 % edge: 20 - 19 = 0.05 * 20
    ((2.0, 19.0 - 2.0 ** -40, 1.5), '', ''),                     # just outside the 5 %
    ((4.55, 10.0, 1.99), 'vùng ngoài biên chưa khảo sát bằng GA: c:H [kiểm ngoài miền: 1 điểm: không cờ 1, '
     'chưa kết luận 0, cờ 0] | TABLE:kp_wz:H', 'kp_wz:H:trần bảng chặn | c:H:hết lưới'),   # both kinds
    ((2.0, 10.0, 1.02), 'TABLE:c:L', 'c:L:trần bảng chặn'),      # the source decides, on the flagged side
])
def test_v68_edge_report_table_ceiling_limit(x, limit, src):
    """S04 v6.8 section 2 (C3-7): a flagged bound whose source is 'trần bảng chặn' adds the limit
    EDGE_LIMIT_TABLE to edge_limit; a grid end keeps the old text; the two kinds are joined by ' | ';
    no flag, no limit."""
    dom = [dict(name='kp_wz', L=0.6, H=4.6, scale='lin', L_src='hết lưới dưới', H_src='trần bảng chặn',
                L_kiem='0 điểm'),
           dict(name='b', L=0.0, H=20.0, scale='lin', L_src='0', H_src='trần bảng chặn'),
           dict(name='c', L=1.0, H=2.0, scale='lin', L_src='trần bảng chặn', H_src='hết lưới',
                H_kiem='1 điểm: không cờ 1, chưa kết luận 0, cờ 0')]
    rep = s4.edge_report(dom, np.array(x))
    assert rep['edge'] == ' '.join(s4.edge_flags(dom, np.array(x)))              # the old flag, unchanged
    assert rep['edge_side'] == ' '.join(f'{n}:{s}' for n, s in s4.edge_sides(dom, np.array(x)))
    assert rep['edge_src'] == src
    assert rep['edge_limit'] == limit.replace('TABLE:', _TABLE_LIMIT + ': ')


def test_v68_selected_rows_carry_the_table_ceiling_limit(tmp_path):
    """S04 v6.8 section 2 (C3-7): through read_domain + summarise the table-ceiling limit reaches
    selected.csv (edge_limit) for a selected point at the flagged bound."""
    d = _s03_dir(tmp_path)
    _edit_levels(d, L_nguon='0', H_nguon='trần bảng chặn', b='8.0')
    dom = s4.read_domain(str(d), 'PID')
    lo, hi = s4.ga_bounds(dom)
    xga = (lo + hi) / 2
    xga[0] = hi[0]                                              # the FIRST factor at its H bound
    x = [s4.decode(dom, xga)[dd['name']] for dd in dom]
    finals = {1: [P(0.020, 9.0, x=x, seed=1)], 2: [P(0.022, 4.0, x=x, seed=2)]}
    for p in finals[1] + finals[2]:
        p['xga'] = list(xga)
    rows, _ = s4.summarise('PID', dom, finals)
    for r in rows:
        assert r['edge_side'] == f"{dom[0]['name']}:H" and \
            r['edge_limit'] == _TABLE_LIMIT + f": {dom[0]['name']}:H", r['kind']


def test_v68_table_ceiling_texts_are_those_of_the_spec():
    """S04 v6.8 section 2 (C3-7): the source label of S03 v8.6 and the limit text of the spec, literally."""
    assert s4.TABLE_CEILING_SOURCE == 'trần bảng chặn'
    assert s4.EDGE_LIMIT_TABLE == _TABLE_LIMIT


def test_mu_rule_mid_when_every_point_satisfies(tmp_path):
    d = _s03_dir(tmp_path, _runs())
    dec, rep = s4.mu_rule(str(d), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])
    assert dec == 'mid' and rep['points_checked'] == 257 * 3 and rep['points_failing'] == 0
    assert rep['n_point_ids'] == 257 and rep['point_ids'] == list(range(257))     # S04 v6 9.4: to lock.yaml


@pytest.mark.parametrize('mod, why', [
    (lambda r: r.update(status='track_err') if r['point_id'] == 129 and r['mu'] == 0.6 and r['v'] == 0.8 else None,
     'track_err'),
    (lambda r: r.update(completed='inconclusive') if r['point_id'] == 1 and r['mu'] == 0.4 else None, 'inconclusive'),
    (lambda r: r.update(completed='false') if r['point_id'] == 0 and r['mu'] == 0.6 else None, 'differs'),
    (lambda r: r.update(e_perp_max=0.0411) if r['point_id'] == 1 and r['mu'] == 0.4 else None, 'dY2'),
    (lambda r: r.update(tv_duty=6.01) if r['point_id'] == 0 and r['mu'] == 0.6 else None, 'dY3'),
    # v6.7 9.7: a G3, G4 or G5 gate state that changes over mu fails the rule
    (lambda r: r.update(g3_state='khong_dat') if r['point_id'] == 1 and r['mu'] == 0.4 else None, 'g3 state differs'),
    (lambda r: r.update(g4_state='chua_ket_luan') if r['point_id'] == 0 and r['mu'] == 0.6 else None,
     'g4 state differs'),
    (lambda r: r.update(g5_state='khong_dat') if r['point_id'] == 129 and r['mu'] == 0.5 else None,
     'g5 state differs'),
])
def test_mu_rule_all_when_one_point_fails(tmp_path, mod, why):
    """S04 3 (v4, B8b-4, B8b-5): one failing point of any box -> evaluate at the three mu."""
    d = _s03_dir(tmp_path, _runs(mod))
    dec, rep = s4.mu_rule(str(d), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])
    assert dec == 'all' and rep['points_failing'] >= 1 and any(why in w for w in rep['first'][0]['why'])


@pytest.mark.parametrize('e, tv, dec', [
    # S04 v6.8: |dY3| <= 0.4 1/s absolute (no S03 y3_reference any more); same inclusive-bound convention
    (0.04, 5.4, 'mid'),            # exactly on both bounds: 0.04 - 0.03 = 0.010000000000000002, 5.4 - 5.0 =
                                   # 0.40000000000000036 in binary64 (S4C-14)
    (0.02, 4.6, 'mid'),            # the bounds below the middle mu
    (0.0401, 5.0, 'all'),          # just outside dY2
    (0.03, 5.4001, 'all'),         # just outside dY3
    (0.03, 5.4 + 1e-9, 'all'),     # 1e-9 beyond the bound is outside: the tolerance covers rounding only
])
def test_mu_rule_tolerances_are_inclusive_at_the_bound(tmp_path, e, tv, dec):
    def mod(r):
        if r['point_id'] == 1 and r['mu'] == 0.4:
            r.update(e_perp_max=e, tv_duty=tv)
    assert s4.mu_rule(str(_s03_dir(tmp_path, _runs(mod))), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])[0] == dec


@pytest.mark.parametrize('key, mid, low, dec', [
    ('tv_duty', 4.0, 4.4000000000002, 'all'),      # Codex C3-6: above 0.4, inside the old 0.4 * (1 + 1e-12)
    ('tv_duty', 4.0, 4.4, 'mid'),                  # exactly on the bound (binary64 difference 0.40000000000000036)
    ('tv_duty', 4.0, 4.3999999999998, 'mid'),      # just below
    ('tv_duty', 4.0, 4.400000001, 'all'),          # 1e-9 above
    ('tv_duty', 4.0, 3.6, 'mid'),                  # the bound below the middle mu
    ('tv_duty', 4.0, 3.5999999999998, 'all'),
    ('e_perp_max', 0.03, 0.0400000000005, 'all'),  # inside the old additive band 0.01 + 1e-12
    ('e_perp_max', 0.03, 0.04, 'mid'),
    ('e_perp_max', 0.03, 0.0399999999995, 'mid'),
    ('e_perp_max', 0.03, 0.040000001, 'all'),
    ('e_perp_max', 0.03, 0.02, 'mid'),
    ('e_perp_max', 0.03, 0.0199999999995, 'all'),
    ('tv_duty', 4.0, float('nan'), 'all'),         # not finite: not comparable
    ('e_perp_max', 0.03, float('inf'), 'all'),
])
def test_mu_rule_bounds_use_the_s03_exact_convention(tmp_path, key, mid, low, dec):
    """C3-6: the mu-rule Y2 / Y3 bounds are compared as S03 compares (s03_doe.not_above_by: inclusive,
    exact on the decimal of each number as written in runs.csv, no tolerance of its own); a Y2 or Y3
    that is not finite does not satisfy the rule. The mod changes point 1 at ONE speed (v == 0.5):
    mu 0.5 and 0.6 at mid, mu 0.4 at low."""
    def mod(r):
        if r['point_id'] == 1 and r['v'] == 0.5:
            r[key] = low if r['mu'] == 0.4 else mid
    d = _s03_dir(tmp_path, _runs(mod))
    decision, rep = s4.mu_rule(str(d), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])
    assert decision == dec
    if dec == 'all':
        assert rep['points_failing'] == 1
        first = rep['first'][0]
        assert first['point_id'] == '1' and first['v'] == 0.5
        text = 'dY3 > 0.4 1/s' if key == 'tv_duty' else 'dY2 > 0.01 m'
        assert (text if math.isfinite(low) else f'{key} not finite') in first['why']
    # with the gate [0.2, 0.8] the changed speed 0.5 is not in the gate: 'mid' whatever the bound
    assert s4.mu_rule(str(d), 'PID', [0.2, 0.8], [0.4, 0.5, 0.6])[0] == 'mid'


def test_mu_rule_refuses_missing_or_duplicate_runs(tmp_path):
    """S4C-5: a design point missing at a gate speed, a duplicate run or a point not in design.csv
    stops the rule; it never concludes 'mid' on a partial S03 table."""
    runs = [r for r in _runs() if not (r['point_id'] == 129 and r['v'] == 0.8)]
    with pytest.raises(SystemExit, match='no run of PID point 129'):
        s4.mu_rule(str(_s03_dir(tmp_path / 'a', runs)), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])
    runs = _runs() + [dict(_runs()[0], status='ok')]
    with pytest.raises(SystemExit, match='duplicate'):
        s4.mu_rule(str(_s03_dir(tmp_path / 'b', runs)), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])
    runs = _runs() + [dict(r, point_id=999) for r in _runs() if r['point_id'] == 1]
    with pytest.raises(SystemExit, match='not in sealed/design.csv'):
        s4.mu_rule(str(_s03_dir(tmp_path / 'c', runs)), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])


def test_mu_rule_ignores_v8_axial_and_test_points(tmp_path):
    """S04 v6 section 9 item 4: axial rows (in sealed/runs.csv) and test rows are filtered out by
    box in both files; a failing axial run does not change the rule, and the 257 point_id are
    locked (a missing centre / outer / inner point stops)."""
    axial = [dict(controller='PID', point_id=p, box='axial', mu=mu, v=v, status='track_err', completed='false',
                  e_perp_max=0.5, tv_duty=50.0) for p in range(257, 271) for mu in (0.4, 0.5, 0.6) for v in s3.SPEEDS]
    dec, rep = s4.mu_rule(str(_s03_dir(tmp_path / 'a', _runs() + axial)), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])
    assert dec == 'mid' and rep['points_checked'] == 257 * 3
    d = _s03_dir(tmp_path / 'b', _runs())
    dz = d / 'sealed' / 'design.csv'
    rows = dz.read_text().splitlines()
    dz.write_text('\n'.join(r for r in rows if r != 'PID,200,inner') + '\n')
    with pytest.raises(SystemExit, match='point_id 0..256'):
        s4.mu_rule(str(d), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])


def test_mu_rule_only_gate_speeds(tmp_path):
    d = _s03_dir(tmp_path, _runs(lambda r: r.update(status='track_err') if r['v'] == 0.8 else None))
    assert s4.mu_rule(str(d), 'PID', [0.2, 0.5], [0.4, 0.5, 0.6])[0] == 'mid'
    assert s4.mu_rule(str(d), 'PID', [0.2, 0.5, 0.8], [0.4, 0.5, 0.6])[0] == 'all'


# ── seeds, summary (sections 3, 5) ───────────────────────────────────────────────────

def test_summary_union_and_infeasible_seeds():
    dom = [dict(name='a', L=1.0, H=2.0, scale='lin')]
    finals = {1: [P(0.020, 9.0, x=(1.1,), seed=1), P(0.05, 3.0, x=(1.2,), seed=1)],
              2: [P(0.022, 4.0, x=(1.3,), seed=2)],
              3: [P(0.01, 1.0, x=(1.4,), seed=3, feasible=False)]}
    rows, stats = s4.summarise('PID', dom, finals)
    per = [r for r in rows if r['kind'] == 'per_seed']
    assert [r['feasible'] for r in per] == [True, True, False]
    send = rows[-1]
    # per seed: seed 1 picks (0.020, 9.0) (0.05 is out of tolerance); union: (0.022, 4.0) of seed 2
    assert per[0]['f1'] == 0.020 and send['kind'] == 'send_to_HIL' and send['f1'] == 0.022 and send['ga_seed'] == 2
    assert send['a'] == repr(1.3) and float(send['a_f32']) == float(np.float32(1.3))
    assert stats['n_feasible_seeds'] == 2 and stats['per_seed_selected']['f1']['median'] == pytest.approx(0.021)


@pytest.mark.parametrize('order', [0, 1, 2])
def test_summary_union_is_not_two_level(order):
    """S4C-15: seed 1 has A (0.020, 3) and B (0.029, 1) -> picks B; seed 2 has C (0.015, 4). On the
    union of the sets f1* = 0.015, keep f1 <= 0.025: A wins (f2 3). Choosing among the per-seed picks
    (B, C) would give C. Point order inside a seed must not matter."""
    dom = [dict(name='a', L=1.0, H=2.0, scale='lin')]
    s1 = [P(0.020, 3.0, x=(1.1,), seed=1), P(0.029, 1.0, x=(1.2,), seed=1)]
    if order:
        s1 = s1[::-1]
    finals = {1: s1, 2: [P(0.015, 4.0, x=(1.3,), seed=2)]}
    if order == 2:
        finals = {2: finals[2], 1: finals[1]}
    rows, _ = s4.summarise('PID', dom, finals)
    per = {r['ga_seed']: r for r in rows if r['kind'] == 'per_seed'}
    assert per[1]['f1'] == 0.029 and per[2]['f1'] == 0.015
    send = rows[-1]
    assert (send['f1'], send['f2'], send['ga_seed']) == (0.020, 3.0, 1)


def test_summary_stops_when_more_than_half_infeasible():
    dom = [dict(name='a', L=1.0, H=2.0, scale='lin')]
    finals = {s: [P(0.02, 4.0, seed=s, feasible=s <= 2)] for s in range(1, 6)}
    rows, stats = s4.summarise('PID', dom, finals)
    assert rows[-1]['feasible'] is False and 'tell Liam' in rows[-1]['note'] and stats['n_feasible_seeds'] == 2
    finals = {s: [P(0.02, 4.0, seed=s, feasible=s <= 3)] for s in range(1, 6)}
    assert s4.summarise('PID', dom, finals)[0][-1]['feasible'] is True


def test_seed_error_rate_counts_first_attempts_only():
    ev = [dict(controller='PID', ga_seed=1, attempt=a, completed=c, status=st)
          for a, c, st in [(1, 'true', 'ok')] * 9 + [(1, '', 'run_error'), (2, 'true', 'ok'), (1, 'inconclusive', 'track_err')]]
    r = s4.seed_error_rates(ev, 'PID')[1]
    assert r['n'] == 11 and r['rate'] == pytest.approx(2 / 11) and r['over_10pct']


def test_meta_value():
    t = 'trial: true     # x\nmu_levels: [0.5]\nstatus: "ran to the end"\n'
    assert s4.meta_value(t, 'trial') is True and s4.meta_value(t, 'mu_levels') == [0.5]
    assert s4.meta_value(t, 'status') == 'ran to the end'


# ── guards (sections 4.2, 4.3, L56) ──────────────────────────────────────────────────

@pytest.mark.parametrize('extra, msg', [
    ([], 'main S03 run'),                                          # trial S03 without --tool-check
])
def test_guard_needs_main_s03(tmp_path, monkeypatch, extra, msg):
    d = _s03_dir(tmp_path, trial=True, mu=(0.5,))
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial', '--controller', 'PID', '--s03', str(d),
                                      '--out', str(tmp_path / 'o')] + extra)
    with pytest.raises(SystemExit, match=msg):
        s4.main()


def test_guard_l56_and_preconditions_and_budget(tmp_path, monkeypatch):
    d = _s03_dir(tmp_path, _runs())
    base = ['s04_ga.py', '--phase', 'trial', '--controller', 'PID', '--s03', str(d), '--out', str(tmp_path / 'o')]
    monkeypatch.setattr(sys, 'argv', base)
    with pytest.raises(SystemExit, match='L56'):
        s4.main()
    monkeypatch.setattr(sys, 'argv', base + ['--pop', '6'])
    with pytest.raises(SystemExit, match='budget is locked'):
        s4.main()
    monkeypatch.setattr(sys, 'argv', base + ['--seeds', '1', '2'])
    with pytest.raises(SystemExit, match='seeds 1-5'):
        s4.main()
    monkeypatch.setattr(sys, 'argv', base + ['--s03-unsealed', '--gate-speeds', '0.2'])       # S4C-3
    with pytest.raises(SystemExit, match='gate speeds are those of the S03 run'):
        s4.main()
    monkeypatch.setattr(sys, 'argv', base + ['--tool-check', '--seeds', '1', '1'])            # S4C-13
    with pytest.raises(SystemExit, match='repeated seed'):
        s4.main()
    if os.path.isfile(_DLL):
        for bad in ('', None, False, 0, [], {}):                                             # S4C-4
            pre = tmp_path / 'pre.json'
            pre.write_text(json.dumps(dict(s03_run_and_locked='x', final_test_set_locked=bad,
                                           ctrl_imu_checked_on_board='y', x3_compared='z')))
            monkeypatch.setattr(sys, 'argv', base + ['--s03-unsealed', '--dll', _DLL, '--preconditions', str(pre)])
            with pytest.raises(SystemExit, match='final_test_set_locked'):
                s4.main()


def _lock(out, d, tool_check_common, tool_check_ctrl=None, fp=None, pre=None, budget=None):
    pop, base, mx = budget or (s4.POP, s4.GEN_BASE, s4.GEN_MAX)
    L = dict(tool_check=tool_check_common if tool_check_ctrl is None else tool_check_ctrl,
             domain=s4.read_domain(str(d), 'PID'), s03_run=os.path.abspath(str(d)), speeds=list(s3.SPEEDS),
             eval_mu=[0.5], pop=pop, n_gen=base, gen_rule=dict(base=base, max=mx), seeds=list(s4.SEEDS),
             preconditions=pre if pre is not None else {k: 'x' for k in s4.PRECONDITIONS}, hv_ref=[s4.D, 50.0],
             stop=None)
    out.mkdir(parents=True, exist_ok=True)
    unsealed = not tool_check_common
    lock = dict(controllers=dict(PID=L), common=dict(tool_check=tool_check_common, s03_run=os.path.abspath(str(d)),
                                                    fingerprint=fp or s4.fingerprint(_DLL, str(d), unsealed=unsealed),
                                                    s03_mu_levels=[0.4, 0.5, 0.6], unsealed=unsealed),
                s03_mu_levels=[0.4, 0.5, 0.6])
    (out / 'lock.yaml').write_text(json.dumps(lock))


@needs_dll
def test_guard_one_mode_and_one_source_per_folder(tmp_path, monkeypatch):
    """S4C-1: the trial and main guards refuse a lock.yaml (written by hand here) whose mode, budget,
    preconditions or fingerprint do not fit the run. The fingerprint against real changed files,
    through a real trial and the post phase: test_lock_from_a_real_trial_catches_changed_sources."""
    d = _s03_dir(tmp_path, _runs())
    out = tmp_path / 'o'
    pre = tmp_path / 'pre.json'
    pre.write_text(json.dumps({k: 'checked in x' for k in s4.PRECONDITIONS}))
    main_trial = ['s04_ga.py', '--phase', 'trial', '--controller', 'ADRC', '--s03', str(d), '--s03-unsealed',
                  '--preconditions', str(pre), '--dll', _DLL, '--out', str(out)]
    _lock(out, d, True)                                    # PID trial was a tool check
    monkeypatch.setattr(sys, 'argv', main_trial)
    with pytest.raises(SystemExit, match='never share a folder'):
        s4.main()
    main_pid = ['s04_ga.py', '--phase', 'main', '--controller', 'PID', '--s03', str(d), '--s03-unsealed',
                '--dll', _DLL, '--out', str(out)]
    _lock(out, d, False, tool_check_ctrl=True)             # common flag forged to false, PID locked as a tool check
    monkeypatch.setattr(sys, 'argv', main_pid)
    with pytest.raises(SystemExit, match='differ in --tool-check'):
        s4.main()
    _lock(out, d, False, budget=(6, 3, 5))                 # main flag but a small budget in the lock
    with pytest.raises(SystemExit, match='not a main-run lock'):
        s4.main()
    _lock(out, d, False, pre={})                           # main flag but no preconditions
    with pytest.raises(SystemExit, match='not a main-run lock'):
        s4.main()
    fp = s4.fingerprint(_DLL, str(d), unsealed=True)
    _lock(out, d, False, fp=dict(fp, dll_sha256='0' * 64))
    with pytest.raises(SystemExit, match='dll_sha256'):
        s4.main()
    _lock(out, d, False, fp=dict(fp, s03_files=dict(fp['s03_files'], **{'levels.csv': 'x'})))
    with pytest.raises(SystemExit, match='s03_files'):
        s4.main()


@needs_dll
def test_main_stops_when_the_domain_source_differs(tmp_path, monkeypatch):
    """S04 v6.7 9.6: nguon_mien and ly_do are locked at the trial and copied into meta.yaml; a main
    run whose lock records another domain source than the sealed levels stops (S04 7)."""
    d = _s03_dir(tmp_path, _runs())
    out = tmp_path / 'o'
    _lock(out, d, False)
    lock = json.loads((out / 'lock.yaml').read_text(encoding='utf-8'))
    lock['controllers']['PID'].update(nguon_mien='du_phong', ly_do='')   # sealed file says sang_loc
    (out / 'lock.yaml').write_text(json.dumps(lock), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'main', '--controller', 'PID', '--s03', str(d),
                                    '--s03-unsealed', '--dll', _DLL, '--out', str(out)])
    with pytest.raises(SystemExit, match='source'):
        s4.main()


def _levels_near_defaults():
    fw = fwsim.FirmwareSim(_DLL)                  # a domain around the firmware defaults: both controllers feasible
    return {c: {n: ((v * 0.8, v * 1.25) if v > 0 else (0.0, 0.1)) for n, v in s3.defaults(fw, c).items()}
            for c in s3.CONTROLLERS}


@needs_dll
def test_lock_from_a_real_trial_catches_changed_sources(tmp_path, monkeypatch):
    """S4C-2, S4C-16: the lock is written by a real trial; one changed byte in a real S03 file
    (levels.csv) stops main, and with the file restored main runs; one changed byte in
    sealed/design.csv stops post, and with it restored post runs; a trial whose S03 meta.yaml
    records another DLL
    SHA-256 stops (the S03 record is changed here, not the DLL file). The other S03 source checks
    of the first trial: test_first_trial_refuses_a_source_s03_did_not_use."""
    d = _s03_dir(tmp_path, trial=True, mu=(0.5,), levels=_levels_near_defaults(), gate=(0.2,))
    out = tmp_path / 'run'
    common = ['--tool-check', '--controller', 'PID', '--s03', str(d), '--pop', '4', '--gen-base', '1',
              '--gen-max', '1', '--dll', _DLL, '--out', str(out)]
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial'] + common)
    assert s4.main() in (0, 2)
    lv = d / 'levels.csv'
    orig = lv.read_bytes()
    lv.write_bytes(orig + b'\n')                                       # same content, other bytes
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'main', '--seeds', '1'] + common)
    with pytest.raises(SystemExit, match='s03_files'):
        s4.main()
    lv.write_bytes(orig)
    s4.main()                                                          # unchanged sources: main runs
    dz = d / 'sealed' / 'design.csv'
    dorig = dz.read_bytes()
    dz.write_bytes(dorig + b'\n')
    with pytest.raises(SystemExit, match='s03_files'):
        s4.post(s4_args(out))
    dz.write_bytes(dorig)
    assert s4.post(s4_args(out)) == 0                                  # unchanged sources: post runs
    meta = d / 'meta.yaml'
    meta.write_text(meta.read_text(encoding='utf-8').replace(s3.sha256_file(_DLL), '0' * 64), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial', '--tool-check', '--controller', 'ADRC',
                                      '--s03', str(d), '--pop', '4', '--gen-base', '1', '--gen-max', '1',
                                      '--dll', _DLL, '--out', str(tmp_path / 'run2')])
    with pytest.raises(SystemExit, match='not the DLL of the S03 run'):
        s4.main()


def _other_blob(b):
    return ('0' if b[0] != '0' else '1') + b[1:]


@needs_dll
@pytest.mark.parametrize('kw, msg', [
    (dict(plant_src='plant'), 'not the S03 plant.src'),                       # params.py changed after S03
    (dict(o1c='D'), 'not the S03 o1'),                                         # an O1 constant changed after S03
    (dict(imu='rate'), 'not the S03 imu'),                                     # s03_doe.IMU changed after S03
    (dict(script='dirty'), 'no clean script commit'),
    (dict(script='unknown'), 'cannot be read in this repository'),             # git cannot read the commit
])
def test_first_trial_refuses_a_source_s03_did_not_use(tmp_path, monkeypatch, kw, msg):
    """S4C-2 (Codex rounds 3, 4): the first trial checks the S03 meta.yaml, so it cannot lock a
    plant, O1 constants or IMU configuration that the S03 run did not use, nor run when the S03 commit
    is not clean or not readable; these stop a --tool-check too, before any lock is written."""
    src = dict(fwsim.PLANT_SRC)
    arg = dict(plant_src=dict(src, params_py=_other_blob(src['params_py'])) if 'plant_src' in kw else None,
               o1c=dict(s4.o1_constants(), D=0.2) if 'o1c' in kw else None,
               imu=dict(s3.IMU, rate_hz=10.0) if 'imu' in kw else None,
               script={'dirty': dict(commit=_head(), dirty=True), 'unknown': dict(commit='0' * 40, dirty=False)}
               .get(kw.get('script')))
    d = _s03_dir(tmp_path, trial=True, mu=(0.5,), **arg)
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial', '--tool-check', '--controller', 'PID',
                                      '--s03', str(d), '--pop', '4', '--gen-base', '1', '--gen-max', '1',
                                      '--dll', _DLL, '--out', str(tmp_path / 'run')])
    with pytest.raises(SystemExit, match=msg):
        s4.main()
    assert not (tmp_path / 'run' / 'lock.yaml').exists()


@needs_dll
@pytest.mark.parametrize('path', s4.SIM_SOURCES)
def test_each_sim_source_is_compared(tmp_path, monkeypatch, capsys, path):
    """S4C-R5-1: a blob at the S03 commit that differs for this one module (and no other) is
    reported as exactly that module; a main start stops, a --tool-check warns and returns it."""
    meta = (_s03_dir(tmp_path, trial=True, mu=(0.5,)) / 'meta.yaml').read_text(encoding='utf-8')
    fp = s4.fingerprint(_DLL, str(tmp_path / 's03'))
    real = s4.blob_at
    monkeypatch.setattr(s4, 'blob_at', lambda c, p: _other_blob(real(c, p)) if p == path else real(c, p))
    assert [n for n, _, _ in s4.source_diff(_head())] == [path]
    with pytest.raises(SystemExit, match=f'not the sources of the S03 script commit: {path} '):
        s4.check_s03_source(meta, fp, _DLL)
    assert [g['path'] for g in s4.check_s03_source(meta, fp, _DLL, tool_check=True)] == [path]
    assert f'WARNING (--tool-check): not the sources of the S03 script commit: {path} ' in capsys.readouterr().out


S03_COMMIT_OLD_FWSIM = '49bbb50'   # feat/s03-doe, pushed: fwsim/fwsim.py there is older than on feat/s04-ga


@needs_dll
def test_sources_compared_by_class_with_the_s03_commit(tmp_path, monkeypatch, capsys):
    """S4C-2 (Codex round 4): every module of SIM_SOURCES (all S04 imports for simulation and
    scoring except s04_ga.py) is compared, through git, with its blob at the S03 script commit. At
    HEAD nothing differs; at the fixed older commit S03_COMMIT_OLD_FWSIM fwsim/fwsim.py differs
    (real git history, no mock): a main trial stops before any lock, a --tool-check only warns and
    returns the difference. A source that differs only through a mocked blob is caught too, and an
    S03 meta.yaml without the plant line is refused."""
    assert set(s4.SIM_SOURCES) == set(s4.SOURCES) | {s4.S03_SCREEN_SCRIPT}   # v6.7 9.6: this script + the
    assert s4.source_diff(_head()) == []                                   # screening script compared too
    old = s4.source_diff(S03_COMMIT_OLD_FWSIM)
    assert 'fwsim/fwsim.py' in [n for n, _, _ in old] and all(now != then for _, now, then in old)
    d = _s03_dir(tmp_path, _runs(), script=dict(commit=S03_COMMIT_OLD_FWSIM, dirty=False))
    pre = tmp_path / 'pre.json'
    pre.write_text(json.dumps({k: 'checked in x' for k in s4.PRECONDITIONS}))
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial', '--controller', 'PID', '--s03', str(d),
                                      '--s03-unsealed', '--preconditions', str(pre), '--dll', _DLL,
                                      '--out', str(tmp_path / 'run')])
    with pytest.raises(SystemExit, match='not the sources of the S03 script commit: .*fwsim/fwsim.py'):
        s4.main()
    assert not (tmp_path / 'run' / 'lock.yaml').exists()
    meta = (d / 'meta.yaml').read_text(encoding='utf-8')
    fp = s4.fingerprint(_DLL, str(d))
    got = s4.check_s03_source(meta, fp, _DLL, tool_check=True)
    assert 'fwsim/fwsim.py' in [g['path'] for g in got] and 'WARNING (--tool-check)' in capsys.readouterr().out
    head_meta = (_s03_dir(tmp_path / 'h', trial=True, mu=(0.5,)) / 'meta.yaml').read_text(encoding='utf-8')
    assert s4.check_s03_source(head_meta, fp, _DLL) == []
    real = s4.blob_at
    monkeypatch.setattr(s4, 'blob_at', lambda c, p: _other_blob(real(c, p)) if p == 'rpi5/imu_model.py' else real(c, p))
    with pytest.raises(SystemExit, match='rpi5/imu_model.py'):
        s4.check_s03_source(head_meta, fp, _DLL)
    monkeypatch.setattr(s4, 'blob_at', real)
    no_plant = '\n'.join(l for l in head_meta.splitlines() if not l.startswith('plant:'))
    with pytest.raises(SystemExit, match="no 'plant' line"):
        s4.check_s03_source(no_plant, fp, _DLL)


# ── GA loop: generation rule, re-evaluation, run_error (sections 3, 4.3) ─────────────

class FakeEv:
    """Deterministic candidate scores from the gains, no fwsim."""
    ctrl = 'PID'

    def __init__(self):
        self.calls = []

    def candidate(self, g, ga_seed, gen, cand):
        v = list(g.values())
        self.calls.append((gen, cand))
        f1 = 0.01 + 0.05 * abs(v[0] - 1.5)
        f2 = 1.0 + abs(v[1] - 2.0) + 0.1 * v[0]
        return dict(f1=f1, f2=f2, g1=0.0, g2=(f1 - s4.D) / s4.D, g3=-0.5, g4=-0.5, g5=-0.5, status='ok', n_runs=1,
                    n_false=0, n_inconclusive=0,
                    n_track_err=0, n_run_error=0)


DOM2 = [dict(name='a', L=1.0, H=2.0, scale='lin'), dict(name='b', L=1.0, H=3.0, scale='lin')]


@pytest.mark.parametrize('hv_seq, extended, n_gen', [
    ([1.0, 1.0, 1.0], False, 3),           # HV_3 vs HV_1 (gen_compare(3) = 1): no gain -> stop at gen_base
    ([1.0, 1.0, 1.5], True, 5),            # (1.5 - 1.0) / 1.5 > 1 % -> go on to gen_max
    ([1.0, 1.0, 1.005], False, 3),         # 0.5 % -> stop
    ([0.0, 0.0, 0.0], True, 5),            # HV_base = 0 -> go on
])
def test_generation_rule_branches(monkeypatch, hv_seq, extended, n_gen):
    """S4C-16: each branch of the S04 4.3 rule forced through a fake HV sequence."""
    seq = iter(hv_seq + [9.0] * 10)
    monkeypatch.setattr(s4, 'hypervolume', lambda feas, ref: (next(seq), 0))
    hist, hv_rows = [], []
    _, g, ext = s4.run_seed(FakeEv(), DOM2, 1, 6, None, (s4.D, 50.0), hist, hv_rows, ext=(3, 5), log=lambda *_: None)
    assert (ext, g, len(hv_rows)) == (extended, n_gen, n_gen)


def test_gen_compare_is_gen_base_minus_5():
    assert s4.gen_compare(20) == 15 and s4.gen_compare(3) == 1 and s4.gen_compare(1) == 1


def test_extended_trial_equals_a_straight_run(monkeypatch):
    """S4C-16: the trial that goes on to gen_max is the same NSGA-II run as one of gen_max generations."""
    n = iter(range(1, 100))
    monkeypatch.setattr(s4, 'hypervolume', lambda feas, ref: (float(next(n)), 0))     # rising HV: extends
    ev1, ev2 = FakeEv(), FakeEv()
    h1, h2 = [], []
    f1, g1, e1 = s4.run_seed(ev1, DOM2, 3, 6, None, (s4.D, 1e9), h1, [], ext=(2, 5), log=lambda *_: None)
    f2, g2, _ = s4.run_seed(ev2, DOM2, 3, 6, 5, (s4.D, 1e9), h2, [], log=lambda *_: None)
    assert e1 and g1 == g2 == 5
    assert [r['x_a'] for r in h1] == [r['x_a'] for r in h2] and ev1.calls == ev2.calls
    assert [p['xga'] for p in f1] == [p['xga'] for p in f2]
    # every final point carries its own evaluation id, and that evaluation holds the point's F, G
    hist = {(r['gen'], r['cand_id']): r for r in h1}
    for p in f1:
        assert p['gen'] != '' and 'does not match' not in p['id_note'] and 'no eval_id' not in p['id_note']
        h = hist[(p['gen'], p['cand_id'])]
        assert (h['f1'], h['f2']) == (p['f1'], p['f2'])


def test_points_of_uses_the_eval_id_on_the_individual():
    """S4C-6: a vector evaluated twice with the SAME F, G (for example two run_error penalties) maps
    to the evaluation id the Individual carries, not to the first evaluation of the vector."""
    from pymoo.core.individual import Individual
    X, F, G = np.array([1.2, 2.0]), np.array([1e6, 1e6]), np.array([3.0, 1e6])
    ind = Individual(X=X, F=F, G=G)
    ind.set('eval_id', (3, 2))
    seen = {X.tobytes(): [(1, 4, (1e6, 1e6, 3.0, 1e6)), (3, 2, (1e6, 1e6, 3.0, 1e6))]}
    p = s4.points_of([ind], DOM2, 1, seen)[0]
    assert (p['gen'], p['cand_id']) == (3, 2) and 'more than once' in p['id_note']
    bad = Individual(X=X, F=np.array([0.5, 0.5]), G=G)
    bad.set('eval_id', (3, 2))
    assert 'does not match' in s4.points_of([bad], DOM2, 1, seen)[0]['id_note']
    assert 'no eval_id' in s4.points_of([Individual(X=X, F=F, G=G)], DOM2, 1, seen)[0]['id_note']


def test_pareto_rows_feasible_and_infeasible_and_empty(tmp_path):
    """S4C-R-2: pareto.csv holds the non-dominated feasible points and, separately, the non-dominated
    infeasible ones; an empty set still gives a file with the header, and the merged file exists."""
    pts = [P(0.02, 5.0, x=(1.1,)), P(0.03, 6.0, x=(1.2,)), P(0.01, 1.0, x=(1.3,), feasible=False),
           P(0.02, 2.0, x=(1.4,), feasible=False)]
    rows = s4.pareto_rows('PID', {1: pts})
    assert [(r['feasible'], r['f1']) for r in rows] == [(True, 0.02), (False, 0.01)]
    flags = [(r['feasible'], r['nondominated_feasible'], r['nondominated_in_group']) for r in rows]
    assert flags == [(True, True, True), (False, False, True)]                      # S4C-R3-1: both flags
    full = s4.population_rows('PID', {1: pts})                                      # dominated points too
    assert [(r['f1'], r['nondominated_feasible'], r['nondominated_in_group']) for r in full] == [
        (0.02, True, True), (0.03, False, False), (0.01, False, True), (0.02, False, False)]
    s4.write_rows(str(tmp_path / 'pareto_PID.csv'), s4.pareto_rows('PID', {1: []}), s4.POP_COLS)
    s4.merge_controllers(str(tmp_path))
    with open(tmp_path / 'pareto.csv', newline='') as f:
        r = list(csv.reader(f))
    assert r == [s4.POP_COLS]


def test_root_meta_keeps_code_notes(tmp_path):
    """S4C-R-1: notes of CODE in code_notes.json survive every rewrite of meta.yaml."""
    yaml = pytest.importorskip('yaml')
    (tmp_path / 'code_notes.json').write_text(json.dumps({'PID seed 3': 'rate 12 %: lane reversal, no tool error'}))
    for _ in range(2):
        s4.write_root_meta(str(tmp_path), _DLL, dict(controllers={}, common=dict(tool_check=True)))
    m = yaml.safe_load(open(tmp_path / 'meta.yaml', encoding='utf-8'))
    assert m['code_notes'] == {'PID seed 3': 'rate 12 %: lane reversal, no tool error'}
    assert set(m['pymoo_files']) == set(s4.PYMOO_FILES)                  # S4C-18
    assert m['spec'] == s4.SPEC and m['spec'].startswith('S04 v6.6 ')    # S83-2


def test_default_gains_must_give_a_valid_reference(monkeypatch):
    """S4C-7: no HV reference from a run_error penalty or a non-positive f2."""
    rows = iter([(dict(status='run_error', completed='', error='x'), None)])
    monkeypatch.setattr(s4, 'safe_run_gate', lambda *a, **k: next(rows))
    monkeypatch.setattr(s3, 'run_in_new_process', lambda dll, jobs, work: [dict(row=dict(status='ok'), digest=None)])
    monkeypatch.setattr(s3, 'defaults', lambda fw, c: dict(G_PID))   # v6.8 2: candidate() checks the gain set
    assert s4.default_f2(None, 'x.dll', 'PID', [0.2], 0.5, 'w')[3] is False
    for tv in (0.0, -3.0, float('nan'), float('inf')):                 # f2 not finite or not > 0
        rows = iter([(dict(status='ok', completed='true', e_perp_max=0.02, tv_duty=tv), None)])
        assert s4.default_f2(None, 'x.dll', 'PID', [0.2], 0.5, 'w')[3] is False, tv
    rows = iter([(dict(status='ok', completed='true', e_perp_max=0.02, tv_duty=3.0), None)])
    assert s4.default_f2(None, 'x.dll', 'PID', [0.2], 0.5, 'w')[3] is True


def test_check_eval_keys_refuses_duplicates():
    """S4C-13."""
    e = dict(controller='PID', ga_seed=1, gen=1, cand_id=0, v=0.2, mu=0.5, attempt=1)
    s4.check_eval_keys([e, dict(e, attempt=2)])
    with pytest.raises(SystemExit, match='not unique'):
        s4.check_eval_keys([e, dict(e)])


def test_run_error_retry_ok_keeps_the_first_attempt_penalty(monkeypatch):
    """S04 3 (B8b-20): a run_error is run once more in a new process for diagnosis; even when that
    attempt succeeds, the candidate keeps the penalty of the first attempt."""
    ok = dict(status='ok', completed='true', e_perp_max=0.02, tv_duty=4.0)
    calls = iter([(ok, None), (dict(status='run_error', completed='', error='x'), None)])
    monkeypatch.setattr(s4, 'safe_run_gate', lambda *a, **k: next(calls))
    monkeypatch.setattr(s3, 'run_in_new_process', lambda dll, jobs, work: [dict(row=ok, digest=None)])
    ev = s4.Evaluator(None, 'x.dll', 'PID', [0.2, 0.5], [0.5], 'w')
    s = ev.candidate(G_PID, 1, 1, 0)
    assert (s['status'], s['f1'], s['g1']) == ('run_error', s4.PENALTY, 2.0)
    assert [(e['v'], e['attempt'], e['status']) for e in ev.evals] == [(0.2, 1, 'ok'), (0.5, 1, 'run_error'),
                                                                       (0.5, 2, 'ok')]


def test_three_mu_findings_flag_track_err_even_when_completed():
    """S4C-9 (v6.7 9.7: the pass comes from the gate state columns, not the maximum)."""
    dat = dict(g1_state='dat', g2_state='dat', g3_state='dat', g4_state='dat', g5_state='dat')
    rows = [dict(controller='PID', v=0.2, mu=0.4, status='track_err', completed='true', e_perp_max=0.03, **dat),
            dict(controller='PID', v=0.5, mu=0.4, status='ok', completed='true', e_perp_max=0.03, **dat),
            dict(controller='PID', v=0.8, mu=0.6, status='ok', completed='true', e_perp_max=0.2,
                 **dict(dat, g2_state='khong_dat'))]
    f = s4.three_mu_findings(rows, 'PID')
    assert len(f) == 2 and 'track_err' in f[0] and 'v=0.8' in f[1] and 'G2 khong_dat' in f[1]


def test_plant_dump_keeps_nested_values():
    """S4C-8: every attribute, nested objects included."""
    p = fwsim.plant_params(__import__('argparse').Namespace(ground='traction', mu=0.5))
    d = s4.plant_dump(p)
    assert set(d) == set(vars(p))
    assert isinstance(d['slip'], dict) and d['slip']                      # a nested object, not dropped
    json.dumps(d)


# ── end to end, tiny budget (needs the DLL) ──────────────────────────────────────────

@needs_dll
def test_tool_check_three_phases(tmp_path, monkeypatch, capsys):
    """trial -> lock.yaml, main (2 seeds) -> selected.csv with a send_to_HIL row whose gains read back
    from the file reproduce every GA run bit for bit in a new process (post)."""
    yaml = pytest.importorskip('yaml')
    d = _s03_dir(tmp_path, trial=True, mu=(0.5,), levels=_levels_near_defaults(), gate=(0.2,))
    out = tmp_path / 'run'
    for ctrl in s3.CONTROLLERS:
        common = ['--tool-check', '--controller', ctrl, '--s03', str(d), '--pop', '4', '--gen-base', '1',
                  '--gen-max', '2', '--dll', _DLL, '--out', str(out)]
        monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial'] + common)
        assert s4.main() in (0, 2)
        assert yaml.safe_load(open(out / 'meta.yaml', encoding='utf-8'))['tool_check'] is True    # S4C-12
    lock = json.load(open(out / 'lock.yaml'))
    assert lock['common']['s03_source_diff'] == []                         # S4C-2: S03 record of this checkout
    for ctrl in s3.CONTROLLERS:
        L = lock['controllers'][ctrl]
        assert L['eval_mu'] == [0.5] and L['n_gen'] in (1, 2) and L['hv_ref'][0] == s4.D and L['tool_check']
        assert L['nguon_mien'] == 'sang_loc' and 'ly_do' in L            # S04 v6.7 9.6: locked source
        assert L['gen_rule']['compared_with'] == 1
        # S04 v6.8 4.2: the fixed reference (D; 800 1/s), the default gains no longer run
        assert L['hv_ref'] == [s4.D, 800.0] and 'default_f2' not in L
    assert lock['common']['tool_check'] is True
    for ctrl in s3.CONTROLLERS:                                              # S04 v6.8 4.2: no default-gains run
        tr = list(csv.DictReader(open(out / 'trial' / ctrl / 'evals.csv')))
        assert tr and all(e['ga_seed'] != '0' for e in tr)
    for ctrl in s3.CONTROLLERS:
        monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'main', '--tool-check', '--controller', ctrl,
                                          '--s03', str(d), '--seeds', '1', '2', '--dll', _DLL, '--out', str(out)])
        s4.main()
    m_adrc = yaml.safe_load(open(out / 'meta_ADRC.yaml', encoding='utf-8'))
    # S04 v6.8 2: the outside-triangle count of every seed (0 here: the domain lies inside the triangle)
    assert m_adrc['run']['n_ngoai_bang_per_seed'] == {'1': 0, '2': 0}
    assert set(m_adrc['run']['seed_error_rates']) == {'1', '2'}                 # C3-10: one entry per seed run
    evals = list(csv.DictReader(open(out / 'evals.csv')))
    keys = [(e['controller'], e['ga_seed'], e['gen'], e['cand_id'], e['v'], e['mu'], e['attempt']) for e in evals]
    assert len(keys) == len(set(keys)) and {e['controller'] for e in evals} == set(s3.CONTROLLERS)   # S04 8
    for e in evals:                                                    # S03 v8.6 11.6: KEP columns, ADRC empty
        for k in s4.KEP_FRAC_COLS + ['kep_note']:
            assert k in e
        if e['controller'] == 'ADRC':
            assert all(e[k] == '' for k in s4.KEP_FRAC_COLS + ['kep_note'])
        else:
            assert all(e[k] == '' or math.isfinite(float(e[k])) for k in s4.KEP_FRAC_COLS)
    sel = list(csv.DictReader(open(out / 'selected.csv')))
    hist = list(csv.DictReader(open(out / 'history.csv')))
    for send in (r for r in sel if r['kind'] == 'send_to_HIL'):
        h = next(r for r in hist if (r['controller'], r['ga_seed'], r['gen'], r['cand_id']) ==
                 (send['controller'], send['ga_seed'], send['gen'], send['cand_id']))
        assert float(h['f1']) == float(send['f1'])                       # selected row is a GA candidate
    par = list(csv.DictReader(open(out / 'pareto.csv')))
    pop = list(csv.DictReader(open(out / 'final_population.csv')))
    assert par and len(pop) >= len(par)                                  # S4C-17
    assert all(r['nondominated_in_group'] == 'true' for r in par)        # S4C-R3-1: both groups, flag per group
    assert all(r['nondominated_feasible'] == r['feasible'] for r in par)
    assert sum(r['nondominated_in_group'] == 'true' for r in pop) == len(par)
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'post', '--dll', _DLL, '--out', str(out)])
    rc = s4.post(s4_args(out))
    pc = out / 'post_checks'
    det = list(csv.DictReader(open(pc / 'determinism.csv')))
    assert det and all(r['equal'] == 'true' and r['tool_check'] == 'true' for r in det)
    sens = list(csv.DictReader(open(pc / 'sensitivity.csv')))
    assert {r['r_factor'] for r in sens if r['check'] == 'r_factor'} == {repr(k) for k in s4.R_FACTORS}
    assert {r['b0_factor'] for r in sens if r['check'] == 'b0_factor'} == {repr(k) for k in s4.B0_FACTORS}
    changed = [json.loads(r['plant_changed']) for r in sens if r['check'] == 'r_factor' and r['r_factor'] != '1.0']
    assert changed and all(set(c) == {'motor_R'} for c in changed)       # only R changes (S04 6)
    full = json.load(open(pc / 'plant_params.json'))['entries']          # S4C-8
    assert any('r_factor=1.2' in k for k in full) and all(set(v) == {'before', 'after'} for v in full.values())
    notes = json.load(open(pc / 'notes.json'))
    assert notes['tool_check'] is True                                   # S4C-10
    assert rc == (2 if notes['cho_liam'] else 0) and 'cho_liam_note' in notes  # a cho_liam finding -> rc 2
    assert all(c['check'] in ('pre_hil', 'three_mu', 'spectral_radius', 'b0_factor', 'r_factor')
               for c in notes['cho_liam'])
    assert notes['notes']['PID three_mu']['done'] is False and 'NOT done' in notes['notes']['PID three_mu']['note']
    for r in sens:                                                       # S04 v6.7 9.7: gate state columns
        assert all(r[f'{c}_state'] in s4.og.STATES for c in s4.og.CONDS)
        assert all(k in r for k in ('e_trk_max', 'e_bao_max', 'e_theta_cmd_max_deg'))
    mu3 = list(csv.DictReader(open(pc / 'three_mu.csv')))
    assert all(all(f'{c}_state' in r for c in s4.og.CONDS) for r in mu3)
    mh = s4.mo_hinh_con
    for c in s3.CONTROLLERS:                                             # S04 v6.8 6: pre-HIL check ran
        ph = notes['notes'][f'{c} pre_hil']
        assert ph['ok'] is True and isinstance(ph['spectral_radius'], float)
        send = next(s for s in sel if s['controller'] == c and s['kind'] == 'send_to_HIL')
        gains = {n: float(send[n]) for n in s3.FACTORS[c]}
        assert ph['spectral_radius'] == mh.ban_kinh_pho(c, gains)
        assert ph['spectral_radius_states'] == list(mh.TEN[c]) or ph['spectral_radius_dropped']
        assert ph['spectral_radius_state'] in ('on_dinh', 'khong_on_dinh')
        assert ph['spectral_radius_model']['module'] == 'tools/mo_hinh_con.py'
    assert 'G3, G4 or G5 state' in notes['criterion']
    root_meta = yaml.safe_load(open(out / 'meta.yaml', encoding='utf-8'))
    assert set(root_meta['pre_hil']) == {'PID pre_hil', 'ADRC pre_hil'}
    for c in s3.CONTROLLERS:                                             # S04 v6.7 9.6: in meta.yaml too
        assert root_meta['controllers'][c]['nguon_mien'] == 'sang_loc'
        assert 'ly_do' in root_meta['controllers'][c]
    for p in [out / 'meta.yaml', out / 'meta_PID.yaml', out / 'meta_ADRC.yaml', out / 'trial' / 'PID' / 'meta.yaml',
              out / 'trial' / 'ADRC' / 'meta.yaml', out / 'lock.yaml']:
        m = yaml.safe_load(open(p, encoding='utf-8'))                    # S4C-11: valid YAML on Windows paths
        if p.name.startswith('meta'):
            assert m['spec'] == s4.SPEC and m['spec'].startswith('S04 v6.6 '), p   # S83-2: one version line

    # S04 v6.8 6: --phase sens_after (report only) adds sens_after/ and changes no hashed input
    hashed = [out / 'phases.json', out / 'meta.yaml', out / 'lock.yaml'] + sorted(pc.iterdir())
    snap = {p: p.read_bytes() for p in hashed if p.is_file()}
    assert s4.sens_after(argparse.Namespace(out=str(out), dll=_DLL, io_delay_ms=None)) == 0
    rows = list(csv.DictReader(open(out / 'sens_after' / 'delay.csv')))
    for c in s3.CONTROLLERS:
        mine = [r for r in rows if r['controller'] == c]
        assert len(mine) == 3 and [r['io_delay_ms'] for r in mine] == ['0', '1', '2']   # 1 speed x 1 mu
        assert [r['baseline'] for r in mine] == ['true', 'false', 'false']
        assert all(r[f'{c_}_state'] in s4.og.STATES for r in mine for c_ in s4.og.CONDS)
        send = next(s for s in sel if s['controller'] == c and s['kind'] == 'send_to_HIL')
        e = next(x for x in evals if (x['controller'], x['ga_seed'], x['gen'], x['cand_id'], x['attempt'])
                 == (c, send['ga_seed'], send['gen'], send['cand_id'], '1'))
        assert float(mine[0]['e_perp_max']) == float(e['e_perp_max'])    # io_delay_ms 0 = the GA condition
    n = json.load(open(out / 'sens_after' / 'notes.json'))
    assert n['report_only'] is True and n['io_delay_ms_run'] == [1, 2] and n['measured_io_delay_ms'] is None
    assert 'not_in_this_script' not in n                                 # C3 note: key dropped
    assert n['variants'] == ['io_delay_ms'] + list(s4.PLANT_VARIANTS)    # S04 v6.1 6, RB-1
    assert set(n['plant_variants']) == set(s4.PLANT_VARIANTS) and 's01_identify' in n['plant_basis']
    assert 'seal is opened' in n['when']
    # S04 v6.1 6, RB-1: plant.csv = 1 baseline + one row per variant per (controller, v, mu)
    pl = list(csv.DictReader(open(out / 'sens_after' / 'plant.csv')))
    assert list(pl[0].keys()) == s4.PLANT_SENS_COLS
    for c in s3.CONTROLLERS:
        mine = [r for r in pl if r['controller'] == c]
        assert len(mine) == 1 + len(s4.PLANT_VARIANTS)                   # 1 speed x 1 mu x (base + 4)
        assert mine[0]['variant'] == 'baseline' and mine[0]['baseline'] == 'true'
        assert mine[0]['plant_changed'] == '[]' and mine[0]['tv_duty'] != ''
        assert [r['variant'] for r in mine[1:]] == list(s4.PLANT_VARIANTS)
        assert all(r['baseline'] == 'false' and r['tv_duty'] != '' for r in mine[1:])
        assert all(r[f'{c_}_state'] in s4.og.STATES for r in mine for c_ in s4.og.CONDS)
        send = next(s for s in sel if s['controller'] == c and s['kind'] == 'send_to_HIL')
        e = next(x for x in evals if (x['controller'], x['ga_seed'], x['gen'], x['cand_id'], x['attempt'])
                 == (c, send['ga_seed'], send['gen'], send['cand_id'], '1'))
        assert float(mine[0]['e_perp_max']) == float(e['e_perp_max'])    # baseline = the GA condition
        changed = {r['variant']: json.loads(r['plant_changed']) for r in mine[1:]}
        assert changed['R0.82_only'] == ['motor_R'] and changed['Vbatt11.1'] == ['V_batt']
        for vn in ('R0.8_JbTcTs', 'R1.25_JbTcTs'):
            assert 'motor_R' in changed[vn] and 'motor_K' not in changed[vn]
    pp = json.load(open(out / 'sens_after' / 'plant_params.json'))
    assert set(pp) >= {'note', 'definitions', 'entries'} and set(pp['definitions']) == set(s4.PLANT_VARIANTS)
    assert {k.split(' ', 1)[1].rsplit(' mu=', 1)[0] for k in pp['entries']} == set(s4.PLANT_VARIANTS)
    for k, ent in pp['entries'].items():
        vn = k.split(' ', 1)[1].rsplit(' mu=', 1)[0]
        assert set(ent['changed']) == set(ent['expected']) | set(ent['derived'])
        assert len(ent['slip_nsub']) == 2 and all(isinstance(x, int) for x in ent['slip_nsub'])  # [before, after]
        if vn in ('R0.8_JbTcTs', 'R1.25_JbTcTs'):
            assert 'motor_K' not in ent['changed'] and 'motor_u0' not in ent['changed']
    assert all(p.read_bytes() == b for p, b in snap.items())             # report only: no hashed input changes
    with pytest.raises(SystemExit, match='already holds data'):
        s4.sens_after(argparse.Namespace(out=str(out), dll=_DLL, io_delay_ms=None))


def s4_args(out):
    import argparse
    return argparse.Namespace(out=str(out), dll=_DLL)


# ── g3-g5 and penalty flags (S04 v6.2 SC-8, v6.3 RS2-8; R134 package) ─────────────────

def test_score_g3_g4_g5_are_maxima_over_runs():
    rows = [(0.5, 0.2, R(trk=0.03, bao=0.05, th=2.0)), (0.5, 0.5, R(trk=0.12, bao=0.04, th=9.0))]
    s = s4.score(rows, False)
    assert s['g3'] == pytest.approx((0.12 - s4.D) / s4.D) and s['g4'] == pytest.approx((0.05 - s4.D) / s4.D)
    assert s['g5'] == pytest.approx((9.0 - 7.5) / 7.5) and s4.THETA_MAX_DEG == 7.5
    assert (s['pen_f1'], s['n_pen_runs']) == (0, 0)


@pytest.mark.parametrize('key', s4.PEN_KEYS)
@pytest.mark.parametrize('bad', [None, '', 'nan', float('nan'), float('inf'), 'x'])
def test_missing_or_non_finite_quantity_is_penalised_and_flagged(key, bad):
    row = R()
    if bad is None:
        del row[key]
    else:
        row[key] = bad
    vals, flags = s4.run_values(row)
    assert vals[key] == s4.PENALTY and flags['pen_' + key] == 1
    assert all(flags['pen_' + k] == 0 and vals[k] == float(R()[k]) for k in s4.PEN_KEYS if k != key)
    s = s4.score([(0.5, 0.2, row), (0.5, 0.5, R())], False)
    gi = {'e_perp_max': 'g2', 'e_trk_max': 'g3', 'e_bao_max': 'g4', 'e_theta_cmd_max_deg': 'g5'}[key]
    lim = s4.THETA_MAX_DEG if key == 'e_theta_cmd_max_deg' else s4.D
    assert s[gi] == pytest.approx((s4.PENALTY - lim) / lim) and s['n_pen_runs'] == 1
    assert s['pen_f1'] == int(key == 'e_perp_max') and all(math.isfinite(s[f'g{i}']) for i in range(1, 6))


def test_run_error_flags_every_quantity_and_penalises_every_constraint():
    vals, flags = s4.run_values(dict(status='run_error', completed='', e_trk_max=0.01))
    assert all(v == s4.PENALTY for v in vals.values()) and all(f == 1 for f in flags.values())
    rows = [(0.5, 0.2, R()), (0.5, 0.5, dict(completed='', status='run_error', error='x'))]
    s = s4.score(rows, True)
    assert (s['f1'], s['f2'], s['g1'], s['status'], s['pen_f1'], s['n_pen_runs']) == (
        s4.PENALTY, s4.PENALTY, 2.0, 'run_error', 1, 1)
    assert s['g3'] == s['g4'] == pytest.approx((s4.PENALTY - s4.D) / s4.D)
    assert s['g5'] == pytest.approx((s4.PENALTY - 7.5) / 7.5)


def test_score_refuses_a_constraint_that_is_not_finite(monkeypatch):
    """S04 v6.2 section 3: G is checked finite before the optimiser (forced through run_values)."""
    monkeypatch.setattr(s4, 'run_values', lambda row: ({k: float('inf') for k in s4.PEN_KEYS},
                                                       {'pen_' + k: 0 for k in s4.PEN_KEYS}))
    with pytest.raises(RuntimeError, match='not finite'):
        s4.score([(0.5, 0.2, R())], False)


def test_evals_rows_carry_e_bao_and_the_flags(monkeypatch):
    calls = iter([(R(bao=float('nan')), None), (R(), None)])
    monkeypatch.setattr(s4, 'safe_run_gate', lambda *a, **k: next(calls))
    ev = s4.Evaluator(None, 'x.dll', 'PID', [0.2, 0.5], [0.5], 'w')
    s = ev.candidate(G_PID, 1, 1, 0)
    assert [e['pen_e_bao_max'] for e in ev.evals] == [1, 0] and s['g4'] == pytest.approx((s4.PENALTY - s4.D) / s4.D)
    assert all(set(s4.PEN_COLS) <= set(e) for e in ev.evals) and {'e_bao_max'} <= set(s4.EVAL_COLS)


def test_g5_alone_makes_the_point_infeasible():
    from pymoo.core.individual import Individual
    X = np.array([1.2, 2.0])
    G = np.array([0.0, -0.5, -0.5, -0.5, 0.2])
    ind = Individual(X=X, F=np.array([0.05, 3.0]), G=G)
    ind.set('eval_id', (1, 0))
    seen = {X.tobytes(): [(1, 0, (0.05, 3.0, 0.0, -0.5, -0.5, -0.5, 0.2))]}
    p = s4.points_of([ind], DOM2, 1, seen)[0]
    assert p['feasible'] is False and p['g5'] == 0.2 and p['id_note'] == ''
    row = s4.population_rows('PID', {1: [p]})[0]
    assert (row['g3'], row['g4'], row['g5']) == (-0.5, -0.5, 0.2) and set(s4.POP_COLS) >= {'g3', 'g4', 'g5'}


GATE_FIELD = re.compile(r'^(e_bao_max|gate_[a-z_]+|g[1-5]_[a-z_]+)$')     # G1-G5 fields an s03_doe row may carry


def _same_fields(a, b):
    """run_gate row b against s03_doe.run_one row a: b adds e_bao_max and the kep report fields
    (S03 v8.6 11.6); every field of b keeps the value of a; a field of a that b has not is a G1-G5 field
    of an s03_doe that writes them (hil feat/s03-gate, S3G-5; CODE 15/11), and an e_bao_max in a equals
    the one of b bit for bit."""
    added = {'e_bao_max'} | set(s4.KEP_FRAC_COLS) | {'kep_note'}
    assert 'e_bao_max' in b and set(b) - set(a) <= added
    # the gate columns s03_doe writes (v8.2 GATE_RUN_COLS, B81-3 G1_RUN_COLS 15/11): by name, not a pattern (the
    # pattern missed g1_t1, g1_t_done_1)
    gate_cols = set(getattr(s3, 'GATE_RUN_COLS', ())) | set(getattr(s3, 'G1_RUN_COLS', ()))
    assert set(a) - set(b) <= gate_cols and all(GATE_FIELD.match(k) or k in gate_cols for k in set(a) - set(b)), \
        sorted(set(a) - set(b) - gate_cols)
    keys = [k for k in b if k not in added]
    assert s3.row_text({k: a[k] for k in keys}) == s3.row_text({k: b[k] for k in keys})
    if 'e_bao_max' in a:
        assert s3._bits(a['e_bao_max']) == s3._bits(b['e_bao_max'])


@needs_dll
def test_run_gate_matches_run_one_and_adds_e_bao():
    """The gate run is s03_doe.run_one with o1_gate_eval: every field of run_one keeps its value."""
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    import o1_feasibility as o1
    path = o1.Path8()
    g = s3.defaults(fw, 'PID')
    a, la = s3.run_one(fw, path, 'PID', g, 0.5, 0.436)
    b, lb = s4.run_gate(fw, path, 'PID', g, 0.5, 0.436)
    assert s3.log_digest(la) == s3.log_digest(lb)
    _same_fields(a, b)
    assert math.isfinite(b['e_bao_max'])


# ── after Codex round 1 (R134C-7, R134C-13; Liam 14/11 CODE_R134 "Sua + vong 2 (Recommended)") ──

def test_safe_run_gate_lets_s03stop_through(monkeypatch):
    """R134C-7: a tool / configuration error stops the GA as in s03_doe.safe_run, not a run_error row."""
    def stop(*a, **k):
        raise s3.S03Stop('PID kd != 0')
    monkeypatch.setattr(s4, 'run_gate', stop)
    with pytest.raises(s3.S03Stop):
        s4.safe_run_gate(None, None, 'PID', {}, 0.5, 0.5)
    ev = s4.Evaluator(None, 'x.dll', 'PID', [0.2], [0.5], 'w')
    with pytest.raises(s3.S03Stop):
        ev.candidate(G_PID, 1, 1, 0)
    assert ev.evals == []

    def boom(*a, **k):
        raise ValueError('evaluator')
    monkeypatch.setattr(s4, 'run_gate', boom)
    row, log = s4.safe_run_gate(None, None, 'PID', {}, 0.5, 0.5)
    assert row['status'] == 'run_error' and 'ValueError' in row['error'] and log is None


def test_g3_g4_g5_maxima_found_at_another_mu():
    """R134C-13: the largest e_trk, e_bao, e_theta of the candidate is at the second mu, not the first."""
    rows = [(0.4, 0.2, R(trk=0.03, bao=0.04, th=2.0)), (0.4, 0.5, R(trk=0.04, bao=0.03, th=3.0)),
            (0.6, 0.2, R(trk=0.11, bao=0.02, th=1.0)), (0.6, 0.5, R(trk=0.05, bao=0.09, th=8.5))]
    s = s4.score(rows, False)
    assert s['g3'] == pytest.approx((0.11 - s4.D) / s4.D) and s['g4'] == pytest.approx((0.09 - s4.D) / s4.D)
    assert s['g5'] == pytest.approx((8.5 - 7.5) / 7.5)


class InfeasibleEv(FakeEv):
    """Every candidate breaks one constraint only (g5 > 0 by default), through the real score()."""

    def __init__(self, bad='g5'):
        super().__init__()
        self.bad = bad

    def candidate(self, g, ga_seed, gen, cand):
        self.calls.append((gen, cand))
        r = R(e=0.02, tv=5.0, trk=0.03, bao=0.04, th=2.0)
        r.update({'g3': dict(e_trk_max=0.2), 'g4': dict(e_bao_max=0.2), 'g5': dict(e_theta_cmd_max_deg=9.0)}[self.bad])
        return s4.score([(0.5, 0.2, r)], False)


@pytest.mark.parametrize('bad', ['g3', 'g4', 'g5'])
def test_one_constraint_alone_leaves_no_feasible_point_and_no_send(bad):
    """R134C-13: candidates that break only G3, G4 or G5, evaluated through run_seed: no feasible point,
    HV 0 every generation, the trial goes on to gen_max (HV_base = 0), and summarise sends nothing."""
    hist, hv_rows = [], []
    pts, n_gen, ext = s4.run_seed(InfeasibleEv(bad), DOM2, 1, 6, None, (s4.D, 50.0), hist, hv_rows, ext=(3, 5),
                                  log=lambda *_: None)
    assert pts and not any(p['feasible'] for p in pts) and all(p[bad] > 0 for p in pts)
    assert all(h['hv'] == 0.0 and h['n_feasible'] == 0 for h in hv_rows) and (ext, n_gen) == (True, 5)
    rows, st = s4.summarise('PID', DOM2, {1: pts, 2: pts})
    send = [r for r in rows if r['kind'] == 'send_to_HIL']
    assert len(send) == 1 and send[0]['feasible'] is False and st['n_feasible_seeds'] == 0


@needs_dll
@pytest.mark.parametrize('ctrl, v, mu', [('PID', 0.2, 0.6), ('ADRC', 0.5, 0.436), ('ADRC', 0.8, 0.3)])
def test_run_gate_matches_run_one_for_both_controllers(ctrl, v, mu):
    """R134C-13: run_gate = s03_doe.run_one + e_bao_max for both controllers, other speeds and mu."""
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    import o1_feasibility as o1
    path = o1.Path8()
    g = s3.defaults(fw, ctrl)
    a, la = s3.run_one(fw, path, ctrl, g, v, mu)
    b, lb = s4.run_gate(fw, path, ctrl, g, v, mu)
    assert s3.log_digest(la) == s3.log_digest(lb)
    _same_fields(a, b)
    assert math.isfinite(b['e_bao_max'])


@pytest.mark.parametrize('bad, row_mod', [('g3', dict(e_trk_max=0.2)), ('g4', dict(e_bao_max=0.2)),
                                          ('g5', dict(e_theta_cmd_max_deg=9.0))])
def test_each_constraint_through_the_real_evaluator(monkeypatch, bad, row_mod):
    """R134C2-10: G3, G4, G5 through Evaluator.candidate (only the simulation row is made up), then
    run_seed: no feasible point, and the breaking constraint is the only one above 0."""
    def fake_gate(fw, path, ctrl, g, v, mu):
        return dict(R(), **row_mod), None
    monkeypatch.setattr(s4, 'safe_run_gate', fake_gate)
    ev = s4.Evaluator(None, 'x.dll', 'PID', [0.2, 0.5], [0.4, 0.6], 'w')
    s = ev.candidate(G_PID, 1, 1, 0)
    assert s[bad] > 0 and all(s[g] <= 0 for g in ('g1', 'g2', 'g3', 'g4', 'g5') if g != bad)
    assert len(ev.evals) == 4 and all(e['attempt'] == 1 for e in ev.evals)
    pts, _, _ = s4.run_seed(ev, DOM_PID7, 1, 4, 2, (s4.D, 50.0), [], [], log=lambda *_: None)  # v6.8: 7 names
    assert pts and not any(p['feasible'] for p in pts)


# ── S04 v6.8: gain table, GA condition, pre-HIL check, sens_after ─────────────────────

def test_gain_check_table_and_triangle():
    """S04 v6.8 section 2: tools/gain_table.check on the float32 values the firmware gets; the range
    violations start with '<gain>=', the ADRC triangle violation is the only other kind; the warnings
    of check() are not violations."""
    assert s4.gain_check('PID', G_PID) == ([], [])
    assert s4.gain_check('PID', dict(G_PID, kp_xy=41.0)) == ([], [])          # the bound is allowed
    assert s4.gain_check('PID', dict(G_PID, kp_xy=41.0000001)) == ([], [])    # float32 is exactly 41.0
    t, tri = s4.gain_check('PID', dict(G_PID, kp_xy=41.00001))
    assert len(t) == 1 and t[0].startswith('kp_xy=') and tri == []
    assert s4.gain_check('PID', dict(G_PID, kp_wz=-0.1))[0]
    t, _ = s4.gain_check('PID', dict(G_PID, ki_xy=float('nan')))
    assert len(t) == 1 and 'is not finite' in t[0]
    assert s4.gain_check('ADRC', G_ADRC) == ([], [])
    t, tri = s4.gain_check('ADRC', dict(G_ADRC, omega_c_xy=150.0, omega_o_xy=48.0))
    assert t == [] and len(tri) == 1 and 'omega_c_xy' in tri[0]
    # binary64 sum 199.999999999 < 200, but the float32 value of 104.0 - 1e-9 is 104.0: sum 200 -> outside
    assert s4.gain_check('ADRC', dict(G_ADRC, omega_c_xy=104.0 - 1e-9, omega_o_xy=48.0))[1]
    assert s4.gain_check('ADRC', dict(G_ADRC, omega_c_xy=103.9, omega_o_xy=48.0)) == ([], [])
    for bad in ({k: v for k, v in G_PID.items() if k != 'k_psi'}, {}):        # check() cannot read it
        with pytest.raises(s3.S03Stop):
            s4.gain_check('PID', bad)
    assert 'tools/gain_table.py' in s4.SOURCES and 'tools/gain_table.py' in s4.SIM_SOURCES


_Verdict = collections.namedtuple('Verdict', 'trong_bang trong_tam_giac vi_pham_bang vi_pham_tam_giac')


def _fake_classify(ctrl, gains):
    """C3a gain_table.classify interface (the Verdict NamedTuple), with deliberately different wording:
    the texts check() writes are NOT what gain_check reports — the Verdict fields decide."""
    gt = s4.gain_table
    g = gt._gains(ctrl, gains)                 # float32 values; ValueError on a malformed set
    bang = [f'ngoai bang: {n} ({g[n]!r})' for i, n in enumerate(gt.NAMES[ctrl])
            if not (math.isfinite(g[n]) and gt._f32(gt.LO[ctrl][i]) <= g[n] <= gt._f32(gt.HI[ctrl][i]))]
    tam, ok = [], True
    if ctrl == 'ADRC':
        for ax in ('xy', 'wz'):
            c, o = g[f'omega_c_{ax}'], g[f'omega_o_{ax}']
            if not (math.isfinite(c) and math.isfinite(o)):
                ok = False
            elif not c + 2.0 * o < 200.0:      # binary64 sum of the float32 values, both axes hard
                tam.append(f'ngoai tam giac truc {ax}')
                ok = False
    return _Verdict(not bang, ok, tuple(bang), tuple(tam))


def test_gain_check_uses_classify_when_the_table_has_it(monkeypatch):
    """C3-11: with classify(), its Verdict fields decide what a table or a triangle violation is,
    never the wording of a message; a table without it is a tool error (C3-25, next tests — the
    real table of this worktree has it, the fake here only changes the wording)."""
    monkeypatch.setattr(s4.gain_table, 'classify', _fake_classify, raising=False)
    assert s4.gain_check('PID', G_PID) == ([], [])
    assert s4.gain_check('ADRC', G_ADRC) == ([], [])
    t, tri = s4.gain_check('PID', dict(G_PID, kp_xy=41.00001))
    assert len(t) == 1 and t[0].startswith('ngoai bang: kp_xy') and tri == []
    assert s4.gain_check('PID', dict(G_PID, kp_xy=41.0000001)) == ([], [])    # float32 is exactly 41.0
    # Codex: 150 + 2 x 25 = 200 must block (the fake writes its own text for it)
    assert s4.gain_check('ADRC', dict(G_ADRC, omega_c_wz=150.0, omega_o_wz=25.0)) == \
        ([], ['ngoai tam giac truc wz'])
    # Codex's distinguishing float32 case: binary64 sum 199.99999237060547 < 200
    assert s4.gain_check('ADRC', dict(G_ADRC, omega_c_xy=103.99999237060547, omega_o_xy=48.0)) == ([], [])
    t, tri = s4.gain_check('ADRC', dict(G_ADRC, omega_c_xy=float('nan')))
    assert len(t) == 1 and tri == []
    with pytest.raises(s3.S03Stop):                        # a malformed set (a key missing)
        s4.gain_check('PID', {k: v for k, v in G_PID.items() if k != 'k_psi'})
    # the flags decide without texts: empty violation tuples still give one text each
    monkeypatch.setattr(s4.gain_table, 'classify', lambda c, g: _Verdict(True, False, (), ()))
    t, tri = s4.gain_check('ADRC', G_ADRC)
    assert t == [] and len(tri) == 1
    monkeypatch.setattr(s4.gain_table, 'classify', lambda c, g: _Verdict(False, True, (), ()))
    t, tri = s4.gain_check('ADRC', G_ADRC)
    assert len(t) == 1 and tri == []
    monkeypatch.setattr(s4.gain_table, 'classify', _fake_classify)             # back to the fake
    # through the callers (with _fake_classify)
    calls = []
    monkeypatch.setattr(s4, 'safe_run_gate', lambda *a, **k: calls.append(a))
    ev = s4.Evaluator(None, 'x.dll', 'ADRC', [0.2, 0.5], [0.5], 'w')
    s = ev.candidate(dict(G_ADRC, omega_c_wz=150.0, omega_o_wz=25.0), 1, 1, 0)
    assert s['status'] == 'ngoai_bang' and calls == [] and ev.evals == []
    ph = s4.pre_hil_check('ADRC', dict(G_ADRC, omega_c_wz=150.0, omega_o_wz=25.0))
    assert (ph['ok'], ph['in_table'], ph['in_triangle']) == (False, True, False)
    with pytest.raises(s3.S03Stop, match='range table'):
        s4.Evaluator(None, 'x.dll', 'PID', [0.2, 0.5], [0.5], 'w').candidate(dict(G_PID, kp_wz=5.0), 1, 1, 0)


def test_gain_check_without_classify_is_a_tool_error(monkeypatch):
    """C3-25: a tools/gain_table without callable classify() (stand-in exposing only check and NAMES) is a
    tool error (S03Stop) in gain_check, Evaluator.candidate and pre_hil_check — never a text fallback, and
    no simulation is called."""
    real = s4.gain_table
    monkeypatch.setattr(s4, 'gain_table', types.SimpleNamespace(check=real.check, NAMES=real.NAMES))
    assert not hasattr(s4.gain_table, 'classify')
    for f in (lambda: s4.gain_check('PID', dict(G_PID, kp_xy=41.00001)),
              lambda: s4.gain_check('ADRC', dict(G_ADRC, omega_c_xy=150.0, omega_o_xy=48.0)),
              lambda: s4.gain_check('PID', {})):
        with pytest.raises(s3.S03Stop, match='classify'):
            f()
    calls = []
    monkeypatch.setattr(s4, 'safe_run_gate', lambda *a, **k: calls.append(a))
    with pytest.raises(s3.S03Stop, match='classify'):
        s4.Evaluator(None, 'x.dll', 'ADRC', [0.2], [0.5], 'w').candidate(G_ADRC, 1, 1, 0)
    with pytest.raises(s3.S03Stop, match='classify'):            # not swallowed into a table violation
        s4.pre_hil_check('PID', G_PID)
    assert calls == []


def test_triangle_cases_through_the_real_classify(monkeypatch):
    """C3-25 through the REAL tools/gain_table.classify (no stand-in): ADRC omega_c_wz 150 + 2 x 25
    = 200 is a triangle violation in gain_check, a ngoai_bang candidate with no run in
    Evaluator.candidate and in_triangle False in pre_hil_check; omega_c_xy = float32
    103.99999237060547 with omega_o_xy 48 (binary64 sum 199.99999237060547 < 200) passes all three."""
    calls = []
    monkeypatch.setattr(s4, 'safe_run_gate', lambda *a, **k: calls.append(a))
    g = dict(G_ADRC, omega_c_wz=150.0, omega_o_wz=25.0)
    t, tri = s4.gain_check('ADRC', g)
    assert t == [] and len(tri) == 1 and 'omega_c_wz' in tri[0]
    ev = s4.Evaluator(None, 'x.dll', 'ADRC', [0.2, 0.5], [0.5], 'w')
    s = ev.candidate(g, 1, 1, 0)
    assert s['status'] == 'ngoai_bang' and calls == [] and ev.evals == []
    ph = s4.pre_hil_check('ADRC', g)
    assert (ph['ok'], ph['in_table'], ph['in_triangle']) == (False, True, False)
    g2 = dict(G_ADRC, omega_c_xy=float(np.float32(103.99999237060547)), omega_o_xy=48.0)
    assert s4.gain_check('ADRC', g2) == ([], [])
    monkeypatch.setattr(s4, 'safe_run_gate', lambda *a, **k: (calls.append(a), (R(), None))[1])
    s = ev.candidate(g2, 1, 1, 0)
    assert s['status'] == 'ok' and len(calls) == 2               # 1 eval mu x 2 gate speeds
    ph = s4.pre_hil_check('ADRC', g2)
    assert (ph['ok'], ph['in_table'], ph['in_triangle']) == (True, True, True)


def test_main_post_sens_after_stop_without_classify(tmp_path, monkeypatch):
    """C3-25: with a gain_table stand-in without classify, main() (trial, --tool-check), post() and
    sens_after() raise SystemExit matching 'classify' and FirmwareSim is never created."""
    real = s4.gain_table
    monkeypatch.setattr(s4, 'gain_table', types.SimpleNamespace(check=real.check, NAMES=real.NAMES))
    calls = []
    monkeypatch.setattr(fwsim, 'FirmwareSim', lambda *a, **k: calls.append(a))
    d = _s03_dir(tmp_path, trial=True)
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial', '--tool-check', '--controller',
                                    'PID', '--s03', str(d), '--out', str(tmp_path / 'o')])
    with pytest.raises(SystemExit, match='classify'):
        s4.main()
    out = _lock_dir(tmp_path / 'r', tool_check=True, s03=d, unsealed=False)
    for f in (lambda: s4.post(argparse.Namespace(out=str(out), dll='x.dll')),
              lambda: s4.sens_after(argparse.Namespace(out=str(out), dll='x.dll', io_delay_ms=None))):
        with pytest.raises(SystemExit, match='classify'):
            f()
    assert calls == []


def test_candidate_outside_the_table_stops_before_any_run(monkeypatch):
    """S04 v6.8 section 2: a gain outside the range table of the campaign binary is a tool error, so
    the GA domain lies inside the table: the run stops before any simulation of the candidate."""
    calls = []
    monkeypatch.setattr(s4, 'safe_run_gate', lambda *a, **k: calls.append(a))
    ev = s4.Evaluator(None, 'x.dll', 'PID', [0.2, 0.5], [0.5], 'w')
    with pytest.raises(s3.S03Stop, match='range table'):
        ev.candidate(dict(G_PID, kp_wz=5.0), 1, 1, 0)
    assert calls == [] and ev.evals == []


@pytest.mark.parametrize('c', [150.0, 104.0 - 1e-9])
def test_candidate_outside_the_triangle_is_not_run(monkeypatch, c):
    """S04 v6.8 section 2 (G-Q2): an ADRC candidate outside the triangle is not run; it gets the
    penalty of a run_error candidate of section 3 with status ngoai_bang and no evals row."""
    calls = []
    monkeypatch.setattr(s4, 'safe_run_gate', lambda *a, **k: calls.append(a))
    ev = s4.Evaluator(None, 'x.dll', 'ADRC', [0.2, 0.5], [0.4, 0.5, 0.6], 'w')
    s = ev.candidate(dict(G_ADRC, omega_c_xy=c, omega_o_xy=48.0), 1, 1, 0)
    assert s['status'] == 'ngoai_bang' and calls == [] and ev.evals == []
    assert (s['g1'], s['n_runs'], s['n_run_error']) == (6.0, 0, 0)            # g1 = the expected runs
    ref = s4.score([(m, v, dict(status='run_error', completed=''))
                    for m in (0.4, 0.5, 0.6) for v in (0.2, 0.5)], True)
    assert all(s[k] == ref[k] for k in ('f1', 'f2', 'g1', 'g2', 'g3', 'g4', 'g5')) and set(s) == set(ref)
    assert all(math.isfinite(s[g]) for g in ('g1', 'g2', 'g3', 'g4', 'g5'))
    # the pass side of the same check: inside the triangle the candidate runs
    calls.clear()
    monkeypatch.setattr(s4, 'safe_run_gate', lambda *a, **k: (calls.append(a), (R(), None))[1])
    s = ev.candidate(dict(G_ADRC, omega_c_xy=103.9, omega_o_xy=48.0), 1, 2, 0)
    assert s['status'] == 'ok' and len(calls) == 6 and len(ev.evals) == 6


def test_run_seed_with_candidates_outside_the_triangle(monkeypatch):
    """S04 v6.8 section 2: through run_seed an outside-triangle candidate writes a history.csv row
    with status ngoai_bang and no evals.csv row, is never feasible, and is in neither part of the
    seed error rate (CC1-3)."""
    monkeypatch.setattr(s4, 'safe_run_gate', lambda *a, **k: (R(), None))
    ev = s4.Evaluator(None, 'x.dll', 'ADRC', [0.2, 0.5], [0.4, 0.6], 'w')
    hist, hv = [], []
    pts, _, _ = s4.run_seed(ev, DOM_ADRC7, 1, 12, 2, (s4.D, s4.HV_REF_F2), hist, hv, log=lambda *_: None)
    out = [h for h in hist if h['status'] == 'ngoai_bang']
    ok = [h for h in hist if h['status'] != 'ngoai_bang']
    for h in hist:
        g = {d['name']: h[d['name']] for d in DOM_ADRC7}
        assert (h['status'] == 'ngoai_bang') == bool(s4.gain_check('ADRC', g)[1])
    assert out and ok
    assert all(h['f1'] == s4.PENALTY and h['g1'] == 4.0 for h in out)         # 2 mu x 2 speeds expected
    assert {(e['gen'], e['cand_id']) for e in ev.evals} == {(h['gen'], h['cand_id']) for h in ok}
    assert s4.ngoai_bang_counts(hist, 'ADRC') == {1: len(out)}
    r = s4.seed_error_rates(ev.evals, 'ADRC')[1]
    assert r['n'] == 4 * len(ok) and r['rate'] == 0.0
    assert all(not p['feasible'] for p in pts if p['f1'] == s4.PENALTY)


def test_seed_error_rate_ignores_ngoai_bang_rows():
    """S04 v6.8 section 2 (CC1-3): a ngoai_bang row is in neither the numerator nor the denominator."""
    ev = [dict(controller='PID', ga_seed=1, attempt=a, completed=c, status=st)
          for a, c, st in [(1, 'true', 'ok')] * 9 + [(1, '', 'run_error'), (2, 'true', 'ok'),
                                                     (1, 'inconclusive', 'track_err')]]
    ev += [dict(controller='PID', ga_seed=1, attempt=1, completed='', status='ngoai_bang')] * 5
    r = s4.seed_error_rates(ev, 'PID')[1]
    assert r['n'] == 11 and r['rate'] == pytest.approx(2 / 11)
    hist = [dict(controller='ADRC', ga_seed=1, status='ok'),
            dict(controller='ADRC', ga_seed=1, status='ngoai_bang'),
            dict(controller='ADRC', ga_seed=2, status='ok'),
            dict(controller='PID', ga_seed=1, status='ngoai_bang')]
    assert s4.ngoai_bang_counts(hist, 'ADRC') == {1: 1, 2: 0}


def test_seed_error_rate_reports_a_seed_without_runs():
    """C3-10: a seed of the locked list with no first-attempt run (every candidate ngoai_bang, not
    run) is reported with n = 0 and rate None (JSON null), not left out."""
    ev = [dict(controller='ADRC', ga_seed=1, attempt=1, completed='true', status='ok')] * 10
    ev += [dict(controller='ADRC', ga_seed=1, attempt=1, completed='', status='run_error'),
           dict(controller='ADRC', ga_seed=1, attempt=2, completed='true', status='ok')]
    ev += [dict(controller='ADRC', ga_seed=3, attempt=1, completed='', status='ngoai_bang')] * 4
    r = s4.seed_error_rates(ev, 'ADRC', seeds=[1, 2, 3])
    assert list(r) == [1, 2, 3]
    assert r[2] == r[3] == dict(rate=None, n=0, over_10pct=False)
    assert r[1]['n'] == 11 and r[1]['rate'] == pytest.approx(1 / 11)
    assert json.loads(json.dumps(r))['2']['rate'] is None
    assert list(s4.seed_error_rates(ev, 'ADRC', seeds=[2])) == [1, 2]   # a seed in evals is still reported
    assert set(s4.seed_error_rates(ev, 'ADRC')) == {1}                  # without seeds: as before


def test_seed_with_only_ngoai_bang_candidates_is_reported(monkeypatch):
    """C3-10: a seed whose candidates are all outside the LADRC triangle runs nothing (no evals.csv
    row, ngoai_bang in history); seed_error_rates(evals, ctrl, [seed]) still reports it, n = 0."""
    calls = []
    monkeypatch.setattr(s4, 'safe_run_gate', lambda *a, **k: calls.append(a))
    ev = s4.Evaluator(None, 'x.dll', 'ADRC', [0.2, 0.5], [0.4, 0.6], 'w')
    dom = [dict(d) for d in DOM_ADRC7]
    dom[0]['L'], dom[0]['H'] = 150.0, 190.0                    # omega_c_xy: every candidate has xy sum >= 230
    dom[1]['L'], dom[1]['H'] = 40.0, 48.0                      # omega_o_xy
    hist, hv = [], []
    s4.run_seed(ev, dom, 1, 8, 2, (s4.D, s4.HV_REF_F2), hist, hv, log=lambda *_: None)
    assert calls == [] and ev.evals == []
    assert hist and all(h['status'] == 'ngoai_bang' for h in hist)
    assert s4.ngoai_bang_counts(hist, 'ADRC') == {1: len(hist)}
    assert s4.seed_error_rates(ev.evals, 'ADRC') == {}
    assert s4.seed_error_rates(ev.evals, 'ADRC', [1]) == {1: dict(rate=None, n=0, over_10pct=False)}


@pytest.mark.parametrize('bad', [None, '', float('nan'), float('inf'), 'x'])
def test_tv_duty_missing_or_not_finite_is_a_run_error_row(monkeypatch, bad):
    """S04 v6.7 section 3 (9.7): a run that is not a run_error but whose tv_duty is missing or not
    finite is recorded with status run_error at the run level (the other fields kept), before the
    diagnosis rerun decision, and then treated as any run_error."""
    bad_row = R()
    if bad is None:
        del bad_row['tv_duty']
    else:
        bad_row['tv_duty'] = bad
    calls = iter([(R(), None), (bad_row, None)])
    monkeypatch.setattr(s4, 'safe_run_gate', lambda *a, **k: next(calls))
    monkeypatch.setattr(s3, 'run_in_new_process', lambda dll, jobs, work: [dict(row=R(), digest=None)])
    ev = s4.Evaluator(None, 'x.dll', 'PID', [0.2, 0.5], [0.5], 'w')
    s = ev.candidate(G_PID, 1, 1, 0)
    assert [(e['v'], e['attempt'], e['status']) for e in ev.evals] == [(0.2, 1, 'ok'), (0.5, 1, 'run_error'),
                                                                       (0.5, 2, 'ok')]   # diagnosis ran
    conv = ev.evals[1]
    assert 'tv_duty' in conv['error'] and conv['completed'] == 'true'
    assert all(conv['pen_' + k] == 1 for k in s4.PEN_KEYS)
    assert (s['status'], s['f1'], s['f2'], s['g1']) == ('run_error', s4.PENALTY, s4.PENALTY, 2.0)
    r = s4.seed_error_rates(ev.evals, 'PID')[1]
    assert r['n'] == 2 and r['rate'] == 0.5


def test_score_counts_a_missing_tv_duty_as_run_error():
    """S04 v6.7 section 3: score() counts a missing / not finite tv_duty as a run_error also when the
    caller did not convert the row."""
    row = R()
    del row['tv_duty']
    s = s4.score([(0.5, 0.2, R()), (0.5, 0.5, row)], False)
    assert (s['status'], s['f1'], s['f2'], s['g1']) == ('run_error', s4.PENALTY, s4.PENALTY, 2.0)
    s = s4.score([(0.5, 0.2, R()), (0.5, 0.5, R(tv=float('nan')))], False)
    assert (s['status'], s['f1'], s['f2'], s['g1']) == ('run_error', s4.PENALTY, s4.PENALTY, 2.0)


@pytest.mark.parametrize('bad', [None, '', float('nan'), float('inf'), float('-inf'), 'x'])
def test_score_fallback_uses_the_converted_row(bad):
    """C3-9: score() converts every row itself (tv_duty_run_error): the SAME converted row gives the
    score, the penalty flags and the counts, also when the caller did not convert (the fallback path)."""
    bad_row = R()
    if bad is None:
        del bad_row['tv_duty']
    else:
        bad_row['tv_duty'] = bad
    raw = [(0.5, 0.2, R()), (0.5, 0.5, bad_row)]
    s = s4.score(raw, False)
    ref = s4.score([(mu, v, s4.tv_duty_run_error(r)) for mu, v, r in raw], True)
    assert s == ref
    assert (s['status'], s['f1'], s['f2'], s['g1']) == ('run_error', s4.PENALTY, s4.PENALTY, 2.0)
    assert s['n_run_error'] == 1 and s['n_pen_runs'] == 1 and s['pen_f1'] == 1   # old code: 0, 0
    assert s['g3'] == s['g4'] == (s4.PENALTY - s4.D) / s4.D
    assert bad_row.get('status') == 'ok'                                       # the input row not mutated


@pytest.mark.parametrize('bad', ['', float('nan'), float('inf'), float('-inf')])
def test_tv_duty_error_then_successful_diagnosis_keeps_the_penalty(monkeypatch, bad):
    """S04 v6.7 section 3 / C3-9: a run whose tv_duty is missing or not finite is a run_error at the
    run level; the diagnosis rerun may succeed, but the candidate keeps the penalty, and the fallback
    path of score (the row not converted) agrees with the converted main path."""
    bad_row = R(tv=bad)
    calls = iter([(R(), None), (bad_row, None)])
    monkeypatch.setattr(s4, 'safe_run_gate', lambda *a, **k: next(calls))
    monkeypatch.setattr(s3, 'run_in_new_process', lambda dll, jobs, work: [dict(row=R(), digest=None)])
    ev = s4.Evaluator(None, 'x.dll', 'PID', [0.2, 0.5], [0.5], 'w')
    s = ev.candidate(G_PID, 1, 1, 0)
    assert [(e['v'], e['attempt'], e['status']) for e in ev.evals] == [(0.2, 1, 'ok'), (0.5, 1, 'run_error'),
                                                                       (0.5, 2, 'ok')]
    assert (s['status'], s['n_run_error'], s['n_pen_runs'], s['pen_f1'], s['g1']) == \
        ('run_error', 1, 1, 1, 2.0)
    # the main path (candidate converts the row) and the fallback path (score converts it) agree
    assert s == s4.score([(0.5, 0.2, R()), (0.5, 0.5, bad_row)], False)


def test_hypervolume_reference_d_800():
    """S04 v6.8 section 4.2: the reference is fixed at (D; 800 1/s); a point on or beyond it adds 0
    and is counted in n_nd_on_or_beyond_ref."""
    assert s4.HV_REF_F2 == 800.0
    ref = (s4.D, s4.HV_REF_F2)
    assert s4.hypervolume([P(s4.D, 1.0)], ref) == (0.0, 1)                 # f1 = D exactly adds 0, counted
    assert s4.hypervolume([P(0.05, 800.0)], ref) == (0.0, 1)
    hv, out = s4.hypervolume([P(0.05, 800.0 - 1e-9)], ref)
    assert out == 0 and hv > 0
    hv, out = s4.hypervolume([P(s4.D - 1e-12, 5.0)], ref)
    assert out == 0 and hv > 0
    hv, out = s4.hypervolume([P(0.02, 100.0), P(s4.D, 50.0), P(0.01, 800.0)], ref)
    assert hv == pytest.approx((s4.D - 0.02) * 700.0) and out == 2


def test_mu_rule_reads_the_gate_state_columns(tmp_path):
    """S04 v6.7 9.7: the mu rule compares the gate state columns g3_state .. g5_state of
    sealed/runs.csv; a state that is the same at the three mu satisfies the rule whatever it is; a
    non-run_error row without a state column stops the script. S04 v6.8: no S03 y3_reference read."""
    for name, st in (('a', dict(g4_state='khong_dat')), ('b', dict(g5_state='chua_ket_luan'))):
        def mod(r, st=st):
            if r['point_id'] == 1:
                r.update(st)
        d = _s03_dir(tmp_path / name, _runs(mod))
        dec, _ = s4.mu_rule(str(d), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])
        assert dec == 'mid', name                                   # the same state at the three mu passes

    def mod(r):                                                     # (c) no g5_state column at all
        r.pop('g5_state')
    d = _s03_dir(tmp_path / 'c', _runs(mod))
    with pytest.raises(SystemExit, match='g5_state'):
        s4.mu_rule(str(d), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])

    def mod(r):                                                     # (d) a state text that is not a state
        if r['point_id'] == 1 and r['mu'] == 0.5:
            r.update(g3_state='x')
    d = _s03_dir(tmp_path / 'd', _runs(mod))
    with pytest.raises(SystemExit, match='g3_state'):
        s4.mu_rule(str(d), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])
    d = _s03_dir(tmp_path / 'e', _runs())                           # (e) meta.yaml without y3_reference
    mt = d / 'meta.yaml'
    mt.write_text('\n'.join(l for l in mt.read_text(encoding='utf-8').splitlines()
                            if not l.startswith('y3_reference:')) + '\n', encoding='utf-8')
    dec, _ = s4.mu_rule(str(d), 'PID', list(s3.SPEEDS), [0.4, 0.5, 0.6])
    assert dec == 'mid'


def test_judge_reads_the_gate_state_columns():
    """S04 v6.2 section 6 (v6.7 9.7): a change of the G3, G4 or G5 gate state against factor 1.0
    makes the sensitivity row 'sensitive'."""
    def J(k, **kw):
        return dict(dict(v=0.2, mu=0.5, r_factor=k, completed='true', e_perp_max=0.03,
                         g3_state='dat', g4_state='dat', g5_state='dat'), **kw)
    assert s4.judge([J(1.0), J(0.8), J(1.2)], 'r_factor') == (False, [])
    for c, st in (('g3', 'khong_dat'), ('g4', 'chua_ket_luan'), ('g5', 'khong_dat')):
        sens, why = s4.judge([J(1.0), J(0.8, **{f'{c}_state': st})], 'r_factor')
        assert sens is True and any(f'{c.upper()} state dat -> {st}' in w for w in why)
    row = J(0.8)
    del row['g5_state']
    with pytest.raises(KeyError):
        s4.judge([J(1.0), row], 'r_factor')


def test_three_mu_findings_read_states_not_maxima():
    """S04 v6.2 section 6 (v6.7 9.7): a three-mu run passes only with status ok and the five gate
    states dat, read from the state columns — a maximum above its threshold with states dat passes,
    a state not dat with every maximum below does not."""
    def T(**kw):
        return dict(dict(controller='PID', v=0.2, mu=0.4, status='ok', completed='true', e_perp_max=0.03,
                         e_trk_max=0.03, e_bao_max=0.03, e_theta_cmd_max_deg=2.0, g1_state='dat',
                         g2_state='dat', g3_state='dat', g4_state='dat', g5_state='dat'), **kw)
    assert s4.three_mu_findings([T()], 'PID') == []
    f = s4.three_mu_findings([T(g3_state='chua_ket_luan')], 'PID')
    assert len(f) == 1 and 'G3 chua_ket_luan' in f[0]
    assert s4.three_mu_findings([T(e_perp_max=0.2)], 'PID') == []    # the state column decides, not the max
    row = T()
    del row['g5_state']
    f = s4.three_mu_findings([row], 'PID')
    assert len(f) == 1 and 'G5 missing' in f[0]
    assert len(s4.three_mu_findings([T(status='run_error')], 'PID')) == 1
    assert s4.three_mu_findings([dict(T(), controller='ADRC')], 'PID') == []


def _fake_post(tmp_path, monkeypatch, gains=G_PID, ctrl='PID', eval_mu=(0.5,), levels=(0.4, 0.5, 0.6),
               tool_check=False, mu3_mod=None, sens_mod=None):
    """The real s4.post without a DLL: GA outputs (lock.yaml, selected.csv with one send_to_HIL row,
    evals.csv attempt-1 rows) and the S03 meta.yaml check_s03_trial reads are written; every sim call is
    stubbed; write_root_meta and log_phase run for real. mu3_mod(row, v, mu) modifies a three-mu run row;
    sens_mod(row, kind, factor, v, mu) modifies a b0/R run row. Returns (return code, notes.json, out,
    selected.csv bytes before post)."""
    out = tmp_path / 'run'
    out.mkdir(parents=True)
    dll = tmp_path / 'fake.dll'
    dll.write_bytes(b'fake dll')
    s03 = tmp_path / 's03'
    s03.mkdir(exist_ok=True)
    (s03 / 'meta.yaml').write_text(f'trial: {str(tool_check).lower()}\n', encoding='utf-8')
    speeds = [0.2, 0.5]
    lock = dict(common=dict(tool_check=tool_check, s03_run=str(s03), fingerprint={},
                            unsealed=not tool_check),
                s03_mu_levels=list(levels),
                controllers={ctrl: dict(eval_mu=list(eval_mu), speeds=speeds)})
    (out / 'lock.yaml').write_text(json.dumps(lock), encoding='utf-8')
    s3.write_csv(str(out / 'selected.csv'),
                 [dict(kind='send_to_HIL', controller=ctrl, ga_seed=1, feasible='true', gen=1, cand_id=0,
                       f1=0.03, f2=5.0, **{n: repr(float(v)) for n, v in gains.items()})])
    s3.write_csv(str(out / 'evals.csv'),
                 [dict(controller=ctrl, ga_seed=1, gen=1, cand_id=0, v=v, mu=mu, attempt=1, log_sha256='')
                  for mu in eval_mu for v in speeds])

    def row(**kw):
        return dict(dict(status='ok', completed='true', e_perp_max=0.03, tv_duty=5.0, e_trk_max=0.03,
                         e_bao_max=0.03, e_theta_cmd_max_deg=2.0,
                         **{f'{c}_state': 'dat' for c in s4.og.CONDS}), **kw)

    def fake_safe(fw, path, c, g, v, mu):
        r = row()
        if mu3_mod:
            mu3_mod(r, v, mu)
        return r, None

    def fake_custom(fw, path, c, g, v, mu, **kw):
        r = row()
        if sens_mod:
            for kind in ('b0_factor', 'r_factor'):
                if kind in kw:
                    sens_mod(r, kind, kw[kind], v, mu)
        return r, None, {}, dict(motor_R=1.0), dict(motor_R=1.0)

    monkeypatch.setattr(s4, 'fingerprint', lambda *a, **k: {})
    monkeypatch.setattr(fwsim, 'FirmwareSim', lambda dll_: None)
    monkeypatch.setattr(s3, 'run_in_new_process',
                        lambda dll_, jobs, work: [dict(row=row(), digest=None) for _ in jobs])
    monkeypatch.setattr(s4, 'run_custom', fake_custom)
    monkeypatch.setattr(s3, 'safe_run', fake_safe)
    sel = (out / 'selected.csv').read_bytes()
    rc = s4.post(argparse.Namespace(out=str(out), dll=str(dll)))
    notes = json.load(open(out / 'post_checks' / 'notes.json', encoding='utf-8'))
    return rc, notes, out, sel


@pytest.mark.parametrize('case', 'abcdefg')
def test_post_three_mu_findings_wait_for_liam(tmp_path, monkeypatch, case):
    """C3-8: a run of the three-mu check not passing G1-G5 (three mu levels, check done) is listed in
    post_checks/notes.json cho_liam with the pre-HIL failures, and post returns 2 (no reselection:
    the selected gains are kept, never chosen again here)."""
    if case == 'a':
        rc, notes, out, sel = _fake_post(
            tmp_path / 'a', monkeypatch,
            mu3_mod=lambda r, v, mu: r.update(g4_state='khong_dat') if (mu, v) == (0.4, 0.2) else None)
        assert rc == 2                                                  # Codex C3-8 (old code: 0)
        assert notes['cho_liam'] == [dict(controller='PID', check='three_mu',
                                        reasons=notes['notes']['PID three_mu']['findings'])]
        assert len(notes['cho_liam'][0]['reasons']) == 1
        assert 'mu=0.4' in notes['cho_liam'][0]['reasons'][0]
        assert 'G4 khong_dat' in notes['cho_liam'][0]['reasons'][0]
        assert notes['notes']['PID three_mu']['done'] is True
        assert notes['notes']['PID pre_hil']['ok'] is True
        assert 'cho_liam_note' in notes
        assert (out / 'selected.csv').read_bytes() == sel               # no reselection
    elif case == 'b':
        rc, notes, _, _ = _fake_post(tmp_path / 'b', monkeypatch)
        assert rc == 0 and notes['cho_liam'] == []
    elif case == 'c':
        rc, notes, _, _ = _fake_post(
            tmp_path / 'c', monkeypatch,
            mu3_mod=lambda r, v, mu: r.update(g5_state='chua_ket_luan') if mu == 0.6 else None)
        assert rc == 2
        cl = [c for c in notes['cho_liam'] if c['check'] == 'three_mu']
        assert len(cl) == 1 and cl[0]['reasons']
    elif case == 'd':
        rc, notes, _, _ = _fake_post(
            tmp_path / 'd', monkeypatch,
            mu3_mod=lambda r, v, mu: r.update(status='track_err') if (mu, v) == (0.6, 0.5) else None)
        assert rc == 2
        assert any(c['check'] == 'three_mu' for c in notes['cho_liam'])
    elif case == 'e':
        # the GA already evaluated three mu: no three-mu check at all
        rc, notes, _, _ = _fake_post(
            tmp_path / 'e', monkeypatch, eval_mu=(0.4, 0.5, 0.6),
            mu3_mod=lambda r, v, mu: r.update(g4_state='khong_dat') if (mu, v) == (0.4, 0.2) else None)
        assert rc == 0 and notes['cho_liam'] == []
        assert 'PID three_mu' not in notes['notes']
    elif case == 'f':
        # a tool check on an S03 run with one mu level: the check was NOT done (unchanged path)
        rc, notes, _, _ = _fake_post(
            tmp_path / 'f', monkeypatch, levels=(0.5,), tool_check=True,
            mu3_mod=lambda r, v, mu: r.update(g4_state='khong_dat'))
        assert rc == 0 and notes['cho_liam'] == []
        assert notes['notes']['PID three_mu']['done'] is False
    else:  # g: the selected gains outside the range table (kp_wz 5.0 > the ceiling 4.6)
        rc, notes, _, _ = _fake_post(tmp_path / 'g', monkeypatch, gains=dict(G_PID, kp_wz=5.0))
        assert rc == 2                                                  # as before this change
        cl = notes['cho_liam']
        assert len(cl) == 1 and cl[0]['check'] == 'pre_hil' and cl[0]['reasons']


def test_pre_hil_check():
    """S04 v6.8 section 6: range table and (LADRC) triangle on the float32 values before HIL, plus the
    spectral radius of the linear sub-model (tools/mo_hinh_con.py, report only) with the kept states."""
    mh = s4.mo_hinh_con
    ph = s4.pre_hil_check('PID', G_PID)
    assert ph['ok'] is True and ph['in_table'] is True and ph['in_triangle'] is None
    assert ph['spectral_radius'] == mh.ban_kinh_pho('PID', G_PID) and ph['spectral_radius_state'] == 'on_dinh'
    assert ph['spectral_radius_states'] == list(mh.TEN['PID']) and ph['spectral_radius_dropped'] == []
    m = ph['spectral_radius_model']
    assert (m['G'], m['tau'], m['T'], m['b0'], m['module']) == (mh.G_MAC_DINH, mh.TAU, mh.T_TICK, None,
                                                              'tools/mo_hinh_con.py')
    assert m['sha256'] == s4.s3.sha256_file(os.path.join(s4._TOOLS, 'mo_hinh_con.py'))
    assert 'report only' in ph['spectral_radius_note']
    ph = s4.pre_hil_check('PID', dict(G_PID, pos_kp=0.0, pos_ki=3.0))
    assert ph['ok'] is True and ph['spectral_radius_state'] == 'khong_on_dinh'  # report only: ok unchanged
    ph = s4.pre_hil_check('PID', dict(G_PID, kp_wz=5.0))
    assert ph['ok'] is False and ph['in_table'] is False and ph['table_violations']
    ph = s4.pre_hil_check('ADRC', dict(G_ADRC, omega_c_xy=150.0, omega_o_xy=48.0))
    assert ph['ok'] is False and ph['in_table'] is True and ph['in_triangle'] is False
    ph = s4.pre_hil_check('ADRC', G_ADRC)
    assert ph['ok'] is True and ph['in_triangle'] is True
    assert ph['spectral_radius'] == mh.ban_kinh_pho('ADRC', G_ADRC)
    assert ph['spectral_radius_model']['b0'] == mh.B0_XY
    assert s4.submodel_spectral_radius('PID', G_PID) == mh.ban_kinh_pho('PID', G_PID)
    assert s4.submodel_spectral_radius('PID', {k: v for k, v in G_PID.items() if k != 'k_psi'}) is None


def test_delay_steps():
    """S04 v6.8 section 6: the io_delay_ms values of the loop-delay variant and their rule text."""
    assert s4.delay_steps(None)[0] == [1, 2]
    assert s4.delay_steps(2.0)[0] == [2]
    assert s4.delay_steps(1.9999999995)[0] == [2]                    # within 1e-9 ms of a whole number
    assert s4.delay_steps(2.0000000005)[0] == [2]
    assert s4.delay_steps(1.4)[0] == [1, 2]                          # rounding down and up
    assert s4.delay_steps(0.3)[0] == [0, 1]
    assert s4.delay_steps(2.000001)[0] == [2, 3]
    assert s4.delay_steps(0)[0] == [0]
    for bad in (float('nan'), float('inf'), -1.0):
        with pytest.raises(SystemExit, match='io-delay-ms'):
            s4.delay_steps(bad)
    for x in (None, 2.0, 1.4, 0.3, 0):
        steps, rule = s4.delay_steps(x)
        assert steps and isinstance(rule, str) and rule


@pytest.mark.parametrize('ph', ('trial', 'main', 'post'))
def test_io_delay_only_with_sens_after(tmp_path, monkeypatch, ph):
    """S04 v6.8 section 6: --io-delay-ms belongs to --phase sens_after only."""
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', ph, '--out', str(tmp_path / 'o'),
                                      '--io-delay-ms', '1.4'])
    with pytest.raises(SystemExit, match='io-delay-ms only with --phase sens_after'):
        s4.main()


def test_run_gate_stops_on_another_slip_omega_s(monkeypatch):
    """S04 v6.8 section 6 / S03 v8.6 section 8 condition 12: the GA condition is slip_omega_s 0.1
    and io_delay_ms 0; another plant slip_omega_s stops as a tool error, also through safe_run_gate."""
    assert (s4.GA_IO_DELAY_MS, s4.GA_SLIP_OMEGA_S) == (0, 0.1)
    monkeypatch.setattr(s4.fwsim, 'plant_params',
                        lambda a: types.SimpleNamespace(slip_omega_s=0.2, lx=0.1, ly=0.1))
    with pytest.raises(s3.S03Stop, match='slip_omega_s'):
        s4.run_gate(None, None, 'PID', G_PID, 0.2, 0.5)
    with pytest.raises(s3.S03Stop, match='slip_omega_s'):
        s4.safe_run_gate(None, None, 'PID', G_PID, 0.2, 0.5)


@needs_dll
def test_run_gate_runs_the_ga_condition(monkeypatch):
    """S04 v6.8 section 6: run_gate simulates with io_delay_ms = 0 and slip_omega_s = 0.1 (the GA
    condition), explicit in the fwsim.simulate call."""
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    import o1_feasibility as o1
    seen = []
    real = s4.fwsim.simulate

    def spy(fw_, sp, dur, params=None, **kw):
        seen.append((kw.get('io_delay_ms'), params.slip_omega_s))
        return real(fw_, sp, dur, params=params, **kw)
    monkeypatch.setattr(s4.fwsim, 'simulate', spy)
    s4.run_gate(fw, o1.Path8(), 'PID', s3.defaults(fw, 'PID'), 0.2, 0.436)
    assert seen == [(0, 0.1)]


@needs_dll
def test_run_custom_gate_states_and_io_delay():
    """S04 v6.7 9.7 / v6.8 6: run_custom io_delay_ms = 0 gives the same log as s03_doe.run_one and
    the gate state columns of o1_gate.gate; io_delay_ms = 2 gives another log with valid states."""
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    import o1_feasibility as o1
    path, g = o1.Path8(), s3.defaults(fw, 'PID')
    a, la = s3.run_one(fw, path, 'PID', g, 0.2, 0.436)
    res, log, back, before, after = s4.run_custom(fw, path, 'PID', g, 0.2, 0.436)
    assert s3.log_digest(log) == s3.log_digest(la)
    assert all(res[f'{c}_state'] == a[f'{c}_state'] for c in s4.og.CONDS)
    assert (res['e_perp_max'], res['tv_duty'], res['e_bao_max']) == (a['e_perp_max'], a['tv_duty'],
                                                                   a['e_bao_max'])
    assert before == after                                          # no plant variant asked for
    res2, log2, *_ = s4.run_custom(fw, path, 'PID', g, 0.2, 0.436, io_delay_ms=2)
    assert s3.log_digest(log2) != s3.log_digest(log)
    assert all(res2[f'{c}_state'] in s4.og.STATES for c in s4.og.CONDS)


# ── S04 v6.7 section 8 / 9.6, v6.8 (sealed inputs, classify, kep, spectral radius, plant) ──

def test_v67_read_domain_reads_the_sealed_levels_only(tmp_path):
    """S04 v6.7 9.6: the domain comes from sealed/levels_sangloc.csv only; the public levels.csv is
    ignored (other numbers) and may be absent; no sealed file -> sys.exit (an S03 run before v8.5)."""
    d = _s03_dir(tmp_path)
    dom = s4.read_domain(str(d), 'PID')
    with open(d / 'levels.csv', 'w', newline='') as f:                 # a public file with other numbers
        w = csv.writer(f)
        w.writerow(['controller', 'factor', 'L', 'H', 'centre', 'scale', 'default', 'rule_note'])
        for c in s3.CONTROLLERS:
            for n in s3.FACTORS[c]:
                w.writerow([c, n, 9.0, 99.0, '', 'lin', '', ''])
    assert s4.read_domain(str(d), 'PID') == dom                        # the sealed values, not 9..99
    os.remove(str(d / 'levels.csv'))
    assert s4.read_domain(str(d), 'PID') == dom                        # no public file at all
    d2 = _s03_dir(tmp_path / 'm', sealed_levels=False)
    with pytest.raises(SystemExit, match='sealed/levels_sangloc.csv'):
        s4.read_domain(str(d2), 'PID')
    with pytest.raises(SystemExit, match='sealed/levels_sangloc.csv'):
        s4.domain_source(str(d2), 'PID')


@pytest.mark.parametrize('upd', [dict(nguon_mien=''), dict(nguon_mien='khac')])
def test_v67_nguon_mien_must_be_one_known_value(tmp_path, upd):
    """S04 v6.7 9.6: nguon_mien empty or unknown stops; a mixed domain stops too."""
    d = _s03_dir(tmp_path)
    _edit_levels(d, **upd)
    for f in (s4.read_domain, s4.domain_source):
        with pytest.raises(SystemExit, match='nguon_mien'):
            f(str(d), 'PID')
    d2 = _s03_dir(tmp_path / 'x')
    _edit_levels(d2, mod=lambda r: r.update(nguon_mien='du_phong')
                 if (r['controller'], r['factor']) == ('PID', 'kp_xy') else None)
    with pytest.raises(SystemExit, match='nguon_mien'):
        s4.read_domain(str(d2), 'PID')


def test_v67_domain_source(tmp_path):
    """domain_source: the identical nguon_mien and the unique non-empty ly_do joined by ' | '."""
    d = _s03_dir(tmp_path)
    assert s4.domain_source(str(d), 'PID') == dict(nguon_mien='sang_loc', ly_do='')
    d = _s03_dir(tmp_path / 'a', ly_do='sang loc 1')
    assert s4.domain_source(str(d), 'ADRC') == dict(nguon_mien='sang_loc', ly_do='sang loc 1')
    _edit_levels(d, mod=lambda r: r.update(ly_do='a' if r['factor'] == 'kp_xy' else
                                           ('b' if r['factor'] == 'ki_xy' else '')))
    assert s4.domain_source(str(d), 'PID') == dict(nguon_mien='sang_loc', ly_do='a | b')


def _tran_levels(tmp_path, ctrl='PID', name='pos_kp', b=None, bn=None, h=None,
                 h_nguon='trần bảng chặn'):
    """du_phong levels with <name> at H = T (the table ceiling), b_n = H, b within TRAN_RTOL of T."""
    T = s3.ceiling(ctrl, name)
    if b is None:
        b = repr(math.nextafter(T, 0.0))
    d = _s03_dir(tmp_path, nguon_mien='du_phong')
    _edit_levels(d, mod=lambda r: r.update(H=repr(h if h is not None else T), b=b,
                                           b_n=repr(bn if bn is not None else T),
                                           H_nguon=h_nguon)
                 if (r['controller'], r['factor']) == (ctrl, name) else None)
    return d, T


def _sweep(d, ctrl, name, rows):
    """A public sweep.csv: rows = (value, v, tran)."""
    with open(d / 'sweep.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['controller', 'factor', 'value', 'v', 'tran'])
        for val, v, tr in rows:
            w.writerow([ctrl, name, val, v, tr])


@pytest.mark.parametrize('ctrl, name', [('PID', 'pos_kp'), ('ADRC', 'omega_c_xy')])
@pytest.mark.parametrize('b', ['199.9999999', None])
def test_v67_du_phong_table_ceiling_extension_accepted(tmp_path, ctrl, name, b):
    """C3-7 / C3-26 (S03 v8.6 3.1 rule (2)): du_phong with H = b_n = T, b within TRAN_RTOL of T
    (the exact S03 output b = 199.9999999, b_n = H = T = 200.0, and the nextafter boundary),
    H_nguon 'trần bảng chặn' and every sweep row at H 'tran' is accepted; a feasible point at
    the H edge makes the send_to_HIL row of summarise() carry EDGE_LIMIT_TABLE as '<name>:H'."""
    d, T = _tran_levels(tmp_path, ctrl=ctrl, name=name, b=b)
    assert T == 200.0                                            # b is the exact S03 number
    _sweep(d, ctrl, name, [(repr(T), v, 'true') for v in s3.SPEEDS] + [('', 0.2, 'false')])
    dom = s4.read_domain(str(d), ctrl)
    assert {x['name']: x for x in dom}[name]['H'] == T
    lo, hi = s4.ga_bounds(dom)
    rep = s4.edge_report(dom, hi)
    assert f'{name}:H' in rep['edge_limit'] and s4.EDGE_LIMIT_TABLE in rep['edge_limit']
    assert f'{name}:H:{s4.TABLE_CEILING_SOURCE}' in rep['edge_src']
    rows, _ = s4.summarise(ctrl, dom, {1: [P(0.02, 5.0, x=list(hi))]})
    send = [r for r in rows if r['kind'] == 'send_to_HIL'][0]
    assert send['feasible'] and f'{name}:H' in send['edge_limit']
    assert s4.EDGE_LIMIT_TABLE in send['edge_limit']


def test_v67_du_phong_table_ceiling_extension_refused(tmp_path):
    """C3-7 / C3-26: every failing condition of the exception falls back to the v6.6 refusal."""
    d, T = _tran_levels(tmp_path)
    with pytest.raises(SystemExit, match='extension'):                        # no sweep.csv
        s4.read_domain(str(d), 'PID')
    _sweep(d, 'PID', 'pos_kp', [(repr(T), v, 'true') for v in s3.SPEEDS[:1]] +
           [(repr(T), s3.SPEEDS[1], 'false')])
    with pytest.raises(SystemExit, match='extension'):                        # one tran false row
        s4.read_domain(str(d), 'PID')
    _sweep(d, 'PID', 'pos_kp', [(repr(T / 2), v, 'true') for v in s3.SPEEDS])
    with pytest.raises(SystemExit, match='extension'):                        # no sweep row at H
        s4.read_domain(str(d), 'PID')
    for i, kw in enumerate((dict(b='199.999999'), dict(bn='250.0'),
                            dict(h=math.nextafter(200.0, math.inf)), dict(h_nguon='hết lưới'))):
        d2, T2 = _tran_levels(tmp_path / f'r{i}', **kw)
        _sweep(d2, 'PID', 'pos_kp', [(repr(T2), v, 'true') for v in s3.SPEEDS])
        with pytest.raises(SystemExit, match='extension'):
            s4.read_domain(str(d2), 'PID')
    # kp_wz at H written through float32, not the binary64 4.6 of the ceiling: H != T -> refused (b
    # within TRAN_RTOL of 4.6, so only the H != T condition fails). The lower float32 neighbour of
    # 4.6 (4.599999904632568) lies below every b within TRAN_RTOL of T (b >= 4.6 - 4.6e-9), so the
    # float32 neighbour ABOVE 4.6 is the one that is on the extension (H > b) and != T.
    h = float(np.nextafter(np.float32(4.6), np.float32(np.inf)))
    assert h > 4.6 * (1 - 5e-10) and h != 4.6
    d3, _ = _tran_levels(tmp_path / 'w', name='kp_wz', b=repr(4.6 * (1 - 5e-10)), bn=h, h=h)
    _sweep(d3, 'PID', 'kp_wz', [(repr(h), v, 'true') for v in s3.SPEEDS])
    with pytest.raises(SystemExit, match='extension'):
        s4.read_domain(str(d3), 'PID')


def test_v67_fingerprint_locks_the_sealed_files_and_the_root_number(tmp_path):
    """S04 v6.7 8: fingerprint hashes meta.yaml, levels.csv, sweep.csv, the four S03_SEALED_LOCK
    files (the root design.csv is NOT hashed), sealed/runs.csv when unsealed, kiem_ngoai_mien.csv,
    plus so_goc; one changed byte in each sealed file and in the number file is caught."""
    d = _s03_dir(tmp_path)
    dll = str(d / 'levels.csv')
    fp = s4.fingerprint(dll, str(d))
    assert set(fp['s03_files']) == {'meta.yaml', 'levels.csv', 'sweep.csv', 'kiem_ngoai_mien.csv'} \
        | set(s4.S03_SEALED_LOCK)
    assert 'design.csv' not in fp['s03_files'] and fp['s03_files']['sweep.csv'] is None
    assert all(fp['s03_files'][n] for n in s4.S03_SEALED_LOCK)
    sg = fp['so_goc']
    assert sg['file'] == str(tmp_path / 'so_goc.txt') \
        and sg['s03_sha256'] == hashlib.sha256(b'so goc\n').hexdigest()
    for n in s4.S03_SEALED_LOCK:
        p = d / n
        orig = p.read_bytes()
        p.write_bytes(orig + b'\n')
        assert s4.fingerprint(dll, str(d))['s03_files'][n] != fp['s03_files'][n]
        p.write_bytes(orig)
    fpu = s4.fingerprint(dll, str(d), unsealed=True)
    assert 'sealed/runs.csv' in fpu['s03_files'] and 'sealed/runs.csv' not in fp['s03_files']
    num = tmp_path / 'so_goc.txt'
    num.write_bytes(b'so goc\r\n')                                    # CRLF form of the recorded bytes
    assert s4.so_goc(str(d))['file'] == sg['file']
    num.write_bytes(b'other number\n')
    with pytest.raises(SystemExit, match='not the number file'):
        s4.so_goc(str(d))
    num.write_bytes(b'so goc\n')
    d2 = _s03_dir(tmp_path / 'n', screen_u=False)
    assert s4.so_goc(str(d2)) is None
    fp2 = s4.fingerprint(dll, str(d2))
    s4.check_s03_sealed(fp2, tool_check=True)                          # tool-check: two sealed files only
    with pytest.raises(SystemExit, match='sealed'):
        s4.check_s03_sealed(fp2, tool_check=False)                     # a main run needs so_goc too
    d3 = _s03_dir(tmp_path / 'd', sealed_design=False)
    with pytest.raises(SystemExit, match='sealed'):
        s4.check_s03_sealed(s4.fingerprint(dll, str(d3)), tool_check=True)


def test_v67_mu_rule_needs_the_sealed_design(tmp_path):
    """S04 v6.7 9.6: a public design.csv alone cannot stand in for sealed/design.csv."""
    d = _s03_dir(tmp_path, _runs(), sealed_design=False)
    with open(d / 'design.csv', 'w', newline='') as f:
        csv.writer(f).writerow(['controller', 'point_id', 'box'])
    fp = s4.fingerprint(str(d / 'levels.csv'), str(d))
    with pytest.raises(SystemExit, match='sealed'):
        s4.check_s03_sealed(fp, tool_check=False)


def _lock_dir(tmp_path, tool_check, s03, unsealed):
    """A run folder with the files post() / sens_after() read before the S03 gate: lock.yaml,
    selected.csv, evals.csv."""
    out = tmp_path
    out.mkdir(parents=True)
    lock = dict(common=dict(tool_check=tool_check, s03_run=str(s03), fingerprint={}, unsealed=unsealed),
                s03_mu_levels=[0.5], controllers=dict(PID=dict(eval_mu=[0.5], speeds=[0.2])))
    (out / 'lock.yaml').write_text(json.dumps(lock), encoding='utf-8')
    s3.write_csv(str(out / 'selected.csv'),
                 [dict(kind='send_to_HIL', controller='PID', feasible='true',
                       **{n: '1.0' for n in s3.FACTORS['PID']})])
    s3.write_csv(str(out / 'evals.csv'), [dict(controller='PID', log_sha256='')])
    return out


def test_v67_gates_open_nothing_under_sealed(tmp_path, monkeypatch):
    """S04 v6.7 8: --tool-check on an S03 run that is not --trial exits before read_domain /
    fingerprint (nothing under sealed/ is opened); post() and sens_after() do the same through the
    lock, and a main lock needs an unsealed S03 run."""
    import builtins
    d = _s03_dir(tmp_path, trial=False)                                # an S03 main run
    opened = []
    real = builtins.open

    def spy(f, *a, **kw):
        opened.append(str(f))
        return real(f, *a, **kw)
    monkeypatch.setattr(builtins, 'open', spy)
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial', '--tool-check', '--controller',
                                      'PID', '--s03', str(d), '--out', str(tmp_path / 'o')])
    with pytest.raises(SystemExit, match='tool-check'):
        s4.main()
    assert not any('/sealed/' in p.replace('\\', '/') for p in opened)
    out = _lock_dir(tmp_path / 'r1', tool_check=True, s03=d, unsealed=False)
    for f in (lambda: s4.post(argparse.Namespace(out=str(out), dll='x.dll')),
              lambda: s4.sens_after(argparse.Namespace(out=str(out), dll='x.dll', io_delay_ms=None))):
        with pytest.raises(SystemExit, match='not an S03 --trial'):
            f()
    assert not any('/sealed/' in p.replace('\\', '/') for p in opened)
    out2 = _lock_dir(tmp_path / 'r2', tool_check=False, s03=d, unsealed=False)
    with pytest.raises(SystemExit, match='unsealed'):
        s4.post(argparse.Namespace(out=str(out2), dll='x.dll'))
    assert not any('/sealed/' in p.replace('\\', '/') for p in opened)


def test_v67_source_diff_covers_this_script_and_the_screening_script(monkeypatch):
    """S04 v6.7 9.6: source_diff compares the working-tree blob of fwsim/s04_ga.py and of
    fwsim/s03_sangloc.py against the S03 commit; a script missing at the commit (blob_at None) is
    reported."""
    real_w, real_b = s4.work_blob, s4.blob_at
    for path in ('fwsim/s04_ga.py', s4.S03_SCREEN_SCRIPT):
        monkeypatch.setattr(s4, 'work_blob',
                            lambda p, path=path: _other_blob(real_w(p) or 'a') if p == path else real_w(p))
        assert [n for n, _, _ in s4.source_diff(_head())] == [path]
        monkeypatch.setattr(s4, 'work_blob', real_w)
        monkeypatch.setattr(s4, 'blob_at', lambda c, p, path=path: None if p == path else real_b(c, p))
        assert [n for n, _, _ in s4.source_diff(_head())] == [path]     # missing at the S03 commit
        monkeypatch.setattr(s4, 'blob_at', real_b)


def test_kep_log_hash_uses_only_the_bit_exact_keys(monkeypatch):
    """S03 v8.6 11.6: digest_sha hashes fwsim.BIT_EXACT_KEYS only (adding/removing 'kep' changes
    nothing; the old formula for a digest without it); an unknown key is a ValueError."""
    dig = {k: [0.0, 1.0] for k in fwsim.BIT_EXACT_KEYS}
    old = hashlib.sha256(json.dumps(dig, sort_keys=True).encode()).hexdigest()
    assert s4.digest_sha(dig) == old
    assert s4.digest_sha(dict(dig, kep=[1, 2, 3])) == old            # the channel is not compared
    assert s4.digest_sha(dict(dig, kep=[])) == old
    with pytest.raises(ValueError, match='BIT_EXACT_KEYS'):
        s4.digest_sha(dict(dig, other=[0.0]))
    assert s4.log_sha(None) == ''
    monkeypatch.setattr(s3, 'log_digest', lambda log: dict(dig, **({'kep': [1]} if log == 'with' else {})))
    assert s4.log_sha('with') == s4.log_sha('plain') == old


def test_kep_fields_report_only(monkeypatch):
    """kep_fields: ADRC all ''; PID the fwsim.kep_indicator values; a ValueError gives '' fractions
    and the message, never a change of status or score."""
    assert s4.kep_fields('ADRC', 'log', 1.0) == dict(kep_frac_vx='', kep_frac_vy='', kep_frac_wz='',
                                                    kep_note='')
    monkeypatch.setattr(fwsim, 'kep_indicator',
                        lambda log, t: dict(kep_frac_vx=0.1, kep_frac_vy=0.2, kep_frac_wz=0.3))
    assert s4.kep_fields('PID', 'log', 1.0) == dict(kep_frac_vx=0.1, kep_frac_vy=0.2, kep_frac_wz=0.3,
                                                   kep_note='')
    monkeypatch.setattr(fwsim, 'kep_indicator',
                        lambda log, t: (_ for _ in ()).throw(ValueError('no kep channel in log')))
    k = s4.kep_fields('PID', 'log', 1.0)
    assert k['kep_frac_vx'] == '' and 'no kep channel' in k['kep_note']


def test_plant_variants_change_exactly_the_declared_keys(monkeypatch):
    """S04 v6.1 6 (RB-1): apply_plant_variant of each variant changes exactly the declared keys
    (motor_K and motor_u0 never); an unexpected change is an S03Stop."""
    assert s4.PLANT_VARIANTS == ('R0.8_JbTcTs', 'R1.25_JbTcTs', 'R0.82_only', 'Vbatt11.1')

    def plant():
        return fwsim.plant_params(argparse.Namespace(ground='traction', mu=0.5))

    for name, k in (('R0.8_JbTcTs', 0.8), ('R1.25_JbTcTs', 1.25)):
        p = plant()
        b = s4.plant_dump(p)
        rep = s4.apply_plant_variant(p, name)
        a = s4.plant_dump(p)
        assert set(rep['expected']) == {'motor_R'} | set(s4.PLANT_COUPLED)
        assert set(rep['changed']) == set(rep['expected']) | set(rep['derived'])
        assert rep['derived'] and set(rep['derived']) <= set(s4.PLANT_DERIVED)
        assert np.allclose(np.asarray(a['motor_R']), np.asarray(b['motor_R']) * k)
        for n in s4.PLANT_COUPLED:
            assert np.allclose(np.asarray(a[n]), np.asarray(b[n]) / k)
        assert np.array_equal(np.asarray(a['motor_K']), np.asarray(b['motor_K']))
        assert np.array_equal(np.asarray(a['motor_u0']), np.asarray(b['motor_u0']))
    p = plant()
    b = s4.plant_dump(p)
    rep = s4.apply_plant_variant(p, 'R0.82_only')
    assert rep['changed'] == ['motor_R'] and rep['derived'] == []
    assert np.allclose(np.asarray(s4.plant_dump(p)['motor_R']), np.asarray(b['motor_R']) * 0.82)
    p = plant()
    rep = s4.apply_plant_variant(p, 'Vbatt11.1')
    assert rep['changed'] == ['V_batt'] and s4.plant_dump(p)['V_batt'] == 11.1
    with pytest.raises(s3.S03Stop, match='unknown plant variant'):
        s4.apply_plant_variant(plant(), 'R2')
    p = plant()                                                     # an unexpected key changes
    monkeypatch.setattr(p, 'update_physical',
                        lambda **kw: setattr(p, 'motor_K', np.asarray(p.motor_K) * 2.0))
    with pytest.raises(s3.S03Stop, match='not the expected'):
        s4.apply_plant_variant(p, 'R0.8_JbTcTs')


def test_sens_after_pre_checks_every_variant_before_any_simulation(tmp_path, monkeypatch):
    """S04 v6.8 6: sens_after runs the plant-variant pre-check of EVERY controller's eval mu and
    EVERY variant before the first simulation of any controller (the pre-check is its own loop over
    all send_to_HIL rows, not one per controller inside the run loop)."""
    d = _s03_dir(tmp_path, trial=True)
    out, dll = tmp_path / 'r', tmp_path / 'x.dll'
    out.mkdir()
    dll.write_bytes(b'x')
    lock = dict(common=dict(tool_check=True, s03_run=str(d), fingerprint={}, unsealed=False),
                s03_mu_levels=[0.5],
                controllers={c: dict(eval_mu=[0.5], speeds=[0.2]) for c in s3.CONTROLLERS})
    (out / 'lock.yaml').write_text(json.dumps(lock), encoding='utf-8')
    s3.write_csv(str(out / 'selected.csv'),
                 [dict(kind='send_to_HIL', controller=c, feasible='true',
                       **{n: '1.0' for n in s3.FACTORS[c]}) for c in s3.CONTROLLERS])
    calls, real_variant = [], s4.apply_plant_variant
    monkeypatch.setattr(s4, 'fingerprint', lambda *a, **k: {})
    monkeypatch.setattr(s4.fwsim, 'FirmwareSim', lambda *a, **k: object())
    monkeypatch.setattr(s4, 'apply_plant_variant',
                        lambda p, name: (calls.append(('pre', name)), real_variant(p, name))[1])
    res = {c: '0' for c in s4.SENS_AFTER_COLS[7:]}
    res.update(status='ok', completed='true')
    monkeypatch.setattr(s4, 'run_custom',
                        lambda *a, **k: (calls.append(('run', k.get('plant_variant'))),
                                         (res, None, None, {}, {}))[1])
    assert s4.sens_after(argparse.Namespace(out=str(out), dll=str(dll), io_delay_ms=None)) == 0
    n_pre = len(s3.CONTROLLERS) * len(s4.PLANT_VARIANTS)               # one eval mu, all variants
    pre = [i for i, c in enumerate(calls) if c[0] == 'pre']
    run = [i for i, c in enumerate(calls) if c[0] == 'run']
    assert len(pre) == n_pre and run and max(pre) < min(run)


def test_post_spectral_radius_waits_for_liam(tmp_path, monkeypatch):
    """S04 v6.8 6: rho >= 1 (or not computable although inside the table) is a cho_liam item —
    post returns 2, selected.csv unchanged, meta.yaml pre_hil holds the radius and the states;
    rho == 1.0 is the unstable boundary."""
    g = dict(G_PID, pos_kp=0.0, pos_ki=3.0)                          # inside the table, rho >= 1
    rc, notes, out, sel = _fake_post(tmp_path, monkeypatch, gains=g)
    assert rc == 2 and len(notes['cho_liam']) == 1
    assert notes['cho_liam'][0]['check'] == 'spectral_radius' and notes['cho_liam'][0]['reasons']
    assert (out / 'selected.csv').read_bytes() == sel                # the selected gains are kept
    yaml = pytest.importorskip('yaml')
    ph = yaml.safe_load(open(out / 'meta.yaml', encoding='utf-8'))['pre_hil']['PID pre_hil']
    assert ph['spectral_radius'] >= 1 and ph['spectral_radius_state'] == 'khong_on_dinh'
    assert ph['spectral_radius_states'] or ph['spectral_radius_dropped']
    mh = s4.mo_hinh_con
    real = mh.ban_kinh_pho_va_trang_thai

    def fake_rho(rho):
        def f(ctrl, g_):
            m = dict(real(ctrl, g_))
            m['ban_kinh_pho'], m['trang_thai'] = rho, mh.trang_thai(rho)
            return m
        return f
    monkeypatch.setattr(mh, 'ban_kinh_pho_va_trang_thai', fake_rho(1.0))
    rc, notes, *_ = _fake_post(tmp_path / 'b', monkeypatch)
    assert rc == 2 and notes['cho_liam'][0]['check'] == 'spectral_radius'   # rho == 1 is not stable
    monkeypatch.setattr(mh, 'ban_kinh_pho_va_trang_thai', fake_rho(math.nextafter(1.0, 0.0)))
    rc, notes, *_ = _fake_post(tmp_path / 'c', monkeypatch)
    assert rc == 0 and notes['cho_liam'] == []
    monkeypatch.setattr(mh, 'ban_kinh_pho_va_trang_thai',             # inside the table, not computable
                        lambda c, g_: (_ for _ in ()).throw(ValueError('bad gains')))
    rc, notes, *_ = _fake_post(tmp_path / 'd', monkeypatch)
    assert rc == 2 and notes['cho_liam'][0]['check'] == 'spectral_radius'


def test_post_sensitivity_waits_for_liam(tmp_path, monkeypatch):
    """S04 v6.8 6: judge() 'sensitive' for b0 (LADRC) or motor R appends a cho_liam item and post
    returns 2; unchanged runs give rc 0."""
    def mod(r, kind, f, v, mu):
        if f != 1.0:
            r.update(completed='false')
    rc, notes, *_ = _fake_post(tmp_path / 'a', monkeypatch, sens_mod=mod)
    assert rc == 2 and [c['check'] for c in notes['cho_liam']] == ['r_factor']
    assert notes['cho_liam'][0]['reasons']
    rc, notes, *_ = _fake_post(tmp_path / 'b', monkeypatch, ctrl='ADRC', gains=G_ADRC, sens_mod=mod)
    assert rc == 2 and {c['check'] for c in notes['cho_liam']} == {'b0_factor', 'r_factor'}
    rc, notes, *_ = _fake_post(tmp_path / 'c', monkeypatch)
    assert rc == 0 and notes['cho_liam'] == []


def test_post_findings_gate_the_gain_send(tmp_path, monkeypatch, capsys):
    """S04 6 hand-over: a post with a cho_liam finding makes gain_send --s04-run refuse (5, nothing
    sent); --liam-da-quyet '<reason>' sends once and logs the reason; a clean post sends."""
    import gain_send as gs
    rc, notes, out, _ = _fake_post(
        tmp_path / 'a', monkeypatch,
        mu3_mod=lambda r, v, mu: r.update(g3_state='khong_dat') if mu == 0.4 else None)
    assert rc == 2 and notes['cho_liam']
    calls = []

    def exch(ip, port, make_packet, timeout, tries, log=print):
        calls.append(make_packet(7))
        return dict(ok='1', req_id='1', id='1', ctrl='PID', mac='aa', boot='1', sha='s', env='e',
                    applied_n='0', **{n: repr(float(np.float32(v))) for n, v in G_PID.items()}), 1
    monkeypatch.setattr(gs, 'exchange', exch)
    gains = ','.join(f'{n}={v}' for n, v in G_PID.items())
    base = ['--ip', '127.0.0.1', '--port', '9', '--ctrl', 'PID', '--id', '1', '--gains', gains]
    assert gs.main(base + ['--s04-run', str(out)]) == 5 and calls == []
    txt = capsys.readouterr()
    assert '# cho_liam:' in txt.err and txt.out.rstrip().endswith('#ket_qua,ma=5')
    assert gs.main(base + ['--s04-run', str(out), '--liam-da-quyet', 'Liam checked']) == 0
    assert len(calls) == 1
    txt = capsys.readouterr().out
    assert '#ban_giao_s04,' in txt and 'Liam checked' in txt and '# ban_kinh_pho=' in txt
    assert txt.count('# ack ') == 1 and txt.rstrip().endswith('#ket_qua,ma=0')
    rc, notes, out2, _ = _fake_post(tmp_path / 'b', monkeypatch)
    assert rc == 0 and notes['cho_liam'] == []
    assert gs.main(base + ['--s04-run', str(out2)]) == 0
    txt = capsys.readouterr().out
    assert 'liam_da_quyet=""' in txt and txt.rstrip().endswith('#ket_qua,ma=0')
    assert gs.main(base + ['--liam-da-quyet', 'x']) == 1               # reason without --s04-run


@needs_dll
def test_run_gate_records_the_kep_columns(monkeypatch):
    """S03 v8.6 11.6: the PID evals row holds the three kep fractions of fwsim.kep_indicator (report
    only); the ADRC row has '' everywhere; a ValueError gives '' + kep_note, status/score unchanged."""
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    import o1_feasibility as o1
    path, g = o1.Path8(), s3.defaults(fw, 'PID')
    res, log = s4.run_gate(fw, path, 'PID', g, 0.2, 0.436)
    if getattr(fw, 'kep_cols', 0):
        k = fwsim.kep_indicator(log, res['t_end'])
        assert (res['kep_frac_vx'], res['kep_frac_vy'], res['kep_frac_wz']) == \
            (k['kep_frac_vx'], k['kep_frac_vy'], k['kep_frac_wz']) and res['kep_note'] == ''
    else:                                       # an old DLL without the export: note, report only
        assert res['kep_frac_vx'] == '' and res['status'] != 'run_error'
    res2, _ = s4.run_gate(fw, path, 'ADRC', s3.defaults(fw, 'ADRC'), 0.2, 0.436)
    assert res2['kep_frac_vx'] == res2['kep_frac_vy'] == res2['kep_frac_wz'] == res2['kep_note'] == ''
    if getattr(fw, 'kep_cols', 0):
        monkeypatch.setattr(fwsim, 'kep_indicator',
                            lambda *a: (_ for _ in ()).throw(ValueError('no kep channel in log')))
        res3, _ = s4.run_gate(fw, path, 'PID', g, 0.2, 0.436)
        assert res3['kep_frac_vx'] == '' and 'no kep channel' in res3['kep_note']
        assert res3['status'] != 'run_error' and math.isfinite(float(res3['e_perp_max']))


def test_main_refuses_a_dll_without_the_kep_channel(tmp_path, monkeypatch):
    """S03 v8.6 section 11 item 6 (c): a PID trial stops when the DLL has no clamp export
    (kep_cols 0); an ADRC trial is not stopped by that check."""
    d = _s03_dir(tmp_path, trial=True)
    calls = []

    class _NoKep:
        kep_cols = 0
        build_info = {}

    monkeypatch.setattr(fwsim, 'FirmwareSim', lambda dll: (calls.append(dll), _NoKep())[1])
    monkeypatch.setattr(s4, 'fingerprint', lambda *a, **k: sys.exit('past the kep check'))
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial', '--tool-check', '--controller',
                                    'PID', '--s03', str(d), '--out', str(tmp_path / 'o'), '--dll', 'x.dll'])
    with pytest.raises(SystemExit, match='no kep channel'):
        s4.main()
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial', '--tool-check', '--controller',
                                    'ADRC', '--s03', str(d), '--out', str(tmp_path / 'o2'), '--dll', 'x.dll'])
    with pytest.raises(SystemExit, match='past the kep check'):       # stopped later, not by kep_cols
        s4.main()
    assert len(calls) == 2


# CODE 23/11 (gop thu dot 2): so_goc doc dung dong 'sang_loc:' ma fwsim/s03_doe.py ghi (D2a 13d0a76)
def test_so_goc_doc_dong_sang_loc_cua_s03_doe(tmp_path):
    num = tmp_path / 'so.csv'
    num.write_bytes(b'set,row\r\nS0,1\r\n')
    raw = num.read_bytes()
    plan = dict(che_do='chinh', m_min=86, ghi_chu='kế hoạch, không có kết quả',
                file_so=dict(file=str(num), sha256=hashlib.sha256(raw).hexdigest(),
                             sha256_lf=hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()))
    (tmp_path / 'meta.yaml').write_text(
        'spec: S03 v8.6\n'
        f"sang_loc: {json.dumps(plan, ensure_ascii=False)}   # PLAN of S03 3.1b (item 10, 8.10)\n",
        encoding='utf-8')
    got = s4.so_goc(str(tmp_path))
    assert got['file'] == str(num) and got['s03_sha256'] == plan['file_so']['sha256']
    (tmp_path / 'meta.yaml').write_text(
        'sang_loc: ' + json.dumps(dict(che_do='khong_chay: --trial mot muc mu')) + '\n', encoding='utf-8')
    assert s4.so_goc(str(tmp_path)) is None
    (tmp_path / 'meta.yaml').write_text('spec: S03 v8.6\n', encoding='utf-8')
    assert s4.so_goc(str(tmp_path)) is None
