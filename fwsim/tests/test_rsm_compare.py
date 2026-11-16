"""Tests of fwsim/rsm_compare.py (S04 v6.4 section 12). Synthetic S03 and S04 folders generated here
(S04 v6 C Q2: no sealed/ or sealed_test/ of a real S03 run is opened), responses from known quadratic
functions; the confirmation runs go through the real s04_ga.Evaluator with s04_ga.safe_run_gate replaced
by a synthetic run, except the last tests (need fwsim/build/fwsim.dll), which run real fwsim and a real
s04_ga --tool-check. Run: python -m pytest fwsim/tests/test_rsm_compare.py -q"""
import builtins
import csv
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import fwsim  # noqa: E402
import rsm_compare as rc  # noqa: E402
import s03_doe as s3  # noqa: E402
import s04_ga as s4  # noqa: E402

_DLL = os.path.join(fwsim.HERE, 'build', 'fwsim.dll')
needs_dll = pytest.mark.skipif(not os.path.isfile(_DLL), reason='fwsim.dll not built')
MU = (0.4, 0.5, 0.6)
SPEEDS = (0.2, 0.5, 0.8)
LEVELS = {c: {n: ((0.5, 8.0) if k % 2 == 0 else (1.0, 3.0)) for k, n in enumerate(s3.FACTORS[c])}
          for c in s3.CONTROLLERS}


# ── synthetic truth ──────────────────────────────────────────────────────────────────

def q1_convex(c, mu=0.5, v=0.2):
    c = np.asarray(c, dtype=float)
    return 0.03 + 0.002 * float(np.sum(c ** 2)) + 0.001 * c[0]


def q1_flat(c, mu=0.5, v=0.2):
    c = np.asarray(c, dtype=float)
    return 0.03 + 0.0005 * float(np.sum(c ** 2)) + 0.001 * c[0]


def q2_convex(c, mu=0.5, v=0.2):
    c = np.asarray(c, dtype=float)
    return 5.0 + 0.5 * float(np.sum((c - 0.3) ** 2))


W = np.array([0.011, 0.023, 0.037, 0.041, 0.053, 0.067, 0.071])


def q2_concave(c, mu=0.5, v=0.2):
    """minima at the box corners, all different: many candidates, ordered."""
    c = np.asarray(c, dtype=float)
    return 6.0 - 0.5 * float(np.sum(c ** 2)) + float(W @ c)


W_MIXED = W * np.array([1, -1, 1, -1, 1, -1, 1])


def q2_mixed(c, mu=0.5, v=0.2):
    """RSM3-6: weights of both signs: the f2 minimum is the corner (-1, 1, -1, 1, -1, 1, -1), H on 3 factors."""
    c = np.asarray(c, dtype=float)
    return 6.0 - 0.5 * float(np.sum(c ** 2)) + float(W_MIXED @ c)


def encode(dom, g):
    out = []
    for d in dom:
        L, H, x = d['L'], d['H'], g[d['name']]
        if d['scale'] == 'log':
            out.append(2.0 * math.log(x / math.sqrt(L * H)) / math.log(H / L))
        else:
            out.append((x - (L + H) / 2.0) / ((H - L) / 2.0))
    return out


# ── synthetic folders ────────────────────────────────────────────────────────────────

def _head():
    return subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()


def make_repo(tmp_path):
    r = tmp_path / 'repo'
    r.mkdir()
    for a in (['init', '-q'], ['config', 'user.email', 't@t'], ['config', 'user.name', 't']):
        subprocess.run(['git', '-C', str(r)] + a, check=True)
    (r / 'README').write_text('x\n')
    subprocess.run(['git', '-C', str(r), 'add', 'README'], check=True)
    subprocess.run(['git', '-C', str(r), 'commit', '-q', '-m', 'init'], check=True)
    return r


def _test_points(n=100, seed=7):
    rng = np.random.default_rng(seed)
    return [(s3.TEST_FIRST + i, tuple(float(x) for x in rng.uniform(-1, 1, 7))) for i in range(n)]


def make_s03(tmp_path, q1=q1_convex, q2=q2_convex, mods=None, fake=True, dll=None, levels=None, gate=SPEEDS,
             test_mods=None, trial=False):
    """S03 v8 folder: levels, design (centre, outer, inner, axial, test), sealed/runs.csv for every box but
    test, sealed_test/runs_test.csv; mods(row) changes a sealed row; rsm_fake: true unless fake=False."""
    d = tmp_path / 's03'
    (d / 'sealed').mkdir(parents=True)
    (d / 'sealed_test').mkdir()
    lv = levels or LEVELS
    with open(d / 'levels.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['controller', 'factor', 'L', 'H', 'centre', 'scale', 'default', 'rule_note'])
        for c in s3.CONTROLLERS:
            for n in s3.FACTORS[c]:
                L, H = lv[c][n]
                w.writerow([c, n, repr(L), repr(H), '', 'log' if L > 0 and H / L >= 4 else 'lin', '', ''])
    pts = ([(0, 'centre', (0.0,) * 7)] + [(p, 'outer', c) for p, c in s3.design_points(1.0, 1)] +
           [(p, 'inner', c) for p, c in s3.design_points(0.5, 129)] + [(p, 'axial', c) for p, c in s3.axial_points()] +
           [(p, 'test', c) for p, c in _test_points()])
    design, runs, trows = [], [], []
    for c in s3.CONTROLLERS:
        for pid, box, codes in pts:
            design.append(dict(controller=c, point_id=pid, box=box, **{f'X{k + 1}': x for k, x in enumerate(codes)}))
            for mu in MU:
                for v in SPEEDS:
                    if box == 'test' and mu != MU[1]:
                        continue
                    r = dict(controller=c, point_id=pid, box=box, mu=mu, v=v, status='ok', completed='true',
                             e_perp_max=q1(codes, mu, v), tv_duty=q2(codes, mu, v), sat_frac=0.0,
                             e_theta_cmd_max_deg=2.0, e_trk_max=0.02, e_bao_max=0.02, n_upper_frames=1,
                             n_back_frames=0, error='')
                    if box == 'test':
                        if test_mods:
                            test_mods(r)
                        trows.append(r)
                    else:
                        if mods:
                            mods(r)
                        runs.append(r)
    s3.write_csv(str(d / 'design.csv'), design)
    s3.write_csv(str(d / 'sealed' / 'runs.csv'), runs)
    s3.write_csv(str(d / 'sealed_test' / 'runs_test.csv'), trows)
    (d / 'sealed' / 'notes.json').write_text(json.dumps(dict(elapsed_s_main_design=123, run_error_reruns=[])))
    (d / 'sealed_test' / 'notes.json').write_text(json.dumps(dict(elapsed_s_test=45, run_error_reruns=[])))
    dll_sha = s3.sha256_file(dll) if dll and os.path.isfile(dll) else 'no dll'
    y3 = {f'{c} v={v:g}': 5.0 for c in s3.CONTROLLERS for v in SPEEDS}
    (d / 'meta.yaml').write_text(
        f'trial: {str(trial).lower()}     # x\nstatus: "ran to the end"\nmu_levels: {json.dumps(list(MU))}\n'
        + ('rsm_fake: true\n' if fake else '') +
        f'script: {json.dumps(dict(commit=_head(), dirty=False))}\nfirmware:\n  dll_sha256: {dll_sha}\n'
        f'plant: {json.dumps(dict(motor_model="dc", src=fwsim.PLANT_SRC))}\n'
        f'o1: {json.dumps(s4.o1_constants())}\nimu: {json.dumps(s3.IMU)}\n'
        f'gate_speeds: {json.dumps(list(gate))}\ny3_reference: {json.dumps(y3)}\n', encoding='utf-8')
    return d


def run_row(**kw):
    r = dict(status='ok', completed='true', e_perp_max=0.03, tv_duty=5.0, sat_frac=0.0, t_end=10.0,
             e_theta_lane_max_deg=1.0, e_theta_cmd_max_deg=2.0, ds_absmax=0.0, e_trk_max=0.02, e_bao_max=0.02)
    r.update(kw)
    return r


def make_s04(repo, s03, dll, eval_mu=(0.5,), speeds=SPEEDS, ctrls=('PID',), x_ga=(0.035, 5.5), ga=None,
             three_mu_mod=None, three_mu_drop=None, q1=q1_convex, q2=q2_convex):
    """S04 folder as s04_ga writes it (tool check): lock.yaml, selected.csv, phases.json, history/evals,
    trial/, post_checks/. ga: list of (code, mods dict for evals rows or None) visited candidates."""
    out = repo / '05_DATA' / 'results' / 'S04' / 'run01'
    (out / 'post_checks').mkdir(parents=True)
    ctl = {}
    for c in ctrls:
        dom = s4.read_domain(str(s03), c)
        ctl[c] = dict(tool_check=True, domain=dom, speeds=list(speeds), eval_mu=list(eval_mu), pop=2, n_gen=1,
                      seeds=[1], mu_decision='mid')
    lock = dict(common=dict(tool_check=True, s03_run=str(s03), fingerprint=s4.fingerprint(dll, str(s03)),
                            s03_mu_levels=list(MU), unsealed=False, s03_source_diff=[]),
                controllers=ctl, s03_mu_levels=list(MU))
    (out / 'lock.yaml').write_text(json.dumps(lock, indent=1, default=str))
    (out / 'meta.yaml').write_text('spec: x\n')
    sel, phases = [], []
    for c in ctrls:
        dom = ctl[c]['domain']
        g = {d['name']: s3.decode(0.1, d['L'], d['H'], d['scale']) for d in dom}
        sel.append(dict(kind='per_seed', controller=c, ga_seed=1, feasible='true'))
        sel.append(dict(kind='send_to_HIL', controller=c, ga_seed=1, feasible='true', gen=1, cand_id=0, f1=x_ga[0],
                        f2=x_ga[1], **{n: repr(v) for n, v in g.items()}))
        phases += [dict(phase='trial', controller=c, elapsed_s=11), dict(phase='main', controller=c, elapsed_s=22)]
        hist, evals = [], []
        lo, hi = s4.ga_bounds(dom)
        for k, (code, mod) in enumerate(ga or [((0.2,) * 7, None), ((-0.4, 0.1, 0.0, 0.3, -0.2, 0.5, 0.0), None)]):
            gk = {d['name']: s3.decode(ci, d['L'], d['H'], d['scale']) for d, ci in zip(dom, code)}
            xga = [a + (ci + 1) / 2 * (b - a) for ci, a, b in zip(code, lo, hi)]
            hist.append(dict(controller=c, ga_seed=1, gen=1, cand_id=k, **gk,
                             **{f'x_{d["name"]}': x for d, x in zip(dom, xga)}))
            for mu in eval_mu:
                for v in speeds:
                    r = dict(controller=c, ga_seed=1, gen=1, cand_id=k, v=v, mu=mu, attempt=1,
                             **run_row(e_perp_max=q1(code), tv_duty=q2(code)), cause='')
                    for p in ('e_perp_max', 'e_trk_max', 'e_bao_max', 'e_theta_cmd_max_deg'):
                        r['pen_' + p] = 0
                    if mod:
                        for extra in mod(r, v) or []:
                            evals.append(extra)
                    evals.append(r)
        s3.write_csv(str(out / f'history_{c}.csv'), hist)
        s3.write_csv(str(out / f'evals_{c}.csv'), evals)
        (out / 'trial' / c).mkdir(parents=True)
        s3.write_csv(str(out / 'trial' / c / 'evals.csv'), [dict(controller=c, attempt=1, v=0.2, mu=0.5)])
    phases.append(dict(phase='post', elapsed_s=33))
    s3.write_csv(str(out / 'selected.csv'), sel)
    (out / 'phases.json').write_text(json.dumps(phases))
    det = [dict(controller=c, v=0.2, mu=0.5, equal='true') for c in ctrls]
    s3.write_csv(str(out / 'post_checks' / 'determinism.csv'), det)
    s3.write_csv(str(out / 'post_checks' / 'sensitivity.csv'), [dict(controller=c, check='r_factor') for c in ctrls])
    tm = []
    for c in ctrls:
        for mu in MU:
            for v in speeds:
                r = dict(tool_check='true', controller=c, v=v, mu=mu, **run_row())
                if three_mu_mod:
                    three_mu_mod(r)
                tm.append(r)
    if three_mu_drop:
        tm = [{k: v for k, v in r.items() if k != three_mu_drop} for r in tm]
    s3.write_csv(str(out / 'post_checks' / 'three_mu.csv'), tm)
    (out / 'post_checks' / 'notes.json').write_text('{}')
    return out


@pytest.fixture
def dummy_dll(tmp_path):
    p = tmp_path / 'fake.dll'
    p.write_bytes(b'not a dll')
    return str(p)


@pytest.fixture
def fake_eval(monkeypatch):
    """The real s04_ga.Evaluator, with safe_run_gate / run_in_new_process synthetic. behaviour(idx, code, mu, v)
    -> run row (idx = order in which the candidate gains were first seen)."""
    state = dict(seen=[], behaviour=None, dom=None, calls=0)

    def safe_run_gate(fw, path, ctrl, g, v, mu):
        key = tuple(sorted(g.items()))
        if key not in state['seen']:
            state['seen'].append(key)
        state['calls'] += 1
        code = encode(state['dom'][ctrl], g)
        return state['behaviour'](state['seen'].index(key), code, mu, v), None

    def rinp(dll, runs, work):
        return [dict(row=run_row(status='run_error', completed='', error='again'), digest=None) for _ in runs]
    monkeypatch.setattr(s4, 'safe_run_gate', safe_run_gate)
    monkeypatch.setattr(s3, 'run_in_new_process', rinp)

    def factory(inp, ctrl, speeds, mus, work):
        state['dom'] = {c: inp.ctrl_lock(c)['domain'] for c in inp.ctrls}
        return s4.Evaluator(None, inp.dll, ctrl, speeds, mus, work)
    monkeypatch.setattr(rc, 'EVALUATOR', factory)
    state.update(factory=factory, srg=safe_run_gate, rinp=rinp)
    state['behaviour'] = lambda i, code, mu, v: run_row(e_perp_max=q1_convex(code), tv_duty=q2_convex(code))
    return state


def setup(tmp_path, dll, **kw):
    s03_kw = {k: kw.pop(k) for k in list(kw) if k in ('q1', 'q2', 'mods', 'fake', 'test_mods')}
    repo = make_repo(tmp_path)
    s03 = make_s03(tmp_path, dll=dll, **s03_kw)
    q = {k: s03_kw[k] for k in ('q1', 'q2') if k in s03_kw}
    s04 = make_s04(repo, s03, dll, **q, **kw)
    return repo, s03, s04


def att(s04):
    return list(csv.DictReader(open(s04 / 'rsm' / 'attempts.csv', encoding='utf-8')))


def git_out(repo, *a):
    return subprocess.run(['git', '-C', str(repo)] + list(a), capture_output=True, text=True, check=True).stdout


# ── model ────────────────────────────────────────────────────────────────────────────

def test_36_terms_and_ccd_rank():
    assert len(rc.TERMS) == 36 and rc.TERMS[0] == '1' and rc.TERMS[-1] == 'X6*X7'
    exp = rc.expected_codes()
    C = np.array([exp[p][1] for p in rc.FIT_IDS])
    assert C.shape == (143, 7) and np.linalg.matrix_rank(rc.design_matrix(C)) == 36
    for a in rc.AXIAL_IDS:                                    # 12.6: dropping one axial point keeps rank 36
        keep = [i for i, p in enumerate(rc.FIT_IDS) if p != a]
        assert np.linalg.matrix_rank(rc.design_matrix(C[keep])) == 36
    # every axial point gone: the seven X_k^2 columns are the same (corners only), rank < 36
    keep = [i for i, p in enumerate(rc.FIT_IDS) if p not in rc.AXIAL_IDS]
    assert np.linalg.matrix_rank(rc.design_matrix(C[keep])) < 36


def test_fit_recovers_a_quadratic_and_gradient_is_exact():
    rng = np.random.default_rng(1)
    beta = rng.normal(size=36)
    exp = rc.expected_codes()
    C = np.array([exp[p][1] for p in rc.FIT_IDS])
    b, d = rc.fit_ls(C, rc.design_matrix(C) @ beta)
    assert d['rank'] == 36 and np.allclose(b, beta, atol=1e-10)
    c = rng.uniform(-1, 1, 7)
    num = np.array([(rc.predict(beta, c + h) - rc.predict(beta, c - h)) / 2e-6 for h in np.eye(7) * 1e-6])
    assert np.allclose(rc.gradient(beta, c), num, atol=1e-6)


@pytest.mark.parametrize('ga,ra,f2ga,f2r,cls', [
    ((0.029, 9.0), (0.038, 5.0), 9.0, 5.0, 'R3'),          # 12.5 example: {B, C} -> C, not within tolerance
    ((0.020, 10.0), (0.050, 5.0), 10.0, 5.0, 'R1'),        # C outside f1* + 0.01: GA chosen
    ((0.030, 5.0), (0.035, 5.9), 5.0, 5.9, 'R2'),          # within 0.01 m and 20 %
    ((0.030, 0.0), (0.031, 0.0), 0.0, 0.0, 'R2'),          # f2(x_GA) = 0: only when equal
    ((0.030, 0.0), (0.031, 0.1), 0.0, 0.1, 'R1'),
])
def test_classify(ga, ra, f2ga, f2r, cls):
    a = dict(f1=ga[0], f2=ga[1], x=[1.0], ga_seed=1)
    b = dict(f1=ra[0], f2=ra[1], x=[2.0], ga_seed=0)
    assert rc.classify(a, b) == cls


def test_classify_example_scope():
    """12.5 S46-9: {B, C} gives R3, but on {A, B, C} the section 5 rule still chooses B."""
    A, B, C = (dict(f1=f1, f2=f2, x=[k], ga_seed=1) for k, (f1, f2) in enumerate(((0.020, 10), (0.029, 9), (0.038, 5))))
    assert rc.classify(B, C) == 'R3'
    assert s4.select([A, B, C]) is B


def R(**kw):
    r = dict(controller='PID', phase='confirm', cand_id=1, v=0.2, mu=0.5, attempt=1, **run_row(), pen_e_perp_max=0,
             pen_e_trk_max=0, pen_e_bao_max=0, pen_e_theta_cmd_max_deg=0)
    r.update(kw)
    return r


@pytest.mark.parametrize('rows,lab', [
    ([R()], 'pass'),
    ([R(completed='false')], 'violation_observed'),
    ([R(completed='false', status='track_err')], 'violation_observed'),          # Y1 conclusion also with track_err
    ([R(e_perp_max=0.2)], 'violation_observed'),
    ([R(status='track_err', e_perp_max=0.2)], 'inconclusive'),                   # Y2 of a track_err row is flagged
    ([R(status='track_err', e_trk_max=0.2)], 'violation_observed'),              # G3 trusted under track_err
    ([R(status='track_err', e_theta_cmd_max_deg=9.0)], 'violation_observed'),    # G5 too
    ([R(e_bao_max=0.2)], 'violation_observed'),
    ([R(status='track_err', e_bao_max=0.2)], 'inconclusive'),                    # G4 not under track_err
    ([R(e_perp_max=1e6, pen_e_perp_max=1)], 'run_error_excluded'),               # a penalty is no evidence
    ([R(status='run_error', completed='')], 'run_error_excluded'),
    ([R(completed='inconclusive')], 'inconclusive'),
    ([R(status='run_error', completed=''), R(completed='false', v=0.5)], 'violation_observed'),   # order of 12.4
    ([R(status='track_err'), R(status='run_error', completed='', v=0.5)], 'run_error_excluded'),
])
def test_label_order(rows, lab):
    assert rc.label(rows)[0] == lab


# ── phase 1 ──────────────────────────────────────────────────────────────────────────

def test_fit_phase_recovers_truth_and_commits_only_its_paths(tmp_path, dummy_dll):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    (repo / 'other.txt').write_text('staged by someone else\n')
    subprocess.run(['git', '-C', str(repo), 'add', 'other.txt'], check=True)
    rc.run_phase('fit', str(s04), dummy_dll)
    pts = list(csv.DictReader(open(s04 / 'rsm' / 'fit' / 'fit_points.csv')))
    assert len(pts) == 143 and {p['box'] for p in pts} == {'centre', 'outer', 'axial'}
    co = {(r['model'], r['variant'], r['response'], r['term']): float(r['coef'])
          for r in csv.DictReader(open(s04 / 'rsm' / 'fit' / 'coefs.csv'))}
    assert co[('main', 'base', 'f1', '1')] == pytest.approx(0.03) and co[('main', 'base', 'f1', 'X1')] == pytest.approx(0.001)
    assert co[('main', 'base', 'f1', 'X3^2')] == pytest.approx(0.002) and abs(co[('main', 'base', 'f1', 'X1*X2')]) < 1e-12
    assert co[('main', 'base', 'f2', 'X1^2')] == pytest.approx(0.5)
    assert {k[1] for k in co} == {'base', 'no_flag', 'cens05'} | {f'drop_axial_{a}' for a in rc.AXIAL_IDS}
    rows = att(s04)
    assert [(r['kind'], r['phase'], r['status']) for r in rows] == [('start', 'fit', ''), ('end', 'fit', 'done')]
    log = git_out(repo, 'log', '--format=%s', '-3').splitlines()
    assert log[0].startswith('fw: rsm_compare fit done') and log[1].startswith('fw: rsm_compare start fit')
    for rev in ('HEAD', 'HEAD~1'):
        names = git_out(repo, 'show', '--name-only', '--format=', rev).split()
        assert names and all(n.startswith('05_DATA/results/S04/run01/rsm/') for n in names)
    assert 'other.txt' in git_out(repo, 'diff', '--cached', '--name-only')            # left staged, not taken
    lock = json.load(open(s04 / 'rsm' / 'fit' / 'lock.yaml'))
    assert set(lock['files']) == {'fit/fit_points.csv', 'fit/coefs.csv', 'fit/fit_diag.yaml'}
    assert lock['solver'] == rc.SOLVER and lock['eps'] == rc.EPS and lock['K'] == 5


@pytest.mark.parametrize('mod,msg', [
    (lambda r: r.update(status='run_error') if (r['point_id'], r['v'], r['mu']) == (3, 0.2, 0.5) else None, 'run_error'),
    (lambda r: r.update(e_perp_max='nan') if r['point_id'] == 260 else None, 'not finite'),
    (lambda r: r.update(completed='maybe') if r['point_id'] == 0 else None, 'not usable'),
    (lambda r: r.update(point_id=500) if r['point_id'] == 270 else None, 'not in the 143'),
    (lambda r: r.update(n_upper_frames='') if r['point_id'] == 5 else None, 'n_upper_frames'),
    (lambda r: r.update(box='axial') if r['point_id'] == 7 else None, "has box 'axial'"),
    (lambda r: r.update(box='outer') if r['point_id'] == 263 else None, "has box 'outer'"),
])
def test_fit_input_checks_stop_and_close_the_attempt(tmp_path, dummy_dll, mod, msg):
    repo, s03, s04 = setup(tmp_path, dummy_dll, mods=mod)
    with pytest.raises(rc.Stop, match=msg):
        rc.run_phase('fit', str(s04), dummy_dll)
    rows = att(s04)
    assert [(r['kind'], r['status']) for r in rows] == [('start', ''), ('end', 'stopped')] and msg in rows[1]['reason']
    assert not (s04 / 'rsm' / 'fit').exists()
    assert 'attempts.csv' in git_out(repo, 'show', '--name-only', '--format=', 'HEAD')    # the stop is committed


def test_fit_design_codes_and_missing_point(tmp_path, dummy_dll):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    rows = list(csv.DictReader(open(s03 / 'design.csv')))
    for r in rows:
        if r['point_id'] == '262':
            r['X3'] = '0.5'
    s3.write_csv(str(s03 / 'design.csv'), rows)
    with pytest.raises(rc.Stop):                     # the S03 file differs from the S04 fingerprint first
        rc.run_phase('fit', str(s04), dummy_dll)
    d = rc.read_design(str(s03), 'PID')
    with pytest.raises(rc.Stop, match='3.2 pattern'):
        rc.check_fit_design(d, 'PID')
    d.pop(262)
    with pytest.raises(rc.Stop, match='missing'):
        rc.check_fit_design(d, 'PID')


def _row(**kw):
    r = dict(controller='PID', box='outer', point_id='1', v='0.2', mu='0.5', status='ok', completed='true',
             e_perp_max='0.03', tv_duty='5.0')
    r.update(kw)
    return r


@pytest.mark.parametrize('rows', [
    [_row(), _row()],                                   # exact duplicate
    [_row(), _row(box='inner')],                        # RSM-7: the same condition under another box
    [_row(), _row(v='0.2000000001')],                   # RSM-7: two rows matching one speed by isclose
])
def test_more_than_one_row_for_a_condition_stops(rows):
    by = rc.index_runs(rows, 'PID', {1}, [0.2, 0.5, 0.8], [0.4, 0.5, 0.6])
    with pytest.raises(rc.Stop, match='2 runs'):
        rc.need_run(by, 1, 0.2, 0.5, 'PID', 'x', 'outer')


def test_one_row_per_condition_passes_and_other_levels_are_left_aside():
    by = rc.index_runs([_row(), _row(v='0.65')], 'PID', {1}, [0.2, 0.5, 0.8], [0.4, 0.5, 0.6])
    assert rc.need_run(by, 1, 0.2, 0.5, 'PID', 'x', 'outer')['v'] == '0.2'


def test_tool_check_refuses_a_real_s03(tmp_path, dummy_dll):
    repo, s03, s04 = setup(tmp_path, dummy_dll, fake=False)
    with pytest.raises(rc.Stop, match='rsm_fake'):
        rc.run_phase('fit', str(s04), dummy_dll)


def test_other_dll_than_the_s04_run_stops(tmp_path, dummy_dll):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    other = tmp_path / 'other.dll'
    other.write_bytes(b'other')
    with pytest.raises(rc.Stop, match='dll_sha256'):
        rc.run_phase('fit', str(s04), str(other))


def test_canonical_folder(tmp_path, dummy_dll):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    bad = repo / 'elsewhere' / 'run01'
    bad.parent.mkdir()
    os.rename(s04, bad)
    with pytest.raises(rc.Stop, match='canonical'):
        rc.run_phase('fit', str(bad), dummy_dll)


def test_no_flag_variant_not_fitted_when_rank_drops(tmp_path, dummy_dll):
    # every axial point inconclusive: the no_flag fit has rank < 36, reported, the main fit keeps them
    mod = (lambda r: r.update(completed='inconclusive') if r['point_id'] in rc.AXIAL_IDS else None)
    repo, s03, s04 = setup(tmp_path, dummy_dll, mods=mod)
    rc.run_phase('fit', str(s04), dummy_dll)
    diag = open(s04 / 'rsm' / 'fit' / 'fit_diag.yaml', encoding='utf-8').read()
    assert 'main/no_flag/f1: rank' in diag and '"inconclusive": 14' in diag
    co = list(csv.DictReader(open(s04 / 'rsm' / 'fit' / 'coefs.csv')))
    assert not any(r['variant'] == 'no_flag' for r in co) and any(r['variant'] == 'base' for r in co)


def test_main_and_mid_differ_with_three_mu(tmp_path, dummy_dll):
    q1 = (lambda c, mu=0.5, v=0.2: q1_convex(c) + 0.05 * (mu - 0.5) ** 2)
    repo, s03, s04 = setup(tmp_path, dummy_dll, q1=q1, eval_mu=MU)
    rc.run_phase('fit', str(s04), dummy_dll)
    co = {(r['model'], r['response'], r['term']): float(r['coef'])
          for r in csv.DictReader(open(s04 / 'rsm' / 'fit' / 'coefs.csv')) if r['variant'] == 'base'}
    assert co[('main', 'f1', '1')] == pytest.approx(0.03 + 0.05 * 0.01) and co[('mid', 'f1', '1')] == pytest.approx(0.03)


def test_fit_never_opens_sealed_test(tmp_path, dummy_dll, monkeypatch, fake_eval):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    real = builtins.open

    def guard(p, *a, **k):
        if 'sealed_test' in str(p):
            raise AssertionError(f'opened {p}')
        return real(p, *a, **k)
    monkeypatch.setattr(builtins, 'open', guard)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    monkeypatch.setattr(builtins, 'open', real)
    rc.run_phase('test', str(s04), dummy_dll)


# ── phase 2 ──────────────────────────────────────────────────────────────────────────

def test_opt_confirm_classify_default(tmp_path, dummy_dll, fake_eval):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    cands = list(csv.DictReader(open(s04 / 'rsm' / 'opt' / 'candidates.csv')))
    assert len(cands) == 1                                    # convex: every start, one solution after dedup
    c = json.loads(cands[0]['code'])
    assert np.allclose([float(x) for x in c], 0.3, atol=1e-4)
    assert float(cands[0]['f2_hat']) == pytest.approx(5.0, abs=1e-6)
    res = list(csv.DictReader(open(s04 / 'rsm' / 'confirm' / 'result.csv')))[0]
    assert res['state'] == 'x_RSM' and res['rsm_class'] == 'R2' and res['n_tried'] == '1'
    assert res['mu_label_rsm'] == 'robust' and res['mu_label_ga'] == 'robust'
    assert res['mu_label_ga_source'].startswith('post_checks/three_mu.csv:')
    ev = list(csv.DictReader(open(s04 / 'rsm' / 'confirm' / 'evals_rsm.csv')))
    assert {r['phase'] for r in ev} == {'confirm', 'mu3'}
    assert sum(r['phase'] == 'mu3' for r in ev) == 2 * 3 and {r['mu'] for r in ev if r['phase'] == 'mu3'} == {'0.4', '0.6'}
    assert all(r['ga_seed'] == '' and r['gen'] == '' and r['gain_source'] == 'rsm/opt/candidates.csv:2' for r in ev)
    # gains confirmed are the ones read back from the file, bit for bit
    g = {n: float(cands[0][n]) for n in s3.FACTORS['PID']}
    assert fake_eval['seen'][0] == tuple(sorted(g.items()))
    tried = list(csv.DictReader(open(s04 / 'rsm' / 'check' / 'candidates_tried.csv')))
    assert [(t['cand_id'], t['label']) for t in tried] == [('1', 'pass')]
    lock = json.load(open(s04 / 'rsm' / 'opt_lock.yaml'))
    assert all(p.split('/')[0] in ('opt', 'confirm', 'check') for p in lock['files'])
    cost = list(csv.DictReader(open(s04 / 'rsm' / 'check' / 'cost.csv')))
    assert {r['part'] for r in cost} == {'a', 'b', 'c', 'mixed'} and not any('test' in r['item'] for r in cost)
    a = next(r for r in cost if r['part'] == 'a')
    assert a['n_runs_planned'] == '2313' and a['n_runs_actual'] == str(257 * 9)
    ax = list(csv.DictReader(open(s04 / 'rsm' / 'check' / 'axial_sens.csv')))
    assert len(ax) == 14 and all(abs(float(r['d_f1_hat'])) < 1e-9 for r in ax)   # exact quadratic: no change


def test_candidates_order_dedup_and_failure_labels(tmp_path, dummy_dll, fake_eval):
    """Concave f2: corner candidates, ordered by f2_hat; candidates 1-4 fail four ways, 5 passes."""
    repo, s03, s04 = setup(tmp_path, dummy_dll, q1=q1_flat, q2=q2_concave)
    rc.run_phase('fit', str(s04), dummy_dll)

    def beh(i, code, mu, v):
        base = run_row(e_perp_max=q1_flat(code), tv_duty=q2_concave(code))
        if i == 0:
            return dict(base, completed='false')
        if i == 1:
            return dict(base, status='run_error', completed='') if v == 0.5 else base
        if i == 2:
            return dict(base, status='track_err')
        if i == 3:
            return dict(base, e_bao_max=0.5)
        return base
    fake_eval['behaviour'] = beh
    rc.run_phase('opt', str(s04), dummy_dll)
    cands = list(csv.DictReader(open(s04 / 'rsm' / 'opt' / 'candidates.csv')))
    keys = [(float(c['f2_hat']), float(c['f1_hat'])) for c in cands]
    assert len(cands) > 5 and keys == sorted(keys)
    codes = [np.array([float(x) for x in json.loads(c['code'])]) for c in cands]
    assert all(np.linalg.norm(a - b) >= rc.DEDUP for i, a in enumerate(codes) for b in codes[i + 1:])
    assert all(c['neg_f2_hat'] == 'false' for c in cands) and all(c['edge'] for c in cands)   # corners: edge flag
    tried = list(csv.DictReader(open(s04 / 'rsm' / 'check' / 'candidates_tried.csv')))
    assert [t['label'] for t in tried] == ['violation_observed', 'run_error_excluded', 'inconclusive',
                                           'violation_observed', 'pass']
    res = list(csv.DictReader(open(s04 / 'rsm' / 'confirm' / 'result.csv')))[0]
    assert (res['x_rsm_cand_id'], res['n_tried'], res['n_violation_observed'], res['n_run_error_excluded'],
            res['n_inconclusive']) == ('5', '5', '2', '1', '1')
    ev = list(csv.DictReader(open(s04 / 'rsm' / 'confirm' / 'evals_rsm.csv')))
    a2 = [r for r in ev if r['attempt'] == '2']
    assert len(a2) == 1 and a2[0]['cand_id'] == '2' and a2[0]['cause'] == ''          # diagnosis only
    assert res['n_run_error_runs'] == '1'


def test_r0_iii_no_candidate_passes(tmp_path, dummy_dll, fake_eval):
    repo, s03, s04 = setup(tmp_path, dummy_dll, q1=q1_flat, q2=q2_concave)
    rc.run_phase('fit', str(s04), dummy_dll)
    fake_eval['behaviour'] = lambda i, code, mu, v: run_row(completed='inconclusive')
    rc.run_phase('opt', str(s04), dummy_dll)
    res = list(csv.DictReader(open(s04 / 'rsm' / 'confirm' / 'result.csv')))[0]
    assert res['state'].startswith('R0 (iii)') and res['rsm_class'] == 'R0' and res['n_tried'] == '5'
    assert res['n_inconclusive'] == '5' and res['mu_label_rsm'] == ''
    ax = list(csv.DictReader(open(s04 / 'rsm' / 'check' / 'axial_sens.csv')))
    assert {r['at'] for r in ax} == {'top candidate'}


@pytest.mark.parametrize('kind', ['i', 'ii'])
def test_r0_i_and_ii(tmp_path, dummy_dll, fake_eval, monkeypatch, kind):
    import scipy.optimize as so
    real = so.minimize

    def fake_min(fun, x0, **kw):
        r = real(fun, x0, **kw)
        if kind == 'i' or kw.get('constraints'):
            r.success = kind != 'i'
            if kind == 'ii':
                r.x = np.ones(7)                     # converged on a point that breaks f1_hat <= f1* + 0.01
        return r
    monkeypatch.setattr(so, 'minimize', fake_min)
    repo, s03, s04 = setup(tmp_path, dummy_dll, q1=lambda c, mu=0.5, v=0.2: q1_convex(c) + 0.01 * float(np.sum(c)))
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    res = list(csv.DictReader(open(s04 / 'rsm' / 'confirm' / 'result.csv')))[0]
    assert res['state'].startswith(f'R0 ({kind})') and res['n_tried'] == '0' and fake_eval['calls'] == 0
    ax = list(csv.DictReader(open(s04 / 'rsm' / 'check' / 'axial_sens.csv')))
    assert ax[0]['at'] == 'none'


def test_not_robust_over_mu_and_ga_label(tmp_path, dummy_dll, fake_eval):
    repo, s03, s04 = setup(tmp_path, dummy_dll, three_mu_mod=lambda r: r.update(e_trk_max=0.5) if r['mu'] == 0.6 else None)
    rc.run_phase('fit', str(s04), dummy_dll)
    fake_eval['behaviour'] = (lambda i, code, mu, v: run_row(e_perp_max=q1_convex(code), tv_duty=q2_convex(code),
                                                             completed='false' if mu == 0.4 else 'true'))
    rc.run_phase('opt', str(s04), dummy_dll)
    res = list(csv.DictReader(open(s04 / 'rsm' / 'confirm' / 'result.csv')))[0]
    assert res['mu_label_rsm'] == 'not robust over mu' and res['mu_label_ga'] == 'not robust over mu'
    assert res['rsm_class'] == 'R2'                           # the label does not change the class


def test_three_mu_without_g3_g5_columns_stops(tmp_path, dummy_dll, fake_eval):
    repo, s03, s04 = setup(tmp_path, dummy_dll, three_mu_drop='e_bao_max')
    rc.run_phase('fit', str(s04), dummy_dll)
    with pytest.raises(rc.Stop, match='e_bao_max'):
        rc.run_phase('opt', str(s04), dummy_dll)
    assert att(s04)[-1]['status'] == 'stopped' and not (s04 / 'rsm' / 'opt').exists()
    assert any((s04 / 'rsm' / 'aborted').iterdir())


def test_ga_points_labels(tmp_path, dummy_dll, fake_eval):
    def tool(r, v):                                          # run_error at v 0.5 replaced: cause tool: on the rerun
        if v == 0.5:
            again = dict(r, attempt=2, cause='tool: shim crash')
            r.update(status='run_error', completed='')
            return [again]

    def cause_on_first(r, v):                                # RSM-8: a cause on attempt 1 does not count
        if v == 0.5:
            again = dict(r, attempt=2)
            r.update(status='run_error', completed='', cause='tool: shim crash')
            return [again]

    def stored_pen(r, v):                                    # RSM-9: finite 1e6 with its stored flag
        if v == 0.2:
            r.update(e_perp_max=1e6, pen_e_perp_max=1)

    def notool(r, v):
        if v == 0.5:
            r.update(status='run_error', completed='')
            return [dict(r, attempt=2, status='ok', completed='true')]

    def pen(r, v):
        if v == 0.8:
            r.update(e_trk_max='', pen_e_trk_max=1)
    c0 = (0.2,) * 7
    ga = [(c0, None), (c0, None), ((0.1,) * 7, tool), ((0.0,) * 7, notool), ((-0.1,) * 7, pen),
          ((0.3,) * 7, cause_on_first), ((-0.3,) * 7, stored_pen)]
    repo, s03, s04 = setup(tmp_path, dummy_dll, ga=ga)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    pts = list(csv.DictReader(open(s04 / 'rsm' / 'check' / 'ga_points.csv')))
    assert [p['cand_id'] for p in pts] == ['0', '2']          # 1 duplicate of 0; 3, 5 not replaced; 4, 6 penalised
    summ = next(r for r in csv.DictReader(open(s04 / 'rsm' / 'check' / 'ga_errors.csv')) if r['kind'] == 'summary')
    assert summ['n_labelled'] == '2' and summ['n_unlabelled'] == '4'
    assert json.loads(summ['unlabelled_by_reason']) == {'run_error not replaced': 2, 'penalised quantity': 2}
    assert float(summ['maxabs_f1']) < 1e-9 and float(pts[0]['X1']) == pytest.approx(0.2)


# ── locks, attempts, no overwrite (12.7) ─────────────────────────────────────────────

def test_order_and_no_overwrite(tmp_path, dummy_dll, fake_eval):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    with pytest.raises(rc.Stop, match='fit/lock.yaml missing'):
        rc.run_phase('opt', str(s04), dummy_dll)
    with pytest.raises(rc.Stop, match='fit/lock.yaml missing'):
        rc.run_phase('test', str(s04), dummy_dll)
    rc.run_phase('fit', str(s04), dummy_dll)
    with pytest.raises(rc.Stop, match='no overwrite'):
        rc.run_phase('fit', str(s04), dummy_dll)
    with pytest.raises(rc.Stop, match='opt_lock.yaml missing'):
        rc.run_phase('test', str(s04), dummy_dll)


def test_fit_file_changed_after_lock_stops_opt(tmp_path, dummy_dll, fake_eval):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    rc.run_phase('fit', str(s04), dummy_dll)
    p = s04 / 'rsm' / 'fit' / 'coefs.csv'
    p.write_text(p.read_text() + 'PID,main,x,f1,1,0\n')
    with pytest.raises(rc.Stop, match='output commit'):
        rc.run_phase('opt', str(s04), dummy_dll)


def test_start_row_must_be_committed(tmp_path, dummy_dll, monkeypatch):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    monkeypatch.setattr(rc, 'commit_paths', lambda *a, **k: None)
    with pytest.raises(rc.Stop, match='not in HEAD'):
        rc.run_phase('fit', str(s04), dummy_dll)
    assert not (s04 / 'rsm' / 'fit').exists()


def test_dangling_attempt_and_abort_previous(tmp_path, dummy_dll, monkeypatch, fake_eval):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    real = rc.run_fit

    def crash(inp, ctx):
        real(inp, ctx)
        raise KeyboardInterrupt                     # power loss: no end row
    monkeypatch.setattr(rc, 'run_fit', crash)
    with pytest.raises(KeyboardInterrupt):
        rc.run_phase('fit', str(s04), dummy_dll)
    monkeypatch.setattr(rc, 'run_fit', real)
    with pytest.raises(rc.Stop, match='without an end row'):
        rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('fit', str(s04), dummy_dll, abort_reason='power loss in test')
    rows = att(s04)
    assert [(r['kind'], r['status']) for r in rows] == [('start', ''), ('end', 'stopped'), ('moved', 'moved'),
                                                        ('note', ''), ('start', ''), ('end', 'done')]
    ab = list((s04 / 'rsm' / 'aborted').iterdir())
    assert len(ab) == 1 and (ab[0] / 'fit' / 'coefs.csv').is_file() and 'power loss' in (ab[0] / 'reason.txt').read_text()
    # RSM2-4: the partial files of the run that died stay under aborted/, not committed (named in the note row)
    st = git_out(repo, 'status', '--porcelain', '--untracked-files=all', '--', '05_DATA/results/S04/run01/rsm').split('\n')
    st = [x for x in st if x.strip() and not x.endswith('/rsm/' + rc.RUN_LOCK)]   # RSM3-3: the lock file stays
    assert st and all(x.startswith('?? ') and '/rsm/aborted/' in x and '/fit/' in x for x in st)
    assert 'fit/coefs.csv' in rows[3]['reason']
    # a committed, done phase moved aside: its removal and its moved copy are committed
    rc.run_phase('opt', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll, abort_reason='exploratory re-run')
    st2 = [x for x in git_out(repo, 'status', '--porcelain', '--untracked-files=all', '--',
                                '05_DATA/results/S04/run01/rsm').split('\n')
           if x.strip() and not x.endswith('/rsm/' + rc.RUN_LOCK)]
    assert sorted(st2) == sorted(st)                       # nothing new uncommitted
    rc.run_phase('test', str(s04), dummy_dll)


def test_test_phase_errors_and_main_kind(tmp_path, dummy_dll, fake_eval):
    """Non-quadratic truth: the mid model has errors on the test points; misclassification counted."""
    q1 = (lambda c, mu=0.5, v=0.2: q1_convex(c) + 0.04 * float(np.sin(3 * np.asarray(c)[1])))
    repo, s03, s04 = setup(tmp_path, dummy_dll, q1=q1,
                           test_mods=lambda r: r.update(e_trk_max=0.5) if r['point_id'] == 275 else None)
    for ph in rc.PHASES:
        rc.run_phase(ph, str(s04), dummy_dll)
    te = list(csv.DictReader(open(s04 / 'rsm' / 'test' / 'test_errors.csv')))[0]
    assert te['n_points'] == '100' and float(te['rmse_f1_mid']) > 1e-4
    tp = list(csv.DictReader(open(s04 / 'rsm' / 'test' / 'test_points.csv')))
    p275 = next(p for p in tp if p['point_id'] == '275')
    assert p275['true_feasible_mid'] == 'false' and p275['pred_feasible'] == 'true'
    assert int(te['misclassified']) == sum(p['pred_feasible'] != p['true_feasible_mid'] for p in tp) >= 1
    meta = open(s04 / 'rsm' / 'meta.yaml', encoding='utf-8').read()
    assert 'comparison_kind: "main"' in meta and 'tool_check: true' in meta and 'coefs_commit_before_test: "' in meta
    costs = list(csv.DictReader(open(s04 / 'rsm' / 'test' / 'cost_test.csv')))
    assert costs[0]['n_runs_planned'] == '300' and costs[0]['n_runs_actual'] == '300'


def test_test_set_missing_run_stops(tmp_path, dummy_dll, fake_eval):
    repo, s03, s04 = setup(tmp_path, dummy_dll, test_mods=lambda r: r.update(status='run_error')
                           if (r['point_id'], r['v']) == (300, 0.8) else None)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    with pytest.raises(rc.Stop, match='run_error'):
        rc.run_phase('test', str(s04), dummy_dll)


def test_test_set_missing_row_stops(tmp_path, dummy_dll, fake_eval):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    p = s03 / 'sealed_test' / 'runs_test.csv'
    rows = [r for r in csv.DictReader(open(p)) if not (r['point_id'] == '301' and r['v'] == '0.5')]
    s3.write_csv(str(p), rows)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    with pytest.raises(rc.Stop, match='0 runs of PID point 301'):
        rc.run_phase('test', str(s04), dummy_dll)


# ── Codex round 1 fixes (RSM-1 .. RSM-16) ────────────────────────────────────────────

def test_f2_is_max_over_mu_of_the_mean_over_speeds(tmp_path, dummy_dll):
    """RSM-16: duty [0, 30, 0], [9, 9, 9], [8, 8, 8] over the speeds at the three mu: f2 = 10 (not 30, not 16)."""
    duty = {0.4: (0.0, 30.0, 0.0), 0.5: (9.0, 9.0, 9.0), 0.6: (8.0, 8.0, 8.0)}
    rows = {mu: [dict(e_perp_max=0.01 * (i + 1), tv_duty=t) for i, t in enumerate(ts)] for mu, ts in duty.items()}
    assert rc.aggregate(rows) == (pytest.approx(0.03), pytest.approx(10.0))
    q2 = (lambda c, mu=0.5, v=0.2: duty[mu][SPEEDS.index(v)])
    repo, s03, s04 = setup(tmp_path, dummy_dll, q2=q2, eval_mu=MU)
    rc.run_phase('fit', str(s04), dummy_dll)
    co = {(r['model'], r['response'], r['term']): float(r['coef'])
          for r in csv.DictReader(open(s04 / 'rsm' / 'fit' / 'coefs.csv')) if r['variant'] == 'base'}
    assert co[('main', 'f2', '1')] == pytest.approx(10.0) and co[('mid', 'f2', '1')] == pytest.approx(9.0)


def _start_in_head(repo):
    '''A start row without an end row is in HEAD's attempts.csv (read through git, not open).'''
    r = subprocess.run(['git', '-C', str(repo), 'show', 'HEAD:05_DATA/results/S04/run01/rsm/attempts.csv'],
                       capture_output=True)
    if r.returncode:
        return False
    rows = list(csv.DictReader(r.stdout.decode('utf-8').splitlines()))
    ended = {x['attempt_id'] for x in rows if x['kind'] in ('end', 'moved')}
    return any(x['kind'] == 'start' and x['attempt_id'] not in ended for x in rows)


def _sealed_ids(s03):
    """(st_dev, st_ino) of every sealed S03 file: an alias (hard link) is caught too (RSM2-7, RSM3-2)."""
    out = set()
    for d in ('sealed', 'sealed_test'):
        for n in os.listdir(os.path.join(str(s03), d)):
            st = os.stat(os.path.join(str(s03), d, n))
            out.add((st.st_dev, st.st_ino))
    return out


def _io_log(monkeypatch, repo=None, s03=None):
    events, real_open, real_commit = [], builtins.open, rc.commit_paths
    ids = _sealed_ids(s03) if s03 is not None else set()

    def op(p, *a, **k):
        q = str(p).replace('\\', '/')
        same = False
        if ids and isinstance(p, (str, os.PathLike)):
            try:
                st = os.stat(p)
                same = (st.st_dev, st.st_ino) in ids
            except OSError:
                pass
        if '/s03/sealed/' in q or '/s03/sealed_test/' in q or same:
            events.append(('open', str(p), _start_in_head(repo) if repo else None))
        return real_open(p, *a, **k)

    def cm(repo, rels, msg):
        events.append(('commit', msg))
        return real_commit(repo, rels, msg)
    monkeypatch.setattr(builtins, 'open', op)
    monkeypatch.setattr(rc, 'commit_paths', cm)
    return events


def test_no_sealed_file_is_opened_before_the_start_commit(tmp_path, dummy_dll, monkeypatch, fake_eval):
    """RSM-1: in every phase the first open of a sealed file comes after the commit of the start row; phases 1-2
    never open sealed_test/."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    for ph in rc.PHASES:
        ev = _io_log(monkeypatch, repo, s03)
        rc.run_phase(ph, str(s04), dummy_dll)
        start = next(i for i, e in enumerate(ev) if e[0] == 'commit' and f'start {ph}' in e[1])
        first = next(i for i, e in enumerate(ev) if e[0] == 'open')
        assert start < first, ev[:5]
        assert all(e[2] for e in ev if e[0] == 'open')           # RSM2-7: the start row is in HEAD at each open
        if ph != 'test':
            assert not any('/s03/sealed_test/' in e[1].replace('\\', '/') for e in ev if e[0] == 'open')
        monkeypatch.undo()
        monkeypatch.setattr(rc, 'EVALUATOR', fake_eval['factory'])
        monkeypatch.setattr(s4, 'safe_run_gate', fake_eval['srg'])
        monkeypatch.setattr(s3, 'run_in_new_process', fake_eval['rinp'])


def test_tool_check_on_a_real_s03_opens_no_sealed_file(tmp_path, dummy_dll, monkeypatch):
    repo, s03, s04 = setup(tmp_path, dummy_dll, fake=False)
    ev = _io_log(monkeypatch, s03=s03)
    with pytest.raises(rc.Stop, match='rsm_fake'):
        rc.run_phase('fit', str(s04), dummy_dll)
    assert not [e for e in ev if e[0] == 'open'] and not [e for e in ev if e[0] == 'commit']


def _commit_all(repo, path, msg='x'):
    subprocess.run(['git', '-C', str(repo), 'add', '--', str(path)], check=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '-q', '-m', msg, '--', str(path)], check=True)


def test_lock_edited_and_recommitted_stops(tmp_path, dummy_dll, fake_eval):
    """RSM-3: change coefs.csv, update fit/lock.yaml to match, commit both: the next phase stops (the files are
    not the blobs of the fit output commit)."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    fit = s04 / 'rsm' / 'fit'
    (fit / 'coefs.csv').write_text((fit / 'coefs.csv').read_text().replace('0.03', '0.031', 1))
    lk = json.load(open(fit / 'lock.yaml'))
    lk['files']['fit/coefs.csv'] = dict(zip(('sha256', 'sha256_lf'), rc.file_sha(str(fit / 'coefs.csv'))))
    (fit / 'lock.yaml').write_text(json.dumps(lk))
    _commit_all(repo, fit, 'edit')
    with pytest.raises(rc.Stop, match='output commit'):
        rc.run_phase('test', str(s04), dummy_dll)


def test_rerun_after_the_test_set_is_exploratory(tmp_path, dummy_dll, fake_eval):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    for ph in rc.PHASES:
        rc.run_phase(ph, str(s04), dummy_dll)
    meta = (s04 / 'rsm' / 'meta.yaml').read_text(encoding='utf-8')
    assert 'comparison_kind: "main"' in meta
    rc.run_phase('opt', str(s04), dummy_dll, abort_reason='re-run after the test set')
    assert 'comparison_kind: "exploratory"' in (s04 / 'rsm' / 'meta.yaml').read_text(encoding='utf-8')
    rc.run_phase('test', str(s04), dummy_dll)
    assert 'comparison_kind: "exploratory"' in (s04 / 'rsm' / 'meta.yaml').read_text(encoding='utf-8')


def test_attempts_append_only_and_interrupted_end(tmp_path, dummy_dll, monkeypatch, fake_eval):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    real = rc.write_rsm_meta

    def crash(*a, **k):
        raise KeyboardInterrupt                         # after the end row, before the commit
    monkeypatch.setattr(rc, 'write_rsm_meta', crash)
    with pytest.raises(KeyboardInterrupt):
        rc.run_phase('fit', str(s04), dummy_dll)
    monkeypatch.setattr(rc, 'write_rsm_meta', real)
    rc.release_run_lock(rc.acquire_run_lock(str(s04 / 'rsm' / rc.RUN_LOCK)))     # RSM3-3: the lock was freed
    with pytest.raises(rc.Stop, match='not committed'):
        rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('fit', str(s04), dummy_dll, abort_reason='interrupted before the commit')
    rc.run_phase('opt', str(s04), dummy_dll)
    p = s04 / 'rsm' / 'attempts.csv'
    lines = p.read_text().splitlines(True)
    p.write_text(''.join(lines[:1] + lines[2:]))            # a committed row removed
    with pytest.raises(rc.Stop, match='altered or removed'):
        rc.run_phase('test', str(s04), dummy_dll)


def test_no_flag_with_every_point_flagged(tmp_path, dummy_dll):
    """RSM-6: no point left for no_flag: reported as not fitted, the main fit goes on."""
    repo, s03, s04 = setup(tmp_path, dummy_dll, mods=lambda r: r.update(status='track_err') if r['v'] == 0.2 else None)
    rc.run_phase('fit', str(s04), dummy_dll)
    diag = (s04 / 'rsm' / 'fit' / 'fit_diag.yaml').read_text(encoding='utf-8')
    assert 'main/no_flag/f1: rank 0 < 36' in diag


@pytest.mark.parametrize('rows,lab', [
    ({0.4: [R(status='run_error', completed=''), R(completed='false', v=0.5)], 0.6: [R()]}, 'not robust over mu'),
    ({0.4: [R(e_perp_max=1e6, pen_e_perp_max=1)], 0.6: [R()]}, 'undetermined'),
    ({0.4: [R(status='track_err')], 0.6: [R()]}, 'inconclusive over mu'),
    # RSM2-5: a stored flag is kept even when another flag column is missing
    ({0.4: [{k: v for k, v in R(e_perp_max=1e6, pen_e_perp_max=1).items() if k != 'pen_e_bao_max'}], 0.6: [R()]},
     'undetermined'),
    ({0.4: [R()], 0.6: [R()]}, 'robust'),
    ({0.4: [{k: v for k, v in R(e_trk_max=0.5).items() if not k.startswith('pen_')}], 0.6: [R()]},
     'not robust over mu'),                                  # three_mu.csv rows: no pen_* columns
])
def test_mu_label_order(rows, lab):
    """RSM-10: an observed violation is never hidden by a run_error at another speed; a penalty alone is not a
    violation."""
    assert rc.mu_label_from_rows(rows)[0].startswith(lab)


def test_dll_changed_during_the_phase_stops(tmp_path, dummy_dll, fake_eval, monkeypatch):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    rc.run_phase('fit', str(s04), dummy_dll)
    base = rc.EVALUATOR

    def swap(inp, *a):
        open(inp.dll, 'wb').write(b'another dll')
        return base(inp, *a)
    monkeypatch.setattr(rc, 'EVALUATOR', swap)
    with pytest.raises(rc.Stop, match='DLL'):
        rc.run_phase('opt', str(s04), dummy_dll)


def test_one_run_at_a_time_and_no_stray_file(tmp_path, dummy_dll, fake_eval, monkeypatch):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    rc.run_phase('fit', str(s04), dummy_dll)
    base = rc.EVALUATOR

    def stray(inp, *a):
        os.makedirs(os.path.join(inp.out, 'confirm'), exist_ok=True)
        open(os.path.join(inp.out, 'confirm', 'stray.txt'), 'w').write('x')
        return base(inp, *a)
    monkeypatch.setattr(rc, 'EVALUATOR', stray)
    with pytest.raises(rc.Stop, match='not written by this run'):
        rc.run_phase('opt', str(s04), dummy_dll)
    tracked = git_out(repo, 'ls-files', '05_DATA/results/S04/run01/rsm')
    assert 'stray.txt' not in tracked and 'reason.txt' in tracked             # RSM2-4: the stray file not committed
    assert 'not committed (not written by this run)' in att(s04)[-1]['reason']


def test_clip_beyond_rounding_is_not_converged(tmp_path, dummy_dll, fake_eval, monkeypatch):
    import scipy.optimize as so
    real = so.minimize

    def fake_min(fun, x0, **kw):
        r = real(fun, x0, **kw)
        r.x = np.array(r.x, dtype=float)
        r.x[0] = 1.01
        r.success = True
        return r
    monkeypatch.setattr(so, 'minimize', fake_min)
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    st = list(csv.DictReader(open(s04 / 'rsm' / 'opt' / 'starts.csv')))
    assert all(r['converged'] == 'false' and float(r['clip_move']) == pytest.approx(0.01) for r in st)
    res = list(csv.DictReader(open(s04 / 'rsm' / 'confirm' / 'result.csv')))[0]
    assert res['state'].startswith('R0 (i)')


def test_candidate_filter_bounds_and_dedup():
    f1s, e = 0.03, rc.EPS
    pts = [dict(code=[0.0] * 7, f1_hat=f1s + rc.F1_TOL + e, f2_hat=1.0),
           dict(code=[0.5] * 7, f1_hat=f1s + rc.F1_TOL + 2 * e, f2_hat=0.5)]
    assert [c['f2_hat'] for c in rc.candidates(pts, f1s)] == [1.0]            # eps inclusive, 2 eps out
    f1s = 0.095
    pts = [dict(code=[0.0] * 7, f1_hat=rc.D + e, f2_hat=1.0), dict(code=[0.5] * 7, f1_hat=rc.D + 2 * e, f2_hat=0.5)]
    assert [c['f2_hat'] for c in rc.candidates(pts, f1s)] == [1.0]
    a = [0.0] * 7
    at = [rc.DEDUP] + [0.0] * 6
    near = [rc.DEDUP * 0.99] + [0.0] * 6
    got = rc.candidates([dict(code=a, f1_hat=0.03, f2_hat=1.0), dict(code=at, f1_hat=0.03, f2_hat=1.0),
                         dict(code=near, f1_hat=0.03, f2_hat=1.0)], 0.03)
    assert [c['code'] for c in got] == [a, at]                                 # same f: codes in order; near dropped
    got = rc.candidates([dict(code=near, f1_hat=0.03, f2_hat=1.0), dict(code=at, f1_hat=0.03, f2_hat=1.0),
                         dict(code=a, f1_hat=0.03, f2_hat=1.0)], 0.03)
    assert [c['code'] for c in got] == [a, at]                                 # RSM2-7: input order reversed


def test_edge_side_and_source_columns(tmp_path, dummy_dll, fake_eval):
    repo, s03, s04 = setup(tmp_path, dummy_dll, q1=q1_flat, q2=q2_concave)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    c = list(csv.DictReader(open(s04 / 'rsm' / 'opt' / 'candidates.csv')))[0]
    sides = c['edge_side'].split()
    assert len(sides) == 7 and all(s.endswith((':L', ':H')) for s in sides)
    code = [float(x) for x in json.loads(c['code'])]
    assert [s.endswith(':H') for s in sides] == [x > 0 for x in code]


def test_two_controllers(tmp_path, dummy_dll, fake_eval):
    repo, s03, s04 = setup(tmp_path, dummy_dll, ctrls=('PID', 'ADRC'))
    for ph in rc.PHASES:
        rc.run_phase(ph, str(s04), dummy_dll)
    res = list(csv.DictReader(open(s04 / 'rsm' / 'confirm' / 'result.csv')))
    assert [r['controller'] for r in res] == ['PID', 'ADRC'] and all(r['rsm_class'] == 'R2' for r in res)
    te = list(csv.DictReader(open(s04 / 'rsm' / 'test' / 'test_errors.csv')))
    assert [r['controller'] for r in te] == ['PID', 'ADRC']


def test_links_are_refused(tmp_path, dummy_dll):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    try:
        os.symlink(str(s03 / 'sealed_test' / 'runs_test.csv'), str(s04 / 'post_checks' / 'leak.csv'))
    except (OSError, NotImplementedError):
        pytest.skip('no symlink permission on this machine')
    with pytest.raises(rc.Stop, match='link'):
        rc.run_phase('fit', str(s04), dummy_dll)


def _main_mode(tmp_path, dummy_dll, monkeypatch):
    repo = make_repo(tmp_path)
    s03 = make_s03(tmp_path, dll=dummy_dll, fake=False)
    s04 = make_s04(repo, s03, dummy_dll)
    lk = json.load(open(s04 / 'lock.yaml'))
    lk['common'].update(tool_check=False, unsealed=True, fingerprint=s4.fingerprint(dummy_dll, str(s03), unsealed=True))
    for c in lk['controllers'].values():
        c['tool_check'] = False
    (s04 / 'lock.yaml').write_text(json.dumps(lk, indent=1, default=str))
    monkeypatch.setattr(s3, 'git_state', lambda repo: dict(commit='abc1234', branch='x', dirty=False, modified=[]))
    return repo, s03, s04


def test_main_run_needs_the_root_repo_and_committed_s04(tmp_path, dummy_dll, monkeypatch, fake_eval):
    """RSM-2: a main run refuses a repository whose origin is not liam-zuu/hil-thesis and S04 files that are
    not committed; with both, the fit runs."""
    repo, s03, s04 = _main_mode(tmp_path, dummy_dll, monkeypatch)
    monkeypatch.setattr(rc, 'check_script_blob', lambda hil: None)
    with pytest.raises(rc.Stop, match='main RSM runs only in'):          # not the root folder
        rc.run_phase('fit', str(s04), dummy_dll)
    monkeypatch.setattr(rc, 'ROOT_PATH', str(repo))
    with pytest.raises(rc.Stop, match='origin'):
        rc.run_phase('fit', str(s04), dummy_dll)
    subprocess.run(['git', '-C', str(repo), 'remote', 'add', 'origin', 'https://github.com/liam-zuu/hil-thesis.git'],
                   check=True)
    with pytest.raises(rc.Stop, match='not committed'):
        rc.run_phase('fit', str(s04), dummy_dll)
    _commit_all(repo, s04, 's04 run')
    rc.run_phase('fit', str(s04), dummy_dll)
    assert 'tool_check: false' in (s04 / 'rsm' / 'meta.yaml').read_text(encoding='utf-8')


def test_script_blob_check(tmp_path):
    """RSM-12: an edit hidden by assume-unchanged is still caught (content against the HEAD blob)."""
    r = make_repo(tmp_path)
    (r / 'fwsim').mkdir()
    f = r / 'fwsim' / 'rsm_compare.py'
    f.write_text('x = 1\n')
    _commit_all(r, f)
    rc.check_script_blob(str(r))
    subprocess.run(['git', '-C', str(r), 'update-index', '--assume-unchanged', 'fwsim/rsm_compare.py'], check=True)
    f.write_text('x = 2\n')
    assert subprocess.run(['git', '-C', str(r), 'status', '--porcelain'], capture_output=True, text=True).stdout == ''
    with pytest.raises(rc.Stop, match='HEAD blob'):
        rc.check_script_blob(str(r))


# ── Codex round 2 fixes (RSM2-1 .. RSM2-7) ───────────────────────────────────────────

def test_attempts_history_edited_and_recommitted_stops(tmp_path, dummy_dll, fake_eval):
    """RSM2-2: remove the test rows of attempts.csv and commit that: the next run stops (history not append only),
    so no new 'main' test can be made."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    for ph in rc.PHASES:
        rc.run_phase(ph, str(s04), dummy_dll)
    p = s04 / 'rsm' / 'attempts.csv'
    rows = list(csv.DictReader(open(p, encoding='utf-8')))
    s3.write_csv(str(p), [r for r in rows if r['phase'] != 'test'])
    _commit_all(repo, p, 'drop the test rows')
    with pytest.raises(rc.Stop, match='history'):
        rc.run_phase('opt', str(s04), dummy_dll, abort_reason='try a new main test')


def test_output_commit_ignores_a_quoted_message(tmp_path, dummy_dll, fake_eval):
    """RSM2-3: a later commit whose message quotes the fit output message does not count as an output commit."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    rc.run_phase('fit', str(s04), dummy_dll)
    aid = att(s04)[0]['attempt_id']
    rc.run_phase('opt', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll,
                 abort_reason=f'checking rsm_compare fit done 05_DATA/results/S04/run01 ({aid})')
    assert (s04 / 'rsm' / 'opt_lock.yaml').is_file()                        # the fit lock still passed
    inp = rc.Inputs(str(s04), dummy_dll)
    c = rc.output_commit(inp, 'fit', aid)
    assert git_out(repo, 'log', '-1', '--format=%s', c).startswith('fw: rsm_compare fit done')


_HOLD = """import os, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)
if os.name == 'nt':
    import msvcrt
    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
else:
    import fcntl
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
print('held', flush=True)
sys.stdin.read()
"""


def test_lock_held_by_another_process_then_killed(tmp_path, dummy_dll, fake_eval):
    """RSM3-3, RSM3-4, RSM2-7: a process holding the OS lock (any content) stops a run, with or without
    --abort-previous, before any row is written; once that process is killed the system has freed the lock: the
    next run goes on, no file to delete by hand, no --abort-previous."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    (s04 / 'rsm').mkdir()
    lk = s04 / 'rsm' / rc.RUN_LOCK
    p = subprocess.Popen([sys.executable, '-c', _HOLD, str(lk)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True)
    try:
        assert p.stdout.readline().strip() == 'held'
        with pytest.raises(rc.Stop, match='is locked'):
            rc.run_phase('fit', str(s04), dummy_dll)
        with pytest.raises(rc.Stop, match='is locked'):
            rc.run_phase('fit', str(s04), dummy_dll, abort_reason='the other run looks dead')
        assert not (s04 / 'rsm' / 'attempts.csv').exists()
    finally:
        p.kill()                                            # a forced death: no cleanup code runs
        p.wait()
    rc.run_phase('fit', str(s04), dummy_dll)
    assert att(s04)[-1]['status'] == 'done' and lk.exists()


def test_lock_file_left_behind_is_no_lock_and_a_phase_holds_it(tmp_path, dummy_dll, fake_eval, monkeypatch):
    """RSM3-3: an empty lock file (power loss right after it was made) or one naming a live process id does not
    block the next run; RSM3-4: a second run started while a phase runs is refused for the whole phase."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    (s04 / 'rsm').mkdir()
    lk = s04 / 'rsm' / rc.RUN_LOCK
    lk.write_text('')
    rc.run_phase('fit', str(s04), dummy_dll)
    lk.write_text(f'{os.getpid()} 2026-11-16 00:00:00\n')
    real, inner = rc.run_opt, []

    def nested(inp, ctx, log=print):
        with pytest.raises(rc.Stop, match='is locked'):
            rc.run_phase('test', str(s04), dummy_dll, abort_reason='a second run')
        out = real(inp, ctx, log)
        with pytest.raises(rc.Stop, match='is locked'):
            rc.run_phase('opt', str(s04), dummy_dll)
        inner.append(1)
        return out
    monkeypatch.setattr(rc, 'run_opt', nested)
    rc.run_phase('opt', str(s04), dummy_dll)
    assert inner and [(r['phase'], r['status']) for r in att(s04) if r['kind'] == 'end'] == [('fit', 'done'),
                                                                                             ('opt', 'done')]


def test_merge_that_drops_the_test_rows_stops(tmp_path, dummy_dll, fake_eval):
    """RSM3-1: test run on one branch, then a merge that keeps the attempts.csv of a branch made before the test
    (git log without --full-history follows that parent only): no new test, main or not, can be made."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    main = git_out(repo, 'rev-parse', '--abbrev-ref', 'HEAD').strip()
    git_out(repo, 'branch', 'before_test')
    rc.run_phase('test', str(s04), dummy_dll)                # the test set is opened on <main>
    git_out(repo, 'checkout', '-q', 'before_test')
    git_out(repo, 'merge', '-q', '--no-ff', '-s', 'ours', main, '-m', 'merge, attempts.csv of before_test')
    assert git_out(repo, 'merge-base', '--is-ancestor', main, 'HEAD') == ''
    assert 'test' not in {r['phase'] for r in att(s04)}       # HEAD's attempts.csv has no test row
    with pytest.raises(rc.Stop, match='history'):
        rc.run_phase('test', str(s04), dummy_dll)
    with pytest.raises(rc.Stop, match='history'):
        rc.run_phase('opt', str(s04), dummy_dll, abort_reason='a new main test')


def test_links_refused_before_any_open(tmp_path, dummy_dll, monkeypatch):
    """RSM2-1: a link at rsm/attempts.csv (simulated: os.path.islink true) stops the run before the file is opened."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    (s04 / 'rsm').mkdir()
    (s04 / 'rsm' / 'attempts.csv').write_text('x\n')
    target = str(s04 / 'rsm' / 'attempts.csv')
    real_islink, real_open = os.path.islink, builtins.open
    opened = []
    monkeypatch.setattr(os.path, 'islink', lambda p: os.path.normcase(str(p)) == os.path.normcase(target)
                        or real_islink(p))

    def op(p, *a, **k):
        if os.path.normcase(str(p)) == os.path.normcase(target):
            opened.append(p)
        return real_open(p, *a, **k)
    monkeypatch.setattr(builtins, 'open', op)
    with pytest.raises(rc.Stop, match='link'):
        rc.run_phase('fit', str(s04), dummy_dll)
    assert not opened


def test_edge_source_and_limit(tmp_path, dummy_dll, fake_eval):
    """RSM2-7, RSM3-6: edge_src and edge_limit from the S03 v8.3 level sources (H of every factor: 'het luoi noi');
    the candidate has factors on both sides, so the H branch is checked with a non-empty edge_limit."""
    repo = make_repo(tmp_path)
    s03 = make_s03(tmp_path, q1=q1_flat, q2=q2_mixed, dll=dummy_dll)
    rows = list(csv.DictReader(open(s03 / 'levels.csv')))
    for r in rows:
        r['L_nguon'], r['H_nguon'] = '0', 'hết lưới nối'
    s3.write_csv(str(s03 / 'levels.csv'), rows)
    s04 = make_s04(repo, s03, dummy_dll, q1=q1_flat, q2=q2_mixed)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    c = list(csv.DictReader(open(s04 / 'rsm' / 'opt' / 'candidates.csv', encoding='utf-8')))[0]
    code = [float(x) for x in json.loads(c['code'])]
    n_high = sum(x > 0 for x in code)
    assert 0 < n_high < 7                                      # RSM3-6: both sides present
    assert c['edge_src'].count(':H:hết lưới nối') == n_high and c['edge_src'].count(':L:0') == 7 - n_high
    assert (c['edge_limit'] != '') == (n_high > 0) and c['edge_limit'].count(':H') == n_high


def test_hard_links_are_refused(tmp_path, dummy_dll):
    """RSM3-2: a second hard link (islink false, realpath unchanged) to a sealed_test/ file, at rsm/.run_lock with
    --abort-previous or at an S04 file, stops the run before the start row; the sealed file is not changed."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    src = s03 / 'sealed_test' / 'runs_test.csv'
    before = src.read_bytes()
    (s04 / 'rsm').mkdir()
    try:
        os.link(str(src), str(s04 / 'rsm' / rc.RUN_LOCK))
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip('no hard link on this file system')
    with pytest.raises(rc.Stop, match='hard link'):
        rc.run_phase('fit', str(s04), dummy_dll, abort_reason='a run died')
    assert src.read_bytes() == before
    (s04 / 'rsm' / rc.RUN_LOCK).unlink()
    os.link(str(src), str(s04 / 'post_checks' / 'leak.csv'))
    with pytest.raises(rc.Stop, match='hard link'):
        rc.run_phase('fit', str(s04), dummy_dll)
    assert not (s04 / 'rsm' / 'attempts.csv').exists() and src.read_bytes() == before


def _dir_link(target, link):
    """A directory symlink, or a junction on Windows when symlinks need a privilege; False when neither works."""
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        pass
    if os.name == 'nt':
        import _winapi
        try:
            _winapi.CreateJunction(str(target), str(link))
            return True
        except OSError:
            pass
    return False


def test_nested_directory_link_is_refused(tmp_path, dummy_dll, fake_eval):
    """RSM3-2: rsm/confirm/_work made a directory link (or junction) before the opt phase stops it before the start
    row, whatever the link points to (here an empty folder outside rsm/)."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    rc.run_phase('fit', str(s04), dummy_dll)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (s04 / 'rsm' / 'confirm').mkdir()
    if not _dir_link(outside, s04 / 'rsm' / 'confirm' / '_work'):
        pytest.skip('no directory link on this machine')
    n = len(att(s04))
    with pytest.raises(rc.Stop, match='directory link'):
        rc.run_phase('opt', str(s04), dummy_dll)
    assert len(att(s04)) == n and not list(outside.iterdir())


# ── real fwsim (needs the DLL) ───────────────────────────────────────────────────────

def _levels_near_defaults():
    fw = fwsim.FirmwareSim(_DLL)
    return {c: {n: ((v * 0.8, v * 1.25) if v > 0 else (0.0, 0.1)) for n, v in s3.defaults(fw, c).items()}
            for c in s3.CONTROLLERS}


@needs_dll
def test_confirm_uses_the_ga_evaluator(tmp_path, monkeypatch):
    """The default evaluator is s04_ga.Evaluator on the DLL: the same log SHA-256 as a GA evaluation of the
    same gains."""
    repo = make_repo(tmp_path)
    s03 = make_s03(tmp_path, dll=_DLL, levels=_levels_near_defaults(), gate=(0.2,))
    s04 = make_s04(repo, s03, _DLL, speeds=(0.2,))
    inp = rc.Inputs(str(s04), _DLL)
    cand = s04 / 'cands.csv'
    dom = inp.ctrl_lock('PID')['domain']
    g, edge, _ = rc.to_gains(inp, 'PID', [0.0] * 7)
    s3.write_csv(str(cand), [dict(controller='PID', cand_id=1, code=json.dumps(['0.0'] * 7),
                                  **s4.gain_cols(dom, dict(x=[g[d['name']] for d in dom])))])
    x, tried, ev, _ = rc.confirm(inp, 'PID', str(cand), str(tmp_path / 'w'), {}, print)
    ref = s4.Evaluator(fwsim.FirmwareSim(_DLL), _DLL, 'PID', [0.2], [0.5], str(tmp_path / 'w2'))
    s = ref.candidate({n: float(repr(v)) for n, v in g.items()}, 1, 1, 0)
    assert [e['log_sha256'] for e in ev] == [e['log_sha256'] for e in ref.evals] and ev[0]['log_sha256']
    assert all(tried[0][k] == s[k] for k in ('f1', 'f2', 'g1', 'g2', 'g3', 'g4', 'g5'))       # RSM-16: same score
    assert (tried[0]['label'] == 'pass') == rc.feasible(s)


@needs_dll
def test_end_to_end_with_a_real_s04_tool_check(tmp_path, monkeypatch, capsys):
    """Tool check end to end: synthetic S03 (rsm_fake) -> s04_ga --tool-check trial, main, post (real fwsim)
    -> rsm fit, opt (real confirmation), test. Never main numbers (L82)."""
    repo = make_repo(tmp_path)
    lv = _levels_near_defaults()
    s03 = make_s03(tmp_path, dll=_DLL, levels=lv, gate=(0.2,))
    out = repo / '05_DATA' / 'results' / 'S04' / 'run01'
    common = ['--tool-check', '--controller', 'PID', '--s03', str(s03), '--pop', '4', '--gen-base', '1',
              '--gen-max', '1', '--dll', _DLL, '--out', str(out)]
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'trial'] + common)
    assert s4.main() in (0, 2)
    monkeypatch.setattr(sys, 'argv', ['s04_ga.py', '--phase', 'main', '--seeds', '1'] + common)
    if s4.main() != 0:
        pytest.skip('the tiny GA found no feasible send_to_HIL row')
    import argparse
    assert s4.post(argparse.Namespace(out=str(out), dll=_DLL)) == 0
    for ph in rc.PHASES:
        assert rc.main(['--phase', ph, '--s04', str(out), '--dll', _DLL]) == 0
    res = list(csv.DictReader(open(out / 'rsm' / 'confirm' / 'result.csv')))
    assert res and res[0]['rsm_class'] in ('R0', 'R1', 'R2', 'R3')
    tried = list(csv.DictReader(open(out / 'rsm' / 'check' / 'candidates_tried.csv')))
    n = int(res[0]['n_tried'])
    assert len(tried) == n <= 5
    assert sum(int(res[0][k]) for k in ('n_violation_observed', 'n_run_error_excluded', 'n_inconclusive')) == \
        n - (res[0]['state'] == 'x_RSM')
    meta = open(out / 'rsm' / 'meta.yaml', encoding='utf-8').read()
    assert 'not_main_numbers: true' in meta
    st = git_out(repo, 'status', '--porcelain', '--', '05_DATA/results/S04/run01/rsm').splitlines()
    assert st == ['?? 05_DATA/results/S04/run01/rsm/' + rc.RUN_LOCK]      # RSM3-3: only the lock file, untracked
    cost = list(csv.DictReader(open(out / 'rsm' / 'check' / 'cost.csv')))
    main = next(r for r in cost if r['item'] == 'S04 main')
    assert main['n_runs_planned'] == main['n_runs_actual']          # the seeds run (1), not the locked ones (5)
