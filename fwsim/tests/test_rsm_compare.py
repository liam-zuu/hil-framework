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
# S04 v6.8: the synthetic levels lie inside the range table (kp_wz ceiling 4.6)
LEVELS = {c: {n: (((0.5, 4.0) if n == 'kp_wz' else (0.5, 8.0)) if k % 2 == 0 else (1.0, 3.0))
              for k, n in enumerate(s3.FACTORS[c])} for c in s3.CONTROLLERS}


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


# S04 v6.8 12.3: an ADRC domain the LADRC triangle cuts (corner 180 + 2 x 40 = 260), and responses pulled to it
LEVELS_TRI = dict(LEVELS, ADRC=dict(zip(s3.FACTORS['ADRC'], [(20.0, 180.0), (10.0, 40.0), (0.5, 8.0), (1.0, 3.0),
                                                             (0.5, 8.0), (1.0, 3.0), (0.5, 8.0)])))
T_TRI = np.array([1.0, 1.0, 0.3, 0.3, 0.3, 0.3, 0.3])


def q1_tri(c, mu=0.5, v=0.2):
    c = np.asarray(c, dtype=float)
    return 0.03 + 0.0005 * float(np.sum(c ** 2)) - 0.001 * float(c[0] + c[1])


def q2_tri(c, mu=0.5, v=0.2):
    c = np.asarray(c, dtype=float)
    return 5.0 + 0.5 * float(np.sum((c - T_TRI) ** 2))


def _dom(levels, ctrl):
    return [dict(name=n, L=L, H=H, scale='log' if L > 0 and H / L >= 4 else 'lin')
            for n, (L, H) in ((n, levels[n]) for n in s3.FACTORS[ctrl])]


def _gains(dom, code):
    return {d['name']: s3.decode(float(c), d['L'], d['H'], d['scale']) for d, c in zip(dom, code)}


def _code_at_sum(dom, total):
    """A code whose decoded xy sum omega_c + 2 omega_o is `total` up to rounding: omega_o_xy at H (code 1),
    omega_c_xy solved from its log coding; the other factors at 0."""
    code = [0.0] * 7
    code[1] = 1.0
    code[0] = 2.0 * math.log((total - 2.0 * dom[1]['H']) / math.sqrt(dom[0]['L'] * dom[0]['H'])) / \
        math.log(dom[0]['H'] / dom[0]['L'])
    return code


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
    s03_kw = {k: kw.pop(k) for k in list(kw) if k in ('q1', 'q2', 'mods', 'fake', 'test_mods', 'levels')}
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


@pytest.mark.parametrize('where', ['both', 'step2', 'mixed', 'corner'])
def test_adrc_solver_result_over_the_margin_is_not_used(tmp_path, dummy_dll, fake_eval, monkeypatch, where):
    """C3-11: a solver result above the 199.99 margin on the binary64 gains is not converged whatever
    SLSQP reports: it sets neither f1* nor a candidate."""
    import scipy.optimize as so
    dom = _dom(LEVELS_TRI['ADRC'], 'ADRC')
    over, inside = _code_at_sum(dom, 199.9900000005), _code_at_sum(dom, 199.9899999995)
    s = rc.triangle_sums('ADRC', _gains(dom, over))['xy']                     # the premises of the case
    assert rc.TRI_MAX < s < rc.TRI_MAX + 1e-9
    assert s == pytest.approx(199.9900000005, abs=1e-11)
    assert s4.gain_check('ADRC', _gains(dom, over)) == ([], [])               # passes the float32 check
    assert rc.triangle_sums('ADRC', _gains(dom, inside))['xy'] < rc.TRI_MAX

    def pick(step2, x0):
        if where == 'both':
            return over
        if where == 'corner':
            return list(T_TRI) if step2 else [0.0] * 7
        if not step2:
            return [0.0] * 7
        if where == 'step2':
            return over
        return over if x0[0] > 0 else inside                                # 'mixed'

    def fake_min(fun, x0, **kw):
        step2 = len(kw.get('constraints') or []) == 4
        return so.OptimizeResult(x=np.array(pick(step2, x0), dtype=float), success=True, status=0,
                                 message='fake', nit=1)
    monkeypatch.setattr(so, 'minimize', fake_min)
    repo, s03, s04 = setup(tmp_path, dummy_dll, ctrls=('ADRC',), levels=LEVELS_TRI, q1=q1_tri, q2=q2_tri)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    res = list(csv.DictReader(open(s04 / 'rsm' / 'confirm' / 'result.csv')))[0]
    st = list(csv.DictReader(open(s04 / 'rsm' / 'opt' / 'starts.csv')))
    cands = list(csv.DictReader(open(s04 / 'rsm' / 'opt' / 'candidates.csv')))
    if where == 'both':                                                     # old code: M = 1
        assert res['state'].startswith('R0 (i)') and res['M'] == '0' and res['f1_hat_star'] == ''
        assert res['n_tried'] == '0' and res['n_ngoai_tam_giac'] == '0' and fake_eval['calls'] == 0
        assert not cands
        assert all(r['success'] == 'true' and r['converged'] == 'false' and r['tri_ok'] == 'false'
                   and json.loads(r['tri_sums'])['xy'] > rc.TRI_MAX for r in st)
    elif where == 'step2':
        assert res['state'].startswith('R0 (i)') and res['M'] == '0' and res['f1_hat_star'] != ''
        assert all((r['converged'] == 'true' and r['tri_ok'] == 'true') if r['step'] == '1'
                   else r['converged'] == 'false' for r in st)
    elif where == 'mixed':                                                  # old code keeps `over`
        assert res['M'] == '1' and res['n_ngoai_tam_giac'] == '0' and len(cands) == 1
        assert float(cands[0]['omega_c_xy']) + 2 * float(cands[0]['omega_o_xy']) <= rc.TRI_MAX
    else:                                                                   # 'corner'
        assert res['state'].startswith('R0 (i)') and res['M'] == '0' and res['n_ngoai_tam_giac'] == '0'


def test_adrc_candidate_outside_the_triangle_is_dropped_before_m(tmp_path, dummy_dll, fake_eval, monkeypatch):
    """S04 v6.8 12.3 (ii), reworked for C3-11: the corner used before is now refused by the binary64
    check; the float32 drop is exercised with a stubbed s4.gain_check, since with the 0.01 margin no
    real point passes binary64 and fails float32. The candidate with the smaller f2_hat is dropped
    before M is counted; the kept boundary solution is still ranked cand_id 1."""
    import scipy.optimize as so
    dom = _dom(LEVELS_TRI['ADRC'], 'ADRC')
    kept = _code_at_sum(dom, 199.5)
    drop = _code_at_sum(dom, 199.98)                            # both inside the margin; drop smaller f2_hat

    def fake_min(fun, x0, **kw):
        step2 = len(kw.get('constraints') or []) == 4
        x = drop if step2 and x0[0] > 0 else kept               # step 2, part of the starts
        return so.OptimizeResult(x=np.array(x, dtype=float), success=True, status=0, message='fake', nit=1)
    monkeypatch.setattr(so, 'minimize', fake_min)
    real_check = s4.gain_check

    def stub(ctrl, g):
        t, tri = real_check(ctrl, g)
        return t, tri + (['float32 triangle (stub)'] if g['omega_c_xy'] > 119.9 else [])
    monkeypatch.setattr(s4, 'gain_check', stub)
    repo, s03, s04 = setup(tmp_path, dummy_dll, ctrls=('ADRC',), levels=LEVELS_TRI, q1=q1_tri, q2=q2_tri)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    cands = list(csv.DictReader(open(s04 / 'rsm' / 'opt' / 'candidates.csv')))
    assert len(cands) == 1 and cands[0]['cand_id'] == '1'
    row = cands[0]
    assert float(row['omega_c_xy']) + 2 * float(row['omega_o_xy']) == pytest.approx(199.5, abs=1e-9)
    assert float(row['omega_c_xy']) + 2 * float(row['omega_o_xy']) <= rc.TRI_MAX
    res = list(csv.DictReader(open(s04 / 'rsm' / 'confirm' / 'result.csv')))[0]
    assert res['state'] == 'x_RSM' and res['M'] == '1' and res['n_ngoai_tam_giac'] == '1'
    assert res['n_tried'] == '1' and res['ga_in_tri_band'] == 'false'
    assert set(json.loads(res['ga_tri_sums'])) == {'xy', 'wz'}


def test_adrc_every_candidate_outside_the_triangle_is_state_ii(tmp_path, dummy_dll, fake_eval, monkeypatch):
    """S04 v6.8 12.3 (ii), reworked for C3-11 (the float32 drop stubbed, as above): M = 0 after the
    drop is failure state (ii): no evaluator call, no candidate row, the axial sensitivity has no
    point; the GA point inside the band still reports ga_in_tri_band true."""
    import scipy.optimize as so
    dom = _dom(LEVELS_TRI['ADRC'], 'ADRC')
    kept = _code_at_sum(dom, 199.5)
    drop = _code_at_sum(dom, 199.98)

    def fake_min(fun, x0, **kw):
        step2 = len(kw.get('constraints') or []) == 4           # step 2 of ADRC, every start
        return so.OptimizeResult(x=np.array(drop if step2 else kept, dtype=float), success=True, status=0,
                                 message='fake', nit=1)
    monkeypatch.setattr(so, 'minimize', fake_min)
    real_check = s4.gain_check

    def stub(ctrl, g):
        t, tri = real_check(ctrl, g)
        return t, tri + (['float32 triangle (stub)'] if g['omega_c_xy'] > 119.9 else [])
    monkeypatch.setattr(s4, 'gain_check', stub)
    repo, s03, s04 = setup(tmp_path, dummy_dll, ctrls=('ADRC',), levels=LEVELS_TRI, q1=q1_tri, q2=q2_tri)
    p = s04 / 'selected.csv'
    sel = list(csv.DictReader(open(p, encoding='utf-8')))
    row = next(r for r in sel if r['kind'] == 'send_to_HIL')
    row['omega_c_xy'] = repr(199.995 - 2 * float(row['omega_o_xy']))   # x_GA inside the band (v6.8 12.3)
    s3.write_csv(str(p), sel)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    res = list(csv.DictReader(open(s04 / 'rsm' / 'confirm' / 'result.csv')))[0]
    assert res['state'].startswith('R0 (ii)') and res['rsm_class'] == 'R0'
    assert res['M'] == '0' and res['n_ngoai_tam_giac'] == '1' and res['n_tried'] == '0'
    assert fake_eval['calls'] == 0
    assert not list(csv.DictReader(open(s04 / 'rsm' / 'opt' / 'candidates.csv')))
    ax = list(csv.DictReader(open(s04 / 'rsm' / 'check' / 'axial_sens.csv')))
    assert ax[0]['at'] == 'none'
    assert res['ga_in_tri_band'] == 'true'


def test_pid_result_has_no_triangle_columns_filled(tmp_path, dummy_dll, fake_eval):
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    res = list(csv.DictReader(open(s04 / 'rsm' / 'confirm' / 'result.csv')))[0]
    assert res['n_ngoai_tam_giac'] == '0' and res['ga_tri_sums'] == '' and res['ga_in_tri_band'] == ''
    assert res['state'] == 'x_RSM'


def test_ga_points_count_ngoai_bang_as_unlabelled(tmp_path, dummy_dll, fake_eval):
    """S04 v6.8 2: a GA candidate with status ngoai_bang was not run (no evals row): unlabelled, its own reason."""
    repo, s03, s04 = setup(tmp_path, dummy_dll, ctrls=('ADRC',))
    p = s04 / 'history_ADRC.csv'
    rows = list(csv.DictReader(open(p, encoding='utf-8')))
    rows.append(dict(rows[0], cand_id='99', status='ngoai_bang'))
    s3.write_csv(str(p), rows)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    summ = next(r for r in csv.DictReader(open(s04 / 'rsm' / 'check' / 'ga_errors.csv')) if r['kind'] == 'summary')
    assert json.loads(summ['unlabelled_by_reason']) == {'ngoai_bang (not run)': 1}
    assert summ['n_unlabelled'] == '1' and summ['n_labelled'] == '2'


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
    st = [x for x in st if x.strip()]
    assert st and all(x.startswith('?? ') and '/rsm/aborted/' in x and '/fit/' in x for x in st)
    assert 'fit/coefs.csv' in rows[3]['reason']
    # a committed, done phase moved aside: its removal and its moved copy are committed
    rc.run_phase('opt', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll, abort_reason='exploratory re-run')
    st2 = [x for x in git_out(repo, 'status', '--porcelain', '--untracked-files=all', '--',
                                '05_DATA/results/S04/run01/rsm').split('\n') if x.strip()]
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
    """{(st_dev, st_ino): 'sealed' | 'sealed_test'} of every sealed S03 file: an alias (hard link) is caught too and
    keeps the kind of the file it aliases (RSM2-7, RSM3-2, RSM4-3)."""
    out = {}
    for d in ('sealed', 'sealed_test'):
        for n in os.listdir(os.path.join(str(s03), d)):
            st = os.stat(os.path.join(str(s03), d, n))
            out[(st.st_dev, st.st_ino)] = d
    return out


def _sealed_kind(p, ids):
    """'sealed' | 'sealed_test' | None for an opened path: by file identity, else by the path text."""
    if isinstance(p, (str, bytes, os.PathLike)):                 # RSM5-3: bytes paths too
        try:
            st = os.stat(p)
            if (st.st_dev, st.st_ino) in ids:
                return ids[(st.st_dev, st.st_ino)]
        except OSError:
            pass
        q = os.fsdecode(os.fspath(p)).replace('\\', '/')
        if '/s03/sealed_test/' in q:
            return 'sealed_test'
        if '/s03/sealed/' in q:
            return 'sealed'
    return None


def _io_log(monkeypatch, repo=None, s03=None):
    """Events ('open', path, start row in HEAD, kind) of every builtins.open and os.open of a sealed file, and
    ('commit', message) of every commit of the script (RSM4-3: os.open too)."""
    events, real_open, real_osopen, real_commit = [], builtins.open, os.open, rc.commit_paths
    ids = _sealed_ids(s03) if s03 is not None else {}

    def note(p):
        kind = _sealed_kind(p, ids)
        if kind:
            events.append(('open', str(p), _start_in_head(repo) if repo else None, kind))

    def op(p, *a, **k):
        note(p)
        return real_open(p, *a, **k)

    def oso(p, *a, **k):
        note(p)
        return real_osopen(p, *a, **k)

    def cm(repo, rels, msg):
        events.append(('commit', msg))
        return real_commit(repo, rels, msg)
    monkeypatch.setattr(builtins, 'open', op)
    monkeypatch.setattr(os, 'open', oso)
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
            assert not any(e[3] == 'sealed_test' for e in ev if e[0] == 'open')     # RSM4-3: by kind
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
    rc.release_run_lock(rc.acquire_run_lock(str(s04 / 'rsm')))     # RSM3-3: the lock was freed
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


def test_triangle_constraints_pid_none_adrc_two_axes():
    """S04 v6.8 12.3 (G-Q4): the SLSQP constraints omega_c + 2 omega_o <= 199.99 of each axis on the decoded real
    values, the analytic Jacobian included (decode_slope: log/log on xy, log/lin on wz); () for PID."""
    assert rc.TRI_MAX == 199.99
    assert rc.triangle_constraints('PID', _dom(LEVELS['PID'], 'PID')) == ()
    dom = _dom(LEVELS_TRI['ADRC'], 'ADRC')
    cons = rc.triangle_constraints('ADRC', dom)
    assert len(cons) == 2 and all(c['type'] == 'ineq' for c in cons)
    c = np.array([0.4, -0.2, 0.1, 0.7, 0.0, 0.0, 0.0])
    g = _gains(dom, c)
    assert cons[0]['fun'](c) == pytest.approx(rc.TRI_MAX - (g['omega_c_xy'] + 2 * g['omega_o_xy']), abs=1e-12)
    assert cons[1]['fun'](c) == pytest.approx(rc.TRI_MAX - (g['omega_c_wz'] + 2 * g['omega_o_wz']), abs=1e-12)
    num = np.array([[(con['fun'](c + h) - con['fun'](c - h)) / 2e-6 for h in np.eye(7) * 1e-6] for con in cons])
    assert np.allclose([con['jac'](c) for con in cons], num, atol=1e-5)
    assert rc.triangle_sums('PID', {}) == {}
    assert rc.triangle_sums('ADRC', dict(omega_c_xy=100.0, omega_o_xy=40.0, omega_c_wz=8.0,
                                       omega_o_wz=3.0)) == {'xy': 180.0, 'wz': 14.0}
    h = 1e-6
    assert rc.decode_slope(0.3, 1.0, 3.0, 'lin') == pytest.approx(1.0)
    assert rc.decode_slope(0.3, 0.5, 8.0, 'log') == pytest.approx(
        (s3.decode(0.3 + h, 0.5, 8.0, 'log') - s3.decode(0.3 - h, 0.5, 8.0, 'log')) / (2 * h), rel=1e-6)
    c2 = np.zeros(7)                                             # omega_c_xy = 119.99, omega_o_xy = 40: on the bound
    c2[0] = 2 * math.log((rc.TRI_MAX - 80.0) / 60.0) / math.log(9.0)
    c2[1] = 1.0
    assert abs(cons[0]['fun'](c2)) < 1e-9


def test_optimise_keeps_both_steps_inside_the_triangle():
    """S04 v6.8 12.3: with the triangle constraints both SLSQP steps stay inside the band and the candidate sits
    on the bound; without them the same problem ends at the corner 180 + 2 x 40 = 260."""
    exp = rc.expected_codes()
    C = np.array([exp[p][1] for p in rc.FIT_IDS])
    dom = _dom(LEVELS_TRI['ADRC'], 'ADRC')
    b1 = rc.fit_ls(C, [q1_tri(c) for c in C])[0]
    b2 = rc.fit_ls(C, [q2_tri(c) for c in C])[0]
    starts = [C[i] for i in range(0, len(C), 12)]
    state, f1s, rows, cands = rc.optimise(b1, b2, starts, rc.triangle_constraints('ADRC', dom))
    assert state is None and cands
    assert {r['step'] for r in rows} == {1, 2}
    for row in rows:
        if row['converged']:
            g = _gains(dom, json.loads(row['code']))
            assert max(rc.triangle_sums('ADRC', g).values()) <= rc.TRI_MAX + 1e-6
    g = _gains(dom, cands[0]['code'])
    assert rc.triangle_sums('ADRC', g)['xy'] == pytest.approx(rc.TRI_MAX, abs=1e-4)
    assert s4.gain_check('ADRC', g) == ([], [])
    state2, _, _, cands2 = rc.optimise(b1, b2, starts)
    assert cands2 and rc.triangle_sums('ADRC', _gains(dom, cands2[0]['code']))['xy'] > 200


def test_optimise_margin_check_is_inclusive_and_exact():
    """C3-11: the binary64 margin check of triangle_check is inclusive and exact: a sum of exactly
    TRI_MAX is inside, the next float above it is out, on either axis; a NaN sum fails."""
    exp = rc.expected_codes()
    C = np.array([exp[p][1] for p in rc.FIT_IDS])
    b1 = rc.fit_ls(C, [q1_convex(c) for c in C])[0]
    b2 = rc.fit_ls(C, [q2_convex(c) for c in C])[0]
    starts = [C[0], C[1]]
    nxt = math.nextafter(rc.TRI_MAX, math.inf)
    for sums, ok in (({'xy': rc.TRI_MAX, 'wz': 10.0}, True),
                     ({'xy': nxt, 'wz': 10.0}, False),
                     ({'xy': 10.0, 'wz': nxt}, False),
                     ({'xy': float('nan'), 'wz': 10.0}, False),
                     ({'xy': 199.9900000005, 'wz': 10.0}, False)):
        state, f1s, rows, cands = rc.optimise(b1, b2, starts, tri=lambda code, s=sums: dict(s))
        if ok:
            assert state is None and cands
            assert all(r['converged'] and r['tri_ok'] is True for r in rows)
        else:
            assert state == 'i' and f1s is None and cands == []
            assert all(r['success'] and not r['converged'] and r['tri_ok'] is False for r in rows)
    state, f1s, rows, cands = rc.optimise(b1, b2, starts)
    assert all(r['tri_sums'] == '' and r['tri_ok'] == '' for r in rows)
    assert rc.triangle_check('PID', _dom(LEVELS['PID'], 'PID')) is None
    dom = _dom(LEVELS_TRI['ADRC'], 'ADRC')
    c = np.array([0.4, -0.2, 0.1, 0.7, 0.0, 0.0, 0.0])
    assert rc.triangle_check('ADRC', dom)(c) == rc.triangle_sums('ADRC', _gains(dom, c))


def test_optimise_with_the_margin_check_never_uses_a_result_over_it():
    """C3-11: with triangle_check the margin is kept by the result check, whatever SLSQP reports: a
    converged result is inside, every candidate is inside, f1* comes from converged step-1 results
    only. The state is not asserted (solver rounding)."""
    exp = rc.expected_codes()
    C = np.array([exp[p][1] for p in rc.FIT_IDS])
    dom = _dom(LEVELS_TRI['ADRC'], 'ADRC')
    b1 = rc.fit_ls(C, [q1_tri(c) for c in C])[0]
    b2 = rc.fit_ls(C, [q2_tri(c) for c in C])[0]
    starts = [C[i] for i in range(0, len(C), 12)]
    state, f1s, rows, cands = rc.optimise(b1, b2, starts, rc.triangle_constraints('ADRC', dom),
                                          rc.triangle_check('ADRC', dom))
    for row in rows:
        s = json.loads(row['tri_sums'])
        assert row['tri_ok'] == (max(s.values()) <= rc.TRI_MAX)
        assert not row['converged'] or row['tri_ok']
    for c in cands:
        assert max(rc.triangle_sums('ADRC', _gains(dom, c['code'])).values()) <= rc.TRI_MAX
    if f1s is not None:
        assert f1s == min(r['f1_hat'] for r in rows if r['step'] == 1 and r['converged'])


@pytest.mark.parametrize('lv,code,ok', [
    (dict(kp_xy=(0.5, 41.0)), [1.0, 0, 0, 0, 0, 0, 0], True),           # exactly the ceiling
    (dict(kp_xy=(0.5, 41.0000001)), [1.0, 0, 0, 0, 0, 0, 0], True),     # binary64 above 41, float32 exactly 41.0
    (dict(kp_xy=(0.5, 41.00001)), [1.0, 0, 0, 0, 0, 0, 0], False),      # float32 above the ceiling
    (dict(kp_wz=(0.5, 8.0)), [0, 0, 1.0, 0, 0, 0, 0], False),           # float32 8.0 above the ceiling 4.6
    (dict(kp_wz=(0.5, 8.0)), [0.0] * 7, True),                         # inside the table
])
def test_to_gains_checks_the_range_table_on_float32(tmp_path, dummy_dll, lv, code, ok):
    """S04 v6.8 12.3 (ii): to_gains compares the float32 gains with the range table (a binary64 just above the
    ceiling can round down to an accepted float32)."""
    levels = dict(LEVELS, PID=dict(LEVELS['PID'], **lv))
    repo, s03, s04 = setup(tmp_path, dummy_dll, levels=levels)
    inp = rc.Inputs(str(s04), dummy_dll)
    if ok:
        g, edge, dom = rc.to_gains(inp, 'PID', [float(x) for x in code])
        assert s4.gain_check('PID', g) == ([], [])
    else:
        with pytest.raises(rc.Stop, match='range table'):
            rc.to_gains(inp, 'PID', [float(x) for x in code])


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


_HOLD = """import sys
sys.path.insert(0, sys.argv[1])
import rsm_compare as rc
rc.acquire_run_lock(sys.argv[2])
print('held', flush=True)
sys.stdin.read()
"""


def test_lock_held_by_another_process_then_killed(tmp_path, dummy_dll, fake_eval):
    """RSM3-3, RSM3-4, RSM2-7: a process holding the OS lock (any content) stops a run, with or without
    --abort-previous, before any row is written; once that process is killed the system has freed the lock: the
    next run goes on, no file to delete by hand, no --abort-previous."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    (s04 / 'rsm').mkdir()
    p = subprocess.Popen([sys.executable, '-c', _HOLD, os.path.join(HERE, '..'), str(s04 / 'rsm')],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
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
    assert att(s04)[-1]['status'] == 'done'
    assert not [p for p in os.listdir(s04 / 'rsm') if 'lock' in p and p != 'opt_lock.yaml']   # RSM4-2: no lock file


def test_lock_file_left_behind_is_no_lock_and_a_phase_holds_it(tmp_path, dummy_dll, fake_eval, monkeypatch):
    """RSM3-3: a lock file left by an older version (empty, or naming a live process id) does not block the next
    run; RSM3-4: a second run started while a phase runs is refused for the whole phase (same process too)."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    (s04 / 'rsm').mkdir()
    lk = s04 / 'rsm' / '.run_lock'
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


def test_two_commits_bringing_the_same_done_row_stop(tmp_path, dummy_dll, fake_eval):
    """RSM4-1: a branch from the fit start commit copies the fit output commit's attempts.csv, changes coefs.csv and
    the lock hash, commits with the same message; a merge keeps that branch's fit files: two commits bring the fit
    done row in, so no output commit is chosen and the opt phase stops (git log order does not decide)."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    rc.run_phase('fit', str(s04), dummy_dll)
    rel_rsm = '05_DATA/results/S04/run01/rsm'
    main = git_out(repo, 'rev-parse', '--abbrev-ref', 'HEAD').strip()
    f_commit = git_out(repo, 'rev-parse', 'HEAD').strip()
    msg = git_out(repo, 'log', '-1', '--format=%s', 'HEAD').strip()
    start = git_out(repo, 'rev-parse', 'HEAD~1').strip()
    git_out(repo, 'checkout', '-q', '-b', 'other', start)
    git_out(repo, 'checkout', f_commit, '--', rel_rsm + '/attempts.csv', rel_rsm + '/fit', rel_rsm + '/meta.yaml')
    fit = s04 / 'rsm' / 'fit'
    (fit / 'coefs.csv').write_text((fit / 'coefs.csv').read_text().replace('0.03', '0.031', 1))
    lk = json.load(open(fit / 'lock.yaml'))
    lk['files']['fit/coefs.csv'] = dict(zip(('sha256', 'sha256_lf'), rc.file_sha(str(fit / 'coefs.csv'))))
    (fit / 'lock.yaml').write_text(json.dumps(lk))
    git_out(repo, 'add', '--', rel_rsm)
    git_out(repo, 'commit', '-q', '-m', msg)
    git_out(repo, 'checkout', '-q', main)
    git_out(repo, 'merge', '-q', '--no-ff', '--no-commit', '-s', 'ours', 'other')
    git_out(repo, 'checkout', 'other', '--', rel_rsm + '/fit')
    git_out(repo, 'commit', '-q', '-m', 'merge, fit files of other')
    inp = rc.Inputs(str(s04), dummy_dll)
    aid = att(s04)[0]['attempt_id']
    with pytest.raises(rc.Stop, match='more than one commit'):
        rc.output_commit(inp, 'fit', aid)
    with pytest.raises(rc.Stop, match='more than one commit'):
        rc.run_phase('opt', str(s04), dummy_dll)


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
    """RSM2-7, RSM3-6: edge_src and edge_limit from the S03 v8.4 level sources (H of every factor: 'het luoi');
    the candidate has factors on both sides, so the H branch is checked with a non-empty edge_limit."""
    repo = make_repo(tmp_path)
    s03 = make_s03(tmp_path, q1=q1_flat, q2=q2_mixed, dll=dummy_dll)
    rows = list(csv.DictReader(open(s03 / 'levels.csv')))
    for r in rows:
        r['L_nguon'], r['H_nguon'] = '0', 'hết lưới'
    s3.write_csv(str(s03 / 'levels.csv'), rows)
    s04 = make_s04(repo, s03, dummy_dll, q1=q1_flat, q2=q2_mixed)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    c = list(csv.DictReader(open(s04 / 'rsm' / 'opt' / 'candidates.csv', encoding='utf-8')))[0]
    code = [float(x) for x in json.loads(c['code'])]
    n_high = sum(x > 0 for x in code)
    assert 0 < n_high < 7                                      # RSM3-6: both sides present
    assert c['edge_src'].count(':H:hết lưới') == n_high and c['edge_src'].count(':L:0') == 7 - n_high
    assert (c['edge_limit'] != '') == (n_high > 0) and c['edge_limit'].count(':H') == n_high


def test_edge_limit_names_the_table_ceiling(tmp_path, dummy_dll, fake_eval):
    """S04 v6.8 section 2 (C3-7): the same construction as test_edge_source_and_limit, but the flagged
    bound of ki_xy has source 'trần bảng chặn' (H = T): edge_limit names it with EDGE_LIMIT_TABLE,
    after the grid-end limits, and ki_xy:H appears in edge_limit only there."""
    repo = make_repo(tmp_path)
    s03 = make_s03(tmp_path, q1=q1_flat, q2=q2_mixed, dll=dummy_dll)
    rows = list(csv.DictReader(open(s03 / 'levels.csv', encoding='utf-8')))
    for r in rows:
        r['L_nguon'] = '0'
        r['H_nguon'] = 'trần bảng chặn' if r['factor'] == 'ki_xy' else 'hết lưới'
    s3.write_csv(str(s03 / 'levels.csv'), rows)
    s04 = make_s04(repo, s03, dummy_dll, q1=q1_flat, q2=q2_mixed)
    rc.run_phase('fit', str(s04), dummy_dll)
    rc.run_phase('opt', str(s04), dummy_dll)
    c = list(csv.DictReader(open(s04 / 'rsm' / 'opt' / 'candidates.csv', encoding='utf-8')))[0]
    table_limit = 'vùng ngoài bảng chặn không thuộc thí nghiệm (DUT từ chối gain đó)'   # S04 v6.8 section 2
    assert c['edge_limit'].endswith(' | ' + table_limit + ': ki_xy:H')
    assert c['edge_limit'].startswith('vùng ngoài biên chưa khảo sát bằng GA: ')
    assert c['edge_limit'].count('ki_xy:H') == 1                  # only in the table message
    assert ':H:trần bảng chặn' in c['edge_src']


def _hard_link(src, dst):
    try:
        os.link(str(src), str(dst))
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip('no hard link on this file system')


def test_io_guard_catches_an_alias(tmp_path, dummy_dll, monkeypatch):
    """RSM4-3, RSM5-3: the I/O guard records an open of a sealed_test/ file, directly and by a hard link alias, by
    builtins.open and by os.open, with a str, Path or bytes path, with the kind of the file it aliases."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    alias = tmp_path / 'alias.csv'
    _hard_link(s03 / 'sealed_test' / 'runs_test.csv', alias)
    direct = s03 / 'sealed_test' / 'runs_test.csv'
    ev = _io_log(monkeypatch, s03=s03)
    open(alias, 'rb').close()
    os.close(os.open(str(alias), os.O_RDONLY))
    for p in (os.fsencode(str(alias)), os.fsencode(str(direct))):
        open(p, 'rb').close()
        os.close(os.open(p, os.O_RDONLY))
    assert [(e[0], e[3]) for e in ev] == [('open', 'sealed_test')] * 6


def test_run_lock_name_is_global(tmp_path):
    """RSM5-1: the mutex name is in the Global namespace (one per folder for every Windows logon session) and
    depends on the real rsm/ path only."""
    a = rc.run_lock_name(str(tmp_path / 'rsm'))
    assert a.startswith('Global\\' + rc.RUN_LOCK_PREFIX) and a == rc.run_lock_name(str(tmp_path / 'x' / '..' / 'rsm'))
    assert a != rc.run_lock_name(str(tmp_path / 'other' / 'rsm'))


@pytest.mark.parametrize('tag,refused', [(0xA000000C, True),     # IO_REPARSE_TAG_SYMLINK
                                         (0xA0000003, True),     # IO_REPARSE_TAG_MOUNT_POINT (junction)
                                         (0x80000013, False),    # IO_REPARSE_TAG_DEDUP: storage, no other name
                                         (0x9000001A, False)])   # IO_REPARSE_TAG_CLOUD_A (OneDrive placeholder)
def test_reparse_tags(tmp_path, monkeypatch, tag, refused):
    """RSM5-2: only name-surrogate reparse points are links; a storage reparse point is a plain file (simulated:
    os.lstat of that path gives the tag)."""
    import types
    f = tmp_path / 'f.csv'
    f.write_text('x\n')
    real = os.lstat
    fake = types.SimpleNamespace(st_mode=0o100644, st_file_attributes=0x400, st_reparse_tag=tag, st_nlink=1)
    monkeypatch.setattr(os, 'lstat', lambda p, *a, **k: fake if os.path.normcase(str(p)) == os.path.normcase(str(f))
                        else real(p, *a, **k))
    monkeypatch.delattr(os.path, 'isjunction', raising=False)
    assert rc.is_dir_link(str(f)) is refused


def test_hard_links_are_refused(tmp_path, dummy_dll, monkeypatch):
    """RSM3-2, RSM4-2, RSM4-3: a second hard link to a sealed_test/ file at an S04 file stops the run before the
    start row; one at the old lock path rsm/.run_lock is never opened (the lock opens no file); under the I/O guard,
    no sealed_test/ file is opened, by builtins.open or os.open."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    src = s03 / 'sealed_test' / 'runs_test.csv'
    before = src.read_bytes()
    (s04 / 'rsm').mkdir()
    _hard_link(src, s04 / 'rsm' / '.run_lock')
    ev = _io_log(monkeypatch, repo, s03)
    rc.run_phase('fit', str(s04), dummy_dll, abort_reason='a run died')
    assert not [e for e in ev if e[0] == 'open' and e[3] == 'sealed_test'] and src.read_bytes() == before
    (s04 / 'rsm' / '.run_lock').unlink()
    _hard_link(src, s04 / 'post_checks' / 'leak.csv')
    ev.clear()
    with pytest.raises(rc.Stop, match='hard link'):
        rc.run_phase('opt', str(s04), dummy_dll)
    assert not [e for e in ev if e[0] == 'open'] and src.read_bytes() == before
    assert [r['kind'] for r in att(s04)] == ['start', 'end']                  # stopped before the opt start row


def _dir_link(kind, target, link):
    """A directory symlink or a junction (Windows); False when the machine cannot make it."""
    if kind == 'symlink':
        try:
            os.symlink(str(target), str(link), target_is_directory=True)
            return True
        except (OSError, NotImplementedError):
            return False
    if os.name != 'nt':
        return False
    import _winapi
    try:
        _winapi.CreateJunction(str(target), str(link))
        return True
    except OSError:
        return False


@pytest.mark.parametrize('kind', ['symlink', 'junction', 'junction_without_isjunction'])
def test_nested_directory_link_is_refused(tmp_path, dummy_dll, fake_eval, monkeypatch, kind):
    """RSM3-2, RSM4-4: rsm/confirm/_work made a directory symlink or a junction before the opt phase stops it
    before the start row, whatever the link points to (here an empty folder outside rsm/), and also on a Python
    without os.path.isjunction (simulated: the attribute removed)."""
    repo, s03, s04 = setup(tmp_path, dummy_dll)
    rc.run_phase('fit', str(s04), dummy_dll)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (s04 / 'rsm' / 'confirm').mkdir()
    if not _dir_link(kind.split('_')[0], outside, s04 / 'rsm' / 'confirm' / '_work'):
        pytest.skip(f'no {kind.split("_")[0]} on this machine')
    if kind == 'junction_without_isjunction':
        monkeypatch.delattr(os.path, 'isjunction', raising=False)
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
    assert git_out(repo, 'status', '--porcelain', '--', '05_DATA/results/S04/run01/rsm').strip() == ''   # RSM4-2
    cost = list(csv.DictReader(open(out / 'rsm' / 'check' / 'cost.csv')))
    main = next(r for r in cost if r['item'] == 'S04 main')
    assert main['n_runs_planned'] == main['n_runs_actual']          # the seeds run (1), not the locked ones (5)
