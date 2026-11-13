"""Tests of fwsim/o1_feas_search.py (S05 C12, O1 3.9 (5); design 03_CODE/docs/2026-11-14_Thiet_ke_tim_kha_thi_C12_v1.md,
Liam 14/11 CODE_FEAS "Duyệt, code luôn"; Codex round 1 C12-1..13, Liam "Sửa cả 13 + vòng 2"). Most tests use a
FAKE runner (invented level results): no fwsim number is made or printed. The smoke test on the real DLL only
checks files and exit codes."""
import argparse
import contextlib
import csv
import io
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.join(HERE, '..')
for _p in (FW, os.path.join(FW, '..', 'tools'), os.path.join(FW, '..', 'rpi5')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import o1_feas_search as F               # noqa: E402
import o1_gate as G                      # noqa: E402

DLL = os.path.join(FW, 'build', 'fwsim.dll')
FAKE_FW = 'fakefw1234'
DEFAULTS = {'PID': dict(kp_xy=1.0, ki_xy=0.5, kp_wz=2.0, ki_wz=0.4, pos_kp=6.0, pos_ki=3.0, k_psi=1.0),
            'ADRC': dict(omega_c_xy=8.0, omega_o_xy=30.0, omega_c_wz=8.0, omega_o_wz=30.0, pos_kp=6.0, pos_ki=3.0,
                         k_psi=1.0)}


def _row(v, mu, q=(0.05, 0.05, 0.05, 5.0), g1='dat', status='ok', run_bad=0, n_undef=0):
    r = dict(v=v, mu=mu, status=status, run_bad=run_bad, error='', g1_state=g1, g1_q='true')
    for k, qk in zip(G.MEASURED, q):
        st = 'dat' if (math.isfinite(qk) and qk <= G.THRESHOLDS[k] and n_undef == 0) else 'khong_dat'
        r.update({f'{k}_state': st, f'{k}_q': qk, f'{k}_n_undef': n_undef})
    states = [g1] + [r[f'{k}_state'] for k in G.MEASURED]
    r['outcome_decide'] = G.run_outcome(states)
    return F.finish_level(r)


class FakeRunner(F.Runner):
    """Level results from a rule on the gains: PID passes every level when its first gain > 2 x default;
    ADRC passes only v = 0.2, when its first gain > default. log_sha from the inputs (deterministic).
    Class switches (set by fake_run_jobs for post-search / recheck jobs): salt changes log_sha;
    gate_fail turns the outcome to khong_dat with the same log_sha; only: controllers they apply to."""
    calls = []
    salt = ''
    gate_fail = False
    only = ('PID', 'ADRC')

    def __init__(self, dll):
        self._cmd = {}

    def defaults(self, ctrl):
        return dict(DEFAULTS[ctrl])

    def level(self, ctrl, g, v, mu, keep_log=False):
        FakeRunner.calls.append((ctrl, v, mu, F.o1.A_R))
        first = g[F.s3.FACTORS[ctrl][0]]
        d = DEFAULTS[ctrl][F.s3.FACTORS[ctrl][0]]
        ok = first > 2 * d if ctrl == 'PID' else (first > d and v == 0.2)
        hit = ctrl in FakeRunner.only
        r = _row(v, mu, q=(0.05, 0.06, 0.07, 6.0) if ok else (0.15, 0.06, 0.07, 6.0))
        if hit and FakeRunner.gate_fail:
            r['outcome_decide'] = 'khong_dat'
        r['log_sha'] = json.dumps([ctrl, sorted(g.items()), v, mu, FakeRunner.salt if hit else ''])
        return r, None


def fake_run_jobs(jobs, work, n_parallel, say=None):
    """In-process run_jobs: same job files, job_main with the (patched) Runner."""
    os.makedirs(work, exist_ok=True)
    out = []
    for i, job in enumerate(jobs):
        if say:
            say(f'job {i + 1}/{len(jobs)} started')
        p = os.path.join(work, f"job_{job['name']}.json")
        F.dump_json(job, p)
        post = job['name'].startswith('post_')
        FakeRunner.salt = fake_run_jobs.salt if post else ''
        FakeRunner.gate_fail = fake_run_jobs.gate_fail if post else False
        F.job_main(p)
        FakeRunner.salt, FakeRunner.gate_fail = '', False
        with open(p + '.out.json', encoding='utf-8') as f:
            out.append(json.load(f))
    return out


@pytest.fixture
def fake(monkeypatch, tmp_path):
    dll = tmp_path / 'fwsim.dll'
    dll.write_bytes(b'fake')
    (tmp_path / 'fwsim_build.json').write_text(json.dumps(dict(esp32=dict(commit=FAKE_FW, dirty=False))))
    monkeypatch.setattr(F, 'Runner', FakeRunner)
    monkeypatch.setattr(F, 'run_jobs', fake_run_jobs)
    monkeypatch.setattr(F, 'POP', 6)
    monkeypatch.setattr(F, 'N_GEN', 3)
    monkeypatch.setattr(F.o1, 'A_R', 0.5)
    fake_run_jobs.salt, fake_run_jobs.gate_fail = '', False
    FakeRunner.calls, FakeRunner.only = [], ('PID', 'ADRC')
    return dll


def _main(args):
    buf = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buf):
        try:
            F.main(args)
        except SystemExit as e:
            code = e.code
    return code, buf.getvalue()


def _search(dll, out, tool=False, fw=FAKE_FW):
    return _main(['search', '--out', str(out), '--dll', str(dll), '--jobs', '1', '--expect-fw', fw]
                 + (['--tool-check'] if tool else []))


def _csv(p):
    with open(p, encoding='utf-8') as f:
        return list(csv.DictReader(f))


# ── V (design 4.3) ───────────────────────────────────────────────────────────────────

def test_V_zero_only_when_every_condition_passes():
    assert _row(0.2, 0.436)['V_level'] == 0.0
    edge = _row(0.2, 0.436, q=(0.10, 0.10, 0.10, 7.5))                  # q equal to the threshold passes
    assert edge['V_level'] == 0.0 and edge['c'] == 0
    assert _row(0.2, 0.436, q=(0.11, 0.05, 0.05, 5.0))['V_level'] == pytest.approx(0.1)
    assert _row(0.2, 0.436, q=(0.05, 0.05, 0.05, 15.0))['V_level'] == pytest.approx(1.0)
    nan = _row(0.2, 0.436, q=(float('nan'), 0.05, 0.05, 5.0))
    assert nan['g2_h'] == 1.0 and nan['V_level'] >= 1.0
    assert _row(0.2, 0.436, n_undef=1)['c'] == 1
    assert _row(0.2, 0.436, status='track_err')['c'] == 1
    assert _row(0.2, 0.436, run_bad=1)['c'] == 1
    assert _row(0.2, 0.436, run_bad=None)['c'] == 1
    assert _row(0.2, 0.436, g1='chua_ket_luan')['c'] == 1
    err = F.error_level(0.2, 0.436, 'FloatingPointError: x')
    assert err['c'] == 1 and err['V_level'] == 5.0 and not F.gate_pass(err)


def test_V_zero_iff_real_gate_passes():
    """level_from_gate on o1_gate.gate with invented frames: V = 0 <=> gate_pass (design 4.3)."""
    rng = np.random.default_rng(7)
    seen = set()
    for _ in range(400):
        n = 20
        fr = {G.FRAME_KEYS[k]: G.THRESHOLDS[k] * rng.uniform(0.3, 1.0, n) for k in G.MEASURED}
        if rng.random() < 0.3:                                         # one frame of one condition above
            k = G.MEASURED[int(rng.integers(4))]
            fr[G.FRAME_KEYS[k]][5] = G.THRESHOLDS[k] * 1.02
        if rng.random() < 0.15:
            fr['e_perp'][3] = np.nan
        completed = rng.choice(['true', 'true', 'true', 'false', 'inconclusive'])
        status = 'ok' if rng.random() < 0.85 else 'track_err'
        run_bad = 0 if rng.random() < 0.9 else 1
        gt = G.gate(fr, completed, v_cmd_max=1.0, wz_cmd_max=1.0, u_t=0.0, status=status)
        r = F.level_from_gate(0.5, 0.436, gt, status, run_bad)
        assert r['outcome_decide'] == gt['outcome_decide']
        assert (r['V_level'] == 0.0) == F.gate_pass(r)
        seen.add(r['V_level'] == 0.0)
    assert seen == {True, False}


def test_strict_json_no_nan(tmp_path):
    p = tmp_path / 'x.json'
    F.dump_json(dict(a=float('nan'), b=[np.float64('inf'), np.int64(3), np.bool_(True)], c=1.5), p)
    txt = p.read_text()
    assert 'NaN' not in txt and 'Infinity' not in txt
    assert json.loads(txt) == dict(a=None, b=[None, 3, True], c=1.5)


# ── Runner.level with its dependencies stopped (C12-9: arguments really used) ─────────

def test_runner_level_passes_configuration(monkeypatch):
    seen = {}

    def fake_sim(fw, setpoint, duration, p, **kw):
        seen.update(duration=duration, mu=np.asarray(p.mu_hd).tolist(), ground=p.ground_model, **kw)
        n = 5
        return dict(t=np.arange(n) * 0.01, pose=np.zeros((n, 3)), v_true=np.zeros((n, 3)), duty=np.zeros((n, 4)),
                    pwm=np.zeros((n, 4)), pos=np.zeros((n, 1)))
    frames = {G.FRAME_KEYS[k]: np.full(3, 0.5 * G.THRESHOLDS[k]) for k in G.MEASURED}
    monkeypatch.setattr(F.s3, 'sim_kwargs', lambda fw, ctrl, g: dict(gains_marker=dict(g)))
    monkeypatch.setattr(F.fwsim, 'simulate', fake_sim)
    monkeypatch.setattr(F.E, 'evaluate_fig8', lambda log, path, sch, lsum: (dict(completed='true', status='ok'),
                                                                            [], frames))
    monkeypatch.setattr(F.fwsim, 'run_validity', lambda log: dict(run_bad=0))
    monkeypatch.setattr(F.o1, 'A_R', 0.4)
    r = F.Runner.__new__(F.Runner)
    r.fw, r.path, r._cmd = None, None, {}
    row, _ = r.level('PID', dict(kp_xy=1.0), 0.5, 0.413)
    sch = F.o1.Schedule(0.5)
    assert sch.Tr == pytest.approx(sch.vmax / 0.4)                    # --a-r reaches the schedule
    assert seen['duration'] == pytest.approx(sch.Tcmd + F.o1.T_LATE)
    assert seen['mu'] == [0.413] * 4 and seen['ground'] == 'traction'
    assert seen['seed'] == F.s3.SEED and seen['imu'] == F.s3.IMU and seen['mode'] == 'PID'
    assert seen['gains_marker'] == dict(kp_xy=1.0)
    assert row['V_level'] == 0.0 and F.gate_pass(row) and row['log_sha']


# ── domain ───────────────────────────────────────────────────────────────────────────

def test_box_positive_log_and_default_inside():
    for ctrl in F.CONTROLLERS:
        box = F.box_of(ctrl, DEFAULTS[ctrl])
        assert [d['name'] for d in box] == F.s3.FACTORS[ctrl]
        for d in box:
            assert 0.0 < d['a'] <= d['default'] <= d['b']
            assert d['scale'] == ('log' if d['b'] / d['a'] >= 4.0 * (1 - 1e-14) else 'lin')
            if d['name'].startswith('omega_o'):
                assert d['b'] <= F.s3.OMEGA_O_MAX
        X = F.first_population(box)
        assert X.shape == (F.POP, 7)
        assert F.decode(box, X[0]) == F.default_gains(box)                 # row 0 is the default, exactly
        for x in X[1:]:
            g = F.decode(box, x)
            assert all(d['a'] <= g[d['name']] <= d['b'] for d in box)
    with pytest.raises(F.FeasStop):
        F.box_of('PID', dict(DEFAULTS['PID'], ki_xy=0.0))


def test_first_population_same_coded_draw_for_both_controllers():
    bp, ba = (F.box_of(c, DEFAULTS[c]) for c in F.CONTROLLERS)
    up = [(F.first_population(bp)[1:, i] - d['xl']) / (d['xu'] - d['xl']) for i, d in enumerate(bp)]
    ua = [(F.first_population(ba)[1:, i] - d['xl']) / (d['xu'] - d['xl']) for i, d in enumerate(ba)]
    assert np.allclose(up, ua, atol=1e-12)


def test_inside_uses_s03_tolerance():
    tol = F.s3.BOUND_RTOL * (2.0 - 1.0)
    assert F.inside(2.0 + 0.5 * tol, 1.0, 2.0) and F.inside(1.0 - 0.5 * tol, 1.0, 2.0)
    assert not F.inside(2.0 + 3 * tol, 1.0, 2.0) and not F.inside(1.0 - 3 * tol, 1.0, 2.0)
    big = F.s3.BOUND_RTOL * 1e6                       # a tolerance scaled by max(|L|, |H|) would let this pass
    assert not F.inside(1e6 + 0.5 * big, 1e6 - 1e-3, 1e6)


# ── GA: witness, budget, early end of pymoo (C12-8) ─────────────────────────────────

def test_witness_is_first_V_zero_and_whole_budget_runs():
    box = F.box_of('PID', DEFAULTS['PID'])
    hist, wit, notes = F.ga_search(FakeRunner(None), 'PID', (0.2, 0.5, 0.8), 0.436, box, pop=6, n_gen=4)
    assert notes['n_evaluated'] == 24 and notes['generations'] == 4 and notes['end'] == 'budget'
    assert notes['per_generation'] == [6, 6, 6, 6]
    assert len({h['eval_id'] for h in hist}) == 24 and len(hist) == 3 * 24
    zero = sorted({h['eval_id'] for h in hist if h['V'] == 0.0})
    assert wit is not None and wit['eval_id'] == zero[0]
    assert len(zero) > 1                                               # later V = 0 candidates did not replace it


def test_default_already_feasible_still_runs_whole_budget():
    box = F.box_of('PID', DEFAULTS['PID'])

    class AllPass(FakeRunner):
        def level(self, ctrl, g, v, mu, keep_log=False):
            r = _row(v, mu)
            r['log_sha'] = 'x'
            return r, None
    hist, wit, notes = F.ga_search(AllPass(None), 'PID', (0.2,), 0.436, box, pop=5, n_gen=3)
    assert wit['eval_id'] == 0 and wit['gains'] == {k: DEFAULTS['PID'][k] for k in F.s3.FACTORS['PID']}
    assert notes['n_evaluated'] == 15


def test_pymoo_no_new_offspring_is_recorded():
    box = F.box_of('PID', DEFAULTS['PID'])

    def factory(pop, seed_pop):
        ga = F.make_ga(pop, seed_pop)
        real_ask, n = ga.ask, [0]

        def ask():
            n[0] += 1
            return None if n[0] == 3 else real_ask()
        ga.ask = ask
        return ga
    hist, wit, notes = F.ga_search(FakeRunner(None), 'PID', (0.2,), 0.436, box, pop=4, n_gen=5, algorithm=factory)
    assert notes['end'] == 'no_new_offspring' and notes['generations'] == 2 and notes['n_evaluated'] == 8


# ── labels (design 4.5) ──────────────────────────────────────────────────────────────

def _wit(levels, ok=True):
    return dict(eval_id=0, gen=1, idx=0, gains={}, levels=[_row(v, 0.436, q=(0.05, 0.05, 0.05, 5.0) if ok
                                                                  else (0.2, 0.05, 0.05, 5.0)) for v in levels])


def test_labels_inherit_and_common_set():
    w = {('PID', 'T3'): _wit((0.2, 0.5, 0.8)), ('PID', 'T2'): None, ('PID', 'T1'): None,
         ('ADRC', 'T3'): None, ('ADRC', 'T2'): _wit((0.2, 0.5)), ('ADRC', 'T1'): _wit((0.2,))}
    rows, inh, chosen = F.labels(w)
    lab = {(r['controller'], r['set']): (r['own_search'], r['label']) for r in rows}
    assert lab[('PID', 'T2')] == (F.NOT_FOUND, F.FOUND) and lab[('PID', 'T1')] == (F.NOT_FOUND, F.FOUND)
    assert lab[('ADRC', 'T3')] == (F.NOT_FOUND, F.NOT_FOUND)
    assert chosen == 'T2'
    assert {(r['controller'], r['set'], r['from_set']) for r in inh} == {('PID', 'T2', 'T3'), ('PID', 'T1', 'T3')}


def test_labels_empty_and_never_impossible():
    rows, _, chosen = F.labels({(c, s): None for c in F.CONTROLLERS for s in F.SET_NAMES})
    assert chosen == F.EMPTY_SET and {r['label'] for r in rows} == {F.NOT_FOUND}
    w = {(c, s): None for c in F.CONTROLLERS for s in F.SET_NAMES}
    w[('PID', 'T1')] = _wit((0.2,))
    rows, _, chosen = F.labels(w)
    assert chosen == F.EMPTY_SET                                       # ADRC has nothing
    assert not any('kha_thi_co' in str(r) for r in rows)


def test_restrict_V_and_by_level_check_level_keys():
    rows = [_row(0.2, 0.436), _row(0.5, 0.436), _row(0.8, 0.436, q=(0.2, 0.05, 0.05, 5.0))]
    assert F.restrict_V(rows, (0.2, 0.5)) == 0.0
    assert F.restrict_V(rows, (0.5, 0.2)) == 0.0                       # order does not matter
    assert F.restrict_V(rows, (0.2, 0.5, 0.8)) > 0.0
    with pytest.raises(ValueError):
        F.restrict_V(rows[:1], (0.2, 0.5))                             # missing level
    with pytest.raises(ValueError):
        F.restrict_V([_row(0.2, 0.436), _row(0.2, 0.436)], (0.2, 0.5))  # duplicate instead of 0.5 (C12-7)
    with pytest.raises(ValueError):
        F.by_level([_row(0.2, 0.413)], (0.2,), 0.436)                  # wrong mu
    with pytest.raises(ValueError):
        F.by_level([_row(0.2, 0.436), _row(0.2, 0.436)], (0.2, 0.2))   # duplicate keys even when counted twice


# ── whole search with the fake runner: fence, files ──────────────────────────────────

ALLOWED = [r'o1_feas_search search: mu 0\.436, a_r 0\.5, firmware fakefw1234, out \S+', r'job [1-6]/6 started',
           r'labels \(O1 3\.9 \(5\); tim_thay = feasible on this fwsim configuration, not on the car\):',
           r'  (PID |ADRC) T[123] \{(0\.2 0\.5 0\.8|0\.2 0\.5|0\.2)\}: (tim_thay|chua_tim_thay)',
           r'feasibility level set: (rong|T[123] \{[0-9. ]+\})',
           r'default gains at mu 0\.436 \(G1 state; G2-G5 state and q; G5 in deg\):',
           r'  (PID |ADRC) v 0\.[258]: G1 (dat|khong_dat|chua_ket_luan), G2 \w+ ([0-9.]+|nan), '
           r'G3 \w+ ([0-9.]+|nan), G4 \w+ ([0-9.]+|nan), G5 \w+ ([0-9.]+|nan), status \w+']
PUBLIC_FILES = {'box.csv', 'default_gains.csv', 'labels.csv', 'meta.yaml', 'console.txt', 'sealed'}


def test_search_fence_and_files(fake, tmp_path):
    out = tmp_path / 'run'
    code, text = _search(fake, out)
    assert code == 0, text
    for line in text.splitlines():
        assert any(re.fullmatch(p, line) for p in ALLOWED), line
    assert 'feasibility level set: T1 {0.2}' in text                   # rule of FakeRunner: ADRC only at 0.2
    assert set(os.listdir(out)) == PUBLIC_FILES                         # logs_default/ only with real logs
    wit = _csv(out / 'sealed' / 'witness.csv')
    assert len(wit) == 6 and all(r['nonce'] for r in wit)              # always 6 rows, nonce (C12-1)
    public = text + ''.join((out / f).read_text(encoding='utf-8') for f in ('box.csv', 'default_gains.csv',
                                                                           'labels.csv', 'meta.yaml'))
    for r in wit:                                                      # no witness gain outside sealed/
        for k, v in r.items():
            if k.startswith('gain_') and v and float(v) not in DEFAULTS[r['controller']].values():
                assert v not in public and f'{float(v):.4f}' not in public
    meta = yaml.safe_load((out / 'meta.yaml').read_text(encoding='utf-8'))
    assert meta['status'] == 'complete' and meta['tool_check'] is False
    assert set(meta['sealed_sha256']) == {'witness.csv', 'manifest.json'}    # no per-file hash in public (C12-1)
    assert meta['sealed_sha256']['witness.csv'] == F.s3.sha256_file(out / 'sealed' / 'witness.csv')
    assert 'elapsed' not in (out / 'meta.yaml').read_text(encoding='utf-8')
    assert (out / 'console.txt').read_text(encoding='utf-8') == text
    lab = _csv(out / 'labels.csv')
    assert lab[-1]['levels'] == 'T1'
    dg_text = (out / 'default_gains.csv').read_text(encoding='utf-8')
    assert dg_text.splitlines()[0] == ','.join(F.DEFAULT_COLS)          # fixed public columns (C12-10)
    dg = _csv(out / 'default_gains.csv')
    assert len(dg) == 6 and all(r['mu'] == '0.436' for r in dg)
    assert not [n for n in os.listdir(out / 'sealed' / 'jobs') if n.endswith('.out.json')]
    man = json.loads((out / 'sealed' / 'manifest.json').read_text())
    assert man['nonce'] == wit[0]['nonce'] and 'mu_lohi.csv' in man['files']


def test_default_table_written_only_at_the_end(fake, tmp_path, monkeypatch):
    seen = {}

    def spy(jobs, work, n, say=None):
        seen.setdefault('public_at_first_job', sorted(os.listdir(tmp_path / 'run')))
        return fake_run_jobs(jobs, work, n, say)
    monkeypatch.setattr(F, 'run_jobs', spy)
    _search(fake, tmp_path / 'run')
    assert 'default_gains.csv' not in seen['public_at_first_job'] and 'labels.csv' not in seen['public_at_first_job']


def test_every_search_gets_its_whole_budget(fake, tmp_path):
    out = tmp_path / 'run'
    _search(fake, out)
    notes = json.loads((out / 'sealed' / 'notes.json').read_text())
    assert len(notes['jobs']) == 6
    for name, n in notes['jobs'].items():
        assert n['n_evaluated'] == F.POP * F.N_GEN and n['end'] == 'budget', name


def test_post_schedule_is_fixed(fake, tmp_path, monkeypatch):
    """Same number of post-search runs whatever the witnesses (C12-3): 6 positions x levels x 3 mu."""
    counts = []
    for rule in ('normal', 'none'):
        FakeRunner.calls = []
        if rule == 'none':
            monkeypatch.setattr(FakeRunner, 'only', ())
            monkeypatch.setattr(F, 'labels', F.labels)
            orig = FakeRunner.level

            def never(self, ctrl, g, v, mu, keep_log=False):
                r, _ = orig(self, ctrl, g, v, mu, keep_log)
                r = dict(r, outcome_decide='khong_dat', V_level=1.0)
                return r, None
            monkeypatch.setattr(FakeRunner, 'level', never)
        _search(fake, tmp_path / f'run_{rule}')
        n_search = 2 * F.POP * F.N_GEN * (3 + 2 + 1)
        counts.append(len(FakeRunner.calls) - n_search - 6)              # minus search and default runs
    assert counts[0] == counts[1] == 2 * (3 + 2 + 1) * 3


def test_determinism_mismatch_stops_and_writes_no_witness(fake, tmp_path):
    fake_run_jobs.salt = 'different'
    out = tmp_path / 'run'
    code, text = _search(fake, out)
    assert code == 2 and 'determinism check failed' in text
    assert all(r['found'] == 'false' for r in _csv(out / 'sealed' / 'witness.csv'))
    assert not (out / 'labels.csv').exists() and not (out / 'default_gains.csv').exists()
    assert 'tim_thay' not in text
    assert yaml.safe_load((out / 'meta.yaml').read_text(encoding='utf-8'))['status'] == 'stopped'


def test_same_log_but_gate_fails_stops(fake, tmp_path):
    fake_run_jobs.gate_fail = True
    code, text = _search(fake, tmp_path / 'run')
    assert code == 2 and 'determinism check failed' in text


def test_one_witness_failing_stops_the_run(fake, tmp_path):
    FakeRunner.only = ('ADRC',)
    fake_run_jobs.salt = 'different'
    out = tmp_path / 'run'
    code, _ = _search(fake, out)
    assert code == 2
    rc = _csv(out / 'sealed' / 'recheck.csv')
    assert {r['ok'] for r in rc if r['controller'] == 'PID'} == {'true'}
    assert {r['ok'] for r in rc if r['controller'] == 'ADRC'} == {'false'}


def test_tool_check_prints_no_label(fake, tmp_path):
    code, text = _search(fake, tmp_path / 'run', tool=True)
    assert code == 0
    assert 'tim_thay' not in text and 'feasibility level set' not in text and 'default gains' not in text
    assert 'tool-check done' in text
    assert all(c[2] == F.MU_TOOL for c in FakeRunner.calls)
    assert yaml.safe_load((tmp_path / 'run' / 'meta.yaml').read_text(encoding='utf-8'))['tool_check'] is True


def test_firmware_must_be_clean_and_expected(fake, tmp_path):
    code, text = _search(fake, tmp_path / 'a', fw='0123456')
    assert code == 2 and 'expect-fw' in text
    assert yaml.safe_load((tmp_path / 'a' / 'meta.yaml').read_text(encoding='utf-8'))['status'] == 'stopped'
    (fake.parent / 'fwsim_build.json').write_text(json.dumps(dict(esp32=dict(commit=FAKE_FW, dirty=True))))
    code, text = _search(fake, tmp_path / 'b')
    assert code == 2 and 'dirty' in text


def test_existing_folder_refused(fake, tmp_path):
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'x').write_text('x')
    code, _ = _search(fake, out)
    assert code not in (0, None)


def _recheck(fake, src, out, levels, a_r='0.4'):
    return _main(['recheck', '--from', str(src), '--out', str(out), '--levels', levels, '--a-r', a_r,
                  '--dll', str(fake), '--jobs', '1', '--expect-fw', FAKE_FW])


def test_recheck_intersection_set_and_a_r(fake, tmp_path):
    out = tmp_path / 'run'
    _search(fake, out)
    FakeRunner.calls = []
    code, text = _recheck(fake, out, tmp_path / 'rc', '0.8,0.2')        # an intersection set (C12-2)
    assert code == 0, text
    res = {r['controller']: r['result'] for r in _csv(tmp_path / 'rc' / 'labels_recheck.csv')}
    assert res == {'PID': 'con_dat', 'ADRC': 'khong_con_dat'}          # only T3 holds {0.2, 0.8}; ADRC has none
    assert FakeRunner.calls and {c[3] for c in FakeRunner.calls} == {0.4}       # a_r reached every run
    assert {(c[0], c[1]) for c in FakeRunner.calls} == {(c, v) for c in F.CONTROLLERS for v in (0.2, 0.8)}
    meta = yaml.safe_load((tmp_path / 'rc' / 'meta.yaml').read_text(encoding='utf-8'))
    assert meta['a_r'] == 0.4 and meta['status'] == 'complete' and meta['levels'] == [0.2, 0.8]
    assert 'gain' not in text
    code, _ = _recheck(fake, out, tmp_path / 'rc2', '0.5')
    assert code == 0                                                    # {0.5}: T3, T2 hold it


@pytest.mark.parametrize('levels', ['0.3', '0.2,0.2', '', '0.2,x'])
def test_recheck_rejects_bad_levels(fake, tmp_path, levels):
    out = tmp_path / 'run'
    _search(fake, out)
    code, _ = _recheck(fake, out, tmp_path / 'rc', levels)
    assert code not in (0, None)


def test_recheck_refuses_tool_check_stopped_or_changed_source(fake, tmp_path):
    _search(fake, tmp_path / 'tc', tool=True)
    code, _ = _recheck(fake, tmp_path / 'tc', tmp_path / 'r1', '0.2')
    assert code not in (0, None)
    fake_run_jobs.salt = 'different'
    _search(fake, tmp_path / 'stop')
    fake_run_jobs.salt = ''
    code, _ = _recheck(fake, tmp_path / 'stop', tmp_path / 'r2', '0.2')
    assert code not in (0, None)
    _search(fake, tmp_path / 'ok')
    with open(tmp_path / 'ok' / 'sealed' / 'history.csv', 'a', encoding='utf-8') as f:
        f.write('tampered\n')
    code, _ = _recheck(fake, tmp_path / 'ok', tmp_path / 'r3', '0.2')
    assert code not in (0, None)


def test_inbox(fake, tmp_path):
    out = tmp_path / 'run'
    _search(fake, out)
    s03 = tmp_path / 'S03 run'                                         # a space and a backslash path below
    s03.mkdir()
    with open(s03 / 'levels.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['controller', 'factor', 'L', 'H', 'centre', 'scale', 'default', 'rule_note'])
        for c in F.CONTROLLERS:
            for n in F.s3.FACTORS[c]:
                w.writerow([c, n, 1e-9, 1e9, 1, 'log', 1, ''])
    code, text = _main(['inbox', '--from', str(out), '--out', str(tmp_path / 'ib'), '--s03', str(s03)])
    assert code == 0 and text.strip() == 'inbox written (sealed)'
    meta = yaml.safe_load((tmp_path / 'ib' / 'meta.yaml').read_text(encoding='utf-8'))   # Windows paths (C12-11)
    assert meta['s03_run'] == str(s03) and meta['source_run'] == str(out)
    assert set(meta['sealed_sha256']) == {'manifest.json'}
    rows = _csv(tmp_path / 'ib' / 'sealed' / 'inbox.csv')
    assert rows and all(r['inside'] == 'true' for r in rows)


def test_child_failure_prints_one_fixed_sentence(tmp_path):
    with pytest.raises(F.FeasStop) as e:
        F.run_jobs([dict(name='secret_job_PID_T3', kind='search', dll=str(tmp_path / 'missing.dll'))],
                   str(tmp_path / 'work'), 1)
    assert str(e.value) == F.CHILD_FAILED


# ── smoke on the real DLL (files and exit code only; no number read) ─────────────────

SIM_LOCK = ['fwsim/s03_doe.py', 'fwsim/o1_feasibility.py', 'fwsim/fwsim.py', 'fwsim/tune_ga.py',
            'rpi5/trajectory_pub.py', 'rpi5/imu_model.py', 'rpi5/plant_blob.py', 'tools/run_validity.py',
            'rpi5/plant_step.py', 'rpi5/params.py', 'fwsim/o1_gate.py', 'fwsim/o1_gate_eval.py', 'tools/o1_lanes.py']


@pytest.mark.skipif(not os.path.isfile(DLL), reason='fwsim.dll not built')
def test_tool_check_smoke_real_dll(tmp_path):
    with open(os.path.join(FW, 'build', 'fwsim_build.json'), encoding='utf-8') as f:
        fw = json.load(f)['esp32']['commit']
    out = tmp_path / 'tc'
    r = subprocess.run([sys.executable, os.path.join(FW, 'o1_feas_search.py'), 'search', '--tool-check', '--out',
                        str(out), '--dll', DLL, '--jobs', '6', '--expect-fw', fw], capture_output=True, text=True,
                       timeout=3000)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    assert 'tim_thay' not in r.stdout and 'tool-check done' in r.stdout
    assert len(_csv(out / 'sealed' / 'witness.csv')) == 6
    assert len(_csv(out / 'default_gains.csv')) == 6
    meta = yaml.safe_load((out / 'meta.yaml').read_text(encoding='utf-8'))
    assert meta['status'] == 'complete' and meta['imu'] == F.s3.IMU and meta['a_r'] == 0.5
    root = os.path.abspath(os.path.join(FW, '..'))
    st = subprocess.run(['git', '-C', root, 'status', '--porcelain', '--'] + SIM_LOCK, capture_output=True, text=True)
    assert st.stdout.strip() == ''                                     # S04 lock and evaluator untouched
