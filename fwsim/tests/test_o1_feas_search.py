"""Tests of fwsim/o1_feas_search.py (S05 C12, O1 3.9 (5); design 03_CODE/docs/2026-11-14_Thiet_ke_tim_kha_thi_C12_v1.md,
Liam 14/11 CODE_FEAS "Duyệt, code luôn"). Most tests use a FAKE runner (invented level results): no fwsim
number is made or printed. The smoke test on the real DLL only checks files and exit codes."""
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

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.join(HERE, '..')
for _p in (FW, os.path.join(FW, '..', 'tools'), os.path.join(FW, '..', 'rpi5')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import o1_feas_search as F               # noqa: E402
import o1_gate as G                      # noqa: E402

DLL = os.path.join(FW, 'build', 'fwsim.dll')
DEFAULTS = {'PID': dict(kp_xy=1.0, ki_xy=0.5, kp_wz=2.0, ki_wz=0.4, pos_kp=6.0, pos_ki=3.0, k_psi=1.0),
            'ADRC': dict(omega_c_xy=8.0, omega_o_xy=30.0, omega_c_wz=8.0, omega_o_wz=30.0, pos_kp=6.0, pos_ki=3.0,
                         k_psi=1.0)}


def _row(v, mu, q=(0.05, 0.05, 0.05, 5.0), g1='dat', status='ok', run_bad=0, n_undef=0):
    r = dict(v=v, mu=mu, status=status, run_bad=run_bad, error='', g1_state=g1, g1_q='true')
    for k, qk in zip(G.MEASURED, q):
        st = 'dat' if (math.isfinite(qk) and qk <= G.THRESHOLDS[k] and n_undef == 0) else 'khong_dat'
        r.update({f'{k}_state': st, f'{k}_q': qk, f'{k}_n_undef': n_undef})
    r = F.finish_level(r)
    states = [g1] + [r[f'{k}_state'] for k in G.MEASURED]
    r['outcome_decide'] = G.run_outcome(states)
    return r


class FakeRunner(F.Runner):
    """Level results from a rule on the gains: PID passes every level when its first gain > 2 x default;
    ADRC passes only v = 0.2, when its first gain > default. log_sha from the inputs (deterministic)."""
    calls = []
    salt = ''

    def __init__(self, dll):
        self._cmd = {}

    def defaults(self, ctrl):
        return dict(DEFAULTS[ctrl])

    def level(self, ctrl, g, v, mu, keep_log=False):
        FakeRunner.calls.append((ctrl, v, mu))
        first = g[F.s3.FACTORS[ctrl][0]]
        d = DEFAULTS[ctrl][F.s3.FACTORS[ctrl][0]]
        ok = first > 2 * d if ctrl == 'PID' else (first > d and v == 0.2)
        r = _row(v, mu, q=(0.05, 0.06, 0.07, 6.0) if ok else (0.15, 0.06, 0.07, 6.0))
        r['log_sha'] = json.dumps([ctrl, sorted(g.items()), v, mu, FakeRunner.salt])
        return r, None


def fake_run_jobs(jobs, work, n_parallel, say=None):
    """In-process run_jobs: same job files, job_main with the (patched) Runner."""
    os.makedirs(work, exist_ok=True)
    out = []
    for i, job in enumerate(jobs):
        if say:
            say(f'job {i + 1}/{len(jobs)} started')
        p = os.path.join(work, f"job_{i:02d}_{job['name']}.json")
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(job, f)
        if job['kind'] == 'levels' and job['name'].startswith('recheck_'):
            FakeRunner.salt = getattr(fake_run_jobs, 'recheck_salt', '')
        F.job_main(p)
        FakeRunner.salt = ''
        with open(p + '.out.json', encoding='utf-8') as f:
            out.append(json.load(f))
    return out


@pytest.fixture
def fake(monkeypatch, tmp_path):
    dll = tmp_path / 'fwsim.dll'
    dll.write_bytes(b'fake')
    (tmp_path / 'fwsim_build.json').write_text(json.dumps(dict(esp32=dict(commit='fake', dirty=False))))
    monkeypatch.setattr(F, 'Runner', FakeRunner)
    monkeypatch.setattr(F, 'run_jobs', fake_run_jobs)
    monkeypatch.setattr(F, 'POP', 6)
    monkeypatch.setattr(F, 'N_GEN', 3)
    fake_run_jobs.recheck_salt = ''
    FakeRunner.calls = []
    return dll


def _search(dll, out, tool=False):
    args = ['search', '--out', str(out), '--dll', str(dll), '--jobs', '1'] + (['--tool-check'] if tool else [])
    buf = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buf):
        try:
            F.main(args)
        except SystemExit as e:
            code = e.code
    return code, buf.getvalue()


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
    assert _row(0.2, 0.436, run_bad=float('nan'))['c'] == 1
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
        r['outcome_decide'] = gt['outcome_decide']
        assert (r['V_level'] == 0.0) == F.gate_pass(r)
        seen.add(r['V_level'] == 0.0)
    assert seen == {True, False}


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
        assert F.decode(box, X[0]) == {d['name']: d['default'] for d in box}     # row 0 is the default, exactly
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


# ── GA: witness, budget ──────────────────────────────────────────────────────────────

def test_witness_is_first_V_zero_and_whole_budget_runs():
    box = F.box_of('PID', DEFAULTS['PID'])
    hist, wit, n = F.ga_search(FakeRunner(None), 'PID', (0.2, 0.5, 0.8), 0.436, box, pop=6, n_gen=4)
    assert n == 24                                                     # no early stop: 6 x 4
    assert len({h['eval_id'] for h in hist}) == n and len(hist) == 3 * n
    zero = sorted({h['eval_id'] for h in hist if h['V'] == 0.0})
    assert wit is not None and wit['eval_id'] == zero[0]
    assert len(zero) > 1                                               # later V = 0 candidates did not replace it


def test_default_already_feasible_still_runs_whole_budget():
    d = dict(DEFAULTS['PID'], kp_xy=1.0)
    box = F.box_of('PID', d)

    class AllPass(FakeRunner):
        def level(self, ctrl, g, v, mu, keep_log=False):
            r = _row(v, mu)
            r['log_sha'] = 'x'
            return r, None
    hist, wit, n = F.ga_search(AllPass(None), 'PID', (0.2,), 0.436, box, pop=5, n_gen=3)
    assert wit['eval_id'] == 0 and wit['gains'] == {k: d[k] for k in F.s3.FACTORS['PID']}
    assert n == 15


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


def test_restrict_V_uses_only_subset_levels():
    w = dict(levels=[_row(0.2, 0.436), _row(0.5, 0.436), _row(0.8, 0.436, q=(0.2, 0.05, 0.05, 5.0))])
    assert F.restrict_V(w['levels'], (0.2, 0.5)) == 0.0
    assert F.restrict_V(w['levels'], (0.2, 0.5, 0.8)) > 0.0
    with pytest.raises(ValueError):
        F.restrict_V(w['levels'][:1], (0.2, 0.5))


# ── whole search with the fake runner: fence, files ──────────────────────────────────

ALLOWED = [r'^o1_feas_search search: ', r'^job \d+/6 started$', r'^labels \(O1 3\.9 \(5\);',
           r'^  (PID |ADRC) T[123] \{[0-9. ]+\}: (tim_thay|chua_tim_thay)$', r'^feasibility level set: ',
           r'^default gains at mu ', r'^  (PID |ADRC) v 0\.[258]: G1 ']


def test_search_fence_and_files(fake, tmp_path):
    out = tmp_path / 'run'
    code, text = _search(fake, out)
    assert code == 0, text
    for line in text.splitlines():
        assert any(re.match(p, line) for p in ALLOWED), line
    assert 'feasibility level set: T1' in text                         # rule of FakeRunner: ADRC only at 0.2
    wit = _csv(out / 'sealed' / 'witness.csv')
    assert len(wit) == 6                                               # always 6 rows (design 5)
    found = [r for r in wit if r['found'] == 'true']
    for r in found:                                                    # no witness gain on the console
        for k, v in r.items():
            if k.startswith('gain_') and v:
                assert v not in text and f'{float(v):.4f}' not in text
    assert 'eval' not in text.lower() and ' gen' not in text
    for f in ('box.csv', 'default_gains.csv', 'labels.csv', 'meta.yaml', 'console.txt'):
        assert (out / f).is_file(), f
    for f in ('history.csv', 'recheck.csv', 'inherit.csv', 'mu_lohi.csv', 'notes.json', 'manifest.json'):
        assert (out / 'sealed' / f).is_file(), f
    man = json.loads((out / 'sealed' / 'manifest.json').read_text())
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    assert json.dumps(man, sort_keys=True) in meta                    # SHA-256 in meta.yaml (O1 3.9 (5))
    assert 'elapsed' not in meta                                       # time only in sealed/notes.json
    assert (out / 'console.txt').read_text(encoding='utf-8') == text
    lab = _csv(out / 'labels.csv')
    assert lab[-1]['levels'] == 'T1'
    dg = _csv(out / 'default_gains.csv')
    assert len(dg) == 6 and all(r['mu'] == '0.436' for r in dg)
    assert not [n for n in os.listdir(out / 'sealed' / 'jobs') if n.endswith('.out.json')]


def test_search_runs_every_search_even_when_found(fake, tmp_path):
    _search(fake, tmp_path / 'run')
    pid_t3 = [c for c in FakeRunner.calls if c[0] == 'PID' and c[1] == 0.8]
    assert len(pid_t3) >= F.POP * F.N_GEN                              # T3 search: whole budget


def test_determinism_mismatch_stops_and_writes_no_witness(fake, tmp_path):
    fake_run_jobs.recheck_salt = 'different'
    out = tmp_path / 'run'
    code, text = _search(fake, out)
    assert code == 2 and 'determinism check failed' in text
    assert all(r['found'] == 'false' for r in _csv(out / 'sealed' / 'witness.csv'))
    assert not (out / 'labels.csv').exists()
    assert 'tim_thay' not in text


def test_tool_check_prints_no_label(fake, tmp_path):
    code, text = _search(fake, tmp_path / 'run', tool=True)
    assert code == 0
    assert 'tim_thay' not in text and 'feasibility level set' not in text and 'default gains' not in text
    assert 'tool-check done' in text
    assert all(c[2] == F.MU_TOOL for c in FakeRunner.calls)
    assert 'tool_check: true' in (tmp_path / 'run' / 'meta.yaml').read_text(encoding='utf-8')


def test_dirty_firmware_is_not_a_main_run(fake, tmp_path):
    (fake.parent / 'fwsim_build.json').write_text(json.dumps(dict(esp32=dict(commit='fake', dirty=True))))
    code, text = _search(fake, tmp_path / 'run')
    assert code == 2 and 'dirty' in text


def test_existing_folder_refused(fake, tmp_path):
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'x').write_text('x')
    code, _ = _search(fake, out)
    assert code not in (0, None)


def test_recheck_and_inbox(fake, tmp_path):
    out = tmp_path / 'run'
    _search(fake, out)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        F.main(['recheck', '--from', str(out), '--out', str(tmp_path / 'rc'), '--levels', '0.2',
                '--a-r', '0.4', '--dll', str(fake), '--jobs', '1'])
    text = buf.getvalue()
    res = {r['controller']: r['result'] for r in _csv(tmp_path / 'rc' / 'labels_recheck.csv')}
    assert res == {'PID': 'con_dat', 'ADRC': 'con_dat'}
    assert F.o1.A_R == 0.4
    assert 'a_r: 0.4' in (tmp_path / 'rc' / 'meta.yaml').read_text(encoding='utf-8')
    assert 'gain' not in text
    F.o1.A_R = 0.5
    s03 = tmp_path / 's03'
    s03.mkdir()
    with open(s03 / 'levels.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['controller', 'factor', 'L', 'H', 'centre', 'scale', 'default', 'rule_note'])
        for c in F.CONTROLLERS:
            for n in F.s3.FACTORS[c]:
                w.writerow([c, n, 1e-9, 1e9, 1, 'log', 1, ''])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        F.main(['inbox', '--from', str(out), '--out', str(tmp_path / 'ib'), '--s03', str(s03)])
    assert buf.getvalue().strip() == 'inbox written (sealed)'
    rows = _csv(tmp_path / 'ib' / 'sealed' / 'inbox.csv')
    assert rows and all(r['inside'] == 'true' for r in rows)


def test_recheck_rejects_unknown_level_set(fake, tmp_path):
    with pytest.raises(SystemExit):
        F.main(['recheck', '--from', str(tmp_path), '--out', str(tmp_path / 'rc'), '--levels', '0.5',
                '--a-r', '0.5', '--dll', str(fake)])


def test_child_failure_reports_class_only(tmp_path):
    log = tmp_path / 'j.log'
    log.write_text('Traceback (most recent call last):\n  x\nValueError: secret 0.1234 kp_xy=3.3\n')
    assert F._exc_class(str(log)) == 'ValueError'


# ── smoke on the real DLL (files and exit code only; no number read) ─────────────────

SIM_LOCK = ['fwsim/s03_doe.py', 'fwsim/o1_feasibility.py', 'fwsim/fwsim.py', 'fwsim/tune_ga.py',
            'rpi5/trajectory_pub.py', 'rpi5/imu_model.py', 'rpi5/plant_blob.py', 'tools/run_validity.py',
            'rpi5/plant_step.py', 'rpi5/params.py', 'fwsim/o1_gate.py', 'fwsim/o1_gate_eval.py', 'tools/o1_lanes.py']


@pytest.mark.skipif(not os.path.isfile(DLL), reason='fwsim.dll not built')
def test_tool_check_smoke_real_dll(tmp_path):
    out = tmp_path / 'tc'
    r = subprocess.run([sys.executable, os.path.join(FW, 'o1_feas_search.py'), 'search', '--tool-check', '--out',
                        str(out), '--dll', DLL, '--jobs', '6'], capture_output=True, text=True, timeout=3000)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    assert 'tim_thay' not in r.stdout and 'tool-check done' in r.stdout
    assert len(_csv(out / 'sealed' / 'witness.csv')) == 6
    assert len(_csv(out / 'default_gains.csv')) == 6
    root = os.path.abspath(os.path.join(FW, '..'))
    st = subprocess.run(['git', '-C', root, 'status', '--porcelain', '--'] + SIM_LOCK, capture_output=True, text=True)
    assert st.stdout.strip() == ''                                     # S04 lock and evaluator untouched
