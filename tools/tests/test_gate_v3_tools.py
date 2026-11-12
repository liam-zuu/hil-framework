"""Tune stop gate v3 in the tools: hil_o1_score.py (gate_v3 block), rq3_log.py (GATE_COLS, --gate-json,
outcome check, summary guard). 12/11/2026, Liam CODE local_87282b70 "làm ngay đi"; after Codex round 1
(GV3-1, 2, 3, 5, 6, 8; Liam "Cho sửa 7 chỗ")."""
import csv
import datetime as dt
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'fwsim')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import hil_o1_score as H                                     # noqa: E402
import o1_gate as G                                          # noqa: E402
import o1_lanes as L                                         # noqa: E402
import protocol as P                                         # noqa: E402
import rq3_log as rl                                         # noqa: E402
from test_hil_o1_score import write_frame_log, k_log, DT     # noqa: E402


def test_hil_score_has_gate_block_g1_g2_decide(tmp_path):
    p = str(tmp_path / 'run.npz')
    write_frame_log(p, 'O1-F05')
    out = H.score(p, 'O1-F05')
    g = out['gate_v3']
    assert g['version'] == G.GATE_VERSION and g['decide'] == ['g1', 'g2'] and g['u_t'] is None
    assert g['theta_key'] == 'e_theta_cmd_max_deg'
    assert g['g1']['state'] == 'dat' and g['g2']['state'] == 'dat' and g['g4']['state'] == 'chua_tinh'
    assert g['g3']['state'] == g['g5']['state'] == 'chua_ket_luan'          # U_t not known on HIL
    assert g['g3']['q'] == pytest.approx(out['e_trk_max']) and g['g5']['q'] == pytest.approx(out['e_theta_cmd_max_deg'])
    assert g['v_cmd_max'] == pytest.approx(0.729, abs=1e-3)
    js = H._json_safe(out)
    json.dumps(js)                                            # serialisable as hil_o1_score prints it
    rl.validate_gate(js['gate_v3'], 'HIL')                    # and accepted by the RQ3 log for a HIL run
    with pytest.raises(rl.LogError, match='on platform'):
        rl.validate_gate(js['gate_v3'], 'HW')


def write_k_frame_log(path, task, yaw, k0=300):
    """Frame log of a K run: the car on the command point (k_log) with a constant yaw offset."""
    lg = k_log(task)
    pose = np.column_stack([lg['pose'][:, :2], np.full(len(lg['t']), yaw)])
    pose = np.vstack([np.zeros((k0, 3)), pose])
    v = np.vstack([np.zeros((k0, 3)), lg['v_true']])
    n = len(pose)
    mosi = np.zeros((n, 40), dtype=np.uint8)
    for k in range(n):
        st = [0.0] * 4 + [pose[k, 0], pose[k, 1], pose[k, 2], v[k, 0], v[k, 1], v[k, 2]]
        on = k >= k0 + 3
        mosi[k] = np.frombuffer(P.encode_state(st, traj_ref=(0.3, 0.0, 0.0) if on else None,
                                               traj_flags=P.TRAJ_VALID if on else 0, seq=k), dtype=np.uint8)
    manifest = dict(period_us=1000, steps=n, plant='MecanumPlant', miso_check=dict(crc_bad=0, empty=0),
                    lifecycle=dict(outcome='completed', transitions=[dict(step=0, t=0, phase='ARM'),
                                                                     dict(step=k0, t=0.3, phase='RUN')]))
    np.savez(path, manifest=np.array(json.dumps(manifest)), mosi=mosi, miso=np.zeros((n, 40), dtype=np.uint8))


@pytest.mark.parametrize('task', L.K_TASKS)
def test_hil_score_k_task_uses_start_heading(tmp_path, task):
    # GV3-8: through the K branch of H.score, with a yaw that is not zero
    p = str(tmp_path / 'k.npz')
    yaw = math.radians(3.0)
    write_k_frame_log(p, task, yaw)
    out = H.score(p, task, allow_final_set=True)
    g = out['gate_v3']
    assert g['theta_key'] == 'e_theta_start_max_deg' and g['wz_cmd_max'] == 0.0
    assert g['g5']['q'] == pytest.approx(out['e_theta_start_max_deg']) == pytest.approx(3.0, abs=0.05)
    assert g['v_cmd_max'] == pytest.approx(L.KSchedule(task).vmax)


# ── rq3_log ──
@pytest.fixture
def ds(tmp_path, monkeypatch):
    clock = [dt.datetime(2026, 11, 20, 8, 0, 0)]
    monkeypatch.setattr(rl, 'now', lambda: clock[0])
    d = str(tmp_path)
    rl.migrate(d)
    return d, clock


def _gate(decide=G.DECIDE_ALL, u_t=0.0, **kw):
    s = dict(completed='true', status='ok', e_perp_max=0.02, e_trk_max=0.03, e_theta_cmd_max_deg=3.0)
    s.update(kw)
    return H._json_safe(G.gate_v3(s, v_cmd_max=0.729, wz_cmd_max=1.634, u_t=u_t, decide=decide))


def _open(d, clock, platform='HW'):
    rl.begin(d, 'S', 'A-PID', platform, 'PID', 'O1-F05', '0.5', 'screen')
    rl.step(d, 'run')
    clock[0] += dt.timedelta(seconds=60)


def _rows(d):
    with open(os.path.join(d, rl.TUNING), newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_gate_columns_written(ds):
    d, clock = ds
    _open(d, clock)
    rl.end(d, 'khong_dat', gate=_gate(e_trk_max=0.12), results_dir='res/x', gate_sha256='ab' * 32)
    r = _rows(d)[-1]
    assert r['gate_version'] == G.GATE_VERSION and r['gate_decide'] == 'g1,g2,g3,g4,g5'
    assert r['gate_json_sha256'] == 'ab' * 32 and r['results_dir'] == 'res/x'
    assert r['g1_q'] == 'true' and r['g1_state'] == 'dat'
    assert float(r['g3_q']) == pytest.approx(0.12) and r['g3_state'] == 'khong_dat' and float(r['g3_u']) == 0.0
    assert r['g4_state'] == 'chua_tinh' and r['g4_q'] == '' and r['g4_u'] == ''
    assert float(r['g5_q']) == pytest.approx(3.0) and r['g5_state'] == 'dat'


def test_gate_needs_results_dir(ds):
    d, clock = ds
    _open(d, clock)
    with pytest.raises(rl.LogError, match='results-dir'):                  # GV3-6
        rl.end(d, 'khong_dat', gate=_gate(e_trk_max=0.12))


def test_no_dat_on_incomplete_gate_and_deciding_conditions(ds):
    d, clock = ds
    _open(d, clock)
    with pytest.raises(rl.LogError, match='not complete'):                 # GV3-1: G4 missing
        rl.end(d, 'dat', gate=_gate(), results_dir='r')
    with pytest.raises(rl.LogError, match='disagrees'):                    # G3 certainly failed
        rl.end(d, 'chua_ket_luan', gate=_gate(e_trk_max=0.12), results_dir='r')
    with pytest.raises(rl.LogError, match='on platform'):                  # GV3-R2-2: HW run, HIL deciding set
        rl.end(d, 'chua_ket_luan', gate=_gate(decide=G.DECIDE_HIL, u_t=None), results_dir='r')
    rl.end(d, 'chua_ket_luan', gate=_gate(completed='inconclusive'), results_dir='r')
    _open(d, clock, 'HIL')
    with pytest.raises(rl.LogError, match='disagrees'):                    # GV3-2: HIL, G2 certain fail decides
        rl.end(d, 'chua_ket_luan', gate=_gate(decide=G.DECIDE_HIL, u_t=None, e_perp_max=0.2), results_dir='r')
    with pytest.raises(rl.LogError, match='on platform'):                  # GV3-R2-2: HIL needs exactly g1, g2
        rl.end(d, 'chua_ket_luan', gate=_gate(e_trk_max=0.5), results_dir='r')
    rl.end(d, 'chua_ket_luan', gate=_gate(decide=G.DECIDE_HIL, u_t=None, e_trk_max=0.5), results_dir='r')
    _open(d, clock)
    rl.end(d, 'hong_he_do', fail_cause='mat_khung', gate=_gate(e_trk_max=0.12), results_dir='r')
    assert [r['run_outcome'] for r in _rows(d)] == ['chua_ket_luan', 'chua_ket_luan', 'hong_he_do']
    rl.summary(d)                                                          # the rows written by end pass


@pytest.mark.parametrize('edit,match', [
    (dict(run_outcome='dat'), 'no run is dat'),                            # GV3-1
    (dict(g4_state='dat', g4_q='0.01', g4_u='0.0', run_outcome='dat'), 'does not compute G4'),   # GV3-R2-1
    (dict(g2_q='0.2', g2_state='dat'), 'recomputed from q, U'),           # GV3-R2-1: state lied about
    (dict(gate_decide='g1,g3,g4,g5'), 'G1 and G2'),                        # GV3-R2-2
    (dict(gate_decide='g1,g2'), 'on platform'),                            # HW row with the HIL set
    (dict(gate_version='gate_v3_full'), 'only'),
    (dict(g3_u='-0.01'), 'negative'),
    (dict(g3_q='abc'), 'not a number'),
])
def test_summary_recomputes_gate_rows(ds, edit, match):
    # GV3-1, GV3-R2-1: summary recomputes every gate row from q, U, does not trust the state columns
    d, clock = ds
    _open(d, clock)
    rl.end(d, 'chua_ket_luan', gate=_gate(completed='inconclusive'), results_dir='r')
    path = os.path.join(d, rl.TUNING)
    rows = _rows(d)
    rows[-1].update(edit)
    rl._write_all(path, rl.TUNING_COLS, rows)
    with pytest.raises(rl.LogError, match=match):
        rl.summary(d)


@pytest.mark.parametrize('patch,match', [
    (lambda g: g.update(outcome_decide='dat', outcome='dat', any_certain_fail=False), 'recomputed'),
    (lambda g: g['g3'].update(state='dat'), 'recomputed from q, U'),
    (lambda g: g['g3'].update(q=None, u=-1, state='dat'), 'negative'),
    (lambda g: g['g3'].update(q='0.1'), 'not null or a finite number'),
    (lambda g: g['g2'].update(u=True), 'not null or a finite number'),
    (lambda g: g.pop('g2'), 'g2: missing'),
    (lambda g: g.update(version='gate_v3_other'), 'only'),
    (lambda g: g.update(decide=[]), 'decide'),
    (lambda g: g['g5'].update(threshold=10.0), 'threshold'),
    (lambda g: g['g1'].update(q='false'), 'g1'),
    (lambda g: g['g4'].update(state='dat'), 'g4'),
    # GV3-R2-3: a missing key is not 'unknown'; booleans by type
    (lambda g: (g['g3'].pop('u'), g['g3'].update(state='chua_ket_luan'),
                g.update(outcome='chua_ket_luan', outcome_decide='chua_ket_luan', any_certain_fail=False)), 'without'),
    (lambda g: g['g2'].pop('q'), 'without'),
    (lambda g: g.pop('complete'), 'complete: missing'),
    (lambda g: g.update(complete=0), 'not a boolean'),
    (lambda g: g.update(any_certain_fail=1), 'not a boolean'),
    (lambda g: g['g2'].update(threshold=True), 'threshold'),
    (lambda g: g['g1'].update(q=True), 'g1'),                          # JSON g1 q is a string
    # GV3-R2-2: G1, G2 always decide
    (lambda g: g.update(decide=['g1', 'g3', 'g4', 'g5']), 'G1 and G2'),
    (lambda g: g.update(decide=['g1', 'g2', 2]), 'decide'),
])
def test_validate_gate_rejects_inconsistent_json(patch, match):
    # GV3-5: the JSON is checked and recomputed, its outcome fields are not trusted
    g = _gate(e_trk_max=0.12)
    patch(g)
    with pytest.raises(rl.LogError, match=match):
        rl.validate_gate(g)


def test_read_gate_json_forms(tmp_path):
    g = _gate(e_trk_max=float('nan'))                        # NaN -> null, as the scorer writes
    a, b, c = (str(tmp_path / n) for n in ('a.json', 'b.json', 'c.json'))
    for path, obj in ((a, dict(task='O1-F05', gate_v3=g)), (b, g), (c, dict(task='O1-F05'))):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f)
    assert rl.read_gate_json(a) == rl.read_gate_json(b) == g
    with pytest.raises(rl.LogError, match='no gate_v3 block'):
        rl.read_gate_json(c)
    rl.validate_gate(g)
    cols = rl.gate_columns(g)
    assert cols['g3_q'] == '' and cols['g3_state'] == 'chua_ket_luan'


def test_migrate_adds_gate_columns_and_keeps_every_cell(ds):
    # GV3-8: every old cell, with values that tell columns apart
    d, clock = ds
    _open(d, clock)
    rl.end(d, 'khong_dat', notes='ghi chú, có dấu phẩy', fail_cause='', results_dir='res/a')
    path = os.path.join(d, rl.TUNING)
    old_cols = [c for c in rl.TUNING_COLS if c not in rl.GATE_COLS]
    rows = _rows(d)
    for i, c in enumerate(old_cols):
        rows[0][c] = rows[0][c] or f'v{i}_{c}'
    rl._write_all(path, old_cols, rows)                                    # a log of the 11/11 columns
    assert rl.migrate(d)[rl.TUNING] == 'migrated'
    new = _rows(d)
    assert list(new[0]) == rl.TUNING_COLS and len(new) == 1
    assert all(new[0][c] == rows[0][c] for c in old_cols)
    assert all(new[0][c] == '' for c in rl.GATE_COLS)


def test_cli_end_with_gate_json(ds, tmp_path):
    d, clock = ds
    f = str(tmp_path / 'score.json')
    with open(f, 'w', encoding='utf-8') as fh:
        json.dump(dict(gate_v3=_gate(e_trk_max=0.12)), fh)
    _open(d, clock)
    assert rl.main(['--dataset', d, 'end', '--outcome', 'khong_dat', '--gate-json', f, '--results-dir', 'r']) == 0
    r = _rows(d)[-1]
    assert r['g3_state'] == 'khong_dat' and r['gate_json_sha256'] == rl.sha256_file(f)
