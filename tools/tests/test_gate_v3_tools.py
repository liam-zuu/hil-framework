"""Tune stop gate v3 in the tools: hil_o1_score.py (gate_v3 block), o1_lanes K tasks, rq3_log.py
(GATE_COLS, --gate-json, outcome check). 12/11/2026, Liam CODE local_87282b70 "làm ngay đi"."""
import csv
import datetime as dt
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'fwsim')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import hil_o1_score as H                                     # noqa: E402
import o1_gate as G                                          # noqa: E402
import o1_lanes as L                                         # noqa: E402
import rq3_log as rl                                         # noqa: E402
from test_hil_o1_score import write_frame_log, k_log         # noqa: E402


def test_hil_score_has_gate_block_not_for_decision(tmp_path):
    p = str(tmp_path / 'run.npz')
    write_frame_log(p, 'O1-F05')
    out = H.score(p, 'O1-F05')
    g = out['gate_v3']
    assert g['version'] == G.GATE_VERSION and g['usable_for_decision'] is False
    assert g['theta_key'] == 'e_theta_cmd_max_deg'
    assert g['g1']['state'] == 'dat' and g['g2']['state'] == 'dat' and g['g4']['state'] == 'chua_tinh'
    assert g['g3']['q'] == pytest.approx(out['e_trk_max']) and g['g5']['q'] == pytest.approx(out['e_theta_cmd_max_deg'])
    assert g['v_cmd_max'] == pytest.approx(0.729, abs=1e-3)
    json.dumps(H._json_safe(out))                            # serialisable as hil_o1_score prints it


def test_gate_on_k_task_uses_start_heading():
    out, _ = L.evaluate_k(k_log('O1-K1'), 'O1-K1')
    v, wz = G.k_cmd_max(L.KSchedule('O1-K1'))
    g = G.gate_v3(out, v_cmd_max=v, wz_cmd_max=wz, theta_key='e_theta_start_max_deg')
    assert wz == 0.0 and g['g5']['q'] == pytest.approx(out['e_theta_start_max_deg'])
    assert g['g5']['state'] == 'dat'


# ── rq3_log ──
@pytest.fixture
def ds(tmp_path, monkeypatch):
    clock = [dt.datetime(2026, 11, 20, 8, 0, 0)]
    monkeypatch.setattr(rl, 'now', lambda: clock[0])
    d = str(tmp_path)
    rl.migrate(d)
    return d, clock


def _gate(usable=True, **kw):
    s = dict(completed='true', status='ok', e_perp_max=0.02, e_trk_max=0.03, e_theta_cmd_max_deg=3.0)
    s.update(kw)
    return G.gate_v3(s, v_cmd_max=0.729, wz_cmd_max=1.634, usable_for_decision=usable)


def _open(d, clock):
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen')
    rl.step(d, 'run')
    clock[0] += dt.timedelta(seconds=60)


def _rows(d):
    with open(os.path.join(d, rl.TUNING), newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_gate_columns_written(ds):
    d, clock = ds
    _open(d, clock)
    rl.end(d, 'khong_dat', gate=_gate(e_trk_max=0.12))
    r = _rows(d)[-1]
    assert r['gate_version'] == G.GATE_VERSION and r['gate_usable'] == '1'
    assert r['g1_q'] == 'true' and r['g1_state'] == 'dat'
    assert float(r['g3_q']) == pytest.approx(0.12) and r['g3_state'] == 'khong_dat' and float(r['g3_u']) == 0.0
    assert r['g4_state'] == 'chua_tinh' and r['g4_q'] == '' and r['g4_u'] == ''
    assert float(r['g5_q']) == pytest.approx(3.0) and r['g5_state'] == 'dat'


def test_outcome_checked_against_gate(ds):
    d, clock = ds
    _open(d, clock)
    with pytest.raises(rl.LogError, match='disagrees with the gate'):
        rl.end(d, 'dat', gate=_gate(e_trk_max=0.12))                       # G3 certainly failed
    with pytest.raises(rl.LogError, match='disagrees with the gate'):
        rl.end(d, 'dat', gate=_gate(completed='inconclusive'))               # chua_ket_luan rules out dat
    rl.end(d, 'chua_ket_luan', gate=_gate(completed='inconclusive'))
    _open(d, clock)
    rl.end(d, 'dat', gate=_gate())                                          # G4 missing: dat not refuted
    _open(d, clock)
    rl.end(d, 'hong_he_do', fail_cause='mat_khung', gate=_gate(e_trk_max=0.12))   # lost runs: not checked
    _open(d, clock)
    rl.end(d, 'dat', gate=_gate(usable=False, e_trk_max=0.12))             # HIL report-only gate: not checked
    assert [r['run_outcome'] for r in _rows(d)] == ['chua_ket_luan', 'dat', 'hong_he_do', 'dat']


def test_complete_gate_needs_equal_outcome(ds):
    d, clock = ds
    g = dict(_gate(), complete=True, outcome='dat')
    _open(d, clock)
    with pytest.raises(rl.LogError):
        rl.end(d, 'khong_dat', gate=g)
    rl.end(d, 'dat', gate=g)


def test_read_gate_json_forms(tmp_path):
    g = H._json_safe(_gate(e_trk_max=float('nan')))                        # NaN -> null, as the scorer writes
    a, b, c = (str(tmp_path / n) for n in ('a.json', 'b.json', 'c.json'))
    json.dump(dict(task='O1-F05', gate_v3=g), open(a, 'w', encoding='utf-8'))
    json.dump(g, open(b, 'w', encoding='utf-8'))
    json.dump(dict(task='O1-F05'), open(c, 'w', encoding='utf-8'))
    assert rl.read_gate_json(a) == rl.read_gate_json(b) == g
    with pytest.raises(rl.LogError, match='no gate_v3 block'):
        rl.read_gate_json(c)
    cols = rl.gate_columns(g)
    assert cols['g3_q'] == '' and cols['g3_state'] == 'chua_ket_luan'


def test_migrate_adds_gate_columns_and_keeps_rows(ds):
    d, clock = ds
    _open(d, clock)
    rl.end(d, 'dat')
    path = os.path.join(d, rl.TUNING)
    old_cols = [c for c in rl.TUNING_COLS if c not in rl.GATE_COLS]
    rows = _rows(d)
    rl._write_all(path, old_cols, rows)                                     # a log of the 11/11 columns
    assert rl.migrate(d)[rl.TUNING] == 'migrated'
    new = _rows(d)
    assert list(new[0]) == rl.TUNING_COLS and new[0]['run_uid'] == rows[0]['run_uid'] and new[0]['g3_state'] == ''


def test_cli_end_with_gate_json(ds, tmp_path):
    d, clock = ds
    f = str(tmp_path / 'score.json')
    json.dump(dict(gate_v3=H._json_safe(_gate())), open(f, 'w', encoding='utf-8'))
    _open(d, clock)
    assert rl.main(['--dataset', d, 'end', '--outcome', 'dat', '--gate-json', f]) == 0
    assert _rows(d)[-1]['g2_state'] == 'dat'
