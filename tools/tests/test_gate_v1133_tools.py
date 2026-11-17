"""O1 v11.33 on the tools side (CODE_GATE 17/11/2026, design CLAUDE_THESIS 03_CODE/docs/2026-11-17_Thiet_ke_CODE_GATE_v1.md
sections 2, 4, 6; Liam 17/11 at CODE_GATE "Duyệt, gửi CODE luôn (Recommended)", K1/K2 "Không áp, cả năm quyết
(Recommended)"): the C8 gate file (tools/gate_c8.py), the required set per run, rq3_log end / summary against the gate
file (OG2-7), hil_o1_score --gate-file. Every number is invented (FAKE)."""
import copy
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import gate_c8 as GC          # noqa: E402
import rq3_log as rl          # noqa: E402
import hil_o1_score as H      # noqa: E402
import o1_gate                # noqa: E402

LEVELS = ('O1-F02', 'O1-F05', 'O1-F08')


def _good():
    return GC.fake_gate(LEVELS, u_p_m=0.002, u_psi_deg=2.0, u_t_s=0.014)     # G5 report only at F08 (4.10 > 3.75 deg)


def _write(tmp_path, d, name='gate_c8.json'):
    p = str(tmp_path / name)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(d, f)
    return p


# ── 6.4: the gate file ──

def test_fake_gate_numbers_and_report_only():
    d = _good()
    GC.check(d)
    pl = d['dieu_2b']['per_level']
    assert pl['O1-F08']['report_only'] == ['g5'] and pl['O1-F05']['report_only'] == [] == pl['O1-F02']['report_only']
    assert pl['O1-F08']['uhat']['g5'] > 3.75 > pl['O1-F05']['uhat']['g5']


def _edit(path, value):
    def f(d):
        o = d
        for k in path[:-1]:
            o = o[k]
        if value is KeyError:
            del o[path[-1]]
        else:
            o[path[-1]] = value
    return f


F08 = ('dieu_2b', 'per_level', 'O1-F08')


@pytest.mark.parametrize('edit, msg', [
    (_edit(('format',), 'gate_c8_v0'), 'format'),
    (_edit(('levels',), []), 'levels'),
    (_edit(('levels',), ['O1-F05', 'O1-F05']), 'levels'),
    (_edit(('levels',), ['O1-K1']), 'levels'),
    (_edit(('u_t_branch',), 'tick'), 'u_t_branch'),
    (_edit(('dieu_2b', 'thresholds', 'g5'), 7.0), 'thresholds'),
    (_edit(('dieu_2b', 'u_p_m'), -0.001), 'u_p_m'),
    (_edit(('dieu_2b', 'u_psi_deg'), True), 'u_psi_deg'),
    (_edit(('dieu_2b', 'per_level', 'O1-F02'), KeyError), 'per_level'),
    (_edit(F08 + ('report_only',), []), 'report_only'),                    # the file says it decides: refused
    (_edit(F08 + ('report_only',), ['g5', 'g3']), 'report_only'),
    (_edit(F08 + ('uhat', 'g5'), 3.0), 'uhat'),
    (_edit(F08 + ('u_t_c8_s',), 0.013), 'u_t_c8_s'),
    (_edit(F08 + ('runs',), [dict(run_id='a', controller='PID', u_t_s=0.014)]), 'LADRC'),
    (_edit(F08 + ('runs',), [dict(run_id='a', controller='PID', u_t_s=0.014),
                             dict(run_id='a', controller='LADRC', u_t_s=0.014)]), 'once'),
    (_edit(F08 + ('runs',), [dict(run_id='a', controller='PID', u_t_s=float('nan')),
                             dict(run_id='b', controller='LADRC', u_t_s=0.014)]), 'u_t_s'),
    (_edit(F08 + ('v_cmd_max',), 1.0), 'v_cmd_max'),
    (_edit(F08 + ('wz_cmd_max',), 2.0), 'wz_cmd_max'),
])
def test_gate_file_refusals(tmp_path, edit, msg):
    d = _good()
    edit(d)
    with pytest.raises(GC.GateFileError, match=msg):
        GC.read_gate_c8(_write(tmp_path, d))


def test_half_threshold_equal_still_decides():
    # Uhat exactly half the threshold: '>' strict, the condition still decides (O1 3.9 rule 2b)
    uh = {'g2': 0.05, 'g3': 0.05, 'g5': 3.75}
    assert GC.report_only_of(uh) == ()
    assert GC.report_only_of({'g2': 0.0500001, 'g3': 0.0, 'g5': 3.7500001}) == ('g2', 'g5')


def test_csv_gate_file_refused(tmp_path):
    p = str(tmp_path / 'gate.csv')
    with open(p, 'w', encoding='utf-8') as f:
        f.write('level\nO1-F05\n')
    with pytest.raises(GC.GateFileError, match='JSON'):
        GC.read_gate_c8(p)


@pytest.mark.parametrize('level, platform, stage, campaign, want', [
    ('O1-F08', 'SIM', 'script', 'B-chung', o1_gate.CONDS),
    ('O1-F08', 'HW', 'screen', 'A-PID', ('g1', 'g2', 'g3', 'g4')),
    ('O1-F08', 'HW', 'final', 'A-LADRC', ('g1', 'g2', 'g3', 'g4')),
    ('O1-F05', 'HW', 'confirm', 'A-PID', o1_gate.CONDS),
    ('O1-K1', 'HW', 'final', 'A-PID', o1_gate.CONDS),                    # Liam 17/11: K1, K2 not masked
    ('O1-F08', 'HIL', 'hil_check', 'B-PID', ('g1', 'g2', 'g3', 'g4')),    # Liam 17/11 THESIS: same mask for B
    ('O1-F08', 'HIL', 'hil_check', 'B-LADRC', ('g1', 'g2', 'g3', 'g4')),
    ('O1-F08', 'HIL', 'final_hil', 'RQ2', o1_gate.CONDS),
    ('O1-F08', 'HIL', 'screen', 'A-PID', o1_gate.CONDS),
])
def test_required_by_run(tmp_path, level, platform, stage, campaign, want):
    gf = GC.read_gate_c8(_write(tmp_path, _good()))
    assert gf.required(level, platform, stage, campaign) == tuple(want)


def test_write_gate_sha(tmp_path):
    p = str(tmp_path / 'g.json')
    sha = GC.write_gate(p, _good())
    assert GC.read_gate_c8(p).sha256 == sha == GC.sha256_file(p)


# ── 6.5: rq3_log against the gate file ──

@pytest.fixture
def ds(tmp_path):
    d = str(tmp_path / 'ds')
    os.makedirs(d)
    rl.migrate(d)
    p = str(tmp_path / 'gate_c8.json')
    GC.write_gate(p, _good())
    return d, rl.read_gate_file(p), tmp_path


def _hil_b(d, task='O1-F08'):
    rl.begin(d, 'S', 'B-PID', 'HIL', 'PID', task, str(rl.FIG8[task]), 'hil_check', gain_set='B-PID/GA1')
    rl.step(d, 'run')


def _gate(gf=None, level='O1-F08', platform='HIL', stage='hil_check', campaign='B-PID', th=9.0, **kw):
    return rl.fake_gate(gate_file=gf, level=level, platform=platform, stage=stage, campaign=campaign, e_theta_deg=th,
                        **kw)


def test_end_hil_b_with_g5_report_only_is_dat(ds):
    d, gf, tmp = ds
    _hil_b(d)
    g = _gate(gf)
    assert g['g5']['state'] == 'khong_dat' and g['outcome_decide'] == 'dat'
    row = rl.end(d, 'dat', completed=1, results_dir=str(tmp), gate=g, gate_sha256='x', gate_file=gf)
    assert row['gate_version'] == 'gate_v3.3' and row['gate_report_only'] == 'g5'
    assert row['gate_file_sha256'] == gf.sha256 and row['gate_decide'] == 'g1,g2,g3,g4'
    rl.summary(d, gate_file=gf)                                               # the row passes the summary checks


def test_end_refuses_khong_dat_by_a_report_only_condition(ds):
    d, gf, tmp = ds
    _hil_b(d)
    with pytest.raises(rl.LogError, match='gate outcome dat'):
        rl.end(d, 'khong_dat', completed=1, results_dir=str(tmp), gate=_gate(gf), gate_file=gf)


def test_end_masked_gate_needs_the_gate_file(ds):
    d, gf, tmp = ds
    _hil_b(d)
    with pytest.raises(rl.LogError, match='not given here'):
        rl.end(d, 'dat', completed=1, results_dir=str(tmp), gate=_gate(gf))


def test_end_refuses_another_gate_file(ds):
    d, gf, tmp = ds
    p2 = str(tmp / 'other.json')
    GC.write_gate(p2, GC.fake_gate(LEVELS, u_psi_deg=1.0, u_t_s=0.014))
    _hil_b(d)
    with pytest.raises(rl.LogError, match='scored with the gate file'):
        rl.end(d, 'dat', completed=1, results_dir=str(tmp), gate=_gate(gf), gate_file=rl.read_gate_file(p2))


def test_end_refuses_a_self_declared_exemption(ds):
    # OG2-7: the run says G5 is report only at F05, the gate file says it decides there
    d, gf, tmp = ds
    _hil_b(d, 'O1-F05')
    n = 20
    frames = {'e_perp': [0.02] * n, 'e_trk': [0.03] * n, 'e_bao': [0.04] * n, 'e_theta_cmd_deg': [9.0] * n}
    g = rl._plain(o1_gate.gate(frames, 'true', v_cmd_max=0.729, wz_cmd_max=1.634, u_t=0.0015, report_only=['g5'],
                               level='O1-F05', gate_file_sha256=gf.sha256))
    with pytest.raises(rl.LogError, match='gate file gives'):
        rl.end(d, 'dat', completed=1, results_dir=str(tmp), gate=g, gate_file=gf)


def test_end_refuses_an_exemption_without_any_gate_file(ds):
    d, gf, tmp = ds
    _hil_b(d)
    g = _gate(gf)
    g['gate_file_sha256'] = None
    with pytest.raises(rl.LogError, match='no condition is exempt'):
        rl.end(d, 'dat', completed=1, results_dir=str(tmp), gate=g)


def test_end_final_hil_cannot_carry_the_mask(ds):
    d, gf, tmp = ds
    rl.begin(d, 'S', 'RQ2', 'HIL', 'PID', 'O1-F08', '0.8', 'final_hil', gain_set='A-PID/G00')
    rl.step(d, 'run')
    g = _gate(gf)                                                             # masked as for a B candidate check
    with pytest.raises(rl.LogError, match='gate file gives'):
        rl.end(d, 'dat', completed=1, results_dir=str(tmp), gate=g, gate_file=gf,
               queries=rl.fake_queries())


def test_end_unmapped_hil_b_still_a_lost_run(ds):
    d, gf, tmp = ds
    _hil_b(d)
    g = _gate(gf, mapped=False)
    assert g['decide'] == ['g1', 'g2'] and g['complete'] is False
    with pytest.raises(rl.LogError, match='lost run'):
        rl.end(d, 'dat', completed=1, results_dir=str(tmp), gate=g, gate_file=gf)
    row = rl.end(d, 'hong_he_do', completed=1, fail_cause=rl.MAP_REFUSED, results_dir=str(tmp), gate=g,
                 gate_file=gf)
    assert row['gate_decide'] == 'g1,g2'


def test_summary_refuses_a_floor_campaign_run_scored_without_the_gate_file(ds):
    d, gf, tmp = ds
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G01')
    rl.step(d, 'run')
    rl.end(d, 'khong_dat', completed=1, e_perp_max=0.2, results_dir=str(tmp),
           gate=rl.floor_gate(rl.fake_gate(e_trk=0.5)), queries=rl.fake_queries())       # v3.2, no gate file
    rl.summary(d)                                                             # without the gate file: as before
    with pytest.raises(rl.LogError, match='scored without the C8 gate file'):
        rl.summary(d, gate_file=gf)


def test_summary_refuses_a_row_whose_report_only_was_edited(ds):
    d, gf, tmp = ds
    _hil_b(d)
    rl.end(d, 'dat', completed=1, results_dir=str(tmp), gate=_gate(gf), gate_file=gf)
    p = os.path.join(d, rl.TUNING)
    rows = rl.read_rows(p)
    rows[-1]['gate_report_only'] = 'g3,g5'
    rows[-1]['gate_decide'] = 'g1,g2,g4'
    rl._write_all(p, rl.TUNING_COLS, rows)
    with pytest.raises(rl.LogError, match='gate file gives'):
        rl.summary(d, gate_file=gf)


def test_floor_run_masked_at_f08(ds):
    d, gf, tmp = ds
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F08', '0.8', 'screen', gain_set='A-PID/G01')
    rl.step(d, 'run')
    g = rl.floor_gate(_gate(gf, platform='HW', stage='screen', campaign='A-PID'))
    assert g['outcome_decide'] == 'dat' and g['g5']['state'] == 'khong_dat'
    row = rl.end(d, 'dat', completed=1, e_perp_max=0.02, results_dir=str(tmp), gate=g, gate_file=gf,
                 queries=rl.fake_queries())
    assert row['gate_report_only'] == 'g5'
    rl.summary(d, gate_file=gf)


# ── 6.6: hil_o1_score --gate-file ──

def _f08_log(tmp_path, monkeypatch):
    sys.path.insert(0, HERE)
    import test_hil_o1_score as TH
    p = str(tmp_path / 'run.npz')
    TH.write_frame_log(p, 'O1-F08')
    monkeypatch.setattr(H.hil_edge_map, 'read_esp_log', lambda path: {})
    monkeypatch.setattr(H.hil_edge_map, 'load_frames', lambda path: {})
    monkeypatch.setattr(H.hil_edge_map, 'map_edge',
                        lambda *a, **k: dict(ok=True, edge_step=400.0, u_t_s=0.008, offset_ms=0.0, reason=''))
    return p


def test_hil_score_with_and_without_the_gate_file(tmp_path, monkeypatch):
    p = _f08_log(tmp_path, monkeypatch)
    gp = _write(tmp_path, _good())
    plain = H.score(p, 'O1-F08', esp_log='esp.csv')['gate_v3']
    b = H.score(p, 'O1-F08', esp_log='esp.csv', gate_file=gp, stage='hil_check', campaign='B-PID')['gate_v3']
    fin = H.score(p, 'O1-F08', esp_log='esp.csv', gate_file=gp, stage='final_hil', campaign='RQ2')['gate_v3']
    assert plain['version'] == b['version'] == fin['version'] == 'gate_v3.3'   # per-frame U (U_t known)
    assert plain['u_mode'] == 'khung' and plain['gate_file_sha256'] is None and plain['report_only'] == []
    assert b['report_only'] == ['g5'] and b['decide'] == ['g1', 'g2', 'g3', 'g4'] and b['complete'] is True
    assert b['gate_file_sha256'] == GC.sha256_file(gp) and b['level'] == 'O1-F08'
    assert fin['report_only'] == [] and fin['decide'] == list(o1_gate.CONDS)
    for c in o1_gate.MEASURED:                                                # every condition still computed, same
        for k in ('q', 'lo', 'hi', 'n_undef', 'state'):
            assert plain[c][k] == b[c][k] == fin[c][k] or all(
                isinstance(x, float) and math.isnan(x) for x in (plain[c][k], b[c][k], fin[c][k]))
    # the per-frame U is not above the constant one, and below it somewhere (frames near t = 0)
    v, wz = o1_gate.fig8_cmd_max(H.Schedule(H.FIG8['O1-F08']))
    assert plain['g3']['u'] <= v * 0.008 * (1 + 1e-9) and plain['g5']['u'] <= math.degrees(wz * 0.008) * (1 + 1e-9)


def test_hil_score_gate_accepted_by_rq3_log(tmp_path, monkeypatch):
    p = _f08_log(tmp_path, monkeypatch)
    gp = _write(tmp_path, _good())
    g = rl._plain(H._json_safe(H.score(p, 'O1-F08', esp_log='esp.csv', gate_file=gp, stage='hil_check',
                                       campaign='B-PID')['gate_v3']))
    req = rl.read_gate_file(gp).required('O1-F08', 'HIL', 'hil_check', 'B-PID')
    ev = rl.validate_gate(g, 'HIL', req)
    assert ev['complete'] is True


def test_hil_score_cli_stage_without_gate_file(tmp_path):
    out = H.main(['--frame-log', str(tmp_path / 'x.npz'), '--task', 'O1-F08', '--stage', 'hil_check'])
    assert out == 2
