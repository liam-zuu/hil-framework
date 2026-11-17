"""O1 v11.33 on the tools side (CODE_GATE 17/11/2026, design CLAUDE_THESIS 03_CODE/docs/2026-11-17_Thiet_ke_CODE_GATE_v1.md
sections 2, 4, 6; Liam 17/11 at CODE_GATE "Duyệt, gửi CODE luôn (Recommended)", K1/K2 "Không áp, cả năm quyết
(Recommended)"): the C8 gate file (tools/gate_c8.py), the required set per run, rq3_log end / summary against the gate
file (OG2-7), hil_o1_score --gate-file. Every number is invented (FAKE)."""
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


# 6.4 (the gate file itself): tools/tests/test_gate_c8.py


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


# ── design v2: the C8b time grid block 'luoi' (Liam 17/11 at CODE_GATE "Duyệt, code luôn (Recommended)"; the exception
# before t1 "Khi mọi nút đều có (Recommended)") ──

def _grid_gate(gf, g1_nut, hong=None, cat=None, nut_tam=None, base=None):
    """A FAKE floor gate on a grid of len(g1_nut) nodes; the G1 fields of the gate are those of node nut_tam."""
    g = base if base is not None else rl.floor_gate(_gate(gf, level='O1-F05', platform='HW', stage='screen', campaign='A-PID', th=3.0))
    n = len(g1_nut)
    k = n // 2 if nut_tam is None else nut_tam
    g['luoi'] = dict(format='luoi_v1', gio_sha256='cd' * 32, n_nut=n, d_dau_s=0.010,
                     d_cuoi_s=0.010 + 0.001 * (n - 1), buoc_max_s=0.001 if n > 1 else 0.0, nut_tam=k,
                     g1_nut=list(g1_nut), hong_nut=list(hong or [False] * n), cat_nut=list(cat or [False] * n),
                     g1_state=rl.luoi_g1(g1_nut))
    st = g['luoi']['g1_state']
    g['g1'].update(state=st, q={'dat': 'true', 'khong_dat': 'false'}.get(st, 'inconclusive'))
    g.update(o1_gate.evaluate_outcomes(g, g['decide'], g['required']))
    return g


def _floor_run(d, task='O1-F05'):
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', task, str(rl.FIG8[task]), 'screen', gain_set='A-PID/G01')
    rl.step(d, 'run')


@pytest.mark.parametrize('nodes, want', [(['dat'], 'dat'), (['chua_ket_luan'], 'chua_ket_luan'),
                                         (['dat'] * 3, 'dat'), (['khong_dat'] * 3, 'khong_dat'),
                                         (['dat', 'dat', 'khong_dat'], 'chua_ket_luan'),
                                         (['khong_dat', 'chua_ket_luan', 'khong_dat'], 'chua_ket_luan')])
def test_luoi_g1_rule(nodes, want):
    assert rl.luoi_g1(nodes) == want


def test_grid_run_g1_from_the_nodes_end_and_summary(ds):
    d, gf, tmp = ds
    _floor_run(d)
    g = _grid_gate(gf, ['dat', 'dat', 'khong_dat'])                         # centre node dat, one node not
    assert g['g1']['state'] == 'chua_ket_luan' and g['outcome_decide'] == 'chua_ket_luan'
    with pytest.raises(rl.LogError, match='gate outcome chua_ket_luan'):
        rl.end(d, 'dat', completed=1, results_dir=str(tmp), gate=g, gate_file=gf, queries=rl.fake_queries())
    row = rl.end(d, 'chua_ket_luan', completed=1, results_dir=str(tmp), gate=g, gate_file=gf,
                 queries=rl.fake_queries())
    assert (row['luoi_g1_nut'], row['luoi_g1_state'], row['luoi_n_nut']) == ('ddk', 'chua_ket_luan', '3')
    rl.summary(d, gate_file=gf)                                               # the row read back and recomputed


def test_grid_g1_state_must_be_the_grid_one(ds):
    d, gf, tmp = ds
    _floor_run(d)
    g = _grid_gate(gf, ['dat', 'dat', 'khong_dat'])
    g['g1'].update(state='dat', q='true')
    g.update(o1_gate.evaluate_outcomes(g, g['decide'], g['required']))
    with pytest.raises(rl.LogError, match='grid gives'):
        rl.end(d, 'dat', completed=1, results_dir=str(tmp), gate=g, gate_file=gf, queries=rl.fake_queries())


def test_grid_centre_fields_must_match_their_node(ds):
    d, gf, tmp = ds
    _floor_run(d)
    g = _grid_gate(gf, ['dat', 'khong_dat', 'dat'])                         # centre khong_dat, its fields say dat
    with pytest.raises(rl.LogError, match='grid node nut_tam'):
        rl.end(d, 'chua_ket_luan', completed=1, results_dir=str(tmp), gate=g, gate_file=gf,
               queries=rl.fake_queries())


def test_grid_every_node_lost_is_a_lost_run(ds):
    d, gf, tmp = ds
    _floor_run(d)
    base = rl.floor_gate(_gate(gf, level='O1-F05', platform='HW', stage='screen', campaign='A-PID', th=3.0),
                         t_done=(10.0, None))                              # centre fields: G1 chua_ket_luan
    g = _grid_gate(gf, ['chua_ket_luan'] * 3, hong=[True] * 3, base=base)
    with pytest.raises(rl.LogError, match='every grid node lost'):
        rl.end(d, 'chua_ket_luan', completed=1, results_dir=str(tmp), gate=g, gate_file=gf, queries=rl.fake_queries())
    row = rl.end(d, 'hong_he_do', completed=1, fail_cause='mat_khung_FAKE', results_dir=str(tmp), gate=g,
                 gate_file=gf, queries=rl.fake_queries())
    assert row['luoi_hong_nut'] == '111'


def _cat_base(gf):
    g = rl.floor_gate(_gate(gf, level='O1-F05', platform='HW', stage='screen', campaign='A-PID', th=3.0, e_trk=0.5),
                      t_done=(None, None))
    g['g1'].update(t1=6.0, cat_t1=True, state='chua_ket_luan', q='inconclusive')
    return g


def test_grid_exception_before_t1_only_when_every_node_has_it(ds):
    d, gf, tmp = ds
    _floor_run(d)
    part = _grid_gate(gf, ['chua_ket_luan'] * 3, cat=[True, True, False], nut_tam=0, base=_cat_base(gf))
    assert part['outcome_decide'] == 'khong_dat'                              # G3 certainly above D
    with pytest.raises(rl.LogError, match='every node has it'):
        rl.end(d, 'khong_dat', completed=1, fail_cause=rl.VIOL_BEFORE, results_dir=str(tmp), gate=part,
               gate_file=gf, queries=rl.fake_queries())
    full = _grid_gate(gf, ['chua_ket_luan'] * 3, cat=[True] * 3, base=_cat_base(gf))
    row = rl.end(d, 'khong_dat', completed=1, fail_cause=rl.VIOL_BEFORE, results_dir=str(tmp), gate=full,
                 gate_file=gf, queries=rl.fake_queries())
    assert row['luoi_cat_nut'] == '111'
    rl.summary(d, gate_file=gf)


def test_grid_skips_the_one_domain_equality(ds):
    d, gf, tmp = ds
    g = rl.floor_gate(_gate(gf, level='O1-F05', platform='HW', stage='screen', campaign='A-PID', th=3.0))
    g['g3']['chung']['lo'] = g['g3']['lo'] - 0.001                            # grid numbers: common below union
    g['g3']['chung']['q'] = g['g3']['q'] - 0.001
    with pytest.raises(rl.LogError, match='one frame domain'):
        rl.validate_gate(g, 'HW')
    rl.validate_gate(_grid_gate(gf, ['dat'] * 2, base=g), 'HW')


@pytest.mark.parametrize('edit, msg', [
    (lambda lb: lb.update(buoc_max_s=0.0015), 'buoc_max_s'),
    (lambda lb: lb.update(d_cuoi_s=lb['d_dau_s'] + 0.005), 'buoc_max_s'),     # 3 nodes cannot cover 5 ms at 1 ms
    (lambda lb: lb.update(n_nut=1, g1_nut=['dat'], hong_nut=[False], cat_nut=[False], nut_tam=0), 'd_dau_s'),
    (lambda lb: lb.update(nut_tam=3), 'nut_tam'),
    (lambda lb: lb.update(g1_nut=['dat', 'dat']), 'g1_nut'),
    (lambda lb: lb.update(hong_nut=[0, 0, 0]), 'hong_nut'),
    (lambda lb: lb.update(hong_nut=[True, False, False], cat_nut=[True, False, False]), 'both lost'),
    (lambda lb: lb.update(hong_nut=[True, False, False]), 'not chua_ket_luan'),   # a lost node with G1 dat
    (lambda lb: lb.update(cat_nut=[False, False, True]), 'not chua_ket_luan'),
    (lambda lb: lb.update(gio_sha256='x'), 'gio_sha256'),
    (lambda lb: lb.update(format='luoi_v0'), 'format'),
    (lambda lb: lb.update(extra=1), 'exactly'),
])
def test_grid_block_refusals(ds, edit, msg):
    d, gf, tmp = ds
    g = _grid_gate(gf, ['dat'] * 3)
    edit(g['luoi'])
    with pytest.raises(rl.LogError, match=msg):
        rl.validate_gate(g, 'HW')


def test_grid_only_on_the_floor(ds):
    d, gf, tmp = ds
    g = _grid_gate(gf, ['dat'] * 3)
    with pytest.raises(rl.LogError, match='floor gate|2 way'):          # refused either way
        rl.validate_gate(g, 'HIL')
