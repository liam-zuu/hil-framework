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
    base = _undef(_t1_base(gf, t_done=(None, None)))                         # GT1-1: the centre lost by its t1
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


# ── design v2 section 6: reruns after the C8b time check (O1 v11.37 D1, S05 v3.43 D2, D3; Liam 17/11 THESIS "Chỉ lượt
# trượt kiểm giờ", "Tổng", "Bắt buộc file", "Luật §4.6: 3 hỏng liên tiếp"; the code: Liam 17/11 at CODE_GATE) ──

def _gio(tmp, name, ok=False, fmt='c8b_gio_v1', **extra):
    """A FAKE C8b time file; its name goes into it, so two files never share a sha256 (GT1-8)."""
    p = os.path.join(str(tmp), f'{name}.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(dict(format=fmt, ok=ok, nodes_s=[0.01, 0.011], FAKE=True, name=name, **extra), f)
    return p


def _lost_gio(d, tmp, name, task='O1-F05', stage='screen', gs='A-PID/G01', gains=''):
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', task, str(rl.FIG8[task]), stage, gain_set=gs, gains=gains)
    rl.step(d, 'run')
    return rl.end(d, 'hong_he_do', completed=1, fail_cause=rl.C8B_GIO_FAIL, c8b_gio=_gio(tmp, name),
                  results_dir=str(tmp), queries=rl.fake_queries())


def test_c8b_check_needs_its_file_ok_false(ds):
    d, gf, tmp = ds
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G01')
    rl.step(d, 'run')
    kw = dict(completed=1, results_dir=str(tmp), queries=rl.fake_queries())
    with pytest.raises(rl.LogError, match='go together'):
        rl.end(d, 'hong_he_do', fail_cause=rl.C8B_GIO_FAIL, **kw)
    with pytest.raises(rl.LogError, match='ok false'):
        rl.end(d, 'hong_he_do', fail_cause=rl.C8B_GIO_FAIL, c8b_gio=_gio(tmp, 'okt', ok=True), **kw)
    with pytest.raises(rl.LogError, match='ok false'):
        rl.end(d, 'hong_he_do', fail_cause=rl.C8B_GIO_FAIL, c8b_gio=_gio(tmp, 'fmt', fmt='x'), **kw)
    with pytest.raises(rl.LogError, match='go together'):
        rl.end(d, 'hong_he_do', fail_cause='mat_khung_FAKE', c8b_gio=_gio(tmp, 'g0'), **kw)
    p = _gio(tmp, 'g1')
    row = rl.end(d, 'hong_he_do', fail_cause=rl.C8B_GIO_FAIL, c8b_gio=p, **kw)
    assert row['c8b_gio_sha256'] == rl.sha256_file(p)


def test_c8b_check_only_on_the_floor_lost(ds):
    d, gf, tmp = ds
    _hil_b(d)
    with pytest.raises(rl.LogError, match='floor run lost'):
        rl.end(d, 'hong_he_do', completed=1, fail_cause=rl.C8B_GIO_FAIL, c8b_gio=_gio(tmp, 'h'),
               gate=_gate(gf), gate_file=gf, results_dir=str(tmp))


def test_c8b_fourth_failure_stops_the_level_in_total_not_in_a_row(ds):
    d, gf, tmp = ds
    for i in range(2):
        _lost_gio(d, tmp, f'a{i}', gains='kp_xy=1.0')                         # gains logged (GT1-7 round 3)
    _lost_gio(d, tmp, 'other_gain', gs='A-PID/G02')                          # another gain: its own count
    _lost_gio(d, tmp, 'other_level', task='O1-F08')                           # another level: its own count
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G01')   # a run that passes
    rl.step(d, 'run')
    rl.end(d, 'hong_he_do', completed=1, fail_cause='mat_khung_FAKE', queries=rl.fake_queries())   # other cause
    for i in range(2, 4):                                                     # total 4 (not reset by the runs between)
        row = _lost_gio(d, tmp, f'a{i}', gains='kp_xy=1.0')
    assert 'chua ket luan' in rl.refusal_note(d, row)
    with pytest.raises(rl.LogError, match='stopped after 3 reruns for the C8b time check'):
        rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'confirm', gain_set='A-PID/G01')
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G02',
             gains='kp_xy=2.0')                                               # other gain still runs (its gains given:
    rl.cancel(d)                                                              # GT1-7 round 2)
    s = {r['campaign']: r for r in rl.summary(d)}['A-PID']
    assert s['san_gio_c8b_n'] == '6' and s['san_gio_c8b_dung'] == 'A-PID/G01 O1-F05@0.5'


def test_c8b_final_follows_the_lost_in_a_row_rule(ds):
    d, gf, tmp = ds
    for i in range(5):                                                        # final: no limit of 3 reruns here
        rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'final', gain_set='A-PID/G00')
        rl.step(d, 'run')
        if i < 2:
            rl.end(d, 'hong_he_do', completed=1, fail_cause=rl.C8B_GIO_FAIL, c8b_gio=_gio(tmp, f'f{i}'),
                   results_dir=str(tmp), queries=rl.fake_queries())
        else:
            rl.end(d, 'dat', completed=1, e_perp_max=0.02, results_dir=str(tmp),
                   gate=rl.floor_gate(rl.fake_gate(gate_file=gf)), gate_file=gf, queries=rl.fake_queries())
    assert rl.c8b_refusals(rl.read_rows(os.path.join(d, rl.TUNING))) == {}


def test_c8b_summary_refuses_an_edited_row(ds):
    d, gf, tmp = ds
    _lost_gio(d, tmp, 'x')
    p = os.path.join(d, rl.TUNING)
    rows = rl.read_rows(p)
    rows[-1]['c8b_gio_sha256'] = ''
    rl._write_all(p, rl.TUNING_COLS, rows)
    with pytest.raises(rl.LogError, match='sha256 of its C8b time file'):
        rl.summary(d)
    rows[-1].update(c8b_gio_sha256='ab' * 32, fail_cause='mat_khung_FAKE')
    rl._write_all(p, rl.TUNING_COLS, rows)
    with pytest.raises(rl.LogError, match='without fail cause'):
        rl.summary(d)


# ── CC1-2 (Codex round 1 of CODE_CAM c7_calib_drift; Liam 17/11 at CODE_GATE "Sửa, gộp nhánh này (Recommended)"): a floor
# run lost to the measurement whose scorer file has gate_v3 null is logged with the file's sha256, so so-h finds it ──

def _lost_score(d, tmp, name):
    import s05_fake_runs as fr
    folder = str(tmp / name)
    fr.write_final_run(folder, 'HW', dict(ctrl='PID', id=1, gains=dict(kp_xy=1.0, ki_xy=0.5, kp_wz=1.0, ki_wz=0.2, pos_kp=6.0, pos_ki=3.0, k_psi=1.0)),
                       score_extra=dict(outcome='hong_he_do', gate_v3=None))
    return folder, os.path.join(folder, 'score.json')


def test_cc1_2_lost_run_without_gate_through_end_and_so_h(ds):
    d, gf, tmp = ds
    folder, sp = _lost_score(d, tmp, 'lost1')
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G01')
    rl.step(d, 'run')
    assert rl.main(['--dataset', d, 'end', '--outcome', 'hong_he_do', '--fail-cause', 'mat_khung_FAKE',
                    '--results-dir', folder, '--gate-json', sp]) == 0      # GT2-1: no gate file, as CC1-2
    row = rl.read_rows(os.path.join(d, rl.TUNING))[-1]
    assert row['gate_json_sha256'] == rl.sha256_file(sp) and not rl.has_gate(row)
    assert (row['gate_file_sha256'], row['gate_report_only']) == ('', '')
    r = rl.so_h(d, row['run_uid'], rl.fake_score_final(folder, 0.8))
    assert r['outcome_sau'] == 'hong_he_do'
    rl.summary(d)


def test_cc1_2_gate_null_only_for_a_lost_run(ds):
    d, gf, tmp = ds
    folder, sp = _lost_score(d, tmp, 'lost2')
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G01')
    rl.step(d, 'run')
    with pytest.raises(rl.LogError, match='no gate_v3 block'):
        rl.read_gate_json(sp)                                                 # judged outcomes: as before
    with pytest.raises(rl.LogError, match='only for a run lost'):
        rl.end(d, 'chua_ket_luan', completed=1, results_dir=folder, gate=None, gate_sha256=rl.sha256_file(sp),
               queries=rl.fake_queries())
    with pytest.raises(rl.LogError, match='results-dir'):
        rl.end(d, 'hong_he_do', completed=1, fail_cause='mat_khung_FAKE', gate=None, gate_sha256=rl.sha256_file(sp))
    rl.end(d, 'hong_he_do', completed=1, fail_cause='mat_khung_FAKE', results_dir=folder, gate=None,
           gate_sha256=rl.sha256_file(sp))
    p = os.path.join(d, rl.TUNING)
    rows = rl.read_rows(p)
    rows[-1].update(run_outcome='khong_dat', campaign='')                    # edited by hand: refused
    rl._write_all(p, rl.TUNING_COLS, rows)
    with pytest.raises(rl.LogError, match='lost to the measurement only'):
        rl.summary(d)


# ── Codex round 1 of CODE_GATE (GT1-1, -4, -5, -6, -7, -8; Liam 17/11 at CODE_GATE "Sửa theo bảng (Recommended)"). Each
# test fails on 9f27de8 (the branch before the fixes) ──

def _f05(gf, **kw):
    return _gate(gf, level='O1-F05', platform='HW', stage='screen', campaign='A-PID', th=3.0, **kw)


def _req(gf, level='O1-F05'):
    return gf.required(level, 'HW', 'screen', 'A-PID')


def _t1_base(gf, t_done=(10.0, 10.0), cat=False):
    """A FAKE floor gate whose G1 fields carry t1 (a tracking error or a gap above t_trong) and cat_t1."""
    g = rl.floor_gate(_f05(gf), t_done=t_done)
    g['g1'].update(t1=6.0, cat_t1=cat)
    st = rl.gate_g1_state(rl.gate_g1_fields(g['g1'], 'HW'))
    g['g1'].update(state=st, q={'dat': 'true', 'khong_dat': 'false'}.get(st, 'inconclusive'))
    g.update(o1_gate.evaluate_outcomes(g, g['decide'], g['required']))
    return g


def _undef(g):
    """Every measured condition given one undefined frame of the union (what a lost grid node brings, O1 L1); states
    and outcomes recomputed."""
    for c, thr in o1_gate.THRESHOLDS.items():
        g[c]['n_undef'] = 1
        com = g[c].get('chung') or g[c]
        g[c]['state'] = rl.cond_state2(com['lo'], g[c]['lo'], g[c]['hi'], 1, thr)
    g.update(o1_gate.evaluate_outcomes(g, g['decide'], g['required']))
    return g


def test_gt1_1_a_grid_of_one_node_keeps_the_rule_of_t1(ds):
    d, gf, tmp = ds
    plain = _t1_base(gf)                                                     # t1 without the exception, no grid
    with pytest.raises(rl.LogError, match='hong_he_do'):
        rl.check_outcome_vs_gate('dat', plain, 'HW', '', _req(gf))
    rl.check_outcome_vs_gate('hong_he_do', plain, 'HW', 'mat_khung_FAKE', _req(gf))
    one = _grid_gate(gf, ['dat'], base=_t1_base(gf))                         # Codex GT1-1: the node not marked lost
    with pytest.raises(rl.LogError, match='nut_tam'):
        rl.check_outcome_vs_gate('dat', one, 'HW', '', _req(gf))
    lost = _grid_gate(gf, ['chua_ket_luan'], hong=[True], base=_undef(_t1_base(gf)))
    with pytest.raises(rl.LogError, match='hong_he_do'):
        rl.check_outcome_vs_gate('chua_ket_luan', lost, 'HW', '', _req(gf))
    rl.check_outcome_vs_gate('hong_he_do', lost, 'HW', 'mat_khung_FAKE', _req(gf))


def test_gt1_1_a_lost_centre_node_is_lost_by_its_t1(ds):
    d, gf, tmp = ds
    nodes, hong = ['dat', 'chua_ket_luan', 'dat'], [False, True, False]
    g = _grid_gate(gf, nodes, hong=hong, nut_tam=1, base=_undef(_t1_base(gf, t_done=(None, None))))
    assert g['outcome_decide'] == 'chua_ket_luan'
    rl.check_outcome_vs_gate('chua_ket_luan', g, 'HW', '', _req(gf))         # the fields alone would say khong_dat
    no_t1 = _grid_gate(gf, nodes, hong=hong, nut_tam=1, base=_undef(rl.floor_gate(_f05(gf), t_done=(10.0, None))))
    with pytest.raises(rl.LogError, match='nut_tam'):                       # lost, but its fields have no t1
        rl.check_outcome_vs_gate('chua_ket_luan', no_t1, 'HW', '', _req(gf))
    cat = _grid_gate(gf, nodes, cat=[False, True, False], nut_tam=1, base=rl.floor_gate(_f05(gf), t_done=(10.0, None)))
    with pytest.raises(rl.LogError, match='nut_tam'):                       # the exception, but no cat_t1 in the fields
        rl.check_outcome_vs_gate('chua_ket_luan', cat, 'HW', '', _req(gf))


def test_gt1_4_a_lost_node_leaves_every_measured_condition_undefined(ds):
    d, gf, tmp = ds
    nodes, hong = ['dat', 'chua_ket_luan', 'dat'], [False, True, False]
    g = _grid_gate(gf, nodes, hong=hong, nut_tam=0, base=rl.floor_gate(_f05(gf)))
    assert all(g[c]['state'] == 'dat' for c in o1_gate.MEASURED)
    with pytest.raises(rl.LogError, match='n_undef'):
        rl.check_outcome_vs_gate('chua_ket_luan', g, 'HW', '', _req(gf))
    g = _grid_gate(gf, nodes, hong=hong, nut_tam=0, base=_undef(rl.floor_gate(_f05(gf))))
    rl.check_outcome_vs_gate('chua_ket_luan', g, 'HW', '', _req(gf))


def _judged(d, tmp, name, gate, outcome, gf, task='O1-F05'):
    """A FAKE floor run logged with its scorer file (outcome, gate_v3) in its own folder."""
    folder = os.path.join(str(tmp), name)
    os.makedirs(folder)
    sp = os.path.join(folder, 'score.json')
    with open(sp, 'w', encoding='utf-8') as f:
        json.dump(dict(FAKE=True, outcome=outcome, gate_v3=gate), f)
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', task, str(rl.FIG8[task]), 'screen', gain_set='A-PID/G01')
    rl.step(d, 'run')
    row = rl.end(d, outcome, completed=1, results_dir=folder, gate=rl.read_gate_json(sp),
                 gate_sha256=rl.sha256_file(sp), gate_file=gf, queries=rl.fake_queries())
    return row, folder


def test_gt1_5_so_h_reads_the_exception_from_every_node(ds):
    d, gf, tmp = ds
    row, folder = _judged(d, tmp, 'h5', rl.floor_gate(_f05(gf), t_done=(10.0, None)), 'chua_ket_luan', gf)
    base = rl.floor_gate(_f05(gf), t_done=(None, None))
    base['g1'].update(t1=6.0, cat_t1=True)                                   # the centre node has the exception
    h1 = _grid_gate(gf, ['dat', 'chua_ket_luan', 'dat'], cat=[False, True, False], nut_tam=1, base=base)
    assert h1['outcome_decide'] == 'chua_ket_luan'
    rl.check_outcome_vs_gate('chua_ket_luan', h1, 'HW', '', _req(gf))        # 'end' takes it without a fail cause
    sf = rl.fake_score_final(folder, 3.0, h1=dict(FAKE=True, outcome='chua_ket_luan', gate_v3=h1))
    assert rl.so_h(d, row['run_uid'], sf)['outcome_sau'] == 'chua_ket_luan'


def test_gt1_5_one_rule_for_the_exception(ds):
    d, gf, tmp = ds
    base = rl.floor_gate(_f05(gf), t_done=(None, None))
    base['g1'].update(t1=6.0, cat_t1=True)
    part = _grid_gate(gf, ['dat', 'chua_ket_luan', 'dat'], cat=[False, True, False], nut_tam=1, base=base)
    assert not rl.viol_before_exception(part)
    assert rl.viol_before_exception(_grid_gate(gf, ['chua_ket_luan'] * 3, cat=[True] * 3, base=_cat_base(gf)))
    assert rl.viol_before_exception(_cat_base(gf)) and not rl.viol_before_exception(rl.floor_gate(_f05(gf)))


def test_gt1_6_a_lost_run_without_gate_keeps_its_gate_file_for_so_h(ds):
    d, gf, tmp = ds
    folder, sp = _lost_score(d, tmp, 'lost6')
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F08', str(rl.FIG8['O1-F08']), 'screen', gain_set='A-PID/G01')
    rl.step(d, 'run')
    assert rl.main(['--dataset', d, 'end', '--outcome', 'hong_he_do', '--fail-cause', 'mat_khung_FAKE',
                    '--results-dir', folder, '--gate-json', sp, '--gate-file', str(tmp / 'gate_c8.json')]) == 0
    row = rl.read_rows(os.path.join(d, rl.TUNING))[-1]
    assert (row['gate_file_sha256'], row['gate_report_only']) == (gf.sha256, 'g5') and not rl.has_gate(row)
    h1 = rl.floor_gate(rl.fake_gate(gate_file=gf, level='O1-F08', platform='HW', stage='screen', campaign='A-PID'))
    assert h1['report_only'] == ['g5'] and h1['outcome_decide'] == 'dat'
    sf = rl.fake_score_final(folder, 3.0, h1=dict(FAKE=True, outcome='dat', gate_v3=h1))   # above eps_H
    assert rl.so_h(d, row['run_uid'], sf)['outcome_sau'] == 'hong_he_do'
    rl.summary(d, gate_file=gf)
    p = os.path.join(d, rl.TUNING)
    rows = rl.read_rows(p)
    rows[-1]['gate_report_only'] = ''                                        # edited by hand: refused (by the
    rl._write_all(p, rl.TUNING_COLS, rows)                                   # so-h check again, or the row check)
    with pytest.raises(rl.LogError, match='GT1-6|report-only'):
        rl.summary(d, gate_file=gf)
    rows[-1]['gate_file_sha256'] = 'ab' * 32
    rows[-1]['gate_report_only'] = 'g5'
    rl._write_all(p, rl.TUNING_COLS, rows)
    with pytest.raises(rl.LogError, match='GT1-6|gate file SHA'):
        rl.summary(d, gate_file=gf)


G01 = 'kp_xy=1.0,ki_xy=0.5,kp_wz=1.0'


def test_gt1_7_a_floor_campaign_run_carries_its_gain_set(ds):
    d, gf, tmp = ds
    with pytest.raises(rl.LogError, match='--gain-set'):
        rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'confirm')
    p = os.path.join(d, rl.TUNING)                                           # and in the summary
    _lost_gio(d, tmp, 'x')
    rows = rl.read_rows(p)
    rows[-1]['gain_set_id'] = ''
    rl._write_all(p, rl.TUNING_COLS, rows)
    with pytest.raises(rl.LogError, match='gain set id'):
        rl.summary(d)


def test_gt1_7_another_name_or_other_values_do_not_reopen_a_stopped_level(ds):
    d, gf, tmp = ds
    for i in range(4):
        _lost_gio(d, tmp, f'r{i}', gains=G01)
    with pytest.raises(rl.LogError, match='stopped'):                        # the same gains under another name
        rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'confirm', gain_set='A-PID/G09', gains=G01)
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F08', str(rl.FIG8['O1-F08']), 'confirm', gain_set='A-PID/G09',
             gains=G01)                                                      # another level: still runs
    rl.cancel(d)
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'confirm', gain_set='A-PID/G09',
             gains='kp_xy=1.1,ki_xy=0.5,kp_wz=1.0')                          # other gains: another candidate
    rl.cancel(d)
    _lost_gio(d, tmp, 'g2', gs='A-PID/G02', gains='kp_xy=2.0,ki_xy=0.5,kp_wz=1.0')
    with pytest.raises(rl.LogError, match='other values'):                   # one gain set, one gain
        rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G02', gains=G01)


def test_gt1_8_the_time_file_belongs_to_the_run(ds):
    d, gf, tmp = ds
    run1, run2 = (os.path.join(str(tmp), n) for n in ('run1', 'run2'))
    os.makedirs(run1)
    os.makedirs(run2)
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G01')
    rl.step(d, 'run')
    kw = dict(completed=1, fail_cause=rl.C8B_GIO_FAIL, queries=rl.fake_queries())
    with pytest.raises(rl.LogError, match='results-dir'):
        rl.end(d, 'hong_he_do', c8b_gio=_gio(tmp, 'outside'), results_dir=run1, **kw)   # not under the run folder
    with pytest.raises(rl.LogError, match='results-dir'):
        rl.end(d, 'hong_he_do', c8b_gio=_gio(run1, 'nodir'), **kw)
    with pytest.raises(rl.LogError, match='task'):
        rl.end(d, 'hong_he_do', c8b_gio=_gio(run1, 'f08', task='O1-F08'), results_dir=run1, **kw)
    p = _gio(run1, 'own', task='O1-F05')
    rl.end(d, 'hong_he_do', c8b_gio=p, results_dir=run1, **kw)
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G01')
    rl.step(d, 'run')
    q = os.path.join(run2, 'copy.json')
    with open(p, 'rb') as a, open(q, 'wb') as b:
        b.write(a.read())
    with pytest.raises(rl.LogError, match='another run'):                    # the same file again
        rl.end(d, 'hong_he_do', c8b_gio=q, results_dir=run2, **kw)
    rl.end(d, 'hong_he_do', c8b_gio=_gio(run2, 'own2'), results_dir=run2, **kw)
    path = os.path.join(d, rl.TUNING)
    rows = rl.read_rows(path)
    rows[-1]['c8b_gio_sha256'] = rows[-2]['c8b_gio_sha256']                  # edited by hand: refused
    rl._write_all(path, rl.TUNING_COLS, rows)
    with pytest.raises(rl.LogError, match='another run'):
        rl.summary(d)


# ── Codex round 2 of CODE_GATE (GT1-7 rest, GT2-1, GT2-3; Liam 17/11 at CODE_GATE "Sửa 4 + Codex vòng 3
# (Recommended)"). Each test fails on 7623570 ──

def _lost_plain(d, gs, gains='', task='O1-F05'):
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', task, str(rl.FIG8[task]), 'screen', gain_set=gs, gains=gains)
    rl.step(d, 'run')
    return rl.end(d, 'hong_he_do', completed=1, fail_cause='mat_khung_FAKE', queries=rl.fake_queries())


def test_gt1_7_round2_a_known_name_without_its_gains_does_not_reopen_a_stopped_level(ds):
    d, gf, tmp = ds
    _lost_plain(d, 'A-PID/G09', gains=G01)                                   # G09 known with the gains G01
    for i in range(4):
        _lost_gio(d, tmp, f'r{i}', gains=G01)                                # G01 (same gains) stops at F05
    with pytest.raises(rl.LogError, match='stopped'):                        # Codex round 2: --gains left out
        rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G09')
    with pytest.raises(rl.LogError, match='cannot be told'):                 # a new name with no gains at all
        rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G10')
    with pytest.raises(rl.LogError, match='cannot be told'):                 # part of the gains, all equal
        rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G10', gains='kp_xy=1.0')
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G10',
             gains='kp_xy=1.0,ki_xy=0.7,kp_wz=1.0')                          # another candidate
    rl.cancel(d)
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F08', str(rl.FIG8['O1-F08']), 'screen', gain_set='A-PID/G10')
    rl.cancel(d)                                                             # no stop at F08: no gains needed


def test_gt1_7_round3_no_gain_name_in_common_is_no_evidence(ds):
    # GT1-7 (Codex round 3; Liam 17/11 at CODE_GATE "Sửa + Codex vòng 4 (Recommended)"): fails on 1d32334
    d, gf, tmp = ds
    for i in range(4):
        _lost_gio(d, tmp, f'k{i}', gains='kp_xy=1.0')                         # G01 stops with kp_xy only
    with pytest.raises(rl.LogError, match='cannot be told'):                 # names apart: no evidence
        rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G09', gains='ki_xy=0.5')
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G09',
             gains='kp_xy=1.5,ki_xy=0.5')                                    # kp_xy in common, another value
    rl.cancel(d)
    for i in range(4):                                                       # G03 stops at F08 with no gains logged
        _lost_gio(d, tmp, f'e{i}', task='O1-F08', gs='A-PID/G03')
    with pytest.raises(rl.LogError, match='cannot be told'):                 # nothing can tell another set from it
        rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F08', str(rl.FIG8['O1-F08']), 'screen', gain_set='A-PID/G04',
                 gains='kp_xy=9.0')


def test_gt2_1_a_lost_run_without_gate_needs_no_gate_file(ds):
    d, gf, tmp = ds
    folder, sp = _lost_score(d, tmp, 'lost21')
    rl.begin(d, 'S', 'A-PID', 'HW', 'PID', 'O1-F08', str(rl.FIG8['O1-F08']), 'screen', gain_set='A-PID/G01')
    rl.step(d, 'run')
    row = rl.end(d, 'hong_he_do', completed=1, fail_cause='mat_khung_FAKE', results_dir=folder, gate=None,
                 gate_sha256=rl.sha256_file(sp))                             # CC1-2 without the gate file
    assert (row.get('gate_file_sha256', ''), row.get('gate_report_only', '')) == ('', '')
    rl.summary(d)
    rl.summary(d, gate_file=gf)


def test_gt2_3_rows_without_campaign_are_not_checked_for_one_gain(ds):
    d, gf, tmp = ds
    _lost_plain(d, 'A-PID/G01', gains='kp_xy=1.0')
    _lost_plain(d, 'A-PID/G02', gains='kp_xy=2.0')
    p = os.path.join(d, rl.TUNING)
    rows = rl.read_rows(p)
    for r in rows:                                                           # historic rows: no campaign, no gain set
        r.update(campaign='', gain_set_id='')
    rl._write_all(p, rl.TUNING_COLS, rows)
    rl.summary(d)
