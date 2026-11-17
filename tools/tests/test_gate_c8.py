"""tools/gate_c8.py: the C8 gate file gate_c8_v1 (reader, checks, required set per run). CODE_GATE 17/11/2026, design
CLAUDE_THESIS 03_CODE/docs/2026-11-17_Thiet_ke_CODE_GATE_v1.md section 2, 4 (Liam 17/11 at CODE_GATE "Duyệt, gửi CODE
luôn (Recommended)"; K1, K2: "Không áp, cả năm quyết (Recommended)"). Every number is invented (FAKE)."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import gate_c8 as GC          # noqa: E402
import o1_gate                # noqa: E402

LEVELS = ('O1-F02', 'O1-F05', 'O1-F08')


def _good():
    return GC.fake_gate(LEVELS, u_p_m=0.002, u_psi_deg=2.0, u_t_s=0.014)     # G5 report only at F08 (4.10 > 3.75 deg)


def _write(tmp_path, d, name='gate_c8.json'):
    p = str(tmp_path / name)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(d, f)
    return p


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


def test_unknown_platform_refused(tmp_path):
    gf = GC.read_gate_c8(_write(tmp_path, _good()))
    with pytest.raises(GC.GateFileError, match='platform'):
        gf.required('O1-F08', 'CAR', 'final', 'A-PID')


def test_write_gate_sha(tmp_path):
    p = str(tmp_path / 'g.json')
    sha = GC.write_gate(p, _good())
    assert GC.read_gate_c8(p).sha256 == sha == GC.sha256_file(p)
    with pytest.raises(GC.GateFileError):
        d = _good()
        d['levels'] = []
        GC.write_gate(str(tmp_path / 'bad.json'), d)                          # never written when it does not check
    assert not os.path.exists(str(tmp_path / 'bad.json'))


# ── Codex Sol round 1 (17/11, account 1, session 01a1e87f): GC1-1 ... GC1-6 ──

def test_gc1_1_tolerance_cannot_exempt_a_condition_at_half(tmp_path):
    # G2 recomputed exactly at half the threshold must decide; a stored uhat 5e-10 above it may not exempt it
    d = GC.fake_gate(('O1-F08',), u_p_m=0.05, u_psi_deg=0.0, u_t_s=0.001)
    e = d['dieu_2b']['per_level']['O1-F08']
    assert 'g2' not in e['report_only']
    e['uhat']['g2'] = 0.050000000025
    e['report_only'] = ['g2'] + [c for c in e['report_only'] if c != 'g2']
    with pytest.raises(GC.GateFileError, match='report_only'):
        GC.read_gate_c8(_write(tmp_path, d))
    with pytest.raises(GC.GateFileError, match='report_only'):
        GC.write_gate(str(tmp_path / 'w.json'), d)


def test_gc1_1_stored_peak_within_tolerance_does_not_move_the_mask(tmp_path):
    # GC2-4 (round 2): the canonical G3 just below half, a stored peak within REL_TOL that would put it above half;
    # the other report-only conditions kept as they are, so only G3 differs
    sch_v = GC.schedule_peaks('O1-F02')[0]
    u_t = (0.05 - 1e-12 - 0.002) / sch_v                                     # canonical G3 = half - 1e-12
    d = GC.fake_gate(('O1-F02',), u_p_m=0.002, u_psi_deg=0.0, u_t_s=u_t)
    e = d['dieu_2b']['per_level']['O1-F02']
    assert 'g3' not in e['report_only']
    stored_v = e['v_cmd_max'] * (1 + 5e-10)
    stored_g3 = 0.002 + u_t * stored_v
    assert stored_g3 > 0.05 and GC._close(stored_g3, e['uhat']['g3'])        # above half, within REL_TOL
    e['v_cmd_max'] = stored_v
    e['uhat']['g3'] = stored_g3
    e['report_only'] = sorted(set(e['report_only']) | {'g3'}, key=GC.MASKABLE.index)
    with pytest.raises(GC.GateFileError, match='report_only'):
        GC.check(d)


def test_gc2_1_strict_greater_than_half(tmp_path):
    # G2 canonical bound 0.050000000025 > 0.05: report only (no tolerance band in the rule, GC2-1)
    d = GC.fake_gate(('O1-F08',), u_p_m=0.050000000025, u_psi_deg=0.0, u_t_s=0.001)
    e = d['dieu_2b']['per_level']['O1-F08']
    assert 'g2' in e['report_only']
    GC.read_gate_c8(_write(tmp_path, d))
    e['report_only'] = [c for c in e['report_only'] if c != 'g2']
    with pytest.raises(GC.GateFileError, match='report_only'):
        GC.check(d)
    assert GC.report_only_of({'g2': 0.05, 'g3': 0.05, 'g5': 3.75}) == ()


def test_gc2_2_writer_never_writes_a_file_the_reader_refuses(tmp_path):
    d = GC.fake_gate()
    d['camera'] = {1: 0, '1': 1}                                             # two keys that serialise to one
    p = str(tmp_path / 'w.json')
    with pytest.raises(GC.GateFileError, match='duplicate'):
        GC.write_gate(p, d)
    assert not os.path.exists(p)


@pytest.mark.parametrize('bad', [True, '0.014', float('nan'), 0.0, 10 ** 400])
def test_gc2_3_level_block_checks_types(bad):
    with pytest.raises(GC.GateFileError):
        GC.level_block('O1-F08', 0.002, 2.0, [dict(run_id='p', controller='PID', u_t_s=bad),
                                              dict(run_id='l', controller='LADRC', u_t_s=0.014)])
    with pytest.raises(GC.GateFileError):
        GC.level_block('O1-F08', True, 2.0, [dict(run_id='p', controller='PID', u_t_s=0.01),
                                             dict(run_id='l', controller='LADRC', u_t_s=0.014)])


def test_gc1_2_overflow_in_the_recomputed_bound(tmp_path):
    d = GC.fake_gate(('O1-F02',))
    e = d['dieu_2b']['per_level']['O1-F02']
    for r in e['runs']:
        r['u_t_s'] = 1e307
    e['u_t_c8_s'] = 1e307
    e['uhat']['g3'] = GC.uhat_of(0.002, 2.0, 1e307, *GC.schedule_peaks('O1-F02'))['g3']    # finite, matches
    e['uhat']['g5'] = 0.0                                                    # its recomputed value overflows to inf
    with pytest.raises(GC.GateFileError, match='not finite'):
        GC.read_gate_c8(_write(tmp_path, d))
    assert not GC._close(1.0, float('inf')) and not GC._close(float('inf'), float('inf'))


def test_gc1_3_peak_without_the_margin():
    # F02: the speed peak is sqrt(2) A w0 = Schedule.vmax (tau = 0, pi on the hold); U_t that puts the true G3 just
    # below / above half: decides / report only (the margin of fig8_cmd_max must not decide it)
    vmax = GC.Schedule(GC.FIG8['O1-F02']).vmax
    assert GC.schedule_peaks('O1-F02')[0] == pytest.approx(vmax, rel=1e-12)
    below = GC.level_block('O1-F02', 0.002, 0.0, [dict(run_id='a', controller='PID', u_t_s=(0.04999999 - 0.002) / vmax),
                                                    dict(run_id='b', controller='LADRC', u_t_s=0.001)])
    above = GC.level_block('O1-F02', 0.002, 0.0, [dict(run_id='a', controller='PID', u_t_s=(0.05000001 - 0.002) / vmax),
                                                    dict(run_id='b', controller='LADRC', u_t_s=0.001)])
    assert 'g3' not in below['report_only'] and 'g3' in above['report_only']


@pytest.mark.parametrize('old, new', [('"g5": 7.5', '"g5": 0, "g5": 7.5'),
                                      ('"run_id": "FAKE_O1-F08_PID"', '"run_id": "x", "run_id": "FAKE_O1-F08_PID"')])
def test_gc1_4_duplicate_json_keys_refused(tmp_path, old, new):
    raw = json.dumps(_good())
    assert old in raw
    p = str(tmp_path / 'dup.json')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(raw.replace(old, new, 1))
    with pytest.raises(GC.GateFileError, match='duplicate'):
        GC.read_gate_c8(p)


def test_gc1_4_duplicate_level_block_refused(tmp_path):
    d = _good()
    blk = json.dumps(d['dieu_2b']['per_level']['O1-F05'])
    raw = json.dumps(d).replace('"O1-F05": ' + blk, '"O1-F05": {"x": 1}, "O1-F05": ' + blk, 1)
    p = str(tmp_path / 'dup.json')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(raw)
    with pytest.raises(GC.GateFileError, match='duplicate'):
        GC.read_gate_c8(p)


def test_gc1_5_integer_beyond_float(tmp_path):
    d = _good()
    d['dieu_2b']['u_p_m'] = 10 ** 400
    with pytest.raises(GC.GateFileError, match='float range'):
        GC.check(d)
    with pytest.raises(GC.GateFileError):
        GC.read_gate_c8(_write(tmp_path, d))


@pytest.mark.parametrize('uts', [(0.010, 0.014), (0.014, 0.010)])
def test_gc1_6_u_t_c8_is_the_largest_of_different_runs(tmp_path, uts):
    d = GC.fake_gate(('O1-F08',), u_t_s={'O1-F08': uts})
    assert d['dieu_2b']['per_level']['O1-F08']['u_t_c8_s'] == 0.014
    GC.read_gate_c8(_write(tmp_path, d))
    d['dieu_2b']['per_level']['O1-F08']['u_t_c8_s'] = 0.010
    with pytest.raises(GC.GateFileError, match='u_t_c8_s'):
        GC.check(d)


@pytest.mark.parametrize('level, platform, stage, campaign, want', [
    ('O1-F08', 'HIL', 'final_hil', 'B-PID', o1_gate.CONDS),             # B campaign but not the candidate check
    ('O1-F08', 'HIL', 'hil_check', 'A-PID', o1_gate.CONDS),             # candidate check stage but not B
    ('O1-F08', 'HIL', 'hil_check', 'B-chung', o1_gate.CONDS),
    ('O1-K2', 'HW', 'final', 'A-LADRC', o1_gate.CONDS),
    ('O1-F02', 'HW', 'screen', 'A-PID', o1_gate.CONDS),                 # a gate level with nothing report only
])
def test_gc1_6_required_more_cases(tmp_path, level, platform, stage, campaign, want):
    gf = GC.read_gate_c8(_write(tmp_path, _good()))
    assert gf.required(level, platform, stage, campaign) == tuple(want)
    d = _good()
    d['levels'] = ['O1-F05', 'O1-F08']
    del d['dieu_2b']['per_level']['O1-F02']
    gf2 = GC.read_gate_c8(_write(tmp_path, d, 'two.json'))
    assert gf2.required('O1-F02', 'HW', 'screen', 'A-PID') == o1_gate.CONDS   # a level outside the gate levels
