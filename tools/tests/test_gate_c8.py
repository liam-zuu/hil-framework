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
