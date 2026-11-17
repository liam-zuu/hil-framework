"""GT1-9 (Codex round 1 of CODE_GATE; Liam 17/11 at CODE_GATE "Sửa theo bảng (Recommended)"): fwsim/o1_gate.gate called
as before O1 v11.33 (fwsim, S03, S04) gives, byte for byte and type for type, the gate_v3.2 dict and the S03 run fields
of the base 1b41cca. The reference is not computed by this code: data/gate_v32_golden_1b41cca.json holds the sha256 of
every case (gate_v32_cases.py) as run with the modules of 1b41cca (made once, the command is in the file). Every number
is invented."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
import o1_gate as G                      # noqa: E402
import o1_feasibility as O               # noqa: E402
import o1_tasks as OT                    # noqa: E402
import s03_doe as s3                     # noqa: E402
import gate_v32_cases as C               # noqa: E402

GOLDEN = os.path.join(HERE, 'data', 'gate_v32_golden_1b41cca.json')


def _golden():
    with open(GOLDEN, encoding='utf-8') as f:
        return json.load(f)


def test_reference_is_of_the_base():
    d = _golden()
    assert d['source_commit'] == '1b41cca79565226942f16b18b01944d6c897a920' and d['n'] == len(d['digests'])


def test_old_path_is_byte_for_byte_the_base():
    want = _golden()['digests']
    got = C.digests(G, O, OT, s3)
    assert sorted(got) == sorted(want)
    bad = sorted(k for k in want if got[k] != want[k])
    assert not bad, f'{len(bad)} case(s) differ from 1b41cca, first: {bad[:5]}'


def test_cases_cover_the_five_tasks_and_every_state():
    out = C.outputs(G, O, OT, s3)
    gates = {k: v for k, v in out.items() if isinstance(v, dict) and v.get('version') == G.GATE_VERSION}
    assert {k.split('/')[0] for k in gates} == set(C.TASKS)
    for c in G.CONDS:
        assert {g[c]['state'] for g in gates.values()} == set(G.STATES), c
    assert {g['outcome'] for g in gates.values()} == set(G.STATES)
    assert any(g[c]['n_undef'] > 0 for g in gates.values() for c in G.MEASURED)
    assert all(not ({'u_mode', 'level', 'gate_file_sha256', 'report_only', 'required'} & set(g)) for g in gates.values())
    assert sum(k.endswith('/row_text') for k in out) == len(C.TASKS) * len(C.VARIANTS) * len(C.COMPLETED)
