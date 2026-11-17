"""Tune stop gate v3.2 in the tools: hil_o1_score.py (gate_v3 block), rq3_log.py (GATE_COLS, --gate-json,
outcome check, summary guard). 12/11/2026, Liam CODE local_87282b70 "làm ngay đi"; after Codex rounds 1-4
(GV3-1, 2, 3, 5, 6, 8; GV3-R2-n, GV3-R3-n); 13/11 gate v3.2 (G4, per-frame q, lo, hi, n_undef SC-1, SC3-1;
HIL deciding sets with the run_active edge mapping RS-4; Liam CODE local_db99d287, local_7f7b0ef4)."""
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
from test_hil_o1_score import write_frame_log, k_log         # noqa: E402


def test_hil_score_without_esp_log_decides_g1_g2(tmp_path):
    p = str(tmp_path / 'run.npz')
    write_frame_log(p, 'O1-F05')
    out = H.score(p, 'O1-F05')
    g = out['gate_v3']
    assert g['version'] == G.GATE_VERSION and g['decide'] == ['g1', 'g2'] and g['u_t'] is None
    assert g['g1']['state'] == g['g2']['state'] == g['g4']['state'] == 'dat'
    assert g['g3']['state'] == g['g5']['state'] == 'chua_ket_luan'          # U_t not known without RS-4
    assert g['g3']['q'] == pytest.approx(out['e_trk_max']) and g['g5']['q'] == pytest.approx(out['e_theta_cmd_max_deg'])
    assert g['g4']['q'] == pytest.approx(out['e_bao_max'])
    assert g['g3']['n_undef'] == g['g3']['n_frames'] > 0 and g['g2']['n_undef'] == 0
    assert g['v_cmd_max'] == pytest.approx(0.729, abs=1e-3) and not g['complete']
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
    mosi = np.zeros((n, P.FRAME_SIZE), dtype=np.uint8)
    for k in range(n):
        st = [0.0] * 4 + [pose[k, 0], pose[k, 1], pose[k, 2], v[k, 0], v[k, 1], v[k, 2]]
        on = k >= k0 + 3
        mosi[k] = np.frombuffer(P.encode_state(st, traj_ref=(0.3, 0.0, 0.0) if on else None,
                                               traj_flags=P.TRAJ_VALID if on else 0, seq=k), dtype=np.uint8)
    manifest = dict(period_us=1000, steps=n, plant='MecanumPlant', miso_check=dict(crc_bad=0, empty=0),
                    state_out_of_range=None,                        # B81-15 check ran on the Pi
                    lifecycle=dict(outcome='completed', transitions=[dict(step=0, t=0, phase='ARM'),
                                                                     dict(step=k0, t=0.3, phase='RUN')]))
    np.savez(path, manifest=np.array(json.dumps(manifest)), mosi=mosi, miso=np.zeros((n, P.FRAME_SIZE), dtype=np.uint8))


@pytest.mark.parametrize('task', L.K_TASKS)
def test_hil_score_k_task_uses_start_heading(tmp_path, task):
    # GV3-8: through the K branch of H.score, with a yaw that is not zero
    p = str(tmp_path / 'k.npz')
    yaw = math.radians(3.0)
    write_k_frame_log(p, task, yaw)
    out = H.score(p, task, allow_final_set=True)
    g = out['gate_v3']
    assert g['wz_cmd_max'] == 0.0
    assert g['g5']['q'] == pytest.approx(out['e_theta_start_max_deg']) == pytest.approx(3.0, abs=0.05)
    assert g['v_cmd_max'] == pytest.approx(L.KSchedule(task).vmax)
    assert g['g4']['q'] == pytest.approx(out['e_bao_max']) and out['e_bao_max'] > 0.0


# ── rq3_log ──
@pytest.fixture
def ds(tmp_path, monkeypatch):
    clock = [dt.datetime(2026, 11, 20, 8, 0, 0)]
    monkeypatch.setattr(rl, 'now', lambda: clock[0])
    d = str(tmp_path)
    rl.migrate(d)
    return d, clock


def _gate(decide=G.DECIDE_ALL, u_t=0.0, completed='true', n=20, **peak):
    f = dict(e_perp=np.full(n, 0.02), e_trk=np.full(n, 0.03), e_bao=np.full(n, 0.04), e_theta_cmd_deg=np.full(n, 3.0))
    for k, v in peak.items():
        f[k] = f[k].copy()
        f[k][n // 2] = v
    return H._json_safe(G.gate(f, completed, v_cmd_max=0.729, wz_cmd_max=1.634, u_t=u_t, decide=decide))


def _open(d, clock, platform='HW'):
    rl.begin(d, 'S', 'A-PID', platform, 'PID', 'O1-F05', '0.5', 'screen',
             gain_set='A-PID/G1')         # a counted HIL run names its gain set
    rl.step(d, 'run')
    clock[0] += dt.timedelta(seconds=60)


def _rows(d):
    with open(os.path.join(d, rl.TUNING), newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _end(d, outcome, gate=None, **kw):
    """rl.end with what a car run needs since B8-1 (S05 v3.20 C9 (a)-(c)): its gate as a camera scorer output
    (FAKE: rl.floor_gate, the two frame domains equal; B81-1, B81-3); a judged car or final_hil run both boot
    answers (B81-2).
    Other platforms: rl.end unchanged. Tests of the B8-1 rules themselves call rl.end."""
    row = (rl._load_open(d) or {}).get('row', {})
    if row.get('platform') == 'HW' and gate is not None and gate['g1'].get('t_done') is None:
        gate = rl.floor_gate(gate)
    if outcome in rl.VALID and rl.boot_needed(row.get('platform'), row.get('stage')):     # car, final_hil
        if 'boot_before' not in kw and 'queries' not in kw:      # R134C4-1: two answers, not two strings
            kw['queries'] = rl.fake_queries('ab')
    return rl.end(d, outcome, gate=gate, **kw)


def test_gate_columns_written(ds):
    d, clock = ds
    _open(d, clock)
    _end(d, 'khong_dat', gate=_gate(e_trk=0.12), results_dir='res/x', gate_sha256='ab' * 32)
    r = _rows(d)[-1]
    assert r['gate_version'] == G.GATE_VERSION and r['gate_decide'] == 'g1,g2,g3,g4,g5'
    assert r['gate_json_sha256'] == 'ab' * 32 and r['results_dir'] == 'res/x' and float(r['gate_u_t']) == 0.0
    assert r['g1_q'] == 'true' and r['g1_state'] == 'dat'
    assert float(r['g3_q']) == float(r['g3_lo']) == float(r['g3_hi']) == pytest.approx(0.12)
    assert r['g3_n_undef'] == '0' and r['g3_state'] == 'khong_dat'
    assert float(r['g4_q']) == pytest.approx(0.04) and r['g4_state'] == 'dat'
    assert float(r['g5_q']) == pytest.approx(3.0) and r['g5_state'] == 'dat'
    assert not any(c.endswith('_u') for c in rl.GATE_COLS)                 # SC-1: no single (q, U) pair


def test_hil_columns_without_mapping(ds):
    d, clock = ds
    _open(d, clock, 'HIL')
    _end(d, 'hong_he_do', fail_cause=rl.MAP_REFUSED, gate=_gate(decide=G.DECIDE_HIL, u_t=None), results_dir='r')
    r = _rows(d)[-1]
    assert r['gate_u_t'] == '' and r['g3_lo'] == r['g3_hi'] == '' and r['g3_n_undef'] == '20'
    assert float(r['g3_q']) == pytest.approx(0.03) and r['g3_state'] == 'chua_ket_luan'
    rl.summary(d)


def test_gate_needs_results_dir(ds):
    d, clock = ds
    _open(d, clock)
    with pytest.raises(rl.LogError, match='results-dir'):                  # GV3-6
        _end(d, 'khong_dat', gate=_gate(e_trk=0.12))


def test_outcome_against_the_gate(ds):
    d, clock = ds
    _open(d, clock)
    with pytest.raises(rl.LogError, match='disagrees'):                    # G3 certainly failed
        _end(d, 'chua_ket_luan', gate=_gate(e_trk=0.12), results_dir='r')
    with pytest.raises(rl.LogError, match='on platform'):                  # GV3-R2-2: HW run, HIL deciding set
        _end(d, 'chua_ket_luan', gate=_gate(decide=G.DECIDE_HIL, u_t=None), results_dir='r')
    with pytest.raises(rl.LogError, match='disagrees'):                    # complete gate: dat is dat
        _end(d, 'chua_ket_luan', gate=_gate(), results_dir='r')
    _end(d, 'dat', gate=_gate(), results_dir='r')                        # all five decide and pass
    _open(d, clock)
    _end(d, 'chua_ket_luan', gate=_gate(completed='inconclusive'), results_dir='r')
    _open(d, clock, 'HIL')
    unmapped = _gate(decide=G.DECIDE_HIL, u_t=None)
    # S05 v3.15 4.5 step 2 (replaces GV3-1, GV3-2): HIL without the RS-4 mapping is a lost run, whatever G1, G2 say
    for outcome, gate in (('dat', unmapped), ('chua_ket_luan', unmapped), ('khong_dat', unmapped),
                          ('khong_dat', _gate(decide=G.DECIDE_HIL, u_t=None, e_perp=0.2))):   # G2 certain fail too
        with pytest.raises(rl.LogError, match='not mapped'):
            _end(d, outcome, gate=gate, results_dir='r')
    with pytest.raises(rl.LogError, match='not mapped'):                   # another fail cause
        _end(d, 'hong_he_do', fail_cause='mat_khung', gate=unmapped, results_dir='r')
    _end(d, 'hong_he_do', fail_cause=rl.MAP_REFUSED, gate=_gate(decide=G.DECIDE_HIL, u_t=None, e_bao=0.2),
           results_dir='r')
    _open(d, clock, 'HIL')
    with pytest.raises(rl.LogError, match='has the mapping'):              # mapped gate, cause says refused
        _end(d, 'hong_he_do', fail_cause=rl.MAP_REFUSED, gate=_gate(u_t=0.0015), results_dir='r')
    with pytest.raises(rl.LogError, match='gate-json'):                    # refused needs the scorer output
        _end(d, 'hong_he_do', fail_cause=rl.MAP_REFUSED, results_dir='r')
    _end(d, 'dat', gate=_gate(u_t=0.0015), results_dir='r')             # HIL with the mapping: all five
    _open(d, clock)
    with pytest.raises(rl.LogError, match='gate-json'):                    # not a HIL run
        _end(d, 'hong_he_do', fail_cause=rl.MAP_REFUSED, gate=_gate(e_trk=0.12), results_dir='r')
    _end(d, 'hong_he_do', fail_cause='mat_khung', gate=_gate(e_trk=0.12), results_dir='r')
    assert [r['run_outcome'] for r in _rows(d)] == ['dat', 'chua_ket_luan', 'hong_he_do', 'dat', 'hong_he_do']
    rl.summary(d)                                                          # the rows written by end pass


def _hil_check(d, clock, gain_set='B-PID/GA1', task='O1-F05', speed='0.5', stage='hil_check', campaign='B-PID',
               mapped=False, gains=''):
    rl.begin(d, 'H', campaign, 'HIL', 'PID', task, speed, stage, gain_set=gain_set, gains=gains)
    rl.step(d, 'run')
    clock[0] += dt.timedelta(seconds=60)
    if mapped:
        return _end(d, 'dat', gate=_gate(u_t=0.0015), results_dir='r')
    return _end(d, 'hong_he_do', fail_cause=rl.MAP_REFUSED, gate=_gate(decide=G.DECIDE_HIL, u_t=None),
                  results_dir='r')


def test_refused_mapping_reruns_capped(ds):
    # S05 v3.15 4.5 step 2 (Q1 (a), Liam 14/11): at most 3 reruns of one gain at one level, then chua_ket_luan
    d, clock = ds
    for _ in range(rl.MAP_RERUNS_MAX):
        _hil_check(d, clock)
    _hil_check(d, clock, gain_set='B-PID/GA2')                             # another gain: its own count
    _hil_check(d, clock, task='O1-F08', speed='0.8')                       # another level: its own count
    s = {r['campaign']: r for r in rl.summary(d)}['B-PID']
    assert s['hil_map_refused_n'] == '5' and s['hil_map_stopped'] == ''
    _hil_check(d, clock)                                                   # 4th refused = 3 reruns used up
    s = {r['campaign']: r for r in rl.summary(d)}['B-PID']
    assert s['hil_map_refused_n'] == '6' and s['hil_map_stopped'] == 'B-PID/GA1 O1-F05@0.5'
    assert s['N_HIL'] == '6'                                               # lost runs still count as HIL runs
    with pytest.raises(rl.LogError, match='stopped after 3 reruns'):        # no fifth run of that level
        rl.begin(d, 'H', 'B-PID', 'HIL', 'PID', 'O1-F05', '0.5', 'hil_check', gain_set='B-PID/GA1')
    _hil_check(d, clock, gain_set='B-PID/GA2', mapped=True)                # a rerun that maps decides the level
    assert 'hil_map_refused_n' not in {r['campaign']: r for r in rl.summary(d)}['A-PID']
    rows = _rows(d)                                                        # a row written past the stop
    rows.append(dict(rows[0], run_uid='H:99', iter_no='99', wall_start='2026-11-21T08:00:00'))
    rl._write_all(os.path.join(d, rl.TUNING), rl.TUNING_COLS, rows)
    with pytest.raises(rl.LogError, match='no run after that'):
        rl.summary(d)


def test_refused_mapping_cap_not_on_final_hil(ds):
    # final_hil keeps S05 4.6 (FINAL_LOST_N lost runs in a row); refused runs there are still hong_he_do. Codex
    # S315 tests: begin refuses the next run itself (not only summary)
    d, clock = ds
    fh = dict(stage='final_hil', campaign='RQ2', gain_set='RQ2-INIT-PID/G0')
    _hil_check(d, clock, **fh)
    _hil_check(d, clock, mapped=True, **fh)                                # a valid run between: streak restarts
    for _ in range(rl.FINAL_LOST_N - 1):
        _hil_check(d, clock, **fh)
    assert rl.map_refusals(_rows(d)) == {}
    rl.summary(d)                                                          # 2 in a row: not stopped
    _hil_check(d, clock, **fh)                                             # 3 in a row: stopped
    with pytest.raises(rl.LogError, match='lost runs in a row'):
        rl.begin(d, 'H', 'RQ2', 'HIL', 'PID', 'O1-F05', '0.5', 'final_hil', gain_set='RQ2-INIT-PID/G0')


def _final_hil_tool_error(d, clock, gain_set='RQ2-INIT-PID/G0'):
    """A final_hil run lost to a HIL tool error of the bench (L50: CRC MISO, empty frame, SPIfix stuck)."""
    rl.begin(d, 'H', 'RQ2', 'HIL', 'PID', 'O1-F05', '0.5', 'final_hil', gain_set=gain_set)
    rl.step(d, 'run')
    clock[0] += dt.timedelta(seconds=60)
    return _end(d, 'hong_he_do', fail_cause='crc_miso_FAKE', results_dir='r')


@pytest.mark.parametrize('order', [('map', 'tool', 'map'), ('tool', 'map', 'tool'), ('tool', 'tool', 'map')])
def test_v3_33_final_hil_refused_mapping_counts_with_tool_errors(ds, order):
    # S05 v3.33 §4.6 (B82-10; Liam 15/11 THESIS_B8 "Đếm chung N = 3 liên tiếp (Recommended)"): a final_hil run
    # refused by the mapping is lost to the measurement and is counted together with the HIL tool errors: after
    # N = 3 lost runs in a row (any cause of the bench) in one condition of one gain set, the condition stops
    d, clock = ds
    fh = dict(stage='final_hil', campaign='RQ2', gain_set='RQ2-INIT-PID/G0')
    for k, cause in enumerate(order):
        if cause == 'map':
            _hil_check(d, clock, **fh)
        else:
            _final_hil_tool_error(d, clock)
        if k < len(order) - 1:
            rl.summary(d)                                                  # fewer than 3 in a row: not stopped
    with pytest.raises(rl.LogError, match='lost runs in a row'):
        rl.begin(d, 'H', 'RQ2', 'HIL', 'PID', 'O1-F05', '0.5', 'final_hil', gain_set='RQ2-INIT-PID/G0')
    _final_hil_tool_error(d, clock, gain_set='RQ2-INIT-PID/G1')           # another gain set: its own streak
    rl.summary(d)


def test_v3_33_a_judged_final_hil_run_restarts_the_streak(ds):
    d, clock = ds
    fh = dict(stage='final_hil', campaign='RQ2', gain_set='RQ2-INIT-PID/G0')
    _hil_check(d, clock, **fh)
    _final_hil_tool_error(d, clock)
    _hil_check(d, clock, mapped=True, **fh)                                # judged: the streak restarts
    _final_hil_tool_error(d, clock)
    _hil_check(d, clock, **fh)
    rl.begin(d, 'H', 'RQ2', 'HIL', 'PID', 'O1-F05', '0.5', 'final_hil', gain_set='RQ2-INIT-PID/G0')   # 2 in a row
    rl.summary(d)


def test_s315_1_hil_judged_without_gate(ds):
    # Codex S315-1: a HIL run judged dat / khong_dat / chua_ket_luan needs its gate, at end and in summary
    d, clock = ds
    _open(d, clock, 'HIL')
    for o in ('dat', 'khong_dat', 'chua_ket_luan'):
        with pytest.raises(rl.LogError, match='needs its scorer output'):
            _end(d, o, results_dir='r')
    _end(d, 'hong_he_do', fail_cause='mat_khung', results_dir='r')       # a lost run without gate: allowed
    _open(d, clock)
    for o in ('dat', 'khong_dat', 'chua_ket_luan'):                        # R134 (d), B81-1: no car run judged
        with pytest.raises(rl.LogError, match='camera scorer'):            # without its gate
            _end(d, o)
    _end(d, 'hong_he_do', fail_cause='mat_khung')                          # a lost car run without gate
    rl.summary(d)
    path = os.path.join(d, rl.TUNING)
    base = _rows(d)
    for edit, match in ((dict(run_outcome='dat', fail_cause=''), 'without its gate'),                 # HIL row
                        (dict(platform='HW', fail_cause=rl.MAP_REFUSED), 'not a HIL hong_he_do'),
                        (dict(fail_cause=rl.MAP_REFUSED), 'not a HIL hong_he_do')):               # HIL, no gate
        rows = [dict(r) for r in base]
        rows[0].update(edit)
        rl._write_all(path, rl.TUNING_COLS, rows)
        with pytest.raises(rl.LogError, match=match):
            rl.summary(d)


def test_s315_2_same_gain_same_level_as_numbers(ds):
    # Codex S315-2: '1' and '1.0', '0.5' and '0.50' are the same candidate, the same level; a gain set id is needed,
    # and one gain set id cannot carry other gain values at the same level
    d, clock = ds
    for g in ('kp_xy=1', 'kp_xy=1.0', 'kp_xy=1e0', 'kp_xy=1'):
        _hil_check(d, clock, gains=g, speed='0.5')
    with pytest.raises(rl.LogError, match='stopped after 3 reruns'):
        rl.begin(d, 'H', 'B-PID', 'HIL', 'PID', 'O1-F05', '0.50', 'hil_check', gain_set='B-PID/GA1',
                 gains='kp_xy=1.00')
    with pytest.raises(rl.LogError, match='other values of kp_xy'):
        rl.begin(d, 'H', 'B-PID', 'HIL', 'PID', 'O1-F05', '0.5', 'hil_check', gain_set='B-PID/GA1', gains='kp_xy=2')
    with pytest.raises(rl.LogError, match='needs --gain-set'):
        rl.begin(d, 'H', 'B-PID', 'HIL', 'PID', 'O1-F05', '0.5', 'hil_check')
    assert rl.map_refusals(_rows(d)) == {('B-PID', 'B-PID/GA1', 'O1-F05', '0.5'): 4}


def test_s315_3_stage_does_not_restart_the_count(ds):
    d, clock = ds
    for st in ('hil_check', 'screen', 'confirm', 'explore'):               # Codex S315-3: one count over the stages
        _hil_check(d, clock, stage=st)
    for st in ('hil_check', 'screen', 'explore'):
        with pytest.raises(rl.LogError, match='stopped after 3 reruns'):
            rl.begin(d, 'H', 'B-PID', 'HIL', 'PID', 'O1-F05', '0.5', st, gain_set='B-PID/GA1')


def test_s315_4_log_order_not_the_clock(ds):
    # Codex S315-4: a row added after the stop with an earlier wall_start is still after the stop
    d, clock = ds
    for _ in range(rl.MAP_RERUNS_MAX + 1):
        _hil_check(d, clock)
    rows = _rows(d)
    late = dict(rows[0], run_uid='H:9', iter_no='9', wall_start='2026-11-20T07:59:00', run_outcome='dat')
    late.update(rl.gate_columns(_gate(u_t=0.0015)))
    late['fail_cause'] = ''
    rl._write_all(os.path.join(d, rl.TUNING), rl.TUNING_COLS, rows + [late])
    with pytest.raises(rl.LogError, match='no run after that'):
        rl.summary(d)


def test_s315_5_log_changed_between_begin_and_end(ds):
    # Codex S315-5: the log as it will be is checked before the row is written; nothing written on refusal
    d, clock = ds
    for _ in range(rl.MAP_RERUNS_MAX):
        _hil_check(d, clock)
    rl.begin(d, 'H', 'B-PID', 'HIL', 'PID', 'O1-F05', '0.5', 'hil_check', gain_set='B-PID/GA1')
    rl.step(d, 'run')
    path = os.path.join(d, rl.TUNING)
    rows = _rows(d)                                                        # another window logged a 4th refusal
    rl._write_all(path, rl.TUNING_COLS, rows + [dict(rows[-1], run_uid='X:1', session_id='X', iter_no='1')])
    before = _rows(d)
    with pytest.raises(rl.LogError, match='not logged'):
        _end(d, 'hong_he_do', fail_cause=rl.MAP_REFUSED, gate=_gate(decide=G.DECIDE_HIL, u_t=None),
               results_dir='r')
    assert _rows(d) == before and os.path.exists(os.path.join(d, rl.OPEN_RUN))   # still open, log unchanged
    rl._write_all(path, rl.TUNING_COLS, rows)                              # back to 3 refusals
    rows = _rows(d)                       # another window: same set, kp_xy=2, an L50 lost run without gate (R2-4)
    l50 = dict(rows[-1], run_uid='X:1', session_id='X', iter_no='1', g_kp_xy='2', fail_cause='mat_khung')
    l50.update({c: '' for c in rl.GATE_COLS})
    rl._write_all(path, rl.TUNING_COLS, rows + [l50])
    rl.summary(d)                                                          # a valid log so far
    os.remove(os.path.join(d, rl.OPEN_RUN))
    rl.begin(d, 'H', 'B-PID', 'HIL', 'PID', 'O1-F05', '0.5', 'hil_check', gain_set='B-PID/GA1', gains='kp_xy=2')
    rl.step(d, 'run')
    rows = _rows(d)                                                        # and changes it to kp_xy=1 meanwhile
    rows[-1]['g_kp_xy'] = '1'
    rl._write_all(path, rl.TUNING_COLS, rows)
    before = _rows(d)
    with pytest.raises(rl.LogError, match='not logged.*other values of kp_xy'):
        _end(d, 'hong_he_do', fail_cause=rl.MAP_REFUSED, gate=_gate(decide=G.DECIDE_HIL, u_t=None),
               results_dir='r')
    assert _rows(d) == before


def test_s315_6_cli_note_by_stage(ds, tmp_path, capsys):
    # Codex S315-6: the CLI note follows 4.5 step 2 for hil_check and 4.6 for final_hil
    d, clock = ds
    f = str(tmp_path / 'unmapped.json')
    with open(f, 'w', encoding='utf-8') as fh:
        json.dump(dict(gate_v3=_gate(decide=G.DECIDE_HIL, u_t=None)), fh)
    end = ['--dataset', d, 'end', '--outcome', 'hong_he_do', '--fail-cause', rl.MAP_REFUSED, '--gate-json', f,
           '--results-dir', 'r']
    notes = []
    for stage, campaign, gs, n in (('hil_check', 'B-PID', 'B-PID/GA1', 4), ('final_hil', 'RQ2', 'RQ2-INIT-PID/G0', 3)):
        for _ in range(n):
            rl.begin(d, 'H', campaign, 'HIL', 'PID', 'O1-F05', '0.5', stage, gain_set=gs)
            rl.step(d, 'run')
            assert rl.main(end) == 0
            notes.append([ln for ln in capsys.readouterr().out.splitlines() if 'tu choi' in ln][0])
    assert notes[0].startswith('anh xa bi tu choi lan 1: chay lai cung gain (con 3 lan)')
    assert notes[3].startswith('anh xa bi tu choi lan 4: qua 3 lan chay lai')
    # Codex S315-R2-4: the exact branch, before and at the stop
    assert notes[4] == 'final_hil: anh xa bi tu choi, 1 luot hong lien (S05 4.6): chay lai (dung khi du 3 luot hong lien)'
    assert notes[5] == 'final_hil: anh xa bi tu choi, 2 luot hong lien (S05 4.6): chay lai (dung khi du 3 luot hong lien)'
    assert notes[6] == 'final_hil: anh xa bi tu choi, 3 luot hong lien (S05 4.6): dieu kien nay dung'


def test_s315_r2_1_level_not_the_typed_speed(ds):
    # Codex S315-R2-1: a speed within check_task's 1e-9 is the same level; nan is refused
    d, clock = ds
    for _ in range(rl.MAP_RERUNS_MAX + 1):
        _hil_check(d, clock)
    with pytest.raises(rl.LogError, match='stopped after 3 reruns'):
        rl.begin(d, 'H', 'B-PID', 'HIL', 'PID', 'O1-F05', '0.5000000001', 'hil_check', gain_set='B-PID/GA1')
    for bad in ('nan', 'inf'):
        with pytest.raises(rl.LogError, match='not a finite number'):
            rl.begin(d, 'H', 'B-PID', 'HIL', 'PID', 'O1-F05', bad, 'hil_check', gain_set='B-PID/GA1')
    rows = _rows(d)                                                        # the same by hand: summary refuses
    rl._write_all(os.path.join(d, rl.TUNING), rl.TUNING_COLS,
                  rows + [dict(rows[-1], run_uid='H:9', iter_no='9', speed_mps='0.5000000001')])
    with pytest.raises(rl.LogError, match='no run after that'):
        rl.summary(d)


def test_s315_r2_3_final_hil_one_order(ds, tmp_path, capsys):
    # Codex S315-R2-3: the PC clock goes back between runs; the CLI note, begin and summary agree (log order)
    d, clock = ds
    fh = dict(stage='final_hil', campaign='RQ2', gain_set='RQ2-INIT-PID/G0')
    _hil_check(d, clock, **fh)                                             # F1 R
    _hil_check(d, clock, **fh)                                             # F2 R
    clock[0] -= dt.timedelta(hours=1)
    _hil_check(d, clock, mapped=True, **fh)                                # F3 M, earlier wall_start
    row = _hil_check(d, clock, **fh)                                       # F4 R
    assert rl.refusal_note(d, row).startswith('final_hil: anh xa bi tu choi, 1 luot hong lien')
    rl.begin(d, 'H', 'RQ2', 'HIL', 'PID', 'O1-F05', '0.5', 'final_hil', gain_set='RQ2-INIT-PID/G0')   # not stopped
    rl.cancel(d)
    rl.summary(d)


def test_sc3_1_example_end_to_end(ds):
    # THESIS 13/11 (Codex TX-4): a frame (q, lo, hi) = (0.12, 0.11, +inf), D = 0.10, is khong_dat; the log
    # keeps lo, so summary recomputes khong_dat, not chua_ket_luan (on a car run since S05 v3.15: an unmapped HIL
    # run is hong_he_do whatever G2 says)
    d, clock = ds
    g = _gate()
    g['g2'].update(q=0.12, lo=0.11, hi=0.06, n_undef=1, state='khong_dat')  # hi: max over the finite frames
    g.update(outcome='khong_dat', outcome_decide='khong_dat', any_certain_fail=True)
    _open(d, clock)
    _end(d, 'khong_dat', gate=g, results_dir='r')
    r = _rows(d)[-1]
    assert (r['g2_lo'], r['g2_n_undef'], r['g2_state']) == ('0.11', '1', 'khong_dat')
    rl.summary(d)


@pytest.mark.parametrize('edit,match', [
    (dict(platform='HIL', run_outcome='dat', gate_decide='g1,g2', gate_u_t='', g3_lo='', g3_hi='', g3_n_undef='20',
          g3_state='chua_ket_luan', g5_lo='', g5_hi='', g5_n_undef='20', g5_state='chua_ket_luan'), 'no run is dat'),
    (dict(g2_lo='0.2', g2_q='0.2', g2_hi='0.2', g2_state='dat'), 'recomputed from lo'),   # state lied about
    (dict(g4_n_undef='1'), 'recomputed from lo'),
    (dict(g4_n_undef='-1'), 'whole number'),
    (dict(g4_n_undef='0.5'), 'whole number'),
    (dict(g3_lo='0.05'), 'lo'),                                            # lo > q
    (dict(gate_decide='g1,g3,g4,g5'), 'decide'),
    (dict(gate_decide='g1,g2'), 'on platform'),                            # HW row, HIL set
    (dict(platform='HIL', gate_decide='g1,g2'), 'U_t'),                    # HIL set with U_t given
    (dict(gate_u_t=''), 'U_t'),                                            # all five decide, U_t unknown
    (dict(gate_version='gate_v3_g1235'), 'only'),
    (dict(gate_u_t='-0.01'), 'negative'),
    (dict(g3_q='abc'), 'not a number'),
    (dict(gate_version='', g2_q='0.2', g2_state='dat', run_outcome='dat'), 'only'),   # GV3-R3-1
    (dict(gate_version='', gate_decide='', run_outcome='dat'), 'only'),              # any gate cell left
])
def test_summary_recomputes_gate_rows(ds, edit, match):
    # GV3-1, GV3-R2-1: summary recomputes every gate row, does not trust the state columns. A one-domain row
    # (SIM; B81-3: the common-domain columns are a copy of the union), each number edited in both domains
    d, clock = ds
    _open(d, clock, 'SIM')
    _end(d, 'chua_ket_luan', gate=_gate(completed='inconclusive'), results_dir='r')
    path = os.path.join(d, rl.TUNING)
    rows = _rows(d)
    rows[-1].update(edit)
    rows[-1].update({k + '_chung': v for k, v in edit.items() if k[:3] in ('g2_', 'g3_', 'g4_', 'g5_')
                     and k[3:] in rl.DOMAIN_FIELDS})
    rl._write_all(path, rl.TUNING_COLS, rows)
    with pytest.raises(rl.LogError, match=match):
        rl.summary(d)


@pytest.mark.parametrize('patch,match', [
    (lambda g: g.update(outcome_decide='dat', outcome='dat', any_certain_fail=False), 'recomputed'),
    (lambda g: g['g3'].update(state='dat'), 'recomputed from lo'),
    (lambda g: g['g3'].update(n_undef=-1), 'whole number'),
    (lambda g: g['g3'].update(n_undef=True), 'whole number'),
    (lambda g: g['g3'].update(q='0.1'), 'not null or a finite number'),
    (lambda g: g['g2'].update(lo=True), 'not null or a finite number'),
    (lambda g: g['g2'].update(hi=float('inf')), 'not null or a finite number'),
    (lambda g: g['g2'].update(lo=0.5), 'lo'),
    (lambda g: g.pop('g2'), 'g2: missing'),
    (lambda g: g.update(version='gate_v3_g1235'), 'only'),
    (lambda g: g.update(decide=[]), 'decide'),
    (lambda g: g['g5'].update(threshold=10.0), 'threshold'),
    (lambda g: g['g4'].update(threshold=0.2), 'threshold'),
    (lambda g: g['g1'].update(q='false'), 'g1'),
    # GV3-R2-3: a missing key is not 'unknown'; booleans by type
    (lambda g: g['g3'].pop('hi'), 'without'),
    (lambda g: g['g4'].pop('n_undef'), 'without'),
    (lambda g: g.pop('u_t'), 'u_t: missing'),
    (lambda g: g.pop('complete'), 'complete: missing'),
    (lambda g: g.update(complete=0), 'not a boolean'),
    (lambda g: g.update(any_certain_fail=1), 'not a boolean'),
    (lambda g: g['g2'].update(threshold=True), 'threshold'),
    (lambda g: g['g1'].update(q=True), 'g1'),                          # JSON g1 q is a string
    (lambda g: g.update(decide=['g1', 'g3', 'g4', 'g5']), 'decide'),
    (lambda g: g.update(decide=['g1', 'g2', 2]), 'decide'),
    (lambda g: g.update(u_t=None), 'U_t'),                             # all five decide, U_t unknown
])
def test_validate_gate_rejects_inconsistent_json(patch, match):
    # GV3-5: the JSON is checked and recomputed, its outcome fields are not trusted
    g = _gate(e_trk=0.12)
    patch(g)
    with pytest.raises(rl.LogError, match=match):
        rl.validate_gate(g)


def test_bool_completed_from_the_scorer_end_to_end(ds, tmp_path):
    # GV3-R3-2: gate output (completed as a Python bool, O1 v5 evaluate) -> JSON -> end -> summary
    d, clock = ds
    for completed, outcome in ((True, 'khong_dat'), (False, 'hong_he_do')):
        f = str(tmp_path / f'score_{completed}.json')
        with open(f, 'w', encoding='utf-8') as fh:
            json.dump(dict(gate_v3=_gate(completed=completed, e_trk=0.12)), fh)
        g = rl.read_gate_json(f)
        assert g['g1']['q'] == ('true' if completed else 'false')
        _open(d, clock)
        _end(d, outcome, fail_cause='mat_khung' if outcome == 'hong_he_do' else '', results_dir='r', gate=g)
    assert [r['g1_q'] for r in _rows(d)] == ['true', 'false']
    rl.summary(d)


@pytest.mark.parametrize('col', rl.GATE_COLS)
def test_any_single_gate_cell_makes_a_gate_row(ds, col):
    # GV3-R3-1 (Codex round 4, test gap): each gate column alone triggers the check; all empty does not
    d, clock = ds
    _open(d, clock, 'SIM')                                                 # B81-1: a judged car run has a gate
    _end(d, 'khong_dat')                                                 # no gate: a legacy-style row
    path = os.path.join(d, rl.TUNING)
    rows = _rows(d)
    rl.summary(d)                                                          # all gate cells empty: not checked
    rows[-1][col] = 'x'
    rl._write_all(path, rl.TUNING_COLS, rows)
    with pytest.raises(rl.LogError):
        rl.summary(d)


def test_read_gate_json_forms(tmp_path):
    g = _gate(e_trk=float('nan'))                             # NaN frame -> n_undef 1, as the scorer writes
    a, b, c = (str(tmp_path / n) for n in ('a.json', 'b.json', 'c.json'))
    for path, obj in ((a, dict(task='O1-F05', gate_v3=g)), (b, g), (c, dict(task='O1-F05'))):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f)
    assert rl.read_gate_json(a) == rl.read_gate_json(b) == g
    with pytest.raises(rl.LogError, match='no gate_v3 block'):
        rl.read_gate_json(c)
    rl.validate_gate(g)
    cols = rl.gate_columns(g)
    assert cols['g3_n_undef'] == '1' and cols['g3_state'] == 'chua_ket_luan'


def test_migrate_adds_gate_columns_and_keeps_every_cell(ds):
    # GV3-8: every old cell, with values that tell columns apart
    d, clock = ds
    _open(d, clock, 'SIM')                                                 # B81-1: a judged car run has a gate
    _end(d, 'khong_dat', notes='ghi chú, có dấu phẩy', fail_cause='', results_dir='res/a')
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
        json.dump(dict(gate_v3=rl.floor_gate(_gate(e_trk=0.12))), fh)  # a car run: camera scorer output
    _open(d, clock)
    qb, qa = _query_files(tmp_path)                                        # a judged car run: R134 (b)
    assert rl.main(['--dataset', d, 'end', '--outcome', 'khong_dat', '--gate-json', f, '--results-dir', 'r',
                    '--query-before', qb, '--query-after', qa]) == 0
    r = _rows(d)[-1]
    assert r['g3_state'] == 'khong_dat' and r['gate_json_sha256'] == rl.sha256_file(f)
    assert r['boot_before'] == r['boot_after'] == '4753dc47'


def test_g32_1_unknown_q_and_lo_never_pass(ds):
    # Codex G32-1: q, lo unknown, hi finite, n_undef 0, state dat: refused at end and at summary
    g = _gate()
    g['g2'].update(q=None, lo=None, hi=0.05, n_undef=0, state='dat')
    with pytest.raises(rl.LogError, match='n_undef 0'):
        rl.validate_gate(g)
    d, clock = ds
    _open(d, clock)
    _end(d, 'dat', gate=_gate(), results_dir='r')
    rows = _rows(d)
    rows[-1].update(g2_q='', g2_lo='')
    rl._write_all(os.path.join(d, rl.TUNING), rl.TUNING_COLS, rows)
    with pytest.raises(rl.LogError, match='n_undef 0'):
        rl.summary(d)


def test_g32_11_migrate_drops_empty_old_u_columns(ds):
    # a log written by hil 7768134 (gate_v3_g1235): g1_u..g5_u columns; empty -> dropped, a value -> refused
    d, clock = ds
    _open(d, clock, 'SIM')                                                 # B81-1: a judged car run has a gate
    _end(d, 'khong_dat', results_dir='r')
    path = os.path.join(d, rl.TUNING)
    rows = _rows(d)
    old = rl.TUNING_COLS + list(rl.OLD_GATE_U)
    rl._write_all(path, old, rows)
    assert rl.migrate(d)[rl.TUNING] == 'migrated' and list(_rows(d)[0]) == rl.TUNING_COLS
    rows = _rows(d)
    rows[0]['g3_u'] = '0.01'
    rl._write_all(path, old, rows)
    with pytest.raises(rl.LogError, match='gate_v3_g1235'):
        rl.migrate(d)


def test_s315_r3_1_begin_extra_cannot_set_identity(ds):
    # Codex S315-R3-1: an extra field cannot replace a field begin sets and checks (session_id: SKIP_PREFIX rows
    # are left out of every count), nor a gain column (gains go through --gains)
    d, clock = ds
    for k, v in (('session_id', 'EXAMPLE-R3'), ('run_uid', 'EXAMPLE-R3:1'), ('iter_no', '1'), ('g_kp_xy', 'x'),
                 ('wall_start', '2026-11-01T00:00:00')):
        with pytest.raises(rl.LogError, match='set by begin'):
            rl.begin(d, 'H1', 'B-PID', 'HIL', 'PID', 'O1-F05', '0.5', 'hil_check', gain_set='B-PID/GA1', **{k: v})
    rl.begin(d, 'H1', 'B-PID', 'HIL', 'PID', 'O1-F05', '0.5', 'hil_check', gain_set='B-PID/GA1', candidate_no='2')
    assert not _rows(d)


def _final_car(d, clock, outcome, minutes, replaces=''):
    clock[0] = dt.datetime(2026, 11, 20, 8, 0, 0) + dt.timedelta(minutes=minutes)
    rl.begin(d, 'F', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'final', gain_set='A-PID/G0', replaces_run_uid=replaces)
    rl.step(d, 'run')
    clock[0] += dt.timedelta(seconds=30)
    gate = dict(gate=_gate(), results_dir='r') if outcome == 'dat' else {}     # R134 (d) SC3-2
    return _end(d, outcome, fail_cause='mat_khung' if outcome == 'hong_he_do' else '', **gate)['run_uid']


def test_s315_r3_2_replacement_by_log_order(ds):
    # Codex S315-R3-2: a replacement is later in the log than the run it replaces, whatever the PC clock says
    d, clock = ds
    lost = _final_car(d, clock, 'hong_he_do', 10)
    _final_car(d, clock, 'dat', -60, replaces=lost)                        # clock went back: a real replacement
    rl.summary(d)
    path = os.path.join(d, rl.TUNING)
    rows = _rows(d)                                                        # the replacement written before its run
    rl._write_all(path, rl.TUNING_COLS, [rows[1], rows[0]])
    with pytest.raises(rl.LogError, match='not an earlier run'):
        rl.summary(d)


def test_s315_r3_3_summary_refuses_speed_not_finite(ds):
    # Codex S315-R3-3: summary checks the speed of every HIL / HW row
    d, clock = ds
    _open(d, clock)
    _end(d, 'hong_he_do', fail_cause='mat_khung')                          # a HW row (B81-1: judged needs a gate)
    path = os.path.join(d, rl.TUNING)
    base = _rows(d)
    for bad in ('nan', 'inf'):
        rl._write_all(path, rl.TUNING_COLS, [dict(base[0], speed_mps=bad)])
        with pytest.raises(rl.LogError, match='not a finite number'):
            rl.summary(d)


def test_s315_r4_1_replacement_uid_duplicate_or_missing(ds):
    # Codex S315-R4-1: the position of a row is its own index; a run_uid on two rows, or a replacement row
    # without run_uid, is an error
    d, clock = ds
    lost = _final_car(d, clock, 'hong_he_do', 10)                         # F:1 lost
    _final_car(d, clock, 'dat', 20, replaces=lost)                         # F:2 replaces F:1
    rl.summary(d)
    path = os.path.join(d, rl.TUNING)
    rows = _rows(d)
    # duplicate: the replacement first, then the lost run, then a second row with the replacement's uid
    rl._write_all(path, rl.TUNING_COLS, [rows[1], rows[0], dict(rows[1], replaces_run_uid='')])
    with pytest.raises(rl.LogError, match='on two rows'):
        rl.summary(d)
    rl._write_all(path, rl.TUNING_COLS, [dict(rows[1], run_uid=''), rows[0]])   # missing uid, replacement first
    with pytest.raises(rl.LogError, match='has no run_uid'):
        rl.summary(d)


# ── R134 package after Codex round 1 (Liam 14/11 CODE_R134 "Sua + vong 2 (Recommended)") ──

def _query_files(tmp_path, before='4753dc47', after='4753dc47'):
    """Two saved gain_send --query outputs as the writers save them (R134C3-7: with the '#query_pc' line)."""
    out = []
    for name, boot, t in (('qb.txt', before, (1.0, 2.0)), ('qa.txt', after, (3.0, 4.0))):
        p = tmp_path / name
        p.write_text(f'#query_pc,t_start={t[0]},t_end={t[1]}\n'
                     f'#gain_running,sha=16702ff0e688,env=pid,mac=aa:bb,boot={boot},id=7,applied_n=1,ctrl=PID\n'
                     '#ket_qua,ma=0\n', encoding='utf-8')
        out.append(str(p))
    return out


@pytest.mark.parametrize('platform', ['HW', 'SIM', 'HIL'])
def test_sc3_2_no_dat_without_the_gate_at_end(ds, platform):
    """R134 (d) SC3-2: a campaign run is never dat without its gate, on every platform."""
    d, clock = ds
    _open(d, clock, platform)
    match = {'SIM': 'SC3-2', 'HIL': 'needs its scorer output', 'HW': 'camera scorer'}[platform]   # B81-1 on HW
    with pytest.raises(rl.LogError, match=match):
        _end(d, 'dat', results_dir='r')
    _end(d, 'dat', gate=_gate(u_t=0.0015 if platform == 'HIL' else 0.0), results_dir='r')
    assert _rows(d)[-1]['run_outcome'] == 'dat'
    rl.summary(d)


def test_sc3_2_summary_refuses_a_campaign_dat_row_without_gate_columns(ds):
    """R134 (d) SC3-2: at the summary too; a row outside the campaigns (historic) is counted apart."""
    d, clock = ds
    _open(d, clock)
    _end(d, 'dat', gate=_gate(), results_dir='r')
    path = os.path.join(d, rl.TUNING)
    rows = _rows(d)
    bare = dict(rows[0], **{c: '' for c in rl.GATE_COLS})
    rl._write_all(path, rl.TUNING_COLS, [bare])
    with pytest.raises(rl.LogError, match=rl.NO_GATE_EVIDENCE):
        rl.summary(d)
    old = dict(bare, campaign='', session_id='OLD', run_uid='OLD:1')
    rl._write_all(path, rl.TUNING_COLS, [old])
    rl.summary(d)
    assert rl.count_no_gate(_rows(d)) == 1


def test_sc3_2_summary_cli_reports_the_historic_rows(ds, capsys):
    """R134C2-11: the CLI summary prints the historic dat rows without gate columns, apart."""
    d, clock = ds
    _open(d, clock)
    _end(d, 'dat', gate=_gate(), results_dir='r')
    rows = _rows(d)
    old = dict(rows[0], campaign='', session_id='OLD', run_uid='OLD:1', **{c: '' for c in rl.GATE_COLS})
    rl._write_all(os.path.join(d, rl.TUNING), rl.TUNING_COLS, rows + [old])
    assert rl.main(['--dataset', d, 'summary']) == 0
    assert f'{rl.NO_GATE_EVIDENCE}: 1 dat row' in capsys.readouterr().out


@pytest.mark.parametrize('platform, stage, outcome, needed', [
    ('HW', 'screen', 'khong_dat', True), ('HW', 'final', 'dat', True), ('HW', 'screen', 'hong_he_do', False),
    ('HIL', 'final_hil', 'khong_dat', True), ('HIL', 'hil_check', 'khong_dat', False),
    ('SIM', 'screen', 'khong_dat', False)])
def test_cli_end_requires_the_query_files(ds, tmp_path, capsys, platform, stage, outcome, needed):
    """R134 (b), R134C2-10, R134C3-7: the CLI 'end' of a judged car run (every stage) or HIL final_hil run
    needs both files; each row opens a valid run of its platform, stage and gain set, ends with its own
    outcome and a gate that agrees, and must close (rc 0, the row written) when the files are given or not
    needed."""
    assert rl.query_files_needed(dict(platform=platform, stage=stage), outcome) is needed
    d, clock = ds
    campaign, gs = {('HW', 'screen'): ('A-PID', 'A-PID/G1'), ('HW', 'final'): ('A-PID', 'A-PID/G0'), ('HIL', 'final_hil'): ('RQ2', 'RQ2-INIT-PID/G0'), ('HIL', 'hil_check'): ('B-PID', 'B-PID/GA1'), ('SIM', 'screen'): ('B-PID', '')}[(platform, stage)]
    rl.begin(d, 'S', campaign, platform, 'PID', 'O1-F05', '0.5', stage, gain_set=gs)
    rl.step(d, 'run')
    clock[0] += dt.timedelta(seconds=60)
    f = str(tmp_path / 'score.json')
    with open(f, 'w', encoding='utf-8') as fh:
        g = _gate(u_t=0.0015 if platform == 'HIL' else 0.0, e_trk=0.12 if outcome == 'khong_dat' else 0.03)
        json.dump(dict(gate_v3=rl.floor_gate(g) if platform == 'HW' else g), fh)          # B81-1, B81-3
    extra = ['--fail-cause', 'mat_khung'] if outcome == 'hong_he_do' else ['--gate-json', f, '--results-dir', 'r']
    qb, qa = _query_files(tmp_path)
    for q in ([], ['--query-before', qb]):
        rc = rl.main(['--dataset', d, 'end', '--outcome', outcome] + extra + q)
        err = capsys.readouterr().err
        assert ('needs --query-before and --query-after' in err) is needed, (q, err)
        if needed:
            assert rc == 2 and rl._load_open(d) is not None               # refused, the run still open
        else:
            break
    if needed:
        rc = rl.main(['--dataset', d, 'end', '--outcome', outcome] + extra + ['--query-before', qb, '--query-after', qa])
        assert 'needs --query-before' not in capsys.readouterr().err
    assert rc == 0 and rl._load_open(d) is None, capsys.readouterr()      # closed
    r = _rows(d)[-1]
    assert (r['platform'], r['stage'], r['run_outcome']) == (platform, stage, outcome)
    assert (r['boot_before'], r['boot_after']) == (('4753dc47',) * 2 if needed else ('', ''))
