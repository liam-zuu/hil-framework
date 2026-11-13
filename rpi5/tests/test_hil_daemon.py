"""hil_daemon (stage E reduced, L79): the run manager behind the panel. run_hil.py is replaced by
tests/fake_run_hil.py (same output lines); STOP goes through its stdin. Checks the RunControl
contract of panel_link.py, the run folder (meta.json mode panel, data_valid false), the H7 light
(probe, ARM, MISO), errors on the FAULT screen, and one whole run driven from the reference
panel client over PanelLink. No hardware, no SPI."""
import json
import logging
import os
import sys
import time

import pytest

import panel_proto as pp
import run_hil
from hil_daemon import E_NO_SUMMARY, E_RUN, O1_PANEL, HilDaemon, panel_tests

STUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fake_run_hil.py')


def stdin_stop(proc, hard):
    if hard:
        proc.kill()
    else:
        proc.stdin.write('STOP\n')
        proc.stdin.flush()


def make(tmp_path, busy=False, **kw):
    d = HilDaemon(str(tmp_path), tests=[('T-1', ['--test', 'A-L1']), ('T-2', ['--o1', 'O1-F05'])],
                  child_cmd=lambda args: [sys.executable, '-u', STUB, *args], stop_fn=stdin_stop,
                  busy_check=lambda pid: busy, **kw)
    return d


def wait_for(d, states, timeout=10.0):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        s = d.snapshot()
        if s.run_state in states and (s.run_state in pp.ST_ACTIVE or d.proc is None):
            return s
        time.sleep(0.01)
    raise AssertionError(f'state {d.snapshot().run_state} never in {states}')


def run_meta(d):
    with open(os.path.join(d.run_dir, 'meta.json'), encoding='utf-8') as f:
        return json.load(f)


def test_panel_tests_are_group_a_then_o1_floor_tasks_never_k1_k2():
    from trajectory_pub import GROUP_A
    tests = panel_tests()
    names = [n for n, _ in tests]
    assert names == list(GROUP_A) + list(O1_PANEL)
    assert not any('K1' in n and n.startswith('O1') or n == 'O1-K2' for n in names)
    assert all(len(n) <= pp.NAME_LEN and n.isascii() for n in names)
    assert all(a[0] in ('--test', '--o1') and a[1] == n for n, a in tests)


def test_whole_run_completed_result_and_run_folder(tmp_path, monkeypatch):
    out = tmp_path / 'argv.json'
    monkeypatch.setenv('FAKE_RUN_HIL', 'ok')
    monkeypatch.setenv('FAKE_ARGV_OUT', str(out))
    d = make(tmp_path / 'runs', ctrl='LADRC')
    assert d.start(1) == pp.RES_OK
    assert d.snapshot().run_state == pp.ST_PREPARE and d.snapshot().run_no == 1
    assert d.start(0) == pp.RES_BUSY and d.select(0) == pp.RES_BUSY
    wait_for(d, (pp.ST_RUN,))
    assert d.snapshot().h7_ok and d.snapshot().loop_us == 42           # p99 41.5 of the stats line
    s = wait_for(d, (pp.ST_END,))
    assert s.outcome == pp.OUT_COMPLETED and s.result_run == 1 and s.run_no == 1
    assert s.result[0].startswith('completed #1') and s.loop_us == 43 and s.overruns == 1
    assert all(len(x) <= pp.RESULT_LEN for x in s.result)
    assert s.ctrl == 'LADRC' and s.ctrl_src == pp.CTRL_SRC_PROFILE and s.quality & pp.Q_DUT_UNCONF
    assert json.loads(out.read_text()) == ['--o1', 'O1-F05', '--finite-run']
    m = run_meta(d)
    assert m['mode'] == 'panel' and m['data_valid'] is False and m['exit_code'] == 0
    assert m['outcome'] == 'completed' and m['run_no'] == 1 and m['test'] == 'T-2'
    assert m['run_summary']['outcome'] == 'completed' and m['miso'] == [1200, 0, 0]
    assert '#run_end' in open(os.path.join(d.run_dir, 'run_hil.log'), encoding='utf-8').read()
    assert d.stop() == pp.RES_NOT_IDLE                                  # nothing to stop in END
    assert d.start(0) == pp.RES_OK and d.snapshot().run_no == 2         # END is not active
    wait_for(d, (pp.ST_END,))


def test_stop_in_run_is_user_stop_and_again_is_harmless(tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_RUN_HIL', 'ok')
    monkeypatch.setenv('FAKE_RUN_S', '30')
    d = make(tmp_path)
    d.start(0)
    wait_for(d, (pp.ST_RUN,))
    assert d.stop() == pp.RES_OK and d.stop() == pp.RES_OK
    s = wait_for(d, (pp.ST_END,))
    assert s.outcome == pp.OUT_USER_STOP and run_meta(d)['outcome'] == 'user_stop'


def test_stop_before_the_first_phase_line_is_sent_when_it_comes(tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_RUN_HIL', 'ok')
    monkeypatch.setenv('FAKE_RUN_S', '30')
    d = make(tmp_path)
    d.start(0)
    assert d.stop() == pp.RES_OK and d._stop_pending                   # child has printed nothing yet
    s = wait_for(d, (pp.ST_END,))
    assert s.outcome == pp.OUT_USER_STOP


def test_arm_timeout_no_h7_is_a_fault_h7_off_and_stop_acknowledges(tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_RUN_HIL', 'no_h7')
    d = make(tmp_path)
    d.h7_ok = True
    d.start(0)
    s = wait_for(d, (pp.ST_ERROR,))
    assert s.error_code == E_RUN and s.error_text.startswith('arm_timeout no_h7') and not s.h7_ok
    assert len(s.error_text) <= pp.ERROR_TEXT_LEN and s.outcome == pp.OUT_ERROR
    assert run_meta(d)['exit_code'] == 3
    assert d.stop() == pp.RES_OK and d.snapshot().run_state == pp.ST_IDLE and d.snapshot().error_code == 0


def test_crash_without_summary_is_its_own_fault(tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_RUN_HIL', 'crash')
    d = make(tmp_path)
    s = (d.start(0), wait_for(d, (pp.ST_ERROR,)))[1]
    assert s.error_code == E_NO_SUMMARY and 'rc 1' in s.error_text


def test_end_tail_cut_exit_3_is_a_fault_with_the_outcome_named(tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_RUN_HIL', 'tail_cut')
    d = make(tmp_path)
    d.start(0)
    s = wait_for(d, (pp.ST_ERROR,))
    assert s.error_code == E_RUN and s.error_text.startswith('completed exit 3')


def test_bad_miso_frames_mark_spi_quality_and_turn_h7_off(tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_RUN_HIL', 'miso_bad')
    d = make(tmp_path)
    d.start(0)
    s = wait_for(d, (pp.ST_END,))
    assert s.quality & pp.Q_SPI_STALE and not s.h7_ok


def test_mock_mode_ignores_the_fake_h7_miso_counters(tmp_path, monkeypatch):
    """Pi check 11/11: run_hil.py --mock's fake H7 has bad MISO CRCs; with --mock the H7 light stays on."""
    monkeypatch.setenv('FAKE_RUN_HIL', 'miso_bad')
    d = make(tmp_path, mock=True)
    d.h7_ok = True
    d.start(0)
    s = wait_for(d, (pp.ST_END,))
    assert s.h7_ok and not s.quality & pp.Q_SPI_STALE
    assert '--mock' in run_meta(d)['argv']


def test_refusals_without_side_effects(tmp_path):
    d = make(tmp_path, busy=True)
    assert d.start(0) == pp.RES_BUSY and d.snapshot().run_no == 0 and d.proc is None
    assert d.start(5) == pp.RES_BAD_PROFILE and d.select(-1) == pp.RES_BAD_PROFILE
    assert d.select(1) == pp.RES_OK and d.snapshot().profile == 1
    assert d.stop() == pp.RES_NOT_IDLE
    d.run_no = 65535
    d.busy_check = lambda pid: False
    assert d.start(0) == pp.RES_FAILED and d.run_no == 65535 and d.state == pp.ST_IDLE


@pytest.mark.parametrize('probe,ok', [('ok', True), ('bad', False), ('none', False)])
def test_h7_probe(tmp_path, monkeypatch, probe, ok):
    monkeypatch.setenv('FAKE_PROBE', probe)
    d = make(tmp_path)
    assert d.probe_h7() is ok and d.snapshot().h7_ok is ok


def test_probe_refused_while_another_run_hil_runs(tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_PROBE', 'ok')
    d = make(tmp_path, busy=True)
    assert d.probe_h7() is False and d.h7_ok is False


def test_whole_run_from_the_reference_panel(tmp_path, monkeypatch):
    """PanelLink + PanelClient (as in test_panel_link.Wire): START from the panel, the panel
    sees RUN, then END with the result of its own run number."""
    from test_panel_link import Wire
    monkeypatch.setenv('FAKE_RUN_HIL', 'ok')
    d = make(tmp_path)
    d.h7_ok = True
    w = Wire(d)
    w.step(30)
    assert w.panel.profiles == {0: 'T-1', 1: 'T-2'}
    w.press(pp.CMD_START, 0)
    seen, t_end = set(), time.monotonic() + 10
    st = None
    while time.monotonic() < t_end:
        w.step(2)
        st = [x for _, x in w.frames if isinstance(x, pp.Status)][-1]
        seen.add(st.run_state)
        if st.run_state == pp.ST_END and w.panel.info is not None and w.panel.info.result_run == st.run_no == 1:
            break
        time.sleep(0.01)
    assert pp.ST_RUN in seen and st.run_state == pp.ST_END
    assert w.panel.info.result[0].startswith('completed #1')
    assert ('START', pp.CMD_START, pp.RES_OK) in w.panel.shown or any(
        s[1] == pp.CMD_START and s[2] == pp.RES_OK for s in w.panel.shown)


def test_run_hil_phase_reporter_prints_each_change_once(caplog):
    class Box:
        seq = iter(['PREPARE', 'PREPARE', 'ARM', 'RUN', 'RUN', 'STOPPING', 'END'])

        def phase(self):
            return next(self.seq)

    rep = run_hil._PhaseReporter(Box())
    with caplog.at_level(logging.INFO, logger='hil.main'):
        for _ in range(7):
            rep()
    got = [r.getMessage().split(',')[1] for r in caplog.records if r.getMessage().startswith('#phase,')]
    assert got == ['PREPARE', 'ARM', 'RUN', 'STOPPING', 'END']


# ── CODE_PANEL 14/11: daemon started with the Pi (Liam Q3), plan 2026-11-14 items 2, 7 ──

def test_run_meta_records_the_swa_mode(tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_SCEN', 'ok')
    d = make(tmp_path)
    assert d.start(0) == pp.RES_OK
    wait_for(d, (pp.ST_END, pp.ST_ERROR))
    assert run_meta(d)['swa_mode'] == 'phase'


def test_o1_panel_is_the_figure_8_levels_of_o1_tasks():
    import o1_tasks
    assert O1_PANEL == tuple(o1_tasks.FIG8) and not set(O1_PANEL) & set(o1_tasks.FINAL_SET)


def test_start_is_busy_while_the_probe_runs(tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_PROBE', 'ok')
    d = make(tmp_path)
    seen = []
    real = d.child_cmd

    def spy(args):
        if '--fixed-omega' in args:
            seen.append(d.start(0))               # a panel START while the probe child is spawned
        return real(args)
    d.child_cmd = spy
    assert d.probe_h7() is True
    assert seen == [pp.RES_BUSY] and d.state == pp.ST_IDLE and d.run_no == 0
    assert d.start(0) == pp.RES_OK                # after the probe: allowed
    wait_for(d, (pp.ST_END, pp.ST_ERROR))


def test_reprobe_only_when_idle_light_off_and_due(tmp_path, monkeypatch):
    t = [100.0]
    d = make(tmp_path, clock=lambda: t[0])
    monkeypatch.setenv('FAKE_PROBE', 'none')
    assert d.maybe_reprobe(every_s=15) is False     # never probed yet: probe now
    t[0] += 10
    assert d.maybe_reprobe(every_s=15) is None      # not due
    t[0] += 6
    monkeypatch.setenv('FAKE_PROBE', 'ok')
    assert d.maybe_reprobe(every_s=15) is True and d.h7_ok
    t[0] += 100
    assert d.maybe_reprobe(every_s=15) is None      # light on: no probe


def test_reprobe_never_in_mock(tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_PROBE', 'none')
    d = make(tmp_path, mock=True)
    assert d.maybe_reprobe(every_s=0) is None


class FakeLgpio:
    def __init__(self, fail_open=False):
        self.calls, self.fail_open = [], fail_open

    def gpiochip_open(self, chip):
        if self.fail_open:
            raise OSError('no chip')
        self.calls.append(('open', chip))
        return 7

    def gpio_claim_output(self, h, g, level):
        self.calls.append(('out', g, level))

    def gpio_free(self, h, g):
        self.calls.append(('free', g))

    def gpio_claim_input(self, h, g):
        self.calls.append(('in', g))

    def gpiochip_close(self, h):
        self.calls.append(('close', h))


def test_reset_panel_pulls_en_low_then_leaves_an_input():
    from hil_daemon import reset_panel
    lg, slept = FakeLgpio(), []
    assert reset_panel(22, 4, 0.1, lg=lg, sleep=slept.append) == 'done'
    assert lg.calls == [('open', 4), ('out', 22, 0), ('free', 22), ('in', 22), ('close', 7)]
    assert slept == [0.1]
    assert not any(c[0] == 'out' and c[2] == 1 for c in lg.calls)   # never drives EN high


def test_reset_panel_without_the_chip_is_skipped_not_fatal():
    from hil_daemon import reset_panel
    assert reset_panel(22, 4, 0.1, lg=FakeLgpio(fail_open=True), sleep=lambda s: None).startswith('skipped')
