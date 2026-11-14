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


def stdin_stop(proc, level):
    """Stand-in for _posix_stop on any OS: level 0 = soft STOP line, 1 and 2 = kill."""
    if level:
        proc.kill()
    else:
        proc.stdin.write('STOP\n')
        proc.stdin.flush()


def make(tmp_path, busy=False, stop_fn=stdin_stop, **kw):
    d = HilDaemon(str(tmp_path), tests=[('T-1', ['--test', 'A-L1']), ('T-2', ['--o1', 'O1-F05'])],
                  child_cmd=lambda args: [sys.executable, '-u', STUB, *args], stop_fn=stop_fn,
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
    hold = tmp_path / 'go'
    monkeypatch.setenv('FAKE_RUN_HIL', 'ok')
    monkeypatch.setenv('FAKE_RUN_S', '30')
    monkeypatch.setenv('FAKE_HOLD_FILE', str(hold))                     # PPI-10: no race
    d = make(tmp_path / 'runs')
    d.start(0)
    assert d.stop() == pp.RES_OK and d._stop_pending                   # child has printed nothing yet
    hold.write_text('')
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
    monkeypatch.setenv('FAKE_RUN_HIL', 'ok')
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
    """lgpio stand-in that models the pin, not only the calls (PPI-5): the direction and level
    set last stay after the chip is closed, as RP1 may keep outputs (persist_gpio_outputs).
    fail: name of the call that raises."""
    SET_OPEN_DRAIN = 8

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail
        self.pin = {'dir': 'in', 'level': None, 'flags': 0}

    def _do(self, name, *a):
        self.calls.append((name, *a))
        if self.fail == name:
            raise OSError(f'{name} failed')

    def gpiochip_open(self, chip):
        self._do('open', chip)
        return 7

    def gpio_claim_output(self, h, g, level, flags=0):
        self._do('out', g, level, flags)
        self.pin.update(dir='out', level=level, flags=flags)

    def gpio_free(self, h, g):
        self._do('free', g)                       # free alone does not change the pin

    def gpio_claim_input(self, h, g, flags=0):
        self._do('in', g)
        self.pin.update(dir='in', level=None)

    def gpiochip_close(self, h):
        self._do('close', h)

    def en_held_low(self):
        return self.pin['dir'] == 'out' and self.pin['level'] == 0


def test_reset_panel_pulls_en_low_then_leaves_an_input():
    from hil_daemon import reset_panel
    lg, slept = FakeLgpio(), []
    assert reset_panel(22, 4, 0.1, lg=lg, sleep=slept.append) == 'done'
    assert lg.calls[:2] == [('open', 4), ('out', 22, 0, FakeLgpio.SET_OPEN_DRAIN)] and lg.calls[-1] == ('close', 7)
    assert slept == [0.1] and not lg.en_held_low()
    assert not any(c[0] == 'out' and c[2] == 1 for c in lg.calls)   # never drives EN high


def test_reset_panel_ctrl_c_during_the_pulse_still_releases_en():
    from hil_daemon import reset_panel
    lg = FakeLgpio()

    def interrupted(_s):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        reset_panel(22, 4, 0.1, lg=lg, sleep=interrupted)
    assert not lg.en_held_low() and lg.calls[-1] == ('close', 7)


def test_reset_panel_failed_release_is_reported():
    from hil_daemon import reset_panel
    lg = FakeLgpio(fail='in')
    r = reset_panel(22, 4, 0.1, lg=lg, sleep=lambda s: None)
    assert r.startswith('failed: GPIO22 release') and lg.calls[-1] == ('close', 7)


def test_reset_panel_without_the_chip_is_skipped_not_fatal():
    from hil_daemon import reset_panel
    assert reset_panel(22, 4, 0.1, lg=FakeLgpio(fail='open'), sleep=lambda s: None).startswith('skipped')


# ── Codex round 1 (14/11), PPI-1, -3, -6, -8 ──

def test_stop_escalates_on_its_own_when_the_run_ignores_it(tmp_path, monkeypatch):
    """PPI-1: one STOP press; the child ignores the soft STOP; the daemon's timer sends level 1,
    then level 2, without further presses, and the run ends."""
    monkeypatch.setenv('FAKE_RUN_HIL', 'hang')
    levels = []

    def stop_fn(proc, level):
        levels.append(level)
        if level == 2:
            proc.kill()                           # level 1 'ignored' too: only SIGKILL works
    d = make(tmp_path, stop_fn=stop_fn)
    d.stop_term_s, d.stop_kill_s = 0.3, 0.3
    d.start(0)
    t_end = time.monotonic() + 10
    while not d._seen_phase and time.monotonic() < t_end:
        time.sleep(0.01)
    assert d.stop() == pp.RES_OK
    s = wait_for(d, (pp.ST_ERROR, pp.ST_END), timeout=10)
    assert levels == [0, 1, 2] and d.proc is None
    assert run_meta(d)['stop_signals'] == [0, 1, 2]
    assert s.run_state == pp.ST_ERROR and s.error_code == E_NO_SUMMARY


@pytest.mark.skipif(os.name != 'posix', reason='process groups and signals: run on the Pi')
def test_posix_stop_kills_the_whole_group(tmp_path):
    """PPI-1/PPI-10, PPI-R2-7: the real _posix_stop on a child that ignores SIGINT and SIGTERM
    and has a grandchild in its group; level 2 alone (no _reap_group here) must take the whole
    group, so a level 2 sent to the owner's pid only fails. Ready signal = a line on stdout."""
    import subprocess
    from hil_daemon import _posix_stop, _spawn_kw, _group_alive
    code = ('import signal, subprocess, sys, time;'
            'signal.signal(signal.SIGINT, signal.SIG_IGN); signal.signal(signal.SIGTERM, signal.SIG_IGN);'
            'g = subprocess.Popen([sys.executable, "-c", "import signal,sys,time;'
            'signal.signal(signal.SIGTERM, signal.SIG_IGN); print(1, flush=True); time.sleep(60)"],'
            ' stdout=subprocess.PIPE); g.stdout.readline(); print("ready", flush=True); time.sleep(60)')
    p = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True, **_spawn_kw())
    assert p.stdout.readline().strip() == 'ready'
    _posix_stop(p, 0)
    _posix_stop(p, 1)
    time.sleep(0.3)
    assert p.poll() is None and _group_alive(p.pid)      # both ignored
    _posix_stop(p, 2)
    p.wait(timeout=5)
    t_end = time.monotonic() + 2.0
    while _group_alive(p.pid) and time.monotonic() < t_end:
        time.sleep(0.02)
    assert not _group_alive(p.pid)                       # grandchild gone too, without a reap


@pytest.mark.skipif(os.name != 'posix', reason='process groups and signals: run on the Pi')
def test_owner_exit_reaps_a_child_holding_the_pipe(tmp_path, monkeypatch):
    """PPI-R2-1: the owner prints a phase and exits; its child (same group) ignores SIGTERM and
    keeps the stdout pipe. The daemon must still end the run (reap with SIGKILL) and not stay
    ACTIVE waiting for EOF."""
    import hil_daemon
    from hil_daemon import _group_alive
    monkeypatch.setattr(hil_daemon, 'REAP_S', 0.5)
    script = tmp_path / 'owner.py'
    script.write_text(
        'import signal, subprocess, sys, time\n'
        'print("02:00:00 INFO     hil.main: #phase,PREPARE,t_ns=1", flush=True)\n'
        'subprocess.Popen([sys.executable, "-c", "import signal,time; '
        'signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])\n'
        'time.sleep(0.3)\n', encoding='utf-8')
    d = HilDaemon(str(tmp_path / 'runs'), tests=[('T-1', ['x'])], busy_check=lambda pid: False,
                  child_cmd=lambda args: [sys.executable, '-u', str(script)])
    assert d.start(0) == pp.RES_OK
    pid = d.proc.pid
    s = wait_for(d, (pp.ST_ERROR, pp.ST_END), timeout=8)
    assert d.proc is None and s.error_code == E_NO_SUMMARY and not _group_alive(pid)


@pytest.mark.parametrize('argv,expected', [
    ([b'python3', b'run_hil.py', b'--test', b'A-L1'], True),
    ([b'/usr/bin/python3', b'-u', b'-B', b'/home/x/rpi5/run_hil.py', b'--finite-run'], True),
    ([b'python3.13', b'-W', b'ignore', b'run_hil.py'], True),
    ([b'python3', b'-X', b'dev', b'-u', b'run_hil.py'], True),
    ([b'vim', b'/home/x/rpi5/run_hil.py'], False),
    ([b'python3', b'-c', b'print(1)', b'run_hil.py'], False),
    ([b'python3', b'-m', b'pytest', b'tests/test_run_hil.py'], False),
    ([b'python3', b'hil_daemon.py'], False),
    ([b'python3', b'not_run_hil.py'], False),
    ([b''], False),
])
def test_run_hil_argv_detection(argv, expected):
    """PPI-3: options of the interpreter are skipped; an editor on the file is not a run."""
    from hil_daemon import is_run_hil_argv
    assert is_run_hil_argv(argv) is expected


def test_spawn_worker_of_this_folder_is_a_run(tmp_path):
    from hil_daemon import is_run_hil_worker
    argv = [b'/usr/bin/python3', b'-c', b'from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5)',
            b'--multiprocessing-fork']
    assert is_run_hil_worker(argv, str(tmp_path), folder=str(tmp_path))
    assert not is_run_hil_worker(argv, str(tmp_path / 'other'), folder=str(tmp_path))
    assert not is_run_hil_worker([b'python3', b'x.py'], str(tmp_path), folder=str(tmp_path))


def test_log_that_cannot_be_written_does_not_leave_the_panel_active(tmp_path, monkeypatch):
    """PPI-6: run_hil.log cannot be opened (full card): the run is still followed to its end."""
    import builtins
    import hil_daemon
    real_open = builtins.open

    def bad_open(path, *a, **k):
        if str(path).endswith('run_hil.log'):
            raise OSError(28, 'No space left on device')
        return real_open(path, *a, **k)
    monkeypatch.setattr(hil_daemon, 'open', bad_open, raising=False)
    monkeypatch.setenv('FAKE_RUN_HIL', 'ok')
    d = make(tmp_path)
    assert d.start(0) == pp.RES_OK
    s = wait_for(d, (pp.ST_END, pp.ST_ERROR))
    assert s.run_state == pp.ST_END and s.outcome == pp.OUT_COMPLETED and d.proc is None
    assert any('No space' in e for e in run_meta(d)['reader_errors'])
    assert d.start(1) == pp.RES_OK                # not stuck BUSY
    wait_for(d, (pp.ST_END, pp.ST_ERROR))


def test_meta_starts_with_swa_mode_null(tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_RUN_HIL', 'crash')
    d = make(tmp_path)
    d.start(0)
    wait_for(d, (pp.ST_ERROR,))
    assert 'swa_mode' in run_meta(d) and run_meta(d)['swa_mode'] is None


def test_shutdown_stops_the_run_and_closes_meta(tmp_path, monkeypatch):
    """PPI-2: the daemon's way out waits for the run and writes meta.json before returning."""
    monkeypatch.setenv('FAKE_RUN_HIL', 'ok')
    monkeypatch.setenv('FAKE_RUN_S', '30')
    d = make(tmp_path)
    d.start(0)
    t_end = time.monotonic() + 10
    while d.snapshot().run_state != pp.ST_RUN and time.monotonic() < t_end:
        time.sleep(0.01)
    assert d.shutdown(wait_s=10) is True and d.proc is None
    m = run_meta(d)
    assert m['outcome'] == 'user_stop' and m['exit_code'] == 0 and m['stop_signals'] == [0]


def test_uart_wait_retries_without_spending_anything():
    """PPI-9: the UART not there yet is waited for, one line said, then opened."""
    from hil_daemon import open_uart_when_ready
    tries, said, slept = [], [], []

    class SerialException(OSError):
        pass

    class Port:
        def __init__(self, **kw):
            self.port = None

        def open(self):
            tries.append(self.port)
            if len(tries) < 3:
                raise SerialException('no such device')

    class Serial:
        pass
    Serial.Serial, Serial.SerialException = Port, SerialException
    port = open_uart_when_ready(Serial, '/dev/ttyAMA0', 115200, retry_s=2, say=lambda *a, **k: said.append(a),
                                sleep=slept.append)
    assert len(tries) == 3 and slept == [2, 2] and len(said) == 2 and port.port == '/dev/ttyAMA0'


# ── Codex round 2 (14/11): PPI-R2-2 … R2-7 ──

@pytest.mark.parametrize('argv,expected', [
    ([b'python3', b'-uW', b'ignore', b'run_hil.py'], True),
    ([b'python3', b'-Wignore', b'run_hil.py'], True),
    ([b'python3', b'-BXdev', b'run_hil.py'], True),
    ([b'python3', b'-BX', b'dev', b'-u', b'/x/rpi5/run_hil.py'], True),
    ([b'python3', b'-uc', b'import x', b'run_hil.py'], False),
    ([b'python3', b'-Bm', b'pytest', b'run_hil.py'], False),
    ([b'python3', b'--check-hash-based-pycs', b'always', b'run_hil.py'], True),
    ([b'python3', b'-', b'run_hil.py'], False),
])
def test_combined_short_options(argv, expected):
    from hil_daemon import is_run_hil_argv
    assert is_run_hil_argv(argv) is expected


def test_ctrl_c_during_the_probe_still_kills_and_reaps_it(tmp_path, monkeypatch):
    """PPI-R2-2: KeyboardInterrupt while waiting for the probe: the probe child is killed, the
    probe flag cleared only once it is gone."""
    import subprocess
    monkeypatch.setenv('FAKE_PROBE', 'hang')
    levels = []

    def stop_fn(proc, level):
        levels.append(level)
        proc.kill()
    d = make(tmp_path, stop_fn=stop_fn)
    seen = {}
    real = subprocess.Popen.communicate

    def interrupted(self, *a, **k):
        if k.get('timeout') is not None:
            seen['p'] = self
            raise KeyboardInterrupt
        return real(self, *a, **k)
    monkeypatch.setattr(subprocess.Popen, 'communicate', interrupted)
    with pytest.raises(KeyboardInterrupt):
        d.probe_h7()
    assert levels == [2] and seen['p'].poll() is not None
    assert not d._probing and not d._wedged and d._probe_proc is None


def test_shutdown_kills_a_probe_in_progress_and_blocks_new_work(tmp_path, monkeypatch):
    """PPI-R2-2: shutdown during a background probe: the probe is killed, then nothing new."""
    import threading
    monkeypatch.setenv('FAKE_PROBE', 'hang')
    d = make(tmp_path)
    t = threading.Thread(target=d.probe_h7, daemon=True)
    t.start()
    t_end = time.monotonic() + 10
    while d._probe_proc is None and time.monotonic() < t_end:
        time.sleep(0.01)
    assert d.shutdown(wait_s=10) is True
    t.join(5)
    assert not t.is_alive() and not d._probing
    assert d.start(0) == pp.RES_BUSY and d.probe_h7() is False and d.maybe_reprobe(every_s=0) is None


def test_wedged_blocks_start_probe_reprobe_and_shutdown_says_so(tmp_path, monkeypatch):
    """PPI-R2-6: after E_GROUP nothing new runs, even when /proc sees nothing."""
    monkeypatch.setenv('FAKE_PROBE', 'ok')
    d = make(tmp_path)
    d._wedged = True
    assert d.start(0) == pp.RES_BUSY and d.probe_h7() is False and d.maybe_reprobe(every_s=0) is None
    assert d.shutdown(wait_s=1) is False


def test_reset_panel_ctrl_c_and_failed_release_is_still_said():
    """PPI-R2-5: Ctrl+C during the pulse and the release fails: the error is printed even
    though the KeyboardInterrupt goes on and nothing is returned."""
    from hil_daemon import reset_panel
    said = []

    def interrupted(_s):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        reset_panel(22, 4, 0.1, lg=FakeLgpio(fail='in'), sleep=interrupted,
                    say=lambda *a, **k: said.append(' '.join(map(str, a))))
    assert len(said) == 1 and 'release' in said[0] and 'EN may stay low' in said[0]


def test_epoch_is_taken_only_once_the_uart_opens(tmp_path, monkeypatch):
    """PPI-9, PPI-R2-7: _serve_panel opens the UART (retrying) before it takes an epoch."""
    import argparse
    import hil_daemon
    events = []

    class SerialException(OSError):
        pass

    class Port:
        def __init__(self, **kw):
            self.port = None

        def open(self):
            events.append('open')
            if events.count('open') < 3:
                raise SerialException('not yet')

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Serial:
        pass
    Serial.Serial, Serial.SerialException = Port, SerialException
    real_wait = hil_daemon.open_uart_when_ready
    monkeypatch.setattr(hil_daemon, 'open_uart_when_ready',
                        lambda s, p, b: real_wait(s, p, b, say=lambda *a, **k: None, sleep=lambda x: None))
    monkeypatch.setattr(hil_daemon.pp, 'next_daemon_epoch', lambda path, create=False: events.append('epoch') or 5)
    a = argparse.Namespace(panel_reset_gpio=-1, mock=True, no_probe=True, port='/dev/x', baud=115200,
                           epoch_file=str(tmp_path / 'e'), create_epoch=False, runs_dir=str(tmp_path))
    d = make(tmp_path / 'runs')
    hil_daemon._serve_panel(a, d, Serial, lambda epoch, daemon, w, baud: None, lambda port: None,
                            lambda port, link: events.append('serve'))
    assert events == ['open', 'open', 'open', 'epoch', 'serve']


# ── Codex round 3 (14/11): PPI-R3-1 … R3-3 ──

def test_group_that_cannot_be_reaped_finishes_the_run_without_eof(tmp_path, monkeypatch):
    """PPI-R3-2: the owner exits, a child keeps the pipe and (as patched here) the group cannot
    be reaped: the run is finished at once with E_GROUP, START and probes are refused, and the
    reader's later EOF finishes nothing again."""
    import hil_daemon
    monkeypatch.setattr(hil_daemon, '_reap_group', lambda pgid, wait_s=0: False)
    monkeypatch.setenv('FAKE_RUN_HIL', 'orphan')
    monkeypatch.setenv('FAKE_ORPHAN_S', '3')
    monkeypatch.setenv('FAKE_PROBE', 'ok')
    d = make(tmp_path)
    t0 = time.monotonic()
    assert d.start(0) == pp.RES_OK
    reader = d._reader
    s = wait_for(d, (pp.ST_ERROR,), timeout=5)
    assert time.monotonic() - t0 < 2.5 and reader.is_alive()           # before the pipe's EOF
    assert s.error_code == hil_daemon.E_GROUP and d._wedged and d.proc is None
    assert run_meta(d)['group_gone'] is False
    assert d.start(0) == pp.RES_BUSY and d.probe_h7() is False
    reader.join(10)
    assert not reader.is_alive() and d.snapshot().error_code == hil_daemon.E_GROUP


def test_probe_started_while_shutdown_begins_is_killed(tmp_path, monkeypatch):
    """PPI-R3-1: shutdown begins between the probe's Popen and the publication of its handle:
    the probe sees closing and kills its own child."""
    import subprocess
    import hil_daemon
    monkeypatch.setenv('FAKE_PROBE', 'hang')
    levels = []

    def stop_fn(proc, level):
        levels.append(level)
        proc.kill()
    d = make(tmp_path, stop_fn=stop_fn)
    real_popen = subprocess.Popen

    class Gate(real_popen):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            d._closing = True                     # shutdown arrives right here
    monkeypatch.setattr(hil_daemon.subprocess, 'Popen', Gate)
    t0 = time.monotonic()
    assert d.probe_h7() is False
    assert time.monotonic() - t0 < 10 and 2 in levels and not d._probing and not d._wedged


def test_reset_panel_failing_say_still_closes_the_chip_and_keeps_ctrl_c():
    """PPI-R3-3: Ctrl+C during the pulse, the release fails and printing fails too: the chip is
    still closed and the KeyboardInterrupt is what comes out."""
    from hil_daemon import reset_panel
    lg = FakeLgpio(fail='in')

    def interrupted(_s):
        raise KeyboardInterrupt

    def broken_say(*a, **k):
        raise BrokenPipeError('stdout gone')
    with pytest.raises(KeyboardInterrupt):
        reset_panel(22, 4, 0.1, lg=lg, sleep=interrupted, say=broken_say)
    assert lg.calls[-1] == ('close', 7)
