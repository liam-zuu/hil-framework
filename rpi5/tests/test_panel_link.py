"""panel_link (RPi5 side of the panel UART, v6): STATUS / INFO mapping from RunSnapshot, INFO
rate, PROFILE list, commands reaching RunControl exactly once over a lossy wire, failures of
RunControl reported as RES_FAILED. Panel = the reference PanelClient. No hardware."""
import random

import pytest

import panel_proto as pp
from panel_link import INFO_MIN_GAP_S, INFO_PERIOD_S, FakeRun, PanelLink, RunSnapshot, info_of, status_of


class Recorder:
    """RunControl that records calls and returns what the test sets."""

    def __init__(self, names=('A', 'B')):
        self.names = list(names)
        self.calls = []
        self.snap = RunSnapshot()
        self.result = pp.RES_OK
        self.raise_on = None

    def _call(self, what, *a):
        self.calls.append((what,) + a)
        if self.raise_on == what:
            raise RuntimeError('boom')
        return self.result

    def profiles(self):
        return self.names

    def select(self, i):
        return self._call('select', i)

    def start(self, i):
        return self._call('start', i)

    def stop(self):
        return self._call('stop')

    def snapshot(self):
        return self.snap


class Wire:
    """PanelLink <-> PanelClient, 10 ms steps, frames dropped with probability p each way."""

    def __init__(self, run, p=0.0, seed=0):
        self.t = 0
        self.rnd = random.Random(seed)
        self.p = p
        self.down = bytearray()
        self.link = PanelLink(5, run, self._w, clock=lambda: self.t / 1000)
        self.panel = pp.PanelClient(iter(range(0x300, 0x400)).__next__)
        self.parser = pp.StreamParser(clock=lambda: self.t / 1000)
        self.up = []
        self.frames = []

    def _w(self, data):
        self.down += data
        return len(data)

    def step(self, n=1):
        for _ in range(n):
            self.t += 10
            for f in self.up:
                if self.rnd.random() >= self.p:
                    self.link.feed(f)
            self.up = []
            self.link.poll()
            data, self.down = bytes(self.down), bytearray()
            for x in self.parser.feed(data):
                self.frames.append((self.t, x))
                if self.rnd.random() < self.p:
                    continue
                if isinstance(x, pp.Status):
                    self.up += self.panel.on_status(x, self.t)
                elif isinstance(x, pp.Profile):
                    self.panel.on_profile(x)
                else:
                    self.panel.on_info(x)
            self.up += self.panel.tick(self.t)

    def press(self, cmd, arg=0):
        self.up += self.panel.press(cmd, arg, self.t)


def test_status_mapping_unknowns_and_link_bits():
    sess = pp.DaemonSession(1)
    st = status_of(RunSnapshot(), sess)
    assert st.profile == pp.PROFILE_NONE and st.data_age_ms == 0xFFFF and st.links == pp.L_RPI
    st = status_of(RunSnapshot(profile=2, data_age_ms=70000, h7_ok=True, esp32_ok=False, phase_s=1.27), sess)
    assert st.profile == 2 and st.links == pp.L_RPI | pp.L_H7 | pp.L_ESP32_KNOWN and st.phase_ds == 12
    assert pp.decode(pp.encode_status(st)).data_age_ms == 0xFFFF            # saturates
    assert status_of(RunSnapshot(esp32_ok=True), sess).links & pp.L_ESP32


def test_info_mapping_valid_flags():
    i = info_of(RunSnapshot(), 9)
    assert i.flags == 0 and i.ctrl_src == pp.CTRL_SRC_NONE
    i = info_of(RunSnapshot(loop_us=27, overruns=3, result=('x', '', ''), ctrl='LADRC',
                            ctrl_src=pp.CTRL_SRC_FW), 9)
    assert i.flags == pp.I_LOOP_VALID | pp.I_RESULT_VALID and i.overruns == 3 and i.ctrl == 'LADRC'


def test_handshake_profiles_and_info_at_once():
    w = Wire(Recorder())
    w.step(30)
    assert w.panel.sid and w.panel.profiles == {0: 'A', 1: 'B'} and w.panel.info is not None


def test_empty_profile_list_is_one_frame_with_count_zero():
    w = Wire(Recorder(names=()))
    w.step(30)
    profs = [x for _, x in w.frames if isinstance(x, pp.Profile)]
    assert profs and all(p.count == 0 for p in profs) and w.panel.profiles == {}


def test_info_rate_constant_and_changing():
    run = Recorder()
    run.snap = RunSnapshot(loop_us=27)
    w = Wire(run)
    w.step(20)                                                  # session up
    t0, n0 = w.t, sum(isinstance(x, pp.Info) for _, x in w.frames)
    w.step(300)                                                 # 3 s, INFO constant
    n_const = sum(isinstance(x, pp.Info) for _, x in w.frames) - n0
    assert 3 <= n_const <= 4
    k = [0]

    def changing():
        k[0] += 1
        return RunSnapshot(loop_us=20 + k[0] % 9)
    run.snapshot = changing
    n1 = sum(isinstance(x, pp.Info) for _, x in w.frames)
    w.step(300)
    n_chg = sum(isinstance(x, pp.Info) for _, x in w.frames) - n1
    assert n_chg <= 3 / INFO_MIN_GAP_S + 1 and n_chg >= 3 / INFO_PERIOD_S


@pytest.mark.parametrize('seed', range(10))
def test_each_command_reaches_run_control_once_over_a_lossy_wire(seed):
    run = Recorder()
    w = Wire(run, p=0.3, seed=seed)
    w.step(200)
    w.press(pp.CMD_SELECT, 1)
    w.step(300)
    w.press(pp.CMD_START, 1)
    w.step(300)
    assert [c for c in run.calls if c[0] == 'select'] in ([('select', 1)], [])
    assert len([c for c in run.calls if c[0] == 'start']) <= 1
    for what, *_ in run.calls:
        assert what in ('select', 'start')
    shown = {cmd: what for what, cmd, _ in w.panel.shown}
    for cmd, name in ((pp.CMD_SELECT, 'select'), (pp.CMD_START, 'start')):
        if shown.get(cmd) == 'done':
            assert (name, 1) in run.calls                       # 'done' only when it really ran


def test_run_control_raising_is_res_failed_and_link_keeps_going():
    run = Recorder()
    run.raise_on = 'start'
    w = Wire(run)
    w.step(30)
    w.press(pp.CMD_START, 0)
    w.step(30)
    assert w.panel.shown == [('done', pp.CMD_START, pp.RES_FAILED)]
    run.raise_on = None
    w.press(pp.CMD_STOP)
    w.step(30)
    assert ('stop',) in run.calls and w.panel.stop_state in ('acked', 'confirmed')


def test_fake_run_full_cycle_and_fault_ack():
    t = [0.0]
    run = FakeRun(run_s=1.0, fault_at=None, clock=lambda: t[0])
    assert run.start(0) == pp.RES_OK and run.start(0) == pp.RES_BUSY
    for state in (pp.ST_ARM, pp.ST_RUN, pp.ST_END):
        t[0] += 1.01
        assert run.snapshot().run_state == state
    assert run.snapshot().result[0].startswith('FAKE') and run.stop() == pp.RES_NOT_IDLE
    run = FakeRun(run_s=5.0, fault_at=0.5, clock=lambda: t[0])
    run.start(1)
    for _ in range(3):
        t[0] += 0.6
        run.snapshot()
    assert run.state == pp.ST_ERROR and run.stop() == pp.RES_OK and run.snapshot().run_state == pp.ST_IDLE
    assert run.select(9) == pp.RES_BAD_PROFILE
