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
                            ctrl_src=pp.CTRL_SRC_FW, run_no=7, result_run=7), 9)
    assert i.flags == pp.I_LOOP_VALID | pp.I_RESULT_VALID and i.overruns == 3 and i.ctrl == 'LADRC'
    assert i.result_run == 7


def test_handshake_profiles_and_info_at_once():
    w = Wire(Recorder())
    w.step(30)
    assert w.panel.sid and w.panel.profiles == {0: 'A', 1: 'B'} and w.panel.info is not None


def test_empty_profile_list_is_one_frame_with_count_zero():
    w = Wire(Recorder(names=()))
    w.step(30)
    profs = [x for _, x in w.frames if isinstance(x, pp.Profile)]
    assert profs and all(p.count == 0 for p in profs) and w.panel.profiles == {}


@pytest.mark.parametrize('n', [0, 1, 64, 65, 255])
def test_whole_profile_list_arrives_codex_p14(n):
    w = Wire(Recorder(names=['T%d' % i for i in range(n)]))
    w.step(100)
    assert w.panel.profiles == {i: 'T%d' % i for i in range(n)}
    assert w.panel.profile_count == n


def test_new_session_does_not_send_info_of_the_old_one():
    run = Recorder()
    w = Wire(run)
    w.step(30)
    old = w.link.sess.sid
    w.panel = pp.PanelClient(iter(range(0x500, 0x600)).__next__)       # panel reboot: new HELLO
    w.step(30)
    infos = [x for t, x in w.frames if isinstance(x, pp.Info) and t > w.t - 300]
    assert w.panel.sid != old and w.panel.info is not None and infos[-1].sid == w.panel.sid


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


def test_no_result_while_a_run_is_active_codex_q1():
    i = info_of(RunSnapshot(run_state=pp.ST_RUN, result=('old', '', ''), run_no=5, result_run=4), 1)
    assert not any(i.result) and not i.flags & pp.I_RESULT_VALID and i.result_run == 0
    i = info_of(RunSnapshot(run_state=pp.ST_END, result=('new', '', ''), run_no=5, result_run=5), 1)
    assert i.result[0] == 'new' and i.flags & pp.I_RESULT_VALID and i.result_run == 5


def test_result_lines_without_their_run_are_not_sent_as_valid_r1_r5():
    i = info_of(RunSnapshot(run_state=pp.ST_END, result=('x', '', ''), run_no=5, result_run=0), 1)
    assert not any(i.result) and not i.flags & pp.I_RESULT_VALID and i.result_run == 0
    i = info_of(RunSnapshot(run_state=pp.ST_END, result=(), run_no=5, result_run=5), 1)
    assert not i.flags & pp.I_RESULT_VALID and i.result_run == 0          # no lines: no result run


def test_fake_run_counts_runs_and_names_the_result_r1_r5():
    t = [0.0]
    run = FakeRun(run_s=1.0, clock=lambda: t[0])
    assert (run.snapshot().run_no, run.snapshot().result_run) == (0, 0)
    run.start(0)
    for _ in range(3):
        t[0] += 1.01
        run.snapshot()
    s = run.snapshot()
    assert s.run_state == pp.ST_END and (s.run_no, s.result_run) == (1, 1)
    run.start(1)
    s = run.snapshot()
    assert (s.run_no, s.result_run) == (2, 0) and s.result == ()
    run.stop()
    t[0] += 1.0
    s = run.snapshot()
    assert s.run_state == pp.ST_END and (s.run_no, s.result_run) == (2, 0)   # user stop: no result


def test_every_status_carries_the_run_it_is_about_s1_s6():
    """Codex round 4 S1-S6 (Liam 04/11: "A: run_no vào STATUS")."""
    assert status_of(RunSnapshot(run_no=9), pp.DaemonSession(1)).run_no == 9
    t = {'w': None}
    w = Wire(FakeRun(('A', 'B'), run_s=0.5, clock=lambda: t['w'].t / 1000 if t['w'] else 0.0))
    t['w'] = w
    w.step(30)
    w.link.run.start(0)                                     # from the PC: run 1
    w.step(250)
    w.link.run.start(1)                                     # run 2
    w.step(250)
    st = [(x.run_state, x.run_no) for _, x in w.frames if isinstance(x, pp.Status)]
    assert st[0][1] == 0 and (pp.ST_END, 1) in st and st[-1] == (pp.ST_END, 2)
    first_2 = next(k for k, s in enumerate(st) if s[1] == 2)
    assert all(n == 2 for _, n in st[first_2:]) and all(n in (0, 1) for _, n in st[:first_2])
    assert st[first_2][0] in pp.ST_ACTIVE                   # run 2 is named from its first STATUS


def test_reference_panel_drops_old_names_when_the_list_count_changes_codex_q4():
    w = Wire(Recorder(names=('A', 'B', 'C')))
    w.step(30)
    sid = w.panel.sid
    w.panel.on_profile(pp.Profile(sid, 0, 1, 'X'))
    assert w.panel.profiles == {0: 'X'}
    w.panel.on_profile(pp.Profile(sid, 0, 3, 'Y'))
    assert w.panel.profiles == {0: 'Y'} and w.panel.profile_count == 3     # old 'C' not reused


def test_nonblocking_writer_maps_eagain_to_zero_codex_q6(monkeypatch):
    import panel_link

    class Port:
        def fileno(self):
            return 99
    calls = []

    def fake_write(fd, data):
        calls.append((fd, len(data)))
        if len(calls) == 1:
            raise BlockingIOError
        return 3
    monkeypatch.setattr(panel_link.os, 'name', 'posix')
    monkeypatch.setattr(panel_link.os, 'write', fake_write)
    w = panel_link.nonblocking_writer(Port())
    assert w(b'abcdef') == 0 and w(b'abcdef') == 3 and calls == [(99, 6), (99, 6)]


class WireFifo:
    """UART buffer that takes everything at once and a wire that sends it at 115200 8N1:
    returns the time each byte leaves the wire."""

    def __init__(self, clock):
        self.clock = clock
        self.free = 0.0
        self.data = bytearray()
        self.t_out = []

    def __call__(self, data):
        t = max(self.clock(), self.free)
        for _ in data:
            t += 10 / 115200
            self.t_out.append(t)
        self.free = t
        self.data += data
        return len(data)


@pytest.mark.parametrize('n', [11, 255])
def test_status_keeps_flowing_on_the_wire_during_a_long_profile_list_codex_q9(n):
    t = [0.0]
    fifo = WireFifo(lambda: t[0])
    link = PanelLink(5, Recorder(names=['T%d' % i for i in range(n)]), fifo, clock=lambda: t[0])
    panel = pp.PanelClient(iter(range(0x600, 0x700)).__next__)
    for f in panel.tick(0):
        link.feed(f)                                        # HELLO: session + whole list queued
    for _ in range(300):                                    # 3 s, poll every 10 ms
        t[0] += 0.01
        link.poll()
    ends, i, buf = [], 0, bytes(fifo.data)
    while i < len(buf):                                     # frames are written back to back
        n_f = {pp.START_STATUS: pp.LEN_STATUS, pp.START_PROFILE: pp.LEN_PROFILE,
               pp.START_INFO: pp.LEN_INFO}[buf[i]]
        if buf[i] == pp.START_STATUS:
            ends.append(fifo.t_out[i + n_f - 1])
        i += n_f
    gaps = [b - a for a, b in zip(ends, ends[1:])]
    assert len(ends) > 20 and max(gaps) < 0.25
