"""tools/hil_bench_session.py: the pure judging functions (no serial port, no SSH)."""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
import hil_bench_session as hb  # noqa: E402
from protocol import encode_state, encode_torques  # noqa: E402


def frames(n=4000, v=0.2, w=0.2, ref_from=500, duty=0.2, jump_at=None):
    """Circle: the plant follows the setpoint exactly; MOSI carries world-frame velocity like hil_node."""
    mosi, miso = np.zeros((n, 40), np.uint8), np.zeros((n, 40), np.uint8)
    th = 0.0
    for k in range(n):
        on = k >= ref_from
        vb, wz = (v, w) if on else (0.0, 0.0)
        th = (th + wz * 1e-3 + math.pi) % (2 * math.pi) - math.pi
        st = [vb / 0.0485] * 4 + [0.0, 0.0, th, vb * math.cos(th), vb * math.sin(th), wz]
        mosi[k] = np.frombuffer(encode_state(st, traj_ref=(v, 0.0, w) if on else (0.0, 0.0, 0.0), seq=k & 0xFF), np.uint8)
        d = duty if on else 0.0
        if jump_at is not None and k == jump_at:
            d = 0.9
        b = encode_torques([d] * 4)
        miso[k, :len(b)] = np.frombuffer(b, np.uint8)
    return np.arange(n, dtype=np.int64) * 1_000_000, mosi, miso


def test_tracking_body_frame():
    t, mosi, miso = frames()
    r = hb.tracking(t, mosi, miso)
    assert r['moving_frames'] > 2000
    assert r['body'][0] == pytest.approx(0.2, abs=2e-3) and r['body'][2] == pytest.approx(0.2, abs=2e-3)
    assert r['rms_err'][0] < 3e-3 and r['rms_err'][1] < 3e-3     # world-frame vx alone would be ~0.2 off
    assert r['duty_max'] == pytest.approx(0.2, abs=1e-3) and r['duty_sat'] == 0


def test_tracking_duty_jump_reported():
    t, mosi, miso = frames(jump_at=3000)
    assert hb.tracking(t, mosi, miso)['duty_step_max'] == pytest.approx(0.7, abs=1e-3)


def test_tracking_no_setpoint():
    t, mosi, miso = frames(ref_from=10**9)
    assert hb.tracking(t, mosi, miso)['moving_frames'] == 0


def _rows(n=600, spike_at=None, motor_on_from=0):
    out = []
    for i in range(n):
        rpm = 40.0 + (25.0 if spike_at is not None and i == spike_at else 0.0)
        out.append(dict(motor_on='1' if i >= motor_on_from else '0', sp_vx='0.2', sp_vy='0', sp_wz='0',
                        fl_rpm=str(rpm), fr_rpm='40', rl_rpm='40', rr_rpm='40'))
    return out


def test_rpm_spike_found_and_clean():
    r = hb.rpm_spikes(_rows(spike_at=300))
    assert r['wheels']['fl']['over10'] == 1 and r['wheels']['fl']['dev_max'] == pytest.approx(25.0)
    assert r['wheels']['fr']['over10'] == 0
    assert hb.rpm_spikes(_rows())['wheels']['fl']['dev_max'] == 0.0
    assert hb.rpm_spikes(_rows(spike_at=50))['wheels']['fl']['over10'] == 0      # first 1 s skipped
    assert hb.rpm_spikes(_rows(motor_on_from=10**6))['moving_rows'] == 0


def test_h7_window_rise():
    L = [(0, 'Frame: unread=0  new seen=0 taken=0 repl=0 drop=0  applied=10 skipped=3 age_max=1us'),
         (0, 'SPIfix: stuck=5 stuck_err=0x0 restart_fail=0 down_req=0 svc=1..2 cyc (480 cyc/us)'),
         (0, 'Loop: max=400.0 us prn_max=400.0 us over500us=0 over1ms=0 passes=10'),
         (1, 'Frame: unread=0  new seen=0 taken=0 repl=0 drop=0  applied=20 skipped=3 age_max=1us'),
         (1, 'SPIfix: stuck=6 stuck_err=0x0 restart_fail=0 down_req=0 svc=1..2 cyc (480 cyc/us)'),
         (1, 'Loop: max=1200.5 us prn_max=1200.5 us over500us=2 over1ms=1 passes=10')]
    h = hb.h7_window(L)
    assert h == dict(lines=2, skipped_rise=0, stuck_rise=1, loop_max_us=1200.5, loop_over1ms=1)
    assert hb.h7_window([])['stuck_rise'] is None


def test_esp_window_and_judge():
    L = [(0, 'I (1) MOTOR: Motor enabled'), (1, 'I (2) MOTOR: Motor disabled'), (2, 'I (3) MOTOR: Motor enabled'),
         (3, 'I (4) MOTOR: Motor disabled')]
    e = hb.esp_window(L)
    assert e['motor_disabled_inside'] == 1 and e['bad']['POWERON'] == 0
    ok, why = hb.judge([], dict(stuck_rise=0, skipped_rise=0), e)
    assert not ok and 'Motor disabled' in why[0]
    assert hb.judge([], dict(stuck_rise=0, skipped_rise=0), hb.esp_window(L[:2])) == (True, [])
    assert hb.judge(['2 frame MISO sai CRC'], None, None)[0] is False
    assert hb.judge([], dict(stuck_rise=2, skipped_rise=0), None)[0] is False


def test_run_cmd_and_bad_args():
    c = hb.run_cmd('~/run14', 'b2_AL1', '--test A-L1', True)
    assert 'agv_logger.py --out ~/run14/b2_AL1.csv' in c and 'kill -INT $L' in c
    # 2026-11-09_run01: `&` must apply to the logger alone, not to a `mkdir && cd && logger` list ($! = a
    # background shell that ignores SIGINT -> logger never stops, ssh hangs)
    assert '|| exit 2; python3 tools/agv_logger.py' in c and '&& python3 tools/agv_logger.py' not in c
    assert '--test A-L1 --finite-run --frame-log ~/run14/b2_AL1_frames.npz' in c
    assert 'agv_logger' not in hb.run_cmd('~/r', 'x', '--test A-L1', False)
    for bad in ('--test A-L1; rm -rf ~', '--test $(x)', '--test `x`', '--x | y', '--x && y'):
        assert hb.BAD_ARG.search(bad)
    assert not hb.BAD_ARG.search('--traj circle --speed 0.3 --radius 0.5 --duration 60')


def test_i8_rules_only_for_i8_runs():
    one = [(0, 'I (1) MOTOR: Motor enabled'), (1, 'I (2) MOTOR: Motor disabled')]
    e = hb.esp_window(one)
    assert e['motor_enabled'] == 1 and e['off_after_on']
    sw = dict(cmd_on_frames=100, held_mismatch=0, unfollowed=0, unobserved=0, settle_frames=20)
    assert hb.judge([], None, e, sw, i8=True) == (True, [])
    # on with no off after it (I8-9)
    ok, why = hb.judge([], None, hb.esp_window(one[:1]), sw, i8=True)
    assert not ok and 'Motor disabled sau' in why[0]
    # the DUT never ran although the Pi sent SWA on
    ok, why = hb.judge([], None, hb.esp_window(one[1:]), sw, i8=True)
    assert not ok and 'Motor enabled 0' in why[0]
    # SWA off the whole run: the motors must not come on
    ok, why = hb.judge([], None, e, dict(sw, cmd_on_frames=0), i8=True)
    assert not ok and 'cần 0' in why[0]
    # the H7 status does not follow (I8-7)
    ok, why = hb.judge([], None, e, dict(sw, held_mismatch=5, mismatch_first=[30, 31, 32]), i8=True)
    assert not ok and 'status SWA' in why[0]
    ok, why = hb.judge([], None, e, dict(sw, unfollowed=1), i8=True)
    assert not ok and 'không làm theo' in why[0]
    # a run from before I8 (no swa_mode in the manifest): none of these rules (I8-9)
    assert hb.judge([], None, hb.esp_window(one[1:]), dict(sw, unfollowed=3), i8=False) == (True, [])


def test_i8_boot_motor_on_counted():
    assert hb.boot_motor_on(['I (1) MAIN: Boot done', 'I (2) MOTOR: Motor enabled']) == 1
    assert hb.boot_motor_on(['I (1) MAIN: Boot done', 'I (2) MOTOR: Motor disabled']) == 0


def test_i8_r2_3_unfinished_end_tail_is_not_clean():
    e = hb.esp_window([(0, 'I (1) MOTOR: Motor enabled'), (1, 'I (2) MOTOR: Motor disabled')])
    sw = dict(cmd_on_frames=100, held_mismatch=0, unfollowed=0, unobserved=0, settle_frames=20)
    done = dict(swa_mode='phase', swa_tail_sent=30, swa_tail_complete=True)
    assert hb.judge([], None, e, sw, i8=True, man=done) == (True, [])
    for man in (dict(done, swa_tail_complete=False, swa_tail_sent=7),
                {k: v for k, v in done.items() if k != 'swa_tail_complete'}):   # key missing: not clean either
        ok, why = hb.judge([], None, e, sw, i8=True, man=man)
        assert not ok and 'đuôi END' in why[0]
    # 'on' / 'off' have no tail rule (no lifecycle tail), and neither has a run from before I8
    assert hb.judge([], None, e, sw, i8=True, man=dict(swa_mode='on', swa_tail_complete=None)) == (True, [])
    assert hb.judge([], None, e, sw, i8=False, man={}) == (True, [])


class _FakeSerial:
    lines = {}

    def __init__(self, port, path):
        self.port = port

    def start(self):
        pass

    def stop(self):
        pass

    def pulse_rts(self):
        pass

    def since(self, t0, t1=None):
        return [x for x in self.lines.get(self.port, []) if x[0] >= t0 and (t1 is None or x[0] <= t1)]


def test_i8_10_boot_counts_motor_on_from_the_pi_start(tmp_path, monkeypatch):
    # Codex round 2 I8-10: the old ESP32 run sees the Pi start 4 s before the reset; a `Motor enabled`
    # there must stop the session too, not only one after the reset
    clock = dict(t=100.0)
    monkeypatch.setattr(hb, 'time', SimpleNamespace(time=lambda: clock['t'],
                                                   sleep=lambda s: clock.__setitem__('t', clock['t'] + s)))
    monkeypatch.setattr(hb, 'ssh', lambda *a, **k: (0, '## HEAD\nabc'))
    monkeypatch.setattr(hb.subprocess, 'Popen', lambda *a, **k: SimpleNamespace(wait=lambda timeout=None: 0))
    monkeypatch.setattr(hb, 'SerialLog', _FakeSerial)
    boot_ok = [(105.0, 'BNO085 init OK'), (105.1, 'Controller: LADRC'), (105.2, 'Boot done')]
    a = SimpleNamespace(out=str(tmp_path / 's1'), host='h', pi_dir='~/r', h7_port='COM3', esp_port='COM9',
                        boot=True, run=[], no_logger=True, run_timeout=10, pi_kill=[], pi_stop=[],
                        expect_esp32=None, esp32_ip=None, logger_loose=False)
    _FakeSerial.lines = {'COM9': [(102.0, 'I (9) MOTOR: Motor enabled')] + boot_ok}   # before the reset (104.0)
    assert hb.bench(a) == (2, [])
    clock['t'] = 100.0
    a.out = str(tmp_path / 's2')                     # one session per folder (A)
    _FakeSerial.lines = {'COM9': [(99.0, 'I (9) MOTOR: Motor enabled')] + boot_ok}    # before the Pi start
    assert hb.bench(a) == (0, [])


# ── A, B (Liam 09/11, "Làm cả A và B") ──

def _bench_env(monkeypatch, tmp_path, ssh_reply=None, out='s', **kw):
    """bench() with fake ssh/scp/serial/clock/Timer; returns (args, ssh calls, timers)."""
    calls, timers = [], []
    clock = dict(t=100.0)
    monkeypatch.setattr(hb, 'time', SimpleNamespace(time=lambda: clock['t'],
                                                   sleep=lambda s: clock.__setitem__('t', clock['t'] + s)))

    def fake_ssh(host, cmd, timeout=None):
        calls.append(cmd)
        return (ssh_reply or (lambda c: (0, '## HEAD\nabc')))(cmd)
    monkeypatch.setattr(hb, 'ssh', fake_ssh)
    monkeypatch.setattr(hb, 'scp', lambda *a: 0)
    monkeypatch.setattr(hb, 'SerialLog', _FakeSerial)

    class FakeTimer:
        def __init__(self, sec, fn):
            self.sec, self.fn, self.daemon = sec, fn, False
            timers.append(self)

        def start(self):
            self.fn()                                # fire at once: the stop goes out before the run returns

        def cancel(self):
            pass

        def join(self, t=None):
            pass
    monkeypatch.setattr(hb.threading, 'Timer', FakeTimer)
    _FakeSerial.lines = {}
    a = SimpleNamespace(out=str(tmp_path / out), host='h', pi_dir='~/r', h7_port='COM3', esp_port='COM9',
                        boot=False, run=[], no_logger=True, run_timeout=10, pi_kill=[], pi_stop=[],
                        expect_esp32=None, esp32_ip=None, logger_loose=False)
    for k, v in kw.items():
        setattr(a, k, v)
    return a, calls, timers


# ── #fw: ESP32 firmware id (Liam 09/11, CODE) ──

FW_BOOT = 'I (1301) MAIN: #fw sha=1a2b3c4d5e6f dirty=0 env=ladrc'
FW_UDP = '2026-11-09T12:00:00.000001 #fw,sha=1a2b3c4d5e6f,dirty=0,env=ladrc'


def test_fw_lines_both_forms_and_distinct():
    fw = dict(sha='1a2b3c4d5e6f', dirty=0, env='ladrc')
    assert hb.fw_lines([FW_BOOT, 'x', FW_UDP, FW_UDP]) == [fw]
    assert hb.fw_lines(['I (1) S06: #fw sha=unknown dirty=1 env=s06_load']) == [dict(sha='unknown', dirty=1, env='s06_load')]
    assert hb.fw_lines(['#run_end,seq=3', 'Controller: ADRC']) == []


@pytest.mark.parametrize('fws, expect, ok', [
    ([], None, True),                                                      # no --expect-esp32: no rule
    ([dict(sha='1a2b3c4d5e6f', dirty=0, env='ladrc')], '1a2b3c4', True),   # short prefix
    ([dict(sha='1a2b3c4d5e6f', dirty=0, env='ladrc')], '1a2b3c4d5e6f7788', True),   # longer full SHA
    ([dict(sha='1a2b3c4d5e6f', dirty=0, env='ladrc')], 'ffd4a07', False),
    ([dict(sha='1a2b3c4d5e6f', dirty=1, env='ladrc')], '1a2b3c4', False),  # dirty build
    ([], '1a2b3c4', False),                                                # old firmware: no line
    ([dict(sha='unknown', dirty=1, env='ladrc')], 'unknown', False),
    ([dict(sha='1a2b3c4d5e6f', dirty=0, env='ladrc'), dict(sha='aaaaaaaaaaaa', dirty=0, env='ladrc')],
     '1a2b3c4', False),                                                    # two ids in one log
])
def test_fw_problem(fws, expect, ok):
    assert (hb.fw_problem(fws, expect) is None) == ok


def test_fw_boot_mismatch_stops_and_is_recorded(tmp_path, monkeypatch):
    boot = [(105.0, 'BNO085 init OK'), (105.1, 'Controller: ADRC'), (105.15, FW_BOOT), (105.2, 'Boot done')]
    for expect, code in (('ffd4a07', 2), ('1a2b3c4', 0)):
        a, calls, _ = _bench_env(monkeypatch, tmp_path, out=f'b_{expect}', boot=True, expect_esp32=expect)
        monkeypatch.setattr(hb.subprocess, 'Popen', lambda *x, **k: SimpleNamespace(wait=lambda timeout=None: 0))
        _FakeSerial.lines = {'COM9': boot}
        assert hb.bench(a)[0] == code
        import json as _j
        s = _j.load(open(os.path.join(a.out, 'session.json'), encoding='utf-8'))
        assert s['boot']['esp32_fw'] == [dict(sha='1a2b3c4d5e6f', dirty=0, env='ladrc')]


def test_fw_from_the_udp_events_of_a_run(tmp_path, monkeypatch):
    def fake_scp(host, remote, local_dir):
        if remote.endswith('.csv*'):                 # agv_logger files of the run
            with open(os.path.join(local_dir, 'x.csv.events.txt'), 'w', encoding='utf-8') as f:
                f.write(FW_UDP + '\n' + FW_UDP + '\n2026-11-09T12:00:09 #run_end,seq=9\n')
        return 0
    reply = lambda c: (0, '## HEAD\nabc') if 'git' in c else (1, '')  # noqa: E731
    a, calls, _ = _bench_env(monkeypatch, tmp_path, reply, run=['x=--test A-L1'], no_logger=False,
                             expect_esp32='ffd4a07')
    monkeypatch.setattr(hb, 'scp', fake_scp)
    hb.bench(a)
    import json as _j
    s = _j.load(open(os.path.join(a.out, 'session.json'), encoding='utf-8'))
    assert s['runs'][0]['esp32_fw'] == [dict(sha='1a2b3c4d5e6f', dirty=0, env='ladrc')]


@pytest.mark.parametrize('old', ['session.json', 'bench_summary.json', 'h7_COM3.txt', 'esp32_COM5.txt',
                                 'frames/b2_AL1_frames.npz', 'pi/b2_AL1_pi.txt'])
def test_a_folder_with_results_is_refused_before_anything(tmp_path, monkeypatch, old):
    a, calls, _ = _bench_env(monkeypatch, tmp_path, run=['x=--test A-L1'])
    p = os.path.join(a.out, old)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        f.write('kept')
    assert hb.bench(a) == (2, [])
    assert calls == []                               # no SSH, no port opened
    assert open(p).read() == 'kept'


def test_a_flash_folder_alone_is_allowed(tmp_path, monkeypatch):
    a, calls, _ = _bench_env(monkeypatch, tmp_path, run=['x=--test A-L1'])
    os.makedirs(os.path.join(a.out, 'flash'))
    open(os.path.join(a.out, 'flash', 'flash.json'), 'w').close()
    assert hb.existing_results(a.out) == []
    hb.bench(a)
    assert any('run_hil.py --test A-L1' in c for c in calls)
    assert os.path.exists(os.path.join(a.out, 'session.json'))


def test_a_serial_log_never_overwritten(tmp_path, monkeypatch):
    # second guard: even past existing_results(), the log file opens create-only
    class FakePort:
        def __init__(self):
            self.closed = False

        def open(self):
            pass

        def close(self):
            self.closed = True
    port = FakePort()
    monkeypatch.setitem(sys.modules, 'serial', SimpleNamespace(Serial=lambda: port))
    p = tmp_path / 'h7_COM3.txt'
    p.write_bytes(b'old')
    with pytest.raises(FileExistsError):
        hb.SerialLog('COM3', str(p))
    assert p.read_bytes() == b'old' and port.closed


def test_a_plan_refuses_repeats_and_bad_stops():
    assert hb.plan_runs(['a=--x', 'b=--y'], ['b=11']) == [('a', '--x', None), ('b', '--y', ('kill', 11.0))]
    assert hb.plan_runs(['a=--x'], [], ['a=2.5']) == [('a', '--x', ('stop', 2.5))]
    for runs, kills, stops in ((['a=--x', 'a=--y'], [], []),        # same name twice: files overwritten
                               (['a=--x'], ['b=3'], []), (['a=--x'], ['a=0'], []), (['a=--x'], ['a=x'], []),
                               (['a=--x'], ['a=3'], ['a=4']), (['a;b=--x'], [], []), (['a=--x; rm'], [], [])):
        with pytest.raises(ValueError):
            hb.plan_runs(runs, kills, stops)


def test_b_run_cmd_writes_the_pid_of_run_hil_itself():
    import re
    for logger in (True, False):
        c = hb.run_cmd('~/r', 'b5', '--test A-L1', logger)
        # run_hil.py is the single command put in the background (after `;`), so $! is its own PID
        assert re.search(r'\|\| exit 2; python3 run_hil\.py --test A-L1 --finite-run --frame-log ~/r/b5_frames\.npz '
                         r'& H=\$!; echo \$H > ~/r/b5\.pid; wait \$H; RC=\$\?', c), c
        assert 'rm -f ~/r/b5.pid' in c and c.endswith('exit $RC') and 'pkill' not in c


def test_b_stop_cmd_uses_the_pid_file_and_checks_it():
    k = hb.pi_stop_cmd('~/r', 'b5', 'kill')
    s = hb.pi_stop_cmd('~/r', 'b5', 'stop')
    for c in (k, s):
        assert 'cat ~/r/b5.pid' in c and '/proc/$P/cmdline' in c and "grep -q 'run_hil.py'" in c
        assert 'pkill' not in c and '|| exit 3' in c and '|| exit 4' in c
    assert 'kill -KILL $T' in k and 'T="$(d $P) $P"' in k           # descendants too
    assert 'kill -TERM $T' in s and 'T="$P"' in s                    # the main process only
    with pytest.raises(KeyError):
        hb.pi_stop_cmd('~/r', 'b5', 'int')


def test_b_leftovers():
    txt = ('812 python3 /usr/lib/x/other.py\n'
           '4750 python3 tools/agv_logger.py --out /home/u/run18/b3_kill.csv\n'
           '4751 python3 run_hil.py --test A-L1 --finite-run\n')
    assert hb.leftovers(txt) == ['4750 python3 tools/agv_logger.py --out /home/u/run18/b3_kill.csv',
                                 '4751 python3 run_hil.py --test A-L1 --finite-run']
    assert hb.leftovers('') == [] and 'python3' in hb.LEFT_CMD and 'run_hil' not in hb.LEFT_CMD


def test_b_leftover_before_the_session_stops_it(tmp_path, monkeypatch):
    reply = lambda c: (0, '4751 python3 run_hil.py --test A-L1') if c == hb.LEFT_CMD else (0, '## HEAD\nabc')  # noqa: E731
    a, calls, _ = _bench_env(monkeypatch, tmp_path, reply, run=['x=--test A-L1'])
    assert hb.bench(a) == (2, [])
    assert not any('run_hil.py --test A-L1 --finite-run' in c for c in calls)


def test_b_leftover_after_a_run_stops_the_next_runs(tmp_path, monkeypatch):
    n = dict(left=0)

    def reply(c):
        if c == hb.LEFT_CMD:
            n['left'] += 1
            return (0, '') if n['left'] == 1 else (0, '4750 python3 tools/agv_logger.py --out x')
        return (0, '## HEAD\nabc')
    a, calls, _ = _bench_env(monkeypatch, tmp_path, reply, run=['x=--test A-L1', 'y=--test A-L1'])
    code, _ = hb.bench(a)
    assert code == 2
    assert sum('python3 run_hil.py' in c for c in calls) == 1                   # y never ran
    import json as _j
    s = _j.load(open(os.path.join(a.out, 'session.json'), encoding='utf-8'))
    assert s['runs'][0]['pi_leftovers'] == ['4750 python3 tools/agv_logger.py --out x']


def test_b_pi_kill_sends_the_stop_and_records_it(tmp_path, monkeypatch):
    def reply(c):
        if c.startswith('P=$(cat'):
            return (0, 'pid=4905 targets=4907 4908 4905\n1794197831.1')
        if 'run_hil.py' in c and 'finite-run' in c:
            return (137, 'killed')
        return (0, '## HEAD\nabc') if 'git' in c else (1, '')
    a, calls, timers = _bench_env(monkeypatch, tmp_path, reply, run=['k=--test A-L1'], pi_kill=['k=11'])
    code, _ = hb.bench(a)
    assert [t.sec for t in timers] == [11.0]
    stop = [c for c in calls if c.startswith('P=$(cat')]
    assert stop == [hb.pi_stop_cmd('~/r', 'k', 'kill')]
    import json as _j
    s = _j.load(open(os.path.join(a.out, 'session.json'), encoding='utf-8'))
    r = s['runs'][0]
    assert r['ssh_exit'] == 137 and r['pi_stop']['kind'] == 'kill' and r['pi_stop']['after_s'] == 11.0
    assert r['pi_stop']['ssh_exit'] == 0 and 'pid=4905' in r['pi_stop']['out'] and 't_pc' in r['pi_stop']
    assert code == 1                                 # no frame log: expected with --pi-kill, still not clean


# ── A5: strict logger (Liam 09/11, CODE, STATUS_CODE L61 b) ──

def test_a5_run_cmd_strict_waits_for_ready():
    c = hb.run_cmd('~/r', 'b1', '--test A-L1', True, '192.168.1.181')
    assert ('python3 tools/agv_logger.py --out ~/r/b1.csv --strict --src-ip 192.168.1.181 '
            '--ready-file ~/r/b1.ready > ~/r/b1_logger.txt 2>&1 & L=$!;') in c
    # run_hil.py starts only after the ready check; no ready file = no run, exit 2
    i_ready, i_hil = c.index('[ -e ~/r/b1.ready ] ||'), c.index('python3 run_hil.py')
    assert i_ready < i_hil and 'echo logger_not_ready' in c[i_ready:i_hil] and 'exit 2; }' in c[i_ready:i_hil]
    assert 'kill -0 $L' in c[:i_hil]                 # a logger that died stops the wait at once
    loose = hb.run_cmd('~/r', 'b1', '--test A-L1', True)
    assert '--strict' not in loose and '.ready' not in loose
    assert loose == hb.run_cmd('~/r', 'b1', '--test A-L1', True, None)


def test_a5_logger_src():
    lines = ['I (2069) WIFI: Got IP: 192.168.1.181 (after 0 failed attempts)', 'x',
             'I (9) WIFI: Got IP: 192.168.1.190 (after 1 failed attempts)']
    assert hb.esp32_ip(lines) == '192.168.1.190'     # the last one: a reconnect may change it
    assert hb.esp32_ip(['I (1) MAIN: Controller: ADRC']) is None
    assert hb.logger_src('10.0.0.5', lines) == ('10.0.0.5', 'arg')
    assert hb.logger_src(None, lines) == ('192.168.1.190', 'serial')
    assert hb.logger_src(None, []) == ('first', 'first_packet')


def _lsum(**kw):
    s = dict(mode='strict', status='closed', src_ip='192.168.1.181', src_lock='fixed', data_lines=10, rejects={},
             segments=[dict(seq_holes=0), dict(seq_holes=2)])
    s.update(kw)
    return s


@pytest.mark.parametrize('summary, txt, strict, n_why', [
    (_lsum(), 'logger_exit=0', True, 0),
    (_lsum(rejects={'source': 5}), 'logger_exit=0', True, 0),          # other sender: refused, data unharmed
    (_lsum(rejects={'n_cols_66': 3}), 'logger_exit=0', True, 1),       # a data line missing from the CSV
    (_lsum(status='open'), 'logger_exit=3', True, 2),
    (_lsum(mode='loose'), 'logger_exit=0', True, 1),
    (_lsum(mode='loose'), 'logger_exit=0', False, 0),                  # --logger-loose: loose is expected
    (None, 'logger_not_ready\nlogger_exit=0', True, 2),
    (None, '', True, 1),
])
def test_a5_logger_problems(summary, txt, strict, n_why):
    assert len(hb.logger_problems(summary, txt, strict)) == n_why


def test_a5_logger_info_and_summary_line():
    info = hb.logger_info(_lsum())
    assert info['seq_holes'] == 2 and info['segments'] == 2 and info['mode'] == 'strict'
    assert hb.logger_info(None) is None


def test_a5_bench_uses_serial_ip_and_judges_the_logger(tmp_path, monkeypatch):
    import json as _j

    def fake_scp(host, remote, local_dir):
        if remote.endswith('.csv*'):
            with open(os.path.join(local_dir, 'x.csv.summary.json'), 'w', encoding='utf-8') as f:
                _j.dump(_lsum(rejects={'type': 1}), f)
        return 0
    reply = lambda c: (0, '## HEAD\nabc') if 'git' in c else (0, 'logger_exit=0')  # noqa: E731
    a, calls, _ = _bench_env(monkeypatch, tmp_path, reply, run=['x=--test A-L1'], no_logger=False)
    monkeypatch.setattr(hb, 'scp', fake_scp)
    _FakeSerial.lines = {'COM9': [(50.0, 'I (2069) WIFI: Got IP: 192.168.1.181 (after 0 failed attempts)')]}
    hb.bench(a)
    run = [c for c in calls if 'run_hil.py' in c and 'agv_logger' in c]
    assert run and '--strict --src-ip 192.168.1.181 --ready-file ~/r/x.ready' in run[0]
    s = _j.load(open(os.path.join(a.out, 'session.json'), encoding='utf-8'))
    assert s['runs'][0]['logger_cmd'] == dict(strict=True, src_ip='192.168.1.181', src_from='serial')
    a2, calls2, _ = _bench_env(monkeypatch, tmp_path, reply, out='s2', run=['x=--test A-L1'], no_logger=False,
                               logger_loose=True)
    monkeypatch.setattr(hb, 'scp', fake_scp)
    hb.bench(a2)
    assert all('--strict' not in c for c in calls2)
