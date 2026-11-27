"""Panel demo profiles, schedule clock, DUT reset and firmware checks on the host."""
import json
import sys
import threading
from fractions import Fraction
from pathlib import Path

import pytest

import fault_schedule
import hil_daemon as hd
import o1_tasks
import panel_proto as pp
import run_hil
from hil_node import HILConfig
from panel_link import extra_of
from test_hil_daemon import FakeLgpio, STUB, run_meta, stdin_stop, wait_for

SHA = '18820107799e'
OTHER = '18820107799f'  # same first eight digits as SHA
IP = '192.0.2.42'


def daemon(tmp_path, **kwargs):
    d = hd.HilDaemon(str(tmp_path / 'runs'), child_cmd=lambda args: [sys.executable, '-u', STUB, *args],
                     stop_fn=stdin_stop, busy_check=lambda pid: False, **kwargs)
    # Firmware-query tests exercise the daemon without binding the host UDP port.
    d._logger_cmd = lambda folder: [sys.executable, '-c', 'pass']
    return d


@pytest.mark.parametrize('name,fault,target,end_ms,params', [
    ('DEMO-ENC', 'enc_loss', 'FL', 7000, {}),
    ('DEMO-IMU', 'imu_loss', 'imu', 7000, {}),
    ('DEMO-GYRO', 'gyro_scale', 'gz', 9000, {'s': Fraction('0.3')}),
    ('DEMO-JAM', 'wheel_jam', 'FL', 7000, {'b': Fraction('0.04'), 'tc': Fraction('0.20')}),
])
def test_schedules_load_on_default_o1_plant(monkeypatch, name, fault, target, end_ms, params):
    args = dict(hd.panel_tests())[name]
    monkeypatch.setattr(sys, 'argv', ['run_hil.py', *args, '--finite-run', '--frame-log', 'frames.npz',
                                     '--step-log', 'steps.csv'])
    parsed = run_hil._parse_args()
    assert parsed.o1 == 'O1-F05' and parsed.finite_run and not parsed.mock
    cfg = HILConfig(imu_static=parsed.imu_static, imu_profile=parsed.imu_profile)
    cfg.lifecycle, _ = run_hil._finite_run_config(parsed, o1_tasks.make_spec(parsed.o1))
    # Calls fault_schedule.load with the lifecycle's RUN length, IMU source and ground above,
    # and fault_plant.check_params with the nominal wheels, just as in run_hil.main.
    row, = run_hil.load_faults(parsed, n_omega_modes=0, hil_cfg=cfg, load_schedule=False).rows
    assert (row.fault, row.target, row.start_ms, row.end_ms, dict(row.params)) == (
        fault, target, 4000, end_ms, params)


def test_jam_slows_fl_at_the_same_duty():
    import numpy as np
    from fault_plant import PlantVariant
    from params import MecanumParams
    from plant_step import plant_step

    nominal = MecanumParams()
    args = dict(hd.panel_tests())['DEMO-JAM']
    duration, _ = o1_tasks.task('O1-F05')
    rows = fault_schedule.load(args[args.index('--faults') + 1], round(duration * 1000), 'plant', 'traction').rows
    initial = np.zeros(10)
    initial[3], initial[6:] = 0.5, 0.5 / nominal.r
    jammed, _ = PlantVariant(rows, nominal).at((0,), rows[0].start_ms, initial)
    speeds = []
    for params in (nominal, jammed):
        x = initial
        for _ in range(200):
            x = plant_step(x, [-0.35] * 4, params)
        speeds.append(x[6])
    assert 0 <= speeds[1] < 0.75 * speeds[0]


def test_fault_argv_and_meta_name_the_same_run_files(tmp_path, monkeypatch):
    out = tmp_path / 'argv.json'
    monkeypatch.setenv('FAKE_ARGV_OUT', str(out))
    d = daemon(tmp_path)
    cmd = d.child_cmd

    def child_cmd(args):
        assert Path(d.run_dir).is_dir()
        return cmd(args)

    d.child_cmd = child_cmd
    assert d.start(d.profiles().index('DEMO-ENC')) == pp.RES_OK
    initial = run_meta(d)
    wait_for(d, (pp.ST_END,))
    argv = json.loads(out.read_text())
    expected = dict(hd.panel_tests())['DEMO-ENC']
    assert argv == [*expected, '--finite-run', '--pose-log', str(Path(d.run_dir) / 'pose.csv'),
                    '--frame-log', str(Path(d.run_dir) / 'frames.npz'),
                    '--step-log', str(Path(d.run_dir) / 'steps.csv')]
    assert initial['argv'] == run_meta(d)['argv'] == [sys.executable, '-u', STUB, *argv]
    assert not initial['data_valid'] and initial['mode'] == 'panel'


def test_flag_uses_run_timestamp_earliest_end_and_is_cut_at_phase_change(tmp_path, monkeypatch):
    both = tmp_path / 'both.csv'
    both.write_text('# hil-fault-schedule v1\nid,fault,target,start_ms,end_ms,params\n'
                    'g,gyro_scale,gz,4000,9000,s=0.3\ne,enc_loss,FL,4000,7000,\n')
    d = daemon(tmp_path, tests=[('both', ['--o1', 'O1-F05', '--faults', str(both)])])
    monkeypatch.setenv('FAKE_RUN_S', '30')
    assert d.start(0) == pp.RES_OK
    try:
        wait_for(d, (pp.ST_RUN,))
        origin_ns = 987654321012345
        d._parse(f'#phase,RUN,t_ns={origin_ns}', {})
        now = [origin_ns]
        monkeypatch.setattr(hd.time, 'monotonic_ns', lambda: now[0])
        for delta, text in [(3_999_999_999, ''), (4_000_000_000, 'LOI ENC FL 3.0s'),
                            (4_900_000_000, 'LOI ENC FL 2.1s'), (6_999_999_999, 'LOI ENC FL 0.1s'),
                            (7_000_000_000, 'LOI GYRO GZ 2.0s'), (9_000_000_000, '')]:
            now[0] = origin_ns + delta
            extra = extra_of(d.snapshot(), 1)
            assert (extra.flags, extra.fault_text) == (pp.X_FAULT_ACTIVE if text else 0, text)
        now[0] = origin_ns + 5_000_000_000
        d._parse(f'#phase,STOPPING,t_ns={now[0]}', {})
        assert extra_of(d.snapshot(), 1).flags == 0 and not d.snapshot().fault_text
    finally:
        assert d.shutdown()


def last_run(d):
    snap = d.snapshot()
    return snap.run_no, snap.outcome, snap.result_run, snap.result


@pytest.mark.parametrize('busy', ['run', '_probing', '_checking', 'PC', '_closing', '_wedged'])
def test_busy_blocks_reset_and_queries_without_touching_last_run(tmp_path, monkeypatch, busy):
    d = daemon(tmp_path, dut_ip=IP)
    d.child_cmd = lambda args: pytest.fail('busy run')
    monkeypatch.setattr(hd, 'reset_panel', lambda *args: pytest.fail('busy GPIO'))
    monkeypatch.setattr(hd.gain_send, 'exchange', lambda *a, **kw: pytest.fail('busy gain_get'))
    d.run_no, d.result_run, d.outcome, d.result = 4, 4, pp.OUT_COMPLETED, ('completed #4',)
    if busy == 'run':
        d.state = pp.ST_RUN
    elif busy == 'PC':
        d.busy_check = lambda pid: pid is None
    else:
        setattr(d, busy, True)
    before = last_run(d)
    d.maybe_query_dut()
    assert d.start(d.reset_index) == d.start(0) == pp.RES_BUSY
    assert d.note.startswith('DUT reset skipped: ') and len(d.note.encode('ascii')) <= 22
    assert last_run(d) == before


@pytest.mark.parametrize('failure,code,note', [
    (None, pp.RES_OK, 'DUT reset done'), ('no wire', pp.RES_FAILED, 'DUT reset: no wire'),
    ('open', pp.RES_FAILED, 'DUT reset skipped: I/O'), ('out', pp.RES_FAILED, 'DUT reset failed: GPIO'),
    ('free', pp.RES_FAILED, 'DUT reset failed: GPIO'), ('close', pp.RES_FAILED, 'DUT reset failed: GPIO'),
])
def test_reset_gpio_and_note_preserve_completed_run(tmp_path, monkeypatch, failure, code, note):
    gpio = -1 if failure == 'no wire' else 27
    d = daemon(tmp_path, dut_reset_gpio=gpio, expect_dut=SHA)
    assert d.start(0) == pp.RES_OK
    wait_for(d, (pp.ST_END,))
    before, meta = last_run(d), run_meta(d)
    d.h7_ok = False
    d.on_fw_line(OTHER, 0, 'ladrc')
    lg = FakeLgpio(failure)
    monkeypatch.setitem(sys.modules, 'lgpio', lg)
    d.child_cmd = lambda _: pytest.fail('reset started a process')
    index = d.profiles().index('RESET DUT')
    assert index == d.snapshot().reset_index == len(d.profiles()) - 1
    assert d.start(index) == code and d.note == note
    assert last_run(d) == before and run_meta(d) == meta
    x = extra_of(d.snapshot(), 7)
    assert x.check_text == note and not x.flags & pp.X_CHECK_VALID
    if failure is None:
        assert lg.calls == [('open', 4), ('out', 27, 0, lg.SET_OPEN_DRAIN), ('free', 27), ('in', 27), ('close', 7)]
        assert not lg.en_held_low() and x.sha_state == pp.SHA_UNKNOWN and x.dut_sha == x.dut_env == ''
    if failure == 'no wire':
        assert not lg.calls
    if failure == 'free':
        assert ('in', 27) in lg.calls and not lg.en_held_low()


@pytest.fixture
def udp(monkeypatch):
    # The real gain_send.exchange selects the nonce, builds the datagram and parses the ack.
    class Dut:
        sha = SHA
        ok = True

        def __init__(self):
            self.packets = []

        def settimeout(self, timeout):
            assert 0 < timeout <= 0.3

        def sendto(self, data, dest):
            assert dest == (IP, 5014)
            self.packets.append(data)
            prefix, tx = data.decode('ascii').strip().rsplit('=', 1)
            assert prefix == '#gain_get,v=1,tx' and int(tx) > 0
            self.tx = tx

        def recvfrom(self, size):
            if self.sha is None:
                raise OSError('DUT unreachable')
            ack = (f'#gain_ack,tx={self.tx},req_id=0,ok={int(self.ok)},err=ok,sha={self.sha},env=ladrc,'
                   'mac=aa:bb:cc:dd:ee:ff,boot=00abcdef,id=0,applied_n=0,ctrl=PID,'
                   'kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1\n')
            return ack.encode('ascii'), (IP, 5014)

        def close(self):
            pass

    dut = Dut()
    monkeypatch.setattr(hd.gain_send.socket, 'socket', lambda *args: dut)
    return dut


@pytest.mark.parametrize('expected,answer,state', [(SHA.upper(), SHA, pp.SHA_OK), (SHA, OTHER, pp.SHA_MISMATCH),
                                                  (SHA, None, pp.SHA_UNKNOWN), (None, SHA, pp.SHA_NONE)])
def test_idle_identity_and_fresh_start_gate(tmp_path, monkeypatch, udp, expected, answer, state):
    t = [1.0]
    d = daemon(tmp_path, clock=lambda: t[0], dut_ip=IP, expect_dut=expected, ctrl='declared')
    d.maybe_query_dut()
    t[0] += hd.REPROBE_S - 0.01
    d.maybe_query_dut()
    assert len(udp.packets) == 1
    t[0] += 0.02
    d.maybe_query_dut()
    assert len(udp.packets) == 2
    udp.sha = answer
    code = d.start(0)
    assert len(udp.packets) == 3
    snap = d.snapshot()
    assert (snap.sha_state, snap.dut_sha, snap.dut_env) == (state, answer or '', 'ladrc' if answer else '')
    assert (snap.ctrl, snap.ctrl_src) == (('ladrc', pp.CTRL_SRC_FW) if answer else ('declared', pp.CTRL_SRC_PROFILE))
    if state == pp.SHA_MISMATCH:
        assert code == pp.RES_BAD_PROFILE and snap.run_no == 0 and d.proc is None
        monkeypatch.setitem(sys.modules, 'lgpio', FakeLgpio())
        assert d.start(d.reset_index) == pp.RES_OK and len(udp.packets) == 3
        assert d.snapshot().sha_state == pp.SHA_UNKNOWN
    else:
        assert code == pp.RES_OK
        wait_for(d, (pp.ST_END,))


def test_unread_rejected_or_unconfigured(tmp_path, monkeypatch, udp):
    d = daemon(tmp_path)
    d.maybe_query_dut()
    assert not udp.packets and d.snapshot().sha_state == pp.SHA_NONE
    d = daemon(tmp_path, dut_ip=IP, expect_dut=SHA)
    udp.ok = False
    d.maybe_query_dut()
    assert d.snapshot().sha_state == pp.SHA_UNKNOWN
    monkeypatch.setattr(hd.gain_send, 'exchange', lambda *args, **kw: (None, 1))
    assert d.start(0) == pp.RES_OK and d.snapshot().sha_state == pp.SHA_UNKNOWN
    wait_for(d, (pp.ST_END,))


def test_background_query_keeps_snapshot_available_and_reserves_dut(tmp_path, monkeypatch):
    d = daemon(tmp_path, dut_ip=IP)
    entered, release = threading.Event(), threading.Event()

    def exchange(ip, port, packet, timeout, tries, log):
        assert timeout <= 0.3 and tries == 1
        entered.set()
        assert release.wait(5)
        return None, 1

    monkeypatch.setattr(hd.gain_send, 'exchange', exchange)
    worker = threading.Thread(target=d.maybe_query_dut)
    worker.start()
    try:
        assert entered.wait(2)
        assert d.snapshot().run_state == pp.ST_IDLE
        assert d.start(0) == d.start(d.reset_index) == pp.RES_BUSY
        assert not d.probe_h7()
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()


def test_motor_on_report_arriving_before_phase_stdout_stops_at_run(tmp_path, monkeypatch):
    gate = tmp_path / 'release'
    monkeypatch.setenv('FAKE_HOLD_FILE', str(gate))
    monkeypatch.setenv('FAKE_RUN_S', '30')
    d = daemon(tmp_path, expect_dut=SHA)
    assert d.start(0) == pp.RES_OK
    try:
        d.on_fw_line(OTHER, False, 'ladrc')
        assert d.state == pp.ST_PREPARE and not d._meta.get('stop_signals')
        gate.touch()
        snap = wait_for(d, (pp.ST_ERROR,))
        assert snap.error_text == 'DUT FW wrong' and run_meta(d)['stop_signals'] == [0]
    finally:
        gate.touch()
        assert d.shutdown()


@pytest.mark.parametrize('sha,dirty,state,reason', [(SHA, '1', pp.SHA_DIRTY, 'DUT FW dirty'),
                                                    (OTHER, '0', pp.SHA_MISMATCH, 'DUT FW wrong')])
def test_motor_on_firmware_fault_stops_run_and_records_error(tmp_path, monkeypatch, udp, sha, dirty, state, reason):
    d = daemon(tmp_path, dut_ip=IP, expect_dut=SHA)
    monkeypatch.setenv('FAKE_RUN_S', '30')
    assert d.start(0) == pp.RES_OK
    try:
        wait_for(d, (pp.ST_RUN,))
        d.on_fw_line(SHA, '0', 'ladrc')
        assert d.snapshot().run_state == pp.ST_RUN and not d._meta.get('stop_signals')
        d.on_fw_line(sha, dirty, 'ladrc')
        assert d.snapshot().outcome == pp.OUT_ERROR and d._meta['stop_signals'] == [0]
        snap = wait_for(d, (pp.ST_ERROR,))
        assert snap.sha_state == state and snap.error_text == reason and snap.outcome == pp.OUT_ERROR
        meta = run_meta(d)
        assert meta['outcome'] == 'error' and meta['error'] == reason and meta['run_summary']['outcome'] == 'user_stop'
        if dirty == '1':
            assert d.stop() == pp.RES_OK  # acknowledges the FAULT
            assert d.start(0) == pp.RES_BAD_PROFILE and d.snapshot().sha_state == pp.SHA_DIRTY
            monkeypatch.setitem(sys.modules, 'lgpio', FakeLgpio())
            assert d.start(d.reset_index) == pp.RES_OK and d.snapshot().sha_state == pp.SHA_UNKNOWN
            assert d.start(0) == pp.RES_OK and d.snapshot().sha_state == pp.SHA_OK
    finally:
        assert d.shutdown()


def test_logger_ready_before_run_then_dirty_event_stops_fake_owner(tmp_path, monkeypatch, udp):
    d = daemon(tmp_path, dut_ip=IP, expect_dut=SHA)
    monkeypatch.setenv('FAKE_RUN_S', '30')
    seen = []
    on_fw_line = d.on_fw_line

    def record(*identity):
        seen.append(identity)
        on_fw_line(*identity)

    d.on_fw_line = record
    # Keep the production argv and session/stop handling. Substitute only the logger script.
    popen = hd.subprocess.Popen

    def spawn(cmd, **kw):
        if STUB in cmd:
            assert (Path(d.run_dir) / 'logger_ready.json').is_file()
        elif any(Path(s).name == 'agv_logger.py' for s in cmd):
            argv = list(cmd)
            idx = argv.index(str(hd.panel_result.TOOLS / 'agv_logger.py'))
            argv[idx] = str(Path(__file__).with_name('fake_panel_logger.py'))
            return popen(argv, **kw)
        return popen(cmd, **kw)

    del d._logger_cmd  # use the actual class method for this run
    monkeypatch.setattr(hd.subprocess, 'Popen', spawn)
    assert d.start(0) == pp.RES_OK
    try:
        wait_for(d, (pp.ST_RUN,))
        folder = Path(d.run_dir)
        (folder / 'motor_on').touch()
        snap = wait_for(d, (pp.ST_ERROR,))
        assert snap.sha_state == pp.SHA_DIRTY and snap.error_text == 'DUT FW dirty'
        assert seen == [(SHA, '0', 'pid'), (SHA, '1', 'ladrc')]
        meta = run_meta(d)
        logger = meta['logger']
        assert logger['argv'][-7:] == ['--strict', '--src-ip', IP, '--out', str(folder / 'esp32.csv'),
                                       '--ready-file', str(folder / 'logger_ready.json')]
        assert 'first' not in logger['argv']
        assert logger['ready'] and logger['group_gone'] and logger['exit_code'] == 0
        assert meta['stop_signals'] == logger['stop_signals'] == [0]
        assert meta['run_summary']['outcome'] == 'user_stop'
        assert snap.tracks[pp.TR_DUT] == (((0, 0), (60, 0)),)
        assert meta['check']['pos_reset_mid'] == 0
        assert d.start(0) == pp.RES_BAD_PROFILE
    finally:
        assert d.shutdown()


def test_unready_logger_is_reaped_and_run_proceeds(tmp_path, monkeypatch, udp):
    d = daemon(tmp_path, dut_ip=IP)
    d._logger_cmd = lambda folder: [sys.executable, '-c', 'import sys; sys.stdin.readline()']
    monkeypatch.setattr(hd, 'LOGGER_READY_S', .15)
    assert d.start(0) == pp.RES_OK
    wait_for(d, (pp.ST_END,))
    logger = run_meta(d)['logger']
    assert not logger['ready'] and logger['reason'] == 'logger not ready'
    assert logger['group_gone'] and logger['stop_signals'] == [0]


def test_without_ip_no_logger_and_both_starts_refused_until_check_finishes(tmp_path, monkeypatch):
    now = [1.0]
    d = daemon(tmp_path, clock=lambda: now[0])
    d._logger_cmd = lambda folder: pytest.fail('unexpected UDP logger')
    entered, release = threading.Event(), threading.Event()
    run_check = hd.panel_result.run_check

    def check(*args):
        entered.set()
        assert release.wait(5)
        return run_check(*args)

    monkeypatch.setattr(hd.panel_result, 'run_check', check)
    assert d.start(0) == pp.RES_OK
    try:
        assert entered.wait(5)
        assert d.start(0) == d.start(d.reset_index) == pp.RES_BUSY
        x = extra_of(d.snapshot(), 1)
        assert x.flags & pp.X_CHECK_VALID and (x.check_code, x.check_run) == (pp.CHK_RUNNING, 1)
        release.set()
        s = wait_for(d, (pp.ST_END,))
        m = run_meta(d)
        assert m['logger'] == dict(ready=False, reason='no --dut-ip')
        assert not (Path(d.run_dir) / 'esp32.csv').exists()
        assert not s.tracks and m['check']['pos_reset_mid'] is None
        assert m['check']['code'] == s.check_code == pp.CHK_INCONCLUSIVE
        now[0] += 3  # the transient RESET DUT feedback relinquishes the shared text field
        x = extra_of(d.snapshot(), 1)
        assert x.flags & pp.X_CHECK_VALID and (x.check_run, x.check_code, x.check_text) == (
            1, m['check']['code'], m['check']['text'])
    finally:
        release.set()
        assert d.shutdown()
