"""Tests of tools/gain_send.py (no hardware). Run: python -m pytest tools/tests/test_gain_send.py -q

FakeEsp32 is a small UDP server on 127.0.0.1 that answers like AGV_MECANUM src/gain_rx/gain_rx.c after
the Codex round-1 fixes (12/11): '#gain_set,v=1,tx=..' / '#gain_get,v=1,tx=..' in,
'#gain_ack,tx=..,req_id=..,ok=..,err=..,sha=..,env=..,mac=..,boot=..[,id=..,applied_n=..,ctrl=..,<7 gains>]'
out, with the gains running after the request; switches for the motors on, silence, a lost answer
after the set was applied, a wrong value, a stale answer, an answer from another port, an incomplete
answer. It is a copy of the protocol, not of the firmware: gain_set.c has its own host test
(fwsim/tests/test_gain_set.py) and the board is checked on the bench."""
import csv
import json
import os
import socket
import struct
import subprocess
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import gain_send as gs  # noqa: E402

PID = dict(kp_xy='1.2', ki_xy='0.4', kp_wz='1.0', ki_wz='0.05', pos_kp='6', pos_ki='3', k_psi='1')
DEFAULT_PID = dict(kp_xy=1.5, ki_xy=0.05, kp_wz=1.0, ki_wz=0.05, pos_kp=6.0, pos_ki=3.0, k_psi=1.0)
DEFAULT_ADRC = dict(omega_c_xy=8.0, omega_o_xy=30.0, omega_c_wz=8.0, omega_o_wz=30.0, pos_kp=6.0, pos_ki=3.0, k_psi=1.0)
MAC = 'aa:bb:cc:00:11:22'


class FakeEsp32:
    def __init__(self, mode='ok', ctrl='PID'):
        self.mode, self.ctrl = mode, ctrl
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', 0))
        self.sock.settimeout(0.05)
        self.port = self.sock.getsockname()[1]
        self.other = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)    # answers from another port
        default = DEFAULT_PID if ctrl == 'PID' else DEFAULT_ADRC
        self.running = dict(id=0, gains={k: gs.f32(v) for k, v in default.items()})
        self.applied_n = 0
        self.received = []
        self._stop = False
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def close(self):
        self._stop = True
        self.t.join(1)
        self.sock.close()
        self.other.close()

    def _ack(self, src, tx, req_id, err, with_gains=True):
        line = (f'#gain_ack,tx={tx},req_id={req_id},ok={int(err == "")},err={err},sha=abc1234,env=test,'
                f'mac={MAC},boot=0000abcd')
        if with_gains:
            g = self.running['gains']
            line += (f',id={self.running["id"]},applied_n={self.applied_n},ctrl={self.ctrl},'
                     + ','.join(f'{n}={"%.9g" % g[n]}' for n in gs.NAMES[self.ctrl]))
        (self.other if self.mode == 'wrong_port' else self.sock).sendto((line + '\n').encode(), src)

    def _loop(self):
        while not self._stop:
            try:
                raw, src = self.sock.recvfrom(1024)
            except (socket.timeout, OSError):
                continue
            text = raw.decode().strip()
            self.received.append(text)
            head, *rest = text.split(',')
            parts = dict(p.split('=', 1) for p in rest if '=' in p)
            tx = parts.get('tx', '0')
            if self.mode == 'silent' or (self.mode == 'silent_first' and len(self.received) == 1):
                continue
            if self.mode == 'stale_first':
                self._ack(src, '999', parts.get('id', '0'), '')
            if head == '#gain_get':
                self._ack(src, tx, '0', '', with_gains=self.mode != 'incomplete')
                continue
            req_id = parts.get('id', '0')
            if parts.get('v') != '1' or tx == '0':
                self._ack(src, tx, '0', 'format', with_gains=False)
                continue
            if self.mode == 'motor_on':
                self._ack(src, tx, req_id, 'motor_on')
                continue
            if parts.get('ctrl') != self.ctrl:
                self._ack(src, tx, req_id, 'ctrl')
                continue
            gains = {n: struct.unpack('<f', struct.pack('<f', float(parts[n])))[0] for n in gs.NAMES[self.ctrl]}
            if self.mode == 'wrong_value':
                gains['k_psi'] = gs.f32(gains['k_psi'] + 0.5)
            self.running = dict(id=int(req_id), gains=gains)
            self.applied_n += 1
            if self.mode == 'ack_lost_first' and len(self.received) == 1:
                continue                                   # applied, but the answer is lost
            self._ack(src, tx, req_id, '')


@pytest.fixture
def fake(request):
    f = FakeEsp32(**getattr(request, 'param', {}))
    yield f
    f.close()


def args(fake, *extra):
    # C3-8: a manual send declares its source; --s04-run, --query and an explicit --nguon-gain-tay
    # replace the default declaration
    src = [] if any(x in extra for x in ('--s04-run', '--nguon-gain-tay', '--query')) \
        else ['--nguon-gain-tay', 'test bench']
    return ['--ip', '127.0.0.1', '--port', str(fake.port), '--timeout', '0.3', '--tries', '3', *extra,
            *src]


def gains_arg(d):
    return ','.join(f'{k}={v}' for k, v in d.items())


def sent_tx(text):
    return dict(p.split('=', 1) for p in text.split(',')[1:])['tx']


def spy_exchange(monkeypatch):
    """Record calls to gs.exchange while still running the real one; returns the calls list."""
    calls = []
    real = gs.exchange
    def spy(*a, **kw):
        calls.append(a)
        return real(*a, **kw)
    monkeypatch.setattr(gs, 'exchange', spy)
    return calls


def test_float32_format_round_trips_bits():
    for x in (0.1, 3.3333333, 1e-7, 19.999998, 0.3, 7.1, 2.718281828):
        s = gs.fmt(x)
        assert struct.pack('<f', float(s)) == struct.pack('<f', x)


def test_normalize_aliases_and_errors():
    g = gs.normalize('ADRC', dict(wc_xy=10, wo_xy=35, omega_c_wz=12, wo_wz=40, pos_kp=5, pos_ki=2, k_psi=1))
    assert list(g) == gs.NAMES['ADRC']
    with pytest.raises(gs.UsageError, match='missing k_psi'):
        gs.normalize('PID', {k: v for k, v in PID.items() if k != 'k_psi'})
    with pytest.raises(gs.UsageError, match='not for PID'):
        gs.normalize('PID', dict(PID, omega_c_xy=3))
    with pytest.raises(gs.UsageError, match='not finite'):
        gs.normalize('PID', dict(PID, kp_xy='nan'))
    with pytest.raises(gs.UsageError, match='twice'):
        gs.normalize('ADRC', dict(wc_xy=1, omega_c_xy=1, wo_xy=35, omega_c_wz=12, wo_wz=40, pos_kp=5, pos_ki=2, k_psi=1))
    with pytest.raises(gs.UsageError, match='float32'):
        gs.normalize('PID', dict(PID, kp_xy='1e100'))                       # Codex L90-11
    with pytest.raises(gs.UsageError, match='object'):
        gs.normalize('PID', [])
    with pytest.raises(gs.UsageError, match='not a number'):
        gs.normalize('PID', dict(PID, kp_xy=True))


def test_gains_arg_rejects_duplicates():
    with pytest.raises(gs.UsageError, match='twice'):
        gs.parse_gains_arg('kp_xy=1,kp_xy=2')


def test_packet_text():
    p = gs.build_packet(12, 'PID', gs.normalize('PID', PID), 77)
    assert p == ('#gain_set,v=1,tx=77,id=12,ctrl=PID,kp_xy=1.20000005,ki_xy=0.400000006,kp_wz=1,'
                 'ki_wz=0.0500000007,pos_kp=6,pos_ki=3,k_psi=1')


def test_applied_and_verified(fake, capsys):
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '12', '--gains', gains_arg(PID))) == 0
    out = capsys.readouterr().out
    assert out.rstrip().endswith('#ket_qua,ma=0') and 'applied: id=12' in out and f'mac={MAC}' in out
    assert fake.running['id'] == 12 and len(fake.received) == 1


def test_resend_uses_new_tx(fake, capsys):
    fake.mode = 'silent_first'
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '5', '--gains', gains_arg(PID))) == 0
    assert len(fake.received) == 2 and sent_tx(fake.received[0]) != sent_tx(fake.received[1])


@pytest.mark.parametrize('fake', [dict(mode='ack_lost_first')], indirect=True)
def test_applied_but_answer_lost_then_resent(fake, capsys):
    """Codex L90-14: the first set is applied but its answer is lost; the resend applies it again and the
    answer shows the running set, so the tool reports it correctly."""
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '5', '--gains', gains_arg(PID))) == 0
    assert fake.applied_n == 2 and 'applied_n=2' in capsys.readouterr().out


@pytest.mark.parametrize('fake', [dict(mode='silent')], indirect=True)
def test_no_answer_is_3(fake, capsys):
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '5', '--gains', gains_arg(PID))) == 3
    assert len(fake.received) == 3
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=3')


@pytest.mark.parametrize('fake', [dict(mode='wrong_port')], indirect=True)
def test_answer_from_other_port_is_ignored(fake, capsys):
    """Codex L90-8: only answers from the ESP32's port count."""
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '5', '--gains', gains_arg(PID))) == 3


@pytest.mark.parametrize('fake', [dict(mode='motor_on')], indirect=True)
def test_motor_on_is_rejected_without_retry(fake, capsys):
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '5', '--gains', gains_arg(PID))) == 2
    out = capsys.readouterr().out
    assert 'turn SWA off' in out and len(fake.received) == 1


@pytest.mark.parametrize('fake', [dict(mode='wrong_value')], indirect=True)
def test_running_value_differs_is_4(fake, capsys):
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '5', '--gains', gains_arg(PID))) == 4
    assert 'k_psi: running' in capsys.readouterr().out


@pytest.mark.parametrize('fake', [dict(mode='stale_first')], indirect=True)
def test_stale_answer_is_ignored(fake, capsys):
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '6', '--gains', gains_arg(PID))) == 0


@pytest.mark.parametrize('fake', [dict(ctrl='ADRC')], indirect=True)
def test_wrong_controller_is_rejected(fake, capsys):
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '5', '--gains', gains_arg(PID))) == 2
    assert 'err=ctrl' in capsys.readouterr().out


def test_mac_check(fake, capsys):
    assert gs.main(args(fake, '--mac', MAC.upper(), '--ctrl', 'PID', '--id', '5', '--gains', gains_arg(PID))) == 0
    assert gs.main(args(fake, '--mac', '11:22:33:44:55:66', '--ctrl', 'PID', '--id', '6', '--gains', gains_arg(PID))) == 4
    assert 'WRONG BOARD' in capsys.readouterr().out


def test_file_forms_and_conflicts(fake, tmp_path, capsys):
    f1 = tmp_path / 'gain_A-PID_03.json'
    f1.write_text(json.dumps(dict(ctrl='PID', id=3, gains={k: float(v) for k, v in PID.items()})))
    assert gs.main(args(fake, '--file', str(f1))) == 0 and fake.running['id'] == 3
    f2 = tmp_path / 'gain_flat.json'
    f2.write_text(json.dumps({k: float(v) for k, v in PID.items()}))
    assert gs.main(args(fake, '--file', str(f2))) == 1                        # no ctrl, no id
    assert gs.main(args(fake, '--file', str(f2), '--ctrl', 'PID', '--id', '4')) == 0
    assert gs.main(args(fake, '--file', str(f1), '--id', '9')) == 1           # id disagrees with file
    assert gs.main(args(fake, '--file', str(f1), '--gains', gains_arg(PID))) == 1


@pytest.mark.parametrize('body', [
    '{"ctrl": "PID", "id": 3.9, "gains": {}}',                    # id not an integer (Codex L90-11)
    '{"ctrl": "PID", "id": true, "gains": {}}',
    '{"ctrl": "PID", "id": 3, "gains": []}',                      # gains not an object
    '{"ctrl": "PID", "id": 3, "id": 4, "gains": {}}',             # duplicate JSON key
    '{"ctrl": "PID", "id": 3, "gains": {"kp_xy": 1, "kp_xy": 2}}',
    '{"ctrl": "PID", "id": null, "gains": {}}',                   # id key present but null (round 2)
    '{"ctrl": "PID", "id": 3, "gains": {"kp_xy": 1' + '0' * 400 + '}}',   # integer too big for a float
])
def test_bad_files_are_usage_errors(fake, tmp_path, capsys, body):
    f = tmp_path / 'gain_bad.json'
    f.write_text(body)
    assert gs.main(args(fake, '--file', str(f))) == 1
    assert gs.main(args(fake, '--file', str(f), '--id', '3')) == 1           # --id does not hide a bad file id
    assert fake.received == []


def test_null_id_with_full_gains_is_refused(fake, tmp_path, capsys):
    """Codex round 2 (L90-11): "id": null must not be read as 'no id' and then filled by --id."""
    f = tmp_path / 'gain_null_id.json'
    f.write_text(json.dumps(dict(ctrl='PID', id=None, gains={k: float(v) for k, v in PID.items()})))
    assert gs.main(args(fake, '--file', str(f), '--id', '3')) == 1
    assert fake.received == []


def test_bad_arguments_send_nothing(fake, capsys):
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '0', '--gains', gains_arg(PID))) == 1
    assert gs.main(args(fake, '--ctrl', 'PID', '--gains', gains_arg(PID))) == 1
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '3', '--gains', 'kp_xy=1')) == 1
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '3', '--gains', 'kp_xy=1,kp_xy=2')) == 1
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '3', '--gains', gains_arg(dict(PID, kp_xy='1e100')))) == 1
    assert fake.received == []


def test_query_reads_running_set(fake, capsys):
    assert gs.main(args(fake, '--query')) == 0
    out = capsys.readouterr().out
    assert '#gain_running,' in out and 'id=0' in out and 'kp_xy=1.5' in out
    assert fake.running['id'] == 0 and fake.received[0].startswith('#gain_get,v=1,tx=')


@pytest.mark.parametrize('fake', [dict(mode='incomplete')], indirect=True)
def test_query_incomplete_answer_is_4(fake, capsys):
    """Codex L90-12: an answer without the running set is not a successful query."""
    assert gs.main(args(fake, '--query')) == 4
    assert 'INCOMPLETE' in capsys.readouterr().out


def test_gain_table_violation_sends_nothing(fake, tmp_path, capsys, monkeypatch):
    calls = spy_exchange(monkeypatch)
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '3',
                        '--gains', gains_arg(dict(PID, kp_xy='41.01')))) == 1
    cap = capsys.readouterr()
    assert cap.out.rstrip().endswith('#ket_qua,ma=1')
    assert 'kp_xy' in cap.err and 'upper bound' in cap.err and 'nothing sent' in cap.err
    assert len(calls) == 0 and fake.received == []
    adrc = FakeEsp32(ctrl='ADRC')
    try:
        a = dict(omega_c_xy='100', omega_o_xy='50', omega_c_wz='8', omega_o_wz='30',
                 pos_kp='6', pos_ki='3', k_psi='1')
        assert gs.main(args(adrc, '--ctrl', 'ADRC', '--id', '3', '--gains', gains_arg(a))) == 1
        cap = capsys.readouterr()
        assert cap.out.rstrip().endswith('#ket_qua,ma=1')
        assert 'omega_c_xy + 2*omega_o_xy' in cap.err and 'nothing sent' in cap.err
        assert len(calls) == 0 and adrc.received == []
        # S03 v8.6: the wz triangle is a hard violation like xy
        a = dict(omega_c_xy='8', omega_o_xy='30', omega_c_wz='100', omega_o_wz='50',
                 pos_kp='6', pos_ki='3', k_psi='1')
        assert gs.main(args(adrc, '--ctrl', 'ADRC', '--id', '3', '--gains', gains_arg(a))) == 1
        cap = capsys.readouterr()
        assert cap.out.rstrip().endswith('#ket_qua,ma=1')
        assert 'omega_c_wz + 2*omega_o_wz' in cap.err and 'nothing sent' in cap.err
        assert len(calls) == 0 and adrc.received == []
    finally:
        adrc.close()
    f = tmp_path / 'gain_violating.json'
    f.write_text(json.dumps(dict(ctrl='PID', id=4,
                                 gains={k: float(v) for k, v in dict(PID, kp_xy='41.01').items()})))
    assert gs.main(args(fake, '--file', str(f))) == 1
    cap = capsys.readouterr()
    assert cap.out.rstrip().endswith('#ket_qua,ma=1')
    assert 'upper bound' in cap.err and 'nothing sent' in cap.err
    assert len(calls) == 0 and fake.received == []


@pytest.mark.parametrize('fake', [dict(ctrl='ADRC')], indirect=True)
def test_gain_table_warning_still_sends(fake, capsys, monkeypatch):
    calls = spy_exchange(monkeypatch)
    # S03 v8.6: the wz triangle is a violation now; the only warning left is the pos_kp/pos_ki wedge
    a = dict(omega_c_xy='8', omega_o_xy='30', omega_c_wz='8', omega_o_wz='30',
             pos_kp='10', pos_ki='55', k_psi='1')
    assert gs.main(args(fake, '--ctrl', 'ADRC', '--id', '3', '--gains', gains_arg(a))) == 0
    out = capsys.readouterr().out
    assert '# warning: pos_kp' in out and len(fake.received) == 1
    assert len(calls) == 1                       # the spy sits on the real send path


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.parametrize('argv', [
    ('tools/gain_send.py', '--help'),
    ('-m', 'tools.gain_send', '--help'),
    ('-c', 'import tools.gain_send'),
])
def test_runs_as_script_module_and_import(argv):
    """gain_send must run as tools/gain_send.py, as -m tools.gain_send, and import as tools.gain_send."""
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    env['PYTHONIOENCODING'] = 'utf-8'            # the --help docstring has Vietnamese chars
    r = subprocess.run([sys.executable, *argv], cwd=ROOT, capture_output=True, encoding='utf-8', env=env)
    assert r.returncode == 0, r.stderr
    if '--help' in argv:
        assert 'gain_send.py' in r.stdout


# ── S04 section 6 hand-over (--s04-run, --liam-da-quyet) and the spectral indicator ──

ADRC = dict(omega_c_xy='8', omega_o_xy='30', omega_c_wz='8', omega_o_wz='30',
            pos_kp='6', pos_ki='3', k_psi='1')


def _sel_row(ctrl='PID', **kw):
    """A send_to_HIL row with the float32 values of the normalized default gains."""
    g = gs.normalize(ctrl, PID if ctrl == 'PID' else ADRC)
    r = dict(kind='send_to_HIL', controller=ctrl, feasible='true',
             **{n: repr(g[n]) for n in gs.NAMES[ctrl]})
    r.update(kw)
    return r


def _s04_run(tmp_path, ctrl='PID', cho=None, notes='default', rows='default', tool_check=False):
    """<run>/post_checks/notes.json + <run>/selected.csv of a fake S04 run. notes 'default': the C3-8
    record of a finished post -- dict(tool_check, post_trang_thai='hoan_tat', selected_sha256,
    selected_sha256_lf of the file's bytes, cho_liam); rows 'default': one send_to_HIL row of ctrl."""
    import hashlib
    run = tmp_path / 's04'
    (run / 'post_checks').mkdir(parents=True)
    raw = b''
    if rows is not None:
        cols = ['kind', 'controller', 'feasible'] + gs.NAMES[ctrl]
        with open(run / 'selected.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in ([_sel_row(ctrl)] if rows == 'default' else rows):
                w.writerow({n: r.get(n, '') for n in cols})
        raw = (run / 'selected.csv').read_bytes()
    if notes == 'default':
        (run / 'post_checks' / 'notes.json').write_text(
            json.dumps(dict(tool_check=tool_check, post_trang_thai='hoan_tat',
                            selected_sha256=hashlib.sha256(raw).hexdigest(),
                            selected_sha256_lf=hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest(),
                            cho_liam=list(cho or []))), encoding='utf-8')
    elif notes is not None:
        (run / 'post_checks' / 'notes.json').write_text(notes, encoding='utf-8')
    return run


def _gains(ctrl='PID'):
    return gs.normalize(ctrl, PID if ctrl == 'PID' else ADRC)


def test_s04_hand_over_refused_by_bad_notes_and_selected(tmp_path):
    """s04_hand_over code 5 on every malformed input; nothing may be sent."""
    g = _gains('PID')
    assert gs.s04_hand_over(str(tmp_path / 'none'), 'PID', g, None)[0] == 5        # missing run
    run = _s04_run(tmp_path / 'a', notes='not json {{')
    assert gs.s04_hand_over(str(run), 'PID', g, None)[0] == 5                     # not JSON
    for i, body in enumerate(('[1]', '{}', '{"cho_liam": "x"}')):                 # list, no key, not a list
        run = _s04_run(tmp_path / f'j{i}', notes=body)
        code, _, err = gs.s04_hand_over(str(run), 'PID', g, None)
        assert code == 5 and 'cho_liam' in err[0]
    run = _s04_run(tmp_path / 'b', rows=None)                                     # selected.csv missing
    assert gs.s04_hand_over(str(run), 'PID', g, None)[0] == 5
    for rows in ([dict(_sel_row('ADRC'), controller='ADRC')],                     # none of this ctrl
                 [_sel_row('PID'), _sel_row('PID')],                              # two
                 [_sel_row('PID', feasible='false')]):                            # not feasible
        run = _s04_run(tmp_path / f'r{len(rows)}{rows[0]["feasible"][:2]}', rows=rows)
        code, _, err = gs.s04_hand_over(str(run), 'PID', g, None)
        assert code == 5 and 'exactly one' in err[0]
    bad = _sel_row('PID', kp_xy=repr(_f32_next(g['kp_xy'])))                      # one float32 ulp off
    run = _s04_run(tmp_path / 'u', rows=[bad])
    code, _, err = gs.s04_hand_over(str(run), 'PID', g, None)
    assert code == 5 and 'kp_xy' in err[0]


def _f32_next(x):
    """One float32 ulp above x."""
    return struct.unpack('<f', struct.pack('<i', struct.unpack('<i', struct.pack('<f', x))[0] + 1))[0]


def test_s04_hand_over_float32_equality(tmp_path):
    """The gains sent are compared as float32: a binary64 value that rounds to the same float32 of the
    selected row is accepted; one ulp of float32 difference is refused (above)."""
    g = dict(_gains('PID'), kp_xy=1.50000001)              # rounds to the same float32 as 1.5
    row = _sel_row('PID', kp_xy=repr(gs.f32(1.5)))
    run = _s04_run(tmp_path, rows=[row])
    assert gs.s04_hand_over(str(run), 'PID', g, None)[0] == 0


def test_s04_hand_over_cho_liam_blocks_and_overrides(tmp_path):
    """cho_liam non-empty: code 5 without a reason (an item of ANOTHER controller blocks too, every
    item listed); code 1 with a reason on an empty list; code 0 lines otherwise."""
    g = _gains('PID')
    items = [dict(controller='ADRC', check='spectral_radius', reasons=['r1', 'r2']),
             dict(controller='PID', check='three_mu', reasons=['r3'])]
    run = _s04_run(tmp_path / 'a', cho=items)
    code, out, err = gs.s04_hand_over(str(run), 'PID', g, None)
    assert code == 5 and out == []
    assert any('ADRC' in l and 'spectral_radius' in l and '2' in l for l in err)
    assert any('PID' in l and 'three_mu' in l for l in err)
    run = _s04_run(tmp_path / 'b')
    assert gs.s04_hand_over(str(run), 'PID', g, 'Liam checked')[0] == 1           # nothing to override
    code, out, err = gs.s04_hand_over(str(run), 'PID', g, None)                   # clean: code 0
    assert code == 0 and err == [] and len(out) == 1
    assert out[0].startswith('#ban_giao_s04,run=') and 'tool_check=false' in out[0]   # C3-47: never a tool check
    assert 'cho_liam=0' in out[0] and 'liam_da_quyet=""' in out[0]
    code, out, err = gs.s04_hand_over(str(_s04_run(tmp_path / 'c', cho=items)), 'PID', g,
                                      'Liam đã xem và duyệt')
    assert code == 0 and 'cho_liam=2' in out[0] and 'đã' not in out[0] and '\\u0111' in out[0]
    assert len([l for l in out if l.startswith('# cho_liam:')]) == 2


def test_main_s04_run_refusal_and_override(fake, tmp_path, capsys):
    """Through main() + FakeEsp32: refused sends nothing (5); the override applies the set (0)."""
    items = [dict(controller='ADRC', check='b0_factor', reasons=['r'])]
    run = _s04_run(tmp_path / 'a', cho=items)
    base = args(fake, '--ctrl', 'PID', '--id', '3', '--gains', gains_arg(PID), '--s04-run', str(run))
    assert gs.main(base) == 5
    txt = capsys.readouterr()
    assert txt.out.rstrip().endswith('#ket_qua,ma=5') and '# cho_liam:' in txt.err
    assert fake.received == []
    assert gs.main(base + ['--liam-da-quyet', 'Liam checked']) == 0
    out = capsys.readouterr().out
    assert '#ban_giao_s04,' in out and 'Liam checked' in out and 'cho_liam=1' in out
    assert fake.received and fake.running['id'] == 3
    run2 = _s04_run(tmp_path / 'b')                                               # clean post
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '4', '--gains', gains_arg(PID),
                        '--s04-run', str(run2))) == 0
    out = capsys.readouterr().out
    assert 'liam_da_quyet=""' in out and out.rstrip().endswith('#ket_qua,ma=0')


def test_main_liam_da_quyet_usage_errors(fake, tmp_path, capsys):
    """--liam-da-quyet without --s04-run, a blank reason, and a reason on an empty cho_liam: 1."""
    base = args(fake, '--ctrl', 'PID', '--id', '3', '--gains', gains_arg(PID))
    assert gs.main(base + ['--liam-da-quyet', 'x']) == 1
    run = _s04_run(tmp_path)
    assert gs.main(base + ['--s04-run', str(run), '--liam-da-quyet', '  ']) == 1  # blank reason
    assert gs.main(base + ['--s04-run', str(run), '--liam-da-quyet', 'x']) == 1   # empty cho_liam
    assert fake.received == []
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=1')


def _check_send_output(out):
    """Every successful send: '# ban_kinh_pho=' before '# send', exactly one '# ack ' line,
    '#ket_qua,ma=0' last."""
    lines = out.splitlines()
    assert sum(l.startswith('# ack ') for l in lines) == 1
    assert out.index('# ban_kinh_pho=') < out.index('# send')
    assert lines[-1] == '#ket_qua,ma=0'


def test_spectral_indicator_pid_and_adrc(fake, capsys):
    """S04 v6.8 6: the '# ban_kinh_pho=' indicator line is printed before '# send' (report only)."""
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '3', '--gains', gains_arg(PID))) == 0
    out = capsys.readouterr().out
    _check_send_output(out)
    assert 'trang_thai=on_dinh' in out and 'bien_trang_thai=' in out
    adrc = FakeEsp32(ctrl='ADRC')
    try:
        assert gs.main(args(adrc, '--ctrl', 'ADRC', '--id', '3', '--gains', gains_arg(ADRC))) == 0
        out = capsys.readouterr().out
        _check_send_output(out)
        assert 'trang_thai=on_dinh' in out
    finally:
        adrc.close()


def test_spectral_indicator_unstable_still_sends(fake, capsys):
    """rho >= 1 (PID pos_kp=0, pos_ki=3, inside the table) warns and still sends."""
    g = dict(PID, pos_kp='0', pos_ki='3')
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '3', '--gains', gains_arg(g))) == 0
    out = capsys.readouterr().out
    _check_send_output(out)
    assert '# warning: ban_kinh_pho >= 1' in out


def test_spectral_indicator_failure_still_sends(fake, capsys, monkeypatch):
    """A mo_hinh_con failure gives one 'not computed' line and the send goes on."""
    import mo_hinh_con
    monkeypatch.setattr(mo_hinh_con, 'ban_kinh_pho_va_trang_thai',
                        lambda *a, **k: (_ for _ in ()).throw(ValueError('boom')))
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '3', '--gains', gains_arg(PID))) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert sum(l.startswith('# ack ') for l in lines) == 1 and lines[-1] == '#ket_qua,ma=0'
    assert out.index('# ban_kinh_pho: not computed (') < out.index('# send')
    assert '# warning: ban_kinh_pho' not in out


# ── C3-8 / C3-47: the declared gain source and the hardened hand-over ──


def _manual_args(fake, *extra):
    """args() without the default '--nguon-gain-tay' the helper adds (the source rules are on test)."""
    return ['--ip', '127.0.0.1', '--port', str(fake.port), '--timeout', '0.3', '--tries', '3', *extra]


def test_c3_8_gain_send_needs_a_declared_source(fake, capsys):
    """C3-8: no --s04-run and no --nguon-gain-tay -> refused (5), nothing sent; a manual gain set
    names its source with --nguon-gain-tay; both flags or a blank description are usage errors (1)."""
    base = _manual_args(fake, '--ctrl', 'PID', '--id', '5', '--gains', gains_arg(PID))
    assert gs.main(base) == 5 and fake.received == []
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=5')
    assert gs.main(base + ['--nguon-gain-tay', 'thu ban']) == 0
    out = capsys.readouterr().out
    assert '#nguon_gain,loai=tay,mo_ta="thu ban"' in out
    assert out.index('#nguon_gain') < out.index('# send')
    assert gs.main(base + ['--nguon-gain-tay', 'x', '--s04-run', 'y']) == 1
    assert gs.main(base + ['--nguon-gain-tay', '   ']) == 1


def test_c3_47_s04_run_refusals_at_unit_level(fake, tmp_path, capsys):
    """C3-8/C3-47 through gs.main: a tool_check post, a changed selected.csv, broken cho_liam items
    and a post that did not run to its end are all refused (5), nothing sent; the LF/CRLF variant of
    the same content is accepted."""
    def base(run):
        return _manual_args(fake, '--ctrl', 'PID', '--id', '5', '--gains', gains_arg(PID),
                            '--s04-run', str(run))

    run = _s04_run(tmp_path / 'a', tool_check=True)
    assert gs.main(base(run)) == 5 and fake.received == []
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=5')
    run = _s04_run(tmp_path / 'b')
    p = run / 'selected.csv'
    raw = p.read_bytes()
    p.write_bytes(raw + b'\n')
    assert gs.main(base(run)) == 5 and fake.received == []
    p.write_bytes(raw.replace(b'\r\n', b'\n'))                     # same content, LF ends: accepted
    assert gs.main(base(run)) == 0
    for i, bad in enumerate([[None], [{}], [{'controller': 'PID', 'check': 'x'}],
                             [{'controller': 'PID', 'check': 'x', 'reasons': 'r'}]]):
        run = _s04_run(tmp_path / f'c{i}')
        np_ = run / 'post_checks' / 'notes.json'
        n = json.loads(np_.read_text(encoding='utf-8'))
        n['cho_liam'] = bad
        np_.write_text(json.dumps(n), encoding='utf-8')
        n_sent = len(fake.received)
        assert gs.main(base(run)) == 5 and len(fake.received) == n_sent
        assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=5')
    for i, st in enumerate(('dang_chay', None)):
        run = _s04_run(tmp_path / f'd{i}')
        np_ = run / 'post_checks' / 'notes.json'
        n = json.loads(np_.read_text(encoding='utf-8'))
        if st is None:
            del n['post_trang_thai']
        else:
            n['post_trang_thai'] = st
        np_.write_text(json.dumps(n), encoding='utf-8')
        n_sent = len(fake.received)
        assert gs.main(base(run)) == 5 and len(fake.received) == n_sent


def test_c3_47_hand_over_exception_is_code_5(fake, tmp_path, capsys, monkeypatch):
    """C3-47: a crash inside s04_hand_over is a refused send (5) ending with #ket_qua,ma=5, never an
    exception without the result line."""
    run = _s04_run(tmp_path)
    monkeypatch.setattr(gs, 's04_hand_over',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    assert gs.main(_manual_args(fake, '--ctrl', 'PID', '--id', '5', '--gains', gains_arg(PID),
                                '--s04-run', str(run))) == 5
    assert fake.received == [] and capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=5')
