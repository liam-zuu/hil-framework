"""Tests of tools/gain_send.py (no hardware). Run: python -m pytest tools/tests/test_gain_send.py -q

FakeEsp32 is a small UDP server on 127.0.0.1 that answers like AGV_MECANUM src/gain_rx/gain_rx.c after
the Codex round-1 fixes (12/11): '#gain_set,v=1,tx=..' / '#gain_get,v=1,tx=..' in,
'#gain_ack,tx=..,req_id=..,ok=..,err=..,sha=..,env=..,mac=..,boot=..[,id=..,applied_n=..,ctrl=..,<7 gains>]'
out, with the gains running after the request; switches for the motors on, silence, a lost answer
after the set was applied, a wrong value, a stale answer, an answer from another port, an incomplete
answer. It is a copy of the protocol, not of the firmware: gain_set.c has its own host test
(fwsim/tests/test_gain_set.py) and the board is checked on the bench."""
import json
import os
import socket
import struct
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
    return ['--ip', '127.0.0.1', '--port', str(fake.port), '--timeout', '0.3', '--tries', '3', *extra]


def gains_arg(d):
    return ','.join(f'{k}={v}' for k, v in d.items())


def sent_tx(text):
    return dict(p.split('=', 1) for p in text.split(',')[1:])['tx']


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
