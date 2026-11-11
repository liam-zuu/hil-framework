"""Tests of tools/gain_send.py (no hardware). Run: python -m pytest tools/tests/test_gain_send.py -q

FakeEsp32 is a small UDP server on 127.0.0.1 that answers like AGV_MECANUM src/gain_rx/gain_rx.c
('#gain_ack,req_id=..,ok=..,err=..,sha=..,env=..,id=..,applied_n=..,ctrl=..,<7 gains>' with the
gains running after the request), with switches for the motors on, silence, a wrong value and a
stale answer. It is a copy of the protocol, not of the firmware: gain_set.c has its own host test
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


class FakeEsp32:
    def __init__(self, mode='ok', ctrl='PID'):
        self.mode, self.ctrl = mode, ctrl
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', 0))
        self.sock.settimeout(0.05)
        self.port = self.sock.getsockname()[1]
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

    def _ack(self, src, req_id, err):
        g = self.running['gains']
        line = (f'#gain_ack,req_id={req_id},ok={int(err == "")},err={err},sha=abc1234,env=test,'
                f'id={self.running["id"]},applied_n={self.applied_n},ctrl={self.ctrl},'
                + ','.join(f'{n}={"%.9g" % g[n]}' for n in gs.NAMES[self.ctrl]) + '\n')
        self.sock.sendto(line.encode(), src)

    def _loop(self):
        while not self._stop:
            try:
                raw, src = self.sock.recvfrom(1024)
            except (socket.timeout, OSError):
                continue
            text = raw.decode().strip()
            self.received.append(text)
            parts = dict(p.split('=', 1) for p in text.split(',')[1:] if '=' in p)
            req_id = parts.get('id', '0')
            if self.mode == 'silent' or (self.mode == 'silent_first' and len(self.received) == 1):
                continue
            if self.mode == 'stale_first':
                self._ack(src, '999', '')                   # answer to some earlier packet
            if req_id == '0' or parts.get('v') != '1':
                self._ack(src, '0', 'format')
                continue
            if self.mode == 'motor_on':
                self._ack(src, req_id, 'motor_on')
                continue
            if parts.get('ctrl') != self.ctrl:
                self._ack(src, req_id, 'ctrl')
                continue
            gains = {n: struct.unpack('<f', struct.pack('<f', float(parts[n])))[0] for n in gs.NAMES[self.ctrl]}
            if self.mode == 'wrong_value':
                gains['k_psi'] = gs.f32(gains['k_psi'] + 0.5)
            self.running = dict(id=int(req_id), gains=gains)
            self.applied_n += 1
            self._ack(src, req_id, '')


@pytest.fixture
def fake(request):
    f = FakeEsp32(**getattr(request, 'param', {}))
    yield f
    f.close()


def args(fake, *extra):
    return ['--ip', '127.0.0.1', '--port', str(fake.port), '--timeout', '0.3', '--tries', '3', *extra]


def gains_arg(d):
    return ','.join(f'{k}={v}' for k, v in d.items())


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


def test_packet_text():
    p = gs.build_packet(12, 'PID', gs.normalize('PID', PID))
    assert p == ('#gain_set,v=1,id=12,ctrl=PID,kp_xy=1.20000005,ki_xy=0.400000006,kp_wz=1,ki_wz=0.0500000007,'
                 'pos_kp=6,pos_ki=3,k_psi=1')


def test_applied_and_verified(fake, capsys):
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '12', '--gains', gains_arg(PID))) == 0
    out = capsys.readouterr().out
    assert out.rstrip().endswith('#ket_qua,ma=0') and 'applied: id=12' in out
    assert fake.running['id'] == 12 and len(fake.received) == 1


def test_resend_when_silent_first(fake, capsys):
    fake.mode = 'silent_first'
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '5', '--gains', gains_arg(PID))) == 0
    assert len(fake.received) == 2 and fake.received[0] == fake.received[1]


@pytest.mark.parametrize('fake', [dict(mode='silent')], indirect=True)
def test_no_answer_is_3(fake, capsys):
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '5', '--gains', gains_arg(PID))) == 3
    assert len(fake.received) == 3
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=3')


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


def test_bad_arguments_send_nothing(fake, capsys):
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '0', '--gains', gains_arg(PID))) == 1
    assert gs.main(args(fake, '--ctrl', 'PID', '--gains', gains_arg(PID))) == 1
    assert gs.main(args(fake, '--ctrl', 'PID', '--id', '3', '--gains', 'kp_xy=1')) == 1
    assert fake.received == []


def test_query_reads_running_set(fake, capsys):
    assert gs.main(args(fake, '--query')) == 0
    out = capsys.readouterr().out
    assert '#gain_running,' in out and 'id=0' in out and 'kp_xy=1.5' in out
    assert fake.running['id'] == 0
