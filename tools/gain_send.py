"""
gain_send.py: send one gain set to the ESP32 (car or HIL bench DUT) over WiFi, no USB, no reflash.

L90, Liam 12/11/2026 (CODE session): "coi như từ đầu mình thiết kế nạp gain qua wifi đi".
Design: 03_CODE/docs/2026-11-12_Thiet_ke_doi_gain_qua_WiFi.md of the thesis repo.
Firmware side: AGV_MECANUM src/controller/gain_set.c (packet, ranges, apply) and src/gain_rx/
(UDP port 5014). The ESP32 applies the set only while the motors are off (SWA off); with the motors
on, or for a packet it received before it saw the motors off, it answers err=motor_on and changes
nothing.

  python tools/gain_send.py --ip 192.168.1.50 --ctrl PID --id 12 \
      --gains kp_xy=1.2,ki_xy=0.4,kp_wz=1.0,ki_wz=0.05,pos_kp=6,pos_ki=3,k_psi=1
  python tools/gain_send.py --ip 192.168.1.50 --file 05_DATA/.../gain_A-PID_03.json --id 3
  python tools/gain_send.py --ip 192.168.1.50 --query        (read the running set, change nothing)
  add --mac aa:bb:cc:dd:ee:ff to refuse an answer from any other board (the MAC is in every answer)

Gains: 7 per controller, the fwsim names (fwsim/tune_ga.py VARS_BY_MODE):
  PID : kp_xy, ki_xy, kp_wz, ki_wz, pos_kp, pos_ki, k_psi
  ADRC: omega_c_xy, omega_o_xy, omega_c_wz, omega_o_wz, pos_kp, pos_ki, k_psi
  Aliases accepted (tools/rq3_log.py names): wc_xy, wo_xy, wc_wz, wo_wz.
  Fixed on the ESP32 as in fwsim: PID kd = 0 and the limits of pid.c; LADRC b0, out_limit of adrc.c.
--file: JSON {"ctrl": "PID", "id": 3, "gains": {name: value, ...}} or a flat {name: value} (then
--ctrl is needed); --id / --ctrl on the command line must agree with the file when both are given.
Every value is rounded to float32 and sent as %.9g, so the ESP32 gets exactly those bits.

Packets (protocol after Codex round 1, 12/11): '#gain_set,v=1,tx=<nonce>,id=..,ctrl=..,<7 gains>' and
'#gain_get,v=1,tx=<nonce>'. tx is a new random number for every packet sent; only an answer from the
ESP32's IP AND port 5014 with the same tx counts. The answer
'#gain_ack,tx=..,req_id=..,ok=..,err=..,sha=..,env=..,mac=..,boot=..,id=..,applied_n=..,ctrl=..,<7 gains>'
carries the gains RUNNING after the request (snapshot taken by the ESP32's control task). The set
counts as applied only when ok=1, the running id is the one sent, and all 7 running values equal the
sent float32 values bit for bit. No answer, or err=busy / timeout: sent again with a new tx, at most
--tries times (the ESP32 may apply the same id twice; it only resets the controller state again, with
the motors off). err=motor_on, ctrl, range, format: no retry.
The running set is not kept over a reset: after a reboot the ESP32 runs the built-in gains (id=0) and
'boot' changes. Each run log has '#gain,id=..,first_seq=..' lines at the start of the run.

Exit code / last line '#ket_qua,ma=N': 0 applied and verified (or --query read a complete set);
2 rejected by the ESP32 (err=..); 3 no answer; 4 answer does not match what was sent (or another
board's MAC); 1 bad arguments.
"""
import argparse
import json
import math
import secrets
import socket
import struct
import sys
import time

PORT = 5014
NAMES = {'PID': ['kp_xy', 'ki_xy', 'kp_wz', 'ki_wz', 'pos_kp', 'pos_ki', 'k_psi'],
         'ADRC': ['omega_c_xy', 'omega_o_xy', 'omega_c_wz', 'omega_o_wz', 'pos_kp', 'pos_ki', 'k_psi']}
ALIASES = {'wc_xy': 'omega_c_xy', 'wo_xy': 'omega_o_xy', 'wc_wz': 'omega_c_wz', 'wo_wz': 'omega_o_wz'}
ID_MAX = 2147483647
RETRY_ERRS = ('busy', 'timeout')


class UsageError(Exception):
    pass


def f32(x: float) -> float:
    """Nearest float32, as a Python float; UsageError when it does not fit (e.g. 1e100)."""
    try:
        v = struct.unpack('<f', struct.pack('<f', float(x)))[0]
    except (OverflowError, struct.error, TypeError, ValueError):
        raise UsageError(f'value {x!r} does not fit a float32')
    if not math.isfinite(v):
        raise UsageError(f'value {x!r} is not finite as a float32')
    return v


def fmt(x: float) -> str:
    """%.9g of the float32 value: strtof on the ESP32 gives back the same bits."""
    return '%.9g' % f32(x)


def new_tx() -> int:
    return secrets.randbelow(2**32 - 1) + 1          # 1 .. 2^32-1


def parse_gains_arg(text: str) -> dict:
    out = {}
    for part in (text or '').split(','):
        part = part.strip()
        if not part:
            continue
        k, sep, v = part.partition('=')
        if not sep:
            raise UsageError(f'--gains: "{part}" is not name=value')
        k = k.strip()
        if k in out:
            raise UsageError(f'--gains: {k} given twice')
        out[k] = v.strip()
    return out


def normalize(ctrl: str, gains: dict) -> dict:
    """Map aliases, check the 7 names of ctrl, values finite numbers; returns {name: float32}."""
    if ctrl not in NAMES:
        raise UsageError(f'controller {ctrl!r}: PID or ADRC')
    if not isinstance(gains, dict):
        raise UsageError('gains must be an object of name: value')
    g = {}
    for k, v in gains.items():
        name = ALIASES.get(k, k)
        if name in g:
            raise UsageError(f'gain {name} given twice')
        if isinstance(v, bool):
            raise UsageError(f'gain {k}={v!r} is not a number')
        try:
            x = float(v)
        except (TypeError, ValueError, OverflowError):          # OverflowError: JSON integer like 10**400
            raise UsageError(f'gain {k}={v!r} is not a number that fits a float')
        if not math.isfinite(x):
            raise UsageError(f'gain {k}={v!r} is not finite')
        g[name] = f32(x)
    want = NAMES[ctrl]
    missing = [n for n in want if n not in g]
    extra = [n for n in g if n not in want]
    if missing or extra:
        raise UsageError(f'{ctrl} needs exactly {", ".join(want)}'
                         + (f'; missing {", ".join(missing)}' if missing else '')
                         + (f'; not for {ctrl}: {", ".join(extra)}' if extra else ''))
    return {n: g[n] for n in want}


def _no_dup_keys(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise UsageError(f'JSON key {k!r} given twice')
        d[k] = v
    return d


def check_id(v, where):
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= ID_MAX:
        raise UsageError(f'{where}: id must be an integer 1..{ID_MAX}, got {v!r}')
    return v


def load_file(path: str):
    with open(path, encoding='utf-8') as f:
        d = json.load(f, object_pairs_hook=_no_dup_keys)
    if not isinstance(d, dict):
        raise UsageError(f'{path}: JSON object expected')
    if 'gains' in d:
        if not isinstance(d['gains'], dict):
            raise UsageError(f'{path}: "gains" must be an object')
        # a key "id" that is present must be a valid id, even null (Codex round 2, L90-11)
        return d.get('ctrl'), (check_id(d['id'], path) if 'id' in d else None), d['gains']
    return None, None, d


def build_packet(set_id: int, ctrl: str, gains: dict, tx: int) -> str:
    return (f'#gain_set,v=1,tx={tx},id={set_id},ctrl={ctrl},'
            + ','.join(f'{n}={fmt(gains[n])}' for n in NAMES[ctrl]))


def parse_ack(text: str) -> dict:
    line = text.strip().splitlines()[0] if text.strip() else ''
    parts = line.split(',')
    if not parts or parts[0] != '#gain_ack':
        raise ValueError(f'not a #gain_ack line: {line[:80]!r}')
    out = {}
    for p in parts[1:]:
        k, sep, v = p.partition('=')
        if not sep or k in out:
            raise ValueError(f'bad field {p!r} in {line[:80]!r}')
        out[k] = v
    return out


def running_gains(ack: dict) -> dict:
    """The 7 running gains of the ack as floats; ValueError if missing or not finite."""
    ctrl = ack.get('ctrl')
    if ctrl not in NAMES:
        raise ValueError(f'ack ctrl={ctrl!r}')
    out = {}
    for n in NAMES[ctrl]:
        if n not in ack:
            raise ValueError(f'ack has no {n}')
        x = float(ack[n])
        if not math.isfinite(x):
            raise ValueError(f'ack {n}={ack[n]!r} not finite')
        out[n] = x
    return out


def check_running(ack: dict) -> list:
    """Problems with the running-set part of an ack (id, ctrl, 7 gains); [] = complete."""
    errs = []
    if not ack.get('id', '').isdigit():
        errs.append(f"running id={ack.get('id')!r}")
    try:
        running_gains(ack)
    except (KeyError, ValueError) as e:
        errs.append(str(e))
    return errs


def verify(ack: dict, set_id: int, ctrl: str, gains: dict) -> list:
    """Differences between the ack and the request; [] = applied as sent."""
    errs = []
    if ack.get('ok') != '1':
        errs.append(f"ok={ack.get('ok')} err={ack.get('err')}")
    if ack.get('req_id') != str(set_id):
        errs.append(f"req_id={ack.get('req_id')} (sent {set_id})")
    if ack.get('id') != str(set_id):
        errs.append(f"running id={ack.get('id')} (sent {set_id})")
    if ack.get('ctrl') != ctrl:
        errs.append(f"running ctrl={ack.get('ctrl')} (sent {ctrl})")
        return errs
    try:
        run = running_gains(ack)
    except (KeyError, ValueError) as e:
        return errs + [f'gains in ack: {e}']
    for n in NAMES[ctrl]:
        if struct.pack('<f', run[n]) != struct.pack('<f', gains[n]):
            errs.append(f'{n}: running {run[n]!r}, sent {gains[n]!r}')
    return errs


def exchange(ip: str, port: int, make_packet, timeout: float, tries: int, log=print):
    """make_packet(tx) -> text. Each try uses a new tx; only an answer from (ip, port) with that tx is
    taken. busy / timeout answers and silence lead to the next try. Returns (ack | None, n_sent)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(0.05)
        for k in range(1, tries + 1):
            tx = new_tx()
            sock.sendto((make_packet(tx) + '\n').encode('ascii'), (ip, port))
            t_end = time.monotonic() + timeout
            retry = False
            while time.monotonic() < t_end:
                try:
                    raw, src = sock.recvfrom(2048)
                except socket.timeout:
                    continue
                except ConnectionResetError:          # Windows: ICMP port unreachable from an earlier send
                    continue
                if src[0] != ip or src[1] != port:
                    continue
                try:
                    ack = parse_ack(raw.decode('ascii', 'replace'))
                except ValueError as e:
                    log(f'# ignored answer from {src[0]}:{src[1]}: {e}')
                    continue
                if ack.get('tx') != str(tx):
                    continue                         # late answer to an earlier packet
                if ack.get('err') in RETRY_ERRS and k < tries:
                    log(f'# try {k}: ESP32 err={ack.get("err")}, sending again')
                    retry = True
                    break
                return ack, k
            if not retry:
                log(f'# try {k}: no answer in {timeout} s')
        return None, tries
    finally:
        sock.close()


def board_mismatch(ack: dict, want_mac):
    if want_mac and ack.get('mac', '').lower() != want_mac.lower():
        return f"answer from mac={ack.get('mac')}, expected {want_mac}"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', required=True, help='ESP32 address (its UART prints it at boot; one board only)')
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--mac', help='expected MAC of the ESP32 (aa:bb:cc:dd:ee:ff); other boards are refused')
    ap.add_argument('--ctrl', choices=sorted(NAMES))
    ap.add_argument('--id', type=int, help=f'gain set id 1..{ID_MAX} (written into every run log as #gain,id=)')
    ap.add_argument('--gains', help='name=value,... (7 names of the controller)')
    ap.add_argument('--file', help='JSON gain file (frozen gain_*.json)')
    ap.add_argument('--query', action='store_true', help='only read the running set (#gain_get)')
    ap.add_argument('--timeout', type=float, default=1.0)
    ap.add_argument('--tries', type=int, default=3)
    a = ap.parse_args(argv)

    if a.query:
        ack, _ = exchange(a.ip, a.port, lambda tx: f'#gain_get,v=1,tx={tx}', a.timeout, a.tries)
        if ack is None:
            print('#ket_qua,ma=3')
            return 3
        bad = board_mismatch(ack, a.mac)
        if bad or ack.get('ok') != '1':
            print(f"# {bad or 'rejected: err=' + str(ack.get('err'))}")
            print('#ket_qua,ma=4' if bad else '#ket_qua,ma=2')
            return 4 if bad else 2
        errs = check_running(ack)
        if errs:
            print('# INCOMPLETE answer: ' + '; '.join(errs))
            print('#ket_qua,ma=4')
            return 4
        print('#gain_running,' + ','.join(f'{k}={v}' for k, v in ack.items() if k not in ('tx', 'req_id', 'ok', 'err')))
        print('#ket_qua,ma=0')
        return 0

    try:
        ctrl, set_id, raw = a.ctrl, a.id, {}
        if a.file:
            if a.gains:
                raise UsageError('give --file or --gains, not both')
            f_ctrl, f_id, raw = load_file(a.file)
            if f_ctrl is not None and ctrl is not None and f_ctrl != ctrl:
                raise UsageError(f'--ctrl {ctrl} but the file says {f_ctrl}')
            if f_id is not None and set_id is not None and f_id != set_id:
                raise UsageError(f'--id {set_id} but the file says {f_id}')
            ctrl = ctrl or f_ctrl
            set_id = set_id if set_id is not None else f_id
        else:
            raw = parse_gains_arg(a.gains or '')
        if ctrl is None:
            raise UsageError('--ctrl PID|ADRC needed (or "ctrl" in the file)')
        if set_id is None:
            raise UsageError(f'--id 1..{ID_MAX} needed (or "id" in the file)')
        check_id(set_id, '--id')
        gains = normalize(ctrl, raw)
    except (UsageError, OSError, json.JSONDecodeError) as e:
        print(f'error: {e}', file=sys.stderr)
        print('#ket_qua,ma=1')
        return 1

    print(f'# send {a.ip}:{a.port} ' + build_packet(set_id, ctrl, gains, 0).replace(',tx=0,', ',tx=<new each try>,'))
    ack, n = exchange(a.ip, a.port, lambda tx: build_packet(set_id, ctrl, gains, tx), a.timeout, a.tries)
    if ack is None:
        print(f'# no answer from {a.ip}:{a.port} after {n} tries (ESP32 on? same network? right IP?)')
        print('#ket_qua,ma=3')
        return 3
    print('# ack ' + ','.join(f'{k}={v}' for k, v in ack.items()))
    bad = board_mismatch(ack, a.mac)
    if bad:
        print('# WRONG BOARD: ' + bad)
        print('#ket_qua,ma=4')
        return 4
    if ack.get('ok') != '1':
        print(f"# rejected: err={ack.get('err')}"
              + (' (turn SWA off first: gains change only with the motors off)' if ack.get('err') == 'motor_on' else ''))
        print('#ket_qua,ma=2')
        return 2
    errs = verify(ack, set_id, ctrl, gains)
    if errs:
        print('# MISMATCH: ' + '; '.join(errs))
        print('#ket_qua,ma=4')
        return 4
    print(f"# applied: id={set_id} ctrl={ctrl} applied_n={ack.get('applied_n')} mac={ack.get('mac')} "
          f"boot={ack.get('boot')} fw={ack.get('sha')} env={ack.get('env')}")
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
