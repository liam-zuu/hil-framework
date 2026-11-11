"""
gain_send.py: send one gain set to the ESP32 (car or HIL bench DUT) over WiFi, no USB, no reflash.

L90, Liam 12/11/2026 (CODE session): "coi như từ đầu mình thiết kế nạp gain qua wifi đi".
Design: 03_CODE/docs/2026-11-12_Thiet_ke_doi_gain_qua_WiFi.md of the thesis repo.
Firmware side: AGV_MECANUM src/controller/gain_set.c (packet, ranges, apply) and src/gain_rx/
(UDP port 5014). The ESP32 applies the set only while the motors are off (SWA off); with the motors
on it answers err=motor_on and changes nothing.

  python tools/gain_send.py --ip 192.168.1.50 --ctrl PID --id 12 \
      --gains kp_xy=1.2,ki_xy=0.4,kp_wz=1.0,ki_wz=0.05,pos_kp=6,pos_ki=3,k_psi=1
  python tools/gain_send.py --ip 192.168.1.50 --file 05_DATA/.../gain_A-PID_03.json --id 3
  python tools/gain_send.py --ip 192.168.1.50 --query        (read the running set, change nothing)

Gains: 7 per controller, the fwsim names (fwsim/tune_ga.py VARS_BY_MODE):
  PID : kp_xy, ki_xy, kp_wz, ki_wz, pos_kp, pos_ki, k_psi
  ADRC: omega_c_xy, omega_o_xy, omega_c_wz, omega_o_wz, pos_kp, pos_ki, k_psi
  Aliases accepted (tools/rq3_log.py names): wc_xy, wo_xy, wc_wz, wo_wz.
  Fixed on the ESP32 as in fwsim: PID kd = 0 and the limits of pid.c; LADRC b0, out_limit of adrc.c.
--file: JSON {"ctrl": "PID", "id": 3, "gains": {name: value, ...}} or a flat {name: value} (then
--ctrl is needed); --id / --ctrl on the command line must agree with the file when both are given.
Every value is rounded to float32 and sent as %.9g, so the ESP32 gets exactly those bits.

The ESP32 answers "#gain_ack,req_id=..,ok=..,err=..,sha=..,env=..,id=..,applied_n=..,ctrl=..,<7 gains>"
with the gains RUNNING after the request (read back from the controller). The set counts as applied
only when ok=1, the running id is the one sent, and all 7 running values equal the sent float32
values bit for bit. No answer in --timeout s: the same packet is sent again, at most --tries times
(the ESP32 applies the same id again; harmless). err=motor_on, ctrl, range, format: no retry.
The running set is not kept over a reset: after a reboot the ESP32 runs the built-in gains (id=0).
Each run log has '#gain,id=..' lines at the start of the run (agv_logger.py events file).

Exit code / last line '#ket_qua,ma=N': 0 applied and verified; 2 rejected by the ESP32 (err=..);
3 no answer; 4 answer does not match what was sent; 1 bad arguments.
"""
import argparse
import json
import math
import socket
import struct
import sys
import time

PORT = 5014
NAMES = {'PID': ['kp_xy', 'ki_xy', 'kp_wz', 'ki_wz', 'pos_kp', 'pos_ki', 'k_psi'],
         'ADRC': ['omega_c_xy', 'omega_o_xy', 'omega_c_wz', 'omega_o_wz', 'pos_kp', 'pos_ki', 'k_psi']}
ALIASES = {'wc_xy': 'omega_c_xy', 'wo_xy': 'omega_o_xy', 'wc_wz': 'omega_c_wz', 'wo_wz': 'omega_o_wz'}
ID_MAX = 2147483647


class UsageError(Exception):
    pass


def f32(x: float) -> float:
    """Nearest float32, as a Python float."""
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def fmt(x: float) -> str:
    """%.9g of the float32 value: strtof on the ESP32 gives back the same bits."""
    return '%.9g' % f32(x)


def parse_gains_arg(text: str) -> dict:
    out = {}
    for part in (text or '').split(','):
        part = part.strip()
        if not part:
            continue
        k, sep, v = part.partition('=')
        if not sep:
            raise UsageError(f'--gains: "{part}" is not name=value')
        out[k.strip()] = v.strip()
    return out


def normalize(ctrl: str, gains: dict) -> dict:
    """Map aliases, check the 7 names of ctrl, values finite numbers; returns {name: float32}."""
    if ctrl not in NAMES:
        raise UsageError(f'controller {ctrl!r}: PID or ADRC')
    g = {}
    for k, v in gains.items():
        name = ALIASES.get(k, k)
        if name in g:
            raise UsageError(f'gain {name} given twice')
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise UsageError(f'gain {k}={v!r} is not a number')
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


def load_file(path: str):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise UsageError(f'{path}: JSON object expected')
    if 'gains' in d:
        return d.get('ctrl'), d.get('id'), d['gains']
    return None, None, d


def build_packet(set_id: int, ctrl: str, gains: dict) -> str:
    return (f'#gain_set,v=1,id={set_id},ctrl={ctrl},'
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
    ctrl = ack.get('ctrl')
    if ctrl not in NAMES:
        raise ValueError(f'ack ctrl={ctrl!r}')
    return {n: float(ack[n]) for n in NAMES[ctrl]}


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


def exchange(ip: str, port: int, packet: str, req_id: int, timeout: float, tries: int, log=print):
    """Send packet, wait for the #gain_ack with req_id; resend on silence. Returns (ack | None, n_sent)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(0.05)
        data = (packet + '\n').encode('ascii')
        for k in range(1, tries + 1):
            sock.sendto(data, (ip, port))
            t_end = time.monotonic() + timeout
            while time.monotonic() < t_end:
                try:
                    raw, src = sock.recvfrom(2048)
                except socket.timeout:
                    continue
                if src[0] != ip:
                    continue
                try:
                    ack = parse_ack(raw.decode('ascii', 'replace'))
                except ValueError as e:
                    log(f'# ignored answer from {src[0]}: {e}')
                    continue
                if ack.get('req_id') != str(req_id):
                    continue                         # late answer to an earlier packet
                if ack.get('err') in ('busy', 'timeout') and k < tries:
                    log(f'# try {k}: ESP32 err={ack.get("err")}, sending again')
                    break
                return ack, k
            else:
                log(f'# try {k}: no answer in {timeout} s')
        return None, tries
    finally:
        sock.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', required=True, help='ESP32 address (its UART prints it at boot; one board only)')
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--ctrl', choices=sorted(NAMES))
    ap.add_argument('--id', type=int, help=f'gain set id 1..{ID_MAX} (written into every run log as #gain,id=)')
    ap.add_argument('--gains', help='name=value,... (7 names of the controller)')
    ap.add_argument('--file', help='JSON gain file (frozen gain_*.json)')
    ap.add_argument('--query', action='store_true', help='only read the running set (sends an invalid id 0 '
                    'request, which the ESP32 rejects with err=format and answers with the running set)')
    ap.add_argument('--timeout', type=float, default=1.0)
    ap.add_argument('--tries', type=int, default=3)
    a = ap.parse_args(argv)

    if a.query:
        ack, _ = exchange(a.ip, a.port, '#gain_set,v=1,id=0', 0, a.timeout, a.tries)
        if ack is None:
            print('#ket_qua,ma=3')
            return 3
        print('#gain_running,' + ','.join(f'{k}={v}' for k, v in ack.items() if k not in ('req_id', 'ok', 'err')))
        print('#ket_qua,ma=0')
        return 0

    try:
        ctrl, set_id, raw = a.ctrl, a.id, {}
        if a.file:
            f_ctrl, f_id, raw = load_file(a.file)
            if f_ctrl is not None and ctrl is not None and f_ctrl != ctrl:
                raise UsageError(f'--ctrl {ctrl} but the file says {f_ctrl}')
            if f_id is not None and set_id is not None and int(f_id) != set_id:
                raise UsageError(f'--id {set_id} but the file says {f_id}')
            ctrl = ctrl or f_ctrl
            set_id = set_id if set_id is not None else (int(f_id) if f_id is not None else None)
            if a.gains:
                raise UsageError('give --file or --gains, not both')
        else:
            raw = parse_gains_arg(a.gains or '')
        if ctrl is None:
            raise UsageError('--ctrl PID|ADRC needed (or "ctrl" in the file)')
        if set_id is None or not 1 <= set_id <= ID_MAX:
            raise UsageError(f'--id 1..{ID_MAX} needed (or "id" in the file)')
        gains = normalize(ctrl, raw)
    except (UsageError, OSError, json.JSONDecodeError) as e:
        print(f'error: {e}', file=sys.stderr)
        print('#ket_qua,ma=1')
        return 1

    packet = build_packet(set_id, ctrl, gains)
    print(f'# send {a.ip}:{a.port} {packet}')
    ack, n = exchange(a.ip, a.port, packet, set_id, a.timeout, a.tries)
    if ack is None:
        print(f'# no answer from {a.ip}:{a.port} after {n} tries (ESP32 on? same network? right IP?)')
        print('#ket_qua,ma=3')
        return 3
    print('# ack ' + ','.join(f'{k}={v}' for k, v in ack.items()))
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
    print(f"# applied: id={set_id} ctrl={ctrl} applied_n={ack.get('applied_n')} fw={ack.get('sha')} env={ack.get('env')}")
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
