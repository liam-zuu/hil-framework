"""
gain_send.py: send one gain set to the ESP32 (car or HIL bench DUT) over WiFi, no USB, no reflash.

L90, Liam 12/11/2026 (CODE session): "coi như từ đầu mình thiết kế nạp gain qua wifi đi".
Design: 03_CODE/docs/2026-11-12_Thiet_ke_doi_gain_qua_WiFi.md of the thesis repo.
Firmware side: AGV_MECANUM src/controller/gain_set.c (packet, ranges, apply) and src/gain_rx/
(UDP port 5014). The ESP32 applies the set only while the motors are off (SWA off); with the motors
on, or for a packet it received before it saw the motors off, it answers err=motor_on and changes
nothing.

  python tools/gain_send.py --ip 192.168.1.50 --ctrl PID --id 12 \
      --gains kp_xy=1.2,ki_xy=0.4,kp_wz=1.0,ki_wz=0.05,pos_kp=6,pos_ki=3,k_psi=1 \
      --nguon-gain-tay "manual tuning on the bench"
  python tools/gain_send.py --ip 192.168.1.50 --file 05_DATA/.../gain_A-PID_03.json --id 3 \
      --nguon-gain-tay "frozen gain file of <campaign>"
  python tools/gain_send.py --ip 192.168.1.50 --ctrl PID --id 3 --gains <7 gains> --s04-run <S04 run>
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
Before sending, the set is checked against the PC gain table tools/gain_table.py (range; ADRC
omega_c + 2*omega_o < 200 on each axis, xy and wz); a violation sends nothing (exit 1, like bad
arguments); warnings (the pos_kp / pos_ki wedge) are printed and the set is still sent.
Every send declares where its gains come from (C3-8): a campaign gain set with
--s04-run <S04 run folder>, any other send with --nguon-gain-tay "<where the gains come from>"
(logged on the '#nguon_gain,loai=tay' line); both flags or a blank description: usage error
(exit 1); no source at all: refused, nothing sent (exit 5). --query needs no source.
--s04-run <S04 run folder>: before the send, the S04 section 6 hand-over check runs
(s04_hand_over): post_checks/notes.json cho_liam must be a list of objects with str 'controller',
str 'check' and list 'reasons' (a broken record refuses, never overridable), tool_check must be
exactly false (a tool-check run is never handed over), post_trang_thai must be 'hoan_tat' (the post
checks ran to their end; notes.json carries selected_sha256 / selected_sha256_lf of the selected.csv
bytes the post hashed — the file read for this send must hash to one of them, else it changed after
post and the send is refused), and selected.csv must hold exactly one send_to_HIL row of this
controller with feasible 'true' whose 7 gains are the sent ones bit for
bit as float32; a non-empty cho_liam refuses the send (exit 5) unless --liam-da-quyet "<reason>"
overrides it (every item is listed, the reason is logged on the '#ban_giao_s04' line, which also
carries selected_sha256 of the sent file).
--liam-da-quyet without --s04-run, a blank reason, or a reason while cho_liam is empty: usage error
(exit 1). Right before the '# send' line the spectral radius of the one-axis sub-model
(tools/mo_hinh_con.ban_kinh_pho_va_trang_thai) is printed as an indicator: report only, it NEVER
blocks the send ('(chi bao, khong chan)'; a '# warning: ban_kinh_pho >= 1' line when unstable, a
'not computed' line when it fails).

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
board's MAC); 5 refused by the S04 hand-over check (--s04-run) or no gain source declared (nothing
sent); 1 bad arguments or refused by the PC gain table.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import secrets
import socket
import struct
import sys
import time

if __package__:
    from . import gain_table                 # run as -m tools.gain_send (tools is a namespace package)
else:
    import gain_table                        # run as tools/gain_send.py

PORT = 5014
NAMES = {'PID': ['kp_xy', 'ki_xy', 'kp_wz', 'ki_wz', 'pos_kp', 'pos_ki', 'k_psi'],
         'ADRC': ['omega_c_xy', 'omega_o_xy', 'omega_c_wz', 'omega_o_wz', 'pos_kp', 'pos_ki', 'k_psi']}
ALIASES = {'wc_xy': 'omega_c_xy', 'wo_xy': 'omega_o_xy', 'wc_wz': 'omega_c_wz', 'wo_wz': 'omega_o_wz'}
ID_MAX = 2147483647
RETRY_ERRS = ('busy', 'timeout')
SHA_RE = re.compile(r'[0-9a-f]{64}')
POST_DONE = 'hoan_tat'                    # post_trang_thai of a notes.json whose post checks ran to the end


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


def s04_hand_over(run_dir, ctrl, gains, reason):
    """S04 section 6 hand-over check before one gain set goes to the board. Reads
    <run_dir>/post_checks/notes.json and <run_dir>/selected.csv (the latter ONCE, as bytes: hashed
    and parsed from the same bytes, C3-8). Returns (code, out_lines, err_lines):
      5  refused, nothing may be sent: notes.json missing / not JSON / not an object / 'cho_liam'
         missing or not a list; a cho_liam item not an object with str 'controller', str 'check',
         list 'reasons' (a broken hand-over record, C3-47 — never overridable); tool_check not
         exactly false (a tool-check run is never handed over); post_trang_thai not 'hoan_tat' (the
         post checks did not run to their end); selected_sha256 / selected_sha256_lf missing, not a
         SHA-256, or not matching the selected.csv bytes (C3-8: changed after post); selected.csv
         missing, not UTF-8 or without exactly one row kind=send_to_HIL of controller ctrl with
         feasible 'true'; the 7 sent gains differ bit for bit as float32 from that row; or cho_liam
         is non-empty and reason is None (ANY item blocks, whatever its controller; each item is
         listed: controller, check, number of reasons)
      1  usage error: a reason is given but cho_liam is empty (nothing to override)
      0  send; out_lines: '#ban_giao_s04,run=<json abs path>,tool_check=<json>,cho_liam=<n>,
         liam_da_quyet=<json reason or "">,selected_sha256=<hex of the raw bytes>' plus one
         '# cho_liam: ...' line per item overridden.
    All text is ASCII (free text through json.dumps ensure_ascii=True)."""
    p = os.path.join(run_dir, 'post_checks', 'notes.json')
    try:
        with open(p, encoding='utf-8') as f:
            notes = json.load(f)
    except FileNotFoundError:
        return 5, [], [f'# error: {p} missing: the S04 hand-over check needs post_checks/notes.json '
                       f'(S04 section 6)']
    except (OSError, ValueError) as e:
        return 5, [], [f'# error: {p}: {e}']
    if not isinstance(notes, dict) or not isinstance(notes.get('cho_liam'), list):
        return 5, [], [f"# error: {p}: 'cho_liam' missing or not a list: not the notes.json of an S04 "
                       f'post check (S04 section 6)']
    cho = notes['cho_liam']
    for i, it in enumerate(cho):                # C3-47: a broken hand-over record can never be overridden
        if not isinstance(it, dict) or not isinstance(it.get('controller'), str) \
                or not isinstance(it.get('check'), str) or not isinstance(it.get('reasons'), list):
            return 5, [], [f'# error: {p}: cho_liam item {i} is not an object with str "controller", str '
                           f'"check" and list "reasons": a broken hand-over record (C3-47)']
    if notes.get('tool_check') is not False:    # C3-8: a tool-check S04 run (or no tool_check: false) is never
        return 5, [], [f'# error: {p}: tool_check is {notes.get("tool_check")!r}, not false: a tool-check S04 '
                       f'run is never handed over to the board (S04 section 6; C3-8)']
    if notes.get('post_trang_thai') != POST_DONE:   # C3-8: the post phase did not run to its end
        return 5, [], [f'# error: {p}: post_trang_thai is {notes.get("post_trang_thai")!r}, not '
                       f'{POST_DONE!r}: the post checks did not run to their end, no hand-over (C3-8)']
    sp = os.path.join(run_dir, 'selected.csv')
    try:
        with open(sp, 'rb') as f:
            raw = f.read()                      # C3-8: read ONCE; hashed and parsed from the same bytes
    except OSError as e:
        return 5, [], [f'# error: {sp}: {e}']
    want, want_lf = notes.get('selected_sha256'), notes.get('selected_sha256_lf')
    if not (isinstance(want, str) and SHA_RE.fullmatch(want)) \
            or not (isinstance(want_lf, str) and SHA_RE.fullmatch(want_lf)):
        return 5, [], [f'# error: {p}: selected_sha256 / selected_sha256_lf missing or not a SHA-256: not the '
                       f'notes.json of an S04 post check that ran to the end (C3-8)']
    if hashlib.sha256(raw).hexdigest() != want \
            and hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() != want_lf:
        return 5, [], [f'# error: {sp} is not the file the post checks hashed (changed after post): '
                       f'selected_sha256 {want[:12]}... of {p}, file {hashlib.sha256(raw).hexdigest()[:12]}... '
                       f'(C3-8); run the post checks again, nothing sent']
    try:
        rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8'), newline='')))
    except (OSError, UnicodeDecodeError) as e:
        return 5, [], [f'# error: {sp}: {e}']
    sel = [r for r in rows if r.get('kind') == 'send_to_HIL' and r.get('controller') == ctrl
           and r.get('feasible') == 'true']
    if len(sel) != 1:
        return 5, [], [f"# error: {sp}: {len(sel)} rows kind=send_to_HIL controller={ctrl} "
                       f"feasible='true' (exactly one needed; S04 section 6)"]
    diff = []
    for n in NAMES.get(ctrl, ()):
        try:
            same = struct.pack('<f', float(sel[0][n])) == struct.pack('<f', f32(gains[n]))
        except (KeyError, TypeError, ValueError, OverflowError, UsageError):
            same = False
        if not same:
            diff.append(n)
    if diff:
        return 5, [], [f'# error: the gains sent differ bit for bit (float32) from the selected.csv '
                       f'send_to_HIL row ({", ".join(diff)}): send what S04 selected, nothing else']
    if cho and reason is None:
        err = ['# error: the S04 post checks have cho_liam items: Liam must see them before HIL; '
               'send only with --liam-da-quyet "<reason>" (S04 section 6)']
        for it in cho:
            err.append(f'# cho_liam: controller={it.get("controller")} check={it.get("check")} '
                       f'reasons={len(it.get("reasons") or [])}')
        return 5, [], err
    if reason is not None and not cho:
        return 1, [], ['# error: --liam-da-quyet given but cho_liam of ' + p + ' is empty: nothing to '
                       'override (usage error)']
    out = ['#ban_giao_s04,run=' + json.dumps(os.path.abspath(run_dir), ensure_ascii=True)
           + ',tool_check=' + json.dumps(notes.get('tool_check'))
           + f',cho_liam={len(cho)}'
           + ',liam_da_quyet=' + json.dumps(reason or '', ensure_ascii=True)
           + f',selected_sha256={hashlib.sha256(raw).hexdigest()}']
    for it in cho:
        out.append('# cho_liam: ' + json.dumps(it.get('controller'), ensure_ascii=True)
                   + ' ' + json.dumps(it.get('check'), ensure_ascii=True)
                   + f' ({len(it.get("reasons") or [])} reasons)')
    return 0, out, []


def spectral_lines(ctrl, gains):
    """S04 v6.8 section 6: the spectral-radius indicator line(s) printed before '# send'. Report only,
    NEVER blocks: mo_hinh_con is imported lazily inside the try (package form under tools/, plain import
    beside it); any exception gives one '# ban_kinh_pho: not computed (<reason>)' line and the send goes
    on. '# warning: ban_kinh_pho >= 1' when the sub-model is not stable."""
    try:
        if __package__:
            from . import mo_hinh_con           # run as -m tools.gain_send
        else:
            import mo_hinh_con                  # run as tools/gain_send.py
        m = mo_hinh_con.ban_kinh_pho_va_trang_thai(ctrl, gains)
    except Exception as e:                      # noqa: BLE001  the indicator may never stop the send
        return ['# ban_kinh_pho: not computed (' + json.dumps(f'{type(e).__name__}: {e}',
                                                            ensure_ascii=True) + ')']
    lines = [f"# ban_kinh_pho={m['ban_kinh_pho']!r} trang_thai={m['trang_thai']} "
             f"bien_trang_thai={','.join(m['bien_trang_thai'])} G={m['G']!r} tau={m['tau']!r} "
             f"(chi bao, khong chan)"]
    if m['trang_thai'] == mo_hinh_con.KHONG_ON_DINH:
        lines.append('# warning: ban_kinh_pho >= 1: the one-axis sub-model is not stable '
                     '(tools/mo_hinh_con.py; report only, the send goes on)')
    return lines


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
    ap.add_argument('--s04-run',
                    help='S04 run folder: the section 6 hand-over check (post_checks/notes.json cho_liam, '
                         'selected.csv send_to_HIL row) runs before the send; a non-empty cho_liam refuses '
                         '(exit 5) unless --liam-da-quyet overrides it')
    ap.add_argument('--nguon-gain-tay', metavar='MO_TA',
                    help='a MANUAL gain set: where these gains come from (logged on the #nguon_gain line); '
                         'needed for every send that is not a campaign S04 hand-over')
    ap.add_argument('--liam-da-quyet', metavar='LY_DO',
                    help='the reason Liam gave for sending although the S04 post checks list cho_liam items '
                         '(logged on the #ban_giao_s04 line)')
    ap.add_argument('--timeout', type=float, default=1.0)
    ap.add_argument('--tries', type=int, default=3)
    a = ap.parse_args(argv)

    if a.liam_da_quyet is not None and (a.s04_run is None or not a.liam_da_quyet.strip()):
        print('error: --liam-da-quyet needs --s04-run and a non-empty reason', file=sys.stderr)
        print('#ket_qua,ma=1')
        return 1

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

    # C3-8: every send declares where its gains come from (a query needs none)
    if a.s04_run is not None and a.nguon_gain_tay is not None:
        print('error: give --s04-run or --nguon-gain-tay, not both', file=sys.stderr)
        print('#ket_qua,ma=1')
        return 1
    if a.nguon_gain_tay is not None and not a.nguon_gain_tay.strip():
        print('error: --nguon-gain-tay needs a non-empty description of where the gains come from',
              file=sys.stderr)
        print('#ket_qua,ma=1')
        return 1
    if a.s04_run is None and a.nguon_gain_tay is None:
        print('error: no gain source declared: a campaign gain set needs --s04-run <S04 run folder> (S04 '
              'hand-over check); any other send needs --nguon-gain-tay "<where the gains come from>"; '
              'nothing sent', file=sys.stderr)
        print('#ket_qua,ma=5')
        return 5

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

    violations, warnings = gain_table.check(ctrl, gains)
    if violations:
        for v in violations:
            print(f'error: {v}', file=sys.stderr)
        print(f'error: refused by the PC gain table ({gain_table.TABLE_VERSION}); nothing sent',
              file=sys.stderr)
        print('#ket_qua,ma=1')
        return 1
    for w in warnings:
        print(f'# warning: {w}')

    if a.s04_run is not None:                    # S04 section 6: the hand-over check before anything is sent
        try:
            code, out_lines, err_lines = s04_hand_over(a.s04_run, ctrl, gains, a.liam_da_quyet)
        except Exception as e:                   # noqa: BLE001  C3-8: a broken record never sends and never
            code, out_lines, err_lines = 5, [], ['# error: the S04 hand-over check failed: ' +  # ends without #ket_qua
                                                 json.dumps(f'{type(e).__name__}: {e}', ensure_ascii=True)]
        for t in out_lines:
            print(t)
        for t in err_lines:
            print(t, file=sys.stderr)
        if code:
            print(f'#ket_qua,ma={code}')
            return code
    else:                                        # C3-8: a manual send logs its declared source
        print('#nguon_gain,loai=tay,mo_ta=' + json.dumps(a.nguon_gain_tay.strip(), ensure_ascii=True))
    for t in spectral_lines(ctrl, gains):        # S04 v6.8 section 6: report only, never blocks
        print(t)

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
