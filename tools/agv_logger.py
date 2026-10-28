#!/usr/bin/env python3
"""
agv_logger.py — nhận UDP từ ESP32 (hoặc fake_esp32.py), lưu CSV
Không thay đổi gì giữa test laptop và hardware thật.

Usage:
    python agv_logger.py                    # lắng nghe 0.0.0.0:5005
    python agv_logger.py --port 5006        # đổi port
    python agv_logger.py --out my_log.csv   # đặt tên file output
    python agv_logger.py --strict --src-ip 192.168.1.50 --out run.csv --ready-file run.ready
                                            # chế độ đo (luồng làm việc HIL bản 3.1, mục 9.5)

Files next to the CSV. All five paths are checked before anything is created and each is
created exclusively, so the logger never overwrites or appends to another run:
    <out>                CSV, accepted data lines in arrival order (header = COLUMNS); kept as
                         received: duplicates, reordering and DUT restarts are reported, not fixed
    <out>.events.txt     '#' lines from the DUT ('#run_end', ...) and '#logger,...' events
    <out>.raw.jsonl      every datagram as received (base64), rejected ones too, with source
                         ip/port, length, UTC and monotonic receive time
    <out>.summary.json   written at open ("open") and rewritten at the end: counters per DUT boot
    --ready-file         JSON {state: LOGGER_OPEN, pid, out}; removed at a clean exit

--strict (measurement mode, Codex rounds 2-3 #25, #43; B0 review B3-B10): only 66-column
lines with the DUT's printf types and C type ranges are kept; a header-like line that is not
COLUMNS name by name blocks data until a matching header arrives. Without --strict, old
28/37-column lines are padded as before. LOGGER_OPEN is not DUT READY (#35).

DUT restarts: within one boot log_seq and timestamp_us grow together, so a line belongs to
a boot when it fits between that boot's neighbouring seqs in time. A line that fits no boot
starts a new one, reported as 'dut_restart_suspected': without a boot id from the DUT it
cannot be confirmed (a boot with few lines is flagged in the summary).
"""

import argparse
import base64
import bisect
import csv
import datetime
import json
import math
import os
import signal
import socket
import sys
import time

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DEFAULT_PORT = 5005
BUFFER_SIZE  = 65535        # whole UDP datagram, never truncated

COLUMNS = [
    "timestamp_us",
    "fl_count", "fr_count", "rl_count", "rr_count",
    "fl_rpm",   "fr_rpm",   "rl_rpm",   "rr_rpm",
    "vx", "vy", "wz",
    "sp_vx", "sp_vy", "sp_wz",
    "pwm_fl", "pwm_fr", "pwm_rl", "pwm_rr",
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "yaw", "pitch", "roll",
    # AGV_MECANUM vel_ctrl metric counters (spec S02): tick, cumulative |dpwm|, saturated ticks
    "tick", "dpwm_fl", "dpwm_fr", "dpwm_rl", "dpwm_rr", "sat_fl", "sat_fr", "sat_rl", "sat_rr",
    # AGV_MECANUM ctrl-imu (controller-IMU spec v2.2 section 8): log loss counters, motor_on
    # (0 = 10 Hz line while the motor is off), imu_init code, IMU fusion / heading loop.
    # Old columns keep their meaning: wz = encoder wz, gz = raw chip gyro, yaw in degrees.
    "log_seq", "log_qdrop", "log_send_err", "motor_on", "imu_init",
    "wz_meas", "wz_cmd", "gz_fused", "yaw_rad", "age_g_ms", "age_r_ms", "gyro_n", "rv_n", "reset_n",
    "g_valid", "r_valid", "wz_src", "src_switch_count", "psi_ref", "e_psi", "head_on", "psi_reset_count",
    "slip_flag", "slip_on_count", "r", "bias_gz",
    # Added 28/10 (Codex review of ctrl-imu #11, #8 (c)): slip_eval 0 = r could not be checked
    # this tick (gyro invalid or wheels below omega_min), so slip_flag 0 does not mean "no slip".
    "src_switch_dwz", "bias_updates", "slip_eval",
]
N_OLD = 28      # firmware before the counters: same columns without the last 9 + 29
N_S02 = 37      # firmware a6644a5 .. before ctrl-imu: without the last 29

# C type of each integer column in the DUT's log_data_t (AGV ctrl-imu 030a836 logger.h:17-62,
# printed by logger.c:131-161); every other column is a float printed with %f.
_I64 = (-2**63, 2**63 - 1)
_I32 = (-2**31, 2**31 - 1)
_U32 = (0, 2**32 - 1)
_U8  = (0, 255)
_I8  = (-128, 127)
INT_RANGES = {"timestamp_us": _I64}
INT_RANGES.update({c: _I32 for c in ("fl_count", "fr_count", "rl_count", "rr_count",
                                     "pwm_fl", "pwm_fr", "pwm_rl", "pwm_rr")})
INT_RANGES.update({c: _U32 for c in ("tick", "dpwm_fl", "dpwm_fr", "dpwm_rl", "dpwm_rr",
                                     "sat_fl", "sat_fr", "sat_rl", "sat_rr",
                                     "log_seq", "log_qdrop", "log_send_err", "gyro_n", "rv_n", "reset_n",
                                     "src_switch_count", "psi_reset_count", "slip_on_count",
                                     "bias_updates")})
INT_RANGES.update({c: _U8 for c in ("motor_on", "g_valid", "r_valid", "wz_src", "head_on",
                                    "slip_flag", "slip_eval")})
INT_RANGES["imu_init"] = _I8
INT_COLUMNS = frozenset(INT_RANGES)
COL = {c: i for i, c in enumerate(COLUMNS)}

RUN_END_KEYS = ("seq", "last_data_seq", "log_qdrop", "log_send_err")   # logger.c:123
CONFIRM_LINES = 10          # a boot with fewer data lines is flagged 'few_lines' [ĐỀ XUẤT]
# UDP reordering only delays a datagram by a few others; a line more than this many seqs
# below the highest seq already received cannot be a late line of that boot [ĐỀ XUẤT]
REORDER_WINDOW = 64


def _int(s):
    try:
        return int(s)
    except ValueError:
        return None


def _float(s):
    try:
        return float(s)
    except ValueError:
        return None


def parse_run_end(line):
    """'#run_end,seq=..,last_data_seq=..,log_qdrop=..,log_send_err=..' -> dict, or None if malformed."""
    parts = line.split(',')
    if parts[0] != '#run_end':
        return None
    kv = {}
    for p in parts[1:]:
        k, sep, v = p.partition('=')
        if not sep or _int(v) is None:
            return None
        kv[k] = int(v)
    return kv if all(k in kv for k in RUN_END_KEYS) else None


class _Boot:
    """Lines of one DUT boot: log_seq -> timestamp_us, kept sorted by seq."""

    def __init__(self, index):
        self.index = index
        self.ts = {}
        self.seqs = []
        self.n = 0
        self.dup = 0
        self.reorder = 0
        self.late = 0               # lines placed here after a later boot had started
        self.footers = []           # (seq, last_data_seq) of valid '#run_end' lines
        self.counters = {}          # name -> [first, last] of log_qdrop / log_send_err
        self.header = 'not_seen'

    def fits(self, seq, ts):
        """seq/ts consistent with this boot's lines, and not further back than UDP reordering."""
        if seq in self.ts or (self.seqs and seq < self.seqs[-1] - REORDER_WINDOW):
            return False
        i = bisect.bisect_left(self.seqs, seq)
        before = self.ts[self.seqs[i - 1]] if i > 0 else None
        after = self.ts[self.seqs[i]] if i < len(self.seqs) else None
        return (before is None or before <= ts) and (after is None or ts <= after)

    def add(self, seq, ts):
        if self.seqs and seq < self.seqs[-1]:
            self.reorder += 1
        bisect.insort(self.seqs, seq)
        self.ts[seq] = ts
        self.n += 1

    def summary(self):
        tss = [self.ts[s] for s in self.seqs]
        gaps = [b - a for a, b in zip(tss, tss[1:])]
        footer_seqs = {f[0] for f in self.footers}
        span = (self.seqs[-1] - self.seqs[0] + 1) if self.seqs else 0
        holes = span - len(self.seqs) - sum(1 for s in footer_seqs if self.seqs and self.seqs[0] <= s <= self.seqs[-1])
        return dict(boot=self.index, data_lines=self.n,
                    first_seq=self.seqs[0] if self.seqs else None, last_seq=self.seqs[-1] if self.seqs else None,
                    first_ts=tss[0] if tss else None, last_ts=tss[-1] if tss else None,
                    max_gap_us=max(gaps) if gaps else 0,   # between timestamps sorted by seq
                    seq_holes=holes, duplicates=self.dup, reordered=self.reorder, late_lines=self.late,
                    footers=[dict(seq=s, last_data_seq=l) for s, l in self.footers],
                    counters=self.counters, header=self.header,
                    few_lines=self.n < CONFIRM_LINES)


class LoggerCore:
    """Datagram checks without sockets or files (tested in tools/tests/test_agv_logger.py).

    feed() returns (kind, payload):
      ('data', fields)          keep in the CSV (fields padded to COLUMNS in loose mode)
      ('dut_event', line)       '#' line from the DUT, goes to the events file as is
      ('dut_event_bad', line)   malformed '#run_end': goes to the events file escaped
      ('header', None)          header-like line (not written to the CSV)
      ('reject', reason)        not kept in the CSV (the raw file still has it)
    Logger events collect in self.events as '#logger,...' lines; the caller drains them.
    """

    def __init__(self, strict=False, src_ip=None):
        self.strict = strict
        self.src_ip = src_ip
        self.header = 'not_seen'            # last header-like line: not_seen / match / old_prefix / mismatch
        self.schema_blocked = False
        self.events = []
        self.n_packets = 0
        self.n_data = 0
        self.n_dut_events = 0
        self.rejects = {}
        self.n_nonfinite = 0
        self.n_restart_suspected = 0
        self.boots = []
        self._last_ts_noseq = None
        self._header_since_data = None      # header seen after the last data line (belongs to the next boot)

    # ── helpers ───────────────────────────────────────────────────────────
    def _reject(self, reason):
        self.rejects[reason] = self.rejects.get(reason, 0) + 1
        return ('reject', reason)

    def _event(self, text):
        self.events.append('#logger,' + text)

    def _counter(self, boot, name, v, seq):
        c = boot.counters.get(name)
        if c is None:
            boot.counters[name] = [v, v]
            return
        if v > c[1]:
            self._event(f'{name},from={c[1]},to={v},log_seq={seq},boot={boot.index}')
            c[1] = v

    # ── main entry ────────────────────────────────────────────────────────
    def feed_datagram(self, data, src_ip=None):
        """Raw UDP payload -> feed(); undecodable or multi-line datagrams are rejected."""
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            self.n_packets += 1
            return self._reject('decode')
        text = text.rstrip('\r\n')
        if '\n' in text or '\r' in text:
            self.n_packets += 1
            return self._reject('multiline')
        return self.feed(text, src_ip)

    def feed(self, line, src_ip=None):
        self.n_packets += 1
        if self.src_ip is not None and src_ip != self.src_ip:
            return self._reject('source')
        line = line.lstrip('﻿')
        if not line.strip():
            return self._reject('empty')

        if line.startswith('#'):
            if line.startswith('#run_end'):
                kv = parse_run_end(line)
                if kv is None:
                    self._event('bad_dut_event,raw=' + line.replace(',', ';'))
                    return ('dut_event_bad', line)
                self._footer(kv)
            self.n_dut_events += 1
            return ('dut_event', line)

        fields = line.split(',')
        if all(_float(f) is None for f in fields):
            return self._header(fields)

        if self.strict:
            if self.schema_blocked:
                return self._reject('schema_header')
            if len(fields) != len(COLUMNS):
                return self._reject(f'n_cols_{len(fields)}')
            for name, val in zip(COLUMNS, fields):
                if name in INT_RANGES:
                    v = _int(val)
                    if v is None:
                        return self._reject('type')
                    lo, hi = INT_RANGES[name]
                    if not lo <= v <= hi:
                        return self._reject('range')
                elif _float(val) is None:
                    return self._reject('type')
        else:
            if len(fields) in (N_OLD, N_S02):
                fields = fields + [""] * (len(COLUMNS) - len(fields))
            if len(fields) != len(COLUMNS):
                return self._reject(f'n_cols_{len(fields)}')

        if any((x := _float(v)) is not None and not math.isfinite(x) for v in fields):
            self.n_nonfinite += 1
        self._track(fields)
        self.n_data += 1
        return ('data', fields)

    def _header(self, names):
        names = [n.strip() for n in names]
        if names == COLUMNS:
            self.header = 'match'
            self.schema_blocked = False
        elif names in (COLUMNS[:N_OLD], COLUMNS[:N_S02]):
            self.header = 'old_prefix'
            self.schema_blocked = self.strict
            self._event(f'header_old,n_cols={len(names)}')
        else:
            self.header = 'mismatch'
            self.schema_blocked = self.strict
            bad = next((i for i, (a, b) in enumerate(zip(names, COLUMNS)) if a != b),
                       min(len(names), len(COLUMNS)))
            self._event(f'header_mismatch,n_cols={len(names)},first_diff_col={bad}')
        self._header_since_data = self.header
        return ('header', None)

    def _footer(self, kv):
        """A valid '#run_end' belongs to the boot whose data its last_data_seq continues."""
        boot = next((b for b in reversed(self.boots) if kv['last_data_seq'] in b.ts), None)
        if boot is None:
            boot = self.boots[-1] if self.boots else None
        if boot is None:
            self._event(f"run_end_without_data,last_data_seq={kv['last_data_seq']}")
            return
        if (kv['seq'], kv['last_data_seq']) not in boot.footers:
            boot.footers.append((kv['seq'], kv['last_data_seq']))
        self._counter(boot, 'log_qdrop', kv['log_qdrop'], kv['seq'])
        self._counter(boot, 'log_send_err', kv['log_send_err'], kv['seq'])

    def _new_boot(self):
        b = _Boot(len(self.boots))
        self.boots.append(b)
        return b

    def _track(self, fields):
        """Place a data line into a DUT boot; report duplicates, reordering, restarts, counters."""
        ts = _int(fields[COL['timestamp_us']])
        seq = _int(fields[COL['log_seq']])
        if ts is None:
            return
        if seq is None:                       # old formats: time order only
            if self._last_ts_noseq is not None and ts < self._last_ts_noseq:
                self._event(f'ts_backwards,ts_us={ts},after_ts_us={self._last_ts_noseq}')
            self._last_ts_noseq = ts if self._last_ts_noseq is None else max(ts, self._last_ts_noseq)
            return

        for b in self.boots:                  # the same datagram twice
            if b.ts.get(seq) == ts:
                b.dup += 1
                self._event(f'dup_seq,seq={seq},boot={b.index}')
                return

        cur = self.boots[-1] if self.boots else None
        # A late line of an earlier boot also "continues" the current boot forward, so an
        # earlier boot that takes it back (within its reordering window) wins.
        old = next((b for b in reversed(self.boots[:-1])
                    if b.seqs and seq < b.seqs[-1] and b.fits(seq, ts)), None)
        if cur is None:
            boot = self._new_boot()
        elif old is not None:
            boot = old
            old.late += 1
            self._event(f'late_line_of_earlier_boot,seq={seq},ts_us={ts},boot={old.index}')
        elif cur.fits(seq, ts):
            boot = cur
            if cur.seqs and seq < cur.seqs[-1]:
                self._event(f'seq_reorder,seq={seq},after_seq={cur.seqs[-1]},boot={cur.index}')
        else:
            self.n_restart_suspected += 1
            self._event(f'dut_restart_suspected,seq={seq},ts_us={ts},prev_last_seq={cur.seqs[-1]},'
                        f'prev_last_ts_us={cur.ts[cur.seqs[-1]]}')
            boot = self._new_boot()
        boot.add(seq, ts)
        if self._header_since_data is not None:   # the DUT sends its header once, at boot
            if boot.n > 1:
                self._event(f'header_mid_boot,boot={boot.index}')
            boot.header = self._header_since_data
            self._header_since_data = None
        for name in ('log_qdrop', 'log_send_err'):
            v = _int(fields[COL[name]])
            if v is not None:
                self._counter(boot, name, v, seq)

    def summary(self):
        return dict(
            mode='strict' if self.strict else 'loose',
            src_ip=self.src_ip,
            header=self.header,
            packets=self.n_packets,
            data_lines=self.n_data,
            dut_event_lines=self.n_dut_events,
            rejects=self.rejects,
            lines_with_nonfinite=self.n_nonfinite,
            dut_restarts_suspected=self.n_restart_suspected,
            boots=[b.summary() for b in self.boots],
            note='boots split by seq/time consistency; no DUT boot id, so a restart is suspected, not confirmed',
        )


def _progress(fields):
    """Progress line read by column name (old version printed vy/wz/sp_vx/rr_count as vx/vy/wz/fl_rpm)."""
    try:
        ts_ms = int(fields[COL['timestamp_us']]) // 1000
        vals = {c: float(fields[COL[c]]) for c in ('vx', 'vy', 'wz', 'fl_rpm')}
    except (ValueError, IndexError):
        return None
    return (f"[{ts_ms:8d}ms] vx={vals['vx']:+.3f} vy={vals['vy']:+.3f} wz={vals['wz']:+.3f} "
            f"| fl_rpm={vals['fl_rpm']:+6.1f}")


def _raw_record(data, addr):
    return json.dumps(dict(t_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           t_mono_ns=time.monotonic_ns(), ip=addr[0], port=addr[1],
                           len=len(data), b64=base64.b64encode(data).decode('ascii')))


def _fsync_dir(path):
    if os.name == 'nt':                       # no directory fsync on Windows
        return
    fd = os.open(os.path.dirname(os.path.abspath(path)) or '.', os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main(argv=None):
    parser = argparse.ArgumentParser(description="AGV UDP Logger")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--bind", default="0.0.0.0", help="Địa chỉ bind (mặc định mọi card mạng)")
    parser.add_argument("--out",  default=None, help="Tên file CSV output")
    parser.add_argument("--strict", action="store_true",
                        help="Chế độ đo: chỉ giữ dòng 66 cột đúng kiểu, header DUT phải khớp")
    parser.add_argument("--src-ip", default=None, help="Chỉ nhận gói từ IP này (ESP32 của bàn HIL)")
    parser.add_argument("--ready-file", default=None,
                        help="Tạo file này (JSON, PID) khi đã bind cổng và mở file (LOGGER_OPEN)")
    parser.add_argument("--duration", type=float, default=None, help="Tự dừng sau N giây")
    args = parser.parse_args(argv)

    # Piped stdout on Windows is cp1252 and cannot print Vietnamese (hil_session reads it)
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    # Tên file tự động nếu không chỉ định
    if args.out is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agv_log_{ts}.csv"
    else:
        filename = args.out
    paths = {"csv": filename, "events": filename + ".events.txt",
             "raw": filename + ".raw.jsonl", "summary": filename + ".summary.json"}
    if args.ready_file:
        paths["ready"] = args.ready_file
    norm = [os.path.normcase(os.path.abspath(p)) for p in paths.values()]
    if len(set(norm)) != len(norm):
        print(f"[ERROR] Hai file output trùng đường dẫn: {paths}", file=sys.stderr)
        return 2
    existing = [p for p in paths.values() if os.path.exists(p)]
    if existing:
        print(f"[ERROR] File đã có, không ghi đè: {existing}", file=sys.stderr)
        return 2

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.5)

    opened = {}
    try:                                       # exclusive create, summary placeholder first
        opened["summary"] = open(paths["summary"], "x", encoding="utf-8")
        opened["summary"].write('{"status": "open"}\n')
        opened["summary"].close()
        opened["csv"] = open(paths["csv"], "x", newline="")
        opened["events"] = open(paths["events"], "x", encoding="utf-8")
        opened["raw"] = open(paths["raw"], "x", encoding="utf-8")
    except OSError as exc:
        for f in opened.values():
            f.close()
        sock.close()
        print(f"[ERROR] Không tạo được file: {exc}", file=sys.stderr)
        return 2
    csvfile, evfile, rawfile = opened["csv"], opened["events"], opened["raw"]
    writer = csv.writer(csvfile)
    writer.writerow(COLUMNS)
    csvfile.flush()

    core = LoggerCore(strict=args.strict, src_ip=args.src_ip)

    stop = {"flag": False}

    def _on_signal(signum, _frame):
        stop["flag"] = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    print(f"[INFO] Lắng nghe UDP {args.bind}:{args.port}  mode={'strict' if args.strict else 'loose'}"
          f"  src_ip={args.src_ip or 'any'}")
    print(f"[INFO] Lưu vào: {filename}")
    ready_made = False
    if args.ready_file:
        with open(args.ready_file, "x", encoding="utf-8") as f:
            json.dump(dict(state="LOGGER_OPEN", pid=os.getpid(), out=os.path.abspath(filename),
                           t_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), f)
        ready_made = True
    print("LOGGER_OPEN", flush=True)
    print("[INFO] Ctrl+C để dừng\n", flush=True)

    t_end = None if args.duration is None else time.monotonic() + args.duration
    last_wait = time.monotonic()
    errors = []
    try:
        while not stop["flag"] and (t_end is None or time.monotonic() < t_end):
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                if time.monotonic() - last_wait > 5.0:
                    print(f"[WAIT] Chưa nhận được data... ({core.n_data} lines saved)")
                    last_wait = time.monotonic()
                continue
            except InterruptedError:
                continue
            last_wait = time.monotonic()
            now = datetime.datetime.now().isoformat()
            rawfile.write(_raw_record(data, addr) + "\n")

            kind, payload = core.feed_datagram(data, addr[0])
            if kind == 'dut_event':
                print(f"[RUN] {payload}")
                evfile.write(f"{now} {payload}\n")
            elif kind == 'header':
                print(f"[INFO] Header từ {addr[0]}: {core.header}")
            elif kind == 'data':
                writer.writerow(payload)
                if core.n_data % 100 == 0:
                    p = _progress(payload)
                    if p:
                        print(f"{p} | saved={core.n_data} rejects={sum(core.rejects.values())}")
            for ev in core.events:             # malformed DUT events arrive here, escaped
                print(f"[LOG] {ev}")
                evfile.write(f"{now} {ev}\n")
            core.events.clear()
            csvfile.flush()
            evfile.flush()
            rawfile.flush()
    finally:
        # Close every resource on its own: one failing flush/fsync must not skip the rest
        for name, f in (("csv", csvfile), ("events", evfile), ("raw", rawfile)):
            try:
                f.flush()
                os.fsync(f.fileno())
            except OSError as exc:
                errors.append(f"{name}: {exc}")
            finally:
                f.close()
        sock.close()
        try:
            summ = core.summary()
            summ["status"] = "closed" if not errors else "closed_with_errors"
            summ["close_errors"] = errors
            with open(paths["summary"], "w", encoding="utf-8") as f:
                json.dump(summ, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            _fsync_dir(paths["csv"])
        except OSError as exc:
            errors.append(f"summary: {exc}")
        if ready_made:
            try:
                os.remove(args.ready_file)
            except OSError as exc:
                errors.append(f"ready: {exc}")

    print(f"\n[DONE] Dừng — đã lưu {core.n_data} dòng, từ chối {sum(core.rejects.values())} "
          f"{core.rejects or ''}")
    print(f"[DONE] File: {filename}")
    if errors:
        print(f"[ERROR] Lỗi khi đóng: {errors}", file=sys.stderr)
        return 3
    if core.n_data > 0:
        print("\n[NEXT] Plot data:")
        print(f"       python plot_log.py {filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
