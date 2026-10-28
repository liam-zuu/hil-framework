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

Files next to the CSV (opened exclusively: the logger refuses to overwrite a run):
    <out>                CSV, accepted data lines only (header = COLUMNS)
    <out>.events.txt     '#' lines from the DUT ('#run_end', ...) and '#logger,...' events
    <out>.raw.txt        every packet received, rejected ones too: t_recv_iso,src_ip,line
    <out>.summary.json   counters at the end (schema, restarts, reorder, gaps, rejects)

--strict (measurement mode, Codex vòng 2–3 #25, #43): only 66-column lines with the right
number types are kept; a DUT header that does not match COLUMNS name by name rejects data
until a matching header arrives. Without --strict, old 28/37-column lines are padded as before.
LOGGER_OPEN (socket bound, files open) is not DUT READY (#35): the logger never claims it.
"""

import argparse
import csv
import datetime
import json
import os
import socket
import sys
import time

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DEFAULT_PORT = 5005
BUFFER_SIZE  = 2048

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

# Columns the DUT prints with an integer format (AGV ctrl-imu logger.c:131-161); the rest are %f
INT_COLUMNS = frozenset([
    "timestamp_us", "fl_count", "fr_count", "rl_count", "rr_count",
    "pwm_fl", "pwm_fr", "pwm_rl", "pwm_rr",
    "tick", "dpwm_fl", "dpwm_fr", "dpwm_rl", "dpwm_rr", "sat_fl", "sat_fr", "sat_rl", "sat_rr",
    "log_seq", "log_qdrop", "log_send_err", "motor_on", "imu_init", "gyro_n", "rv_n", "reset_n",
    "g_valid", "r_valid", "wz_src", "src_switch_count", "head_on", "psi_reset_count",
    "slip_flag", "slip_on_count", "bias_updates", "slip_eval",
])
COL = {c: i for i, c in enumerate(COLUMNS)}

# DUT restart = timestamp_us AND log_seq both go back near zero, and the time goes back by
# more than any UDP reordering could [ĐỀ XUẤT; boot + WiFi can take ~15 s before the first
# line]. Any other drop of log_seq is UDP reordering (#43).
RESTART_TS_MAX_US      = 120_000_000
RESTART_SEQ_MAX        = 2000
RESTART_TS_DROP_MIN_US = 1_000_000


def _is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def _is_float(s):
    try:
        float(s)            # "nan" is accepted here and counted apart
        return True
    except ValueError:
        return False


class LoggerCore:
    """Packet checks without sockets or files (tested in tools/tests/test_agv_logger.py).

    feed() returns (kind, payload):
      ('data', fields)    keep in the CSV (fields padded to COLUMNS in loose mode)
      ('dut_event', line) '#' line from the DUT, goes to the events file
      ('header', None)    DUT header line (not written to the CSV)
      ('reject', reason)  not kept in the CSV (the raw file still has it)
    Logger events (restart, reorder, counter increases, header mismatch) collect in
    self.events as '#logger,...' lines; the caller drains them into the events file.
    """

    def __init__(self, strict=False, src_ip=None):
        self.strict = strict
        self.src_ip = src_ip
        self.header = 'not_seen'            # not_seen / match / old_prefix / mismatch
        self.schema_blocked = False         # strict mode after a header that is not COLUMNS
        self.events = []
        self.n_packets = 0
        self.n_data = 0
        self.n_dut_events = 0
        self.rejects = {}
        self.n_nan = 0
        self.n_restart = 0
        self.n_reorder = 0
        self.n_dup = 0
        self.boots = []                     # per boot segment: first/last ts and seq, max gap
        self._seen = set()
        self._last_ts = None
        self._last_seq = None
        self._qdrop = None
        self._send_err = None

    # ── helpers ───────────────────────────────────────────────────────────
    def _reject(self, reason):
        self.rejects[reason] = self.rejects.get(reason, 0) + 1
        return ('reject', reason)

    def _event(self, text):
        self.events.append('#logger,' + text)

    def _new_boot(self, ts, seq):
        self.boots.append(dict(first_ts=ts, last_ts=ts, first_seq=seq, last_seq=seq, max_gap_us=0, n=0))
        self._seen = set()
        self._qdrop = None
        self._send_err = None

    # ── main entry ────────────────────────────────────────────────────────
    def feed(self, line, src_ip=None):
        self.n_packets += 1
        if self.src_ip is not None and src_ip != self.src_ip:
            return self._reject('source')

        if line.startswith('#'):
            self.n_dut_events += 1
            return ('dut_event', line)

        if line.startswith('timestamp_us'):
            names = line.split(',')
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
            return ('header', None)

        fields = line.split(',')
        if self.strict:
            if self.schema_blocked:
                return self._reject('schema_header')
            if len(fields) != len(COLUMNS):
                return self._reject(f'n_cols_{len(fields)}')
            for name, val in zip(COLUMNS, fields):
                ok = _is_int(val) if name in INT_COLUMNS else _is_float(val)
                if not ok:
                    return self._reject('type')
        else:
            if len(fields) in (N_OLD, N_S02):
                fields = fields + [""] * (len(COLUMNS) - len(fields))
            if len(fields) != len(COLUMNS):
                return self._reject(f'n_cols_{len(fields)}')

        if any(v.strip().lower() == 'nan' for v in fields):
            self.n_nan += 1
        self._track(fields)
        self.n_data += 1
        return ('data', fields)

    def _track(self, fields):
        """Restart / reorder / duplicate / gap / counters, on lines that carry log_seq."""
        try:
            ts = int(fields[COL['timestamp_us']])
        except ValueError:
            return
        seq_s = fields[COL['log_seq']]
        seq = int(seq_s) if _is_int(seq_s) else None

        if not self.boots:
            self._new_boot(ts, seq)
        elif seq is not None and self._last_seq is not None and seq < self._last_seq:
            if (ts < self._last_ts - RESTART_TS_DROP_MIN_US and ts <= RESTART_TS_MAX_US
                    and seq <= RESTART_SEQ_MAX):
                self.n_restart += 1
                self._event(f'dut_restart,prev_seq={self._last_seq},prev_ts_us={self._last_ts},'
                            f'seq={seq},ts_us={ts}')
                self._new_boot(ts, seq)
            else:
                self.n_reorder += 1
                self._event(f'seq_reorder,seq={seq},after_seq={self._last_seq}')
                if seq in self._seen:
                    self.n_dup += 1
                self._seen.add(seq)
                self.boots[-1]['n'] += 1
                return                          # keep the newest seq/ts as reference
        elif seq is None and self._last_ts is not None and ts < self._last_ts:
            self.n_reorder += 1                 # old formats: no seq to tell restart from reorder
            self._event(f'ts_backwards,ts_us={ts},after_ts_us={self._last_ts}')
            return

        b = self.boots[-1]
        if seq is not None:
            if seq in self._seen:
                self.n_dup += 1
                self._event(f'dup_seq,seq={seq}')
            self._seen.add(seq)
            b['last_seq'] = seq
        if self._last_ts is not None and b['n'] > 0:
            b['max_gap_us'] = max(b['max_gap_us'], ts - self._last_ts)
        b['last_ts'] = ts
        b['n'] += 1
        self._last_ts, self._last_seq = ts, seq

        for name, attr in (('log_qdrop', '_qdrop'), ('log_send_err', '_send_err')):
            v = fields[COL[name]]
            if not _is_int(v):
                continue
            v = int(v)
            prev = getattr(self, attr)
            if prev is not None and v > prev:
                self._event(f'{name},from={prev},to={v},log_seq={seq}')
            setattr(self, attr, v if prev is None else max(prev, v))

    def summary(self):
        return dict(
            mode='strict' if self.strict else 'loose',
            src_ip=self.src_ip,
            header=self.header,
            packets=self.n_packets,
            data_lines=self.n_data,
            dut_event_lines=self.n_dut_events,
            rejects=self.rejects,
            lines_with_nan=self.n_nan,
            dut_restarts=self.n_restart,
            seq_reorder=self.n_reorder,
            seq_duplicates=self.n_dup,
            boots=self.boots,
            log_qdrop_last=self._qdrop,
            log_send_err_last=self._send_err,
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
                        help="Ghi file này khi đã bind cổng và mở file (LOGGER_OPEN), cho hil_session")
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

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.5)

    # Exclusive create: never append to or overwrite another run's files
    try:
        csvfile = open(filename, "x", newline="")
        evfile  = open(filename + ".events.txt", "x", encoding="utf-8")
        rawfile = open(filename + ".raw.txt", "x", encoding="utf-8")
    except FileExistsError as exc:
        sock.close()
        print(f"[ERROR] File đã có, không ghi đè: {exc.filename}", file=sys.stderr)
        return 2
    writer = csv.writer(csvfile)
    writer.writerow(COLUMNS)
    csvfile.flush()

    core = LoggerCore(strict=args.strict, src_ip=args.src_ip)

    print(f"[INFO] Lắng nghe UDP {args.bind}:{args.port}  mode={'strict' if args.strict else 'loose'}"
          f"  src_ip={args.src_ip or 'any'}")
    print(f"[INFO] Lưu vào: {filename}")
    print("LOGGER_OPEN", flush=True)
    if args.ready_file:
        with open(args.ready_file, "w") as f:
            f.write(f"LOGGER_OPEN {datetime.datetime.now().isoformat()} {os.path.abspath(filename)}\n")
    print("[INFO] Ctrl+C để dừng\n", flush=True)

    t_end = None if args.duration is None else time.monotonic() + args.duration
    last_wait = time.monotonic()
    try:
        while t_end is None or time.monotonic() < t_end:
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                if time.monotonic() - last_wait > 5.0:
                    print(f"[WAIT] Chưa nhận được data... ({core.n_data} lines saved)")
                    last_wait = time.monotonic()
                continue
            last_wait = time.monotonic()
            now = datetime.datetime.now().isoformat()

            line = data.decode("utf-8", errors="replace").strip()
            rawfile.write(f"{now},{addr[0]},{line}\n")

            kind, payload = core.feed(line, addr[0])
            if kind == 'dut_event':
                print(f"[RUN] {line}")
                evfile.write(f"{now} {line}\n")
            elif kind == 'header':
                print(f"[INFO] Header từ {addr[0]}: {core.header}")
            elif kind == 'data':
                writer.writerow(payload)
                if core.n_data % 100 == 0:
                    p = _progress(payload)
                    if p:
                        print(f"{p} | saved={core.n_data} rejects={sum(core.rejects.values())}")
            for ev in core.events:
                print(f"[LOG] {ev}")
                evfile.write(f"{now} {ev}\n")
            core.events.clear()
            csvfile.flush()
            evfile.flush()
            rawfile.flush()

    except KeyboardInterrupt:
        pass
    finally:
        for f in (csvfile, evfile, rawfile):
            f.flush()
            os.fsync(f.fileno())
            f.close()
        sock.close()
        with open(filename + ".summary.json", "x", encoding="utf-8") as f:
            json.dump(core.summary(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    print(f"\n[DONE] Dừng — đã lưu {core.n_data} dòng, từ chối {sum(core.rejects.values())} "
          f"{core.rejects or ''}")
    print(f"[DONE] File: {filename}")
    if core.n_data > 0:
        print("\n[NEXT] Plot data:")
        print(f"       python plot_log.py {filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
