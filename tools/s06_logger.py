#!/usr/bin/env python3
"""
s06_logger.py — bộ nhận log ESP32 env `s06_load` và điều khiển một lượt S06 (chạy trên RPi5).

Spec S06 v3.4 mục 8 (thiết kế: 03_CODE/docs/2026-10-29_Thiet_ke_env_s06_load.md; Liam 29/10:
Q-S1 = 1 ms, Q-S2 = lệnh UDP). Nhận hai luồng UDP:
  - DATA 100 Hz của logger.c (cổng 5005, CSV + dòng '#run_end' của đường motor_off);
  - sự kiện của s06_load.c (cổng 5007: J, I, R, P, E, S và dòng '#...').
Gửi lệnh tới ESP32 (cổng 5011): `START <zero|heavy|t6> <s> <0|1>`, `STOP <lý do>`.

Ghi log thô nguyên văn: mỗi dòng nhận được ghi thành `<t_rx_us>\t<dòng>`, t_rx_us là
time.monotonic_ns() // 1000 trên RPi5 lúc nhận gói (dùng cho luật T9: t_rx(#run_end) − t_rx(#run_start)).
Không lọc, không sửa dòng; kiểm lượt bằng s06_esp32_check.py.
Nhiệt độ CPU RPi5 (T5, Liam 30/10): mỗi 1 s đọc `--temp-path` (mặc định thermal_zone0, m°C), ghi
`<t_us>\t<m°C>` vào `pi_temp.txt` cùng đồng hồ với t_rx_us; không đọc được thì capture.yaml ghi lý do.

Usage:
    python tools/s06_logger.py --esp32 192.168.1.50 --profile heavy --duration 60 --out <thư mục lượt>
    python tools/s06_logger.py --esp32 ... --profile zero --duration 610 --no-auto-end   # T9: Pi gửi STOP
"""

import argparse
import datetime
import os
import selectors
import socket
import subprocess
import sys
import time

DATA_PORT = 5005          # logger.c LOGGER_UDP_PORT
EVT_PORT = 5007           # s06_load.c S06_EVT_PORT
CMD_PORT = 5011           # s06_load.c S06_CMD_PORT
RCVBUF = 4 * 1024 * 1024  # bộ đệm nhận lớn: mất gói phía Pi cũng làm lượt không đủ điều kiện đo
START_TRIES = 5           # gửi lại START mỗi 1 s tới khi thấy #ack hoặc #run_start
STOP_TRIES = 5            # gửi lại STOP mỗi 0,2 s tới khi thấy #run_end
END_GRACE_S = 5.0         # lượt tự kết thúc: chờ #run_end tới thời lượng + chừng này, quá thì gửi STOP
TEMP_PATH = "/sys/class/thermal/thermal_zone0/temp"
TEMP_PERIOD_US = 1_000_000


def now_us():
    return time.monotonic_ns() // 1000


def git_rev(path):
    """Commit và trạng thái sạch của repo chứa script (ghi vào capture.yaml)."""
    try:
        sha = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain", "--", "."],
                               capture_output=True, text=True, timeout=5).stdout.strip() != ""
        return sha or "unknown", dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", True


def text_field(line, key):
    """Giá trị `key=...` trong dòng '#...,k=v,...' (None nếu không có)."""
    for part in line.split(","):
        if part.startswith(key + "="):
            return part[len(key) + 1:]
    return None


class Capture:
    """Hai socket nhận, hai file thô, và trạng thái lượt đọc từ các dòng '#'."""

    def __init__(self, out_dir, data_port, evt_port, bind="0.0.0.0", temp_path=TEMP_PATH):
        os.makedirs(out_dir, exist_ok=True)
        self.temp_path = temp_path
        self.temp_file = open(os.path.join(out_dir, "pi_temp.txt"), "w", encoding="utf-8", newline="\n")
        self.temp_next = 0
        self.temp_n = 0
        self.temp_error = None    # lần đọc lỗi đầu tiên (ghi vào capture.yaml)
        self.out_dir = out_dir
        self.sel = selectors.DefaultSelector()
        self.files = {}
        self.lines = {}
        for name, port in (("data", data_port), ("events", evt_port)):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF)
            s.bind((bind, port))
            s.setblocking(False)
            self.sel.register(s, selectors.EVENT_READ, name)
            self.files[name] = open(os.path.join(out_dir, f"esp32_{name}.txt"), "w",
                                    encoding="utf-8", newline="\n")
            self.lines[name] = 0
        self.socks = [k.fileobj for k in self.sel.get_map().values()]
        self.peer = None
        self.acks = []            # (t_rx_us, dòng)
        self.run_start = None     # (t_rx_us, dòng) bản đầu tiên
        self.run_end = None       # (t_rx_us, dòng) bản đầu tiên của luồng sự kiện
        self.fw = None

    def sample_temp(self):
        t = now_us()
        if t < self.temp_next:
            return
        self.temp_next = t + TEMP_PERIOD_US
        try:
            with open(self.temp_path, encoding="ascii") as f:
                mc = int(f.read().strip())
        except (OSError, ValueError) as exc:
            if self.temp_error is None:
                self.temp_error = f"{type(exc).__name__}: {exc}"
            return
        self.temp_file.write(f"{t}\t{mc}\n")
        self.temp_n += 1

    def poll(self, timeout_s):
        self.sample_temp()
        for key, _ in self.sel.select(timeout_s):
            name = key.data
            while True:
                try:
                    pkt, addr = key.fileobj.recvfrom(65535)
                except BlockingIOError:
                    break
                t = now_us()
                if self.peer is None:
                    self.peer = addr[0]
                f = self.files[name]
                for line in pkt.decode("utf-8", errors="replace").splitlines():
                    if not line:
                        continue
                    f.write(f"{t}\t{line}\n")
                    self.lines[name] += 1
                    if name == "events" and line.startswith("#"):
                        self._on_text(t, line)

    def _on_text(self, t, line):
        if line.startswith("#ack,"):
            self.acks.append((t, line))
            print(f"[ESP32] {line}")
        elif line.startswith("#run_start,") and self.run_start is None:
            self.run_start = (t, line)
            print(f"[ESP32] {line}")
        elif line.startswith("#run_end,") and self.run_end is None:
            self.run_end = (t, line)
            print(f"[ESP32] {line}")
        elif line.startswith("#fw,"):
            self.fw = line
            print(f"[ESP32] {line}")

    def close(self):
        self.temp_file.close()
        for f in self.files.values():
            f.close()
        for s in self.socks:
            s.close()


def send_cmd(sock, addr, text):
    sock.sendto(text.encode("ascii"), addr)
    print(f"[PI] -> {addr[0]}:{addr[1]} {text}")
    return now_us()


def run(args):
    cap = Capture(args.out, args.data_port, args.evt_port, args.bind, args.temp_path)
    cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.esp32, args.cmd_port)
    auto_end = 0 if args.no_auto_end else 1
    start_text = f"START {args.profile} {args.duration} {auto_end}"
    sent = []                 # (t_us, lệnh)
    stop_reason = None
    result = "ok"
    t0_wall = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    t0 = now_us()
    try:
        # Nghe trước một lúc để thấy luồng đã chảy (và #fw nếu ESP32 vừa khởi động)
        t_end = time.monotonic() + args.listen_before
        while time.monotonic() < t_end:
            cap.poll(0.05)

        # START, gửi lại tới khi ESP32 trả lời
        for _ in range(START_TRIES):
            n_ack = len(cap.acks)
            sent.append((send_cmd(cmd, dest, start_text), start_text))
            t_end = time.monotonic() + 1.0
            while time.monotonic() < t_end and cap.run_start is None and len(cap.acks) == n_ack:
                cap.poll(0.05)
            if cap.run_start is not None or len(cap.acks) > n_ack:
                break
        t_end = time.monotonic() + 1.0
        while cap.run_start is None and time.monotonic() < t_end:
            cap.poll(0.05)
        if cap.run_start is None:
            result = "no_run_start"
            print("[LỖI] Không thấy #run_start (xem các dòng #ack ở trên).", file=sys.stderr)
        else:
            # Chờ hết lượt: tự kết thúc, hoặc Pi gửi STOP lúc t_rx(#run_start) + thời lượng (T9)
            t_rs = cap.run_start[0]
            limit = args.duration * 1e6 + (END_GRACE_S * 1e6 if auto_end else 0)
            last_report = time.monotonic()
            while cap.run_end is None and now_us() - t_rs < limit:
                cap.poll(0.01)
                if time.monotonic() - last_report >= 10:
                    last_report = time.monotonic()
                    print(f"[PI] {(now_us() - t_rs) / 1e6:.0f} s, "
                          f"data {cap.lines['data']} dòng, events {cap.lines['events']} dòng")
            if cap.run_end is None:
                stop_reason = "pi_stop" if not auto_end else "pi_timeout"
    except KeyboardInterrupt:
        stop_reason = "operator"
        result = "operator"
    try:
        if cap.run_start is not None and cap.run_end is None and stop_reason:
            for _ in range(STOP_TRIES):
                sent.append((send_cmd(cmd, dest, f"STOP {stop_reason}"), f"STOP {stop_reason}"))
                t_end = time.monotonic() + 0.2
                while cap.run_end is None and time.monotonic() < t_end:
                    cap.poll(0.01)
                if cap.run_end is not None:
                    break
            if cap.run_end is None and result == "ok":
                result = "no_run_end"
        # Nhận nốt phần đuôi (report IMU tới callback sau #run_end, dòng S)
        t_end = time.monotonic() + args.linger
        while time.monotonic() < t_end:
            cap.poll(0.05)
    finally:
        cap.close()
        cmd.close()
    write_capture_yaml(args, cap, sent, t0_wall, t0, result, stop_reason)
    print(f"[PI] Kết quả: {result}; data {cap.lines['data']} dòng, events {cap.lines['events']} dòng "
          f"-> {args.out}")
    return 0 if result == "ok" else 1


def yq(v):
    """Chuỗi YAML trong nháy kép."""
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_capture_yaml(args, cap, sent, t0_wall, t0, result, stop_reason):
    sha, dirty = git_rev(os.path.dirname(os.path.abspath(__file__)))
    env = text_field(cap.fw, "env") if cap.fw else None
    if cap.run_start is not None:
        env = env or text_field(cap.run_start[1], "env")
    out = [
        "# Ghi bởi tools/s06_logger.py: thông tin bộ nhận biết được. meta.yaml của lượt chép từ đây",
        "# và thêm phần người đo ghi (commit H7/RPi5, sha256 bản .bin đã nạp, thời gian bàn đã chạy).",
        f"script: tools/s06_logger.py",
        f"script_commit: {sha}",
        f"script_dirty: {str(dirty).lower()}",
        f"started_wall: {yq(t0_wall)}",
        f"t0_rx_us: {t0}",
        f"clock_rx: time.monotonic_ns() // 1000 trên máy chạy bộ nhận",
        f"esp32_ip: {yq(args.esp32)}",
        f"esp32_seen_from: {yq(cap.peer or '')}",
        f"ports: {{data: {args.data_port}, events: {args.evt_port}, cmd: {args.cmd_port}}}",
        f"profile: {args.profile}",
        f"duration_s: {args.duration}",
        f"auto_end: {0 if args.no_auto_end else 1}",
        f"result: {result}",
        f"pi_stop_reason: {yq(stop_reason or '')}",
        f"fw_line: {yq(cap.fw or '')}",
        f"run_start: {{t_rx_us: {cap.run_start[0] if cap.run_start else 'null'}, "
        f"line: {yq(cap.run_start[1] if cap.run_start else '')}}}",
        f"run_end: {{t_rx_us: {cap.run_end[0] if cap.run_end else 'null'}, "
        f"line: {yq(cap.run_end[1] if cap.run_end else '')}}}",
        f"lines: {{data: {cap.lines['data']}, events: {cap.lines['events']}}}",
        f"pi_temp: {{file: pi_temp.txt, path: {yq(cap.temp_path)}, samples: {cap.temp_n}, "
        f"error: {yq(cap.temp_error or '')}}}",
        "commands:",
    ]
    out += [f"  - {{t_us: {t}, cmd: {yq(c)}}}" for t, c in sent]
    out.append("acks:")
    out += [f"  - {{t_rx_us: {t}, line: {yq(l)}}}" for t, l in cap.acks]
    if env == "s06_load":
        # THESIS ghi spec S06 mục 2.2 (Liam duyệt 29/10, "Ghi cả 2"): khác build xe, ghi mỗi lượt
        out.append('build_diff_vs_car: ["WiFi power save off (esp_wifi_set_ps(WIFI_PS_NONE), s06_load.c)"]')
    with open(os.path.join(args.out, "capture.yaml"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description="Nhận log ESP32 s06_load và chạy một lượt S06")
    p.add_argument("--esp32", required=True, help="IP của ESP32 (nhận lệnh UDP)")
    p.add_argument("--profile", required=True, choices=["zero", "heavy", "t6"])
    p.add_argument("--duration", required=True, type=int, help="thời lượng lượt [s]")
    p.add_argument("--no-auto-end", action="store_true",
                   help="ESP32 không tự kết thúc; Pi gửi STOP lúc t_rx(#run_start) + thời lượng (T9)")
    p.add_argument("--out", default=None, help="thư mục ghi (mặc định s06_capture_<giờ>)")
    p.add_argument("--listen-before", type=float, default=2.0, help="nghe trước khi gửi START [s]")
    p.add_argument("--linger", type=float, default=2.0, help="nghe tiếp sau #run_end [s]")
    p.add_argument("--data-port", type=int, default=DATA_PORT)
    p.add_argument("--evt-port", type=int, default=EVT_PORT)
    p.add_argument("--cmd-port", type=int, default=CMD_PORT)
    p.add_argument("--bind", default="0.0.0.0", help="địa chỉ nghe (test dùng 127.0.0.1)")
    p.add_argument("--temp-path", default=TEMP_PATH, help="file nhiệt độ CPU RPi5 (m°C), đọc mỗi 1 s")
    args = p.parse_args(argv)
    if args.duration <= 0:
        p.error("--duration phải > 0")
    if args.out is None:
        args.out = "s06_capture_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
