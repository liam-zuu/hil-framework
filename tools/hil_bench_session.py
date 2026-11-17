#!/usr/bin/env python3
"""
hil_bench_session.py — một buổi vòng kín HIL bằng một lệnh, chạy trên PC (Liam 08/11: "làm 1 2 trước nhé",
việc 2: gom lượt bàn thành một lệnh). Làm đúng trình tự đã chạy tay ở CLOSED_LOOP/2026-11-08_run12, run13
(hướng dẫn `00_OVERVIEW/Setup_Guides/2026-10-30_Huong_dan_Vong_kin_HIL.md` của repo luận văn):

  1. ghi log serial H7 (ST-Link VCP) và ESP32 (USB) suốt buổi, mỗi dòng có giờ PC (`HH:MM:SS.mmm > dòng`),
     mở cổng với DTR/RTS tắt (không reset board);
  2. `--boot`: Pi chạy `run_hil.py --imu-static` 20 s, sau 4 s reset ESP32 bằng RTS, kiểm dòng khởi động
     (`BNO085 init OK`, `Controller:`, `Boot done`), không đạt thì dừng (mã 2);
  3. mỗi `--run TÊN=ĐỐI_SỐ`: trên Pi chạy `tools/agv_logger.py` (log WiFi ESP32, trừ khi `--no-logger`) cùng
     `run_hil.py ĐỐI_SỐ --finite-run --frame-log`, hết lượt dừng logger bằng SIGINT; scp frame log + CSV về;
  4. chấm từng lượt (cũng là `--analyze-only` cho thư mục đã có):
     - L50 (STATUS_CODE L50): `s06_frames_decode.decode()` không có mục nào trong `problems()`, và trong lúc
       lượt chạy dòng `SPIfix: stuck=` của H7 không tăng; kèm `Frame: skipped` không tăng, `Loop: over1ms`;
     - bám setpoint: vận tốc plant trong MOSI là hệ thế giới, xoay về hệ thân xe theo θ rồi so với setpoint
       (bỏ 1 s đầu), duty lớn nhất, số frame |duty| ≥ 0,99, bước duty lớn nhất giữa hai frame;
     - rpm (CSV ESP32): lệch so với trung vị 11 mẫu khi motor bật và setpoint khác 0 (bỏ 1 s đầu);
     - ESP32: `Motor disabled` giữa lượt, `POWERON`, `IMU_INIT_FAIL`, `IMU FAIL STATE`;
     - I8 (Liam 09/11; Codex vòng 1 I8-6..10), chỉ với lượt có `swa_mode` trong manifest frame log:
       status H7 (MISO byte 16 bit 0) bằng bit SWA Pi gửi (MOSI byte 20 bit 3) ở mọi frame mà lệnh đã giữ
       hơn 20 frame, không lần đổi nào H7 không theo (`s06_frames_decode.swa_summary`); ESP32 `Motor enabled`
       đúng 1 lần và có `Motor disabled` sau đó khi Pi có gửi SWA bật, 0 lần khi không. `--boot` chạy omega
       mode nên `--swa auto` = tắt: thấy `Motor enabled` (tính từ lúc khởi chạy Pi, I8-10) thì dừng mã 2.
       Lượt `swa_mode = phase` có đuôi END chưa xong (`swa_tail_complete` không phải true) là không sạch
       (Codex vòng 2 I8-R2-3).
     Chỉ L50 và các lỗi ESP32 làm lượt "không sạch"; số bám setpoint, rpm chỉ báo, không có ngưỡng
     đạt/không (kiểm chức năng, gain chưa tune). Không viết `meta.yaml` (người chạy viết, kèm commit 3 board).

Đầu ra trong `--out`: `h7_COM3.txt`, `esp32_COM5.txt` (tên theo cổng), `pi/`, `frames/`, `session.json`
(lệnh, commit Pi, giờ PC đầu/cuối mỗi lượt, mã thoát, lệnh dừng Pi), `bench_summary.json` (số chấm). Dòng cuối in
`#ket_qua,ma=<0|1|2>`: 0 mọi lượt sạch, 1 có lượt không sạch, 2 lỗi chuẩn bị (cổng, SSH, khởi động ESP32,
thư mục đã có kết quả, tiến trình sót trên Pi).

Chống mất dữ liệu (A, Liam 09/11 "Làm cả A và B"; bài học CLOSED_LOOP/2026-11-09_run05: bốn phiên vào một thư
mục, log serial ba phiên đầu bị ghi đè): mỗi phiên một thư mục. `--out` đã có `session.json`, `bench_summary.json`,
`h7_*.txt`, `esp32_*.txt` hay file trong `frames/`, `pi/` thì dừng mã 2 trước khi mở cổng hay SSH; mọi file của
phiên mở ở chế độ chỉ tạo mới; hai `--run` cùng tên bị từ chối. `flash/` (`tools/h7_flash.py`) được để sẵn.
`--analyze-only --write-summary` ghi `bench_summary_analyze.json`.

Nhận dạng firmware ESP32 (Liam 09/11, CODE): AGV in `#fw sha=.. dirty=.. env=..` lúc khởi động và gửi
`#fw,sha=..` qua UDP ở đầu mỗi lượt (AGV `scripts/build_id.py`, mọi env). Công cụ ghi `boot.esp32_fw`,
`runs[].esp32_fw` vào `session.json` và in ở dòng tổng kết; `--expect-esp32 SHA`: khác SHA, dirty=1 hay không
thấy dòng thì `--boot` dừng mã 2, lượt không sạch (`meta.yaml` ghi commit ESP32 theo dòng này).

Nhận dạng firmware H7 (cùng lời Liam, 2e): H7 in `#fw sha=.. dirty=.. env=Debug` lúc khởi động và mỗi 10 s
(`stm32h7/scripts/fw_id.py`, bước pre-build CubeIDE). `--expect-h7 SHA`: trước lượt đầu chờ tối đa 11 s một dòng
`#fw` của H7; khác SHA, dirty=1 hay không có thì dừng mã 2; lượt nào thấy `#fw` H7 khác thì không sạch. Ghi
`h7_fw` (trước lượt đầu), `runs[].h7_fw`, `h7_fw_all` vào `session.json`.

Logger chế độ đo (A5, Liam 09/11, CODE, STATUS_CODE L61 b): mỗi lượt chạy `agv_logger.py --strict --src-ip IP
--ready-file <pi-dir>/TÊN.ready`; IP = `--esp32-ip`, không có thì dòng `Got IP:` ESP32 in ở serial trong phiên,
không có nữa thì `first` (khóa theo gói đầu). `run_hil.py` chỉ chạy khi file ready đã có (chờ tối đa 5 s, không
có thì không chạy lượt, Pi thoát 2). Lượt không sạch khi logger không sẵn sàng, mã thoát khác 0, `summary.json`
thiếu hay `status` khác `closed`, `mode` không phải `strict`, hoặc logger loại dòng vì lý do khác `source`;
`seq_holes`, số đoạn chỉ báo. `--logger-loose` giữ logger cũ (firmware trước 85 cột).

Dừng Pi giữa lượt (B, cùng lời Liam; bài học run05 b3_kill: `pkill -f` khớp cả shell bọc ngoài, TERM thành STOP
êm): `--pi-kill TÊN=GIÂY` (SIGKILL `run_hil.py` và mọi tiến trình con: Pi chết, không STOP, không frame log),
`--pi-stop TÊN=GIÂY` (SIGTERM: STOP êm). Công cụ đọc PID `run_hil.py` từ `<pi-dir>/TÊN.pid` (do lệnh lượt ghi), kiểm
`/proc/PID/cmdline` có `run_hil.py`, không dò theo chuỗi lệnh. Trước phiên và sau mỗi lượt kiểm trên Pi không còn
`run_hil.py`, `agv_logger.py` (`pgrep -a python3`); còn thì dừng mã 2 và in PID.

Lượt S06 (Liam 15/11, CODE_S06: "Mở rộng hil_bench_session", "Code cả hai"): `--s06 TÊN=PROFILE,GIÂY[,logger-first]`
cho một `--run` chạy `tools/s06_logger.py --esp32 IP --profile PROFILE --duration GIÂY --out <pi-dir>/TÊN_esp` thay
`agv_logger.py` (IP = `--esp32-ip`, không có thì dòng `Got IP:` ESP32 in ở serial; không có nữa thì dừng mã 2). Mặc
định plant trước, logger sau 2 s (mẫu `s06_run.sh` buổi thử 15/11); `logger-first`: logger trước, plant sau 4 s (T5,
mẫu `s06_run_lf.sh`). `--steps` của lượt phải ≥ (GIÂY + 10) × 1000, để plant chạy dài hơn lượt ESP32 (bài học buổi
thử: IMU ngừng, frame không phủ lượt). Sau lượt: scp `TÊN_esp/` về `pi/`, bộ giải ghép report với `esp32_events.txt`
(`match_esp32`), `tools/s06_esp32_check.py` ghi `check/TÊN/`; mã khác 0, `s06_logger` mã thoát khác 0 thì lượt
không sạch. Không dùng cùng `--gain-query`. S06 T3 (thiết kế T3 mục 4.1; hồ sơ t3: Liam 17/11 CODE_S06 "Hồ sơ t3
riêng (Recommended)"): lượt hồ sơ `t3` cần `--omega-schedule` (lịch `rpi5/t3_schedule.py`) và chạy `run_hil.py --marker
t3`; lượt khác `--marker 10hz` (chân đánh dấu cho đồng hồ ESP32 ↔ Pi); thiếu `--marker` thì công cụ tự thêm, khác thì
từ chối.

Usage:
  python tools/hil_bench_session.py --out <thư mục lượt> --pi-dir '~/run14' --boot \
      --run b2_AL1='--test A-L1' --run b3_AC1='--test A-C1' \
      --run b4_PCNT='--traj circle --speed 0.3 --radius 0.5 --duration 60'
  python tools/hil_bench_session.py --out <thư mục lượt mới> --pi-dir '~/run21' \
      --run b5_kill='--test A-L1' --pi-kill b5_kill=11
  python tools/hil_bench_session.py --analyze-only --out <thư mục lượt có frames/>

Giới hạn: Pi chạy không sudo (như run12); không nạp board; ESP32 phải đã nạp đúng env; PuTTY không được mở
cùng cổng. Cửa sổ thời gian của lượt (cho log H7, ESP32) chỉ có khi chạy trực tiếp; `--analyze-only` chấm
log serial cả file nếu không có `session.json`.
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'rpi5'))
import s06_frames_decode as dec                      # noqa: E402
from protocol import decode_state, decode_torques, decode_traj_ref  # noqa: E402

BAD_ARG = re.compile(r'[;&|$`<>\\\n]')
BOOT_NEED = ('BNO085 init OK', 'Controller:', 'Boot done')
ESP_BAD = ('POWERON', 'IMU_INIT_FAIL', 'IMU FAIL STATE')


# ── serial capture ──

class SerialLog(threading.Thread):
    """Lines of one COM port with PC time; DTR/RTS False at open (no reset)."""

    def __init__(self, port, path):
        super().__init__(daemon=True)
        import serial
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = port, 115200, 0.1
        self.s.dtr = False
        self.s.rts = False
        self.s.open()
        try:
            self.f = open(path, 'xb')                # A (Liam 09/11): never overwrite a log of an earlier session
        except FileExistsError:
            self.s.close()
            raise
        self.lines = []                              # (time.time(), text)
        self.lock = threading.Lock()
        self.stop_ev = threading.Event()

    def run(self):
        pending = b''
        while not self.stop_ev.is_set():
            pending += self.s.read(4096)
            while b'\n' in pending:
                line, pending = pending.split(b'\n', 1)
                t = time.time()
                txt = line.rstrip(b'\r')
                ts = datetime.datetime.fromtimestamp(t).strftime('%H:%M:%S.%f')[:-3].encode()
                self.f.write(ts + b' > ' + txt + b'\n')
                self.f.flush()
                with self.lock:
                    self.lines.append((t, txt.decode('utf-8', 'replace')))
        self.s.close()
        self.f.close()

    def pulse_rts(self):
        self.s.rts = True
        time.sleep(0.15)
        self.s.rts = False

    def since(self, t0, t1=None):
        with self.lock:
            return [x for x in self.lines if x[0] >= t0 and (t1 is None or x[0] <= t1)]

    def stop(self):
        self.stop_ev.set()
        self.join(2.0)


# ── analysis (pure functions, tests/test_hil_bench_session.py) ──

def tracking(t_send_ns, mosi, miso, skip_frames=1000):
    """Body-frame plant velocity vs the setpoint carried in MOSI, duty from MISO."""
    S = np.array([decode_state(bytes(m)) for m in mosi])
    R = np.array([decode_traj_ref(bytes(m))[:3] for m in mosi])
    D = np.array([decode_torques(bytes(m)) for m in miso])
    act = np.flatnonzero(np.abs(R).sum(1) > 1e-6)
    if len(act) == 0 or act[-1] - act[0] <= skip_frames + 1:
        return dict(moving_frames=0)
    sl = slice(int(act[0]) + skip_frames, int(act[-1]))
    th, vx, vy, wz = S[sl, 6], S[sl, 7], S[sl, 8], S[sl, 9]
    vb = np.cos(th) * vx + np.sin(th) * vy
    vl = -np.sin(th) * vx + np.cos(th) * vy
    d = D[sl]
    dd = np.abs(np.diff(d, axis=0))
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))  # noqa: E731
    return dict(moving_frames=int(sl.stop - sl.start),
                ref=[float(R[sl, k].mean()) for k in range(3)],
                body=[float(vb.mean()), float(vl.mean()), float(wz.mean())],
                rms_err=[rms(vb - R[sl, 0]), rms(vl - R[sl, 1]), rms(wz - R[sl, 2])],
                wheel_mean=[float(x) for x in S[sl, :4].mean(0)],
                duty_max=float(np.abs(d).max()), duty_sat=int((np.abs(d) >= 0.99).sum()),
                duty_step_max=float(dd.max()) if len(dd) else 0.0)


def rpm_spikes(rows, skip_rows=100, win=5):
    """ESP32 CSV rows (dicts): per wheel max |rpm - median of 2*win+1 rows| while motor on and setpoint set."""
    if not rows:
        return None
    mo = np.array([r.get('motor_on') in ('1', 'true', 'True') for r in rows])
    sp = np.array([abs(float(r['sp_vx'])) + abs(float(r['sp_vy'])) + abs(float(r['sp_wz'])) for r in rows])
    idx = np.flatnonzero(mo & (sp > 1e-6))
    if len(idx) == 0:
        return dict(rows=len(rows), moving_rows=0)
    idx = idx[(idx > idx[0] + skip_rows) & (idx < idx[-1] - 10)]
    out = dict(rows=len(rows), moving_rows=int(len(idx)), wheels={})
    for w in ('fl', 'fr', 'rl', 'rr'):
        rpm = np.array([float(r[f'{w}_rpm']) for r in rows])
        dev = np.array([rpm[i] - np.median(rpm[max(i - win, 0):i + win + 1]) for i in idx])
        out['wheels'][w] = dict(mean=float(rpm[idx].mean()) if len(idx) else None,
                                dev_max=float(np.abs(dev).max()) if len(dev) else 0.0,
                                over10=int((np.abs(dev) > 10).sum()))
    return out


_RE = dict(skipped=re.compile(r'Frame:.* skipped=(\d+)'), stuck=re.compile(r'SPIfix: stuck=(\d+)'),
           loop=re.compile(r'Loop: max=([\d.]+) us .*over1ms=(\d+)'))


def h7_window(lines):
    """H7 lines of one run: rise of the cumulative 'skipped' and 'stuck' counters, Loop: max and over1ms."""
    sk = [int(m[1]) for _, s in lines for m in [_RE['skipped'].search(s)] if m]
    st = [int(m[1]) for _, s in lines for m in [_RE['stuck'].search(s)] if m]
    lp = [(float(m[1]), int(m[2])) for _, s in lines for m in [_RE['loop'].search(s)] if m]
    return dict(lines=len(sk), skipped_rise=(sk[-1] - sk[0]) if sk else None,
                stuck_rise=(st[-1] - st[0]) if st else None,
                loop_max_us=max(x[0] for x in lp) if lp else None,
                loop_over1ms=sum(x[1] for x in lp) if lp else None)


FW_RE = re.compile(r'#fw[ ,]sha=([0-9a-f]+|unknown)[ ,]dirty=([01])[ ,]env=([A-Za-z0-9_\-]+)')


def fw_lines(lines):
    """ESP32 firmware id (Liam 09/11, CODE: AGV scripts/build_id.py, every env): the boot line
    `#fw sha=.. dirty=.. env=..` (serial) and the UDP line `#fw,sha=..,dirty=..,env=..` (agv_logger
    events file, 3 copies at each run start). Distinct ids in order: [{sha, dirty, env}]."""
    out = []
    for s in lines:
        m = FW_RE.search(s)
        if m:
            d = dict(sha=m[1], dirty=int(m[2]), env=m[3])
            if d not in out:
                out.append(d)
    return out


def fw_problem(fws, expect):
    """Why the ESP32 firmware seen does not match --expect-esp32 (a SHA prefix), else None. Not seen, more
    than one id, another SHA, or dirty=1 are all refused: meta.yaml takes the commit from this line."""
    if not expect:
        return None
    if not fws:
        return 'không thấy dòng #fw của ESP32 (firmware chưa có nhận dạng, hay log thiếu)'
    if len(fws) > 1:
        return f'ESP32 có {len(fws)} nhận dạng firmware khác nhau: {fws}'
    f, e = fws[0], expect.lower()
    if f['sha'] == 'unknown' or not (f['sha'].startswith(e) or e.startswith(f['sha'])):   # either is a prefix
        return f"ESP32 chạy {f['sha']} (env {f['env']}), --expect-esp32 {expect}"
    if f['dirty']:
        return f"ESP32 build từ cây bẩn (dirty=1, {f['sha']} env {f['env']})"
    return None


LIFTED_TEST_RE = re.compile(r'--test\s+A-K\d')
LIFTED_ENV_TAG = 'wheels_lifted'
CAMPAIGN_ENVS = ('esp32-s3-devkitc-1', 'ladrc')     # AGV env chinh (PID) va ladrc: binary chien dich (QD 6)


def lifted_env_problem(args, fws, campaign_env=False):
    """L55 (Liam 09/11, CODE: the lifted-wheel flag is chosen by env): a lifted test (--test A-K*) must run on
    an ESP32 env with wheels_lifted in its name (AGV env wheels_lifted / ladrc_wheels_lifted), and a ground
    run must not; None when it fits or the ESP32 id is not known (fw_problem judges a missing id).
    campaign_env (--campaign-env, C5; Liam 10/11 CODE local_85caf8b8 "Them co C5", after "Env chien dich"
    for C5 the same day): a lifted test must run on a campaign env (CAMPAIGN_ENVS) instead; the
    wheels_lifted flag only turns off the gyro-bias estimator, which is off anyway while sigma = 0."""
    if not fws:
        return None
    lifted_run = bool(LIFTED_TEST_RE.search(args or ''))
    envs = sorted({f['env'] for f in fws})
    lifted_env = [LIFTED_ENV_TAG in e for e in envs]
    if lifted_run and campaign_env:
        if not all(e in CAMPAIGN_ENVS for e in envs):
            return f'bài kê bánh C5 (--campaign-env) nhưng ESP32 chạy env {envs}, cần env chiến dịch {list(CAMPAIGN_ENVS)}'
        return None
    if lifted_run and not all(lifted_env):
        return f'bài kê bánh (A-K) nhưng ESP32 chạy env {envs}, cần env *{LIFTED_ENV_TAG} (C5: --campaign-env)'
    if not lifted_run and any(lifted_env):
        return f'bài trên sàn nhưng ESP32 chạy env kê bánh {envs}'
    return None


H7_FW_WAIT_S = 11.0                 # --expect-h7: the H7 prints #fw every 10 s (main.c debug block)
IP_RE = re.compile(r'Got IP: (\d{1,3}(?:\.\d{1,3}){3})')
LOGGER_EXIT_RE = re.compile(r'^logger_exit=(\d+)', re.M)
LOGGER_SAFE_REJECTS = ('source',)     # datagrams from another sender: refused by the source lock, data unharmed


GAIN_SEND = os.path.join(HERE, 'gain_send.py')
GAIN_QUERY_TIMEOUT_S = 20.0


def gain_query(ip, path, run=subprocess.run):
    """S05 4.7 (iv), R3V-9 (R134 package, Liam 14/11 CODE_R134 local_eb9f3b5c "duyet goi sua code
    fix/r134-answers"): gain_send.py --query to the ESP32 (UDP 5014; answered with the motors on or off,
    AGV 16702ff main.c 623-627), its whole output to `path` (new file); returns the fields of the
    '#gain_running' line (boot, mac, ...; gain_line_check.read_query: one answer line, output ending with
    '#ket_qua,ma=0'), or None (no answer, refused, no IP)."""
    import gain_line_check as glc
    t_start = time.time()
    if not ip:
        txt = '# no ESP32 IP (--esp32-ip or a "Got IP:" line): boot not asked\n#ket_qua,ma=1\n'
    else:
        try:
            p = run([sys.executable, GAIN_SEND, '--ip', ip, '--query'], capture_output=True, text=True,
                    timeout=GAIN_QUERY_TIMEOUT_S)
            txt = (p.stdout or '') + (p.stderr or '')
        except (OSError, subprocess.TimeoutExpired) as e:
            txt = f'# gain_send.py --query failed: {e}\n#ket_qua,ma=1\n'
    with open(path, 'x', encoding='utf-8') as f:
        f.write(glc.query_pc_line(t_start, time.time()) + txt)      # R134C2-1: PC times of the question
    return glc.read_query(path)


def gain_set(ip, gain_file, path, run=subprocess.run):
    """B8-1 B81-14 (S05 v3.20 4.7 (iv); Liam 14/11 CODE_R134 local_076fc0aa, hop chon "Cong cu tu dat va luu
    (Recommended)"): gain_send.py --file <gain_file> to the ESP32 just before the run's question 'before' (the
    ESP32 applies a set only with the motors off), its whole output to `path` (new file, never overwritten);
    returns the fields of its '# ack' line (gain_line_check.read_ack: ok=1, output ending with '#ket_qua,ma=0'),
    or None (not applied, refused, no answer, no IP)."""
    import gain_line_check as glc
    if not ip:
        txt = '# no ESP32 IP (--esp32-ip or a "Got IP:" line): gain not set\n#ket_qua,ma=1\n'
    else:
        try:
            p = run([sys.executable, GAIN_SEND, '--ip', ip, '--file', gain_file], capture_output=True, text=True,
                    timeout=GAIN_QUERY_TIMEOUT_S)
            txt = (p.stdout or '') + (p.stderr or '')
        except (OSError, subprocess.TimeoutExpired) as e:
            txt = f'# gain_send.py --file failed: {e}\n#ket_qua,ma=1\n'
    with open(path, 'x', encoding='utf-8') as f:
        f.write(txt)
    return glc.read_ack(path)


def ack_of_run(out_dir, name):
    """The saved '#gain_ack' of a run's gain setting (pi/<name>_gain_ack.txt), read again (None = missing)."""
    import gain_line_check as glc
    return glc.read_ack(os.path.join(out_dir, 'pi', f'{name}_gain_ack.txt'))


def timed_query(ip, path):
    """gain_query with the PC times around it: (answer fields or None, record for session.json)."""
    t_a = time.time()
    q = gain_query(ip, path)
    return q, dict(t_start=t_a, t_end=time.time(), boot=q and q['boot'], mac=q and q.get('mac'),
                   file=os.path.basename(path))


def boot_problem(before, after, out_dir=None, name=None):
    """None, or why the two answers (gain_query fields, None = missing) do not show one boot of one board
    over the run (R134C-1): same boot and MAC, and the boot of the run's '#us,ev=start' in its UDP log
    pi/<name>.csv (gain_line_check.run_boot_of; out_dir None: that last check is skipped)."""
    import gain_line_check as glc
    run_boot, why = (None, None) if out_dir is None else \
        glc.run_boot_of(os.path.join(out_dir, 'pi', f'{name}.csv'))
    bad = glc.query_problem(before, after, run_boot, check_run_boot=out_dir is not None)
    if bad and why and run_boot is None and before is not None and after is not None:
        bad += f' ({why})'
    return bad


def queries_of_run(out_dir, name):
    """The two saved answers of a run (pi/<name>_gain_query_before.txt, _after.txt), read again."""
    import gain_line_check as glc
    return tuple(glc.read_query(os.path.join(out_dir, 'pi', f'{name}_gain_query_{w}.txt')) for w in ('before', 'after'))


def esp32_ip(lines):
    """A5: the last `Got IP: a.b.c.d` the ESP32 printed on its serial port (AGV wifi.c), else None."""
    ips = [m[1] for s in lines for m in [IP_RE.search(s)] if m]
    return ips[-1] if ips else None


def logger_src(arg_ip, esp_lines):
    """A5: (src_ip, source) for agv_logger --strict --src-ip: --esp32-ip, else the ESP32 serial `Got IP:` of
    this session, else 'first' (agv_logger locks on the sender of the first packet)."""
    if arg_ip:
        return arg_ip, 'arg'
    ip = esp32_ip(esp_lines)
    return (ip, 'serial') if ip else ('first', 'first_packet')


def logger_problems(summary, pi_txt, strict):
    """A5 (Liam 09/11, CODE, STATUS_CODE L61 b): why the WiFi log of a run is not usable, [] if it is.
    Not usable: the logger never got ready (strict), its exit code is not 0, no summary or status not
    'closed', mode not 'strict' when asked, lines refused for any reason but the source lock (a refused
    data line is a line missing from the CSV). seq holes and segments are reported, not judged here."""
    why = []
    if 'logger_not_ready' in pi_txt:
        why.append('logger không sẵn sàng (không có file ready trong 5 s): lượt không chạy')
    m = LOGGER_EXIT_RE.search(pi_txt)
    if m and m[1] != '0':
        why.append(f'logger mã thoát {m[1]}')
    if summary is None:
        return why + ['không có summary.json của logger']
    if summary.get('status') != 'closed':
        why.append(f"logger status {summary.get('status')}")
    if strict and summary.get('mode') != 'strict':
        why.append(f"logger mode {summary.get('mode')} (cần strict)")
    bad = {k: n for k, n in (summary.get('rejects') or {}).items() if k not in LOGGER_SAFE_REJECTS and n}
    if bad:
        why.append(f'logger loại dòng: {bad}')
    return why


def logger_info(summary):
    """A5: short logger record for bench_summary.json and the summary line."""
    if summary is None:
        return None
    seg = summary.get('segments') or []
    return dict(mode=summary.get('mode'), src_ip=summary.get('src_ip'), src_lock=summary.get('src_lock'),
                status=summary.get('status'), data_lines=summary.get('data_lines'),
                rejects=summary.get('rejects'), segments=len(seg),
                seq_holes=sum(s.get('seq_holes', 0) for s in seg))


def boot_motor_on(lines):
    """I8-10: `Motor enabled` lines of the ESP32 in the --boot step (texts)."""
    return sum(1 for s in lines if 'Motor enabled' in s)


def esp_window(lines):
    """ESP32 lines of one run: motor disabled before the last line (inside the run) and boot/IMU failures.
    I8: motor_enabled = number of `Motor enabled` lines; off_after_on = a `Motor disabled` line comes after
    the last `Motor enabled` line."""
    txt = [s for _, s in lines]
    dis = [i for i, s in enumerate(txt) if 'Motor disabled' in s]
    en = [i for i, s in enumerate(txt) if 'Motor enabled' in s]
    return dict(motor_disabled_inside=len(dis[:-1]) if dis else 0,
                motor_enabled=len(en),
                off_after_on=bool(en and dis and dis[-1] > en[-1]),
                bad={k: sum(1 for s in txt if k in s) for k in ESP_BAD})


def judge(dec_problems, h7, esp, swa=None, i8=False, man=None):
    """Run clean: L50 (no decode problem, SPIfix stuck did not rise) and no ESP32 failure. Reasons listed.
    I8 (Liam 09/11; Codex round 1 I8-6, -7, -9): only for a run whose frame-log manifest has swa_mode
    (i8=True): the H7 status follows the SWA bit (s06_frames_decode.swa_summary: no held mismatch, no
    unfollowed change), and the ESP32 turns the motors on once and off after it when the Pi sent SWA on,
    never when it did not. Logs from before I8 keep the older rules.
    I8-R2-3 (Codex round 2): an I8 run in swa_mode 'phase' whose END tail did not finish (manifest
    swa_tail_complete not true, also when the key is missing) is not clean: the off command may not have
    been followed to the end."""
    why = list(dec_problems)
    man = man or {}
    if i8 and man.get('swa_mode') == 'phase' and man.get('swa_tail_complete') is not True:
        why.append(f"đuôi END chưa xong (swa_tail_sent {man.get('swa_tail_sent')}, "
                   f"swa_tail_complete {man.get('swa_tail_complete')})")
    if h7 and h7.get('stuck_rise'):
        why.append(f"SPIfix stuck tăng {h7['stuck_rise']}")
    if h7 and h7.get('skipped_rise'):
        why.append(f"H7 Frame skipped tăng {h7['skipped_rise']}")
    if esp:
        if esp['motor_disabled_inside']:
            why.append(f"ESP32 Motor disabled giữa lượt {esp['motor_disabled_inside']} lần")
        why += [f'ESP32 {k} {n} lần' for k, n in esp['bad'].items() if n]
    if i8 and swa:
        if swa.get('held_mismatch'):
            why.append(f"H7 status SWA khác lệnh Pi ở {swa['held_mismatch']} frame (lệnh đã giữ "
                       f"> {swa.get('settle_frames')} frame; đầu tiên {swa.get('mismatch_first', [])[:3]})")
        if swa.get('unfollowed'):
            why.append(f"H7 không làm theo bit SWA ({swa['unfollowed']} lần đổi không thấy trong status)")
        if esp:
            want = 1 if swa.get('cmd_on_frames') else 0
            if esp['motor_enabled'] != want:
                why.append(f"ESP32 Motor enabled {esp['motor_enabled']} lần (cần {want})")
            elif want and not esp['off_after_on']:
                why.append('ESP32 không có Motor disabled sau Motor enabled trong lượt')
    return (not why), why


def analyze_run(out, name, h7_lines=None, esp_lines=None, s06=False, check_sub='check'):
    """s06: an S06 run (--s06): the decoder also matches the H7 reports with the ESP32 events of
    pi/<name>_esp (match_esp32), and tools/s06_esp32_check.py judges that log; any code but 0 = not clean."""
    z = np.load(os.path.join(out, 'frames', f'{name}_frames.npz'))
    ev = os.path.join(out, 'pi', f'{name}_esp', 'esp32_events.txt')
    pre = []
    try:
        res = dec.decode(z['t_send_ns'], z['mosi'], z['miso'], ev if s06 and os.path.exists(ev) else None)
    except dec.stm.RunProblem as ex:              # T3T-8: e.g. two #run_start in the events: judged, not raised
        pre = [f'không xác định được một lượt ở log ESP32 ({ex}): không ghép report']
        res = dec.decode(z['t_send_ns'], z['mosi'], z['miso'], None)
    probs = pre + dec.problems(res)
    if s06:
        probs += dec.conditions(res)              # T3T-6: missing evidence (±1 s pad, TICK) is not clean either
        if not os.path.exists(ev):
            probs.append(f'lượt S06 không có pi/{name}_esp/esp32_events.txt')
        c = s06_check(out, name, check_sub) if check_sub else 0
        if c != 0:
            probs.append(f's06_esp32_check mã {c} ({check_sub}/{name}/)')
    trk = tracking(z['t_send_ns'], z['mosi'], z['miso'])
    p = os.path.join(out, 'pi', f'{name}.csv')
    rpm = rpm_spikes(list(csv.DictReader(open(p, encoding='utf-8')))) if os.path.exists(p) else None
    h7 = h7_window(h7_lines) if h7_lines is not None else None
    esp = esp_window(esp_lines) if esp_lines is not None else None
    man = json.loads(str(z['manifest'])) if 'manifest' in z.files else {}
    ok, why = judge(probs, h7, esp, res.get('swa'), i8='swa_mode' in man, man=man)
    f = res['frames']
    return dict(name=name, clean=ok, why=why,
                frames=dict(n=f['frames'], miso_crc_bad=f['miso_crc_bad'], empty_in_run=f['empty_in_run'],
                            unanswered=f['unanswered'], start_leftover=f.get('start_leftover')),
                tracking=trk, rpm=rpm, h7=h7, esp32=esp, swa=res.get('swa'))


def summary_line(r):
    t = r['tracking']
    s = f"{r['name']}: {'SẠCH' if r['clean'] else 'KHÔNG SẠCH'}"
    if t.get('moving_frames'):
        s += (f" | vx {t['body'][0]:.4f}/{t['ref'][0]:.3f} wz {t['body'][2]:.4f}/{t['ref'][2]:.3f}"
              f" (RMS {t['rms_err'][0]:.4f}, {t['rms_err'][2]:.4f}) duty max {t['duty_max']:.2f}"
              f" sat {t['duty_sat']} bước {t['duty_step_max']:.3f}")
    if r['rpm'] and r['rpm'].get('wheels'):
        s += f" | rpm lệch max {max(w['dev_max'] for w in r['rpm']['wheels'].values()):.1f}"
    if r['h7'] and r['h7']['loop_max_us'] is not None:
        s += f" | H7 Loop max {r['h7']['loop_max_us']:.1f} µs"
    for f in r.get('esp32_fw') or []:
        s += f" | ESP32 #fw {f['sha']} {f['env']}" + (' DIRTY' if f['dirty'] else '')
    for f in r.get('h7_fw') or []:
        s += f" | H7 #fw {f['sha']} {f['env']}" + (' DIRTY' if f['dirty'] else '')
    lg = r.get('logger')
    if lg:
        s += f" | logger {lg['mode']} src {lg['src_ip']} seq hở {lg['seq_holes']}"
    sw = r.get('swa')
    if sw and sw.get('cmd_on_frames'):
        s += (f" | SWA bật {sw['cmd_on_frames']} frame, đổi {len(sw['cmd_changes'])} lần,"
              f" H7 theo sau max {sw['follow_lag_max']} frame")
    if r['why']:
        s += ' | ' + '; '.join(r['why'])
    return s


# ── bench session ──

def ssh(host, cmd, timeout=None):
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', host, cmd],
                       capture_output=True, timeout=timeout)
    return r.returncode, r.stdout.decode('utf-8', 'replace') + r.stderr.decode('utf-8', 'replace')


def scp(host, remote, local_dir, recursive=False):
    return subprocess.run(['scp', '-o', 'BatchMode=yes', '-q'] + (['-r'] if recursive else [])
                          + [f'{host}:{remote}', local_dir]).returncode


# ── S06 runs with tools/s06_logger.py (Liam 15/11, CODE_S06 'Mo rong hil_bench_session', 'Code ca hai') ──
# Template: the trial scripts of S06/2026-11-15_run01_thu/flash (s06_run.sh: plant first, logger 2 s after;
# s06_run_lf.sh: logger first, plant 4 s after). Lesson of that run (NOTES.md mục 3): the plant must outlast the
# ESP32 run, or the IMU stops and the frames do not cover the run (S38).
S06_PROFILES = ('zero', 'heavy', 't6', 't3')     # t3: Liam 17/11 CODE_S06 "Hồ sơ t3 riêng (Recommended)"
S06_LOGGER_AFTER_S = 2          # plant first: the logger starts this long after run_hil.py
S06_PLANT_AFTER_S = 4           # logger first: run_hil.py starts this long after the logger
S06_PLANT_MARGIN_S = 10         # [ĐỀ XUẤT CODE_S06]: --steps >= (logger --duration + this) x 1000
# T3T-5 (Codex 15/11; Liam 'Sua ca 10 + vong 2'): an S06 run's ssh limit is at least the plant time (--steps / 1 kHz)
# plus this (start, the logger's START wait and linger, frame-log save) [ĐỀ XUẤT CODE_S06, not measured]
S06_TIMEOUT_EXTRA_S = 120


def marker_of(args: str):
    """S06 T3: --marker of a run_hil.py argument string (shlex, `--marker X` or `--marker=X`, the last wins);
    None without one."""
    import shlex
    toks, val = shlex.split(args), None
    for i, t in enumerate(toks):
        if t == '--marker':
            val = toks[i + 1] if i + 1 < len(toks) else ''
        elif t.startswith('--marker='):
            val = t[len('--marker='):]
    return val


def has_opt(args: str, opt: str) -> bool:
    import shlex
    return any(t == opt or t.startswith(opt + '=') for t in shlex.split(args))


def steps_of(args: str):
    """T3T-9: --steps of a run_hil.py argument string as argparse reads it: shlex tokens, `--steps N` or
    `--steps=N`, the LAST one wins. None without one; ValueError on a value that is not a whole number."""
    import shlex
    toks, val = shlex.split(args), None
    for i, t in enumerate(toks):
        if t == '--steps':
            if i + 1 >= len(toks):
                raise ValueError('--steps không có giá trị')
            val = toks[i + 1]
        elif t.startswith('--steps='):
            val = t[len('--steps='):]
    if val is None:
        return None
    try:                                     # T3T-9 (vòng 2): as argparse type=int: int() takes '+70000', '70_000'
        return int(val)
    except ValueError:
        raise ValueError(f'--steps {val!r} không phải số nguyên') from None


def plan_s06(specs, plan, gain_query=False):
    """--s06 TÊN=PROFILE,GIÂY[,logger-first] -> {name: dict(profile, duration, logger_first, steps)}. Raises
    ValueError on a run not in --run, a run given twice, a bad profile or duration, run_hil.py args without --steps
    or with fewer than (GIÂY + S06_PLANT_MARGIN_S) x 1000 steps (as run_hil.py's argparse reads them, T3T-9), or
    --gain-query (it reads the agv_logger UDP log)."""
    args_of = {n: a for n, a, _ in plan}
    out = {}
    for spec in specs:
        name, _, rest = spec.partition('=')
        parts = rest.split(',')
        if name not in args_of or name in out or len(parts) not in (2, 3) or parts[0] not in S06_PROFILES \
                or not parts[1].isdigit() or int(parts[1]) <= 0 or (len(parts) == 3 and parts[2] != 'logger-first'):
            raise ValueError(f'--s06 không hợp lệ: {spec!r} (cần TÊN_LƯỢT=zero|heavy|t6|t3,GIÂY[,logger-first], '
                             'mỗi lượt một lần, lượt có ở --run)')
        dur = int(parts[1])
        steps = steps_of(args_of[name])
        need = (dur + S06_PLANT_MARGIN_S) * 1000
        if steps is None or steps < need:
            raise ValueError(f'--s06 {name}: run_hil.py cần --steps ≥ {need} (lượt ESP32 {dur} s + '
                             f'{S06_PLANT_MARGIN_S} s), có {steps if steps is not None else "không có --steps"}')
        if gain_query:
            raise ValueError('--gain-query không dùng với lượt --s06 (cần log UDP của agv_logger)')
        # S06 T3 (thiết kế T3 mục 4.1): chân đánh dấu ở mọi lượt S06; t3 nâng ở frame có bậc, lượt khác 10 Hz.
        # Thiếu --marker thì thêm (không có thì Pi phát xung mỗi frame: 1000 ngắt/s ở GPIO38 của s06_load)
        want = 't3' if parts[0] == 't3' else '10hz'
        got = marker_of(args_of[name])
        if got not in (None, want):
            raise ValueError(f'--s06 {name}: hồ sơ {parts[0]} cần run_hil.py --marker {want}, có --marker {got}')
        if parts[0] == 't3' and not has_opt(args_of[name], '--omega-schedule'):
            raise ValueError(f'--s06 {name}: hồ sơ t3 cần run_hil.py --omega-schedule <T3_schedule.csv>')
        if parts[0] == 't3' and len(parts) != 3:
            # T3C-14 (Codex vòng 1): lịch T3 có bậc từ giây 1; plant trước thì bậc đầu chạy khi ESP32 chưa START
            raise ValueError(f'--s06 {name}: hồ sơ t3 cần logger-first (ESP32 bật ngắt PF3 trước bậc đầu)')
        out[name] = dict(profile=parts[0], duration=dur, logger_first=len(parts) == 3, steps=steps,
                         marker=want, add_marker=got is None)
    return out


def s06_timeout(run_timeout: float, s06: dict) -> float:
    """T3T-5: ssh limit of an S06 run: --run-timeout, raised to the plant time + S06_TIMEOUT_EXTRA_S."""
    return max(run_timeout, s06['steps'] / 1000.0 + S06_TIMEOUT_EXTRA_S)


def s06_run_cmd(pi_dir, name, args, s06, esp_ip):
    """Pi command of one S06 run: run_hil.py (PID file <name>.pid as run_cmd, B) and tools/s06_logger.py (PID file
    <name>_logger.pid, T3T-5) into <pi_dir>/<name>_esp, in the order of `s06`; prints logger_exit=, exits with the
    run_hil.py code. T3T-1: no --finite-run (an S06 run ends by --steps; --finite-run excludes --steps,
    --load-schedule and the omega modes). T3T-4: exit 2 before starting anything when an output of this run name
    is already in pi_dir (an earlier session with the same --pi-dir)."""
    outs = ' '.join(f'{pi_dir}/{name}{s}' for s in ('_esp', '_logger.txt', '_frames.npz'))
    if s06.get('add_marker'):
        args = f'{args} --marker {s06["marker"]}'
    guard = (f'mkdir -p {pi_dir} || exit 2; for f in {outs}; do [ -e "$f" ] && {{ echo "s06_pi_exists=$f"; exit 2; }}; '
             f'done; rm -f {pi_dir}/{name}.pid {pi_dir}/{name}_logger.pid; ')
    hil = (f'cd ~/hil-framework/rpi5 || exit 2; python3 run_hil.py {args} '
           f'--frame-log {pi_dir}/{name}_frames.npz & H=$!; echo $H > {pi_dir}/{name}.pid')
    log = (f'cd ~/hil-framework || exit 2; python3 tools/s06_logger.py --esp32 {esp_ip} --profile {s06["profile"]} '
           f'--duration {s06["duration"]} --out {pi_dir}/{name}_esp > {pi_dir}/{name}_logger.txt 2>&1 & L=$!; '
           f'echo $L > {pi_dir}/{name}_logger.pid')
    if s06['logger_first']:
        return (f'{guard}{log}; sleep {S06_PLANT_AFTER_S}; {hil}; wait $H; RC=$?; '
                f'wait $L; echo logger_exit=$?; exit $RC')
    return (f'{guard}{hil}; sleep {S06_LOGGER_AFTER_S}; {log}; wait $L; echo logger_exit=$?; '
            f'wait $H; RC=$?; exit $RC')


def s06_cleanup_cmd(pi_dir, name):
    """T3T-5: after an S06 run that did not end cleanly (ssh time-out, lost connection, non-zero code): for the
    run_hil.py and s06_logger.py of this run, by their PID files, when still alive and the PID is that script (as
    pi_stop_cmd checks /proc/PID/cmdline). T3T-R2-1: first list its process tree (`pgrep -P`, recursive, as
    pi_stop_cmd kind 'kill': run_hil.py runs the plant in multiprocessing 'spawn' children, whose cmdline does not
    name run_hil.py, and a child left by a dead parent can no longer be found from the PID file), then SIGTERM the
    script only (run_hil.py turns it into STOP), wait up to 5 s for the whole tree, then SIGKILL every process of
    that tree still alive; prints what it did."""
    # d() prints each child before going down into it: a function variable in bash is global, so printing $c after
    # the recursive call would print the grandchild's PID a second time in place of the child's.
    tree = 'd() { for c in $(pgrep -P "$1"); do echo "$c"; d "$c"; done; }; '
    alive = 'a() { A=""; for c in $T; do kill -0 $c 2>/dev/null && A="$A $c"; done; }; '
    one = ('P=$(cat {f} 2>/dev/null); if [ -n "$P" ] && kill -0 "$P" 2>/dev/null && '
           "tr '\\0' ' ' < /proc/$P/cmdline | grep -q '{s}'; then T=\"$P $(d $P)\"; "
           'echo "stop {s} pid=$P tree=$(echo $T)"; kill -TERM $P; '
           'for i in 1 2 3 4 5 6 7 8 9 10; do a; [ -z "$A" ] && break; sleep 0.5; done; a; '
           '[ -n "$A" ] && {{ echo "kill {s} pids=$(echo $A)"; kill -KILL $A; }}; fi; ')
    return (tree + alive + one.format(f=f'{pi_dir}/{name}.pid', s='run_hil.py')
            + one.format(f=f'{pi_dir}/{name}_logger.pid', s='s06_logger.py') + 'true')


def s06_check(out, name, sub='check'):
    """tools/s06_esp32_check.py on pi/<name>_esp, output in <sub>/<name> (not inside the raw folder, NOTES.md
    mục 5; --analyze-only: check_analyze/, the session's check/ is kept). Returns its exit code (2 when the
    folder is missing)."""
    d = os.path.join(out, 'pi', f'{name}_esp')
    if not os.path.isdir(d):
        return 2
    import contextlib
    import s06_esp32_check as sec
    o = os.path.join(out, sub, name)
    os.makedirs(o, exist_ok=True)
    with open(os.path.join(o, 'stdout.txt'), 'w' if sub != 'check' else 'x', encoding='utf-8') as f, contextlib.redirect_stdout(f):
        return sec.main([d, '--out', o])


def run_cmd(pi_dir, name, args, logger, src_ip=None):
    # A5 (Liam 09/11, CODE, STATUS_CODE L61 b): src_ip given = measurement mode: agv_logger --strict
    # --src-ip <src_ip> --ready-file <pi_dir>/<name>.ready, and run_hil.py starts only once the ready file
    # exists (the logger has every output open); none within 5 s (or the logger died) = no run, exit 2.
    # src_ip None = the old loose logger, unchanged.
    # B (Liam 09/11, "Làm cả A và B"): run_hil.py runs as a background job of its own (a single command after
    # `;`, so $! is its PID, not a subshell's), the PID goes to <pi_dir>/<name>.pid for --pi-kill/--pi-stop,
    # and `wait` gives its exit code (137 when killed). No stop by `pkill -f <pattern>`: that matched the
    # wrapper shell too (CLOSED_LOOP/2026-11-09_run05, b3_kill).
    hil = (f'rm -f {pi_dir}/{name}.pid; cd ~/hil-framework/rpi5 || exit 2; python3 run_hil.py {args} --finite-run '
           f'--frame-log {pi_dir}/{name}_frames.npz & H=$!; echo $H > {pi_dir}/{name}.pid; wait $H; RC=$?')
    if not logger:
        return f'mkdir -p {pi_dir} || exit 2; {hil}; exit $RC'
    # `&` on the logger alone (a `;` before it): with `a && b && logger &` the whole list went to a background
    # subshell, $! was that shell, and a non-interactive shell ignores SIGINT in background jobs, so the
    # logger never stopped and ssh waited for ever (2026-11-09_run01). agv_logger closes on SIGINT/SIGTERM.
    ready = f'{pi_dir}/{name}.ready'
    strict = f'--strict --src-ip {src_ip} --ready-file {ready} ' if src_ip else ''
    wait_ready = (f'for i in $(seq 1 50); do [ -e {ready} ] && break; kill -0 $L 2>/dev/null || break; sleep 0.1; '
                  f'done; [ -e {ready} ] || {{ echo logger_not_ready; kill -TERM $L 2>/dev/null; wait $L; '
                  f'echo logger_exit=$?; exit 2; }}; ') if src_ip else ''
    return (f'mkdir -p {pi_dir} && cd ~/hil-framework || exit 2; python3 tools/agv_logger.py --out {pi_dir}/{name}.csv '
            f'{strict}> {pi_dir}/{name}_logger.txt 2>&1 & L=$!; {wait_ready}{hil}; sleep 1; kill -INT $L; '
            f'for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $L 2>/dev/null || break; sleep 0.5; done; '
            f'kill -TERM $L 2>/dev/null; wait $L; echo logger_exit=$?; exit $RC')


def pi_stop_cmd(pi_dir, name, kind):
    """B: stop the run_hil.py of run `name` by the PID in <pi_dir>/<name>.pid (written by run_cmd).
    kind 'kill' = SIGKILL to it and every descendant (the Pi dies: no STOP, no END tail, no frame log);
    'stop' = SIGTERM to it only (run_hil.py turns it into STOP: zero setpoint, then the end of the run).
    Refuses (exit 3) without a PID file or a live process, (exit 4) when that PID is not run_hil.py.
    d() prints each child before going down into it (Liam 16/11, CODE_S06 'Sua, kem test'): a function variable in
    bash is global, so printing $c after the recursive call printed a grandchild's PID in place of its parent's,
    and a child with children of its own got no signal."""
    sig = {'kill': 'KILL', 'stop': 'TERM'}[kind]
    tgt = '$(d $P) $P' if kind == 'kill' else '$P'
    return (f'P=$(cat {pi_dir}/{name}.pid 2>/dev/null); [ -n "$P" ] && kill -0 "$P" 2>/dev/null || exit 3; '
            f"tr '\\0' ' ' < /proc/$P/cmdline | grep -q 'run_hil.py' || exit 4; "
            f'd() {{ for c in $(pgrep -P "$1"); do echo "$c"; d "$c"; done; }}; T="{tgt}"; '
            f'echo "pid=$P targets=$(echo $T)"; date +%s.%N; kill -{sig} $T')


LEFT_CMD = 'pgrep -u "$(id -u)" -a python3'          # process name python3: the ssh bash itself never matches
# T3T-R2-1: and multiprocessing children (run_hil.py sets 'spawn': `python3 -c from multiprocessing.spawn import
# spawn_main; ...`, plus multiprocessing.resource_tracker): a plant child left by a dead run_hil.py still holds the SPI
LEFT_PAT = ('run_hil.py', 'agv_logger.py', 's06_logger.py', 'from multiprocessing.')


def leftovers(pgrep_text):
    """B: `pgrep -a python3` lines of run_hil.py / agv_logger.py / s06_logger.py processes, and of multiprocessing
    child processes (T3T-R2-1), still on the Pi."""
    return [ln.strip() for ln in pgrep_text.splitlines() if any(p in ln for p in LEFT_PAT)]


def existing_results(out):
    """A (Liam 09/11): files of an earlier bench session in `out` (a bench run never writes into them).
    flash/ (tools/h7_flash.py, run before the bench) and anything else the operator put there are allowed."""
    found = [os.path.join(out, f) for f in ('session.json', 'bench_summary.json') if os.path.exists(os.path.join(out, f))]
    found += glob.glob(os.path.join(out, 'h7_*.txt')) + glob.glob(os.path.join(out, 'esp32_*.txt'))
    for sub in ('frames', 'pi', 'check'):
        d = os.path.join(out, sub)
        if os.path.isdir(d):
            found += [os.path.join(d, f) for f in os.listdir(d)]
    return sorted(found)


def plan_runs(runs, kills=(), stops=()):
    """Run specs -> [(name, args, stop)], stop = None or (kind, seconds). Raises ValueError on a bad name,
    a bad argument, a repeated name (its Pi files and frame log would be overwritten, A), or a --pi-kill /
    --pi-stop for a run that is not there, given twice, or not after a positive time."""
    out, names = [], set()
    for spec in runs:
        name, _, args = spec.partition('=')
        if not re.fullmatch(r'[A-Za-z0-9_\-]+', name) or BAD_ARG.search(args):
            raise ValueError(f'--run không hợp lệ: {spec!r}')
        if name in names:
            raise ValueError(f'--run trùng tên {name!r} (file Pi, frame log sẽ bị ghi đè)')
        names.add(name)
        out.append([name, args, None])
    by = {r[0]: r for r in out}
    for kind, specs in (('kill', kills), ('stop', stops)):
        for spec in specs:
            name, _, s = spec.partition('=')
            try:
                sec = float(s)
            except ValueError:
                sec = -1.0
            if name not in by or not sec > 0:
                raise ValueError(f'--pi-{kind} không hợp lệ: {spec!r} (cần TÊN_LƯỢT=GIÂY, giây > 0)')
            if by[name][2] is not None:
                raise ValueError(f'lượt {name!r} có hai lệnh dừng')
            by[name][2] = (kind, sec)
    return [tuple(r) for r in out]


def plan_run_gains(specs, plan):
    """--run-gain TÊN=FILE (CODE_GAIN0 17/11; Liam 17/11 at the CODE session "GAIN0 sua sau (A) (Recommended)"):
    {run name: gain file} of the session's runs. Raises ValueError on a spec without '=' or file, a name that is
    not a --run of the session, two --run-gain for one run, or a file that is not there. A run without --run-gain
    uses --gain-file (if any), as before."""
    names = {r[0] for r in plan}
    out = {}
    for spec in specs or ():
        name, sep, f = spec.partition('=')
        if not sep or not name or not f:
            raise ValueError(f'--run-gain không hợp lệ: {spec!r} (cần TÊN=FILE)')
        if name not in names:
            raise ValueError(f'--run-gain {spec!r}: không có lượt {name!r} trong các --run')
        if name in out:
            raise ValueError(f'lượt {name!r} có hai --run-gain')
        if not os.path.isfile(f):
            raise ValueError(f'--run-gain {spec!r}: không có file {f}')
        out[name] = f
    return out


def gain_src(path):
    """--run-gain: the file and the sha256 of its bytes, read just before the set (recorded with the run's ack)."""
    with open(path, 'rb') as fh:
        return dict(gain_file=path, gain_sha256=hashlib.sha256(fh.read()).hexdigest())


def pi_leftovers(host):
    """B: run_hil.py / agv_logger.py still running on the Pi (None if SSH failed)."""
    rc, txt = ssh(host, LEFT_CMD, 30)
    return leftovers(txt) if rc in (0, 1) else None    # pgrep: 1 = no process matched


def _stop_later(a, name, kind, sec, rec):
    """B: after `sec` s, stop the run's run_hil.py by its PID file; what happened goes into rec."""
    def fire():
        rec.update(t_pc=time.time())
        try:
            rc, txt = ssh(a.host, pi_stop_cmd(a.pi_dir, name, kind), 30)
        except subprocess.TimeoutExpired:
            rc, txt = -1, 'ssh quá 30 s'
        rec.update(ssh_exit=rc, out=txt.strip())
        print(f"{name}: --pi-{kind} sau {sec:g} s: mã {rc} ({txt.strip().splitlines()[0] if txt.strip() else ''})",
              flush=True)
    t = threading.Timer(sec, fire)
    t.daemon = True
    return t


def bench(a):
    # A (Liam 09/11, "Làm cả A và B"): one bench session per output folder (CLOSED_LOOP/2026-11-09_run05 lost
    # the serial logs of three sessions run into one folder)
    old = existing_results(a.out)
    if old:
        print(f'DỪNG: {a.out} đã có kết quả của một phiên trước ({len(old)} file, ví dụ '
              f'{", ".join(os.path.relpath(p, a.out) for p in old[:3])}); dùng thư mục runNN mới')
        return 2, []
    try:
        plan = plan_runs(a.run, a.pi_kill, a.pi_stop)
        s06 = plan_s06(getattr(a, 's06', None) or [], plan, getattr(a, 'gain_query', False))
        run_gains = plan_run_gains(getattr(a, 'run_gain', None), plan)
    except ValueError as ex:
        print(ex)
        return 2, []
    os.makedirs(os.path.join(a.out, 'pi'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'frames'), exist_ok=True)
    sess = dict(started=datetime.datetime.now().isoformat(timespec='seconds'), host=a.host, pi_dir=a.pi_dir,
                runs=[], argv=sys.argv)
    rc, txt = ssh(a.host, 'cd ~/hil-framework && git status -sb | head -1 && git log -1 --format=%H', 30)
    if rc != 0:
        print(f'SSH tới Pi lỗi: {txt.strip()}')
        return 2, []
    sess['pi_git'] = txt.strip().splitlines()
    left = pi_leftovers(a.host)
    if left:                                         # B: an older run_hil / logger would share the SPI, UDP port
        print('DỪNG: trên Pi còn tiến trình của lần trước, dừng tay rồi chạy lại:\n  ' + '\n  '.join(left))
        return 2, []
    try:
        h7 = SerialLog(a.h7_port, os.path.join(a.out, f'h7_{a.h7_port}.txt'))
        esp = SerialLog(a.esp_port, os.path.join(a.out, f'esp32_{a.esp_port}.txt'))
    except Exception as ex:                          # serial.SerialException, OSError
        print(f'không mở được cổng serial: {ex}')
        return 2, []
    h7.start()
    esp.start()
    results, code = [], 0
    try:
        if a.boot:
            boot_cmd = (f'mkdir -p {a.pi_dir} && cd ~/hil-framework/rpi5 && python3 run_hil.py --fixed-omega 0,0,0,0 '
                        f'--imu-static 0,0,0,9.81,0 --steps 20000 --no-traj')
            t_pi = time.time()           # I8-10 (Codex round 2): motor check from here, not from the reset
            p = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', a.host, boot_cmd],
                                 stdout=open(os.path.join(a.out, 'pi', 'w0_imustatic_pi.txt'), 'xb'),
                                 stderr=subprocess.STDOUT)
            time.sleep(4.0)
            t_reset = time.time()
            esp.pulse_rts()
            boot_rc = p.wait(timeout=120)
            boot = [s for _, s in esp.since(t_reset)]
            miss = [k for k in BOOT_NEED if not any(k in s for s in boot)]
            # I8-10: the 4 s before the reset count too (the old ESP32 run sees the Pi start first)
            motor_on = boot_motor_on([s for _, s in esp.since(t_pi)])
            fws = fw_lines(boot)
            sess['boot'] = dict(t_pi=t_pi, t_reset=t_reset, pi_exit=boot_rc, missing=miss, motor_enabled=motor_on,
                                controller=next((s.strip() for s in boot if 'Controller:' in s), None),
                                esp32_fw=fws)
            print(f"khởi động ESP32: {sess['boot']['controller']}, thiếu {miss or 'không'}, Pi mã {boot_rc}, "
                  f"Motor enabled {motor_on} lần, #fw {fws or 'không có'}")
            if miss or boot_rc != 0:
                return 2, []
            bad = fw_problem(fws, a.expect_esp32)
            if bad:
                print(f'DỪNG: {bad}')
                return 2, []
            if motor_on:
                # I8-10: the boot step runs an omega mode, so --swa auto sends SWA off; motors on means the H7
                # has no I8 (or the Pi sent SWA on): stop before any run
                print('DỪNG: ESP32 bật motor lúc --boot (H7 chưa có I8, hoặc Pi gửi SWA bật)')
                return 2, []
        if a.expect_h7:
            # #fw H7 (Liam 09/11, CODE): the H7 prints its id at boot and every 10 s; wait for one line
            # before the first run, then the same rules as --expect-esp32
            t_w = time.time()
            while not fw_lines([s for _, s in h7.since(0)]) and time.time() - t_w < H7_FW_WAIT_S:
                time.sleep(0.5)
            sess['h7_fw'] = fw_lines([s for _, s in h7.since(0)])
            print(f"H7 #fw {sess['h7_fw'] or 'không có'}")
            bad = fw_problem(sess['h7_fw'], a.expect_h7)
            if bad:
                print('DỪNG: ' + bad.replace('ESP32', 'H7'))
                return 2, []
        for name, args, stop in plan:
            stop_rec = dict(kind=stop[0], after_s=stop[1]) if stop else None
            timer = _stop_later(a, name, stop[0], stop[1], stop_rec) if stop else None
            # A5: strict logger locked on the ESP32 address (unless --logger-loose / --no-logger)
            src, src_from = (None, None) if (a.no_logger or a.logger_loose or name in s06) else \
                logger_src(a.esp32_ip, [s for _, s in esp.since(0)])
            s_ip = None
            if name in s06:                          # the ESP32 address s06_logger sends START to
                s_ip = a.esp32_ip or esp32_ip([s for _, s in esp.since(0)])
                if not s_ip:
                    print(f'{name}: DỪNG: lượt --s06 cần IP ESP32 (--esp32-ip hoặc dòng "Got IP:" ở serial ESP32)')
                    sess['runs'].append(dict(name=name, args=args, s06=s06[name], esp32_ip=None))
                    code = 2
                    break
            boots, qrec, ack, gsrc = None, None, None, {}
            if getattr(a, 'gain_query', False):                         # S05 4.7 (iv): ask boot just before the run
                q_ip = a.esp32_ip or esp32_ip([s for _, s in esp.since(0)])
                gfile = run_gains.get(name) or getattr(a, 'gain_file', None)   # --run-gain of the run, else --gain-file
                if name in run_gains:                                   # its file and sha256 go with the run's ack
                    gsrc = gain_src(gfile)
                if gfile:                                               # B81-14: set the gain first, keep its ack
                    ack = gain_set(q_ip, gfile, os.path.join(a.out, 'pi', f'{name}_gain_ack.txt'))
                    if ack is None:
                        print(f'{name}: DỪNG: không đặt được gain {gfile} (pi/{name}_gain_ack.txt); '
                              'không chạy lượt này và các lượt sau')
                        sess['runs'].append(dict(name=name, args=args, gain_ack=dict(file=f'{name}_gain_ack.txt',
                                                                                     applied=False, **gsrc)))
                        code = 2
                        break
                q, rb = timed_query(q_ip, os.path.join(a.out, 'pi', f'{name}_gain_query_before.txt'))
                boots, qrec = [q, None], dict(before=rb)
                if ack is not None:                  # R134C4-5: a restart between the set and the question
                    import gain_line_check as glc
                    abad = glc.ack_problem(ack, q, {'id': (q or {}).get('id')})      # R134C5-4: id too
                    if abad:
                        print(f'{name}: DỪNG: {abad}; không chạy lượt này và các lượt sau')
                        sess['runs'].append(dict(name=name, args=args, boot=qrec, gain_ack=dict(
                            file=f'{name}_gain_ack.txt', applied=True, boot=ack.get('boot'), problem=abad, **gsrc)))
                        code = 2
                        break
            t0 = time.time()                         # R134C-9: the run (and its --pi-stop / --pi-kill clock)
            if timer:                                # starts after the question before it
                timer.start()
            lim = s06_timeout(a.run_timeout, s06[name]) if name in s06 else a.run_timeout      # T3T-5
            try:
                cmd = s06_run_cmd(a.pi_dir, name, args, s06[name], s_ip) if name in s06 else \
                    run_cmd(a.pi_dir, name, args, not a.no_logger, src)
                rc, txt = ssh(a.host, cmd, timeout=lim)
            except subprocess.TimeoutExpired:
                how = f'--run-timeout, lượt S06: --steps/1000 + {S06_TIMEOUT_EXTRA_S}' if name in s06 \
                    else '--run-timeout'                                             # T3T-R2-2
                rc, txt = -1, f'ssh quá {lim:.0f} s ({how}); kiểm tiến trình trên Pi'
                print(f'{name}: {txt}')
            if name in s06 and rc != 0:                # T3T-5: do not leave this run's processes on the Pi
                try:
                    crc, ctxt = ssh(a.host, s06_cleanup_cmd(a.pi_dir, name), timeout=30)
                except subprocess.TimeoutExpired:
                    crc, ctxt = -1, 'ssh quá 30 s'
                txt += f'\n# s06_cleanup (ssh mã {crc}):\n{ctxt}'
                print(f'{name}: dọn tiến trình S06 trên Pi (ssh mã {crc}): {ctxt.strip() or "không còn tiến trình"}')
            t1 = time.time()
            if timer:                                # R134C2-5: the run ended first: no stop sent, also not
                timer.cancel()                       # during the question after
            if boots is not None:                    # ... and just after it, before any file is copied (R134C-9),
                boots[1], qrec['after'] = timed_query(q_ip, os.path.join(a.out, 'pi', f'{name}_gain_query_after.txt'))
            if timer:                                # not after waiting for a stop already on its way (R134C3-5)
                timer.join(35.0)
                if 't_pc' not in stop_rec:
                    stop_rec['note'] = 'lượt xong trước khi tới giờ dừng: không gửi'
            with open(os.path.join(a.out, 'pi', f'{name}_pi.txt'), 'x', encoding='utf-8') as f:
                f.write(txt)
            scp(a.host, f'{a.pi_dir}/{name}_frames.npz', os.path.join(a.out, 'frames'))
            if name in s06:
                scp(a.host, f'{a.pi_dir}/{name}_esp', os.path.join(a.out, 'pi'), recursive=True)
                scp(a.host, f'{a.pi_dir}/{name}_logger.txt', os.path.join(a.out, 'pi'))
            elif not a.no_logger:
                scp(a.host, f'{a.pi_dir}/{name}.csv*', os.path.join(a.out, 'pi'))
                scp(a.host, f'{a.pi_dir}/{name}_logger.txt', os.path.join(a.out, 'pi'))
            run_rec = dict(name=name, args=args, t0=t0, t1=t1, ssh_exit=rc)
            if boots is not None:
                run_rec['boot'] = qrec
            if ack is not None:
                run_rec['gain_ack'] = dict(file=f'{name}_gain_ack.txt', applied=True, boot=ack.get('boot'),
                                           mac=ack.get('mac'), id=ack.get('id'), **gsrc)
            if name in s06:
                run_rec['s06'] = dict(s06[name], esp32_ip=s_ip)
            elif not a.no_logger:
                run_rec['logger_cmd'] = dict(strict=src is not None, src_ip=src, src_from=src_from)    # A5
            evs = [os.path.join(a.out, 'pi', f'{name}_esp', f) for f in ('esp32_data.txt', 'esp32_events.txt')] \
                if name in s06 else [os.path.join(a.out, 'pi', f'{name}.csv.events.txt')]
            run_fw = fw_lines([ln for ev in evs if os.path.exists(ev)
                               for ln in open(ev, encoding='utf-8', errors='replace').read().splitlines()])
            run_rec['esp32_fw'] = run_fw                 # from the UDP log of this run (none with --no-logger)
            if stop_rec:
                run_rec['pi_stop'] = stop_rec
            sess['runs'].append(run_rec)
            time.sleep(1.0)                          # last H7 lines of the run
            left = pi_leftovers(a.host)
            if left is None or left:                 # B: nothing of this run may stay on the Pi
                run_rec['pi_leftovers'] = left
                print(f'{name}: DỪNG CÁC LƯỢT SAU: ' + ('không kiểm được tiến trình trên Pi (SSH)' if left is None else
                      'trên Pi còn tiến trình, dừng tay:\n  ' + '\n  '.join(left)))
                code = 2
            if not os.path.exists(os.path.join(a.out, 'frames', f'{name}_frames.npz')):
                print(f'{name}: không có frame log (ssh mã {rc})'
                      + (' — đúng dự kiến với --pi-kill' if stop and stop[0] == 'kill' else '')
                      + (' — logger không sẵn sàng, run_hil.py không chạy (A5)' if 'logger_not_ready' in txt else ''))
                code = max(code, 1)
                if code == 2:
                    break
                continue
            r = analyze_run(a.out, name, h7.since(t0, t1 + 1.0), esp.since(t0, t1 + 1.0), s06=name in s06)
            r['ssh_exit'] = rc
            r['esp32_fw'] = run_fw
            r['h7_fw'] = fw_lines([s for _, s in h7.since(t0, t1 + 1.0)])
            run_rec['h7_fw'] = r['h7_fw']
            # a short run may hold no H7 #fw line (one every 10 s): only another id makes it not clean
            hbad = fw_problem(r['h7_fw'], a.expect_h7) if (a.expect_h7 and r['h7_fw']) else None
            if hbad:
                r['clean'] = False
                r['why'].append(hbad.replace('ESP32', 'H7'))
            if rc != 0:
                r['clean'] = False
                r['why'].append(f'Pi mã thoát {rc}')
            if boots is not None:
                r['boot'] = run_rec['boot']
                bbad = boot_problem(boots[0], boots[1], a.out, name)
                if not bbad and ack is not None:                   # B81-14: the set's ack of the same boot
                    import gain_line_check as glc
                    bbad = glc.ack_problem(ack, boots[0], {'id': (boots[0] or {}).get('id')})
                if bbad:
                    r['clean'] = False
                    r['why'].append(bbad)
            bad = fw_problem(run_fw, a.expect_esp32) if (name in s06 or not a.no_logger) else None
            if bad:
                r['clean'] = False
                r['why'].append(bad)
            ebad = lifted_env_problem(args, run_fw or sess.get('boot', {}).get('esp32_fw') or [], a.campaign_env)
            r['campaign_env'] = bool(a.campaign_env)
            if ebad:
                r['clean'] = False
                r['why'].append(ebad)
            if name in s06:                          # S06: s06_logger exit code (its log judged in analyze_run)
                m = LOGGER_EXIT_RE.search(txt)
                if not m or m[1] != '0':
                    r['clean'] = False
                    r['why'].append(f"s06_logger mã thoát {m[1] if m else 'không có'}")
            elif not a.no_logger:                    # A5: the WiFi log of the run must be usable
                sp = os.path.join(a.out, 'pi', f'{name}.csv.summary.json')
                lsum = json.load(open(sp, encoding='utf-8')) if os.path.exists(sp) else None
                r['logger'] = logger_info(lsum)
                lbad = logger_problems(lsum, txt, src is not None)
                if lbad:
                    r['clean'] = False
                    r['why'] += lbad
            if stop_rec:
                r['pi_stop'] = stop_rec
            results.append(r)
            print(summary_line(r), flush=True)
            code = max(code, 0 if r['clean'] else 1)
            if code == 2:
                break
    finally:
        time.sleep(1.0)
        h7.stop()
        esp.stop()
        sess['h7_fw_all'] = fw_lines([s for _, s in h7.since(0)])      # every H7 id seen in the session
        sess['ended'] = datetime.datetime.now().isoformat(timespec='seconds')
        with open(os.path.join(a.out, 'session.json'), 'x', encoding='utf-8') as f:     # A
            json.dump(sess, f, ensure_ascii=False, indent=1)
    return code, results


def analyze_only(a):
    names = sorted(os.path.basename(p)[:-len('_frames.npz')] for p in glob.glob(os.path.join(a.out, 'frames', '*_frames.npz')))
    sp = os.path.join(a.out, 'session.json')
    sess = json.load(open(sp, encoding='utf-8')) if os.path.exists(sp) else None
    if not names:
        print('không có frames/*_frames.npz')
        return 2, []
    results, code = [], 0
    runs = {x.get('name'): x for x in (sess or {}).get('runs', [])}
    # T3T-10: s06_esp32_check of an --s06 run writes its outputs: with --write-summary into a NEW check_analyze*/
    # folder (an earlier one is kept), else into a temporary folder outside --out (nothing written there)
    chk_sub = None
    if any('s06' in runs.get(n, {}) for n in names):
        if getattr(a, 'write_summary', False):
            chk_sub = 'check_analyze'
            k = 2
            while os.path.exists(os.path.join(a.out, chk_sub)):
                chk_sub, k = f'check_analyze_{k}', k + 1
        else:
            import tempfile
            chk_sub = tempfile.mkdtemp(prefix='s06_check_')
    for n in names:
        # no PC time windows: serial logs not judged per run; an --s06 run: events matched again, s06_esp32_check
        # into chk_sub (the session's check/ kept), and the s06_logger exit code read again (T3T-7)
        r = analyze_run(a.out, n, s06=True, check_sub=chk_sub) if 's06' in runs.get(n, {}) \
            else analyze_run(a.out, n)
        if 'boot' in runs.get(n, {}):                # R134C-10: a session made with --gain-query, checked again
            qb, qa = queries_of_run(a.out, n)        # from the saved answers (older sessions: no such record)
            r['boot'] = runs[n]['boot']
            bbad = boot_problem(qb, qa, a.out, n)
            if not bbad and 'gain_ack' in runs[n]:   # B81-14: a session made with --gain-file, ack read again
                import gain_line_check as glc
                bbad = glc.ack_problem(ack_of_run(a.out, n), qb, {'id': (qb or {}).get('id')})
            if bbad:
                r['clean'] = False
                r['why'].append(bbad)
        if 's06' in runs.get(n, {}):                 # T3T-7: as the run itself judged it (s06_logger exit code)
            tp = os.path.join(a.out, 'pi', f'{n}_pi.txt')
            txt = open(tp, encoding='utf-8', errors='replace').read() if os.path.exists(tp) else ''
            m = LOGGER_EXIT_RE.search(txt)
            if not m or m[1] != '0':
                r['clean'] = False
                r['why'].append(f"s06_logger mã thoát {m[1] if m else 'không có'} (pi/{n}_pi.txt)")
        lc = runs.get(n, {}).get('logger_cmd')
        if lc is not None:                           # A5: only sessions made with it, old folders keep their verdict
            sp = os.path.join(a.out, 'pi', f'{n}.csv.summary.json')
            tp = os.path.join(a.out, 'pi', f'{n}_pi.txt')
            lsum = json.load(open(sp, encoding='utf-8')) if os.path.exists(sp) else None
            txt = open(tp, encoding='utf-8', errors='replace').read() if os.path.exists(tp) else ''
            r['logger'] = logger_info(lsum)
            lbad = logger_problems(lsum, txt, lc.get('strict'))
            if lbad:
                r['clean'] = False
                r['why'] += lbad
        results.append(r)
        print(summary_line(r))
        code = max(code, 0 if r['clean'] else 1)
    if sess is None:
        print('(không có session.json: log serial H7, ESP32 không chấm theo lượt)')
    return code, results


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding='utf-8')     # Vietnamese text when stdout is a pipe (CLOSED_LOOP 2026-11-09_run03)
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description='Một buổi vòng kín HIL bằng một lệnh (chạy trên PC)')
    ap.add_argument('--out', required=True, help='thư mục lượt, ví dụ 05_DATA/results/CLOSED_LOOP/<ngày>_runNN')
    ap.add_argument('--host', default='liam-zuu@192.168.1.76')
    ap.add_argument('--pi-dir', default='~/bench', help="thư mục trên Pi (để trong nháy đơn: '~/run14')")
    ap.add_argument('--h7-port', default='COM3')
    ap.add_argument('--esp-port', default='COM5')
    ap.add_argument('--boot', action='store_true', help='--imu-static + reset ESP32 bằng RTS trước các lượt')
    ap.add_argument('--run', action='append', default=[], metavar='TÊN=ĐỐI_SỐ', help='đối số run_hil.py của một lượt')
    ap.add_argument('--no-logger', action='store_true', help='không chạy agv_logger trên Pi')
    ap.add_argument('--s06', action='append', default=[], metavar='TÊN=PROFILE,GIÂY[,logger-first]',
                    help='lượt S06: chạy tools/s06_logger.py (--profile, --duration) thay agv_logger; mặc định plant '
                         f'trước, logger sau {S06_LOGGER_AFTER_S} s; logger-first: logger trước, plant sau '
                         f'{S06_PLANT_AFTER_S} s (T5); --steps của lượt ≥ (GIÂY + {S06_PLANT_MARGIN_S}) × 1000')
    ap.add_argument('--expect-h7', metavar='SHA',
                    help='commit H7 phải chạy (tiền tố SHA): chờ dòng #fw của H7 (tối đa 11 s) trước lượt đầu; '
                         'khác SHA, dirty=1 hay không có thì dừng mã 2; lượt có #fw H7 khác thì không sạch')
    ap.add_argument('--esp32-ip', metavar='IP',
                    help='A5: IP nguồn khóa cho agv_logger --strict (mặc định: dòng "Got IP:" ESP32 in trong phiên, '
                         'không có thì khóa theo gói đầu tiên)')
    ap.add_argument('--logger-loose', action='store_true',
                    help='A5: logger chế độ cũ (không --strict, không khóa nguồn, không file ready), cho firmware cũ')
    ap.add_argument('--run-timeout', type=float, default=600.0, help='giới hạn mỗi lượt ssh, giây')
    ap.add_argument('--pi-kill', action='append', default=[], metavar='TÊN=GIÂY',
                    help='sau GIÂY từ lúc bắt đầu lượt TÊN: SIGKILL run_hil.py và mọi tiến trình con (rút Pi)')
    ap.add_argument('--campaign-env', action='store_true',
                    help='C5 (Liam 10/11 "Thêm cờ C5"): bài kê bánh A-K chạy trên env chiến dịch (esp32-s3-devkitc-1, '
                         'ladrc) thay cho env *wheels_lifted (L55); ghi campaign_env vào bản tóm tắt lượt')
    ap.add_argument('--expect-esp32', metavar='SHA',
                    help='commit ESP32 phải chạy (tiền tố SHA): dòng #fw lúc --boot hoặc trong log UDP của lượt '
                         'khác SHA, dirty=1, hay không có thì --boot dừng mã 2, lượt không sạch')
    ap.add_argument('--pi-stop', action='append', default=[], metavar='TÊN=GIÂY',
                    help='sau GIÂY từ lúc bắt đầu lượt TÊN: SIGTERM run_hil.py (STOP êm: setpoint 0 rồi hết lượt)')
    ap.add_argument('--gain-query', action='store_true',
                    help='S05 4.7 (iv) (R134): gain_send.py --query ngay trước lượt (rồi mới đếm giờ --pi-stop, '
                         '--pi-kill) và ngay khi lượt xong (trước scp), ghi pi/<TÊN>_gain_query_before.txt, _after.txt; '
                         'thiếu, khác boot hay MAC, hay boot khác dòng #us,ev=start của lượt thì lượt không sạch')
    ap.add_argument('--gain-file', metavar='FILE',
                    help='B81-14 (S05 4.7 (iv)): với --gain-query, trước câu hỏi "trước lượt" của mỗi lượt đặt bộ gain '
                         'FILE bằng gain_send.py --file (motor phải tắt), lưu nguyên đầu ra thành pi/<TÊN>_gain_ack.txt; '
                         'đặt không được thì dừng (mã 2), không chạy lượt; boot của #gain_ack khác hai lần hỏi thì '
                         'lượt không sạch')
    ap.add_argument('--run-gain', action='append', default=[], metavar='TÊN=FILE',
                    help='CODE_GAIN0 17/11 (kiểm cuối S05 4.6, sáu bộ gain trộn ngẫu nhiên): với --gain-query, lượt TÊN '
                         'đặt bộ gain FILE (thay --gain-file cho riêng lượt đó; cách đặt, ack, dừng như --gain-file); '
                         'file và sha256 ghi vào gain_ack của lượt; tên không phải --run, hai lần một lượt hay file '
                         'không có thì dừng trước mọi bước')
    ap.add_argument('--analyze-only', action='store_true',
                    help='chỉ chấm thư mục đã có frames/ (không ghi gì vào thư mục, trừ khi --write-summary; '
                         'lượt --s06: s06_esp32_check ghi vào thư mục tạm, với --write-summary vào check_analyze*/ mới)')
    ap.add_argument('--write-summary', action='store_true',
                    help='với --analyze-only: ghi bench_summary_analyze.json (không đụng bench_summary.json của phiên)')
    a = ap.parse_args(argv)
    if not a.analyze_only and not a.run and not a.boot:
        ap.error('cần --run, --boot hoặc --analyze-only')
    if a.analyze_only and (a.pi_kill or a.pi_stop):
        ap.error('--pi-kill, --pi-stop chỉ dùng khi chạy lượt')
    if a.gain_query and not os.path.isfile(GAIN_SEND):
        ap.error(f'--gain-query cần {GAIN_SEND} (nhánh feat/wifi-gain, L90; chưa gộp vào nhánh này)')
    if a.gain_query and a.analyze_only:
        ap.error('--gain-query chỉ dùng khi chạy lượt')
    if a.gain_query and a.no_logger:
        ap.error('--gain-query cần log UDP của lượt (dòng #us,ev=start mang boot của lượt, R134C-1): bỏ --no-logger')
    if a.gain_file and not a.gain_query:
        ap.error('--gain-file đi cùng --gain-query (boot của #gain_ack so với hai lần hỏi, B81-14)')
    if a.gain_file and not os.path.isfile(a.gain_file):
        ap.error(f'--gain-file {a.gain_file}: không có file')
    if a.run_gain and not a.gain_query:
        ap.error('--run-gain đi cùng --gain-query (boot của #gain_ack so với hai lần hỏi, B81-14)')
    code, results = analyze_only(a) if a.analyze_only else bench(a)
    if results and a.analyze_only and a.write_summary:
        with open(os.path.join(a.out, 'bench_summary_analyze.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
    elif results and not a.analyze_only:
        with open(os.path.join(a.out, 'bench_summary.json'), 'x', encoding='utf-8') as f:    # A
            json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'#ket_qua,ma={code}')
    return code


if __name__ == '__main__':
    sys.exit(main())
