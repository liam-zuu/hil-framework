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
(lệnh, commit Pi, giờ PC đầu/cuối mỗi lượt, mã thoát), `bench_summary.json` (số chấm). Dòng cuối in
`#ket_qua,ma=<0|1|2>`: 0 mọi lượt sạch, 1 có lượt không sạch, 2 lỗi chuẩn bị (cổng, SSH, khởi động ESP32).

Usage:
  python tools/hil_bench_session.py --out <thư mục lượt> --pi-dir '~/run14' --boot \
      --run b2_AL1='--test A-L1' --run b3_AC1='--test A-C1' \
      --run b4_PCNT='--traj circle --speed 0.3 --radius 0.5 --duration 60'
  python tools/hil_bench_session.py --analyze-only --out <thư mục lượt có frames/>

Giới hạn: Pi chạy không sudo (như run12); không nạp board; ESP32 phải đã nạp đúng env; PuTTY không được mở
cùng cổng. Cửa sổ thời gian của lượt (cho log H7, ESP32) chỉ có khi chạy trực tiếp; `--analyze-only` chấm
log serial cả file nếu không có `session.json`.
"""
import argparse
import csv
import datetime
import glob
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
        self.f = open(path, 'wb')
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


def analyze_run(out, name, h7_lines=None, esp_lines=None):
    z = np.load(os.path.join(out, 'frames', f'{name}_frames.npz'))
    res = dec.decode(z['t_send_ns'], z['mosi'], z['miso'])
    probs = dec.problems(res)
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


def scp(host, remote, local_dir):
    return subprocess.run(['scp', '-o', 'BatchMode=yes', '-q', f'{host}:{remote}', local_dir]).returncode


def run_cmd(pi_dir, name, args, logger):
    hil = (f'cd ~/hil-framework/rpi5 && python3 run_hil.py {args} --finite-run '
           f'--frame-log {pi_dir}/{name}_frames.npz')
    if not logger:
        return f'mkdir -p {pi_dir} && {hil}'
    # `&` on the logger alone (a `;` before it): with `a && b && logger &` the whole list went to a background
    # subshell, $! was that shell, and a non-interactive shell ignores SIGINT in background jobs, so the
    # logger never stopped and ssh waited for ever (2026-11-09_run01). agv_logger closes on SIGINT/SIGTERM.
    return (f'mkdir -p {pi_dir} && cd ~/hil-framework || exit 2; python3 tools/agv_logger.py --out {pi_dir}/{name}.csv '
            f'> {pi_dir}/{name}_logger.txt 2>&1 & L=$!; {hil}; RC=$?; sleep 1; kill -INT $L; '
            f'for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $L 2>/dev/null || break; sleep 0.5; done; '
            f'kill -TERM $L 2>/dev/null; wait $L; echo logger_exit=$?; exit $RC')


def bench(a):
    os.makedirs(os.path.join(a.out, 'pi'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'frames'), exist_ok=True)
    sess = dict(started=datetime.datetime.now().isoformat(timespec='seconds'), host=a.host, pi_dir=a.pi_dir,
                runs=[], argv=sys.argv)
    rc, txt = ssh(a.host, 'cd ~/hil-framework && git status -sb | head -1 && git log -1 --format=%H', 30)
    if rc != 0:
        print(f'SSH tới Pi lỗi: {txt.strip()}')
        return 2, []
    sess['pi_git'] = txt.strip().splitlines()
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
                                 stdout=open(os.path.join(a.out, 'pi', 'w0_imustatic_pi.txt'), 'wb'),
                                 stderr=subprocess.STDOUT)
            time.sleep(4.0)
            t_reset = time.time()
            esp.pulse_rts()
            boot_rc = p.wait(timeout=120)
            boot = [s for _, s in esp.since(t_reset)]
            miss = [k for k in BOOT_NEED if not any(k in s for s in boot)]
            # I8-10: the 4 s before the reset count too (the old ESP32 run sees the Pi start first)
            motor_on = boot_motor_on([s for _, s in esp.since(t_pi)])
            sess['boot'] = dict(t_pi=t_pi, t_reset=t_reset, pi_exit=boot_rc, missing=miss, motor_enabled=motor_on,
                                controller=next((s.strip() for s in boot if 'Controller:' in s), None))
            print(f"khởi động ESP32: {sess['boot']['controller']}, thiếu {miss or 'không'}, Pi mã {boot_rc}, "
                  f"Motor enabled {motor_on} lần")
            if miss or boot_rc != 0:
                return 2, []
            if motor_on:
                # I8-10: the boot step runs an omega mode, so --swa auto sends SWA off; motors on means the H7
                # has no I8 (or the Pi sent SWA on): stop before any run
                print('DỪNG: ESP32 bật motor lúc --boot (H7 chưa có I8, hoặc Pi gửi SWA bật)')
                return 2, []
        for spec in a.run:
            name, _, args = spec.partition('=')
            if not re.fullmatch(r'[A-Za-z0-9_\-]+', name) or BAD_ARG.search(args):
                print(f'--run không hợp lệ: {spec!r}')
                return 2, results
            t0 = time.time()
            try:
                rc, txt = ssh(a.host, run_cmd(a.pi_dir, name, args, not a.no_logger), timeout=a.run_timeout)
            except subprocess.TimeoutExpired:
                rc, txt = -1, f'ssh quá {a.run_timeout} s (--run-timeout); kiểm tiến trình trên Pi'
                print(f'{name}: {txt}')
            t1 = time.time()
            with open(os.path.join(a.out, 'pi', f'{name}_pi.txt'), 'w', encoding='utf-8') as f:
                f.write(txt)
            scp(a.host, f'{a.pi_dir}/{name}_frames.npz', os.path.join(a.out, 'frames'))
            if not a.no_logger:
                scp(a.host, f'{a.pi_dir}/{name}.csv*', os.path.join(a.out, 'pi'))
                scp(a.host, f'{a.pi_dir}/{name}_logger.txt', os.path.join(a.out, 'pi'))
            sess['runs'].append(dict(name=name, args=args, t0=t0, t1=t1, ssh_exit=rc))
            time.sleep(1.0)                          # last H7 lines of the run
            if not os.path.exists(os.path.join(a.out, 'frames', f'{name}_frames.npz')):
                print(f'{name}: không có frame log (ssh mã {rc})')
                code = 1
                continue
            r = analyze_run(a.out, name, h7.since(t0, t1 + 1.0), esp.since(t0, t1 + 1.0))
            r['ssh_exit'] = rc
            if rc != 0:
                r['clean'] = False
                r['why'].append(f'Pi mã thoát {rc}')
            results.append(r)
            print(summary_line(r), flush=True)
            code = max(code, 0 if r['clean'] else 1)
    finally:
        time.sleep(1.0)
        h7.stop()
        esp.stop()
        sess['ended'] = datetime.datetime.now().isoformat(timespec='seconds')
        with open(os.path.join(a.out, 'session.json'), 'w', encoding='utf-8') as f:
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
    for n in names:
        r = analyze_run(a.out, n)                    # no PC time windows: serial logs not judged per run
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
    ap.add_argument('--run-timeout', type=float, default=600.0, help='giới hạn mỗi lượt ssh, giây')
    ap.add_argument('--analyze-only', action='store_true',
                    help='chỉ chấm thư mục đã có frames/ (không ghi gì vào thư mục, trừ khi --write-summary)')
    ap.add_argument('--write-summary', action='store_true', help='với --analyze-only: ghi bench_summary.json')
    a = ap.parse_args(argv)
    if not a.analyze_only and not a.run and not a.boot:
        ap.error('cần --run, --boot hoặc --analyze-only')
    code, results = analyze_only(a) if a.analyze_only else bench(a)
    if results and (not a.analyze_only or a.write_summary):
        with open(os.path.join(a.out, 'bench_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'#ket_qua,ma={code}')
    return code


if __name__ == '__main__':
    sys.exit(main())
