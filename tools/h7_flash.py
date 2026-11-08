#!/usr/bin/env python3
"""
h7_flash.py — build and flash the H7 in one command, with evidence (Liam 09/11, phiên CODE `local_d45a2c5c…`:
"Duyệt cả 5", Q-F1 of 03_CODE/docs/2026-11-09_De_xuat_I8_bit_SWA_va_reset_panel.md; standing permission to
build and flash the H7, STATUS_CODE L56). Runs on the PC, from any worktree; builds the CubeIDE workspace
(main folder of hil-framework, `--repo`), as done by hand on 09/11 (CLOSED_LOOP/2026-11-09_run04/flash/):

  1. checks: CubeIDE not running (it locks the workspace), `--repo` HEAD (and `--expect` prefix), clean tree
     (`git status --porcelain`; dirty only with `--allow-dirty`, recorded `dirty: true`);
  2. clean build of stm32h7/Debug with the headless CubeIDE builder; needs `Build Finished. 0 errors` and
     `Headless build completed successfully`; records the `arm-none-eabi-size` line, SHA-256 of the .elf and
     of its loadable image (`objcopy -O binary`);
  3. opens the ST-Link VCP (COM3, DTR/RTS False: no reset) before flashing, so the boot lines are caught;
  4. STM32_Programmer_CLI `-c port=SWD sn=... -w <elf> -v -rst`; needs `Download verified successfully`;
  5. waits for `--- H753 HIL boot ---`, `Init done, entering main loop` and a `Loop:` line after the flash
     started, then logs `--log-after` seconds more.

Output in `--out`: headless_build.log, flash.log, h7_COM3.txt (`HH:MM:SS.mmm > line`), flash.json.
Last line `#ket_qua,ma=<0|1|2>`: 0 built, flashed, booted; 1 flashed but the boot lines did not come;
2 stopped before or during flashing (the board keeps whatever it had, unless flash.log says otherwise).

Usage:
  python tools/h7_flash.py --repo D:/CLAUDE_THESIS/03_CODE/hil-framework_plantmodel_matlab \
      --out <folder, e.g. 05_DATA/results/<RUN>/flash> [--expect 896164b]
  python tools/h7_flash.py ... --no-build     # flash the existing Debug/stm32h7.elf (still records SHAs)

Limits: Windows paths of the tools are the defaults below (CubeIDE 2.1.1, CubeCLT 1.21.0); the build uses
the Debug configuration only; the board must be powered over the ST-Link USB; PuTTY must not hold COM3.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time

CUBEIDEC = r'C:\ST\STM32CubeIDE_2.1.1\STM32CubeIDE\stm32cubeidec.exe'
PROG_CLI = r'C:\ST\STM32CubeCLT_1.21.0\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe'
OBJCOPY = r'C:\ST\STM32CubeCLT_1.21.0\GNU-tools-for-STM32\bin\arm-none-eabi-objcopy.exe'
STLINK_SN = '002F003A3532511131333430'
ELF_REL = os.path.join('stm32h7', 'Debug', 'stm32h7.elf')
BOOT_MARK = '--- H753 HIL boot ---'
INIT_MARK = 'Init done, entering main loop'
LOOP_MARK = 'Loop:'


# ── pure functions (tools/tests/test_h7_flash.py) ──

def cubeide_running(tasklist_csv):
    """True if `tasklist /FO CSV /NH` lists the CubeIDE GUI (stm32cubeide.exe; not the headless stm32cubeidec)."""
    for line in tasklist_csv.splitlines():
        name = line.split(',')[0].strip().strip('"').lower()
        if name == 'stm32cubeide.exe':
            return True
    return False


_SIZE = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9a-fA-F]+)\s+stm32h7\.elf\s*$', re.M)
_ERRS = re.compile(r'Build Finished\.\s+(\d+) errors?,\s+(\d+) warnings?')


def parse_build_log(text):
    """ok only with 0 errors and the headless success line; size from the last size table."""
    m = _ERRS.findall(text)
    errors, warnings = (int(m[-1][0]), int(m[-1][1])) if m else (None, None)
    sizes = _SIZE.findall(text)
    size = None
    if sizes:
        t, d, b, dec, _ = sizes[-1]
        size = dict(text=int(t), data=int(d), bss=int(b), dec=int(dec))
    cleaned = 'Clean-only build' in text
    ok = (errors == 0 and 'Headless build completed successfully' in text and size is not None)
    return dict(ok=ok, errors=errors, warnings=warnings, size=size, cleaned=bool(cleaned))


def parse_flash_log(text):
    """ok needs the verify line; reset seen separately (the board runs the new code only after it)."""
    return dict(ok='Download verified successfully' in text,
                reset='Software reset is performed' in text or 'MCU Reset' in text,
                board=(re.search(r'Board\s*:\s*(\S+)', text) or [None, None])[1],
                erased='Erasing' in text)


def boot_check(lines, t_flash):
    """lines: [(pc_time, text)]. Boot, init, then a Loop: line, all at or after t_flash, in that order."""
    out = dict(boot_t=None, init_t=None, loop_t=None, boot_lines=[])
    for t, s in lines:
        if t < t_flash:
            continue
        if out['boot_t'] is None:
            if BOOT_MARK in s:
                out['boot_t'] = t
            continue
        if out['init_t'] is None:
            if s.strip():
                out['boot_lines'].append(s.strip())
            if INIT_MARK in s:
                out['init_t'] = t
            continue
        if out['loop_t'] is None and s.startswith(LOOP_MARK):
            out['loop_t'] = t
            out['first_loop'] = s.strip()
            break
    out['ok'] = out['loop_t'] is not None
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


# ── serial capture ──

class SerialLog(threading.Thread):
    """Lines of one COM port with PC time; DTR/RTS False at open (no reset). Same format as hil_bench_session."""

    def __init__(self, port, path):
        super().__init__(daemon=True)
        import serial
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = port, 115200, 0.1
        self.s.dtr = False
        self.s.rts = False
        self.s.open()
        self.f = open(path, 'wb')
        self.lines = []
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

    def snapshot(self):
        with self.lock:
            return list(self.lines)

    def stop(self):
        self.stop_ev.set()
        self.join(2.0)


# ── steps ──

def _run(cmd, log_path, timeout):
    t0 = time.time()
    with open(log_path, 'wb') as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    with open(log_path, 'rb') as f:
        text = f.read().decode('utf-8', 'replace')
    return p.returncode, text, time.time() - t0


def _git(repo, *args):
    return subprocess.run(['git', '-C', repo] + list(args), capture_output=True, text=True, check=True).stdout


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding='utf-8')     # Vietnamese text when stdout is a pipe (run03, 09/11)
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--repo', required=True, help='CubeIDE workspace = main folder of hil-framework')
    ap.add_argument('--out', required=True)
    ap.add_argument('--expect', default=None, help='HEAD must start with this SHA')
    ap.add_argument('--allow-dirty', action='store_true')
    ap.add_argument('--no-build', action='store_true')
    ap.add_argument('--port', default='COM3')
    ap.add_argument('--sn', default=STLINK_SN)
    ap.add_argument('--boot-timeout', type=float, default=10.0)
    ap.add_argument('--log-after', type=float, default=3.0)
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    rec = dict(started=datetime.datetime.now().isoformat(timespec='seconds'), argv=sys.argv,
               repo=os.path.abspath(a.repo), tools=dict(cubeidec=CUBEIDEC, programmer=PROG_CLI))
    js = os.path.join(a.out, 'flash.json')

    def finish(code, why=None):
        rec['ended'] = datetime.datetime.now().isoformat(timespec='seconds')
        rec['result'] = code
        if why:
            rec['why'] = why
            print(f'DỪNG: {why}')
        with open(js, 'w', encoding='utf-8') as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        print(f'#ket_qua,ma={code}')
        return code

    # 1. checks
    tl = subprocess.run(['tasklist', '/FO', 'CSV', '/NH'], capture_output=True, text=True).stdout
    if cubeide_running(tl):
        return finish(2, 'CubeIDE đang mở (khóa workspace): đóng CubeIDE rồi chạy lại')
    head = _git(a.repo, 'rev-parse', 'HEAD').strip()
    porcelain = _git(a.repo, 'status', '--porcelain').strip()
    rec['git'] = dict(head=head, dirty=bool(porcelain), status=porcelain.splitlines()[:20],
                      describe=_git(a.repo, 'log', '-1', '--oneline').strip())
    print(f"repo HEAD {rec['git']['describe']}  dirty={rec['git']['dirty']}")
    if a.expect and not head.startswith(a.expect):
        return finish(2, f'HEAD {head[:9]} khác --expect {a.expect}')
    if porcelain and not a.allow_dirty:
        return finish(2, 'cây git bẩn (thêm --allow-dirty nếu cố ý; khi đó không làm số chính)')

    # 2. build
    elf = os.path.join(a.repo, ELF_REL)
    if not a.no_build:
        cmd = [CUBEIDEC, '--launcher.suppressErrors', '-nosplash', '-application',
               'org.eclipse.cdt.managedbuilder.core.headlessbuild', '-data', os.path.abspath(a.repo),
               '-cleanBuild', 'stm32h7/Debug', '-printErrorMarkers']
        rc, text, dt = _run(cmd, os.path.join(a.out, 'headless_build.log'), 900)
        b = parse_build_log(text)
        b.update(exit=rc, seconds=round(dt, 1))
        rec['build'] = b
        print(f"build: exit {rc}, {b['errors']} errors, {b['warnings']} warnings, size {b['size']}, {dt:.1f} s")
        if rc != 0 or not b['ok']:
            return finish(2, 'build không đạt (xem headless_build.log)')
    else:
        rec['build'] = dict(skipped=True)
    if not os.path.isfile(elf):
        return finish(2, f'không có {elf}')
    rec['elf'] = dict(path=elf, sha256=sha256_file(elf), mtime=datetime.datetime.fromtimestamp(
        os.path.getmtime(elf)).isoformat(timespec='seconds'))
    binp = os.path.join(a.out, 'stm32h7.bin')
    try:
        subprocess.run([OBJCOPY, '-O', 'binary', elf, binp], check=True, capture_output=True)
        rec['elf']['image_sha256'] = sha256_file(binp)
        os.remove(binp)                                  # the hash is the evidence; the image is rebuilt from git
    except (OSError, subprocess.CalledProcessError) as e:
        rec['elf']['image_sha256'] = None
        rec['elf']['objcopy_error'] = str(e)

    # 3. serial, 4. flash
    try:
        log = SerialLog(a.port, os.path.join(a.out, f'h7_{a.port}.txt'))
    except Exception as e:                               # noqa: BLE001 (serial.SerialException and OSError)
        return finish(2, f'không mở được {a.port}: {e} (PuTTY đang giữ cổng?)')
    log.start()
    time.sleep(0.5)
    t_flash = time.time()
    rec['t_flash'] = datetime.datetime.fromtimestamp(t_flash).isoformat(timespec='milliseconds')
    cmd = [PROG_CLI, '-c', 'port=SWD', f'sn={a.sn}', '-w', elf, '-v', '-rst']
    try:
        rc, text, dt = _run(cmd, os.path.join(a.out, 'flash.log'), 120)
    except subprocess.TimeoutExpired:
        log.stop()
        return finish(2, 'STM32_Programmer_CLI quá 120 s')
    f = parse_flash_log(text)
    f.update(exit=rc, seconds=round(dt, 1))
    rec['flash'] = f
    print(f"flash: exit {rc}, verified={f['ok']}, reset={f['reset']}, {dt:.1f} s")
    if rc != 0 or not f['ok']:
        log.stop()
        return finish(2, 'nạp không đạt (xem flash.log; board có thể đã bị xóa một phần)')

    # 5. boot lines
    t_end = time.time() + a.boot_timeout
    bc = boot_check(log.snapshot(), t_flash)
    while not bc['ok'] and time.time() < t_end:
        time.sleep(0.2)
        bc = boot_check(log.snapshot(), t_flash)
    if bc['ok']:
        time.sleep(a.log_after)
    log.stop()
    for k in ('boot_t', 'init_t', 'loop_t'):
        if bc.get(k) is not None:
            bc[k] = datetime.datetime.fromtimestamp(bc[k]).isoformat(timespec='milliseconds')
    rec['boot'] = bc
    if not bc['ok']:
        return finish(1, f'đã nạp nhưng không thấy đủ dòng khởi động trong {a.boot_timeout:.0f} s')
    print(f"boot: {bc['boot_t']}  {bc.get('first_loop', '')}")
    return finish(0)


if __name__ == '__main__':
    sys.exit(main())
