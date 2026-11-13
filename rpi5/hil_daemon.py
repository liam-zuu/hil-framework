"""
hil_daemon — stage E of the HIL workflow, reduced version (L79, Liam 11/11, CODE session
local_b2daa0f4: "Panel trước", "Tạm thời làm thu gọn").

The run manager behind the panel: it implements panel_link.RunControl, so the panel (ESP32-S3,
UART ttyAMA0, protocol v6) can list the tests, start one and stop it. One run at a time. Each
run is `run_hil.py <test args> --finite-run` started as a child process; the daemon follows it
from the lines it prints and never touches the 1 kHz loop (QĐ 17, workflow 9.6).

    panel <--UART--> PanelLink (panel_link.py) <--RunControl--> HilDaemon --Popen--> run_hil.py

What a panel run is (workflow 03_CODE/docs/2026-10-28_Luong_du_lieu_HIL_2_che_do.md, use 1):
a quick look / demo, NOT thesis data. meta.json of every run says mode: panel, data_valid:
false. Thesis numbers still come from the PC (tools/hil_bench_session.py, use 2).

Reduced version, what it leaves for the full one of 28/10 (sections 8.4-8.5, 9.4):
  * tests: a fixed list (spec S02 group A + the O1 floor tasks of C6, not K1/K2) instead of
    profiles rpi5/profiles/<id>.yaml; no load profile check, no manifest, no hil_ctl over SSH,
    no hil_session on the PC, no request_id; the controller is only what --ctrl declares.
  * H7 state while idle: no SPI frame is exchanged between runs, so the 'H7' light is the
    evidence of the last H7 check: a short probe run at daemon start (probe_h7; --no-probe skips it),
    then every finished run (it reached RUN only if the H7 answered for 1 s in ARM; MISO CRC /
    empty frames counted by hil_node). The panel only sends START while that light is on.
  * LOOP on the panel: p99 of the plant-step lateness of the last stats window (rt_monitor,
    printed every 10 s and for the whole run), not a loop turnaround time.
  * One run at a time on the Pi: START (and the probe) are refused while any other
    run_hil.py runs (e.g. one started from the PC), checked in /proc. Nothing else stops the
    PC from starting run_hil.py during a panel run: do not mix the two uses.

Run on the Pi (rpi5/ folder of the checkout):
    python hil_daemon.py --port /dev/ttyAMA0 --ctrl LADRC      # H7 probe first, then serve the panel
    python hil_daemon.py --port /dev/ttyAMA0 --mock            # run_hil.py --mock (fake H7), no probe
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, List, Optional, Sequence, Tuple

import panel_proto as pp
from panel_link import RunSnapshot

HERE = os.path.dirname(os.path.abspath(__file__))

# Tests the panel lists, in this order: spec S02 group A (trajectory_pub.GROUP_A, same order)
# and the C6 floor tasks (run_hil.py --o1; K1, K2 are the final test set, O1 3.7: never here).
def _o1_panel() -> Tuple[str, ...]:
    import o1_tasks                               # the figure-8 levels, never FINAL_SET (K1, K2)
    return tuple(t for t in o1_tasks.FIG8 if t not in o1_tasks.FINAL_SET)


O1_PANEL = _o1_panel()

# Idle H7 re-check (CODE_PANEL 14/11, Liam Q3 "tự chạy": the daemon starts with the Pi, the H7
# may still be off or booting): while idle and the H7 light is off, probe again this often.
REPROBE_S = 15.0   # [ĐỀ XUẤT]

# Panel reset line (CG2): Pi GPIO22 (pin 15) -> 1 kOhm -> EN of the WeAct board, open drain:
# the Pi only pulls EN low, then lets the line go (input); the board's own pull-up raises EN.
PANEL_RESET_GPIO = 22
PANEL_RESET_LOW_S = 0.1   # [ĐỀ XUẤT]
RPI5_GPIOCHIP = 4         # as gpio_sync.py

# error_code on the FAULT screen (panel shows the number and error_text)
E_RUN = 1          # run_hil.py ended in error:<reason> or with exit code != 0
E_SPAWN = 2        # run_hil.py could not be started
E_NO_SUMMARY = 3   # run_hil.py ended without a #run_summary line (killed, crashed early)

STOP_KILL_S = 5.0  # STOP again this long after the first one: SIGTERM (run_hil: 2nd Ctrl+C)

_PHASE_ST = {'PREPARE': pp.ST_PREPARE, 'ARM': pp.ST_ARM, 'RUN': pp.ST_RUN,
             'STOPPING': pp.ST_STOPPING, 'END': pp.ST_STOPPING}   # END of the loop: child still closing
_RE_PHASE = re.compile(r'#phase,([A-Z]+)')
_RE_SUMMARY = re.compile(r'#run_summary,(\{.*\})\s*$')
_RE_MISO = re.compile(r'#miso,checked=(\d+),crc_bad=(\d+),empty=(\d+)')
_RE_SWA = re.compile(r'#swa,mode=(\w+)')
_RE_STATS = re.compile(r'p99=([0-9.]+|None) us\s+max=([0-9.]+|None) us\s+>warn: \d+\s+overruns=(\d+)')
_RE_WHOLE = re.compile(r'whole run .*?p99=([0-9.]+|None) us .*?max=([0-9.]+) us .*?overruns=(\d+)')


def panel_tests() -> List[Tuple[str, List[str]]]:
    """(name shown on the panel, run_hil.py arguments) of every test, fixed order."""
    from trajectory_pub import GROUP_A
    tests = [(tid, ['--test', tid]) for tid in GROUP_A]
    tests += [(tid, ['--o1', tid]) for tid in O1_PANEL]
    return tests


def other_run_hil_running(exclude_pid: Optional[int] = None) -> bool:
    """True if a run_hil.py process runs on this machine (Linux /proc; False elsewhere)."""
    if not os.path.isdir('/proc'):
        return False
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) in (os.getpid(), exclude_pid):
            continue
        try:
            with open(f'/proc/{d}/cmdline', 'rb') as f:
                argv = f.read().split(b'\0')
        except OSError:
            continue
        if any(a.endswith(b'run_hil.py') for a in argv[:3]):
            return True
    return False


def checkout_id(folder: str = HERE) -> dict:
    """SHA and dirty flag of the git checkout the daemon runs from (None if not a checkout)."""
    def git(*a):
        return subprocess.run(['git', *a], cwd=folder, capture_output=True, text=True, timeout=10)
    try:
        sha = git('rev-parse', 'HEAD')
        st = git('status', '--porcelain', '--untracked-files=no')
    except (OSError, subprocess.TimeoutExpired):
        return {'sha': None, 'dirty': None}
    if sha.returncode:
        return {'sha': None, 'dirty': None}
    return {'sha': sha.stdout.strip(), 'dirty': bool(st.stdout.strip())}


def reset_panel(gpio: int = PANEL_RESET_GPIO, chip: int = RPI5_GPIOCHIP, low_s: float = PANEL_RESET_LOW_S,
                lg=None, sleep: Callable[[float], None] = time.sleep) -> str:
    """Pull the panel's EN low for low_s, then free the line (input, the board's pull-up wins).
    Returns 'done', or why it was skipped: the daemon still serves the panel without the wire
    (the panel then handshakes on its own when it sees the new epoch). lg: lgpio module (tests)."""
    if lg is None:
        try:
            import lgpio as lg
        except ImportError:
            return 'skipped: lgpio not installed'
    try:
        h = lg.gpiochip_open(chip)
    except Exception as exc:                      # lgpio raises its own error type
        return f'skipped: gpiochip{chip}: {exc}'
    try:
        lg.gpio_claim_output(h, gpio, 0)
        sleep(low_s)
        lg.gpio_free(h, gpio)
        lg.gpio_claim_input(h, gpio)              # leave it an input, never drive EN high
    except Exception as exc:
        return f'failed: GPIO{gpio}: {exc}'
    finally:
        lg.gpiochip_close(h)
    return 'done'


def _num(s: str) -> Optional[float]:
    return None if s == 'None' else float(s)


def _posix_stop(proc, hard: bool) -> None:
    """SIGINT = STOP for run_hil.py --finite-run (setpoint 0, then END); SIGTERM = kill."""
    os.kill(proc.pid, signal.SIGTERM if hard else signal.SIGINT)


class HilDaemon:
    """RunControl of panel_link.py (see its docstring for the contract). Thread-safe: the
    panel link calls select/start/stop/snapshot from its loop, a reader thread per run
    parses the child's output. No call waits for a run."""

    def __init__(self, runs_dir: str, tests: Optional[Sequence[Tuple[str, List[str]]]] = None,
                 extra_args: Sequence[str] = (), ctrl: str = '', mock: bool = False,
                 child_cmd: Optional[Callable[[List[str]], List[str]]] = None,
                 stop_fn: Callable = _posix_stop, clock: Callable[[], float] = time.monotonic,
                 busy_check: Callable[[Optional[int]], bool] = other_run_hil_running):
        self.tests = list(tests) if tests is not None else panel_tests()
        self.runs_dir = runs_dir
        self.mock = mock
        self.extra_args = list(extra_args) + (['--mock'] if mock else [])
        self.ctrl = ctrl
        self.child_cmd = child_cmd or (lambda args: [sys.executable, '-u', os.path.join(HERE, 'run_hil.py'), *args])
        self.stop_fn, self.clock, self.busy_check = stop_fn, clock, busy_check
        self._lock = threading.Lock()
        self.state, self.t_phase = pp.ST_IDLE, clock()
        self.profile: Optional[int] = None
        self.outcome = pp.OUT_NONE
        self.error_code, self.error_text = 0, ''
        self.result: Tuple[str, ...] = ()
        self.run_no = self.result_run = 0
        self.h7_ok = False                  # evidence of the last H7 check (probe or run)
        self.loop_us: Optional[int] = None
        self.overruns: Optional[int] = None
        self.quality = 0
        self.proc = None
        self.run_dir: Optional[str] = None
        self._t_stop: Optional[float] = None
        self._stop_pending = False
        self._seen_phase = False
        self._reader: Optional[threading.Thread] = None
        self._meta: dict = {}
        self._probing = False               # a probe run_hil.py is running: START is BUSY
        self._t_probe: Optional[float] = None
        self._checkout = checkout_id()         # read once: the checkout must not change under the daemon

    # ── RunControl ────────────────────────────────────────────────────────
    def profiles(self) -> list:
        return [name for name, _ in self.tests]

    def select(self, index: int) -> int:
        with self._lock:
            if self.state in pp.ST_ACTIVE:
                return pp.RES_BUSY
            if not 0 <= index < len(self.tests):
                return pp.RES_BAD_PROFILE
            self.profile = index
            return pp.RES_OK

    def start(self, index: int) -> int:
        with self._lock:
            if self.state in pp.ST_ACTIVE:
                return pp.RES_BUSY
            if not 0 <= index < len(self.tests):
                return pp.RES_BAD_PROFILE
            if self._probing or self.busy_check(None):
                return pp.RES_BUSY                 # another run_hil.py (e.g. from the PC, or the probe) owns the SPI
            try:
                run_no = pp.next_run_no(self.run_no)
            except OverflowError:                  # no number left: refuse, change nothing
                return pp.RES_FAILED
            name, args = self.tests[index]
            self.profile, self.run_no = index, run_no
            self.outcome, self.error_code, self.error_text = pp.OUT_NONE, 0, ''
            self.result, self.result_run = (), 0
            self.quality = pp.Q_DUT_UNCONF              # DUT READY is never checked in a panel run
            self.loop_us = self.overruns = None
            self._t_stop, self._stop_pending, self._seen_phase = None, False, False
            argv = self.child_cmd([*args, '--finite-run', *self.extra_args])
            self._open_run_dir(name, argv)
            try:
                self.proc = subprocess.Popen(argv, cwd=HERE, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                                             encoding='utf-8', errors='replace')
            except OSError as exc:
                self._fail(E_SPAWN, f'spawn {exc.errno}', f'spawn failed: {exc}')
                self._close_meta(rc=None)
                return pp.RES_FAILED
            self._meta['pid'] = self.proc.pid
            self._enter(pp.ST_PREPARE)
            self._reader = threading.Thread(target=self._read, args=(self.proc,), daemon=True)
            self._reader.start()
            return pp.RES_OK

    def stop(self) -> int:
        with self._lock:
            if self.state == pp.ST_ERROR:          # STOP acknowledges the fault
                self._enter(pp.ST_IDLE)
                self.error_code, self.error_text = 0, ''
                return pp.RES_OK
            if self.state not in pp.ST_ACTIVE or self.proc is None:
                return pp.RES_NOT_IDLE
            now = self.clock()
            if self._t_stop is None:
                self._t_stop = now
                self.outcome = pp.OUT_USER_STOP
                if not self._seen_phase:
                    self._stop_pending = True      # run_hil installs its STOP handler late: send at #phase
                    return pp.RES_OK
                return self._send_stop(hard=False)
            if now - self._t_stop >= STOP_KILL_S:
                return self._send_stop(hard=True)
            return pp.RES_OK                       # STOP again: harmless

    def snapshot(self) -> RunSnapshot:
        with self._lock:
            return RunSnapshot(
                run_state=self.state, phase_s=self.clock() - self.t_phase, profile=self.profile,
                outcome=self.outcome, quality=self.quality, lost=0, data_age_ms=None,
                h7_ok=self.h7_ok, esp32_ok=None, error_code=self.error_code, error_text=self.error_text,
                ctrl=self.ctrl, ctrl_src=pp.CTRL_SRC_PROFILE if self.ctrl else pp.CTRL_SRC_NONE,
                loop_us=self.loop_us, overruns=self.overruns, result=self.result,
                run_no=self.run_no, result_run=self.result_run)

    # ── H7 check while idle ───────────────────────────────────────────────
    def probe_h7(self, steps: int = 1500, timeout_s: float = 30.0) -> bool:
        """Short run with no trajectory and no plant dynamics (wheel speeds 0, IMU static and
        level, SWA off: the DUT motors stay off) just to see the H7 answer; h7_ok = MISO
        frames were checked and none had a bad CRC or was empty (hil_node '#miso' line).
        Refused (False, h7_ok unchanged) while a run is active or another run_hil.py runs."""
        with self._lock:
            if self.state in pp.ST_ACTIVE or self._probing or self.busy_check(None):
                return False
            self._probing, self._t_probe = True, self.clock()
        argv = self.child_cmd(['--fixed-omega', '0,0,0,0', '--imu-static', '0,0,0,9.81,0',
                               '--steps', str(steps), '--no-traj', '--swa', 'off', *self.extra_args])
        try:
            try:
                out = subprocess.run(argv, cwd=HERE, capture_output=True, text=True, timeout=timeout_s,
                                     encoding='utf-8', errors='replace')
                text = out.stdout + out.stderr
            except (OSError, subprocess.TimeoutExpired):
                text = ''
            m = _RE_MISO.search(text)
            ok = bool(m) and int(m.group(1)) > 0 and m.group(2) == '0' and m.group(3) == '0'
            with self._lock:
                self.h7_ok = ok
            return ok
        finally:
            with self._lock:
                self._probing, self._t_probe = False, self.clock()

    def maybe_reprobe(self, every_s: float = REPROBE_S) -> Optional[bool]:
        """Idle re-check of the H7 (daemon started before the H7 was up): probe when idle, the H7
        light is off and the last probe ended at least every_s ago. None = no probe this time.
        Never in --mock (the fake H7 always answers)."""
        with self._lock:
            if (self.mock or self.h7_ok or self.state in pp.ST_ACTIVE or self._probing
                    or (self._t_probe is not None and self.clock() - self._t_probe < every_s)):
                return None
        return self.probe_h7()

    # ── internals (called with the lock held unless noted) ───────────────
    def _enter(self, st: int) -> None:
        self.state, self.t_phase = st, self.clock()

    def _fail(self, code: int, text: str, detail: str) -> None:
        self.outcome, self.error_code = pp.OUT_ERROR, code
        self.error_text = text[:pp.ERROR_TEXT_LEN]
        self._meta['error'] = detail
        self._enter(pp.ST_ERROR)

    def _send_stop(self, hard: bool) -> int:
        try:
            self.stop_fn(self.proc, hard)
        except (OSError, ProcessLookupError):
            return pp.RES_FAILED                   # effect unknown
        return pp.RES_OK

    def _open_run_dir(self, name: str, argv: List[str]) -> None:
        now = _dt.datetime.now()
        run_id = f'{now:%Y%m%d_%H%M%S}_{name}_{secrets.token_hex(2)}'
        self.run_dir = os.path.join(self.runs_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=False)
        self._meta = {'run_id': run_id, 'run_no': self.run_no, 'mode': 'panel', 'data_valid': False,
                      'data_valid_why': 'panel run = quick look / demo (L79, workflow use 1)',
                      'test': name, 'argv': argv, 'ctrl_declared': self.ctrl or None,
                      'pi_checkout': self._checkout, 'started': now.isoformat(timespec='seconds')}
        self._write_meta()

    def _write_meta(self) -> None:
        path = os.path.join(self.run_dir, 'meta.json')
        with open(path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(self._meta, f, indent=1)
        os.replace(path + '.tmp', path)

    def _close_meta(self, rc: Optional[int]) -> None:
        self._meta.update(ended=_dt.datetime.now().isoformat(timespec='seconds'), exit_code=rc,
                          outcome={pp.OUT_COMPLETED: 'completed', pp.OUT_USER_STOP: 'user_stop',
                                   pp.OUT_ERROR: 'error', pp.OUT_NONE: None}[self.outcome],
                          h7_ok_after=self.h7_ok, result=list(self.result))
        self._write_meta()

    def _read(self, proc) -> None:
        """Reader thread: every line to run_hil.log, parsed for the panel; then the end."""
        summary, miso, last = None, None, None
        with open(os.path.join(self.run_dir, 'run_hil.log'), 'w', encoding='utf-8') as logf:
            for line in proc.stdout:
                logf.write(line)
                with self._lock:
                    m = _RE_PHASE.search(line)
                    if m and m.group(1) in _PHASE_ST:
                        self._seen_phase = True
                        if self._stop_pending:
                            self._stop_pending = False
                            self._send_stop(hard=False)
                        st = _PHASE_ST[m.group(1)]
                        if m.group(1) == 'RUN':
                            self.h7_ok = True      # ARM leaves only after 1 s of fresh H7 answers
                        if st != self.state:
                            self._enter(st)
                        continue
                    m = _RE_WHOLE.search(line) or _RE_STATS.search(line)
                    if m:
                        p99 = _num(m.group(1))
                        self.loop_us = None if p99 is None else int(round(p99))
                        self.overruns = int(m.group(3))
                        if _RE_WHOLE.search(line):
                            last = (p99, float(m.group(2)), int(m.group(3)))
                        continue
                    m = _RE_SUMMARY.search(line)
                    if m:
                        try:
                            summary = json.loads(m.group(1))
                        except ValueError:
                            summary = {'outcome': 'error:bad_summary'}
                        continue
                    m = _RE_MISO.search(line)
                    if m:
                        miso = tuple(int(g) for g in m.groups())
                        continue
                    m = _RE_SWA.search(line)
                    if m:
                        self._meta['swa_mode'] = m.group(1)   # what SWA did in this panel run (I8)
        rc = proc.wait()
        with self._lock:
            self._finish(rc, summary, miso, last)

    def _finish(self, rc: int, summary: Optional[dict], miso, last) -> None:
        self._meta.update(run_summary=summary, miso=miso)
        # --mock: the fake H7 of run_hil.py does not build valid MISO CRCs (Pi check 11/11: 11093 of
        # 11096 frames 'bad'), so its counters say nothing about an H7; the light stays as it was
        if miso is not None and not self.mock:
            checked, crc_bad, empty = miso
            if crc_bad or empty:
                self.quality |= pp.Q_SPI_STALE
                self.h7_ok = False
        outcome = (summary or {}).get('outcome') or ''
        lines = [f'{outcome or "no summary"} #{self.run_no}'[:pp.RESULT_LEN]]
        if last is not None:
            p99 = '-' if last[0] is None else f'{last[0]:.0f}'
            lines.append(f'p99 {p99} max {last[1]:.0f}us'[:pp.RESULT_LEN])
            lines.append(f'ovr {last[2]} miso {"-" if miso is None else miso[1] + miso[2]}'[:pp.RESULT_LEN])
        self.result, self.result_run = tuple(lines), self.run_no
        if summary is None:
            self._fail(E_NO_SUMMARY, f'no summary rc {rc}', f'run_hil.py exit {rc} without #run_summary')
        elif outcome.startswith('error:') or rc != 0:
            detail = summary.get('detail') or ''
            # rc 3 with a non-error outcome: END reached but its tail was cut (hil_node.run_failed)
            why = outcome[6:] if outcome.startswith('error:') else f'{outcome} exit {rc}'
            self._fail(E_RUN, f'{why} {detail}'.strip(), f'{outcome} {detail} exit {rc}')
            if outcome == 'error:arm_timeout' and detail == 'no_h7':
                self.h7_ok = False
        else:
            self.outcome = pp.OUT_USER_STOP if outcome == 'user_stop' else pp.OUT_COMPLETED
            self._enter(pp.ST_END)
        self._close_meta(rc)
        self.proc = None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Panel run manager (stage E, reduced, L79)')
    ap.add_argument('--port', default='/dev/ttyAMA0', help='panel UART')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--epoch-file', default=os.path.expanduser('~/.hil_panel_epoch'))
    ap.add_argument('--create-epoch', action='store_true', help='first start only: create the epoch file')
    ap.add_argument('--runs-dir', default=os.path.expanduser('~/hil_panel_runs'))
    ap.add_argument('--ctrl', default='', help='controller flashed on the DUT, as declared (shown "declared")')
    ap.add_argument('--mock', action='store_true', help='run_hil.py --mock (fake H7, no SPI)')
    ap.add_argument('--no-probe', action='store_true',
                    help='skip the H7 check at start (the H7 light, so START, stays off until a run reaches RUN)')
    ap.add_argument('--panel-reset-gpio', type=int, default=PANEL_RESET_GPIO,
                    help='BCM GPIO wired to the panel EN (pulled low once at start); -1 = no wire')
    a = ap.parse_args(argv)
    os.makedirs(a.runs_dir, exist_ok=True)
    daemon = HilDaemon(a.runs_dir, ctrl=a.ctrl, mock=a.mock)
    if a.panel_reset_gpio >= 0:
        print(f'panel reset GPIO{a.panel_reset_gpio}:', reset_panel(a.panel_reset_gpio), flush=True)
    if a.mock:
        daemon.h7_ok = True                        # fake H7 of run_hil.py --mock always answers
    elif not a.no_probe:
        print('H7 probe ...', flush=True)
        print('H7 probe:', 'answers' if daemon.probe_h7()
              else f'NO ANSWER (panel START blocked; probing again every {REPROBE_S:.0f} s while idle)', flush=True)

        def reprobe_loop():
            while True:
                r = daemon.maybe_reprobe()
                if r is not None:
                    print('H7 re-probe:', 'answers' if r else 'no answer', flush=True)
                time.sleep(1.0)
        threading.Thread(target=reprobe_loop, name='h7-reprobe', daemon=True).start()
    import serial  # pyserial
    from panel_link import PanelLink, nonblocking_writer, serve
    owner = pp._FileLock(a.epoch_file + '.owner', timeout_s=0.5)
    with owner:                                    # one link per UART / epoch file
        epoch = pp.next_daemon_epoch(a.epoch_file, create=a.create_epoch)
        port = serial.Serial(baudrate=a.baud, timeout=0, write_timeout=0)
        port.dtr = port.rts = False
        port.port = a.port
        port.open()
        with port:
            link = PanelLink(epoch, daemon, nonblocking_writer(port), baud=a.baud)
            print(f'hil_daemon on {a.port}, epoch {epoch}, {len(daemon.tests)} tests, runs in {a.runs_dir}; '
                  'Ctrl+C to quit', flush=True)
            try:
                serve(port, link)
            except KeyboardInterrupt:
                if daemon.proc is not None:
                    daemon.stop()
                    print('STOP sent to the running run_hil.py', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
