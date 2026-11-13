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
    a new probe every REPROBE_S while idle with the light off (the H7 may boot after the Pi),
    then every finished run (it reached RUN only if the H7 answered for 1 s in ARM; MISO CRC /
    empty frames counted by hil_node). The panel only sends START while that light is on.
  * LOOP on the panel: p99 of the plant-step lateness of the last stats window (rt_monitor,
    printed every 10 s and for the whole run), not a loop turnaround time.
  * One run at a time on the Pi: START (and the probe) are refused while any other
    run_hil.py runs (e.g. one started from the PC), checked in /proc (is_run_hil_argv,
    is_run_hil_worker). That is a check, not a lock: nothing stops the PC from starting
    run_hil.py a moment later. Do not mix the two uses: stop the service before a PC session.
  * Each run and each probe is its own session / process group (POSIX). STOP = SIGINT to
    run_hil.py, then the daemon's timer: SIGTERM to the group after STOP_TERM_S, SIGKILL after
    STOP_KILL_S more; a run is finished only once its whole group is gone.

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
E_GROUP = 4        # a process of the run survived SIGKILL (PPI-1/4): START refused until restart

# STOP escalation (PPI-1, Codex 14/11 round 1), run by the daemon's own timer from the first STOP,
# whatever the child prints: level 0 = SIGINT to run_hil.py (soft STOP: setpoint 0, then END);
# STOP_TERM_S later level 1 = SIGTERM to the run's whole process group; STOP_KILL_S after that
# level 2 = SIGKILL to the group. A run ends only once its whole group is gone (_reap_group).
STOP_TERM_S = 5.0   # [ĐỀ XUẤT]
STOP_KILL_S = 3.0   # [ĐỀ XUẤT]
REAP_S = 3.0        # after the owner exits: the group's last processes get this long, then SIGKILL
SHUTDOWN_S = 12.0   # daemon stop (Ctrl+C, systemd; TimeoutStopSec 15): wait this long for the run

_PHASES = ('PREPARE', 'ARM', 'RUN', 'STOPPING', 'END')
_PHASE_ST = {'PREPARE': pp.ST_PREPARE, 'ARM': pp.ST_ARM, 'RUN': pp.ST_RUN,
             'STOPPING': pp.ST_STOPPING, 'END': pp.ST_STOPPING}   # END of the loop: child still closing
_RE_PHASE = re.compile(r'#phase,(' + '|'.join(_PHASES) + r'),t_ns=')   # PPI-11: whole field only
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


# Short options of the interpreter that take a value (CPython getopt: '-W x', '-Wx', '-uW x');
# 'c' and 'm' end the options with no script. Long options that take the next token as value.
_PY_SHORT_ARG = b'WX'
_PY_SHORT_END = b'cm'
_PY_LONG_ARG = {b'--check-hash-based-pycs'}


def is_run_hil_argv(argv: Sequence[bytes]) -> bool:
    """True if argv (a /proc cmdline split on NUL) is a Python interpreter running the script
    run_hil.py (PPI-3, PPI-R2-3): options are read as CPython does, letter by letter inside a
    token ('-uW ignore', '-BXdev', '-uc …'); the first non-option argument is the script; '-c'
    or '-m' (alone or inside a token) mean no script. 'vim run_hil.py' is not a run."""
    argv = [a for a in argv if a != b'']
    if not argv or not os.path.basename(argv[0]).startswith(b'python'):
        return False
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == b'--':
            i += 1
            break
        if a == b'-' or not a.startswith(b'-'):
            break                                  # '-' = script from stdin: no run_hil.py
        if a.startswith(b'--'):
            i += 2 if a in _PY_LONG_ARG else 1
            continue
        i += 1
        for k in range(1, len(a)):
            ch = a[k:k + 1]
            if ch in _PY_SHORT_END:
                return False
            if ch in _PY_SHORT_ARG:
                if k + 1 == len(a):
                    i += 1                         # its value is the next token
                break                              # else the rest of this token is its value
    return i < len(argv) and os.path.basename(argv[i]) == b'run_hil.py'


def is_run_hil_worker(argv: Sequence[bytes], cwd: Optional[str], folder: str = HERE) -> bool:
    """A multiprocessing 'spawn' worker (HIL-Plant, HIL-TrajPub: run_hil.py sets 'spawn', so the
    cmdline does not name run_hil.py) started from this rpi5 folder (PPI-3/PPI-4: an orphan of a
    killed run_hil.py may still hold the SPI)."""
    joined = b' '.join(argv)
    return (b'multiprocessing' in joined and b'spawn_main' in joined and cwd is not None
            and os.path.realpath(cwd) == os.path.realpath(folder))


def other_run_hil_running(exclude_pid: Optional[int] = None) -> bool:
    """True if a run_hil.py process, or a spawn worker of one started from this rpi5 folder,
    runs on this machine (Linux /proc; False elsewhere). Not a lock: a PC run can still start a
    moment later (documented: never mix the two uses; stop the service before a PC session)."""
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
        if is_run_hil_argv(argv):
            return True
        try:
            cwd = os.readlink(f'/proc/{d}/cwd')
        except OSError:
            cwd = None
        if is_run_hil_worker(argv, cwd):
            return True
    return False


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reap_group(pgid: Optional[int], wait_s: float = REAP_S) -> bool:
    """POSIX: give what is left of the process group pgid wait_s to exit, then SIGKILL it.
    True once the group is gone (always True off POSIX or without a group)."""
    if pgid is None or os.name != 'posix':
        return True
    t_end = time.monotonic() + wait_s
    while _group_alive(pgid) and time.monotonic() < t_end:
        time.sleep(0.05)
    if _group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        t_end = time.monotonic() + 2.0
        while _group_alive(pgid) and time.monotonic() < t_end:
            time.sleep(0.05)
    return not _group_alive(pgid)


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
                lg=None, sleep: Callable[[float], None] = time.sleep,
                say: Callable[..., None] = print) -> str:
    """Pull the panel's EN low for low_s, then let it go: the line is requested open drain
    (writing 1 = released, never driven high) and left an input before the chip is closed, on
    every path once it was pulled low, including an exception or Ctrl+C during the wait (PPI-5).
    The code cannot see whether the wire is there: 'done' only means the GPIO calls succeeded.
    Returns 'done', 'skipped: …' (no lgpio, no chip: the daemon still serves the panel), or
    'failed: …'. A failed release is said at once through say (PPI-R2-5: also when Ctrl+C is
    on its way out and nothing is returned), and the original exception (if any) still raised.
    lg: lgpio module, say: where the release error is printed (tests)."""
    if lg is None:
        try:
            import lgpio as lg
        except ImportError:
            return 'skipped: lgpio not installed'
    try:
        h = lg.gpiochip_open(chip)
    except Exception as exc:                      # lgpio raises its own error type
        return f'skipped: gpiochip{chip}: {exc}'
    pulled, result = False, 'done'
    try:
        try:
            lg.gpio_claim_output(h, gpio, 0, getattr(lg, 'SET_OPEN_DRAIN', 0))
            pulled = True
            sleep(low_s)
        except Exception as exc:
            result = f'failed: GPIO{gpio}: {exc}'
    finally:
        if pulled:
            try:
                lg.gpio_free(h, gpio)
                lg.gpio_claim_input(h, gpio)      # left an input after close (RP1 may keep outputs)
            except Exception as exc:
                result = f'failed: GPIO{gpio} release: {exc} (EN may stay low: check the wire)'
                say('panel reset ' + result, flush=True)
        try:
            lg.gpiochip_close(h)
        except Exception:
            pass
    return result


def _num(s: str) -> Optional[float]:
    return None if s == 'None' else float(s)


def _posix_stop(proc, level: int) -> None:
    """level 0: SIGINT to run_hil.py only (soft STOP: setpoint 0, then END). level 1: SIGTERM,
    level 2: SIGKILL, both to the whole process group of the run (start_new_session: group id =
    the owner's pid), so HIL-Plant / HIL-TrajPub go too (PPI-1)."""
    if level == 0:
        os.kill(proc.pid, signal.SIGINT)
    else:
        os.killpg(proc.pid, signal.SIGTERM if level == 1 else signal.SIGKILL)


def _spawn_kw() -> dict:
    """Popen arguments: on POSIX the child leads its own session / process group, so the
    terminal's or systemd's SIGINT to the daemon does not reach it twice (PPI-2) and the
    daemon can signal the whole group (PPI-1, PPI-4)."""
    return {'start_new_session': True} if os.name == 'posix' else {}


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
        self._wedged = False                # a run's / probe's process survived SIGKILL: START, probe refused
        self._closing = False               # shutdown started: no new START, no new probe
        self._probe_proc = None
        self.meta_error: Optional[str] = None
        self.stop_term_s, self.stop_kill_s = STOP_TERM_S, STOP_KILL_S
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
            if self._closing or self._probing or self._wedged or self.busy_check(None):
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
            try:
                self._open_run_dir(name, argv)
                self.proc = subprocess.Popen(argv, cwd=HERE, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                                             encoding='utf-8', errors='replace', **_spawn_kw())
            except OSError as exc:
                self._fail(E_SPAWN, f'spawn {exc.errno}', f'spawn failed: {exc}')
                self._close_meta(rc=None)
                return pp.RES_FAILED
            self._meta['pid'] = self.proc.pid
            self._enter(pp.ST_PREPARE)
            threading.Thread(target=self._watch_owner, args=(self.proc,), name='run-owner', daemon=True).start()
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
            if self._t_stop is not None:
                return pp.RES_OK                   # STOP again: harmless, the timer escalates
            self._t_stop = self.clock()
            self.outcome = pp.OUT_USER_STOP
            proc = self.proc
            threading.Thread(target=self._escalate, args=(proc,), name='stop-escalate', daemon=True).start()
            if not self._seen_phase:
                self._stop_pending = True          # run_hil installs its STOP handler late: send at #phase
                return pp.RES_OK
            return self._signal(proc, 0)

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
            if (self.state in pp.ST_ACTIVE or self._probing or self._closing or self._wedged
                    or self.busy_check(None)):
                return False
            self._probing, self._t_probe = True, self.clock()
        argv = self.child_cmd(['--fixed-omega', '0,0,0,0', '--imu-static', '0,0,0,9.81,0',
                               '--steps', str(steps), '--no-traj', '--swa', 'off', *self.extra_args])
        p, gone = None, False                       # PPI-R2-2: 'gone' only from a real reap
        try:
            try:
                p = subprocess.Popen(argv, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding='utf-8', errors='replace', **_spawn_kw())
            except OSError:
                gone = True                         # nothing was started
                return self._probe_result('')
            with self._lock:
                self._probe_proc = p
            try:
                text, _ = p.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self._kill_probe(p)                 # PPI-4: the whole group, not only the owner
                text = p.communicate()[0] or ''
            gone = _reap_group(p.pid if _spawn_kw() else None)
            return self._probe_result(text)
        finally:
            if p is not None and not gone:          # exception, Ctrl+C, shutdown: still reap it
                self._kill_probe(p)
                try:
                    p.wait(timeout=5)
                except Exception:
                    pass
                gone = _reap_group(p.pid if _spawn_kw() else None) and p.poll() is not None
            with self._lock:
                self._probe_proc = None
                # a probe that would not die keeps START and probes refused until the daemon restarts
                self._probing, self._t_probe = False, self.clock()
                if not gone:
                    self._wedged = True

    def _probe_result(self, text: str) -> bool:
        m = _RE_MISO.search(text or '')
        ok = bool(m) and int(m.group(1)) > 0 and m.group(2) == '0' and m.group(3) == '0'
        with self._lock:
            self.h7_ok = ok
        return ok

    def _kill_probe(self, p) -> None:
        try:
            self.stop_fn(p, 2)
        except OSError:
            pass

    def maybe_reprobe(self, every_s: float = REPROBE_S) -> Optional[bool]:
        """Idle re-check of the H7 (daemon started before the H7 was up): probe when idle, the H7
        light is off and the last probe ended at least every_s ago. None = no probe this time.
        Never in --mock (the fake H7 always answers)."""
        with self._lock:
            if (self.mock or self.h7_ok or self.state in pp.ST_ACTIVE or self._probing or self._closing
                    or self._wedged
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

    def _signal(self, proc, level: int) -> int:
        """Send STOP level 0 / 1 / 2 to proc (see _posix_stop); recorded in meta.json."""
        self._meta.setdefault('stop_signals', []).append(level)
        try:
            self.stop_fn(proc, level)
        except (OSError, ProcessLookupError):
            return pp.RES_FAILED                   # effect unknown
        return pp.RES_OK

    def _escalate(self, proc) -> None:
        """Timer thread from the first STOP (PPI-1): soft STOP may not end the run (child hung,
        handler not installed, plant stuck), so level 1 after stop_term_s and level 2 after
        stop_term_s + stop_kill_s, whatever the child prints. Ends when the owner exits; the
        group's remaining processes are then reaped at once by _watch_owner (PPI-R2-1)."""
        t0, level = time.monotonic(), 0
        while proc.poll() is None:
            el = time.monotonic() - t0
            with self._lock:
                if self.proc is not proc:
                    return
                if level < 1 and el >= self.stop_term_s:
                    level = 1
                    self._stop_pending = False
                    self._signal(proc, 1)
                elif level < 2 and el >= self.stop_term_s + self.stop_kill_s:
                    level = 2
                    self._signal(proc, 2)
            time.sleep(0.05)

    def _watch_owner(self, proc) -> None:
        """Run thread (PPI-R2-1): as soon as the owner run_hil.py exits, by itself or killed,
        reap its process group (wait REAP_S, then SIGKILL), without waiting for the reader's EOF:
        a child of the group that still holds the stdout pipe would otherwise keep the reader,
        so the run, open for ever. The reader then sees EOF and finishes the run."""
        try:
            proc.wait()
        except Exception:
            return
        _reap_group(proc.pid if _spawn_kw() else None)

    def shutdown(self, wait_s: float = SHUTDOWN_S) -> bool:
        """Daemon stop (Ctrl+C, systemd SIGINT, a fatal error of the panel link; PPI-2, PPI-R2-2):
        no new START or probe from now on (closing), STOP the run, kill a probe in progress, then
        wait for the reader / probe to finish them (escalation and reaping included) so meta.json
        is closed. True only if nothing of the daemon's is left running (not after E_GROUP)."""
        with self._lock:
            self._closing = True
            active = self.proc is not None
            probe = self._probe_proc
        if active:
            self.stop()
        if probe is not None:
            self._kill_probe(probe)
        t_end = time.monotonic() + wait_s
        while time.monotonic() < t_end:
            with self._lock:
                if self.proc is None and not self._probing:
                    return not self._wedged
            time.sleep(0.05)
        return False

    def _open_run_dir(self, name: str, argv: List[str]) -> None:
        now = _dt.datetime.now()
        run_id = f'{now:%Y%m%d_%H%M%S}_{name}_{secrets.token_hex(2)}'
        self.run_dir = os.path.join(self.runs_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=False)
        self._meta = {'run_id': run_id, 'run_no': self.run_no, 'mode': 'panel', 'data_valid': False,
                      'data_valid_why': 'panel run = quick look / demo (L79, workflow use 1)',
                      'test': name, 'argv': argv, 'ctrl_declared': self.ctrl or None, 'swa_mode': None,
                      'pi_checkout': self._checkout, 'started': now.isoformat(timespec='seconds')}
        self._write_meta()

    def _write_meta(self) -> None:
        """meta.json; a write error (full card) is kept in memory, never stops the run (PPI-6)."""
        path = os.path.join(self.run_dir, 'meta.json')
        try:
            with open(path + '.tmp', 'w', encoding='utf-8') as f:
                json.dump(self._meta, f, indent=1)
            os.replace(path + '.tmp', path)
        except OSError as exc:
            self.meta_error = str(exc)

    def _close_meta(self, rc: Optional[int]) -> None:
        self._meta.update(ended=_dt.datetime.now().isoformat(timespec='seconds'), exit_code=rc,
                          outcome={pp.OUT_COMPLETED: 'completed', pp.OUT_USER_STOP: 'user_stop',
                                   pp.OUT_ERROR: 'error', pp.OUT_NONE: None}[self.outcome],
                          h7_ok_after=self.h7_ok, result=list(self.result))
        self._write_meta()

    def _parse(self, line: str, acc: dict) -> None:
        """One output line of run_hil.py (lock held): pha, stats, summary, MISO, SWA into the
        snapshot or acc (summary, miso, last). The single parser used by the reader and the tests."""
        m = _RE_PHASE.search(line)
        if m:
            self._seen_phase = True
            if self._stop_pending:
                self._stop_pending = False
                self._signal(self.proc, 0)
            st = _PHASE_ST[m.group(1)]
            if m.group(1) == 'RUN':
                self.h7_ok = True                  # ARM leaves only after 1 s of fresh H7 answers
            if st != self.state:
                self._enter(st)
            return
        m = _RE_WHOLE.search(line) or _RE_STATS.search(line)
        if m:
            p99 = _num(m.group(1))
            self.loop_us = None if p99 is None else int(round(p99))
            self.overruns = int(m.group(3))
            if _RE_WHOLE.search(line):
                acc['last'] = (p99, float(m.group(2)), int(m.group(3)))
            return
        m = _RE_SUMMARY.search(line)
        if m:
            try:
                acc['summary'] = json.loads(m.group(1))
            except ValueError:
                acc['summary'] = {'outcome': 'error:bad_summary'}
            return
        m = _RE_MISO.search(line)
        if m:
            acc['miso'] = tuple(int(g) for g in m.groups())
            return
        m = _RE_SWA.search(line)
        if m:
            self._meta['swa_mode'] = m.group(1)    # what SWA did in this panel run (I8)

    def _read(self, proc) -> None:
        """Reader thread: every line to run_hil.log, parsed for the panel; then the end. A log
        that cannot be written (full card), or a line the parser chokes on, never stops the
        reader (PPI-6): the output is still drained, the run still finished, the error kept."""
        acc = {'summary': None, 'miso': None, 'last': None}
        logf, errors = None, []
        try:
            try:
                logf = open(os.path.join(self.run_dir, 'run_hil.log'), 'w', encoding='utf-8')
            except OSError as exc:
                errors.append(f'log open: {exc}')
            for line in proc.stdout:
                if logf is not None:
                    try:
                        logf.write(line)
                    except OSError as exc:
                        errors.append(f'log write: {exc}')
                        try:
                            logf.close()
                        except OSError:
                            pass
                        logf = None
                try:
                    with self._lock:
                        self._parse(line, acc)
                except Exception as exc:           # keep draining, whatever the line
                    errors.append(f'parse: {type(exc).__name__}: {exc}')
        except Exception as exc:                   # stdout itself failed
            errors.append(f'read: {type(exc).__name__}: {exc}')
        finally:
            if logf is not None:
                try:
                    logf.close()
                except OSError:
                    pass
            try:
                rc = proc.wait()
            except Exception:
                rc = None
            gone = _reap_group(proc.pid if _spawn_kw() else None)
            with self._lock:
                if errors:
                    self._meta['reader_errors'] = errors[:20]
                self._finish(rc, acc['summary'], acc['miso'], acc['last'], gone)

    def _finish(self, rc: Optional[int], summary: Optional[dict], miso, last, group_gone: bool = True) -> None:
        self._meta.update(run_summary=summary, miso=miso, group_gone=group_gone)
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
        if not group_gone:
            # a process of the run would not die even by SIGKILL: it may hold the SPI
            self._fail(E_GROUP, 'run process left', 'process group of the run still alive after SIGKILL')
            self._wedged = True                    # START stays BUSY until the daemon restarts
        elif summary is None:
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
                    help='skip only the H7 check at start; the idle re-check still turns the light on (PPI-8)')
    ap.add_argument('--panel-reset-gpio', type=int, default=PANEL_RESET_GPIO,
                    help='BCM GPIO wired to the panel EN (pulled low once at start); -1 = no wire')
    ap.add_argument('--lock-dir', default=os.path.expanduser('~/.hil_panel_locks'),
                    help='one daemon per machine: its lock file lives here (tests only change it)')
    a = ap.parse_args(argv)
    os.makedirs(a.runs_dir, exist_ok=True)
    import serial  # pyserial
    from panel_link import PanelLink, nonblocking_writer, serve
    # PPI-7, PPI-R2-4: ONE daemon on this machine (one panel, one H7 per bench), whatever the
    # port name or epoch file: a fixed lock (--lock-dir, default ~/.hil_panel_locks), plus the
    # epoch file's owner lock; both taken before anything touches the panel or the H7.
    os.makedirs(a.lock_dir, exist_ok=True)
    with pp._FileLock(os.path.join(a.lock_dir, 'hil_daemon.lock'), timeout_s=0.5), \
            pp._FileLock(a.epoch_file + '.owner', timeout_s=0.5):
        daemon = HilDaemon(a.runs_dir, ctrl=a.ctrl, mock=a.mock)
        try:
            return _serve_panel(a, daemon, serial, PanelLink, nonblocking_writer, serve)
        except KeyboardInterrupt:
            pass
        finally:
            # PPI-2: one way out for Ctrl+C, systemd (KillSignal SIGINT, KillMode mixed: only the
            # daemon gets it, the run is in its own session) and errors: STOP the run, wait for it
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if daemon.proc is not None or daemon._probing:
                print('daemon stopping: STOP to the running run_hil.py, waiting for it ...', flush=True)
            ok = daemon.shutdown()
            print('daemon stopped' + ('' if ok else ': the run did not end in time (systemd kills the rest)'),
                  flush=True)
    return 0


def open_uart_when_ready(serial, port_path: str, baud: int, retry_s: float = 2.0,
                         say=print, sleep=time.sleep):
    """PPI-9: wait for the UART instead of failing (and restarting, and spending an epoch);
    one line when it starts waiting, one when it opens."""
    told = False
    while True:
        port = serial.Serial(baudrate=baud, timeout=0, write_timeout=0)
        port.dtr = port.rts = False
        port.port = port_path
        try:
            port.open()
            if told:
                say(f'{port_path}: open', flush=True)
            return port
        except (OSError, serial.SerialException) as exc:
            if not told:
                say(f'{port_path}: not ready ({exc}); retrying every {retry_s:.0f} s', flush=True)
                told = True
            sleep(retry_s)


def _serve_panel(a, daemon, serial, PanelLink, nonblocking_writer, serve) -> int:
    if a.panel_reset_gpio >= 0:
        print(f'panel reset GPIO{a.panel_reset_gpio}:', reset_panel(a.panel_reset_gpio), flush=True)
    if a.mock:
        daemon.h7_ok = True                        # fake H7 of run_hil.py --mock always answers
    else:
        if not a.no_probe:
            print('H7 probe ...', flush=True)
            print('H7 probe:', 'answers' if daemon.probe_h7()
                  else f'NO ANSWER (panel START blocked; probing again every {REPROBE_S:.0f} s while idle)',
                  flush=True)

        def reprobe_loop():
            while not daemon._closing:              # PPI-R2-2: no probe once shutdown started
                r = daemon.maybe_reprobe()
                if r is not None:
                    print('H7 re-probe:', 'answers' if r else 'no answer', flush=True)
                time.sleep(1.0)
        threading.Thread(target=reprobe_loop, name='h7-reprobe', daemon=True).start()
    port = open_uart_when_ready(serial, a.port, a.baud)
    with port:
        epoch = pp.next_daemon_epoch(a.epoch_file, create=a.create_epoch)   # only once the UART is up
        link = PanelLink(epoch, daemon, nonblocking_writer(port), baud=a.baud)
        print(f'hil_daemon on {a.port}, epoch {epoch}, {len(daemon.tests)} tests, runs in {a.runs_dir}; '
              'Ctrl+C to quit', flush=True)
        serve(port, link)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
