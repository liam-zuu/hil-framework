"""Stand-in for run_hil.py in test_hil_daemon.py: prints the lines hil_daemon.py parses, in the
format of the real one (logging 'HH:MM:SS INFO     hil.main: ...'), and nothing else. The
scenario comes from FAKE_RUN_HIL; STOP is the line 'STOP' on stdin (test stop_fn) or SIGINT.

  ok         PREPARE, ARM, RUN (RUN_S s or until STOP), STOPPING, END; completed / user_stop; exit 0
  no_h7      PREPARE, ARM, STOPPING, END; error:arm_timeout no_h7; exit 3
  crash      PREPARE, then exit 1 without #run_summary
  miso_bad   as ok, with crc_bad=2 in the #miso line
  tail_cut   as ok (completed), but exit 3 (END tail cut, hil_node.run_failed)
  hang       PREPARE, then ignores STOP (stdin and SIGINT) and sleeps 60 s: only a kill ends it
  orphan     PREPARE, then exits 0 at once; a child holds stdout FAKE_ORPHAN_S s (default 3)
FAKE_HOLD_FILE: wait until that file exists before the first line (a STOP before any #phase).
Probe calls (--fixed-omega in argv): '#miso' line only, crc_bad from FAKE_PROBE (ok / bad / none;
hang = sleeps 60 s).
"""
import json
import os
import signal
import sys
import threading
import time

stop = threading.Event()
RUN_S = float(os.environ.get('FAKE_RUN_S', '0.4'))


def say(msg):
    print(time.strftime('%H:%M:%S') + ' INFO     hil.main: ' + msg, flush=True)


def phase(p):
    say(f'#phase,{p},t_ns={time.monotonic_ns()}')


def watch_stdin():
    for line in sys.stdin:
        if line.strip() == 'STOP':
            stop.set()


def main():
    if '--fixed-omega' in sys.argv:
        probe = os.environ.get('FAKE_PROBE', 'ok')
        if probe == 'hang':                     # a probe that never ends by itself
            time.sleep(60)
            return 9
        if probe != 'none':
            say('#miso,checked=1490,crc_bad=%d,empty=0,first_crc_bad=None,first_empty=None'
                % (0 if probe == 'ok' else 3))
        return 0
    scen = os.environ.get('FAKE_RUN_HIL', 'ok')
    with open(os.environ['FAKE_ARGV_OUT'], 'w') if os.environ.get('FAKE_ARGV_OUT') else open(os.devnull, 'w') as f:
        f.write(json.dumps(sys.argv[1:]))
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, lambda *_: stop.set())
    threading.Thread(target=watch_stdin, daemon=True).start()
    hold = os.environ.get('FAKE_HOLD_FILE')
    while hold and not os.path.exists(hold):
        time.sleep(0.01)
    phase('PREPARE')
    if scen == 'hang':
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        time.sleep(60)
        return 9
    if scen == 'orphan':                        # owner exits, a child keeps the stdout pipe
        import subprocess
        # FAKE_ORPHAN_GATE: the child waits for that file, then prints a stale '#phase,RUN' line
        # (a line still in the pipe after the run was finished), then holds the pipe on
        child = ('import os, sys, time\n'
                 'g = os.environ.get("FAKE_ORPHAN_GATE")\n'
                 'while g and not os.path.exists(g): time.sleep(0.01)\n'
                 'if g: print("00:00:00 INFO     hil.main: #phase,RUN,t_ns=2", flush=True)\n'
                 'time.sleep(float(os.environ.get("FAKE_ORPHAN_S", "3")))\n')
        subprocess.Popen([sys.executable, '-c', child], stdout=sys.stdout, stderr=sys.stdout)
        return 0
    time.sleep(0.05)
    if scen == 'crash':
        return 1
    say('#run_start,timestamp_us=1,max_steps=40000,frame_log=-')
    phase('ARM')
    time.sleep(0.1)
    if scen == 'no_h7':
        phase('STOPPING')
        say('#run_summary,' + json.dumps({'outcome': 'error:arm_timeout', 'detail': 'no_h7'}))
        say('#miso,checked=0,crc_bad=0,empty=0,first_crc_bad=None,first_empty=None')
        phase('END')
        return 3
    if not stop.is_set():
        phase('RUN')
        say('step=10000    t= 10.0 s  rate=  1000 Hz  jitter (last 10000 ticks) p99=41.5 us  max=120.0 us  '
            '>warn: 0  overruns=0')
        stop.wait(RUN_S)
    phase('STOPPING')
    time.sleep(0.05)
    outcome = 'user_stop' if stop.is_set() else 'completed'
    say('#run_summary,' + json.dumps({'outcome': outcome, 'detail': None, 'steps': 1234}))
    say('Plant loop jitter, whole run (n=1234): p99=43.2 us  p99.9=80.0 us  max=150.0 us  >warn: 0  overruns=1')
    say('#swa,mode=phase,events=[],tail={}')
    say('#miso,checked=1200,crc_bad=%d,empty=0,first_crc_bad=None,first_empty=None'
        % (2 if scen == 'miso_bad' else 0))
    say('#run_end,timestamp_us=2,steps=1234,reason=end')
    phase('END')
    return 3 if scen == 'tail_cut' else 0


if __name__ == '__main__':
    sys.exit(main())
