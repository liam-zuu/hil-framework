"""File and readiness interface of agv_logger for daemon tests. No UDP socket."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict', action='store_true', required=True)
    parser.add_argument('--src-ip', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--ready-file', required=True, type=Path)
    args = parser.parse_args()
    stopped = threading.Event()

    def watch_stdin():
        if sys.stdin.readline().strip() == 'STOP':
            stopped.set()

    threading.Thread(target=watch_stdin, daemon=True).start()
    args.out.write_text('timestamp_us,run_active,pos_reset_n,motor_on,pos_x_ref,pos_y_ref,pos_x,pos_y\n'
                        '1000,1,2,1,0,0,0,0\n51000,1,2,1,.05,0,.06,0\n')
    with Path(str(args.out) + '.events.txt').open('w', encoding='utf-8') as events:
        args.ready_file.write_text(json.dumps(dict(state='LOGGER_OPEN', pid=os.getpid())))
        gate = args.out.parent / 'motor_on'
        while not gate.is_file():
            if stopped.wait(.01):
                return
        events.write('2026-11-28T10:00:00 #fw,sha=18820107799e,dirty=0,env=pid\n')
        events.write('2026-11-28T10:00:01 #fw,sha=18820107799e,dirty=')
        events.flush()
        time.sleep(.2)  # the tailer can observe the partial line
        events.write('1,env=ladrc\n')
        events.flush()
        if not stopped.wait(10):
            raise RuntimeError('daemon did not stop its logger')


if __name__ == '__main__':
    main()
