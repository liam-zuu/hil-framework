"""
s02_plan.py: spec S02 group A on HIL or the car: the run list (runs.csv frame) in block order with the test
order drawn from a seed, and a check of the filled-in runs.csv.

Design: 03_CODE/docs/2026-11-15_Thiet_ke_CODE_S02_v1.md (repo luan van) section 12; Liam 15/11, CODE_S02
local_ff3bd6e3, "Gom theo dieu kien (Recommended)"; spec S02 v1.10 P2-2 (Thu tu), P2-4, P2-5 (Khoi), P2-6
(runs.csv), P2-7 (A-S after the S05 final check).

    python tools/s02_plan.py plan --platform hil|real --order-seed N --out <run_dir>
                                  [--controllers PID,LADRC] [--reps 5] [--with-strafe] [--prep-s 3]
    python tools/s02_plan.py check <run_dir>

plan: a block is one controller x one condition (floor or lifted) x one repetition (P2-5). Blocks grouped by
condition (Liam 15/11): every floor block of the N repetitions, then every lifted block; inside a
condition, repetition 1..N, and the two controllers alternate, the first one switching with the
repetition (rep 1: PID, LADRC; rep 2: LADRC, PID; ...). The tests of a block are shuffled by one
numpy default_rng(order_seed) stream, drawn block after block in this order. seed = noise_seed = the
repetition (S02 v1.4). A-S1, A-S2 only with --with-strafe. Each change of block is a change of ESP32 binary
(env column: P2-2 floor and lifted binaries). Writes <run_dir>/runs.csv (never over an existing file) and
<run_dir>/s02_plan.json (arguments, seed, the block list); meta.yaml stays the operator's (order_seed goes
there, P2-2).
  runs.csv columns: run_id, block, order_in_block, platform, controller, condition, test_id, speed_mps, rep,
  seed, noise_seed, env, command, file, then for the operator: v_batt_start, v_batt_end (REAL), aborted,
  result, reason, cause_group. tools/s02_metrics.py reads run_id, platform, controller, test_id, speed_mps,
  rep, file, aborted.
check: every row has result completed | broken | replaced; replaced needs cause_group 1, 2 or 3 (the closed
list of external causes, P2-5) and a later row of the same cell (test x controller x repetition); broken
needs a reason; aborted rows are not completed; per cell at most one completed or broken row (the
5 runs are fixed, a broken run is not re-run, P2-4). Prints the problems; exit 0 none, 1 some, 2 error.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rpi5'))
from trajectory_pub import GROUP_A, LIFTED_TESTS  # noqa: E402

FLOOR = ('A-L1', 'A-L2', 'A-C1', 'A-C2', 'A-F1', 'A-F2')
STRAFE = ('A-S1', 'A-S2')
ENV = {('PID', 'floor'): 'esp32-s3-devkitc-1', ('LADRC', 'floor'): 'ladrc',
       ('PID', 'lifted'): 'wheels_lifted', ('LADRC', 'lifted'): 'ladrc_wheels_lifted'}   # AGV platformio.ini
COLUMNS = ['run_id', 'block', 'order_in_block', 'platform', 'controller', 'condition', 'test_id', 'speed_mps',
           'rep', 'seed', 'noise_seed', 'env', 'command', 'file',
           'v_batt_start', 'v_batt_end', 'aborted', 'result', 'reason', 'cause_group']
RESULTS = ('completed', 'broken', 'replaced')
HIL_ARGS = '--ground traction --mu 0.436 --imu-noise 20'     # S02 P2-2: traction, mu_hd 0.436, IMU 20 Hz


def blocks(controllers, reps):
    """[(condition, rep, controller)] in run order (module docstring)."""
    out = []
    for cond in ('floor', 'lifted'):
        for rep in range(1, reps + 1):
            order = controllers if rep % 2 == 1 else controllers[::-1]
            out += [(cond, rep, c) for c in order]
    return out


def command(platform, test_id, run_id, rep, prep_s):
    if platform == 'hil':
        return (f'--run {run_id}="--test {test_id} --s02-prep {prep_s:g} {HIL_ARGS} --seed {rep}"')
    return f'python rpi5/s02_sender.py --ip <IP xe> --test {test_id} --prep-s {prep_s:g}' + (
        ' --allow-strafe' if test_id in STRAFE else '')


def plan(platform, controllers, reps, order_seed, with_strafe=False, prep_s=3.0):
    """runs.csv rows and the block list."""
    if platform not in ('hil', 'real'):
        raise ValueError(f'--platform {platform}: hil or real (SIM orders its runs itself)')
    if sorted(controllers) != ['LADRC', 'PID']:
        raise ValueError(f'--controllers {controllers}: PID and LADRC (S02 P2-2)')
    rng = np.random.default_rng(order_seed)
    tests = {'floor': list(FLOOR) + (list(STRAFE) if with_strafe else []), 'lifted': list(LIFTED_TESTS)}
    rows, blist = [], []
    for b, (cond, rep, ctrl) in enumerate(blocks(controllers, reps), start=1):
        order = list(tests[cond])
        rng.shuffle(order)
        blist.append(dict(block=b, condition=cond, rep=rep, controller=ctrl, env=ENV[(ctrl, cond)], tests=order))
        for i, tid in enumerate(order, start=1):
            run_id = f'{tid}_{ctrl}_r{rep}'
            rows.append(dict(run_id=run_id, block=b, order_in_block=i, platform=platform, controller=ctrl,
                             condition=cond, test_id=tid, speed_mps=GROUP_A[tid].v, rep=rep, seed=rep,
                             noise_seed=rep if platform == 'hil' else '', env=ENV[(ctrl, cond)],
                             command=command(platform, tid, run_id, rep, prep_s),
                             file=('pi/' if platform == 'hil' else 'logs/') + f'{run_id}.csv',
                             v_batt_start='', v_batt_end='', aborted='', result='', reason='', cause_group=''))
    return rows, blist


def check(rows):
    """Problems of a filled-in runs.csv (module docstring)."""
    out = []
    if not rows:
        return ['runs.csv has no row']
    miss = [c for c in COLUMNS if c not in rows[0]]
    if miss:
        return [f'runs.csv lacks columns {miss}']
    done = {}
    for i, r in enumerate(rows, start=2):
        cell = (r['test_id'], r['controller'], r['rep'])
        res = r['result'].strip()
        if res not in RESULTS:
            out.append(f'line {i} {r["run_id"]}: result {res!r} not in {RESULTS}')
            continue
        if res == 'replaced':
            if r['cause_group'].strip() not in ('1', '2', '3'):
                out.append(f'line {i} {r["run_id"]}: replaced needs cause_group 1, 2 or 3 (P2-5)')
            if not any((x['test_id'], x['controller'], x['rep']) == cell for x in rows[i - 1:]):
                out.append(f'line {i} {r["run_id"]}: replaced but no later run of {cell}')
            continue
        if res == 'broken' and not r['reason'].strip():
            out.append(f'line {i} {r["run_id"]}: broken needs a reason (P2-5)')
        if res == 'completed' and r['aborted'].strip() not in ('', '0'):
            out.append(f'line {i} {r["run_id"]}: aborted run marked completed')
        if cell in done:
            out.append(f'line {i} {r["run_id"]}: {cell} already has a {done[cell]} run (5 fixed runs, P2-4)')
        done[cell] = res
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--platform', required=True, choices=['hil', 'real'])
    p.add_argument('--controllers', default='PID,LADRC')
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--order-seed', type=int, required=True)
    p.add_argument('--with-strafe', action='store_true', help='A-S1, A-S2 (S02 P2-7: after the S05 final check)')
    p.add_argument('--prep-s', type=float, default=3.0)
    p.add_argument('--out', required=True)
    c = sub.add_parser('check')
    c.add_argument('run_dir')
    a = ap.parse_args(argv)
    if a.cmd == 'plan':
        path = os.path.join(a.out, 'runs.csv')
        js = os.path.join(a.out, 's02_plan.json')
        if os.path.exists(path) or os.path.exists(js):
            print(f'{a.out} already has runs.csv or s02_plan.json: use a new folder')
            return 2
        try:
            rows, blist = plan(a.platform, a.controllers.split(','), a.reps, a.order_seed, a.with_strafe, a.prep_s)
        except ValueError as exc:
            print(exc)
            return 2
        os.makedirs(a.out, exist_ok=True)
        with open(path, 'x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        with open(js, 'x', encoding='utf-8') as f:
            json.dump(dict(argv=sys.argv, platform=a.platform, controllers=a.controllers, reps=a.reps,
                           order_seed=a.order_seed, with_strafe=a.with_strafe, prep_s=a.prep_s,
                           numpy=np.__version__, blocks=blist), f, indent=1)
        print(f'{len(rows)} runs in {len(blist)} blocks -> {path}; order_seed {a.order_seed} goes into meta.yaml')
        return 0
    try:
        with open(os.path.join(a.run_dir, 'runs.csv'), newline='') as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        print(exc)
        return 2
    probs = check(rows)
    for x in probs:
        print(x)
    print(f'#ket_qua,ma={1 if probs else 0}')
    return 1 if probs else 0


if __name__ == '__main__':
    raise SystemExit(main())
