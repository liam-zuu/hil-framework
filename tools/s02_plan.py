"""
s02_plan.py: spec S02 group A on HIL or the car: the run list (runs.csv frame) in block order with the test
order drawn from a seed, and a check of the filled-in runs.csv.

Design: 03_CODE/docs/2026-11-15_Thiet_ke_CODE_S02_v1.md (repo luan van) section 12; Liam 15/11, CODE_S02
local_ff3bd6e3, "Gom theo dieu kien (Recommended)"; spec S02 v1.10 P2-2 (Thu tu), P2-4, P2-5 (Khoi), P2-6
(runs.csv), P2-7 (A-S after the S05 final check).

    python tools/s02_plan.py plan --platform hil|real --order-seed N --out <run_dir>
                                  [--controllers PID,LADRC] [--reps 5] [--with-strafe] [--prep-s 3] [--trial]
    python tools/s02_plan.py check <run_dir> [--trial]

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
check (S2P1-8, -9, -14, Codex round 1): every row has result completed | broken | replaced; replaced needs
cause_group 1, 2 or 3 (the closed list of external causes, P2-5) and a later row of the same cell (test x
controller x repetition, rep as a number); broken needs a reason; aborted rows are not completed; per cell
exactly one completed or broken row, and with s02_plan.json in the folder every planned cell and no other
(the N runs are fixed, a broken run is not re-run, P2-4). Battery (REAL, P2-5): each block has the rest
voltage before its first and after its last run; a sag > 0.3 V needs block_status battery_rerun on every row
of the block and a later block of the same condition, repetition and controller; battery_rerun rows stay as
history and are not counted (tools/s02_metrics.py skips them, and replaced rows). Codex round 2: the main
numbers need s02_plan.json and repetitions exactly 1..5 (S2P1-8; another N only with plan --trial, checked
with check --trial); battery_rerun only on a REAL block whose own rest voltages sagged > 0.3 V, compared as
decimals of the written values (12.4 -> 12.1 is not over), and the later block runs every test of it again
(S2P2-3, S2P2-4); a completed run has missing_pct 0..100 and gen_pause_n >= 0 filled in (S2P1-14). Codex round 1
of v3 (S2M1-6): every row of a block has the same platform, controller, condition and repetition, the rows of a
block are together, each test belongs to its condition; with s02_plan.json each planned block keeps its key and a
row for every planned test, and a later block (not planned) has the key of a planned block and only its tests.
Prints the problems; exit 0 none, 1 some, 2 error.
"""
import argparse
import csv
import json
import os
import sys
from decimal import Decimal, InvalidOperation

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
           'v_batt_start', 'v_batt_end', 'block_status', 'aborted', 'result', 'reason', 'cause_group',
           'missing_pct', 'gen_pause_n']          # S2P1-14: P2-6 missing_pct, generation pauses
RESULTS = ('completed', 'broken', 'replaced')
BLOCK_STATUS = ('', 'ok', 'battery_rerun')     # S2P1-9: a block whose battery sagged > 0.3 V (P2-5)
SAG_MAX_V = Decimal('0.3')                     # P2-5; compared as decimals of the written values (S2P2-4)
REPS_MAIN = 5                                  # P2-4: 5 runs per cell; another N only as a declared trial (S2P1-8)
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


def plan(platform, controllers, reps, order_seed, with_strafe=False, prep_s=3.0, trial=False):
    """runs.csv rows and the block list. reps other than REPS_MAIN only with trial (S2P1-8, Codex round 2)."""
    if reps != REPS_MAIN and not trial:
        raise ValueError(f'--reps {reps}: the main numbers have {REPS_MAIN} runs per cell (P2-4); a trial with '
                         'another N needs --trial')
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
                             v_batt_start='', v_batt_end='', block_status='', aborted='', result='', reason='',
                             cause_group='', missing_pct='', gen_pause_n=''))
    return rows, blist


def _num(x):
    try:
        v = float(str(x).strip())
    except ValueError:
        return None
    return v if v == v and abs(v) != float('inf') else None     # S2P3-5: inf, 1e309 are not numbers here


def _dec(x):
    """The written value as a Decimal (12.4 - 12.1 is exactly 0.3, S2P2-4), None if not a finite number."""
    try:
        v = Decimal(str(x).strip())
    except InvalidOperation:
        return None
    return v if v.is_finite() else None


def _volts(rs):
    v0 = next((_dec(r['v_batt_start']) for r in rs if _dec(r['v_batt_start']) is not None), None)
    v1 = next((_dec(r['v_batt_end']) for r in reversed(rs) if _dec(r['v_batt_end']) is not None), None)
    return v0, v1


def _block_problems(rows, blocks, planned):
    """S2M1-6 (Codex round 1 of v3; S2P3-4): a block is one platform x one controller x one condition x one
    repetition on every row, not only the first (P2-5 "Khoi"); its rows are together in runs.csv; each test belongs
    to the row's condition; with s02_plan.json, a planned block keeps its controller, condition, repetition and a row
    for every planned test of it (a test moved to another block is refused), and a block not in the plan (a later
    block: battery re-run, replacement runs) has the key of a planned block and only its tests."""
    out = []
    cond_tests = {'floor': set(FLOOR) | set(STRAFE), 'lifted': set(LIFTED_TESTS)}
    idx = {}
    for i, r in enumerate(rows):
        idx.setdefault(r['block'].strip(), []).append(i)
    key = {}
    for b, rs in blocks.items():
        ix = idx[b]
        if ix != list(range(ix[0], ix[-1] + 1)):
            out.append(f'block {b}: its rows are not together in runs.csv (lines {[i + 2 for i in ix]}; P2-5)')
        ks = {(r['platform'].strip().lower(), r['controller'].strip(), r['condition'].strip(), r['_rep']) for r in rs}
        if len(ks) > 1:
            out.append(f'block {b}: rows of {len(ks)} (platform, controller, condition, rep) {sorted(ks, key=str)}: '
                       'one block is one controller x one condition x one repetition (P2-5)')
        key[b] = (rs[0]['controller'].strip(), rs[0]['condition'].strip(), rs[0]['_rep'])
    for i, r in enumerate(rows, start=2):
        if r['test_id'].strip() not in cond_tests.get(r['condition'].strip(), set()):
            out.append(f'line {i} {r["run_id"]}: test {r["test_id"]!r} is not a {r["condition"]!r} test')
    if planned is None:
        return out
    pb = {str(bl['block']).strip(): bl for bl in planned}
    pkey = {b: (bl['controller'], bl['condition'], int(bl['rep'])) for b, bl in pb.items()}
    for b, rs in blocks.items():
        tests = [r['test_id'].strip() for r in rs]
        if b in pb:
            bl = pb[b]
            if key[b] != pkey[b]:
                out.append(f'block {b}: {key[b]} in runs.csv, {pkey[b]} in s02_plan.json')
            extra, miss = set(tests) - set(bl['tests']), set(bl['tests']) - set(tests)
            if extra:
                out.append(f'block {b}: tests {sorted(extra)} not planned in this block (s02_plan.json)')
            if miss:
                out.append(f'block {b}: planned tests {sorted(miss)} have no row in this block')
        else:
            same = [b2 for b2, k in pkey.items() if k == key[b]]
            if not same:
                out.append(f'block {b}: not in s02_plan.json and no planned block {key[b]}')
            elif not set(tests) <= set(pb[same[0]]['tests']):
                out.append(f'block {b}: tests {sorted(set(tests) - set(pb[same[0]]["tests"]))} not in the planned '
                           f'block {same[0]} {key[b]}')
    for b in pb:
        if b not in blocks:
            out.append(f'block {b}: planned (s02_plan.json), no row in runs.csv')
    return out


def check(rows, planned=None, n_reps=None):
    """Problems of a filled-in runs.csv (module docstring). planned: the s02_plan.json block list, when the
    folder has it: the exact set of cells (test x controller x repetition) the session must cover. n_reps: the
    planned repetitions must be exactly 1..n_reps (main numbers: REPS_MAIN; None for a declared trial)."""
    out = []
    if not rows:
        return ['runs.csv has no row']
    miss = [c for c in COLUMNS if c not in rows[0]]
    if miss:
        return [f'runs.csv lacks columns {miss}']
    # repetitions as numbers (S2P1-8: "1" and "01" are one cell)
    for i, r in enumerate(rows, start=2):
        v = _num(r['rep'])
        ok = v is not None and v == int(v) and v >= 1
        if not ok:
            out.append(f'line {i} {r["run_id"]}: rep {r["rep"]!r} is not a repetition number')
        r['_rep'] = int(v) if ok else None
        if r['block_status'].strip() not in BLOCK_STATUS:
            out.append(f'line {i} {r["run_id"]}: block_status {r["block_status"]!r} not in {BLOCK_STATUS}')
    # battery per block (S2P1-9, P2-5): rest voltage before the first and after the last run of the block
    blocks = {}
    for r in rows:
        blocks.setdefault(r['block'].strip(), []).append(r)
    order = {b: rows.index(rs[0]) for b, rs in blocks.items()}
    out += _block_problems(rows, blocks, planned)
    for b, rs in blocks.items():
        st = {r['block_status'].strip() for r in rs}
        if 'battery_rerun' in st and st != {'battery_rerun'}:
            out.append(f'block {b}: battery_rerun on some rows only {sorted(st)}')
        real = rs[0]['platform'].strip().lower() == 'real'
        v0, v1 = _volts(rs) if real else (None, None)
        sagged = v0 is not None and v1 is not None and v0 - v1 > SAG_MAX_V
        if st == {'battery_rerun'}:
            # S2P2-3 (Codex round 2): only a block whose own rest voltages sagged > 0.3 V may be set aside, and the
            # later block of the same condition, repetition and controller runs every test of it again
            if not sagged:
                why = 'not a REAL block' if not real else f'{v0} -> {v1} V'
                out.append(f'block {b}: battery_rerun without its own sag > {SAG_MAX_V} V ({why}; P2-5)')
            key = (rs[0]['condition'], rs[0]['_rep'], rs[0]['controller'])
            tests = {r['test_id'] for r in rs}
            later = [b2 for b2, r2 in blocks.items() if order[b2] > order[b]
                     and {x['block_status'].strip() for x in r2} != {'battery_rerun'}
                     and (r2[0]['condition'], r2[0]['_rep'], r2[0]['controller']) == key]
            if not later:
                out.append(f'block {b}: battery_rerun but no later block of {key}')
            elif len(later) > 1:
                # S2P3-4 (Codex round 3): one whole replacement block, its own battery checked as a block; small
                # blocks whose union covers the tests would each pass the sag rule while the whole re-run sags
                out.append(f'block {b}: battery_rerun replaced by {len(later)} blocks {sorted(later)}: one whole '
                           'block of the same condition, repetition and controller (P2-5)')
            else:
                again = {x['test_id'] for x in blocks[later[0]]}
                if not tests <= again:
                    out.append(f'block {b}: the later block {later[0]} of {key} does not run again '
                               f'{sorted(tests - again)}')
        if not real:
            continue
        if v0 is None or v1 is None:
            out.append(f'block {b}: rest voltage before / after the block missing (P2-5)')
        elif sagged and st != {'battery_rerun'}:
            out.append(f'block {b}: battery sagged {v0 - v1} V > {SAG_MAX_V} V: mark the block battery_rerun '
                       'and run it again (P2-5)')
    # runs (P2-4, P2-5): outside battery_rerun blocks, each cell exactly one completed or broken run
    done = {}
    live = [r for r in rows if r['block_status'].strip() != 'battery_rerun']
    for j, r in enumerate(live):
        i = rows.index(r) + 2
        cell = (r['test_id'], r['controller'], r['_rep'])
        res = r['result'].strip()
        if res not in RESULTS:
            out.append(f'line {i} {r["run_id"]}: result {res!r} not in {RESULTS}')
            continue
        if res == 'replaced':
            if r['cause_group'].strip() not in ('1', '2', '3'):
                out.append(f'line {i} {r["run_id"]}: replaced needs cause_group 1, 2 or 3 (P2-5)')
            if not any((x['test_id'], x['controller'], x['_rep']) == cell for x in live[j + 1:]):
                out.append(f'line {i} {r["run_id"]}: replaced but no later run of {cell}')
            continue
        if res == 'broken' and not r['reason'].strip():
            out.append(f'line {i} {r["run_id"]}: broken needs a reason (P2-5)')
        if res == 'completed' and r['aborted'].strip() not in ('', '0'):
            out.append(f'line {i} {r["run_id"]}: aborted run marked completed')
        if res == 'completed':
            # S2P1-14 (Codex round 2): the P2-6 loss fields filled in (empty is not 0), finite, in range
            mp, gp = _num(r['missing_pct']), _num(r['gen_pause_n'])
            if mp is None or not 0.0 <= mp <= 100.0:
                out.append(f'line {i} {r["run_id"]}: missing_pct {r["missing_pct"]!r} not a percentage 0..100')
            if gp is None or gp != int(gp) or gp < 0:
                out.append(f'line {i} {r["run_id"]}: gen_pause_n {r["gen_pause_n"]!r} not a count >= 0')
        if cell in done:
            out.append(f'line {i} {r["run_id"]}: {cell} already has a {done[cell]} run (5 fixed runs, P2-4)')
        done[cell] = res
    if planned is not None:
        want = {(t, bl['controller'], int(bl['rep'])) for bl in planned for t in bl['tests']}
        for c in sorted(want - set(done), key=str):
            out.append(f'cell {c}: planned, no completed or broken run (P2-4)')
        for c in sorted(set(done) - want, key=str):
            out.append(f'cell {c}: not in s02_plan.json')
        reps = {int(bl['rep']) for bl in planned}
        if reps != set(range(1, max(reps) + 1)):
            out.append(f'planned repetitions {sorted(reps)} are not 1..N')
        elif n_reps is not None and reps != set(range(1, n_reps + 1)):
            out.append(f'planned repetitions {sorted(reps)} are not 1..{n_reps} (P2-4; a trial needs --trial)')
        if n_reps is not None:
            # S2P3-6 (Codex round 3): every planned (test, controller) has exactly the repetitions 1..n_reps
            per = {}
            for bl in planned:
                for t in bl['tests']:
                    per.setdefault((t, bl['controller']), []).append(int(bl['rep']))
            for tc, rr in sorted(per.items()):
                if sorted(rr) != list(range(1, n_reps + 1)):
                    out.append(f'{tc}: planned repetitions {sorted(rr)}, not 1..{n_reps} (P2-4)')
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
    p.add_argument('--trial', action='store_true', help=f'a trial plan (--reps other than {REPS_MAIN}): not main numbers')
    p.add_argument('--out', required=True)
    c = sub.add_parser('check')
    c.add_argument('run_dir')
    c.add_argument('--trial', action='store_true', help='check a trial folder (no plan, or a trial plan, accepted)')
    a = ap.parse_args(argv)
    if a.cmd == 'plan':
        path = os.path.join(a.out, 'runs.csv')
        js = os.path.join(a.out, 's02_plan.json')
        if os.path.exists(path) or os.path.exists(js):
            print(f'{a.out} already has runs.csv or s02_plan.json: use a new folder')
            return 2
        try:
            rows, blist = plan(a.platform, a.controllers.split(','), a.reps, a.order_seed, a.with_strafe, a.prep_s,
                               a.trial)
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
                           order_seed=a.order_seed, with_strafe=a.with_strafe, prep_s=a.prep_s, trial=a.trial,
                           numpy=np.__version__, blocks=blist), f, indent=1)
        print(f'{len(rows)} runs in {len(blist)} blocks -> {path}; order_seed {a.order_seed} goes into meta.yaml')
        return 0
    try:
        with open(os.path.join(a.run_dir, 'runs.csv'), newline='') as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        print(exc)
        return 2
    planned, pre = None, []
    js = os.path.join(a.run_dir, 's02_plan.json')
    if os.path.isfile(js):
        with open(js, encoding='utf-8') as f:
            pj = json.load(f)
        planned = pj['blocks']
        if pj.get('trial') and not a.trial:
            pre.append('s02_plan.json is a trial plan: not main numbers (check --trial to check it as a trial)')
        # S2P3-6 (Codex round 3): the manifest must be the plan of its own arguments (an edited block list, e.g.
        # one cell dropped from it and from runs.csv, is refused)
        try:
            again = plan(pj['platform'], pj['controllers'].split(','), int(pj['reps']), int(pj['order_seed']),
                         bool(pj['with_strafe']), float(pj['prep_s']), trial=True)[1]
        except (KeyError, ValueError, TypeError) as exc:
            again = None
            pre.append(f's02_plan.json: cannot rebuild the plan from its arguments ({exc})')
        if again is not None and json.loads(json.dumps(again)) != planned:
            pre.append('s02_plan.json: the block list is not the plan of its own arguments (edited?)')
    elif a.trial:
        print('no s02_plan.json: the planned cells are not checked (trial)')
    else:
        # S2P1-8 (Codex round 2): the main numbers need the plan, or a missing cell passes
        pre.append('no s02_plan.json: the planned cells cannot be checked; main numbers need it (P2-4)')
    probs = pre + check(rows, planned, None if a.trial else REPS_MAIN)
    for x in probs:
        print(x)
    print(f'#ket_qua,ma={1 if probs else 0}')
    return 1 if probs else 0


if __name__ == '__main__':
    raise SystemExit(main())
