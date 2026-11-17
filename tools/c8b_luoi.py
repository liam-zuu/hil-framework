#!/usr/bin/env python3
"""
c8b_luoi.py: score one O1 floor run on the clock grid of c8b_gio.py and combine the node results into the three
states of each condition (C8b design v4 section 3 "Luoi gio", approved: Liam 17/11, CODE_C8B local_cccce6cb,
"Duyet (Recommended)"; CLAUDE_THESIS 03_CODE/docs/2026-11-17_Thiet_ke_C8b_v4.md). Written by CODE_C8B
local_21b7debc on feat/c8b-u.

At every node D_n of the clock file the WHOLE camera scorer runs (frame set, common / union domains, t_deadline,
G1-G5) with U_t of section 3 (the node step 1 ms in it). Combination per condition (C8B4R2-2 rounds 3, 4; C8B4R5-1,
R5-2):
  dat           dat at every node;
  khong_dat     G2-G5: every grid cell (two adjacent nodes) has a violating frame (finite lo_k above the threshold,
                lo_k of either node: each node's U_t holds the whole step, so it bounds the whole cell) that is in the
                common domain at BOTH nodes of the cell (so in the domain all through the cell; two different frames
                violating at the two nodes are not enough);
                G1 (no lo_k): khong_dat at every node and the G1 state the same at the two nodes of every cell;
  chua_ket_luan otherwise. Zero-width interval (one node): the state of that node. A node not scored is never
                dropped: every condition is chua_ket_luan at that node. Two kinds (hong_nut / loi_nut): the run lost to
                the measurement system at that node (scorer outcome hong_he_do), or a tool problem (scorer refused,
                files missing or without the C8b columns) [CACH HIEU: design 3 "nut loi"; only the first counts as
                lost].
Run outcome (CODE_GATE v2 section 2, luoi_v1; Liam 17/11 at CODE_GATE "Duyet, code luon", "Khi moi nut deu co"):
lost to the measurement system when EVERY node is lost, or the clock file says so (design 7; O1 v11.34 3.5 Q2: the
per-run checks); the clock file without a finite bound for another reason: G1-G5 chua_ket_luan; khong_dat
(vi_pham_truoc_loi_he_do) when EVERY node has the violation-before-t1 exception (cat_t1); else O1 order over the
DECIDING conditions (all five, minus the report-only ones of the level from the C8 gate file): one khong_dat ->
khong_dat; else one chua_ket_luan -> chua_ket_luan; else dat.
Gate block written (gate_v3, read by rq3_log, which recomputes the states): the block of node nut_tam (the scored
node nearest the centre: G1 fields, constants) with G1 state = the grid state, G2-G5 numbers: common lo = smallest
over the cells of the largest cell-violation value above (so khong_dat <=> lo > threshold, the grid rule); hi,
n_undef, q = largest over the nodes, n_undef at least 1 when a node is not scored; u_t = largest node U_t; plus the
luoi_v1 block (gio_sha256, n_nut, d_dau_s, d_cuoi_s, buoc_max_s, nut_tam, g1_nut, hong_nut, cat_nut, g1_state).
'bieu_dien' (self-check) says per condition whether cond_state2 on these numbers gives the grid state.

Rows of each node (scorer --rows) must hold t_cam_s (frame PTS, joins a frame across nodes), mien_chung, mien_hop
(0/1: the frame in the common / union domain at that node; on the 'cat_t1' branch the domain before t1) and lo_g2 ..
lo_g5. Interface asked of CODE 17/11 (c7_score --gio-c8b FILE --nut K): see the scorer.

  python tools/c8b_luoi.py --gio gio.json --frames frames.csv --esp32 run.csv --sender run_sender.txt --task O1-F08 \
      --hang-so hang_so.json --out-dir luoi/ [--gate-file gate_c8.json] [--scorer-arg ARG ...]
  python tools/c8b_luoi.py --gop-only --gio gio.json --out-dir luoi/ [--gate-file ...]   (combine node files again)
Last line: #ket_qua,ma=0,outcome=... / 2 refused or error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c8_u_imu as cu                                                    # noqa: E402

VERSION = 'c8b_luoi_v1'
MEASURED = ('g2', 'g3', 'g4', 'g5')
CONDS = ('g1',) + MEASURED
THRESHOLDS = {'g2': 0.10, 'g3': 0.10, 'g4': 0.10, 'g5': 7.5}     # o1_gate.THRESHOLDS (checked in tests)
STATES = ('dat', 'khong_dat', 'chua_ket_luan')
KEY_DEC = 6                       # t_cam_s rounded to 1 us to join a frame across nodes


class LuoiError(Exception):
    pass


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


# ── node results ─────────────────────────────────────────────────────────────────────────────

def node_from_files(score_path, rows_path):
    """A node result from the scorer's score.json and rows CSV:
      kind 'ok'    scored: states{c}, nums{c}, rows{t_cam key: {mien_chung, lo_c}}, gate (the gate_v3 block), cat
      kind 'hong'  the run is lost to the measurement system at this node (outcome hong_he_do)
      kind 'loi'   the files are missing or do not hold what is needed (a tool problem, not the measurement)
    ok = (kind == 'ok')."""
    try:
        with open(score_path, encoding='utf-8') as f:
            sc = json.load(f)
    except (OSError, ValueError) as exc:
        return dict(ok=False, kind='loi', why=f'score file: {exc}')
    out = dict(ok=True, kind='ok', outcome=sc.get('outcome'), fail_cause=sc.get('fail_cause', ''),
               reasons=sc.get('outcome_reasons', []), u_t_s=sc.get('U_t_s'), cat=bool(sc.get('cat_t1')))
    if out['outcome'] == 'hong_he_do':
        return dict(out, ok=False, kind='hong', why='run lost at this node: ' + '; '.join(map(str, out['reasons'])))
    gate = sc.get('gate_v3')
    if not isinstance(gate, dict):
        return dict(out, ok=False, kind='loi', why='no gate_v3 block')
    st, nums = {}, {}
    for c in CONDS:
        b = gate.get(c)
        if not isinstance(b, dict) or b.get('state') not in STATES:
            return dict(out, ok=False, kind='loi', why=f'gate {c}: no state')
        st[c] = b['state']
        if c != 'g1':
            ch = b.get('chung') or {}
            nums[c] = dict(q=_f(b.get('q')), lo=_f(b.get('lo')), hi=_f(b.get('hi')), n_undef=b.get('n_undef'),
                           q_chung=_f(ch.get('q')), lo_chung=_f(ch.get('lo')), hi_chung=_f(ch.get('hi')),
                           n_undef_chung=ch.get('n_undef'), n_frames=b.get('n_frames'))
    out.update(states=st, nums=nums, gate=gate)
    try:
        with open(rows_path, newline='', encoding='utf-8') as f:
            rr = list(csv.DictReader(f))
    except OSError as exc:
        return dict(out, ok=False, kind='loi', why=f'rows file: {exc}')
    need = ('t_cam_s', 'mien_chung') + tuple(f'lo_{c}' for c in MEASURED)
    if rr and any(k not in rr[0] for k in need):
        return dict(out, ok=False, kind='loi', why=f'rows file without {[k for k in need if k not in rr[0]]} '
                                                   f'(interface c7_score --rows for C8b)')
    rows = {}
    for r in rr:
        k = round(_f(r['t_cam_s']), KEY_DEC)
        if not math.isfinite(k) or k in rows:
            return dict(out, ok=False, kind='loi', why=f'rows file: t_cam_s {r["t_cam_s"]!r} not finite or repeated')
        rows[k] = dict(mien_chung=r['mien_chung'] == '1', **{c: _f(r[f'lo_{c}']) for c in MEASURED})
    out['rows'] = rows
    return out


# ── combination ──────────────────────────────────────────────────────────────────────────────

def cell_violation(a, b, c):
    """Largest lo of condition c over the frames in the common domain (the domain before t1 on the cat_t1 branch)
    at both nodes a, b (either node's lo: each node's U_t holds the whole step); NaN when a node is not scored or no
    such frame has a finite lo."""
    if not (a['ok'] and b['ok']):
        return math.nan
    best = math.nan
    for k, ra in a['rows'].items():
        rb = b['rows'].get(k)
        if rb is None or not (ra['mien_chung'] and rb['mien_chung']):
            continue
        vals = [v for v in (ra[c], rb[c]) if math.isfinite(v)]
        if vals:
            m = max(vals)
            best = m if not math.isfinite(best) else max(best, m)
    return best


def g1_rule(node_st):
    """G1 over the grid (C8B4R5-1, R5-2; luoi_v1 of CODE_GATE v2 section 2)."""
    if len(node_st) == 1:
        return node_st[0]
    if all(s == 'dat' for s in node_st):
        return 'dat'
    if all(s == 'khong_dat' for s in node_st):
        return 'khong_dat'
    return 'chua_ket_luan'


def combine(nodes, report_only=frozenset()):
    """Grid states of G1-G5, the run outcome and the gate numbers from the node results (list, in node order)."""
    if not nodes:
        raise LuoiError('no node')
    n = len(nodes)
    ok = [x['ok'] for x in nodes]
    hong = [x.get('kind') == 'hong' for x in nodes]
    cat = [bool(x['ok'] and x.get('cat')) for x in nodes]
    states, lo_cells, bieu = {}, {}, {}
    node_st = {c: [x['states'][c] if x['ok'] else 'chua_ket_luan' for x in nodes] for c in CONDS}
    states['g1'] = g1_rule(node_st['g1'])
    for c in MEASURED:
        st = node_st[c]
        if n == 1:
            states[c] = st[0]
        elif all(s == 'dat' for s in st):
            states[c] = 'dat'
        else:
            cells = [cell_violation(nodes[i], nodes[i + 1], c) for i in range(n - 1)]
            lo_cells[c] = cells
            states[c] = 'khong_dat' if all(math.isfinite(v) and v > THRESHOLDS[c] for v in cells) else 'chua_ket_luan'
    # numbers for the gate block (CODE_GATE v2 section 2 point 3: a node not scored adds n_undef >= 1)
    nums = {}
    for c in MEASURED:
        good = [x['nums'][c] for x in nodes if x['ok']]
        if n == 1:
            lo_c = good[0]['lo_chung'] if good else math.nan
        else:
            cells = lo_cells.get(c)
            if cells is None:                        # dat everywhere: the cell values from the rows
                cells = [cell_violation(nodes[i], nodes[i + 1], c) for i in range(n - 1)]
            lo_c = min(cells) if cells and all(math.isfinite(v) for v in cells) else math.nan

        def mx(k):
            v = [g[k] for g in good if math.isfinite(g[k])]
            return max(v) if v else math.nan

        def mxn(k):
            v = [g[k] for g in good if isinstance(g[k], int) and not isinstance(g[k], bool)]
            m = max(v) if v else None
            if not all(ok):
                m = max(m or 0, 1)
            return m
        nums[c] = dict(chung=dict(q=mx('q_chung'), lo=lo_c, hi=mx('hi_chung'), n_undef=mxn('n_undef_chung')),
                       q=mx('q'), lo=mx('lo'), hi=mx('hi'), n_undef=mxn('n_undef'), threshold=THRESHOLDS[c])
        bieu[c] = _state_from_numbers(lo_c, nums[c]['lo'], nums[c]['hi'], nums[c]['n_undef'],
                                      THRESHOLDS[c]) == states[c]
    decide = [c for c in CONDS if c not in report_only]
    ds = [states[c] for c in decide]
    fail_cause = ''
    if all(hong):
        outcome, fail_cause = 'hong_he_do', 'he_do'
    elif all(cat):
        # violation before t1 at every node (CODE_GATE v2 section 2 point 5; t1 and the frame times move together
        # with D, so "before t1" is the same frame set at every node)
        outcome, fail_cause = 'khong_dat', 'vi_pham_truoc_loi_he_do'
    elif 'khong_dat' in ds:
        outcome = 'khong_dat'
    elif 'chua_ket_luan' in ds:
        outcome = 'chua_ket_luan'
    else:
        outcome = 'dat'
    centre = n // 2
    scored = [i for i in range(n) if ok[i]]
    nut_tam = min(scored, key=lambda i: (abs(i - centre), i)) if scored else None
    return dict(states=states, outcome=outcome, fail_cause=fail_cause, decide=decide,
                report_only=sorted(report_only), nums=nums, lo_cells={c: v for c, v in lo_cells.items()},
                bieu_dien=bieu, nodes_ok=ok, n_nodes=n, nut_tam=nut_tam,
                g1_nut=node_st['g1'], hong_nut=hong, cat_nut=cat,
                loi_nut=[None if x.get('kind') != 'loi' else x.get('why') for x in nodes],
                node_states=[x.get('states') if x['ok'] else None for x in nodes],
                u_t_nodes=[x.get('u_t_s') for x in nodes],
                node_errors=[None if x['ok'] else x.get('why') for x in nodes])


def luoi_block(res, gio, gio_sha):
    """luoi_v1 (CODE_GATE v2 section 2)."""
    nd = gio['nodes_s']
    return dict(format='luoi_v1', gio_sha256=gio_sha, n_nut=len(nd), d_dau_s=nd[0], d_cuoi_s=nd[-1],
                buoc_max_s=max((b - a for a, b in zip(nd[:-1], nd[1:])), default=0.0), nut_tam=res['nut_tam'],
                g1_nut=res['g1_nut'], hong_nut=res['hong_nut'], cat_nut=res['cat_nut'], g1_state=res['states']['g1'])


def gate_block(nodes, res, gio, gio_sha):
    """The combined gate block of the run: the gate_v3 block of the node nut_tam (G1 fields, constants) with G2-G5
    replaced by the combined numbers and states, G1 state = the combined one, u_t = the largest node U_t, and the
    luoi block. None when no node is scored."""
    if res['nut_tam'] is None:
        return None
    g = json.loads(json.dumps(nodes[res['nut_tam']]['gate']))
    g['g1'] = dict(g.get('g1') or {}, state=res['states']['g1'],
                   q={'dat': 'true', 'khong_dat': 'false'}.get(res['states']['g1'], 'inconclusive'))
    for c in MEASURED:
        nm = res['nums'][c]
        hi = nm['hi']
        g[c] = dict(q=nm['q'], lo=nm['lo'], hi=hi, n_undef=nm['n_undef'],
                    n_frames=max((x['nums'][c].get('n_frames') or 0) for x in nodes if x['ok']),
                    ratio=hi / nm['threshold'] if math.isfinite(hi) else math.nan, threshold=nm['threshold'],
                    state=res['states'][c], chung=nm['chung'])
    ut = [x for x in res['u_t_nodes'] if isinstance(x, (int, float)) and math.isfinite(x)]
    g['u_t'] = max(ut) if ut else math.nan
    try:
        import o1_gate
        g.update(o1_gate.evaluate_outcomes(g, g.get('decide') or o1_gate.DECIDE_ALL))
        r = [g[c]['ratio'] for c in MEASURED if o1_gate._finite(g[c]['ratio'])]
        g['max_ratio'] = max(r) if r else math.nan
    except (ImportError, KeyError, ValueError):
        pass
    g['luoi'] = luoi_block(res, gio, gio_sha)
    g['note'] = (g.get('note') or '') + f'; c8b_luoi: {len(nodes)} nodes, node {res["nut_tam"]} for G1 fields'
    return g


def _state_from_numbers(lo_c, lo_u, hi_u, nu, thr):
    """rq3_log.cond_state2 (common lo, union lo, union hi, union n_undef), one copy here so the module stands alone
    (a test checks it against rq3_log)."""
    if math.isfinite(lo_c) and lo_c > thr:
        return 'khong_dat'
    if math.isfinite(lo_u) and lo_u > thr:
        return 'chua_ket_luan'
    if not isinstance(nu, int) or nu != 0 or not (math.isfinite(lo_u) and math.isfinite(hi_u)) or hi_u > thr:
        return 'chua_ket_luan'
    return 'dat'


def not_bounded(gio):
    """The clock file is not ok: lost (count pairs, log hole) or G1-G5 not concluded (design 3, 7)."""
    lost = gio.get('ket_luan') == 'hong_he_do'
    st = {c: 'chua_ket_luan' for c in CONDS}
    return dict(states=st, outcome='hong_he_do' if lost else 'chua_ket_luan', fail_cause='he_do' if lost else '',
                gate_v3=None, decide=list(CONDS), report_only=[],
                nums=None, lo_cells={}, bieu_dien={c: False for c in CONDS}, nodes_ok=[], n_nodes=0,
                node_states=[], node_errors=[], u_t_nodes=[], ly_do=gio.get('ly_do', []))


# ── run the scorer at each node ─────────────────────────────────────────────────────────────────

def scorer_cmd(gio_path, k, a, score, rows, task):
    cmd = [sys.executable, a.scorer or os.path.join(HERE, 'c7_score.py'), '--frames', a.frames, '--esp32', a.esp32,
           '--task', task, '--hang-so', a.hang_so, '--gio-c8b', gio_path, '--nut', str(k),
           '--out', score, '--rows', rows]
    if a.sender:
        cmd += ['--sender', a.sender]
    for x in a.scorer_arg or []:
        cmd.append(x)
    return cmd


def run_nodes(gio, gio_path, a, out_dir):
    res = []
    for k, D in enumerate(gio['nodes_s']):
        score = os.path.join(out_dir, f'nut_{k:03d}.json')
        rows = os.path.join(out_dir, f'nut_{k:03d}_rows.csv')
        p = subprocess.run(scorer_cmd(gio_path, k, a, score, rows, gio['task']), capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
        with open(os.path.join(out_dir, f'nut_{k:03d}_stdout.txt'), 'w', encoding='utf-8') as f:
            f.write(f'D_s={D!r}\nexit={p.returncode}\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}')
        last = (p.stdout.strip().splitlines() or [''])[-1]
        if p.returncode != 0 or not last.startswith('#ket_qua,ma=0'):
            res.append(dict(ok=False, why=f'scorer exit {p.returncode}: {(p.stderr.strip().splitlines() or [last])[-1]}'))
        else:
            res.append(node_from_files(score, rows))
    return res


def load_nodes(gio, out_dir):
    res = []
    for k in range(len(gio['nodes_s'])):
        res.append(node_from_files(os.path.join(out_dir, f'nut_{k:03d}.json'),
                                   os.path.join(out_dir, f'nut_{k:03d}_rows.csv')))
    return res


def report_only_of(gate_file, task):
    if not gate_file:
        return frozenset(), None
    import gate_c8                                       # tools/gate_c8.py of CODE_GATE (feat/gate-v1133)
    g = gate_c8.read_gate_c8(gate_file)
    return frozenset(g.report_only(task) if task in g.levels else ()), g.sha256


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gio', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--frames')
    ap.add_argument('--esp32')
    ap.add_argument('--sender')
    ap.add_argument('--task')
    ap.add_argument('--hang-so')
    ap.add_argument('--gate-file', default=None)
    ap.add_argument('--scorer-arg', action='append', help='extra argument passed to c7_score (repeat)')
    ap.add_argument('--gop-only', action='store_true', help='combine the node files already in --out-dir')
    ap.add_argument('--scorer', default=None, help='scorer script (default tools/c7_score.py; tests)')
    a = ap.parse_args(argv)
    try:
        for p in (a.gio, a.out_dir):
            cu.check_path(p)
        with open(a.gio, encoding='utf-8') as f:
            gio = json.load(f)
        if gio.get('format') != 'c8b_gio_v1':
            raise LuoiError(f'{a.gio}: not a c8b_gio_v1 file')
        task = a.task or gio['task']
        if task != gio['task']:
            raise LuoiError(f'--task {task} but the clock file is for {gio["task"]}')
        ro, gsha = report_only_of(a.gate_file, task)
        if not gio.get('ok'):
            res = not_bounded(gio)
        else:
            if not a.gop_only:
                if not all((a.frames, a.esp32, a.hang_so)):
                    raise LuoiError('--frames, --esp32, --hang-so are needed (or --gop-only)')
                os.makedirs(a.out_dir, exist_ok=False)
                nodes = run_nodes(gio, a.gio, a, a.out_dir)
            else:
                nodes = load_nodes(gio, a.out_dir)
            res = combine(nodes, ro)
            res['gate_v3'] = gate_block(nodes, res, gio, cu.sha256(a.gio))
        res.update(version=VERSION, task=task, nodes_s=gio.get('nodes_s', []), gio_sha256=cu.sha256(a.gio),
                   gate_file_sha256=gsha, script=cu.script_state())
        os.makedirs(a.out_dir, exist_ok=True)
        out = os.path.join(a.out_dir, 'luoi.json')
        with open(out, 'x', encoding='utf-8') as f:
            f.write(json.dumps(_clean(res), indent=1, ensure_ascii=False, allow_nan=False))
    except (LuoiError, cu.InputError, OSError, ValueError, KeyError) as exc:
        print(f'c8b_luoi: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print(' '.join(f'{c}={res["states"][c]}' for c in CONDS) + f'; {res["n_nodes"]} nut')
    print(f'#ket_qua,ma=0,outcome={res["outcome"]}')
    return 0


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


if __name__ == '__main__':
    sys.exit(main())
