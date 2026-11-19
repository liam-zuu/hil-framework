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
      kind 'khong_cham_c8'  the scorer marks the node cham_c8 false: not a C8 scoring, not taken (Codex CAM CT3-3)
    ok = (kind == 'ok')."""
    try:
        with open(score_path, encoding='utf-8') as f:
            sc = json.load(f)
    except (OSError, ValueError) as exc:
        return dict(ok=False, kind='loi', why=f'score file: {exc}')
    out = dict(ok=True, kind='ok', outcome=sc.get('outcome'), fail_cause=sc.get('fail_cause', ''),
               reasons=sc.get('outcome_reasons', []), u_t_s=sc.get('U_t_s'), cat=bool(sc.get('cat_t1')),
               nhanh_tick=(sc.get('U_t_parts') or {}).get('nhanh_tick'))
    if out['outcome'] == 'hong_he_do':
        return dict(out, ok=False, kind='hong', why='run lost at this node: ' + '; '.join(map(str, out['reasons'])))
    # Codex CAM CT3-3 (CODE 19/11): a node the scorer marks cham_c8 false (its clock not from c8b_gio_v1, constants not
    # locked for C8, ...) is not a C8 scoring: the node is not scored, its result not taken, the reason kept
    if sc.get('cham_c8') is False:
        return dict(out, ok=False, kind='khong_cham_c8',
                    why='cham_c8 false: ' + str(sc.get('cham_c8_ly_do') or 'khong ghi ly do'))
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
    for c in MEASURED:                                  # C8BS-9: the recorded state must follow the stored numbers
        nm = nums[c]
        if _state_from_numbers(nm['lo_chung'], nm['lo'], nm['hi'], nm['n_undef'], THRESHOLDS[c]) != st[c]:
            return dict(out, ok=False, kind='loi', why=f'gate {c}: state {st[c]} does not follow its numbers')
    try:
        bien = domain_key(sc, gate)
    except LuoiError as exc:
        return dict(out, ok=False, kind='loi', why=str(exc))
    out.update(states=st, nums=nums, gate=gate, bien=bien)
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


def domain_key(sc, gate):
    """The frame-dependent ends of the scoring domains at a node, on the CAMERA clock (C8BS-6; design 3: an end that
    depends on the frames themselves, as t_done, must not change between two adjacent nodes): the camera time of each
    t_done (None = not completed: the end is t_deadline, fixed in command time) and, on the cat_t1 branch, of t1.
    Command time t <-> camera time t + t0 - D."""
    g1 = gate.get('g1') or {}
    t0 = _f(sc.get('t0_log_us')) * 1e-6
    D = _f((sc.get('gio_c8b') or {}).get('D'))
    if not (math.isfinite(t0) and math.isfinite(D)):
        raise LuoiError('score file without t0_log_us or gio_c8b.D: the domain ends cannot be compared (C8BS-6)')

    def cam(x):
        if x is None:
            return None
        v = _f(x) + t0 - D
        if not math.isfinite(v):
            raise LuoiError(f'domain end {x!r} not finite (C8BS-6)')
        return round(v, KEY_DEC)
    key = [cam(x) for x in (g1.get('t_done') or [])]
    if sc.get('cat_t1'):
        key.append(('t1', cam(g1.get('t1'))))
    return tuple(key)


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


def _cat_check(x, report_only):
    """C8BS-7 (O1 3.5 OG2-3): the exception 'violation before t1' counts only for a DECIDING condition. A node whose
    exception rests only on report-only conditions is a node where the run is lost (t1 without a deciding certain
    violation before it)."""
    if not (x['ok'] and x.get('cat')) or not report_only:
        return x
    dec = [c for c in MEASURED if c not in report_only]
    hit = any(r['mien_chung'] and math.isfinite(r[c]) and r[c] > THRESHOLDS[c] for r in x['rows'].values() for c in dec)
    if hit:
        return x
    return dict(x, ok=False, kind='hong', cat=False,
                why='violation before t1 only in report-only conditions: run lost at this node (OG2-3)')


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
    nodes = [_cat_check(x, report_only) for x in nodes]
    ok = [x['ok'] for x in nodes]
    hong = [x.get('kind') == 'hong' for x in nodes]
    # C8BS-6: a frame-dependent domain end that changes between two adjacent scored nodes
    bien_doi = [bool(nodes[i]['ok'] and nodes[i + 1]['ok'] and nodes[i].get('bien') != nodes[i + 1].get('bien'))
                for i in range(n - 1)]
    cat = [bool(x['ok'] and x.get('cat')) for x in nodes]
    states, lo_cells, bieu = {}, {}, {}
    node_st = {c: [x['states'][c] if x['ok'] else 'chua_ket_luan' for x in nodes] for c in CONDS}
    states['g1'] = g1_rule(node_st['g1'])
    for c in MEASURED:
        st = node_st[c]
        if n == 1:
            states[c] = st[0]
        elif any(bien_doi):
            states[c] = 'chua_ket_luan'                  # design 3: the domain moves inside a cell
        elif all(s == 'dat' for s in st):
            states[c] = 'dat'
        else:
            cells = [math.nan if bien_doi[i] else cell_violation(nodes[i], nodes[i + 1], c) for i in range(n - 1)]
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
                cells = [math.nan if bien_doi[i] else cell_violation(nodes[i], nodes[i + 1], c)
                         for i in range(n - 1)]
            # C8BS-R2-1: a cell whose domain moves gives no finite violation, so the combined lo is not finite
            lo_c = min(cells) if cells and all(math.isfinite(v) for v in cells) else math.nan

        def mx(k):
            v = [g[k] for g in good if math.isfinite(g[k])]
            return max(v) if v else math.nan

        def mxn(k):
            v = [g[k] for g in good if isinstance(g[k], int) and not isinstance(g[k], bool)]
            m = max(v) if v else None
            if not all(ok) or any(bien_doi):          # not scored / domain moves: at least one undefined frame
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
                bieu_dien=bieu, nodes_ok=ok, n_nodes=n, nut_tam=nut_tam, bien_doi=bien_doi,
                nhanh_tick=sorted({str(x.get('nhanh_tick')) for x in nodes if x['ok']}),
                g1_nut=node_st['g1'], hong_nut=hong, cat_nut=cat,
                loi_nut=[None if x.get('kind') != 'loi' else x.get('why') for x in nodes],
                node_states=[x.get('states') if x['ok'] else None for x in nodes],
                u_t_nodes=[x.get('u_t_s') for x in nodes],
                node_errors=[None if x['ok'] else x.get('why') for x in nodes])


def _order(states):
    st = list(states)
    return 'khong_dat' if 'khong_dat' in st else ('chua_ket_luan' if 'chua_ket_luan' in st else 'dat')


def luoi_block(res, gio, gio_sha):
    """luoi_v1 (CODE_GATE v2 section 2)."""
    nd = gio['nodes_s']
    return dict(format='luoi_v1', gio_sha256=gio_sha, n_nut=len(nd), d_dau_s=nd[0], d_cuoi_s=nd[-1],
                buoc_max_s=max((b - a for a, b in zip(nd[:-1], nd[1:])), default=0.0), nut_tam=res['nut_tam'],
                g1_nut=res['g1_nut'], hong_nut=res['hong_nut'], cat_nut=res['cat_nut'], g1_state=res['states']['g1'])


def gate_block(nodes, res, gio, gio_sha, task='', gate_sha=None):
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
    # C8BS-8: the deciding set and the outcome fields of the COMBINED block (gate_v3.3 fields when a mask applies;
    # CODE_GATE v1 section 4: complete = decide is the required set)
    dec = list(res['decide'])
    g['decide'] = dec
    if res['report_only']:
        g.update(report_only=list(res['report_only']), required=dec, level=task, gate_file_sha256=gate_sha)
    g.update(outcome=_order(res['states'][c] for c in CONDS), outcome_decide=_order(res['states'][c] for c in dec),
             complete=True, any_certain_fail=any(res['states'][c] == 'khong_dat' for c in dec))
    r = [g[c]['ratio'] for c in MEASURED if math.isfinite(g[c]['ratio'])]
    g['max_ratio'] = max(r) if r else math.nan
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
        import c8b_gio
        scorer_path = a.scorer or os.path.join(HERE, 'c7_score.py')
        before = c8b_gio.repo_state(scorer_path)
        p = subprocess.run(scorer_cmd(gio_path, k, a, score, rows, gio['task']), capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
        after = c8b_gio.repo_state(scorer_path)
        # C8BS-16: the scorer state of THIS node scoring, written beside its results (read by --gop-only)
        same = before.get('commit') == after.get('commit') and before.get('dirty') == after.get('dirty')
        with open(os.path.join(out_dir, f'nut_{k:03d}_nguon.json'), 'x', encoding='utf-8') as f:
            json.dump(dict(scorer=dict(before, dirty=before.get('dirty') if same else True, changed=not same),
                           gio_sha256=cu.sha256(gio_path), D_s=D), f)
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


def node_provenance(gio, gio_sha, out_dir):
    """C8BS-16: the scorer commit and clean state of every node, as recorded when that node was scored
    (nut_KKK_nguon.json); a node without it, or scored for another clock file, is refused."""
    out = []
    for k, D in enumerate(gio['nodes_s']):
        p = os.path.join(out_dir, f'nut_{k:03d}_nguon.json')
        try:
            with open(p, encoding='utf-8') as f:
                d = json.load(f)
        except (OSError, ValueError) as exc:
            raise LuoiError(f'node {k}: no provenance of its scoring ({exc}): score the nodes again (C8BS-16)')
        if d.get('gio_sha256') != gio_sha or d.get('D_s') != D:
            raise LuoiError(f'node {k}: provenance for another clock file or node (C8BS-16)')
        out.append(d['scorer'])
    return out


LOAI_HONG = ('thao_tac', 'cua_so_tham_chieu', 'duong_lenh', 'he_do')    # J v6 section 5 order (operator first)


def loai_hong_of(score):
    """F19 (GATE 8705918, CODE 19/11): the lost-run kind one node's score.json gives, or None (scored): thao_tac
    (fail_cause 'thao_tac'), cua_so_tham_chieu (hong_cua_so_tc true), duong_lenh (fail_cause 'duong_lenh'), he_do (any
    other fail_cause 'he_do'). Read from the scorer's fields only, nothing inferred."""
    if not isinstance(score, dict):
        return None
    fc = score.get('fail_cause') or ''
    if fc == 'thao_tac' or score.get('thao_tac'):
        return 'thao_tac'
    if score.get('hong_cua_so_tc') is True:
        return 'cua_so_tham_chieu'
    if fc == 'duong_lenh':
        return 'duong_lenh'
    if fc == 'he_do' or score.get('outcome') == 'hong_he_do':
        return 'he_do'
    return None


def node_sources(gio, out_dir):
    """F19: nguon_nut [{nut, duong_dan, sha256}] of every node score.json present, and loai_hong_nut per node."""
    src, kinds = [], []
    for k in range(len(gio.get('nodes_s') or [])):
        p = os.path.join(out_dir, f'nut_{k:03d}.json')
        if not os.path.isfile(p):
            kinds.append(None)
            continue
        src.append(dict(nut=k, duong_dan=os.path.abspath(p), sha256=cu.sha256(p)))
        try:
            with open(p, encoding='utf-8') as f:
                kinds.append(loai_hong_of(json.load(f)))
        except (OSError, ValueError):
            kinds.append(None)
    return src, kinds


def gio_loai_hong(gio):
    """Lost-run kinds the clock file itself gives: thao_tac (truoc_boot.thao_tac, from the scorer's score.json, one
    source). A clock not bounded with a measurement-system loss gives he_do through the run outcome (main)."""
    out = []
    tt = ((gio.get('kiem_gio_d1') or {}).get('truoc_boot') or {}).get('thao_tac')
    if tt:
        out.append('thao_tac')
    return out


J_REPORT_ONLY = {'O1-F08': frozenset({'g5'})}   # J v6 section 6, L130 (Liam 19/11 "chi bao cao, khong lam bu")


def _file_ref(path):
    if not path:
        return None
    cu.check_path(path)
    if not os.path.isfile(path):
        raise LuoiError(f'{path}: no such file')
    return dict(duong_dan=os.path.abspath(path), sha256=cu.sha256(path))


def chain_files(a):
    """F19 (CODE 19/11): clip {ten, clip_sha256}, file_so_h, file_cham_h1, file_gio_h1 {duong_dan, sha256} of this run
    (null when not given: GATE refuses a cell chain that lacks them)."""
    clip = None
    if a.clip or a.clip_sha256:
        sha = a.clip_sha256.lower() if a.clip_sha256 else None
        if sha is not None and (len(sha) != 64 or any(ch not in '0123456789abcdef' for ch in sha)):
            raise LuoiError(f'--clip-sha256 {a.clip_sha256!r} is not a SHA-256')
        if a.clip and os.path.isfile(a.clip):
            got = cu.sha256(a.clip)
            if sha is not None and sha != got:
                raise LuoiError(f'--clip-sha256 is not the SHA-256 of {a.clip}')
            sha = got
        if sha is None:
            raise LuoiError(f'--clip {a.clip}: no such file and no --clip-sha256')
        clip = dict(ten=os.path.basename(a.clip) if a.clip else None, clip_sha256=sha)
    if bool(a.file_cham_h1) != bool(a.file_gio_h1):
        raise LuoiError('--file-cham-h1 and --file-gio-h1 go together (the H1 scoring has its own clock file)')
    return dict(clip=clip, file_so_h=_file_ref(a.file_so_h), file_cham_h1=_file_ref(a.file_cham_h1),
                file_gio_h1=_file_ref(a.file_gio_h1))


def report_only_of(gate_file, task):
    """Conditions that only report at this level: those of the J source (G5 of the figure 8 at F08, always, Codex CAM
    CT4-4: an empty or missing gate mask must not decide it again from the grid) plus the gate file's 2b mask."""
    j = J_REPORT_ONLY.get(task, frozenset())
    if not gate_file:
        return j, None
    import gate_c8                                       # tools/gate_c8.py of CODE_GATE (feat/gate-v1133)
    g = gate_c8.read_gate_c8(gate_file)
    return j | frozenset(g.report_only(task) if task in g.levels else ()), g.sha256


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
    # F19 (CODE 19/11): the cell chain GATE builds from the luoi.json of every run and thu_tu_clip
    ap.add_argument('--clip', default=None, help='video clip of this run (name and SHA-256 written as clip)')
    ap.add_argument('--clip-sha256', default=None, help='SHA-256 of the clip when the file is not given')
    ap.add_argument('--file-so-h', default=None,
                    help='drift.json (c7_calib_drift so) of the part of the session holding this run: (H dau, H giua) '
                         'before the middle H clip, (H giua, H cuoi) after')
    ap.add_argument('--file-cham-h1', default=None, help='score file of the scoring again with H1 (when done)')
    ap.add_argument('--file-gio-h1', default=None, help='clock file of the H1 scoring (H1 has its own clock file)')
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
        prov_scorer = None
        if not gio.get('ok'):
            res = not_bounded(gio)
            res.update(n_nut_cham=0, trang_thai_luot=('hong_he_do' if res['outcome'] == 'hong_he_do'
                                                       else 'chua_ket_luan_khong_nut'))
        else:
            if not a.gop_only:
                if not all((a.frames, a.esp32, a.hang_so)):
                    raise LuoiError('--frames, --esp32, --hang-so are needed (or --gop-only)')
                os.makedirs(a.out_dir, exist_ok=False)
                nodes = run_nodes(gio, a.gio, a, a.out_dir)
            else:
                nodes = load_nodes(gio, a.out_dir)
            prov = node_provenance(gio, cu.sha256(a.gio), a.out_dir)
            commits = {x.get('commit') for x in prov}
            prov_scorer = dict(commit=commits.pop() if len(commits) == 1 else '',
                               dirty=not (len({x.get('commit') for x in prov}) == 1
                                          and all(x.get('dirty') is False for x in prov)),
                               nut=prov)
            res = combine(nodes, ro)
            n_cham = sum(1 for x in res['nodes_ok'] if x)
            res['n_nut_cham'] = n_cham
            if n_cham == 0:
                # F11 (design C8B v7 section 7a): no node scored. Every node lost -> run lost (rerun); otherwise (tool
                # errors) the run is closed not concluded by GATE: no gate block, the reason of every node kept
                lost = all(res['hong_nut'])
                res.update(gate_v3=None, trang_thai_luot='hong_he_do' if lost else 'chua_ket_luan_khong_nut',
                           outcome='hong_he_do' if lost else 'chua_ket_luan', fail_cause='he_do' if lost else '',
                           ly_do_nut=[x.get('why') or x.get('fail_cause') for x in nodes])
            else:
                bad = [c for c in MEASURED if not res['bieu_dien'][c]]
                if bad:
                    raise LuoiError(f'combined numbers do not give the grid state for {bad} (C8BS-9): not written')
                res['trang_thai_luot'] = 'cham'
                res['gate_v3'] = gate_block(nodes, res, gio, cu.sha256(a.gio), task, gsha)
        import c8b_gio
        ut = [x for x in res.get('u_t_nodes') or [] if isinstance(x, (int, float)) and not isinstance(x, bool)]
        # F17 (GATE v6_C8 section 6): the largest node U_t (rule 2b takes h_s of the clock file + u_max_s)
        res['u_max_s'] = (max(ut) if ut and len(ut) == len(res.get('u_t_nodes') or [])
                          and all(math.isfinite(x) and x >= 0 for x in ut) else None)
        # F19 (GATE 8705918): the score.json of every node (path, SHA) and the lost-run kinds gathered from them (and
        # from the clock file), in the J v6 section 5 order; GATE takes the first as the run's kind
        src, kinds_nut = node_sources(gio, a.out_dir) if os.path.isdir(a.out_dir) else ([], [])
        kinds = set(gio_loai_hong(gio)) | {x for x in kinds_nut if x}
        if res.get('outcome') == 'hong_he_do' and not kinds:
            kinds.add('he_do')
        res.update(nguon_nut=src, loai_hong_nut=kinds_nut, loai_hong=[x for x in LOAI_HONG if x in kinds])
        res.update(chain_files(a))
        res.update(version=VERSION, task=task, nodes_s=gio.get('nodes_s', []), gio_sha256=cu.sha256(a.gio),
                   gate_file_sha256=gsha, script=c8b_gio.repo_state(), scorer=prov_scorer)
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
