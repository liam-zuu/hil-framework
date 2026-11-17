"""GT1-9 (Codex round 1 of CODE_GATE; Liam 17/11 at CODE_GATE "Sửa theo bảng (Recommended)"): the cases whose
gate_v3.2 dicts (fwsim/o1_gate.gate called as before O1 v11.33: no v_k, w_k, report_only) and S03 run fields
(fwsim/s03_doe.gate_fields) must stay byte for byte those of the base 1b41cca. The reference digests
(data/gate_v32_golden_1b41cca.json) were made by running digests() with the modules of 1b41cca (git archive of that
commit; the command is in the reference file); test_gate_v32_golden.py runs it with the modules of the branch.

This module imports nothing of the project: the caller passes the modules (o1_gate G, o1_feasibility O, o1_tasks OT,
s03_doe s3), so the same cases run on either tree. Every number is invented (frames made up from a seed)."""
import hashlib
import json
import struct

import numpy as np

TASKS = ('O1-F02', 'O1-F05', 'O1-F08', 'O1-K1', 'O1-K2')
# frame errors as multiples of the threshold of each condition: well below (dat), straddling it within the bound
# (chua_ket_luan once U_t > 0), far above (khong_dat), and with undefined frames (NaN: n_undef)
VARIANTS = {'below': (0.2, 0.6, None), 'edge': (0.95, 1.04, None), 'above': (1.5, 3.0, None),
            'nan': (0.3, 0.9, 0.2)}
COMPLETED = ('true', 'false', 'inconclusive')
U_TS = (None, 0.0, 0.014)


def enc(x):
    """A value as text that keeps its type and every bit (floats by their IEEE-754 bytes, NaN payload too)."""
    if isinstance(x, dict):
        return {str(k): enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    if isinstance(x, np.ndarray):
        return ['a' + str(x.dtype)] + [enc(v) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return f'b{bool(x)}:{type(x).__name__}'
    if isinstance(x, (int, np.integer)):
        return f'i{int(x)}:{type(x).__name__}'
    if isinstance(x, (float, np.floating)):
        return 'f' + struct.pack('<d', float(x)).hex() + ':' + type(x).__name__
    if x is None:
        return 'n'
    if isinstance(x, str):
        return 's' + x
    return 'r' + type(x).__name__ + ':' + repr(x)


def text(x) -> str:
    """The canonical text of a value (sorted keys, no spaces), whose sha256 is the digest of a case."""
    return json.dumps(enc(x), sort_keys=True, separators=(',', ':'))


def frames(G, variant: str, seed: int, n: int = 25) -> dict:
    lo, hi, nan_share = VARIANTS[variant]
    rng = np.random.default_rng(seed)
    out = {}
    for c in G.MEASURED:
        e = G.THRESHOLDS[c] * rng.uniform(lo, hi, n)
        if nan_share:
            e[rng.random(n) < nan_share] = np.nan
        out[G.FRAME_KEYS[c]] = e
    return out


def g1_fields(completed: str) -> dict:
    """B81-3 G1 fields of one way that agree with `completed` (o1_gate.check_g1_fields)."""
    if completed == 'true':
        return dict(t_done=[5.0], t_deadline=20.0, t1=None, prog_before_t1=None, cond25_before_deadline=None)
    if completed == 'false':
        return dict(t_done=[None], t_deadline=20.0, t1=None, prog_before_t1=None, cond25_before_deadline=None)
    return dict(t_done=[None], t_deadline=20.0, t1=7.0, prog_before_t1=False, cond25_before_deadline=True)


def cmd_max(G, O, OT, task: str) -> tuple:
    if task in OT.FIG8:
        return G.fig8_cmd_max(O.Schedule(OT.FIG8[task]))
    return float(OT.K_V), 0.0


def outputs(G, O, OT, s3) -> dict:
    """name -> the gate dict (or S03 run fields) of every case, computed with the given modules."""
    out = {}
    seed = 0
    for task in TASKS:
        v, wz = cmd_max(G, O, OT, task)
        out[f'{task}/cmd_max'] = (v, wz)
        for variant in VARIANTS:
            seed += 1
            fr = frames(G, variant, seed)
            for i, u_t in enumerate(U_TS):
                for decide in (G.DECIDE_ALL, G.DECIDE_HIL):
                    completed = COMPLETED[(seed + i + len(decide)) % 3]
                    name = f'{task}/{variant}/u{u_t}/{"all" if decide == G.DECIDE_ALL else "hil"}/{completed}'
                    out[name] = G.gate(fr, completed, v_cmd_max=v, wz_cmd_max=wz, u_t=u_t, decide=decide,
                                       note='FAKE')
            for completed in COMPLETED:                        # the fwsim / S03 path: G1 fields, U_t = 0
                gt = G.gate(fr, completed, v_cmd_max=v, wz_cmd_max=wz, u_t=s3.GATE_U_T, decide=G.DECIDE_ALL,
                            g1_fields=g1_fields(completed))
                out[f'{task}/{variant}/s03/{completed}/gate'] = gt
                res = dict(e_bao_max=float(np.nanmax(fr['e_bao'])) if np.any(np.isfinite(fr['e_bao']))
                           else float('nan'))
                out[f'{task}/{variant}/s03/{completed}/row'] = s3.gate_fields(res, gt)
                out[f'{task}/{variant}/s03/{completed}/row_text'] = s3.row_text(s3.gate_fields(res, gt))
    out['cols'] = (list(s3.GATE_RUN_COLS), list(s3.G1_RUN_COLS))
    return out


def digests(G, O, OT, s3) -> dict:
    """name -> sha256 of the canonical text of every case."""
    return {k: hashlib.sha256(text(v).encode('utf-8')).hexdigest() for k, v in outputs(G, O, OT, s3).items()}
