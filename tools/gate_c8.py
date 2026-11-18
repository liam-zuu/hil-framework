"""
gate_c8.py: the C8 gate file (format gate_c8_v1), its reader and checker, and the set of conditions that decide
a run (O1 v11.33 3.9 rule 2b, line 321; OG2-6, OG2-7, OG2-10). CODE_GATE, 17/11/2026, design
CLAUDE_THESIS 03_CODE/docs/2026-11-17_Thiet_ke_CODE_GATE_v1.md section 2 (Liam 17/11 at CODE_GATE local_e2e02399,
"Duyệt, gửi CODE luôn (Recommended)"; K1, K2 on the floor: "Không áp, cả năm quyết (Recommended)").

One JSON file written once on the day of C8 (05_DATA/results/C8/<date>_runNN/gate_c8.json); its SHA-256 (of the
bytes) is "the gate file SHA" carried by S05_summary.csv, meta.yaml and every run record. The part read here:

  format        'gate_c8_v1'
  levels        the gate levels (figure-8 tasks, O1 3.9 rules 2, 5), each once
  u_t_branch    'bo_tick' / 'giu_tick' (O1 SC-12, OG3-4), recorded only
  dieu_2b       thresholds {g2, g3, g5} (= fwsim/o1_gate.THRESHOLDS), u_p_m, u_psi_deg (locked at C8),
                per_level {level: v_cmd_max, wz_cmd_max (schedule_peaks: the peaks of the level's schedule, the
                           refined maxima of o1_gate.fig8_cmd_max without its upper-bound margin, GC1-3),
                           runs [{run_id, controller, u_t_s}] (C8 figure-8 runs not lost to the measurement, U_t
                           finite; at least one PID and one LADRC run), u_t_c8_s (= max u_t_s),
                           uhat {g2, g3, g5}, report_only [subset of g2, g3, g5 in that order]}
  camera        other C8 numbers (CODE_C8B, CODE_CAM): not read here, covered by the SHA

Every number is recomputed: uhat = uhat_of(...) (the time maximum of the per-frame bound is the bound at the peak
of the schedule, so the peak formula) from the RECOMPUTED peaks, never the stored ones; report_only =
report_only_of(that canonical uhat) (strict '>' against half the threshold, GC1-1, GC2-1). Any difference refuses the whole file (GateFileError):
stored numbers further than REL_TOL from the recomputed ones, non-finite results (GC1-2), a duplicate JSON key at any
level (GC1-4), a number that does not convert (GC1-5). The writer (C8b script) builds each level with level_block
and calls check() (write_gate does) before the file leaves C8.
Codex Sol rounds 1, 2 (17/11, account 1, session 01a1e87f): GC1-1 ... GC1-6, GC2-1 ... GC2-4 fixed here.

required(level, platform, stage, campaign): the conditions that decide a run (design section 4):
  SIM                                   all five
  HW at a gate level                    all five minus report_only(level)
  HW at another task (K1, K2, ...)      all five (Liam 17/11, "Không áp, cả năm quyết")
  HIL, stage hil_check, campaign B-*    all five minus report_only(level) (S05 4.5; Liam 17/11 THESIS
                                        "HIL dùng cùng mặt nạ (Recommended)")
  HIL otherwise                         all five
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from typing import Dict, Iterable, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'fwsim'), os.path.join(HERE, '..', 'rpi5')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import o1_gate                                                          # noqa: E402
from o1_feasibility import Schedule                                     # noqa: E402
from o1_tasks import FIG8                                               # noqa: E402

FORMAT = 'gate_c8_v1'
MASKABLE = ('g2', 'g3', 'g5')           # O1 3.9 rule 2b: G1, G4 always decide
CONTROLLERS = ('PID', 'LADRC')
U_T_BRANCHES = ('bo_tick', 'giu_tick')
B_CAMPAIGNS = ('B-PID', 'B-LADRC')
MASK_HIL_STAGE = 'hil_check'            # S05 4.5: the HIL candidate check of procedure B
PLATFORMS = ('SIM', 'HIL', 'HW')
REL_TOL = 1e-9                          # recomputed numbers (cmd maxima, uhat): relative difference allowed
TOP_KEYS = ('format', 'levels', 'u_t_branch', 'dieu_2b')
LEVEL_KEYS = ('v_cmd_max', 'wz_cmd_max', 'u_t_c8_s', 'runs', 'uhat', 'report_only')


class GateFileError(ValueError):
    pass


def _num(x, what: str, positive: bool = False) -> float:
    """A JSON number (int / float, not bool), finite, >= 0 (> 0 when positive)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise GateFileError(f'{what} {x!r}: a finite number')
    try:
        v = float(x)
    except (OverflowError, ValueError):                 # GC1-5: an integer beyond the float range
        raise GateFileError(f'{what}: a number beyond the float range')
    if not math.isfinite(v):
        raise GateFileError(f'{what} {x!r}: a finite number')
    if v < 0.0 or (positive and v == 0.0):
        raise GateFileError(f'{what} {x!r}: {"> 0" if positive else ">= 0"}')
    return v


def _close(a: float, b: float) -> bool:
    """Equal within REL_TOL, both finite (GC1-2: inf never matches)."""
    a, b = float(a), float(b)
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= REL_TOL * max(abs(a), abs(b), 1e-300)


def schedule_peaks(level: str) -> Tuple[float, float]:
    """(v, wz) peaks [m/s, rad/s] of the figure-8 schedule of a level: the refined maxima of o1_gate.fig8_cmd_max
    WITHOUT its upper-bound margin CMD_MARGIN (GC1-3: rule 2b classifies the peak itself; the margin belongs to the
    bounds used when scoring frames). The refinement (grid, golden section) is accurate far below REL_TOL."""
    v, wz = o1_gate.fig8_cmd_max(Schedule(FIG8[level]))
    return v / (1.0 + o1_gate.CMD_MARGIN), wz / (1.0 + o1_gate.CMD_MARGIN)


def uhat_of(u_p_m: float, u_psi_deg: float, u_t_s: float, v_cmd_max: float, wz_cmd_max: float) -> Dict[str, float]:
    """Base bound of the measurement per condition at one level (O1 3.9 rule 2b): G2 U_p, G3 U_p + U_t v_cmd_max,
    G5 U_psi + U_t wz_cmd_max in degrees."""
    return {'g2': float(u_p_m), 'g3': float(u_p_m) + float(u_t_s) * float(v_cmd_max),
            'g5': float(u_psi_deg) + math.degrees(float(u_t_s) * float(wz_cmd_max))}


def report_only_of(uhat: Dict[str, float]) -> Tuple[str, ...]:
    """Conditions whose base bound is above half the threshold ('>' strict, O1 rule 2b: equal still decides;
    GC2-1: no tolerance band in the rule itself; check() classifies the canonical bound, never a stored one).
    A bound that is not finite is an error."""
    out = []
    for c in MASKABLE:
        x = float(uhat[c])
        if not math.isfinite(x):
            raise GateFileError(f'uhat {c} {x!r}: not finite')
        if x > 0.5 * o1_gate.THRESHOLDS[c]:
            out.append(c)
    return tuple(out)


def level_block(level: str, u_p_m: float, u_psi_deg: float, runs) -> dict:
    """The per_level block of a gate level as check() wants it, for the writer (C8b script): runs [{run_id,
    controller, u_t_s}]; peaks, U_t,C8, uhat and report_only computed here."""
    v, wz = schedule_peaks(level)
    runs = [dict(run_id=r['run_id'], controller=r['controller'],          # GC2-3: types checked, not converted
                 u_t_s=_num(r['u_t_s'], f'{level} runs[{i}].u_t_s', positive=True)) for i, r in enumerate(runs)]
    u_p_m = _num(u_p_m, f'{level} u_p_m')
    u_psi_deg = _num(u_psi_deg, f'{level} u_psi_deg')
    u_t = max(r['u_t_s'] for r in runs)
    uh = uhat_of(u_p_m, u_psi_deg, u_t, v, wz)
    return dict(v_cmd_max=v, wz_cmd_max=wz, u_t_c8_s=u_t, runs=runs, uhat=uh, report_only=list(report_only_of(uh)))


def required_set(report_only: Iterable[str]) -> Tuple[str, ...]:
    ro = set(report_only)
    if ro - set(MASKABLE):
        raise GateFileError(f'report_only {sorted(ro)!r}: a subset of {MASKABLE}')
    return tuple(c for c in o1_gate.CONDS if c not in ro)


def mask_applies(platform: str, stage: str, campaign: str) -> bool:
    """Whether the report-only list of the level applies to a run (module docstring)."""
    if platform not in PLATFORMS:
        raise GateFileError(f'platform {platform!r}: one of {PLATFORMS}')
    if platform == 'HW':
        return True
    return platform == 'HIL' and stage == MASK_HIL_STAGE and campaign in B_CAMPAIGNS


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def check(d) -> dict:
    """The gate file content checked field by field and recomputed (module docstring). Returns it."""
    if not isinstance(d, dict):
        raise GateFileError('not a JSON object')
    for k in TOP_KEYS:
        if k not in d:
            raise GateFileError(f'{k}: missing')
    if d['format'] != FORMAT:
        raise GateFileError(f'format {d["format"]!r}: {FORMAT!r}')
    lv = d['levels']
    if (not isinstance(lv, list) or not lv or any(not isinstance(x, str) or x not in FIG8 for x in lv)
            or len(set(lv)) != len(lv)):
        raise GateFileError(f'levels {lv!r}: a non-empty list of figure-8 tasks {sorted(FIG8)}, each once')
    if d['u_t_branch'] not in U_T_BRANCHES:
        raise GateFileError(f'u_t_branch {d["u_t_branch"]!r}: one of {U_T_BRANCHES}')
    b = d['dieu_2b']
    if not isinstance(b, dict) or any(k not in b for k in ('thresholds', 'u_p_m', 'u_psi_deg', 'per_level')):
        raise GateFileError('dieu_2b: an object with thresholds, u_p_m, u_psi_deg, per_level')
    th = b['thresholds']
    if not isinstance(th, dict) or set(th) != set(MASKABLE):
        raise GateFileError(f'dieu_2b.thresholds {th!r}: keys {MASKABLE}')
    for c in MASKABLE:
        if _num(th[c], f'dieu_2b.thresholds.{c}') != o1_gate.THRESHOLDS[c]:
            raise GateFileError(f'dieu_2b.thresholds.{c} {th[c]!r}: expected {o1_gate.THRESHOLDS[c]} (fwsim/o1_gate.py)')
    u_p = _num(b['u_p_m'], 'dieu_2b.u_p_m')
    u_psi = _num(b['u_psi_deg'], 'dieu_2b.u_psi_deg')
    pl = b['per_level']
    if not isinstance(pl, dict) or set(pl) != set(lv):
        raise GateFileError(f'dieu_2b.per_level keys {sorted(pl) if isinstance(pl, dict) else pl!r}: exactly the '
                            f'levels {lv}')
    for name in lv:
        e, w = pl[name], f'dieu_2b.per_level.{name}'
        if not isinstance(e, dict) or any(k not in e for k in LEVEL_KEYS):
            raise GateFileError(f'{w}: an object with {LEVEL_KEYS}')
        v_max, wz_max = schedule_peaks(name)
        if not _close(_num(e['v_cmd_max'], f'{w}.v_cmd_max'), v_max):
            raise GateFileError(f'{w}.v_cmd_max {e["v_cmd_max"]!r}: schedule_peaks gives {v_max!r}')
        if not _close(_num(e['wz_cmd_max'], f'{w}.wz_cmd_max'), wz_max):
            raise GateFileError(f'{w}.wz_cmd_max {e["wz_cmd_max"]!r}: schedule_peaks gives {wz_max!r}')
        runs = e['runs']
        if not isinstance(runs, list) or not runs:
            raise GateFileError(f'{w}.runs: a non-empty list')
        ids, ctrls, uts = set(), set(), []
        for i, r in enumerate(runs):
            if not isinstance(r, dict) or set(r) != {'run_id', 'controller', 'u_t_s'}:
                raise GateFileError(f'{w}.runs[{i}]: an object with run_id, controller, u_t_s')
            if not isinstance(r['run_id'], str) or not r['run_id'] or r['run_id'] in ids:
                raise GateFileError(f'{w}.runs[{i}].run_id {r["run_id"]!r}: a non-empty name, once')
            if r['controller'] not in CONTROLLERS:
                raise GateFileError(f'{w}.runs[{i}].controller {r["controller"]!r}: one of {CONTROLLERS}')
            ids.add(r['run_id'])
            ctrls.add(r['controller'])
            uts.append(_num(r['u_t_s'], f'{w}.runs[{i}].u_t_s', positive=True))
        if ctrls != set(CONTROLLERS):
            raise GateFileError(f'{w}.runs: no run of {sorted(set(CONTROLLERS) - ctrls)} (O1 3.9 rule 2b: one run '
                                'of each controller at each level, else C8 is not done)')
        u_t = _num(e['u_t_c8_s'], f'{w}.u_t_c8_s', positive=True)
        if u_t != max(uts):
            raise GateFileError(f'{w}.u_t_c8_s {u_t!r}: the largest u_t_s of the runs is {max(uts)!r}')
        uh = e['uhat']
        if not isinstance(uh, dict) or set(uh) != set(MASKABLE):
            raise GateFileError(f'{w}.uhat {uh!r}: keys {MASKABLE}')
        want = uhat_of(u_p, u_psi, u_t, v_max, wz_max)                # GC1-1: the recomputed peaks, not the file's
        for c in MASKABLE:
            if not math.isfinite(want[c]):                            # GC1-2
                raise GateFileError(f'{w}.uhat.{c}: the recomputed bound is not finite ({want[c]!r})')
            if not _close(_num(uh[c], f'{w}.uhat.{c}'), want[c]):
                raise GateFileError(f'{w}.uhat.{c} {uh[c]!r}: recomputed {want[c]!r}')
        ro = e['report_only']
        if not isinstance(ro, list) or any(not isinstance(c, str) for c in ro):
            raise GateFileError(f'{w}.report_only {ro!r}: a list of condition names')
        if tuple(ro) != report_only_of(want):                         # GC1-1: classified on the canonical bound
            raise GateFileError(f'{w}.report_only {ro!r}: from the recomputed uhat (> half the threshold) it is '
                                f'{list(report_only_of(want))!r}')
    return d


class GateC8:
    """A checked gate file (read_gate_c8)."""

    def __init__(self, d: dict, sha256: str, path: str = ''):
        self.data = check(d)
        self.sha256 = sha256
        self.path = path
        self.levels = list(d['levels'])
        self.u_t_branch = d['u_t_branch']

    def report_only(self, level: str) -> Tuple[str, ...]:
        """The report-only conditions of a gate level; () for a task that is not a gate level."""
        e = self.data['dieu_2b']['per_level'].get(level)
        return tuple(e['report_only']) if e is not None else ()

    def run_report_only(self, level: str, platform: str, stage: str, campaign: str) -> Tuple[str, ...]:
        """The report-only conditions that apply to a run (module docstring)."""
        return self.report_only(level) if mask_applies(platform, stage, campaign) else ()

    def required(self, level: str, platform: str, stage: str, campaign: str) -> Tuple[str, ...]:
        return required_set(self.run_report_only(level, platform, stage, campaign))


def _hex64(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(ch in '0123456789abcdef' for ch in x)


def _finite(x, what: str, lo: float, strict: bool = False) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or \
            (x <= lo if strict else x < lo):
        raise GateFileError(f'{what} {x!r}: a finite number {">" if strict else ">="} {lo}')
    return float(x)


def car_pose_of(cp, what: str) -> dict:
    """The car_pose configuration {tag_offset, side_m, h_m} (C8B 18/11, c8b_u check_khoa): side_m > 0, h_m >= 0 finite;
    tag_offset null or [dx_m, dy_m, dpsi_rad] three finite numbers, null read as [0, 0, 0]."""
    if not isinstance(cp, dict) or set(cp) != {'tag_offset', 'side_m', 'h_m'}:
        raise GateFileError(f'{what}: car_pose must be {{tag_offset, side_m, h_m}}, got {cp!r}')
    to = cp['tag_offset']
    if to is None:
        to = [0.0, 0.0, 0.0]
    elif not isinstance(to, list) or len(to) != 3:
        raise GateFileError(f'{what}: car_pose.tag_offset {to!r}: null or [dx_m, dy_m, dpsi_rad]')
    else:
        to = [_finite(v, f'{what}: car_pose.tag_offset', -math.inf) for v in to]
    return dict(tag_offset=to, side_m=_finite(cp['side_m'], f'{what}: car_pose.side_m', 0.0, strict=True),
                h_m=_finite(cp['h_m'], f'{what}: car_pose.h_m', 0.0))


def moc_dich_of(gate: 'GateC8') -> dict:
    """The C8 lock of the shifted-anchor bound (C8b proposal v3 sections 3-5, approved 18/11; CODE_GATE design
    03_CODE/docs/2026-11-18_Thiet_ke_rq3_u_moc_v4.md section 3.3): c8b_u cong copies the c8b_khoa_v4 lock into
    camera.c8b; read here camera.c8b.moc_dich {U_moc_xy_m, U_moc_psi_deg (before the margin), nguon {bo_lay_mau,
    car_pose, tap_moc}} and camera.c8b.khoa_sha256. GateFileError when any is missing or of the wrong type."""
    w = f'gate file {gate.path!r}: camera.c8b'
    cam = gate.data.get('camera')
    c8b = cam.get('c8b') if isinstance(cam, dict) else None
    md = c8b.get('moc_dich') if isinstance(c8b, dict) else None
    if not isinstance(md, dict):
        raise GateFileError(f'{w}.moc_dich: missing (a gate file of a C8 lock before c8b_khoa_v4)')
    ng = md.get('nguon')
    if not isinstance(ng, dict):
        raise GateFileError(f'{w}.moc_dich.nguon: missing')
    blm = ng.get('bo_lay_mau')
    if not isinstance(blm, str) or not blm:
        raise GateFileError(f'{w}.moc_dich.nguon.bo_lay_mau {blm!r}: a version string')
    if not _hex64(ng.get('tap_moc')):
        raise GateFileError(f'{w}.moc_dich.nguon.tap_moc {ng.get("tap_moc")!r}: sha256 of tap_moc.json (64 hex)')
    if not _hex64(c8b.get('khoa_sha256')):
        raise GateFileError(f'{w}.khoa_sha256 {c8b.get("khoa_sha256")!r}: sha256 of the lock file (64 hex)')
    return dict(U_moc_xy_m=_finite(md.get('U_moc_xy_m'), f'{w}.moc_dich.U_moc_xy_m', 0.0),
                U_moc_psi_deg=_finite(md.get('U_moc_psi_deg'), f'{w}.moc_dich.U_moc_psi_deg', 0.0),
                bo_lay_mau=blm, car_pose=car_pose_of(ng.get('car_pose'), f'{w}.moc_dich.nguon'),
                tap_moc=ng['tap_moc'], khoa_sha256=c8b['khoa_sha256'])


def _no_duplicate_keys(pairs):
    """GC1-4: a JSON object with a key twice is refused (json keeps the last one silently)."""
    keys = [k for k, _ in pairs]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise GateFileError(f'duplicate JSON key(s) {dup}')
    return dict(pairs)


def read_gate_c8(path: str) -> GateC8:
    """Read and check a gate file; GateFileError (a ValueError) on any problem."""
    if not os.path.isfile(path):
        raise GateFileError(f'gate file {path!r}: no such file')
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        d = json.loads(raw.decode('utf-8'), object_pairs_hook=_no_duplicate_keys)
    except GateFileError as exc:
        raise GateFileError(f'gate file {path!r}: {exc}')
    except (UnicodeDecodeError, ValueError) as exc:
        raise GateFileError(f'gate file {path!r}: not UTF-8 JSON ({exc})')
    try:
        return GateC8(d, hashlib.sha256(raw).hexdigest(), path)
    except GateFileError as exc:
        raise GateFileError(f'gate file {path!r}: {exc}')


def fake_gate(levels=('O1-F05',), u_p_m=0.002, u_psi_deg=2.0, u_t_s=0.010) -> dict:
    """A gate file content for tests and the rq3_log demo (FAKE numbers, not measured). u_t_s: one number, or a
    dict {level: U_t,C8 or (U_t PID, U_t LADRC)}; the blocks are built by level_block, as the C8b script must."""
    per = {}
    for name in levels:
        ut = u_t_s[name] if isinstance(u_t_s, dict) else u_t_s
        uts = tuple(ut) if isinstance(ut, (tuple, list)) else (ut, ut)       # (PID, LADRC)
        per[name] = level_block(name, u_p_m, u_psi_deg,
                                [dict(run_id=f'FAKE_{name}_PID', controller='PID', u_t_s=uts[0]),
                                 dict(run_id=f'FAKE_{name}_LADRC', controller='LADRC', u_t_s=uts[1])])
    return dict(format=FORMAT, c8_dir='FAKE', written_by=dict(script='FAKE', commit='', dirty=True),
                levels=list(levels), u_t_branch='bo_tick',
                dieu_2b=dict(thresholds={c: o1_gate.THRESHOLDS[c] for c in MASKABLE}, u_p_m=float(u_p_m),
                             u_psi_deg=float(u_psi_deg), per_level=per),
                camera={})


def write_gate(path: str, d: dict) -> str:
    """Write a gate content and return the SHA-256 of the bytes. GC2-2: the serialised bytes are read back in memory
    as read_gate_c8 reads them (duplicate keys refused) and checked before the file is opened, so a file the reader
    refuses is never written."""
    check(d)
    raw = (json.dumps(d, indent=1, ensure_ascii=False) + '\n').encode('utf-8')
    try:
        check(json.loads(raw.decode('utf-8'), object_pairs_hook=_no_duplicate_keys))
    except GateFileError as exc:
        raise GateFileError(f'not written: {exc}')
    with open(path, 'wb') as f:
        f.write(raw)
    return hashlib.sha256(raw).hexdigest()
