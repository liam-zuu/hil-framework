#!/usr/bin/env python3
"""
c8b_u.py: lock the C8b numbers after C8 and write the C8 gate file (C8b design v4 sections 3, 4b, 5, 9, approved:
Liam 17/11, CODE_C8B local_cccce6cb, "Duyet (Recommended)"; CLAUDE_THESIS 03_CODE/docs/2026-11-17_Thiet_ke_C8b_v4.md;
gate file format gate_c8_v1: 03_CODE/docs/2026-11-17_Thiet_ke_CODE_GATE_v1.md section 2; O1 v11.33 3.9 rule 2b).
Written by CODE_C8B local_21b7debc on feat/c8b-u.

khoa (after the C8 clips are tracked and the C8 runs have their clock files):
  troi_ppm       drift rate bound [ppm] = LE x the largest |D_end - D_start| / distance of the third centres over the
                 two bound channels of every C8 run (design 3 table, C8B4R2-4; a convention: fit noise may hide drift);
                 name of the camera scorer's constants file (CODE 17/11)
  dai            per level: [min, max] over the C8 runs of the width U~ - L and of D_gyro - D_RV (design 3 "Kiem moi
                 luot": a campaign run outside the C8 range of its level is not concluded)
  man_trap       {alpha_r_s_px, alpha_c_s_px, nguon}: the rolling-shutter read time per row / column x LE from test 3b
                 when it was usable; null = one frame (design 3 table, 4b; the scorer's names, CODE 17/11)
  U_p  (m)       = LE x [largest per-frame error of the 9 check tags (test 1) + check-tag placement by tape + body-centre
                 residual of test 2 after the tag-body offset + offset-parameter difference origin / far place of test
                 2 + re-seating repeatability (position) + tag-height error rho |dh| / (H - h) + d_p_quay (3b)]
  U_psi (deg)    = LE x [heading residual of test 2 after the mounting angle + reference heading error of test 2 (car
                 set on the tape) + re-seating repeatability (angle) + d_psi_quay (3b)]          (design 5, C8B4-11)
                 Parts may overlap (conservative), none subtracted. Test 5 (ruler at the stop points) is an end-to-end
                 check: every difference <= U_p, else 'xem_lai' (not widened by itself).
  The camera inputs (tests 1, 2, re-seating, height) come from CODE_CAM as one JSON (fields below, interface asked
  of CODE 17/11); d_p_quay, d_psi_quay and alpha from c8b_3b.py; 3b with ga_ok false: refused (set-up redone).
cong (after every C8 figure-8 run is scored on its clock grid with the locked constants):
  O1 3.9 rule 2b: for each gate level, the runs of the C8 plan written before C8 (slot, task, controller); a rerun
  only replaces a lost run of its slot; runs lost to the measurement system or without a finite U_t are left out;
  each level needs at least one PID and one LADRC run (else C8 is not finished: refused).
  u_t_s of a run = the largest node U_t of its grid + the half width of its clock interval (design 9: so that U-hat
  keeps the clock part the grid carries; node scoring itself uses U_t without the half width).
  U_t,C8(level) = max u_t_s; U-hat: G2 = U_p, G3 = U_p + U_t,C8 v_cmd_max, G5 = U_psi + deg(U_t,C8 wz_cmd_max)
  (the time maximum of the per-frame bound is the bound at the schedule peak, o1_gate.fig8_cmd_max); report-only:
  U-hat > half the threshold (strict). G1, G4 always decide. Written in gate_c8_v1, then read back with
  tools/gate_c8.read_gate_c8 (CODE_GATE) before the command reports success; a file it refuses is renamed
  *.TU_CHOI and the command fails.

  python tools/c8b_u.py khoa --gio r1=gio_r1.json ... --tb 3b.json --cam-u c8_cam_u_vao.json \
      [--u-imu u_imu_dong.json] [--phep5 phep5.csv] --out c8b_khoa.json
  python tools/c8b_u.py cong --khoa c8b_khoa.json --ke-hoach ke_hoach_c8.csv --levels O1-F02,O1-F05,O1-F08 \
      --u-t-branch bo_tick --luot SLOT=RUN_ID,gio.json,luoi.json ... [--cam cam.json] --c8-dir 05_DATA/results/C8/<d> \
      --out gate_c8.json
Last line: #ket_qua,ma=0 written / 2 refused or error.

c8_cam_u_vao.json (CODE_CAM, proposed names; m and deg): {"phep1": {"loi_khung_max_m"}, "dat_ma_kiem_m",
  "phep2": {"du_tam_m", "du_huong_deg", "chenh_tham_so_m", "huong_chuan_deg"}, "thao_dat": {"vi_tri_m", "goc_deg"},
  "do_cao": {"rho_max_m", "dh_m", "H_m", "h_m"}, "nguon": [...]}
ke_hoach_c8.csv: slot, task, controller (PID / LADRC), written before C8.
phep5.csv: diem, chenh_m (ruler minus camera at a stop point, absolute or signed).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c8_u_imu as cu                                                    # noqa: E402

VERSION = 'c8b_khoa_v1'
LE = 1.25
CONTROLLERS = ('PID', 'LADRC')
MASKABLE = ('g2', 'g3', 'g5')
BRANCHES = ('bo_tick', 'giu_tick')


class UError(Exception):
    pass


def _num(x, what, lo=0.0):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) or float(x) < lo:
        raise UError(f'{what}: {x!r} is not a finite number >= {lo}')
    return float(x)


def _get(d, path):
    cur = d
    for k in path.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            raise UError(f'camera input: no field {path}')
        cur = cur[k]
    return cur


# ── khoa ─────────────────────────────────────────────────────────────────────────────────────

def u_p_u_psi(cam, tb):
    """U_p (m), U_psi (deg) with their parts (design 5)."""
    if not tb.get('ga_ok'):
        raise UError('test 3b: ga_ok false (set-up not within 2 mm at the slowest part): redo 3b, numbers not used')
    dpq = _num(tb.get('d_p_quay_m'), '3b d_p_quay_m')
    dsq = _num(tb.get('d_psi_quay_deg'), '3b d_psi_quay_deg')
    H, h = _num(_get(cam, 'do_cao.H_m'), 'do_cao.H_m'), _num(_get(cam, 'do_cao.h_m'), 'do_cao.h_m')
    if not H > h:
        raise UError(f'do_cao: camera height {H} m not above the tag height {h} m')
    cao = _num(_get(cam, 'do_cao.rho_max_m'), 'do_cao.rho_max_m') * _num(_get(cam, 'do_cao.dh_m'), 'do_cao.dh_m') / (H - h)
    p = dict(phep1_khung_max=_num(_get(cam, 'phep1.loi_khung_max_m'), 'phep1.loi_khung_max_m'),
             dat_ma_kiem=_num(_get(cam, 'dat_ma_kiem_m'), 'dat_ma_kiem_m'),
             phep2_du_tam=_num(_get(cam, 'phep2.du_tam_m'), 'phep2.du_tam_m'),
             phep2_chenh_tham_so=_num(_get(cam, 'phep2.chenh_tham_so_m'), 'phep2.chenh_tham_so_m'),
             thao_dat_vi_tri=_num(_get(cam, 'thao_dat.vi_tri_m'), 'thao_dat.vi_tri_m'),
             do_cao=cao, d_p_quay=dpq)
    s = dict(phep2_du_huong=_num(_get(cam, 'phep2.du_huong_deg'), 'phep2.du_huong_deg'),
             phep2_huong_chuan=_num(_get(cam, 'phep2.huong_chuan_deg'), 'phep2.huong_chuan_deg'),
             thao_dat_goc=_num(_get(cam, 'thao_dat.goc_deg'), 'thao_dat.goc_deg'),
             d_psi_quay=dsq)
    return LE * sum(p.values()), p, LE * sum(s.values()), s


def khoa(gios, tb, cam, u_imu=None, phep5=None, tb_name='', tb_sha=''):
    """gios: {run_id: c8b_gio_v1 dict} of the C8 runs."""
    if not gios:
        raise UError('no C8 clock file')
    rates, dai = [], {}
    for rid, g in gios.items():
        if g.get('format') != 'c8b_gio_v1':
            raise UError(f'{rid}: not a c8b_gio_v1 file')
        if not g.get('ok'):
            continue                                  # a run without a finite clock bound gives no range, no rate
        if g.get('rate_luot') is not None:
            rates.append(_num(g['rate_luot'], f'{rid} rate_luot'))
        d = dai.setdefault(g['task'], dict(width_s=[math.inf, -math.inf], gyro_minus_rv_s=[math.inf, -math.inf],
                                           runs=[]))
        for k in ('width_s', 'gyro_minus_rv_s'):
            v = float(g[k])
            d[k] = [min(d[k][0], v), max(d[k][1], v)]
        d['runs'].append(rid)
    if not rates:
        raise UError('no C8 run with a drift rate (every run short or not ok)')
    U_p, p_parts, U_psi, s_parts = u_p_u_psi(cam, tb)
    sh = (tb.get('B') or {}).get('shutter') or {}
    man_trap = None
    if sh.get('use_alpha'):
        man_trap = dict(alpha_r_s_px=_num(abs(sh['alpha_r_le_s_px']), 'alpha_r'),
                        alpha_c_s_px=_num(abs(sh['alpha_c_le_s_px']), 'alpha_c'),
                        nguon=dict(file=tb_name, sha256=tb_sha, le=LE,
                                   do=dict(alpha_r_s_px=sh['alpha_r_s_px'], alpha_c_s_px=sh['alpha_c_s_px'])))
    out = dict(version=VERSION, le=LE, troi_ppm=LE * max(rates) * 1e6, troi_rate_luot=sorted(rates),
               dai=dai, man_trap=man_trap, man_trap_kieu='alpha' if man_trap else 'mot_khung',
               mot_khung_s=sh.get('mot_khung_s'), U_p_m=U_p, U_p_phan_m=p_parts, U_psi_deg=U_psi,
               U_psi_phan_deg=s_parts)
    if u_imu is not None:
        if u_imu.get('version') != 'u_imu_dong_v2':
            raise UError('u_imu file: version is not u_imu_dong_v2')
        if u_imu.get('thu'):
            raise UError('u_imu file marked thu (trial tau): not lockable')
        out['U_IMU'] = {m: v.get('U_IMU_dong_rad') for m, v in (u_imu.get('muc') or {}).items()}
    if phep5 is not None:
        over = [r for r in phep5 if abs(r[1]) > U_p + 1e-12]
        out['phep5'] = dict(n=len(phep5), max_m=max((abs(r[1]) for r in phep5), default=None),
                            ket_qua='xem_lai' if over else 'dat', vuot=[r[0] for r in over])
    return out


# ── cong (rule 2b, gate file) ─────────────────────────────────────────────────────────────────────

def read_plan(path):
    cu.check_path(path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    plan = {}
    for r in rows:
        if r.get('controller') not in CONTROLLERS:
            raise UError(f'{path}: controller {r.get("controller")!r} (PID / LADRC)')
        if r['slot'] in plan:
            raise UError(f'{path}: slot {r["slot"]} twice')
        plan[r['slot']] = dict(task=r['task'], controller=r['controller'])
    if not plan:
        raise UError(f'{path}: empty plan')
    return plan


def run_u_t_parts(gio, luoi):
    """(u_t_s, parts) of one C8 run for rule 2b (O1 v11.34 3.9 2b: h + u_max), or (None, why): lost, or U_t not
    finite at every node."""
    if not gio.get('ok') or luoi.get('outcome') == 'hong_he_do':
        return None, 'lost to the measurement system'
    ut = luoi.get('u_t_nodes') or []
    if not ut or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in ut):
        return None, 'U_t not finite at every node'
    h = float(gio['half_s'])
    return max(ut) + h, dict(u_max_s=max(ut), h_s=h, n_nut=len(ut))


def run_u_t(gio, luoi):
    return run_u_t_parts(gio, luoi)[0]


def slot_runs(plan, luot):
    """luot: [(slot, run_id, gio, luoi)] in the order run. Returns {slot: (run_id, u_t_s or None)} with the rerun
    rule: a later run of a slot only after a lost one."""
    by = {}
    for slot, rid, g, l in luot:
        if slot not in plan:
            raise UError(f'run {rid}: slot {slot} not in the C8 plan (no run added after the plan, O1 3.9 2b)')
        if g.get('task') != plan[slot]['task']:
            raise UError(f'run {rid}: task {g.get("task")} but slot {slot} is {plan[slot]["task"]}')
        by.setdefault(slot, []).append((rid, g, l))
    out = {}
    for slot, lst in by.items():
        chosen = None
        for i, (rid, g, l) in enumerate(lst):
            lost = l.get('outcome') == 'hong_he_do'
            if chosen is not None:
                raise UError(f'slot {slot}: run {rid} after a run that was not lost (a rerun only replaces a lost run)')
            if not lost:
                chosen = (rid,) + run_u_t_parts(g, l)
        out[slot] = chosen if chosen is not None else (lst[-1][0], None, 'every run of the slot lost')
    return out


def uhat(u_p, u_psi_deg, u_t, v, wz):
    return dict(g2=u_p, g3=u_p + u_t * v, g5=u_psi_deg + math.degrees(u_t * wz))


def cong(k, plan, luot, levels, branch, c8_dir, cam=None, written_by=None):
    import o1_gate
    from o1_feasibility import Schedule
    from o1_tasks import FIG8
    if branch not in BRANCHES:
        raise UError(f'u_t_branch {branch!r}: {BRANCHES}')
    if not levels or len(set(levels)) != len(levels) or any(lv not in FIG8 for lv in levels):
        raise UError(f'levels {levels!r}: distinct figure-8 tasks')
    chosen = slot_runs(plan, luot)
    thr = {c: o1_gate.THRESHOLDS[c] for c in MASKABLE}
    per, parts = {}, {}
    for lv in levels:
        runs = []
        for slot, p in plan.items():
            if p['task'] != lv:
                continue
            if slot not in chosen:
                raise UError(f'slot {slot} ({lv}, {p["controller"]}) has no run')
            rid, ut, part = chosen[slot]
            parts[rid] = dict(slot=slot, task=lv, controller=p['controller'], u_t_s=ut,
                              **(part if isinstance(part, dict) else dict(bo_ra=part)))
            if ut is not None:
                runs.append(dict(run_id=rid, controller=p['controller'], u_t_s=ut))
        ctl = {r['controller'] for r in runs}
        if not set(CONTROLLERS) <= ctl:
            raise UError(f'{lv}: needs a run of each controller not lost and with a finite U_t (has {sorted(ctl)}): '
                         f'C8 not finished (O1 3.9 2b)')
        v, wz = o1_gate.fig8_cmd_max(Schedule(FIG8[lv]))
        utc = max(r['u_t_s'] for r in runs)
        uh = uhat(k['U_p_m'], k['U_psi_deg'], utc, v, wz)
        per[lv] = dict(v_cmd_max=v, wz_cmd_max=wz, u_t_c8_s=utc, runs=runs, uhat=uh,
                       report_only=[c for c in MASKABLE if uh[c] > 0.5 * thr[c]])
    return dict(format='gate_c8_v1', c8_dir=c8_dir, written_by=written_by or {},
                written_at=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                levels=list(levels), u_t_branch=branch,
                dieu_2b=dict(rule='report_only = {c in g2,g3,g5 : uhat[c] > 0.5 * threshold[c]}', thresholds=thr,
                             u_p_m=k['U_p_m'], u_psi_deg=k['U_psi_deg'], per_level=per),
                camera=dict(c8b=dict(k, u_t_luot_c8=parts), cam=cam or {}))


def write_gate(path, d):
    """Write, then read back with gate_c8.read_gate_c8 (CODE_GATE); refused: renamed *.TU_CHOI, UError."""
    try:
        import gate_c8
    except ImportError:
        raise UError('tools/gate_c8.py (CODE_GATE, feat/gate-v1133) is not on this branch: the gate file cannot be '
                     'checked, not written')
    cu.check_path(path)
    text = json.dumps(d, indent=1, ensure_ascii=False, allow_nan=False)
    with open(path, 'x', encoding='utf-8') as f:
        f.write(text)
    try:
        return gate_c8.read_gate_c8(path)
    except Exception as exc:                            # any refusal of the reader
        os.replace(path, path + '.TU_CHOI')
        raise UError(f'gate_c8.read_gate_c8 refused the file ({exc}); kept as {path}.TU_CHOI')


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────

def _json(path):
    cu.check_path(path)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('khoa')
    a1.add_argument('--gio', action='append', required=True, help='RUN_ID=gio.json (every C8 run)')
    a1.add_argument('--tb', required=True)
    a1.add_argument('--cam-u', required=True)
    a1.add_argument('--u-imu')
    a1.add_argument('--phep5')
    a1.add_argument('--out', required=True)
    a2 = sub.add_parser('cong')
    a2.add_argument('--khoa', required=True)
    a2.add_argument('--ke-hoach', required=True)
    a2.add_argument('--levels', required=True)
    a2.add_argument('--u-t-branch', required=True, choices=BRANCHES)
    a2.add_argument('--luot', action='append', required=True, help='SLOT=RUN_ID,gio.json,luoi.json (in run order)')
    a2.add_argument('--cam')
    a2.add_argument('--c8-dir', required=True)
    a2.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'khoa':
            gios, src = {}, []
            for s in a.gio:
                rid, p = s.split('=', 1)
                if rid in gios:
                    raise UError(f'run {rid} twice')
                gios[rid] = _json(p)
                src.append(p)
            phep5 = None
            if a.phep5:
                with open(a.phep5, newline='', encoding='utf-8') as f:
                    phep5 = [(r['diem'], float(r['chenh_m'])) for r in csv.DictReader(f)]
                src.append(a.phep5)
            out = khoa(gios, _json(a.tb), _json(a.cam_u), _json(a.u_imu) if a.u_imu else None, phep5,
                       os.path.basename(a.tb), cu.sha256(a.tb))
            src += [a.tb, a.cam_u] + ([a.u_imu] if a.u_imu else [])
            out['nguon'] = [dict(file=os.path.basename(p), sha256=cu.sha256(p)) for p in src]
            out['script'] = cu.script_state()
            cu.check_path(a.out)
            with open(a.out, 'x', encoding='utf-8') as f:
                f.write(json.dumps(_clean(out), indent=1, ensure_ascii=False, allow_nan=False))
            print(f'U_p {out["U_p_m"] * 1e3:.2f} mm, U_psi {out["U_psi_deg"]:.3f} deg, troi '
                  f'{out["troi_ppm"]:.0f} ppm, man trap {out["man_trap_kieu"]}')
        else:
            k = _json(a.khoa)
            if k.get('version') != VERSION:
                raise UError(f'{a.khoa}: not a {VERSION} file')
            plan = read_plan(a.ke_hoach)
            luot = []
            for s in a.luot:
                slot, rest = s.split('=', 1)
                rid, g, l = rest.split(',')
                luot.append((slot, rid, _json(g), _json(l)))
            st = cu.script_state()
            d = cong(k, plan, luot, [x.strip() for x in a.levels.split(',')], a.u_t_branch, a.c8_dir,
                     _json(a.cam) if a.cam else None,
                     dict(script='tools/c8b_u.py', repo='hil-framework', commit=st.get('commit'), dirty=st.get('dirty')))
            g = write_gate(a.out, _clean(d))
            print(f'gate file {a.out} sha256 {g.sha256}; ' + '; '.join(
                f'{lv}: report_only {d["dieu_2b"]["per_level"][lv]["report_only"]}' for lv in d['levels']))
    except (UError, cu.InputError, OSError, ValueError, KeyError) as exc:
        print(f'c8b_u: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
