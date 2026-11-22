"""
sens_tre_slip.py (task C2): sensitivity of the fwsim responses (s03_doe.RESPONSES, the
same evaluation as S03: run_one -> evaluate_fig8 -> o1_gate) to two run parameters
that are not gains:

  io_delay_ms    pure delay on the duty, 1 ms per slot (fwsim.py 'Not modelled': the
                 SPI/UART latency the option stands for)
  slip_omega_s   traction smoothing band [rad/s] (params.py [ASSUMPTION]): it also sets
                 traction_nsub = the number of plant sub-steps per 1 ms

Combos: 3 delays x 3 omega_s, one of them the reference (io_delay_ms 0, omega_s 0.1,
the params.py default). Conditions: 3 mu x 3 speeds as S03, firmware default gains
(s03_doe.defaults read back from the DLL), seeds as S03 (SEED for the plant and the
IMU noise replay). 2 controllers x 9 combos x 9 conditions = 162 runs.

The reference combo is s03_doe.run_one called through patched_simulate with the
defaults: identical to run_one bit for bit (test_sens_tre_slip.py test_ref_equals_direct).

Outputs in --out (CSV: s03_doe.write_csv / fmt):
  bang.csv                  one row per run
  tom_tat_theo_dieu_kien.csv  per run x response: value, reference value, hieu = value - ref
  tom_tat.csv               per controller x combo x response over the 9 conditions
  meta.json

  python fwsim/sens_tre_slip.py --out ket_qua_c2 --dll ket_qua_c2/build/fwsim.dll --workers 8
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fwsim                                            # noqa: E402
import o1_feasibility as o1                             # noqa: E402
import s03_doe as s03                                   # noqa: E402

DELAYS = (0, 1, 2)                                      # io_delay_ms
OMEGAS = (0.05, 0.1, 0.2)                               # slip_omega_s [rad/s]
REF = (0, 0.1)                                          # (io_delay_ms, slip_omega_s) reference combo
CONTROLLERS = s03.CONTROLLERS
SPEEDS = s03.SPEEDS                                     # (0.2, 0.5, 0.8)
# mu_levels of S03 main: D:/CLAUDE_THESIS/05_DATA/results/S03/2026-11-10_run02/meta.yaml
# line 18 (mu_levels, = MU_FLOOR/2026-11-09_run02 min/mean/max)
MU_LEVELS = (0.413, 0.436, 0.473)
MU_SOURCE = ('05_DATA/results/S03/2026-11-10_run02/meta.yaml line 18 mu_levels '
             '(MU_FLOOR/2026-11-09_run02 min/mean/max)')
SEED = s03.SEED                                         # plant seed and IMU noise seed (s03.IMU)
RESPONSES = s03.RESPONSES                               # e_perp_max, e_perp_max_cens, tv_duty, sat_frac
THRESH_C2 = {'e_perp_max': 0.01, 'e_perp_max_cens': 0.01, 'sat_frac': 0.01, 'tv_duty': 0.4}
TEN_C2 = {'e_perp_max': 'Y1', 'e_perp_max_cens': 'Y1 (cens)', 'sat_frac': 'Y2', 'tv_duty': 'Y3'}
TEN_S03 = {'e_perp_max': 'Y2', 'e_perp_max_cens': 'Y2 (cens)', 'tv_duty': 'Y3', 'sat_frac': 'Y4'}
DON_VI = {'e_perp_max': 'm', 'e_perp_max_cens': 'm', 'tv_duty': '1/s',
          'sat_frac': 'phan so (tick x banh)'}

BANG_COLS = ['controller', 'io_delay_ms', 'slip_omega_s', 'la_moc', 'mu', 'v',
             'seed_plant', 'seed_imu_noise', 'n_seed', 'slip_nsub', 'status', 'completed',
             'e_perp_max', 'e_perp_max_cens', 'tv_duty', 'sat_frac', 'sat_any_frac',
             'e_perp_global_max', 'e_perp_rms', 'pos_reset_mid', 'gate_outcome', 'error', 't_chay_s']
DK_COLS = ['controller', 'io_delay_ms', 'slip_omega_s', 'la_moc', 'mu', 'v', 'dap_ung',
           'ten_c2', 'ten_s03', 'don_vi', 'gia_tri_moc', 'gia_tri', 'hieu', 'hieu_tuong_doi',
           'nguong', 'vuot_nguong', 'completed_moc', 'completed', 'status_moc', 'status']
TT_COLS = ['controller', 'io_delay_ms', 'slip_omega_s', 'la_moc', 'dap_ung', 'ten_c2', 'ten_s03',
           'don_vi', 'nguong', 'n_dieu_kien', 'hieu_tb', 'hieu_max', 'dk_max',
           'hieu_tuong_doi_tai_max', 'max_abs_hieu_tuong_doi', 'vuot_nguong', 'n_vuot', 'dk_vuot',
           'n_completed_doi', 'n_status_doi']
SHA_FILES = ['fwsim/s03_doe.py', 'fwsim/o1_feasibility.py', 'fwsim/o1_gate.py', 'fwsim/o1_gate_eval.py',
             'fwsim/fwsim.py', 'rpi5/plant_step.py', 'rpi5/params.py', 'fwsim/sens_tre_slip.py']


@contextlib.contextmanager
def patched_simulate(delay_ms, omega_s, rec=None):
    """fwsim.simulate with io_delay_ms = delay_ms and params.slip_omega_s = omega_s (in-memory
    attribute of that run's params only; params.py untouched). rec: records the values that
    reach the call. s03_doe.run_one calls fwsim.simulate via the module attribute."""
    orig = fwsim.simulate

    def sim(fw_, setpoint, duration, params=None, *a, **kw):
        assert not a and 'io_delay_ms' not in kw
        params.slip_omega_s = omega_s                      # in-memory attribute only
        if rec is not None:
            rec.append(dict(slip_omega_s=params.slip_omega_s, slip_nsub=fwsim.traction_nsub(params),
                            io_delay_ms=delay_ms, mu_hd=float(params.mu_hd[0])))
        return orig(fw_, setpoint, duration, params, io_delay_ms=delay_ms, **kw)

    fwsim.simulate = sim
    try:
        yield
    finally:
        fwsim.simulate = orig


def run_case(fw, path, ctrl, v, mu, delay, omega_s):
    """One s03_doe.safe_run (run_one inside: evaluate_fig8 + gate) under patched_simulate.
    Returns (row, log); the row gains slip_omega_s / slip_nsub as used."""
    rec = []
    with patched_simulate(delay, omega_s, rec):
        # defaults() calls fw.reset(); argument evaluation puts it before simulate() - fine
        row, log = s03.safe_run(fw, path, ctrl, s03.defaults(fw, ctrl), v, mu)
    assert len(rec) == 1 or (not rec and row.get('status') == 'run_error'), \
        f'{len(rec)} simulate calls'
    if rec:
        row['slip_omega_s'] = rec[0]['slip_omega_s']
        row['slip_nsub'] = rec[0]['slip_nsub']
    else:                                   # run_error before simulate
        row['slip_omega_s'] = omega_s
        row['slip_nsub'] = ''
    return row, log


# ── pool ─────────────────────────────────────────────────────────────────────────────

_FW = None
_PATH = None


def _init(dll):
    global _FW, _PATH
    _FW = fwsim.FirmwareSim(dll)
    _PATH = o1.Path8()


def _job(job):
    ctrl, delay, omega_s, mu, v = job
    t0 = time.time()
    row, _ = run_case(_FW, _PATH, ctrl, v, mu, delay, omega_s)
    row.update(controller=ctrl, io_delay_ms=delay, mu=mu, v=v,
               la_moc=(delay == REF[0] and omega_s == REF[1]),
               seed_plant=SEED, seed_imu_noise=s03.IMU['noise_seed'], n_seed=1,
               t_chay_s=time.time() - t0)
    return row


# ── summaries ───────────────────────────────────────────────────────────────────────

def _hieu(g, gm):
    """value - reference value; None when either side is missing."""
    try:
        return float(g) - float(gm)
    except (TypeError, ValueError):
        return None


def _rel(hieu, gm):
    """hieu / |ref|; None when ref == 0 or non-finite (or hieu missing)."""
    try:
        gm = float(gm)
    except (TypeError, ValueError):
        return None
    if hieu is None or gm == 0.0 or not math.isfinite(gm):
        return None
    return hieu / abs(gm)


def _dk(mu, v):
    return f'mu={mu};v={v}'


def per_condition_rows(rows, refs):
    """tom_tat_theo_dieu_kien.csv: one row per run x response."""
    out = []
    for r in rows:
        rm = refs[(r['controller'], r['mu'], r['v'])]
        for resp in RESPONSES:
            g, gm = r.get(resp, ''), rm.get(resp, '')
            h = _hieu(g, gm)
            rel = _rel(h, gm)
            vuot = '' if h is None or not math.isfinite(h) else abs(h) > THRESH_C2[resp]
            out.append(dict(controller=r['controller'], io_delay_ms=r['io_delay_ms'],
                            slip_omega_s=r['slip_omega_s'], la_moc=r['la_moc'], mu=r['mu'], v=r['v'],
                            dap_ung=resp, ten_c2=TEN_C2[resp], ten_s03=TEN_S03[resp],
                            don_vi=DON_VI[resp], gia_tri_moc=gm, gia_tri=g,
                            hieu='' if h is None else h, hieu_tuong_doi='' if rel is None else rel,
                            nguong=THRESH_C2[resp], vuot_nguong=vuot,
                            completed_moc=rm.get('completed', ''), completed=r.get('completed', ''),
                            status_moc=rm.get('status', ''), status=r.get('status', '')))
    return out


def summary_rows(rows, refs):
    """tom_tat.csv: one row per controller x combo (incl. the reference) x response."""
    by_key = {(r['controller'], r['io_delay_ms'], r['slip_omega_s'], r['mu'], r['v']): r
              for r in rows}
    out = []
    for ctrl in CONTROLLERS:
        for delay in DELAYS:
            for w in OMEGAS:
                rs = [by_key[(ctrl, delay, w, mu, v)] for mu in MU_LEVELS for v in SPEEDS]
                ms = [refs[(ctrl, mu, v)] for mu in MU_LEVELS for v in SPEEDS]
                for resp in RESPONSES:
                    per = [(r['mu'], r['v'], _hieu(r.get(resp, ''), rm.get(resp, '')),
                            _rel(_hieu(r.get(resp, ''), rm.get(resp, '')), rm.get(resp, '')))
                           for r, rm in zip(rs, ms)]
                    hs = [(mu, v, h) for mu, v, h, _ in per if h is not None and math.isfinite(h)]
                    rels = [x for *_, x in per if x is not None and math.isfinite(x)]
                    vuot = [(mu, v) for mu, v, h, _ in per
                            if h is not None and math.isfinite(h) and abs(h) > THRESH_C2[resp]]
                    imax = max(hs, key=lambda t: abs(t[2])) if hs else None
                    out.append(dict(
                        controller=ctrl, io_delay_ms=delay, slip_omega_s=w,
                        la_moc=(delay == REF[0] and w == REF[1]), dap_ung=resp,
                        ten_c2=TEN_C2[resp], ten_s03=TEN_S03[resp], don_vi=DON_VI[resp],
                        nguong=THRESH_C2[resp], n_dieu_kien=len(hs),
                        hieu_tb='' if not hs else sum(h for _, _, h in hs) / len(hs),
                        hieu_max='' if imax is None else imax[2],
                        dk_max='' if imax is None else _dk(imax[0], imax[1]),
                        hieu_tuong_doi_tai_max='' if imax is None else next(
                            ('' if x is None else x) for mu, v, _, x in per
                            if (mu, v) == (imax[0], imax[1])),
                        max_abs_hieu_tuong_doi='' if not rels else max(abs(x) for x in rels),
                        vuot_nguong='' if not hs else bool(vuot),
                        n_vuot=len(vuot), dk_vuot=' '.join(_dk(mu, v) for mu, v in vuot),
                        n_completed_doi=sum(r.get('completed', '') != rm.get('completed', '')
                                            for r, rm in zip(rs, ms)),
                        n_status_doi=sum(r.get('status', '') != rm.get('status', '')
                                         for r, rm in zip(rs, ms))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='ket_qua_c2')
    ap.add_argument('--dll', default=os.path.join('ket_qua_c2', 'build', 'fwsim.dll'))
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args(argv)
    bang = os.path.join(args.out, 'bang.csv')
    if os.path.exists(bang):
        sys.exit(f'{bang} exists: refusing to overwrite')
    os.makedirs(args.out, exist_ok=True)
    t0, t_iso = time.time(), time.strftime('%Y-%m-%dT%H:%M:%S')

    fw0 = fwsim.FirmwareSim(args.dll)
    if not fw0.pos_cols:
        sys.exit('fwsim.dll has no pos_ctrl.c: C2 needs the campaign firmware')
    gains = {c: s03.defaults(fw0, c) for c in CONTROLLERS}

    jobs = [(c, d, w, mu, v) for c in CONTROLLERS for d in DELAYS for w in OMEGAS
            for mu in MU_LEVELS for v in SPEEDS]
    rows = []
    n_workers = min(args.workers, 8)
    with mp.Pool(processes=n_workers, initializer=_init, initargs=(args.dll,)) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.append(row)
            print(f"[{i}/{len(jobs)}] {row['controller']} delay={row['io_delay_ms']} "
                  f"omega_s={row['slip_omega_s']} mu={row['mu']} v={row['v']} "
                  f"status={row['status']} t={row['t_chay_s']:.1f}s", flush=True)
    rows.sort(key=lambda r: (CONTROLLERS.index(r['controller']), r['io_delay_ms'],
                             r['slip_omega_s'], r['mu'], r['v']))
    refs = {(r['controller'], r['mu'], r['v']): r for r in rows if r['la_moc']}
    assert len(refs) == len(CONTROLLERS) * len(MU_LEVELS) * len(SPEEDS), 'missing reference runs'

    s03.write_csv(bang, rows, BANG_COLS)
    s03.write_csv(os.path.join(args.out, 'tom_tat_theo_dieu_kien.csv'),
                  per_condition_rows(rows, refs), DK_COLS)
    s03.write_csv(os.path.join(args.out, 'tom_tat.csv'), summary_rows(rows, refs), TT_COLS)

    root = os.path.join(HERE, '..')
    build_json = os.path.join(os.path.dirname(os.path.abspath(args.dll)), 'fwsim_build.json')
    with open(build_json, encoding='utf-8') as f:
        build = json.load(f)
    meta = dict(
        command=' '.join(sys.argv), started=t_iso, ended=time.strftime('%Y-%m-%dT%H:%M:%S'),
        elapsed_s=time.time() - t0, n_runs=len(rows), n_workers=n_workers,
        cpu_count=os.cpu_count(),
        dll=dict(path=os.path.abspath(args.dll), sha256=s03.sha256_file(args.dll)),
        fwsim_build_json=build_json, fwsim_build=build,
        esp32_dirty=fw0.build_info.get('esp32', {}).get('dirty'),
        gains=gains,
        seeds=dict(plant=SEED, imu_noise=s03.IMU['noise_seed'],
                   note='no other random source'),
        imu=dict(s03.IMU),
        mu_levels=list(MU_LEVELS), mu_source=MU_SOURCE,
        speeds=list(SPEEDS), delays=list(DELAYS), omegas=list(OMEGAS), ref=list(REF),
        thresholds=THRESH_C2, ten_c2=TEN_C2, ten_s03=TEN_S03, don_vi=DON_VI,
        plant_src=fwsim.PLANT_SRC, libraries=fwsim.lib_versions(),
        sha256={p: s03.sha256_file(os.path.join(root, p)) for p in SHA_FILES},
        counts=dict(status=dict(Counter(str(r.get('status', '')) for r in rows)),
                    completed=dict(Counter(str(r.get('completed', '')) for r in rows))),
    )
    with open(os.path.join(args.out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    print(f'done: {len(rows)} runs in {time.time() - t0:.0f} s -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
