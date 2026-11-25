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
  chay_lai.csv              both attempts of every run_error rerun (attempt 1, 2)
  meta.json                 incl. command_full, the DLL path/sha256/fwsim_build and warnings

Run (default --dll = fwsim/build/fwsim.dll, the output of python fwsim/build_fwsim.py;
meta.json records command_full with every option and absolute paths, the DLL sha256, its
fwsim_build.json, and a WARNING when that file is missing or the build is dirty):
  python fwsim/build_fwsim.py
  python fwsim/sens_tre_slip.py --out <new folder> --workers 8 [--ctrl ADRC] [--dll <dll>]
--out must be a new folder (an existing bang.csv is refused).

Rerun of SENS_TRE_SLIP/2026-11-22_run01 (its meta.yaml line 18 'lenh' has no --dll: at hil
53f0e2f the default was ket_qua_c2/build/fwsim.dll relative to the cwd = repo root):
  python fwsim/sens_tre_slip.py --out <new folder> --dll ket_qua_c2/build/fwsim.dll --workers 8
  DLL sha256 4675ff9e79933703567d6c902a1a7610d1867798c595e76fce5544551683bae7,
  AGV_MECANUM 5bde4c5 dirty false, gcc MinGW-W64 13.2.0 -std=gnu11 -O2 -ffp-contract=off
  (run01 meta.yaml lines 14, 16; ket_qua_c2/build/fwsim_build.json). If that DLL is gone:
  python fwsim/build_fwsim.py --fw <AGV_FIRMWARE clean at 5bde4c5> --out <folder>, then
  --dll <folder>/fwsim.dll, and use the numbers only if meta.json dll.sha256 equals the one
  above (a byte-identical rebuild is not checked).

Confirmation run when an input changes (S03 v8.6 section 8 condition 12): if the S03 main
run differs from run01 in rpi5/plant_step.py, rpi5/params.py, rpi5/imu_model.py, the DLL
(beyond the limits table, L142), the mu levels, the speed set, the replayed IMU noise data,
the scoring (s03_doe.py, O1 gate), fwsim/fwsim.py or this file: run the same grid on the new
inputs before S03 main:
  python fwsim/build_fwsim.py
  python fwsim/sens_tre_slip.py --out <new folder> --workers 8
New mu levels / speeds are an edit of MU_LEVELS, MU_SOURCE (SPEEDS = s03_doe.SPEEDS) first.
Thresholds are the S03 section 5.2 ones locked at run time, read from s03_doe.THRESH (the
same source as S03; thresholds(), recorded in meta.json thresholds; C3-21): no tv_duty entry
there = refused. Condition 12 rules (C3-19): a run_error is run once more in a new pool,
both attempts go to chay_lai.csv, a success after an error replaces the row in bang.csv and
is listed in meta.json loi_chua_ro_nguyen_nhan; every grid key, reference run and pair is
checked (completeness(), meta.json kiem_cap): still failing, missing or duplicated =
ket_luan 'chưa kết luận' (report to Liam). A pair counts only when both runs hold Y1
(completed), the gate outcome and finite Y2 (e_perp_max), Y3 (tv_duty) (C3-29; meta.json
kiem_cap cap_khong_so_duoc, khoa_thieu_dai_luong). An exception raised by a worker or the
pool iterator does not stop the script: its key goes to meta.json loi_worker. A worker
exception keeps its key and is run once more in a new pool like a run_error (C3-19 round 3:
both attempts in chay_lai.csv, status 'loi_worker' for an attempt that raised); still failing =
the key is missing, 'chưa kết luận'. The wait for each worker result is bounded
(HAN_CHO_WORKER_S): a worker dying in native code cannot hang the script; past the bound the
keys not returned go to loi_worker (ly_do 'het_han_cho_worker'), the pool is terminated,
'chưa kết luận'. The build origin is verified only when fwsim_build.json dll_sha256
equals the DLL sha256 and the json holds the ESP32 source (esp32.commit a 7-40 character hex
commit, of the --fw AGV repo when that folder is readable, and esp32.dirty) (C3-20,
meta.json dll.sidecar_khop_sha, dll.nguon_esp32_du, nguon_build); an unverified build or an
inconclusive grid is not a confirmation run before S03 main (meta.json
dung_lam_luot_xac_nhan). This script does not compare with run01.

LADRC run after S06 (condition 12 (ii)): when the measured HIL loop delay is >= 2 ms, the
same grid for LADRC only (3 delays x 3 omega_s x 9 conditions = 81 runs):
  python fwsim/sens_tre_slip.py --out <new folder> --workers 8 --ctrl ADRC
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
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
DEFAULT_DLL = os.path.join(HERE, 'build', 'fwsim.dll')  # output of fwsim/build_fwsim.py
DEFAULT_FW = os.path.join(HERE, '..', '..', 'AGV_FIRMWARE')  # the build_fwsim.py --fw default
# C3-19 round 3: longest wait [s] for the next worker result. Basis: the longest run of
# 05_DATA/results/SENS_TRE_SLIP/2026-11-22_run01/bang.csv is t_chay_s 27.25 s (PID,
# io_delay_ms 0, slip_omega_s 0.05, mu 0.473, v 0.2; 162 runs, 8 workers); 600 s is about
# 22 x that run (convention: a 20x margin, rounded up). imap_unordered returns results as they
# finish, so with every worker alive the gap between two results is at most about one run.
HAN_CHO_WORKER_S = 600.0
HET_HAN = 'het_han_cho_worker'
LOI_STATUS = ('run_error', 'loi_worker')               # attempts that are rerun once (C3-19)
COMMIT_RE = re.compile(r'[0-9a-fA-F]{7,40}')            # esp32.commit: short or full git SHA-1
GIT_TIMEOUT_S = 60                                      # git cat-file bound (convention)
THRESH_SOURCE = 'fwsim/s03_doe.py THRESH (S03 section 5.2, read at run time)'
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


def _job_safe(job):
    """_job, or {'_loi_worker': reason, '_khoa': job} when it raises (C3-19): the exception
    stays tied to its grid key instead of leaving the pool iterator."""
    try:
        return _job(job)
    except Exception as exc:                 # noqa: BLE001 (recorded in meta.json loi_worker)
        return dict(_loi_worker=f'{type(exc).__name__}: {exc}', _khoa=tuple(job))


def _loi_row(k, error, het_han=False):
    """Stand-in row of a key whose attempt gave no row (worker exception or no result within
    HAN_CHO_WORKER_S): status 'loi_worker', so it is rerun (exception) and written as an
    attempt of chay_lai.csv; main drops it from the final rows (the key is then missing)."""
    ctrl, delay, omega_s, mu, v = k
    return dict(controller=ctrl, io_delay_ms=delay, slip_omega_s=omega_s,
                la_moc=(delay == REF[0] and omega_s == REF[1]), mu=mu, v=v,
                status='loi_worker', completed='', error=error, slip_nsub='', t_chay_s='',
                _khong_ket_qua=True, _het_han=het_han)


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


def thresholds():
    """S03 section 5.2 thresholds locked at run time, read from s03_doe.THRESH (the same source
    as S03, C3-21), one per response. SystemExit when s03_doe holds none for a response (e.g.
    no absolute tv_duty threshold of S03 v8.6): no private copy is used instead."""
    missing = [r for r in RESPONSES if r not in s03.THRESH]
    if missing:
        sys.exit(f's03_doe.THRESH has no threshold for {", ".join(missing)} (S03 section 5.2): '
                 'the locked S03 thresholds are not in this tree; refusing')
    return {r: float(s03.THRESH[r]) for r in RESPONSES}


def key_of(r):
    """Grid key of a row or job: (controller, io_delay_ms, slip_omega_s, mu, v)."""
    return (r['controller'], r['io_delay_ms'], r['slip_omega_s'], r['mu'], r['v'])


def _ks(k):
    return f'{k[0]};io_delay_ms={k[1]};slip_omega_s={k[2]};mu={k[3]};v={k[4]}'


Y1_VALUES = ('true', 'false', 'inconclusive')          # o1_feasibility evaluate_v7 'completed'
GATE_VALUES = ('dat', 'khong_dat', 'chua_ket_luan')     # o1_gate.run_outcome
Y_FINITE = ('e_perp_max', 'tv_duty')                    # S03 Y2, Y3: the differences compared


def missing_quantities(r):
    """Fields a condition 12 comparison needs that row r lacks (C3-29): Y1 ('completed' one
    of Y1_VALUES), the gate outcome (one of GATE_VALUES), finite Y2 and Y3; [] = comparable."""
    bad = [f for f, vals in (('completed', Y1_VALUES), ('gate_outcome', GATE_VALUES))
           if r.get(f) not in vals]
    for f in Y_FINITE:
        try:
            x = float(r.get(f))
        except (TypeError, ValueError):
            x = math.nan
        if not math.isfinite(x):
            bad.append(f)
    return bad


def completeness(rows, ctrls, loi_worker=()):
    """S03 section 8 condition 12 check of the final rows (C3-19): every key of make_jobs(ctrls)
    exactly once, no key outside the grid, no run_error left, every reference run and every pair
    (comparison run + reference run of the same controller, mu, v) present without run_error.
    C3-29: a present pair counts only when both runs hold Y1, the gate outcome and finite Y2, Y3
    (missing_quantities; else cap_khong_so_duoc, the runs in khoa_thieu_dai_luong). loi_worker:
    the pool errors left unresolved (main: iterator errors, het_han_cho_worker, exceptions whose
    key has no successful row; an exception rerun with success is not passed). Any failure:
    ket_luan 'chưa kết luận'."""
    want = [tuple(j) for j in make_jobs(ctrls)]
    wset = set(want)
    cnt = Counter(key_of(r) for r in rows)
    got = {key_of(r): r for r in rows}

    def ok(k):
        return cnt[k] == 1 and got[k].get('status') != 'run_error'

    def moc(k):
        return (k[0], REF[0], REF[1], k[3], k[4])

    ref = [k for k in want if (k[1], k[2]) == REF]
    so = [k for k in want if (k[1], k[2]) != REF]
    thieu = [k for k in want if cnt[k] == 0]
    trung = [k for k in want if cnt[k] > 1]
    la = [k for k in cnt if k not in wset]
    loi = [k for k in want if cnt[k] and got[k].get('status') == 'run_error']
    moc_thieu = [k for k in ref if not ok(k)]
    cap_thieu = [k for k in so if not (ok(k) and ok(moc(k)))]
    dl = {k: missing_quantities(got[k]) for k in want if ok(k)}
    dl = {k: f for k, f in dl.items() if f}
    cap_ks = [k for k in so if k not in cap_thieu and (k in dl or moc(k) in dl)]
    bad = thieu or trung or la or loi or moc_thieu or cap_thieu or cap_ks or loi_worker
    return dict(ket_luan='chưa kết luận' if bad else 'đủ cặp',
                n_khoa_can=len(want), n_khoa_co=len(want) - len(thieu),
                n_moc_can=len(ref), n_moc_thieu=len(moc_thieu),
                n_cap_can=len(so), n_cap_du=len(so) - len(cap_thieu) - len(cap_ks),
                n_cap_thieu=len(cap_thieu), n_cap_khong_so_duoc=len(cap_ks),
                khoa_thieu=[_ks(k) for k in thieu], khoa_trung=[_ks(k) for k in trung],
                khoa_la=[_ks(k) for k in la], khoa_run_error=[_ks(k) for k in loi],
                moc_thieu=[_ks(k) for k in moc_thieu], cap_thieu=[_ks(k) for k in cap_thieu],
                cap_khong_so_duoc=[_ks(k) for k in cap_ks],
                khoa_thieu_dai_luong=[dict(khoa=_ks(k), truong=dl[k]) for k in want if k in dl],
                n_loi_worker=len(loi_worker),
                khoa_loi_worker=[e['khoa'] for e in loi_worker if e.get('khoa')])


def merge_retries(rows, retry_rows):
    """Rows of the first pass with each run_error or loi_worker row (LOI_STATUS) replaced by its
    rerun (same key) when the rerun returned; a run_error row is kept when its rerun gave only a
    loi_worker stand-in (C3-19 round 3). Returns (rows, attempts, loi_chua_ro): attempts = both
    attempts of every rerun key (attempt 1, 2; S03 section 8 condition 12 'báo cả hai lần'),
    loi_chua_ro = keys whose rerun succeeded after an error ('lỗi chưa rõ nguyên nhân')."""
    retry = {key_of(r): r for r in retry_rows}
    out, attempts, loi_chua_ro = [], [], []
    for r in rows:
        k = key_of(r)
        if r.get('status') not in LOI_STATUS or k not in retry:
            out.append(r)
            continue
        r2 = retry[k]
        out.append(r if r2.get('_khong_ket_qua') and not r.get('_khong_ket_qua') else r2)
        attempts += [dict(r, attempt=1), dict(r2, attempt=2)]
        if r2.get('status') not in LOI_STATUS:
            loi_chua_ro.append(dict(khoa=_ks(k), error_lan_1=r.get('error', ''),
                                    status_lan_2=r2.get('status', ''),
                                    ghi_chu='lỗi chưa rõ nguyên nhân'))
    return out, attempts, loi_chua_ro


def per_condition_rows(rows, refs, thr=None):
    """tom_tat_theo_dieu_kien.csv: one row per run x response (a missing reference run gives
    empty hieu: C3-19). thr: thresholds() by default."""
    thr = thresholds() if thr is None else thr
    out = []
    for r in rows:
        rm = refs.get((r['controller'], r['mu'], r['v']), {})
        for resp in RESPONSES:
            g, gm = r.get(resp, ''), rm.get(resp, '')
            h = _hieu(g, gm)
            rel = _rel(h, gm)
            vuot = '' if h is None or not math.isfinite(h) else abs(h) > thr[resp]
            out.append(dict(controller=r['controller'], io_delay_ms=r['io_delay_ms'],
                            slip_omega_s=r['slip_omega_s'], la_moc=r['la_moc'], mu=r['mu'], v=r['v'],
                            dap_ung=resp, ten_c2=TEN_C2[resp], ten_s03=TEN_S03[resp],
                            don_vi=DON_VI[resp], gia_tri_moc=gm, gia_tri=g,
                            hieu='' if h is None else h, hieu_tuong_doi='' if rel is None else rel,
                            nguong=thr[resp], vuot_nguong=vuot,
                            completed_moc=rm.get('completed', ''), completed=r.get('completed', ''),
                            status_moc=rm.get('status', ''), status=r.get('status', '')))
    return out


def summary_rows(rows, refs, controllers=CONTROLLERS, thr=None):
    """tom_tat.csv: one row per controller x combo (incl. the reference) x response. Only the
    conditions where both the run and its reference run exist enter (a missing one is reported
    by completeness(), not raised: C3-19). thr: thresholds() by default."""
    thr = thresholds() if thr is None else thr
    by_key = {(r['controller'], r['io_delay_ms'], r['slip_omega_s'], r['mu'], r['v']): r
              for r in rows}
    out = []
    for ctrl in controllers:
        for delay in DELAYS:
            for w in OMEGAS:
                pairs = [(by_key.get((ctrl, delay, w, mu, v)), refs.get((ctrl, mu, v)))
                         for mu in MU_LEVELS for v in SPEEDS]
                pairs = [(r, rm) for r, rm in pairs if r is not None and rm is not None]
                rs, ms = [r for r, _ in pairs], [rm for _, rm in pairs]
                for resp in RESPONSES:
                    per = [(r['mu'], r['v'], _hieu(r.get(resp, ''), rm.get(resp, '')),
                            _rel(_hieu(r.get(resp, ''), rm.get(resp, '')), rm.get(resp, '')))
                           for r, rm in zip(rs, ms)]
                    hs = [(mu, v, h) for mu, v, h, _ in per if h is not None and math.isfinite(h)]
                    rels = [x for *_, x in per if x is not None and math.isfinite(x)]
                    vuot = [(mu, v) for mu, v, h, _ in per
                            if h is not None and math.isfinite(h) and abs(h) > thr[resp]]
                    imax = max(hs, key=lambda t: abs(t[2])) if hs else None
                    out.append(dict(
                        controller=ctrl, io_delay_ms=delay, slip_omega_s=w,
                        la_moc=(delay == REF[0] and w == REF[1]), dap_ung=resp,
                        ten_c2=TEN_C2[resp], ten_s03=TEN_S03[resp], don_vi=DON_VI[resp],
                        nguong=thr[resp], n_dieu_kien=len(hs),
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


# ── command line and DLL provenance ─────────────────────────────────────────────────

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='ket_qua_c2')
    ap.add_argument('--dll', default=DEFAULT_DLL)
    ap.add_argument('--fw', default=DEFAULT_FW,
                    help='AGV_FIRMWARE git tree; when this folder is readable, esp32.commit of '
                         'fwsim_build.json must be a commit of it (C3-20)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--ctrl', nargs='+', choices=CONTROLLERS, default=list(CONTROLLERS),
                    help='controllers to run (default both); --ctrl ADRC = LADRC only, 81 runs')
    return ap


def select_controllers(ctrl):
    """--ctrl value -> tuple in CONTROLLERS order, duplicates removed."""
    return tuple(c for c in CONTROLLERS if c in ctrl)


def make_jobs(ctrls):
    """(controller, io_delay_ms, slip_omega_s, mu, v) of the grid for the selected ctrls."""
    return [(c, d, w, mu, v) for c in ctrls for d in DELAYS for w in OMEGAS
            for mu in MU_LEVELS for v in SPEEDS]


def full_command(args, ctrls):
    """The command of this run with every option explicit and absolute paths (meta.json
    command_full_argv): a rerun needs only to copy it."""
    return [sys.executable, os.path.abspath(__file__),
            '--out', os.path.abspath(args.out), '--dll', os.path.abspath(args.dll),
            '--fw', os.path.abspath(args.fw),
            '--workers', str(args.workers), '--ctrl', *ctrls]


def fw_readable(fw):
    """True when fw names a readable folder (the AGV tree to resolve esp32.commit in)."""
    return fw is not None and os.path.isdir(fw) and os.access(fw, os.R_OK)


def commit_in_repo(fw, commit):
    """True when commit names a commit object of the git repo fw (C3-20):
    'git -C <fw> cat-file -e <commit>^{commit}' exits 0. False on any other exit code, when
    git cannot be run or does not end within GIT_TIMEOUT_S."""
    try:
        r = subprocess.run(['git', '-C', fw, 'cat-file', '-e', f'{commit}^{{commit}}'],
                           capture_output=True, text=True, timeout=GIT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def dll_info(dll, fw=None):
    """Provenance of --dll: absolute path, sha256, the fwsim_build.json next to it and
    warnings; never exits (a sensitivity trial continues on an unknown or dirty build).
    build_verified (= sidecar_khop_sha) confirms only the DLL SHA-256: the json's dll_sha256
    equals the sha256 of this DLL (C3-20). It says nothing of the ESP32 source; a reader must
    also require nguon_esp32_du (nguon_build 'đã xác minh') before using the results. Without
    the SHA match: 'chưa xác minh' and esp32_dirty None (the json's fields stay in fwsim_build,
    as read, not as a result).
    nguon_esp32_du (C3-20 rounds 2, 3): the SHA matches and the json holds esp32.commit as a
    7-40 character hex string (COMMIT_RE; not 'UNKNOWN', empty or other text) and a boolean
    esp32.dirty; when fw (--fw) is a readable folder the commit must also be a commit of that
    repo (commit_in_repo; esp32_commit_trong_repo True / False, None = not checked).
    nguon_build 'đã xác minh' needs both, 'khớp SHA, thiếu nguồn ESP32' when only the SHA."""
    path = os.path.abspath(dll)
    sha = s03.sha256_file(dll)
    build_json = os.path.join(os.path.dirname(path), 'fwsim_build.json')
    build, warnings = None, []
    if os.path.isfile(build_json):
        with open(build_json, encoding='utf-8') as f:
            build = json.load(f)
    else:
        warnings.append('no fwsim_build.json next to the DLL: the origin of the DLL is '
                        'unknown; sensitivity trial, run continues')
    esp32_dirty = None
    if build is not None:
        esp32 = build.get('esp32') or {}
        if 'dirty' not in esp32:
            warnings.append('fwsim_build.json has no esp32.dirty: the dirty state of the '
                            'firmware tree at build time is unknown; sensitivity trial, '
                            'run continues')
        else:
            esp32_dirty = esp32['dirty']
            if esp32_dirty:
                warnings.append('fwsim.dll was built from a dirty firmware tree '
                                '(esp32.dirty = true); sensitivity trial, run continues')
        hil = build.get('hil_framework') or {}
        if hil.get('dirty'):
            warnings.append('the hil_framework repo was dirty at build time '
                            f"(modified: {', '.join(hil.get('modified') or ['?'])}); "
                            'sensitivity trial, run continues')
        recorded = build.get('dll_sha256')
        if recorded and recorded != sha:
            warnings.append('fwsim_build.json dll_sha256 differs from the DLL on disk: '
                            'fwsim_build.json does not describe this DLL; sensitivity '
                            'trial, run continues')
        elif not recorded:
            warnings.append('fwsim_build.json has no dll_sha256: it cannot be tied to this '
                            'DLL, the build origin is unverified; sensitivity trial, run '
                            'continues')
    verified = build is not None and build.get('dll_sha256') == sha
    if not verified:
        esp32_dirty = None
    esp32 = (build or {}).get('esp32') or {}
    commit = esp32.get('commit')
    esp32_commit = commit if verified and isinstance(commit, str) \
        and COMMIT_RE.fullmatch(commit) else None
    fw_dir = os.path.abspath(fw) if fw is not None else None
    trong_repo = commit_in_repo(fw_dir, esp32_commit) \
        if esp32_commit is not None and fw_readable(fw_dir) else None
    # C3-20: the ESP32 source of a sidecar tied to this DLL is complete with a commit and a
    # boolean dirty state (dirty true still counts: no "clean build" rule is decided)
    esp32_ok = esp32_commit is not None and isinstance(esp32_dirty, bool) and trong_repo is not False
    if trong_repo is False:
        warnings.append(f'esp32.commit {esp32_commit} of fwsim_build.json is not a commit of the '
                        f'AGV repo {fw_dir} (git cat-file -e)')
    if verified and not esp32_ok:
        warnings.append('fwsim_build.json matches the DLL sha256 but lacks the ESP32 source '
                        '(esp32.commit as a 7-40 character hex commit, of the --fw repo when '
                        'readable, esp32.dirty): the firmware the DLL was built from is '
                        'unknown; sensitivity trial, run continues')
    return dict(path=path, sha256=sha, fwsim_build_json=build_json, fwsim_build=build,
                esp32_dirty=esp32_dirty, build_verified=verified,
                sidecar_khop_sha=verified, esp32_commit=esp32_commit, fw_dir=fw_dir,
                esp32_commit_trong_repo=trong_repo, nguon_esp32_du=esp32_ok,
                nguon_build=('đã xác minh' if verified and esp32_ok else
                             'khớp SHA, thiếu nguồn ESP32' if verified else 'chưa xác minh'),
                warnings=warnings)


def main(argv=None):
    args = build_parser().parse_args(argv)
    ctrls = select_controllers(args.ctrl)
    bang = os.path.join(args.out, 'bang.csv')
    if os.path.exists(bang):
        sys.exit(f'{bang} exists: refusing to overwrite')
    thr = thresholds()
    os.makedirs(args.out, exist_ok=True)
    t0, t_iso = time.time(), time.strftime('%Y-%m-%dT%H:%M:%S')

    fw0 = fwsim.FirmwareSim(args.dll)
    if not fw0.pos_cols:
        sys.exit('fwsim.dll has no pos_ctrl.c: C2 needs the campaign firmware')
    info = dll_info(args.dll, args.fw)
    for w in info['warnings']:
        print(f'WARNING: {w}', flush=True)
    gains = {c: s03.defaults(fw0, c) for c in ctrls}

    jobs = make_jobs(ctrls)
    n_workers = min(args.workers, 8)

    loi_worker = []          # C3-19: exceptions out of the pool, recorded instead of exiting

    def run_pool(js, tag='', lan=1):
        """Rows of js; a worker exception or a key without result within HAN_CHO_WORKER_S
        gives a _loi_row stand-in (C3-19 round 3) and a loi_worker entry."""
        out, n_exc, ve = [], 0, set()
        with mp.Pool(processes=n_workers, initializer=_init, initargs=(args.dll,)) as pool:
            it = pool.imap_unordered(_job_safe, js)
            i = 0
            while True:
                try:
                    row = it.next(timeout=HAN_CHO_WORKER_S)
                except StopIteration:
                    break
                except mp.TimeoutError:
                    cho = [tuple(j) for j in js if tuple(j) not in ve]
                    err = f'{HET_HAN}: no worker result within {HAN_CHO_WORKER_S} s'
                    for k in cho:
                        loi_worker.append(dict(khoa=_ks(k), lan=lan, error=err, ly_do=HET_HAN))
                        out.append(_loi_row(k, err, het_han=True))
                    print(f'{tag}WARNING: no worker result within {HAN_CHO_WORKER_S} s, '
                          f'{len(cho)} keys not returned, pool terminated', flush=True)
                    pool.terminate()
                    break
                except Exception as exc:     # noqa: BLE001 (raised by the iterator itself)
                    n_exc += 1
                    loi_worker.append(dict(khoa='', lan=lan,
                                           error=f'{type(exc).__name__}: {exc}'))
                    print(f'{tag}WARNING: pool iterator raised {type(exc).__name__}', flush=True)
                    if n_exc > len(js):
                        break
                    continue
                i += 1
                if '_loi_worker' in row:
                    k = tuple(row['_khoa'])
                    ve.add(k)
                    loi_worker.append(dict(khoa=_ks(k), lan=lan, error=row['_loi_worker']))
                    out.append(_loi_row(k, row['_loi_worker']))
                    print(f'{tag}[{i}/{len(js)}] {_ks(k)} worker exception', flush=True)
                    continue
                ve.add(key_of(row))
                out.append(row)
                print(f"{tag}[{i}/{len(js)}] {row['controller']} delay={row['io_delay_ms']} "
                      f"omega_s={row['slip_omega_s']} mu={row['mu']} v={row['v']} "
                      f"status={row['status']} t={row['t_chay_s']:.1f}s", flush=True)
        return out

    rows = run_pool(jobs)
    # S03 section 8 condition 12: a run_error is run once more, in a new pool (new processes);
    # C3-19 round 3: so is a worker exception (not a key past HAN_CHO_WORKER_S)
    retry_jobs = [key_of(r) for r in rows
                  if r.get('status') in LOI_STATUS and not r.get('_het_han')]
    n_run_error_1 = sum(r.get('status') == 'run_error' for r in rows)
    retry_rows = run_pool(retry_jobs, 'rerun ', 2) if retry_jobs else []
    rows, attempts, loi_chua_ro = merge_retries(rows, retry_rows)
    rows = [r for r in rows if not r.get('_khong_ket_qua')]     # still no row: key missing
    rows.sort(key=lambda r: (CONTROLLERS.index(r['controller']), r['io_delay_ms'],
                             r['slip_omega_s'], r['mu'], r['v']))
    refs = {(r['controller'], r['mu'], r['v']): r for r in rows if r['la_moc']}
    ok_keys = {_ks(key_of(r)) for r in rows if r.get('status') != 'run_error'}
    loi_con = [e for e in loi_worker
               if not e['khoa'] or e.get('ly_do') == HET_HAN or e['khoa'] not in ok_keys]
    kiem_cap = completeness(rows, ctrls, loi_con)

    s03.write_csv(bang, rows, BANG_COLS)
    s03.write_csv(os.path.join(args.out, 'tom_tat_theo_dieu_kien.csv'),
                  per_condition_rows(rows, refs, thr), DK_COLS)
    s03.write_csv(os.path.join(args.out, 'tom_tat.csv'), summary_rows(rows, refs, ctrls, thr),
                  TT_COLS)
    s03.write_csv(os.path.join(args.out, 'chay_lai.csv'), attempts, BANG_COLS + ['attempt'])
    ly_do = ([] if info['build_verified'] else ['nguon build chua xac minh (C3-20)']) + \
        ([] if not info['build_verified'] or info['nguon_esp32_du']
         else ['nguon ESP32 chua du: esp32.commit, esp32.dirty (C3-20)']) + \
        ([] if kiem_cap['ket_luan'] == 'đủ cặp' else ['chua ket luan: thieu cap hay loi (C3-19)'])

    root = os.path.join(HERE, '..')
    cmd_full = full_command(args, ctrls)
    meta = dict(
        command=' '.join(sys.argv), command_full=subprocess.list2cmdline(cmd_full),
        command_full_argv=cmd_full, cwd=os.getcwd(), controllers=list(ctrls),
        started=t_iso, ended=time.strftime('%Y-%m-%dT%H:%M:%S'),
        elapsed_s=time.time() - t0, n_runs=len(rows), n_workers=n_workers,
        cpu_count=os.cpu_count(),
        dll=info,
        fwsim_build_json=info['fwsim_build_json'], fwsim_build=info['fwsim_build'],
        esp32_dirty=info['esp32_dirty'], warnings=info['warnings'],
        nguon_build=info['nguon_build'],
        kiem_cap=kiem_cap, n_run_error_lan_1=n_run_error_1,
        loi_chua_ro_nguyen_nhan=loi_chua_ro, chay_lai_csv='chay_lai.csv', loi_worker=loi_worker,
        dung_lam_luot_xac_nhan=not ly_do, ly_do_khong_dung=ly_do,
        gains=gains,
        seeds=dict(plant=SEED, imu_noise=s03.IMU['noise_seed'],
                   note='no other random source'),
        imu=dict(s03.IMU),
        mu_levels=list(MU_LEVELS), mu_source=MU_SOURCE,
        speeds=list(SPEEDS), delays=list(DELAYS), omegas=list(OMEGAS), ref=list(REF),
        thresholds=thr, thresholds_source=THRESH_SOURCE, ten_c2=TEN_C2, ten_s03=TEN_S03, don_vi=DON_VI,
        plant_src=fwsim.PLANT_SRC, libraries=fwsim.lib_versions(),
        sha256={p: s03.sha256_file(os.path.join(root, p)) for p in SHA_FILES},
        counts=dict(status=dict(Counter(str(r.get('status', '')) for r in rows)),
                    completed=dict(Counter(str(r.get('completed', '')) for r in rows))),
    )
    with open(os.path.join(args.out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    for w in info['warnings']:
        print(f'WARNING: {w}', flush=True)
    if kiem_cap['ket_luan'] != 'đủ cặp':
        print(f"WARNING: inconclusive (S03 section 8 condition 12): {kiem_cap['n_cap_thieu']} of "
              f"{kiem_cap['n_cap_can']} pairs missing, {len(kiem_cap['khoa_thieu'])} keys missing, "
              f"{len(kiem_cap['khoa_run_error'])} run_error left, "
              f"{kiem_cap['n_cap_khong_so_duoc']} pairs not comparable, "
              f"{kiem_cap['n_loi_worker']} worker exceptions; see meta.json kiem_cap; "
              'report to Liam', flush=True)
    if loi_chua_ro:
        print(f'WARNING: {len(loi_chua_ro)} run_error rerun ok (cause unknown): rerun values used, '
              'both attempts in chay_lai.csv', flush=True)
    print(f'done: {len(rows)} runs in {time.time() - t0:.0f} s -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
