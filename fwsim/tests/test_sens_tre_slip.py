"""Tests of fwsim/sens_tre_slip.py (task C2): patched_simulate puts io_delay_ms and
slip_omega_s into the run while s03_doe.run_one stays the evaluation unchanged.
DLL: ket_qua_c2/build/fwsim.dll, fallback fwsim/build/fwsim.dll (skip if neither).
Run: python -m pytest -q fwsim/tests/test_sens_tre_slip.py"""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import fwsim                                            # noqa: E402
import o1_feasibility as o1                             # noqa: E402
import s03_doe as s03                                   # noqa: E402
import sens_tre_slip as sens                            # noqa: E402

_DLL = os.path.join(fwsim.HERE, '..', 'ket_qua_c2', 'build', 'fwsim.dll')
if not os.path.isfile(_DLL):
    _DLL = os.path.join(fwsim.HERE, 'build', 'fwsim.dll')
needs_dll = pytest.mark.skipif(not os.path.isfile(_DLL), reason='fwsim.dll not built')


@pytest.fixture(scope='module')
def path():
    return o1.Path8()


@pytest.fixture(scope='module')
def fw():
    f = fwsim.FirmwareSim(_DLL)
    if not f.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    return f


def _same(a, b):
    """Bit-exact equality of two row values (NaN == NaN; repr for floats)."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return repr(a) == repr(b)


@needs_dll
@pytest.mark.parametrize('ctrl', ('PID', 'ADRC'))
def test_ref_equals_direct(ctrl, fw, path):
    """The reference combo (delay 0, omega_s 0.1 = params.py default) through run_case is
    s03_doe.run_one bit for bit: same row values, same log arrays; fwsim.simulate restored."""
    orig = fwsim.simulate
    row1, log1 = sens.run_case(fw, path, ctrl, 0.8, 0.436, 0, 0.1)
    assert fwsim.simulate is orig
    row2, log2 = s03.run_one(fw, path, ctrl, s03.defaults(fw, ctrl), 0.8, 0.436)
    for k in row2:
        assert _same(row1[k], row2[k]), f'{k}: {row1[k]!r} != {row2[k]!r}'
    assert set(log1) == set(log2)
    for k in log2:
        assert np.array_equal(log1[k], log2[k], equal_nan=True), k


@needs_dll
def test_slip_omega_reaches_plant(monkeypatch, fw):
    """patched_simulate sets slip_omega_s on the params of that run only: plant_step sees it
    at every call, slip_nsub follows traction_nsub, MecanumParams() keeps the 0.1 default."""
    seen = []
    orig_step = fwsim.plant_step

    def rec_step(x, u, p, dt, rng):
        seen.append(p.slip_omega_s)
        return orig_step(x, u, p, dt, rng)

    monkeypatch.setattr(fwsim, 'plant_step', rec_step)
    nsub = {}
    for w in (0.05, 0.2):
        seen.clear()
        rec = []
        p = fwsim.plant_params(argparse.Namespace(ground='traction', mu=0.436))
        with sens.patched_simulate(0, w, rec):
            fwsim.simulate(fw, lambda t: (0.2, 0.0, 0.0), 0.05, p, seed=1, mode='PID',
                           imu=dict(s03.IMU))
        assert seen and all(x == w for x in seen)
        assert len(rec) == 1 and rec[0]['slip_omega_s'] == w
        p_w = fwsim.plant_params(argparse.Namespace(ground='traction', mu=0.436))
        p_w.slip_omega_s = w
        nsub[w] = fwsim.traction_nsub(p_w)
        assert rec[0]['slip_nsub'] == nsub[w]
        assert p.slip_omega_s == w                       # in-memory on that params object
    assert nsub[0.05] != nsub[0.2]
    assert fwsim.MecanumParams().slip_omega_s == 0.1     # params.py default untouched


@needs_dll
def test_io_delay_2ms(monkeypatch, fw):
    """io_delay_ms is a pure 1 ms-per-slot duty pipeline: with delay 0 the duty of the tick
    at t_j reaches the plant in the step at t_j, with delay 2 in the step at t_j + 2."""
    orig_step = fwsim.plant_step
    runs = {}
    for delay in (0, 2):
        u_rec = []

        def rec_step(x, u, p, dt, rng, u_rec=u_rec):
            u_rec.append(np.array(u, dtype=float).copy())
            return orig_step(x, u, p, dt, rng)

        monkeypatch.setattr(fwsim, 'plant_step', rec_step)
        p = fwsim.plant_params(argparse.Namespace(ground='traction', mu=0.436))
        with sens.patched_simulate(delay, 0.1):
            log = fwsim.simulate(fw, lambda t: (0.2, 0.0, 0.0), 0.05, p, seed=1, mode='PID',
                                 imu=dict(s03.IMU))
        runs[delay] = (u_rec, log)

    u0, log0 = runs[0]
    u2, log2 = runs[2]
    j = next(i for i, d in enumerate(log0['duty']) if np.any(d != 0))
    t_j = int(round(log0['t'][j] * 1000))                # plant call index of the tick
    duty_j = log0['duty'][j]
    prev = log0['duty'][j - 1] if j else np.zeros(4)     # all zeros: j is the first nonzero tick
    # plant input is zero before t_j in both runs -> the duty decided up to t_j is identical
    assert np.array_equal(log0['duty'][:j + 1], log2['duty'][:j + 1])
    # delay 0: the tick duty is applied in the step starting at t_j
    assert np.array_equal(u0[t_j], duty_j) and not np.array_equal(u0[t_j - 1], duty_j)
    # delay 2: the tick duty is applied two steps later
    assert np.array_equal(u2[t_j], prev) and np.array_equal(u2[t_j + 1], prev)
    assert np.array_equal(u2[t_j + 2], duty_j)


# ── command line and DLL provenance (no DLL needed) ─────────────────────────────────

def test_default_dll():
    """--dll default = fwsim/build/fwsim.dll of this repo (the build_fwsim.py --out default),
    not the ket_qua_c2 results folder the run01 command was launched from."""
    dll = sens.build_parser().parse_args([]).dll
    repo = os.path.dirname(fwsim.HERE)
    want = os.path.join(repo, 'fwsim', 'build', 'fwsim.dll')
    assert os.path.normcase(os.path.abspath(dll)) == os.path.normcase(want)
    assert dll == sens.DEFAULT_DLL
    assert 'ket_qua_c2' not in dll


def test_full_command(tmp_path, monkeypatch):
    """command_full_argv: the script, every option explicit, absolute paths (a rerun needs
    only to copy it); parse_args(argv[2:]) round-trips to the same options."""
    monkeypatch.chdir(tmp_path)
    rel_out, rel_dll = os.path.join('sub dir', 'out'), os.path.join('sub dir', 'fwsim.dll')
    args = sens.build_parser().parse_args(['--out', rel_out, '--dll', rel_dll, '--ctrl', 'ADRC'])
    argv = sens.full_command(args, ('ADRC',))
    assert argv[0] == sys.executable and argv[1] == os.path.abspath(sens.__file__)
    for opt in ('--out', '--dll', '--workers', '--ctrl'):
        assert opt in argv
    assert os.path.isabs(argv[argv.index('--out') + 1])
    assert argv[argv.index('--out') + 1] == os.path.abspath(rel_out)
    assert argv[argv.index('--dll') + 1] == os.path.abspath(rel_dll)
    rt = sens.build_parser().parse_args(argv[2:])
    assert rt.out == os.path.abspath(rel_out) and rt.dll == os.path.abspath(rel_dll)
    assert rt.workers == args.workers and rt.ctrl == ['ADRC']
    argv_def = sens.full_command(sens.build_parser().parse_args([]), sens.CONTROLLERS)
    assert argv_def[-3:] == ['--ctrl', 'PID', 'ADRC']
    assert f'"{os.path.abspath(rel_dll)}"' in subprocess.list2cmdline(argv)


def _mk_dll(tmp_path, data=b'fake-dll', build=None):
    """tmp_path/fwsim.dll holding data, plus fwsim_build.json (dict) when build is given."""
    dll = tmp_path / 'fwsim.dll'
    dll.write_bytes(data)
    if build is not None:
        (tmp_path / 'fwsim_build.json').write_text(json.dumps(build), encoding='utf-8')
    return str(dll), data


def test_dll_info_no_json(tmp_path):
    """No fwsim_build.json next to the DLL: build None, dirty None, one warning."""
    dll, data = _mk_dll(tmp_path)
    info = sens.dll_info(dll)
    assert info['path'] == os.path.abspath(dll)
    assert info['sha256'] == hashlib.sha256(data).hexdigest()
    assert info['fwsim_build_json'] == os.path.join(os.path.dirname(info['path']),
                                                  'fwsim_build.json')
    assert info['fwsim_build'] is None and info['esp32_dirty'] is None
    assert len(info['warnings']) == 1
    assert 'fwsim_build.json' in info['warnings'][0]
    assert 'sensitivity trial' in info['warnings'][0]


def test_dll_info_clean(tmp_path):
    """A clean build json that matches the DLL: no warning. (C3-20 round 2: 'clean' includes
    the ESP32 commit; without it a warning, see test_c3_20_esp32_source.)"""
    dll, data = _mk_dll(tmp_path, build=dict(
        esp32=dict(commit='5bde4c5', dirty=False), hil_framework=dict(dirty=False),
        dll_sha256=hashlib.sha256(b'fake-dll').hexdigest()))
    info = sens.dll_info(dll)
    assert info['warnings'] == []
    assert info['esp32_dirty'] is False
    assert info['fwsim_build']['esp32']['dirty'] is False


def test_dll_info_esp32_dirty(tmp_path):
    """esp32.dirty true in a json tied to this DLL (dll_sha256 matches): a warning, esp32_dirty
    True. (C3-20: the json carries dll_sha256 now; without it esp32_dirty is None.)"""
    dll, _ = _mk_dll(tmp_path, build=dict(esp32=dict(dirty=True),
                                          hil_framework=dict(dirty=False),
                                          dll_sha256=hashlib.sha256(b'fake-dll').hexdigest()))
    info = sens.dll_info(dll)
    assert info['esp32_dirty'] is True and info['build_verified'] is True
    assert any('dirty' in w for w in info['warnings'])
    assert all('sensitivity trial' in w for w in info['warnings'])


def test_dll_info_no_esp32(tmp_path):
    """esp32 section missing in a present json: dirty state unknown warning, dirty None."""
    dll, _ = _mk_dll(tmp_path, build=dict(compiler='gcc x'))
    info = sens.dll_info(dll)
    assert info['esp32_dirty'] is None
    assert any('unknown' in w for w in info['warnings'])


def test_dll_info_hil_dirty(tmp_path):
    """hil_framework.dirty true: a warning that lists the modified files."""
    dll, _ = _mk_dll(tmp_path, build=dict(
        esp32=dict(dirty=False),
        hil_framework=dict(dirty=True, modified=['fwsim/sens_tre_slip.py'])))
    info = sens.dll_info(dll)
    assert any('fwsim/sens_tre_slip.py' in w for w in info['warnings'])


def test_dll_info_sha_mismatch(tmp_path):
    """dll_sha256 in the json != the DLL on disk: the json does not describe this DLL."""
    dll, _ = _mk_dll(tmp_path, build=dict(esp32=dict(dirty=False), dll_sha256='0' * 64))
    info = sens.dll_info(dll)
    assert any('dll_sha256' in w for w in info['warnings'])


def test_ctrl_filter():
    """--ctrl ADRC selects LADRC only: 81 jobs, the ADRC subset of the full grid in order."""
    args = sens.build_parser().parse_args(['--ctrl', 'ADRC'])
    ctrls = sens.select_controllers(args.ctrl)
    assert ctrls == ('ADRC',)
    jobs = sens.make_jobs(ctrls)
    assert len(jobs) == 81 and all(j[0] == 'ADRC' for j in jobs)
    full = sens.make_jobs(sens.CONTROLLERS)
    assert jobs == [j for j in full if j[0] == 'ADRC']
    assert len(full) == 162
    args2 = sens.build_parser().parse_args(['--ctrl', 'ADRC', 'PID', 'ADRC'])
    assert sens.select_controllers(args2.ctrl) == ('PID', 'ADRC')
    with pytest.raises(SystemExit):
        sens.build_parser().parse_args(['--ctrl', 'LADRC'])


class _FakePool:
    """mp.Pool stand-in: imap_unordered runs each job in-process (no _init, no DLL)."""

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, func, jobs):
        return (func(j) for j in jobs)


def _fake_job(job):
    """A deterministic full row for one (ctrl, delay, omega_s, mu, v) job."""
    ctrl, delay, omega_s, mu, v = job
    return dict(controller=ctrl, io_delay_ms=delay, slip_omega_s=omega_s,
                la_moc=(delay == sens.REF[0] and omega_s == sens.REF[1]),
                mu=mu, v=v, seed_plant=sens.SEED, seed_imu_noise=1, n_seed=1, slip_nsub=1,
                status='ok', completed='true',
                e_perp_max=0.01 + 0.001 * delay, e_perp_max_cens=0.02,
                tv_duty=1.0 + v, sat_frac=0.0, sat_any_frac=0.0,
                e_perp_global_max=0.03, e_perp_rms=0.005, pos_reset_mid=0,
                gate_outcome='dat', error='', t_chay_s=0.0)


class _FakeFW:
    """FirmwareSim stand-in: a campaign DLL (pos_cols) with no build info."""

    def __init__(self, dll):
        self.pos_cols = True
        self.build_info = {}


def _read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_main_ctrl_and_meta(tmp_path, monkeypatch, capsys):
    """main end to end with fakes: --ctrl ADRC runs 81 jobs; meta.json holds controllers,
    command_full_argv, the DLL sha256/fwsim_build and the warnings; a missing
    fwsim_build.json only warns; an existing bang.csv is still refused."""
    monkeypatch.setattr(sens.fwsim, 'FirmwareSim', _FakeFW)
    monkeypatch.setattr(sens.s03, 'defaults', lambda fw, c: {'g': c})
    monkeypatch.setattr(sens.mp, 'Pool', _FakePool)
    monkeypatch.setattr(sens, '_job', _fake_job)
    monkeypatch.setitem(sens.s03.THRESH, 'tv_duty', 0.4)   # C3-21: main reads s03_doe.THRESH

    dll = tmp_path / 'fwsim.dll'
    dll.write_bytes(b'fake-dll')
    sha = hashlib.sha256(b'fake-dll').hexdigest()
    (tmp_path / 'fwsim_build.json').write_text(json.dumps(
        dict(esp32=dict(dirty=True), hil_framework=dict(dirty=False), dll_sha256=sha)),
        encoding='utf-8')
    out = tmp_path / 'out_adrc'
    sens.main(['--out', str(out), '--dll', str(dll), '--ctrl', 'ADRC', '--workers', '1'])
    meta = json.loads((out / 'meta.json').read_text(encoding='utf-8'))
    assert meta['controllers'] == ['ADRC'] and meta['n_runs'] == 81
    argv = meta['command_full_argv']
    assert argv[0] == sys.executable and argv[1] == os.path.abspath(sens.__file__)
    assert argv[argv.index('--out') + 1] == os.path.abspath(str(out))
    assert argv[argv.index('--dll') + 1] == os.path.abspath(str(dll))
    assert argv[argv.index('--ctrl') + 1:] == ['ADRC']
    assert meta['command_full'] == subprocess.list2cmdline(argv)
    assert meta['dll']['sha256'] == sha
    assert meta['dll']['esp32_dirty'] is True
    assert meta['fwsim_build']['esp32']['dirty'] is True
    assert meta['esp32_dirty'] is True
    assert meta['warnings'] and any('dirty' in w for w in meta['warnings'])
    assert 'WARNING' in capsys.readouterr().out
    bang = _read_csv(out / 'bang.csv')
    assert len(bang) == 81 and {r['controller'] for r in bang} == {'ADRC'}
    tt = _read_csv(out / 'tom_tat.csv')
    assert len(tt) == 9 * len(sens.RESPONSES) and {r['controller'] for r in tt} == {'ADRC'}

    # no fwsim_build.json next to the DLL: only a warning, the run still completes
    dll2 = tmp_path / 'alone' / 'fwsim.dll'
    dll2.parent.mkdir()
    dll2.write_bytes(b'other-dll')
    out2 = tmp_path / 'out_nojson'
    sens.main(['--out', str(out2), '--dll', str(dll2), '--ctrl', 'ADRC', '--workers', '1'])
    meta2 = json.loads((out2 / 'meta.json').read_text(encoding='utf-8'))
    assert meta2['fwsim_build'] is None
    assert any('fwsim_build.json' in w for w in meta2['warnings'])

    # an existing bang.csv is still refused
    with pytest.raises(SystemExit):
        sens.main(['--out', str(out), '--dll', str(dll), '--ctrl', 'ADRC'])


# ── review round 1 (C3-19, C3-20, C3-21) ────────────────────────────────────────────

X_SO = ('ADRC', 1, 0.05, 0.413, 0.2)          # a comparison key of the --ctrl ADRC grid
X_MOC = ('ADRC', 0, 0.1, 0.436, 0.5)          # a reference key
_SHA = hashlib.sha256(b'fake-dll').hexdigest()   # sha256 of the _mk_dll default DLL


def _err_row(job):
    ctrl, delay, omega_s, mu, v = job
    return dict(controller=ctrl, io_delay_ms=delay, slip_omega_s=omega_s,
                la_moc=(delay == sens.REF[0] and omega_s == sens.REF[1]), mu=mu, v=v,
                status='run_error', completed='', error='RuntimeError: boom', t_chay_s=0.0)


def _drop_pool(drop):
    """_FakePool that never returns the jobs in drop (a missing key)."""
    class P(_FakePool):
        def imap_unordered(self, func, jobs):
            return (func(j) for j in jobs if tuple(j) not in drop)
    return P


def _main_fakes(monkeypatch, tmp_path, job=_fake_job, pool=_FakePool, build='match'):
    """main with fakes, --ctrl ADRC; build: 'match' (json with the DLL sha256 and the ESP32
    commit, dirty), 'nosha' ({"esp32": {"dirty": false}}), 'badsha', 'shaonly'
    ({"dll_sha256": <right>}, C3-20 round 2) or a dict (written as is). Returns the out folder."""
    monkeypatch.setattr(sens.fwsim, 'FirmwareSim', _FakeFW)
    monkeypatch.setattr(sens.s03, 'defaults', lambda fw, c: {'g': c})
    monkeypatch.setattr(sens.mp, 'Pool', pool)
    monkeypatch.setattr(sens, '_job', job)
    monkeypatch.setitem(sens.s03.THRESH, 'tv_duty', 0.4)
    b = dict(esp32=dict(dirty=False))
    if isinstance(build, dict):
        b = build
    elif build == 'match':
        b = dict(esp32=dict(commit='5bde4c5', dirty=False), dll_sha256=_SHA)
    elif build == 'shaonly':
        b = dict(dll_sha256=_SHA)
    elif build == 'badsha':
        b['dll_sha256'] = '0' * 64
    dll, _ = _mk_dll(tmp_path, build=b)
    out = tmp_path / 'out'
    sens.main(['--out', str(out), '--dll', dll, '--ctrl', 'ADRC', '--workers', '1'])
    return out


def _meta(out):
    return json.loads((out / 'meta.json').read_text(encoding='utf-8'))


def test_c3_19_missing_pair_no_crash(tmp_path, monkeypatch):
    """C3-19 (Codex): a comparison run missing, every reference present. Old code: KeyError in
    summary_rows after bang.csv was written, no meta.json. Now every file is written and
    meta.json kiem_cap says 'chưa kết luận' with the missing key and pair."""
    out = _main_fakes(monkeypatch, tmp_path, pool=_drop_pool({X_SO}))
    meta = _meta(out)
    kc = meta['kiem_cap']
    assert kc['ket_luan'] == 'chưa kết luận'
    assert kc['n_cap_can'] == 72 and kc['n_cap_thieu'] == 1 and kc['n_cap_du'] == 71
    assert kc['khoa_thieu'] == [sens._ks(X_SO)] and kc['cap_thieu'] == [sens._ks(X_SO)]
    assert kc['n_moc_thieu'] == 0 and kc['n_khoa_can'] == 81 and kc['n_khoa_co'] == 80
    assert meta['dung_lam_luot_xac_nhan'] is False
    assert len(_read_csv(out / 'bang.csv')) == 80
    tt = [r for r in _read_csv(out / 'tom_tat.csv')
          if (r['io_delay_ms'], r['slip_omega_s'], r['dap_ung']) == ('1', '0.05', 'e_perp_max')]
    assert len(tt) == 1 and tt[0]['n_dieu_kien'] == '8'
    assert len(_read_csv(out / 'tom_tat_theo_dieu_kien.csv')) == 80 * len(sens.RESPONSES)


def test_c3_19_missing_reference(tmp_path, monkeypatch):
    """A reference run missing: the 8 pairs of its condition are missing; still no crash
    (old code: assert 'missing reference runs', nothing written)."""
    out = _main_fakes(monkeypatch, tmp_path, pool=_drop_pool({X_MOC}))
    kc = _meta(out)['kiem_cap']
    assert kc['ket_luan'] == 'chưa kết luận'
    assert kc['moc_thieu'] == [sens._ks(X_MOC)] and kc['n_cap_thieu'] == 8
    assert all(';mu=0.436;v=0.5' in k for k in kc['cap_thieu'])
    dk = _read_csv(out / 'tom_tat_theo_dieu_kien.csv')
    miss = [r for r in dk if (r['mu'], r['v']) == ('0.436', '0.5')]
    assert miss and all(r['hieu'] == '' and r['gia_tri_moc'] == '' for r in miss)


def test_c3_19_run_error_rerun_ok(tmp_path, monkeypatch):
    """A run_error is run once more; the rerun succeeds: bang.csv holds the rerun row,
    chay_lai.csv both attempts, meta.json 'lỗi chưa rõ nguyên nhân', grid complete.
    Old code: no rerun (81 calls), the run_error row stays, no kiem_cap."""
    calls = []

    def job(j):
        calls.append(tuple(j))
        return _err_row(j) if tuple(j) == X_SO and calls.count(X_SO) == 1 else _fake_job(j)

    out = _main_fakes(monkeypatch, tmp_path, job=job)
    assert len(calls) == 82 and calls.count(X_SO) == 2
    meta = _meta(out)
    assert meta['kiem_cap']['ket_luan'] == 'đủ cặp' and meta['kiem_cap']['n_cap_thieu'] == 0
    assert meta['n_run_error_lan_1'] == 1
    assert meta['loi_chua_ro_nguyen_nhan'] == [dict(
        khoa=sens._ks(X_SO), error_lan_1='RuntimeError: boom', status_lan_2='ok',
        ghi_chu='lỗi chưa rõ nguyên nhân')]
    assert meta['dung_lam_luot_xac_nhan'] is True and meta['ly_do_khong_dung'] == []
    bang = _read_csv(out / 'bang.csv')
    assert len(bang) == 81 and all(r['status'] == 'ok' for r in bang)
    cl = _read_csv(out / 'chay_lai.csv')
    assert [(r['attempt'], r['status']) for r in cl] == [('1', 'run_error'), ('2', 'ok')]
    assert all((r['io_delay_ms'], r['slip_omega_s'], r['mu'], r['v']) == ('1', '0.05', '0.413', '0.2')
               for r in cl)


def test_c3_19_run_error_twice(tmp_path, monkeypatch):
    """A reference run fails twice: rerun exactly once, the row stays run_error, the 8 pairs
    of its condition are missing, 'chưa kết luận'; old code: no kiem_cap."""
    calls = []

    def job(j):
        calls.append(tuple(j))
        return _err_row(j) if tuple(j) == X_MOC else _fake_job(j)

    out = _main_fakes(monkeypatch, tmp_path, job=job)
    assert calls.count(X_MOC) == 2 and len(calls) == 82
    meta = _meta(out)
    kc = meta['kiem_cap']
    assert kc['ket_luan'] == 'chưa kết luận'
    assert kc['khoa_run_error'] == [sens._ks(X_MOC)] and kc['moc_thieu'] == [sens._ks(X_MOC)]
    assert kc['n_cap_thieu'] == 8 and kc['khoa_thieu'] == []
    assert meta['loi_chua_ro_nguyen_nhan'] == [] and meta['dung_lam_luot_xac_nhan'] is False
    cl = _read_csv(out / 'chay_lai.csv')
    assert [(r['attempt'], r['status']) for r in cl] == [('1', 'run_error'), ('2', 'run_error')]


def test_c3_19_completeness_dup_and_foreign():
    """completeness: full grid 'đủ cặp'; a duplicated key or a key outside the grid is
    'chưa kết luận' even when every pair exists."""
    full = [_fake_job(j) for j in sens.make_jobs(('ADRC',))]
    kc = sens.completeness(full, ('ADRC',))
    assert kc['ket_luan'] == 'đủ cặp' and (kc['n_cap_can'], kc['n_moc_can']) == (72, 9)
    kc = sens.completeness(full + [dict(full[5])], ('ADRC',))
    assert kc['ket_luan'] == 'chưa kết luận' and kc['khoa_trung'] == [sens._ks(sens.key_of(full[5]))]
    kc = sens.completeness(full + [_fake_job(('ADRC', 3, 0.1, 0.413, 0.2))], ('ADRC',))
    assert kc['ket_luan'] == 'chưa kết luận' and len(kc['khoa_la']) == 1


def test_c3_20_sidecar_without_sha(tmp_path):
    """C3-20 (Codex): {"esp32": {"dirty": false}} with no dll_sha256 is not tied to the DLL:
    unverified, esp32_dirty None, a warning (old code: esp32_dirty False, no warning)."""
    dll, _ = _mk_dll(tmp_path, build=dict(esp32=dict(dirty=False)))
    info = sens.dll_info(dll)
    assert info['esp32_dirty'] is None
    assert any('no dll_sha256' in w for w in info['warnings'])
    assert info['build_verified'] is False and info['nguon_build'] == 'chưa xác minh'
    assert info['fwsim_build'] == dict(esp32=dict(dirty=False))       # recorded as read


def test_c3_20_sidecar_wrong_sha(tmp_path):
    """C3-20 (Codex): a sidecar whose dll_sha256 differs: unverified, its esp32.dirty is not
    a result (old code: esp32_dirty False taken from that json)."""
    dll, _ = _mk_dll(tmp_path, build=dict(esp32=dict(dirty=False), dll_sha256='0' * 64))
    info = sens.dll_info(dll)
    assert info['esp32_dirty'] is None
    assert info['build_verified'] is False and info['nguon_build'] == 'chưa xác minh'
    (tmp_path / 'ok').mkdir()
    dll2, _ = _mk_dll(tmp_path / 'ok', build=dict(
        esp32=dict(commit='5bde4c5', dirty=False),
        dll_sha256=hashlib.sha256(b'fake-dll').hexdigest()))
    info2 = sens.dll_info(dll2)
    assert info2['build_verified'] is True and info2['nguon_build'] == 'đã xác minh'
    assert info2['esp32_dirty'] is False


@pytest.mark.parametrize('build', ('nosha', 'badsha'))
def test_c3_20_main_unverified_not_confirmation(build, tmp_path, monkeypatch):
    """main on an unverified build: meta.json nguon_build 'chưa xác minh', esp32_dirty None,
    the run is not a confirmation run before S03 main even with every pair present."""
    meta = _meta(_main_fakes(monkeypatch, tmp_path, build=build))
    assert meta['esp32_dirty'] is None
    assert meta['nguon_build'] == 'chưa xác minh' and meta['dll']['build_verified'] is False
    assert meta['kiem_cap']['ket_luan'] == 'đủ cặp'
    assert meta['dung_lam_luot_xac_nhan'] is False
    assert meta['ly_do_khong_dung'] == ['nguon build chua xac minh (C3-20)']


def _pair_rows(g):
    """One reference row (e_perp_max 0.0) and one comparison row (e_perp_max g), same cond."""
    base = dict(controller='ADRC', mu=0.413, v=0.2, status='ok', completed='true',
                e_perp_max_cens=0.0, tv_duty=1.0, sat_frac=0.0)
    ref = dict(base, io_delay_ms=0, slip_omega_s=0.1, la_moc=True, e_perp_max=0.0)
    so = dict(base, io_delay_ms=1, slip_omega_s=0.05, la_moc=False, e_perp_max=g)
    return [ref, so], {('ADRC', 0.413, 0.2): ref}


@pytest.mark.parametrize('g, want', [(0.009, True), (0.008, False), (0.008 + 1e-9, True),
                                     (0.007, False)])
def test_c3_21_threshold_from_s03(g, want, monkeypatch):
    """C3-21 (Codex): the threshold is s03_doe.THRESH at run time. With the Y2 threshold
    locked at 0.008 a 0.009 m difference exceeds it (old code: THRESH_C2 copy 0.01, False);
    equal to the threshold does not exceed, 1e-9 above does."""
    monkeypatch.setitem(sens.s03.THRESH, 'e_perp_max', 0.008)
    monkeypatch.setitem(sens.s03.THRESH, 'tv_duty', 0.4)
    rows, refs = _pair_rows(g)
    dk = [r for r in sens.per_condition_rows(rows, refs)
          if r['dap_ung'] == 'e_perp_max' and not r['la_moc']]
    assert len(dk) == 1 and dk[0]['vuot_nguong'] is want
    assert dk[0]['nguong'] == 0.008
    tt = [r for r in sens.summary_rows(rows, refs, ('ADRC',))
          if r['dap_ung'] == 'e_perp_max' and (r['io_delay_ms'], r['slip_omega_s']) == (1, 0.05)]
    assert len(tt) == 1 and tt[0]['vuot_nguong'] is want and tt[0]['nguong'] == 0.008
    assert sens.thresholds()['e_perp_max'] == 0.008


def test_c3_21_meta_records_thresholds(tmp_path, monkeypatch):
    """meta.json thresholds = the values used (read from s03_doe.THRESH), with the source."""
    monkeypatch.setitem(sens.s03.THRESH, 'e_perp_max', 0.008)
    meta = _meta(_main_fakes(monkeypatch, tmp_path))
    assert meta['thresholds'] == {'e_perp_max': 0.008, 'e_perp_max_cens': 0.01,
                                  'tv_duty': 0.4, 'sat_frac': 0.01}
    assert 's03_doe.py THRESH' in meta['thresholds_source']
    tt = _read_csv(tmp_path / 'out' / 'tom_tat.csv')
    assert {r['nguong'] for r in tt if r['dap_ung'] == 'e_perp_max'} == {'0.008'}


def test_c3_21_refuse_without_tv_threshold(tmp_path, monkeypatch):
    """s03_doe.THRESH without tv_duty (no locked absolute Y3 threshold): main refuses before
    any run or output folder; no private copy is used."""
    monkeypatch.delitem(sens.s03.THRESH, 'tv_duty', raising=False)
    with pytest.raises(SystemExit, match='tv_duty'):
        sens.thresholds()
    out = tmp_path / 'out'
    with pytest.raises(SystemExit, match='tv_duty'):
        sens.main(['--out', str(out), '--dll', str(tmp_path / 'fwsim.dll'), '--ctrl', 'ADRC'])
    assert not out.exists()


# ── review round 2 (C3-19/C3-29, C3-20) ─────────────────────────────────────────────

Y3_DONG = 'chua ket luan: thieu cap hay loi (C3-19)'
ESP_DONG = 'nguon ESP32 chua du: esp32.commit, esp32.dirty (C3-20)'


def _job_with(key, **fields):
    """_fake_job, with fields overridden in the row of key (status stays 'ok')."""
    def job(j):
        r = _fake_job(j)
        if tuple(j) == key:
            r.update(fields)
        return r
    return job


@pytest.mark.parametrize('bad', [float('nan'), '', None, float('inf'), float('-inf'), 'abc',
                                 np.float32('nan')])
def test_c3_29_main_y3_bad_in_ok_run(bad, tmp_path, monkeypatch):
    """C3-29 (Codex): all 81 keys present, one comparison run status 'ok' with tv_duty NaN or
    empty. Old code: kiem_cap 'đủ cặp' with 72 pairs and dung_lam_luot_xac_nhan true while
    tom_tat holds 8 conditions there. Now 'chưa kết luận', the pair and the field named."""
    out = _main_fakes(monkeypatch, tmp_path, job=_job_with(X_SO, tv_duty=bad))
    meta = _meta(out)
    kc = meta['kiem_cap']
    assert kc['n_khoa_can'] == 81 and kc['n_khoa_co'] == 81 and kc['khoa_thieu'] == []
    assert kc['khoa_run_error'] == [] and kc['n_cap_thieu'] == 0
    assert kc['ket_luan'] == 'chưa kết luận'
    assert kc['n_cap_du'] == 71 and kc['n_cap_khong_so_duoc'] == 1
    assert kc['cap_khong_so_duoc'] == [sens._ks(X_SO)]
    assert kc['khoa_thieu_dai_luong'] == [dict(khoa=sens._ks(X_SO), truong=['tv_duty'])]
    assert meta['dung_lam_luot_xac_nhan'] is False and meta['ly_do_khong_dung'] == [Y3_DONG]
    bang = _read_csv(out / 'bang.csv')
    assert len(bang) == 81 and all(r['status'] == 'ok' for r in bang)
    tt = [r for r in _read_csv(out / 'tom_tat.csv')
          if (r['io_delay_ms'], r['slip_omega_s'], r['dap_ung']) == ('1', '0.05', 'tv_duty')]
    assert len(tt) == 1 and tt[0]['n_dieu_kien'] == '8'


def test_c3_29_main_y3_nan_reference(tmp_path, monkeypatch):
    """tv_duty NaN in a reference run: its 8 pairs are not comparable, the run is named once."""
    out = _main_fakes(monkeypatch, tmp_path, job=_job_with(X_MOC, tv_duty=float('nan')))
    kc = _meta(out)['kiem_cap']
    assert kc['ket_luan'] == 'chưa kết luận' and kc['n_moc_thieu'] == 0
    assert kc['n_cap_khong_so_duoc'] == 8 and kc['n_cap_du'] == 64
    assert all(';mu=0.436;v=0.5' in k for k in kc['cap_khong_so_duoc'])
    assert kc['khoa_thieu_dai_luong'] == [dict(khoa=sens._ks(X_MOC), truong=['tv_duty'])]


@pytest.mark.parametrize('fields, truong', [
    (dict(completed=''), ['completed']),
    (dict(completed='True'), ['completed']),
    (dict(gate_outcome=''), ['gate_outcome']),
    (dict(gate_outcome='pass'), ['gate_outcome']),
    (dict(e_perp_max=float('nan')), ['e_perp_max']),
    (dict(e_perp_max=float('inf'), tv_duty=''), ['e_perp_max', 'tv_duty']),
])
def test_c3_29_main_each_quantity(fields, truong, tmp_path, monkeypatch):
    """Each quantity of a pair (Y1, gate outcome, finite Y2, Y3) missing in one 'ok' run:
    'chưa kết luận', the fields listed."""
    meta = _meta(_main_fakes(monkeypatch, tmp_path, job=_job_with(X_SO, **fields)))
    kc = meta['kiem_cap']
    assert kc['ket_luan'] == 'chưa kết luận' and kc['n_cap_khong_so_duoc'] == 1
    assert kc['khoa_thieu_dai_luong'] == [dict(khoa=sens._ks(X_SO), truong=truong)]
    assert meta['dung_lam_luot_xac_nhan'] is False


@pytest.mark.parametrize('fields', [
    dict(tv_duty=0.0), dict(tv_duty=5e-324), dict(tv_duty=1.7976931348623157e308),
    dict(tv_duty=np.float32(3.4028235e38)), dict(tv_duty=np.float32(0.4)),
    dict(tv_duty=0.4 + 1e-9), dict(tv_duty='1.25'),
    dict(completed='false'), dict(completed='inconclusive'),
    dict(gate_outcome='khong_dat'), dict(gate_outcome='chua_ket_luan'),
    dict(e_perp_max_cens=float('nan')), dict(sat_frac=float('nan'))])
def test_c3_29_comparable_edges(fields, tmp_path, monkeypatch):
    """Edge values that still compare: finite extremes (0, smallest subnormal, largest double,
    largest float32), values at / 1e-9 above the Y3 threshold, a numeric string, every Y1 and
    gate value; NaN in e_perp_max_cens or sat_frac (not a quantity of the pair). 'đủ cặp'."""
    meta = _meta(_main_fakes(monkeypatch, tmp_path, job=_job_with(X_SO, **fields)))
    kc = meta['kiem_cap']
    assert kc['ket_luan'] == 'đủ cặp' and kc['n_cap_du'] == 72
    assert kc['cap_khong_so_duoc'] == [] and kc['khoa_thieu_dai_luong'] == []
    assert meta['dung_lam_luot_xac_nhan'] is True and meta['ly_do_khong_dung'] == []


def test_c3_29_missing_quantities_unit():
    """missing_quantities on a full row, a run_error row and float32 NaN / inf."""
    r = _fake_job(X_SO)
    assert sens.missing_quantities(r) == []
    assert sens.missing_quantities(_err_row(X_SO)) == ['completed', 'gate_outcome',
                                                       'e_perp_max', 'tv_duty']
    assert sens.missing_quantities(dict(r, tv_duty=np.float32('inf'))) == ['tv_duty']
    assert sens.missing_quantities(dict(r, e_perp_max=np.float32('nan'))) == ['e_perp_max']


def _raise_on(key, exc=RuntimeError('boom')):
    def job(j):
        if tuple(j) == key:
            raise exc
        return _fake_job(j)
    return job


class _RaisingIter:
    """IMapIterator stand-in: next() raises exc in place of the result of a job in bad (as
    multiprocessing re-raises an unpicklable result), the other results still follow (CPython
    IMapIterator goes on after a raised item)."""

    def __init__(self, func, jobs, bad, exc):
        self.func, self.jobs, self.bad, self.exc = func, iter(jobs), bad, exc

    def __iter__(self):
        return self

    def __next__(self):
        j = next(self.jobs)
        if tuple(j) in self.bad:
            raise self.exc
        return self.func(j)


def _raising_pool(bad, exc=ValueError('result not picklable')):
    class P(_FakePool):
        def imap_unordered(self, func, jobs):
            return _RaisingIter(func, jobs, bad, exc)
    return P


def test_c3_19_worker_exception_recorded(tmp_path, monkeypatch, capsys):
    """C3-19 (Codex): _job raises in the worker for one key. Old code: the exception left
    main through the pool iterator, no meta.json. Now every file is written, meta.json
    loi_worker holds the key and the error, 'chưa kết luận', not a confirmation run."""
    out = _main_fakes(monkeypatch, tmp_path, job=_raise_on(X_SO))
    meta = _meta(out)
    assert meta['loi_worker'] == [dict(khoa=sens._ks(X_SO), lan=1, error='RuntimeError: boom')]
    kc = meta['kiem_cap']
    assert kc['ket_luan'] == 'chưa kết luận'
    assert kc['n_loi_worker'] == 1 and kc['khoa_loi_worker'] == [sens._ks(X_SO)]
    assert kc['khoa_thieu'] == [sens._ks(X_SO)] and kc['cap_thieu'] == [sens._ks(X_SO)]
    assert meta['dung_lam_luot_xac_nhan'] is False and meta['ly_do_khong_dung'] == [Y3_DONG]
    assert meta['n_run_error_lan_1'] == 0                 # a worker exception is not rerun
    assert len(_read_csv(out / 'bang.csv')) == 80
    for f in ('tom_tat.csv', 'tom_tat_theo_dieu_kien.csv', 'chay_lai.csv'):
        assert (out / f).exists()
    assert 'worker exceptions' in capsys.readouterr().out


def test_c3_19_worker_exception_reference(tmp_path, monkeypatch):
    """A reference run raising: its 8 pairs are missing, recorded, no crash."""
    kc = _meta(_main_fakes(monkeypatch, tmp_path, job=_raise_on(X_MOC)))['kiem_cap']
    assert kc['ket_luan'] == 'chưa kết luận' and kc['moc_thieu'] == [sens._ks(X_MOC)]
    assert kc['n_cap_thieu'] == 8 and kc['khoa_loi_worker'] == [sens._ks(X_MOC)]


def test_c3_19_iterator_exception_recorded(tmp_path, monkeypatch):
    """The pool iterator itself raises (no key known): the remaining results are still read,
    meta.json loi_worker holds the error with an empty key, the key is missing in kiem_cap."""
    out = _main_fakes(monkeypatch, tmp_path, pool=_raising_pool({X_SO}))
    meta = _meta(out)
    assert meta['loi_worker'] == [dict(khoa='', lan=1, error='ValueError: result not picklable')]
    kc = meta['kiem_cap']
    assert kc['ket_luan'] == 'chưa kết luận' and kc['n_loi_worker'] == 1
    assert kc['khoa_loi_worker'] == [] and kc['khoa_thieu'] == [sens._ks(X_SO)]
    assert kc['n_khoa_co'] == 80 and len(_read_csv(out / 'bang.csv')) == 80
    assert meta['dung_lam_luot_xac_nhan'] is False


def test_c3_19_rerun_exception(tmp_path, monkeypatch):
    """A run_error whose rerun raises: lan 2 in loi_worker, the run_error row stays,
    'chưa kết luận'."""
    calls = []

    def job(j):
        calls.append(tuple(j))
        if tuple(j) == X_SO:
            if calls.count(X_SO) == 1:
                return _err_row(j)
            raise OSError('worker died')
        return _fake_job(j)

    meta = _meta(_main_fakes(monkeypatch, tmp_path, job=job))
    assert calls.count(X_SO) == 2 and meta['n_run_error_lan_1'] == 1
    assert meta['loi_worker'] == [dict(khoa=sens._ks(X_SO), lan=2, error='OSError: worker died')]
    kc = meta['kiem_cap']
    assert kc['ket_luan'] == 'chưa kết luận' and kc['khoa_run_error'] == [sens._ks(X_SO)]
    assert meta['dung_lam_luot_xac_nhan'] is False


def test_c3_19_no_exception_meta_empty(tmp_path, monkeypatch):
    """Nothing raises: loi_worker [] and kiem_cap n_loi_worker 0 (the new fields always present)."""
    meta = _meta(_main_fakes(monkeypatch, tmp_path))
    assert meta['loi_worker'] == [] and meta['kiem_cap']['n_loi_worker'] == 0
    assert meta['kiem_cap']['ket_luan'] == 'đủ cặp' and meta['dung_lam_luot_xac_nhan'] is True


def test_c3_20_main_sidecar_sha_only(tmp_path, monkeypatch, capsys):
    """C3-20 (Codex round 2): sidecar {"dll_sha256": <right sha>} only. Old code:
    nguon_build 'đã xác minh', dung_lam_luot_xac_nhan true. Now the two states are apart:
    sidecar_khop_sha true, nguon_esp32_du false, not a confirmation run, a warning."""
    meta = _meta(_main_fakes(monkeypatch, tmp_path, build='shaonly'))
    assert meta['kiem_cap']['ket_luan'] == 'đủ cặp'
    assert meta['nguon_build'] != 'đã xác minh'                       # old code: 'đã xác minh'
    assert meta['dung_lam_luot_xac_nhan'] is False                    # old code: True
    assert meta['ly_do_khong_dung'] == [ESP_DONG]
    assert meta['nguon_build'] == 'khớp SHA, thiếu nguồn ESP32'
    d = meta['dll']
    assert d['sidecar_khop_sha'] is True and d['build_verified'] is True
    assert d['nguon_esp32_du'] is False and d['esp32_commit'] is None
    assert any('ESP32 source' in w for w in meta['warnings'])
    assert 'ESP32 source' in capsys.readouterr().out


@pytest.mark.parametrize('esp32, du', [
    (dict(commit='5bde4c5', dirty=False), True),
    (dict(commit='5bde4c5', dirty=True), True),          # dirty does not block (no rule decided)
    (dict(dirty=False), False),
    (dict(commit='', dirty=False), False),
    (dict(commit='  ', dirty=False), False),
    (dict(commit='UNKNOWN', dirty=True), False),          # build_fwsim.git_state on a git error
    (dict(commit=None, dirty=False), False),
    (dict(commit='5bde4c5'), False),
    (dict(commit='5bde4c5', dirty='false'), False),       # not a boolean
    (dict(commit='5bde4c5', dirty=None), False),
    (None, False),
])
def test_c3_20_esp32_source(esp32, du, tmp_path):
    """dll_info with a sidecar tied to the DLL: nguon_esp32_du needs esp32.commit (non-empty,
    not 'UNKNOWN') and a boolean esp32.dirty; nguon_build 'đã xác minh' only then."""
    b = dict(dll_sha256=_SHA)
    if esp32 is not None:
        b['esp32'] = esp32
    dll, _ = _mk_dll(tmp_path, build=b)
    info = sens.dll_info(dll)
    assert info['sidecar_khop_sha'] is True
    assert info['nguon_esp32_du'] is du
    assert info['nguon_build'] == ('đã xác minh' if du else 'khớp SHA, thiếu nguồn ESP32')
    assert info['esp32_commit'] == ('5bde4c5' if (esp32 or {}).get('commit') == '5bde4c5'
                                    else None)
    assert any('ESP32 source' in w for w in info['warnings']) is (not du)


def test_c3_20_esp32_commit_wrong_sha(tmp_path):
    """A sidecar with the ESP32 commit but a wrong dll_sha256: no state taken from it."""
    dll, _ = _mk_dll(tmp_path, build=dict(esp32=dict(commit='5bde4c5', dirty=False),
                                          dll_sha256='0' * 64))
    info = sens.dll_info(dll)
    assert info['sidecar_khop_sha'] is False and info['nguon_esp32_du'] is False
    assert info['esp32_commit'] is None and info['nguon_build'] == 'chưa xác minh'


def test_c3_20_main_dirty_with_commit_confirms(tmp_path, monkeypatch):
    """SHA match, ESP32 commit, dirty true: still a confirmation run (no 'not dirty' rule is
    added); the dirty warning stays."""
    meta = _meta(_main_fakes(monkeypatch, tmp_path, build=dict(
        esp32=dict(commit='5bde4c5', dirty=True), dll_sha256=_SHA)))
    assert meta['nguon_build'] == 'đã xác minh' and meta['esp32_dirty'] is True
    assert meta['dung_lam_luot_xac_nhan'] is True and meta['ly_do_khong_dung'] == []
    assert any('dirty' in w for w in meta['warnings'])


def test_c3_20_main_both_reasons(tmp_path, monkeypatch):
    """SHA-only sidecar and a non-comparable pair: both reasons listed."""
    meta = _meta(_main_fakes(monkeypatch, tmp_path, job=_job_with(X_SO, tv_duty=''),
                             build='shaonly'))
    assert meta['dung_lam_luot_xac_nhan'] is False
    assert meta['ly_do_khong_dung'] == [ESP_DONG, Y3_DONG]
