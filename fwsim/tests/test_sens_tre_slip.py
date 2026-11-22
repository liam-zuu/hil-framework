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
    """A clean build json that matches the DLL: no warning."""
    dll, data = _mk_dll(tmp_path, build=dict(
        esp32=dict(dirty=False), hil_framework=dict(dirty=False),
        dll_sha256=hashlib.sha256(b'fake-dll').hexdigest()))
    info = sens.dll_info(dll)
    assert info['warnings'] == []
    assert info['esp32_dirty'] is False
    assert info['fwsim_build']['esp32']['dirty'] is False


def test_dll_info_esp32_dirty(tmp_path):
    """esp32.dirty true: a warning, esp32_dirty True."""
    dll, _ = _mk_dll(tmp_path, build=dict(esp32=dict(dirty=True),
                                          hil_framework=dict(dirty=False)))
    info = sens.dll_info(dll)
    assert info['esp32_dirty'] is True
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
                gate_outcome='pass', error='', t_chay_s=0.0)


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
