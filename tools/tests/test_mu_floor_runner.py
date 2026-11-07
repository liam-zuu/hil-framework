"""Tests of tools/mu_floor_runner.py (no hardware). Run: python -m pytest tools/tests -q"""
import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import mu_floor_runner as mf  # noqa: E402


def test_mu_from_force_is_kgf_over_kg():
    assert mf.mu_from_force(2.4, 4.0) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        mf.mu_from_force(1.0, 0.0)
    with pytest.raises(ValueError):
        mf.mu_from_force(-1.0, 4.0)


def test_summarize_levels_are_min_mean_max_of_good_trials():
    tr = [dict(surface='floor', direction='XP', mu=m) for m in (0.5, 0.6, 0.7)] + \
         [dict(surface='floor', direction='YP', mu=m) for m in (0.4, None)] + \
         [dict(surface='mica', direction='XP', mu=0.2)]
    s = {(r['surface'], r['direction']): r for r in mf.summarize(tr)}
    assert s[('floor', 'all')]['n'] == 4                               # the bad (None) trial is left out
    assert (s[('floor', 'all')]['mu_min'], s[('floor', 'all')]['mu_max']) == (0.4, 0.7)
    assert s[('floor', 'all')]['mu_mean'] == pytest.approx((0.5 + 0.6 + 0.7 + 0.4) / 4)
    assert s[('floor', 'YP')]['n'] == 1 and s[('mica', 'all')]['mu_mean'] == 0.2


def test_parse_lines():
    d = mf.parse_data('D,120,101,3,-3,-3,3,0.0123,-0.0040,9.8061,0.00012,24')
    assert d == dict(t_ms=120, pwm=101, c_fl=3, c_fr=-3, c_rl=-3, c_rr=3, ax=0.0123, ay=-0.004, az=9.8061,
                     gz=0.00012, imu_n=24)
    assert mf.parse_data('D,1,2,3') is None and mf.parse_data('EV START t_ms=0') is None
    assert mf.parse_kv('DONE reason=slip t_ms=8340 pwm=183') == dict(reason='slip', t_ms='8340', pwm='183')
    assert mf.parse_kv('EV SLIP t_ms=8340 pwm=183 w=3.41') == dict(t_ms='8340', pwm='183', w='3.41')


def test_ask_force_accepts_comma_and_blank_is_bad():
    seq = iter(['abc', '-1', '2,35'])
    assert mf.ask_force('', lambda p: next(seq)) == pytest.approx(2.35)
    assert mf.ask_force('', lambda p: '') is None


def test_dry_run_writes_trials_summary_meta_and_resumes(tmp_path):
    yaml = pytest.importorskip('yaml')
    out = tmp_path / 'run'
    answers = iter(['', '2.4', '', ''] * 2)                            # Enter to start, F, ...; 2nd trial bad
    a = mf.argparse.Namespace(out=str(out), dirs=['XP'], reps=2, surface='floor', pwm0=100, rate=50,
                              pwm_max=1000, w_slip=3.0, mass_kg=4.0, dry_run=True, esp32_commit='', esp32_dirty=False,
                              gauge='test')
    dev = mf.FakeDevice(mass_kg=4.0, mu=0.6)
    os.makedirs(out)
    mf.write_meta(a, dev, os.path.dirname(mf.__file__))
    trials, summ = mf.run(a, dev, inp=lambda p: next(answers))
    rows = list(csv.DictReader(open(out / 'trials.csv', encoding='utf-8')))
    assert [r['trial'] for r in rows] == ['floor_XP_01', 'floor_XP_02']
    assert rows[0]['reason'] == 'slip' and float(rows[0]['mu']) == pytest.approx(0.6) and rows[1]['mu'] == ''
    raw = list(csv.DictReader(open(out / 'raw' / 'floor_XP_01.csv', encoding='utf-8')))
    assert raw and list(raw[0]) == mf.RAW_COLS
    assert yaml.safe_load(open(out / 'meta.yaml', encoding='utf-8'))['mass_kg'] == 4.0
    # resume: only the bad trial is run again
    seen = []
    again = iter(['', '2.0'])
    mf.run(a, dev, inp=lambda p: (seen.append(p), next(again))[1])
    assert any('floor_XP_02' in p for p in seen) and not any('floor_XP_01' in p for p in seen)
    rows = list(csv.DictReader(open(out / 'trials.csv', encoding='utf-8')))
    assert len(rows) == 2 and float(rows[1]['mu']) == pytest.approx(0.5)
    s = {r['direction']: r for r in csv.DictReader(open(out / 'summary.csv', encoding='utf-8'))}
    assert int(s['all']['n']) == 2 and float(s['all']['mu_mean']) == pytest.approx(0.55)


def test_main_needs_port_and_commit_unless_dry_run(tmp_path):
    with pytest.raises(SystemExit, match='--port and --esp32-commit'):
        mf.main(['--out', str(tmp_path / 'o'), '--mass-kg', '4'])
