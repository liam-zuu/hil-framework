"""Tests for tools/imu_capture.py --phases yawscale (Liam 11/11, guide 2026-11-11_Huong_dan_Buoi_xe_san_thang_momen_yaw.md).
The sensor here is FakeSource: invented, NOT a measurement."""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import imu_capture as ic  # noqa: E402


def test_unwrap_follows_turns_past_180():
    assert ic.unwrap_deg([170, -170, -10, 90, 170, -170]) == pytest.approx([170, 190, 350, 450, 530, 550])
    assert ic.unwrap_deg([-170, 170]) == pytest.approx([-170, -190])


@pytest.mark.parametrize('turns,scale', [(3, 1.0), (-3, 1.005), (5, 0.997), (-5, 1.0)])
def test_yaw_scale_check_recovers_an_invented_scale(turns, scale):
    src = ic.FakeSource(50000, seed=1)
    rows, _ = src.read(180 if abs(turns) == 3 else 45, turns=turns, scale=scale)
    c = ic.yaw_scale_check(rows, turns)
    assert c['ok'], c['warn']
    assert c['yaw_scale_err_pct'] == pytest.approx(100 * (scale - 1), abs=0.03)
    assert c['gz_scale_err_pct'] == pytest.approx(100 * (scale - 1), abs=0.1)     # gyro quantised to 1/512


def test_yaw_scale_check_warns_on_a_wrong_count_and_on_missing_reports():
    rows, _ = ic.FakeSource(50000).read(45, turns=4)
    c = ic.yaw_scale_check(rows, 5)                                    # one turn short
    assert not c['ok'] and any('far from 1800' in w for w in c['warn'])
    rows, _ = ic.FakeSource(50000).read(45, turns=5)
    holed = [r for r in rows if not (r[0] == 'G' and 300 <= r[2] < 303)]
    c = ic.yaw_scale_check(holed, 5)
    assert not c['ok'] and c['missing']['G'] == 3


def test_yaw_scale_check_warns_when_an_end_is_not_still():
    rows, _ = ic.FakeSource(50000).read(45, turns=5)
    t_end = rows[-1][1]
    moved = [r[:5] + [0.5] + r[6:] if r[0] == 'G' and t_end - r[1] < 1e6 else r for r in rows]
    c = ic.yaw_scale_check(moved, 5)
    assert 'not still at the end' in c['warn']


def test_dry_run_yawscale_writes_files_and_only_runs_the_chosen_phases(tmp_path):
    yaml = pytest.importorskip('yaml')
    out = tmp_path / 'ys'
    ic.main(['--dry-run', '--rate', '20', '--out', str(out), '--phases', 'yawscale', '--only', 'fast_ccw', 'fast_cw',
             '--dry-scale', '1.004'])
    assert sorted(os.listdir(out)) == ['meta.yaml', 'raw_fast_ccw.csv', 'raw_fast_cw.csv', 'summary.csv', 'yawscale.csv']
    ys = list(csv.DictReader(open(out / 'yawscale.csv')))
    assert [r['phase'] for r in ys] == ['fast_ccw', 'fast_cw'] and all(r['ok'] == 'True' for r in ys)
    assert float(ys[0]['yaw_scale_err_pct']) == pytest.approx(0.4, abs=0.03)
    m = yaml.safe_load(open(out / 'meta.yaml', encoding='utf-8'))
    assert m['dry_run'] is True and m['setup'] == 'real_agv_imu_on_turntable'
    assert m['firmware']['esp32']['pio_env'] == 'imu_log_20'


def test_only_refuses_unknown_phase(tmp_path):
    with pytest.raises(SystemExit):
        ic.main(['--dry-run', '--rate', '20', '--out', str(tmp_path / 'x'), '--phases', 'yawscale', '--only', 'static'])


def test_noise_set_is_unchanged():
    assert [(n, d) for n, d, _ in ic.PHASE_SETS['noise']] == [('static', 300), ('yaw90', 20)]
