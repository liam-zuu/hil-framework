"""s02_compare.py (CODE_S02 design v3 section 22, spec S02 v1.13 P2-6): Welch and one-sample intervals against
scipy, cells with fewer than 3 values, delta %, the encoder resolution label, rows not ok ignored."""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest
from scipy import stats as sst

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import s02_compare as C  # noqa: E402
import s02_metrics as M  # noqa: E402

BASE = ['run_id', 'platform', 'controller', 'test_id', 'speed_mps', 'rep', 'status', 'status_reason']


def write(d, rows, extra=()):
    os.makedirs(d, exist_ok=True)
    keys = BASE + [k for k in rows[0] if k not in BASE]
    with open(os.path.join(d, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
    with open(os.path.join(d, 'metrics_columns.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(columns={k: M.column_info(k) for k in keys}), f)
    return d


def runs(platform, ctrl, vals, metric='e_p_rms', test='A-L2', status='ok', **extra):
    return [dict(run_id=f'{platform}{i}', platform=platform, controller=ctrl, test_id=test, speed_mps='0.5',
                 rep=str(i + 1), status=status, status_reason='', **{metric: repr(v)}, **extra)
            for i, v in enumerate(vals)]


def line(lines, metric, pa, pb, test='A-L2'):
    return next(x for x in lines if x['metric'] == metric and x['platform_a'] == pa and x['platform_b'] == pb
                and x['test_id'] == test)


def test_welch_matches_scipy(tmp_path):
    a, b = [0.010, 0.012, 0.011, 0.013, 0.009], [0.020, 0.018, 0.025, 0.022, 0.019]
    rows, info = C.load([write(str(tmp_path / 's'), runs('sim', 'ADRC', a)),
                         write(str(tmp_path / 'h'), runs('HIL', 'LADRC', b))])
    x = line(C.compare(rows, info), 'e_p_rms', 'sim', 'hil')
    ci = sst.ttest_ind(b, a, equal_var=False).confidence_interval(0.95)
    assert x['method'] == 'welch' and x['controller'] == 'LADRC'
    assert x['delta'] == pytest.approx(np.mean(b) - np.mean(a))
    assert (x['ci_low'], x['ci_high']) == pytest.approx((ci.low, ci.high), rel=1e-9)
    assert x['delta_pct'] == pytest.approx((np.mean(b) - np.mean(a)) / np.mean(a) * 100)
    assert x['n_a'] == 5 and x['n_b'] == 5 and x['label'] == ''


def test_one_sample_when_sim_repeats_bit_for_bit(tmp_path):
    a, b = [0.010] * 5, [0.020, 0.018, 0.025, 0.022, 0.019]
    rows, info = C.load([write(str(tmp_path / 's'), runs('sim', 'PID', a)),
                         write(str(tmp_path / 'h'), runs('hil', 'PID', b))])
    x = line(C.compare(rows, info), 'e_p_rms', 'sim', 'hil')
    ci = sst.ttest_1samp(b, 0.010).confidence_interval(0.95)
    assert x['method'] == 'one_sample' and x['df'] == 4
    assert (x['ci_low'], x['ci_high']) == pytest.approx((ci.low - 0.010, ci.high - 0.010), rel=1e-9)


def test_fewer_than_three_values_and_undefined_values(tmp_path):
    rows_h = runs('hil', 'PID', [0.2, 0.21, float('nan'), float('nan'), 0.19], metric='rise_s_seg2', test='A-K1')
    rows_s = runs('sim', 'PID', [0.15, 0.16, 0.15], metric='rise_s_seg2', test='A-K1')
    rows, info = C.load([write(str(tmp_path / 's'), rows_s), write(str(tmp_path / 'h'), rows_h)])
    x = line(C.compare(rows, info), 'rise_s_seg2', 'sim', 'hil', 'A-K1')
    assert x['n_b'] == 3 and x['n_undef_b'] == 2 and x['label'] == ''
    rows_h[0]['rise_s_seg2'] = 'nan'
    rows, info = C.load([write(str(tmp_path / 's2'), rows_s), write(str(tmp_path / 'h2'), rows_h)])
    x = line(C.compare(rows, info), 'rise_s_seg2', 'sim', 'hil', 'A-K1')
    assert x['label'] == 'insufficient' and 'delta' not in x and x['n_b'] == 2


def test_rows_not_ok_are_ignored(tmp_path):
    h = runs('hil', 'PID', [0.02, 0.021, 0.019]) + runs('hil', 'PID', [5.0, 5.0], status='aborted')
    rows, info = C.load([write(str(tmp_path / 's'), runs('sim', 'PID', [0.01, 0.011, 0.012])),
                         write(str(tmp_path / 'h'), h)])
    x = line(C.compare(rows, info), 'e_p_rms', 'sim', 'hil')
    assert x['n_b'] == 3 and x['mean_b'] == pytest.approx(0.02)


def test_encoder_label_and_no_delta_pct_under_two_qrms(tmp_path):
    q = M.QRMS['e_vx']
    a = [1.5 * q, 1.5 * q + 1e-5, 1.5 * q - 1e-5]
    b = [1.5 * q + 2e-5, 1.5 * q + 3e-5, 1.5 * q + 1e-5]
    rows, info = C.load([write(str(tmp_path / 's'), runs('sim', 'PID', a, metric='e_vx')),
                         write(str(tmp_path / 'h'), runs('hil', 'PID', b, metric='e_vx'))])
    x = line(C.compare(rows, info), 'e_vx', 'sim', 'hil')
    assert x['label'] == 'below_encoder_resolution' and 'delta_pct' not in x
    # e_wz is an encoder metric only when every run used the encoder source
    qz = M.QRMS['e_wz']
    za, zb = [3 * qz, 3 * qz + 1e-5, 3 * qz - 1e-5], [3 * qz + 1e-5, 3 * qz, 3 * qz + 2e-5]
    for frac, want in (('1', 'below_encoder_resolution'), ('0.5', '')):
        d = tmp_path / f'z{frac}'
        rows, info = C.load([write(str(d / 's'), runs('sim', 'PID', za, metric='e_wz', wz_src_enc_frac=frac)),
                             write(str(d / 'h'), runs('hil', 'PID', zb, metric='e_wz', wz_src_enc_frac=frac))])
        assert line(C.compare(rows, info), 'e_wz', 'sim', 'hil')['label'] == want


def test_flags_are_not_compared_and_pairs(tmp_path):
    s = runs('sim', 'PID', [0.1, 0.11, 0.12], metric='e_vx', no90_seg2='0')
    h = runs('hil', 'PID', [0.1, 0.12, 0.13], metric='e_vx', no90_seg2='1')
    r = runs('real', 'PID', [0.2, 0.22, 0.21], metric='e_vx', no90_seg2='1')
    rows, info = C.load([write(str(tmp_path / d), x) for d, x in (('s', s), ('h', h), ('r', r))])
    lines = C.compare(rows, info)
    assert {(x['platform_a'], x['platform_b']) for x in lines} == {('sim', 'hil'), ('hil', 'real'), ('sim', 'real')}
    assert all(x['metric'] == 'e_vx' for x in lines)


def test_cli_never_overwrites(tmp_path):
    s = write(str(tmp_path / 's'), runs('sim', 'PID', [0.1, 0.11, 0.12]))
    h = write(str(tmp_path / 'h'), runs('hil', 'PID', [0.1, 0.12, 0.13]))
    assert C.main(['--out', str(tmp_path / 'o'), '--date', '2026-11-16', s, h]) == 0
    with open(tmp_path / 'o' / 'compare_2026-11-16.csv', newline='') as f:
        got = list(csv.DictReader(f))
    assert got[0]['metric'] == 'e_p_rms' and got[0]['unit'] == 'm' and got[0]['family'] == 'R'
    assert C.main(['--out', str(tmp_path / 'o'), '--date', '2026-11-16', s, h]) == 2


def test_old_metrics_without_status_refused(tmp_path):
    d = tmp_path / 'old'
    d.mkdir()
    (d / 'metrics.csv').write_text('run_id,platform,controller,test_id,e_vx\nr,sim,PID,A-L2,0.1\n', encoding='utf-8')
    with pytest.raises(ValueError, match='no status column'):
        C.load([str(d)])


def test_label_needs_the_whole_ci_inside_plus_minus_qrms(tmp_path):
    """A CI that only overlaps +-qrms (one end outside) gets no below-resolution label."""
    q = M.QRMS['e_vx']
    a = [10 * q, 10 * q + 1e-5, 10 * q - 1e-5]
    b = [10 * q + 0.8 * q, 10 * q + 0.9 * q, 10 * q + 1.1 * q]       # delta about 0.9 qrms, CI past qrms
    rows, info = C.load([write(str(tmp_path / 's'), runs('sim', 'PID', a, metric='e_vx')),
                         write(str(tmp_path / 'h'), runs('hil', 'PID', b, metric='e_vx'))])
    x = line(C.compare(rows, info), 'e_vx', 'sim', 'hil')
    assert x['ci_low'] < q < x['ci_high'] and x['label'] == ''
