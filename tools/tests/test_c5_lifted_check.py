"""tools/c5_lifted_check.py (C5, section 4 of the paper run): a synthetic log built from the
schedule passes; each criterion fails on its own mutation; the real HIL log of
CLOSED_LOOP/2026-11-09_run14 passes when it is on this machine."""
import csv
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
import c5_lifted_check as c5  # noqa: E402
from agv_logger import COLUMNS  # noqa: E402
from trajectory_pub import GROUP_A, get_body_velocity  # noqa: E402

SHA, ENV = '54b7269ce21d', 'esp32-s3-devkitc-1'
RPM_PER_MPS = 200.0


def _rows(test='A-K1', pre=1.0, tail=1.5):
    spec = GROUP_A[test]
    rows, seq = [], 100
    n = int(round((pre + spec.duration + tail) / 0.01))
    on0, on1 = int(round(0.5 / 0.01)), int(round((pre + spec.duration + 1.0) / 0.01))
    for k in range(n):
        t = k * 0.01
        u = t - pre
        sp = get_body_velocity(u, spec)[0] if 0.0 <= u < spec.duration else 0.0
        on = on0 <= k < on1
        r = {c: '0' for c in COLUMNS}
        r.update(timestamp_us=str(1_000_000 + k * 10_000), log_seq=str(seq), sp_vx=f'{sp:.6f}',
                 vx=f'{sp:.6f}', motor_on=str(int(on)), imu_init='0', wz_src='1' if on else '0',
                 run_active=str(int(on)), pos_reset_n='1' if k >= on0 else '0')   # counter only grows
        for w in c5.WHEELS:
            r[f'{w}_rpm'] = f'{sp * RPM_PER_MPS:.3f}'
        rows.append(r)
        seq += 1
    return rows, seq


def _write(tmp_path, rows, last_seq, summary=None, events=None, name='run.csv'):
    p = tmp_path / name
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    seg = dict(segment=0, data_lines=len(rows), seq_holes=0, duplicates=0, conflicts=0, reordered=0,
               run_end_groups=[dict(last_data_seq=last_seq - 1, footer_seqs=[last_seq], footers_missing=0,
                                    tail_data_missing=0)],
               counters=dict(log_qdrop=[0, 0]))
    s = dict(status='closed', segments=[seg])
    if summary:
        summary(s)
    (tmp_path / (name + '.summary.json')).write_text(json.dumps(s), encoding='utf-8')
    ev = events if events is not None else [f'2026-11-10T20:00:00 #fw,sha={SHA},dirty=0,env={ENV}'] * 3 + \
        ['2026-11-10T20:00:12 #run_end,seq=1']
    (tmp_path / (name + '.events.txt')).write_text('\n'.join(ev) + '\n', encoding='utf-8')
    return str(p)


def _run(path, test='A-K1', sha='54b7269', env=ENV):
    return c5.main(['--log', path, '--test', test, '--expect-sha', sha, '--expect-env', env])


@pytest.mark.parametrize('test', ['A-K1', 'A-K2'])
def test_clean_synthetic_log_passes(tmp_path, capsys, test):
    rows, last = _rows(test)
    assert _run(_write(tmp_path, rows, last), test) == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1].startswith('#ket_qua,ma=0')
    n_hold = len(c5.expected_holds(test))
    assert n_hold == (1 if test == 'A-K1' else 3)


def test_expected_holds_from_schedule():
    assert c5.expected_holds('A-K1') == [pytest.approx((3.0, 6.0, 0.3))]
    assert [h[2] for h in c5.expected_holds('A-K2')] == pytest.approx([0.3, -0.3, 0.3])


def _mutate(tmp_path, capsys, fn=None, summary=None, events=None, test='A-K1', **kw):
    rows, last = _rows(test)
    if fn:
        fn(rows)
    rc = _run(_write(tmp_path, rows, last, summary=summary, events=events), test, **kw)
    return rc, capsys.readouterr().out.strip().splitlines()[-1]


def _set(rows, t0, t1, **vals):
    for k, r in enumerate(rows):
        if t0 <= k * 0.01 < t1:
            r.update({c: str(v) for c, v in vals.items()})


@pytest.mark.parametrize('case, expect', [
    (dict(sha='408d44b'), 'fw'),
    (dict(env='ladrc'), 'fw'),
    (dict(events=['#fw,sha=54b7269ce21d,dirty=1,env=esp32-s3-devkitc-1', '#run_end,seq=1']), 'fw'),
    (dict(events=['#run_end,seq=1']), 'fw'),                                     # no #fw line
    (dict(summary=lambda s: s['segments'][0].update(seq_holes=2)), 'logger'),
    (dict(summary=lambda s: s.update(status='open')), 'logger'),
    (dict(summary=lambda s: s['segments'].append(dict(s['segments'][0]))), 'logger'),
    (dict(summary=lambda s: s['segments'][0]['run_end_groups'][0].update(tail_data_missing=1)), 'logger'),
    (dict(summary=lambda s: s['segments'][0]['counters'].update(log_qdrop=[0, 3])), 'logger'),
    (dict(fn=lambda r: _set(r, 6.0, 6.2, motor_on=0)), 'motor'),                 # second rise
    (dict(fn=lambda r: _set(r, 0.0, 99.0, imu_init=3)), 'imu'),
    (dict(fn=lambda r: _set(r, 2.0, 2.2, wz_src=0)), 'imu'),                     # 20 of ~1050 lines < 99 %
    (dict(fn=lambda r: _set(r, 5.0, 5.01, slip_flag=1)), 'imu'),
    (dict(fn=lambda r: _set(r, 5.0, 5.01, pos_reset_n=2)), 'pos'),
    (dict(fn=lambda r: _set(r, 6.2, 7.0, vx=0.32)), 'track'),                    # last 1 s off by 0.02
    (dict(fn=lambda r: _set(r, 4.0, 7.0, rr_rpm=-60)), 'track'),                 # one wheel the wrong way
    (dict(fn=lambda r: _set(r, 5.0, 5.05, sat_fl=1)), 'track'),
    (dict(fn=lambda r: _set(r, 4.0, 7.0, sp_vx=0.25, vx=0.25)), 'track'),       # not the schedule
])
def test_each_criterion_fails_on_its_mutation(tmp_path, capsys, case, expect):
    rc, last = _mutate(tmp_path, capsys, **case)
    assert rc == 1 and last.startswith('#ket_qua,ma=1')
    assert expect in last.split('khong_dat=')[1].split(',')


def test_small_wz_src_dropout_is_tolerated(tmp_path, capsys):
    rc, last = _mutate(tmp_path, capsys, fn=lambda r: _set(r, 2.0, 2.05, wz_src=0))   # 5 lines < 1 %
    assert rc == 0


def test_stop_reverse_spin_is_recorded_not_scored(tmp_path, capsys):
    rc, _ = _mutate(tmp_path, capsys, fn=lambda r: _set(r, 7.0, 7.5, fl_rpm=-25))
    assert rc == 0


def test_input_errors(tmp_path, capsys):
    assert _run(str(tmp_path / 'nope.csv')) == 2
    rows, last = _rows()
    p = _write(tmp_path, rows, last)
    os.remove(p + '.summary.json')
    assert _run(p) == 2
    with pytest.raises(SystemExit):
        c5.main(['--log', p, '--test', 'A-K3', '--expect-sha', 'x', '--expect-env', ENV])    # A-K3 not scored


REAL = r'D:\CLAUDE_THESIS\05_DATA\results\CLOSED_LOOP\2026-11-09_run14\pi\b1_AK1.csv'


@pytest.mark.skipif(not os.path.isfile(REAL), reason='HIL log of 09/11 not on this machine')
def test_real_hil_log_passes():
    assert c5.main(['--log', REAL, '--test', 'A-K1', '--expect-sha', 'b64095e', '--expect-env', 'wheels_lifted']) == 0
    assert c5.main(['--log', REAL, '--test', 'A-K1', '--expect-sha', '54b7269', '--expect-env', 'wheels_lifted']) == 1
