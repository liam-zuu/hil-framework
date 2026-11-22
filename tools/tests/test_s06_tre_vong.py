"""tools/s06_tre_vong.py on synthetic T3_result.csv / T6_follow.csv / summaries: known
stats (hand-computed median/p95 'higher'/max, max of a sum = max_a + max_b), flat
transitions excluded from the sums but reported, excluded verdicts counted, io floor/ceil
around the 1 ms grid, input errors, and the evidence gates (sync manual, limit 0.0055,
u_us). All files under tmp_path. Run: python -m pytest <this file>"""
import csv
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'rpi5')]
import s06_t6_check as t6m      # noqa: E402
import s06_t3_score as t3m      # noqa: E402
import s06_tre_vong as tv       # noqa: E402


def t3_row(vi='do_duoc', vii='', delay_us=None, quant_us=None):
    r = dict.fromkeys(t3m.COLUMNS)
    r.update(verdict_i=vi, verdict_ii=vii, delay_us=delay_us, quant_us=quant_us)
    return r


def t6_row(duty_to=0.7, leave_ms='', follow_ms='', settled=False, span=50.0):
    return dict(idx=0, wheel='FL', duty_from=0.3, duty_to=duty_to, t_cmd_us=0,
                t_mark_span_us=span, frame_leave='', frame_settle='',
                follow_ms=follow_ms, leave_ms=leave_ms, settled=settled)


def write_inputs(tmp, rows3=None, rows6=None):
    p3 = str(tmp / 'T3_result.csv')
    t3m.write_rows(p3, rows3 if rows3 is not None else [
        t3_row(delay_us=3000.0, vii='do_duoc', quant_us=100.0),
        t3_row(delay_us=5000.0),
        t3_row(delay_us=8000.0, vii='do_duoc', quant_us=200.0),
        t3_row(vi='het_han'), t3_row(vi='no_data'), t3_row(vi='chua_ghep'),
        t3_row(vi='khong_dat')])
    p6 = str(tmp / 'T6_follow.csv')
    t6m.write_csv(p6, rows6 if rows6 is not None else [
        t6_row(duty_to=0.7, leave_ms=4.0, follow_ms=12.0, settled=True),
        t6_row(duty_to=0.9, leave_ms=6.0, follow_ms=20.0, settled=True),
        t6_row(duty_to=0.5, settled=False),                       # not settled, no leave
        t6_row(duty_to=0.0, leave_ms=3.0, follow_ms=8.0, settled=True),   # flat -> rieng
        t6_row(duty_to=1.0, leave_ms=9.0, settled=False)],        # flat leave, not settled
        t6m.FOLLOW_COLS)
    return p3, p6


def write_summaries(tmp, sync='manual', limit=0.0055, u_us=12.3):
    t6s = str(tmp / 'T6_summary.json')
    json.dump({'sync': {'kind': sync}, 'limit': limit}, open(t6s, 'w'))
    t3s = str(tmp / 'T3_summary.json')
    json.dump({'u_us': u_us}, open(t3s, 'w'))
    return t6s, t3s


def read_out(tmp):
    rows = {r['thanh_phan']: r for r in csv.DictReader(
        open(tmp / 'tre_vong.csv', newline='', encoding='utf-8'))}
    summ = json.load(open(tmp / 'tre_vong_summary.json', encoding='utf-8'))
    return rows, summ


def run_cli(capsys, argv):
    code = tv.main(argv)
    out = capsys.readouterr().out.strip().splitlines()
    return code, out[-1] if out else ''


def test_known_stats(tmp_path, capsys):
    p3, p6 = write_inputs(tmp_path)
    t6s, t3s = write_summaries(tmp_path)
    code, last = run_cli(capsys, ['--t3-result', p3, '--t6-follow', p6,
                                  '--t6-summary', t6s, '--t3-summary', t3s,
                                  '--out', str(tmp_path / 'o')])
    assert code == 0 and last == '#ket_qua,ma=0'
    r, summ = read_out(tmp_path / 'o')
    # t3_i = [3,5,8]; t3_i_ii = [3.1, 8.2]
    assert int(r['t3_i']['n']) == 3
    assert float(r['t3_i']['median_ms']) == 5.0 and float(r['t3_i']['max_ms']) == 8.0
    assert float(r['t3_i']['min_ms']) == 3.0 and float(r['t3_i']['p95_ms']) == 8.0
    assert float(r['t3_i_ii']['median_ms']) == pytest.approx(5.65)
    assert float(r['t3_i_ii']['max_ms']) == pytest.approx(8.2)
    # t6 non-flat: leave [4,6], follow [12,20]
    assert int(r['t6_leave']['n']) == 2 and float(r['t6_leave']['max_ms']) == 6.0
    assert int(r['t6_follow']['n']) == 2 and float(r['t6_follow']['median_ms']) == 16.0
    # flat tach rieng, khong vao tong
    assert int(r['t6_flat_leave']['n']) == 2 and float(r['t6_flat_leave']['max_ms']) == 9.0
    assert int(r['t6_flat_follow']['n']) == 1 and float(r['t6_flat_follow']['max_ms']) == 8.0
    # S1 = [7,9,9,11,12,14]: med 10, p95/max 14 = 8+6
    s1 = r['tre_vong_chet']
    assert int(s1['n']) == 6 and float(s1['median_ms']) == 10.0
    assert float(s1['p95_ms']) == 14.0 and float(s1['max_ms']) == 14.0
    assert float(s1['min_ms']) == 7.0
    # S2 = [15.1,20.2,23.1,28.2]: med 21.65, max 28.2 = 8.2+20
    s2 = r['tre_vong_tren']
    assert int(s2['n']) == 4 and float(s2['median_ms']) == pytest.approx(21.65)
    assert float(s2['max_ms']) == pytest.approx(28.2)
    # io tren luoi 1 ms: 5.0/8.0 nguyen -> (v,v); 10.0/14.0 idem
    assert int(r['t3_i']['io_floor_median']) == 5 and int(r['t3_i']['io_ceil_median']) == 5
    assert int(r['t3_i']['io_floor_max']) == 8 and int(r['t3_i']['io_ceil_max']) == 8
    # summary: exclusions, span, sync, u
    assert summ['t3_excluded'] == {'het_han': 1, 'khong_dat': 1, 'no_data': 1,
                                   'chua_ghep': 1}
    assert summ['t6_excluded'] == {'not_settled': 2, 'no_leave': 1}
    assert summ['t_mark_span_us_max'] == 50.0
    assert summ['t6'] == {'sync_kind': 'manual', 'limit': 0.0055}
    assert summ['t3'] == {'u_us': 12.3}
    assert len(summ['thieu']) == 3                     # chi cac manh co dinh khong do duoc


def test_io_pair_edges():
    for x in (3.0, 2.9999999999, 3.0000000001):
        assert tv.io_pair(x) == (3, 3)
    assert tv.io_pair(3.4) == (3, 4)
    assert tv.io_pair(4.0) == (4, 4)
    assert tv.io_pair(0.0) == (0, 0)


def test_header_mismatch(tmp_path):
    p3, p6 = write_inputs(tmp_path)
    with open(p3, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['sai', 'header'])
    assert tv.main(['--t3-result', p3, '--t6-follow', p6,
                    '--out', str(tmp_path / 'o')]) == 2


def test_non_numeric_delay(tmp_path):
    p3, p6 = write_inputs(tmp_path, rows3=[t3_row(delay_us='abc')])
    assert tv.main(['--t3-result', p3, '--t6-follow', p6,
                    '--out', str(tmp_path / 'o')]) == 2


def test_no_s1_samples(tmp_path, capsys):
    p3, p6 = write_inputs(tmp_path, rows3=[t3_row(vi='het_han'), t3_row(vi='no_data')])
    code, last = run_cli(capsys, ['--t3-result', p3, '--t6-follow', p6,
                                  '--out', str(tmp_path / 'o')])
    assert code == 3 and last == '#ket_qua,ma=3,ly_do=thieu_bang_chung'


def test_summary_gates(tmp_path):
    p3, p6 = write_inputs(tmp_path)
    # khong summary nao -> 3
    assert tv.main(['--t3-result', p3, '--t6-follow', p6,
                    '--out', str(tmp_path / 'o1')]) == 3
    summ = json.load(open(tmp_path / 'o1' / 'tre_vong_summary.json', encoding='utf-8'))
    assert any('T6_summary' in s for s in summ['thieu'])
    assert any('T3_summary' in s for s in summ['thieu'])
    # limit 0.006 (s06_t6_check LIMIT v3.4) -> 3
    t6s, t3s = write_summaries(tmp_path, limit=0.006)
    assert tv.main(['--t3-result', p3, '--t6-follow', p6, '--t6-summary', t6s,
                    '--t3-summary', t3s, '--out', str(tmp_path / 'o2')]) == 3
    # sync khong manual -> 3
    t6s, t3s = write_summaries(tmp_path, sync='udp_envelope')
    assert tv.main(['--t3-result', p3, '--t6-follow', p6, '--t6-summary', t6s,
                    '--t3-summary', t3s, '--out', str(tmp_path / 'o3')]) == 3
    # u_us None -> 3
    t6s, t3s = write_summaries(tmp_path, u_us=None)
    assert tv.main(['--t3-result', p3, '--t6-follow', p6, '--t6-summary', t6s,
                    '--t3-summary', t3s, '--out', str(tmp_path / 'o4')]) == 3
    # day du manual + 0.0055 + u -> 0
    t6s, t3s = write_summaries(tmp_path)
    assert tv.main(['--t3-result', p3, '--t6-follow', p6, '--t6-summary', t6s,
                    '--t3-summary', t3s, '--out', str(tmp_path / 'o5')]) == 0
