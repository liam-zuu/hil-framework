"""tools/c7_marks_from_tape.py (L104): tape distances to O, X (and optional Y) -> mark centres, on
synthetic points with known positions; the output CSV is read by c7_clip_probe.load_marks
(what c7_measure calib --markers uses)."""
import csv
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import c7_marks_from_tape as mt  # noqa: E402
import c7_clip_probe as cp  # noqa: E402

DESIGN = {'1': (-1400, 900), '5': (1400, 900), '7': (1400, 0), '10': (0, -900), '13': (-1950, 1450),
          '22': (0, -1450), '27': (0, 0)}
REFS = {'O': (0.0, 0.0), 'X': (1500.0, 0.0)}


def design_csv(tmp_path):
    p = tmp_path / 'design.csv'
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'vai_tro', 'x_cm', 'y_cm', 'canh_den_mm', 'dict'])
        w.writerow(['0', 'noc_xe', '', '', '150.0', 'DICT_4X4_50'])
        for i, (x, y) in DESIGN.items():
            w.writerow([i, 'moc_san', x / 10, y / 10, '150.0' if int(i) >= 13 and int(i) <= 24 else '100.0',
                        'DICT_4X4_50'])
    return str(p)


def row(pid, true, refs=REFS, noise=(0.0, 0.0), extra=None, phia=''):
    r = {'id': pid, 'phia': phia, 'canh_den_mm': '', 'nguoi_do': 'Liam', 'ngay': '2026-11-14', 'ghi_chu': ''}
    for n, (rx, ry) in refs.items():
        d = math.hypot(true[0] - rx, true[1] - ry)
        r[f'd_{n}_mm_1'] = f'{d + noise[0]:.1f}'
        r[f'd_{n}_mm_2'] = f'{d + noise[1]:.1f}'
    r.update(extra or {})
    return r


def by_id(points):
    return {p['id']: p for p in points}


def test_recovers_known_points_on_the_design_side():
    true = {'1': (-1390, 905), '5': (1410, 893), '10': (5, -902), '22': (-8, -1455)}
    rows = [row(k, v) for k, v in true.items()]
    pts, lines, n_err, n_warn, L = mt.process(rows, {k: dict(x_mm=x, y_mm=y, vai_tro='moc_san', canh_den_mm='100',
                                                              dict='D') for k, (x, y) in DESIGN.items()}, REFS)
    P = by_id(pts)
    assert n_err == 0
    for k, (x, y) in true.items():
        assert P[k]['x_mm'] == pytest.approx(x, abs=0.5) and P[k]['y_mm'] == pytest.approx(y, abs=0.5)   # readings rounded to 0.1 mm
    assert P['10']['y_mm'] < 0 and P['22']['y_mm'] < 0                     # side from the design position
    assert P['1']['dev'] == pytest.approx(math.hypot(10, 5), abs=0.5)


def design_dict():
    return {k: dict(x_mm=x, y_mm=y, vai_tro='moc_san', canh_den_mm='100', dict='D') for k, (x, y) in DESIGN.items()}


def test_circles_that_do_not_meet_are_an_error():
    r = {'id': '1', 'd_O_mm_1': '500', 'd_X_mm_1': '600', 'phia': ''}         # 500 + 600 < 1500
    pts, lines, n_err, _, _ = mt.process([r], design_dict(), REFS)
    assert n_err == 1 and 'KHONG_CAT' in lines[-1] and not pts[0]['ok']


def test_two_readings_far_apart_ask_to_measure_again():
    pts, lines, n_err, _, _ = mt.process([row('5', (1410, 893), noise=(0.0, 3.0))], design_dict(), REFS)
    assert n_err == 1 and 'DO_LAI' in lines[-1]
    pts, lines, n_err, _, _ = mt.process([row('5', (1410, 893), noise=(0.0, 1.5))], design_dict(), REFS)
    assert n_err == 0


def test_points_on_the_OX_line_have_a_poor_y_and_a_third_reference_fixes_it():
    pts, lines, n_err, n_warn, _ = mt.process([row('7', (1400, 20))], design_dict(), REFS)
    p = by_id(pts)['7']
    assert n_warn == 1 and 'KEM' in lines[-1] and p['sigma_y'] > 5 and p['sigma_x'] < 1.5
    refs3 = dict(REFS, Y=(0.0, 1000.0))
    pts, lines, n_err, n_warn, _ = mt.process([row('7', (1400, 20), refs=refs3)], design_dict(), refs3)
    p = by_id(pts)['7']
    assert n_warn == 0 and p['sigma_y'] < 3 and p['y_mm'] == pytest.approx(20, abs=0.5)


def test_three_references_with_a_bad_reading_flag_the_residual():
    refs3 = dict(REFS, Y=(0.0, 1000.0))
    r = row('1', (-1390, 905), refs=refs3)
    r['d_Y_mm_1'] = r['d_Y_mm_2'] = str(float(r['d_Y_mm_1']) + 30)
    pts, lines, n_err, _, _ = mt.process([r], design_dict(), refs3)
    assert n_err == 1 and 'DU_LON' in lines[-1]


def test_large_deviation_and_wrong_side_are_warnings():
    pts, lines, n_err, n_warn, _ = mt.process([row('5', (1400, 960))], design_dict(), REFS)
    assert n_err == 0 and n_warn == 1 and 'LECH' in lines[-1]
    pts, lines, n_err, n_warn, _ = mt.process([row('5', (1410, 893), phia='duoi')], design_dict(), REFS)
    assert 'PHIA' in lines[-1]


def test_measured_OX_row_replaces_the_design_length():
    L = 1503.0
    refs = {'O': (0.0, 0.0), 'X': (L, 0.0)}
    rows = [{'id': 'X', 'd_O_mm_1': '1503', 'd_O_mm_2': '1503'}, row('1', (-1390, 905), refs=refs)]
    pts, lines, n_err, _, Lu = mt.process(rows, design_dict(), REFS)
    assert Lu == L and by_id(pts)['1']['x_mm'] == pytest.approx(-1390, abs=0.5)


def test_plumb_point_without_design_uses_phia(tmp_path):
    rows = [row('day_doi', (300, -250), phia='duoi')]
    pts, lines, n_err, _, _ = mt.process(rows, design_dict(), REFS)
    p = by_id(pts)['day_doi']
    assert p['ok'] and p['x_mm'] == pytest.approx(300, abs=0.05) and p['y_mm'] == pytest.approx(-250, abs=0.05)


def test_cli_writes_a_marker_csv_that_load_marks_reads(tmp_path):
    true = {'1': (-1390, 905), '13': (-1945, 1452), '10': (5, -902)}
    rows = [row(k, v) for k, v in true.items()] + [row('day_doi', (310, 120), phia='tren')]
    rows[1]['canh_den_mm'] = '144'
    m = tmp_path / 'do.csv'
    cols = sorted({k for r in rows for k in r}, key=lambda c: (c != 'id', c))
    with open(m, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / 'out'
    rc = mt.main(['--measured', str(m), '--design', design_csv(tmp_path), '--out', str(out), '--warn-sigma', '10'])
    assert rc == 0
    marks = cp.load_marks([str(out / 'moc_san_do.csv')])
    assert set(marks) == {1, 10, 13}
    assert marks[13][2] == pytest.approx(0.144) and marks[1][2] == pytest.approx(0.100)
    assert marks[1][0] == pytest.approx(-1.390, abs=5e-4) and marks[10][1] == pytest.approx(-0.902, abs=5e-4)
    rep = (out / 'bao_cao.txt').read_text(encoding='utf-8')
    import re
    mm = re.search(r'--plumb (-?[0-9.]+),(-?[0-9.]+)', rep)
    assert mm and float(mm.group(1)) == pytest.approx(0.310, abs=5e-4) and float(mm.group(2)) == pytest.approx(0.120, abs=5e-4)
    assert mt.main(['--measured', str(m), '--design', design_csv(tmp_path), '--out', str(out)]) == 2   # no overwrite


def test_default_four_references_with_measured_Y_and_X2():
    refs_true = {'O': (0.0, 0.0), 'X': (1501.0, 0.0), 'Y': (6.0, 998.0), 'X2': (-1497.0, 4.0)}
    rows = [{'id': 'X', 'd_O_mm_1': '1501', 'd_O_mm_2': '1501'},
            row('Y', refs_true['Y'], refs={k: refs_true[k] for k in ('O', 'X')}),
            row('X2', refs_true['X2'], refs={k: refs_true[k] for k in ('O', 'X', 'Y')})]
    true = {'7': (1402, 6), '27': (3, -4), '1': (-1390, 905), '13': (-1945, 1452)}
    rows += [row(k, v, refs=refs_true) for k, v in true.items()]
    pts, lines, n_err, n_warn, _ = mt.process(rows, design_dict(), mt.DEFAULT_REFS)
    P = by_id(pts)
    assert n_err == 0 and n_warn == 0, lines
    for k, (x, y) in true.items():
        assert P[k]['x_mm'] == pytest.approx(x, abs=0.5) and P[k]['y_mm'] == pytest.approx(y, abs=0.5)
        assert max(P[k]['sigma_x'], P[k]['sigma_y']) < 3
    assert any(l.startswith('X2: do duoc') for l in lines) and any(l.startswith('Y: do duoc') for l in lines)


def test_reference_rows_are_not_marks():
    pts, lines, *_ = mt.process([row('Y', (0.0, 1000.0), refs=REFS)], design_dict(), mt.DEFAULT_REFS)
    assert pts == []
