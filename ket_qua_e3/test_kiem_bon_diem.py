"""ket_qua_e3/kiem_bon_diem.py (E3: the four off-grid T5 points, spec S06 v3.5 cau 7)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kiem_bon_diem as k  # noqa: E402

F = np.float32


def test_h7_half_period_timer_round_half_up_and_clamp():
    # encoder_gen.c:47-49: clamp to [1, 65535], then (uint32_t)(hp + 0.5f)
    assert k.h7_half_period_timer(F(952.5)) == 953
    assert k.h7_half_period_timer(np.nextafter(F(952.5), F(0))) == 952
    assert k.h7_half_period_timer(F(952.4999)) == 952
    assert k.h7_half_period_timer(F(65535.0)) == 65535
    assert k.h7_half_period_timer(F(65535.6)) == 65535
    assert k.h7_half_period_timer(F(70000.0)) == 65535
    assert k.h7_half_period_timer(F(1.0)) == 1
    assert k.h7_half_period_timer(F(0.7)) == 1
    assert k.h7_half_period_timer(F(1.5)) == 2


def test_h7_pulses_deadband_is_float32_compare():
    # encoder_gen.c:212-213: pulses only for omega > 0.3f or omega < -0.3f
    assert not k.h7_pulses(F(0.3)) and not k.h7_pulses(-F(0.3))
    assert k.h7_pulses(np.nextafter(F(0.3), F(1.0)))
    assert k.h7_pulses(np.nextafter(-F(0.3), F(-1.0)))
    assert not k.h7_pulses(k.h7_omega(245))     # omega_d_h7 = 0.29908 <= 0.3
    assert k.h7_pulses(k.h7_omega(246))         # omega_d_h7 = 0.30030 > 0.3
    assert not k.h7_pulses(k.h7_omega(238))     # 0.29054: the 0.29 indicator point stays silent


def test_frac_ok_closed_interval():
    assert k.frac_ok(0.4) and k.frac_ok(0.6) and k.frac_ok(0.5)
    assert k.frac_ok(0.4 + 1e-9)
    assert not k.frac_ok(0.4 - 1e-9) and not k.frac_ok(0.6 + 1e-9)


def test_table_26_rows_new_points_inside_frac_band():
    rows = k.table()
    assert len(rows) == 26                       # (1 indicator + 7 old + 4 new + 1 spec point replaced) x 2
    moi = [r for r in rows if r['nhom'] == 'moi']
    assert len(moi) == 8
    assert all(r['frac_trong_04_06'] == 1 for r in moi)
    assert all(r['de_xuat_omega_set'] == '' for r in moi)     # inside the band: no proposal needed
    # C3-18: 9.995 of the spec moved to 9.9942 (int16 8187, timer 953)
    expected = {9.9942: (8187, 953), 19.979: (16366, 477), 31.68: (25951, 301), 38.78: (31768, 245)}
    for r in moi:
        raw, hp_t = expected[abs(r['omega_set'])]
        sign = 1 if r['omega_set'] > 0 else -1
        assert r['int16'] == sign * raw
        assert r['hp_timer'] == hp_t
    frac = {abs(r['omega_set']): r['frac_f32'] for r in moi}
    assert frac[9.9942] == pytest.approx(0.5502, abs=1e-4)
    assert frac[38.78] == pytest.approx(0.4837, abs=1e-4)
    goc = [r for r in rows if r['nhom'] == 'spec_goc']
    assert [r['omega_set'] for r in goc] == [9.995, -9.995]
    for r in goc:
        sign = 1 if r['omega_set'] > 0 else -1
        assert r['int16'] == sign * 8188 and r['hp_timer'] == 952
        assert r['frac_f32'] == pytest.approx(0.4338, abs=1e-4)
        assert r['de_xuat_omega_set'] == sign * 9.9942 and r['de_xuat_int16'] == sign * 8187
        assert abs(r['de_xuat_frac_f32'] - 0.5) < abs(r['frac_f32'] - 0.5)
    assert all(r['phat_xung'] == 0 for r in rows if r['nhom'] == 'chi_bao')
    for r in rows:
        if r['phat_xung']:
            # float32 chain vs float64: a few float32 rounding steps (eps 1.19e-7), relative bound
            assert abs(r['hp_f32'] - r['hp_exact_d']) <= 3e-7 * r['hp_exact_d']
            assert abs(r['err_vs_set']) < 0.006


def test_point_outside_band_gets_proposal(monkeypatch):
    # 10.0 rad/s: frac of the H7 half period ~0.97, outside [0.4; 0.6] -> a proposal must appear
    monkeypatch.setattr(k, 'NEW_POINTS', (10.0,))
    moi = [r for r in k.table() if r['nhom'] == 'moi']
    assert len(moi) == 2
    for r in moi:
        assert r['frac_trong_04_06'] == 0
        prop = r['de_xuat_omega_set']
        assert isinstance(prop, float)
        assert (prop > 0) == (r['omega_set'] > 0)
        assert k.protocol._to_int16(prop, 40.0) == r['de_xuat_int16']
        assert abs(r['de_xuat_frac_f32'] - 0.5) <= 0.05


def test_new_points_match_t5_schedule_lock():
    import t5_schedule as t5
    assert k.NEW_POINTS == t5.OFF_GRID_POINTS
    for r in k.table():
        if r['nhom'] == 'moi':
            raw, wd, hp, timer = t5.OFF_GRID_LOCK[abs(r['omega_set'])]
            sign = 1 if r['omega_set'] > 0 else -1
            assert (r['int16'], r['omega_d_h7'], r['hp_f32'], r['hp_timer']) == (sign * raw, sign * wd, hp, timer)


def test_frame_roundtrip_all_26_omega_set():
    for r in k.table():
        assert k.frame_roundtrip_ok(r['omega_set'])
