"""Tests of tools/mo_hinh_con.py. Run: python -m pytest -q tools/tests/test_mo_hinh_con.py

Check numbers (tables RHO, DINH_TINH, BIEN, KHONG_CO): every number is one the review documents print
for the sub-model; the source is file:line of
  V6 = 03_CODE/docs/review/2026-11-21_Fable_API_can_cu_nhom1/vong6/ket_luan_da_kiem.md
  V4 = .../vong4/ket_luan_da_kiem.md        V3 = .../vong3/ket_luan_da_kiem.md        (thesis repo)
A printed number counts as reproduced when the computed value rounds to it at the printed precision:
|value - printed| <= half a unit of the last printed digit. The relative deviation of every row is
listed by
  python tools/tests/test_mo_hinh_con.py
(ASCII table, the one of BAO_CAO.md). A deviation above 0.5 % can then only come from the rounding of the
document (half a printed unit is more than 0.5 % of the number); the table marks those rows.
Unless a row says otherwise: G = 1, tau = 0.171 s, loop delay 0, velocity loop at the firmware defaults
(PID kp_xy 1.5, ki_xy 0.05: pid.c:12-13; LADRC omega_c 8, omega_o 30: adrc.c:32-34).
"""
import importlib.util
import json
import math
import os
import re
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import mo_hinh_con as mh  # noqa: E402
import gain_table as gt  # noqa: E402

# firmware defaults: pid.c:12-13, 24-25; adrc.c:32-34, 38-40; pos_ctrl.h:37; heading.h:25
PID0 = dict(kp_xy=1.5, ki_xy=0.05, kp_wz=1.0, ki_wz=0.05, pos_kp=6.0, pos_ki=3.0, k_psi=1.0)
ADRC0 = dict(omega_c_xy=8.0, omega_o_xy=30.0, omega_c_wz=8.0, omega_o_wz=30.0, pos_kp=6.0, pos_ki=3.0, k_psi=1.0)
TAU_V3 = 0.018671 / 0.10916          # V3:65, the unrounded tau of round 3 (0.1710 s)
TAT_VI_TRI = dict(pos_kp=0.0, pos_ki=0.0)    # position loop off: the velocity loop alone (V6:36)


def bo(ctrl, **kw):
    """The default gain set of ctrl with the gains of kw replaced."""
    return dict(PID0 if ctrl == 'PID' else ADRC0, **kw)


def on_dinh(ctrl, g, **kw):
    return mh.ban_kinh_pho(ctrl, g, **kw) < 1.0


def khop(value, doc, dec):
    """value rounds to the printed number doc (dec digits after the decimal point; dec < 0: tens, ...)."""
    return abs(value - doc) <= 0.5 * 10.0 ** -dec * (1.0 + 1e-9)


def bien(f, lo, hi, huong):
    """The stability limit of f between lo and hi, by bisection (45 halvings).
    huong 'tren': stable at lo, not at hi (an upper limit); 'duoi': the other way round (a lower limit)."""
    assert f(lo) == (huong == 'tren') and f(hi) == (huong == 'duoi'), (lo, hi, huong)
    for _ in range(45):
        m = 0.5 * (lo + hi)
        if f(m) == (huong == 'tren'):
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def bien_gain(ctrl, g, ten, huong, doc, dec, **kw):
    """The limit of gain ten (the other gains g fixed), looked for within 10 % (at least one printed
    unit) of the printed number doc."""
    w = max(0.1 * doc, 10.0 ** -dec)
    return bien(lambda q: on_dinh(ctrl, dict(g, **{ten: q}), **kw), doc - w, doc + w, huong)


# -- spectral radii printed by the documents: (id, source, what, ctrl, gains, printed, dec, model kw) --
RHO = [
    ('R1', 'V6:69', 'PID, pos_kp=0, pos_ki=3', 'PID', dict(pos_kp=0.0, pos_ki=3.0), 1.0009, 4, {}),
    ('R2', 'V6:69', 'LADRC, pos_kp=0, pos_ki=3', 'ADRC', dict(pos_kp=0.0, pos_ki=3.0), 1.0023, 4, {}),
    ('R3', 'V6:113', 'PID, (pos_kp; pos_ki) = (600; 300)', 'PID', dict(pos_kp=600.0, pos_ki=300.0), 1.151, 3, {}),
    ('R4', 'V6:113', 'LADRC, (600; 300)', 'ADRC', dict(pos_kp=600.0, pos_ki=300.0), 1.009, 3, {}),
    ('R5', 'V4:113', 'LADRC velocity loop, omega_c=50, omega_o=49, G=1.127', 'ADRC',
     dict(omega_c_xy=50.0, omega_o_xy=49.0, **TAT_VI_TRI), 0.75, 2, dict(G=1.127)),
]

# -- stable / not stable as the documents state: (id, source, what, ctrl, gains, stable, model kw) --
DINH_TINH = [
    ('Q1', 'V6:113', 'PID (200; 300) khong on dinh', 'PID', dict(pos_kp=200.0, pos_ki=300.0), False, {}),
    ('Q2', 'V6:113', 'LADRC (200; 300) on dinh', 'ADRC', dict(pos_kp=200.0, pos_ki=300.0), True, {}),
    ('Q3', 'V6:110', 'PID (30; 30) on dinh', 'PID', dict(pos_kp=30.0, pos_ki=30.0), True, {}),
    ('Q4', 'V6:110', 'LADRC (30; 30) on dinh', 'ADRC', dict(pos_kp=30.0, pos_ki=30.0), True, {}),
    ('Q5', 'V6:37', 'PID mac dinh, pos_kp=184 hoi tu', 'PID', dict(pos_kp=184.0), True, {}),
    ('Q6', 'V6:37', 'PID mac dinh, pos_kp=186 phan ky', 'PID', dict(pos_kp=186.0), False, {}),
    ('Q7', 'V6:37', 'LADRC mac dinh, pos_kp=555 hoi tu', 'ADRC', dict(pos_kp=555.0), True, {}),
    ('Q8', 'V6:37', 'LADRC mac dinh, pos_kp=565 phan ky', 'ADRC', dict(pos_kp=565.0), False, {}),
    ('Q9', 'V6:111', 'PID mac dinh (6; 3) on dinh', 'PID', {}, True, {}),
    ('Q10', 'V6:111', 'LADRC mac dinh (6; 3) on dinh', 'ADRC', {}, True, {}),
    ('Q11', 'V6:69', 'PID pos_kp=0, pos_ki=3 khong on dinh', 'PID', dict(pos_kp=0.0, pos_ki=3.0), False, {}),
    ('Q12', 'V6:69', 'LADRC pos_kp=0, pos_ki=3 khong on dinh', 'ADRC', dict(pos_kp=0.0, pos_ki=3.0), False, {}),
]

# -- stability limits: (id, source, what, ctrl, fixed gains, gain looked for, huong, printed, dec, model kw) --
BIEN = []


def _b(ma, nguon, mo_ta, ctrl, g, ten, huong, doc, dec, **kw):
    BIEN.append((ma, nguon, mo_ta, ctrl, g, ten, huong, doc, dec, kw))


# velocity loop alone (V6 a.1 "bo vong vi tri"; V3 1.3)
_b('B01', 'V6:36', 'bien kp_xy, bo vong vi tri, ki_xy=0', 'PID', dict(ki_xy=0.0, **TAT_VI_TRI), 'kp_xy', 'tren', 35.56, 2)
_b('B02', 'V6:36', 'bien kp_xy, bo vong vi tri, ki_xy=0, G=1.127', 'PID', dict(ki_xy=0.0, **TAT_VI_TRI),
   'kp_xy', 'tren', 31.55, 2, G=1.127)
_b('B03', 'V6:149', 'bien kp_xy o tre vong 0 ms', 'PID', dict(ki_xy=0.0, **TAT_VI_TRI), 'kp_xy', 'tren', 35.6, 1)
_b('B04', 'V3:66', 'bien kp_xy, tau=0.018671/0.10916', 'PID', dict(ki_xy=0.0, **TAT_VI_TRI),
   'kp_xy', 'tren', 35.56, 2, tau=TAU_V3)
_b('B05', 'V3:66', 'bien kp_xy, tau=0.018671/0.10916, G=1.127', 'PID', dict(ki_xy=0.0, **TAT_VI_TRI),
   'kp_xy', 'tren', 31.56, 2, tau=TAU_V3, G=1.127)
_b('B06', 'V3:67', 'bien ki_xy tai kp_xy=0, tau=0.018671/0.10916', 'PID', dict(kp_xy=0.0, **TAT_VI_TRI),
   'ki_xy', 'tren', 202, 0, tau=TAU_V3)
_b('B07', 'V3:67', 'bien ki_xy tai kp_xy=14 (lon nhat), tau=0.018671/0.10916', 'PID', dict(kp_xy=14.0, **TAT_VI_TRI),
   'ki_xy', 'tren', 1311, 0, tau=TAU_V3)
for _wo, _doc in ((10, 180), (30, 140), (49, 101), (60, 79), (80, 39)):
    _b(f'B08_{_wo}', 'V4:112', f'LADRC bien omega_c, bo vong vi tri, omega_o={_wo}', 'ADRC',
       dict(omega_o_xy=float(_wo), **TAT_VI_TRI), 'omega_c_xy', 'tren', _doc, 0)

# a.2: limit of pos_kp at pos_ki = 0, loop delay 0 ms
_b('B10', 'V6:44', 'PID bien pos_kp, pos_ki=0', 'PID', dict(pos_ki=0.0), 'pos_kp', 'tren', 185.2, 1)
_b('B11', 'V6:44', 'PID bien pos_kp, pos_ki=0, G=1.127', 'PID', dict(pos_ki=0.0), 'pos_kp', 'tren', 175.0, 1, G=1.127)
_b('B12', 'V6:44', 'LADRC bien pos_kp, pos_ki=0', 'ADRC', dict(pos_ki=0.0), 'pos_kp', 'tren', 559.6, 1)
_b('B13', 'V6:44', 'LADRC bien pos_kp, pos_ki=0, G=1.127', 'ADRC', dict(pos_ki=0.0), 'pos_kp', 'tren', 535.9, 1, G=1.127)
_b('B14', 'V6:53', 'PID bien pos_kp, pos_ki=0, tau=0.0823', 'PID', dict(pos_ki=0.0), 'pos_kp', 'tren', 171.8, 1, tau=0.0823)
_b('B15', 'V6:53', 'PID bien pos_kp, pos_ki=0, tau=0.197', 'PID', dict(pos_ki=0.0), 'pos_kp', 'tren', 187.0, 1, tau=0.197)
_b('B16', 'V6:54', 'LADRC bien pos_kp, pos_ki=0, tau=0.0823', 'ADRC', dict(pos_ki=0.0), 'pos_kp', 'tren', 433.5, 1, tau=0.0823)
_b('B17', 'V6:54', 'LADRC bien pos_kp, pos_ki=0, tau=0.197 (dau duoi 527)', 'ADRC', dict(pos_ki=0.0),
   'pos_kp', 'tren', 527, 0, tau=0.197)
_b('B18', 'V6:54', 'LADRC bien pos_kp, pos_ki=0, tau=0.171 (dau tren 560)', 'ADRC', dict(pos_ki=0.0), 'pos_kp', 'tren', 560, 0)

# a.3: largest pos_ki for a pos_kp
for _kp, _doc, _dec in ((0.6, 9.6, 1), (2, 32, 0), (6, 93, 0), (20, 283, 0), (60, 631, 0), (100, 703, 0), (150, 426, 0)):
    _b(f'B20_{_kp}', 'V6:62', f'PID pos_ki lon nhat tai pos_kp={_kp}', 'PID', dict(pos_kp=float(_kp)), 'pos_ki', 'tren', _doc, _dec)
for _kp, _doc, _dec in ((0.6, 3.3, 1), (2, 11, 0), (6, 35, 0), (20, 127, 0), (60, 451, 0), (100, 823, 0), (150, 1279, 0),
                        (200, 1662, 0), (400, 1805, 0)):
    _b(f'B21_{_kp}', 'V6:63', f'LADRC pos_ki lon nhat tai pos_kp={_kp}', 'ADRC', dict(pos_kp=float(_kp)), 'pos_ki', 'tren', _doc, _dec)
_b('B22', 'V6:65', 'PID pos_ki lon nhat tren moi pos_kp, tai pos_kp=90', 'PID', dict(pos_kp=90.0), 'pos_ki', 'tren', 710, 0)
_b('B23', 'V6:65', 'LADRC pos_ki lon nhat (~2050), tai pos_kp=325', 'ADRC', dict(pos_kp=325.0), 'pos_ki', 'tren', 2050, -1)

# a.3, a.6: smallest pos_kp for a pos_ki
for _ki, _doc, _dec in ((3, 0.19, 2), (30, 1.90, 2), (100, 6.5, 1), (300, 21.4, 1), (600, 54.5, 1)):
    _b(f'B24_{_ki}', 'V6:67' + (', 110' if _ki == 30 else ''), f'PID pos_kp nho nhat tai pos_ki={_ki}', 'PID',
       dict(pos_ki=float(_ki)), 'pos_kp', 'duoi', _doc, _dec)
for _ki, _doc, _dec in ((3, 0.54, 2), (30, 5.21, 2), (100, 16.2, 1), (300, 42.5, 1), (600, 76.3, 1), (1000, 118.9, 1)):
    _b(f'B25_{_ki}', 'V6:68' + (', 110' if _ki == 30 else ''), f'LADRC pos_kp nho nhat tai pos_ki={_ki}', 'ADRC',
       dict(pos_ki=float(_ki)), 'pos_kp', 'duoi', _doc, _dec)

# a.4 PID: limit of pos_kp over kp_xy at ki_xy = 0 (pos_ki = 0); outer box
for _k, _doc in ((0, 202), (0.5, 196), (1.5, 185), (3, 170), (6, 143), (10, 113), (15, 82), (20, 56), (25, 35), (30, 17), (35, 2)):
    _b(f'B30_{_k}', 'V6:79', f'PID bien pos_kp, kp_xy={_k}, ki_xy=0, pos_ki=0', 'PID',
       dict(kp_xy=float(_k), ki_xy=0.0, pos_ki=0.0), 'pos_kp', 'tren', _doc, 0)
_b('B31', 'V6:81', 'PID bien pos_kp, ki_xy=200, kp_xy=10 (lon nhat tren luoi kp_xy cua a.4)', 'PID',
   dict(kp_xy=10.0, ki_xy=200.0, pos_ki=0.0), 'pos_kp', 'tren', 74, 0)
_b('B32', 'V6:81', 'PID bien pos_kp, ki_xy=600, kp_xy=15 (lon nhat tren luoi kp_xy cua a.4)', 'PID',
   dict(kp_xy=15.0, ki_xy=600.0, pos_ki=0.0), 'pos_kp', 'tren', 32, 0)
_HOP = dict(kp_xy=0.0, ki_xy=0.0, pos_ki=0.0)
_b('B33', 'V6:82', 'PID hop ngoai pos_kp (kp_xy=0, ki_xy=0, pos_ki=0)', 'PID', _HOP, 'pos_kp', 'tren', 202.0, 1)
_b('B34', 'V6:82', 'PID hop ngoai pos_kp, tau=0.0823', 'PID', _HOP, 'pos_kp', 'tren', 204.1, 1, tau=0.0823)
_b('B35', 'V6:82', 'PID hop ngoai pos_kp, tau=0.197', 'PID', _HOP, 'pos_kp', 'tren', 201.7, 1, tau=0.197)
_b('B36', 'V6:82', 'PID hop ngoai pos_kp, G=0.9', 'PID', _HOP, 'pos_kp', 'tren', 224, 0, G=0.9)
_b('B37', 'V6:82', 'PID hop ngoai pos_kp, G=1.127', 'PID', _HOP, 'pos_kp', 'tren', 179, 0, G=1.127)
_b('B38', 'V6:82', 'PID hop ngoai pos_kp, G=1.159', 'PID', _HOP, 'pos_kp', 'tren', 174, 0, G=1.159)
_b('B39', 'V6:123', 'PID hop ngoai pos_kp, G=0.984', 'PID', _HOP, 'pos_kp', 'tren', 205, 0, G=0.984)
_b('B40', 'V6:83', 'PID hop ngoai pos_ki (kp_xy=23, ki_xy=0, pos_kp=0)', 'PID',
   dict(kp_xy=23.0, ki_xy=0.0, pos_kp=0.0), 'pos_ki', 'tren', 5265, 0)
_b('B41', 'V6:83', 'PID hop ngoai pos_ki, tau=0.197 (kp_xy=27)', 'PID',
   dict(kp_xy=27.0, ki_xy=0.0, pos_kp=0.0), 'pos_ki', 'tren', 5223, 0, tau=0.197)
_b('B42', 'V6:83', 'PID hop ngoai pos_ki, tau=0.0823 (kp_xy=11)', 'PID',
   dict(kp_xy=11.0, ki_xy=0.0, pos_kp=0.0), 'pos_ki', 'tren', 5517, 0, tau=0.0823)
_b('B43', 'V6:83', 'PID hop ngoai pos_ki, G=1.127 (kp_xy=21)', 'PID',
   dict(kp_xy=21.0, ki_xy=0.0, pos_kp=0.0), 'pos_ki', 'tren', 5220, 0, G=1.127)
_b('B44', 'V6:83', 'PID hop ngoai pos_ki, G=0.984 (kp_xy=24)', 'PID',
   dict(kp_xy=24.0, ki_xy=0.0, pos_kp=0.0), 'pos_ki', 'tren', 5254, 0, G=0.984)

# a.4 LADRC
for _wc, _doc in ((0.5, 3574), (1, 2006), (2, 1204), (4, 784), (8, 560), (15, 446), (25, 390), (50, 345)):
    _b(f'B50_{_wc}', 'V6:87', f'LADRC bien pos_kp, omega_o=30, omega_c={_wc}, pos_ki=0', 'ADRC',
       dict(omega_c_xy=float(_wc), omega_o_xy=30.0, pos_ki=0.0), 'pos_kp', 'tren', _doc, 0)
_b('B51', 'V6:88', 'LADRC bien pos_kp, omega_o=95, omega_c=0.5 (dau duoi 122)', 'ADRC',
   dict(omega_c_xy=0.5, omega_o_xy=95.0, pos_ki=0.0), 'pos_kp', 'tren', 122, 0)
_b('B52', 'V6:88', 'LADRC bien pos_kp, omega_o=95, omega_c=8 (dau tren 133)', 'ADRC',
   dict(omega_c_xy=8.0, omega_o_xy=95.0, pos_ki=0.0), 'pos_kp', 'tren', 133, 0)
_b('B53', 'V6:89', 'LADRC pos_ki dat 4.0e4 tai omega_c=100, omega_o=45, pos_kp=170', 'ADRC',
   dict(omega_c_xy=100.0, omega_o_xy=45.0, pos_kp=170.0), 'pos_ki', 'tren', 4.0e4, -3)

# -- "none": no stable value of a gain: (id, source, what, ctrl, fixed gains, gain, values tried) --
_KI = [0.0] + list(np.geomspace(1e-3, 1e5, 61))
_KP = [0.0] + list(np.geomspace(1e-2, 1e4, 121))
_WC = list(np.geomspace(1e-3, 400.0, 101))
KHONG_CO = [
    ('N1', 'V6:62', 'PID pos_kp=200: khong co pos_ki on dinh', 'PID', dict(pos_kp=200.0), 'pos_ki', _KI),
    ('N2', 'V6:62', 'PID pos_kp=400: khong co pos_ki on dinh', 'PID', dict(pos_kp=400.0), 'pos_ki', _KI),
    ('N3', 'V6:67', 'PID pos_ki=1000: khong co pos_kp on dinh', 'PID', dict(pos_ki=1000.0), 'pos_kp', _KP),
    ('N4', 'V6:67', 'PID pos_ki=3000: khong co pos_kp on dinh', 'PID', dict(pos_ki=3000.0), 'pos_kp', _KP),
    ('N5', 'V4:112', 'LADRC bo vong vi tri, omega_o=100: khong co omega_c on dinh', 'ADRC',
     dict(omega_o_xy=100.0, **TAT_VI_TRI), 'omega_c_xy', _WC),
    ('N6', 'V4:112', 'LADRC bo vong vi tri, omega_o=120: khong co omega_c on dinh', 'ADRC',
     dict(omega_o_xy=120.0, **TAT_VI_TRI), 'omega_c_xy', _WC),
]


def _rho(row):
    ma, nguon, mo_ta, ctrl, g, doc, dec, kw = row
    return mh.ban_kinh_pho(ctrl, bo(ctrl, **g), **kw)


def _bien(row):
    ma, nguon, mo_ta, ctrl, g, ten, huong, doc, dec, kw = row
    return bien_gain(ctrl, bo(ctrl, **g), ten, huong, doc, dec, **kw)


@pytest.mark.parametrize('row', RHO, ids=[r[0] for r in RHO])
def test_so_kiem_ban_kinh_pho(row):
    value, doc, dec = _rho(row), row[5], row[6]
    assert khop(value, doc, dec), (row[:3], value)
    assert mh.trang_thai(value) == (mh.KHONG_ON_DINH if doc >= 1 else mh.ON_DINH)


@pytest.mark.parametrize('row', DINH_TINH, ids=[r[0] for r in DINH_TINH])
def test_so_kiem_on_dinh_hay_khong(row):
    ma, nguon, mo_ta, ctrl, g, stable, kw = row
    rho = mh.ban_kinh_pho(ctrl, bo(ctrl, **g), **kw)
    assert (rho < 1.0) == stable, (row[:3], rho)
    assert mh.trang_thai(rho) == (mh.ON_DINH if stable else mh.KHONG_ON_DINH)


@pytest.mark.parametrize('row', BIEN, ids=[r[0] for r in BIEN])
def test_so_kiem_bien(row):
    value, doc, dec = _bien(row), row[7], row[8]
    assert khop(value, doc, dec), (row[:3], value)


@pytest.mark.parametrize('row', KHONG_CO, ids=[r[0] for r in KHONG_CO])
def test_so_kiem_khong_co_gia_tri_on_dinh(row):
    ma, nguon, mo_ta, ctrl, g, ten, luoi = row
    assert not any(on_dinh(ctrl, bo(ctrl, **dict(g, **{ten: q}))) for q in luoi), row[:3]


def test_so_kiem_cuc_dai_va_ti_so():
    def ki_max(ctrl, **g):
        return bien_gain(ctrl, bo(ctrl, **g), 'pos_ki', 'tren', {'PID': 710, 'ADRC': 2050}[ctrl], 0)
    # V6:65: the largest pos_ki is at pos_kp ~ 90 (PID) and ~ 325 (LADRC; 25 apart: 300 and 350 are lower)
    assert ki_max('PID', pos_kp=80.0) < ki_max('PID', pos_kp=90.0) > ki_max('PID', pos_kp=100.0)
    assert ki_max('ADRC', pos_kp=300.0) < ki_max('ADRC', pos_kp=325.0) > ki_max('ADRC', pos_kp=350.0)

    # V6:83: the outer box of pos_ki is at kp_xy = 23, pos_kp near 0
    def hop(kp_xy, pos_kp=0.0):
        return bien_gain('PID', bo('PID', kp_xy=kp_xy, ki_xy=0.0, pos_kp=pos_kp), 'pos_ki', 'tren', 5265, 0)
    assert max(hop(22.0), hop(24.0), hop(23.0, 1.0)) < hop(23.0)
    # V6:111: the default (6; 3) is 31 times (PID) and 11.6 times (LADRC) below the pos_ki limit
    assert khop(bien_gain('PID', bo('PID'), 'pos_ki', 'tren', 93, 0) / 3.0, 31, 0)
    assert khop(bien_gain('ADRC', bo('ADRC'), 'pos_ki', 'tren', 35, 0) / 3.0, 11.6, 1)
    # V6:112: 4e4 is 7.6 times the PID outer box of pos_ki
    assert khop(4.0e4 / hop(23.0), 7.6, 1)


def test_so_kiem_tai_lieu_lech():
    """Numbers of the documents the model does not give as printed; the cause is in BAO_CAO.md."""
    # V4:113 prints 0.75 without saying which G: that is G = 1.127 (row R5); at G = 1 it is 0.713
    rho = mh.ban_kinh_pho('ADRC', bo('ADRC', omega_c_xy=50.0, omega_o_xy=49.0, **TAT_VI_TRI))
    assert khop(rho, 0.713, 3) and not khop(rho, 0.75, 2)
    # V6:149 prints the triangle as 180 / 140 / 102; the model gives 101 at omega_o = 49 (V4:112, row
    # B08_49), 102 = 200 - 2*49 is the triangle formula
    wc = bien_gain('ADRC', bo('ADRC', omega_o_xy=49.0, **TAT_VI_TRI), 'omega_c_xy', 'tren', 101, 0)
    assert khop(wc, 101.5, 1) and not khop(wc, 102, 0)
    # V6:81 "lon nhat 32 khi ki_xy = 600" is the largest on the kp_xy grid of a.4 (row B32, kp_xy = 15);
    # between the grid points the limit is higher: 33.7 at kp_xy = 13
    b13 = bien_gain('PID', bo('PID', kp_xy=13.0, ki_xy=600.0, pos_ki=0.0), 'pos_kp', 'tren', 33.7, 1)
    assert khop(b13, 33.7, 1)
    # V6:54 "trong dai tau +-15 %": at exactly +15 % (0.19665 s) the limit is 528, the printed 527 is at
    # tau = 0.197 (row B17)
    b = bien_gain('ADRC', bo('ADRC', pos_ki=0.0), 'pos_kp', 'tren', 528, 0, tau=0.171 * 1.15)
    assert khop(b, 528, 0)


# -- the matrix against a time simulation written in the order of the firmware code (V6:37) --
def mo_phong(ctrl, g, s0, n, G=1.0, tau=0.171, b0=5.85):
    """n ticks of one axis, float64, no saturation, reference 0. g: plain floats. s0 and every row of the
    result: the state in the order of mh.TEN[ctrl]. The plant is the exact solution of
    tau*dv/dt = -v + G*u over one tick with u held (not written with alpha)."""
    T = 0.01
    a = math.exp(-T / tau)
    if ctrl == 'PID':
        x, v, i_pos, i_vel, u = s0
    else:
        x, v, i_pos, z1, z2, u = s0
    out = [tuple(s0)]
    for _ in range(n):
        dx = G * u * T + (v - G * u) * tau * (1.0 - a)      # wheel travel over the tick
        v = G * u + (v - G * u) * a
        v_enc = dx / T                                      # encoder.c:122-131: count difference / dt
        x += v_enc * T                                      # pos_ctrl.c:103
        e = 0.0 - x                                         # pos_ctrl.c:108
        i_pos += e * T                                      # pos_ctrl.c:50, 52
        v_cmd = 0.0 + g['pos_kp'] * e + g['pos_ki'] * i_pos    # pos_ctrl.c:53; vel_ctrl.c:132
        if ctrl == 'PID':
            ev = v_cmd - v_enc                              # pid.c:35
            i_vel += ev * T                                 # pid.c:41
            u = v_cmd + g['kp_xy'] * ev + g['ki_xy'] * i_vel    # vel_ctrl.c:168; pid.c:55
            out.append((x, v, i_pos, i_vel, u))
        else:
            err = v_enc - z1                                # adrc.c:50
            dz1 = z2 + 2.0 * g['omega_o_xy'] * err + b0 * u    # adrc.c:53 (u of the tick before)
            dz2 = g['omega_o_xy'] ** 2 * err                # adrc.c:54
            z1 += dz1 * T                                   # adrc.c:56
            z2 += dz2 * T                                   # adrc.c:57
            u = (g['omega_c_xy'] * (v_cmd - z1) - z2) / b0  # adrc.c:80-81
            out.append((x, v, i_pos, z1, z2, u))
    return np.array(out)


@pytest.mark.parametrize('ctrl,g,kw', [
    ('PID', {}, {}),
    ('ADRC', {}, {}),
    ('PID', dict(kp_xy=7.0, ki_xy=40.0, pos_kp=25.0, pos_ki=60.0), dict(G=1.127, tau=0.0823)),
    ('ADRC', dict(omega_c_xy=40.0, omega_o_xy=20.0, pos_kp=90.0, pos_ki=200.0), dict(G=0.9, tau=0.197)),
    ('PID', dict(pos_ki=0.0), {}),                            # i_pos left out
    ('PID', dict(ki_xy=0.0), {}),                             # i_vel left out
    ('PID', dict(ki_xy=0.0, pos_kp=0.0, pos_ki=0.0), {}),     # velocity loop alone: v, u
    ('ADRC', dict(pos_ki=0.0), {}),
    ('ADRC', dict(pos_kp=0.0, pos_ki=0.0), {}),
])
def test_ma_tran_khop_mo_phong_theo_trinh_tu_ma(ctrl, g, kw):
    gains = bo(ctrl, **g)
    A, ten = mh.ma_tran(ctrl, gains, **kw)
    giu = [mh.TEN[ctrl].index(n) for n in ten]
    s0 = np.random.default_rng(20261122).normal(size=len(mh.TEN[ctrl]))
    S = mo_phong(ctrl, mh.chuan_hoa(ctrl, gains), s0, 200, **kw)     # the float32 gains the matrix uses
    s = s0[giu]
    for k in range(1, 201):
        s = A @ s
        assert np.allclose(s, S[k][giu], rtol=1e-9, atol=1e-9 * np.abs(S).max()), k


@pytest.mark.parametrize('ctrl,pos_kp,tang', [('PID', 184.0, False), ('PID', 186.0, True),
                                              ('ADRC', 555.0, False), ('ADRC', 565.0, True)])
@pytest.mark.parametrize('pos_ki', [0.0, 3.0])
def test_mo_phong_hoi_tu_phan_ky_V6_37(ctrl, pos_kp, tang, pos_ki):
    """V6:37: the time simulation converges at pos_kp 184 and diverges at 186 (PID), 555 and 565 (LADRC)."""
    g = bo(ctrl, pos_kp=pos_kp, pos_ki=pos_ki)
    s0 = [0.0, 0.01] + [0.0] * (len(mh.TEN[ctrl]) - 2)
    x = np.abs(mo_phong(ctrl, g, s0, 20000)[:, 0])
    dau, cuoi = x[:2000].max(), x[-2000:].max()
    assert (cuoi > 10.0 * dau) if tang else (cuoi < 0.01 * dau), (dau, cuoi)
    assert on_dinh(ctrl, g) == (not tang)


# -- interface of the module and the CLI -----------------------------------------
def f32(x):
    """Nearest float32 as a Python float."""
    return struct.unpack('<f', struct.pack('<f', x))[0]


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, 'tools')
FW_CTRL_DIR = 'D:/CLAUDE_THESIS/03_CODE/AGV_FIRMWARE/src/controller'
needs_fw = pytest.mark.skipif(not os.path.isfile(os.path.join(FW_CTRL_DIR, 'pid.c')),
                              reason='firmware tree not checked out')


def test_cau_truc_mo_dun():
    """State names, matrix shape, constants."""
    assert mh.TEN == {'PID': ('x', 'v', 'i_pos', 'i_vel', 'u'),
                      'ADRC': ('x', 'v', 'i_pos', 'z1', 'z2', 'u')}
    A, ten = mh.ma_tran('PID', PID0)
    assert A.shape == (5, 5) and ten == mh.TEN['PID']
    A, ten = mh.ma_tran('ADRC', ADRC0)
    assert A.shape == (6, 6) and ten == mh.TEN['ADRC']
    assert mh.T_TICK == 0.01 and mh.TAU == 0.171 and mh.G_MAC_DINH == 1.0 and mh.B0_XY == 5.85
    assert mh.ON_DINH == 'on_dinh' and mh.KHONG_ON_DINH == 'khong_on_dinh'


BO_TICH_PHAN = [
    ('PID', dict(pos_ki=0.0), ('x', 'v', 'i_vel', 'u')),
    ('PID', dict(pos_ki=-0.0), ('x', 'v', 'i_vel', 'u')),
    ('PID', dict(ki_xy=0.0), ('x', 'v', 'i_pos', 'u')),
    ('PID', dict(pos_ki=0.0, ki_xy=0.0), ('x', 'v', 'u')),
    ('PID', dict(pos_kp=0.0, pos_ki=0.0), ('v', 'i_vel', 'u')),
    ('PID', dict(pos_kp=0.0, pos_ki=0.0, ki_xy=0.0), ('v', 'u')),
    ('PID', dict(pos_kp=0.0, pos_ki=3.0), ('x', 'v', 'i_pos', 'i_vel', 'u')),
    ('PID', dict(pos_ki=1e-30), ('x', 'v', 'i_pos', 'i_vel', 'u')),
    ('ADRC', dict(pos_ki=0.0), ('x', 'v', 'z1', 'z2', 'u')),
    ('ADRC', dict(pos_kp=0.0, pos_ki=0.0), ('v', 'z1', 'z2', 'u')),
]


@pytest.mark.parametrize('ctrl,kw,names', BO_TICH_PHAN,
                         ids=[f'{c}-{"_".join(sorted(k))}' for c, k, n in BO_TICH_PHAN])
def test_bo_tich_phan_gain_0(ctrl, kw, names):
    """An integrator whose gain is exactly 0 does not act on the loop: it is left out."""
    A, ten = mh.ma_tran(ctrl, bo(ctrl, **kw))
    assert ten == names and A.shape == (len(names),) * 2


def test_pos_ki_nho_van_giai_duoc():
    """pos_ki = 1.2e-5 > 0 stays in the model and the loop is still resolved (rho < 1)."""
    for ctrl in ('PID', 'ADRC'):
        rho = mh.ban_kinh_pho(ctrl, bo(ctrl, pos_ki=1.2e-5))
        assert rho < 1.0 and mh.trang_thai(rho) == mh.ON_DINH


def test_truc_wz_va_vong_heading_khong_trong_mo_hinh():
    """The wz axis and the heading loop are not in the sub-model: changing their gains changes nothing."""
    A0, _ = mh.ma_tran('PID', PID0)
    g1 = bo('PID', kp_wz=4.6, ki_wz=230.0, k_psi=50.0)
    A1, _ = mh.ma_tran('PID', g1)
    assert np.array_equal(A0, A1)
    assert mh.ban_kinh_pho('PID', g1) == mh.ban_kinh_pho('PID', PID0)
    A0, _ = mh.ma_tran('ADRC', ADRC0)
    g1 = bo('ADRC', omega_c_wz=200.0, omega_o_wz=100.0, k_psi=50.0)
    A1, _ = mh.ma_tran('ADRC', g1)
    assert np.array_equal(A0, A1)
    assert mh.ban_kinh_pho('ADRC', g1) == mh.ban_kinh_pho('ADRC', ADRC0)


def test_biet_danh_omega():
    """The wc_*/wo_* aliases of gain_table name the same gains: same radius."""
    alias = dict(ADRC0)
    for a, n in (('wc_xy', 'omega_c_xy'), ('wo_xy', 'omega_o_xy'),
                 ('wc_wz', 'omega_c_wz'), ('wo_wz', 'omega_o_wz')):
        alias[a] = alias.pop(n)
    assert mh.ban_kinh_pho('ADRC', alias) == mh.ban_kinh_pho('ADRC', ADRC0)


def test_gain_lam_tron_float32():
    """chuan_hoa gives the float32 values in NAMES order; the matrix is built on the rounded set."""
    g = mh.chuan_hoa('PID', PID0)
    assert list(g) == gt.NAMES['PID']
    assert g['ki_xy'] == f32(0.05) and g['ki_xy'] != 0.05
    assert mh.ban_kinh_pho('PID', bo('PID', ki_xy=f32(0.05))) == mh.ban_kinh_pho('PID', PID0)
    # a numeric string is accepted, like in gain_table
    assert mh.ban_kinh_pho('PID', bo('PID', kp_xy='1.5')) == mh.ban_kinh_pho('PID', PID0)


@pytest.mark.parametrize('ctrl', ['LADRC', 'pid', None, 5])
def test_tu_choi_ctrl(ctrl):
    """Not PID or ADRC."""
    for f in (mh.ban_kinh_pho, mh.ma_tran):
        with pytest.raises(ValueError):
            f(ctrl, PID0)


@pytest.mark.parametrize('gains', [None, [1.0] * 7], ids=['None', 'list'])
def test_tu_choi_gains_khong_phai_object(gains):
    """gains that is not an object."""
    for f in (mh.ban_kinh_pho, mh.ma_tran):
        with pytest.raises(ValueError):
            f('PID', gains)


@pytest.mark.parametrize('ctrl,gains', [
    ('PID', {k: v for k, v in PID0.items() if k != 'k_psi'}),
    ('PID', dict(PID0, kd_xy=0.0)),
    ('PID', ADRC0),
    ('ADRC', dict(ADRC0, wc_xy=8.0)),           # alias and full name both given
], ids=['thieu_k_psi', 'thua_kd_xy', 'gains_adrc_cho_pid', 'ca_ten_va_biet_danh'])
def test_tu_choi_ten_gain(ctrl, gains):
    """Wrong names: one missing, one extra, the set of the other controller, alias + full name."""
    for f in (mh.ban_kinh_pho, mh.ma_tran):
        with pytest.raises(ValueError):
            f(ctrl, gains)


@pytest.mark.parametrize('gain', ['pos_kp', 'k_psi'])
@pytest.mark.parametrize('x', ['abc', None, True, float('nan'), float('inf'), -float('inf'), 1e100, [1.0]],
                         ids=['abc', 'None', 'True', 'nan', 'inf', '-inf', '1e100', 'list'])
def test_tu_choi_gia_tri_gain(gain, x):
    """A value that is not a finite float32, on a used gain (pos_kp) and on an unused one (k_psi)."""
    for f in (mh.ban_kinh_pho, mh.ma_tran):
        with pytest.raises(ValueError):
            f('PID', bo('PID', **{gain: x}))


@pytest.mark.parametrize('kw', [dict(G=0), dict(G=-1.0), dict(G=float('nan')), dict(G=float('inf')),
                                dict(G=True), dict(G='1'), dict(G=10 ** 400),
                                dict(tau=0), dict(tau=-0.171), dict(tau=float('nan')),
                                dict(tau=float('inf')), dict(tau=True), dict(tau=None)],
                         ids=['G0', 'G-1', 'Gnan', 'Ginf', 'GTrue', 'Gstr', 'G1e400',
                              'tau0', 'tau-0.171', 'taunan', 'tauinf', 'tauTrue', 'tauNone'])
def test_tu_choi_G_tau(kw):
    """G and tau must be finite numbers > 0."""
    for f in (mh.ban_kinh_pho, mh.ma_tran):
        with pytest.raises(ValueError):
            f('PID', PID0, **kw)


def test_nguong_trang_thai():
    """rho < 1 on_dinh, rho >= 1 khong_on_dinh, the float64 compared with 1 as it is."""
    assert mh.trang_thai(1.0) == mh.KHONG_ON_DINH
    assert mh.trang_thai(math.nextafter(1.0, 0.0)) == mh.ON_DINH
    assert mh.trang_thai(math.nextafter(1.0, 2.0)) == mh.KHONG_ON_DINH
    assert mh.trang_thai(0.0) == mh.ON_DINH
    assert mh.trang_thai(1) == mh.KHONG_ON_DINH
    assert mh.trang_thai(np.float64(0.5)) == mh.ON_DINH
    for bad in (float('nan'), float('inf'), -float('inf'), True, '1.0', None):
        with pytest.raises(ValueError):
            mh.trang_thai(bad)


def test_nguong_dung_1_tu_mo_hinh():
    """omega_o_xy = 0 makes dz2 = 0 (adrc.c:54), so row z2 is a unit row and 1 is an exact eigenvalue."""
    rho = mh.ban_kinh_pho('ADRC', bo('ADRC', omega_o_xy=0.0, **TAT_VI_TRI))
    assert rho == 1.0 and mh.trang_thai(rho) == mh.KHONG_ON_DINH


def test_nguong_khong_co_dai_dung_sai(capsys):
    """Within 1e-9 of 1 there is no tolerance band; the CLI prints the float64 so it reads back."""
    g = bo('PID', pos_kp=185.0, pos_ki=0.0)
    f = lambda G: mh.ban_kinh_pho('PID', g, G=G)
    lo, hi = 1.0, 1.01
    assert f(lo) < 1.0 <= f(hi)
    while hi - lo > 1e-12:
        m = 0.5 * (lo + hi)
        if f(m) < 1.0:
            lo = m
        else:
            hi = m
    r_lo, r_hi = f(lo), f(hi)
    assert 0.0 < 1.0 - r_lo < 1e-9 and 0.0 <= r_hi - 1.0 < 1e-9
    assert mh.trang_thai(r_lo) == mh.ON_DINH and mh.trang_thai(r_hi) == mh.KHONG_ON_DINH
    # the radius itself rounded to float32 crosses the threshold; the state is not changed by it
    assert f32(r_lo) == 1.0
    capsys.readouterr()
    assert mh.main(['--ctrl', 'PID', '--gains', json.dumps(g), '--G', repr(lo)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert float(lines[4].split('=', 1)[1]) == r_lo and lines[-1] == 'trang_thai=on_dinh'


def test_gain_float32_qua_nguong():
    """A float64 gain a hair under the limit is stable unrounded; the float32 the firmware gets is not."""
    g = mh.chuan_hoa('PID', bo('PID', pos_ki=0.0))                    # float32 values
    def f(q):                                                         # q NOT rounded
        return mh._ban_kinh(mh._dung('PID', dict(g, pos_kp=q))[0]) < 1.0
    b = bien(f, 180.0, 190.0, 'tren')
    lo32 = f32(b)
    if lo32 > b:
        lo32 = float(np.nextafter(np.float32(lo32), np.float32(0.0)))
    hi32 = float(np.nextafter(np.float32(lo32), np.float32(np.inf)))
    mid = 0.5 * (lo32 + hi32)
    assert lo32 < b < hi32 and b != mid and f(lo32) and not f(hi32)
    q = 0.5 * (b + mid)                       # between the limit and the rounding midpoint
    q32 = f32(q)
    assert q32 == (hi32 if b > mid else lo32) and f(q) != f(q32)      # rounding changes the side
    rho = mh.ban_kinh_pho('PID', bo('PID', pos_kp=q, pos_ki=0.0))
    assert rho == mh.ban_kinh_pho('PID', bo('PID', pos_kp=q32, pos_ki=0.0))
    assert (rho < 1.0) == f(q32)
    assert mh.trang_thai(rho) == (mh.ON_DINH if f(q32) else mh.KHONG_ON_DINH)


def test_tham_so_plant_tu_params_py():
    """tau and G as computed from rpi5/params.py (V3:65-66, V6:29); the matrix uses them."""
    spec = importlib.util.spec_from_file_location('params', os.path.join(ROOT, 'rpi5', 'params.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = mod.PARAMS
    num = float(p.J_wheels.sum() + p.M * p.r ** 2)
    assert abs(num - 0.018671) < 5e-7                                     # V3:65
    den = float((p.motor_K ** 2 / p.motor_R + p.motor_b).sum())
    assert abs(den - 0.10916) < 5e-6                                      # V3:65
    assert round(num / den, 3) == mh.TAU                                  # V3:65 "0,1710 s"; V6:29
    G = float(p.V_batt * (p.motor_K / p.motor_R).sum() / (34.56 * den))   # 34.56 = W_MAX, kinematics.c:6
    assert round(G, 3) == 1.127                                           # V3:65; V6:29
    a = math.exp(-0.01 / TAU_V3)
    alpha = TAU_V3 * (1 - a) / 0.01
    assert round(a, 5) == 0.94321 and round(alpha, 5) == 0.97133          # V3:66
    A, ten = mh.ma_tran('PID', bo('PID', kp_xy=2.0, ki_xy=0.0, **TAT_VI_TRI), tau=TAU_V3)
    assert ten == ('v', 'u')
    assert np.allclose(A, [[a, 1 - a], [-2.0 * alpha, -2.0 * (1 - alpha)]], rtol=1e-12, atol=1e-15)


@needs_fw
def test_gain_mac_dinh_trong_firmware():
    """The defaults written in the firmware sources are the defaults of the model."""
    def define(path, name):
        with open(path, encoding='utf-8') as f:
            m = re.search(r'#define\s+' + name + r'\s+([0-9.eE+-]+)f\b', f.read())
        return float(m.group(1))
    pid_c = os.path.join(FW_CTRL_DIR, 'pid.c')
    adrc_c = os.path.join(FW_CTRL_DIR, 'adrc.c')
    assert define(pid_c, 'PID_VX_KP') == PID0['kp_xy']
    assert define(pid_c, 'PID_VX_KI') == PID0['ki_xy']
    assert define(pid_c, 'PID_VX_KD') == 0.0
    assert define(adrc_c, 'ADRC_VXY_OMEGA_O') == ADRC0['omega_o_xy']
    assert define(adrc_c, 'ADRC_VXY_B0') == mh.B0_XY
    assert define(adrc_c, 'ADRC_VXY_KP') == ADRC0['omega_c_xy']
    with open(os.path.join(FW_CTRL_DIR, 'pos_ctrl.h'), encoding='utf-8') as f:
        m = re.search(r'POS_PARAMS_DEFAULT.*?\.kp\s*=\s*([0-9.eE+-]+)f.*?\.ki\s*=\s*([0-9.eE+-]+)f',
                      f.read(), re.S)
    assert float(m.group(1)) == PID0['pos_kp'] and float(m.group(2)) == PID0['pos_ki']


def _cli(argv, capsys):
    """Run main(argv); returns (exit code, stdout, stderr)."""
    capsys.readouterr()
    rc = mh.main(argv)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def _cli_tu_choi(argv, capsys):
    rc, out, err = _cli(argv, capsys)
    assert rc == 1 and out == '' and err.startswith('error: ') and err.isascii(), (argv, rc, out, err)


def test_cli_ok_pid(capsys):
    """Happy path: six ASCII lines on stdout, exit 0."""
    rc, out, err = _cli(['--ctrl', 'PID', '--gains', json.dumps(PID0)], capsys)
    assert rc == 0 and err == '' and out.isascii()
    assert out.splitlines() == ['ctrl=PID', 'G=1.0', 'tau=0.171',
                                'bien_trang_thai=x,v,i_pos,i_vel,u',
                                f'ban_kinh_pho={mh.ban_kinh_pho("PID", PID0)!r}',
                                'trang_thai=on_dinh']


def test_cli_khong_on_dinh_chi_bao(capsys):
    """(pos_kp; pos_ki) = (600; 300) is khong_on_dinh: reported, still exit 0."""
    g = bo('PID', pos_kp=600.0, pos_ki=300.0)
    rc, out, err = _cli(['--ctrl', 'PID', '--gains', json.dumps(g)], capsys)
    assert rc == 0 and err == ''
    lines = out.splitlines()
    assert lines[-1] == 'trang_thai=khong_on_dinh'
    rho = float(lines[4].split('=', 1)[1])
    assert rho >= 1.0 and khop(rho, 1.151, 3)


def test_cli_file_phang_va_file_gain(capsys, tmp_path):
    """--gains as a path: a flat object needs --ctrl; a gain file brings its own "ctrl"."""
    flat = tmp_path / 'phang.json'
    flat.write_text(json.dumps(ADRC0), encoding='utf-8')
    rc, out1, _ = _cli(['--ctrl', 'ADRC', '--gains', str(flat)], capsys)
    assert rc == 0 and 'bien_trang_thai=x,v,i_pos,z1,z2,u' in out1.splitlines()
    f_gain = tmp_path / 'gain_A.json'
    f_gain.write_text(json.dumps({'ctrl': 'ADRC', 'id': 3, 'gains': ADRC0}), encoding='utf-8')
    rc, out2, _ = _cli(['--gains', str(f_gain)], capsys)
    assert rc == 0 and out2 == out1
    rc, out3, _ = _cli(['--ctrl', 'ADRC', '--gains', str(f_gain)], capsys)
    assert rc == 0 and out3 == out1


def test_cli_G_tau(capsys):
    """--G and --tau reach the model."""
    rc, out, _ = _cli(['--ctrl', 'PID', '--gains', json.dumps(PID0), '--G', '1.127', '--tau', '0.197'],
                      capsys)
    lines = out.splitlines()
    assert rc == 0 and lines[1] == 'G=1.127' and lines[2] == 'tau=0.197'
    assert float(lines[4].split('=', 1)[1]) == mh.ban_kinh_pho('PID', PID0, G=1.127, tau=0.197)


def test_cli_bo_i_pos(capsys):
    """pos_ki = 0 over the CLI: i_pos is left out of the state line."""
    rc, out, _ = _cli(['--ctrl', 'PID', '--gains', json.dumps(bo('PID', pos_ki=0))], capsys)
    assert rc == 0 and 'bien_trang_thai=x,v,i_vel,u' in out.splitlines()


def test_cli_tu_choi(capsys, tmp_path):
    """Each refused input: exit 1, nothing on stdout, an ASCII 'error: ...' line on stderr."""
    f_pid = tmp_path / 'pid.json'
    f_pid.write_text(json.dumps({'ctrl': 'PID', 'gains': PID0}), encoding='utf-8')
    f_arr = tmp_path / 'mang.json'
    f_arr.write_text('[1, 2]', encoding='utf-8')
    dup = ('{"kp_xy": 1, "kp_xy": 2, "ki_xy": 0.05, "kp_wz": 1, "ki_wz": 0.05,'
           ' "pos_kp": 6, "pos_ki": 3, "k_psi": 1}')
    nan_js = ('{"kp_xy": NaN, "ki_xy": 0.05, "kp_wz": 1, "ki_wz": 0.05,'
              ' "pos_kp": 6, "pos_ki": 3, "k_psi": 1}')
    ladrc = '{"ctrl": "LADRC", "gains": ' + json.dumps(ADRC0) + '}'
    khong_psi = json.dumps({k: v for k, v in PID0.items() if k != 'k_psi'})
    cases = [
        ['--ctrl', 'ADRC', '--gains', str(f_pid)],                      # "ctrl" of the file != --ctrl
        ['--gains', json.dumps(PID0)],                                   # flat JSON, no --ctrl
        ['--ctrl', 'PID', '--gains', '{"kp_xy": 1.5'],                   # broken JSON
        ['--ctrl', 'PID', '--gains', str(tmp_path / 'khong_co.json')],   # file does not exist
        ['--ctrl', 'PID', '--gains', dup],                               # a JSON key given twice
        ['--gains', str(f_arr)],                                         # top level not an object
        ['--gains', '{"ctrl": "PID", "gains": 5}'],                      # "gains" not an object
        ['--ctrl', 'PID', '--gains', khong_psi],                         # a gain name missing
        ['--ctrl', 'PID', '--gains', nan_js],                            # NaN in the JSON
        ['--ctrl', 'PID', '--gains', json.dumps(dict(PID0, pos_kp='abc'))],   # a string value
        ['--ctrl', 'PID', '--gains', json.dumps(PID0), '--G', '0'],
        ['--ctrl', 'PID', '--gains', json.dumps(PID0), '--tau', '-1'],
        ['--ctrl', 'PID', '--gains', json.dumps(PID0), '--G', 'nan'],
        ['--gains', ladrc],                                              # "ctrl" not PID|ADRC, no --ctrl
        ['--ctrl', 'PID', '--gains', str(tmp_path / 'kh\u00f4ng_c\u00f3.json')],   # non-ASCII name
    ]
    for argv in cases:
        _cli_tu_choi(argv, capsys)


def test_cli_argparse_thoat():
    """argparse itself refuses a bad --ctrl choice and a missing --gains (SystemExit 2)."""
    with pytest.raises(SystemExit) as e1:
        mh.main(['--ctrl', 'LADRC', '--gains', json.dumps(PID0)])
    with pytest.raises(SystemExit) as e2:
        mh.main(['--ctrl', 'PID'])
    assert e1.value.code == 2 and e2.value.code == 2


def test_cli_nhu_script_va_module(capsys):
    """python tools/mo_hinh_con.py and python -m tools.mo_hinh_con print the same lines as main()."""
    argv = ['--ctrl', 'PID', '--gains', json.dumps(PID0)]
    rc, out, _ = _cli(argv, capsys)
    assert rc == 0
    for cmd in ([sys.executable, os.path.join(TOOLS, 'mo_hinh_con.py')],
                [sys.executable, '-m', 'tools.mo_hinh_con']):
        r = subprocess.run(cmd + argv, capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0 and r.stdout.splitlines() == out.splitlines(), r.stderr


def test_hai_file_la_ascii():
    """The tool and this test file stay pure ASCII."""
    for path in (os.path.join(TOOLS, 'mo_hinh_con.py'), os.path.abspath(__file__)):
        with open(path, 'rb') as f:
            assert f.read().isascii(), path


def _bang():
    """The check-number table as ASCII lines (for BAO_CAO.md)."""
    out = ['ma | nguon | dai luong | tai lieu | tinh duoc | lech % | khop khi lam tron | lech > 0.5 % do tai lieu lam tron']
    for row in RHO + BIEN:
        doc, dec = row[-3], row[-2]
        value = _rho(row) if len(row) == 8 else _bien(row)
        lech = 100.0 * (value - doc) / doc
        out.append(f'{row[0]} | {row[1]} | {row[2]} | {doc:.{max(dec, 0)}f} | {value:.7g} | {lech:+.2f} | '
                   f'{"co" if khop(value, doc, dec) else "KHONG"} | {"x" if abs(lech) > 0.5 else ""}')
    for ma, nguon, mo_ta, ctrl, g, stable, kw in DINH_TINH:
        rho = mh.ban_kinh_pho(ctrl, bo(ctrl, **g), **kw)
        out.append(f'{ma} | {nguon} | {mo_ta} | {"< 1" if stable else ">= 1"} | {rho:.7g} | | '
                   f'{"co" if (rho < 1.0) == stable else "KHONG"} |')
    for ma, nguon, mo_ta, ctrl, g, ten, luoi in KHONG_CO:
        n = sum(on_dinh(ctrl, bo(ctrl, **dict(g, **{ten: q}))) for q in luoi)
        out.append(f'{ma} | {nguon} | {mo_ta} | khong | {n} gia tri on dinh / {len(luoi)} | | {"co" if n == 0 else "KHONG"} |')
    return out


if __name__ == '__main__':
    print('\n'.join(_bang()))
