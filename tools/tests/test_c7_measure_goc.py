"""Tests of the tag corner columns of c7_measure (CAM design v4 sec 2: tag_psi, c0_x .. c3_y after the parallax, for
c8b_a6; written from 22/11). SYNTHETIC camera and rows, not a measurement."""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c7_measure as cm  # noqa: E402


class FakeCam:
    """Floor = pixels (metres), for the corner bookkeeping only."""
    def pix2floor(self, px, z):
        return np.asarray(px, float)


def _corners(side, ang, c):
    b = cm.car_corners_body(side)
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    return b @ R.T + np.asarray(c)


def test_goc_ma_sau_thi_sai_khop_car_pose():
    side, ang, c = 0.1445, 0.6, (0.3, -0.2)
    px = _corners(side, ang, c)
    plumb, kappa = (-0.48, -0.03), 0.047
    out = cm.car_pose(FakeCam(), px, side, kappa, plumb)
    g = cm.tag_corners_floor(FakeCam(), px, kappa, plumb)
    P = np.array([[g[f'c{i}_x'], g[f'c{i}_y']] for i in range(4)])
    want = px + kappa * (np.asarray(plumb) - px)                           # the parallax step of car_pose
    assert np.allclose(P, want, atol=1e-12)
    assert np.allclose(P.mean(axis=0), [out['tag_x'], out['tag_y']], atol=1e-9)
    assert math.isclose(out['tag_psi'], ang, abs_tol=1e-9)


def _row(i, goc):
    r = {c: float(i) for c in cm.FRAME_COLS}
    r.update(idx=i, why=[], valid=True)
    if goc:
        r.update(tag_psi=0.1 * i, **{f'c{k}_{a}': 0.01 * (i + k) for k in range(4) for a in ('x', 'y')})
    return r


def test_write_frames_noi_cot_goc_o_cuoi_giu_nguyen_dau_dong(tmp_path):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    cm.write_frames(str(a), [_row(i, False) for i in range(3)])
    cm.write_frames(str(b), [_row(i, True) for i in range(3)])
    la, lb = a.read_bytes().splitlines(), b.read_bytes().splitlines()
    assert len(la) == len(lb)
    assert all(y.startswith(x + b',') for x, y in zip(la, lb))
    hdr = next(csv.reader([lb[0].decode()]))
    assert tuple(hdr[-len(cm.GOC_COLS):]) == cm.GOC_COLS


def test_khong_co_goc_thi_khong_them_cot(tmp_path):
    a = tmp_path / 'a.csv'
    cm.write_frames(str(a), [_row(0, False)])
    assert 'c0_x' not in a.read_text(encoding='utf-8').splitlines()[0]
