"""c8_hc_vet.py (CODE_HIEUCHUAN 19/11): swept footprint of 12b / E against the 32 floor tags."""
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import c8_hc_vet as cv  # noqa: E402

MOC = os.path.join('D:' + os.sep, 'CLAUDE_THESIS', '05_DATA', 'results', 'C7_VIDEO',
                   '2026-11-18_do_tam_moc_giua', 'moc_32', 'moc_san_do_32.csv')
MOC = os.environ.get('C8_MOC_32', MOC)


def tags_or_skip(vien=0.0):
    if not os.path.isfile(MOC):
        pytest.skip('moc_san_do_32.csv not reachable (set C8_MOC_32)')
    return cv.load_tags(MOC, vien)


def test_poly_and_disc_distance():
    a = cv.rect(0, 1, 0, 1)
    assert cv.poly_dist(a, cv.rect(2, 3, 0, 1)) == pytest.approx(1.0)
    assert cv.poly_dist(a, cv.rect(0.5, 3, 0.5, 1)) == 0.0
    assert cv.poly_dist(a, cv.rect(2, 3, 2, 3)) == pytest.approx(math.sqrt(2))
    assert cv.disc_dist((0.5, 0.5), 0.1, a) == 0.0                 # centre inside
    assert cv.disc_dist((3.0, 0.5), 0.5, a) == pytest.approx(1.5)


def test_pieces_wheels_stick_out_when_340_across():
    doc, ngang = cv.pieces('340_doc'), cv.pieces('340_ngang')
    assert max(p[:, 0].max() for p in doc[1:]) <= doc[0][:, 0].max()          # wheels inside the body (340 along x)
    assert max(p[:, 0].max() for p in ngang[1:]) > ngang[0][:, 0].max()       # 0.1685 > 0.145 (Codex HC2-7)
    assert cv.reach('340_doc') == pytest.approx(math.hypot(0.17, 0.145))


def test_swept_box_covers_motion():
    p = cv.rect(-0.1, 0.1, -0.05, 0.05)
    s = cv.swept_x(p, -0.3, 0.2, 0.002)
    assert s[:, 0].min() == pytest.approx(-0.4) and s[:, 0].max() == pytest.approx(0.3)
    assert s[:, 1].max() == pytest.approx(0.052)


def test_load_tags_32_and_growth():
    t0, t1 = tags_or_skip(0.0), tags_or_skip(0.01)
    assert len(t0) == 32
    w0 = t0[25][:, 0].max() - t0[25][:, 0].min()
    w1 = t1[25][:, 0].max() - t1[25][:, 0].min()
    assert w1 - w0 == pytest.approx(0.02)


def test_k1_040_touches_and_025_clears():
    tags = tags_or_skip(0.0)
    old = cv.hs.PLACES['K1']
    try:
        cv.hs.PLACES['K1'] = ((0.0, 0.40, -0.40, 0.40, -0.40, 0.40, 0.0), True)
        _, w = cv.sweep_12b('K1', '340_doc', tags, 0.03, 3.0, 0.01, 0.0)
        assert w[0] == 0.0 and w[1] in (29, 30)                                 # Codex HC1-5: +-0.4 m touches
    finally:
        cv.hs.PLACES['K1'] = old
    n, w = cv.sweep_12b('K1', '340_doc', tags, 0.03, 3.0, 0.01, 0.0)
    assert w[0] > 0.04 and n[1] == 30


def test_vet_thuc_flags_a_pose_on_a_tag():
    tags = tags_or_skip(0.0)
    assert cv.vet_thuc([(0.499, 0.0, 0.0)], tags, '340_doc')[0] == 0.0           # body over tag 25
    g, tid = cv.vet_thuc([(0.0, 0.0, math.radians(45))], tags, '340_doc')
    assert g > 0.1
