"""Square-shape gate after ArUco detection (c7_clip_probe.detect_markers; used by c7_measure and
c7_usb_fetch). Real data: clip IMG_3550 (13/11, 24 floor marks, camera straight down) had 5 false
"ID 17" detections on marble veins in 1883 frames, all thin quads. The two fixtures are crops of
frames 845 and 1285 of that clip: left = the real ID 17, right = the vein the raw detector also
decodes as ID 17 (crops made 13/11, CODE_CAM; the raw detector still finds both on the crop)."""
import os
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import c7_clip_probe as cp  # noqa: E402
import c7_measure as cm  # noqa: E402

FIX = [os.path.join(HERE, 'data', f) for f in ('c7_false_id17_f845.png', 'c7_false_id17_f1285.png')]


def test_is_squareish():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert cp.is_squareish(sq)
    assert cp.is_squareish(sq * [1.0, 0.8])                       # mild perspective / aspect
    assert not cp.is_squareish(sq * [1.0, 0.3])                   # thin
    sliver = np.array([[0, 0], [40, -6], [30, 0], [2, 2]], float)
    assert not cp.is_squareish(sliver)
    assert not cp.is_squareish(np.zeros((4, 2)))


@pytest.mark.parametrize('path', FIX)
def test_false_id17_on_marble_dropped(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    det = cm.make_detector()
    _, ids, _ = det.detectMarkers(img)
    assert sorted(ids.ravel().tolist()) == [17, 17]               # the raw detector is fooled
    stats = {}
    got = cp.detect_markers(det, img, stats)
    assert list(got) == [17] and stats == {'not_square': 1}
    assert got[17].mean(axis=0)[0] < 220                           # the kept one is the real tag (left)
    r = cm.detect_refined(det, img)
    assert list(r) == [17] and r[17].mean(axis=0)[0] < 220


def test_same_id_twice_is_dropped():
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cp.DICT), 5, 60)
    img = np.full((120, 260), 255, np.uint8)
    img[30:90, 30:90] = tag
    img[30:90, 170:230] = tag
    stats = {}
    assert cp.detect_markers(cm.make_detector(), img, stats) == {}
    assert stats == {'dup_dropped': 1}
