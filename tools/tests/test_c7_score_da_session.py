"""Tests of c7_score.bind_session (several camera segments, CODE 20/11 khuya chain A): the F2 row of a run names the
session its camera comes from; the frames scored must be tracked with that session. SYNTHETIC values."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c7_score as sc  # noqa: E402

A, B = 'a' * 64, 'b' * 64


def test_cung_session_qua():
    sc.bind_session(dict(luot_id='C8-P-F02-1', session_sha256=A), dict(session_sha256=A))


def test_khac_session_bi_tu_choi():
    with pytest.raises(ValueError, match='another camera segment'):
        sc.bind_session(dict(luot_id='C8-L-F02-1', session_sha256=B), dict(session_sha256=A))


@pytest.mark.parametrize('track', [None, {}, dict(session_sha256=None)])
def test_dong_co_session_ma_track_khong_co_bi_tu_choi(track):
    with pytest.raises(ValueError, match='no track.json'):
        sc.bind_session(dict(luot_id='C8-P-K1-1', session_sha256=A), track)


@pytest.mark.parametrize('track', [None, dict(session_sha256=A)])
def test_dong_cu_khong_session_khong_kiem(track):
    sc.bind_session(dict(luot_id='O1-F02'), track)
