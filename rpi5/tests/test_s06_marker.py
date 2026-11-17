"""S06 T3 marker pin (design 2026-11-15_Thiet_ke_S06_T3_danh_dau_v1 section 4.1; CODE 15/11: only _tick, the
default path must keep its behaviour): marker_due, the order rise -> transfer -> fall, the frame log, the
manifest; the default mode keeps one pulse() before get_state on every frame and no marker array.
Also rpi5/t3_schedule.py: the schedule checks (spec S06 T3, Z7, L7-10) and the H7 half period."""
import json
import multiprocessing as mp

import numpy as np
import pytest

import hil_node
from hil_node import HILConfig, HILNode, marker_due, MARKER_EVERY
from plant_interface import MockPlant
from protocol import encode_state
import t3_schedule as ts


def _mosi(w):
    return encode_state(list(w) + [0.0] * 6, seq=0)


# ── marker_due ─────────────────────────────────────────────────────────────────

def test_marker_due_default_never():
    assert not marker_due('', 5, _mosi([2, 2, 2, 2]), _mosi([39.2, 2, 2, 2]))


def test_marker_due_10hz_every_100th():
    hits = [k for k in range(1000) if marker_due('10hz', k, None, _mosi([0, 0, 0, 0]))]
    assert hits == list(range(0, 1000, MARKER_EVERY))


@pytest.mark.parametrize('old,new,due', [
    ([2, 2, 2, 2], [39.2, 2, 2, 2], True),          # retime, same direction
    ([39.2, 5, 5, 5], [2, 5, 5, 5], True),          # down step
    ([-5, 5, 5, 5], [-10, 5, 5, 5], True),          # same direction, negative
    ([2, 2, 2, 2], [2, 2, 2, 2], False),            # no change
    ([0, 2, 2, 2], [5, 2, 2, 2], False),            # start from rest: _start_wheel, no PF3 edge
    ([5, 2, 2, 2], [0, 2, 2, 2], False),            # stop
    ([5, 2, 2, 2], [-5, 2, 2, 2], False),           # reversal: _stop + _start_wheel
    ([0.25, 2, 2, 2], [2, 2, 2, 2], False),         # old speed under ENC_OMEGA_MIN: a start
    ([2, 2, 2, 2], [0.29, 2, 2, 2], False),         # new speed under ENC_OMEGA_MIN: a stop
])
def test_marker_due_t3(old, new, due):
    assert marker_due('t3', 7, _mosi(old), _mosi(new)) is due


def test_marker_due_t3_first_frame():
    assert not marker_due('t3', 0, None, _mosi([39.2, 2, 2, 2]))


def test_bad_marker_mode_refused():
    with pytest.raises(ValueError):
        HILNode(MockPlant(), config=HILConfig(simulate_hw=True, max_steps=1, marker_mode='5hz'),
                shared_state=mp.Array('d', 10))


# ── _tick with a recording GPIO / SPI ──────────────────────────────────────────

class _ScriptPlant(MockPlant):
    """Wheel speeds of frame k from a list (the node reads get_state before step)."""

    def __init__(self, table):
        super().__init__()
        self._t = table
        self._k = 0

    def get_state(self):
        w = self._t[min(self._k, len(self._t) - 1)]
        return list(w) + [0.0] * 6

    def step(self, torques):
        self._k += 1
        return self.get_state()


def _run(tmp_path, table, mode):
    log = []
    plant = _ScriptPlant(table)
    orig_get = plant.get_state
    plant.get_state = lambda: (log.append('get_state'), orig_get())[1]
    cfg = HILConfig(simulate_hw=True, max_steps=len(table), period_us=1_000, marker_mode=mode,
                    frame_log=str(tmp_path / 'frames.npz'))
    node = HILNode(plant, config=cfg, shared_state=mp.Array('d', 10))
    g = node._gpio
    g.pulse = lambda: log.append('pulse')
    g.rise = lambda: log.append('rise')
    g.fall = lambda: log.append('fall')
    orig_tx = node._spi.transfer
    node._spi.transfer = lambda m: (log.append('transfer'), orig_tx(m))[1]
    node.run()
    z = np.load(tmp_path / 'frames.npz')
    return log, z, json.loads(str(z['manifest']))


TABLE = [[2.0] * 4] * 5 + [[39.2, 2, 2, 2]] * 5 + [[39.2, 5, 2, 2]] * 5 + [[0, 5, 2, 2]] * 5 + [[0, 5, -2, 2]] * 5


def test_default_mode_unchanged(tmp_path):
    """Default: one pulse() per frame, before get_state, no rise/fall, no marker array, manifest ''."""
    log, z, man = _run(tmp_path, TABLE, '')
    assert log.count('pulse') == len(TABLE) and 'rise' not in log and 'fall' not in log
    assert log[0] == 'pulse'
    ticks = ' '.join(log).split('pulse')[1:]
    assert all(t.strip().startswith('get_state') and 'transfer' in t for t in ticks)
    # CODE 17/11: the default frame log and manifest as before T3 (same arrays, no new manifest key)
    assert sorted(z.files) == ['manifest', 'miso', 'mosi', 't_pulse0_ns', 't_pulse1_ns', 't_recv_ns', 't_send_ns']
    assert 'marker_mode' not in man and 'marker_frames' not in man
    assert np.all(z['t_pulse1_ns'] >= z['t_pulse0_ns']) and np.all(z['t_pulse0_ns'] > 0)


def test_t3_mode_marks_retimes_only(tmp_path):
    log, z, man = _run(tmp_path, TABLE, 't3')
    assert 'pulse' not in log
    m = np.nonzero(z['marker'])[0].tolist()
    # frame k carries the state read at tick k: changes at frames 5 (2 -> 39.2) and 10 (2 -> 5); 15 is a
    # stop (39.2 -> 0) and 20 a reversal (2 -> -2): no marker
    assert m == [5, 10]
    assert man['marker_mode'] == 't3' and man['marker_frames'] == 2
    # rise right before the transfer, fall right after it
    for i, x in enumerate(log):
        if x == 'rise':
            assert log[i + 1] == 'transfer' and log[i + 2] == 'fall'
    assert log.count('rise') == log.count('fall') == 2
    # t_pulse0/1 bracket a point before t_send on every frame
    assert np.all(z['t_pulse0_ns'] <= z['t_pulse1_ns']) and np.all(z['t_pulse1_ns'] <= z['t_send_ns'])


def test_10hz_mode(tmp_path):
    table = [[2.0] * 4] * 250
    log, z, man = _run(tmp_path, table, '10hz')
    assert np.nonzero(z['marker'])[0].tolist() == [0, 100, 200]
    assert man['marker_frames'] == 3 and log.count('rise') == 3 and 'pulse' not in log


# ── t3_schedule ────────────────────────────────────────────────────────────────

def test_hp_ticks_matches_h7_formula():
    # 2.0 travels as int16 1638 -> 1.999573 rad/s on the H7: 1e6 * pi / (1.999573 * 330) = 4761.0
    assert ts.hp_ticks(2.0) == 4761
    assert ts.hp_ticks(39.2) == 243          # int16 32112 -> 39.2004: 242.86
    assert ts.hp_ticks(0.29) == 0            # stopped
    assert abs(ts.window_ms(2.0) - 10.522) < 1e-9


def test_locked_schedule_passes_its_checks():
    s = ts.make_schedule(ts.LOCKED_SEED)
    assert ts.check(s) == []
    assert len(s) == 100
    for w in range(4):
        mine = [r for r in s if r.wheel == w]
        assert len(mine) == 25 and mine[0].w_from == 2.0 and mine[0].w_to == 39.2


def test_schedule_same_for_same_seed():
    assert ts.make_schedule(3) == ts.make_schedule(3)


def _mut(s, j, **kw):
    import dataclasses
    s = list(s)
    s[j] = dataclasses.replace(s[j], **kw)
    return s


@pytest.mark.parametrize('j,kw,frag', [
    (10, dict(t0_ms=None), 'cách bậc trước'),
    (5, dict(wheel=0), 'xoay vòng'),
    (0, dict(w_to=2.0), 'int16'),
])
def test_check_catches(j, kw, frag):
    s = ts.make_schedule(1)
    if kw.get('t0_ms', 0) is None:
        kw['t0_ms'] = s[j - 1].t0_ms + 30
    bad = ts.check(_mut(s, j, **kw))
    assert any(frag in b for b in bad), bad


def test_check_catches_no_alternation():
    s = ts.make_schedule(1)
    r = s[4]                                   # wheel 0, its 2nd step (a down step)
    bad = ts.check(_mut(s, 4, w_to=min(39.2, r.w_from + 0.1)))
    assert any('xen kẽ' in b or 'nối liền' in b for b in bad)


def test_schedule_plays_with_marker(tmp_path):
    """The schedule played as run_hil --omega-schedule does: one marker per step, on the frame whose speed
    changes (ScheduleOmegaPlant lags one frame), none at the start nor at the final stop."""
    from t3b_schedule import ScheduleOmegaPlant
    s = ts.make_schedule(1)
    p = ScheduleOmegaPlant(s)
    marks, prev = [], None
    for k in range(p.total_steps + 2):
        st = p.get_state()
        m = _mosi(st[0:4])
        if marker_due('t3', k, prev, m):
            marks.append(k)
        prev = m
        p.step([0] * 4)
    assert marks == [r.t0_ms + 1 for r in s]
