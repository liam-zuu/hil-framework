"""tools/hil_bench_session.py: the pure judging functions (no serial port, no SSH)."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
import hil_bench_session as hb  # noqa: E402
from protocol import encode_state, encode_torques  # noqa: E402


def frames(n=4000, v=0.2, w=0.2, ref_from=500, duty=0.2, jump_at=None):
    """Circle: the plant follows the setpoint exactly; MOSI carries world-frame velocity like hil_node."""
    mosi, miso = np.zeros((n, 40), np.uint8), np.zeros((n, 40), np.uint8)
    th = 0.0
    for k in range(n):
        on = k >= ref_from
        vb, wz = (v, w) if on else (0.0, 0.0)
        th = (th + wz * 1e-3 + math.pi) % (2 * math.pi) - math.pi
        st = [vb / 0.0485] * 4 + [0.0, 0.0, th, vb * math.cos(th), vb * math.sin(th), wz]
        mosi[k] = np.frombuffer(encode_state(st, traj_ref=(v, 0.0, w) if on else (0.0, 0.0, 0.0), seq=k & 0xFF), np.uint8)
        d = duty if on else 0.0
        if jump_at is not None and k == jump_at:
            d = 0.9
        b = encode_torques([d] * 4)
        miso[k, :len(b)] = np.frombuffer(b, np.uint8)
    return np.arange(n, dtype=np.int64) * 1_000_000, mosi, miso


def test_tracking_body_frame():
    t, mosi, miso = frames()
    r = hb.tracking(t, mosi, miso)
    assert r['moving_frames'] > 2000
    assert r['body'][0] == pytest.approx(0.2, abs=2e-3) and r['body'][2] == pytest.approx(0.2, abs=2e-3)
    assert r['rms_err'][0] < 3e-3 and r['rms_err'][1] < 3e-3     # world-frame vx alone would be ~0.2 off
    assert r['duty_max'] == pytest.approx(0.2, abs=1e-3) and r['duty_sat'] == 0


def test_tracking_duty_jump_reported():
    t, mosi, miso = frames(jump_at=3000)
    assert hb.tracking(t, mosi, miso)['duty_step_max'] == pytest.approx(0.7, abs=1e-3)


def test_tracking_no_setpoint():
    t, mosi, miso = frames(ref_from=10**9)
    assert hb.tracking(t, mosi, miso)['moving_frames'] == 0


def _rows(n=600, spike_at=None, motor_on_from=0):
    out = []
    for i in range(n):
        rpm = 40.0 + (25.0 if spike_at is not None and i == spike_at else 0.0)
        out.append(dict(motor_on='1' if i >= motor_on_from else '0', sp_vx='0.2', sp_vy='0', sp_wz='0',
                        fl_rpm=str(rpm), fr_rpm='40', rl_rpm='40', rr_rpm='40'))
    return out


def test_rpm_spike_found_and_clean():
    r = hb.rpm_spikes(_rows(spike_at=300))
    assert r['wheels']['fl']['over10'] == 1 and r['wheels']['fl']['dev_max'] == pytest.approx(25.0)
    assert r['wheels']['fr']['over10'] == 0
    assert hb.rpm_spikes(_rows())['wheels']['fl']['dev_max'] == 0.0
    assert hb.rpm_spikes(_rows(spike_at=50))['wheels']['fl']['over10'] == 0      # first 1 s skipped
    assert hb.rpm_spikes(_rows(motor_on_from=10**6))['moving_rows'] == 0


def test_h7_window_rise():
    L = [(0, 'Frame: unread=0  new seen=0 taken=0 repl=0 drop=0  applied=10 skipped=3 age_max=1us'),
         (0, 'SPIfix: stuck=5 stuck_err=0x0 restart_fail=0 down_req=0 svc=1..2 cyc (480 cyc/us)'),
         (0, 'Loop: max=400.0 us prn_max=400.0 us over500us=0 over1ms=0 passes=10'),
         (1, 'Frame: unread=0  new seen=0 taken=0 repl=0 drop=0  applied=20 skipped=3 age_max=1us'),
         (1, 'SPIfix: stuck=6 stuck_err=0x0 restart_fail=0 down_req=0 svc=1..2 cyc (480 cyc/us)'),
         (1, 'Loop: max=1200.5 us prn_max=1200.5 us over500us=2 over1ms=1 passes=10')]
    h = hb.h7_window(L)
    assert h == dict(lines=2, skipped_rise=0, stuck_rise=1, loop_max_us=1200.5, loop_over1ms=1)
    assert hb.h7_window([])['stuck_rise'] is None


def test_esp_window_and_judge():
    L = [(0, 'I (1) MOTOR: Motor enabled'), (1, 'I (2) MOTOR: Motor disabled'), (2, 'I (3) MOTOR: Motor enabled'),
         (3, 'I (4) MOTOR: Motor disabled')]
    e = hb.esp_window(L)
    assert e['motor_disabled_inside'] == 1 and e['bad']['POWERON'] == 0
    ok, why = hb.judge([], dict(stuck_rise=0, skipped_rise=0), e)
    assert not ok and 'Motor disabled' in why[0]
    assert hb.judge([], dict(stuck_rise=0, skipped_rise=0), hb.esp_window(L[:2])) == (True, [])
    assert hb.judge(['2 frame MISO sai CRC'], None, None)[0] is False
    assert hb.judge([], dict(stuck_rise=2, skipped_rise=0), None)[0] is False


def test_run_cmd_and_bad_args():
    c = hb.run_cmd('~/run14', 'b2_AL1', '--test A-L1', True)
    assert 'agv_logger.py --out ~/run14/b2_AL1.csv' in c and 'kill -INT $L' in c
    # 2026-11-09_run01: `&` must apply to the logger alone, not to a `mkdir && cd && logger` list ($! = a
    # background shell that ignores SIGINT -> logger never stops, ssh hangs)
    assert '|| exit 2; python3 tools/agv_logger.py' in c and '&& python3 tools/agv_logger.py' not in c
    assert '--test A-L1 --finite-run --frame-log ~/run14/b2_AL1_frames.npz' in c
    assert 'agv_logger' not in hb.run_cmd('~/r', 'x', '--test A-L1', False)
    for bad in ('--test A-L1; rm -rf ~', '--test $(x)', '--test `x`', '--x | y', '--x && y'):
        assert hb.BAD_ARG.search(bad)
    assert not hb.BAD_ARG.search('--traj circle --speed 0.3 --radius 0.5 --duration 60')
