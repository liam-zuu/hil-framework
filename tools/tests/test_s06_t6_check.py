"""tools/s06_t6_check.py (S06 T6, H6 duty read back) on a synthetic run built here: the T6 level list of
s06_load (31 levels x 1 s, W15 level), P lines with compares by GPIO (the ESP32 swaps RPWM/LPWM, as
motor.c does), an ESP32 clock 3 s off and 20 ppm fast seen through UDP with 2-6 ms latency, and an H7
that holds the old duty 2 ms then follows with an IIR 0.75/0.25 per frame, int16 truncation. A clean
run passes; faults put in on purpose must be caught. Run: python -m pytest tools/tests"""
import binascii
import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'rpi5')]
import s06_t6_check as t6  # noqa: E402

LV = [0, 30, 50, 100, 300, 330, 500, 550, 700, 770, 900, 950, 970, 1000]
STEPS = [[v] * 4 for v in LV] + [[100, 330, 550, 770], [0] * 4] + [[-v] * 4 for v in LV[1:]] + \
        [[-100, -330, -550, -770], [0] * 4]
OFF_US, PPM = -3e6, 20e-6                       # t_esp = (t_pi + OFF) * (1 + PPM)
# AGV motor.c motor_cfg (rpwm_gpio, lpwm_gpio) after its swap: FL {FL_LPWM 3, FL_RPWM 8}, … (motor.h)
MOTOR_CFG = {0: (3, 8), 1: (41, 40), 2: (9, 46), 3: (21, 47)}
MOTOR_DIR = -1


def motor_set_info(wheel, speed):
    """Port of AGV motor.c motor_set_info (feat/s06-load d549ac3): ((gpio0, cmp0), (gpio1, cmp1))."""
    speed = max(-1000, min(1000, speed)) * MOTOR_DIR
    rpwm = lpwm = 1000
    if speed > 0:
        rpwm = 1000 - speed
    elif speed < 0:
        lpwm = 1000 + speed
    g0, g1 = MOTOR_CFG[wheel]
    return (g0, rpwm), (g1, lpwm)


def test_golden_vectors():
    # S23: FL +30 writes GPIO3 = 1000, GPIO8 = 970 (Codex Sol round 1/2 trace of motor.c): duty +0.03,
    # chopping pin GPIO8 -> PE5; FL -30: GPIO3 = 970 -> PA0; FR +330: GPIO41 = 1000, GPIO40 = 670 -> PE6
    assert motor_set_info(0, 30) == ((3, 1000), (8, 970))
    cmp = dict(motor_set_info(0, 30))
    assert t6.duty_ref(0, cmp) == pytest.approx(0.03) and t6.active_pin(0, cmp) == 'PE5'
    cmp = dict(motor_set_info(0, -30))
    assert t6.duty_ref(0, cmp) == pytest.approx(-0.03) and t6.active_pin(0, cmp) == 'PA0'
    cmp = dict(motor_set_info(1, 330))
    assert cmp == {41: 1000, 40: 670} and t6.duty_ref(1, cmp) == pytest.approx(0.33) and t6.active_pin(1, cmp) == 'PE6'
    cmp = dict(motor_set_info(3, 1000))
    assert t6.duty_ref(3, cmp) == pytest.approx(1.0) and t6.active_pin(3, cmp) == 'PF7'
    assert t6.active_pin(2, dict(motor_set_info(2, 0))) == '-'


def crc(b):
    return binascii.crc_hqx(bytes(b), 0xFFFF)


def esp(t_pi):
    return (t_pi + OFF_US) * (1 + PPM)


def build(tmp_path, steps=STEPS, hold_s=1.0, gain=(1, 1, 1, 1), offset=None, bad_gpio=False, tau_frames=1.0,
          corrupt=(), ppm=None, glitch=None, end_force=True):
    global PPM
    PPM = ppm if ppm is not None else 20e-6
    rng = np.random.default_rng(11)
    n = int(len(steps) * hold_s * 1000) + 500
    t_pi = np.arange(n) * 1000.0 + 200e3                    # µs, frame send times
    mosi = np.zeros((n, 40), dtype=np.uint8)
    mosi[:, 21] = np.arange(n) % 256
    t_cmd = [t_pi[0] + 100e3 + i * hold_s * 1e6 for i in range(len(steps))]
    duty_true = np.zeros((n, 4))
    cur = np.zeros(4)
    a = 0.75 ** (1.0 / tau_frames)
    i = -1
    for k in range(n):
        while i + 1 < len(steps) and t_pi[k] >= t_cmd[i + 1] + 2000:
            i += 1
        target = np.array([s / 1000.0 for s in steps[i]]) if i >= 0 else np.zeros(4)    # FL +30 -> +0.03
        cur = a * cur + (1 - a) * target
        duty_true[k] = cur
    miso = np.zeros((n, 40), dtype=np.uint8)
    for k in range(2, n):
        b = bytearray(40)
        for w in range(4):
            v = duty_true[k - 2, w] * gain[w] + (offset[1] if offset and offset[0] == (k, w) else 0.0)
            if offset and offset[0] == ('level', w) and abs(duty_true[k - 2, w] - offset[2]) < 1e-3:
                v += offset[1]
            if glitch and w == 0 and k == glitch:
                v += 0.05                                   # one frame out of band after settling started
            b[2 * w:2 * w + 2] = struct.pack('>h', int(max(-1.0, min(1.0, v)) * 32767))
        b[17] = mosi[k - 2][21]
        b[8:10] = struct.pack('<H', crc(b[:8] + b[10:]))
        if k in corrupt:
            b[0] ^= 0x40                                    # torque FL hit after the CRC: frame must be skipped
        miso[k] = np.frombuffer(bytes(b), dtype=np.uint8)
    lines = [f'{int(t_pi[0])}\t#run_start,first_data_seq=1,timestamp_us={int(esp(t_pi[0]))},qseq=0']

    def p_line(i, tb, w, speed, t_rx):
        (g0, c0), (g1, c1) = motor_set_info(w, speed)
        if bad_gpio and w == 2:
            g0 = 45
        return f'{t_rx}\tP,{i},{int(tb + 20 * w)},{int(tb + 20 * w + 15)},{w},{speed},{g0},{c0},{g1},{c1}'

    for i, st in enumerate(steps):
        tb = esp(t_cmd[i])
        for w in range(4):
            if i > 0 and steps[i - 1][w] == st[w]:
                continue                                    # s06_apply writes only on change
            lines.append(p_line(i, tb, w, st[w], int(t_cmd[i] + rng.uniform(2000, 6000))))
    t_stop = t_cmd[-1] + hold_s * 1e6                       # end_run: #run_end, then every wheel forced to 0
    lines.append(f'{int(t_stop + 3000)}\t#run_end,last_data_seq=9,timestamp_us={int(esp(t_stop))},qseq=1')
    if end_force:
        for w in range(4):
            lines.append(p_line(len(steps), esp(t_stop + 1000), w, 0, int(t_stop + 5000)))
    for t in np.arange(t_pi[0], t_pi[-1], 10e3):
        e = esp(t)
        lines.append(f'{int(t + rng.uniform(2000, 6000))}\tE,1,{int(e)},{int(e + 12)},0,0,0,0')
    ev = tmp_path / 'esp32_events.txt'
    ev.write_text('\n'.join(lines) + '\n')
    npz = tmp_path / 'frames.npz'
    np.savez(npz, t_send_ns=(t_pi * 1000).astype(np.int64), mosi=mosi, miso=miso)
    return str(npz), str(ev)


def true_affine():
    """t_pi = OFFSET + RATE * t_esp of the clock of build() (what the marker pin would give)."""
    return ('--clock-affine', f'{-OFF_US:.6f}', f'{1.0 / (1 + PPM):.12f}')


def run(tmp_path, extra=None, accept=True, **kw):
    """accept=True: score as an acceptance run (marker-pin clock + reference uncertainty given)."""
    npz, ev = build(tmp_path, **kw)
    out = tmp_path / 'out'
    if extra is None:
        extra = (*true_affine(), '--u-duty', '0') if accept else ()
    p = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 's06_t6_check.py'), npz, '--esp32-events', ev,
                        '--out', str(out), *extra], capture_output=True, text=True)
    summ = json.load(open(out / 'T6_summary.json', encoding='utf-8')) if p.returncode != 2 else None
    return p.returncode, summ, out, p


def test_clean_run_passes(tmp_path):
    code, s, out, p = run(tmp_path)
    assert code == 0, p.stdout + p.stderr
    assert s['levels']['fail'] == s['levels']['short'] == 0 and s['levels_missing'] == {}
    assert s['max_abs_err'] < 1e-4 and s['sync']['kind'] == 'manual'
    # S22: the forced 0 at the end of the run (same duty as the last level) makes no extra segment
    rows = open(out / 'T6.csv').read().splitlines()
    assert rows[0] == 'idx,wheel,pin,segment,duty_cmd,frame,seq,duty_miso,err'
    # each wheel: 29 levels held (incl. W15), 200 frames each
    assert len(rows) - 1 == sum(len([1 for i, st in enumerate(STEPS) if i == 0 or STEPS[i - 1][w] != st[w]])
                                for w in range(4)) * 200
    assert any(',PE5,' in r for r in rows) and any(',PA0,' in r for r in rows)
    f = [x for x in s['follow'] if x['duty_from'] == 0.0 and x['duty_to'] == -0.03]
    # true follow (marker-pin clock): 2 ms hold + IIR into ±0.006 (0.75^n * 0.03 <= 0.006 -> n = 6)
    # + answer 2 frames
    assert f and 6.0 <= f[0]['follow_ms_max'] <= 12.0, f


def test_survey_clock_and_unknown_uncertainty_are_not_a_pass(tmp_path):
    # S21: UDP-envelope clock only for a survey; S19: no --u-duty
    code, s, _, _ = run(tmp_path, accept=False)
    assert code == 3 and s['sync']['kind'] == 'udp_envelope' and s['levels']['fail'] == 0
    assert abs(s['sync']['ppm'] + 20.0) < 3.0                # Pi clock 20 ppm slow against the ESP32
    code, s, _, _ = run(tmp_path, extra=true_affine())
    assert code == 3 and s['u_duty'] is None


def test_leave_counts_from_old_level(tmp_path):
    code, _, out, _ = run(tmp_path)
    import csv
    rows = list(csv.DictReader(open(out / 'T6_follow.csv')))
    leave = [float(r['leave_ms']) for r in rows if r['leave_ms']]
    # H7 holds the old duty 2 ms and answers 2 frames later; the UDP-envelope clock is about 2 ms late
    assert code == 0 and len(leave) == len(rows) and min(leave) > 1.0


def test_glitch_restarts_the_100_frame_hold(tmp_path):
    # FL 0 -> -0.03 commanded at frame 1100, in band from about frame 1110; one bad frame at 1150
    code, s, out, _ = run(tmp_path, glitch=1150)
    import csv
    r = next(r for r in csv.DictReader(open(out / 'T6_follow.csv')) if r['wheel'] == 'FL' and r['duty_to'] == '0.03')
    assert int(r['frame_settle']) == 1151
    assert code == 0                                          # the glitch is inside the 100 skipped frames


def test_fast_esp32_clock_needs_rate(tmp_path):
    # an ESP32 clock 5000 ppm slow (not a real crystal): without the fitted rate the last levels map
    # 150 ms early on the Pi clock, more than the 100 skipped frames, and the scored frames hit the old level
    code, s, _, _ = run(tmp_path, ppm=-5000e-6, accept=False)
    assert code == 3 and s['levels']['fail'] == s['levels']['short'] == 0, s


def test_crc_bad_frames(tmp_path):
    # S2: level i is commanded at frame 100 + 1000 i (+ 2 ms H7 hold); frames 1000 i + 200 … + 399 are scored.
    # Bad CRC among the 100 skipped frames does not matter; among the 200 scored it is lost data: fail,
    # and no later frame is taken to fill in
    code, s, _, _ = run(tmp_path, corrupt=set(range(5120, 5140)))
    assert code == 0 and s['miso_crc_bad_frames'] == 20 + 2     # + the 2 empty frames at the start
    code, s, _, _ = run(tmp_path, corrupt={5300})
    assert code == 1 and s['levels']['fail_data_loss'] == 4      # the 4 wheels of that level


def test_w15_rows_required(tmp_path):
    # S18: without the W15 row every level still shows per wheel, but a wheel mix-up would not
    steps = [s for s in STEPS if s != [100, 330, 550, 770]]
    code, s, _, _ = run(tmp_path, steps=steps)
    assert code == 3 and s['levels_missing'] == {'W15': [[0.1, 0.33, 0.55, 0.77]]}


def test_forced_zero_after_run_end_is_not_a_level(tmp_path):
    # S22: the schedule ends on a non-zero level; end_run forces 0 after #run_end: no short segment
    code, s, _, p = run(tmp_path, steps=STEPS[:-1])
    assert code == 0 and s['levels']['short'] == 0, p.stdout[-1500:]


def test_gain_error_fails_on_that_wheel(tmp_path):
    code, s, out, _ = run(tmp_path, gain=(1, 1, 1.01, 1))
    assert code == 1
    bad = [l for l in open(out / 'T6_levels.csv').read().splitlines()[1:] if l.endswith(',fail')]
    assert bad and all(l.split(',')[1] == 'RL' for l in bad)


def test_offset_at_one_level_fails(tmp_path):
    # FR reads -0.33 twice: speed +330 (ESP32 drives LPWM) and the W15 level
    code, s, _, _ = run(tmp_path, offset=(('level', 1), 0.007, -0.33))
    assert code == 1 and s['levels']['fail'] == 2


def test_inconclusive_with_reference_uncertainty(tmp_path):
    code, s, _, _ = run(tmp_path, offset=(('level', 1), 0.005, -0.33), extra=('--u-duty', '0.002'))
    assert code == 3 and s['levels']['inconclusive'] == 2 and s['levels']['fail'] == 0


def test_missing_level_and_short_hold(tmp_path):
    steps = [s for s in STEPS if s != [-950] * 4]
    code, s, _, _ = run(tmp_path, steps=steps)
    assert code == 3 and set(s['levels_missing']) == {'FL', 'FR', 'RL', 'RR'}
    code, s, _, _ = run(tmp_path, hold_s=0.25)
    assert code == 3 and s['levels']['short'] > 0


def test_wiring_mismatch_is_input_error(tmp_path):
    code, _, _, p = run(tmp_path, bad_gpio=True)
    assert code == 2 and 'RL' in p.stderr


def test_slow_follow_reported_not_scored(tmp_path):
    code, s, out, _ = run(tmp_path, tau_frames=4.0)
    assert code == 0                                          # static part still passes after 100 frames
    f = max(x['follow_ms_max'] for x in s['follow'] if x['follow_ms_max'] is not None)
    assert f > 30.0
