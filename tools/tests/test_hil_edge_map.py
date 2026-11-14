"""RS-4 (tools/hil_edge_map.py), count side after L98/L99/L100: the ESP32 run_active edge bracketed on made-up runs
whose true edge (H7 time) is known. Everything the mapper reads is made the way the bench makes it; the time base
is the H7 clock (L100):
- frames: Pi send times (jitter, optional bursts), the H7 receive after the SPI transfer, the H7 apply window
  (start 18..750 us after the receive, as H7 'age_max'; width 1..80 us, as the 'Apply:' line of
  CLOSED_LOOP/2026-11-13_run14) with the true apply inside, 48-byte frames (RS-4 P1: MOSI run_id + frame_no,
  copied back in MISO 40-47; Liam 14/11 CODE_RS4 "Làm P1, 48 byte"), MISO v6 answering frame j - 2 with its MOSI CRC,
  H7 CYCCNT at the receive (480 MHz), RX_NF, the apply window, the reply count and the duty captured at the H7
  service;
- counts: the unchanged H7 encoder_gen.c (tools/tests/encgen_oracle.py, profile mode) driven by the frame wheel
  speeds at the true apply times, read at the ESP32 tick time + 0..J_READ_US (L99), ESP32 clock off the H7 by
  `ppm` (up to the RHO_MAX_PPM bound, L100);
- ticks 10 ms with some 5 ms late (ESP_TICK/2026-11-13_scan).
Liam 13/11/2026 CODE local_db99d287 ("Làm rs4 + g4 đi"), local_7f7b0ef4 ("Kẹp hai phía", "Sửa firmware" L98),
local_4dc05b46 ("Sửa hết, vòng 4" L99; Codex Sol RS4-R3-1..6; "Sửa gốc ở H7, phiên mới" L100, RS4-R4-1..5),
local_b3611c27 ("Miền rộng ±1000 ppm"). Every number is invented."""
import base64
import csv
import json
import math
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'fwsim'), os.path.join(HERE, '..', '..', 'rpi5')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import encgen_oracle as O                # noqa: E402
import hil_edge_map as M                 # noqa: E402
import hil_o1_score as H                 # noqa: E402
import o1_gate as G                      # noqa: E402
import protocol as P                     # noqa: E402
import rq3_log as rl                     # noqa: E402
from agv_logger import COLUMNS           # noqa: E402
from test_hil_o1_score import fig8_pose  # noqa: E402

ROOT = os.path.join(HERE, '..', '..')
FW = [dict(sha=M.FW_COUNTS_AT_TICK[0], dirty='0', env='esp32-s3-devkitc-1')]
RUN_ID = 0x5EED1234                     # RS-4 P1: MOSI v4 run_id of the made-up runs
OTHER_RUN = 0x0BADF00D                  # a run before this one (replies the H7 may still hold)


def mosi_frame(state10, k, run_id=RUN_ID, **kw):
    """MOSI v4 bytes of frame k as hil_node.py sends them (seq = frame_no = k)."""
    return np.frombuffer(P.encode_state(list(state10), seq=k, run_id=run_id, frame_no=k, **kw), dtype=np.uint8)
needs_gcc = pytest.mark.skipif(not O.available(), reason='gcc not on PATH (H7 EncoderGen oracle)')


def wheel_omega(n, k_move, kind='ramp'):
    """Wheel speeds [rad/s] per frame: 0 before k_move, then by kind."""
    k = np.arange(n, dtype=float)
    u = np.clip((k - k_move) / 400.0, 0.0, 1.0)
    if kind == 'ramp':
        base = 9.0 * u * (1.0 + 0.15 * np.sin(2 * math.pi * np.clip(k - k_move, 0, None) / 1700.0))
    elif kind == 'periodic':                                        # alias test (G32-6)
        base = 6.0 + 3.0 * (np.floor((k - k_move) / 50.0) % 2)
    elif kind == 'slow':                                            # Codex G32-R2-1 (a)
        base = np.where(k < k_move + 17, 1.0, 1.0 + 0.00005 * (k - k_move - 17))
    elif kind == 'stop_start':
        base = 10.0 * u
        base[(k > k_move + 900) & (k < k_move + 1000)] = 0.0
    elif kind == 'onoff':                                           # Codex RS4-R3-1: 1 ms on / 1 ms off
        j = k - k_move
        base = np.where((j >= 0) & (j < 20), np.where(j % 2 == 0, 10.0, 0.0), np.where(j >= 20, 10.0, 0.0))
    elif kind == 'fast':                                            # Codex RS4-R3-1: hp 238 quantisation
        base = np.where(k >= k_move, 39.91821039, 0.0)
    else:
        raise ValueError(kind)
    om = np.column_stack([base, -base * 1.07, base * 0.94, -base * 1.02])
    om[k < k_move] = 0.0
    return np.clip(om, -39.99, 39.99)


def make_miso(mosi, duty, tm, skip=(), lost=(), twice=(), over=(), sat=(), v4=False, crc_bad=(), stage_lag=None,
              no_id=False):
    """MISO v6 (v5 + MISO 40-47 = MOSI 40-47 of the frame answered, RS-4 P1) of each transfer j as the H7 sends it
    (hil_protocol.h offsets, Codex RS4-R5-3 (a)): the reply of the
    latest frame f <= j - 2 whose reply reached the DMA, with the duty of its first transfer in bytes 0-7; with no
    newer reply the DMA sends the same bytes again (spi1_handler.c _tx). tm: rx (H7 us of the receive, integer),
    pre (us from rx to the window start, integer), win (window width, us). Frames in `skip` were received but not
    applied (no reply, RX_NF counts them, no reply count), in `lost` never reached the H7 (no reply, not counted);
    frames in `over` were applied but their reply was replaced before the DMA took it (no reply, reply count
    steps); for frames in `twice` the next transfer repeats the reply with another window; frames in `sat` get a
    saturated window; v4: no v5 mark; no_id: MISO 40-47 left 0 (an H7 before v6); transfers in `crc_bad` arrive with a bad MISO CRC; stage_lag {k: L}: the
    reply of frame k reaches the DMA for transfer k + L (default 2; L 3, 4 when the main loop stages late, Codex
    RS4-R6-3)."""
    n = len(mosi)
    rx, pre, win = tm['rx'], tm['pre'], tm['win']
    skip, lost, over, twice, sat, crc_bad = (set(x) for x in (skip, lost, over, twice, sat, crc_bad))
    stage = np.zeros(n, dtype=int)
    c = 0
    for k in range(n):
        if k not in skip and k not in lost:
            c += 1
        stage[k] = c

    def reply(k, j, extra=0):
        p = int(pre[k])
        dur4 = int(math.ceil((math.ceil(pre[k] + win[k]) - p) / M.APPLY_DUR_UNIT_US)) + extra
        if k in sat:
            dur4 = M.APPLY_DUR_MAX
        b = bytearray(P.FRAME_SIZE)
        if not no_id:
            b[M.MISO_ID:M.MISO_ID + 8] = bytes(mosi[k][M.MOSI_ID:M.MOSI_ID + 8])   # v6: the H7 copies MOSI 40-47
        b[0:8] = np.round(np.clip(duty[j], -1.0, 1.0) * 32767).astype('>i2').tobytes()
        b[10:12] = M.crc16(bytes(mosi[k])).to_bytes(2, 'little')
        cyc = int(round((rx[k] + 1e6) * M.H7_CYC_PER_US)) % 2 ** 32
        b[12:16] = cyc.to_bytes(4, 'little')
        b[16] = 0 if v4 else (M.STATUS_V5 | ((stage[k] % M.STAGE_MOD) << 1) | 1)
        b[17] = mosi[k][P.OFS_SEQ] if hasattr(P, 'OFS_SEQ') else (k & 0xFF)
        b[18:20] = ((k + 1 - sum(1 for x in lost if x <= k)) & 0xFFFF).to_bytes(2, 'little')
        b[20:22] = (p | (dur4 << M.APPLY_DUR_SHIFT)).to_bytes(2, 'little')
        b[8:10] = M.crc16(bytes(b[:8]) + bytes(b[10:])).to_bytes(2, 'little')
        return np.frombuffer(bytes(b), dtype=np.uint8)

    miso = np.zeros((n, P.FRAME_SIZE), dtype=np.uint8)
    send_at = {}                                                 # transfer -> frame whose reply goes out first
    for k in range(n):
        if k not in skip and k not in lost and k not in over:
            send_at[k + (stage_lag or {}).get(k, 2)] = k
    cur = None                                                     # bytes the DMA sends while no newer reply
    for j in range(2, n):
        k = j - 2
        if j in send_at:
            cur = reply(send_at[j], j)
        if cur is not None:
            miso[j] = cur
        if k - 1 in twice and k - 1 not in skip | lost | over:
            miso[j] = reply(k - 1, j, extra=1)
        if j in crc_bad:
            miso[j, 8] ^= 0x5A
    return miso


def sim(edge_ms=415.3, k_run=400, n=5200, kind='ramp', move_after_ms=3.0, ppm=30.0,
        late=(3, 7, 40, 41, 90, 133), write_ms=0.4, glitch_every=97, n_before=6, seed=1, pwm_period=None,
        burst=None, stale_ms=0.0, skip=(), dup=(), win_max=80.0, dts=(), stage_lag=None, crc_twin=(), copy2=()):
    """dict(duty, esp, frames, k_true, timing) of a made-up run with the edge tick at H7 time edge_ms."""
    rng = np.random.default_rng(seed)
    dt = np.full(n, 1000.0)
    if burst:
        dt[burst[0]:burst[0] + burst[1]] = burst[2]                   # catch-up frames closer than 1 ms
    for a0, ln, d0 in dts:                                          # Pi delays then catch-up (Codex RS4-R5-1)
        dt[a0:a0 + ln] = d0
    T = np.r_[0.0, np.cumsum(dt[:-1])] - rng.uniform(0.0, 140.0, n) * (np.arange(n) > 0)
    T[1:] = np.maximum(T[1:], T[:-1] + 50.0)
    rx = np.round(T + 384.0 + rng.uniform(0.0, 50.0, n))             # 48 bytes at 1 MHz, then a little (H7 us)
    win = rng.uniform(1.0, win_max, n) * (rng.uniform(size=n) < 0.3) + rng.uniform(1.0, 5.0, n)
    pre = np.round(rng.uniform(18.0, 750.0, n))
    pre = np.minimum(pre, np.maximum(np.r_[np.diff(rx), 1000.0] - win - 30.0, 1.0))   # applied before the next frame
    apply = rx + pre + np.floor(rng.uniform(0.0, 1.0, n) * np.floor(win))  # the true apply, inside the window
    om = wheel_omega(n, int(edge_ms + move_after_ms), kind)
    mosi = np.array([mosi_frame(list(om[k]) + [0.0] * 6, k) for k in range(n)])
    for k in dup:                                                   # two equal MOSI frames, frame_no included (off the
        mosi[k + 1] = mosi[k]                                       # Pi convention)
    for k in copy2:                                                 # Codex RS4-R7-1: frame k + 2 a whole copy of k
        mosi[k + 2] = mosi[k]
    om32 = M.h7_omega32(mosi)
    app = np.round(apply + 10_000).astype(np.int64)
    for i in range(1, n):
        app[i] = max(app[i], app[i - 1] + 1)
    on = np.array([k not in skip for k in range(n)])                 # skipped frames are never applied
    counts = O.run(app[on], om32[on].astype(np.float64), seed)
    t_e = edge_ms * 1000.0
    ts_e = 5_000_000.0

    def pi_time(ts):                                               # ESP32 clock -> H7 time (truth)
        return t_e + (ts - ts_e) * (1.0 + ppm * 1e-6)
    n_ticks = int((T[-1] - t_e - 60_000) / 10_000)
    ts = [ts_e - i * 100_000.0 for i in range(n_before, 0, -1)]
    ts += [ts_e + j * 10_000.0 + (5_000.0 if j in late else 0.0) for j in range(n_ticks)]
    ts = np.array(ts)
    run = np.r_[np.zeros(n_before, int), np.ones(n_ticks, int)]
    read = pi_time(ts) + rng.uniform(0.0, M.J_READ_US, len(ts)) - stale_ms * 1000.0
    cnt = counts.at(np.round(read + 10_000).astype(np.int64))
    j = np.arange(n_ticks)
    if pwm_period:
        p = 60.0 + 30.0 * ((j // (pwm_period // 2)) % 2)
    else:
        p = np.round(60 + 40 * np.sin(0.13 * j) + 20 * np.sin(0.051 * j + 1.0))
    p[0] = 15.0
    pw = np.r_[np.zeros((n_before, 4)), np.round(np.column_stack([p, p * 0.9, p * 1.1, p]))]
    # MISO duty of transfer f: the pwm written before the H7 serviced frame f - 2
    write_t = pi_time(ts + write_ms * 1000.0)
    serv = np.r_[np.full(2, -1e18), apply[:-2]]
    idx = np.searchsorted(write_t, serv, side='right') - 1
    duty = np.where(idx[:, None] >= 0, pw[np.clip(idx, 0, len(pw) - 1)], 0.0) * (-1.0 / 1000.0)
    if glitch_every:
        duty[k_run + 50::glitch_every, 1] = -1.0                     # capture glitch frames
    esp = dict(timestamp_us=ts.astype(np.int64), log_seq=np.arange(100, 100 + len(ts), dtype=np.int64),
               motor_on=run.copy(), run_active=run.copy(), counts=cnt.astype(float), pwm=pw, fw=list(FW),
               esp_pi_ppm=ppm)
    tm = dict(T=T, rx=rx, pre=pre, win=win, om=om, skip=skip, stage_lag=stage_lag)
    for k in crc_twin:                     # Codex RS4-R6-1: frame k + 2 (not applied) with frame k's speeds and CRC
        mosi[k + 2, 0:8] = mosi[k, 0:8]
        want = M.crc16(bytes(mosi[k]))
        for v in range(65536):
            mosi[k + 2, 31], mosi[k + 2, 32] = v >> 8, v & 0xFF
            if M.crc16(bytes(mosi[k + 2])) == want:
                break
        assert M.crc16(bytes(mosi[k + 2])) == want and mosi[k + 2, 21] != mosi[k, 21]
    miso = make_miso(mosi, duty, tm, skip=skip, stage_lag=stage_lag)
    frames = dict(t_send_ns=np.round((T + 7.0e9) * 1000.0).astype(np.int64),
                  t_pulse0_ns=np.round((T - 60.0 + 7.0e9) * 1000.0).astype(np.int64),
                  mosi=mosi, miso=miso)
    duty_cap = duty                        # duty at the H7 capture; the mapper gets the duty on the wire (R6-3)
    duty = miso[:, 0:8].copy().view('>i2').reshape(n, 4).astype(float) / 32767.0
    # truth: plant step = H7 receive of EVERY frame, applied or not (Codex RS4-R5-1, R5-3)
    return dict(duty=duty, duty_cap=duty_cap, esp=esp, frames=frames,
                k_true=float(np.interp(t_e, rx, np.arange(n, dtype=float))), timing=tm)


def run(s, k_run=400, **over):
    return M.map_edge(s['duty'], k_run, dict(s['esp'], **over), 1e-3, s['frames'])


def _check_bracket(em, edge):
    assert em['ok'], em['reason']
    assert em['early_step'] <= edge <= em['late_step'], (em['early_step'], edge, em['late_step'])
    assert abs(em['edge_step'] - edge) <= em['u_t_s'] / em['period_s'] + 1e-9


def test_constants_match_the_h7_source():
    h = open(os.path.join(ROOT, 'stm32h7', 'Core', 'Inc', 'encoder_gen.h'), encoding='utf-8', errors='replace').read()
    c = open(os.path.join(ROOT, 'stm32h7', 'Core', 'Src', 'encoder_gen.c'), encoding='utf-8', errors='replace').read()
    p = open(os.path.join(ROOT, 'stm32h7', 'Core', 'Inc', 'hil_protocol.h'), encoding='utf-8', errors='replace').read()
    assert int(re.search(r'#define\s+ENC_PPR\s+(\d+)u', h).group(1)) == M.ENC_PPR
    assert float(re.search(r'#define\s+ENC_OMEGA_MIN\s+([0-9.]+)f', c).group(1)) == M.OMEGA_MIN
    f = float(re.search(r'#define\s+ENC_TIMER_FREQ\s+(\d+)UL', h).group(1))
    cal = float(re.search(r'#define\s+ENC_CLOCK_CAL\s+([0-9.]+)f', h).group(1))
    assert f * cal == M.ENC_TIMER_HZ
    assert float(re.search(r'#define\s+HIL_OMEGA_MAX\s+([0-9.]+)f', p).group(1)) == M.HIL_OMEGA_MAX
    for name, val in (('MOSI_CRC', M.MISO_MOSI_CRC), ('RX_NF', M.MISO_RX_NF), ('AGE', M.MISO_AGE),
                      ('STATUS', M.MISO_STATUS)):
        assert int(re.search(rf'#define\s+HIL_MISO_{name}\s+(\d+)', p).group(1)) == val
    for name, val in (('PRE_MAX', M.APPLY_PRE_MAX), ('DUR_SHIFT', M.APPLY_DUR_SHIFT), ('DUR_MAX', M.APPLY_DUR_MAX),
                      ('DUR_UNIT_US', M.APPLY_DUR_UNIT_US)):
        assert int(re.search(rf'#define\s+HIL_APPLY_{name}\s+(\d+)', p).group(1)) == val
    assert re.search(r'#define\s+HIL_STATUS_V5\s+\(1u << 7\)', p) and M.STATUS_V5 == 0x80
    assert re.search(r'#define\s+HIL_STATUS_STAGE_MASK\s+\(0x3Fu << HIL_STATUS_STAGE_SHIFT\)', p)
    assert int(re.search(r'#define\s+HIL_STATUS_STAGE_SHIFT\s+(\d+)', p).group(1)) == 1 and M.STAGE_MOD == 64
    mc = open(os.path.join(ROOT, 'stm32h7', 'Core', 'Src', 'main.c'), encoding='utf-8', errors='replace').read()
    assert 'cyc_per_us = SystemCoreClock / 1000000u' in mc and M.H7_CYC_PER_US == 480.0
    assert P.OFS_MISO_CRC == M.MISO_CRC
    assert M.COUNTS_PER_REV == 1320
    # RS-4 P1: 48-byte frames, run_id + frame_no at MOSI 40, copied at MISO 40
    assert int(re.search(r'#define\s+SPI1_FRAME_LEN\s+(\d+)', p).group(1)) == M.FRAME_V6 == P.FRAME_SIZE
    assert int(re.search(r'#define\s+HIL_MOSI_ID\s+(\d+)', p).group(1)) == M.MOSI_ID == P.OFS_RUN_ID
    assert int(re.search(r'#define\s+HIL_MISO_ID\s+(\d+)', p).group(1)) == M.MISO_ID == P.OFS_MISO_ID
    assert 'memcpy(mi.id, &fi.raw[HIL_MOSI_ID], HIL_MOSI_ID_LEN)' in mc            # B1: from the applied copy
    for gone in ('R_ID_US', 'J_SEND_US', 'E_ID_US', 'START_MARGIN_FRAMES'):      # A1-A3 of ff616ab not used
        assert not hasattr(M, gone)


def test_half_period_as_the_h7():
    om = np.array([[39.91821039, 10.0, 0.31, 0.0]], dtype=np.float32)
    hp, amb = M.h7_half_period(om)
    assert hp[0, 0] == 238.0                                         # Codex RS4-R3-1 (b)
    assert hp[0, 1] == round(1e6 * math.pi / (10.0 * 330))
    assert hp[0, 3] == 65535.0 and not amb[0, 3]


def test_enc_model_events_and_integral():
    mosi = np.zeros((5, P.FRAME_SIZE), dtype=np.uint8)
    om = np.array([[0, 0, 0, 0], [10, 10, 10, -10], [20, 5, 10, 10], [20, 0, 10, 10], [20, 5, 10, 10]], float)
    for k in range(5):
        mosi[k] = mosi_frame(list(om[k]) + [0.0] * 6, k)
    m = M.EncModel(np.array([0.0, 1000.0, 2000.0, 3000.0, 4000.0]), mosi)
    hp, _ = M.h7_half_period(M.h7_omega32(mosi))
    ev, bad = m.events(np.array([1500.0]), np.array([2500.0]))                 # the change at 2000 only
    r = hp[2, 0] / hp[1, 0]
    assert ev[0, 0] == pytest.approx(min(M.RETIME_MAX, 2 * (1 / r - 1)))     # speed-up 10 -> 20
    r1 = hp[2, 1] / hp[1, 1]
    assert ev[0, 1] == pytest.approx(min(M.RETIME_MAX, 2 * (1 - 1 / r1)))    # slow-down 10 -> 5
    assert ev[0, 2] == 0 and bad[0].tolist() == [0, 0, 0, 1]                   # same speed; reversal
    ev1, _ = m.events(np.array([1000.0]), np.array([1000.0]))                  # an event at t0 counts
    assert ev1[0, 0] == pytest.approx(M.E_START)
    _, bad2 = m.events(np.array([2500.0]), np.array([4500.0]))
    assert bad2[0].tolist() == [0, 1, 0, 0]                                    # stop of wheel 1
    q = m.Q(np.array([1500.0]))
    assert q[0, 0] == pytest.approx(0.5e-3 * 2e6 / hp[1, 0])
    assert m.wmax(np.array([1500.0]), np.array([2500.0]))[0] == pytest.approx(2e6 / hp[2:].min())


def test_enc_model_apply_window_widens_events_and_wmax():
    # L100: frame k may apply anywhere in [apply_k, apply_k + width_k]: |rate change| x width joins the event
    # bound of that change, a change whose window reaches into [t1, t2] counts, and wmax looks back one window
    mosi = np.zeros((3, P.FRAME_SIZE), dtype=np.uint8)
    om = np.array([[20, 20, 20, 20], [10, 10, 10, 10], [10, 10, 10, 10]], float)   # slow-down: lookback matters
    for k in range(3):
        mosi[k] = mosi_frame(list(om[k]) + [0.0] * 6, k)
    ap = np.array([0.0, 1000.0, 2000.0])
    m0 = M.EncModel(ap, mosi)
    m1 = M.EncModel(ap, mosi, np.array([0.0, 60.0, 0.0]))
    hp, _ = M.h7_half_period(M.h7_omega32(mosi))
    d = 2e6 / hp[1, 0] - 2e6 / hp[0, 0]
    e0, _ = m0.events(np.array([900.0]), np.array([1100.0]))
    e1, _ = m1.events(np.array([900.0]), np.array([1100.0]))
    assert e1[0, 0] - e0[0, 0] == pytest.approx(abs(d) * 60e-6)
    late0, _ = m0.events(np.array([1030.0]), np.array([1500.0]))       # window 1000..1060 meets [1030, 1500]
    late1, _ = m1.events(np.array([1030.0]), np.array([1500.0]))
    assert late0[0, 0] == 0.0 and late1[0, 0] > 0.0
    assert m1.wmax(np.array([1050.0]), np.array([1055.0]))[0] == pytest.approx(2e6 / hp[0, 0])   # frame 0 may hold
    assert m0.wmax(np.array([1050.0]), np.array([1055.0]))[0] == pytest.approx(2e6 / hp[1, 0])


@needs_gcc
@pytest.mark.parametrize('edge,ppm,write,seed', [(415.3, 30.0, 0.4, 1), (437.85, -900.0, 0.4, 2),
                                                 (408.1, 900.0, 7.4, 3), (420.5, 0.0, 9.0, 4)])
def test_bracket_holds_the_true_edge(edge, ppm, write, seed):
    # L100: ESP32 clock up to +-900 ppm off the H7 (bound RHO_MAX_PPM), apply windows up to 85 us
    s = sim(edge_ms=edge, ppm=ppm, write_ms=write, seed=seed)
    em = run(s)
    _check_bracket(em, s['k_true'])
    assert em['u_t_s'] < 8e-3
    assert em['u_t_s'] == pytest.approx(0.5 * (em['late_step'] - em['early_step']) * 1e-3)   # RS4-R3-5
    assert em['h7_apply']['ok'] and em['h7_apply']['window_us']['max'] <= 88.0


@needs_gcc
@pytest.mark.parametrize('ppm', [900.0, -900.0])
def test_fast_wheels_esp32_clock_off_900_ppm(ppm):
    # L100: at 39.9 rad/s (8400 counts/s) 900 ppm over the 3 s fit is 2.7 ms = 22 counts: only the RHO_MAX_PPM
    # bound (not the 100 ppm of L99) keeps the true edge
    s = sim(kind='fast', ppm=ppm, seed=11)
    _check_bracket(run(s), s['k_true'])


@needs_gcc
def test_h7_apply_reads_the_v5_window():
    # L100: apply = receive + pre (window start, H7 clock), width = 4 us x dur4 >= the window, stage steps
    s = sim(n=2600, skip=(430, 431))
    tm = s['timing']
    ap = M.h7_apply(s['frames'], 380, 2400)
    assert ap['ok'], ap['reason']
    u = ap['applied']
    assert 430 not in u and 431 not in u and ap['n_not_applied'] == 2
    start = tm['rx'][u] + np.floor(tm['pre'][u])
    assert np.allclose(ap['apply_us'][u] - ap['apply_us'][u[0]], start - start[0], atol=1e-6)
    assert np.all(ap['apply_us'][u] + ap['width_us'][u] - ap['apply_us'][u[0]] >= start + tm['win'][u] - start[0] - 1e-6)
    assert np.allclose(ap['t_frame_us'][u] - ap['t_frame_us'][u[0]], tm['rx'][u] - tm['rx'][u[0]], atol=1e-6)


@needs_gcc
def test_wide_apply_windows_bracket_holds():
    # L100: windows up to 116 us (the 4 us field holds 120) on a third of the frames
    s = sim(win_max=110.0, seed=7)
    _check_bracket(run(s), s['k_true'])


@needs_gcc
@pytest.mark.parametrize('kind,after', [('ramp', -40.0), ('stop_start', 3.0), ('fast', 3.0)])
def test_moving_at_the_edge_stop_start_and_fast(kind, after):
    # Codex G32-R2-1 (a): wheels turning at the edge; a stop / restart in the fit span; RS4-R3-1 (b) 39.9 rad/s
    s = sim(kind=kind, move_after_ms=after)
    _check_bracket(run(s), s['k_true'])


@needs_gcc
@pytest.mark.parametrize('kind,after,extra', [('slow', -17.0, {}), ('onoff', 2.7, {}),
                                              ('periodic', 3.0, dict(pwm_period=10, glitch_every=0))])
def test_hard_profiles_are_refused_or_bracketed(kind, after, extra):
    # slow wheels (Codex G32-R2-1 a), 1 ms on / off starts (RS4-R3-1 a), a periodic pattern (G32-6)
    s = sim(kind=kind, move_after_ms=after, **extra)
    em = run(s)
    assert (not em['ok']) or (em['early_step'] <= s['k_true'] <= em['late_step']), em


@needs_gcc
def test_catch_up_frames_and_u_t_in_plant_steps():
    # Codex RS4-R3-5: frames sent 0.5 ms apart near the edge; U_t is in the scorer's plant-step time
    s = sim(burst=(395, 40, 500.0))
    em = run(s)
    _check_bracket(em, s['k_true'])
    assert em['u_t_s'] == pytest.approx(0.5 * (em['late_step'] - em['early_step']) * 1e-3)
    assert em['edge_step'] == pytest.approx(0.5 * (em['early_step'] + em['late_step']))


@needs_gcc
def test_cells_cover_every_fitting_time():
    # Codex RS4-R3-4: every s that fits on a 1 us grid lies inside the kept cells
    s = sim()
    em = run(s)
    assert em['ok'], em['reason']
    esp, fr = s['esp'], s['frames']
    k_lo, k_hi = 400 + int(M.SEARCH[0]) - 20, 4000
    ap = M.h7_apply(fr, k_lo, k_hi)
    u = ap['applied']
    model = M.EncModel(ap['apply_us'][u], fr['mosi'][u], ap['width_us'][u])
    ie = int(M.rising_edges(esp['run_active'])[0])
    c = M.count_side(model, ap['t_frame_us'], 400, esp, ie)
    lines = np.arange(ie, len(esp['timestamp_us']))
    D = (esp['timestamp_us'][lines] - esp['timestamp_us'][ie]).astype(float)
    sel = D <= c['fit_span_s'] * 1e6 + 1
    dc = esp['counts'][lines][sel] - esp['counts'][ie]
    g_lo, g_hi = c['search_us']                                     # Codex RS4-R5-3 (c): the whole search range
    probe = np.arange(g_lo, g_hi, 1.0)
    r = M._ratio(model, probe, D[sel], dc, np.ones((sel.sum(), 4), bool), 0.0)
    fit = probe[r <= 1.0]
    assert len(fit) and fit.min() >= c['s_lo'] and fit.max() <= c['s_hi']


@needs_gcc
def test_refusals_firmware_clock_frames():
    s = sim()
    for fw, word in ((None, 'no events file'), ([], 'no #fw'), ([dict(FW[0], dirty='1')], 'dirty=1'),
                     ([dict(sha='2c026415dfc6', dirty='0')], '2c026415dfc6'),        # L98 without the L99 bound
                     ([dict(sha='cfc892b0ba28', dirty='0')], 'cfc892b0ba28'),
                     (FW + [dict(sha='54b7269abcde', dirty='0')], '54b7269abcde')):
        em = run(s, fw=fw)
        assert not em['ok'] and word in em['reason'], em['reason']
    for ppm in (None, 150.0, -101.0):                                   # L100: the ESP32 / Pi estimate is reported only
        em = run(s, esp_pi_ppm=ppm)
        assert em['ok'] and em['esp_pi_ppm'] == ppm, em['reason']
    assert 'H7 apply times not known' in M.map_edge(s['duty'], 400, s['esp'], 1e-3, None)['reason']
    tm, mo = s['timing'], s['frames']['mosi']

    def refused(word, **kw):
        fr = dict(s['frames'], miso=make_miso(mo, s['duty_cap'], tm, **kw))
        em = M.map_edge(s['duty'], 400, s['esp'], 1e-3, fr)
        assert not em['ok'] and word in em['reason'], em['reason']
    refused('did not reach the H7', lost=(700,))                         # a frame that never reached the H7
    refused('different', twice=(700,))                                   # two different answers for one frame
    refused('lost its reply', over=(700,))                              # Codex RS4-R4-2: applied, reply replaced
    refused('saturated', sat=(700,))                                    # apply window not bounded
    refused('before L100', v4=True)                                     # H7 firmware with MISO v4
    refused('no H7 answer', no_id=True)                                 # RS-4 P1: H7 before v6 (no id copied)
    fr = dict(s['frames'], miso=make_miso(mo, s['duty_cap'], tm, over=(4900,)))
    assert M.map_edge(s['duty'], 400, s['esp'], 1e-3, fr)['ok']          # a lost reply outside the used frames


@needs_gcc
def test_delay_then_catch_up_across_skipped_frames():
    # Codex RS4-R5-1: the Pi is late, then sends frames 0.4 ms apart; the H7 main loop skips 39 of them; the edge
    # falls among the skipped frames, whose receive times are not linear in the index. The bracket must hold the
    # true step (from the receive of every frame) or the run is refused.
    s = sim(edge_ms=431.0, dts=[(395, 10, 4000.0), (405, 30, 400.0)], skip=tuple(range(396, 435)), seed=5)
    em = run(s)
    assert (not em['ok']) or (em['early_step'] <= s['k_true'] <= em['late_step']), (em, s['k_true'])
    tm = s['timing']
    lin = float(np.interp(431000.0, tm['rx'][[395, 435]], [395.0, 435.0]))
    assert abs(lin - s['k_true']) > 5.0                              # the interpolation the fix replaces is wrong here
    lo, hi = M.h7_to_steps(431000.0, np.array([395.0, 435.0]), tm['rx'][[395, 435]].astype(float))
    assert (lo, hi) == (395.0, 435.0)


@needs_gcc
def test_late_replies_and_crc_twins_keep_the_frame_identity():
    # Codex RS4-R6-1: the H7 applies every third frame, its reply first goes out at lag 4 and repeats; around the
    # edge frame k + 2 (not applied) has frame k's speeds and the same MOSI CRC (two IMU bytes chosen), only its
    # seq and frame_no differ. The answers must still land on k (P1: the reply names its frame), so the bracket
    # holds the edge.
    lo, hi = 280, 3900
    skip = tuple(k for k in range(lo, hi) if k % 3)
    lag = {k: 4 for k in range(lo, hi) if k % 3 == 0}
    s = sim(skip=skip, stage_lag=lag, crc_twin=(408, 411, 414, 417, 420), seed=9)
    em = run(s)
    assert (not em['ok']) or (em['early_step'] <= s['k_true'] <= em['late_step']), (em, s['k_true'])
    ap = M.h7_apply(s['frames'], 300, 3800)
    assert ap['ok'], ap['reason']
    assert all(k % 3 == 0 for k in ap['applied'])                   # no answer moved to a twin k + 2


@needs_gcc
def test_small_gap_at_the_edge_widens_the_bracket():
    # Codex RS4-R5-1: three skipped frames right at the edge: the step bracket spans the gap's anchors
    s = sim(skip=(414, 415, 416))
    em = run(s)
    _check_bracket(em, s['k_true'])
    assert em['h7_apply']['n_not_applied'] >= 3


@needs_gcc
def test_lost_replies_refused_including_a_full_reply_count_wrap():
    # Codex RS4-R5-2: replies lost at the Pi (bad MISO CRC). Three lost: the reply count jumps; 64 lost: the count
    # steps by exactly 1 mod 64, refused by the 64-frame anchor gap
    s = sim()
    tm, mo = s['timing'], s['frames']['mosi']
    for bad, word in ((range(702, 705), 'lost its reply'), (range(702, 766), 'more than 64 apart')):
        fr = dict(s['frames'], miso=make_miso(mo, s['duty_cap'], tm, crc_bad=tuple(bad)))
        em = M.map_edge(s['duty'], 400, s['esp'], 1e-3, fr)
        assert not em['ok'] and word in em['reason'], em['reason']


@needs_gcc
def test_frames_the_h7_did_not_apply():
    # the H7 main loop took a newer frame: the skipped frames' speeds never reach EncoderGen (RS4-R3-2)
    s = sim(skip=(430, 431, 520, 777, 1500))
    em = run(s)
    _check_bracket(em, s['k_true'])
    assert em['h7_apply']['n_not_applied'] >= 4


@needs_gcc
def test_repeated_mosi_frames_are_refused():
    # Codex RS4-R7-1 input, P1: two consecutive frames with the same 48 bytes (frame_no included) break the Pi
    # convention frame_no = index; which frame a reply names is then not the logged one: refused, not guessed
    s = sim(dup=(425, 700, 1201))
    em = run(s)
    assert not em['ok'] and 'is not the log run_id' in em['reason'], em['reason']
    ap = M.h7_apply(s['frames'], 300, 3800)
    assert not ap['ok'] and 'frame 426:' in ap['reason'] and 'frame_no 425' in ap['reason'], ap['reason']


@needs_gcc
@pytest.mark.parametrize('every', [3, 9])
def test_copied_frames_before_the_range_are_refused(every):
    # Codex RS4-R7-1 schedule: the H7 applies every `every`-th frame and its reply first goes out at lag
    # `every + 1` (3) or `every` (9 > LAG_MAX); frame k + 2 (not applied) is a whole copy of frame k. Before P1 the
    # copy 281 of the source 279 (just before the range) moved every answer +2 frames. P1: the copy carries
    # frame_no 279 at index 281, refused by the convention check.
    lo, hi = 270, 3900
    applied = [k for k in range(lo, hi) if k % every == 0]
    skip = tuple(k for k in range(lo, hi) if k % every)
    lag = {k: (4 if every == 3 else 9) for k in applied}
    s = sim(skip=skip, stage_lag=lag, copy2=tuple(applied), seed=9)
    k_hi = 3858 if every == 3 else 3850
    ap = M.h7_apply(s['frames'], 280, k_hi)
    assert not ap['ok'] and 'frame 281:' in ap['reason'] and 'frame_no 279' in ap['reason'], ap['reason']
    em = run(s)
    assert not em['ok'] and 'is not the log run_id' in em['reason'], em['reason']


@needs_gcc
@pytest.mark.parametrize('every', [3, 9])
def test_late_replies_without_copies_map(every):
    # the same H7 schedules as above (every 3rd / 9th frame applied, replies at lag 4 / 9 > LAG_MAX) without the
    # copies: P1 reads every transfer after k_lo, so the late replies keep their frames and the bracket holds
    lo, hi = 270, 3900
    applied = [k for k in range(lo, hi) if k % every == 0]
    skip = tuple(k for k in range(lo, hi) if k % every)
    lag = {k: (4 if every == 3 else 9) for k in applied}
    s = sim(skip=skip, stage_lag=lag, seed=9)
    em = run(s)
    assert (not em['ok']) or (em['early_step'] <= s['k_true'] <= em['late_step']), (em, s['k_true'])
    ap = M.h7_apply(s['frames'], 297, 3800)                         # 297: applied for both schedules
    assert ap['ok'] and all(k % every == 0 for k in ap['applied']), ap.get('reason')


def _stale_reply(row, mosi_k, rx_us, rx_nf, stage, pre=400, dur4=5, run_id=None, frame_no=None):
    """MISO v6 bytes as the H7 sends them for a frame with the bytes mosi_k received at H7 time rx_us (duty bytes
    of `row` kept); MISO 40-47 = MOSI 40-47 of mosi_k, or run_id / frame_no when given (a frame of another run)."""
    b = bytearray(bytes(row))
    mk = bytearray(bytes(mosi_k))
    if run_id is not None:
        mk[M.MOSI_ID:M.MOSI_ID + 4] = int(run_id).to_bytes(4, 'little')
    if frame_no is not None:
        mk[M.MOSI_ID + 4:M.MOSI_ID + 8] = int(frame_no).to_bytes(4, 'little')
    b[10:12] = M.crc16(bytes(mk)).to_bytes(2, 'little')
    b[12:16] = (int(round((rx_us + 1e6) * M.H7_CYC_PER_US)) % 2 ** 32).to_bytes(4, 'little')
    b[16] = M.STATUS_V5 | ((stage % M.STAGE_MOD) << 1) | 1
    b[17] = mk[M.MOSI_SEQ]
    b[18:20] = (rx_nf & 0xFFFF).to_bytes(2, 'little')
    b[20:22] = (pre | (dur4 << M.APPLY_DUR_SHIFT)).to_bytes(2, 'little')
    b[M.MISO_ID:M.MISO_ID + 8] = mk[M.MOSI_ID:M.MOSI_ID + 8]
    b[8:10] = M.crc16(bytes(b[:8]) + bytes(b[10:])).to_bytes(2, 'little')
    return np.frombuffer(bytes(b), dtype=np.uint8)


@needs_gcc
def test_reply_of_an_earlier_run_is_skipped():
    # Codex RS4-R8-1 schedule, P1, with the H7 two-buffer order (Codex R9-2): no frame of this run reaches the
    # H7 before 283 (frames 0-282 lost, k_lo 280 inside them; the Pi reads 0 bytes), and the H7 still holds two
    # replies of the run before: X in the DMA buffer (_tx), Y staged (_tx_next). Transfer 283 (the first the H7
    # receives) sends X, its callback moves Y to the DMA buffer; transfer 284 sends Y, its callback takes the
    # reply of 283; transfer 285 sends that. X has the bytes of frame 282 with that run's run_id (frame_no 282
    # too), received 4 ms before 283; Y another frame of that run. Before P1 only the Pi send times showed X was
    # not frame 282's. Now neither is of this run: skipped and counted, the run maps, the bracket holds the edge.
    s = sim()
    tm, mo = s['timing'], s['frames']['mosi']
    lost = tuple(range(0, 283))
    miso = make_miso(mo, s['duty_cap'], tm, lost=lost).copy()
    assert not miso[:285].any()                                       # the two-buffer order: nothing before 285
    nf283 = int(miso[285, 18]) | (int(miso[285, 19]) << 8)            # reply of frame 283 goes out at transfer 285
    st283 = (int(miso[285, 16]) >> 1) & 0x3F
    x = _stale_reply(miso[285], mo[282], float(tm['rx'][283]) - 4000.0, nf283 - 2, st283 - 2, run_id=OTHER_RUN)
    y = _stale_reply(miso[285], mo[282], float(tm['rx'][283]) - 3000.0, nf283 - 1, st283 - 1, run_id=OTHER_RUN,
                     frame_no=283)
    miso[283] = x
    miso[284] = y
    fr = dict(s['frames'], miso=miso)
    ap = M.h7_apply(fr, 280, 3800)
    assert ap['ok'] and ap['applied'][0] == 283 and ap['foreign_n'] == 2, ap.get('reason', ap)
    _check_bracket(M.map_edge(s['duty'], 400, s['esp'], 1e-3, fr), s['k_true'])
    # the same reply carrying THIS run_id and frame 282 is what the H7 sends only if it received frame 282 (B1,
    # B2); a reply naming a frame not yet sent is refused
    early = _stale_reply(miso[283], mo[290], float(tm['rx'][283]), nf283 - 1, st283 - 1)
    m2 = miso.copy()
    m2[284] = early
    bad = M.h7_apply(dict(fr, miso=m2), 280, 3800)
    assert not bad['ok'] and 'frame 290, not yet sent' in bad['reason'], bad.get('reason')


@needs_gcc
def test_reply_with_another_mosi_crc_is_refused():
    # the reply names frame 700 of this run but its MOSI CRC is not that of the logged frame 700: the H7 applied
    # other bytes (a corrupted transfer with the id bytes intact)
    s = sim()
    tm, mo = s['timing'], s['frames']['mosi']
    miso = make_miso(mo, s['duty_cap'], tm).copy()
    other = mo[700].copy()
    other[0] ^= 0x40                                                 # wheel 1 speed changed on the wire
    nf = int(miso[702, 18]) | (int(miso[702, 19]) << 8)
    st = (int(miso[702, 16]) >> 1) & 0x3F
    miso[702] = _stale_reply(miso[702], other, float(tm['rx'][700]), nf, st)
    ap = M.h7_apply(dict(s['frames'], miso=miso), 280, 3800)
    assert not ap['ok'] and 'frame 700: the MOSI CRC' in ap['reason'], ap.get('reason')


@needs_gcc
def test_reply_256_frames_late_keeps_its_frame():
    # Codex RS4-R9-1 (c) input, P1: a reply of frame 700 also goes out 256 frames later (transfer 958). It names
    # frame 700, so it is frame 700's: the same values change nothing; other values refuse (two answers differ).
    # Frame 956 is not applied, so transfer 958 would only resend the reply of 955: no reply is lost.
    s = sim(skip=(956,))
    tm, mo = s['timing'], s['frames']['mosi']
    miso = s['frames']['miso'].copy()
    base = M.h7_apply(dict(s['frames'], miso=miso), 280, 3800)
    assert base['ok'], base.get('reason')
    m2 = miso.copy()
    m2[958] = miso[702]
    same = M.h7_apply(dict(s['frames'], miso=m2), 280, 3800)
    assert same['ok'] and np.array_equal(same['anchor_us'], base['anchor_us']), same.get('reason')
    nf = int(miso[702, 18]) | (int(miso[702, 19]) << 8)
    st = (int(miso[702, 16]) >> 1) & 0x3F
    m2[958] = _stale_reply(miso[702], mo[700], float(tm['rx'][700]) + 256_000.0, nf, st)
    diff = M.h7_apply(dict(s['frames'], miso=m2), 280, 3800)
    assert not diff['ok'] and 'frame 700 answered with different' in diff['reason'], diff.get('reason')


@needs_gcc
def test_bracket_steps_from_the_exact_anchors():
    # P1: the anchors are the receives of the frames the replies name, so the steps come from [s_lo, s_hi] with no
    # widening (E_ID_US of ff616ab is gone)
    s = sim()
    em = run(s)
    _check_bracket(em, s['k_true'])
    a, b = em['h7_apply']['range']
    ap = M.h7_apply(s['frames'], a, b)
    t_lo, t_hi = em['edge_h7_time_us']
    assert em['early_step'] == M.h7_to_steps(t_lo, ap['anchor_k'], ap['anchor_us'])[0]
    assert em['late_step'] == M.h7_to_steps(t_hi, ap['anchor_k'], ap['anchor_us'])[1]
    assert 'e_id_us' not in em['constants'] and em['constants']['frame_len'] == 48
    assert em['h7_apply']['run_id'] == f'{RUN_ID:08x}' and em['h7_apply']['foreign_n'] == 0


def test_exact_anchors_hold_the_true_step():
    # the lemma P1 relies on, against independent true times: every frame has a true H7 receive; the anchors are
    # some of the frames (gaps included) at their exact receives. For any time t the true step (interpolated over
    # ALL frames' true receives) lies in h7_to_steps(t) (the gap rule of Codex RS4-R5-1). Only the lemma: the
    # anchors are given as full times here; turning RX_CYC (mod 2^32) into full times is
    # test_cyccnt_unwrap_across_a_pi_stop_codex_r10_1 (B4, Codex RS4-R10-3).
    rng = np.random.default_rng(5)
    for trial in range(40):
        n = 300
        tr = np.cumsum(rng.uniform(50.0, 1500.0, n))                 # true receives, catch-up frames included
        keep = np.unique(np.r_[0, np.sort(rng.choice(n, size=n // 2, replace=False)), n - 1])
        for t in rng.uniform(tr[0], tr[-1], 200):
            true = float(np.interp(t, tr, np.arange(n, dtype=float)))
            lo, hi = M.h7_to_steps(t, keep.astype(float), tr[keep])
            assert lo - 1e-9 <= true <= hi + 1e-9, (trial, t, lo, true, hi)


def _small_frames(n=400, lost=()):
    """Frames without the EncoderGen oracle (no gcc): seq = frame_no = k, H7 receive 1000 k + 384, replies at lag 2."""
    mosi = np.array([mosi_frame([0.0] * 10, k) for k in range(n)])
    T = np.arange(n) * 1000.0
    tm = dict(T=T, rx=np.round(T + 384.0), pre=np.full(n, 400.0), win=np.full(n, 10.0))
    miso = make_miso(mosi, np.zeros((n, 4)), tm, lost=lost)
    return dict(t_send_ns=np.round((T + 7.0e9) * 1000.0).astype(np.int64), t_pulse0_ns=None, mosi=mosi,
                miso=miso), tm


def test_frame_log_without_t_send_and_40_byte_logs_are_refused():
    fr, _ = _small_frames()
    assert M.h7_apply(fr, 20, 300)['ok']
    no_t = M.h7_apply(dict(fr, t_send_ns=None), 20, 300)            # B4: the CYCCNT unwrap needs the Pi send times
    assert not no_t['ok'] and 'without t_send_ns' in no_t['reason'], no_t['reason']
    short = M.h7_apply(dict(fr, t_send_ns=fr['t_send_ns'][:-1]), 20, 300)
    assert not short['ok'] and 'without t_send_ns' in short['reason'], short['reason']
    old = M.h7_apply(dict(fr, mosi=fr['mosi'][:, :40], miso=fr['miso'][:, :40]), 20, 300)
    assert not old['ok'] and '40-byte frames' in old['reason'] and 'ff616ab' in old['reason'], old['reason']


W_US = 2.0 ** 32 / M.H7_CYC_PER_US                                   # one CYCCNT wrap, 8 947 848.53 us


def _paused_frames(pause_us, at=5, n=400):
    """_small_frames with the Pi stopped pause_us between frames at - 1 and at (Codex RS4-R10-1): the Pi send times
    and the H7 receives both move by pause_us; every frame reaches the H7 and is answered at lag 2."""
    mosi = np.array([mosi_frame([0.0] * 10, k) for k in range(n)])
    T = np.arange(n) * 1000.0 + pause_us * (np.arange(n) >= at)
    tm = dict(T=T, rx=np.round(T + 384.0), pre=np.full(n, 400.0), win=np.full(n, 10.0))
    miso = make_miso(mosi, np.zeros((n, 4)), tm)
    return dict(t_send_ns=np.round((T + 7.0e9) * 1000.0).astype(np.int64), t_pulse0_ns=None, mosi=mosi,
                miso=miso), tm


def test_cyccnt_unwrap_across_a_pi_stop_codex_r10_1():
    # Codex RS4-R10-1 counterexample: the Pi stops W = 2^32 / 480 us between frames 4 and 5; identity, MOSI CRC,
    # MISO CRC, RX_NF and the reply count are all right, but RX_CYC of frame 5 on is one wrap short and the nearest
    # unwrap loses it (before the fix: ok with the anchors W early). B4: refused on the Pi send-time gap.
    fr, tm = _paused_frames(W_US)
    rx = np.asarray(tm['rx'], dtype=float)
    d = (round((rx[5] + 1e6) * M.H7_CYC_PER_US) - round((rx[1] + 1e6) * M.H7_CYC_PER_US)) % 2 ** 32
    assert abs(d - 4000 * M.H7_CYC_PER_US) <= M.H7_CYC_PER_US      # the wrap is invisible in the stamps
    ap = M.h7_apply(fr, 0, 300)
    assert not ap['ok'] and 'answered frames 4 and 5' in ap['reason'] and 'RS4-R10-1' in ap['reason'], ap
    # half a wrap, then send gaps 1 ms above and 1 ms under the bound (the 1000 us step is added to the pause):
    # refused above, exact under (the bound to the ns: test_anchor_send_gap_exactly_at_the_bound, Codex RS4-R11-2)
    for pause, ok in ((0.5 * W_US, False), (M.ANCHOR_GAP_MAX_S * 1e6, False), (M.ANCHOR_GAP_MAX_S * 1e6 - 2000.0, True)):
        fr, tm = _paused_frames(pause)
        ap = M.h7_apply(fr, 0, 300)
        assert ap['ok'] == ok, (pause, ap.get('reason'))
        if ok:
            true = np.asarray(tm['rx'], dtype=float)[ap['applied']]
            assert np.allclose(ap['anchor_us'], true - true[0], atol=0.01), pause
            assert ap['max_anchor_send_gap_s'] < M.ANCHOR_GAP_MAX_S
    # the Pi send times out of order between anchors (not a CLOCK_MONOTONIC read before each transfer)
    fr, _ = _small_frames()
    ts = fr['t_send_ns'].copy()
    ts[101] = ts[100]
    ap = M.h7_apply(dict(fr, t_send_ns=ts), 20, 300)
    assert not ap['ok'] and 'answered frames 100 and 101' in ap['reason'], ap['reason']


@needs_gcc
def test_map_edge_refuses_a_pi_stop_of_one_wrap_inside_the_range():
    # the same stop inside map_edge's range (between frames 300 and 301, k_run 400): refused on B4, not mapped
    s = sim(dts=((300, 1, W_US + 1000.0),), edge_ms=(W_US + 415_300.0) / 1000.0)
    em = run(s)
    assert not em['ok'] and 'answered frames 300 and 301' in em['reason'] and 'B4' in em['reason'], em['reason']
    assert em['constants']['anchor_gap_max_s'] == M.ANCHOR_GAP_MAX_S


def test_anchor_gap_bound_keeps_the_unwrap_inside_the_half_wrap():
    # B4 arithmetic: a send gap under ANCHOR_GAP_MAX_S, at most 1 / (1 - rho) longer on the H7 clock (rho =
    # RHO_MAX_PPM, Codex RS4-R11-1), plus a receive at most 3 s after its send stays inside the half wrap, so the
    # nearest unwrap is the true one
    assert abs(M.CYC_HALF_WRAP_S - 2.0 ** 31 / 480e6) < 1e-12
    assert M.ANCHOR_GAP_MAX_S / (1 - M.RHO_MAX_PPM * 1e-6) + 3.0 < M.CYC_HALF_WRAP_S


@pytest.mark.parametrize('gap_ns, ok', [(10 ** 9 - 1, True), (10 ** 9, False), (10 ** 9 + 1, False)])
def test_anchor_send_gap_exactly_at_the_bound(gap_ns, ok):
    # Codex RS4-R11-2: the Pi send gap between frames 4 and 5 set to the ns around ANCHOR_GAP_MAX_S (the H7 receives
    # follow the same pause): one ns under maps, the bound itself and above are refused
    fr, _ = _paused_frames(gap_ns / 1000.0 - 1000.0)
    ts = fr['t_send_ns'].copy()
    ts[5:] = ts[5:] - ts[5] + ts[4] + gap_ns
    assert int(ts[5] - ts[4]) == gap_ns and M.ANCHOR_GAP_MAX_S == 1.0
    ap = M.h7_apply(dict(fr, t_send_ns=ts), 0, 300)
    assert ap['ok'] == ok, ap.get('reason')
    if not ok:
        assert 'answered frames 4 and 5' in ap['reason'] and 'B4' in ap['reason'], ap['reason']


@pytest.mark.parametrize('stale_dt', [-4000.0, -1000.0, 0.0])
def test_reply_of_an_earlier_run_at_the_log_start(stale_dt):
    # Codex RS4-R8-1, R9-2 schedule, P1: k_lo = 0, frames 0-2 never reach the H7, the H7 still holds a reply of the
    # run before with the bytes of frame 2 (RX_NF, stage one below frame 3's) and sends it at transfers 3, 4, at
    # any receive time. It is not of this run: skipped, frame 3 is the first anchor, nothing assumed about time.
    fr, tm = _small_frames(lost=(0, 1, 2))
    miso = fr['miso'].copy()
    nf3 = int(miso[5, 18]) | (int(miso[5, 19]) << 8)
    st3 = (int(miso[5, 16]) >> 1) & 0x3F
    stale = _stale_reply(miso[5], fr['mosi'][2], float(tm['rx'][3]) + stale_dt, nf3 - 1, st3 - 1, run_id=OTHER_RUN)
    miso[3] = stale
    miso[4] = stale
    ap = M.h7_apply(dict(fr, miso=miso), 0, 300)
    assert ap['ok'] and ap['applied'][0] == 3 and ap['foreign_n'] == 2, ap.get('reason', ap)


def test_frame_id_convention_checked_up_to_k_hi_plus_lag_max():
    # the check covers [k_lo, k_hi + LAG_MAX): a bad frame_no right after k_hi + LAG_MAX - 1 is outside, at it
    # inside; run_id 0 (no run_id) refused
    n = 200
    mosi = np.array([mosi_frame([0.0] * 10, k) for k in range(n)])
    miso = np.zeros((n, P.FRAME_SIZE), dtype=np.uint8)
    out = 'no H7 answer'                                            # no MISO at all: the next refusal
    word = 'is not the log run_id'
    for bad, want in ((100 + M.LAG_MAX - 1, word), (100 + M.LAG_MAX, out), (49, out), (50, word), (75, word)):
        m = mosi.copy()
        m[bad, M.MOSI_ID + 4] ^= 0x01
        ap = M.h7_apply(dict(mosi=m, miso=miso), 50, 100)
        assert not ap['ok'] and want in ap['reason'], (bad, ap['reason'])
    m = mosi.copy()
    m[:, M.MOSI_ID:M.MOSI_ID + 4] = 0
    ap = M.h7_apply(dict(mosi=m, miso=miso), 50, 100)
    assert not ap['ok'] and 'run_id 0' in ap['reason'], ap['reason']
    m = mosi.copy()
    m[60, M.MOSI_ID] ^= 0x01                                          # another run_id inside the range
    ap = M.h7_apply(dict(mosi=m, miso=miso), 50, 100)
    assert not ap['ok'] and 'frame 60:' in ap['reason'] and word in ap['reason'], ap['reason']


@needs_gcc
def test_invalid_count_lines():
    s = sim()
    esp = dict(s['esp'], counts=s['esp']['counts'].copy())
    esp['counts'][[30, 31, 200]] = M.LOG_CNT_INVALID                  # L99: lines read too late are dropped
    em = M.map_edge(s['duty'], 400, esp, 1e-3, s['frames'])
    _check_bracket(em, s['k_true'])
    assert em['count_side']['n_invalid_lines'] == 3
    esp['counts'][6] = M.LOG_CNT_INVALID                               # the edge line itself
    assert 'edge line has no valid counts' in M.map_edge(s['duty'], 400, esp, 1e-3, s['frames'])['reason']


@needs_gcc
def test_refusals():
    s = sim()
    esp = s['esp']
    two = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in esp.items()}
    two['run_active'][len(two['run_active']) // 2] = 0
    assert '2 run_active rising edges' in M.map_edge(s['duty'], 400, two, 1e-3, s['frames'])['reason']
    none = dict(esp, run_active=np.zeros_like(esp['run_active']))
    assert 'need exactly 1' in M.map_edge(s['duty'], 400, none, 1e-3, s['frames'])['reason']
    still = dict(esp, counts=np.zeros_like(esp['counts']))
    assert 'never move' in M.map_edge(s['duty'], 400, still, 1e-3, s['frames'])['reason']
    jump = dict(esp, counts=esp['counts'] + np.where(np.arange(len(esp['counts']))[:, None] > 60, 40.0, 0.0))
    assert 'no H7 time fits' in M.map_edge(s['duty'], 400, jump, 1e-3, s['frames'])['reason']
    nopwm = dict(esp, pwm=np.zeros_like(esp['pwm']))
    assert 'PWM cross-check failed' in M.map_edge(s['duty'], 400, nopwm, 1e-3, s['frames'])['reason']
    early_duty = np.r_[s['duty'][30:], np.repeat(s['duty'][-1:], 30, axis=0)]       # duty 30 ms before its write
    em = M.map_edge(early_duty, 400, esp, 1e-3, s['frames'])
    assert not em['ok'] and 'cannot precede' in em['reason']
    assert not M.map_edge(s['duty'], 1000, esp, 1e-3, s['frames'])['ok']              # RUN step far after the edge
    keep = np.r_[0:6, 8:len(esp['log_seq'])]                                          # G32-4: edge lines lost
    lost = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in esp.items()}
    em = M.map_edge(s['duty'], 400, lost, 1e-3, s['frames'])
    assert not em['ok'] and 'log_seq' in em['reason']


# Counts read outside [t_now, t_now + J_READ_US] (pre-L98 firmware) are outside the mapper's model; the only guard
# is the firmware allow-list FW_COUNTS_AT_TICK (test_refusals_firmware_clock_frames). The earlier one-sided
# "stale counts" test is dropped (Codex RS4-R4-5 (b), RS4-R5-3).


def test_plateau_right_end():
    # G32-R2-2: a step-function cost keeps the right end of its flat minimum
    fine = np.arange(0.0, 2.001, 0.01)
    cf = np.where((fine >= 0.6) & (fine < 1.55), 1.0, 5.0)
    lo, hi = M._plateau(fine, cf)
    assert lo == pytest.approx(0.6) and hi == pytest.approx(1.55, abs=0.011)


def test_esp_clock_ppm_from_raw(tmp_path):
    rng = np.random.default_rng(3)
    ts = np.arange(0, 12_000_000, 10_000) + 1_000_000
    for true in (-40.0, 0.0, 35.0):
        rx = ts * (1 + true * 1e-6) + 5e6 + 300.0 + rng.exponential(600.0, len(ts))   # floor + WiFi delay
        p = tmp_path / f'r{true}.jsonl'
        with open(p, 'w') as f:
            for a, b in zip(ts, rx):
                f.write(json.dumps(dict(t_mono_ns=int(b * 1e3), b64=base64.b64encode(f'{a},0,0'.encode()).decode())) + '\n')
        assert M.esp_clock_ppm(str(p)) == pytest.approx(true, abs=5.0)


def write_esp_csv(path, esp, columns=COLUMNS, extra_rows=(), fw=FW, ppm=None):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(columns)
        rows = []
        for i in range(len(esp['log_seq'])):
            d = dict.fromkeys(columns, '0')
            d.update(timestamp_us=int(esp['timestamp_us'][i]), log_seq=int(esp['log_seq'][i]),
                     motor_on=int(esp['motor_on'][i]), run_active=int(esp['run_active'][i]))
            for name, c in zip(M.WHEELS, esp['counts'][i]):
                d[name] = int(c)
            for name, c in zip(M.PWMS, esp['pwm'][i]):
                d[name] = int(c)
            rows.append([d[c] for c in columns])
        for r in rows + [rows[i] for i in extra_rows]:
            w.writerow(r)
    if fw is not None:
        with open(path + '.events.txt', 'w', encoding='utf-8') as f:
            f.write('2026-11-13T01:45:57.545110 #logger,start\n')
            for d in fw:
                f.write('2026-11-13T01:45:57.545975 #fw,' + ','.join(f'{k}={v}' for k, v in d.items()) + '\n')
    if ppm is not None:                                               # raw packets: Pi receive vs ESP32 timestamp
        rng = np.random.default_rng(5)
        ts = np.arange(esp['timestamp_us'][0], esp['timestamp_us'][0] + 8_000_000, 10_000)
        rx = ts * (1 + ppm * 1e-6) + 300.0 + rng.exponential(600.0, len(ts))
        with open(path + '.raw.jsonl', 'w') as f:
            for a, b in zip(ts, rx):
                f.write(json.dumps(dict(t_mono_ns=int(b * 1e3), b64=base64.b64encode(f'{a},1'.encode()).decode())) + '\n')


@needs_gcc
def test_read_esp_log_order_duplicates_columns_and_fw(tmp_path):
    s = sim(n=3000)
    esp = s['esp']
    p = str(tmp_path / 'r.csv')
    write_esp_csv(p, esp, ppm=20.0)
    got = M.read_esp_log(p)
    assert np.array_equal(got['log_seq'], esp['log_seq']) and np.array_equal(got['pwm'], esp['pwm'])
    assert got['fw'] == FW and M.fw_check(got['fw']) == '' and got['esp_pi_ppm'] == pytest.approx(20.0, abs=8.0)
    write_esp_csv(p, esp, extra_rows=(40,))                            # G32-10: duplicate without segment file
    with pytest.raises(ValueError, match='duplicate line'):
        M.read_esp_log(p)
    with open(p, newline='') as f:
        body = list(csv.reader(f))[1:]
    seg = p + '.segments.csv'
    ti, si = COLUMNS.index('timestamp_us'), COLUMNS.index('log_seq')

    def write_seg(last):
        with open(seg, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(M.SEG_COLS)
            for i, r in enumerate(body, start=1):
                w.writerow([i, r[si], r[ti], 0, last if i == len(body) else 'ok'])
    write_seg('duplicate')
    assert len(M.read_esp_log(p)['log_seq']) == len(esp['log_seq'])
    write_seg('conflict')
    with pytest.raises(ValueError, match='conflict'):
        M.read_esp_log(p)
    os.remove(seg)
    p2 = str(tmp_path / 'small.csv')
    write_esp_csv(p2, esp, columns=list(M.NEED) + ['traj_age_ms'], fw=None)
    g2 = M.read_esp_log(p2)
    assert np.array_equal(g2['counts'], got['counts']) and g2['fw'] is None and g2['esp_pi_ppm'] is None
    p3 = str(tmp_path / 'bad.csv')
    write_esp_csv(p3, esp, columns=[c for c in M.NEED if c != 'pwm_fl'])
    with pytest.raises(ValueError, match='pwm_fl'):
        M.read_esp_log(p3)


def write_frame_log(path, task, k_run, s):
    """Figure-8 frame log from a sim: plant pose from k where omega starts, the sim's wheel speeds in the omega
    slots, the MISO answers (with the duty) rebuilt for these MOSI bytes, the sim's frame times."""
    om, tm = s['timing']['om'], s['timing']
    n = len(om)
    k_move = int(np.argmax(np.any(om != 0, axis=1)))
    t = np.arange(n - k_move) * 1e-3
    pose, _ = fig8_pose(task, t)
    pose = np.vstack([np.zeros((k_move, 3)), pose])
    vel = np.gradient(pose[:, :2], 1e-3, axis=0)
    wz = np.gradient(np.unwrap(pose[:, 2]), 1e-3)
    mosi = np.zeros((n, P.FRAME_SIZE), dtype=np.uint8)
    for k in range(n):
        st = list(om[k]) + [pose[k, 0], pose[k, 1], pose[k, 2], vel[k, 0], vel[k, 1], wz[k]]
        on = k >= k_run + 3
        mosi[k] = mosi_frame(st, k, traj_ref=(0.3, 0.0, 0.1) if on else None, traj_flags=P.TRAJ_VALID if on else 0)
    miso = make_miso(mosi, s['duty_cap'], tm, skip=tm['skip'], stage_lag=tm['stage_lag'])
    manifest = dict(period_us=1000, steps=n, plant='MecanumPlant', miso_check=dict(crc_bad=0, empty=0),
                    lifecycle=dict(outcome='completed', transitions=[dict(step=0, t=0, phase='ARM'),
                                                                     dict(step=k_run, t=0.4, phase='RUN')]))
    np.savez(path, manifest=np.array(json.dumps(manifest)), mosi=mosi, miso=miso,
             t_send_ns=s['frames']['t_send_ns'], t_pulse0_ns=s['frames']['t_pulse0_ns'])


@needs_gcc
def test_scorer_with_esp_log_decides_all_five(tmp_path):
    p, c = str(tmp_path / 'run.npz'), str(tmp_path / 'run.csv')
    n = int((H.Schedule(H.FIG8['O1-F05']).Tcmd + 2.5) / 1e-3) + 420
    s = sim(edge_ms=400.0, k_run=388, n=n)
    write_frame_log(p, 'O1-F05', 388, s)
    write_esp_csv(c, s['esp'], ppm=20.0)
    out = H.score(p, 'O1-F05', esp_log=c)
    em, g = out['edge_map'], out['gate_v3']
    _check_bracket(em, s['k_true'])
    assert M.map_run(p, c)['ok']
    assert out['t0_src'] == 'esp32_run_active_edge' and out['k_run'] == 388
    assert g['decide'] == list(G.CONDS) and g['u_t'] == pytest.approx(em['u_t_s']) and g['complete']
    js = H._json_safe(out)
    json.dumps(js)
    assert rl.validate_gate(js['gate_v3'], 'HIL')['complete']
    plain = H.score(p, 'O1-F05')                                         # no ESP32 log: RUN step, G1, G2 decide
    assert plain['t0_src'] == 'lifecycle_RUN' and plain['gate_v3']['decide'] == list(G.DECIDE_HIL)
    assert plain['edge_map'] == dict(ok=False, reason='no --esp-log') and not plain['gate_v3']['complete']
    os.remove(c + '.raw.jsonl')                                          # L100: ESP32 / Pi estimate reported only
    noclk = H.score(p, 'O1-F05', esp_log=c)
    assert noclk['edge_map']['ok'] and noclk['edge_map']['esp_pi_ppm'] is None
    assert noclk['gate_v3']['decide'] == list(G.CONDS)


@needs_gcc
def test_scorer_cli_esp_log_errors(tmp_path):
    p, c = str(tmp_path / 'run.npz'), str(tmp_path / 'run.csv')
    s = sim(n=3000)
    write_frame_log(p, 'O1-F05', 400, s)
    with open(c, 'w') as f:
        f.write('timestamp_us,log_seq\n1,1\n')
    assert H.main(['--frame-log', p, '--task', 'O1-F05', '--esp-log', c]) == 2       # bad log: an error
    assert M.main(['--frame-log', p, '--esp-log', c]) == 2
