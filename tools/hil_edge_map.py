"""
hil_edge_map.py: RS-4, the ESP32 run_active rising edge of a HIL run mapped to the plant step of the Pi
frame log, so t = 0 of the O1 schedule on HIL is the tick where the firmware starts to apply the command
(O1 v11.9 3.4, dossier v3.2 5.2), as on the floor. Liam 13/11/2026 CODE local_db99d287 ("Làm rs4 + g4 đi");
CODE local_7f7b0ef4 ("Kẹp hai phía", L97; after Codex Sol round 2/2, "Sửa firmware (Recommended)", L98).
Inputs: the ESP32 log of agv_logger.py (with its .events.txt) and the frame log of run_hil.py --frame-log.

L98, L99 (AGV fix/enc-log-tick 16702ff; Liam 13/11 CODE local_4dc05b46 "Sửa hết, vòng 4", Codex L98-1): control_task
reads the four PCNT counts right after t_now and the time again after them; a read later than J_READ_US is
retried, and a line that never fits logs LOG_CNT_INVALID (dropped here). So every logged count was read in
[t_now, t_now + J_READ_US]. Logs of other firmware are refused (FW_COUNTS_AT_TICK, #fw lines of .events.txt).

H7 apply times (h7_apply; L100, MISO v5: Liam 13/11 CODE local_4dc05b46 "Sửa gốc ở H7, phiên mới", Codex
RS4-R4-1, R4-2; RS-4 P1, MOSI v4 / MISO v6, 48-byte frames: Liam 14/11 CODE_RS4 local_c3031475 "Làm P1, 48 byte",
03_CODE/docs/2026-11-14_Thiet_ke_RS4_so_phien_MOSI_v1.md). Every time is on the H7 clock (DWT CYCCNT /
H7_CYC_PER_US, unwrapped); the Pi clock is not used.
Frame identity (P1, replaces the inference of the closed RS-4 ff616ab and its assumptions A1-A3): the Pi writes in
every MOSI the run_id of the run (bytes 40-43, drawn per run, never 0) and the frame_no (bytes 44-47, = the
frame log index = the plant step); the H7 copies MOSI 40-47 of the frame it answers into MISO 40-47, from the same
raw copy as the MOSI CRC, the receive stamp and the apply window. A MISO with a good MISO CRC answers frame f of
this log iff its MISO 40-43 is the run_id of the log and MISO 44-47 is f; its MOSI CRC must then equal the CRC
of the logged MOSI of frame f (else the H7 applied other bytes: refused). Replies carrying another run_id (a run
before this one) are not of this log: skipped and counted (foreign_n). Every MOSI of the range must carry the log
run_id and its own index as frame_no (the Pi convention; else refused). With the CYCCNT of its receive (RX_CYC) and the apply
window: the speeds of that frame took effect in rx + [pre_us, pre_us + 4 dur4] (CYCCNT right before the first
and right after the last EncoderGen_SetOmega, pre rounded down, end rounded up; hil_protocol.h). Status bits
1-6 count the replies the H7 staged, one per applied frame: consecutive answered frames must step it by 1, else
an applied frame lost its reply (replaced before the DMA took it) and the run is refused. A frame without an
answer and without a stage step was received but not applied (the H7 main loop took the newer one; RX_NF of the
applied frames around it steps by the frames sent): the previous speed holds, as on the H7. A frame that did not
reach the H7 (RX_NF), two different answers for one frame, a MISO without the v5 mark (H7 before L100), a
saturated window, or two consecutive answered frames more than STAGE_MOD frames apart (the reply count could
wrap, Codex RS4-R5-2) refuse. Plant step k is placed at the H7 receive of frame k. A frame without an answer has
no receive stamp, only its order (RX_NF): an H7 time between the receives of answered frames a and b > a + 1
maps to steps in [a, b] (both ends taken: early to a, late to b; Codex RS4-R5-1); between a and a + 1 it is
interpolated.

Assumptions of the frame identity (P1; design section 4.3): (B1) the four things the H7 sends for one answered
frame (RX_CYC, the apply window, the MOSI CRC, MISO 40-47) come from the one raw copy of the frame it applied
(main.c, spi1_handler.c; checked on the PC by stm32h7/pc_tests/spi1 id_crc_stamp_same_frame). (B2) The run_id
of the log differs from the run_id of every run the H7 answered since it booted (rpi5/protocol.py new_run_id:
32 random bits, never 0; two runs share one with probability 2^-32). (B3) A MISO that is not what the H7 sent
passes the MISO CRC-16 with probability 2^-16 (as before P1). Nothing is assumed about the Pi clock, the
Pi -> H7 delay or how many replies the H7 holds (A1-A3 of ff616ab are not used). Frame logs of 40-byte frames
(before P1) carry no run_id: refused here; they were mapped by ff616ab (L103: each result names its mapper
commit).

EncoderGen model (EncModel; Codex RS4-R3-1). The count rate of frame k is 2 ENC_TIMER_HZ / hp_k, hp_k the integer
half period encoder_gen.c computes in float32 from the float32 wheel speed the H7 decodes (not the nominal
omega: at 39.9 rad/s, hp 238, the difference is 17 counts/s), on the H7 clock (EncoderGen runs on it). The model
applies frame k at the start of its window; Q(t) is that ideal count. Between two times t0 < t with no stop and no reversal of the wheel applied in [t0, t]:
    |C(t) - C(t0) - (Q(t) - Q(t0))| <= TOL_INT + TOL_MARGIN + EV[t0, t]
EV sums E_START per start (both pins forced low: <= 2 counts; first edge one whole half period in: <= 2) and
per same-direction speed change a retime bound: the pending count is at most one old half period away, so
the phase moves by at most 2 |1/r - 1| (speed-up, capped at RETIME_MAX: the next edge is never later than one
new half period) or 2 (1 - 1/r) (slow-down), r = hp_k / hp_k-1; plus what a 1-tick hp ambiguity of a frame
can make; and (L100) |rate_k - rate_k-1| x the window width of frame k, the most the true apply inside the
window can move the count (a frame counts when its window meets the interval). Checked against the unchanged encoder_gen.c on the PC at every count change (both sides of each
jump), on the frame wheel speeds of 9 bench runs, 6 fixed hard profiles (39.9 rad/s hold, Codex 1 ms on/off,
step, near the dead zone, ramp, x5 steps) and 200 random profiles with jumps, stops, reversals: largest
violation / budget 0.67 (05_DATA/results/ENC_SCHED_MODEL/2026-11-13_run03). Lines across a stop or reversal
of a wheel are not used for that wheel (the first run of one direction only).

Count side (count_side). s = H7 time of the edge tick's t_now. Line i (D_i = its timestamp - the edge's) fits a
cell of s of half width h when, on every used wheel,
    |(count_i - count_edge) - (Q(s + D_i) - Q(s))| <= TOL_INT + TOL_MARGIN + EV + W_e (J + h) + W_i (rho D_i + J + h)
(Lipschitz terms in time, Codex RS4-R3-3: W = the largest |rate| over the frames holding in the uncertain time
interval, both ends, widened by the largest apply window; the ESP32 / H7 clock-rate ratio f_ESP / f_H7 is in
[1 - rho, 1 + rho], rho = RHO_MAX_PPM, so an ESP32 span d is at most d / (1 - rho) H7 us: the read bound J and the
drift use J / (1 - rho) and rho / (1 - rho) D (Codex RS4-R5-4); the rate is unknown inside the bound (Liam 13/11 CODE local_b3611c27, "Miền rộng ±1000 ppm (Recommended)"; replaces the
100 ppm assumption of L99, R4-4); esp_clock_ppm (ESP32 vs Pi from the logger raw packets) is reported only). A
coarse pass (COARSE_US cells) then GRID_US cells inside the kept coarse cells: the Lipschitz term covers each
whole cell, so no fitting s is missed (RS4-R3-4). The kept cells must form ONE interval [s_lo, s_hi] away from
the search ends (no alias). The edge tick is at H7 time in [s_lo, s_hi] (the apply windows are inside the
tolerance), given as plant steps of [s_lo, s_hi] (h7_to_steps: interpolated between consecutive answered frames,
the anchors across a gap; each anchor is the true receive of its frame, read from the reply, so no widening:
P1 drops E_ID_US of ff616ab): early_step, late_step, edge_step the middle; U_t = period x half the width in steps (the scorer's time scale, RS4-R3-5). The O1 schedule and the plant
steps run on the Pi, so no drift to t_deadline enters on HIL. The spacing of the firmware ticks is not added;
the edge tick itself is bracketed (O1 v11.12 3.5, Q2: THESIS 14/11, Liam "Chấp nhận" at THESIS local_701fb2e3;
CODE may drop the label: Liam 14/11 CODE local_9951f4b7 "duyệt luật S05 v3.15 nhé").

PWM side (cross-check only, not in the bracket). The pwm written in a tick (pwm_*) reaches the plant in the
MISO duty of a later frame, never an earlier one. The fit of the pwm steps to the duty is a step function of
the shift (whole frames), so its minimum is an interval; its right end p_hi (G32-R2-2) cannot be before the
edge: early_step <= p_hi + PWM_SLACK_STEPS is required, and p_lo - late_step <= PWM_LAG_MAX_STEPS (write within
the 10 ms tick, H7 capture, frame answer) [ĐỀ XUẤT]. The PWM checks of round 1 (cost, correlation, alias)
still refuse; the PWM side can only refuse more, never move the bracket.

Refused (ok False, reason) when: firmware not in FW_COUNTS_AT_TICK or no events file; frame log without MOSI /
MISO; frames shorter than FRAME_V6 (40-byte logs before P1); a MOSI of the range (to k_hi + LAG_MAX) whose run_id
is 0 or not the log's, or whose frame_no is not its index; a reply of this run naming a frame not yet sent, or
whose MOSI CRC is not that of the frame it names; MISO not v5, an apply window saturated, an applied frame without its reply (stage gap); not exactly one
run_active rising edge; the edge line does not follow a run_active = 0 line with the next log_seq (G32-4); the
edge line has LOG_CNT_INVALID counts; a frame lost before the H7 or answered twice differently; counts never move; no
fitting cell, more than one interval, or one on the search range end; bracket wider than BRACKET_MAX_STEPS;
the PWM fit fails or disagrees with the bracket.

Log reading: the CSV of agv_logger.py by column name; with <log>.segments.csv, lines marked 'duplicate' are
dropped after checking they equal an earlier kept line (L92), 'conflict' / 'ambiguous' or more than one
segment refuse the log; without it, a duplicate line refuses the log (S05 6.3, G32-10). Sorted by log_seq.

  python tools/hil_edge_map.py --frame-log <run>.npz --esp-log <run>.csv [--json out]
Last line: #ket_qua,ma=0 mapped / 3 not mapped (reason) / 2 error.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ENC_PPR = 330                    # stm32h7/Core/Inc/encoder_gen.h ENC_PPR (a test checks)
COUNTS_PER_REV = 4 * ENC_PPR     # quadrature on the ESP32 (AGV encoder.h PPR 1320)
OMEGA_MIN = 0.3                  # rad/s, stm32h7/Core/Src/encoder_gen.c ENC_OMEGA_MIN (a test checks)
WINDOWS_S = (1.0, 2.0, 3.0)      # s after the first non-zero pwm (PWM side)
SEARCH = (-100.0, 400.0)         # steps around the RUN step
COARSE, FINE = 0.25, 0.01        # steps (PWM side)
# count side (L98, L99)
FW_COUNTS_AT_TICK = ('16702ff0e688',)   # AGV firmware: counts read within J_READ_US of t_now, else LOG_CNT_INVALID
LOG_CNT_INVALID = -2 ** 31       # AGV main.c LOG_CNT_INVALID (INT32_MIN): line dropped
J_READ_US = 50.0                 # AGV main.c LOG_CNT_WIN_US (a test checks)
FIT_S = 3.0                      # s of log lines after the first count change
ENC_TIMER_HZ = 1_000_000.0       # encoder_gen.h ENC_TIMER_FREQ x ENC_CLOCK_CAL (a test checks)
HIL_OMEGA_MAX = 40.0             # hil_protocol.h HIL_OMEGA_MAX (a test checks)
H7_CYC_PER_US = 480.0            # H7 DWT cycles per us (SystemCoreClock 480 MHz; main.c cyc_per_us)
TOL_INT = 2.0                    # counts are integers at both ends of a difference
E_START = 4.0                    # a start: both pins forced low (<= 2 counts) + first edge one whole half period in (<= 2)
RETIME_MAX = 2.0                 # a retime moves the next count by at most one new half period (2 counts)
TOL_MARGIN = 1.0                 # [ĐỀ XUẤT]
COARSE_US, GRID_US = 250.0, 10.0  # cells of s: coarse pass, then fine cells inside kept coarse cells
MISO_CRC, MISO_MOSI_CRC, MISO_RX_CYC, MISO_RX_NF, MISO_AGE = 8, 10, 12, 18, 20   # hil_protocol.h (a test checks)
MISO_STATUS, STATUS_V5, STAGE_MOD = 16, 0x80, 64   # v5 (L100): bit 7 mark, bits 1-6 replies staged mod 64
MISO_SEQ, MOSI_SEQ, LAG_MAX = 17, 21, 8   # hil_protocol.h; frames the log must hold after k_hi (replies of the range)
FRAME_V6, MOSI_ID, MISO_ID = 48, 40, 40   # RS-4 P1 (hil_protocol.h SPI1_FRAME_LEN, HIL_MOSI_ID, HIL_MISO_ID; a test checks)
APPLY_PRE_MAX, APPLY_DUR_SHIFT, APPLY_DUR_MAX, APPLY_DUR_UNIT_US = 2047, 11, 31, 4   # bytes 20-21, v5
PWM_LAG_MAX_STEPS = 15.0         # [ĐỀ XUẤT] tick compute (< 10 ms) + capture + frame answer
PWM_SLACK_STEPS = 0.5            # [ĐỀ XUẤT] the PWM fit puts ticks on k x period, frames move <= 0.15 step
PWM_COST_MAX = 100.0             # [ĐỀ XUẤT] capped mean square (pwm units^2)
PWM_CAP = 400.0                  # pwm units^2: one frame's square error is capped (capture noise)
GLITCH = 990.0                   # |duty x 1000| >= this: a capture glitch frame, not used
MIN_CORR = 0.8                   # [ĐỀ XUẤT] pwm vs duty, per wheel
UNIQUE_SEP = 5.0                 # steps
UNIQUE_RATIO = 1.5               # [ĐỀ XUẤT] a real alias costs about the same as the best (ratio ~1);
                                 # the 1 ms frame ripple of the PWM basin gives about 2 (2026-11-11_run01)
BRACKET_MAX_STEPS = 25.0         # Liam 13/11 CODE local_b3611c27 "Nới trần lên 25 bước" (was 20 [ĐỀ XUẤT]; 0.3 m/s runs
                                 # of CLOSED_LOOP/2026-11-13_run14 bracket 21-22 steps with RHO_MAX_PPM)
RIDGE_FRAC = 0.25                # [ĐỀ XUẤT] alias test (_unique, PWM side)
RHO_MAX_PPM = 1000.0             # ESP32 vs H7 clock rate, unknown inside (Liam 13/11, L100)
WHEELS = ('fl_count', 'fr_count', 'rl_count', 'rr_count')
PWMS = ('pwm_fl', 'pwm_fr', 'pwm_rl', 'pwm_rr')
NEED = ('timestamp_us', 'log_seq', 'motor_on', 'run_active') + WHEELS + PWMS
SEG_COLS = ['row', 'log_seq', 'timestamp_us', 'segment', 'status']


def read_esp_log(path: str) -> dict:
    """The columns NEED of an agv_logger CSV as arrays, duplicates handled (docstring), sorted by log_seq."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f'{path}: empty')
    head, body = rows[0], rows[1:]
    miss = [c for c in NEED if c not in head]
    if miss:
        raise ValueError(f'{path}: columns {miss} missing')
    if any(len(r) != len(head) for r in body):
        raise ValueError(f'{path}: lines with another column count')
    seg = path + '.segments.csv'
    keep, seen = [], set()
    if os.path.isfile(seg):
        with open(seg, newline='', encoding='utf-8') as f:
            marks = list(csv.reader(f))
        if not marks or marks[0] != SEG_COLS or len(marks) - 1 != len(body):
            raise ValueError(f'{seg}: header or row count does not match {path}')
        segs = set()
        s, t = head.index('log_seq'), head.index('timestamp_us')
        for i, (r, m) in enumerate(zip(body, marks[1:]), start=1):
            if m[0] != str(i) or m[1] != r[s] or m[2] != r[t]:
                raise ValueError(f'{seg} row {i} does not match the CSV line')
            if m[4] in ('conflict', 'ambiguous'):
                raise ValueError(f'{seg}: {m[4]} lines, log not usable as is')
            if m[4] == 'duplicate':
                if tuple(r) not in seen:
                    raise ValueError(f'{seg} row {i}: marked duplicate, no earlier equal line')
                continue
            seen.add(tuple(r))
            segs.add(m[3])
            keep.append(r)
        if len(segs) > 1:
            raise ValueError(f'{path}: more than one logger segment (DUT restart)')
    else:
        for i, r in enumerate(body, start=1):
            if tuple(r) in seen:
                raise ValueError(f'{path} line {i + 1}: duplicate line and no {os.path.basename(seg)} to '
                                 'confirm dropping it (S05 6.3)')
            seen.add(tuple(r))
            keep.append(r)
    try:
        cols = {c: np.array([int(r[head.index(c)]) for r in keep], dtype=np.int64)
                for c in NEED if c not in PWMS}
        pw = np.array([[float(r[head.index(c)]) for c in PWMS] for r in keep], dtype=float).reshape(-1, 4)
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from None
    order = np.argsort(cols['log_seq'], kind='stable')
    out = {c: v[order] for c, v in cols.items()}
    if len(np.unique(out['log_seq'])) != len(out['log_seq']):
        raise ValueError(f'{path}: two different lines with the same log_seq')
    out['counts'] = np.column_stack([out[w] for w in WHEELS]).astype(float)
    out['pwm'] = pw[order]
    ev = path + '.events.txt'
    out['fw'] = read_fw(ev) if os.path.isfile(ev) else None
    raw = path + '.raw.jsonl'
    out['esp_pi_ppm'] = esp_clock_ppm(raw) if os.path.isfile(raw) else None
    return out


def read_fw(path: str) -> list:
    """The '#fw,sha=..,dirty=..,env=..' lines of an agv_logger events file, as dicts (in file order)."""
    out = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            i = line.find('#fw,')
            if i < 0:
                continue
            d = dict(kv.split('=', 1) for kv in line[i + 4:].strip().split(',') if '=' in kv)
            out.append(d)
    return out


def fw_check(fw) -> str:
    """'' when every #fw line names firmware that logs counts read at t_now (L98), else the reason."""
    if fw is None:
        return 'no events file with #fw lines next to the ESP32 log: firmware not known'
    if not fw:
        return 'no #fw line in the events file: firmware not known'
    for d in fw:
        sha, dirty = d.get('sha', ''), d.get('dirty', '')
        if dirty != '0' or len(sha) < 12 or sha[:12] not in {s[:12] for s in FW_COUNTS_AT_TICK}:
            return (f'firmware sha={sha} dirty={dirty}: its log counts are not known to be read at t_now (L98, '
                    f'{FW_COUNTS_AT_TICK})')
    return ''


def rising_edges(run_active) -> np.ndarray:
    ra = np.asarray(run_active)
    return np.nonzero((ra[1:] == 1) & (ra[:-1] == 0))[0] + 1


def crc16(b) -> int:
    return binascii.crc_hqx(bytes(b), 0xFFFF)


def h7_omega32(mosi) -> np.ndarray:
    """Wheel speeds (n, 4) as the H7 computes them, float32: int16 big-endian x (40.0f / 32767.0f)
    (spi1_handler.c _rd16, hil_protocol.h)."""
    b = np.ascontiguousarray(np.asarray(mosi, dtype=np.uint8)[:, 0:8])
    raw = b.view('>i2').reshape(len(b), 4).astype(np.float32)
    return raw * (np.float32(HIL_OMEGA_MAX) / np.float32(32767.0))


def h7_half_period(om32) -> tuple:
    """(hp, ambiguous): encoder_gen.c _half_period in float32 (ticks), and frames whose float value sits within
    1e-3 of a rounding step (the PC float may round the other way)."""
    a = np.abs(om32).astype(np.float32)
    with np.errstate(divide='ignore', over='ignore'):
        hp_f = (np.float32(ENC_TIMER_HZ) * np.float32(math.pi)) / (np.maximum(a, np.float32(1e-30))
                                                                   * np.float32(ENC_PPR))
    hp_f = np.where(np.isfinite(hp_f), hp_f, np.float32(1e9)).astype(np.float32)
    hp = np.floor((hp_f + np.float32(0.5)).astype(np.float64))
    hp = np.clip(hp, 1.0, 65535.0)
    frac = (hp_f.astype(np.float64) + 0.5) % 1.0
    amb = (np.minimum(frac, 1.0 - frac) < 1e-3) & (hp_f > 1.0) & (hp_f < 65535.0)
    return hp, amb


class EncModel:
    """H7 EncoderGen on the H7 clock for the applied frames. apply[k]: start of the apply window of frame k (H7 us),
    width[k] its width (the true apply lies inside); rate[k, w]: counts per H7 second while frame k holds (sign =
    direction, 0 at or below OMEGA_MIN; 2 x ENC_TIMER_HZ / half_period); Q(t): ideal count with frame k from
    apply[k]. A frame change counts in [t1, t2] when its window meets it; ev[k] adds |rate_k - rate_k-1| x
    width[k] (the true apply inside the window moves the count by at most that).
    Per frame event bound ev[k, w] (counts) at the change from frame k-1 to k: E_START at a start from 0, a
    retime bound at a same-direction speed change: the pending count is at most one old half period away (the
    first edge after a start comes one whole half period in), so the retime shifts the phase by at most
    2 |1/r - 1| (speed-up) or 2 (1 - 1/r) (slow-down) counts, r = hp_k / hp_k-1, capped at RETIME_MAX (the next
    edge is never later than one new half period); plus the count a 1-tick half-period ambiguity can make while
    frame k holds;
    bad[k, w] = 1 at a stop or a reversal (lines across it are not used)."""

    def __init__(self, apply_us, mosi, width_us=None):
        om = h7_omega32(mosi)
        hp, amb = h7_half_period(om)
        d = np.where(om > np.float32(OMEGA_MIN), 1, np.where(om < -np.float32(OMEGA_MIN), -1, 0))
        n = len(apply_us)
        self.apply = np.asarray(apply_us, dtype=float)
        self.width = np.zeros(n) if width_us is None else np.asarray(width_us, dtype=float)
        self.apply_end = np.maximum.accumulate(self.apply + self.width)    # sorted, for "window meets [t1, t2]"
        self.w_max = float(self.width.max()) if n else 0.0
        self.dir, self.hp = d, hp
        self.rate = d * (2.0 * ENC_TIMER_HZ / hp)
        dur = np.diff(np.r_[self.apply, self.apply[-1] + 1000.0]) * 1e-6
        alt = 2.0 * ENC_TIMER_HZ / np.maximum(hp - 1.0, 1.0) - np.abs(self.rate)
        ev = np.where(amb & (d != 0), np.abs(alt) * dur[:, None], 0.0)
        bad = np.zeros((n, 4))
        for k in range(1, n):
            p, c = d[k - 1], d[k]
            st = (p == 0) & (c != 0)
            same = (p == c) & (c != 0) & (hp[k] != hp[k - 1])
            r = hp[k] / hp[k - 1]
            rt = np.minimum(RETIME_MAX, 2.0 * np.where(r < 1.0, 1.0 / r - 1.0, 1.0 - 1.0 / r))
            ev[k] += np.where(st, E_START, 0.0) + np.where(same, rt, 0.0)
            ev[k] += np.abs(self.rate[k] - self.rate[k - 1]) * self.width[k] * 1e-6     # L100 apply window
            bad[k] = (p != c) & (c == 0) & (p != 0) | ((p != 0) & (c != 0) & (p != c))
        self.ev_cum = np.vstack([np.zeros(4), np.cumsum(ev, axis=0)])
        self.bad_cum = np.vstack([np.zeros(4), np.cumsum(bad, axis=0)])
        self.cum = np.vstack([np.zeros(4), np.cumsum(self.rate[:-1] * np.diff(self.apply)[:, None] * 1e-6, axis=0)])
        self._rmq = [np.max(np.abs(self.rate), axis=1)]
        while 2 ** len(self._rmq) <= n:
            p, s = self._rmq[-1], 2 ** (len(self._rmq) - 1)
            self._rmq.append(np.maximum(p[:-s], p[s:]))

    def frame(self, t):
        """Index of the frame holding at time t (-1 before the first apply)."""
        return np.searchsorted(self.apply, t, side='right') - 1

    def Q(self, t):
        """Ideal count (..., 4) at H7 time t (0 before the first frame's apply)."""
        t = np.asarray(t, dtype=float)
        j = self.frame(t)
        jj = np.clip(j, 0, len(self.apply) - 1)
        q = self.cum[jj] + self.rate[jj] * ((t - self.apply[jj]) * 1e-6)[..., None]
        q[j < 0] = 0.0
        return q

    def wmax(self, t1, t2):
        """Largest |rate| (counts/s, over the four wheels) of the frames that may hold in [t1, t2] (same shapes;
        t1 moved back by the widest apply window)."""
        lo = np.clip(self.frame(np.asarray(t1, dtype=float) - self.w_max), 0, len(self.apply) - 1)
        hi = np.clip(self.frame(t2), 0, len(self.apply) - 1)
        L = hi - lo + 1
        lev = np.floor(np.log2(np.maximum(L, 1))).astype(int)
        out = np.zeros(np.shape(lo))
        for v in np.unique(lev):
            m = lev == v
            t = self._rmq[v]
            out[m] = np.maximum(t[lo[m]], t[hi[m] - 2 ** v + 1])
        return out

    def events(self, t1, t2):
        """(event bound (..., 4), bad changes (..., 4)) of the frame changes whose apply window meets [t1, t2]."""
        a = np.searchsorted(self.apply_end, t1, side='left')
        b = np.searchsorted(self.apply, t2, side='right')
        return self.ev_cum[b] - self.ev_cum[a], self.bad_cum[b] - self.bad_cum[a]


def h7_apply(frames, k_lo, k_hi):
    """H7 apply windows of the frames of [k_lo, k_hi) on the H7 clock (module docstring, L100, RS-4 P1). Which frame a
    MISO answers is read from it (P1): a MISO of transfer j with a good MISO CRC, MISO 40-43 = the log run_id and
    MISO 44-47 = f answers frame f of this log; f must be < j (the reply of a frame not yet sent: refused) and its
    MOSI CRC must be that of the logged MOSI of frame f (else refused). Every MOSI of [k_lo, k_hi + LAG_MAX) must
    carry the log run_id (non-zero, the one of MOSI k_lo) and frame_no = its index (else refused). Replies with
    another non-zero run_id are skipped and counted (foreign_n). Every transfer after k_lo is read, so a late reply
    is found wherever it is. Returns dict ok, apply_us / width_us (n, NaN where not applied), applied (frame
    indices), n_not_applied, t_frame_us (n: H7 receive of each frame of the range, unanswered ones interpolated,
    NaN outside), anchors, run_id, foreign_n, window stats; refuses (ok False, reason) as the docstring says."""
    mosi = np.asarray(frames['mosi'], dtype=np.uint8)
    miso = np.asarray(frames['miso'], dtype=np.uint8)
    n = len(mosi)
    if mosi.ndim != 2 or mosi.shape[1] < FRAME_V6 or miso.shape != mosi.shape:
        return dict(ok=False, reason=f'frame log of {mosi.shape[-1]}-byte frames: no run_id / frame_no (MOSI v4, '
                                     f'{FRAME_V6} bytes, RS-4 P1); logs before P1 were mapped by ff616ab (L103)')
    if k_lo < 0 or k_hi > n - LAG_MAX or k_hi - k_lo < 10:
        return dict(ok=False, reason=f'frames {k_lo}..{k_hi} not all inside the frame log (n = {n}) with answers')
    a, b = k_lo, k_hi
    u32 = lambda m, o: int(m[o]) | (int(m[o + 1]) << 8) | (int(m[o + 2]) << 16) | (int(m[o + 3]) << 24)  # noqa: E731
    u16 = lambda m, o: int(m[o]) | (int(m[o + 1]) << 8)  # noqa: E731
    # the Pi writes run_id and frame_no = step in every MOSI (rpi5/hil_node.py) and the frame log index is the step
    run_id = u32(mosi[a], MOSI_ID)
    if run_id == 0:
        return dict(ok=False, reason=f'frame {a}: MOSI run_id 0 (a frame log without run_id: not a P1 run)')
    for k in range(a, b + LAG_MAX):
        if u32(mosi[k], MOSI_ID) != run_id or u32(mosi[k], MOSI_ID + 4) != k:
            return dict(ok=False, reason=f'frame {k}: MOSI run_id {u32(mosi[k], MOSI_ID):08x} / frame_no '
                                         f'{u32(mosi[k], MOSI_ID + 4)} is not the log run_id {run_id:08x} and its '
                                         'index: which frame a reply names is not known')
    crc = {}
    ans, foreign = {}, 0
    for j in range(a + 1, n):
        m = miso[j]
        if not m.any() or u16(m, MISO_CRC) != crc16(bytes(m[:MISO_CRC]) + bytes(m[MISO_CRC + 2:])):
            continue
        rid, f = u32(m, MISO_ID), u32(m, MISO_ID + 4)
        if rid != run_id:
            foreign += rid != 0                       # a reply of another run (B2), or an H7 before v6 (0)
            continue
        if f >= j:
            return dict(ok=False, reason=f'transfer {j} carries a reply of frame {f}, not yet sent: the frame log is '
                                         'not what the H7 answered')
        if not a <= f < b:
            continue
        if f not in crc:
            crc[f] = crc16(mosi[f])
        if u16(m, MISO_MOSI_CRC) != crc[f]:
            return dict(ok=False, reason=f'frame {f}: the MOSI CRC the H7 computed differs from the logged frame: '
                                         'the H7 applied other bytes')
        # status without the SWA bit: a repeated answer carries the same stage
        ans.setdefault(f, []).append((u32(m, MISO_RX_CYC), u16(m, MISO_AGE), u16(m, MISO_RX_NF),
                                      int(m[MISO_STATUS]) >> 1))
    bad = [k for k in range(a, b) if len(set(ans.get(k, ()))) > 1]
    if bad:
        return dict(ok=False, reason=f'frame {bad[0]} answered with different H7 receive / apply values; H7 apply '
                                     'time not known')
    used = [k for k in range(a, b) if k in ans]
    if not used or used[0] > a + 5:
        return dict(ok=False, reason=f'no H7 answer for the first frames of {a}..{b}')
    v4 = [k for k in used if not (ans[k][0][3] << 1) & STATUS_V5]
    if v4:
        return dict(ok=False, reason=f'frame {v4[0]}: MISO without the v5 mark (H7 firmware before L100): apply '
                                     'window and reply count not sent')
    # a frame without an answer was received but not applied, as long as RX_NF of the applied frames around it
    # steps by the frames sent (else a frame did not reach the H7) and the reply count steps by one (else an
    # applied frame lost its reply: its apply time is not known)
    ua = np.array(used, dtype=float)
    far = np.diff(ua) > STAGE_MOD
    if np.any(far):
        i = int(np.argmax(far))
        return dict(ok=False, reason=f'answered frames {used[i]} and {used[i + 1]} more than {STAGE_MOD} apart: the '
                                     'reply count could wrap (Codex RS4-R5-2)')
    nf = np.unwrap(np.array([ans[k][0][2] for k in used], dtype=float), period=65536.0)
    if np.any(np.diff(nf) != np.diff(ua)):
        i = int(np.argmax(np.diff(nf) != np.diff(ua)))
        return dict(ok=False, reason=f'H7 RX_NF {int(nf[i])} -> {int(nf[i + 1])} between frames {used[i]} and '
                                     f'{used[i + 1]}: a frame did not reach the H7')
    stage = np.array([ans[k][0][3] & 0x3F for k in used])
    gap = (np.diff(stage) % STAGE_MOD) != 1
    if np.any(gap):
        i = int(np.argmax(gap))
        return dict(ok=False, reason=f'H7 reply count {int(stage[i])} -> {int(stage[i + 1])} between frames '
                                     f'{used[i]} and {used[i + 1]}: an applied frame lost its reply (Codex RS4-R4-2)')
    raw = np.array([ans[k][0][1] for k in used])
    pre, dur4 = raw & 0x7FF, raw >> APPLY_DUR_SHIFT
    sat = (pre >= APPLY_PRE_MAX) | (dur4 >= APPLY_DUR_MAX)
    if np.any(sat):
        i = int(np.argmax(sat))
        return dict(ok=False, reason=f'frame {used[i]}: H7 apply window saturated (pre {int(pre[i])} us, dur '
                                     f'{int(dur4[i]) * APPLY_DUR_UNIT_US} us): apply time not bounded')
    cyc = np.unwrap(np.array([ans[k][0][0] for k in used], dtype=float), period=2.0 ** 32)
    rx = (cyc - cyc[0]) / H7_CYC_PER_US
    if np.any(np.diff(rx) <= 0):
        return dict(ok=False, reason='H7 receive stamps of the answered frames not increasing')
    lo = rx + pre
    width = dur4 * float(APPLY_DUR_UNIT_US)
    if np.any(np.diff(lo) <= 0):
        return dict(ok=False, reason='H7 apply windows of the answered frames not in frame order')
    apply = np.full(n, np.nan)
    wid = np.full(n, np.nan)
    apply[used], wid[used] = lo, width
    t_frame = np.full(n, np.nan)
    t_frame[a:b] = np.interp(np.arange(a, b, dtype=float), ua, rx)
    return dict(ok=True, apply_us=apply, width_us=wid, applied=np.array(used, dtype=np.int64),
                n_not_applied=int(sum(np.diff(used) - 1)), t_frame_us=t_frame, range=[int(a), int(b)],
                anchor_k=ua, anchor_us=rx, max_gap=int(np.max(np.diff(used))),
                run_id=f'{run_id:08x}', foreign_n=int(foreign),
                window_us=dict(max=float(width.max()), median=float(np.median(width)),
                               pre_max=float(pre.max())))


def h7_to_steps(t, anchor_k, anchor_us):
    """(lo, hi) plant steps of H7 time t from the answered frames (index anchor_k, receive anchor_us): between
    consecutive frames a, a + 1 one interpolated value (lo = hi); across a gap a, b > a + 1 the receives of the
    frames in between are not known, only that they lie in order between the two: [a, b] (Codex RS4-R5-1)."""
    i = int(np.searchsorted(anchor_us, t, side='right')) - 1
    if i < 0 or i >= len(anchor_k) - 1:
        v = float(np.interp(t, anchor_us, anchor_k))
        return v, v
    a, b = anchor_k[i], anchor_k[i + 1]
    if b - a == 1:
        v = float(a + (t - anchor_us[i]) / (anchor_us[i + 1] - anchor_us[i]))
        return v, v
    return float(a), float(b)


def esp_clock_ppm(raw_path: str):
    """ESP32 clock rate vs the Pi clock [ppm] from the logger's raw packets (Pi receive time, ESP32 timestamp):
    slope of the edge of the lower convex hull of (timestamp, receive - timestamp) that spans the mean timestamp
    (one-way delay skew estimate). A coarse check only: on 12 bench runs of 13/11 (12.7 s each) it spread over
    -52 .. +31 ppm. Reported only since L100 (the bound is RHO_MAX_PPM). None when fewer than 50 packets or 5 s."""
    pts = []
    with open(raw_path, encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
                first = base64.b64decode(d['b64']).split(b',', 1)[0]
                pts.append((int(first), d['t_mono_ns'] / 1e3))
            except (ValueError, KeyError, TypeError):
                continue
    if len(pts) < 50:
        return None
    a = np.array(pts, dtype=float)
    a = a[np.argsort(a[:, 0], kind='stable')]
    x, y = a[:, 0], a[:, 1] - a[:, 0]
    if x[-1] - x[0] < 5e6:
        return None
    h = []
    for p in zip(x, y):
        while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1])
                               - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
            h.pop()
        h.append(p)
    xm = float(x.mean())
    for (x1, y1), (x2, y2) in zip(h[:-1], h[1:]):
        if x1 <= xm <= x2 and x2 > x1:
            return float((y2 - y1) / (x2 - x1) * 1e6)
    return None


def _minimise(cost, lo, hi):
    """(a, best cost, coarse grid, coarse costs, fine grid, fine costs): coarse grid then a fine grid around
    the best."""
    grid = np.arange(lo, hi + COARSE / 2, COARSE)
    c = np.array([cost(a) for a in grid])
    a0 = grid[int(np.argmin(c))]
    fine = np.arange(a0 - 1.0, a0 + 1.0 + FINE / 2, FINE)      # 1 step each side: the plateau is <= 1 step
    cf = np.array([cost(a) for a in fine])
    return float(fine[int(np.argmin(cf))]), float(cf.min()), grid, c, fine, cf


def _plateau(fine, cf):
    """(lo, hi): the run of fine-grid points around the minimum whose cost equals it (the PWM cost is a
    step function of the shift: whole frames, G32-R2-2), hi extended by one fine step."""
    i = int(np.argmin(cf))
    flat = cf <= cf[i] + 1e-9 * max(1.0, abs(cf[i]))
    lo = hi = i
    while lo > 0 and flat[lo - 1]:
        lo -= 1
    while hi < len(cf) - 1 and flat[hi + 1]:
        hi += 1
    return float(fine[lo]), float(fine[hi] + FINE)


def _unique(grid, c, a, floor):
    """True unless another local minimum m of the coarse cost, UNIQUE_SEP steps or more from a, is an alias:
    about as low as the best (c_m < UNIQUE_RATIO x best + floor) AND behind a ridge (c_m - best <= RIDGE_FRAC
    x (highest cost between the two - best)). A ripple of the 1 ms frames on the wall of the one basin has
    no such ridge; a repeated pattern (G32-6) has both."""
    ib = int(np.argmin(c))
    best = float(c[ib])
    loc = np.nonzero((c <= np.r_[np.inf, c[:-1]]) & (c <= np.r_[c[1:], np.inf]) & (np.abs(grid - a) >= UNIQUE_SEP))[0]
    for j in loc:
        ridge = float(np.max(c[min(ib, j):max(ib, j) + 1]))
        if c[j] < UNIQUE_RATIO * best + floor and c[j] - best <= RIDGE_FRAC * (ridge - best):
            return False
    return True


def _windows(ts, idx, t_from):
    """Lines of `idx` up to t_from + W, one set per window; a window with the same lines as the one
    before is dropped (a short log, G32-6)."""
    out, last = [], None
    for W in WINDOWS_S:
        sel = idx[ts[idx] <= t_from + W * 1e6]
        if last is not None and len(sel) == len(last):
            continue
        out.append((W, sel))
        last = sel
    return out


def _ratio(model, s, D, dc, use_lines, half):
    """Largest |misfit| / tolerance over lines and wheels for candidate edge times s (centres of cells of half
    width `half`), docstring. Lines (i, w) whose span crosses a stop or reversal of wheel w, or more than one
    start, are not used."""
    rho = RHO_MAX_PPM * 1e-6
    rho_h, j_h = rho / (1.0 - rho), J_READ_US / (1.0 - rho)          # ESP32 spans on the H7 clock (RS4-R5-4)
    out = np.empty(len(s))
    for c0 in range(0, len(s), 400):
        sc = s[c0:c0 + 400][:, None]                                  # (G, 1)
        de = j_h + half                                                # edge read: [s, s + J], cell: +-half
        di = rho_h * D[None, :] + j_h + half                           # line read: +-rho D, + [0, J], cell
        q0 = model.Q(sc[:, 0])                                         # (G, 4)
        qi = model.Q(sc + D[None, :])                                  # (G, L, 4)
        we = model.wmax(sc[:, 0] - half, sc[:, 0] + de)               # (G,)
        wi = model.wmax(sc + D[None, :] - di, sc + D[None, :] + di)    # (G, L)
        ev, bad = model.events(sc - half, sc + D[None, :] + di)         # (G, L, 4)
        tol = TOL_INT + TOL_MARGIN + ev + (we[:, None, None] * de + wi[:, :, None] * di[:, :, None]) * 1e-6
        mis = np.abs(dc[None] - (qi - q0[:, None, :]))
        ok_pair = (bad == 0) & use_lines[None]
        r = np.where(ok_pair, mis / tol, 0.0)
        out[c0:c0 + 400] = r.max(axis=(1, 2))
    return out


def count_side(model, frames_t_us, k_run, esp, ie):
    """Cells of s (H7 time of the edge tick's t_now) that fit every used count within the tolerance; a coarse pass
    then GRID_US cells inside the kept coarse cells (the Lipschitz term covers each cell, so no fitting s is
    missed). frames_t_us: H7 receive time per frame (NaN outside the range). ok with s_lo, s_hi when the kept
    cells form one interval away from the search ends."""
    ts, cnt = esp['timestamp_us'].astype(float), esp['counts']
    fin = np.isfinite(frames_t_us)
    T = frames_t_us[fin]
    kk = np.nonzero(fin)[0].astype(float)
    g_lo = float(np.interp(k_run + SEARCH[0], kk, T))
    g_hi = float(np.interp(k_run + SEARCH[1], kk, T))
    after = np.arange(ie, len(ts))
    after = after[esp['run_active'][after] == 1]
    valid = ~np.any(cnt[after] == LOG_CNT_INVALID, axis=1)
    D = ts[after] - ts[ie]
    moved = np.nonzero(valid & np.any(cnt[after] != cnt[ie], axis=1))[0]
    d_move = D[moved[0]] if len(moved) else 0.0
    d_model = model.apply[-1] - g_hi - 2000.0                       # lines inside the modelled frames
    keep = valid & (D <= min(d_move + FIT_S * 1e6, d_model))
    lines, D = after[keep], D[keep]
    out = dict(n_lines=int(len(lines)), n_invalid_lines=int(np.count_nonzero(~valid)),
               fit_span_s=float(D[-1] * 1e-6) if len(D) else 0.0, search_us=[g_lo, g_hi])
    if not len(moved):
        return dict(out, ok=False, reason='counts never move after the edge')
    dc = cnt[lines] - cnt[ie]
    use = np.ones((len(lines), 4), dtype=bool)
    coarse = np.arange(g_lo + COARSE_US / 2, g_hi, COARSE_US)
    rc = _ratio(model, coarse, D, dc, use, COARSE_US / 2)
    keep_c = np.nonzero(rc <= 1.0)[0]
    out['best_ratio_coarse'] = float(rc.min())
    if not len(keep_c):
        return dict(out, ok=False, reason=f'count side: no H7 time fits every count within the tolerance (best '
                                          f'{rc.min():.2f} x tolerance)')
    m = int(round(COARSE_US / GRID_US))
    fine = (coarse[keep_c][:, None] - COARSE_US / 2 + (np.arange(m)[None, :] + 0.5) * GRID_US).ravel()
    rf = _ratio(model, fine, D, dc, use, GRID_US / 2)
    kf = rf <= 1.0
    out['best_ratio'] = float(rf.min())
    if not kf.any():
        return dict(out, ok=False, reason=f'count side: no H7 time fits every count within the tolerance (best '
                                          f'{rf.min():.2f} x tolerance)')
    cells = fine[kf]
    gaps = np.nonzero(np.diff(cells) > GRID_US * 1.5)[0]
    comps = np.split(cells, gaps + 1)
    out['components_us'] = [[float(c[0] - GRID_US / 2), float(c[-1] + GRID_US / 2)] for c in comps]
    if len(comps) > 1:
        return dict(out, ok=False, reason=f'count side: {len(comps)} separate feasible intervals (alias)')
    s_lo, s_hi = float(cells[0] - GRID_US / 2), float(cells[-1] + GRID_US / 2)
    if s_lo <= g_lo + COARSE_US or s_hi >= g_hi - COARSE_US:
        return dict(out, ok=False, reason='count side: feasible interval on the edge of the search range')
    return dict(out, ok=True, s_lo=s_lo, s_hi=s_hi)


def pwm_side(duty, k_run, esp, ie, period_us):
    ts, pw = esp['timestamp_us'].astype(float), esp['pwm']
    D = np.asarray(duty, dtype=float) * 1000.0
    n = len(D)
    after = np.arange(ie, len(ts))
    after = after[esp['run_active'][after] == 1]
    on = after[np.any(pw[after] != 0, axis=1)]
    if not len(on):
        return dict(ok=False, reason='pwm never leaves 0 after the edge')
    lo, hi = k_run + SEARCH[0], k_run + SEARCH[1]
    good = np.all(np.abs(D) < GLITCH, axis=1)
    fits = {}
    for W, sel in _windows(ts, after, ts[on[0]]):
        t_last = ts[sel][-1]
        k_hi = int(min(n, hi + (t_last - ts[ie]) / period_us + 1))
        frames = np.arange(max(0, int(lo) - 5), k_hi)
        frames = frames[good[frames]]
        lines = np.r_[ie - 1, sel] if ie > 0 else sel        # the 0 before the edge

        def pred(a, frames=frames, lines=lines):
            tick_k = a + (ts[lines] - ts[ie]) / period_us
            j = np.searchsorted(tick_k, frames, side='right') - 1
            P = pw[lines][np.clip(j, 0, len(lines) - 1)].copy()
            P[(j < 0) | (lines[np.clip(j, 0, len(lines) - 1)] < ie)] = 0.0
            return P
        P0 = pred(k_run + 15.0)
        sgn = np.sign(np.sum(P0 * D[frames], axis=0))
        sgn[sgn == 0] = 1.0

        def cost(a):
            e = D[frames] - pred(a) * sgn
            return float(np.mean(np.minimum(e ** 2, PWM_CAP)))
        a, c, grid, cs, fine, cf = _minimise(cost, lo, hi)
        a_lo, a_hi = _plateau(fine, cf)
        P = pred(a) * sgn
        corr = [float(np.corrcoef(P[:, w], D[frames, w])[0, 1]) if np.std(P[:, w]) > 0 and np.std(D[frames, w]) > 0
                else float('nan') for w in range(4)]
        fits[str(W)] = dict(a=a, a_lo=a_lo, a_hi=a_hi, cost=c, corr=corr, sign=[float(s) for s in sgn],
                            n_frames=int(len(frames)),
                            n_glitch=int(np.count_nonzero(~good[max(0, int(lo) - 5):k_hi])), t_end_us=float(t_last),
                            unique=_unique(grid, cs, a, 10.0), on_edge=bool(a - lo < 2 * COARSE or hi - a < 2 * COARSE))
    out = dict(fits=fits)
    if len(fits) < 2:
        return dict(out, ok=False, reason=f'PWM side: {len(fits)} distinct window(s) (need 2)')
    for w, f in fits.items():
        why = ('edge of the search range' if f['on_edge'] else 'cost minimum not unique' if not f['unique']
               else f'cost {f["cost"]:.1f} > {PWM_COST_MAX}' if f['cost'] > PWM_COST_MAX
               else f'pwm-duty correlation {min(f["corr"]):.2f} < {MIN_CORR}'
               if not all(np.isfinite(f['corr'])) or min(f['corr']) < MIN_CORR else '')
        if why:
            return dict(out, ok=False, reason=f'PWM side, window {w} s: {why}')
    return dict(out, ok=True, a_lo=min(f['a_lo'] for f in fits.values()), a_hi=max(f['a_hi'] for f in fits.values()),
                t_end_us=max(f['t_end_us'] for f in fits.values()))


def map_edge(duty, k_run: int, esp: dict, period_s: float, frames=None) -> dict:
    """Plant step of the run_active edge of `esp` (read_esp_log) on a frame log: MISO duty `duty` (n, 4), RUN step
    k_run, frames = dict(mosi, miso, ...) of the frame log (None: refused). Returns a dict, ok True / False with
    reason."""
    period_us = period_s * 1e6
    out = dict(ok=False, reason='', method='count_feasible_set_L100_h7_clock_P1_frame_id', k_run=int(k_run),
               period_s=period_s,
               counts_per_rev=COUNTS_PER_REV, omega_min=OMEGA_MIN, fw=esp.get('fw'),
               esp_pi_ppm=esp.get('esp_pi_ppm'),
               constants=dict(fw_counts_at_tick=list(FW_COUNTS_AT_TICK), j_read_us=J_READ_US, fit_s=FIT_S,
                              tol_int=TOL_INT, e_start=E_START, retime_max=RETIME_MAX, tol_margin=TOL_MARGIN,
                              grid_us=GRID_US, coarse_us=COARSE_US, rho_max_ppm=RHO_MAX_PPM,
                              frame_len=FRAME_V6, bracket_max_steps=BRACKET_MAX_STEPS, pwm_lag_max_steps=PWM_LAG_MAX_STEPS,
                              pwm_windows_s=list(WINDOWS_S), pwm_cost_max=PWM_COST_MAX, min_corr=MIN_CORR,
                              unique_ratio=UNIQUE_RATIO))
    why = fw_check(esp.get('fw'))
    if why:
        out['reason'] = why
        return out
    if frames is None or any(frames.get(k) is None for k in ('mosi', 'miso')):
        out['reason'] = 'frame log without MOSI / MISO: H7 apply times not known'
        return out
    edges = rising_edges(esp['run_active'])
    out['n_edges'] = int(len(edges))
    if len(edges) != 1:
        out['reason'] = f'{len(edges)} run_active rising edges in the log (need exactly 1)'
        return out
    ie = int(edges[0])
    out.update(edge_log_seq=int(esp['log_seq'][ie]), edge_timestamp_us=int(esp['timestamp_us'][ie]))
    if int(esp['log_seq'][ie]) != int(esp['log_seq'][ie - 1]) + 1:
        out['reason'] = (f'log_seq {int(esp["log_seq"][ie - 1])} -> {int(esp["log_seq"][ie])} at the edge: lines lost, '
                         'the first run_active line may not be the edge tick (G32-4)')
        return out
    if np.any(esp['counts'][ie] == LOG_CNT_INVALID):
        out['reason'] = 'the edge line has no valid counts (read later than J_READ_US after t_now)'
        return out
    n = len(frames['mosi'])
    # frames needed: from before the search range to the last line count_side uses (first count change + FIT_S)
    cnt = esp['counts']
    after = np.arange(ie, len(cnt))
    after = after[(esp['run_active'][after] == 1) & ~np.any(cnt[after] == LOG_CNT_INVALID, axis=1)]
    moved = after[np.any(cnt[after] != cnt[ie], axis=1)]
    d_move = float(esp['timestamp_us'][moved[0]] - esp['timestamp_us'][ie]) if len(moved) else 0.0
    d_use = d_move + FIT_S * 1e6
    k_lo = int(max(0, k_run + SEARCH[0] - 20))
    t_need = d_use / (1 - RHO_MAX_PPM * 1e-6) + J_READ_US + COARSE_US + 5000.0       # us after the search end
    k_hi = int(min(n - LAG_MAX, k_run + SEARCH[1] + t_need / period_us + 20))
    ap = h7_apply(frames, k_lo, k_hi)
    out['h7_apply'] = {k: (v if k != 'applied' else len(v)) for k, v in ap.items()
                       if k not in ('apply_us', 'width_us', 't_frame_us', 'anchor_k', 'anchor_us')}
    if not ap['ok']:
        out['reason'] = ap['reason']
        return out
    used = ap['applied']
    model = EncModel(ap['apply_us'][used], np.asarray(frames['mosi'])[used], ap['width_us'][used])
    T_all = ap['t_frame_us']
    c = count_side(model, T_all, k_run, esp, ie)
    p = pwm_side(duty, k_run, esp, ie, period_us)
    out.update(count_side=c, pwm_side=p)
    if not c['ok']:
        out['reason'] = c['reason']
        return out
    early_t, late_t = c['s_lo'], c['s_hi']                            # apply windows are inside the tolerance
    # RS-4 P1: each anchor is the H7 receive of the frame its reply names, so no widening (E_ID_US of ff616ab)
    early = h7_to_steps(early_t, ap['anchor_k'], ap['anchor_us'])[0]
    late = h7_to_steps(late_t, ap['anchor_k'], ap['anchor_us'])[1]
    a_step = 0.5 * (early + late)
    out.update(early_step=early, late_step=late, edge_step=a_step, offset_ms=(a_step - k_run) * period_s * 1e3,
               edge_h7_time_us=[early_t, late_t], u_t_s=0.5 * (late - early) * period_s,
               u_t_parts_s=dict(half_bracket_steps=0.5 * (late - early) * period_s))
    if late - early > BRACKET_MAX_STEPS:
        out['reason'] = f'bracket {late - early:.2f} steps > {BRACKET_MAX_STEPS}'
    elif not p['ok']:
        out['reason'] = f'PWM cross-check failed: {p["reason"]}'
    elif early > p['a_hi'] + PWM_SLACK_STEPS:
        out['reason'] = (f'PWM cross-check: duty follows the pwm at shift <= {p["a_hi"]:.2f}, before the early '
                         f'bound {early:.2f} (a duty cannot precede its write)')
    elif p['a_lo'] - late > PWM_LAG_MAX_STEPS:
        out['reason'] = (f'PWM cross-check: duty follows the pwm {p["a_lo"] - late:.2f} steps after the late '
                         f'bound (> {PWM_LAG_MAX_STEPS})')
    else:
        out['ok'] = True
    return out


def load_frames(frame_log: str) -> dict:
    """t_send_ns, t_pulse0_ns, mosi, miso of a run_hil.py frame log (None for a missing array; L100 uses mosi,
    miso)."""
    with np.load(frame_log, allow_pickle=False) as z:
        return {k: (np.array(z[k]) if k in z.files else None) for k in ('t_send_ns', 't_pulse0_ns', 'mosi', 'miso')}


def map_run(frame_log: str, esp_log: str) -> dict:
    """map_edge on the files (frame log of run_hil.py, ESP32 CSV of agv_logger.py)."""
    import hil_o1_score as H
    manifest, mosi, miso = H.load(frame_log)
    state, ref, flags, duty = H.decode_frames(mosi, miso)
    k_run, src = H.command_start(manifest, ref, flags)
    out = map_edge(duty, k_run, read_esp_log(esp_log), float(manifest.get('period_us', 1000)) * 1e-6,
                   load_frames(frame_log))
    out.update(k_run_src=src, frame_log=os.path.basename(frame_log), esp_log=os.path.basename(esp_log))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frame-log', required=True)
    ap.add_argument('--esp-log', required=True)
    ap.add_argument('--json', default='')
    a = ap.parse_args(argv)
    try:
        out = map_run(a.frame_log, a.esp_log)
    except (ValueError, KeyError, OSError) as exc:
        print('hil_edge_map:', exc, file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
    print(text)
    print(f'#ket_qua,ma={0 if out["ok"] else 3}')
    return 0 if out['ok'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
