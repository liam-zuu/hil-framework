"""
hil_o1_score.py: score one O1 run on the HIL bench from the Pi frame log (run_hil.py --frame-log),
on the TRUE plant state the Pi sent to the H7. CODE session local_7787e22f, 11/11/2026 (Liam:
"Duyệt, làm theo thứ tự"; EA-13 of the B8 early review: no HIL scorer existed).

S05 v3.7 section 4.6 (EA-13): a HIL run (candidate check 4.5 step 2, final check on HIL) is scored
with evaluate_v7 by the S03 section 4 rule; the re-search of the window and the backward flag of
O1 3.3 are for the camera only. So:
  figure 8 (O1-F02/F05/F08)  fwsim/o1_feasibility.evaluate_v7 unchanged (imported, its blob kept:
                             it is in the S04 SIM_SOURCES lock)
  K1, K2 (O1-K1/K2)          tools/o1_lanes.evaluate_k (same rule, lane of O1 3.7)

Frame k of the log holds the plant state at step k before the step (MOSI bytes 0-19:
omega 1-4, x, y, theta, vx, vy, wz, int16, protocol.STATE_RANGES; x, y step 5/32767 m = 0.15 mm)
and the duty the H7 measured (MISO bytes 0-7, duty x 32767, the 'torque' slots the Pi plant reads with
protocol.decode_torques; MISO v4 bytes 8-9 are the CRC: this scorer read bytes 8-15 until 11/11/2026,
so its duty, sat_frac and tv_duty of runs scored before were wrong; pose fields were not affected).
tv_duty here is the H7-measured duty at 1 kHz (written as tv_duty_h7_1khz): capture noise makes it
larger than the S05 4.7 / S03 Y3 tv_duty, which comes from the ESP32 counters (s05_final_table.py).
The plant starts at (0, 0, 0)
held through ARM; t = 0 (command start) is the first RUN step of the lifecycle in the manifest
(the publisher starts its trajectory clock when it sees RUN: trajectory_pub, polled every 2 ms),
else the first frame whose trajectory part is VALID and non-zero (t0_src says which).
t of frame k = (k - k0) x period: the plant time (fixed 1 ms step), not the Pi clock.
[CÁCH HIỂU] pwm for the saturation share = duty x 1000 (firmware PWM scale of fwsim).
Log too short (L85 Q1, Liam 11/11 CODE local_821ee4dc; Fable O1/C6 F-1b): when the last frame comes
before t_deadline - one period and the evaluator says completed 'false', the result is
'inconclusive' (completed_raw keeps 'false'); log_end_t, log_covers_deadline are always written.
run_hil.py --o1 --finite-run keeps the zeros and the log to T_cmd + o1_tasks.O1_TAIL_S (2.5 s).

Gate v3.2 (13/11/2026, Liam CODE local_db99d287 "Làm rs4 + g4 đi", local_7f7b0ef4 "Duyệt gộp luôn
lo/hi"): scored with fwsim/o1_gate_eval.py (evaluate_v7 / evaluate_k unchanged, plus G4 and the per-frame
values) and fwsim/o1_gate.gate (G1-G5, per-frame lo / hi / n_undef), block 'gate_v3' of the output.
RS-4: with --esp-log (the agv_logger CSV of the same run), t = 0 is the ESP32 run_active rising edge
mapped to the plant step (tools/hil_edge_map.py): t of frame k = (k - a) x period, frames k >= a scored,
U_t = the mapping bound, all five conditions decide; t0_src 'esp32_run_active_edge', edge_map block
with offset_ms = edge - RUN step (the "chênh" of O1 3.4). Without --esp-log, or when the mapping is
refused: t = 0 at the RUN step as before, U_t unknown, G3, G5 inconclusive, G4 reported (decide G1, G2;
O1 v11.9 3.5). The mapping (L98, L99) brackets the edge from the encoder counts read within 50 us of t_now
(AGV firmware in FW_COUNTS_AT_TICK; others refused) and the H7 apply times read from the MISO answers; the PWM
side is a cross-check.
Validity (CODE L50, STATUS_CODE CG4) is NOT decided here: the manifest counts (MISO CRC bad,
empty frames) are copied out; the run rule is tools/s06_frames_decode.py on the same log.

  python tools/hil_o1_score.py --frame-log <run>.npz --task O1-F05 [--esp-log <run>.csv] [--out score.json]
  (K1, K2 need --allow-final-set: O1 3.7, final test set only)
Last line: #ket_qua,ma=0 scored / 2 error.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'rpi5'), os.path.join(HERE, '..', 'fwsim'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import protocol as P                                                     # noqa: E402
import o1_lanes                                                          # noqa: E402
import o1_gate                                                           # noqa: E402
import o1_gate_eval                                                      # noqa: E402
import hil_edge_map                                                      # noqa: E402
from o1_feasibility import Path8, Schedule                               # noqa: E402
from o1_tasks import FIG8, FINAL_SET                                     # noqa: E402


def _int16(a, ofs, n):
    """Big-endian int16 columns [ofs, ofs + 2n) of a (frames, 40) uint8 array."""
    b = np.ascontiguousarray(a[:, ofs:ofs + 2 * n])
    return b.view('>i2').astype(np.float64).reshape(len(a), n)


def decode_frames(mosi, miso):
    """(state (n, 10) [omega1-4, x, y, theta, vx, vy, wz], traj_ref (n, 3), traj_flags (n,), duty (n, 4))."""
    state = _int16(mosi, 0, 10) / 32767.0 * np.asarray(P.STATE_RANGES)
    ref = _int16(mosi, P.OFS_TRAJ_REF, 3) / 32767.0 * np.asarray(P.REF_RANGES)
    flags = np.asarray(mosi[:, P.OFS_TRAJ_FLAGS], dtype=np.uint8)
    duty = _int16(miso, 0, 4) / 32767.0 * P.TORQUE_RANGE                # = protocol.decode_torques
    return state, ref, flags, duty


def command_start(manifest, ref, flags):
    """(k0, source): first RUN step of the lifecycle, else first VALID non-zero trajectory frame."""
    lc = manifest.get('lifecycle') or {}
    for tr in lc.get('transitions', []):
        if tr.get('phase') == 'RUN':
            return int(tr['step']), 'lifecycle_RUN'
    on = np.nonzero((flags & P.TRAJ_VALID).astype(bool) & np.any(np.abs(ref) > 0, axis=1))[0]
    if len(on):
        return int(on[0]), 'first_valid_ref'
    raise ValueError('no RUN step in the manifest and no VALID non-zero trajectory frame')


def load(path):
    z = np.load(path, allow_pickle=False)
    manifest = json.loads(str(z['manifest']))
    return manifest, z['mosi'], z['miso']


def log_coverage(out, t_log_end, period):
    """L85 Q1 (Fable O1/C6 F-1b), in place on an evaluator summary: the evaluators score frames
    t < t_deadline, so a log whose last frame is before t_deadline - period cannot say 'false' (the
    completion window may lie after its end): 'false' -> 'inconclusive', completed_raw keeps it."""
    covers = t_log_end >= float(out['t_deadline']) - period - 1e-9
    out.update(log_end_t=float(t_log_end), log_covers_deadline=bool(covers))
    if not covers and out.get('completed') == 'false':
        out.update(completed='inconclusive', completed_raw='false',
                   completed_note=f'log ends at t = {t_log_end:.3f} s, before t_deadline = '
                                  f'{float(out["t_deadline"]):.3f} s; every measure is on [0, log end] only')
    return out


def score(path, task, allow_final_set=False, esp_log=None):
    if task not in FIG8 and task not in FINAL_SET:
        raise ValueError(f'task {task!r}: one of {tuple(FIG8) + FINAL_SET}')
    if task in FINAL_SET and not allow_final_set:
        raise ValueError(f'{task} is the final test set (O1 3.7): only with --allow-final-set')
    manifest, mosi, miso = load(path)
    state, ref, flags, duty = decode_frames(mosi, miso)
    k_run, run_src = command_start(manifest, ref, flags)
    period = float(manifest.get('period_us', 1000)) * 1e-6
    # RS-4: t = 0 at the ESP32 run_active edge mapped to the plant step, when the ESP32 log maps
    em = dict(ok=False, reason='no --esp-log')
    if esp_log:
        em = hil_edge_map.map_edge(duty, k_run, hil_edge_map.read_esp_log(esp_log), period,
                                   hil_edge_map.load_frames(path))
        em['esp_log'] = os.path.basename(esp_log)
    if em['ok']:
        a = float(em['edge_step'])
        k0, t0_src, u_t, decide = int(math.ceil(a - 1e-9)), 'esp32_run_active_edge', float(em['u_t_s']), o1_gate.DECIDE_ALL
        note = (f'HIL: t = 0 at the ESP32 run_active edge, plant step {a:.2f} ({em["offset_ms"]:+.2f} ms from the RUN '
                f'step); U_t = {u_t * 1e3:.2f} ms from the mapping (RS-4)')
    else:
        a, k0, t0_src, u_t, decide = float(k_run), k_run, run_src, None, o1_gate.DECIDE_HIL
        note = (f'HIL: t = 0 at the plant RUN step, run_active edge not mapped ({em["reason"]}): U_t not known, '
                'G3, G5, G4 report only (G1, G2 decide, O1 3.5)')
    sl = slice(k0, len(state))
    log = dict(t=(np.arange(len(state))[sl] - a) * period, pose=state[sl, 4:7], v_true=state[sl, 7:10],
               duty=duty[sl], pwm=np.round(duty[sl] * 1000.0))
    if task in FIG8:
        out, _, frames = o1_gate_eval.evaluate_fig8(log, Path8(), Schedule(FIG8[task]))
        out['task'] = task
    else:
        out, _, frames = o1_gate_eval.evaluate_k(log, task)
    out['tv_duty_h7_1khz'] = out.pop('tv_duty')                          # not the S05 4.7 tv_duty (docstring)
    log_coverage(out, float(log['t'][-1]) if len(log['t']) else -1.0, period)
    # gate v3.2 (o1_gate.py): true plant state, so U_p = U_psi = 0
    if task in FIG8:
        v_max, wz_max = o1_gate.fig8_cmd_max(Schedule(FIG8[task]))
    else:
        v_max, wz_max = o1_gate.k_cmd_max(o1_lanes.KSchedule(task))
    out['gate_v3'] = o1_gate.gate(frames, out['completed'], v_cmd_max=v_max, wz_cmd_max=wz_max, u_t=u_t,
                                  decide=decide, status=out.get('status', ''), note=note)
    mc = manifest.get('miso_check') or {}
    out.update(frame_log=os.path.basename(path), k0=k0, k_run=k_run, t0_step=a, t0_src=t0_src,
               frames_scored=int(len(state) - k0), edge_map=em,
               start_pose=[float(v) for v in state[k0, 4:7]], plant=manifest.get('plant'),
               plant_src=manifest.get('plant_src'), imu_noise=manifest.get('imu_noise'),
               lifecycle_outcome=(manifest.get('lifecycle') or {}).get('outcome'),
               miso_crc_bad=mc.get('crc_bad'), miso_empty=mc.get('empty'),
               validity='see tools/s06_frames_decode.py (L50)',
               rule='S05 v3.7 4.6: evaluate_v7, S03 section 4; gate v3.2: o1_gate_eval + o1_gate')
    return out


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frame-log', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--allow-final-set', action='store_true')
    ap.add_argument('--esp-log', default='', help='agv_logger CSV of the run: t = 0 at the run_active edge (RS-4)')
    ap.add_argument('--out', default='')
    a = ap.parse_args(argv)
    try:
        out = _json_safe(score(a.frame_log, a.task, a.allow_final_set, a.esp_log or None))
    except (ValueError, KeyError, OSError) as exc:
        print('hil_o1_score:', exc, file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
    print(text)
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
