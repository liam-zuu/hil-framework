#!/usr/bin/env python3
"""s05_fake_runs: FAKE run folders for the C9 dry run (rq3_log demo) and the tests. Nothing here is a measurement.

CODE 16/11/2026 to CODE_C13 (Liam 16/11 at CODE, hop chon, nguyen van: "Sửa demo + gộp R134(1) (Recommended)"):
the demo builds every judged run's folder, so s05_final_table runs the whole chain on the dry run (C9 dry04 stopped
with exit 2: the final runs had no query files). The formats follow the writers and the firmware, as the tests of
s05_final_table build them (tools/tests/test_s05_final_table.py: Log, start_events, answer, ack_text, write_log):
  esp32.csv               one data line per motor-on tick (100 Hz: tick +1, log_seq +1, dpwm += |d pwm|), motor-off
                          lines at 10 Hz (log_seq +1, tick unchanged), three '#run_end' lines at an on -> off edge
                          (log_seq +3, not in the CSV; AGV 54b7269 / 16702ff logger.c)
  esp32.csv.events.txt    agv_logger events: the '#gain' line of the gain setting with the motor off (first_seq = 0,
                          AGV 16702ff main.c logger_push_gain(0)), then per motor-on stretch '#fw' x3, '#gain' x3,
                          '#us,ev=start' x3 (run_udp_mark), and the '#run_end' lines
  gain_query_before.txt   gain_send.py --query output just before / after the run (writer's '#query_pc' line,
  gain_query_after.txt    '#gain_running', '#ket_qua,ma=0'; R134C2-1)
  gain_ack.txt            gain_send.py output of the gain setting (--gain-file; B81-14)
  score.json              the scorer keys s05_final_table reads (t_end, e_*; HIL: edge_map.esp_pi_ppm,
                          pi_period_ppm, track_err_frames; car: n_back_frames)
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os

WHEELS = ('dpwm_fl', 'dpwm_fr', 'dpwm_rl', 'dpwm_rr')        # s05_final_table.WHEELS
COLS = ['timestamp_us', 'tick', 'log_seq', 'motor_on', 'run_active'] + list(WHEELS)
QUERY_NAMES = ('gain_query_before.txt', 'gain_query_after.txt')   # rpi5/pc_sender.py QUERY_NAMES
ACK_NAME = 'gain_ack.txt'
SHA = '16702ff0e688'
BOOT = '4753dc47'
MAC = 'aa:bb'


class Log:
    """ESP32 log lines tick by tick (test_s05_final_table.Log; seq0: log_seq before the first line)."""

    def __init__(self, t_us: int = 1_000_000, seq0: int = 50, tick0: int = 100):
        self.t, self.tick, self.seq, self.dp, self.rows, self.idle, self.pwm = t_us, tick0, seq0, [0] * 4, [], 0, [0] * 4
        self.run_ends = []                                  # log_seq of each '#run_end' line

    def _line(self, on: bool, ra: bool) -> None:
        self.rows.append(dict(timestamp_us=str(self.t), tick=str(self.tick), log_seq=str(self.seq),
                              motor_on='1' if on else '0', run_active='1' if ra else '0',
                              **{w: str(self.dp[i]) for i, w in enumerate(WHEELS)}))

    def off(self, n_ticks: int, after_on: bool = False) -> None:
        if after_on:
            for _ in range(3):                              # three '#run_end' lines
                self.seq += 1
                self.run_ends.append(self.seq)
        for _ in range(n_ticks):
            self.t += 10_000
            self.idle += 1
            if self.idle >= 10:
                self.idle = 0
                self.seq += 1
                self._line(False, False)

    def on(self, n_ticks: int, pwm_fn=lambda k: [100 + 10 * (k % 2)] * 4, ra: bool = True) -> None:
        self.idle = 0
        for k in range(n_ticks):
            self.t += 10_000
            p = pwm_fn(k)
            for i in range(4):
                self.dp[i] += abs(p[i] - self.pwm[i])
            self.pwm = p
            self.tick += 1
            self.seq += 1
            self._line(True, ra)


def stamp(lines, t0: str, step_us: int = 1000) -> list:
    """agv_logger events lines: '<iso time> <line>', step_us apart from t0."""
    base = dt.datetime.fromisoformat(t0)
    return [f'{(base + dt.timedelta(microseconds=k * step_us)).isoformat(timespec="microseconds")} {x}'
            for k, x in enumerate(lines)]


def gain_line(gain: dict, first_seq: int, drop: int = 0, boot: str = '') -> str:
    """'#gain' line of a gain file dict(ctrl, id, gains) as the firmware writes it (values compared as float32)."""
    vals = ','.join(f'{k}={v!r}' for k, v in gain['gains'].items())
    return (f'#gain,id={gain["id"]},applied_n=1,ctrl={gain["ctrl"]},{vals},first_seq={first_seq},'
            f'line_drop={drop}' + (f',boot={boot}' if boot else ''))


def stretch_starts(rows) -> list:
    """log_seq of the first data line of every motor-on stretch (test_s05_final_table.stretch_starts)."""
    out = []
    for i, r in enumerate(rows):
        if r['motor_on'] != '1':
            continue
        p = rows[i - 1] if i else None
        if p is None or p['motor_on'] != '1' or (int(r['log_seq']) - int(p['log_seq'])
                                                 != int(r['tick']) - int(p['tick'])):
            out.append(int(r['log_seq']))
    return out


def answer(boot: str, gid: int, t: tuple, mac: str = MAC) -> str:
    """A saved gain_send --query output (test_s05_final_table.answer)."""
    return (f'#query_pc,t_start={t[0]:.6f},t_end={t[1]:.6f}\n'
            f'#gain_running,sha={SHA},env=pid,mac={mac},boot={boot},id={gid},applied_n=1,ctrl=PID\n'
            '#ket_qua,ma=0\n')


def ack_text(boot: str, gid: int, mac: str = MAC) -> str:
    """A saved gain_send.py output of setting set `gid` (test_s05_final_table.ack_text)."""
    return (f'# send 192.168.1.50:5014 #gain_set,v=1,tx=<new each try>,id={gid},ctrl=PID\n'
            f'# ack tx=12345,req_id={gid},ok=1,err=,sha={SHA},env=pid,mac={mac},boot={boot},id={gid},'
            'applied_n=1,ctrl=PID\n'
            f'# applied: id={gid} ctrl=PID applied_n=1 mac={mac} boot={boot}\n#ket_qua,ma=0\n')


def write_gain_file(path: str, gain: dict) -> str:
    """A gain file (gain_send.load_file format: ctrl, id, gains), marked FAKE."""
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(dict(gain, FAKE=True), f)
        f.write('\n')
    return path


def sample_gain(controller='PID'):
    if controller == 'PID':
        return dict(ctrl='PID', id=1,
                    gains=dict(kp_xy=1.0, ki_xy=0.5, kp_wz=1.0, ki_wz=0.2, pos_kp=6.0, pos_ki=3.0, k_psi=1.0))
    return dict(ctrl='ADRC', id=1,
                gains=dict(omega_c_xy=2.0, omega_o_xy=6.0, omega_c_wz=2.0, omega_o_wz=6.0,
                           pos_kp=6.0, pos_ki=3.0, k_psi=1.0))


def write_trial_gain(d, controller='PID', boot=BOOT):
    """FAKE trial gain and its short ESP32 log, for tests of rq3_log end."""
    os.makedirs(d, exist_ok=True)
    gain = sample_gain(controller)
    log = Log()
    log.off(10)
    log.on(5)
    write_esp32(d, log, gain, boot, '2026-11-20T10:00:00', ())
    return write_gain_file(os.path.join(d, 'gain_trial.json'), gain)


def write_queries(d: str, boot: str = BOOT, gid: int = 1, t0: float = 100.0) -> tuple:
    """The two query outputs of a run in folder d (before: t0..t0+1, after: t0+100..t0+101); their paths."""
    os.makedirs(d, exist_ok=True)
    paths = []
    for name, t in zip(QUERY_NAMES, ((t0, t0 + 1.0), (t0 + 100.0, t0 + 101.0))):
        p = os.path.join(d, name)
        with open(p, 'w', encoding='utf-8', newline='\n') as f:
            f.write(answer(boot, gid, t))
        paths.append(p)
    return tuple(paths)


def write_final_run(d: str, platform: str, gain: dict, e_perp: float = 0.05, t_end: float = 10.0,
                    boot: str = BOOT, t0: str = '2026-11-20T10:00:00', log: Log = None, run_end_extra=(),
                    score_extra: dict = None) -> dict:
    """A FAKE run folder with everything s05_final_table asks of a judged run (score.json, esp32.csv and its
    events, both query outputs, the ack of the gain setting); `log`: a Log already filled (default: 0.3 s off, 10 s
    + 3 ticks on, then off). run_end_extra: more '#run_end' log_seq values for the events (tests). score_extra: more
    keys of score.json (C9D-1: 'outcome' and 'gate_v3', so score.json is the scorer file given to rq3_log end).
    Returns the paths of the query outputs."""
    os.makedirs(d, exist_ok=True)
    if log is None:
        log = Log()
        log.off(30)
        log.on(int(round(t_end * 100)) + 3)
        log.off(20, after_on=True)
    score = dict(FAKE=True, e_perp_max=e_perp, e_s_absmax=0.01, e_trk_max=0.02, e_bao_max=0.08,
                 e_theta_lane_max_deg=5.0, e_theta_cmd_max_deg=4.0)
    if platform == 'HIL':
        score.update(t_end=t_end, track_err_frames=0, pi_period_ppm=-12.0, edge_map=dict(esp_pi_ppm=24.6))
    else:
        t0_log = next((int(r['timestamp_us']) for r in log.rows if r['run_active'] == '1'), None)
        score.update(t_end_union=t_end, t0_log_us=t0_log, n_back_frames=0)
    score.update(score_extra or {})
    with open(os.path.join(d, 'score.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(score, f)
    write_esp32(d, log, gain, boot, t0, run_end_extra)
    with open(os.path.join(d, ACK_NAME), 'w', encoding='utf-8', newline='\n') as f:
        f.write(ack_text(boot, gain['id']))
    return write_queries(d, boot, gain['id'])


def write_esp32(d, log, gain, boot, t0, run_end_extra):
    with open(os.path.join(d, 'esp32.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(log.rows)
    base = dt.datetime.fromisoformat(t0)
    ev = stamp([gain_line(gain, 0)], (base - dt.timedelta(seconds=10)).isoformat())     # the gain setting
    env = 'pid' if gain['ctrl'] == 'PID' else 'ladrc'
    for k, s in enumerate(stretch_starts(log.rows)):
        ev += stamp([f'#fw,sha=5bde4c55bf3a,dirty=0,env={env}'] * 3 + [gain_line(gain, s, boot=boot)] * 3
                    + [f'#us,ev=start,boot={boot},on=0,swb=1500,first_seq={s},line_drop=0'] * 3,
                    (base + dt.timedelta(seconds=5 * k)).isoformat())
    ends = list(log.run_ends) + list(run_end_extra)
    ev += stamp([f'#run_end,seq={s},last_data_seq={s - 1},log_qdrop=0,log_send_err=0' for s in ends],
                (base + dt.timedelta(seconds=60)).isoformat())
    with open(os.path.join(d, 'esp32.csv.events.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(''.join(f'{x}\n' for x in ev))
