"""Tests for tools/s05_final_table.py (S05 v3.8 4.7: tv_duty from the ESP32 counters, comparison table).
Synthetic logs follow the firmware (AGV 54b7269): one data line per motor-on tick (100 Hz, tick +1,
log_seq +1, dpwm += |dpwm|), motor-off lines at 10 Hz (log_seq +1, tick unchanged), three '#run_end'
lines at an on -> off edge (log_seq +3, not in the CSV)."""
import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import s05_final_table as T  # noqa: E402

COLS = ['timestamp_us', 'tick', 'log_seq', 'motor_on', 'run_active'] + list(T.WHEELS)


class Log:
    """Builds ESP32 log lines tick by tick."""

    def __init__(self, t_us=1_000_000):
        self.t, self.tick, self.seq, self.dp, self.rows, self.idle, self.pwm = t_us, 100, 50, [0] * 4, [], 0, [0] * 4

    def _line(self, on, ra):
        self.rows.append(dict(timestamp_us=str(self.t), tick=str(self.tick), log_seq=str(self.seq),
                              motor_on='1' if on else '0', run_active='1' if ra else '0',
                              **{w: str(self.dp[i]) for i, w in enumerate(T.WHEELS)}))

    def off(self, n_ticks, after_on=False):
        if after_on:
            self.seq += 3                                     # three #run_end lines
        for _ in range(n_ticks):
            self.t += 10_000
            self.idle += 1
            if self.idle >= 10:
                self.idle = 0
                self.seq += 1
                self._line(False, False)

    def on(self, n_ticks, pwm_fn, ra=True, keep=lambda k: True):
        self.idle = 0
        for k in range(n_ticks):
            self.t += 10_000
            p = pwm_fn(k)
            for i in range(4):
                self.dp[i] += abs(p[i] - self.pwm[i])
            self.pwm = p
            self.tick += 1
            self.seq += 1
            if keep(k):
                self._line(True, ra)


def zigzag(k):
    return [100 + 10 * (k % 2)] * 4                          # |dpwm| = 10 per wheel per tick after the first


def tv(rows, t_end_s, **kw):
    """tv_duty_esp32 without the boot and '#gain' checks (counter tests; those checks have their own tests)."""
    return T.tv_duty_esp32(rows, t_end_s, check_boot=False, check_gain_line=False, **kw)


def test_clean_run_gives_the_rate_of_the_counters():
    L = Log()
    L.off(30)
    L.on(1001 + 3, zigzag)                                    # + tail after t_end (RS-5)
    r = tv(L.rows, t_end_s=10.0)
    # window: first run_active line .. line at +10.00 s: 1000 steps of 10 per wheel x 4 wheels / 1000 / 10 s
    assert r['reason'] == '' and r['d_tick'] == r['d_seq'] == 1000
    assert r['tv_duty'] == pytest.approx(4 * 10 * 1000 / 1000 / 10.0)


def test_lost_lines_inside_the_window_do_not_matter():
    L = Log()
    L.off(30)
    L.on(1001 + 3, zigzag, keep=lambda k: k % 7 != 3)
    r = tv(L.rows, t_end_s=10.0)
    assert r['tv_duty'] == pytest.approx(4.0) and r['lines_lost'] > 100


def test_motor_off_for_a_few_ticks_is_seen_without_an_off_line():
    L = Log()
    L.off(30)
    L.on(400, zigzag)
    L.off(5, after_on=True)                                   # 5 ticks off: no 10 Hz line, but #run_end x 3
    L.on(601, zigzag)
    r = tv(L.rows, t_end_s=10.0)                              # t = 0 found from the run_active edge
    assert r['tv_duty'] is None and 'motor tat' in r['reason']


def test_counter_reset_in_the_window():
    L = Log()
    L.off(30)
    L.on(300, zigzag)
    L.tick = 0                                                # vel_ctrl_set_mode: counters_reset
    L.dp = [0] * 4
    L.on(700, zigzag)
    r = tv(L.rows, t_end_s=9.9)
    assert r['tv_duty'] is None and 'khong tang' in r['reason']


def test_end_gap_and_start_edge_rules():
    L = Log()
    L.off(30)
    L.on(1001, zigzag, keep=lambda k: k < 995)                 # last 6 lines lost: end gap 60 ms
    r = tv(L.rows, t_end_s=10.0)
    assert r['tv_duty'] is None and 'khoang bien' in r['reason']
    L = Log()
    L.off(30)
    L.on(1001, zigzag, keep=lambda k: k >= 4)                  # edge line and 3 after lost: 40 ms + idle gap
    r = tv(L.rows, t_end_s=9.9)
    assert r['tv_duty'] is None and 'canh t = 0' in r['reason']
    L = Log()
    L.off(30)
    L.on(1001, zigzag, keep=lambda k: k >= 1)                  # only the edge line lost, 10 ms after an
    L.rows.insert(3, dict(L.rows[2], timestamp_us=str(int(L.rows[3]['timestamp_us']) - 10_000)))  # idle line
    r = tv(L.rows, t_end_s=9.9)
    assert r['reason'] == '' and r['t0_edge_contiguous'] is False and r['t0_edge_gap_ms'] <= T.EDGE_MAX_MS


def test_two_rising_edges_have_no_number():
    L = Log()
    L.off(30)
    L.on(300, zigzag)
    L.on(5, zigzag, ra=False)
    L.on(300, zigzag)
    r = tv(L.rows, t_end_s=5.0)
    assert r['tv_duty'] is None and '2 lan run_active' in r['reason']


REAL = r'D:\CLAUDE_THESIS\05_DATA\results\CLOSED_LOOP\2026-11-11_run01'


@pytest.mark.skipif(not os.path.isdir(REAL), reason='real HIL run not on this machine')
@pytest.mark.parametrize('task,expect', [('f02', 3.7888), ('f05', 3.9676), ('f08', 5.0209)])
def test_real_hil_run_2026_11_11(task, expect):
    sc = json.load(open(os.path.join(REAL, 'analysis', f'score_{task}.json')))
    rows = T.read_esp32_log(os.path.join(REAL, 'pi', f'{task}.csv'))
    r = tv(rows, sc['t_end'])
    assert r['tv_duty'] == pytest.approx(expect, abs=1e-4) and r['d_tick'] == r['d_seq'] and r['lines_lost'] == 0


# ---------------------------------------------------------------- table

def write_log(d, rows, boots=('4753dc47', '4753dc47'), gain_seq=None, name='esp32.csv'):
    """esp32.csv, its events file with the run's '#gain' line (first_seq = first motor-on line, or gain_seq)
    three times, and gain_send --query outputs before / after (None: file not written)."""
    with open(os.path.join(d, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    first = next(r['log_seq'] for r in rows if r['motor_on'] == '1')
    g = ('#gain,id=7,applied_n=1,ctrl=PID,kp_xy=1.5,ki_xy=0.0500000007,kp_wz=1,ki_wz=0.0500000007,pos_kp=6,'
         f'pos_ki=3,k_psi=1,first_seq={gain_seq if gain_seq is not None else first},line_drop=0')
    with open(os.path.join(d, name + '.events.txt'), 'w', encoding='utf-8') as f:
        f.write(''.join(f'2026-11-14T10:00:00.00000{k} {g}\n' for k in range(3)))
    for when, b in zip(('before', 'after'), boots or (None, None)):
        if b is not None:
            with open(os.path.join(d, f'gain_query_{when}.txt'), 'w', encoding='utf-8') as f:
                f.write(f'#gain_running,sha=16702ff0e688,env=pid,mac=aa:bb,boot={b},id=7,applied_n=1,ctrl=PID\n'
                        '#ket_qua,ma=0\n')


def write_run(base, uid, platform, stage, ctrl, task, gain_set_id, outcome, e_perp, tv_ok=True, flag=0,
              boots=('4753dc47', '4753dc47')):
    d = os.path.join(base, uid)
    os.makedirs(d)
    k = task.startswith('O1-K')
    score = dict(e_perp_max=e_perp, e_s_absmax=0.01, e_trk_max=0.02, t_end=10.0,
                 track_err_frames=flag if platform == 'HIL' else 0, n_back_frames=flag if platform == 'HW' else 0)
    score.update(e_theta_start_max_deg=3.0) if k else score.update(e_theta_lane_max_deg=5.0, e_theta_cmd_max_deg=4.0)
    json.dump(score, open(os.path.join(d, 'score.json'), 'w'))
    L = Log()
    L.off(30)
    L.on(1001 + 3, zigzag) if tv_ok else (L.on(400, zigzag), L.off(5, after_on=True), L.on(601 + 3, zigzag))
    write_log(d, L.rows, boots=boots)
    return dict(run_uid=uid, platform=platform, stage=stage, controller=ctrl, trajectory=task,
                gain_set_id=gain_set_id, run_outcome=outcome, results_dir=d)


def test_table_cells_medians_and_reasons(tmp_path):
    runs = [
        write_run(tmp_path, 'r1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.020),
        write_run(tmp_path, 'r2', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'khong_dat', 0.120, flag=3),
        write_run(tmp_path, 'r3', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.030, tv_ok=False),
        write_run(tmp_path, 'r4', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'hong_he_do', 0.900),
        write_run(tmp_path, 'r5', 'HIL', 'final_hil', 'PID', 'O1-K1', 'RQ2-INIT-PID/G00', 'chua_ket_luan', 0.050,
                  boots=('4753dc47', '0badf00d')),                      # boot differs: tv_duty has no number
        write_run(tmp_path, 'r6', 'HW', 'screen', 'PID', 'O1-F05', 'A-PID/G03', 'dat', 0.010),   # not final
    ]
    ds = tmp_path / 'ds'
    os.makedirs(ds)
    with open(ds / 'm7_tuning_log.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    out = tmp_path / 'out'
    assert T.main(['table', '--dataset', str(ds), '--out', str(out)]) == 0
    tab = list(csv.DictReader(open(out / 'S05_final_table.csv', encoding='utf-8')))
    assert len(tab) == 2 * 5 * 2 * 3 * len(T.QUANTITIES)                  # fixed shape, empty cells kept
    cell = {t['quantity']: t for t in tab if (t['controller'], t['condition'], t['platform'], t['gain_set'])
            == ('PID', 'O1-F05', 'xe', 'A')}
    e = cell['e_perp_max']
    assert (e['n_valid'], e['n_dat'], e['n_khong_dat'], e['n_hong_he_do'], e['n_flag']) == ('3', '2', '1', '1', '1')
    assert float(e['median']) == pytest.approx(0.030) and float(e['max']) == pytest.approx(0.120) and e['n_m'] == '3'
    tv = cell['tv_duty']
    assert tv['n_m'] == '2' and float(tv['median']) == pytest.approx(4.0) and 'r3' in tv['missing']
    k = {t['quantity']: t for t in tab if (t['controller'], t['condition'], t['platform'], t['gain_set'])
         == ('PID', 'O1-K1', 'HIL', 'khoi_dau')}
    assert k['e_theta_lane_max_deg']['n_m'] == '0' and 'khong ap dung' in k['e_theta_lane_max_deg']['missing']
    assert float(k['e_theta_cmd_max_deg']['median']) == pytest.approx(3.0)
    assert k['tv_duty']['n_m'] == '0' and 'khac boot' in k['tv_duty']['missing']
    md = (out / 'S05_final_table.md').read_text(encoding='utf-8')
    assert '| PID | O1-F05 | xe |' in md and '0.03 / 0.12 (n_m 3)' in md
    per_run = list(csv.DictReader(open(out / 'S05_final_runs.csv', encoding='utf-8')))
    assert {r['run_uid'] for r in per_run} == {'r1', 'r2', 'r3', 'r4', 'r5'}


def test_unknown_gain_set_is_reported_not_tabled(tmp_path, capsys):
    r = write_run(tmp_path, 'x', 'HW', 'final', 'PID', 'O1-F05', 'C-PID/G01', 'dat', 0.02)
    ds = tmp_path / 'ds'
    os.makedirs(ds)
    with open(ds / 'm7_tuning_log.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(r))
        w.writeheader()
        w.writerow(r)
    assert T.main(['table', '--dataset', str(ds), '--out', str(tmp_path / 'o')]) == 0
    assert 'khong vao bang' in capsys.readouterr().out


# ---------------------------------------------------------------- R134 package (RS-5, RS2-15, boot, #gain)

def _clean():
    L = Log()
    L.off(30)
    L.on(1001 + 3, zigzag)
    return L


def test_tail_line_after_the_last_sample_is_required():
    """RS-5: the data line after b (log_seq + 1) must be there with motor_on = 1 and a time >= t_end."""
    L = Log()
    L.off(30)
    L.on(1001, zigzag)                                        # the log ends at t_end: no line after b
    r = tv(L.rows, t_end_s=10.0)
    assert r['tv_duty'] is None and 'RS-5' in r['reason'] and 'khong co' in r['reason']
    L = Log()
    L.off(30)
    L.on(1001, zigzag)
    L.off(10, after_on=True)                                  # motor off just after b: 3 #run_end, then a 10 Hz line
    L.rows[-1]['log_seq'] = str(int(L.rows[-2]['log_seq']) + 1)   # make the off line the next log_seq
    r = tv(L.rows, t_end_s=10.0)
    assert r['tv_duty'] is None and 'motor_on = 0' in r['reason']
    r = tv(_clean().rows, t_end_s=10.0)
    assert r['reason'] == '' and r['tail_seq'] and r['tail_ms_after_t_end'] == pytest.approx(10.0)


def test_timestamps_must_increase_in_the_window():
    """RS2-15: a restart whose later lines carry a larger log_seq and tick but an earlier time."""
    L = _clean()
    i = 600
    L.rows[i] = dict(L.rows[i], timestamp_us=str(int(L.rows[i - 1]['timestamp_us']) - 5_000))
    r = tv(L.rows, t_end_s=10.0)
    assert r['tv_duty'] is None and 'tang ngat' in r['reason']


def _segments(path, rows, seg_of=lambda i: 0, status_of=lambda i: 'new'):
    with open(path + '.segments.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['row', 'log_seq', 'timestamp_us', 'segment', 'status'])
        for i, r in enumerate(rows, start=1):
            w.writerow([i, r['log_seq'], r['timestamp_us'], seg_of(i), status_of(i)])


def test_segments_of_the_logger(tmp_path):
    """RS2-15: window lines in two logger segments: no number; duplicates dropped; conflicts refused."""
    L = _clean()
    p = str(tmp_path / 'esp32.csv')
    write_log(str(tmp_path), L.rows)
    _segments(p, L.rows, seg_of=lambda i: 0 if i < 700 else 1)
    r = tv(T.read_esp32_log(p), t_end_s=10.0)
    assert r['tv_duty'] is None and 'doan logger' in r['reason']
    rows = L.rows[:500] + [dict(L.rows[499])] + L.rows[500:]          # one identical copy (L92)
    write_log(str(tmp_path), rows)
    _segments(p, rows, status_of=lambda i: 'duplicate' if i == 501 else 'new')
    assert tv(T.read_esp32_log(p), t_end_s=10.0)['tv_duty'] == pytest.approx(4.0)
    _segments(p, rows, status_of=lambda i: 'conflict' if i == 501 else 'new')
    with pytest.raises(ValueError, match='conflict'):
        T.read_esp32_log(p)


def test_boot_before_and_after_the_run(tmp_path):
    """S05 4.7 (iv), R3V-9: both gain_send --query answers, same boot."""
    L = _clean()
    first = int(next(r['log_seq'] for r in L.rows if r['motor_on'] == '1'))
    for boots, ok, why in ((('ab', 'ab'), True, ''), ((None, 'ab'), False, 'truoc'), (('ab', None), False, 'sau'),
                           (('ab', 'cd'), False, 'khac boot')):
        r = T.tv_duty_esp32(L.rows, 10.0, boots=boots, gain_first_seqs={first})
        assert (r['tv_duty'] is not None) == ok and why in r['reason'], (boots, r['reason'])
    d = str(tmp_path)
    write_log(d, L.rows, boots=('4753DC47', '4753dc47'))
    assert T.read_boot(os.path.join(d, 'gain_query_before.txt')) == '4753DC47'
    assert T.main(['tv', '--log', os.path.join(d, 'esp32.csv'), '--t-end', '10',
                   '--query-before', os.path.join(d, 'gain_query_before.txt'),
                   '--query-after', os.path.join(d, 'gain_query_after.txt')]) == 0
    with open(os.path.join(d, 'gain_query_after.txt'), 'w', encoding='utf-8') as f:
        f.write('# no answer\n#ket_qua,ma=3\n')
    assert T.read_boot(os.path.join(d, 'gain_query_after.txt')) is None


def test_gain_line_of_the_run(tmp_path):
    """S05 6.3: a '#gain' line whose first_seq is the first motor-on data line of the run."""
    L = _clean()
    d = str(tmp_path)
    write_log(d, L.rows, gain_seq=3)                                  # the line of another run
    p = os.path.join(d, 'esp32.csv')
    r = T.tv_duty_esp32(T.read_esp32_log(p), 10.0, boots=('a', 'a'), gain_first_seqs=T.gain_seqs(p))
    assert r['tv_duty'] is None and 'khong co dong #gain' in r['reason']
    os.remove(p + '.events.txt')
    assert T.gain_seqs(p) is None
    r = T.tv_duty_esp32(T.read_esp32_log(p), 10.0, boots=('a', 'a'), gain_first_seqs=None)
    assert 'thieu file events' in r['reason']
    write_log(d, L.rows)
    r = T.tv_duty_esp32(T.read_esp32_log(p), 10.0, boots=('a', 'a'), gain_first_seqs=T.gain_seqs(p))
    assert r['tv_duty'] == pytest.approx(4.0) and r['run_first_seq'] == int(L.rows[3]['log_seq'])
