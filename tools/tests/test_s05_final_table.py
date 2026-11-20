"""Tests for tools/s05_final_table.py (S05 v3.8 4.7: tv_duty from the ESP32 counters, comparison table).
Synthetic logs follow the firmware (AGV 54b7269): one data line per motor-on tick (100 Hz, tick +1,
log_seq +1, dpwm += |dpwm|), motor-off lines at 10 Hz (log_seq +1, tick unchanged), three '#run_end'
lines at an on -> off edge (log_seq +3, not in the CSV)."""
import csv
import re
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import s05_final_table as T  # noqa: E402
import rq3_log as rl  # noqa: E402  (R134C4-2: the table runs the log's row checks)

COLS = ['timestamp_us', 'tick', 'log_seq', 'motor_on', 'run_active'] + list(T.WHEELS)


@pytest.fixture
def no_moc(monkeypatch):
    """C8b v3 §5 (CODE_GATE 18/11, design 03_CODE/docs/2026-11-18_Thiet_ke_rq3_u_moc_v4.md): a car run of a campaign
    without its U / premise files is chua_ket_luan after rq3_log.apply_moc. The tests using this fixture check rules
    written before §5 on car runs that have no such files, so the §5 rule is set aside here; it is tested, with
    summary, freeze and s05_final_table, in test_rq3_u_moc.py."""
    monkeypatch.setattr(rl, 'apply_moc', lambda tun, ds: tun)


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

GAIN7 = ('#gain,id=7,applied_n=1,ctrl=PID,kp_xy=1.5,ki_xy=0.0500000007,kp_wz=1,ki_wz=0.0500000007,pos_kp=6,'
         'pos_ki=3,k_psi=1,first_seq={seq},line_drop={drop}')
US = '#us,ev=start,boot={boot},on=0,swb=1500,first_seq={seq},line_drop=0'


def stamp(lines, t0='2026-11-14T10:00:00', step_us=1000):
    """agv_logger events lines: '<iso time> <line>', step_us apart from t0."""
    import datetime
    base = datetime.datetime.fromisoformat(t0)
    return [f'{(base + datetime.timedelta(microseconds=k * step_us)).isoformat(timespec="microseconds")} {x}'
            for k, x in enumerate(lines)]


def start_events(seq, boot='4753dc47', gain_seq=None, drops=(0, 0, 0), t0='2026-11-14T10:00:00'):
    """Events lines of a run start as the firmware sends them (AGV 16702ff run_udp_mark): #fw x3, #gain x3,
    #us,ev=start x3, 1 ms apart from t0 (first_seq = first motor-on data line; gain_seq: another first_seq)."""
    g = gain_seq if gain_seq is not None else seq
    return stamp(['#fw,sha=16702ff0e688,dirty=0,env=pid'] * 3 + [GAIN7.format(seq=g, drop=d) for d in drops]
                 + [US.format(boot=boot, seq=seq)] * 3, t0)


def answer(boot, mac='aa:bb', t=(100.0, 101.0), gid=7):
    """A saved gain_send --query output: the writer's '#query_pc' line (R134C2-1), then the answer."""
    return (f'#query_pc,t_start={t[0]:.6f},t_end={t[1]:.6f}\n'
            f'#gain_running,sha=16702ff0e688,env=pid,mac={mac},boot={boot},id={gid},applied_n=1,ctrl=PID\n'
            '#ket_qua,ma=0\n')


# R134C6-1: the gain file of the set the runs carry (the values of GAIN7, id 7), frozen as A-PID/G07, and the same
# set given as the initial gain of PID (--init-gain)
GAIN7_FILE = dict(ctrl='PID', id=7, gains=dict(kp_xy=1.5, ki_xy=0.05, kp_wz=1.0, ki_wz=0.05, pos_kp=6.0, pos_ki=3.0,
                                                k_psi=1.0))


def write_gain_file(path, **kw):
    g = dict(GAIN7_FILE, **{k: v for k, v in kw.items() if k != 'gains'})
    g['gains'] = dict(GAIN7_FILE['gains'], **kw.get('gains', {}))
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(g, f)
    return str(path)


def write_gains(ds, frozen=(('A-PID', 'A-PID/G07'),)):
    """S05_freeze.csv of the dataset (rq3_log freeze format) with one gain file per frozen set, and the initial
    gain file of PID; returns the --init-gain map."""
    rows = []
    for camp, gs in frozen:
        f = write_gain_file(os.path.join(ds, f'gain_{gs.replace("/", "_")}.json'))
        rows.append(dict.fromkeys(rl.FREEZE_COLS, '') | dict(campaign=camp, gain_set_id=gs,
                                                              gain_file=os.path.basename(f),
                                                              gain_sha256=rl.sha256_file(f)))
    with open(os.path.join(ds, rl.FREEZE), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=rl.FREEZE_COLS)
        w.writeheader()
        w.writerows(rows)
    return dict(PID=write_gain_file(os.path.join(ds, 'gain_init_PID.json')))


def init_args(ds):
    return ['--init-gain', f'PID={os.path.join(ds, "gain_init_PID.json")}']


def LR(ds, **kw):
    return T.load_runs(ds, 'score.json', 'esp32.csv', init_gain=kw.pop('init_gain', {'PID': os.path.join(
        ds, 'gain_init_PID.json')}), **kw)


def ack_text(boot='4753dc47', mac='aa:bb', gid=7, ok=1, ma=0):
    """A saved gain_send.py output of setting set `gid` (hil feat/wifi-gain 331cd76 main; B81-14)."""
    return (f'# send 192.168.1.50:5014 #gain_set,v=1,tx=<new each try>,id={gid},ctrl=PID\n'
            f'# ack tx=12345,req_id={gid},ok={ok},err=,sha=16702ff0e688,env=pid,mac={mac},boot={boot},id={gid},'
            'applied_n=1,ctrl=PID\n'
            f'# applied: id={gid} ctrl=PID applied_n=1 mac={mac} boot={boot}\n#ket_qua,ma={ma}\n')


def _ack(boot='4753dc47', mac='aa:bb', gid='7'):
    """read_ack fields of the gain setting (B81-14)."""
    return dict(ok='1', boot=boot, mac=mac, id=gid)


def set_line(t0='2026-11-14T09:59:50', gid=7):
    """The events line of a gain setting with the motor off: '#gain' with first_seq = 0 (no '#us' with it)."""
    return stamp([GAIN7.format(seq=0, drop=0).replace('#gain,id=7,', f'#gain,id={gid},')], t0)


SET_LINE = set_line()


def stretch_starts(rows):
    """log_seq of the first data line of every motor-on stretch: after a motor-off line, or after a motor-on line
    with d log_seq != d tick (a short stop leaves no off line; R134C8-1)."""
    out = []
    for i, r in enumerate(rows):
        if r['motor_on'] != '1':
            continue
        p = rows[i - 1] if i else None
        if p is None or p['motor_on'] != '1' or (int(r['log_seq']) - int(p['log_seq'])
                                                 != int(r['tick']) - int(p['tick'])):
            out.append(int(r['log_seq']))
    return out


def write_log(d, rows, boots=('4753dc47', '4753dc47'), gain_seq=None, name='esp32.csv', run_boot='4753dc47',
              events=None, set_line=True):
    """esp32.csv, its events file with the run's start lines (start_events, or `events`), and gain_send
    --query outputs before / after (None: file not written). set_line: first the '#gain' line with first_seq = 0
    the firmware writes when the set is applied with the motor off (AGV 16702ff main.c logger_push_gain(0); the
    gain setting whose ack is written below; R134C9-1)."""
    with open(os.path.join(d, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    ev = events if events is not None else sum(
        (start_events(s, run_boot, gain_seq if k == 0 else None, t0=f'2026-11-14T10:00:{5 * k:02d}')
         for k, s in enumerate(stretch_starts(rows))), [])
    if set_line:
        ev = SET_LINE + ev
    with open(os.path.join(d, name + '.events.txt'), 'w', encoding='utf-8') as f:
        f.write(''.join(f'{t}\n' for t in ev))
    for when, b, t in zip(('before', 'after'), boots or (None, None), ((100.0, 101.0), (200.0, 201.0))):
        q = os.path.join(d, f'gain_query_{when}.txt')
        if os.path.exists(q):
            os.remove(q)
        if b is not None:
            with open(q, 'w', encoding='utf-8') as f:
                f.write(answer(b, t=t))
    with open(os.path.join(d, 'gain_ack.txt'), 'w', encoding='utf-8') as f:      # B81-14: ack of the setting
        f.write(ack_text(run_boot))


def write_run(base, uid, platform, stage, ctrl, task, gain_set_id, outcome, e_perp, tv_ok=True, flag=0,
              boots=('4753dc47', '4753dc47')):
    d = os.path.join(base, uid)
    os.makedirs(d)
    k = task.startswith('O1-K')
    score = dict(e_perp_max=e_perp, e_s_absmax=0.01, e_trk_max=0.02, t_end=10.0,
                 track_err_frames=flag if platform == 'HIL' else 0, n_back_frames=flag if platform == 'HW' else 0)
    score.update(e_theta_start_max_deg=3.0) if k else score.update(e_theta_lane_max_deg=5.0, e_theta_cmd_max_deg=4.0)
    gate = None
    if outcome in T.VALID:      # C9D-1: score.json is the scorer file of the run (outcome, gate_v3), sha in the log
        g = rl.fake_gate(e_trk=0.5 if outcome == 'khong_dat' else 0.03,
                         completed='inconclusive' if outcome == 'chua_ket_luan' else 'true')
        gate = rl.floor_gate(g) if platform == 'HW' else g
        score.update(outcome=outcome, gate_v3=gate)
    with open(os.path.join(d, 'score.json'), 'w', encoding='utf-8') as f:
        json.dump(score, f)
    L = Log()
    L.off(30)
    L.on(1001 + 3, zigzag) if tv_ok else (L.on(400, zigzag), L.off(5, after_on=True), L.on(601 + 3, zigzag))
    write_log(d, L.rows, boots=boots)
    # R134C4-2: the log row as rq3_log writes it (the table runs its row checks): campaign, the gate columns of a
    # judged run (FAKE gate agreeing with the outcome; a car run: camera gate), both boots of a judged car /
    # final_hil run
    row = dict.fromkeys(rl.TUNING_COLS, '')
    row.update(run_uid=uid, platform=platform, stage=stage, controller=ctrl, trajectory=task, gain_set_id=gain_set_id,
               run_outcome=outcome, results_dir=d, fail_cause='mat_khung' if outcome == 'hong_he_do' else '',
               campaign='RQ2' if stage == 'final_hil' else gain_set_id.split('/')[0])
    if outcome in T.VALID:
        row.update(rl.gate_columns(gate, rl.sha256_file(os.path.join(d, 'score.json'))))
        row.update(boot_before='4753dc47', boot_after='4753dc47')
    return row


def test_table_cells_medians_and_reasons(tmp_path, no_moc):
    runs = [
        write_run(tmp_path, 'r1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.020),
        write_run(tmp_path, 'r2', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'khong_dat', 0.120, flag=3),
        write_run(tmp_path, 'r3', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.030, tv_ok=False),
        write_run(tmp_path, 'r4', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'hong_he_do', 0.900),
        write_run(tmp_path, 'r5', 'HIL', 'final_hil', 'PID', 'O1-K1', 'RQ2-INIT-PID/G00', 'chua_ket_luan', 0.050),
        # R134C5-1: a run whose two answers differ is refused at run level (test_c5_1_...), not tabled
        write_run(tmp_path, 'r6', 'HW', 'screen', 'PID', 'O1-F05', 'A-PID/G03', 'dat', 0.010),   # not final
    ]
    ds = _table_ds(tmp_path, runs)
    out = tmp_path / 'out'
    assert T.main(['table', '--dataset', ds, '--out', str(out)] + init_args(ds)) == 0
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
    assert k['tv_duty']['n_m'] == '1' and float(k['tv_duty']['median']) == pytest.approx(4.0)
    md = (out / 'S05_final_table.md').read_text(encoding='utf-8')
    assert '| PID | O1-F05 | xe |' in md and '0.03 / 0.12 (n_m 3)' in md
    per_run = list(csv.DictReader(open(out / 'S05_final_runs.csv', encoding='utf-8')))
    assert {r['run_uid'] for r in per_run} == {'r1', 'r2', 'r3', 'r4', 'r5'}
    assert {r['run_uid']: r['gain_sha256'] for r in per_run}['r1'] == rl.sha256_file(os.path.join(ds,
                                                                                         'gain_A-PID_G07.json'))
    used = list(csv.DictReader(open(out / 'S05_final_gain_files.csv', encoding='utf-8')))      # R134C6-1
    assert [(u['gain_set'], u['source']) for u in used] == [('A-PID/G07', 'S05_freeze.csv'),
                                                            ('RQ2-INIT-PID', '--init-gain')]
    assert used[1]['gain_sha256'] == rl.sha256_file(os.path.join(ds, 'gain_init_PID.json'))
    pr = {r['run_uid']: r for r in per_run}                                  # S05 4.7 (i), Liam Q4 (CODE_RS4)
    assert float(pr['r5']['t_end_err_ms']) == pytest.approx(                          # no esp_pi_ppm (S05 v3.24)
        10.0 * ((1.0 + T.ESP_PI_EMAX_PPM * 1e-6) * (1.0 + T.PI_PERIOD_PPM_BOUND * 1e-6) - 1.0) * 1e3)
    assert pr['r1']['t_end_err_ms'] == '' and pr['r1']['t_end_err_src'] == 'xe: C7b'


def test_t_end_error_estimate():
    """S05 4.7 (i), Liam 14/11 CODE_RS4 Q4: t_end x (|esp_pi_ppm| + noise + |pi_period_ppm|); bounds when missing."""
    v, src = T.t_end_err_ms(12.0, esp_pi_ppm=-52.0, pi_period_ppm=-12.0)
    assert v == pytest.approx(12.0 * (52.0 + 50.0 + 12.0) * 1e-3) and 'pi_period_ppm' in src      # 1.37 ms
    v, src = T.t_end_err_ms(12.0, esp_pi_ppm=24.6)
    assert v == pytest.approx(12.0 * (24.6 + 50.0 + T.PI_PERIOD_PPM_BOUND) * 1e-3) and 'can Pi' in src
    # S05 v3.24 4.7 (i) (B81R2-1, B81R3-1, B81R3-2): no esp_pi_ppm -> T x [(1 + e_max)(1 + |pi| 1e-6) - 1] x 1e3,
    # e_max = (1 + rho)/(1 - rho) - 1, rho = 1e-3: 2002.002 ppm; S05: "about +-24.2 ms at T = 12 s, pi_period_ppm -12"
    assert T.ESP_PI_EMAX_PPM == pytest.approx(2002.002002, abs=1e-6)
    assert T.ESP_PI_EMAX_PPM == pytest.approx(1001.0 / 999.0 * 1e6 - 1e6, rel=1e-12)
    for missing in (None, float('nan'), float('inf'), 'x', 10 ** 400):            # 10**400: OverflowError (B8C-R1-6)
        v, src = T.t_end_err_ms(12.0, esp_pi_ppm=missing, pi_period_ppm=-12.0)
        assert v == pytest.approx(24.168312, abs=1e-6) and 'e_max 2002.002 ppm' in src and 'pi_period_ppm' in src, missing
        assert round(v, 1) == 24.2
    # pi_period_ppm missing too: the Pi bound (Liam 15/11 CODE_RS4 "Cận 100 ppm (Recommended)")
    for pi in (None, float('nan'), 10 ** 400):
        v, src = T.t_end_err_ms(12.0, esp_pi_ppm=None, pi_period_ppm=pi)
        assert v == pytest.approx(12.0 * ((1001.0 / 999.0) * (1.0 + 1e-4) - 1.0) * 1e3) and 'can Pi' in src, pi
    v, _ = T.t_end_err_ms(12.0, esp_pi_ppm=None, pi_period_ppm=+12.0)               # |pi_period_ppm|
    assert v == pytest.approx(24.168312, abs=1e-6)
    assert T.t_end_err_ms(0.0)[0] == 0.0
    # the mapper's assumed range is the rho of e_max (a change there must change this bound)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import hil_edge_map
    assert T.MAPPER_RHO_PPM == hil_edge_map.RHO_MAX_PPM
    # Codex B8C-R1-6: t_end < 0 has no number; a large finite t_end keeps a finite result
    assert T.t_end_err_ms(-12.0)[0] is None and T.t_end_err_ms(10 ** 400)[0] is None
    assert T.t_end_err_ms(1e305)[0] == pytest.approx(1e305 * ((1001.0 / 999.0) * (1.0 + 1e-4) - 1.0) * 1e3)
    assert T.t_end_err_ms(1e306, 1e300)[0] is None
    v, src = T.t_end_err_ms(12.0, esp_pi_ppm=24.6, pi_period_ppm=float('nan'))          # not finite: the Pi bound
    assert v == pytest.approx(12.0 * (24.6 + 50.0 + T.PI_PERIOD_PPM_BOUND) * 1e-3) and 'can Pi' in src
    assert T.t_end_err_ms(None) == (None, 'khong co t_end') and T.t_end_err_ms(float('nan'))[0] is None


def test_unknown_gain_set_is_reported_not_tabled(tmp_path, capsys):
    r = write_run(tmp_path, 'x', 'HW', 'final', 'PID', 'O1-F05', 'C-PID/G01', 'dat', 0.02)
    ds = _table_ds(tmp_path, [r])
    assert T.main(['table', '--dataset', ds, '--out', str(tmp_path / 'o')]) == 0
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


def _q(boot, mac='aa:bb', t=(1.0, 2.0), path=None):
    """read_query fields of one answer; t: (start, end) PC times of the question."""
    return dict(boot=boot, mac=mac, _t_start=t[0] if t else None, _t_end=t[1] if t else None, _path=path)


def _qs(boot='4753dc47'):
    """Two answers, before (1-2 s) and after (3-4 s) the run."""
    return _q(boot, t=(1.0, 2.0)), _q(boot, t=(3.0, 4.0))


def _first(L):
    return int(next(r['log_seq'] for r in L.rows if r['motor_on'] == '1'))


def test_boot_before_and_after_the_run(tmp_path):
    """S05 4.7 (iv), R3V-9, R134C-1, R134C2-1: two answers, asked one after the other, same boot and MAC,
    equal to the boot of the run."""
    L = _clean()
    ev = start_events(_first(L), boot='ab')
    B, A = (1.0, 2.0), (3.0, 4.0)
    for qs, ok, why in (((_q('ab', t=B), _q('ab', t=A)), True, ''),
                        ((None, _q('ab', t=A)), False, 'truoc'),
                        ((_q('ab', t=B), None), False, 'sau'),
                        ((_q('ab', t=B), _q('cd', t=A)), False, 'khac boot'),
                        ((_q('AB', t=B), _q('ab', t=A)), True, ''),
                        ((_q('cd', t=B), _q('cd', t=A)), False, 'khac boot cua luot'),   # old answer / other board
                        ((_q('ab', 'm1', t=B), _q('ab', 'm2', t=A)), False, 'hai board'),
                        ((_q('ab', t=A), _q('ab', t=B)), False, 'khong phai truoc va sau'),   # R134C2-1: order
                        ((_q('ab', t=B), _q('ab', t=B)), False, 'khong phai truoc va sau'),   # one question twice
                        ((_q('ab', path='p'), _q('ab', t=A, path='p')), False, 'cung mot file'),
                        ((_q('ab', t=None), _q('ab', t=A)), False, '#query_pc')):            # no PC times
        r = T.tv_duty_esp32(L.rows, 10.0, queries=qs, events=ev, ack=_ack('ab'))
        assert (r['tv_duty'] is not None) == ok and why in r['reason'], (qs, r['reason'])
    r = T.tv_duty_esp32(L.rows, 10.0, queries=_qs('ab'), events=ev[:6], check_gain_line=False)
    assert r['tv_duty'] is None and 'khong co boot' in r['reason']             # no #us,ev=start: no run boot
    d = str(tmp_path)
    p = os.path.join(d, 'esp32.csv')
    write_log(d, L.rows, boots=('4753DC47', '4753dc47'))
    qb, qa = os.path.join(d, 'gain_query_before.txt'), os.path.join(d, 'gain_query_after.txt')
    assert T.read_boot(qb) == '4753DC47' and T.read_query(qb)['_t_end'] == 101.0
    assert T.main(['tv', '--log', p, '--t-end', '10', '--query-before', qb, '--query-after', qa,
                   '--gain-ack', os.path.join(d, 'gain_ack.txt')]) == 0
    r = T.tv_duty_esp32(T.read_esp32_log(p), 10.0, events=T.read_events(p), queries=(T.read_query(qb),) * 2)
    assert r['tv_duty'] is None and 'cung mot file' in r['reason']             # R134C2-1: one file for both
    for bad in ('# no answer\n#ket_qua,ma=3\n', answer('4753dc47') + '#ket_qua,ma=3\n',     # R134C-1: last line
                answer('4753dc47') + answer('4753dc47')):                                   # two answers
        with open(qa, 'w', encoding='utf-8') as f:
            f.write(bad)
        assert T.read_boot(qa) is None, bad


def test_gain_line_of_the_run(tmp_path):
    """S05 6.3, R134C-2: a '#gain' line whose first_seq is the first motor-on data line, next to the
    run's '#us,ev=start'."""
    L = _clean()
    d = str(tmp_path)
    p = os.path.join(d, 'esp32.csv')
    write_log(d, L.rows, gain_seq=3)                                  # the line of another run
    r = T.tv_duty_esp32(T.read_esp32_log(p), 10.0, queries=_qs(), ack=_ack(), events=T.read_events(p))
    assert r['tv_duty'] is None and 'no #gain line' in r['reason']
    os.remove(p + '.events.txt')
    assert T.read_events(p) is None
    r = T.tv_duty_esp32(T.read_esp32_log(p), 10.0, queries=_qs(), ack=_ack(), events=None)
    assert 'thieu file events' in r['reason']
    write_log(d, L.rows)
    r = T.tv_duty_esp32(T.read_esp32_log(p), 10.0, queries=_qs(), ack=_ack(), events=T.read_events(p))
    assert r['tv_duty'] == pytest.approx(4.0) and r['run_first_seq'] == int(L.rows[3]['log_seq'])
    assert r['run_boot'] == '4753dc47'


def test_old_gain_line_of_an_earlier_boot_is_not_taken():
    """R134C-2, R134C2-2 (Codex counterexamples): the events still hold a '#gain' line of an earlier boot
    with the same first_seq; the current run's three '#gain' lines are lost: no evidence, with or without the
    old start and end lines."""
    L = _clean()
    s0 = _first(L)
    now = [x for x in start_events(s0, '4753dc47', t0='2026-11-14T10:00:30') if '#gain,' not in x]
    old_gain = [x for x in start_events(s0, '0badf00d') if '#gain,' in x]                 # 30 s earlier
    for events in (start_events(s0, '0badf00d') + stamp(['#us_end,boot=0badf00d,on=0'], '2026-11-14T10:00:05')
                   + now, old_gain + now):                                            # every boundary line lost
        r = T.tv_duty_esp32(L.rows, 10.0, queries=_qs(), ack=_ack(), events=events)
        assert r['tv_duty'] is None and ('no #gain line' in r['reason'] or 'cannot be told apart' in r['reason'])
    r = T.tv_duty_esp32(L.rows, 10.0, queries=_qs(), ack=_ack(), events=old_gain + now)
    assert 'no #gain line' in r['reason'] and 'within 0.5 s' in r['reason']
    # two boots on the run's own #us,ev=start lines: refused
    r = T.tv_duty_esp32(L.rows, 10.0, queries=_qs(), ack=_ack(),
                        events=start_events(s0)[:7] + stamp([US.format(boot='0badf00d', seq=s0)], '2026-11-14T10:00:00.007'))
    assert r['tv_duty'] is None and 'cannot be told apart' in r['reason']


def test_order_of_arrival_does_not_matter():
    """R134C2-4 (Codex counterexample): UDP reorders: the run's '#gain', then an old '#run_end', then the run's
    '#us,ev=start'; or a late copy of the start after the end lines. Both are the run, by their times."""
    L = _clean()
    s0 = _first(L)
    ev = start_events(s0)
    reordered = ev[:6] + stamp(['#run_end,seq=90,last_data_seq=89,log_qdrop=0'], '2026-11-14T10:00:00.006') + ev[6:]
    r = T.tv_duty_esp32(L.rows, 10.0, queries=_qs(), ack=_ack(), events=reordered)
    assert r['tv_duty'] == pytest.approx(4.0), r['reason']
    late = ev[:8] + stamp(['#us_end,boot=4753dc47,on=0', '#run_end,seq=1200,last_data_seq=1199',
                           US.format(boot='4753dc47', seq=s0)], '2026-11-14T10:00:10')     # read late: late time
    r = T.tv_duty_esp32(L.rows, 10.0, queries=_qs(), ack=_ack(), events=late)
    assert r['tv_duty'] == pytest.approx(4.0), r['reason']                # an early copy is still there


def test_only_a_late_start_copy_left():
    """R134C3-2 (Codex): only a copy of '#us,ev=start' read late (> 0.5 s) is left: without boot on '#gain'
    (firmware 16702ff) the run is refused (limit, DEFERRED); with boot on '#gain' it is the run."""
    L = _clean()
    s0 = _first(L)
    gains = start_events(s0)[3:6]
    late_us = stamp([US.format(boot='4753dc47', seq=s0)], '2026-11-14T10:00:00.700')
    r = T.tv_duty_esp32(L.rows, 10.0, queries=_qs(), ack=_ack(), events=gains + late_us)
    assert r['tv_duty'] is None and 'within 0.5 s' in r['reason']
    with_boot = [x + ',boot=4753dc47' for x in gains]
    r = T.tv_duty_esp32(L.rows, 10.0, queries=_qs(), ack=_ack(), events=with_boot + late_us)
    assert r['tv_duty'] == pytest.approx(4.0), r['reason']


def test_line_drop_differs_between_copies_of_one_gain_line():
    """R134C-5: line_drop is read live when formatted (logger.c 181-182); not a conflict, largest reported."""
    L = _clean()
    ev = start_events(_first(L), drops=(0, 1, 1))
    r = T.tv_duty_esp32(L.rows, 10.0, queries=_qs(), ack=_ack(), events=ev)
    assert r['tv_duty'] == pytest.approx(4.0) and r['gain_line_drop_max'] == 1
    ev = start_events(_first(L))
    ev[4] = ev[4].replace('id=7', 'id=8')
    r = T.tv_duty_esp32(L.rows, 10.0, queries=_qs(), ack=_ack(), events=ev)
    assert r['tv_duty'] is None and 'conflict' in r['reason']


def test_duplicate_lines_follow_the_logger_marks(tmp_path):
    """R134C-3 (S05 6.3, L92): a copy of a line without the sidecar refuses the log, even outside the
    window; with the sidecar marking it duplicate it is dropped and counted."""
    L = _clean()
    d = str(tmp_path)
    p = os.path.join(d, 'esp32.csv')
    rows = L.rows + [dict(L.rows[0])]                                 # copy of the first off line, at the end
    write_log(d, rows)
    with pytest.raises(ValueError, match='two data lines with log_seq'):
        T.read_esp32_log(p)
    assert T.main(['tv', '--log', p, '--t-end', '10', '--no-boot']) == 2
    _segments(p, rows, status_of=lambda i: 'duplicate' if i == len(rows) else 'ok')
    got, n = T.read_esp32_log_info(p)
    assert n == 1 and len(got) == len(L.rows)
    assert T.main(['tv', '--log', p, '--t-end', '10', '--query-before', os.path.join(d, 'gain_query_before.txt'),
                   '--query-after', os.path.join(d, 'gain_query_after.txt')]) == 0


def test_same_log_seq_in_two_segments_is_refused(tmp_path):
    """R134C2-3 (Codex counterexample, S05 6.3 "cung log_seq ma khac noi dung"): a line with an existing
    log_seq and other content, placed by the logger in a new segment and marked ok: not usable."""
    L = _clean()
    d = str(tmp_path)
    p = os.path.join(d, 'esp32.csv')
    rows = L.rows + [dict(L.rows[2], timestamp_us='1000000', tick='0')]
    write_log(d, rows)
    _segments(p, rows, seg_of=lambda i: 1 if i == len(rows) else 0, status_of=lambda i: 'ok')
    with pytest.raises(ValueError, match='on two kept lines'):
        T.read_esp32_log(p)


def test_tail_must_follow_b_in_the_same_segment(tmp_path):
    """R134C-4 (Codex counterexample): the line with log_seq(b) + 1 is in another logger segment (a new
    boot), or b + 1 has the right log_seq but a reset tick: no number."""
    L = _clean()
    t_end = 10.0
    r = tv(L.rows, t_end)
    b = next(i for i, x in enumerate(L.rows) if int(x['log_seq']) == r['tail_seq']) - 1
    d = str(tmp_path)
    p = os.path.join(d, 'esp32.csv')
    rows = L.rows[:b + 2]
    write_log(d, rows)
    _segments(p, rows, seg_of=lambda i: 1 if i == b + 2 else 0)
    r = tv(T.read_esp32_log(p), t_end)
    assert r['tv_duty'] is None and 'doan logger khac' in r['reason']
    rows = [dict(x) for x in L.rows[:b + 2]]
    rows[b + 1]['tick'] = '1'                                         # counters reset (new boot) at b + 1
    r = tv(rows, t_end)
    assert r['tv_duty'] is None and 'tick 1' in r['reason']


# ---------------------------------------------------------------- B8-1 (S05 v3.20 C9 (d), (e); CODE_R134)

def test_b81_14_gain_ack_boot_matches_the_two_questions(tmp_path):
    """S05 4.7 (iv): the boot of the '#gain_ack' at the gain setting equals the boot of the two questions (and the
    board, and the set the run ran)."""
    L = _clean()
    ev = start_events(_first(L))
    for ack, ok, why in ((_ack(), True, ''), (None, False, 'thieu cau tra loi #gain_ack'),
                         (_ack('0badf00d'), False, 'khoi dong lai sau khi dat gain'),
                         (_ack(mac='cc:dd'), False, 'board khac'), (_ack(gid='8'), False, 'bo gain khac'),
                         (_ack('4753DC47'), True, '')):
        r = T.tv_duty_esp32(L.rows, 10.0, queries=_qs(), events=ev, ack=ack)
        assert (r['tv_duty'] is not None) == ok and why in r['reason'], (ack, r['reason'])
    built_in = [x.replace('id=7', 'id=0') for x in ev]               # R134C5-3: every final run has its ack
    r = T.tv_duty_esp32(L.rows, 10.0, queries=_qs(), events=built_in, ack=None)
    assert r['tv_duty'] is None and 'thieu cau tra loi #gain_ack' in r['reason']


def test_b81_14_read_ack_and_the_table(tmp_path):
    d = str(tmp_path)
    p = os.path.join(d, 'a.txt')
    for text, ok in ((ack_text(), True), (ack_text(ok=0), False), (ack_text(ma=4), False),
                     (ack_text() + ack_text(), False), (ack_text(boot='zz'), False), ('#ket_qua,ma=0\n', False)):
        with open(p, 'w', encoding='utf-8') as f:
            f.write(text)
        assert (T.glc.read_ack(p) is not None) is ok, text
    assert T.glc.read_ack(os.path.join(d, 'none.txt')) is None
    base = os.path.join(d, 'runs')
    os.makedirs(base)
    run = write_run(base, 'F1', 'HIL', 'final_hil', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.03)
    ds = _table_ds(d, [run])
    runs, _ = LR(ds)
    assert runs[0]['values']['tv_duty'] == pytest.approx(4.0) and runs[0]['tv_check']['boot_ack'] == '4753dc47'
    with open(os.path.join(run['results_dir'], 'gain_ack.txt'), 'w', encoding='utf-8') as f:
        f.write(ack_text('0badf00d'))
    with pytest.raises(ValueError, match='khoi dong lai sau khi dat gain'):     # R134C5-3: run level, not tv
        LR(ds)
    os.remove(os.path.join(run['results_dir'], 'gain_ack.txt'))
    with pytest.raises(ValueError, match='thieu gain_ack.txt'):
        LR(ds)
    with open(os.path.join(run['results_dir'], 'gain_ack.txt'), 'w', encoding='utf-8') as f:
        f.write(ack_text(ok=0))                                         # there but not an applied set
    with pytest.raises(ValueError, match='gain_ack.txt hong'):
        LR(ds)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf'), True, 'x'])
def test_b81_16_not_finite_is_no_number(bad):
    """S05 v3.20 C9 (e): a quantity that is not a finite number is missing (with its reason), not counted in n_m."""
    v, miss = T.run_values(dict(e_perp_max=bad, e_s_absmax=0.01, e_trk_max=0.02, e_theta_lane_max_deg=5.0,
                                e_theta_cmd_max_deg=4.0), 'O1-F05', dict(tv_duty=1.0))
    assert v['e_perp_max'] is None and 'khong huu han' in miss['e_perp_max']
    runs = [dict(run_uid='a', controller='PID', condition='O1-F05', platform='HIL', gain_set='A', outcome='dat',
                 values=v, missing=miss),
            dict(run_uid='b', controller='PID', condition='O1-F05', platform='HIL', gain_set='A', outcome='dat',
                 values=dict(v, e_perp_max=0.05), missing={}),
            dict(run_uid='c', controller='PID', condition='O1-F05', platform='HIL', gain_set='A', outcome='dat',
                 values=dict(v, e_perp_max=bad), missing={})]               # a value that skipped run_values
    cell = [t for t in T.build_table(runs) if t['condition'] == 'O1-F05' and t['platform'] == 'HIL'
            and t['gain_set'] == 'A' and t['quantity'] == 'e_perp_max'][0]
    assert cell['n_m'] == 1 and cell['max'] == 0.05 and 'a: gia tri khong huu han' in cell['missing']


# ---------------------------------------------------------------- Codex Sol round 4 (CODE_R134)

def _table_ds(tmp_path, runs, frozen=(('A-PID', 'A-PID/G07'),)):
    ds = os.path.join(tmp_path, 'ds')
    os.makedirs(ds, exist_ok=True)
    with open(os.path.join(ds, 'm7_tuning_log.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    write_gains(ds, frozen)                                            # R134C6-1
    # C9D-2: the clip H comparison (below eps_H, outcome kept) of every judged final car run, by the command so-h;
    # a row the log checks refuse is left without it (the table stops on that row check first). The test's own
    # dataset is built again on each call, so the comparison file of an earlier call goes first.
    if os.path.exists(os.path.join(ds, rl.SO_H)):
        os.remove(os.path.join(ds, rl.SO_H))
    for r in rl.so_h_missing(rl.live_rows(runs), ds, stages=('final',)):
        try:
            rl.so_h(ds, r['run_uid'], rl.fake_score_final(r['results_dir'], 0.5))
        except rl.LogError:
            pass
    return ds


@pytest.mark.parametrize('cut, match', [
    (dict(gate_version=''), 'without gate columns'),          # a judged car row without its camera gate
    (dict(boot_before='', boot_after=''), 'missing'),         # without the two boots
])
def test_c4_2_table_refuses_rows_the_log_refuses(tmp_path, capsys, cut, match):
    """R134C4-2: the table runs the row checks of rq3_log summary; a row without its evidence stops it."""
    r = write_run(tmp_path, 'r1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    if 'gate_version' in cut:
        r.update({c: '' for c in rl.GATE_COLS})
    else:
        r.update(cut)
    ds = _table_ds(tmp_path, [r])
    assert T.main(['table', '--dataset', ds, '--out', str(tmp_path / 'out')]) == 2
    assert match in capsys.readouterr().out


def test_c4_2_final_hil_without_boots_and_rows_outside_the_campaigns(tmp_path, capsys):
    h = write_run(tmp_path, 'h1', 'HIL', 'final_hil', 'PID', 'O1-F05', 'RQ2-INIT-PID/G00', 'dat', 0.02)
    ds = _table_ds(tmp_path, [dict(h, boot_after='')])
    assert T.main(['table', '--dataset', ds, '--out', str(tmp_path / 'o1')]) == 2
    capsys.readouterr()
    c = write_run(tmp_path, 'c1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    old = dict(c, campaign='', run_uid='OLD:1')
    runs, problems = LR(_table_ds(tmp_path, [c, old]))
    assert [r['run_uid'] for r in runs] == ['c1'] and any('OLD:1' in p and 'ngoai cac chien dich' in p
                                                          for p in problems)


def test_c9d2_3_table_refuses_a_run_after_a_known_n3_stop(tmp_path, capsys):
    """C9D2-3 (Codex C9 demo round 2): three lost final car runs in a row (logged so, no clip H needed) stop the
    condition (S05 4.6); a later run is an error for the table as for the summary."""
    runs = [write_run(tmp_path, f'n{k}', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'hong_he_do', 0.9)
            for k in range(3)]
    runs.append(write_run(tmp_path, 'n3', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02))
    ds = _table_ds(tmp_path, runs)
    assert T.main(['table', '--dataset', ds, '--out', str(tmp_path / 'out')]) == 2
    assert 'lost runs in a row' in capsys.readouterr().out


def test_c4_6_cat_t1_counted_per_cell(tmp_path):
    """R134C4-6, O1 3.5 R2V-4: runs khong_dat by the exception before t1 are counted in their cell."""
    r = write_run(tmp_path, 'v1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'khong_dat', 0.12)
    g = rl.floor_gate(rl.fake_gate(e_trk=0.5), t_done=(None, None))
    g['g1'].update(t1=6.0, cat_t1=True, state='chua_ket_luan', q='inconclusive')
    g.update(rl.o1_gate.evaluate_outcomes(g, g['decide']))
    sp = os.path.join(r['results_dir'], 'score.json')                 # C9D-1: the scorer file carries this gate
    with open(sp, encoding='utf-8') as f:
        sc = json.load(f)
    with open(sp, 'w', encoding='utf-8') as f:
        json.dump(dict(sc, gate_v3=g), f)
    r.update(rl.gate_columns(g, rl.sha256_file(sp)), fail_cause=rl.VIOL_BEFORE)
    p = write_run(tmp_path, 'v2', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'khong_dat', 0.12)
    runs, _ = LR(_table_ds(tmp_path, [r, p]))
    cell = [t for t in T.build_table(runs) if (t['condition'], t['platform'], t['gain_set'], t['quantity'])
            == ('O1-F05', 'xe', 'A', 'e_perp_max')][0]
    assert cell['n_valid'] == 2 and cell['n_cat_t1'] == 1


def test_c4_5_an_ack_is_checked_even_with_id_0(tmp_path):
    """R134C4-5: a set applied, then an ESP32 restart: the run's '#gain' has id 0 and the queries the new boot;
    the ack of the older boot is still checked. Without an ack, id 0 (built-in set) needs none."""
    L = _clean()
    ev = [x.replace('id=7', 'id=0').replace('applied_n=1', 'applied_n=0') for x in start_events(_first(L))]
    r = T.tv_duty_esp32(L.rows, 10.0, queries=_qs(), events=ev, ack=_ack('0badf00d'))
    assert r['tv_duty'] is None and 'khoi dong lai sau khi dat gain' in r['reason']
    r = T.tv_duty_esp32(L.rows, 10.0, queries=_qs(), events=ev, ack=_ack())       # same boot, but id 7 != 0
    assert r['tv_duty'] is None and 'bo gain khac' in r['reason']
    r = T.tv_duty_esp32(L.rows, 10.0, queries=_qs(), events=ev, ack=None)   # R134C5-3: no exemption left
    assert r['tv_duty'] is None and 'thieu cau tra loi #gain_ack' in r['reason']


# ---------------------------------------------------------------- Codex Sol round 5 (CODE_R134)

@pytest.mark.parametrize('case, match', [('same_answer', 'khong phai truoc va sau'), ('other_boot', 'khac boot sau'),
                                         ('row_boots', 'boot cua dong log')])
def test_c5_1_run_evidence_is_checked_apart_from_tv_duty(tmp_path, case, match):
    """R134C5-1: two answers that are not two questions (or not the row's boots) refuse the table, even when the
    run has no number for tv_duty (no ESP32 log here)."""
    r = write_run(tmp_path, 'e1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    d = r['results_dir']
    os.remove(os.path.join(d, 'esp32.csv'))                          # tv_duty: no number either way
    qb, qa = os.path.join(d, 'gain_query_before.txt'), os.path.join(d, 'gain_query_after.txt')
    if case == 'same_answer':                                         # 'after' is a copy of 'before': same PC times
        with open(qb, encoding='utf-8') as f:
            txt = f.read()
        with open(qa, 'w', encoding='utf-8') as f:
            f.write(txt)
    elif case == 'other_boot':
        with open(qa, 'w', encoding='utf-8') as f:
            f.write(answer('0badf00d', t=(200.0, 201.0)))
    else:
        r.update(boot_before='0badf00d', boot_after='0badf00d')
    with pytest.raises(ValueError, match=match):
        LR(_table_ds(tmp_path, [r]))


def test_c5_2_a_historic_hil_row_does_not_stop_the_table(tmp_path):
    """R134C5-2: a HIL row outside the campaigns, judged without a gate, is set apart before the row checks."""
    good = write_run(tmp_path, 'g1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    old = dict.fromkeys(rl.TUNING_COLS, '')
    old.update(session_id='OLD', run_uid='OLD:1', campaign='', platform='HIL', stage='hil_check', run_outcome='dat',
               controller='PID', trajectory='O1-F05', speed_mps='0.5')
    runs, problems = LR(_table_ds(tmp_path, [good, old]))
    assert [x['run_uid'] for x in runs] == ['g1'] and problems == []


def test_c5_3_every_final_run_has_its_ack(tmp_path):
    """R134C5-3 (Liam 14/11 "Bat buoc ack moi luot (Recommended)"): the initial gain set too; the ack must be of
    the running id of the question before."""
    r = write_run(tmp_path, 'i1', 'HIL', 'final_hil', 'PID', 'O1-F05', 'RQ2-INIT-PID/G00', 'dat', 0.02)
    os.remove(os.path.join(r['results_dir'], 'gain_ack.txt'))
    with pytest.raises(ValueError, match='thieu gain_ack.txt'):
        LR(_table_ds(tmp_path, [r]))
    with open(os.path.join(r['results_dir'], 'gain_ack.txt'), 'w', encoding='utf-8') as f:
        f.write(ack_text(gid=8))                                      # the question before says id 7
    with pytest.raises(ValueError, match='bo gain khac'):
        LR(_table_ds(tmp_path, [r]))


# ---------------------------------------------------------------- Codex Sol round 6 (CODE_R134)

def _set_id(r, gid, ack=True, query=True, line=True):
    """Make the run of row r show set `gid` in its ack, its two answers and / or its '#gain' line."""
    d = r['results_dir']
    if ack:
        with open(os.path.join(d, 'gain_ack.txt'), 'w', encoding='utf-8') as f:
            f.write(ack_text(gid=gid))
    if query:
        for w, t in (('before', (100.0, 101.0)), ('after', (200.0, 201.0))):
            with open(os.path.join(d, f'gain_query_{w}.txt'), 'w', encoding='utf-8') as f:
                f.write(answer('4753dc47', t=t, gid=gid))
    if line:
        p = os.path.join(d, 'esp32.csv.events.txt')
        with open(p, encoding='utf-8') as f:
            txt = f.read()
        with open(p, 'w', encoding='utf-8') as f:
            f.write(txt.replace('#gain,id=7,', f'#gain,id={gid},'))


def test_c6_1_a_run_of_another_file_than_its_set_is_refused(tmp_path):
    """R134C6-1: the gain file id 8 was given for a run of A-PID/G07 (ack, two answers, '#gain' all id 8): the
    table stops at that run, whatever tv_duty."""
    r = write_run(tmp_path, 'w1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    ds = _table_ds(tmp_path, [r])
    assert len(LR(ds)[0]) == 1                                        # the fixture run itself passes
    _set_id(r, 8)
    with pytest.raises(ValueError, match=r'run w1: id dang chay 8 .* khac id 7 cua file gain'):
        LR(ds)


def test_c6_1_set_changed_after_the_ack_is_refused(tmp_path):
    """R134C6-1 variant: ack and the question before say id 7, the run's '#gain' line says id 8 (set changed
    before the run): no longer only tv_duty empty; the run stops the table."""
    r = write_run(tmp_path, 'w2', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    ds = _table_ds(tmp_path, [r])
    _set_id(r, 8, ack=False, query=False)
    with pytest.raises(ValueError, match=r'run w2: dong #gain cua moc bat dau first_seq=\d+ \(1 moc trong events\) '
                                         r'khac file gain: id=8, gain file id 7'):
        LR(ds)


def test_c6_1_gain_values_must_match_bit_for_bit(tmp_path):
    """R134C6-1: one gain of the file differs from the run's '#gain' line (same id): refused."""
    r = write_run(tmp_path, 'w3', 'HIL', 'final_hil', 'PID', 'O1-F05', 'RQ2-INIT-PID/G00', 'dat', 0.02)
    ds = _table_ds(tmp_path, [r])
    other = write_gain_file(os.path.join(ds, 'gain_init_other.json'), gains=dict(kp_xy=1.25))
    with pytest.raises(ValueError, match=r'run w3: dong #gain cua moc bat dau first_seq=\d+ \(1 moc trong events\) '
                                         r'khac file gain: kp_xy: log 1.5'):
        LR(ds, init_gain={'PID': other})


def test_c6_1_no_file_for_the_set(tmp_path, capsys):
    """R134C6-1: a set with no row in S05_freeze.csv, or an initial gain without --init-gain, stops the table."""
    a = write_run(tmp_path, 'n1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    i = write_run(tmp_path, 'n2', 'HIL', 'final_hil', 'PID', 'O1-F05', 'RQ2-INIT-PID/G00', 'dat', 0.02)
    ds = _table_ds(tmp_path, [a, i], frozen=())
    with pytest.raises(ValueError, match=r'run n1: khong co file gain .*S05_freeze.csv; run n2: .*--init-gain PID'):
        LR(ds, init_gain={})
    assert T.main(['table', '--dataset', ds, '--out', str(tmp_path / 'o')]) == 2
    assert 'cho --init-gain PID' in capsys.readouterr().out


def test_c6_1_frozen_file_must_keep_its_sha(tmp_path):
    r = write_run(tmp_path, 's1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    ds = _table_ds(tmp_path, [r])
    write_gain_file(os.path.join(ds, 'gain_A-PID_G07.json'), gains=dict(kp_xy=1.25))     # changed after freeze
    with pytest.raises(ValueError, match='khac gain_sha256'):
        LR(ds)


def test_c6_1_init_gain_arguments_and_ladrc_ctrl(tmp_path, capsys):
    """--init-gain CTRL=FILE once per controller; a LADRC set is a gain file with ctrl ADRC (gain_send format)."""
    r = write_run(tmp_path, 'l1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    ds = _table_ds(tmp_path, [r])
    ok = os.path.join(ds, 'gain_init_PID.json')
    for bad, why in ((['--init-gain', 'PID'], 'write CTRL=FILE'), (init_args(ds) * 2, 'once per controller'),
                     (['--init-gain', 'XYZ=' + ok], 'controller one of')):
        assert T.main(['table', '--dataset', ds, '--out', str(tmp_path / 'o')] + bad) == 2, bad
        assert why in capsys.readouterr().out, bad
    assert T.main(['table', '--dataset', ds, '--out', str(tmp_path / 'o')] + init_args(ds)) == 0
    capsys.readouterr()
    adrc = os.path.join(ds, 'gain_init_LADRC.json')
    with open(adrc, 'w', encoding='utf-8') as f:
        json.dump(dict(ctrl='ADRC', id=3, gains=dict(omega_c_xy=8, omega_o_xy=30, omega_c_wz=6, omega_o_wz=25,
                                                    pos_kp=6, pos_ki=3, k_psi=1)), f)
    f = T.gain_files(ds, {'LADRC': adrc})
    assert f['RQ2-INIT-LADRC']['want']['ctrl'] == 'ADRC'
    with pytest.raises(ValueError, match='the set is of LADRC'):
        T.gain_files(ds, {'LADRC': os.path.join(ds, 'gain_init_PID.json')})


def test_c6_1_no_esp32_log_no_gain_proof(tmp_path):
    """R134C6-1: without the ESP32 log of a judged run its '#gain' line cannot be compared: the table stops."""
    r = write_run(tmp_path, 'm1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    os.remove(os.path.join(r['results_dir'], 'esp32.csv'))
    with pytest.raises(ValueError, match='run m1: thieu esp32.csv'):
        LR(_table_ds(tmp_path, [r]))


@pytest.mark.parametrize('a_id, q_id', [(None, None), ('', ''), (None, '7'), ('7', None), ('7', ''), ('x', 'x')])
def test_c6_2_both_ids_must_be_there(a_id, q_id):
    """R134C6-2: ids are compared only when both are decimal; two missing or two empty ids are no match."""
    ack, before = dict(ok='1', boot='4753dc47', mac='aa:bb'), dict(boot='4753dc47', mac='aa:bb')
    if a_id is not None:
        ack['id'] = a_id
    why = T.glc.ack_problem(ack, before, {'id': q_id})
    assert why and 'khong so duoc bo gain' in why
    assert T.glc.ack_problem(dict(ack, id='7'), before, {'id': '7'}) is None
    assert T.glc.ack_problem(dict(ack, id='7'), before, None) is None           # no running id shown: not compared


def test_c6_2_table_without_ids(tmp_path):
    """R134C6-2 at the table: ack and the two answers without id: refused."""
    r = write_run(tmp_path, 'k1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    ds = _table_ds(tmp_path, [r])
    d = r['results_dir']
    for name in ('gain_ack.txt', 'gain_query_before.txt', 'gain_query_after.txt'):
        p = os.path.join(d, name)
        with open(p, encoding='utf-8') as f:
            txt = f.read()
        with open(p, 'w', encoding='utf-8') as f:
            f.write(re.sub(r',id=7(?=[,\n])', '', txt))
    with pytest.raises(ValueError, match='run k1: .*khong so duoc bo gain'):
        LR(ds)


# ---------------------------------------------------------------- Codex Sol round 7 (CODE_R134)

def _two_stretches(r, second_gid=7, second_gain=True):
    """Rewrite the ESP32 log of row r: two motor-on stretches (the motor stops, then starts again), each with its
    start lines (#fw, #gain, #us,ev=start) as the firmware sends them; the second '#gain' line with id second_gid,
    or none."""
    L = Log()
    L.off(30)
    L.on(500, zigzag)
    L.off(20, after_on=True)
    L.on(504, zigzag)
    on = [x['motor_on'] == '1' for x in L.rows]
    rises = [int(L.rows[i]['log_seq']) for i in range(1, len(L.rows)) if on[i] and not on[i - 1]]
    assert len(rises) == 2
    second = start_events(rises[1], t0='2026-11-14T10:00:05')
    second = ([x.replace('#gain,id=7,', f'#gain,id={second_gid},') for x in second] if second_gain
              else [x for x in second if ' #gain,' not in x])
    write_log(r['results_dir'], L.rows, events=start_events(rises[0]) + second)


def test_c7_1_every_motor_on_stretch_shows_the_set(tmp_path):
    """R134C7-1 (Liam 15/11 "Kiem gain moi doan (Recommended)"): a log with two motor-on stretches counts when both
    show the set of the file (tv_duty has no number then); a second stretch with id 8, or without its '#gain'
    line, refuses the run."""
    r = write_run(tmp_path, 't1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    ds = _table_ds(tmp_path, [r])
    _two_stretches(r)
    runs, _ = LR(ds)
    assert len(runs) == 1 and runs[0]['values']['tv_duty'] is None
    _two_stretches(r, second_gid=8)
    with pytest.raises(ValueError, match=r'run t1: dong #gain cua moc bat dau first_seq=\d+ \(2 moc .* khac file '
                                         r'gain: id=8'):
        LR(ds)
    _two_stretches(r, second_gain=False)
    with pytest.raises(ValueError, match=r'run t1: dong #gain cua moc bat dau first_seq=\d+ \(2 moc trong events\): '
                                         r'.*no #gain line'):
        LR(ds)


def test_c7_1_question_after_shows_the_file_id(tmp_path):
    """R134C7-1: the question after the run says id 8 (the set changed during the run): refused."""
    r = write_run(tmp_path, 'q1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    ds = _table_ds(tmp_path, [r])
    with open(os.path.join(r['results_dir'], 'gain_query_after.txt'), 'w', encoding='utf-8') as f:
        f.write(answer('4753dc47', t=(200.0, 201.0), gid=8))
    with pytest.raises(ValueError, match=r"run q1: id dang chay '8' cua lan hoi sau khac id 7"):
        LR(ds)


def test_c7_2_a_frozen_set_without_judged_runs_does_not_stop_the_table(tmp_path):
    """R134C7-2: the file of B-LADRC was moved after its freeze; the table of runs of other sets is made; a judged
    run of B-LADRC is refused for it, a hong_he_do run of B-LADRC is not."""
    a = write_run(tmp_path, 'a1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    h = write_run(tmp_path, 'b0', 'HW', 'final', 'LADRC', 'O1-F05', 'B-LADRC/G02', 'hong_he_do', 0.9)
    ds = _table_ds(tmp_path, [a, h], frozen=(('A-PID', 'A-PID/G07'), ('B-LADRC', 'B-LADRC/G02')))
    os.remove(os.path.join(ds, 'gain_B-LADRC_G02.json'))
    runs, _ = LR(ds)
    assert sorted(x['run_uid'] for x in runs) == ['a1', 'b0']
    b = write_run(tmp_path, 'b1', 'HW', 'final', 'LADRC', 'O1-F05', 'B-LADRC/G02', 'dat', 0.02)
    ds = _table_ds(tmp_path, [a, h, b], frozen=(('A-PID', 'A-PID/G07'), ('B-LADRC', 'B-LADRC/G02')))
    os.remove(os.path.join(ds, 'gain_B-LADRC_G02.json'))
    with pytest.raises(ValueError, match=r"run b1: file gain cua 'B-LADRC/G02': .*not found") as e:
        LR(ds)
    assert 'run a1' not in str(e.value) and 'run b0' not in str(e.value)


def test_c7_3_relative_frozen_name_takes_the_candidate_with_the_sha(tmp_path, monkeypatch):
    """R134C7-3: gain_file 'gain_rel.json' written at freeze from the working folder; a copy with the same name but
    other content in the dataset folder does not hide it. No candidate with the sha256: refused."""
    r = write_run(tmp_path, 'c1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    ds = _table_ds(tmp_path, [r], frozen=())
    work = tmp_path / 'work'
    os.makedirs(work)
    good = write_gain_file(work / 'gain_rel.json')
    write_gain_file(os.path.join(ds, 'gain_rel.json'), gains=dict(kp_xy=1.25))
    with open(os.path.join(ds, rl.FREEZE), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=rl.FREEZE_COLS)
        w.writeheader()
        w.writerow(dict.fromkeys(rl.FREEZE_COLS, '') | dict(campaign='A-PID', gain_set_id='A-PID/G07',
                                                             gain_file='gain_rel.json', gain_sha256=rl.sha256_file(good)))
    monkeypatch.chdir(work)
    runs, _ = LR(ds)
    assert runs[0]['gain_file'] == 'gain_rel.json' and runs[0]['gain_sha256'] == rl.sha256_file(good)
    write_gain_file(good, gains=dict(kp_xy=1.75))                    # no candidate keeps the frozen sha256 now
    with pytest.raises(ValueError, match=r'run c1: file gain cua .*khac gain_sha256'):
        LR(ds)


# ---------------------------------------------------------------- Codex Sol round 8 (CODE_R134)

def _short_stop(r, second_gid=7, second_start=True, lose_first=False, extra=None):
    """Rewrite the ESP32 log of row r with a stop of 50 ms (five ticks, no motor-off data line: the firmware writes
    one every 10 ticks; three #run_end lines take log_seq), then a start again; the second start lines with id
    second_gid, or none; lose_first: the first data line after the stop is lost; extra: more events lines."""
    L = Log()
    L.off(30)
    L.on(400, zigzag)
    L.off(5, after_on=True)
    L.on(604, zigzag, keep=(lambda k: k != 0) if lose_first else (lambda k: True))
    rows = L.rows
    seqs = [int(x['log_seq']) for x in rows]
    hidden = [i for i in range(1, len(rows)) if rows[i]['motor_on'] == rows[i - 1]['motor_on'] == '1'
              and seqs[i] - seqs[i - 1] != int(rows[i]['tick']) - int(rows[i - 1]['tick'])]
    assert len(hidden) == 1 and all(x['motor_on'] == '1' for x in rows[rows.index(next(
        x for x in rows if x['motor_on'] == '1')):])                     # no 0 -> 1 edge after the first one
    second_seq = seqs[hidden[0]] - (1 if lose_first else 0)
    first = next(int(x['log_seq']) for x in rows if x['motor_on'] == '1')
    ev = start_events(first)
    if second_start:
        ev += [x.replace('#gain,id=7,', f'#gain,id={second_gid},')
               for x in start_events(second_seq, t0='2026-11-14T10:00:05')]
    write_log(r['results_dir'], rows, events=ev + (extra or []))
    return second_seq


def test_c8_1_a_short_stop_without_an_off_line_is_a_boundary(tmp_path):
    """R134C8-1: a stop shorter than 10 ticks leaves no 0 -> 1 edge; the start of the second stretch is found by
    d log_seq != d tick and must show the set of the file: same set counts (tv_duty empty), id 8 or no start
    lines refuse the run."""
    r = write_run(tmp_path, 'u1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    ds = _table_ds(tmp_path, [r])
    _short_stop(r)
    runs, _ = LR(ds)
    assert len(runs) == 1 and runs[0]['values']['tv_duty'] is None
    _short_stop(r, second_gid=8)
    with pytest.raises(ValueError, match=r'run u1: dong #gain cua moc bat dau .* khac file gain: id=8'):
        LR(ds)
    _short_stop(r, second_start=False)
    with pytest.raises(ValueError, match=r'run u1: motor tat ngan giua hai dong motor bat .* khong co moc bat dau'):
        LR(ds)


def test_c8_1_lost_first_line_of_the_second_stretch(tmp_path):
    """R134C8-1: the first data line after the short stop is lost; its start (first_seq = the lost line) lies in
    the gap and is checked: same set counts, id 8 refuses."""
    r = write_run(tmp_path, 'u2', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    ds = _table_ds(tmp_path, [r])
    _short_stop(r, lose_first=True)
    assert len(LR(ds)[0]) == 1
    _short_stop(r, second_gid=8, lose_first=True)
    with pytest.raises(ValueError, match=r'run u2: .* khac file gain: id=8'):
        LR(ds)


def test_c8_1_a_visible_edge_with_its_first_line_lost(tmp_path):
    """R134C8-1: a 0 -> 1 edge whose first motor-on line is lost: the start shown in the gap proves the stretch."""
    r = write_run(tmp_path, 'u3', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    ds = _table_ds(tmp_path, [r])
    L = Log()
    L.off(30)
    L.on(1004, zigzag, keep=lambda k: k != 0)
    first_lost = int(L.rows[next(i for i, x in enumerate(L.rows) if x['motor_on'] == '1')]['log_seq']) - 1
    write_log(r['results_dir'], L.rows, events=start_events(first_lost))
    assert len(LR(ds)[0]) == 1


def test_c8_1_every_start_in_the_events_shows_the_set(tmp_path):
    """R134C8-1 (1): a start in the events with another set refuses the run even where the data show no
    boundary (one file = one run)."""
    r = write_run(tmp_path, 'u4', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    ds = _table_ds(tmp_path, [r])
    _short_stop(r, extra=[x.replace('#gain,id=7,', '#gain,id=8,') for x in start_events(900, t0='2026-11-14T10:00:20')])
    with pytest.raises(ValueError, match=r'run u4: dong #gain cua moc bat dau first_seq=900 \(3 moc'):
        LR(ds)


def test_c8_2_a_read_error_stays_with_its_candidate(tmp_path, monkeypatch):
    """R134C8-2: a frozen file that cannot be read (PermissionError) stops neither the table of other sets nor a
    run whose set has another readable candidate with the sha256; the candidate after the matching one is not
    read."""
    import rq3_log
    a = write_run(tmp_path, 'p1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    ds = _table_ds(tmp_path, [a], frozen=(('A-PID', 'A-PID/G07'), ('B-LADRC', 'B-LADRC/G02')))
    real, read = rq3_log.sha256_file, []

    name = 'gain_A-PID_G07.json'                                     # the relative name S05_freeze.csv holds
    in_ds = os.path.join(ds, name)                                   # candidate 1; candidate 2 is `name` (work folder)

    def sha(path, bad):
        read.append(path)
        if bad(path):
            raise PermissionError(13, 'Permission denied', path)
        return real(path)
    monkeypatch.setattr(rq3_log, 'sha256_file', lambda p: sha(p, lambda x: x.endswith('gain_B-LADRC_G02.json')))
    assert [x['run_uid'] for x in LR(ds)[0]] == ['p1']             # unused set unreadable: table made
    # good copy in the dataset, unreadable copy of the same name in the work folder: never read
    work = tmp_path / 'work'
    os.makedirs(work)
    write_gain_file(work / name)
    monkeypatch.chdir(work)
    read.clear()
    monkeypatch.setattr(rq3_log, 'sha256_file', lambda p: sha(p, lambda x: x == name))
    assert len(LR(ds)[0]) == 1 and in_ds in read and name not in read
    # the dataset copy unreadable, the work copy has the sha256: taken
    monkeypatch.setattr(rq3_log, 'sha256_file', lambda p: sha(p, lambda x: x == in_ds))
    runs, _ = LR(ds)
    assert runs[0]['gain_file'] == name
    # both unreadable: the run of the set is refused, the reason names both candidates
    monkeypatch.setattr(rq3_log, 'sha256_file', lambda p: sha(p, lambda x: x in (in_ds, name)))
    with pytest.raises(ValueError, match=r'run p1: file gain cua .*khong doc duoc.*khong doc duoc'):
        LR(ds)


# ---------------------------------------------------------------- Codex Sol round 9 (CODE_R134)

def test_c9_1_gain_setting_lines_are_not_starts(tmp_path):
    """R134C9-1 (S05 6.3: the '#gain' line with first_seq = 0 written at a gain setting is not counted): a valid
    run with the setting line before its start, or with the same set applied again during a short stop, is
    tabled; the table does not ask a '#us,ev=start' for first_seq = 0."""
    r = write_run(tmp_path, 'z1', 'HW', 'final', 'PID', 'O1-F05', 'A-PID/G07', 'dat', 0.02)
    ds = _table_ds(tmp_path, [r])
    ev = open(os.path.join(r['results_dir'], 'esp32.csv.events.txt'), encoding='utf-8').read()
    assert 'first_seq=0,' in ev                                       # the fixture writes it, as the firmware does
    assert len(LR(ds)[0]) == 1
    _short_stop(r, extra=set_line('2026-11-14T10:00:03'))            # applied again during the stop
    runs, _ = LR(ds)
    assert len(runs) == 1 and runs[0]['values']['tv_duty'] is None
