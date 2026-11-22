"""tools/hil_gioi_han.py on a synthetic run built here: a frame npz with the real layout
(manifest lifecycle RUN @ k0, period_us 1000, MOSI int16 BE state/traj ref+flags at
protocol offsets, MISO duty int16 BE bytes 0-7, t_send_ns 1 ms apart) and an agv_logger
CSV + .raw.jsonl (b64 of the line, t_mono_ns = Pi time + fixed 700 us latency, ESP32
clock offset and +20 ppm). The H7 model holds the old duty 50 ms after pwm goes to
exactly 0 or +-1000 (PWM_TIMEOUT), restarts the 50 ms at every pin change, and follows
-pwm/1000 with a 2 ms lag otherwise; events with pwm +-1000 held < 50 ms never appear in
MISO -> censored. All data synthetic under tmp_path. Run: python -m pytest <this file>"""
import base64
import csv
import json
import math
import os
import struct
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'rpi5'),
                os.path.join(ROOT, 'fwsim')]
import protocol as P            # noqa: E402
import hil_gioi_han as hg       # noqa: E402

FRAME_LEN, T0_NS = 48, 100_000_000_000          # t_send_ns base (Pi monotonic)
PPM, LAT_US = 20e-6, 700.0                       # ESP32 clock +20 ppm, fixed UDP latency
E0_US, LINE_DT_US = 5_000_000, 10_000            # first timestamp_us, 10 ms log tick
FLAT_HOLD_US, SLEW_LAG_US = 50_000.0, 2_000.0
ESP_COLS = ['timestamp_us', 'log_seq', 'pwm_fl', 'pwm_fr', 'pwm_rl', 'pwm_rr']

# pwm commands by log-line index -> [(wheel, level)]; i=5,10 are before the run (outside),
# i=100,179 give w2 a non-flat +800 excursion so its duty varies and the sign corr exists;
# i=180->182 is a 20 ms flat pulse that never shows in MISO (censored, pin-edge restarts
# the 50 ms timeout), i=179->182 keeps w2's duty at -0.8 so both its ->0 events are real
CHANGES = {5: [(3, 1000)], 10: [(3, 0)], 50: [(0, 1000)], 120: [(0, 0)],
           100: [(2, 800)], 179: [(2, 0)], 180: [(2, 1000)], 182: [(2, 0)],
           200: [(1, -1000)], 260: [(1, 0)], 300: [(3, 1000)]}


def write_npz(path, n, k0=100, period_us=1000, state=None, duty=None, has_manifest=True,
              lifecycle=True, valid_ref=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = state or (lambda k: np.zeros(10))
    duty = duty or (lambda k: np.zeros(4))
    mosi = np.zeros((n, FRAME_LEN), np.uint8)
    miso = np.zeros((n, FRAME_LEN), np.uint8)
    for k in range(n):
        st = np.asarray(state(k), float)
        for i, rng in enumerate(P.STATE_RANGES):
            struct.pack_into('>h', mosi[k], i * 2,
                             max(-32768, min(32767, int(round(st[i] / rng * 32767)))))
        if valid_ref and k >= k0:
            for i, v in enumerate((0.2, 0.0, 0.0)):
                struct.pack_into('>h', mosi[k], P.OFS_TRAJ_REF + i * 2,
                                 int(round(v / P.REF_RANGES[i] * 32767)))
            mosi[k][P.OFS_TRAJ_FLAGS] = P.TRAJ_VALID
        d = np.asarray(duty(k), float)
        for i in range(4):
            struct.pack_into('>h', miso[k], i * 2,
                             max(-32768, min(32767, int(round(d[i] * 32767)))))
        miso[k][16] = 1                                   # status byte: MISO never empty
    kw = dict(mosi=mosi, miso=miso,
              t_send_ns=T0_NS + np.arange(n, dtype=np.int64) * int(period_us * 1000))
    if has_manifest:
        m = {'period_us': period_us}
        if lifecycle:
            m['lifecycle'] = {'transitions': [{'phase': 'RUN', 'step': k0}]}
        kw['manifest'] = json.dumps(m)
    np.savez(path, **kw)


def esp_series(n_lines, changes):
    """changes {line_i: [(wheel, pwm)]} -> (ts, seq, pwm (n,4)) per line before drops."""
    level = np.zeros(4, dtype=np.int64)
    pwm = np.zeros((n_lines, 4), dtype=np.int64)
    for i in range(n_lines):
        for w, lv in changes.get(i, ()):
            level[w] = lv
        pwm[i] = level
    ts = E0_US + np.arange(n_lines, dtype=np.int64) * LINE_DT_US
    return ts, 1000 + np.arange(n_lines, dtype=np.int64), pwm


def write_esp(csv_path, ts, seq, pwm, off_us, drop=(), mon=None):
    """CSV + .raw.jsonl; t_mono_ns = true Pi receive time (esp clock + fixed latency).
    Dropped lines are lost packets (absent from both files). mon: optional motor_on
    per line -> appended as an extra 'motor_on' column."""
    rows = [(int(t), int(s), [int(x) for x in p])
            for i, (t, s, p) in enumerate(zip(ts, seq, pwm)) if i not in drop]
    mons = ([int(m) for i, m in enumerate(mon) if i not in drop] if mon is not None
            else None)
    cols = ESP_COLS + (['motor_on'] if mons is not None else [])
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k, (t, s, p) in enumerate(rows):
            w.writerow([t, s] + p + ([mons[k]] if mons is not None else []))
    with open(csv_path + '.raw.jsonl', 'w', encoding='utf-8') as f:
        for k, (t, s, p) in enumerate(rows):
            line = ','.join(str(x) for x in [t, s] + p
                            + ([mons[k]] if mons is not None else []))
            f.write(json.dumps({'t_mono_ns': int((off_us + (1.0 + PPM) * t + LAT_US) * 1e3),
                                'b64': base64.b64encode(line.encode()).decode()}) + '\n')


def report_level(t, changes):
    """pwm level the H7 reports at Pi time t: the last change matured else the report right
    before it; flat command matures after FLAT_HOLD_US measured from the last pin change."""
    last_tc, last_lv = -math.inf, 0
    for tc, lv in changes:
        if tc > t:
            break
        last_tc, last_lv = tc, lv
    if last_tc == -math.inf:
        return 0
    h = FLAT_HOLD_US if last_lv in (0, 1000, -1000) else SLEW_LAG_US
    if t - last_tc >= h:
        return last_lv
    return report_level(last_tc - 1.0, changes)


def build_run(tmp_path, name='x', sign=-1, drop=(), n_frames=6000, k0=150, n_lines=400,
              changes=CHANGES):
    """npz + esp log where t_pin_i (true Pi time of line i) = T0+2ms + ~10ms*i."""
    ts, seq, pwm = esp_series(n_lines, changes)
    off = T0_NS / 1e3 + 2000.0 - (1.0 + PPM) * E0_US
    esp = str(tmp_path / f'{name}.csv')
    write_esp(esp, ts, seq, pwm, off, drop=drop)
    changes_pi = {}
    for i, ws in changes.items():
        for w, lv in ws:
            changes_pi.setdefault(w, []).append((off + (1.0 + PPM) * ts[i], lv))

    def duty(k):
        t = (T0_NS + k * 1_000_000) / 1e3
        return [sign * report_level(t, sorted(changes_pi.get(w, ()))) / 1000.0
                for w in range(4)]
    npz = str(tmp_path / f'{name}_frames.npz')
    write_npz(npz, n_frames, k0=k0, duty=duty)
    return npz, esp


def build_off_run(tmp_path, name='s', n_frames=6000, k0=150, events_txt=True):
    """Motor-off scenario: motor_on column present; on-lines i=0..340 (10 ms tick), three
    footer seqs 1341-1343 (not data lines), idle 10 Hz pwm-0 lines ~101 ms later. pwm:
    fl 0->300@50 ->0@100 (lenh, doi) ->200@250; fr 0->3@200 ->0@210 (lenh trivial)
    ->-50@300; rl ->-50@300; rr ->40@330; all wheels nonzero at the last on line 340
    -> 4 'tat_motor' events anchored at ts[340] + dt_med (10 ms)."""
    n_on, n_idle = 341, 37
    ch = {50: [(0, 300)], 100: [(0, 0)], 200: [(1, 3)], 210: [(1, 0)],
          250: [(0, 200)], 300: [(1, -50), (2, -50)], 330: [(3, 40)]}
    ts = np.concatenate([E0_US + np.arange(n_on) * LINE_DT_US,
                         E0_US + 340 * LINE_DT_US + 101_308
                         + np.arange(n_idle) * 100_000]).astype(np.int64)
    seq = np.concatenate([1000 + np.arange(n_on),
                          1344 + np.arange(n_idle)]).astype(np.int64)
    mon = np.array([1] * n_on + [0] * n_idle, dtype=np.int64)
    pwm = np.zeros((len(ts), 4), dtype=np.int64)
    lvl = np.zeros(4, dtype=np.int64)
    for i in range(len(ts)):
        for w, v in ch.get(i, ()):
            lvl[w] = v
        pwm[i] = lvl if i < n_on else 0                    # idle lines log pwm 0
    off = T0_NS / 1e3 + 2000.0 - (1.0 + PPM) * E0_US
    esp = str(tmp_path / f'{name}.csv')
    write_esp(esp, ts, seq, pwm, off, mon=mon)
    if events_txt:
        with open(esp + '.events.txt', 'w', encoding='utf-8') as f:
            for s in (1341, 1342, 1343):
                f.write('2026-11-08T23:45:53.1 '
                        f'#run_end,seq={s},last_data_seq=1340,'
                        'log_qdrop=0,log_send_err=0,udp_drop=0,udp_take=0\n')
    changes_pi = {}
    for i, ws in ch.items():
        for w, lv in ws:
            changes_pi.setdefault(w, []).append((off + (1.0 + PPM) * ts[i], lv))
    t_stop = off + (1.0 + PPM) * (ts[340] + LINE_DT_US)    # first motor-off tick
    for w in range(4):
        changes_pi.setdefault(w, []).append((t_stop, 0))   # motor_stop_all -> pin flat

    def duty(k):
        t = (T0_NS + k * 1_000_000) / 1e3
        return [-report_level(t, sorted(changes_pi.get(w, ()))) / 1000.0
                for w in range(4)]
    npz = str(tmp_path / f'{name}_frames.npz')
    write_npz(npz, n_frames, k0=k0, duty=duty)
    return npz, esp


def write_segments(csv_path, marks):
    """<log>.segments.csv: marks = [row, log_seq, timestamp_us, segment, status] per CSV
    body row."""
    with open(csv_path + '.segments.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(hg.SEG_MARK_COLS)
        w.writerows(marks)


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def read_one(path):
    rows = read_rows(path)
    assert len(rows) == 1
    return rows[0]


def run_cli(capsys, argv):
    code = hg.main(argv)
    out = capsys.readouterr().out.strip().splitlines()
    return code, out[-1] if out else ''


def test_band_boundaries():
    band = hg.in_speed_band(np.array([0.010, 0.0146, 0.0146 + 5e-10, 0.010 - 1e-9,
                                      0.0146 + 1e-6, 0.010 - 1.1e-9, 0.009, 0.0147]))
    assert band.tolist() == [True, True, True, True, False, False, False, False]
    om = np.array([0.0, np.float32(0.3), np.float32(0.30000001),
                   np.nextafter(np.float32(0.3), np.float32(1.0)),
                   -np.float32(0.3), np.nextafter(np.float32(-0.3), np.float32(-1.0)),
                   np.float32(1e-6), np.float32(-1e-6)], dtype=np.float32).reshape(-1, 1)
    assert hg.blind_band(om)[:, 0].tolist() == \
        [False, True, True, False, True, False, True, True]
    k = hg.duty_kind(np.array([180 / 32767, 181 / 32767, 32587 / 32767, 32586 / 32767,
                               -32587 / 32767, -32586 / 32767, 0.0055, 0.0055 + 5e-10,
                               0.0055 + 2e-9, 1.0, -1.0, 0.5, -0.4]))
    assert k.tolist() == [0, 2, 1, 2, -1, 2, 0, 0, 2, 1, -1, 2, 2]


def test_o1_window_and_last1s(tmp_path, capsys):
    npz = str(tmp_path / 'o_frames.npz')
    write_npz(npz, 12200, k0=100)
    code, last = run_cli(capsys, ['--frame-log', npz, '--task', 'O1-F08',
                                  '--out', str(tmp_path / 'o')])
    assert code == 3 and last == '#ket_qua,ma=3,ly_do=thieu_bang_chung'
    row = read_one(tmp_path / 'o' / 'o_gioi_han.csv')
    assert row['win_kind'] == 'O1_[Tcmd,deadline)' and int(row['win_n']) == 2000
    assert float(row['t_cmd_s']) == pytest.approx(9.953, abs=0.001)

    npz2 = str(tmp_path / 'w_frames.npz')
    write_npz(npz2, 2000, k0=100)
    assert hg.main(['--frame-log', npz2, '--task', 'A-C1',
                    '--out', str(tmp_path / 'o2')]) == 3
    row = read_one(tmp_path / 'o2' / 'w_gioi_han.csv')
    assert row['win_kind'] == 'last1s_gia_dinh' and int(row['win_n']) == 1001


def test_normal_hold(tmp_path, capsys):
    npz, esp = build_run(tmp_path)
    code, last = run_cli(capsys, ['--frame-log', npz, '--esp-log', esp, '--task', 'A-C1',
                                  '--out', str(tmp_path / 'o')])
    assert code == 0 and last == '#ket_qua,ma=0'
    row = read_one(tmp_path / 'o' / 'x_gioi_han.csv')
    assert row['esp_part'] == 'co' and row['sync_kind'] == 'udp_envelope'
    assert float(row['sync_ppm']) == pytest.approx(20.0, abs=1.0)
    assert float(row['sync_resid_max_us']) < 1.0
    for w in hg.WHEELS:
        assert int(row[f'sign_ok_{w}']) == 1 and float(row[f'corr_{w}']) < -0.8
        assert int(row[f'sign_{w}']) == -1
    assert int(row['n_ev_1']) == 4 and int(row['n_follow_1']) == 3
    assert int(row['n_cens_1']) == 1
    assert int(row['n_ev_0']) == 4 and int(row['n_follow_0']) == 3
    assert int(row['n_cens_0']) == 1
    assert int(row['n_ev_outside']) == 2
    assert int(row['n_ev_seqgap']) == 0 and int(row['n_ev_sign_unknown']) == 0
    assert 45.0 <= float(row['hold_med_ms_1']) <= 55.0
    assert 45.0 <= float(row['hold_max_ms_1']) <= 55.0
    assert 19.0 <= float(row['cens_max_ms_1']) <= 21.0
    ev = read_rows(tmp_path / 'o' / 'x_su_kien_a11.csv')
    assert len(ev) == 8                                 # 4 +-1000 + 4 zero, all in-run
    cens = sorted((e for e in ev if e['followed'] == 'False'),
                  key=lambda e: int(e['log_seq']))
    # ->0 at 179: window ends at the +1000 command (10 ms) while duty is stuck at -0.8
    assert len(cens) == 2 and cens[0]['wheel'] == 'rl' and cens[0]['target'] == '0'
    assert 9.0 <= float(cens[0]['cens_ms']) <= 11.0 and cens[0]['hold_ms'] == ''
    # ->+1000 at 180: 20 ms pulse, duty never leaves -0.8 within the window
    assert cens[1]['wheel'] == 'rl' and cens[1]['target'] == '-1'
    assert 19.0 <= float(cens[1]['cens_ms']) <= 21.0 and cens[1]['hold_ms'] == ''
    w2_zero = sorted((e for e in ev if e['wheel'] == 'rl' and e['target'] == '0'),
                     key=lambda e: int(e['log_seq']))
    assert len(w2_zero) == 2 and w2_zero[0]['followed'] == 'False'
    # ->0 at 182: normal 50 ms flat hold of the -0.8 level
    assert w2_zero[1]['followed'] == 'True'
    assert 45.0 <= float(w2_zero[1]['hold_ms']) <= 55.0
    for e in ev:
        if e['followed'] == 'True' and e['wheel'] != 'rl':
            assert 45.0 <= float(e['hold_ms']) <= 55.0
    seg = read_rows(tmp_path / 'o' / 'x_doan_phang.csv')
    fl_neg = [s for s in seg if s['wheel'] == 'fl' and s['kind'] == '-1']
    assert fl_neg and all(600 <= int(s['len_frames']) <= 750 for s in fl_neg)


def test_seqgap_event_skipped(tmp_path):
    npz, esp = build_run(tmp_path, drop=(119,))          # lost packet before the w0->0 line
    out = tmp_path / 'o'
    assert hg.main(['--frame-log', npz, '--esp-log', esp, '--out', str(out)]) == 0
    row = read_one(out / 'x_gioi_han.csv')
    assert int(row['n_ev_seqgap']) == 1                  # the ->0 event of w0 hit the gap
    assert int(row['n_ev_0']) == 3                       # rl@179, rl@182 and fr@260 remain
    assert int(row['n_follow_0']) == 2 and int(row['n_cens_0']) == 1


def test_wrong_sign_not_scored(tmp_path):
    npz, esp = build_run(tmp_path, sign=+1)              # duty = +pwm/1000: sign flipped
    out = tmp_path / 'o'
    assert hg.main(['--frame-log', npz, '--esp-log', esp, '--out', str(out)]) == 0
    row = read_one(out / 'x_gioi_han.csv')
    for w in hg.WHEELS:
        assert int(row[f'sign_ok_{w}']) == 0 and int(row[f'sign_{w}']) == 1
    assert int(row['n_ev_sign_unknown']) == 4            # all in-run +-1000 events
    assert int(row['n_ev_1']) == 0
    assert int(row['n_ev_0']) == 4                       # target-0 still scored
    assert int(row['n_follow_0']) == 3 and int(row['n_cens_0']) == 1


def test_miso_only(tmp_path):
    npz = str(tmp_path / 'm_frames.npz')
    write_npz(npz, 2000, k0=100)
    assert hg.main(['--frame-log', npz, '--out', str(tmp_path / 'o')]) == 3
    row = read_one(tmp_path / 'o' / 'm_gioi_han.csv')
    assert row['esp_part'] == 'khong_co_log_ESP32' and row['esp_log'] == ''


def test_input_errors(tmp_path):
    npz = str(tmp_path / 'e_frames.npz')
    write_npz(npz, 500)
    esp = str(tmp_path / 'e.csv')
    with open(esp, 'w', newline='', encoding='utf-8') as f:      # pwm_rr column missing
        w = csv.writer(f)
        w.writerow(['timestamp_us', 'log_seq', 'pwm_fl', 'pwm_fr', 'pwm_rl'])
        w.writerow([1, 1, 0, 0, 0])
    assert hg.main(['--frame-log', npz, '--esp-log', esp,
                    '--out', str(tmp_path / 'o1')]) == 2
    write_npz(str(tmp_path / 'n_frames.npz'), 500, has_manifest=False)
    assert hg.main(['--frame-log', str(tmp_path / 'n_frames.npz'),
                    '--out', str(tmp_path / 'o2')]) == 2
    write_npz(str(tmp_path / 'r_frames.npz'), 500, lifecycle=False, valid_ref=False)
    assert hg.main(['--frame-log', str(tmp_path / 'r_frames.npz'),
                    '--out', str(tmp_path / 'o3')]) == 2          # command_start ValueError
    assert hg.main(['--frame-log', npz, '--task', 'O1-F99',
                    '--out', str(tmp_path / 'o4')]) == 2          # bad O1 task id


def test_batch(tmp_path, capsys):
    root = tmp_path / 'root'
    a_npz = root / 'runA' / 'frames' / 'a_frames.npz'
    b_npz = root / 'runB' / 'frames' / 'b_frames.npz'
    write_npz(str(a_npz), 2000, k0=100)
    st = np.zeros(10)
    st[7] = 0.012                                          # vx in the speed band
    write_npz(str(b_npz), 2000, k0=100, state=lambda k: st)
    row_a = hg.count_run(str(a_npz), 'A-C1')[0]
    row_b = hg.count_run(str(b_npz), 'A-C1')[0]
    s1_cols = (['run', 'npz', 'name', 'controller', 'task'] + hg.CMP_SAME
               + [cs for cs, _ in hg.CMP_PAIRS])
    s1 = tmp_path / 'ket_qua_s1.csv'
    with open(s1, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=s1_cols)
        w.writeheader()
        for run, fn, name, row, wrong in (
                ('runA', 'a_frames.npz', 'a', row_a, None),
                ('runB', 'b_frames.npz', 'b', row_b, {'n_vband_win': row_b['n_vband_win'] + 5})):
            r = dict(run=run, npz=fn, name=name, controller='PID', task='A-C1')
            r.update({c: row[c] for c in hg.CMP_SAME})
            r.update({cs: row[ce] for cs, ce in hg.CMP_PAIRS})
            if wrong:
                r.update(wrong)
            w.writerow(r)
    out = tmp_path / 'e2'
    code, last = run_cli(capsys, ['--batch-from-s1', str(s1), '--root', str(root),
                                  '--out', str(out)])
    assert code == 3 and last == '#ket_qua,ma=3,ly_do=thieu_bang_chung'
    e2 = read_rows(out / 'ket_qua_e2.csv')
    assert len(e2) == 2 and all(r['esp_part'] == 'khong_co_log_ESP32' for r in e2)
    assert int(e2[1]['n_vband_win']) == row_b['n_vband_win'] == 1001
    mis = read_rows(out / 'so_khop_s1.csv')
    assert len(mis) == 1
    m = mis[0]
    assert m['run'] == 'runB' and m['cot_s1'] == m['cot_e2'] == 'n_vband_win'
    assert int(m['gia_tri_s1']) == row_b['n_vband_win'] + 5
    assert int(m['gia_tri_e2']) == row_b['n_vband_win']
    assert (out / 'su_kien' / 'runB__b_su_kien_a11.csv').is_file()
    assert (out / 'doan_phang' / 'runB__b_doan_phang.csv').is_file()


def test_motor_off_events(tmp_path, capsys):
    npz, esp = build_off_run(tmp_path)
    code, last = run_cli(capsys, ['--frame-log', npz, '--esp-log', esp,
                                  '--out', str(tmp_path / 'o')])
    assert code == 0 and last == '#ket_qua,ma=0'
    row = read_one(tmp_path / 'o' / 's_gioi_han.csv')
    assert int(row['motor_on_col']) == 1
    assert float(row['tick_dt_min_ms']) == float(row['tick_dt_med_ms']) == \
        float(row['tick_dt_max_ms']) == 10.0
    # 4 'tat_motor' events, all followed ~50 ms after the true stop tick
    assert int(row['n_ev_off']) == 4 and int(row['n_follow_off']) == 4
    assert int(row['n_cens_off']) == 0 and int(row['n_off_unproven']) == 0
    assert 46.0 <= float(row['hold_med_ms_off']) <= 54.0
    assert 46.0 <= float(row['hold_max_ms_off']) <= 54.0
    # 'lenh': fl@100 ->0 (doi, ~50 ms) and fr@210 ->0 (trivial, already within tol)
    assert int(row['n_ev_0']) == 2 and int(row['n_follow_0']) == 2
    assert int(row['n_ev_doi_0']) == 1 and int(row['n_follow_doi_0']) == 1
    assert 45.0 <= float(row['hold_med_doi_ms_0']) <= 55.0
    # the 1340->1344 footer gap and the idle pwm-0 lines create no 'lenh' event
    assert int(row['n_ev_seqgap']) == 0
    ev = read_rows(tmp_path / 'o' / 's_su_kien_a11.csv')
    off_ev = [e for e in ev if e['loai'] == 'tat_motor']
    assert len(off_ev) == 4 and all(e['followed'] == 'True' for e in off_ev)
    assert all(e['log_seq'] == '1340' and 45.0 <= float(e['hold_ms']) <= 55.0
               for e in off_ev)
    assert sorted(int(e['pwm_prev']) for e in off_ev) == [-50, -50, 40, 200]
    lenh_ev = [e for e in ev if e['loai'] == 'lenh']
    assert len(lenh_ev) == 2 and all(e['log_seq'] != '1344' for e in lenh_ev)
    tr = [e for e in lenh_ev if e['wheel'] == 'fr'][0]
    assert tr['pwm_prev'] == '3' and float(tr['hold_ms']) < 2.0     # trivial: hold ~0


def test_motor_off_unproven(tmp_path):
    npz, esp = build_off_run(tmp_path, events_txt=False)
    out = tmp_path / 'o'
    assert hg.main(['--frame-log', npz, '--esp-log', esp, '--out', str(out)]) == 0
    row = read_one(out / 's_gioi_han.csv')
    assert int(row['n_off_unproven']) == 4 and int(row['n_ev_off']) == 0
    ev = read_rows(out / 's_su_kien_a11.csv')
    assert all(e['loai'] == 'lenh' for e in ev)


def test_motor_off_outside(tmp_path):
    # frame log ends at ~3.0 s, the stop tick anchor is at ~3.4 s: not censored
    npz, esp = build_off_run(tmp_path, n_frames=3000)
    out = tmp_path / 'o'
    assert hg.main(['--frame-log', npz, '--esp-log', esp, '--out', str(out)]) == 0
    row = read_one(out / 's_gioi_han.csv')
    assert int(row['n_off_outside']) == 4 and int(row['n_ev_off']) == 0
    assert int(row['n_follow_off']) == 0 and int(row['n_cens_off']) == 0
    ev = read_rows(out / 's_su_kien_a11.csv')
    assert all(e['loai'] == 'lenh' for e in ev)


def test_segments_duplicate_and_conflict(tmp_path, capsys):
    npz, esp = build_run(tmp_path)
    with open(esp, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    body2 = body[:7] + [list(body[4])] + body[7:]          # exact copy of body row 5
    with open(esp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(body2)
    marks = [[i, r[1], r[0], 0, 'duplicate' if i == 8 else 'ok']
             for i, r in enumerate(body2, start=1)]
    write_segments(esp, marks)
    code, last = run_cli(capsys, ['--frame-log', npz, '--esp-log', esp,
                                  '--out', str(tmp_path / 'o1')])
    assert code == 0 and last == '#ket_qua,ma=0'           # duplicate dropped, run scores
    row = read_one(tmp_path / 'o1' / 'x_gioi_han.csv')
    assert row['esp_part'] == 'co'
    marks2 = [m[:] for m in marks]
    marks2[9][4] = 'conflict'
    write_segments(esp, marks2)
    code, last = run_cli(capsys, ['--frame-log', npz, '--esp-log', esp,
                                  '--out', str(tmp_path / 'o2')])
    assert code == 3 and last == '#ket_qua,ma=3,ly_do=thieu_bang_chung'
    row = read_one(tmp_path / 'o2' / 'x_gioi_han.csv')
    assert row['esp_part'].startswith('loi')
