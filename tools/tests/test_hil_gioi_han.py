"""tools/hil_gioi_han.py on a synthetic run built here: a frame npz with the real layout
(manifest lifecycle RUN @ k0, period_us 1000, MOSI int16 BE state/traj ref+flags at
protocol offsets, MISO duty int16 BE bytes 0-7, t_send_ns 1 ms apart) and an agv_logger
CSV + .raw.jsonl (b64 of the line, t_mono_ns = Pi time + fixed 700 us latency, ESP32
clock offset and +20 ppm). The H7 model holds the old duty 50 ms after pwm goes to
exactly 0 or +-1000 (PWM_TIMEOUT), restarts the 50 ms at every pin change, and follows
-pwm/1000 with a 2 ms lag otherwise; events with pwm +-1000 held < 50 ms never appear in
MISO -> censored. All data synthetic under tmp_path. Run: python -m pytest <this file>"""
import base64
import binascii
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


def set_crc(frame):
    """MISO v4 CRC-16/CCITT-FALSE at bytes 8-9 (LE) over bytes 0-7 and 10-end
    (protocol.miso_check)."""
    b = bytes(frame)
    struct.pack_into('<H', frame, P.OFS_MISO_CRC,
                     binascii.crc_hqx(b[:P.OFS_MISO_CRC] + b[P.OFS_MISO_CRC + 2:], 0xFFFF))


def write_npz(path, n, k0=100, period_us=1000, state=None, duty=None, has_manifest=True,
              lifecycle=True, valid_ref=True, post=None):
    """Frames as the Pi logs them: MOSI seq = step & 0xFF, MISO with a good CRC.
    post(mosi, miso, t_send_ns): optional in-place faults (empty MISO, bad CRC, gaps)."""
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
        mosi[k][P.OFS_SEQ] = k & 0xFF
        d = np.asarray(duty(k), float)
        for i in range(4):
            struct.pack_into('>h', miso[k], i * 2,
                             max(-32768, min(32767, int(round(d[i] * 32767)))))
        miso[k][16] = 1                                   # status byte: MISO never empty
        set_crc(miso[k])
    t_send = T0_NS + np.arange(n, dtype=np.int64) * int(period_us * 1000)
    if post is not None:
        post(mosi, miso, t_send)
    kw = dict(mosi=mosi, miso=miso, t_send_ns=t_send)
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
              changes=CHANGES, post=None, duty_wrap=None):
    """npz + esp log where t_pin_i (true Pi time of line i) = T0+2ms + ~10ms*i.
    duty_wrap(k, duty_list) -> duty_list: optional override of the modelled MISO duty."""
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
        d = [sign * report_level(t, sorted(changes_pi.get(w, ()))) / 1000.0
             for w in range(4)]
        return duty_wrap(k, d) if duty_wrap is not None else d
    npz = str(tmp_path / f'{name}_frames.npz')
    write_npz(npz, n_frames, k0=k0, duty=duty, post=post)
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
    assert int(row['n_ev_1']) == 4 and int(row['n_vao_dai_1']) == 3
    assert int(row['n_cens_1']) == 1
    assert int(row['n_ev_0']) == 4 and int(row['n_vao_dai_0']) == 3
    assert int(row['n_cens_0']) == 1
    assert int(row['n_ev_outside']) == 2
    assert int(row['n_ev_seqgap']) == 0 and int(row['n_ev_sign_unknown']) == 0
    assert 45.0 <= float(row['t_vao_dai_med_ms_1']) <= 55.0
    assert 45.0 <= float(row['t_vao_dai_max_ms_1']) <= 55.0
    assert 19.0 <= float(row['cens_max_ms_1']) <= 21.0
    # both censored events end at a command change, not at the end of the log
    assert int(row['n_cat_doi_lenh_0']) == 1 and int(row['n_cat_doi_lenh_1']) == 1
    assert int(row['n_cat_het_log_0']) == int(row['n_cat_het_log_1']) == 0
    assert int(row['n_ev_mat_mau_0']) == int(row['n_ev_mat_mau_1']) == 0
    ev = read_rows(tmp_path / 'o' / 'x_su_kien_a11.csv')
    assert len(ev) == 8                                 # 4 +-1000 + 4 zero, all in-run
    assert all(e['moc_t0'] == 'udp_envelope' for e in ev)
    assert all(float(e['bat_dinh_ms']) == pytest.approx(float(row['sync_resid_max_us']) / 1e3,
                                                        abs=1e-6) for e in ev)
    cens = sorted((e for e in ev if e['vao_dai'] == 'False'),
                  key=lambda e: int(e['log_seq']))
    # ->0 at 179: window ends at the +1000 command (10 ms) while duty is stuck at -0.8
    assert len(cens) == 2 and cens[0]['wheel'] == 'rl' and cens[0]['target'] == '0'
    assert 9.0 <= float(cens[0]['cens_ms']) <= 11.0 and cens[0]['t_vao_dai_ms'] == ''
    assert cens[0]['cat'] == 'doi_lenh'
    # ->+1000 at 180: 20 ms pulse, duty never leaves -0.8 within the window
    assert cens[1]['wheel'] == 'rl' and cens[1]['target'] == '-1'
    assert 19.0 <= float(cens[1]['cens_ms']) <= 21.0 and cens[1]['t_vao_dai_ms'] == ''
    w2_zero = sorted((e for e in ev if e['wheel'] == 'rl' and e['target'] == '0'),
                     key=lambda e: int(e['log_seq']))
    assert len(w2_zero) == 2 and w2_zero[0]['vao_dai'] == 'False'
    # ->0 at 182: normal 50 ms flat hold of the -0.8 level
    assert w2_zero[1]['vao_dai'] == 'True' and w2_zero[1]['cat'] == ''
    assert 45.0 <= float(w2_zero[1]['t_vao_dai_ms']) <= 55.0
    for e in ev:
        if e['vao_dai'] == 'True' and e['wheel'] != 'rl':
            assert 45.0 <= float(e['t_vao_dai_ms']) <= 55.0
            assert int(e['n_lien_vao_dai']) > 100 and int(e['n_khong_bc']) == 0
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
    assert int(row['n_vao_dai_0']) == 2 and int(row['n_cens_0']) == 1


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
    assert int(row['n_vao_dai_0']) == 3 and int(row['n_cens_0']) == 1


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
    # 4 'tat_motor' events, all first seen in band ~50 ms after the true stop tick
    assert int(row['n_ev_off']) == 4 and int(row['n_vao_dai_off']) == 4
    assert int(row['n_cens_off']) == 0 and int(row['n_off_unproven']) == 0
    assert 46.0 <= float(row['t_vao_dai_med_ms_off']) <= 54.0
    assert 46.0 <= float(row['t_vao_dai_max_ms_off']) <= 54.0
    # 'lenh': fl@100 ->0 (doi, ~50 ms) and fr@210 ->0 (trivial, already within tol)
    assert int(row['n_ev_0']) == 2 and int(row['n_vao_dai_0']) == 2
    assert int(row['n_ev_doi_0']) == 1 and int(row['n_vao_dai_doi_0']) == 1
    assert 45.0 <= float(row['t_vao_dai_med_ms_doi_0']) <= 55.0
    # the 1340->1344 footer gap and the idle pwm-0 lines create no 'lenh' event
    assert int(row['n_ev_seqgap']) == 0
    ev = read_rows(tmp_path / 'o' / 's_su_kien_a11.csv')
    off_ev = [e for e in ev if e['loai'] == 'tat_motor']
    assert len(off_ev) == 4 and all(e['vao_dai'] == 'True' for e in off_ev)
    assert all(e['log_seq'] == '1340' and 45.0 <= float(e['t_vao_dai_ms']) <= 55.0
               for e in off_ev)
    # anchor: line time + tick median; uncertainty = fit residual + tick spread (0 here)
    assert all(e['moc_t0'] == 'udp_envelope+tick_dt_med' for e in off_ev)
    assert all(float(e['bat_dinh_ms']) == pytest.approx(float(row['sync_resid_max_us']) / 1e3,
                                                        abs=1e-6) for e in off_ev)
    assert sorted(int(e['pwm_prev']) for e in off_ev) == [-50, -50, 40, 200]
    lenh_ev = [e for e in ev if e['loai'] == 'lenh']
    assert len(lenh_ev) == 2 and all(e['log_seq'] != '1344' for e in lenh_ev)
    tr = [e for e in lenh_ev if e['wheel'] == 'fr'][0]
    assert tr['pwm_prev'] == '3' and float(tr['t_vao_dai_ms']) < 2.0     # trivial: ~0


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
    assert int(row['n_vao_dai_off']) == 0 and int(row['n_cens_off']) == 0
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


# ---- C3-14: empty / bad-CRC MISO frames and sample gaps are not duty-0 evidence ----

def test_c3_14_empty_miso_not_duty0(tmp_path):
    """Codex C3-14 counterexample: 60 consecutive empty MISO frames at 1 kHz inside a run
    whose real duty is 0.5 on every wheel. dem_s1 (compat columns) reads them as a 60 ms
    four-wheel duty-0 run; the evidence columns must not."""
    def post(mosi, miso, t):
        miso[1000:1060] = 0
    npz = str(tmp_path / 'e_frames.npz')
    write_npz(npz, 2000, k0=100, duty=lambda k: [0.5] * 4, post=post)
    assert hg.main(['--frame-log', npz, '--out', str(tmp_path / 'o')]) == 3
    row = read_one(tmp_path / 'o' / 'e_gioi_han.csv')
    # compat (dem_s1) unchanged
    assert int(row['n_all4_duty0']) == 60 and int(row['max_all4_0_ms']) == 60
    assert int(row['n_duty0_any']) == 60 and int(row['n_miso_empty_run']) == 60
    # evidence
    assert int(row['n_khong_bc_run']) == 60 and int(row['n_miso_crc_bad_run']) == 0
    assert int(row['n_mat_mau_run']) == 0
    assert int(row['n_all4_duty0_bc']) == 0 and float(row['max_all4_0_bc_ms']) == 0.0
    assert int(row['n_duty0_any_bc']) == 0 and int(row['n_flat_any_bc']) == 0
    assert float(row['max_flat_bc_ms']) == 0.0
    for w in hg.WHEELS:
        assert int(row[f'n_d0_{w}']) == 0 and int(row[f'n_d0_tol_{w}']) == 0
        assert int(row[f'n_seg_tol_{w}']) == 0
    assert int(row['n_d0_tol_any']) == 0 and int(row['n_flat_tol_any']) == 0
    assert read_rows(tmp_path / 'o' / 'e_doan_phang.csv') == []


def test_c3_14_crc_and_gaps_cut_segments(tmp_path):
    """duty 0 on every wheel for the whole run (k0 = 100 .. 1999). Frame 1000 has a bad
    CRC; t_send steps exactly 1.5 periods before frame 300 (not a gap: '>' strict) and
    1.5 periods + 1 ns before frame 600 (a gap), 5 ms before frame 1500 (a gap); the MOSI
    seq skips one before frame 1700 (a gap). Evidence runs: [100,600) [600,1000)
    [1001,1500) [1500,1700) [1700,2000) -> 500, 400, 499, 200, 300 frames."""
    def post(mosi, miso, t):
        miso[1000][P.OFS_MISO_CRC] ^= 0xFF
        t[300:] += 500_000
        t[600:] += 500_001
        t[1500:] += 4_000_000
        for k in range(1700, len(mosi)):
            mosi[k][P.OFS_SEQ] = (k + 1) & 0xFF
    npz = str(tmp_path / 'g_frames.npz')
    write_npz(npz, 2000, k0=100, post=post)
    assert hg.main(['--frame-log', npz, '--out', str(tmp_path / 'o')]) == 3
    row = read_one(tmp_path / 'o' / 'g_gioi_han.csv')
    assert int(row['n_all4_duty0']) == 1900 and int(row['max_all4_0_ms']) == 1900   # compat
    assert int(row['n_miso_crc_bad_run']) == 1 and int(row['n_khong_bc_run']) == 1
    assert int(row['n_mat_mau_run']) == 3
    assert int(row['n_all4_duty0_bc']) == 1899 and float(row['max_all4_0_bc_ms']) == 500.0
    assert float(row['max_flat_bc_ms']) == 500.0 and float(row['max_flat_tol_ms']) == 500.0
    for w in hg.WHEELS:
        assert int(row[f'n_seg_tol_{w}']) == 5 and float(row[f'max_seg_tol_ms_{w}']) == 500.0
        assert int(row[f'n_d0_tol_{w}']) == 1899 and int(row[f'n_d0_{w}']) == 1899
    seg = read_rows(tmp_path / 'o' / 'g_doan_phang.csv')
    fl = sorted((int(s['k_start']), int(s['len_frames'])) for s in seg if s['wheel'] == 'fl')
    assert fl == [(100, 500), (600, 400), (1001, 499), (1500, 200), (1700, 300)]


def test_c3_14_bad_crc_frame_not_first_in_band(tmp_path):
    """A11: the first in-band frame of the fl ->0 event (log line 120) carries a bad CRC: it
    is not evidence, so the first observed entry moves to the next frame and the event
    counts one unobserved frame."""
    npz, esp = build_run(tmp_path, name='a')
    out = tmp_path / 'o1'
    assert hg.main(['--frame-log', npz, '--esp-log', esp, '--out', str(out)]) == 0
    ev = [e for e in read_rows(out / 'a_su_kien_a11.csv')
          if e['wheel'] == 'fl' and e['target'] == '0']
    assert len(ev) == 1 and ev[0]['vao_dai'] == 'True' and ev[0]['n_khong_bc'] == '0'
    k_in = int(ev[0]['k_vao_dai'])

    def post(mosi, miso, t):
        miso[k_in][P.OFS_MISO_CRC] ^= 0xFF
    npz, esp = build_run(tmp_path, name='b', post=post)
    out = tmp_path / 'o2'
    assert hg.main(['--frame-log', npz, '--esp-log', esp, '--out', str(out)]) == 0
    ev = [e for e in read_rows(out / 'b_su_kien_a11.csv')
          if e['wheel'] == 'fl' and e['target'] == '0']
    assert int(ev[0]['k_vao_dai']) == k_in + 1 and ev[0]['n_khong_bc'] == '1'
    row = read_one(out / 'b_gioi_han.csv')
    assert int(row['n_ev_mat_mau_0']) == 1 and int(row['n_miso_crc_bad_run']) == 1


# ---- C3-15: A11 reports the time to the first observed in-band frame, not "followed" ----

def test_c3_15_single_frame_in_band(tmp_path):
    """Codex C3-15 counterexample: after fl goes 200 -> 0 (log line 120) the MISO reads
    -0.2 -> 0.004 (one frame, in the +-0.0055 band) -> -0.2 again (the 50 ms hold). The
    result is named the first observed entry, with n_lien_vao_dai = 1, not 'followed'."""
    ch = {50: [(0, 200)], 120: [(0, 0)], 300: [(0, 300)]}
    blip = 1210                                  # ~8 ms after the ->0 line reaches the Pi

    def wrap(k, d):
        if k == blip:
            d[0] = 0.004
        return d
    npz, esp = build_run(tmp_path, name='c', changes=ch, duty_wrap=wrap)
    out = tmp_path / 'o'
    assert hg.main(['--frame-log', npz, '--esp-log', esp, '--out', str(out)]) == 0
    head = next(csv.reader(open(out / 'c_su_kien_a11.csv', newline='', encoding='utf-8')))
    assert 'followed' not in head and 'hold_ms' not in head and 'k_follow' not in head
    ev = [e for e in read_rows(out / 'c_su_kien_a11.csv')
          if e['wheel'] == 'fl' and e['target'] == '0']
    assert len(ev) == 1
    e = ev[0]
    assert e['vao_dai'] == 'True' and int(e['k_vao_dai']) == blip
    assert int(e['n_lien_vao_dai']) == 1 and e['n_khong_bc'] == '0' and e['cat'] == ''
    assert 5.0 <= float(e['t_vao_dai_ms']) <= 10.0
    assert e['moc_t0'] == 'udp_envelope' and float(e['bat_dinh_ms']) >= 0.0
    row = read_one(out / 'c_gioi_han.csv')
    assert not any(c.startswith(('n_follow', 'hold_')) for c in row)
    assert int(row['n_vao_dai_0']) == 1 and float(row['t_vao_dai_max_ms_0']) <= 10.0


def test_c3_15_censored_by_log_end(tmp_path):
    """fl 200 -> 0 at log line 560 (~5.6 s); the frame log ends ~20 ms later, inside the
    50 ms hold: censored, cut by the end of the log (cat 'het_log'), not by a new command."""
    ch = {50: [(0, 200)], 560: [(0, 0)]}
    npz, esp = build_run(tmp_path, name='d', changes=ch, n_lines=600, n_frames=5625)
    out = tmp_path / 'o'
    assert hg.main(['--frame-log', npz, '--esp-log', esp, '--out', str(out)]) == 0
    ev = [e for e in read_rows(out / 'd_su_kien_a11.csv')
          if e['wheel'] == 'fl' and e['target'] == '0']
    assert len(ev) == 1 and ev[0]['vao_dai'] == 'False' and ev[0]['cat'] == 'het_log'
    assert ev[0]['n_lien_vao_dai'] == '0' and 15.0 <= float(ev[0]['cens_ms']) <= 25.0
    row = read_one(out / 'd_gioi_han.csv')
    assert int(row['n_cat_het_log_0']) == 1 and int(row['n_cat_doi_lenh_0']) == 0


# ---- C3-17: band count on the completion window of the scorer ----

def _score(**kw):
    s = dict(k0=100, t0_step=100.0, T_cmd=10.0, t_deadline=12.0, t_end=10.5,
             completed='true', frames_scored=12100, frame_log='o_frames.npz',
             t0_src='lifecycle_RUN')
    s.update(kw)
    return s


def _speed_npz(tmp_path, name='o', n=12200):
    st = np.zeros(10)
    st[7] = 0.012                                          # vx in the 10-14.6 mm/s band
    npz = str(tmp_path / f'{name}_frames.npz')
    write_npz(npz, n, k0=100, state=lambda k: st)
    return npz


def test_c3_17_score_window(tmp_path):
    """Codex C3-17 counterexample: completion at t_end = 10.5 s, deadline 12 s. Ticks after
    t_end but before the deadline are in the nominal window (compat n_vband_win) but not in
    the completion window of the scorer (n_vband_ht)."""
    npz = _speed_npz(tmp_path)
    cases = [(_score(), 501),                              # t = 10.000 .. 10.500 inclusive
             (_score(t_end=10.5 - 5e-10), 501),            # within 1e-9 of a frame
             (_score(t_end=10.4995), 500),
             (_score(t_end=None, completed='false'), 2000)]    # [10, 12): deadline strict
    for i, (sc, n_ht) in enumerate(cases):
        sj = tmp_path / f's{i}.json'
        sj.write_text(json.dumps(sc), encoding='utf-8')
        out = tmp_path / f'o{i}'
        assert hg.main(['--frame-log', npz, '--task', 'O1-F08', '--score-json', str(sj),
                        '--out', str(out)]) == 3
        row = read_one(out / 'o_gioi_han.csv')
        assert int(row['n_vband_win']) == 2000 and row['win_kind'] == 'O1_[Tcmd,deadline)'
        assert int(row['win_ht_n']) == n_ht and int(row['n_vband_ht']) == n_ht, i
        assert row['win_ht_src'].startswith('hil_o1_score:lifecycle_RUN')
        assert float(row['t_ht0_s']) == 10.0
    # no scorer result: no completion-window count; the last second stays descriptive
    out = tmp_path / 'on'
    assert hg.main(['--frame-log', npz, '--out', str(out)]) == 3
    row = read_one(out / 'o_gioi_han.csv')
    assert row['win_ht_src'] == 'khong_co_ket_qua_cham' and row['n_vband_ht'] == ''
    assert row['win_kind'] == 'last1s_gia_dinh'
    assert int(row['n_vband_win']) == int(row['n_vband_last1s']) == 1001


def test_c3_17_score_json_mismatch(tmp_path):
    npz = _speed_npz(tmp_path)
    for i, sc in enumerate([_score(frames_scored=12099), _score(frame_log='z_frames.npz'),
                            {k: v for k, v in _score().items() if k != 'T_cmd'}]):
        sj = tmp_path / f'm{i}.json'
        sj.write_text(json.dumps(sc), encoding='utf-8')
        assert hg.main(['--frame-log', npz, '--task', 'O1-F08', '--score-json', str(sj),
                        '--out', str(tmp_path / f'o{i}')]) == 2


def test_c3_17_batch_cham(tmp_path, capsys, monkeypatch):
    """--cham runs the scorer (stubbed here) on O1 rows, keeps its JSON under cham/ and
    takes the mask from it; a scorer failure is recorded, the batch goes on; the dem_s1
    compat columns still match."""
    root = tmp_path / 'root'
    for run in ('runA', 'runB'):
        st = np.zeros(10)
        st[7] = 0.012
        write_npz(str(root / run / 'frames' / 'o_frames.npz'), 12200, k0=100,
                  state=lambda k, s=st: s)

    def fake_score(path, task, **kw):
        if 'runB' in path:
            raise ValueError('khong cham duoc')
        assert kw.get('allow_final_set') and kw.get('allow_pre_b81_15')
        return _score()
    monkeypatch.setattr(hg.hil_o1_score, 'score', fake_score)
    s1 = tmp_path / 'ket_qua_s1.csv'
    with open(s1, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'npz', 'name', 'task'] + hg.CMP_SAME
                           + [cs for cs, _ in hg.CMP_PAIRS])
        w.writeheader()
        for run in ('runA', 'runB'):
            row = hg.count_run(str(root / run / 'frames' / 'o_frames.npz'), 'O1-F08')[0]
            r = dict(run=run, npz='o_frames.npz', name='o', task='O1-F08')
            r.update({c: row[c] for c in hg.CMP_SAME})
            r.update({cs: row[ce] for cs, ce in hg.CMP_PAIRS})
            w.writerow(r)
    out = tmp_path / 'e2'
    code, last = run_cli(capsys, ['--batch-from-s1', str(s1), '--root', str(root),
                                  '--out', str(out), '--cham'])
    assert code == 3
    e2 = {r['run']: r for r in read_rows(out / 'ket_qua_e2.csv')}
    assert e2['runA']['win_ht_src'].startswith('hil_o1_score') and \
        int(e2['runA']['n_vband_ht']) == 501
    assert e2['runB']['win_ht_src'].startswith('loi_cham: ValueError') and \
        e2['runB']['n_vband_ht'] == '' and e2['runB']['error'] == ''
    assert json.load(open(out / 'cham' / 'runA__o_score.json'))['t_end'] == 10.5
    assert not (out / 'cham' / 'runB__o_score.json').exists()
    assert read_rows(out / 'so_khop_s1.csv') == []
