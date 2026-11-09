"""tools/s06_enc_alive.py (spec S06 CB5) on frames and E lines written here: a wheel commanded to turn must show
counts at the ESP32; a wheel at rest is not a failure. No hardware."""
import json
import os
import struct
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
import s06_enc_alive as ea  # noqa: E402
import t3b_schedule as ts  # noqa: E402

T0_US = 2_000_000                 # #run_start on the Pi clock
DUR_S = 20
PAD_MS = 1000
OMEGA = (10.0, -20.0, 5.0, 40.0)


def omega_table(fault=None):
    n = DUR_S * 1000 + 2 * PAD_MS
    w = np.tile(np.array(OMEGA), (n, 1))
    if fault == 'stop_rl':          # wheel RL commanded to stop from second 5 to 8
        w[PAD_MS + 5000:PAD_MS + 8000, 2] = 0.0
    if fault == 'reverse_rl':       # RL changes sign half way through a window
        w[PAD_MS + 5500:, 2] = -5.0
    if fault == 'all_zero':
        w[:] = 0.0
    if fault == 'slow_rl':          # only 0.4 rad/s, under --omega-min
        w[:, 2] = 0.4
    return w


def make_frames(tmp_path, w, drop=None, manifest=True, start_us=T0_US - PAD_MS * 1000):
    n = len(w)
    mosi = np.zeros((n, 40), dtype=np.uint8)
    miso = np.zeros((n, 40), dtype=np.uint8)
    for k in range(n):
        mosi[k, :8] = np.frombuffer(struct.pack('>4h', *[round(x * 32767 / 40) for x in w[k]]), dtype=np.uint8)
        mosi[k, 21] = k & 0xFF
    t = np.arange(n, dtype=np.int64) * 1_000_000 + start_us * 1000
    z = dict(t_send_ns=t, mosi=mosi, miso=miso)
    m = dict(step_first=0, step_last=n - 1, steps=n)
    if drop is not None:
        z = {k: np.delete(v, drop, axis=0) for k, v in z.items()}
    if manifest:
        z['manifest'] = np.array(json.dumps(m))
    p = str(tmp_path / 'f.npz')
    np.savez(p, **z)
    return p


def make_events(tmp_path, w, scale=None, dead=None, gap=None, base=0, marks=True, run=1, extra=(), sign=None):
    """E lines every 1 ms on the commanded ω. `dead` = (wheel, first ms, last ms): counts frozen; `gap` = (first
    ms, last ms): no E line; `scale` = {wheel: factor}; `sign` = {wheel: ±1}; `base`: count offset."""
    K = ts.K_COUNT * 1e-3
    c = np.full(4, float(base))
    lines = []
    if marks:
        lines.append(f'{T0_US}\t#run_start,run={run},first_data_seq=1,timestamp_us={T0_US},qseq=0')
    for ms in range(DUR_S * 1000):
        d = w[PAD_MS + ms] * K
        for i in range(4):
            f = (scale or {}).get(i, 1.0) * (sign or {}).get(i, 1)
            if not (dead and i == dead[0] and dead[1] <= ms < dead[2]):
                c[i] += d[i] * f
        if gap and gap[0] <= ms < gap[1]:
            continue
        t = T0_US + ms * 1000
        vals = ','.join(str((int(x) + 2**31) % 2**32 - 2**31) for x in c)     # int32 of the firmware
        lines.append(f'{t}\tE,{ms},{t},{t + 5},{vals},{run}')
    if marks:
        t = T0_US + DUR_S * 1_000_000
        lines.append(f'{t}\t#run_end,run={run},timestamp_us={t},enc_ovf_run=0,ev_qdrop_run=0,qseq=9')
    p = tmp_path / 'ev.txt'
    p.write_text('\n'.join(list(lines) + list(extra)) + '\n')
    return str(p)


def go(tmp_path, frames, ev, *more):
    return ea.main([frames, '--esp32-events', ev, '--out', str(tmp_path / 'o'), *more])


def summary(tmp_path):
    return json.load(open(tmp_path / 'o' / 'S06_enc_alive.json', encoding='utf-8'))


def last_line(capsys):
    return capsys.readouterr().out.splitlines()[-1]


def test_constant_matches_the_schedule_module():
    assert ea.K_COUNT == pytest.approx(ts.K_COUNT, abs=1e-12)


def test_every_wheel_alive(tmp_path, capsys):
    w = omega_table()
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w)) == 0
    assert last_line(capsys) == '#ket_qua,ma=0'
    s = summary(tmp_path)
    assert all(p['alive'] == DUR_S for p in s['per_wheel'].values())
    assert (tmp_path / 'o' / 'S06_enc_alive.csv').read_text().splitlines()[0].startswith('idx,bin,')


def test_counts_frozen_while_commanded_fails(tmp_path, capsys):
    # the case CB5 names: on time at the Pi, encoder stopped; one wheel, three seconds
    w = omega_table()
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w, dead=(2, 5000, 8000))) == 1
    assert last_line(capsys) == '#ket_qua,ma=1'
    s = summary(tmp_path)
    assert s['per_wheel']['RL']['dead'] == 3 and s['dead'] == 3
    assert s['first_bad'] == dict(bin=5, wheel='RL', state='dead')


def test_wheel_commanded_to_stop_is_no_failure(tmp_path):
    w = omega_table('stop_rl')
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w)) == 0
    assert summary(tmp_path)['per_wheel']['RL']['skip'] == 3


def test_window_with_a_sign_change_is_not_scored(tmp_path):
    w = omega_table('reverse_rl')
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w)) == 0
    assert summary(tmp_path)['per_wheel']['RL']['skip'] == 1


def test_wrong_direction_fails(tmp_path):
    w = omega_table()
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w, sign={1: -1})) == 1
    s = summary(tmp_path)
    assert s['per_wheel']['FR']['dead'] == DUR_S and s['too_few'] == []          # a dead window was scored


def test_count_sign_option(tmp_path):
    w = omega_table()
    ev = make_events(tmp_path, w, sign={1: -1})
    assert go(tmp_path, make_frames(tmp_path, w), ev, '--count-sign', '1,-1,1,1') == 0


@pytest.mark.parametrize('factor, code', [(0.4, 1), (0.6, 0)])
def test_ratio_of_the_expected_count(tmp_path, factor, code):
    w = omega_table()
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w, scale={3: factor})) == code


def test_missing_e_lines_fail(tmp_path, capsys):
    w = omega_table()
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w, gap=(4000, 7000))) == 1
    assert last_line(capsys) == '#ket_qua,ma=1'
    assert summary(tmp_path)['no_reads'] > 0


def test_half_a_window_of_reads_is_enough(tmp_path):
    # the two ends of the reads in the window span just over half of it: scored, not `no_reads`
    w = omega_table()
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w, gap=(4000, 4450))) == 0
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w, gap=(4000, 4600))) == 1


def test_unwraps_int32(tmp_path):
    # the count of FR runs down, the others up through +2**31: raw differences jump by 2**32 inside a window
    w = omega_table()
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w, base=2**31 - 2000)) == 0


def test_e_lines_out_of_order_are_read_in_eseq_order(tmp_path):
    # UDP may deliver out of order: the first and last read of a window swapped in the file
    w = omega_table()
    ev = make_events(tmp_path, w)
    lines = open(ev).read().splitlines()
    a, b = (next(i for i, x in enumerate(lines) if f'\tE,{ms},' in x) for ms in (4000, 4999))
    lines[a], lines[b] = lines[b], lines[a]
    open(ev, 'w').write('\n'.join(lines) + '\n')
    assert go(tmp_path, make_frames(tmp_path, w), ev) == 0


def test_nothing_commanded_is_no_conclusion(tmp_path, capsys):
    w = omega_table('all_zero')
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w)) == 3
    assert last_line(capsys) == '#ket_qua,ma=3,ly_do=khong_du_dieu_kien'
    assert summary(tmp_path)['too_few'] == ['FL', 'FR', 'RL', 'RR']


def test_wheel_under_omega_min_is_not_scored(tmp_path):
    w = omega_table('slow_rl')
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w)) == 3
    assert summary(tmp_path)['too_few'] == ['RL']
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w), '--omega-min', '0.35') == 0


def test_min_bins_edge(tmp_path):
    w = omega_table()
    f, ev = make_frames(tmp_path, w), make_events(tmp_path, w)
    assert go(tmp_path, f, ev, '--min-bins', str(DUR_S)) == 0
    assert go(tmp_path, f, ev, '--min-bins', str(DUR_S + 1)) == 3


def test_no_manifest_is_missing_evidence(tmp_path, capsys):
    w = omega_table()
    assert go(tmp_path, make_frames(tmp_path, w, manifest=False), make_events(tmp_path, w)) == 3
    assert last_line(capsys) == '#ket_qua,ma=3,ly_do=thieu_bang_chung'


def test_frames_lost_fail(tmp_path):
    w = omega_table()
    assert go(tmp_path, make_frames(tmp_path, w, drop=[5000]), make_events(tmp_path, w)) == 1
    assert summary(tmp_path)['frame_log_faults']


def test_frames_not_covering_the_run(tmp_path, capsys):
    w = omega_table()
    late = make_frames(tmp_path, w, start_us=T0_US + 500_000)
    assert go(tmp_path, late, make_events(tmp_path, w)) == 3
    assert 'không phủ lượt' in summary(tmp_path)['cover_note']
    w2 = np.tile(np.array(OMEGA), (DUR_S * 1000 + PAD_MS - 300, 1))
    assert go(tmp_path, make_frames(tmp_path, w2), make_events(tmp_path, w)) == 3
    assert 'không phủ lượt' in summary(tmp_path)['cover_note']


def test_no_run_marks(tmp_path, capsys):
    w = omega_table()
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w, marks=False)) == 3
    assert last_line(capsys) == '#ket_qua,ma=3,ly_do=khong_du_dieu_kien'


def test_two_runs(tmp_path, capsys):
    w = omega_table()
    f = make_frames(tmp_path, w)
    other = ('30000000\t#run_start,run=2,first_data_seq=1,timestamp_us=30000000,qseq=10',)
    ev = make_events(tmp_path, w, extra=other)
    assert go(tmp_path, f, ev) == 3                                              # X18: no run chosen
    assert last_line(capsys) == '#ket_qua,ma=3,ly_do=khong_du_dieu_kien'
    assert go(tmp_path, f, ev, '--run', '1') == 0
    assert go(tmp_path, f, ev, '--run', '7') == 2                                # CB1: a wrong --run


def test_e_lines_of_another_run_are_left_out(tmp_path):
    # run 2 lines inside the time span of run 1, its counts far away: they must not count for run 1
    w = omega_table()
    other = tuple(f'{T0_US + k * 1000}\tE,{100000 + k},{T0_US + k * 1000},{T0_US + k * 1000 + 5},0,0,0,0,2'
                  for k in range(0, DUR_S * 1000, 7))
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w, extra=other), '--run', '1') == 0


def test_loss_counter_without_a_run_is_code_1(tmp_path, capsys):
    # C2-4: the loss read in the file is not hidden by X18
    w = omega_table()
    other = ('30000000\t#run_start,run=2,first_data_seq=1,timestamp_us=30000000,qseq=10',
             '31000000\t#run_end,run=2,timestamp_us=31000000,enc_ovf_run=4,ev_qdrop_run=0,qseq=11')
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w, extra=other)) == 1
    assert last_line(capsys) == '#ket_qua,ma=1'


@pytest.mark.parametrize('args', [['--count-sign', '1,1,1'], ['--count-sign', '1,0,1,1']])
def test_bad_count_sign_is_input_fault(tmp_path, args):
    w = omega_table()
    assert go(tmp_path, make_frames(tmp_path, w), make_events(tmp_path, w), *args) == 2


def test_missing_file_is_input_fault(tmp_path):
    w = omega_table()
    assert go(tmp_path, str(tmp_path / 'thieu.npz'), make_events(tmp_path, w)) == 2
