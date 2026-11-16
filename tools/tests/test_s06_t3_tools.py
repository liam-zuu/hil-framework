"""S06 T3 tools (design 2026-11-15_Thiet_ke_S06_T3_danh_dau_v1 section 4; line format T3,qseq,t_us,src,gpio,lvl,
gseq,run agreed with CODE 17/11): the T3 line in s06_esp32_check.parse_events and s06_stream (qseq whole,
bad = 0), s06_clock_sync, s06_t3_score on a whole simulated run (Pi frames from the real schedule playback and
marker rule, ESP32 events from a model of the H7 and the ESP32 ISR)."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
RPI5 = os.path.join(os.path.dirname(TOOLS), 'rpi5')
for p in (TOOLS, RPI5):
    if p not in sys.path:
        sys.path.insert(0, p)

import s06_esp32_check as chk          # noqa: E402
import s06_stream as stm               # noqa: E402
import s06_clock_sync as cs            # noqa: E402
import s06_t3_score as t3s             # noqa: E402
import t3_schedule as ts               # noqa: E402
from t3b_schedule import ScheduleOmegaPlant, write_csv   # noqa: E402
from hil_node import marker_due        # noqa: E402
from protocol import encode_state, FRAME_SIZE   # noqa: E402

OFF_US, RATE = 7_000_000.0, 1.0 + 30e-6        # t_pi = OFF + RATE * t_esp
T0_PI_NS = 1_000_000_000_000


# ── T3 line in the shared readers (CODE condition 17/11) ──────────────────────

def test_parse_events_t3_qseq_whole_bad_zero():
    raw = [(10, '#run_start,run=1,timestamp_us=5,profile=t3,qseq=1'),
           (11, 'T3,2,100,M,38,1,1,1'), (12, 'J,3,105,2,1,7,0'), (13, 'T3,4,400,D,36,0,2,1'),
           (14, 'T3,5,900,N,4,1,3,1'), (15, 'T3,6,1000,O,39,1,4,1'), (16, 'T3,7,1006,L,37,1,5,1')]
    ev = chk.parse_events(raw)
    assert ev['bad'] == 0
    assert ev['qseq'] == [1, 2, 3, 4, 5, 6, 7]
    assert [r['src'] for r in ev['T3']] == ['M', 'D', 'N', 'O', 'L']
    assert ev['T3'][1] == dict(qseq=4, t_us=400, src='D', gpio=36, lvl=0, gseq=2, run=1, t_rx=13)


@pytest.mark.parametrize('line', ['T3,2,100,X,38,1,1,1', 'T3,2,100,M,38,1,1', 'T3,2,1x0,M,38,1,1,1',
                                  'T3,2,100,MM,38,1,1,1'])
def test_parse_events_t3_malformed_is_bad(line):
    ev = chk.parse_events([(1, line)])
    assert ev['bad'] == 1 and ev['T3'] == [] and ev['qseq'] == []


def test_qseq_stream_counts_t3_lines(tmp_path):
    f = tmp_path / 'esp32_events.txt'
    f.write_text('\n'.join([
        '10\t#run_start,run=1,timestamp_us=5,profile=t3,qseq=1',
        '11\tT3,2,100,M,38,1,1,1',
        '12\tT3,3,400,D,36,0,2,1',
        '13\t#run_end,run=1,ev_qdrop_run=0,enc_ovf_run=0,log_qdrop_run=0,qseq=4',
    ]) + '\n', encoding='utf-8')
    q = stm.qseq_stream(str(f))
    assert q['ok'] and q['missing'] == 0 and q['malformed'] == 0


def test_qseq_stream_bad_t3_is_missing(tmp_path):
    f = tmp_path / 'esp32_events.txt'
    f.write_text('\n'.join([
        '10\t#run_start,run=1,timestamp_us=5,profile=t3,qseq=1',
        '11\tT3,2,100,Q,38,1,1,1',
        '13\t#run_end,run=1,ev_qdrop_run=0,enc_ovf_run=0,log_qdrop_run=0,qseq=3',
    ]) + '\n', encoding='utf-8')
    q = stm.qseq_stream(str(f))
    assert not q['ok'] and q['malformed'] == 1


# ── clock sync ────────────────────────────────────────────────────────────────

def _periodic(n=600, start=50, stop=550, drift=40e-6, lose=()):
    rng = np.random.default_rng(1)
    p = np.arange(n) * 100_000.0 + 5e6 + rng.normal(0, 3, n)
    idx = np.arange(start, stop)
    e = (p[idx] - 123_456.0) / (1 + drift) + rng.normal(0, 5, len(idx))
    rx = p[idx] + rng.uniform(1500, 30000, len(idx))
    keep = np.ones(len(e), bool)
    keep[list(lose)] = False
    return p, e[keep], rx[keep]


def test_clock_sync_periodic_10hz_with_drift():
    """10 Hz marks are periodic (pairing by spacing alone is ambiguous): the t_rx anchor finds the right
    pairs; 40 ppm drift over 50 s is followed."""
    p, e, rx = _periodic()
    s = cs.sync(p, np.full(len(p), 1.0), e, rx)
    assert s['n_pairs'] == len(e) and s['esp_unmatched'] == 0 and s['pi_unmatched_in_span'] == 0
    assert abs(s['rate'] - (1 + 40e-6)) < 2e-6 and s['res_max_us'] < 40
    assert cs.verdict(s) == (0, '')
    pairs = dict(s['_pairs'])
    assert all(pairs[i] == i - 50 for i in range(50, 550))


def test_clock_sync_lost_marks_are_data_loss():
    p, e, rx = _periodic(lose=(10, 300))
    s = cs.sync(p, np.full(len(p), 1.0), e, rx)
    assert s['pi_unmatched_in_span'] == 2 and cs.verdict(s)[0] == 1


def test_clock_sync_no_anchor_is_missing_evidence():
    p, e, rx = _periodic()
    s = cs.sync(p, np.full(len(p), 1.0), e, rx + 10e9)      # logger clock not the Pi's
    assert s['n_pairs'] == 0 and cs.verdict(s) == (3, 'thieu_bang_chung')


# ── a whole simulated T3 run ──────────────────────────────────────────────────

def _make_run(tmp_path, delay=lambda j: 300.0 + 7 * (j % 5), quant_frac=0.5, drop_dbg=(), extra_dbg=(),
              stray_dbg=0, loop=True, gseq_hole=None, profile='t3', pf3_before=1, t3_ovf=0, delay_override=None):
    sched = ts.make_schedule(1)
    write_csv(str(tmp_path / 'T3_schedule.csv'), sched)
    plant = ScheduleOmegaPlant(sched)
    n = plant.total_steps + 20
    mosi = np.zeros((n, FRAME_SIZE), np.uint8)
    marker = np.zeros(n, np.uint8)
    prev = None
    for k in range(n):
        m = encode_state(plant.get_state(), seq=k)
        marker[k] = marker_due('t3', k, prev, m)
        mosi[k] = np.frombuffer(bytes(m), np.uint8)
        prev = m
        plant.step([0] * 4)
    t_send = T0_PI_NS + np.arange(n, dtype=np.int64) * 1_000_000 + 30_000
    p0, p1 = t_send - 25_000, t_send - 23_000
    man = dict(step_first=0, step_last=n - 1, steps=n, period_us=1000, marker_mode='t3',
               marker_frames=int(marker.sum()))
    np.savez(tmp_path / 'frames.npz', t_send_ns=t_send, t_recv_ns=t_send + 400_000, t_pulse0_ns=p0,
             t_pulse1_ns=p1, mosi=mosi, miso=np.zeros((n, FRAME_SIZE), np.uint8), marker=marker,
             manifest=np.array(json.dumps(man)))
    # ESP32 side
    to_esp = lambda t_pi_us: (t_pi_us - OFF_US) / RATE
    ev = []                                            # (t_esp, kind, fields)
    rows = sorted(sched, key=lambda r: r.t0_ms)
    mk = np.nonzero(marker)[0]
    lvl = 0
    for j, k in enumerate(mk):
        t_m = to_esp((p0[k] + p1[k]) / 2000.0) + 4.0
        ev.append((t_m, 'M', 38, 1))
        if j in drop_dbg:
            continue
        d = delay(j) if delay_override is None else delay_override.get(j, delay(j))
        t_d = t_m + d
        lvl ^= 1
        ev.append((t_d, 'D', 36, lvl))
        if j in extra_dbg:
            lvl ^= 1
            ev.append((t_d + 200.0, 'D', 36, lvl))
        r = rows[j]
        # encoder_gen.c _retime: the next edge keeps its old time or moves to now + hp_new, whichever is sooner
        hp = min(ts.hp_ticks(r.w_from), ts.hp_ticks(r.w_to))
        hp_new = ts.hp_ticks(r.w_to)
        a, b = t3s.WHEEL_PINS[r.wheel]
        ev.append((t_d + quant_frac * hp, 'N', a, 1))
        ev.append((t_d + quant_frac * hp + hp_new / 2, 'N', b, 0))
        for other in range(4):                         # other wheels' pins opened by D too
            if other != r.wheel:
                ev.append((t_d + 50.0 + other, 'N', t3s.WHEEL_PINS[other][0], 1))
    t_first = to_esp((p0[mk[0]] + p1[mk[0]]) / 2000.0)
    t_last = to_esp((p0[mk[-1]] + p1[mk[-1]]) / 2000.0)
    for i in range(stray_dbg):
        lvl ^= 1
        ev.append(((rows and t_first) + 30_000.0 + i * 100_000.0, 'D', 36, lvl))
    if loop:
        t = t_first - 500_000.0
        lv = 0
        while t < t_last + 200_000.0:
            lv ^= 1
            ev.append((t, 'O', 39, lv))
            ev.append((t + 6.0 + (int(t) % 3), 'L', 37, lv))
            t += 100_000.0
    t_start = t_first - 900_000.0
    for i in range(pf3_before):
        ev.append((t_start - 1000.0 - i, 'D', 36, 1))
    ev.sort(key=lambda x: x[0])
    lines, q, g = [], 1, 0
    t_rx = lambda te: int(OFF_US + RATE * te + 2500)
    lines.append((t_rx(t_start), f'#run_start,run=1,timestamp_us={int(t_start)},profile={profile},duration_s=20,'
                                 f'auto_end=1,qseq={q}'))
    for (te, src, pin, lv) in ev:
        g += 1
        if gseq_hole is not None and g == gseq_hole:
            g += 1
        q += 1
        # PF3-4 case: a D edge carrying the run id but stamped before #run_start (the tool must drop it)
        run = 1 if te >= t_start or src == 'D' else 0
        lines.append((t_rx(te), f'T3,{q},{int(round(te))},{src},{pin},{lv},{g},{run}'))
    t_end = t_last + 400_000.0
    q += 1
    lines.append((t_rx(t_end), f'#run_end,run=1,timestamp_us={int(t_end)},ev_qdrop_run=0,enc_ovf_run=0,'
                               f'log_qdrop_run=0,qseq={q}'))
    q += 1
    lines.append((t_rx(t_end) + 1, f'#run_stats,run=1,log_send_err_run=0,enc_reads_run=0,t3_ovf_run={t3_ovf},'
                                   f'qseq={q}'))
    with open(tmp_path / 'esp32_events.txt', 'w', encoding='utf-8') as f:
        for t, l in sorted(lines):
            f.write(f'{t}\t{l}\n')
    return tmp_path


def _score(d):
    return t3s.main(['--schedule', str(d / 'T3_schedule.csv'), '--frames', str(d / 'frames.npz'),
                     '--esp32-events', str(d / 'esp32_events.txt'), '--out', str(d / 'out')])


def _summary(d):
    return json.load(open(d / 'out' / 'T3_summary.json', encoding='utf-8'))


def test_t3_pass(tmp_path, capsys):
    d = _make_run(tmp_path)
    assert _score(d) == 0
    s = _summary(d)
    assert s['summary']['measured'] == 100 and s['summary']['stray_dbg'] == 0
    assert s['summary']['dbg_before_run_start'] == 1                 # PF3-4: dropped, counted
    assert abs(s['summary']['delay_max_us'] - (300 + 28)) < 2
    assert s['loop_latency']['n'] >= 10 and s['loop_latency']['u_us'] < 10
    assert s['summary']['per_wheel'] == {'0': 25, '1': 25, '2': 25, '3': 25}
    rows = open(d / 'out' / 'T3_result.csv', encoding='utf-8').read().splitlines()
    assert len(rows) == 101
    assert capsys.readouterr().out.strip().splitlines()[-1] == '#ket_qua,ma=0'


def test_t3_quant_is_first_edge_of_the_wheel_after_d(tmp_path):
    d = _make_run(tmp_path, quant_frac=0.4)
    assert _score(d) == 0
    import csv
    rows = list(csv.DictReader(open(d / 'out' / 'T3_result.csv', encoding='utf-8')))
    for r in rows:
        assert abs(float(r['quant_us']) - 0.4 * min(float(r['hp_new_us']), float(r['hp_old_us']))) < 1.5
        assert int(r['pin_n']) == t3s.WHEEL_PINS[int(r['wheel'])][0]


def test_t3_delay_over_limit_fails(tmp_path):
    d = _make_run(tmp_path, delay_override={42: 1200.0})
    assert _score(d) == 1


def test_t3_delay_near_limit_inconclusive(tmp_path):
    d = _make_run(tmp_path, delay_override={42: 998.0})        # 998 + u (4) > 1000, 998 − u < 1000
    assert _score(d) == 3


def test_t3_missing_dbg_expired_fails(tmp_path):
    d = _make_run(tmp_path, drop_dbg=(17,))
    assert _score(d) == 1
    assert _summary(d)['summary']['expired'] == 1


def test_t3_two_dbg_in_window_fails(tmp_path):
    d = _make_run(tmp_path, extra_dbg=(60,))
    assert _score(d) == 1
    assert _summary(d)['summary']['extra_pulse'] == 1


def test_t3_stray_dbg_fails(tmp_path):
    d = _make_run(tmp_path, stray_dbg=1)
    assert _score(d) == 1
    assert _summary(d)['summary']['stray_dbg'] == 1


def test_t3_no_loop_pairs_missing_evidence(tmp_path, capsys):
    d = _make_run(tmp_path, loop=False)
    assert _score(d) == 3
    assert 'ly_do=thieu_bang_chung' in capsys.readouterr().out


def test_t3_gseq_hole_is_data_loss(tmp_path):
    d = _make_run(tmp_path, gseq_hole=50)
    assert _score(d) == 1


def test_t3_overflow_counter_is_data_loss(tmp_path):
    d = _make_run(tmp_path, t3_ovf=2)
    assert _score(d) == 1


def test_t3_wrong_profile_is_input_error(tmp_path):
    d = _make_run(tmp_path, profile='heavy')
    assert _score(d) == 2


def test_t3_frames_not_following_schedule_is_input_error(tmp_path):
    d = _make_run(tmp_path)
    write_csv(str(d / 'T3_schedule.csv'), ts.make_schedule(2))
    assert _score(d) == 2


# ── hil_bench_session --s06: marker per profile ──────────────────────────────

def test_bench_s06_marker_added_and_checked():
    import hil_bench_session as hb
    plan = hb.plan_runs(['t2=--load-schedule s.csv --steps 70000', 't3=--omega-schedule T3.csv --steps 30000',
                         'x=--steps 70000 --marker t3', 'y=--steps 30000'])
    got = hb.plan_s06(['t2=heavy,60', 't3=t3,20'], plan)
    assert got['t2']['marker'] == '10hz' and got['t2']['add_marker']
    assert got['t3']['marker'] == 't3' and got['t3']['add_marker']
    c = hb.s06_run_cmd('~/r', 't3', '--omega-schedule T3.csv --steps 30000', got['t3'], '10.0.0.2')
    assert 'run_hil.py --omega-schedule T3.csv --steps 30000 --marker t3 --frame-log' in c
    assert '--profile t3' in c
    c = hb.s06_run_cmd('~/r', 't2', '--load-schedule s.csv --steps 70000', got['t2'], '10.0.0.2')
    assert '--marker 10hz --frame-log' in c
    with pytest.raises(ValueError, match='--marker 10hz'):
        hb.plan_s06(['x=heavy,60'], plan)                    # non-t3 profile with --marker t3
    with pytest.raises(ValueError, match='omega-schedule'):
        hb.plan_s06(['y=t3,20'], plan)                       # t3 without the schedule
    explicit = hb.plan_s06(['x=heavy,60'], hb.plan_runs(['x=--steps 70000 --marker 10hz']))
    assert not explicit['x']['add_marker']


# ── existing frame-log readers with the marker array (CODE 17/11 condition) ──

def _node_frames(tmp_path, mode, steps=320):
    import multiprocessing as mp
    from hil_node import HILConfig, HILNode
    from plant_interface import MockPlant
    plant = MockPlant()
    plant.reset([2.0, 2.0, 2.0, 2.0] + [0.0] * 6)
    path = tmp_path / f'frames_{mode or "default"}.npz'
    cfg = HILConfig(simulate_hw=True, max_steps=steps, period_us=1_000, marker_mode=mode, frame_log=str(path))
    HILNode(plant, config=cfg, shared_state=mp.Array('d', 10)).run()
    return path


def _strip_marker(src, dst):
    z = np.load(src)
    man = json.loads(str(z['manifest']))
    man.pop('marker_mode', None)
    man.pop('marker_frames', None)
    arrs = {k: z[k] for k in z.files if k not in ('marker', 'manifest')}
    np.savez(dst, manifest=np.array(json.dumps(man)), **arrs)
    return dst


def _outcome(fn):
    try:
        return ('ok', fn())
    except Exception as ex:                                   # noqa: BLE001 (same outcome either way)
        return ('err', type(ex).__name__, str(ex))


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return a == b


def test_frame_log_readers_unchanged_by_marker_array(tmp_path):
    import types
    import s06_frames_decode as fd
    import hil_edge_map as hem
    import hil_o1_score as hos
    with_m = _node_frames(tmp_path, '10hz')
    z = np.load(with_m)
    assert 'marker' in z.files and int(z['marker'].sum()) == 4            # frames 0, 100, 200, 300
    without = _strip_marker(with_m, tmp_path / 'frames_stripped.npz')
    cal_args = types.SimpleNamespace(d1w_us=None, d1w_span_us=None, delta_h_us=None, ts_us=None)

    def dec(p):
        zz = np.load(p)
        return fd.decode(zz['t_send_ns'], zz['mosi'], zz['miso'], cal=fd.load_cal(zz, cal_args))

    def man_check(p):
        zz = np.load(p)
        return fd.manifest_check(fd.load_manifest(zz), zz['t_send_ns'], zz['mosi'], zz['miso'])

    def o1(p):
        man, mosi, miso = hos.load(p)
        man.pop('marker_mode', None)
        man.pop('marker_frames', None)
        return man, mosi, miso

    for fn in (dec, man_check, hem.load_frames, o1):
        a, b = _outcome(lambda: fn(with_m)), _outcome(lambda: fn(without))
        assert _same(a, b), fn
    # (manifest_check itself may flag t_send here: on Windows time.monotonic_ns is coarse, simulated frames share a
    # time; the point is the same outcome with and without the marker array)
