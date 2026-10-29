"""step_log.py (S06 T2 per-step timestamps, shared ring) and its use in hil_node.py."""
import multiprocessing as mp
import time

import numpy as np

from hil_node import HILConfig, HILNode
from plant_interface import MockPlant
from step_log import StepLog, StepLogCSV, READ_LAG, N_HEAD, SLOT, C_COMMIT


def _fill(log, steps):
    for k in steps:
        log.record(k, 1_000_000 * k, 1_000_000 * k + 10, 1_000_000 * k + 300)


def test_record_and_drain_keeps_lag_until_final():
    log = StepLog(n_slots=64)
    _fill(log, range(10))
    rows, lost = log.drain()
    assert [r[0] for r in rows] == list(range(10 - READ_LAG)) and lost == 0
    rows, lost = log.drain(final=True)
    assert [r[0] for r in rows] == [8, 9] and lost == 0
    step, ts, t0, td, te = rows[-1]
    assert (ts, t0, td) == (9_000_000, 9_000_010, 9_000_300) and te > 0


def test_overflow_counted_and_not_overwritten():
    log = StepLog(n_slots=8)
    _fill(log, range(20))                     # reader never ran: steps 8..19 do not fit
    assert log.counters()['overflow'] == 12
    rows, lost = log.drain(final=True)
    assert [r[0] for r in rows] == list(range(8)) and lost == 12


def test_reader_frees_slots():
    log = StepLog(n_slots=8)
    for start in range(0, 40, 4):
        _fill(log, range(start, start + 4))
        log.drain()
    rows, lost = log.drain(final=True)
    assert log.counters()['overflow'] == 0


def test_uncommitted_slot_is_not_read():
    log = StepLog(n_slots=16)
    _fill(log, range(4))
    log.arr[N_HEAD + 2 * SLOT + C_COMMIT] = 0      # step 2 half written
    rows, lost = log.drain(final=True)
    assert [r[0] for r in rows] == [0, 1, 3] and lost == 1


def _writer(log, n):
    for k in range(n):
        t = time.monotonic_ns()
        log.record(k, t, t + 1, t + 2)
        time.sleep(0.0002)


def test_across_processes(tmp_path):
    ctx = mp.get_context('spawn')
    log = StepLog(n_slots=256)
    csv = StepLogCSV(log, str(tmp_path / 's.csv'))
    p = ctx.Process(target=_writer, args=(log, 3000))
    p.start()
    while p.is_alive():
        p.join(timeout=0.01)
        csv.drain()
    meta = csv.close()
    assert meta['rows'] == 3000 and meta['overflow'] == 0 and meta['lost_or_torn'] == 0
    a = np.genfromtxt(tmp_path / 's.csv', delimiter=',', names=True)
    assert list(a['step'].astype(int)) == list(range(3000))


def test_hil_node_logs_every_step_and_schedules_from_run(tmp_path):
    log = StepLog(n_slots=1024)
    cfg = HILConfig(simulate_hw=True, max_steps=300, period_us=1_000)
    node = HILNode(MockPlant(), config=cfg, shared_state=mp.Array('d', 10), step_log=log)
    time.sleep(0.05)                          # before Y16 the first 50 deadlines were already past
    t_call = time.monotonic_ns()
    node.run()
    rows, lost = log.drain(final=True)
    assert lost == 0 and [r[0] for r in rows] == list(range(300))
    r = np.array(rows, dtype=np.int64)
    assert np.all(np.diff(r[:, 1]) == 1_000_000)          # t_sched every 1 ms
    # Y16: the schedule starts at run(), not at construction (holds at any clock resolution;
    # lateness itself is not checkable on Windows, monotonic ticks ~15.6 ms there)
    assert r[0, 1] >= t_call
    assert np.all(r[:, 2] >= r[:, 1]) and np.all(r[:, 3] >= r[:, 2]) and np.all(r[:, 4] >= r[:, 3])
