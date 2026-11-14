"""S06 T2 B1: memory of the plant process (report only). R134 package, Liam 14/11 CODE_R134 "duyet goi sua code
fix/r134-answers"; after Codex round 1 R134C-12 (Liam 14/11 "Sua + vong 2 (Recommended)"): the plant process
itself records VmHWM and /proc/meminfo after its loop, before it exits; the main process reads that snapshot."""
import inspect
import json
import multiprocessing as mp
import os
import sys

import pytest

import run_hil
from run_hil import MEM_SNAPSHOT_ENV, mem_report, plant_mem_snapshot

STATUS = 'Name:\tpython3\nVmPeak:\t  900000 kB\nVmHWM:\t   81234 kB\nVmRSS:\t   70000 kB\n'
MEMINFO = 'MemTotal:        8245364 kB\nMemFree:  100 kB\nMemAvailable:    7012345 kB\n'


def _files(tmp_path):
    st, mi = tmp_path / 'status', tmp_path / 'meminfo'
    st.write_text(STATUS, encoding='ascii')
    mi.write_text(MEMINFO, encoding='ascii')
    return str(st), str(mi)


def test_snapshot_then_report(tmp_path):
    st, mi = _files(tmp_path)
    p = str(tmp_path / 'snap.json')
    snap = plant_mem_snapshot(p, status=st, meminfo=mi)
    assert snap['vm_hwm_kib'] == 81234 and snap['pid'] == os.getpid()
    m = mem_report(p, os.getpid())
    assert (m['plant_peak_rss_kib'], m['plant_rss_kib']) == (81234, 70000)
    assert (m['mem_total_kib'], m['mem_available_kib']) == (8245364, 7012345)
    assert m['pid_match'] is True and 'VmHWM' in m['method'] and not os.path.exists(p)   # read once, removed


def test_snapshot_of_another_pid_is_flagged(tmp_path):
    st, mi = _files(tmp_path)
    p = str(tmp_path / 'snap.json')
    plant_mem_snapshot(p, status=st, meminfo=mi)
    assert mem_report(p, os.getpid() + 1)['pid_match'] is False


def test_no_snapshot_gives_none_and_a_note(tmp_path):
    m = mem_report(str(tmp_path / 'none.json'), 123)
    assert m['plant_peak_rss_kib'] is None and m['mem_available_kib'] is None and 'no snapshot' in m['note']


def test_snapshot_never_raises_and_needs_a_path(tmp_path, monkeypatch):
    monkeypatch.delenv(MEM_SNAPSHOT_ENV, raising=False)
    assert plant_mem_snapshot() is None                                # no path: nothing written
    assert plant_mem_snapshot(str(tmp_path / 'no_dir' / 'x.json')) is None    # write error swallowed
    snap = plant_mem_snapshot(str(tmp_path / 'y.json'), status=str(tmp_path / 'none'), meminfo=str(tmp_path / 'none'))
    assert snap['vm_hwm_kib'] is None and snap['mem_available_kib'] is None


def test_called_in_the_plant_process_after_its_loop():
    """R134C-12: in _run_plant, after node.run() returns and before the process can exit (run_failed)."""
    src = inspect.getsource(run_hil._run_plant)
    i, j, k = src.index('node.run()'), src.index('plant_mem_snapshot()'), src.index('if node.run_failed')
    assert i < j < k


def _child(path, mib):
    os.environ[MEM_SNAPSHOT_ENV] = path
    b = bytearray(mib * 1024 * 1024)
    b[::4096] = b'\x01' * len(b[::4096])
    plant_mem_snapshot()


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='/proc only on Linux (the Pi)')
def test_peak_of_the_plant_process_not_of_a_bigger_child(tmp_path):
    """Codex R134C-12 counterexample: a bigger child waited for first must not show as the plant peak."""
    ctx = mp.get_context('fork')
    big = ctx.Process(target=_child, args=(str(tmp_path / 'big.json'), 128))
    big.start()
    big.join()
    small = ctx.Process(target=_child, args=(str(tmp_path / 'small.json'), 32))
    small.start()
    small.join()
    m = mem_report(str(tmp_path / 'small.json'), small.pid)
    assert m['pid_match'] and 32 * 1024 <= m['plant_peak_rss_kib'] < 100 * 1024
    json.dumps(m)
