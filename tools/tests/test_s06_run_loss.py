"""C2-4: every scoring CLI turns stm.RunLoss (no run selected, a loss counter read) into mã 1, not mã 3.
The first read of each tool raises it, so each `except` is exercised without building a whole run."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path[:0] = [os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'rpi5')]
import s06_frames_decode as fd  # noqa: E402
import s06_stream as stm  # noqa: E402
import s06_t1_check as t1  # noqa: E402
import s06_t4_values as t4  # noqa: E402
import s06_t5_score as t5  # noqa: E402
import s06_t6_check as t6  # noqa: E402
import t3b_score as t3b  # noqa: E402


def boom(*_a, **_k):
    raise stm.RunLoss('hai lượt; bộ đếm mất dữ liệu: run=1 enc_ovf_run=1 (Z19)')


@pytest.mark.parametrize('mod, attr, argv', [
    (t1, (np, 'load'), ['f.npz', '--out', 'o']),
    (t4, (np, 'load'), ['f.npz', 'e.txt', '--out', 'o']),
    (fd, (np, 'load'), ['f.npz', '--out', 'o']),
    (t6, (np, 'load'), ['f.npz', '--esp32-events', 'e.txt', '--out', 'o']),
    (t3b, (t3b.ts, 'read_csv'), ['--schedule', 's.csv', '--frames', 'f.npz', '--esp32-events', 'e.txt',
                                 '--clock-affine', '0', '1', '--delta-sync-us', '5', '--out', 'o']),
    (t5, (t5.t5, 'read_csv'), ['--schedule', 's.csv', '--esp32-events', 'e.txt', '--out', 'o'])],
    ids=['T1', 'T4', 'decode', 'T6', 'T3b', 'T5'])
def test_loss_read_without_a_run_is_code_1(mod, attr, argv, monkeypatch, capsys):
    monkeypatch.setattr(*attr, boom)
    assert mod.main(argv) == 1
    assert capsys.readouterr().out.splitlines()[-1] == '#ket_qua,ma=1'
