"""tools/s06_load_activity.py on frames written here (MOSI ω, MISO duty, NEW flag, manifest); no hardware."""
import hashlib
import json
import os
import struct
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
import s06_load_activity as la  # noqa: E402
import s06_load_schedule as ls  # noqa: E402

N = 3000


def sched_file(tmp_path):
    p = tmp_path / 's.csv'
    p.write_text('idx,kind,test,start_ms,end_ms\n0,test,A-L1,0,3000\n1,heavy,,1000,2000\n')
    return str(p)


def frames(tmp_path, sched, faults=()):
    mosi = np.zeros((N, 40), dtype=np.uint8)
    miso = np.zeros((N, 40), dtype=np.uint8)
    for k in range(N):
        w = ls.heavy_omegas(k - 1000) if 1000 <= k < 2000 else [0.0, 20.0, 0.1, 20.0]
        if 'wrong_block' in faults and k == 1500:
            w = [1.0] * 4
        mosi[k, :8] = np.frombuffer(struct.pack('>4h', *[round(x * 32767 / 40) for x in w]), dtype=np.uint8)
        mosi[k, 21] = k & 0xFF
        if k % 20 == 0:
            mosi[k, la.OFS_TRAJ_FLAGS] = 3
        d = [0.5, 0.03, 0.0, 1.0] if k < 1500 else [0.03, 0.5, 0.0, 1.0]
        miso[k, :8] = np.frombuffer(struct.pack('>4h', *[round(x * 32767) for x in d]), dtype=np.uint8)
    with open(sched, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    m = dict(step_first=0, step_last=N - 1, steps=N, plant='MecanumPlant', heavy_block=[1000, 2000],
             load_schedule=f'{sched} {sha}')
    if 'other_file' in faults:
        m['load_schedule'] = f'{sched} {"0" * 64}'
    if 'mock' in faults:
        m['plant'] = 'MockPlant'
    if 'block' in faults:
        m['heavy_block'] = [1000, 1999]
    p = str(tmp_path / 'f.npz')
    z = dict(t_send_ns=np.arange(N, dtype=np.int64) * 1_000_000 + 1, mosi=mosi, miso=miso)
    if 'no_manifest' not in faults:
        z['manifest'] = np.array(json.dumps(m))
    np.savez(p, **z)
    return p


def run(tmp_path, faults=()):
    s = sched_file(tmp_path)
    return la.main([frames(tmp_path, s, faults), '--schedule', s, '--out', str(tmp_path / 'o')])


def seg(tmp_path):
    return {r['segment']: r for r in json.load(open(tmp_path / 'o' / 'S06_load_activity.json', encoding='utf-8'))['segments']}


def test_schedule_without_heavy_is_not_a_main_t2_run(tmp_path, capsys):
    # C3-4, CB3 (9): a self-consistent schedule with no heavy block skipped every load check
    s = tmp_path / 'nh.csv'
    s.write_text('idx,kind,test,start_ms,end_ms\n0,test,A-L1,0,3000\n')
    f = frames(tmp_path, str(s))
    assert la.main([f, '--schedule', str(s), '--out', str(tmp_path / 'o')]) == 3
    assert capsys.readouterr().out.splitlines()[-1] == '#ket_qua,ma=3,ly_do=khong_du_dieu_kien'
    assert la.main([f, '--schedule', str(s), '--out', str(tmp_path / 'o'), '--allow-no-heavy']) == 0


def test_constants_match_the_schedule_module():
    """Bản sao hằng số trong công cụ PC phải trùng rpi5/s06_load_schedule.py."""
    assert (la.SWAP_MS, la.W_LOW, la.W_HIGH, la.PHASE_MS) == (ls.SWAP_MS, ls.W_LOW, ls.W_HIGH, ls.PHASE_MS)
    assert la.heavy_omega_table(500).tolist() == [ls.heavy_omegas(k) for k in range(500)]


def test_percentages(tmp_path, capsys):
    assert run(tmp_path) == 0
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=0')
    s = seg(tmp_path)
    assert s['heavy']['frames'] == 1000 and s['nhom_a']['frames'] == 2000
    assert s['heavy']['enc_pct_1'] == 100.0 and s['heavy']['enc_pct_3'] == 100.0
    assert s['nhom_a']['enc_pct_1'] == 0.0 and s['nhom_a']['enc_pct_2'] == 100.0
    assert s['nhom_a']['enc_pct_3'] == 0.0            # 0,1 rad/s ≤ 0,3: no edge (Z1)
    assert s['heavy']['pwm_pct_1'] == 100.0
    assert s['heavy']['pwm_pct_3'] == 0.0 and s['heavy']['pwm_pct_4'] == 0.0      # duty 0 and 1: not chopping
    assert s['heavy']['relay_pct'] == 100.0
    assert s['heavy']['imu_pct'] is None              # no REPORT record in these frames


def test_relay_gap_seen(tmp_path):
    s = sched_file(tmp_path)
    p = frames(tmp_path, s)
    z = dict(np.load(p))
    z['mosi'][1000:1300, la.OFS_TRAJ_FLAGS] = 0
    np.savez(p, **z)
    assert la.main([p, '--schedule', s, '--out', str(tmp_path / 'o')]) == 0
    assert seg(tmp_path)['heavy']['relay_pct'] == pytest.approx(70.0)


@pytest.mark.parametrize('fault', ['wrong_block', 'other_file', 'mock', 'block'])
def test_run_not_on_schedule_fails(tmp_path, capsys, fault):
    assert run(tmp_path, (fault,)) == 1
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=1')


def test_no_manifest_is_missing_evidence(tmp_path, capsys):
    assert run(tmp_path, ('no_manifest',)) == 3
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=3,ly_do=thieu_bang_chung')


def test_frame_lost_from_the_log_fails(tmp_path):
    s = sched_file(tmp_path)
    p = frames(tmp_path, s)
    z = dict(np.load(p))
    for k in ('t_send_ns', 'mosi', 'miso'):
        z[k] = z[k][:-1]
    np.savez(p, **z)
    assert la.main([p, '--schedule', s, '--out', str(tmp_path / 'o')]) == 1


def test_unreadable_input_is_code_2(tmp_path, capsys):
    s = sched_file(tmp_path)
    assert la.main([str(tmp_path / 'none.npz'), '--schedule', s, '--out', str(tmp_path / 'o')]) == 2
    assert '#ket_qua' not in capsys.readouterr().out
