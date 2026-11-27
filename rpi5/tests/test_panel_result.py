import csv
import json
import math
from pathlib import Path
import sys
import threading

import pytest

import panel_proto as pp
import panel_result as pr


def write_csv(path, fields, rows):
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields.split(','))
        w.writerows(rows)


def test_rotated_square_time_alignment_and_decimation(tmp_path):
    pose, esp = tmp_path / 'pose.csv', tmp_path / 'esp32.csv'
    a = 1.234
    c, s = math.cos(a), math.sin(a)
    corners = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    path = []
    for (x, y), (u, v) in zip(corners, corners[1:]):
        path.extend((x + i*(u-x)/101, y + i*(v-y)/101) for i in range(101))
    path.append(corners[-1])
    rows = [(3, 'ARM', 10, 10, -1)]
    esp_rows = [(10, 0, 99, 99, 99, 99)]
    expected = []
    for i, (x, y) in enumerate(path):
        rows.append((77_000_000_000 + i * 50_000_000, 'RUN', 1.5+c*x-s*y, -2+s*x+c*y, a+i/100))
        esp_rows.append((22_000+i*50_000, 1, x, y, x+0.1, y-0.2))
        expected.append((round(x*1000), round(y*1000)))
    rows.append((rows[-1][0] + 50_000_000, 'STOPPING', 9, 9, a))
    write_csv(pose, 't_ns,phase,x,y,th', rows)
    write_csv(esp, 'timestamp_us,run_active,pos_x_ref,pos_y_ref,pos_x,pos_y', esp_rows)
    tracks, maximum, last = pr.trajectory(pose, pr.read_dut(esp))
    assert maximum < 1 and last < 1
    assert set(tracks) == {pp.TR_PLANT, pp.TR_DUT, pp.TR_REF}
    for series, chunks in tracks.items():
        assert len(chunks) == 10 <= pp.TRACK_CHUNKS_MAX
        assert [len(chunk) for chunk in chunks] == [22]*9 + [2]
        all_xy = [p for chunk in chunks for p in chunk]
        for i, (x, y) in enumerate(all_xy):
            u, v = expected[i*(len(path)-1)//199]
            if series == pp.TR_DUT:
                u, v = u + 100, v - 200
            assert math.hypot(x-u, y-v) < 1
    tracks, maximum, last = pr.trajectory(pose, None)
    assert set(tracks) == {pp.TR_PLANT} and tracks[pp.TR_PLANT][0][0] == (0, 0)
    assert tracks[pp.TR_PLANT][-1][-1] != (0, 0) and maximum is last is None


def test_error_uses_nearest_relative_time_before_decimation(tmp_path):
    pose = tmp_path / 'pose.csv'
    write_csv(pose, 't_ns,phase,x,y,th', [(123, 'RUN', 1, 2, 0), (123+55_000_000, 'RUN', 1.1, 2, 0),
                                           (123+75_000_000, 'RUN', 1.115, 2, 0)])
    dut = dict(timestamp_us=[200, 30200, 70200, 80200], run_active=[1]*4, pos_x_ref=[0, 0, .13, .2],
               pos_y_ref=[0]*4, pos_x=[0]*4, pos_y=[0]*4)
    _, maximum, last = pr.trajectory(pose, dut)
    assert (maximum, last) == pytest.approx((30, 15))
    assert pr.error_line(maximum, last) == 'e_max 30 e_end 15'
    assert pr.error_line(None, None) == 'e_max - e_end -'
    assert pr.error_line(3456, 1234) == 'e_max 3e3 e_end 1234'
    assert pr.timing_line(1999, 139398) == 'p99 1999 max 1e5us'


def decoder_double(monkeypatch, folder, rc, empty=0, crc=0, logger_ready=False):
    # Use the real subprocess boundary, output directory and last-line contract.
    (folder / 'frames.npz').touch()
    problems = (['empty frames'] if empty else []) + (['bad CRC'] if crc else [])
    problems += ['lỗi IMU trong report, thêm chi tiết'] if rc == 1 else []
    conditions = ['missing TICK'] if rc in (1, 3) else []
    summary = dict(frames=dict(empty=2 + empty, empty_in_run=empty, miso_crc_bad=crc),
                   problems=problems, conditions=conditions)
    events = ['--esp32-events', str(folder / 'esp32.csv.events.txt')] if logger_ready else []
    script = ("import json,sys,pathlib\n"
              f"assert sys.argv[4:] == {events!r}\n"
              "d=pathlib.Path(sys.argv[3]); d.mkdir()\n"
              f"(d/'frames_summary.json').write_text({json.dumps(summary)!r},encoding='utf-8')\n"
              f"print('#ket_qua,ma={rc}' + (',ly_do=khong_du_dieu_kien' if {rc}==3 else ''))\n"
              f"sys.exit({rc})")
    def cmd(name, *args):
        assert name == 's06_frames_decode.py' and args == (folder / 'frames.npz', '--out', folder / 'decode.json')
        return [sys.executable, '-c', script, *map(str, args)]
    monkeypatch.setattr(pr, 'tool_cmd', cmd)


@pytest.mark.parametrize('fault,rc,empty,crc,counts,code,text', [
    (False, 0, 0, 0, [4, 4, 4], pp.CHK_CLEAN_NO_SPIFIX, 'chua kiem SPIfix'),
    (False, 1, 0, 0, [4, 4, 4], pp.CHK_BAD, 'loi IMU trong report,'),
    (False, 3, 0, 0, [4, 4, 4], pp.CHK_INCONCLUSIVE, 'missing TICK'),
    (True, 1, 0, 0, [4, 4, 4], pp.CHK_CLEAN_NO_SPIFIX, 'chua kiem SPIfix'),
    (True, 3, 0, 0, [4, 4, 4], pp.CHK_CLEAN_NO_SPIFIX, 'chua kiem SPIfix'),
    (True, 1, 1, 0, [4, 4, 4], pp.CHK_BAD, 'empty_in_run 1'),
    (True, 1, 0, 2, [4, 4, 4], pp.CHK_BAD, 'miso_crc_bad 2'),
    (True, 1, 0, 0, [4, 4, 5], pp.CHK_BAD, 'pos_reset_mid 1'),
    (False, 0, 0, 0, [4, 4, 5], pp.CHK_BAD, 'pos_reset_mid 1'),
    (False, 0, 0, 0, None, pp.CHK_INCONCLUSIVE, 'no pos_reset'),
    (True, 1, 0, 0, None, pp.CHK_INCONCLUSIVE, 'no pos_reset'),
    (True, 0, 0, 0, [4, 4, 0], pp.CHK_INCONCLUSIVE, 'pos_reset_n goes back'),
])
def test_verdict(tmp_path, monkeypatch, fault, rc, empty, crc, counts, code, text):
    decoder_double(monkeypatch, tmp_path, rc, empty, crc, logger_ready=counts is not None)
    dut = dict(run_active=[0, 1, 0], motor_on=[0, 1, 1], pos_reset_n=counts) if counts else None
    check, other = pr.run_check(tmp_path, fault, bool(dut), dut, threading.Event())
    assert check['code'] == code and check['text'].startswith(text)
    assert len(check['text'].encode('ascii')) <= 22
    expected = ['lỗi IMU trong report, thêm chi tiết'] if fault and rc == 1 else []
    if fault and rc in (1, 3):
        expected.append('missing TICK')
    assert other == expected


@pytest.mark.parametrize('script,timeout,reason', [
    ('raise RuntimeError("crash")', 10, 'decoder exit 1'),
    ('import time; time.sleep(60)', .1, 'decoder timeout'),
    ("print('#ket_qua,ma=3')", 10, 'decoder exit 0'),
    ("print('#ket_qua,ma=0')", 10, 'frames_summary.json'),
])
def test_decoder_failure_never_clean(tmp_path, monkeypatch, script, timeout, reason):
    (tmp_path / 'frames.npz').touch()
    monkeypatch.setattr(pr, 'tool_cmd', lambda *args: [sys.executable, '-c', script])
    check, _ = pr.run_check(tmp_path, True, False, None, threading.Event(), timeout)
    assert check['code'] == pp.CHK_INCONCLUSIVE and reason in check['reason']
