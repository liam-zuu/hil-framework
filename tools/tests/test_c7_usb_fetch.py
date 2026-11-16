"""tools/c7_usb_fetch.py: new-clip detection, copy checks, manifest and the watch loop, on a fake
phone (no device needed). The real phone path (PowerShell + Shell COM) was tried by hand 13/11
(CODE_CAM): list, get, watch on an iPhone 17 over a short cable."""
import csv
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import c7_usb_fetch as uf  # noqa: E402


def item(name, size, folder='202611_a', ext='.MOV', created='2026-11-13T14:09:02'):
    return {'folder': folder, 'name': name, 'file': name + ext, 'size': size, 'created_utc': created}


class FakePhone:
    """Each list() call pops the next state; copy() writes `size` bytes (or `short` fewer)."""

    def __init__(self, states, folders=('202610_a', '202611_a'), short=0):
        self.states = list(states)
        self.folders = list(folders)
        self.cur = self.states[0]
        self.short = short
        self.copies = []
        self.list_args = []

    def list(self, min_folder=''):
        self.list_args.append(min_folder)
        if min_folder == '~':
            return self.folders, []
        if self.states:
            self.cur = self.states.pop(0)
        if isinstance(self.cur, Exception):
            raise self.cur
        return self.folders, [i for i in self.cur if i['folder'] >= min_folder]

    def copy(self, it, dst_dir, timeout_s=600):
        self.copies.append(it['name'])
        with open(os.path.join(dst_dir, it['file']), 'wb') as f:
            f.write(b'x' * (it['size'] - self.short))
        return {'want': it['size'], 'got': it['size'] - self.short, 'copy_s': 0.5}


def test_is_video():
    assert uf.is_video(item('IMG_1', 5))
    assert uf.is_video(item('IMG_1', 5, ext='.mp4'))
    assert not uf.is_video(item('IMG_1', 5, ext='.HEIC'))
    assert not uf.is_video({'folder': 'a', 'name': 'x', 'file': '', 'size': 1})


def test_tracker_ignores_baseline_and_waits_for_stable_size():
    old = item('IMG_3543', 100)
    tr = uf.NewClipTracker([old])
    assert tr.poll([old, item('IMG_3544', 50)]) == []          # first sight
    assert tr.poll([old, item('IMG_3544', 80)]) == []          # still growing
    got = tr.poll([old, item('IMG_3544', 80)])                 # same size twice -> ready
    assert [g['name'] for g in got] == ['IMG_3544']
    assert tr.poll([old, item('IMG_3544', 80)]) == []          # taken once only


def test_tracker_skips_photos_and_zero_size():
    tr = uf.NewClipTracker([])
    for _ in range(3):
        assert tr.poll([item('IMG_9', 10, ext='.HEIC'), item('IMG_8', 0)]) == []


def test_tracker_new_month_folder():
    tr = uf.NewClipTracker([item('IMG_1', 5)])
    nxt = item('IMG_2', 7, folder='202612_a')
    tr.poll([nxt])
    assert [g['folder'] for g in tr.poll([nxt])] == ['202612_a']


def test_dest_name_and_index(tmp_path):
    assert uf.dest_name('run', 3, item('IMG_3544', 1)) == 'run03_IMG_3544.MOV'
    assert uf.next_index(str(tmp_path)) == 1
    uf.append_manifest(str(tmp_path), {'n': 1})
    uf.append_manifest(str(tmp_path), {'n': 4})
    assert uf.next_index(str(tmp_path)) == 5


def test_fetch_one_hash_and_rename(tmp_path):
    ph = FakePhone([[item('IMG_1', 1000)]])
    row = uf.fetch_one(ph, item('IMG_1', 1000), str(tmp_path), 1)
    dest = tmp_path / 'run01_IMG_1.MOV'
    assert dest.exists() and dest.stat().st_size == 1000
    assert row['bytes'] == 1000 and row['sha256'] == uf.sha256(str(dest))
    assert not (tmp_path / '.incoming' / 'IMG_1.MOV').exists()


def test_fetch_one_rejects_short_copy(tmp_path):
    ph = FakePhone([[item('IMG_1', 1000)]], short=10)
    with pytest.raises(OSError, match='incomplete'):
        uf.fetch_one(ph, item('IMG_1', 1000), str(tmp_path), 1)
    assert not (tmp_path / 'run01_IMG_1.MOV').exists()


def test_fetch_one_never_overwrites(tmp_path):
    (tmp_path / 'run01_IMG_1.MOV').write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        uf.fetch_one(FakePhone([[]]), item('IMG_1', 10), str(tmp_path), 1)
    assert (tmp_path / 'run01_IMG_1.MOV').read_bytes() == b'keep'


def test_watch_end_to_end(tmp_path):
    base = [item('IMG_3543', 100)]
    a = item('IMG_3544', 300, created='2026-11-13T14:20:00')
    b = item('IMG_3545', 200)
    states = [base,                       # baseline
              base,                       # nothing new
              base + [a],                 # a appears
              base + [a],                 # a stable -> copy
              DeviceErrorState(),         # phone briefly unreadable
              base + [a, b],
              base + [a, b]]              # b stable -> copy
    ph = FakePhone(states)
    calls = []

    def measure(video, session, tag_offset, out_dir):
        calls.append((os.path.basename(video), session, tag_offset))
        return 0, '#ket_qua,ma=0'

    logs = []
    done = uf.watch(ph, str(tmp_path), session='S.json', tag_offset='0,0,0', poll_s=0,
                    max_clips=2, sleep=lambda s: None, measure=measure, log=logs.append)
    assert [d['dest'] for d in done] == ['run01_IMG_3544.MOV', 'run02_IMG_3545.MOV']
    assert ph.copies == ['IMG_3544', 'IMG_3545']
    assert calls == [('run01_IMG_3544.MOV', 'S.json', '0,0,0'), ('run02_IMG_3545.MOV', 'S.json', '0,0,0')]
    assert ph.list_args[0] == '~' and set(ph.list_args[1:]) == {'202611_a'}
    with open(tmp_path / uf.MANIFEST, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert [r['src_name'] for r in rows] == ['IMG_3544', 'IMG_3545']
    assert rows[0]['created_utc'] == '2026-11-13T14:20:00' and rows[0]['measure_rc'] == '0'
    assert any('chua doc duoc' in s for s in logs)


def test_watch_without_session_does_not_measure(tmp_path):
    a = item('IMG_2', 50)
    ph = FakePhone([[], [a], [a]])
    done = uf.watch(ph, str(tmp_path), poll_s=0, max_clips=1, sleep=lambda s: None,
                    measure=lambda *x: pytest.fail('measured without session'), log=lambda s: None)
    assert 'measure_rc' not in done[0]


def test_watch_stops_after_max_polls(tmp_path):
    ph = FakePhone([[]] * 5)
    assert uf.watch(ph, str(tmp_path), poll_s=0, sleep=lambda s: None, log=lambda s: None,
                    max_polls=3) == []


def DeviceErrorState():
    return uf.DeviceError('LOCKED_OR_UNTRUSTED')


def test_find_clip_newest_folder_first():
    ph = FakePhone([[item('IMG_1', 5, folder='202610_a'), item('IMG_2', 5)]] * 3)
    assert uf.find_clip(ph, 'IMG_2')['folder'] == '202611_a'
    assert ph.list_args == ['~', '202611_a']
    assert uf.find_clip(ph, 'IMG_1')['folder'] == '202610_a'
    assert uf.find_clip(ph, 'IMG_9') is None


# ---------------------------------------------------------------- frame check
def sess_of(centres, w=1280, h=720):
    return {'width': w, 'height': h, 'ref_centres_px': {str(i): list(c) for i, c in centres.items()}}


def grid(n, dx=0.0, dy=0.0):
    return {i: (100.0 + 110 * ((i - 1) % 10) + dx, 120.0 + 200 * ((i - 1) // 10) + dy) for i in range(1, n + 1)}


def test_check_marks_without_session():
    assert uf.check_marks(grid(8), (720, 1280))['verdict'] == 'OK'
    r = uf.check_marks(grid(7), (720, 1280))
    assert r['verdict'] == 'CANH_BAO' and r['n_floor'] == 7
    r = uf.check_marks({**grid(9), 0: (5, 5), 25: (6, 6)}, (720, 1280))
    assert r['car_seen'] and r['n_check'] == 1 and r['n_floor'] == 9


def test_check_marks_with_session():
    s = sess_of(grid(10))
    r = uf.check_marks(grid(10, dx=2.0), (720, 1280), s)
    assert r['verdict'] == 'OK' and abs(r['shift_px'] - 2.0) < 1e-9
    r = uf.check_marks(grid(10, dx=6.0), (720, 1280), s)
    assert r['verdict'] == 'CANH_BAO' and 'xe dich' in r['reasons'][0]
    seen = grid(10)
    del seen[3]
    r = uf.check_marks(seen, (720, 1280), s)
    assert r['verdict'] == 'CANH_BAO' and r['missing'] == [3]
    assert uf.check_marks(grid(10), (1080, 1920), s)['verdict'] == 'CANH_BAO'
    assert uf.check_marks({}, (720, 1280), s)['shift_px'] is None


def make_clip(path, centres, side=70, w=1280, h=720, frames=3, extra=None):
    import cv2
    import numpy as np
    img = np.full((h, w), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for i, (x, y) in {**centres, **(extra or {})}.items():
        m = cv2.aruco.generateImageMarker(d, i, side)
        x0, y0 = int(round(x - side / 2)), int(round(y - side / 2))
        img[y0:y0 + side, x0:x0 + side] = m
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (w, h))
    assert vw.isOpened()
    for _ in range(frames):
        vw.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    vw.release()
    return str(path)


def test_frame_check_on_video(tmp_path):
    pytest.importorskip('cv2')
    ref = grid(10)
    s = sess_of(ref)
    v = make_clip(tmp_path / 'run01_IMG_1.avi', ref, extra={0: (640, 600), 25: (900, 600)})
    r = uf.frame_check(v, s)
    assert r['verdict'] == 'OK', r
    assert r['n_floor'] == 10 and r['car_seen'] and r['n_check'] == 1 and r['shift_px'] < 1.0
    assert (tmp_path / 'run01_IMG_1_khung.jpg').exists() and (tmp_path / 'run01_IMG_1_khung.json').exists()
    moved = make_clip(tmp_path / 'run02_IMG_2.avi', grid(10, dx=15, dy=-8))
    r = uf.frame_check(moved, s)
    assert r['verdict'] == 'CANH_BAO' and r['shift_px'] > 15
    few = make_clip(tmp_path / 'run03_IMG_3.avi', grid(5))
    assert uf.frame_check(few)['verdict'] == 'CANH_BAO'


def test_frame_check_session_path(tmp_path):
    pytest.importorskip('cv2')
    import json
    ref = grid(10)
    p = tmp_path / 'session.json'
    p.write_text(json.dumps(sess_of(ref)), encoding='utf-8')
    v = make_clip(tmp_path / 'a.avi', ref)
    assert uf.frame_check(v, str(p))['verdict'] == 'OK'


def test_handle_records_frame_check(tmp_path):
    a = item('IMG_2', 50)
    logs = []
    calls = []

    def chk(video, session, out_dir):
        calls.append((os.path.basename(video), session))
        return {'verdict': 'CANH_BAO', 'n_floor': 6, 'n_check': 0, 'car_seen': False, 'shift_px': None,
                'reasons': ['chi thay 6 moc san (< 8)']}
    row = uf.handle(FakePhone([[a]]), a, str(tmp_path), 'run', None, None, log=logs.append, check=chk)
    assert row['frame_check'] == 'CANH_BAO' and calls == [('run01_IMG_2.MOV', None)]
    assert any('!! chi thay 6' in s for s in logs)

    def bad(video, session, out_dir):
        raise OSError('OpenCV cannot open')
    b = item('IMG_3', 40)
    row = uf.handle(FakePhone([[b]]), b, str(tmp_path), 'run', None, None, log=logs.append, check=bad)
    assert row['frame_check'] == 'LOI'
    with open(tmp_path / uf.MANIFEST, newline='', encoding='utf-8') as f:
        assert [r['frame_check'] for r in csv.DictReader(f)] == ['CANH_BAO', 'LOI']
