"""Design 5 and K2/K3 checks on synthetic images, cameras, detections and scoring files."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from concurrent.futures import Future

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import c7_clip_probe as cp
import c7_measure as cm
import c7_nhanh as fast
import c7_score
import c8b_gio
from test_c7_33_moc import CAM, OFFSET, car_px, dets, run_frames, sess
from test_c7_measure import car_texture, floor_texture, render_car, render_floor, write_video


def tagged(x, y, marker=0):
    gray = np.full((250, 340), 255, np.uint8)
    gray[y:y + 60, x:x + 60] = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cp.DICT), marker, 60)
    return gray


@pytest.mark.parametrize('x,y', [(7, 7), (273, 7), (7, 183), (273, 183), (7, 100), (273, 100),
                                  (120, 7), (120, 183), (110, 111), (115, 118)])
def test_clipped_origin_and_integer_translations(x, y):
    det = cm.make_detector()
    gray = tagged(x, y)
    car, reason = fast.crop_car(det, gray, (x + 31.2, y + 29.7), 80)
    assert reason is None
    # design 2.1.5 v1.2: ArUco on the crop, refinement on the full frame -> exactly the full-chain corners
    np.testing.assert_array_equal(car, cm.detect_refined(det, gray)[0])
    base, _ = fast.crop_car(det, tagged(110, 111), (141.2, 140.7), 80)
    np.testing.assert_array_equal(car, base + [x - 110, y - 111])


def test_prediction():
    p, q = np.array([36., 22.]), np.array([32., 25.])
    assert fast.prediction(None, None) is None
    for old, new, expected in [(None, p, p), (p, None, p), (p, q, [28., 28.])]:
        np.testing.assert_array_equal(fast.prediction(old, new), expected)


class SyntheticPool:
    def __init__(self, detect):
        self.detect = detect
        self.calls = []

    def submit(self, func, gray):
        assert func is fast._full_detection
        f = Future()
        self.calls.append(int(gray[0, 0]))
        f.set_result((self.detect(gray), 0.))
        return f


@pytest.mark.parametrize('mode,reason', [('margin', 'sat_mep'), ('floor', 'khong_thay')])
def test_crop_rejection_really_uses_full_detection(monkeypatch, sess, mode, reason):
    det = cm.make_detector()
    first = tagged(70, 90)
    second = tagged(94 if mode == 'margin' else 220, 90)
    if mode == 'floor':
        second[90:150, 70:130] = tagged(70, 90, marker=3)[90:150, 70:130]
    # First centre is (99.5,119.5). The right crop edge is column 159.
    car, why = fast.crop_car(det, second, (99.5, 119.5), 60)
    assert car is None and why == reason
    images = [first, second, second]
    monkeypatch.setattr(cm, 'iter_frames', lambda *args: ((i, i / 60, g) for i, g in enumerate(images)))
    scan = fast.DetectionPass(SyntheticPool(lambda g: cm.detect_refined(det, g)), sess, OFFSET, 60, 0.25)
    frames = scan.run('synthetic', 30, None)
    assert scan.fallback == {reason: 1}
    assert set(scan.fits) == {0, 1, 2}
    np.testing.assert_array_equal(frames[1][2][0], cm.detect_refined(det, second)[0])
    assert scan.check['n_co_xe'] == scan.check['n'] == 3


def translated_camera():
    return cp.Camera(CAM.model, CAM.f, CAM.cx, CAM.cy, CAM.k, CAM.rvec, CAM.tvec + [0.005, 0, 0])


@pytest.mark.parametrize('mode,expected', [('still', {0, 8, 16}), ('step', {0, 4, 5, 6, 8, 16}),
                                           ('untrusted', {0, 4, 6, 7, 8, 9, 10, 12, 16})])
def test_full_detection_schedule_and_one_decode(monkeypatch, sess, mode, expected):
    # A step first present at frame 5 must be bracketed by full frames 4 and 5.
    car = car_px(0.1, 0.2, 0.5)
    before, after = dets(), dets(translated_camera())

    def detect(g):
        i = int(g[0, 0])
        d = after if mode == 'step' and i >= 5 else before
        if mode == 'untrusted' and i == 8:
            d = {j: d[j] for j in (1, 2, 3)}
        return {**d, 0: car}

    passes = []

    def decode(video, limit):
        passes.append(video)
        for i in range(17):
            yield i, i / 60, np.full((4, 4), i, np.uint8)

    monkeypatch.setattr(cm, 'iter_frames', decode)
    pool = SyntheticPool(detect)
    scan = fast.DetectionPass(pool, sess, OFFSET, 150, 0.25)
    monkeypatch.setattr(scan, '_crop', lambda gray, pred: (car, None) if pred is not None else (None, 'chua_du_doan'))
    frames = scan.run('synthetic', 8, None)
    assert passes == ['synthetic']
    assert set(scan.fits) == expected == set(pool.calls)
    assert len(pool.calls) == len(expected)
    assert [t for i, t, _ in frames] == [i / 60 for i in range(17)]
    assert not scan.fallback
    assert scan.check['n'] == len(expected)
    assert scan.dense == expected - {0, 8, 16}
    rows, _ = fast.build_rows(frames, expected, sess, OFFSET, cache=scan.cache)
    for row in rows:
        k = row['idx']
        source = min(expected, key=lambda j: (abs(j - k), j))
        assert row['cam_tu_khung'] == source
        assert (row['xe_nguon'] == 'ca_khung') == (k in expected)
        assert ('kc_ok_0' in row) == (k in expected)
        assert row['shift_px'] == rows[source]['shift_px']


def test_camera_delta_without_any_car_and_bad_fit_flag(sess):
    ctx = cm.ChainCtx(sess)
    args = fast.pose_args(sess, OFFSET)
    obs = {0: dets(), 1: {}, 2: dets(translated_camera())}
    fits = {i: ctx.pose(cm.CHAIN_XE, obs[i], obs[i]) for i in (0, 2)}
    assert fast.needs_split(0, 2, fits, obs, ctx, args, 0.25)
    fits[2] = fits[0]
    assert not fast.needs_split(0, 2, fits, obs, ctx, args, 0.25)
    cam, rms, _, _, cond = fits[0]
    fits[2] = cam, rms, False, 'dieu_kien', cond
    assert fast.needs_split(0, 2, fits, obs, ctx, args, 0.25)


def test_n_one_equals_unscheduled_track_chains(sess):
    frames = run_frames(18, fps=60, car=(0.2, -0.3, 0.1))
    frames[2][2].pop(0)
    frames[6][2].clear()
    frames[11][2][0] += [0.24, -0.6]
    gates = dict(cm.DEFAULT_GATES, scale_min=5)
    off = (0.004, -0.005, math.radians(3))
    rows, info = fast.build_rows(frames, set(range(18)), sess, off, gates, video_name='synthetic', video_sha='V')
    expected, _, expected_info = cm.track_chains(frames, 'synthetic', 'V', sess, off, gates)
    assert cm._json(info) == cm._json(expected_info)
    for actual, want in zip(rows, expected, strict=True):
        assert actual.pop('cam_tu_khung') == want['idx']
        assert actual.pop('xe_nguon') == 'ca_khung'
        assert cm._json(actual) == cm._json(want)
        assert {k: type(v) for k, v in actual.items()} == {k: type(v) for k, v in want.items()}
    assert 0 < info['n_valid'] < len(frames)


def test_track_cli_job_order_and_downstream(tmp_path, sess, capsys):
    base = render_floor(CAM, floor_texture(check=True))
    car = car_texture()
    images = []
    for i in range(13):
        gray = base.copy()
        if i != 2:  # Force a fallback while the right endpoint is already in the pool.
            render_car(gray, CAM, car, (-0.3 + i * 0.004, 0.1, 0.2))
        images.append(gray)
    video, session, hs = (tmp_path / n for n in ('shared.avi', 'session.json', 'constants.json'))
    write_video(video, images)
    fast.write_json(session, sess)
    gates = dict(cm.DEFAULT_GATES, scale_min=4)
    fast.write_json(hs, dict(cong_khung=gates, trang_thai='de_xuat'))
    args = ['track', '--video', str(video), '--session', str(session), '--hang-so', str(hs), '--n-thua', '4',
            '--eps-cam-mm', '3', '--max-frames', '11', '--tag-offset', '0.003,-0.002,0.8']
    for jobs in (1, 2):
        out = tmp_path / str(jobs)
        assert fast.main(args + ['--jobs', str(jobs), '--out', str(out)]) == 0
        assert capsys.readouterr().out.splitlines()[-1] == '#ket_qua,ma=0'
        fp, dp = out / 'frames.csv', out / 'dets_nhanh.npz'
        rows = cm.read_frames(fp)
        info = c7_score._track_beside(fp)
        gate_check = c7_score.check_track_gates(fast.read_json(hs), info, cm.sha256(fp))
        assert gate_check['checked'] and gate_check['frames_sha256_match']
        c7_score.bind_session(dict(session_sha256=cm.sha256(session)), info)
        c7_score.bind_gates(dict(cong_khung=info['gates']), dict(cong_khung=gates), info, cm.sha256(fp))
        assert info['thu'] and info['cham_lai'] == 1 and info['lich_sha256'] is None
        assert info['tag_offset'] == list(OFFSET) and info['hang_so_sha256'] == cm.sha256(hs)
        assert info['moc_loai'] == [] and info['duong'] == 'nhanh_v1'
        assert info['n_valid'] > 0
        assert list(c8b_gio.read_frames(fp)['valid']) == [r['valid'] for r in rows]
        ds, sha, shape = cm.load_dets(dp)
        assert sha == info['video_sha256'] == cm.sha256(video)
        assert shape == base.shape
        assert cm.sha256(dp) == info['dets_sha256'] and Path(info['dets_file']) == dp
        with np.load(dp, allow_pickle=False) as z:
            full = set(z['ca_khung'])
        assert full == {0, 2, 4, 8, 10}
        assert info['n_ca_khung'] == len(full) and info['n_xe_do_ca_khung'] == 1
        assert info['kiem_vung_cat']['n'] == len(full) - 1
        assert [r['idx'] for r in rows] == list(range(11))
        for (i, t, d), row in zip(ds, rows):
            assert abs(t - row['t_s']) <= 0.5e-6
            assert int(row['cam_tu_khung']) == min(full, key=lambda j: (abs(j - i), j))
            assert bool(row['kc_n_0']) == (i in full)
            assert (len(d) > 1) == (i in full)
            assert (row['xe_nguon'] == 'vung_cat') == (i not in full)
    assert (tmp_path / '1/frames.csv').read_bytes() == (tmp_path / '2/frames.csv').read_bytes()
    existing = tmp_path / 'empty'
    existing.mkdir()
    assert fast.main(args + ['--out', str(existing)]) == 2
    output = capsys.readouterr()
    assert 'LOI:' in output.err and output.out.splitlines()[-1] == '#ket_qua,ma=2'
    assert not list(existing.iterdir())


# Trimmed from run02 / C8-L-F08-1 / mot_nut / nut_010.json. The run and gate outcomes differ.
NODE = dict(outcome='hong_he_do', gio_c8b=dict(nut=2), gate_v3=dict(
    outcome='khong_dat', u_p=0.04243766250217123, u_psi_deg=3.197749408164356,
    g1=dict(q='true', state='dat', t_done=[11.483937780725682, 11.483937780725682]),
    g2=dict(q=0.08444631540799737, lo=0.03826116042465006, hi=0.15748391005456897, state='chua_ket_luan'),
    g3=dict(q=0.09100499546974757, lo=0.009786671789513451, hi=0.18732951241534113, state='chua_ket_luan'),
    g4=dict(q=0.12536783565092294, lo=0.0725039610320882, hi=0.32187370812253857, state='chua_ket_luan'),
    g5=dict(q=23.792549302558406, lo=16.34268283893251, hi=31.2424157661843, state='khong_dat')))


def verdict_of(node):
    """The ket_luan_nhanh.json fields compare_verdict reads (node number: CN-7)."""
    return dict(fast.node_verdict(node), nut=node['gio_c8b']['nut'])


def f8_result(cmd, reasons):
    """hieu_chuan writes its out directory and hieu_chuan_neo.json, then exits 1 when the state is not dat."""
    out = Path(cmd[cmd.index('--out') + 1])
    out.mkdir(parents=True)
    fast.write_json(out / 'hieu_chuan_neo.json', dict(trang_thai='chua_ket_luan', ly_do=reasons))


def full_chain(folder, rows, ds):
    """frames.csv + dets.npz of the full chain and the track.json recording their sha256 (CN-6 round 2)."""
    folder.mkdir()
    cm.write_frames(folder / 'frames.csv', rows)
    cm.save_dets(folder / 'dets.npz', ds, 'V', (2, 2))
    fast.write_json(folder / 'track.json', dict(frames_sha256=cm.sha256(folder / 'frames.csv'),
                                                dets_sha256=cm.sha256(folder / 'dets.npz')))
    return ['--day', str(folder / 'frames.csv'), '--dets', str(folder / 'dets.npz')]


def track_fixture(frames, dets, cfg_dir, **change):
    """frames.csv, its dets and a track.json recording the clip, session, frames and dets sha256 (CN-4)."""
    for p in (frames, dets):
        p.write_text('fixture ' + p.name)
    info = dict(video_sha256=cm.sha256(cfg_dir / 'clip'), session_sha256=cm.sha256(cfg_dir / 'session'),
                frames_sha256=cm.sha256(frames), dets_sha256=cm.sha256(dets))
    fast.write_json(frames.with_name('track.json'), dict(info, **change))


def config(tmp_path):
    cfg = dict(task='O1-F08', luot_id='test 1', f8=dict(neo='A:1,2,16,16;B:21,22,16,16', mau_n=500, tam_id='tam_c8_1', mau_clip='clip_h'))
    (tmp_path / 'clip_h').write_text('clip_h')
    for k in fast.CONFIG_FILES:
        (tmp_path / k).write_text(k)
        cfg[k] = k
    return cfg


@pytest.mark.parametrize('key,bad', [('session', 'missing.json'), ('task', 'unknown'), ('sender', ''),
                                     ('f8', None), ('neo', 'A:1,2,3,4'), ('mau_n', True), ('mau_n', -1),
                                     ('tam_id', ''), ('luot_id', None), ('mau_clip', ''), ('mau_clip', 'missing_clip_h')])
def test_config_validation(tmp_path, key, bad):
    cfg = config(tmp_path)
    (cfg['f8'] if key in cfg['f8'] else cfg)[key] = bad
    path = tmp_path / 'config.json'
    fast.write_json(path, cfg)
    with pytest.raises(ValueError):
        fast.load_config(path)


@pytest.mark.parametrize('grid,failed', [(False, False), (True, False), (True, True)])
def test_cham_steps_and_verdict(tmp_path, monkeypatch, grid, failed):
    cfg = config(tmp_path)
    conf, frames, out = [tmp_path / n for n in ('config.json', 'frames.csv', 'scored')]
    fast.write_json(conf, cfg)
    track_fixture(frames, frames.with_name('dets_nhanh.npz'), tmp_path)
    calls = []

    def run(cmd, cwd, stdin, stdout, stderr):
        calls.append(cmd)
        step = len(calls)
        assert cmd[:5] == [sys.executable, '-X', 'utf8', '-I', '-B']
        assert cwd == fast.HERE.parent and stdin == subprocess.DEVNULL and stderr == subprocess.STDOUT
        stdout.write(b'stdout')
        if step == 1:
            fast.write_json(cmd[-1], dict(nodes_s=[-1, 0, 1, 2]))
        if step == 2:
            assert Path(cmd[cmd.index('--dets') + 1]) == frames.with_name('dets_nhanh.npz')
        if step == 3:
            j = cmd.index('--mau')
            assert cmd[j + 1:j + 4] == [str(tmp_path / 'clip_h'), str(tmp_path / 'session'), '500']   # clip H, not the run clip
            f8_result(cmd, ['thieu_neo'])
        if step == 6:
            assert cmd[cmd.index('--nut') + 1] == '2'
            fast.write_json(cmd[cmd.index('--out') + 1], NODE)
        if step == 8:
            assert '--scorer-arg=--tuong-doi' in cmd
            assert '--scorer-arg=' + str(tmp_path / 'khoa_gop') in cmd
        return subprocess.CompletedProcess(cmd, 1 if step == 3 else 2 if failed and step == 4 else 0)

    monkeypatch.setattr(subprocess, 'run', run)
    args = ['cham', '--cau-hinh', str(conf), '--frames', str(frames), '--out', str(out)]
    assert fast.main(args + (['--luoi'] if grid else [])) == (2 if failed else 0)
    result = fast.read_json(out / 'ket_luan_nhanh.json')
    assert len(calls) == len(result['buoc']) == (4 if failed else 8 if grid else 6)
    assert result['buoc']['3_f8_hieu_chuan']['ma'] == 1
    assert result['nhan'] == 'duong_nhanh_v1' and result['sha256']['clip'] == cm.sha256(tmp_path / 'clip')
    for name, rec in result['buoc'].items():
        assert rec['thoi_gian_s'] >= 0 and (out / (name + '.txt')).read_bytes() == b'stdout'
    if failed:
        assert result['buoc']['4_f8_lan_lap']['ma'] == 2 and 'loi' in result
    else:
        assert result['outcome'] == 'hong_he_do'
        assert result['sha256']['mot_nut'] == cm.sha256(out / 'mot_nut/nut.json')
        assert result['g1'] == dict(t_done=NODE['gate_v3']['g1']['t_done'], state='dat', q='true')   # CN-11
        for g in fast.GATES[1:]:
            assert result[g] == NODE['gate_v3'][g]


@pytest.mark.parametrize('nut,ok', [(1, True), (0, True), (4, False), (-1, False)])
def test_cham_node_and_dets_options(tmp_path, monkeypatch, nut, ok):
    """Design 2.2 step 5 (--nut, K3b) and --dets (K3a: frames of the full chain with their own dets.npz)."""
    cfg = config(tmp_path)
    conf, frames, out, other = [tmp_path / n for n in ('config.json', 'frames.csv', 'scored', 'dets.npz')]
    fast.write_json(conf, cfg)
    track_fixture(frames, other, tmp_path)
    calls = []

    def run(cmd, cwd, stdin, stdout, stderr):
        calls.append(cmd)
        if len(calls) == 3:
            f8_result(cmd, ['thieu_neo'])
        if len(calls) == 1:
            fast.write_json(cmd[-1], dict(nodes_s=[-1, 0, 1, 2]))
        if len(calls) == 2:
            assert Path(cmd[cmd.index('--dets') + 1]) == other
        if len(calls) == 6:
            assert cmd[cmd.index('--nut') + 1] == str(nut)
            fast.write_json(cmd[cmd.index('--out') + 1], NODE)
        return subprocess.CompletedProcess(cmd, 1 if len(calls) == 3 else 0)

    monkeypatch.setattr(subprocess, 'run', run)
    args = ['cham', '--cau-hinh', str(conf), '--frames', str(frames), '--out', str(out), '--dets', str(other),
            '--nut', str(nut)]
    assert fast.main(args) == (0 if ok else 2)
    result = fast.read_json(out / 'ket_luan_nhanh.json')
    assert len(calls) == (6 if ok else 5)
    assert result['sha256']['dets'] == cm.sha256(other)
    if ok:
        assert result['nut'] == nut and result['outcome'] == 'hong_he_do'
    else:
        assert 'nut' not in result and '--nut' in result['loi']


T60 = [i / 60 for i in range(2000)]


def test_k3_each_quantity_and_nulls():
    verdict = verdict_of(NODE)
    assert fast._compare_verdict(verdict, NODE, T60)['dat']
    verdict['outcome'] = 'dat'
    verdict['g1']['t_done'] = [NODE['gate_v3']['g1']['t_done'][0], None]
    for g in fast.GATES[1:]:
        verdict[g] = dict(state='dat', q=10., lo=10., hi=10.)
    result = fast._compare_verdict(verdict, NODE, T60)
    assert not result['dat'] and not result['outcome']['dat']
    assert result['g1_t_done_1_s']['dat'] and not result['g1_t_done_2_s']['dat']
    assert result['g1_t_done_2_s']['nguong'] == pytest.approx(1 / 60)
    for g in fast.GATES[1:]:
        assert not result[g + '_state']['dat']
        unit, uncertainty = ('deg', 3.197749408164356) if g == 'g5' else ('m', 0.04243766250217123)
        for k in ('q', 'lo', 'hi'):
            entry = result[f'{g}_{k}_{unit}']
            assert not entry['dat'] and entry['nguong'] == 0.05 * uncertainty
            assert entry['gia_tri'] == abs(10 - NODE['gate_v3'][g][k])
    lost = dict(outcome='hong_he_do', gate_v3=None)          # c7_score.lost_output: no gio_c8b, no node number
    lost_verdict = dict(fast.node_verdict(lost), nut=2)
    assert fast._compare_verdict(lost_verdict, lost, T60)['nut']['nguong'] == 'nut_mat_he_do_khong_ghi_so'
    assert fast._compare_verdict(lost_verdict, lost, T60)['dat']
    with pytest.raises(ValueError, match='node numbers'):
        fast._compare_verdict(lost_verdict, dict(lost, gio_c8b=dict(nut=5)), T60)   # a lost node that has one: same
    lost = dict(lost, gio_c8b=dict(nut=2))
    assert fast._compare_verdict(verdict_of(lost), lost, T60)['dat']
    assert fast._compare_verdict(verdict_of(lost), lost, None)['dat']          # CN-10: no rows.csv needed
    with pytest.raises(ValueError):
        fast._compare_verdict(verdict_of(NODE), NODE, None)                      # a reference t_done needs them
    assert not fast._compare_verdict(verdict, lost, T60)['dat']


def test_so_trial_domain_corner_on_invalid_frame_and_k3(tmp_path):
    n = 10
    base = [dict(idx=i, t_s=i / 60, n_marks=12, why=[], car_seen=True, valid=True,
                 x=1., y=0., psi=math.radians(179.99)) for i in range(n + 2)]
    quick = copy.deepcopy(base[:n])
    # Alternate which side sees a valid frame. All seven belong to one disagreement run.
    for i in range(1, 8):
        (base if i % 2 else quick)[i]['valid'] = False
    for rows in (base, quick):
        rows[8]['valid'] = False
    quick[0].update(x=1.0003, y=0.0004, psi=math.radians(-179.99))
    quick[1]['x'] = 10.
    ref_ds = [(r['idx'], r['t_s'], {0: np.zeros((4, 2))}) for r in base]
    fast_ds = copy.deepcopy(ref_ds[:n])
    fast_ds[1][2].clear()
    fast_ds[8][2][0][1] = [0.09, 0.12]
    cm.write_frames(tmp_path / 'frames.csv', quick)
    day = full_chain(tmp_path / 'day', base, ref_ds)
    cm.save_dets(tmp_path / 'dets_nhanh.npz', fast_ds, 'V', (2, 2))
    shas = dict(frames_sha256=cm.sha256(tmp_path / 'frames.csv'), dets_sha256=cm.sha256(tmp_path / 'dets_nhanh.npz'))
    fast.write_json(tmp_path / 'track.json', dict(shas, thu=True))
    (tmp_path / 'rows.csv').write_text('t,x\n' + ''.join(f'{i / 60},0\n' for i in range(1200)))
    fast.write_json(tmp_path / 'node.json', dict(NODE, khung_tuong_doi_v1=dict(rows_sha256=cm.sha256(tmp_path / 'rows.csv'))))
    fast.write_json(tmp_path / 'cham.json', verdict_of(NODE))
    args = ['so', '--nhanh', str(tmp_path), *day,
            '--cham-day', str(tmp_path / 'node.json'), '--cham-nhanh', str(tmp_path / 'cham.json'),
            '--rows-day', str(tmp_path / 'rows.csv')]
    output = tmp_path / 'so.json'
    assert fast.main(args + ['--out', str(output)]) == 0
    result = fast.read_json(output)
    k2 = result['K2']
    assert not result['dat'] and result['K3']['dat'] and not k2['dat']
    assert (k2['n'], k2['n_hop_le_chung'], k2['n_goc'], k2['n_chi_nhanh'], k2['n_chi_day']) == (10, 2, 9, 4, 3)
    assert k2['xy_max_mm']['gia_tri'] == pytest.approx(0.5)
    assert k2['psi_max_deg']['gia_tri'] == pytest.approx(0.02, abs=0.0001)
    assert k2['goc_max_px'] == dict(gia_tri=0.15, nguong=0.1, dat=False)
    assert k2['hop_le_mot_ben_pct'] == dict(gia_tri=70., nguong=1., dat=False)
    assert k2['lien_tiep'] == dict(gia_tri=7, nguong=6, dat=False)
    (tmp_path / 'track.json').write_text(json.dumps(dict(shas, thu=False)))
    assert fast.main(args + ['--out', str(tmp_path / 'bad.json')]) == 2


def test_k3_g1_threshold_is_one_real_frame_of_a_variable_rate_clip():
    """Design 4.2 v1.2 (Liam 27/11): the G1 t_done threshold is the real frame interval at the reference t_done."""
    t_ref = 32.77264934894151                                      # run02 C8-L-F02-2: next frame 18.334 ms later
    times = [t_ref - 0.05, t_ref - 1 / 60, t_ref, t_ref + 0.018334, t_ref + 0.018334 + 1 / 60]
    assert fast.frame_step(times, t_ref) == pytest.approx(0.018334)
    assert fast.frame_step(times, t_ref + 0.018334) == pytest.approx(0.018334)
    node = copy.deepcopy(NODE)
    node['gate_v3']['g1']['t_done'] = [t_ref, t_ref]
    verdict = verdict_of(node)
    verdict['g1']['t_done'] = [t_ref + 0.0183289, t_ref + 0.0183289]   # one frame later: within one real frame
    assert fast._compare_verdict(verdict, node, times)['dat']
    assert not fast._compare_verdict(verdict, node, [i / 60 for i in range(3000)])['dat']   # 1/60 s spacing: not
    verdict['g1']['t_done'] = [t_ref + 0.04, t_ref + 0.04]               # two frames later: refused
    assert not fast._compare_verdict(verdict, node, times)['dat']
    with pytest.raises(ValueError):
        fast.frame_step([t_ref], t_ref)


def cham_until_f8(tmp_path, monkeypatch, reasons, write=True):
    cfg = config(tmp_path)
    conf, frames, out = [tmp_path / n for n in ('config.json', 'frames.csv', 'scored')]
    fast.write_json(conf, cfg)
    track_fixture(frames, frames.with_name('dets_nhanh.npz'), tmp_path)
    calls = []

    def run(cmd, cwd, stdin, stdout, stderr):
        calls.append(cmd)
        if len(calls) == 1:
            fast.write_json(cmd[-1], dict(nodes_s=[0, 1]))
        if len(calls) == 3 and write:
            f8_result(cmd, reasons)
        return subprocess.CompletedProcess(cmd, 1 if len(calls) == 3 else 0)

    monkeypatch.setattr(subprocess, 'run', run)
    rc = fast.main(['cham', '--cau-hinh', str(conf), '--frames', str(frames), '--out', str(out)])
    return rc, calls, fast.read_json(out / 'ket_luan_nhanh.json')


@pytest.mark.parametrize('reasons,write', [(['r_vuot'], True), (['thieu_neo', 'r_vuot'], True), (['thieu_neo'], False)])
def test_cham_f8_exit_1_goes_on_only_for_missing_anchors(tmp_path, monkeypatch, reasons, write):
    """CN-3 (Codex 27/11): hieu_chuan exits 1 for every state other than dat; only thieu_neo goes on."""
    rc, calls, result = cham_until_f8(tmp_path, monkeypatch, reasons, write)
    assert rc == 2 and len(calls) == 3 and '3_f8_hieu_chuan' in result['loi']


@pytest.mark.parametrize('field', ['video_sha256', 'session_sha256', 'frames_sha256', 'dets_sha256'])
def test_cham_refuses_inputs_of_another_track(tmp_path, monkeypatch, field):
    """CN-4 (Codex 27/11): clip, session, frames and dets of the config must be those of track.json."""
    cfg = config(tmp_path)
    conf, frames, out = [tmp_path / n for n in ('config.json', 'frames.csv', 'scored')]
    fast.write_json(conf, cfg)
    track_fixture(frames, frames.with_name('dets_nhanh.npz'), tmp_path, **{field: '0' * 64})
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('no step may run'))
    assert fast.main(['cham', '--cau-hinh', str(conf), '--frames', str(frames), '--out', str(out)]) == 2
    assert field in fast.read_json(out / 'ket_luan_nhanh.json')['loi']


def so_inputs(tmp_path, n=4):
    rows = [dict(idx=i, t_s=i / 60, n_marks=12, why=[], car_seen=True, valid=True, x=1., y=0., psi=0.) for i in range(n)]
    ds = [(r['idx'], r['t_s'], {0: np.zeros((4, 2))}) for r in rows]
    return rows, ds


def write_so(tmp_path, quick, base, fast_ds, ref_ds):
    tmp_path.mkdir()                                       # every writer here refuses to overwrite
    cm.write_frames(tmp_path / 'frames.csv', quick)
    day = full_chain(tmp_path / 'day', base, ref_ds)
    cm.save_dets(tmp_path / 'dets_nhanh.npz', fast_ds, 'V', (2, 2))
    fast.write_json(tmp_path / 'track.json', dict(frames_sha256=cm.sha256(tmp_path / 'frames.csv'),
                                                  dets_sha256=cm.sha256(tmp_path / 'dets_nhanh.npz')))
    return ['so', '--nhanh', str(tmp_path), *day]


def test_k2_refuses_files_that_do_not_belong_together(tmp_path):
    """CN-5, CN-6 (Codex 27/11): car tag missing on a frame valid on both sides; other frame times; sha256."""
    rows, ds = so_inputs(tmp_path)
    args = write_so(tmp_path / 'a', rows, rows, ds, ds)
    assert fast.main(args + ['--out', str(tmp_path / 'ok.json')]) == 0
    assert fast.read_json(tmp_path / 'ok.json')['K2']['dat']
    fewer = copy.deepcopy(ds)
    fewer[2][2].clear()
    with pytest.raises(ValueError, match='car tag is missing'):
        fast.compare_frames(rows, rows, fewer, ds)
    shifted = [dict(r, t_s=r['t_s'] + 1) for r in rows]
    with pytest.raises(ValueError, match='frame times'):
        fast.compare_frames(rows, shifted, ds, ds)
    late = [(i, t + 1, d) for i, t, d in ds]
    with pytest.raises(ValueError, match='frame times'):
        fast.compare_frames(rows, rows, ds, late)
    args = write_so(tmp_path / 'b', rows, shifted, ds, ds)
    assert fast.main(args + ['--out', str(tmp_path / 'b1.json')]) == 2
    args = write_so(tmp_path / 'c', rows, rows, ds, ds)
    (tmp_path / 'c/frames.csv').write_bytes((tmp_path / 'c/frames.csv').read_bytes() + b'\n')
    assert fast.main(args + ['--out', str(tmp_path / 'b2.json')]) == 2


def test_k3_node_number_rows_file_and_lost_nodes(tmp_path):
    """CN-7, CN-10 (Codex 27/11): same node number; rows.csv of that node; two lost nodes need no rows.csv."""
    rows, ds = so_inputs(tmp_path)
    args = write_so(tmp_path / 'track', rows, rows, ds, ds)
    node = copy.deepcopy(NODE)
    (tmp_path / 'rows.csv').write_text('t,x\n' + ''.join(f'{i / 60},0\n' for i in range(1200)))
    node['khung_tuong_doi_v1'] = dict(rows_sha256=cm.sha256(tmp_path / 'rows.csv'))
    fast.write_json(tmp_path / 'node.json', node)
    fast.write_json(tmp_path / 'cham.json', verdict_of(node))
    k3 = args + ['--cham-day', str(tmp_path / 'node.json'), '--cham-nhanh', str(tmp_path / 'cham.json')]
    assert fast.main(k3 + ['--rows-day', str(tmp_path / 'rows.csv'), '--out', str(tmp_path / 'a.json')]) == 0
    assert fast.read_json(tmp_path / 'a.json')['K3']['dat']
    assert fast.main(k3 + ['--out', str(tmp_path / 'b.json')]) == 2               # a reference t_done: rows.csv needed
    (tmp_path / 'other.csv').write_text('t,x\n' + ''.join(f'{i / 30},0\n' for i in range(1200)))
    assert fast.main(k3 + ['--rows-day', str(tmp_path / 'other.csv'), '--out', str(tmp_path / 'c.json')]) == 2
    other = dict(verdict_of(node), nut=7)
    with pytest.raises(ValueError, match='node numbers'):
        fast._compare_verdict(other, node, [i / 60 for i in range(1200)])
    lost = dict(outcome='hong_he_do', gate_v3=None, gio_c8b=dict(nut=2))
    fast.write_json(tmp_path / 'lost.json', lost)
    fast.write_json(tmp_path / 'cham_lost.json', verdict_of(lost))
    k3 = args + ['--cham-day', str(tmp_path / 'lost.json'), '--cham-nhanh', str(tmp_path / 'cham_lost.json')]
    assert fast.main(k3 +['--out', str(tmp_path / 'd.json')]) == 0
    assert fast.read_json(tmp_path / 'd.json')['K3']['dat']


def test_k3_direct_call_verifies_the_rows_file(tmp_path):
    """CN-7 round 2 (Codex 27/11): compare_verdict reads the node's rows.csv itself and checks its sha256, so a direct
    caller cannot pass another frame spacing; a node with a t_done but no rows_sha256 gives no G1 threshold."""
    t_ref = NODE['gate_v3']['g1']['t_done'][0]
    (tmp_path / 'rows.csv').write_text('t\n' + ''.join(f'{i / 60}\n' for i in range(1200)))
    (tmp_path / 'r30.csv').write_text('t\n' + ''.join(f'{i / 30}\n' for i in range(1200)))
    node = dict(NODE, khung_tuong_doi_v1=dict(rows_sha256=cm.sha256(tmp_path / 'rows.csv')))
    late = verdict_of(node)
    late['g1'] = dict(late['g1'], t_done=[t_ref + 0.025, t_ref + 0.025])       # later than 1/60 s, within 1/30 s
    assert not fast.compare_verdict(late, node, tmp_path / 'rows.csv')['dat']
    with pytest.raises(ValueError, match='rows_sha256'):
        fast.compare_verdict(late, node, tmp_path / 'r30.csv')
    with pytest.raises(ValueError, match='rows_sha256'):
        fast.compare_verdict(late, NODE, tmp_path / 'rows.csv')                  # the node records no rows_sha256
    lost = dict(outcome='hong_he_do', gate_v3=None)
    assert fast.compare_verdict(dict(fast.node_verdict(lost), nut=3), lost, None)['dat']


def test_so_refuses_full_chain_files_other_than_its_track(tmp_path, capsys):
    """CN-6 round 2 (Codex 27/11): --day and --dets must be the files the full chain's track.json records."""
    rows, ds = so_inputs(tmp_path)
    args = write_so(tmp_path / 'a', rows, rows, ds, ds)
    assert fast.main(args + ['--out', str(tmp_path / 'ok.json')]) == 0
    day = tmp_path / 'a/day/frames.csv'
    moved = [dict(r, x=r['x'] + (1e-4 if r['idx'] == 1 else 0)) for r in rows]   # same idx and t_s, x + 0.1 mm
    day.unlink()
    cm.write_frames(day, moved)
    assert fast.main(args + ['--out', str(tmp_path / 'b.json')]) == 2
    args = write_so(tmp_path / 'c', rows, rows, ds, ds)
    # CNV3-2 (Codex 27/11): same detections (K2 alone would pass), another file: refused by its sha256 only
    other = tmp_path / 'c/other.npz'
    cm.save_dets(other, ds, 'V', (3, 3))
    capsys.readouterr()
    assert fast.main(args[:-1] + [str(other), '--out', str(tmp_path / 'd.json')]) == 2
    assert 'dets_sha256' in capsys.readouterr().err


def test_cham_broken_track_json_still_writes_the_verdict(tmp_path, monkeypatch):
    """CNV2-1 (Codex 27/11): a broken track.json stops cham before step 1 with its error in ket_luan_nhanh.json."""
    cfg = config(tmp_path)
    conf, frames, out = [tmp_path / n for n in ('config.json', 'frames.csv', 'scored')]
    fast.write_json(conf, cfg)
    track_fixture(frames, frames.with_name('dets_nhanh.npz'), tmp_path)
    frames.with_name('track.json').write_text('{')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('no step may run'))
    assert fast.main(['cham', '--cau-hinh', str(conf), '--frames', str(frames), '--out', str(out)]) == 2
    result = fast.read_json(out / 'ket_luan_nhanh.json')
    assert result['buoc'] == {} and result['loi']


@pytest.mark.parametrize('missing', ['track.json', 'frames.csv', 'dets_nhanh.npz'])
def test_cham_missing_track_files_still_write_the_verdict(tmp_path, monkeypatch, missing):
    """CNV3-1 (Codex 27/11): a missing track file stops cham before step 1 with its error in ket_luan_nhanh.json."""
    cfg = config(tmp_path)
    conf, frames, out = [tmp_path / n for n in ('config.json', 'frames.csv', 'scored')]
    fast.write_json(conf, cfg)
    track_fixture(frames, frames.with_name('dets_nhanh.npz'), tmp_path)
    frames.with_name(missing).unlink()
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('no step may run'))
    assert fast.main(['cham', '--cau-hinh', str(conf), '--frames', str(frames), '--out', str(out)]) == 2
    result = fast.read_json(out / 'ket_luan_nhanh.json')
    assert result['buoc'] == {} and missing in result['loi']


def test_help_text_shows_the_parser_defaults():
    """CN-12 (Codex 27/11): the defaults written in the module docstring (shown by --help) are the parser's."""
    ap = fast.build_parser()
    a = ap.parse_args(['track', '--video', 'v', '--session', 's', '--hang-so', 'h', '--out', 'o'])
    shown = ap.format_help()                      # CNV3-3 (Codex 27/11): what --help prints, not only the docstring
    for flag, value in (('--n-thua', a.n_thua), ('--nua-vung', a.nua_vung), ('--eps-cam-mm', a.eps_cam_mm)):
        assert f'[{flag} {value}]' in shown


@pytest.mark.parametrize('times,t', [([0, 0.016, 0.032], 2.0), ([0, 0.016, 0.032], -0.01), ([0, 1, 1, 1.03], 1),
                                     ([0, float('nan'), 0.032], 0.016), ([0, 0.016, 0.032], float('nan'))])
def test_frame_step_refuses_times_that_do_not_cover_t_done(times, t):
    """CN-8 (Codex 27/11): finite, strictly increasing frame times that cover the reference t_done."""
    with pytest.raises(ValueError):
        fast.frame_step(times, t)
