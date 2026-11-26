"""Fast C8 camera scoring (design 2026-11-26, sections 2 and 4).

python tools/c7_nhanh.py track --video V --session S --hang-so HS --out DIR
    [--tag-offset DX,DY,DPSI_DEG] [--n-thua 30] [--nua-vung 150] [--eps-cam-mm 0.25]
    [--jobs J] [--max-frames N]
python tools/c7_nhanh.py cham --cau-hinh C.json --frames DIR/frames.csv --out DIR2 [--luoi] [--nut K]
    [--dets D.npz (default dets_nhanh.npz beside the frames; the full chain's dets.npz for its frames)]
python tools/c7_nhanh.py so --nhanh DIR --day frames.csv --dets dets.npz --out so.json
    [--cham-nhanh ket_luan_nhanh.json --cham-day nut.json]
"""
import argparse
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from io import BytesIO
from itertools import islice, pairwise
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from zipfile import ZipFile

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c7_clip_probe as cp
import c7_measure as cm


def read_json(path):
    with open(path, encoding='utf-8') as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return obj


def write_json(path, obj):
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(cm._json(obj), f, ensure_ascii=False, indent=1, allow_nan=False)
        f.write('\n')


def new_dir(path):
    out = Path(path).resolve()
    try:
        out.mkdir(parents=True)
    except FileExistsError:
        raise ValueError(f'{out} exists: use a new directory') from None
    return out


def prediction(older, newer):
    """Use only the last two frame slots, including a miss (design 2.1.5)."""
    if newer is None:
        return older
    return newer if older is None else 2 * newer - older


def crop_car(detector, gray, centre, half):
    if centre is None:
        return None, 'chua_du_doan'
    h, w = gray.shape
    x, y = (int(round(v)) for v in centre)
    x0, y0 = max(0, x - half), max(0, y - half)
    x1, y1 = min(w, x + half), min(h, y + half)
    if x1 <= x0 or y1 <= y0:
        return None, 'ngoai_anh'
    # Design 2.1.5 v1.2 (CAM 26/11, validation run of 5c9bceb): ArUco on the crop only finds the tag; the edge
    # refinement runs on the FULL frame from those corners, as detect_refined does (same half width), because
    # refine_edges on the crop moved a corner by up to 0.10 px from the full-frame result although the ArUco corners
    # were identical (run02 C8-P-F02-1_lai1 frame 2813). Same start, same image -> the full-chain corners.
    raw = cp.detect_markers(detector, gray[y0:y1, x0:x1]).get(cm.CAR_ID)
    if raw is None:
        return None, 'khong_thay'
    raw = raw + [x0, y0]
    side_px = float(np.mean(np.linalg.norm(np.diff(np.vstack([raw, raw[:1]]), axis=0), axis=1)))
    corners = cm.refine_edges(gray, raw, half=min(cm.HALF_CAR_PX, 0.75 * side_px / 6.0))
    if corners is None:
        return None, 'tinh_chinh_bo'
    margin = cm.HALF_CAR_PX + 3
    cx, cy = (corners - [x0, y0]).T
    if (x0 > 0 and cx.min() < margin or x1 < w and x1 - x0 - 1 - cx.max() < margin or
            y0 > 0 and cy.min() < margin or y1 < h and y1 - y0 - 1 - cy.max() < margin):
        return None, 'sat_mep'
    return corners, None


def corner_error(a, b):
    return float(np.linalg.norm(a - b, axis=1).max())


def _worker_init():
    global _detector
    cv2.setNumThreads(1)
    _detector = cm.make_detector()


def _full_detection(gray):
    start = perf_counter()
    d = cm.detect_refined(_detector, gray)
    return d, perf_counter() - start


def pose_args(session, off):
    return (session['car_side_m'], session['kappa'], session['plumb_xy_m'], off)


def needs_split(a, b, fits, detections, ctx, car_args, eps_mm):
    """Design 2.1.3. Compare the same observed tag under both endpoint cameras."""
    cameras = []
    for i in (a, b):
        cam, _, ok, _, _ = fits[i]
        if not ok or len(ctx.universe[cm.CHAIN_XE].intersection(detections[i])) < cm.MIN_TRAIN:
            return True
        cameras.append(cam)
    cars = [detections[i][cm.CAR_ID] for i in range(a, b + 1) if cm.CAR_ID in detections[i]]
    for corners in cars:
        p, q = [cm.car_pose(cam, corners, *car_args) for cam in cameras]
        if 1000 * math.hypot(p['x'] - q['x'], p['y'] - q['y']) > eps_mm:
            return True
    if not cars:
        for i in (a, b):
            for j in ctx.universe[cm.CHAIN_XE].intersection(detections[i]):
                p, q = [cm.mark_centre_floor(cam, detections[i][j]) for cam in cameras]
                if 1000 * np.linalg.norm(p - q) > eps_mm:
                    return True
    return False


def midpoints(a, b, full, split):
    """Bisect a sparse interval. Also test both neighbours of incidental full detections."""
    if b - a <= 1:
        return []
    if split(a, b):
        mid = (a + b) // 2
        if mid not in full:
            return [mid]
        return midpoints(a, mid, full, split) + midpoints(mid, b, full, split)
    inside = sorted(i for i in full if a < i < b)
    if not inside:
        return []
    return [k for lo, hi in pairwise([a, *inside, b]) for k in midpoints(lo, hi, full, split)]


class DetectionPass:
    """Compact detections survive the pass. Grey images and in-flight jobs belong to one interval."""

    def __init__(self, pool, session, off, half, eps_mm):
        self.pool = pool
        self.ctx = cm.ChainCtx(session)
        self.car_args = pose_args(session, off)
        self.half, self.eps_mm = half, eps_mm
        self.detector = cm.make_detector()
        self.dets, self.fits, self.cache = {}, {}, {}
        self.times, self.shapes = [], set()
        self.sparse, self.dense = set(), set()
        self.fallback = Counter()
        self.timing = dict(giai_ma=0., vung_cat=0., do_ca_khung_cong_worker=0., cho_worker=0.,
                           khop_camera=0., xet_day=0.)
        self.check = dict(n=0, n_co_xe=0, n_dung_tam_ca_khung=0, n_loi=0, lech_max_px=None)
        self.history = (None, None)

    def _crop(self, gray, centre):
        start = perf_counter()
        result = crop_car(self.detector, gray, centre, self.half)
        self.timing['vung_cat'] += perf_counter() - start
        return result

    def _resolve(self, idx, future):
        start = perf_counter()
        d, elapsed = future.result()
        self.timing['cho_worker'] += perf_counter() - start
        self.timing['do_ca_khung_cong_worker'] += elapsed
        self.dets[idx] = d
        start = perf_counter()
        anchors = self.ctx.universe[cm.CHAIN_XE].intersection(d)
        self.fits[idx] = self.ctx.pose(cm.CHAIN_XE, anchors, d, self.cache, idx)
        self.timing['khop_camera'] += perf_counter() - start

    def interval(self, block):
        a, b = block[0][0] - 1, block[-1][0]
        self.sparse.add(b)
        pending = {b: self.pool.submit(_full_detection, block[-1][2])}
        full = {a, b} if a >= 0 else {b}
        crops = {}

        def crop(idx, gray, centre):
            # A new midpoint can change the next prediction. Reuse only an identical crop.
            key = None if centre is None else tuple(centre)
            if idx not in crops or crops[idx][0] != key:
                crops[idx] = (key, self._crop(gray, centre))
            return crops[idx][1]

        while True:
            older, newer = self.history
            predictions = {}
            for i, _, gray in block:
                centre = predictions[i] = prediction(older, newer)
                if i not in full:
                    car, reason = crop(i, gray, centre)
                    if reason is None:
                        self.dets[i] = {cm.CAR_ID: car}
                    else:
                        self.fallback[reason] += 1
                        pending[i] = self.pool.submit(_full_detection, gray)
                        full.add(i)
                if i in pending:
                    self._resolve(i, pending.pop(i))
                car = self.dets[i].get(cm.CAR_ID)
                older, newer = newer, None if car is None else car.mean(axis=0)
            start = perf_counter()
            mids = midpoints(a, b, full, lambda l, r: needs_split(
                l, r, self.fits, self.dets, self.ctx, self.car_args, self.eps_mm)) if a >= 0 else []
            self.timing['xet_day'] += perf_counter() - start
            if not mids:
                break
            for i in mids:
                pending[i] = self.pool.submit(_full_detection, block[i - a - 1][2])
            full.update(mids)
            self.dense.update(mids)

        self.history = older, newer
        for i, _, gray in block:
            car = self.dets[i].get(cm.CAR_ID)
            if i not in full or car is None:
                continue
            self.check['n_co_xe'] += 1
            probe, _ = crop(i, gray, predictions[i])
            if probe is None:
                # A diagnostic centred on the full detection is possible even on frame zero.
                self.check['n_dung_tam_ca_khung'] += 1
                probe, _ = crop(i, gray, car.mean(axis=0))
            if probe is None:
                self.check['n_loi'] += 1
            else:
                self.check['n'] += 1
                self.check['lech_max_px'] = max(self.check['lech_max_px'] or 0., corner_error(probe, car))

    def run(self, video, n, limit):
        def decode():
            it = cm.iter_frames(video, limit)
            while True:
                start = perf_counter()
                frame = next(it, None)
                self.timing['giai_ma'] += perf_counter() - start
                if frame is None:
                    return
                i, t, gray = frame
                if i != len(self.times):
                    raise ValueError(f'unexpected decoded frame index {i}')
                self.times.append(t)
                self.shapes.add(gray.shape)
                yield frame

        it = decode()
        first = next(it, None)
        if first is None:
            raise ValueError(f'{video}: no decoded frames')
        self.interval([first])
        del first
        while block := list(islice(it, n)):
            self.interval(block)
            del block
        return [(i, t, self.dets[i]) for i, t in enumerate(self.times)]


def build_rows(frames, fully_detected, session, off=None, gates=None, gates_src=None, *, cache=None,
               video_name=None, video_sha=None, shapes=None, thu=False):
    """Design 2.1.4–7. Obtain the camera-only template rows and metadata from the unscheduled chain."""
    off = session['tag_offset'] if off is None else off
    ctx = cm.ChainCtx(session)
    cache = {} if cache is None else cache
    anchors = [(i, t, {j: c for j, c in d.items() if j != cm.CAR_ID})
               for i, t, d in frames if i in fully_detected]
    if not anchors:
        raise ValueError('no fully detected frames')
    templates, _, info = cm.track_chains(anchors, video_name, video_sha, session, off, gates, gates_src,
                                         cache=cache, shapes=shapes, thu=thu)
    ids = [i for i, _, _ in anchors]
    cameras = [ctx.pose(cm.CHAIN_XE, ctx.universe[cm.CHAIN_XE].intersection(d), d, cache, i)[0] for i, _, d in anchors]
    rows = []
    for i, t, d in frames:
        p = bisect_left(ids, i)
        if p == len(ids) or p and i - ids[p - 1] <= ids[p] - i:
            p -= 1
        r = templates[p].copy()
        if ids[p] != i:
            for key in templates[p]:
                if key.startswith('kc_'):
                    del r[key]
        r.update(idx=i, t_s=t, cam_tu_khung=ids[p], xe_nguon='ca_khung' if i in fully_detected else 'vung_cat')
        cam, car = cameras[p], d.get(cm.CAR_ID)
        if cam is not None and car is not None:
            r.update(cm.car_pose(cam, car, *pose_args(session, off)))
            r.update(cm.tag_corners_floor(cam, car, session['kappa'], session['plumb_xy_m']))
            r.update(car_seen=True, row_px=float(car[:, 1].mean()), col_px=float(car[:, 0].mean()))
        rows.append(r)
    info.update(frames=len(rows), median_scale=cm.apply_gates(rows, session['rms_px_H'], gates),
                cam_z_drift_rel=cm._cam_z_drift(rows, session.get('zc_fit_m')),
                n_khop_khong_tin=sum(not r['khop_ok'] for r in rows))
    info['n_valid'] = sum(r['valid'] for r in rows)
    return rows, info


def write_fast_frames(path, rows):
    # write_frames owns every existing column and all numeric formatting (design 2.1.6).
    base = path.with_suffix('.base.csv')
    cm.write_frames(base, rows)
    try:
        with base.open(newline='', encoding='utf-8') as src, path.open('x', newline='', encoding='utf-8') as dst:
            reader, writer = csv.reader(src), csv.writer(dst)
            writer.writerow([*next(reader), 'cam_tu_khung', 'xe_nguon'])
            for cells, row in zip(reader, rows, strict=True):
                writer.writerow([*cells, row['cam_tu_khung'], row['xe_nguon']])
    finally:
        base.unlink()


def track(a):
    start = perf_counter()
    if min(a.jobs, a.n_thua, a.nua_vung) < 1 or (a.max_frames is not None and a.max_frames < 1):
        raise ValueError('--jobs, --n-thua, --nua-vung and --max-frames must be positive')
    if not math.isfinite(a.eps_cam_mm) or a.eps_cam_mm < 0:
        raise ValueError('--eps-cam-mm must be finite and nonnegative')
    session = read_json(a.session)
    off = session['tag_offset']
    if a.tag_offset is not None:
        dx, dy, deg = cm._pair(a.tag_offset, 3, '--tag-offset')
        off = dx, dy, math.radians(deg)
        if not all(map(math.isfinite, off)):
            raise ValueError('--tag-offset must be finite')
    gates, gsha, state = cm.load_gates(a.hang_so)
    src = dict(file=Path(a.hang_so).name, sha256=gsha, trang_thai=state)
    out = new_dir(a.out)
    with ProcessPoolExecutor(a.jobs, initializer=_worker_init) as pool:
        scan = DetectionPass(pool, session, off, a.nua_vung, a.eps_cam_mm)
        frames = scan.run(a.video, a.n_thua, a.max_frames)
    seconds = scan.timing
    now = perf_counter()
    vsha, ssha = cm.sha256(a.video), cm.sha256(a.session)
    seconds['bam_dau_vao'] = perf_counter() - now
    now = perf_counter()
    full = set(scan.fits)
    rows, info = build_rows(frames, full, session, off, gates, src, cache=scan.cache, video_name=Path(a.video).name,
                            video_sha=vsha, shapes=scan.shapes, thu=a.max_frames is not None)
    seconds['dong_va_cong'] = perf_counter() - now
    now = perf_counter()
    fp, dp = out / 'frames.csv', out / 'dets_nhanh.npz'
    write_fast_frames(fp, rows)
    cm.save_dets(dp, frames, vsha, sorted(scan.shapes)[0])
    with ZipFile(dp, 'a') as z:
        data = BytesIO()
        np.save(data, np.array(sorted(full), dtype=int), allow_pickle=False)
        z.writestr('ca_khung.npy', data.getvalue())
    info.update(duong='nhanh_v1', ket_luan='duong nhanh v1 (cham_lai 1, chua co lich)',
                session=str(Path(a.session).resolve()), session_sha256=ssha, frames_sha256=cm.sha256(fp),
                dets_file=str(dp), dets_sha256=cm.sha256(dp), dets_ghi_chu='Khong dung cho c7_kiem_cheo / lich',
                lich_file=None, lich_file_sha256=None, kiem_cheo_khung_sha256=None,
                tham_so=dict(n_thua=a.n_thua, nua_vung=a.nua_vung, eps_cam_mm=a.eps_cam_mm, jobs=a.jobs,
                             max_frames=a.max_frames), n_ca_khung=len(full), n_thua=len(scan.sparse),
                n_day=len(scan.dense), n_xe_do_ca_khung=sum(scan.fallback.values()), xe_du_phong=dict(scan.fallback),
                kiem_vung_cat=scan.check, thoi_gian_s=seconds)
    seconds['ghi_va_bam_dau_ra'] = perf_counter() - now
    seconds['tong'] = perf_counter() - start
    write_json(out / 'track.json', info)
    summary = (f"{len(rows)} khung, {info['n_valid']} hop le, {len(full)} ca khung, {len(scan.dense)} day, "
               f"{info['n_xe_do_ca_khung']} xe du phong")
    (out / 'nhat_ky.txt').write_text(summary + '\n' + json.dumps(seconds) + '\n', encoding='utf-8')
    print(summary)


CONFIG_FILES = ('esp32', 'sender', 'clip', 'session', 'hang_so', 'a6', 'xoay', 'truc_xe', 'khoa_gop')
GATES = ('g1', 'g2', 'g3', 'g4', 'g5')


def load_config(path):
    """Resolve relative inputs beside the config before changing cwd for the tools (design 2.2)."""
    from c7_nhac_tam import parse_neo
    from c7_score import FIG8, FINAL_SET

    cfg = read_json(path)
    for key in ('task', 'luot_id', *CONFIG_FILES):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f'cau_hinh.{key}: expected nonempty string')
    if cfg['task'] not in (*FIG8, *FINAL_SET):
        raise ValueError(f"unknown task {cfg['task']!r}")
    f8 = cfg.get('f8')
    if not isinstance(f8, dict):
        raise ValueError('cau_hinh.f8 must be an object')
    # f8.mau_clip (design 2.2 v1.2, CAM 26/11): the template clip of hieu_chuan --mau is the clip H at the start of the
    # camera segment, as C8B's empty F8 of run03-run07 (nhac_tam/f8_rong*/chay_f8_rong.sh: clip H, frame 500). The run
    # clip refused on run07 C8-L-F08-2_lai1 ("template frame 500 belongs to held-out window 0").
    for key in ('neo', 'tam_id', 'mau_clip'):
        if not isinstance(f8.get(key), str) or not f8[key].strip():
            raise ValueError(f'f8.{key}: expected nonempty string')
    if type(f8.get('mau_n')) is not int or f8['mau_n'] < 0:
        raise ValueError('f8.mau_n: expected nonnegative frame index')
    parse_neo(f8['neo'])
    base = Path(path).resolve().parent
    for key in CONFIG_FILES:
        file = (base / cfg[key]).resolve()
        if not file.is_file():
            raise ValueError(f'{key}: no input file {file}')
        cfg[key] = str(file)
    file = (base / f8['mau_clip']).resolve()
    if not file.is_file():
        raise ValueError(f'f8.mau_clip: no input file {file}')
    f8['mau_clip'] = str(file)
    return cfg


def node_verdict(node):
    """The run outcome can differ from gate_v3.outcome, e.g. a measurement loss."""
    outcome = node.get('outcome')
    if not isinstance(outcome, str) or not outcome or 'gate_v3' not in node:
        raise ValueError('node requires outcome and gate_v3')
    verdict = dict(outcome=outcome)
    gate = node['gate_v3']
    for g in GATES:
        keys = ('state', 't_done') if g == 'g1' else ('state', 'q', 'lo', 'hi')
        if gate is None:  # c7_score.lost_output
            verdict[g] = dict.fromkeys(keys)
        else:
            block = gate.get(g) if isinstance(gate, dict) else None
            if not isinstance(block, dict) or not all(k in block for k in keys):
                raise ValueError(f'gate_v3.{g} requires {keys}')
            verdict[g] = {k: block[k] for k in keys}
    return verdict


def cham(a):
    cfg = load_config(a.cau_hinh)
    out = new_dir(a.out)
    frames = Path(a.frames).resolve()
    # --dets: only hashed by c7_score cua-so; given when scoring frames of the full chain (K3a, design 4.2)
    dets = Path(a.dets).resolve() if a.dets else frames.with_name('dets_nhanh.npz')
    inputs = {key: cfg[key] for key in CONFIG_FILES}
    inputs['f8_mau_clip'] = cfg['f8']['mau_clip']
    inputs.update(cau_hinh=a.cau_hinh, frames=frames, dets=dets, track=frames.with_name('track.json'))
    for suffix in ('events.txt', 'segments.csv'):
        p = Path(cfg['esp32'] + '.' + suffix)
        if p.is_file():
            inputs[suffix] = p
    result = dict(nhan='duong_nhanh_v1', sha256={k: cm.sha256(p) for k, p in inputs.items()}, buoc={})
    for sub in ('cua_so', 'f12', 'mot_nut'):
        (out / sub).mkdir()
    gio, cua = out / 'gio_so_bo.json', out / 'cua_so/cua_so_v1.jsonl'
    hc, ll = out / 'f8/hc/hieu_chuan_neo.json', out / 'f8/ll/lan_lap.json'
    ref, node = out / 'f12/tham_chieu_v1.json', out / 'mot_nut/nut.json'
    task = ['--task', cfg['task']]
    log = ['--esp32', cfg['esp32']]
    hs = ['--hang-so', cfg['hang_so']]
    run_id = ['--luot-id', cfg['luot_id']]
    session = ['--session', cfg['session']]
    a6 = ['--a6', cfg['a6']]
    tx = ['--truc-xe', cfg['truc_xe']]
    xoay = ['--xoay', cfg['xoay']]
    f8 = cfg['f8']
    sc = ['--tuong-doi', '--c8b-khoa', cfg['khoa_gop'], '--cua-so', cua, '--tham-chieu', ref,
          '--lan-lap', ll, *a6, *xoay, *tx, *run_id]

    def step(name, script, *args):
        cmd = [sys.executable, '-X', 'utf8', '-I', '-B', f'tools/{script}.py', *map(str, args)]
        start = perf_counter()
        with (out / f'{name}.txt').open('xb') as stdout:
            rc = subprocess.run(cmd, cwd=HERE.parent, stdin=subprocess.DEVNULL, stdout=stdout,
                                stderr=subprocess.STDOUT).returncode
        result['buoc'][name] = dict(ma=rc, thoi_gian_s=perf_counter() - start)
        if rc and not (name == '3_f8_hieu_chuan' and rc == 1):
            raise ValueError(f'{name}: exit {rc} (see {out / (name + ".txt")})')

    try:
        step('1_gio_so_bo', 'c8b_gio', '--frames', frames, *log, *task, '--out', gio)
        step('2_cua_so', 'c7_score', 'cua-so', '--frames', frames, *task, *hs, *log, '--gio-c8b', gio,
             *run_id, '--clip', cfg['clip'], *session, '--dets', dets, '--out', cua)
        step('3_f8_hieu_chuan', 'c7_nhac_tam', 'hieu_chuan', '--cua-so', cua, *session, '--neo', f8['neo'],
             '--mau', f8['mau_clip'], cfg['session'], f8['mau_n'], *a6, '--tam-id', f8['tam_id'], '--out', hc.parent)
        step('4_f8_lan_lap', 'c7_nhac_tam', 'lan_lap', '--hieu-chuan', hc, '--cua-so', cua, *session,
             *a6, '--tam-id', f8['tam_id'], '--out', ll.parent)
        step('5_f12', 'c7_score', 'tham-chieu', '--frames', frames, *task, *hs, '--cua-so', cua, *run_id,
             *a6, *xoay, '--lan-lap', ll, *tx, '--out', ref)
        nodes = read_json(gio).get('nodes_s')
        if not isinstance(nodes, list) or not nodes:
            raise ValueError(f'{gio}: no nodes_s')
        k = len(nodes) // 2 if a.nut is None else a.nut         # design 2.2 step 5: middle node, or --nut (K3b)
        if not 0 <= k < len(nodes):
            raise ValueError(f'--nut {k}: the preliminary time file has {len(nodes)} nodes')
        result['nut'] = k
        step('6_mot_nut', 'c7_score', '--frames', frames, *task, *log, '--sender', cfg['sender'], *hs, *sc,
             '--gio-c8b', gio, '--nut', k, '--out', node, '--rows', node.with_name('rows.csv'))
        result.update(node_verdict(read_json(node)))
        if a.luoi:
            final = out / 'gio.json'
            step('7_gio_cuoi', 'c8b_gio', '--frames', frames, *log, *task, '--so-bo', gio, '--cua-so', cua,
                 '--score', node, '--out', final)
            step('8_luoi', 'c8b_luoi', '--gio', final, '--frames', frames, *log, '--sender', cfg['sender'],
                 *task, *hs, '--out-dir', out / 'luoi', *(f'--scorer-arg={v}' for v in sc))
    except (ValueError, OSError) as exc:
        result['loi'] = str(exc)
        raise
    finally:
        # These files become inputs to later steps, including the node consumed here.
        for key, path in (('gio_so_bo', gio), ('cua_so', cua), ('hieu_chuan', hc), ('lan_lap', ll),
                          ('tham_chieu', ref), ('mot_nut', node), ('gio_cuoi', out / 'gio.json')):
            if path.is_file():
                result['sha256'][key] = cm.sha256(path)
        write_json(out / 'ket_luan_nhanh.json', result)


# Design 4.2 K2. All distances are Euclidean, heading is the shortest circular difference.
K2_XY_MM = 0.5
K2_PSI_DEG = 0.05
K2_CORNER_PX = 0.1
K2_ONE_VALID_PERCENT = 1.0
K2_ONE_VALID_RUN = 6
# Design 4.2 K3. The authoritative node records u_p in metres and u_psi_deg in degrees.
K3_U_FRACTION = 0.05
K3_TIME_S = 1 / 60


def limit(value, threshold):
    return dict(gia_tri=value, nguong=threshold, dat=value is None or value <= threshold)


def compare_frames(fast, reference, fast_dets, ref_dets):
    """K2 on the whole comparison domain, including invalid rows and undetected tags."""
    n = len(fast)
    indices = list(range(n))
    if not n or any([r['idx'] for r in rows] != indices for rows in (fast, reference)):
        raise ValueError('frames.csv: expected matching consecutive indices from zero')
    if any([i for i, _, _ in ds] != indices for ds in (fast_dets, ref_dets)):
        raise ValueError('detections do not cover the frame indices')
    xy, psi, corners = [], [], []
    fast_only = ref_only = run = longest = 0
    for a, b, (_, _, da), (_, _, db) in zip(fast, reference, fast_dets, ref_dets):
        if a['valid'] and b['valid']:
            xy.append(1000 * math.hypot(a['x'] - b['x'], a['y'] - b['y']))
            psi.append(abs(math.degrees(cm.cp_wrap(a['psi'] - b['psi']))))
        different = a['valid'] != b['valid']
        if different:
            fast_only += int(a['valid'])
            ref_only += int(b['valid'])
        run = run + 1 if different else 0
        longest = max(longest, run)
        if cm.CAR_ID in da and cm.CAR_ID in db:
            corners.append(corner_error(da[cm.CAR_ID], db[cm.CAR_ID]))
    if not all(math.isfinite(v) for values in (xy, psi, corners) for v in values):
        raise ValueError('nonfinite pose on a valid frame or nonfinite detected corner')
    metrics = dict(xy_max_mm=limit(max(xy, default=None), K2_XY_MM),
                   psi_max_deg=limit(max(psi, default=None), K2_PSI_DEG),
                   goc_max_px=limit(max(corners, default=None), K2_CORNER_PX),
                   hop_le_mot_ben_pct=limit(100 * (fast_only + ref_only) / n, K2_ONE_VALID_PERCENT),
                   lien_tiep=limit(longest, K2_ONE_VALID_RUN))
    return dict(dat=all(m['dat'] for m in metrics.values()), n=n, n_hop_le_chung=len(xy), n_goc=len(corners),
                n_chi_nhanh=fast_only, n_chi_day=ref_only, **metrics)


def same(a, b):
    return dict(gia_tri=a, nguong=b, dat=a == b)


def delta(a, b, threshold):
    if a is None or b is None:
        return dict(gia_tri=None, nguong=threshold, dat=a is b)
    if threshold is None or not all(math.isfinite(v) for v in (a, b)):
        raise ValueError('K3: finite values and a reference uncertainty required')
    return limit(abs(a - b), threshold)


def compare_verdict(fast, node):
    """K3 compares both completion times and the union-domain G2–G5 values of the node."""
    ref = node_verdict(node)
    gate = node['gate_v3']
    checks = {'outcome': same(fast['outcome'], ref['outcome'])}
    for g in GATES:
        a, b = fast[g], ref[g]
        checks[g + '_state'] = same(a['state'], b['state'])
        if g == 'g1':
            def times(row):
                v = row['t_done']
                if v is None:
                    return [None, None]
                if not isinstance(v, list) or len(v) != 2:
                    raise ValueError('K3: expected the two G1 completion times')
                return v
            for k, (x, y) in enumerate(zip(times(a), times(b)), 1):
                checks[f'g1_t_done_{k}_s'] = delta(x, y, K3_TIME_S)
        else:
            uncertainty = None if gate is None else gate['u_psi_deg' if g == 'g5' else 'u_p']
            if gate is not None and (not math.isfinite(uncertainty) or uncertainty <= 0):
                raise ValueError(f'K3: invalid reference uncertainty for {g}')
            threshold = None if uncertainty is None else K3_U_FRACTION * uncertainty
            for key in ('q', 'lo', 'hi'):
                checks[f'{g}_{key}_{"deg" if g == "g5" else "m"}'] = delta(a[key], b[key], threshold)
    return dict(dat=all(m['dat'] for m in checks.values()), **checks)


def so(a):
    if bool(a.cham_nhanh) != bool(a.cham_day):
        raise ValueError('--cham-nhanh and --cham-day go together')
    folder = Path(a.nhanh)
    fast = cm.read_frames(folder / 'frames.csv')
    ref = cm.read_frames(a.day)
    fd, vf, _ = cm.load_dets(folder / 'dets_nhanh.npz')
    rd, vr, _ = cm.load_dets(a.dets)
    if vf != vr:
        raise ValueError('detections belong to different videos')
    if read_json(folder / 'track.json').get('thu') is True:
        ref, rd = ref[:len(fast)], rd[:len(fast)]
    k2 = compare_frames(fast, ref, fd, rd)
    k3 = compare_verdict(read_json(a.cham_nhanh), read_json(a.cham_day)) if a.cham_day else None
    write_json(a.out, dict(K2=k2, K3=k3, dat=k2['dat'] and (k3 is None or k3['dat'])))


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    ap = Parser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('track')
    for key in ('video', 'session', 'hang-so', 'out'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--tag-offset')
    p.add_argument('--n-thua', type=int, default=30)
    # Design 2.1.5 v1.1 (CAM 26/11): ArUco misses the car tag (side ~95 px) in a crop of half 150-160 on real
    # video; 100 % from 170 on 1458 frames of four runs (do_nua_vung.txt); 250 keeps 80 px for the prediction.
    p.add_argument('--nua-vung', type=int, default=250)
    p.add_argument('--eps-cam-mm', type=float, default=0.25)
    p.add_argument('--jobs', type=int, default=max(1, min(4, (os.cpu_count() or 1) - 2)))
    p.add_argument('--max-frames', type=int)
    p = sub.add_parser('cham')
    for key in ('cau-hinh', 'frames', 'out'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--luoi', action='store_true')
    p.add_argument('--nut', type=int)
    p.add_argument('--dets')
    p = sub.add_parser('so')
    for key in ('nhanh', 'day', 'dets', 'out'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--cham-nhanh')
    p.add_argument('--cham-day')
    try:
        a = ap.parse_args(argv)
        {'track': track, 'cham': cham, 'so': so}[a.cmd](a)
    except Exception as exc:
        print(f'LOI: {exc}', file=sys.stderr)
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
