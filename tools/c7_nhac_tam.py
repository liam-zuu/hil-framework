#!/usr/bin/env python3
"""
c7_nhac_tam.py: plate lift test of C8 (trang C8b v10 DA DUYET, P3 / A2, section 3 'Nhac tam') -> repeatability of the
car tag position and heading RELATIVE TO THE CAR BODY over 4 placements of the tag plate, the body watched through
textured tape patches stuck on the body.

Design: CLAUDE_THESIS 03_CODE/docs/2026-11-18_Thiet_ke_c7_nhac_tam_v1.md (CODE_NHACTAM local_f8bebd94, 18/11/2026).
Measurement (guide C8 ban 2, part C2): car standing still at O, control power off; N0 = the plate fitted on its pins
(C2.1), N1..N3 = lifted and put back (C2.2, C2.3); one clip of about 5 s per placement, nobody in the frame.

Per frame (design 2.1): marks of the session's own fit (session['fit_ids']) -> camera pose refitted
(c7_measure.refit_pose); car tag (ID 0) -> c7_measure.car_pose with tag_offset 0 (tag centre and heading, with the
parallax correction of the scoring); each body box: template of the middle frame of N0 matched (TM_CCOEFF_NORMED,
parabolic sub-pixel) within +-search px, its centre -> the plane z = h_than with the frame camera.
Per clip (2.2): 0.5 s dropped at both ends; a frame is used if the tag is seen, >= 8 marks and every box correlates
>= 0.9; median of the used frames; the clip must be still (largest deviation from the median below the thresholds).
Body motion (2.3): rigid 2D fit (c7_measure.rigid_fit, no scale) of the box points N0 -> Nj; residual = rigidity
check; kiem_than true when every Nj has >= 2 boxes and a residual within the threshold.
Repeatability (2.4): largest of the 6 pairs of |p_j - p_k| and |psi_j - psi_k|, raw (floor frame) and with the body
motion removed (tag pose mapped back to the N0 body frame); the value for U = max(raw, body removed); body removed
only when kiem_than.

  python tools/c7_nhac_tam.py khung --clip N0.MOV --session S/session.json --out D
        -> D/khung_N0.png (crop around the car, pixel grid, proposed boxes), D/khung_N0.json
  python tools/c7_nhac_tam.py do --clip N0.MOV N1.MOV N2.MOV N3.MOV --session S/session.json \
        --vung-than "u,v,w,h;u,v,w,h" [--h-than M] [--stride K] --out D
        -> D/nhac_tam.json, D/c8_phan_nhac_tam.json (block for c7_kiem_cheo xuat --phan nhac_tam=...),
           D/frames_nhac_tam.csv, D/bao_cao.txt
Boxes are pixel boxes (left, top, width, height) in the frames of N0. All thresholds are [DE XUAT].
Last line: #ket_qua,ma=0 passed / 1 computed with flags (clip moving, body moved, no body check) / 2 error.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import c7_measure as cm                                                  # noqa: E402

VERSION = 'c7_nhac_tam_v1'
N_PLACEMENTS = 4
DEFAULTS = dict(
    trim_s=0.5,            # 2.2: dropped at both ends of a clip
    min_marks=8,           # 2.2: floor marks for the frame camera
    corr_min=0.9,          # 2.2: every body box must correlate at least this
    min_frames=120,        # 2.2: used frames per clip (after stride)
    min_frac=0.6,          #      and share of the frames after the trim
    still_tag_m=0.0005,    # 2.2: largest deviation from the clip median, tag centre
    still_tag_deg=0.3,     #      tag heading
    still_box_m=0.0005,    #      each body box
    rigid_res_m=0.0003,    # 2.3: rms of the rigid fit of the box points N0 -> Nj
    body_move_m=0.001,     # 2.4: body pushed more than this (or the angle) -> redo C2 at the floor
    body_move_deg=0.2,
    search_px=25,          # 2.1: template search half-window
)
THRESH_STATUS = 'DE XUAT (thiet ke c7_nhac_tam v1 muc 2)'


# ---------------------------------------------------------------- small helpers
def parse_boxes(s):
    """'u,v,w,h;u,v,w,h' -> [(u, v, w, h)] ints, w, h >= 8."""
    out = []
    for part in [p for p in s.replace(' ', '').split(';') if p]:
        v = [int(round(float(x))) for x in part.split(',')]
        if len(v) != 4 or v[2] < 8 or v[3] < 8 or v[0] < 0 or v[1] < 0:
            raise ValueError(f'box {part!r}: need u,v,w,h with w, h >= 8')
        out.append(tuple(v))
    if not out:
        raise ValueError('no body box')
    return out


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def circ_median(angles):
    """Median of angles (rad) unwrapped around their circular mean."""
    a = np.asarray(angles, float)
    m = math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a)))
    return wrap(m + float(np.median([wrap(x - m) for x in a])))


def subpix(r, x, y):
    """Parabolic sub-pixel peak of a correlation map around the integer peak (x, y)."""
    def p(a, b, c):
        den = a - 2.0 * b + c
        return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    sx = p(r[y, x - 1], r[y, x], r[y, x + 1]) if 0 < x < r.shape[1] - 1 else 0.0
    sy = p(r[y - 1, x], r[y, x], r[y + 1, x]) if 0 < y < r.shape[0] - 1 else 0.0
    return x + sx, y + sy


class Box:
    """One body box: template from the reference frame, matched in later frames (design 2.1)."""

    def __init__(self, ref_gray, box, search):
        u, v, w, h = box
        H, W = ref_gray.shape
        if u - search < 0 or v - search < 0 or u + w + search > W or v + h + search > H:
            raise ValueError(f'box {box}: too close to the frame edge for a search of {search} px')
        self.box, self.search = box, search
        self.tpl = ref_gray[v:v + h, u:u + w].astype(np.float32).copy()
        if float(self.tpl.std()) < 2.0:
            raise ValueError(f'box {box}: no texture (grey std {self.tpl.std():.2f})')
        self.u0, self.v0 = u + (w - 1) / 2.0, v + (h - 1) / 2.0      # centre of the box in pixel coordinates

    def match(self, gray):
        """(u, v) of the box centre in this frame, correlation."""
        u, v, w, h = self.box
        s = self.search
        win = gray[v - s:v + h + s, u - s:u + w + s].astype(np.float32)
        r = cv2.matchTemplate(win, self.tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        x, y = subpix(r, *loc)
        x, y = self.ecc(win, x, y)
        return self.u0 + x - s, self.v0 + y - s, float(mx)

    def ecc(self, win, x, y):
        """Sub-pixel translation by ECC (cv2.findTransformECC) started at the correlation peak; the parabolic peak is
        kept when ECC fails or moves more than 1 px from it (pixel-locking of the parabola, design 2.1)."""
        warp = np.array([[1.0, 0.0, x], [0.0, 1.0, y]], np.float32)
        try:
            _, warp = cv2.findTransformECC(self.tpl, win, warp, cv2.MOTION_TRANSLATION,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 1)
        except cv2.error:
            return x, y
        ex, ey = float(warp[0, 2]), float(warp[1, 2])
        if not (math.isfinite(ex) and math.isfinite(ey)) or math.hypot(ex - x, ey - y) > 1.0:
            return x, y
        return ex, ey


def read_frame(video, idx):
    for k, t, g in cm.iter_frames(video):
        if k == idx:
            return g
    raise ValueError(f'{video}: no frame {idx}')


def frame_count(video):
    n = 0
    for _ in cm.iter_frames(video):
        n += 1
    return n


def git_head():
    try:
        c = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        d = subprocess.run(['git', '-C', HERE, 'status', '--porcelain', '--', os.path.basename(__file__)],
                           capture_output=True, text=True, timeout=10)
        return dict(commit=c.stdout.strip() or None, file_dirty=bool(d.stdout.strip()))
    except (OSError, subprocess.SubprocessError):
        return dict(commit=None, file_dirty=None)


# ---------------------------------------------------------------- per frame
def measure_clip(video, session, boxes, h_than, stride=1, max_frames=None):
    """Per-frame rows of one clip (design 2.1). boxes: [Box]."""
    cam = cm.cam_from_dict(session['camera'])
    marks = {int(i): tuple(v) for i, v in session['marks'].items()}
    fit_ids = [int(i) for i in session['fit_ids']]
    side, kappa, plumb = session['car_side_m'], session['kappa'], session['plumb_xy_m']
    det = cm.make_detector()
    rows = []
    for k, t, g in cm.iter_frames(video, max_frames):
        if k % stride:
            continue
        d = cm.detect_refined(det, g)
        obj, img, used = cm.floor_obs(d, marks, ids=fit_ids)
        r = dict(idx=k, t_s=t, n_marks=len(used), rms_px=float('nan'), tag_seen=False, tag_x=float('nan'),
                 tag_y=float('nan'), tag_psi=float('nan'), box=[])
        if len(used) >= 4:
            cam, r['rms_px'] = cm.refit_pose(cam, obj, img)
            if cm.CAR_ID in d:
                p = cm.car_pose(cam, d[cm.CAR_ID], side, kappa, plumb, (0.0, 0.0, 0.0))
                r.update(tag_seen=True, tag_x=p['tag_x'], tag_y=p['tag_y'], tag_psi=p['tag_psi'])
            for b in boxes:
                u, v, c = b.match(g)
                xy = cam.pix2floor(np.array([[u, v]]), h_than)[0]
                r['box'].append((float(xy[0]), float(xy[1]), c, u, v))
        rows.append(r)
    return rows


# ---------------------------------------------------------------- per clip
def clip_summary(rows, n_boxes, cfg):
    """Median pose of one clip, stillness, used frames (design 2.2)."""
    out = dict(n_frames=len(rows), flags=[])
    if not rows:
        out.update(ok=False, flags=['khong_co_khung'])
        return out
    t0, t1 = rows[0]['t_s'], rows[-1]['t_s']
    kept = [r for r in rows if t0 + cfg['trim_s'] <= r['t_s'] <= t1 - cfg['trim_s']]
    used = [r for r in kept if r['tag_seen'] and r['n_marks'] >= cfg['min_marks'] and len(r['box']) == n_boxes
            and all(b[2] >= cfg['corr_min'] for b in r['box'])]
    out.update(n_after_trim=len(kept), n_used=len(used))
    if len(used) < cfg['min_frames'] or len(used) < cfg['min_frac'] * max(1, len(kept)):
        out.update(ok=False, flags=['thieu_khung'])
        return out
    x = np.array([r['tag_x'] for r in used])
    y = np.array([r['tag_y'] for r in used])
    psi = [r['tag_psi'] for r in used]
    mx, my, mp = float(np.median(x)), float(np.median(y)), circ_median(psi)
    dev_pos = float(np.max(np.hypot(x - mx, y - my)))
    dev_ang = float(max(abs(math.degrees(wrap(a - mp))) for a in psi))
    bxy = np.array([[b[:2] for b in r['box']] for r in used])          # (n, boxes, 2)
    bmed = np.median(bxy, axis=0)
    bdev = np.max(np.linalg.norm(bxy - bmed[None], axis=2), axis=0) if n_boxes else np.zeros(0)
    corr_min = [float(min(r['box'][i][2] for r in used)) for i in range(n_boxes)]
    still = dev_pos <= cfg['still_tag_m'] and dev_ang <= cfg['still_tag_deg'] and bool(np.all(bdev <= cfg['still_box_m']))
    if not still:
        out['flags'].append('clip_co_dong')
    out.update(ok=True, still=still, tag=[mx, my, mp], tag_dev_m=dev_pos, tag_dev_deg=dev_ang,
               box=bmed.tolist(), box_dev_m=bdev.tolist(), corr_min=corr_min,
               rms_px_median=float(np.median([r['rms_px'] for r in used])))
    return out


# ---------------------------------------------------------------- body motion, repeatability
def body_motion(ref_pts, pts):
    """Rigid 2D motion of the body N0 -> Nj from the box points: (angle rad, t (2,), rms m) with pts ~ R ref + t."""
    ang, t, _, rms = cm.rigid_fit(np.asarray(ref_pts, float), np.asarray(pts, float))
    return ang, np.asarray(t, float), rms


def to_body0(tag, motion):
    """Tag pose of placement j mapped back to the body frame of N0: p' = R^T (p - t), psi' = psi - ang."""
    ang, t, _ = motion
    c, s = math.cos(ang), math.sin(ang)
    p = np.array(tag[:2]) - t
    return [c * p[0] + s * p[1], -s * p[0] + c * p[1], wrap(tag[2] - ang)]


def pairwise(poses):
    """Largest |p_j - p_k| (m) and |psi_j - psi_k| (deg) over all pairs, with the pair."""
    best_p, best_a, pp, pa = 0.0, 0.0, None, None
    for j, k in itertools.combinations(range(len(poses)), 2):
        dp = math.hypot(poses[j][0] - poses[k][0], poses[j][1] - poses[k][1])
        da = abs(math.degrees(wrap(poses[j][2] - poses[k][2])))
        if dp >= best_p:
            best_p, pp = dp, (j, k)
        if da >= best_a:
            best_a, pa = da, (j, k)
    return dict(vi_tri_m=best_p, cap_vi_tri=pp, goc_deg=best_a, cap_goc=pa)


def analyse(summaries, cfg):
    """Design 2.3, 2.4 on the 4 clip summaries (all ok). Returns the result dict and the flags."""
    flags = []
    raw = [s['tag'] for s in summaries]
    nb = len(summaries[0]['box'])
    motions, kiem_than, why = [], nb >= 2, [] if nb >= 2 else ['it_hon_2_vung']
    for j, s in enumerate(summaries):
        m = body_motion(summaries[0]['box'], s['box']) if nb >= 2 else (0.0, np.zeros(2), float('nan'))
        motions.append(m)
        if nb >= 2 and m[2] > cfg['rigid_res_m']:
            kiem_than = False
            why.append(f'N{j}_du_khop_{m[2] * 1000:.2f}mm')
    rep_raw = pairwise(raw)
    res = dict(tho=rep_raw, kiem_than=kiem_than, ly_do_khong_kiem_than=why)
    body = [dict(lan=f'N{j}', dich_m=float(np.hypot(*m[1])) if nb >= 2 else None,
                 quay_deg=math.degrees(m[0]) if nb >= 2 else None, du_khop_m=m[2] if nb >= 2 else None)
            for j, m in enumerate(motions)]
    res['than'] = body
    if kiem_than:
        corr = [to_body0(raw[j], motions[j]) for j in range(len(raw))]
        res['tru_than'] = pairwise(corr)
        res['tru_than']['tu_the'] = corr
        mv_m = max(b['dich_m'] for b in body)
        mv_d = max(abs(b['quay_deg']) for b in body)
        res.update(than_dich_max_m=mv_m, than_dich_max_deg=mv_d)
        if mv_m > cfg['body_move_m'] or mv_d > cfg['body_move_deg']:
            flags.append('than_dich_nhieu')
        res['vi_tri_m'] = max(rep_raw['vi_tri_m'], res['tru_than']['vi_tri_m'])
        res['goc_deg'] = max(rep_raw['goc_deg'], res['tru_than']['goc_deg'])
    else:
        flags.append('khong_kiem_than')
        res.update(tru_than=None, than_dich_max_m=None, than_dich_max_deg=None,
                   vi_tri_m=rep_raw['vi_tri_m'], goc_deg=rep_raw['goc_deg'])
    return res, flags


GIOI_HAN = ('4 lan dat la mau nho: khoang lon nhat cua 4 lan co the nho hon khoang that cua nhieu lan dat (khong nhan he '
            'so; le 1,25 do c8b_u cong). Lech tam-than do chot mon qua cac buoi chua bao. Do tren mot cho dat xe (goc O).')


def block(res, clips, session_path, session_sha):
    """Block nhac_tam for c8_cam_u_vao (C8B o U v10: nhac_tam.vi_tri_m, nhac_tam.goc_deg; other keys information)."""
    t = res['tru_than']
    return dict(
        vi_tri_m=res['vi_tri_m'], goc_deg=res['goc_deg'],
        vi_tri_tho_m=res['tho']['vi_tri_m'], goc_tho_deg=res['tho']['goc_deg'],
        vi_tri_tru_than_m=None if t is None else t['vi_tri_m'], goc_tru_than_deg=None if t is None else t['goc_deg'],
        kiem_than=res['kiem_than'], than_dich_max_m=res['than_dich_max_m'], than_dich_max_deg=res['than_dich_max_deg'],
        n_lan_dat=N_PLACEMENTS,
        cach='lon nhat qua 6 cap cua 4 lan dat (N0 gan theo chot, N1-N3 nhac dat lai); so dua vao U = lon nhat(tho, '
             'da tru than) (thiet ke c7_nhac_tam v1 muc 2.4)' + ('' if res['kiem_than'] else '; KHONG kiem than: chi so tho'),
        gioi_han=GIOI_HAN + ('' if res['kiem_than'] else ' Khong kiem duoc than: so tho gom ca than dich khi nhac tam.'),
        nguon=[dict(file=c['file'], sha256=c['sha256']) for c in clips] + [dict(file=session_path, sha256=session_sha)])


# ---------------------------------------------------------------- report
def report(out):
    L = [f'Nhấc tấm C8 ({out["version"]}): {out["ket_luan"]}']
    for c in out['clips']:
        s = c['tom_tat']
        if not s.get('ok'):
            L.append(f'  {c["lan"]} {c["file"]}: HỎNG ({", ".join(s["flags"])}; dùng {s.get("n_used", 0)} khung)')
            continue
        L.append(f'  {c["lan"]} {c["file"]}: {s["n_used"]} khung; mã lệch trong clip {s["tag_dev_m"] * 1000:.2f} mm, '
                 f'{s["tag_dev_deg"]:.3f}°; tương quan vùng nhỏ nhất {min(s["corr_min"]) if s["corr_min"] else float("nan"):.3f}'
                 + ('' if s['still'] else '  [CLIP CÓ ĐỘNG]'))
    r = out.get('ket_qua')
    if r:
        L.append(f'Thô (mã trên sàn): vị trí {r["tho"]["vi_tri_m"] * 1000:.2f} mm, góc {r["tho"]["goc_deg"]:.3f}°')
        if r['tru_than']:
            L.append(f'Đã trừ thân: vị trí {r["tru_than"]["vi_tri_m"] * 1000:.2f} mm, góc {r["tru_than"]["goc_deg"]:.3f}°; '
                     f'thân dịch lớn nhất {r["than_dich_max_m"] * 1000:.2f} mm, {r["than_dich_max_deg"]:.3f}°')
        else:
            L.append(f'Không kiểm được thân ({", ".join(r["ly_do_khong_kiem_than"])}): chỉ số thô')
        L.append(f'Đưa vào U: P3 = {r["vi_tri_m"] * 1000:.2f} mm, A2 = {r["goc_deg"]:.3f}°')
    if out['co']:
        L.append('Cờ: ' + ', '.join(out['co']))
        if 'than_dich_nhieu' in out['co'] or any('clip_co_dong' in f for f in out['co']):
            L.append('→ Đề nghị làm lại C2 (khoảng 5 phút).')
    return '\n'.join(L) + '\n'


FRAME_COLS = ('lan', 'idx', 't_s', 'n_marks', 'rms_px', 'tag_seen', 'tag_x', 'tag_y', 'tag_psi')


def write_frames(path, per_clip):
    nb = max((len(r['box']) for rows in per_clip for r in rows), default=0)
    cols = list(FRAME_COLS) + [f'vung{i}_{q}' for i in range(nb) for q in ('x', 'y', 'corr', 'u', 'v')]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for j, rows in enumerate(per_clip):
            for r in rows:
                v = [f'N{j}', r['idx'], f'{r["t_s"]:.6f}', r['n_marks'], f'{r["rms_px"]:.4f}', int(r['tag_seen']),
                     f'{r["tag_x"]:.7f}', f'{r["tag_y"]:.7f}', f'{r["tag_psi"]:.7f}']
                for b in r['box']:
                    v += [f'{b[0]:.7f}', f'{b[1]:.7f}', f'{b[2]:.5f}', f'{b[3]:.4f}', f'{b[4]:.4f}']
                v += [''] * (len(cols) - len(v))
                w.writerow(v)


# ---------------------------------------------------------------- commands
def run_do(clips, session_path, boxes_spec, out_dir, h_than=None, stride=1, max_frames=None, cfg=None):
    """Whole analysis; writes the outputs; returns the exit code (0, 1, 2)."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    if len(clips) != N_PLACEMENTS:
        raise ValueError(f'need exactly {N_PLACEMENTS} clips in placement order (N0 N1 N2 N3), got {len(clips)}')
    cm._new_dir(out_dir)
    with open(session_path, encoding='utf-8') as f:
        session = json.load(f)
    h = float(session['tag_h_m'] if h_than is None else h_than)
    boxes_px = parse_boxes(boxes_spec)
    mid = frame_count(clips[0]) // 2
    ref = read_frame(clips[0], mid)
    boxes = [Box(ref, b, cfg['search_px']) for b in boxes_px]
    per_clip, clip_out = [], []
    for j, v in enumerate(clips):
        rows = measure_clip(v, session, boxes, h, stride, max_frames)
        per_clip.append(rows)
        clip_out.append(dict(lan=f'N{j}', file=os.path.basename(v), sha256=cm.sha256(v),
                             tom_tat=clip_summary(rows, len(boxes), cfg)))
    flags = [f'{c["lan"]}_{x}' for c in clip_out for x in c['tom_tat']['flags']]
    out = dict(version=VERSION, session=os.path.abspath(session_path), session_sha256=cm.sha256(session_path),
               hil=git_head(), h_than_m=h, h_than_src='tag_h_m cua session' if h_than is None else '--h-than',
               vung_than_px=[list(b) for b in boxes_px], khung_mau=dict(clip='N0', idx=mid), stride=stride,
               nguong=dict(cfg, trang_thai=THRESH_STATUS), clips=clip_out, ket_qua=None, co=flags)
    write_frames(os.path.join(out_dir, 'frames_nhac_tam.csv'), per_clip)
    if not all(c['tom_tat']['ok'] for c in clip_out):
        out['ket_luan'] = 'LỖI: có clip không đủ khung'
        code = 2
    else:
        res, f2 = analyse([c['tom_tat'] for c in clip_out], cfg)
        out['ket_qua'] = _jsonable(res)
        out['co'] = flags + f2
        code = 1 if out['co'] else 0
        out['ket_luan'] = 'đạt' if code == 0 else 'tính xong, có cờ'
        blk = block(res, clip_out, out['session'], out['session_sha256'])
        with open(os.path.join(out_dir, 'c8_phan_nhac_tam.json'), 'w', encoding='utf-8') as f:
            json.dump(_jsonable(blk), f, ensure_ascii=False, indent=1)
    out['ma'] = code
    with open(os.path.join(out_dir, 'nhac_tam.json'), 'w', encoding='utf-8') as f:
        json.dump(_jsonable(out), f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, 'bao_cao.txt'), 'w', encoding='utf-8') as f:
        f.write(report(out))
    return code


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, tuple):
        return [_jsonable(v) for v in o]
    return cm._json(o) if not isinstance(o, (dict, list)) else (
        {str(k): _jsonable(v) for k, v in o.items()} if isinstance(o, dict) else [_jsonable(v) for v in o])


def propose_boxes(gray, tag_c, tag_side_px, n=4, size=40, exclude=0.75, reach=3.0):
    """Boxes (u, v, w, h) of strong texture around the car tag, outside a circle of exclude x tag side and within
    reach x tag side; greedy, at least one box size apart. Suggestions only: the scorer picks the tape boxes."""
    g = gray.astype(np.float32)
    ev = cv2.cornerMinEigenVal(g, 3)
    score = cv2.boxFilter(ev, -1, (size, size))
    H, W = g.shape
    vv, uu = np.mgrid[0:H, 0:W]
    rr = np.hypot(uu - tag_c[0], vv - tag_c[1])
    mask = (rr > exclude * tag_side_px + size * 0.71) & (rr < reach * tag_side_px)
    mask &= (uu > size) & (uu < W - size) & (vv > size) & (vv < H - size)
    score = np.where(mask, score, -1.0)
    out = []
    for _ in range(n):
        k = int(np.argmax(score))
        v, u = divmod(k, W)
        if score[v, u] <= 0:
            break
        out.append((int(u - size // 2), int(v - size // 2), size, size))
        score[max(0, v - size):v + size, max(0, u - size):u + size] = -1.0
    return out


def run_khung(clip, session_path, out_dir):
    cm._new_dir(out_dir)
    mid = frame_count(clip) // 2
    g = read_frame(clip, mid)
    d = cm.detect_refined(cm.make_detector(), g)
    if cm.CAR_ID not in d:
        raise ValueError(f'{clip}: car tag not seen in frame {mid}')
    c = d[cm.CAR_ID]
    ctr = c.mean(axis=0)
    side = float(np.mean(np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)))
    half = int(3.2 * side)
    u0, v0 = max(0, int(ctr[0]) - half), max(0, int(ctr[1]) - half)
    u1, v1 = min(g.shape[1], int(ctr[0]) + half), min(g.shape[0], int(ctr[1]) + half)
    crop = g[v0:v1, u0:u1]
    boxes = [(u + u0, v + v0, w, h) for u, v, w, h in propose_boxes(crop, ctr - [u0, v0], side)]
    img = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    for x in range((u0 // 50 + 1) * 50, u1, 50):
        cv2.line(img, (2 * (x - u0), 0), (2 * (x - u0), img.shape[0]), (0, 160, 255), 1)
        cv2.putText(img, str(x), (2 * (x - u0) + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 160, 255), 1)
    for y in range((v0 // 50 + 1) * 50, v1, 50):
        cv2.line(img, (0, 2 * (y - v0)), (img.shape[1], 2 * (y - v0)), (0, 160, 255), 1)
        cv2.putText(img, str(y), (2, 2 * (y - v0) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 160, 255), 1)
    for i, (u, v, w, h) in enumerate(boxes):
        cv2.rectangle(img, (2 * (u - u0), 2 * (v - v0)), (2 * (u + w - u0), 2 * (v + h - v0)), (0, 255, 0), 2)
        cv2.putText(img, f'de xuat {i}', (2 * (u - u0), 2 * (v - v0) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(out_dir, 'khung_N0.png'), img)
    info = dict(version=VERSION, clip=os.path.basename(clip), sha256=cm.sha256(clip), idx=mid,
                tag_centre_px=ctr.tolist(), tag_side_px=side, crop_px=[u0, v0, u1, v1], anh_phong=2,
                o_de_xuat=[list(b) for b in boxes],
                ghi_chu='O de xuat chi la goi y (ket cau manh); chon dung o bang dinh tren than, khong tren tam ma')
    with open(os.path.join(out_dir, 'khung_N0.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('khung')
    a1.add_argument('--clip', required=True)
    a1.add_argument('--session', required=True)
    a1.add_argument('--out', required=True)
    a2 = sub.add_parser('do')
    a2.add_argument('--clip', nargs='+', required=True)
    a2.add_argument('--session', required=True)
    a2.add_argument('--vung-than', required=True)
    a2.add_argument('--h-than', type=float)
    a2.add_argument('--stride', type=int, default=1)
    a2.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'khung':
            info = run_khung(a.clip, a.session, a.out)
            print(json.dumps(info, ensure_ascii=False))
            code = 0
        else:
            if a.stride < 1:
                raise ValueError('--stride >= 1')
            cfg = None
            if a.stride > 1:            # the frame count scales with the stride (design 3: at the floor --stride 3)
                cfg = dict(min_frames=max(20, int(math.ceil(DEFAULTS['min_frames'] / a.stride))))
            code = run_do(a.clip, a.session, a.vung_than, a.out, a.h_than, a.stride, cfg=cfg)
            with open(os.path.join(a.out, 'bao_cao.txt'), encoding='utf-8') as f:
                sys.stdout.write(f.read())
    except (OSError, ValueError, KeyError) as e:
        print(f'LOI: {e}', file=sys.stderr)
        code = 2
    print(f'#ket_qua,ma={code}')
    return code


if __name__ == '__main__':
    sys.exit(main())
