# ── per-edge evidence (design v4) ────────────────────────────────────────────────────────────────

def refine_edges_diag(img, corners, half=6.0, skip=0.18, n_samp=24, iters=2):
    """c7_measure.refine_edges, line for line, plus the evidence of each side at the last iteration: profiles kept
    (ok), points kept after the 2.5 sigma rejection, RMS of the kept points about the line (px), median edge contrast
    (peak of the outward derivative), median spread of the derivative (px, edge width). Returns (corners, diag) or
    (None, None) exactly when refine_edges returns None."""
    c = np.asarray(corners, float).copy()
    f32 = img if img.dtype == np.float32 else img.astype(np.float32)
    ss = np.arange(-half, half + 0.25, 0.5)
    sm = (ss[:-1] + ss[1:]) / 2.0
    diag = None
    for _ in range(iters):
        lines, diag = [], []
        for k in range(4):
            p0, p1 = c[k], c[(k + 1) % 4]
            d = p1 - p0
            length = float(np.hypot(*d))
            if length < 8.0:
                return None, None
            d = d / length
            nrm = np.array([d[1], -d[0]])
            fr = np.linspace(skip, 1.0 - skip, n_samp)
            base = p0[None, :] + fr[:, None] * length * d[None, :]
            px = (base[:, None, 0] + ss[None, :] * nrm[0]).astype(np.float32)
            py = (base[:, None, 1] + ss[None, :] * nrm[1]).astype(np.float32)
            prof = cv2.remap(f32, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            g = np.clip(np.diff(prof, axis=1), 0.0, None)
            peak = g.max(axis=1)
            g = np.where(g >= 0.2 * g.max(axis=1, keepdims=True), g, 0.0)
            wsum = g.sum(axis=1)
            ok = wsum > 1e-6
            if ok.sum() < 6:
                return None, None
            sstar = (g[ok] * sm[None, :]).sum(axis=1) / wsum[ok]
            spread = np.sqrt(np.maximum((g[ok] * (sm[None, :] - sstar[:, None]) ** 2).sum(axis=1) / wsum[ok], 0.0))
            pts = base[ok] + sstar[:, None] * nrm[None, :]
            for _ in range(2):
                m = pts.mean(axis=0)
                _, _, vt = np.linalg.svd(pts - m)
                dirv, nv = vt[0], vt[1]
                r = (pts - m) @ nv
                keep = np.abs(r) <= max(2.5 * r.std(), 0.05)
                if keep.all() or keep.sum() < 6:
                    break
                pts = pts[keep]
            r = (pts - m) @ nv
            lines.append((m, dirv))
            diag.append(dict(ok=int(ok.sum()), kept=int(len(pts)), rms_px=float(np.sqrt(np.mean(r ** 2))),
                             contrast=float(np.median(peak[ok])), width_px=float(np.median(spread))))
        new = np.zeros((4, 2))
        for k in range(4):
            (m1, d1), (m2, d2) = lines[k - 1], lines[k]
            a = np.column_stack([d1, -d2])
            if abs(np.linalg.det(a)) < 1e-9:
                return None, None
            tt = np.linalg.solve(a, m2 - m1)
            new[k] = m1 + tt[0] * d1
        if np.max(np.linalg.norm(new - c, axis=1)) > 4.0 + half:
            return None, None
        c = new
    return c, diag


def detect_diag(det, gray):
    """c7_measure.detect_refined for the car tag only, with the per-side evidence: (corners, diag) or (None, why)."""
    found = cp.detect_markers(det, gray)
    if CAR_ID not in found:
        return None, 'khong_do_lai'
    c0 = found[CAR_ID]
    side = float(np.mean(np.linalg.norm(np.diff(np.vstack([c0, c0[:1]]), axis=0), axis=1)))
    half = min(cm.HALF_CAR_PX, 0.75 * side / 6.0)
    q, dg = refine_edges_diag(gray, c0, half=half)
    if q is None:
        return None, 'refine_none'
    return q, dg


def edge_summary(dg):
    """Per frame: worst side by kept points, largest line RMS, smallest contrast ratio (side / median of the 4)."""
    kept = min(d['kept'] for d in dg)
    rms = max(d['rms_px'] for d in dg)
    con = np.array([d['contrast'] for d in dg])
    ratio = float(con.min() / np.median(con)) if np.median(con) > 0 else 0.0
    wid = np.array([d['width_px'] for d in dg])
    wr = float(wid.max() / np.median(wid)) if np.median(wid) > 0 else math.inf
    return dict(kept_min=kept, rms_max_px=rms, contrast_ratio_min=ratio, width_ratio_max=wr)


KEPT_MIN = 20               # per side: points kept of 24 profiles [DE XUAT]
RMS_MAX_PX = 0.5            # per side: RMS of the edge points about the line [DE XUAT]


def edge_vec(dg):
    """8 photometric numbers of a frame, sides in the tag's own order: log(contrast_j / median contrast) and
    log(width_j / median width), j = 0..3 (ArUco side order is fixed on the tag)."""
    con = np.array([d['contrast'] for d in dg], float)
    wid = np.array([d['width_px'] for d in dg], float)
    return np.r_[np.log(con / np.median(con)), np.log(wid / np.median(wid))]


def edge_basic_ok(dg):
    return min(d['kept'] for d in dg) >= KEPT_MIN and max(d['rms_px'] for d in dg) <= RMS_MAX_PX


def edge_pattern(v, X):
    """X ~ [1, v, v^2] P (3 x 8), Huber weights on the norm of the residual (as huber_pattern)."""
    A = np.column_stack([np.ones_like(v), v, v ** 2])
    w = np.ones(len(v))
    for _ in range(HUBER_IT):
        P, *_ = np.linalg.lstsq(A * w[:, None], X * w[:, None], rcond=None)
        r = np.linalg.norm(X - A @ P, axis=1)
        s = 1.4826 * float(np.median(r)) + 1e-12
        w = np.sqrt(np.minimum(1.0, HUBER_K * s / np.maximum(r, 1e-15)))
    return P


def r_canh(x, v, P):
    """Largest photometric residual of a side after the speed pattern."""
    x = np.asarray(x).reshape(-1, 8)
    v = np.atleast_1d(v)
    return np.max(np.abs(x - np.column_stack([np.ones_like(v), v, v ** 2]) @ P), axis=1)
