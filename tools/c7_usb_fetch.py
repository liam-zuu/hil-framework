#!/usr/bin/env python3
"""
c7_usb_fetch.py: C7/C8 floor video, copy each new iPhone clip to the laptop over the USB cable and
run the C7 measurement on it (Liam 13/11/2026, side session CODE_CAM local_377135de, choice "(a) Tu
chep clip (Recommended)": the laptop finds the new clip on the iPhone, copies it into
05_DATA/results/C7_VIDEO/<date>_.../ and runs the analysis of that run).

Feasibility measured 13/11 (iPhone 17, short cable, Windows 11; notes of CODE_CAM): Windows sees the
phone as a portable device (WPD) "Apple iPhone" > "Internal Storage" > one folder per month
(202611_a, ...) > IMG_xxxx (no drive letter, no DCIM folder). Empty until the phone is unlocked and
"Trust" is tapped. A 6 s 4K 60 fps clip (about 49 MB) copies in about 1.7 s; the clip shows on the
laptop a few seconds after Stop; reading continued while the phone was locked for about 30 s.

Device access goes through the Windows Shell (Shell.Application COM) in a PowerShell child process:
no extra Python package. The phone is only READ: nothing is deleted, renamed or written on it.

Copy rule: the clip is copied into <out>/.incoming/, accepted only when the local size equals the
size the phone reports (System.Size), then renamed to <out>/<prefix><NN>_<IMG_xxxx>.<ext>; an
existing file is never overwritten. A clip is taken only when its reported size is the same on two
polls in a row (the phone finished writing it). Each copy adds one row to <out>/manifest_usb.csv
(name, bytes, SHA-256, creation time on the phone, copy time): the video itself is not tracked by
git (L88), the manifest is.

  python tools/c7_usb_fetch.py list [--n 10]
  python tools/c7_usb_fetch.py get   --name IMG_3544 --out <dir> [--session S --tag-offset ...]
  python tools/c7_usb_fetch.py watch --out <dir> [--session <calib dir>/session.json]
                                     [--tag-offset DX,DY,DPSI_DEG] [--poll 2] [--max-clips N]
                                     [--prefix run]
watch: clips already on the phone when it starts are ignored; every new clip is copied and, with
--session, analysed by `c7_measure.py track` into <out>/<stem>_track/. Ctrl+C stops.
Frame check (Liam 13/11 CODE_CAM "Them kiem khung hinh"; --no-check skips it): on the first frame of
each clip, count the ArUco marks (floor 1-24, check 25-33, car 0) with the c7_measure detector; with
--session, compare the floor-mark centres with clip H (missing marks, median shift > 5 px = the
camera may have moved: WARNING); without, warn below 8 floor marks. Writes <stem>_khung.jpg (marks
drawn, red crosses = clip H) and <stem>_khung.json; result in the manifest column frame_check.
  python tools/c7_usb_fetch.py check --video <clip> [--session S]
Last line: #ket_qua,ma=0 done / 2 error.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXT = ('.mov', '.mp4', '.m4v')
MANIFEST = 'manifest_usb.csv'
MANIFEST_COLS = ['n', 'src_folder', 'src_name', 'bytes', 'sha256', 'created_utc', 'copied_local',
                 'copy_s', 'dest', 'frame_check', 'measure_rc']
# frame check (Liam 13/11 CODE_CAM "Them kiem khung hinh") [DE XUAT, lock at C8]
CHECK_SHIFT_PX = 5.0          # median shift of the floor-mark centres from clip H: camera moved
CHECK_MIN_FLOOR = 8           # without a session: fewer floor marks than the frame gate of c7_measure
THUMB_W = 1280

# ---------------------------------------------------------------- PowerShell (Windows Shell COM)
_PS_COMMON = r'''
$ErrorActionPreference = 'Stop'
$sh = New-Object -ComObject Shell.Application
$ip = $sh.NameSpace(17).Items() | Where-Object { $_.Name -eq 'Apple iPhone' } | Select-Object -First 1
if (-not $ip) { Write-Output '{"error":"NO_DEVICE"}'; exit 0 }
$st = $ip.GetFolder.Items() | Select-Object -First 1
if (-not $st) { Write-Output '{"error":"LOCKED_OR_UNTRUSTED"}'; exit 0 }
$top = @($st.GetFolder.Items())
if ($top.Count -eq 0) { Write-Output '{"error":"LOCKED_OR_UNTRUSTED"}'; exit 0 }
'''

_PS_LIST = _PS_COMMON + r'''
$minf = $env:C7U_MIN_FOLDER
$folders = @($top | Where-Object { $_.IsFolder } | ForEach-Object { $_.Name } | Sort-Object)
$scan = @($folders | Where-Object { [string]::CompareOrdinal($_, $minf) -ge 0 })
$out = New-Object System.Collections.Generic.List[object]
foreach ($f in $scan) {
  $fo = ($top | Where-Object { $_.Name -eq $f }).GetFolder
  foreach ($it in @($fo.Items())) {
    $fn = [string]$it.ExtendedProperty('System.FileName')
    $dc = $it.ExtendedProperty('System.DateCreated')
    $out.Add([pscustomobject]@{ folder = $f; name = [string]$it.Name; file = $fn;
      size = [int64]$it.ExtendedProperty('System.Size');
      created_utc = $(if ($dc) { ([datetime]$dc).ToString('yyyy-MM-ddTHH:mm:ss') } else { '' }) })
  }
}
[pscustomobject]@{ folders = $folders; items = $out.ToArray() } | ConvertTo-Json -Depth 4 -Compress
'''

_PS_COPY = _PS_COMMON + r'''
$fo = ($top | Where-Object { $_.Name -eq $env:C7U_FOLDER }).GetFolder
$it = @($fo.Items()) | Where-Object { $_.Name -eq $env:C7U_NAME } | Select-Object -First 1
if (-not $it) { Write-Output '{"error":"NOT_FOUND"}'; exit 0 }
$want = [int64]$it.ExtendedProperty('System.Size')
$fn = [string]$it.ExtendedProperty('System.FileName')
$dst = $env:C7U_DST
$sw = [Diagnostics.Stopwatch]::StartNew()
$sh.NameSpace($dst).CopyHere($it, 0x414)
$p = Join-Path $dst $fn
$tmo = [double]$env:C7U_TIMEOUT
while ($sw.Elapsed.TotalSeconds -lt $tmo) {
  if ((Test-Path $p) -and ((Get-Item $p).Length -eq $want)) { break }
  Start-Sleep -Milliseconds 100
}
$got = $(if (Test-Path $p) { (Get-Item $p).Length } else { -1 })
[pscustomobject]@{ path = $p; want = $want; got = $got; copy_s = $sw.Elapsed.TotalSeconds } | ConvertTo-Json -Compress
'''


def _run_ps(script, env_extra, timeout):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    enc = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    r = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-EncodedCommand', enc],
                       capture_output=True, text=True, env=env, timeout=timeout)
    txt = r.stdout.strip().splitlines()
    if r.returncode != 0 or not txt:
        raise OSError(f'PowerShell rc={r.returncode}: {r.stderr.strip()[:300]}')
    return json.loads(txt[-1])


class DeviceError(OSError):
    pass


class ShellDevice:
    """The iPhone through the Windows Shell. Read only."""

    def list(self, min_folder=''):
        d = _run_ps(_PS_LIST, {'C7U_MIN_FOLDER': min_folder}, timeout=60)
        if 'error' in d:
            raise DeviceError(d['error'])
        items = d.get('items') or []
        if isinstance(items, dict):            # ConvertTo-Json writes a lone object without []
            items = [items]
        folders = d.get('folders') or []
        if isinstance(folders, str):
            folders = [folders]
        return folders, items

    def copy(self, item, dst_dir, timeout_s=600):
        d = _run_ps(_PS_COPY, {'C7U_FOLDER': item['folder'], 'C7U_NAME': item['name'],
                               'C7U_DST': os.path.abspath(dst_dir), 'C7U_TIMEOUT': timeout_s},
                    timeout=timeout_s + 60)
        if 'error' in d:
            raise DeviceError(d['error'])
        return d


# ---------------------------------------------------------------- pure logic (tested)
def is_video(item):
    return os.path.splitext(item.get('file') or '')[1].lower() in VIDEO_EXT


def key(item):
    return (item['folder'], item['name'])


class NewClipTracker:
    """Clips present at start are 'seen'. A new video is ready when its reported size is > 0 and
    equal on two polls in a row."""

    def __init__(self, baseline):
        self.seen = {key(i) for i in baseline}
        self.last_size = {}

    def poll(self, items):
        ready = []
        for it in items:
            k = key(it)
            if k in self.seen or not is_video(it):
                continue
            s = int(it.get('size') or 0)
            if s > 0 and self.last_size.get(k) == s:
                ready.append(it)
                self.seen.add(k)
                self.last_size.pop(k, None)
            else:
                self.last_size[k] = s
        ready.sort(key=lambda i: (i['folder'], i['name']))
        return ready


def latest_folder(folders):
    return max(folders) if folders else ''


def next_index(out_dir):
    p = os.path.join(out_dir, MANIFEST)
    if not os.path.exists(p):
        return 1
    with open(p, newline='', encoding='utf-8') as f:
        ns = [int(r['n']) for r in csv.DictReader(f) if r.get('n')]
    return max(ns, default=0) + 1


def dest_name(prefix, n, item):
    ext = os.path.splitext(item.get('file') or '')[1] or '.MOV'
    return f'{prefix}{n:02d}_{item["name"]}{ext}'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def append_manifest(out_dir, row):
    p = os.path.join(out_dir, MANIFEST)
    new = not os.path.exists(p)
    with open(p, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, '') for c in MANIFEST_COLS})


def fetch_one(dev, item, out_dir, n, prefix='run'):
    """Copy one clip, check the size, rename, hash. Returns the manifest row (without measure_rc)."""
    final = os.path.join(out_dir, dest_name(prefix, n, item))
    if os.path.exists(final):
        raise FileExistsError(final)
    inc = os.path.join(out_dir, '.incoming')
    os.makedirs(inc, exist_ok=True)
    staged = os.path.join(inc, item.get('file') or item['name'])
    if os.path.exists(staged):
        raise FileExistsError(f'{staged} (left from an earlier copy; check and move it away)')
    d = dev.copy(item, inc)
    want = int(item.get('size') or 0)
    got = os.path.getsize(staged) if os.path.exists(staged) else -1
    if got != int(d.get('want', want)) or got <= 0:
        raise OSError(f'copy incomplete: {item["name"]} {got} of {d.get("want", want)} bytes')
    os.replace(staged, final)
    return {'n': n, 'src_folder': item['folder'], 'src_name': item['name'], 'bytes': got,
            'sha256': sha256(final), 'created_utc': item.get('created_utc', ''),
            'copied_local': dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            'copy_s': f'{float(d.get("copy_s", 0.0)):.2f}', 'dest': os.path.basename(final)}


def run_measure(video, session, tag_offset, out_dir):
    """c7_measure.py track on one clip. Returns (rc, last line)."""
    stem = os.path.splitext(os.path.basename(video))[0]
    cmd = [sys.executable, os.path.join(HERE, 'c7_measure.py'), 'track', '--video', video,
           '--session', session, '--out', os.path.join(out_dir, f'{stem}_track')]
    if tag_offset:
        cmd += ['--tag-offset', tag_offset]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = (r.stdout or '').strip().splitlines()
    return r.returncode, (lines[-1] if lines else (r.stderr or '').strip()[-200:])


def _load_session(session):
    if not session:
        return None
    with open(session, encoding='utf-8') as f:
        return json.load(f)


def _median(v):
    v = sorted(v)
    m = len(v) // 2
    return float(v[m]) if len(v) % 2 else float((v[m - 1] + v[m]) / 2)


def check_marks(seen, shape, sess=None):
    """Verdict on one frame. seen = {id: (x, y) centre px}, shape = (h, w). Pure (tested).
    Returns dict(verdict 'OK' / 'CANH_BAO', reasons, counts, shift_px, missing)."""
    floor = sorted(i for i in seen if 1 <= i <= 24)
    check = sorted(i for i in seen if 25 <= i <= 33)
    out = dict(n_floor=len(floor), n_check=len(check), car_seen=0 in seen, floor_ids=floor,
               shift_px=None, missing=[], reasons=[])
    if sess is None:
        if len(floor) < CHECK_MIN_FLOOR:
            out['reasons'].append(f'chi thay {len(floor)} moc san (< {CHECK_MIN_FLOOR})')
    else:
        if (int(sess['height']), int(sess['width'])) != tuple(shape):
            out['reasons'].append(f'kich thuoc khung {shape[1]}x{shape[0]} khac clip H '
                                  f'{sess["width"]}x{sess["height"]}')
        ref = {int(k): v for k, v in sess['ref_centres_px'].items()}
        out['missing'] = sorted(i for i in ref if i not in seen)
        common = [i for i in ref if i in seen]
        if out['missing']:
            out['reasons'].append(f'thieu {len(out["missing"])} moc so voi clip H: {out["missing"]}')
        if common:
            out['shift_px'] = _median([((seen[i][0] - ref[i][0]) ** 2 + (seen[i][1] - ref[i][1]) ** 2) ** 0.5
                                       for i in common])
            if out['shift_px'] > CHECK_SHIFT_PX:
                out['reasons'].append(f'moc lech {out["shift_px"]:.1f} px so voi clip H '
                                      f'(> {CHECK_SHIFT_PX:g}): may co the da xe dich')
        else:
            out['reasons'].append('khong thay moc nao cua clip H')
    out['verdict'] = 'CANH_BAO' if out['reasons'] else 'OK'
    return out


def frame_check(video, session=None, out_dir=None):
    """First frame of the clip: ArUco marks seen, shift from clip H; writes <stem>_khung.jpg (marks
    drawn, red crosses = clip H centres) and <stem>_khung.json. Same reading as c7_measure (auto
    rotation off, its detector)."""
    import cv2                                                   # heavy, only here
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import c7_measure as cm
    cap = cv2.VideoCapture(video, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise OSError(f'OpenCV cannot open {video}')
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise OSError(f'{video}: no frame read')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    stats = {}
    found = cm.cp.detect_markers(cm.make_detector(), gray, stats)   # same gate as c7_measure
    seen = {i: c.mean(axis=0).tolist() for i, c in found.items()}
    sess = session if (session is None or isinstance(session, dict)) else _load_session(session)
    res = check_marks(seen, gray.shape, sess)
    res['dropped'] = stats
    res['video'] = os.path.basename(video)
    res['ref'] = 'clip H (session.json)' if sess else 'khong co session: chi dem moc'
    res['thresholds'] = dict(shift_px=CHECK_SHIFT_PX, min_floor=CHECK_MIN_FLOOR, status='DE XUAT, khoa o C8')
    out_dir = out_dir or os.path.dirname(os.path.abspath(video))
    stem = os.path.splitext(os.path.basename(video))[0]
    img = (frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)).copy()
    if found:
        import numpy as np
        cv2.aruco.drawDetectedMarkers(img, [c.reshape(1, 4, 2).astype(np.float32) for c in found.values()],
                                      np.array(list(found), np.int32).reshape(-1, 1))
    if sess:
        for x, y in sess['ref_centres_px'].values():
            cv2.drawMarker(img, (int(round(x)), int(round(y))), (0, 0, 255), cv2.MARKER_CROSS, 40, 4)
    if img.shape[1] > THUMB_W:
        sc = THUMB_W / img.shape[1]
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    color = (0, 160, 0) if res['verdict'] == 'OK' else (0, 0, 255)
    txt = f'{res["verdict"]} san {res["n_floor"]} kiem {res["n_check"]} xe {"co" if res["car_seen"] else "khong"}'
    if res['shift_px'] is not None:
        txt += f' lech {res["shift_px"]:.1f}px'
    cv2.putText(img, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(img, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, f'{stem}_khung.jpg'), img)
    with open(os.path.join(out_dir, f'{stem}_khung.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    return res


def handle(dev, item, out_dir, prefix, session, tag_offset, measure=run_measure, log=print, check=None):
    n = next_index(out_dir)
    row = fetch_one(dev, item, out_dir, n, prefix)
    log(f'[{n:02d}] {item["name"]} -> {row["dest"]}  {int(row["bytes"]) / 1e6:.1f} MB  '
        f'chep {row["copy_s"]} s')
    if check:
        try:
            r = check(os.path.join(out_dir, row['dest']), session, out_dir)
            row['frame_check'] = r['verdict']
            msg = f'san {r["n_floor"]}, kiem {r["n_check"]}, xe {"co" if r["car_seen"] else "khong"}'
            if r.get('shift_px') is not None:
                msg += f', lech {r["shift_px"]:.1f} px'
            log(f'     khung hinh: {r["verdict"]} ({msg})')
            for x in r['reasons']:
                log(f'       !! {x}')
        except (OSError, ValueError, KeyError) as e:
            row['frame_check'] = 'LOI'
            log(f'     khung hinh: LOI ({e})')
    if session:
        rc, last = measure(os.path.join(out_dir, row['dest']), session, tag_offset, out_dir)
        row['measure_rc'] = rc
        log(f'     phan tich: rc={rc}  {last}')
    append_manifest(out_dir, row)
    return row


def watch(dev, out_dir, prefix='run', session=None, tag_offset=None, poll_s=2.0, max_clips=None,
          sleep=time.sleep, measure=run_measure, log=print, max_polls=None, check=None):
    folders, _ = dev.list('~')          # '~' sorts after every folder name: folder list only
    minf = latest_folder(folders)
    _, items = dev.list(minf)
    tr = NewClipTracker(items)
    log(f'Dang cho clip moi (bo qua {len(items)} muc da co; thu muc tu {minf}). Ctrl+C de dung.')
    done, polls = [], 0
    while max_clips is None or len(done) < max_clips:
        if max_polls is not None and polls >= max_polls:
            break
        sleep(poll_s)
        polls += 1
        try:
            folders, items = dev.list(minf)
        except DeviceError as e:
            log(f'  (iPhone chua doc duoc: {e}; mo khoa iPhone neu can)')
            continue
        for it in tr.poll(items):
            done.append(handle(dev, it, out_dir, prefix, session, tag_offset, measure, log, check))
            if max_clips is not None and len(done) >= max_clips:
                break
    return done


def find_clip(dev, name, max_folders=6):
    """Look for a clip from the newest month folder backwards (listing every folder is slow)."""
    folders = sorted(dev.list('~')[0], reverse=True)[:max_folders]
    for f in folders:
        _, items = dev.list(f)
        hit = [i for i in items if i['name'] == name and is_video(i)]
        if hit:
            return hit[-1]
    return None


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a0 = sub.add_parser('list', help='list the newest videos on the phone')
    a0.add_argument('--n', type=int, default=10)
    ac = sub.add_parser('check', help='frame check of a clip already on the laptop')
    ac.add_argument('--video', required=True)
    ac.add_argument('--session', default=None)
    for name in ('get', 'watch'):
        a = sub.add_parser(name)
        a.add_argument('--out', required=True, help='e.g. 05_DATA/results/C7_VIDEO/<date>_<what>/')
        a.add_argument('--prefix', default='run')
        a.add_argument('--session', default=None, help='session.json of c7_measure calib: run the analysis')
        a.add_argument('--tag-offset', default=None, help='DX,DY,DPSI_DEG, passed to c7_measure track')
        a.add_argument('--no-check', action='store_true', help='skip the frame check of each clip')
        if name == 'get':
            a.add_argument('--name', required=True, help='IMG_xxxx as listed')
        else:
            a.add_argument('--poll', type=float, default=2.0)
            a.add_argument('--max-clips', type=int, default=None)
    args = ap.parse_args(argv)
    dev = ShellDevice()
    try:
        if args.cmd == 'check':
            r = frame_check(args.video, _load_session(args.session))
            print(f'{r["verdict"]}: san {r["n_floor"]}, kiem {r["n_check"]}, xe {r["car_seen"]}, '
                  f'lech {r["shift_px"]} px')
            for x in r['reasons']:
                print(f'  !! {x}')
        elif args.cmd == 'list':
            folders, items = dev.list(latest_folder(dev.list('~')[0]))
            vids = sorted((i for i in items if is_video(i)), key=lambda i: (i['folder'], i['name']))
            for i in vids[-args.n:]:
                print(f'{i["folder"]}/{i["name"]}  {i["file"]}  {int(i["size"]) / 1e6:.1f} MB  '
                      f'tao {i["created_utc"]} UTC')
        else:
            if args.session and not os.path.exists(args.session):
                raise FileNotFoundError(args.session)
            os.makedirs(args.out, exist_ok=True)
            if args.cmd == 'get':
                hit = find_clip(dev, args.name)
                if not hit:
                    raise FileNotFoundError(f'{args.name} not on the phone')
                handle(dev, hit, args.out, args.prefix, args.session, args.tag_offset,
                       check=None if args.no_check else frame_check)
            else:
                try:
                    watch(dev, args.out, args.prefix, args.session, args.tag_offset, args.poll,
                          args.max_clips, check=None if args.no_check else frame_check)
                except KeyboardInterrupt:
                    print('Dung theo Ctrl+C.')
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f'LOI: {exc}')
        print('#ket_qua,ma=2')
        return 2
    print('#ket_qua,ma=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
