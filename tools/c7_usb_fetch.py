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
                 'copy_s', 'dest', 'measure_rc']

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


def handle(dev, item, out_dir, prefix, session, tag_offset, measure=run_measure, log=print):
    n = next_index(out_dir)
    row = fetch_one(dev, item, out_dir, n, prefix)
    log(f'[{n:02d}] {item["name"]} -> {row["dest"]}  {int(row["bytes"]) / 1e6:.1f} MB  '
        f'chep {row["copy_s"]} s')
    if session:
        rc, last = measure(os.path.join(out_dir, row['dest']), session, tag_offset, out_dir)
        row['measure_rc'] = rc
        log(f'     phan tich: rc={rc}  {last}')
    append_manifest(out_dir, row)
    return row


def watch(dev, out_dir, prefix='run', session=None, tag_offset=None, poll_s=2.0, max_clips=None,
          sleep=time.sleep, measure=run_measure, log=print, max_polls=None):
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
            done.append(handle(dev, it, out_dir, prefix, session, tag_offset, measure, log))
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
    for name in ('get', 'watch'):
        a = sub.add_parser(name)
        a.add_argument('--out', required=True, help='e.g. 05_DATA/results/C7_VIDEO/<date>_<what>/')
        a.add_argument('--prefix', default='run')
        a.add_argument('--session', default=None, help='session.json of c7_measure calib: run the analysis')
        a.add_argument('--tag-offset', default=None, help='DX,DY,DPSI_DEG, passed to c7_measure track')
        if name == 'get':
            a.add_argument('--name', required=True, help='IMG_xxxx as listed')
        else:
            a.add_argument('--poll', type=float, default=2.0)
            a.add_argument('--max-clips', type=int, default=None)
    args = ap.parse_args(argv)
    dev = ShellDevice()
    try:
        if args.cmd == 'list':
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
                handle(dev, hit, args.out, args.prefix, args.session, args.tag_offset)
            else:
                try:
                    watch(dev, args.out, args.prefix, args.session, args.tag_offset, args.poll,
                          args.max_clips)
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
