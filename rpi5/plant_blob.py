"""
plant_blob.py: which plant source a run really used (STATUS_CODE L34, Fable F-ND #7, Liam
03/11 CODE session: "Luật + công cụ").

A SIM run (fwsim) and a HIL run are compared only when both used the same
rpi5/plant_step.py and rpi5/params.py, judged by the git blob of each file (the value of
`git rev-parse <ref>:rpi5/plant_step.py`), not by a commit: two commits can hold the same
plant, and a commit says nothing about a file changed after it. So the blob is computed when
the run starts, from the file the process has imported, and written with the run (fwsim:
meta.yaml 'plant'; HIL: the frame-log manifest and the run log).

The working-tree file is hashed as its repository stores it: `git -C <dir> hash-object <file>`
(that repository's core.autocrlf and attributes; Codex S3C-16), only when <dir> is inside a
git work tree (S3D-5: outside a repository git would use the global configuration); else
git's text=auto rule for core.autocrlf = true (convert.c: binary when there is a NUL, a lone
CR, or (printable >> 7) < non-printable, stored as is; else CRLF -> LF; S3C-16); so the value
equals `git rev-parse HEAD:<file>` whenever the file is unchanged. Limit (S3C-17, deferred):
the file is read at the time of the call (the plant process: just after the import; the
--step-log JSON of run_hil.py: in the parent before the plant process starts and again after
it, both written, flag plant_src_changed when they differ, S3D-3); a change between the read
and the import, or a stale .pyc, is not detected. Any other change,
even a docstring, gives another blob (L34: before main numbers both sides go to one merged
commit).

  python rpi5/plant_blob.py      # prints the blobs of the files next to this script
"""
from __future__ import annotations

import hashlib
import os

FILES = ('plant_step', 'params')        # module names, as rpi5/ imports them


def git_is_binary(data: bytes) -> bool:
    """git's convert_is_binary(gather_stats()) (convert.c): a lone CR, a NUL, or fewer than one
    printable byte in 128 per non-printable byte (BS, HT, ESC, FF count printable; DEL and other
    bytes < 32 non-printable; a final ^Z not counted). Codex S3C-16 round 2."""
    printable = nonprintable = 0
    i, n = 0, len(data)
    while i < n:
        c = data[i]
        if c == 13:
            if i + 1 < n and data[i + 1] == 10:
                i += 2
                continue
            return True                                   # lone CR
        if c == 10:
            pass
        elif c == 0:
            return True                                   # NUL
        elif c == 127 or (c < 32 and c not in (8, 9, 27, 12)):
            nonprintable += 1
        else:
            printable += 1
        i += 1
    if n and data[-1] == 26:
        nonprintable -= 1
    return (printable >> 7) < nonprintable


def python_blob(data: bytes) -> str:
    """Blob without git: git's text=auto rule for core.autocrlf = true (convert.c): a file git
    takes as binary (git_is_binary) is stored as is, otherwise CRLF is stored as LF (Codex
    S3C-16). Equal to git only under that configuration."""
    if not git_is_binary(data):
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def git_blob_with_method(path: str) -> tuple:
    """(blob, method): `git -C <dir> hash-object -- <file>`, which applies that repository's
    configuration and attributes, when <dir> is inside a git work tree (S3D-5); python_blob()
    when git is missing, fails, or <dir> is not in a repository."""
    import subprocess
    d, name = os.path.split(os.path.abspath(path))
    try:
        w = subprocess.run(['git', '-C', d, 'rev-parse', '--is-inside-work-tree'], capture_output=True,
                           text=True, timeout=10)
        if w.returncode != 0 or w.stdout.strip() != 'true':
            raise OSError('not inside a git work tree')
        r = subprocess.run(['git', '-C', d, 'hash-object', '--', name], capture_output=True,
                           text=True, timeout=10)
        out = r.stdout.strip()
        if r.returncode == 0 and len(out) == 40 and all(c in '0123456789abcdef' for c in out):
            return out, 'git hash-object'
    except (OSError, subprocess.SubprocessError):
        pass
    with open(path, 'rb') as f:
        return python_blob(f.read()), 'python (no git): autocrlf=true rule'


def git_blob(path: str) -> str:
    """Git blob id of a file as its repository would store it (git_blob_with_method)."""
    return git_blob_with_method(path)[0]


def plant_source() -> dict:
    """{'plant_step_py': blob, 'params_py': blob, 'method': how} of the modules this process
    imports (sys.modules if already imported, else imported now from sys.path)."""
    import importlib
    out, how = {}, set()
    for name in FILES:
        mod = importlib.import_module(name)
        out[f'{name}_py'], m = git_blob_with_method(os.path.abspath(mod.__file__))
        how.add(m)
    out['method'] = ' / '.join(sorted(how))
    return out


def same_plant(a: dict, b: dict) -> bool:
    """True when two plant_source() records name the same blob for every file (L34: a SIM and
    a HIL run are compared only then; otherwise write down why, and do not compare)."""
    keys = [f'{n}_py' for n in FILES]
    return all(a.get(k) and a.get(k) == b.get(k) for k in keys)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for k, v in plant_source().items():
        print(f'{k}: {v}')
