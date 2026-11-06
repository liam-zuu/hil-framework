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
(that repository's core.autocrlf and attributes; Codex S3C-16), or, without git, git's
text=auto rule for core.autocrlf = true (NUL or lone CR: stored as is; else CRLF -> LF); so
the value equals `git rev-parse HEAD:<file>` whenever the file is unchanged. Limit (S3C-17,
deferred): the file is read just after the import; a change in between, or a stale .pyc, is
not detected. Any other change,
even a docstring, gives another blob (L34: before main numbers both sides go to one merged
commit).

  python rpi5/plant_blob.py      # prints the blobs of the files next to this script
"""
from __future__ import annotations

import hashlib
import os

FILES = ('plant_step', 'params')        # module names, as rpi5/ imports them


def python_blob(data: bytes) -> str:
    """Blob without git: git's text=auto rule for core.autocrlf = true (convert.c): a file with
    a NUL byte or a lone CR is binary and stored as is, otherwise CRLF is stored as LF
    (Codex S3C-16). Equal to git only under that configuration."""
    lone_cr = data.count(b'\r') != data.count(b'\r\n')
    if b'\0' not in data and not lone_cr:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def git_blob_with_method(path: str) -> tuple:
    """(blob, method): `git -C <dir> hash-object -- <file>`, which applies that repository's
    configuration and attributes, or python_blob() when git is missing or fails."""
    import subprocess
    d, name = os.path.split(os.path.abspath(path))
    try:
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
