"""
plant_blob.py: which plant source a run really used (STATUS_CODE L34, Fable F-ND #7, Liam
03/11 CODE session: "Luật + công cụ").

A SIM run (fwsim) and a HIL run are compared only when both used the same
rpi5/plant_step.py and rpi5/params.py, judged by the git blob of each file (the value of
`git rev-parse <ref>:rpi5/plant_step.py`), not by a commit: two commits can hold the same
plant, and a commit says nothing about a file changed after it. So the blob is computed when
the run starts, from the file the process has imported, and written with the run (fwsim:
meta.yaml 'plant'; HIL: the frame-log manifest and the run log).

The working-tree bytes are hashed as git stores them: CRLF -> LF for a text file (the
Windows checkouts have core.autocrlf = true; on the Pi the file is LF already), so the
value equals `git rev-parse HEAD:<file>` whenever the file is unchanged. Any other change,
even a docstring, gives another blob (L34: before main numbers both sides go to one merged
commit).

  python rpi5/plant_blob.py      # prints the blobs of the files next to this script
"""
from __future__ import annotations

import hashlib
import os

FILES = ('plant_step', 'params')        # module names, as rpi5/ imports them


def git_blob(path: str) -> str:
    """SHA-1 git blob id of a file, CRLF stored as LF for text (no NUL byte), as git with
    core.autocrlf = true stores it."""
    with open(path, 'rb') as f:
        data = f.read()
    if b'\0' not in data:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def plant_source() -> dict:
    """{'plant_step_py': blob, 'params_py': blob} of the modules this process imports
    (sys.modules if already imported, else imported now from sys.path)."""
    import importlib
    out = {}
    for name in FILES:
        mod = importlib.import_module(name)
        out[f'{name}_py'] = git_blob(os.path.abspath(mod.__file__))
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
