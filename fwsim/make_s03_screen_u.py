"""make_s03_screen_u.py: writes fwsim/s03_screen_u.csv, the committed root numbers of the S03
screening step (spec S03 v8.6 section 3.1b item 3, section 8 condition 9): five sets S0, S1, S2,
S2b, S3 of independent uniform numbers in [0, 1); set t of rerun r is
default_rng(SEED_BASE + 10*r + t).random((n, 7)) (PCG64); the tool-check file (8.11) uses
TRIAL_SEED_BASE = SEED_BASE + 5 (seeds +5..+9, never a main seed 10*r + t). The files are
checked at every S03 start by s03_sangloc.verify_screen_source (the numbers are regenerated in
memory and compared bit for bit and as the canonical text; Codex C3-36). The committed file
is the reference: this script refuses to overwrite an existing file; it writes a temp file,
reads it back with s03_sangloc.parse_screen_u and compares every u bit for bit before
os.replace. CLI: python fwsim/make_s03_screen_u.py [--r R] [--out PATH] [--thu].
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s03_sangloc as sl                                # noqa: E402


def generate(r=0, base=sl.SEED_BASE):
    """s03_sangloc.generate_sets(r, base): set t of SETS is default_rng(base + 10*r + t) (PCG64);
    the same call always gives the same numbers on every machine."""
    return sl.generate_sets(r, base)


def write(path, r=0, base=sl.SEED_BASE):
    """Write the numbers file at path and return the JSON-able generation record. Refuses when
    path exists (the committed file is the reference, never generated again); the temp file is
    written exactly like s3.make_test_points (header SCREEN_COLS, rows [set, j, repr(u)...]),
    parsed back with s03_sangloc.parse_screen_u and every u compared bit for bit with the drawn
    array before os.replace; a mismatch deletes the temp file and raises."""
    if os.path.exists(path):
        raise FileExistsError(f'{path} already exists: the committed file is the reference; '
                              'it is never generated again')
    sets = generate(r, base)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(sl.SCREEN_COLS)
            for s in sl.SETS:
                for j, u in enumerate(sets[s]):
                    w.writerow([s, j + 1] + [repr(float(x)) for x in u])
        with open(tmp, encoding='utf-8') as f:
            back = sl.parse_screen_u(f.read())
        if not all(struct.pack('<d', float(back[s][j, k])) == struct.pack('<d', float(sets[s][j, k]))
                   for s in sl.SETS for j in range(len(sets[s])) for k in range(7)):
            raise RuntimeError(f'{tmp}: the file read back differs from the drawn array')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    with open(path, 'rb') as f:
        raw = f.read()
    return dict(path=os.path.abspath(path), r=r, base=base,
                seeds={s: base + 10 * r + t for t, s in enumerate(sl.SETS)},
                rows={s: int(len(sets[s])) for s in sl.SETS},
                size=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                sha256_lf=hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest(),
                python=platform.python_version(), numpy=np.__version__,
                bit_generator=np.random.default_rng(0).bit_generator.__class__.__name__)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--r', type=int, default=0, help='rerun index; 0 is the committed reference')
    ap.add_argument('--out', default=None, help='output path (default s03_screen_u.csv for --r 0, '
                    's03_screen_u_r<R>.csv for --r >= 1, s03_screen_u_thu.csv for --thu)')
    ap.add_argument('--thu', action='store_true', help='tool-check file (8.11): seeds +5..+9, '
                    'never a main seed; not with --r')
    a = ap.parse_args(argv)
    if a.thu and a.r:
        ap.error('--thu uses its own seeds, not --r')
    base = sl.TRIAL_SEED_BASE if a.thu else sl.SEED_BASE
    if a.out is None:
        if a.thu:
            out = sl.SCREEN_U_THU
        else:
            out = sl.SCREEN_U if a.r == 0 else os.path.join(HERE, f's03_screen_u_r{a.r}.csv')
    else:
        out = a.out
    print(json.dumps(write(out, r=0 if a.thu else a.r, base=base), indent=1))


if __name__ == '__main__':
    main()
