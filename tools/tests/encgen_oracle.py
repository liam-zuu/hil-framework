"""RS-4 oracle (L98, 13/11/2026): the unchanged H7 encoder_gen.c on the PC (stm32h7/pc_tests/encgen,
profile mode) gives the count the ESP32 PCNT sees at any time for wheel speeds applied at given times, so
tests of tools/hil_edge_map.py get their counts from the H7 code itself, not from the mapper's model (Codex
Sol G32-12, G32-R2-4). Needs gcc on PATH."""
import os
import shutil
import subprocess
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
H7 = os.path.join(ROOT, 'stm32h7')
SIM = os.path.join(H7, 'pc_tests', 'encgen')
_EXE = {}


def build(tmp=None):
    """Path of the profile-mode simulator built from the current encoder_gen.c (cached per process)."""
    if 'exe' in _EXE:
        return _EXE['exe']
    tmp = tmp or tempfile.mkdtemp(prefix='encgen_oracle_')
    for f in (os.path.join(H7, 'Core', 'Inc', 'encoder_gen.h'), os.path.join(SIM, 'stm32h7xx_hal.h'),
              os.path.join(SIM, 'encgen_sim_main.c'), os.path.join(H7, 'Core', 'Src', 'encoder_gen.c')):
        shutil.copy(f, tmp)
    exe = os.path.join(tmp, 'encgen_sim.exe')
    r = subprocess.run(['gcc', '-std=gnu11', '-O2', '-Wall', '-Werror', '-I', tmp,
                        os.path.join(tmp, 'encgen_sim_main.c'), os.path.join(tmp, 'encoder_gen.c'), '-lm', '-o', exe],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    _EXE['exe'] = exe
    return exe


def available():
    return shutil.which('gcc') is not None


class Counts:
    """Decoded count per wheel as a step function of the timer tick (1 us)."""

    def __init__(self, edges):
        self.t = [np.array([e[0] for e in edges[w]], dtype=np.int64) for w in range(4)]
        self.c = [np.array([e[1] for e in edges[w]], dtype=np.int64) for w in range(4)]

    def at(self, tick):
        """Counts (len(tick), 4) seen at the given ticks (edges at a tick count as seen)."""
        tick = np.asarray(tick, dtype=np.int64)
        out = np.zeros((len(tick), 4), dtype=np.int64)
        for w in range(4):
            if len(self.t[w]):
                j = np.searchsorted(self.t[w], tick, side='right') - 1
                out[:, w] = np.where(j >= 0, self.c[w][np.clip(j, 0, None)], 0)
        return out


def run(apply_ticks, omega, seed=1):
    """Counts of the H7 EncoderGen when every wheel's speed omega[i] (rad/s, (n, 4)) is applied at
    apply_ticks[i] (us, increasing)."""
    apply_ticks = np.asarray(apply_ticks, dtype=np.int64)
    omega = np.asarray(omega, dtype=float)
    # %.9g keeps a float32 value exact, so the C float of SetOmega equals the H7's (h7_omega32)
    lines = [str(len(apply_ticks))] + [f'{int(t)} {w[0]:.9g} {w[1]:.9g} {w[2]:.9g} {w[3]:.9g}'
                                       for t, w in zip(apply_ticks, omega)]
    r = subprocess.run([build(), 'profile', str(seed)], input='\n'.join(lines) + '\n', capture_output=True,
                       text=True, check=True)
    edges = [[], [], [], []]
    for line in r.stdout.splitlines():
        if line.startswith('E '):
            _, w, t, c = line.split()
            edges[int(w)].append((int(t), int(c)))
    return Counts(edges)
