"""plant_blob (STATUS_CODE L34): git blobs of plant_step.py / params.py computed at run time,
written into the HIL frame-log manifest. Run: cd rpi5 && python -m pytest tests"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RPI5 = os.path.dirname(HERE)
sys.path.insert(0, RPI5)
import plant_blob  # noqa: E402
from plant_blob import git_blob, plant_source, same_plant  # noqa: E402


def test_crlf_and_lf_give_the_same_blob(tmp_path):
    a, b = tmp_path / 'lf.py', tmp_path / 'crlf.py'
    a.write_bytes(b'a = 1\nb = 2\n')
    b.write_bytes(b'a = 1\r\nb = 2\r\n')
    assert git_blob(str(a)) == git_blob(str(b))


def test_any_change_changes_the_blob(tmp_path):
    a, b = tmp_path / 'a.py', tmp_path / 'b.py'
    a.write_bytes(b'"""doc"""\nk = 1\n')
    b.write_bytes(b'"""doc."""\nk = 1\n')
    assert git_blob(str(a)) != git_blob(str(b))


@pytest.mark.skipif(shutil.which('git') is None, reason='git not on PATH')
def test_blob_matches_git_hash_object(tmp_path):
    """Same value as git itself for a text file (hash-object without filters, LF content)."""
    p = tmp_path / 'f.py'
    p.write_bytes(b'print("hil")\n')
    want = subprocess.run(['git', 'hash-object', '--no-filters', str(p)], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert git_blob(str(p)) == want


@pytest.mark.skipif(shutil.which('git') is None, reason='git not on PATH')
def test_unchanged_files_match_head():
    """In a clean checkout the run-time blob equals git rev-parse HEAD:rpi5/<file> (L34)."""
    src = plant_source()
    for name in plant_blob.FILES:
        rel = f'rpi5/{name}.py'
        st = subprocess.run(['git', '-C', RPI5, 'status', '--porcelain', '--', f'{name}.py'],
                            capture_output=True, text=True)
        if st.returncode != 0 or st.stdout.strip():
            pytest.skip(f'{rel} changed in this checkout or not in git')
        head = subprocess.run(['git', '-C', RPI5, 'rev-parse', f'HEAD:{rel}'], capture_output=True,
                              text=True, check=True).stdout.strip()
        assert src[f'{name}_py'] == head


def test_same_plant():
    a = dict(plant_step_py='1' * 40, params_py='2' * 40)
    assert same_plant(a, dict(a))
    assert not same_plant(a, dict(a, params_py='3' * 40))
    assert not same_plant({}, {})


def test_frame_log_manifest_has_plant_src(tmp_path):
    from hil_node import HILConfig, HILNode
    from mecanum_plant import MecanumPlant
    p = str(tmp_path / 'f.npz')
    cfg = HILConfig(simulate_hw=True, max_steps=20, period_us=1_000, frame_log=p)
    node = HILNode(MecanumPlant(dt=0.001, seed=0), config=cfg)
    node.run()
    man = json.loads(str(np.load(p)['manifest']))
    assert man['plant_src'] == plant_source()
    assert set(man['plant_src']) == {'plant_step_py', 'params_py'}
