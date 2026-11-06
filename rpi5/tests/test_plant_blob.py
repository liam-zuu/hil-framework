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


def _raw_blob(data):
    import hashlib
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def test_python_rule_crlf_lone_cr_nul():
    """Fallback rule (git text=auto with core.autocrlf = true): CRLF stored as LF; a lone CR or a
    NUL makes the file binary, stored as is (Codex S3C-16)."""
    assert plant_blob.python_blob(b'a = 1\r\nb = 2\r\n') == _raw_blob(b'a = 1\nb = 2\n')
    lone = b'k=1\r\n# tail\r'
    assert plant_blob.python_blob(lone) == _raw_blob(lone)
    nul = b'k=1\r\n\0'
    assert plant_blob.python_blob(nul) == _raw_blob(nul)
    assert plant_blob.python_blob(b'"""doc"""\nk = 1\n') != plant_blob.python_blob(b'"""doc."""\nk = 1\n')


@pytest.mark.skipif(shutil.which('git') is None, reason='git not on PATH')
@pytest.mark.parametrize('autocrlf', ['true', 'false'])
def test_blob_follows_the_repository_configuration(tmp_path, autocrlf):
    """In a repository, git_blob is git's own value for that repository's core.autocrlf, for a
    CRLF file and a file with a lone CR (S3C-16); with autocrlf true it equals the fallback."""
    repo = tmp_path / 'r'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'core.autocrlf', autocrlf], check=True)
    for name, data in (('crlf.py', b'a = 1\r\nb = 2\r\n'), ('lone.py', b'k=1\r\n# tail\r')):
        p = repo / name
        p.write_bytes(data)
        want = subprocess.run(['git', '-C', str(repo), 'hash-object', name], capture_output=True, text=True,
                              check=True).stdout.strip()
        blob, how = plant_blob.git_blob_with_method(str(p))
        assert blob == want and how == 'git hash-object'
        if autocrlf == 'true':
            assert blob == plant_blob.python_blob(data)


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
    assert set(man['plant_src']) == {'plant_step_py', 'params_py', 'method'}
