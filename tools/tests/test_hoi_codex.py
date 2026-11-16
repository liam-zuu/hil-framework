"""Tests of tools/stra_a/hoi_codex.py (design 03_CODE/docs/2026-11-15_Thiet_ke_CODE_C13_v1.md section 4, 8).
Codex is replaced by a function in the test, never by an environment variable."""
import csv
import hashlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'stra_a'))
import hoi_codex as H  # noqa: E402

GOOD_LOG = ('OpenAI Codex v0.163.0\n--------\nworkdir: D:\\STRA_A\nmodel: gpt-6.1-sol\nprovider: openai\n'
            'approval: never\nsandbox: custom permissions\nreasoning effort: high\n--------\n')
# D:\STRA_A\.claude\settings.json deny patterns (design 4 step 8)
BLOCKED = ('.codex', 'CLAUDE_THESIS', 'HIL_WT', 'HIL_BACKUP', 'TUNE_A', '.claude/projects', 'Claude-Data', '--cd',
           ' -C ', 'sandbox_mode', 'default_permissions', 'dangerously', 'danger-full-access')


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / 'STRA_A'
    (r / '.codex').mkdir(parents=True)
    (r / '.codex' / 'config.toml').write_text('cfg', encoding='utf-8')
    (r / 'CODEX').mkdir()
    (r / 'HOP_THU' / 'luot' / 'A-PID__A1__007').mkdir(parents=True)
    monkeypatch.setattr(H, 'ROOT', str(r))
    monkeypatch.setattr(H, 'SHA_CODEX_CONFIG', hashlib.sha256(b'cfg').hexdigest())
    exe = tmp_path / 'codex.exe'
    exe.write_bytes(b'x')
    monkeypatch.setattr(H, 'CODEX_EXE', str(exe))
    monkeypatch.setattr(H, 'CODEX_SHA256', hashlib.sha256(b'x').hexdigest())
    monkeypatch.setattr(H, 'codex_version', lambda e: H.CODEX_VERSION)
    calls = []

    def fake(cmd, stdin, cwd, log=GOOD_LOG, code=0, answer='Đề xuất: kp_xy 1,8\n', stdout=''):
        calls.append((cmd, stdin, cwd))
        out = cmd[cmd.index('-o') + 1]
        if answer is not None:
            with open(out, 'w', encoding='utf-8') as f:
                f.write(answer)
        fix = lambda x: x.replace('workdir: D:\\STRA_A', 'workdir: ' + str(r)).encode('utf-8')  # noqa: E731
        return code, fix(stdout), fix(log)
    monkeypatch.setattr(H, 'run_codex', fake)
    yield r, calls
    os.chdir(HERE)


def test_good_call_prints_answer(root, capsys):
    r, calls = root
    assert H.main(['--cau', 'Lượt 7 lệch ngang nhiều, nên đổi gì?', '--luot', 'A-PID__A1__007']) == 0
    out = capsys.readouterr().out
    assert 'kp_xy 1,8' in out and '#ket_qua,ma=0' in out
    cmd, stdin, cwd = calls[0]
    assert cwd == str(r) and cmd[0] == H.CODEX_EXE and cmd[1:9] == ['exec', '--skip-git-repo-check', '-m',
                                                                     'gpt-6.1-sol', '-c', 'model_reasoning_effort=high',
                                                                     '-o', cmd[8]]
    assert os.path.basename(cmd[8]).startswith('.tam_tra_loi_')                  # SA-21: temp, then published
    assert not os.path.exists(cmd[8]) and any(n.startswith('tra_loi_') for n in os.listdir(r / 'CODEX'))
    q = (r / 'CODEX' / 'hoi_' ).parent
    qs = [n for n in os.listdir(q) if n.startswith('hoi_')]
    assert len(qs) == 1 and (q / qs[0]).read_bytes() == stdin
    assert 'Lượt 7 lệch ngang nhiều'.encode('utf-8') in stdin and b'mien_A.csv' in stdin
    rows = list(csv.DictReader(open(r / 'CODEX' / 'thoi_gian.csv', encoding='utf-8')))
    assert rows[0]['model'] == 'gpt-6.1-sol' and rows[0]['effort'] == 'high' and rows[0]['luot'] == 'A-PID__A1__007'


def test_numbering_never_overwrites(root):
    r, _ = root
    for _ in range(3):
        assert H.main(['--thu']) == 0
    names = sorted(n for n in os.listdir(r / 'CODEX') if n.startswith(('hoi_', 'tra_loi_', 'log_')))
    assert not any(n.startswith('.tam_') for n in os.listdir(r / 'CODEX'))
    assert len(names) == 9 and len({os.path.splitext(n)[0].rsplit('_', 1)[-1] for n in names}) == 3


@pytest.mark.parametrize('log', [GOOD_LOG.replace('sandbox: custom permissions', 'sandbox: danger-full-access'),
                                 GOOD_LOG.replace('gpt-6.1-sol', 'gpt-5'),
                                 GOOD_LOG.replace('reasoning effort: high', 'reasoning effort: xhigh'),
                                 GOOD_LOG.replace('workdir: D:\\STRA_A', 'workdir: D:\\CLAUDE_THESIS')])
def test_wrong_lock_hides_answer(root, capsys, monkeypatch, log):
    r, _ = root
    f = H.run_codex
    monkeypatch.setattr(H, 'run_codex', lambda c, s, w: f(c, s, w, log=log))
    assert H.main(['--thu']) == 2
    out = capsys.readouterr().out
    assert 'kp_xy' not in out and 'không chạy đúng khóa' in out
    assert any(n.startswith('.tam_tra_loi_') for n in os.listdir(r / 'CODEX'))     # kept, not published


def test_lock_lines_after_the_header_do_not_count(root, capsys, monkeypatch):
    """SA-2: wrong header; the right lines only in the echoed prompt or the answer."""
    bad = GOOD_LOG.replace('sandbox: custom permissions', 'sandbox: danger-full-access') + 'user\n' + GOOD_LOG
    f = H.run_codex
    monkeypatch.setattr(H, 'run_codex', lambda c, s, w: f(c, s, w, log=bad))
    assert H.main(['--thu']) == 2 and 'kp_xy' not in capsys.readouterr().out
    two = GOOD_LOG.replace('model: gpt-6.1-sol\n', 'model: gpt-6.1-sol\nmodel: gpt-5\n')
    monkeypatch.setattr(H, 'run_codex', lambda c, s, w: f(c, s, w, log=two))
    assert H.main(['--thu']) == 2
    wrong = GOOD_LOG.replace('sandbox: custom permissions', 'sandbox: danger-full-access')
    monkeypatch.setattr(H, 'run_codex', lambda c, s, w: f(c, s, w, log=wrong, stdout=GOOD_LOG))   # right header in stdout
    assert H.main(['--thu']) == 2 and 'kp_xy' not in capsys.readouterr().out


def test_pinned_program_and_version(root, capsys, monkeypatch):
    """SA-1: the program is the pinned path, not PATH; another version is refused before the call."""
    r, calls = root
    monkeypatch.setenv('PATH', str(r))
    monkeypatch.setattr(H, 'codex_version', lambda e: '0.167.0-alpha.4.1')
    assert H.main(['--thu']) == 2 and 'thử rào' in capsys.readouterr().out
    monkeypatch.setattr(H, 'codex_version', lambda e: H.CODEX_VERSION)            # right version, other program
    (r / '..' / 'codex.exe').write_bytes(b'khac')
    assert H.main(['--thu']) == 2 and 'SHA' in capsys.readouterr().out
    monkeypatch.setattr(H, 'CODEX_EXE', str(r / 'khong_co.exe'))
    assert H.main(['--thu']) == 2
    assert calls == []


def test_timeout_is_exit_2(root, capsys, monkeypatch):
    def slow(c, s, w):
        raise H.subprocess.TimeoutExpired(c, 1)
    monkeypatch.setattr(H, 'run_codex', slow)
    assert H.main(['--thu']) == 2 and '#ket_qua,ma=2' in capsys.readouterr().out


def test_codex_folder_link_refused(root, capsys, tmp_path):
    r, calls = root
    os.rename(r / 'CODEX', tmp_path / 'CODEX_ngoai')
    try:
        os.symlink(tmp_path / 'CODEX_ngoai', r / 'CODEX', target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('no symlink right on this machine')
    assert H.main(['--thu']) == 2 and calls == []


def test_exit_code_hides_answer(root, capsys, monkeypatch):
    f = H.run_codex
    monkeypatch.setattr(H, 'run_codex', lambda c, s, w: f(c, s, w, code=1))
    assert H.main(['--thu']) == 2 and 'kp_xy' not in capsys.readouterr().out


def test_refuses_before_call(root, capsys, tmp_path):
    r, calls = root
    assert H.main(['--cau', 'x', '--luot', '..\\..\\x']) == 2
    assert H.main(['--cau', 'x', '--luot', 'khong_co']) == 2
    out = tmp_path / 'ngoai.md'
    out.write_text('x', encoding='utf-8')
    assert H.main(['--cau', 'x', '--doc', str(out)]) == 2
    (r / '.codex' / 'config.toml').write_text('đổi', encoding='utf-8')
    assert H.main(['--thu']) == 2
    assert calls == [] and not any(n.startswith('hoi_') for n in os.listdir(r / 'CODEX'))


def test_no_passthrough_arguments(root):
    for bad in (['--thu', '-s', 'danger-full-access'], ['--thu', '-C', 'D:\\'], ['--thu', '--cd', 'x'],
                ['--thu', '-c', 'sandbox_mode=x'], ['--thu', '--dangerously-bypass-approvals-and-sandbox'],
                ['--thu', '--lu', 'x']):
        with pytest.raises(SystemExit):
            H.main(bad)


def test_command_line_passes_stra_a_fence():
    for line in ('python D:\\STRA_A\\hoi_codex.py --cau "Lượt 7 nên đổi gì?" --luot A-PID__A1__007',
                 'python D:\\STRA_A\\hoi_codex.py --thu'):
        assert not any(b in line for b in BLOCKED)
