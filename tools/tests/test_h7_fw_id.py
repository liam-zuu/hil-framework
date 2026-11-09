"""H7 firmware id (#fw, Liam 09/11, CODE; STATUS_CODE L61 b): stm32h7/scripts/fw_id.py, the CubeIDE
pre-build step that writes stm32h7/Core/Inc/fw_id_gen.h, and the main.c lines that print it."""
import importlib.util
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
spec = importlib.util.spec_from_file_location('fw_id', os.path.join(ROOT, 'stm32h7', 'scripts', 'fw_id.py'))
fw_id = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fw_id)


def test_header_text_matches_bench_regex():
    import sys
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import hil_bench_session as hb
    t = fw_id.header_text('1a2b3c4d5e6f', 0, 'Debug')
    vals = dict(re.findall(r'#define (FW_\w+)\s+"?([^"\n]+)"?', t))
    assert vals == {'FW_GIT_SHA': '1a2b3c4d5e6f', 'FW_GIT_DIRTY': '0', 'FW_ENV': 'Debug'}
    line = f"#fw sha={vals['FW_GIT_SHA']} dirty={vals['FW_GIT_DIRTY']} env={vals['FW_ENV']}"   # main.c format
    assert hb.fw_lines([line]) == [dict(sha='1a2b3c4d5e6f', dirty=0, env='Debug')]


def test_main_writes_only_on_change(tmp_path, monkeypatch):
    out = tmp_path / 'fw_id_gen.h'
    monkeypatch.setattr(fw_id, 'OUT', str(out))
    monkeypatch.setattr(fw_id, 'fw_id', lambda cfg: ('aaaaaaaaaaaa', 0, cfg))
    assert fw_id.main(['Release']) == 0
    assert '"Release"' in out.read_text(encoding='utf-8')
    os.utime(out, (1, 1))
    fw_id.main(['Release'])
    assert os.stat(out).st_mtime == 1                 # same text: untouched, so main.c is not rebuilt
    fw_id.main(['Debug'])
    assert '"Debug"' in out.read_text(encoding='utf-8')


def test_dirty_ignores_only_the_cubeide_env_hash_file(monkeypatch):
    def fake(status):
        return lambda *a: 'abcdefabcdef\n' if a[0] == 'rev-parse' else status
    monkeypatch.setattr(fw_id, 'git', fake(' M stm32h7/.settings/language.settings.xml\n'))
    assert fw_id.fw_id('Debug')[1] == 0
    monkeypatch.setattr(fw_id, 'git', fake(''))
    assert fw_id.fw_id('Debug') == ('abcdefabcdef', 0, 'Debug')
    for st in (' M stm32h7/Core/Src/main.c\n', '?? stm32h7/scripts/x.py\n',
               ' M stm32h7/.settings/language.settings.xml\n M rpi5/protocol.py\n'):
        monkeypatch.setattr(fw_id, 'git', fake(st))
        assert fw_id.fw_id('Debug')[1] == 1, st


def test_git_failure_is_unknown_and_dirty(monkeypatch):
    monkeypatch.setattr(fw_id, 'git', lambda *a: None)
    assert fw_id.fw_id('Debug') == ('unknown', 1, 'Debug')


def test_main_c_prints_the_id_at_boot_and_in_the_block():
    src = open(os.path.join(ROOT, 'stm32h7', 'Core', 'Src', 'main.c'), encoding='utf-8').read()
    fmt = 'printf("#fw sha=%s dirty=%d env=%s\\r\\n", FW_GIT_SHA, (int)FW_GIT_DIRTY, FW_ENV)'
    assert src.count(fmt) == 2
    boot = src.index('--- H753 HIL boot ---')
    assert 0 < src.index(fmt, boot) - boot < 200      # right after the boot banner
    assert 'DS_FW, DS_IBUS, DS_END' in src            # one step of the debug block, before IBUS
    assert 'if (s == DS_FW && !d->fw_line)' in src    # skipped in blocks without the line
    assert '#  define FW_GIT_SHA   "unknown"' in src  # build without the pre-build header


def test_cproject_runs_the_prebuild_in_both_configs():
    cp = open(os.path.join(ROOT, 'stm32h7', '.cproject'), encoding='utf-8').read()
    step = 'prebuildStep="python &quot;${ProjDirPath}/scripts/fw_id.py&quot; ${ConfigName}"'
    assert cp.count(step) == 2
    gi = open(os.path.join(ROOT, '.gitignore'), encoding='utf-8').read()
    assert 'stm32h7/Core/Inc/fw_id_gen.h' in gi.split()
