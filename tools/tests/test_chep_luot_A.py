"""Tests of tools/chep_luot_A.py (design 03_CODE/docs/2026-11-15_Thiet_ke_CODE_C13_v1.md section 8, 10.4)."""
import csv
import hashlib
import json
import os
import shutil
import sys

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import chep_luot_A as C  # noqa: E402

README = ('# Hộp thư\n\n- dòng giới thiệu\n\n'
          '| File | Nguồn | Commit repo luận văn | SHA-256 (16 ký tự đầu) | Ngày chép |\n'
          '|---|---|---|---|---|\n'
          '| `spec.md` | spec | `abc` | `{sha}` | 14/11/2026 |\n'
          '\n**Luật ghi bảng:** câu giữ nguyên byte.\n')
TCOLS = ['session_id', 'iter_no', 'run_uid', 'campaign', 'stage', 'platform', 'results_dir', 'gate_json_sha256',
         'run_outcome', 'speed_mps', 'gain_set_id', 'g_kp_xy', 'g_ki_xy']


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def wcsv(path, cols, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def world(tmp_path):
    """A STRA_A root, a run folder of an A run with its files, a tuning log with its row, an empty freeze table."""
    sa = tmp_path / 'STRA_A'
    hop = sa / 'HOP_THU'
    hop.mkdir(parents=True)
    (hop / 'spec.md').write_text('spec\n', encoding='utf-8')
    (hop / 'README.md').write_bytes(README.format(sha=sha(hop / 'spec.md')[:16]).replace('\n', '\r\n').encode())
    run = tmp_path / 'data' / 'S05' / 'A' / 'r1'
    (run / 'cam').mkdir(parents=True)
    esp = run / 'log.csv'
    wcsv(esp, ['timestamp_us', 'x'], [{'timestamp_us': 1, 'x': 2}])
    seg = run / 'log.csv.segments.csv'
    wcsv(seg, ['a'], [{'a': 1}])
    fr = run / 'cam' / 'frames.csv'
    wcsv(fr, ['t_s', 'psi'], [{'t_s': 0, 'psi': 0}])
    tr = run / 'cam' / 'track.json'
    tr.write_text('{"ok": 1}', encoding='utf-8')
    rows = run / 'rows.csv'
    wcsv(rows, ['t', 'e_perp', 'e_theta_lane', 'e_theta_cmd'], [{'t': 0, 'e_perp': 0, 'e_theta_lane': 0,
                                                                 'e_theta_cmd': 0}])
    sc = run / 'score.json'
    sc.write_text(json.dumps({'esp32_sha256': sha(esp), 'frames_sha256': sha(fr), 'outcome': 'dat',
                              'version': 'c7_v1'}), encoding='utf-8')
    tl = tmp_path / 'm7_tuning_log.csv'
    row = dict(session_id='A1', iter_no='7', run_uid='A1:7', campaign='A-PID', stage='screen', platform='HW',
               results_dir=str(run), gate_json_sha256=sha(sc), run_outcome='dat', speed_mps='0.5',
               gain_set_id='g3', g_kp_xy='2.0', g_ki_xy='0.1')
    wcsv(tl, TCOLS, [row])
    fz = tmp_path / 'S05_freeze.csv'
    wcsv(fz, ['campaign', 'frozen_at'], [])
    return dict(sa=sa, hop=hop, run=run, esp=esp, seg=seg, fr=fr, tr=tr, rows=rows, sc=sc, tl=tl, fz=fz, row=row,
                tmp=tmp_path)


def args(w, **kw):
    a = ['--stra-a', str(w['sa']), 'luot', '--run-uid', kw.pop('run_uid', 'A1:7'), '--tuning-log', str(w['tl']),
         '--freeze', str(w['fz']), '--esp32', str(kw.pop('esp', w['esp'])), '--score', str(kw.pop('sc', w['sc'])),
         '--rows', str(kw.pop('rows', w['rows'])), '--frames', str(kw.pop('fr', w['fr'])),
         '--track', str(kw.pop('tr', w['tr']))]
    if kw.pop('seg', True):
        a += ['--segments', str(w['seg'])]
    for k, v in kw.items():
        a += [f'--{k.replace("_", "-")}'] + ([] if v is True else [v])
    return a


def set_row(w, **kw):
    w['row'].update(kw)
    wcsv(w['tl'], TCOLS, [w['row']])


def test_luot_copies_run(world, capsys):
    w = world
    old = (w['hop'] / 'README.md').read_bytes()
    assert C.main(args(w, ghi_chu='lượt ổn')) == 0
    d = w['hop'] / 'luot' / 'A-PID__A1__007'
    info = json.loads((d / 'luot.json').read_text(encoding='utf-8'))
    assert info['run_uid'] == 'A1:7' and info['files']['score']['sha256'] == sha(w['sc'])
    assert set(info['files']) == {'log_esp32', 'log_segments', 'score', 'sai_so', 'quy_dao', 'theo_doi', 'ghi_chu'}
    assert str(w['tmp']) not in (d / 'luot.json').read_text(encoding='utf-8')          # no thesis path
    new = (w['hop'] / 'README.md').read_bytes()
    k = old.index(b'\r\n\r\n**Luat') if b'**Luat' in old else old.index('\r\n\r\n**Luật'.encode()) + 2
    assert new.startswith(old[:k]) and new.endswith(old[k:])
    add = new[k:len(new) - len(old) + k].decode()
    assert add.count('\r\n') == 8 and '(bộ chấm c7_v1)' in add and 'luot/A-PID__A1__007/luot.json' in add
    _, bad = C.box_check(str(w['hop']))
    assert bad == []


@pytest.mark.parametrize('change, msg', [
    (dict(run_uid='A1:9'), '0 rows'),
    (dict(campaign='B-PID'), 'not a campaign A run'),
    (dict(campaign='RQ2'), 'not a campaign A run'),
    (dict(stage='final'), 'stage'),
    (dict(stage='final_hil'), 'stage'),
    (dict(stage='hil_check'), 'stage'),
    (dict(platform='HIL'), 'platform'),
    (dict(gate_json_sha256='0' * 64), 'gate_json_sha256'),
    (dict(run_outcome='khong_dat'), 'run_outcome'),
])
def test_luot_refuses_run(world, capsys, change, msg):
    w = world
    set_row(w, **change)
    old = (w['hop'] / 'README.md').read_bytes()
    assert C.main(args(w)) == 2
    assert msg in capsys.readouterr().out
    assert (w['hop'] / 'README.md').read_bytes() == old and not (w['hop'] / 'luot').exists()


def test_luot_refuses_two_rows_and_frozen(world, capsys):
    w = world
    wcsv(w['tl'], TCOLS, [w['row'], w['row']])
    assert C.main(args(w)) == 2 and '2 rows' in capsys.readouterr().out
    wcsv(w['tl'], TCOLS, [w['row']])
    wcsv(w['fz'], ['campaign', 'frozen_at'], [{'campaign': 'A-PID', 'frozen_at': 'x'}])
    assert C.main(args(w)) == 2 and 'frozen' in capsys.readouterr().out


@pytest.mark.parametrize('part', ['sealed', 'Sealed_Test', 'O1_FEAS', 'S03', 's04', 'logs_sweep', 'final'])
def test_luot_refuses_banned_path(world, capsys, part):
    w = world
    d = w['run'] / part
    d.mkdir()
    p = d / 'rows.csv'
    shutil.copy(w['rows'], p)
    assert C.main(args(w, rows=p)) == 2
    assert 'fence' in capsys.readouterr().out


def test_luot_refuses_outside_results_dir_and_symlink(world, capsys, tmp_path):
    w = world
    p = tmp_path / 'other_rows.csv'
    shutil.copy(w['rows'], p)
    assert C.main(args(w, rows=p)) == 2 and 'not inside results_dir' in capsys.readouterr().out
    link = w['run'] / 'link_rows.csv'
    try:
        os.symlink(p, link)
    except (OSError, NotImplementedError):
        pytest.skip('no symlink right on this machine')
    assert C.main(args(w, rows=link)) == 2


@pytest.mark.parametrize('ext', ['.mov', '.mp4', '.jpg', '.png'])
def test_luot_refuses_media(world, capsys, ext):
    w = world
    p = w['run'] / ('note' + ext)
    p.write_bytes(b'x')
    assert C.main(args(w, ghi_chu_file=str(p))) == 2


def test_luot_refuses_big_file_and_bad_sha(world, capsys, monkeypatch):
    w = world
    monkeypatch.setattr(C, 'MAX_BYTES', 5)
    assert C.main(args(w)) == 2 and 'bytes' in capsys.readouterr().out
    monkeypatch.setattr(C, 'MAX_BYTES', 1 << 20)
    other = w['run'] / 'log2.csv'
    wcsv(other, ['timestamp_us', 'x'], [{'timestamp_us': 1, 'x': 3}])
    assert C.main(args(w, esp=other, seg=False)) == 2 and 'esp32_sha256' in capsys.readouterr().out
    fr2 = w['run'] / 'cam' / 'frames2.csv'
    wcsv(fr2, ['t_s'], [{'t_s': 1}])
    assert C.main(args(w, fr=fr2)) == 2 and 'frames_sha256' in capsys.readouterr().out


def test_luot_refuses_missing_error_columns(world, capsys):
    w = world
    r2 = w['run'] / 'rows2.csv'
    wcsv(r2, ['t', 'e_perp'], [{'t': 0, 'e_perp': 0}])
    assert C.main(args(w, rows=r2)) == 2 and 'no column' in capsys.readouterr().out


def test_luot_again_needs_thay_and_keeps_old(world, capsys):
    w = world
    assert C.main(args(w)) == 0
    readme1 = (w['hop'] / 'README.md').read_bytes()
    assert C.main(args(w)) == 2 and '--thay' in capsys.readouterr().out
    assert (w['hop'] / 'README.md').read_bytes() == readme1
    assert C.main(args(w, thay=True)) == 0
    names = os.listdir(w['hop'] / 'luot')
    assert 'A-PID__A1__007' in names and any(n.startswith('A-PID__A1__007__cu_') for n in names)
    assert (w['hop'] / 'README.md').read_bytes().startswith(readme1[:readme1.index(b'\r\n\r\n**')])
    _, bad = C.box_check(str(w['hop']))
    assert bad == []


def test_luot_failure_leaves_nothing(world, capsys, monkeypatch):
    w = world
    old = (w['hop'] / 'README.md').read_bytes()
    real = shutil.copyfile
    n = {'k': 0}

    def flaky(a, b):
        n['k'] += 1
        if n['k'] == 3:
            raise OSError('disk full (test)')
        return real(a, b)
    monkeypatch.setattr(C.shutil, 'copyfile', flaky)
    with pytest.raises(OSError):
        C.main(args(w))
    assert os.listdir(w['hop'] / 'luot') == [] and (w['hop'] / 'README.md').read_bytes() == old


def test_luot_marks_gain_out_of_domain(world, capsys):
    w = world
    wcsv(w['hop'] / 'mien_A.csv', C.MIEN_COLS,
         [dict(controller_s03='PID', controller='PID', gain='kp_xy', lo='0.0', hi='1.5', n_diem='5', co_diem_0='true',
               nguon_hi='b'),
          dict(controller_s03='PID', controller='PID', gain='ki_xy', lo='0.0', hi='1.0', n_diem='5', co_diem_0='true',
               nguon_hi='b')])
    assert C.main(args(w)) == 0
    assert 'ngoài miền' in capsys.readouterr().out
    info = json.loads((w['hop'] / 'luot' / 'A-PID__A1__007' / 'luot.json').read_text(encoding='utf-8'))
    assert info['ngoai_mien'] == [dict(gain='kp_xy', value=2.0, lo=0.0, hi=1.5)]


def test_readme_refuses_pipe_in_cell(world):
    with pytest.raises(C.Refused, match=r'\|'):
        C.readme_append(str(world['hop']), [('a', 'b | c', 'x', '0' * 64, 'd')])


# ── them ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_them_tai_lieu_and_replace(world, capsys):
    w = world
    doc = w['tmp'] / 'gain_mac_dinh.md'
    doc.write_text('gain\n', encoding='utf-8')
    base = ['--stra-a', str(w['sa']), 'them', '--loai', 'tai_lieu', '--nguon', str(doc), '--ly-do', 'gain mặc định']
    assert C.main(base) == 0
    assert C.main(base) == 2
    doc.write_text('gain 2\n', encoding='utf-8')
    assert C.main(base + ['--thay']) == 0
    assert any(n.startswith('gain_mac_dinh__cu_') for n in os.listdir(w['hop']))
    _, bad = C.box_check(str(w['hop']))
    assert bad == []
    s = w['tmp'] / 'sealed'
    s.mkdir()
    (s / 'x.md').write_text('x', encoding='utf-8')
    assert C.main(['--stra-a', str(w['sa']), 'them', '--loai', 'tai_lieu', '--nguon', str(s / 'x.md'),
                   '--ly-do', 'x']) == 2


# ── mien ───────────────────────────────────────────────────────────────────────────────────────────────────────────

SCOLS = ['controller', 'factor', 'value', 'v', 'luoi', 'completed']
LCOLS = ['controller', 'factor', 'L', 'H', 'default', 'a', 'b', 'b_n', 'L_nguon', 'H_nguon']
PID_G = ['kp_xy', 'ki_xy', 'kp_wz', 'ki_wz', 'pos_kp', 'pos_ki', 'k_psi']
ADRC_G = ['omega_c_xy', 'omega_o_xy', 'omega_c_wz', 'omega_o_wz', 'pos_kp', 'pos_ki', 'k_psi']


def s03_run(tmp, name='2026-11-20_run03', meta=None, speeds=('0.2', '0.5', '0.8'), drop=None, extra=None,
            omega_hi=48.0, lev_edit=None):
    run = tmp / 'S03x' / name
    run.mkdir(parents=True)
    sw, lv = [], []
    for c, gs in (('PID', PID_G), ('ADRC', ADRC_G)):
        for v in speeds:
            sw.append(dict(controller=c, factor='(default check)', value='', v=v, luoi='mac_dinh', completed='true'))
        for g in gs:
            om = g.startswith('omega_o')
            pts = [0.0, 0.5, 1.0, 2.0] + ([omega_hi] if om else [4.0, 8.0])
            for x in pts:
                for v in speeds:
                    if drop == (c, g, x, v):
                        continue
                    lu = '0' if x == 0 else ('noi' if (not om and x == 8.0) else 'log')
                    sw.append(dict(controller=c, factor=g, value=repr(x), v=v, luoi=lu, completed='true'))
            lv.append(dict(controller=c, factor=g, L='0.5', H='2.0', default='1.0', a='0.5',
                           b=repr(omega_hi) if om else '4.0', b_n='' if om else '8.0', L_nguon='0',
                           H_nguon='trần ω_o' if om else 'hết lưới nối'))
    if extra:
        sw.append(extra)
    if lev_edit:
        lev_edit(lv)
    wcsv(run / 'sweep.csv', SCOLS, sw)
    wcsv(run / 'levels.csv', LCOLS, lv)
    m = dict(trial=False, script=dict(commit='abc', dirty=False), firmware=dict(esp32=dict(dirty=False)))
    m.update(meta or {})
    (run / 'meta.yaml').write_text(yaml.safe_dump(m), encoding='utf-8')
    return run


def mien_args(w, run, out=None, *more):
    return ['mien', '--s03-run', str(run), '--tuning-log', str(w['tl0']), '--out', str(out or w['tmp'] / 'A'), *more]


@pytest.fixture
def w0(world, monkeypatch):
    monkeypatch.setattr(C, 'THESIS', str(world['tmp']))
    tl0 = world['tmp'] / 'tl0.csv'
    wcsv(tl0, TCOLS, [])
    world['tl0'] = tl0
    return world


def test_mien_domain(w0, capsys):
    run = s03_run(w0['tmp'])
    assert C.main(mien_args(w0, run)) == 0
    rows = list(csv.DictReader(open(w0['tmp'] / 'A' / 'mien_A.csv', encoding='utf-8')))
    d = {(r['controller'], r['gain']): r for r in rows}
    assert len(rows) == 14
    assert (d[('PID', 'kp_xy')]['lo'], d[('PID', 'kp_xy')]['hi'], d[('PID', 'kp_xy')]['nguon_hi']) == ('0.0', '8.0', 'b_n')
    assert d[('LADRC', 'omega_o_xy')]['hi'] == '48.0' and d[('LADRC', 'omega_o_xy')]['nguon_hi'] == 'tran_omega_o'
    js = json.load(open(w0['tmp'] / 'A' / 'mien_A.json', encoding='utf-8'))
    assert js['sha256']['sweep.csv'] == sha(run / 'sweep.csv')
    assert C.main(mien_args(w0, run)) == 2                                   # no overwrite without --thay


@pytest.mark.parametrize('kw, msg', [
    (dict(name='2026-11-10_run02'), 'history'),
    (dict(meta=dict(trial=True)), 'tool check'),
    (dict(meta=dict(tool_check=True)), 'tool check'),
    (dict(meta=dict(script=dict(commit='x', dirty=True))), 'dirty'),
    (dict(drop=('PID', 'kp_xy', 1.0, '0.5')), 'speeds'),
    (dict(extra=dict(controller='PID', factor='extra_g', value='1.0', v='0.2', luoi='log', completed='true')),
     'gains'),
    (dict(omega_hi=60.0), '> 48'),
    (dict(lev_edit=lambda lv: lv[0].update(default='99')), 'default'),
    (dict(lev_edit=lambda lv: lv[0].update(H='50')), '[L, H]'),
    (dict(lev_edit=lambda lv: lv[0].update(b_n='9.0')), 'b_n'),
    (dict(lev_edit=lambda lv: lv[1].update(b='8.0', b_n='')), 'disagree'),
])
def test_mien_refuses(w0, capsys, kw, msg):
    run = s03_run(w0['tmp'], **kw)
    assert C.main(mien_args(w0, run)) == 2
    assert msg in capsys.readouterr().out
    assert not (w0['tmp'] / 'A' / 'mien_A.csv').exists()


def test_mien_refuses_sealed_and_old_columns(w0, capsys):
    run = s03_run(w0['tmp'])
    sealed = run / 'sealed'
    shutil.copytree(run, sealed, ignore=shutil.ignore_patterns('sealed'))
    assert C.main(mien_args(w0, sealed)) == 2 and 'fence' in capsys.readouterr().out
    rows = list(csv.DictReader(open(run / 'levels.csv', encoding='utf-8')))
    wcsv(run / 'levels.csv', ['controller', 'factor', 'L', 'H', 'default'],
         [{k: r[k] for k in ('controller', 'factor', 'L', 'H', 'default')} for r in rows])
    assert C.main(mien_args(w0, run)) == 2 and 'v8.3' in capsys.readouterr().out


def test_mien_refuses_after_first_A_run(w0, capsys):
    run = s03_run(w0['tmp'])
    w0['tl0'] = w0['tl']                                                     # has an A-PID row
    assert C.main(mien_args(w0, run)) == 2 and 'locked before the first run' in capsys.readouterr().out


def test_them_mien_A_refuses_hand_edit(w0, capsys, monkeypatch):
    run = s03_run(w0['tmp'])
    monkeypatch.setattr(C, 'THESIS', str(w0['tmp']))
    assert C.main(mien_args(w0, run)) == 0
    p = w0['tmp'] / 'A' / 'mien_A.csv'
    base = ['--stra-a', str(w0['sa']), 'them', '--loai', 'mien_A', '--nguon', str(p), '--ly-do', 'miền A']
    t = p.read_text(encoding='utf-8')
    p.write_text(t.replace('8.0', '9.0', 1), encoding='utf-8')
    assert C.main(base) == 2 and 'differs' in capsys.readouterr().out
    p.write_text(t, encoding='utf-8')
    assert C.main(base) == 0
    assert sha(w0['hop'] / 'mien_A.csv') == sha(p)


# ── ke-khai ────────────────────────────────────────────────────────────────────────────────────────────────────────

def fence_files(sa, bad_line=False):
    (sa / '_codex_test').mkdir()
    lines = ['codex: x', "1 doc STRA_A                 => doc duoc A", '2 ghi CODEX                  => ghi ok',
             "2b ghi HOP_THU               => codex.exe : Set-Content : Access to the path denied",
             '6 python trong STRA_A        => python ok 2.4.1',
             '7 mang                       => ' + ('200' if bad_line else 'loi mang: no')]
    (sa / '_codex_test' / 'ket_qua_2026-11-14.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (sa / C.RAO_CLAUDE).write_text('7/7', encoding='utf-8')
    (sa / 'NHAT_KY').mkdir()
    (sa / 'CODEX').mkdir()
    (sa / 'NHAT_KY' / '2026-11-20.md').write_text('buổi 1\n', encoding='utf-8')


def test_ke_khai(world, capsys, monkeypatch):
    w = world
    sa = w['sa']
    fence_files(sa)
    monkeypatch.setattr(C, 'SHA_CODEX_CONFIG', None)
    out1 = w['tmp'] / 'k1.yaml'
    assert C.main(['--stra-a', str(sa), 'ke-khai', '--out', str(out1)]) == 2      # lock SHAs not there
    d = yaml.safe_load(open(out1, encoding='utf-8'))
    assert any('config.toml' in x for x in d['loi']) and d['nhiem_thong_tin']['co'] is False
    (sa / 'NHAT_KY' / '2026-11-20.md').write_text('buổi 1\nNHIEM: lỡ thấy x\n', encoding='utf-8')
    (w['hop'] / 'la.txt').write_text('x', encoding='utf-8')
    out2 = w['tmp'] / 'k2.yaml'
    C.main(['--stra-a', str(sa), 'ke-khai', '--out', str(out2), '--truoc', str(out1)])
    d2 = yaml.safe_load(open(out2, encoding='utf-8'))
    assert d2['nhiem_thong_tin']['co'] is True and any('strange file' in x for x in d2['loi'])
    assert not any('append-only' in x for x in d2['loi'])
    (sa / 'NHAT_KY' / '2026-11-20.md').write_text('sửa đầu\n', encoding='utf-8')
    out3 = w['tmp'] / 'k3.yaml'
    C.main(['--stra-a', str(sa), 'ke-khai', '--out', str(out3), '--truoc', str(out2)])
    assert any('append-only' in x for x in yaml.safe_load(open(out3, encoding='utf-8'))['loi'])


def test_rao_line_wrong(world):
    fence_files(world['sa'], bad_line=True)
    _, bad = C.rao_check(str(world['sa']))
    assert any('test 7' in x for x in bad)


def test_box_check_sha_mismatch(world):
    (world['hop'] / 'spec.md').write_text('đổi\n', encoding='utf-8')
    _, bad = C.box_check(str(world['hop']))
    assert any('differs from its latest row' in x for x in bad)
