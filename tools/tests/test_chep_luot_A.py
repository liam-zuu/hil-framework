"""Tests of tools/chep_luot_A.py (design 03_CODE/docs/2026-11-15_Thiet_ke_CODE_C13_v1.md section 8, 10.4; Codex Sol
round 1 counter-examples SA-n)."""
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
def world(tmp_path, monkeypatch):
    """A thesis folder (tmp) with an A run in 05_DATA/results, a STRA_A root, a tuning log, an empty freeze table."""
    monkeypatch.setattr(C, 'THESIS', str(tmp_path))
    sa = tmp_path / 'STRA_A'
    hop = sa / 'HOP_THU'
    hop.mkdir(parents=True)
    (hop / 'spec.md').write_text('spec\n', encoding='utf-8')
    (hop / 'README.md').write_bytes(README.format(sha=sha(hop / 'spec.md')[:16]).replace('\n', '\r\n').encode())
    run = tmp_path / '05_DATA' / 'results' / 'S05' / 'A' / 'r1'
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
               results_dir='05_DATA/results/S05/A/r1', gate_json_sha256=sha(sc), run_outcome='dat', speed_mps='0.5',
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


def box_ok(w):
    _, bad = C.box_check(str(w['hop']))
    assert bad == []


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
    k = old.index('\r\n\r\n**Luật'.encode()) + 2
    assert new.startswith(old[:k]) and new.endswith(old[k:])
    add = new[k:len(new) - len(old) + k].decode()
    assert add.count('\r\n') == 8 and '(bộ chấm c7_v1)' in add and 'luot/A-PID__A1__007/luot.json' in add
    box_ok(w)


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
    (dict(results_dir='.'), 'not inside 05_DATA/results'),                         # SA-7
    (dict(results_dir='05_DATA/results/S05'), 'not a run folder'),                 # SA-7
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


def test_luot_refuses_outside_results_dir_and_links(world, capsys, tmp_path):
    w = world
    p = tmp_path / 'other_rows.csv'
    shutil.copy(w['rows'], p)
    assert C.main(args(w, rows=p)) == 2 and 'not inside results_dir' in capsys.readouterr().out
    hard = w['run'] / 'hard_rows.csv'                                              # SA-9
    os.link(p, hard)
    assert C.main(args(w, rows=hard)) == 2 and 'hard link' in capsys.readouterr().out
    link = w['run'] / 'link_rows.csv'
    try:
        os.symlink(p, link)
    except (OSError, NotImplementedError):
        return
    assert C.main(args(w, rows=link)) == 2


def test_luot_note_file_must_be_in_the_run(world, capsys):
    w = world                                                                      # SA-8
    other = w['tmp'] / '05_DATA' / 'results' / 'S05' / 'B' / 'r9'
    other.mkdir(parents=True)
    (other / 'ghi_chu.md').write_text('gain B', encoding='utf-8')
    assert C.main(args(w, ghi_chu_file=str(other / 'ghi_chu.md'))) == 2
    assert 'not inside results_dir' in capsys.readouterr().out
    (w['run'] / 'ghi_chu.md').write_text('ổn', encoding='utf-8')
    assert C.main(args(w, ghi_chu_file=str(w['run'] / 'ghi_chu.md'))) == 0


@pytest.mark.parametrize('ext', ['.mov', '.mp4', '.jpg', '.png'])
def test_luot_refuses_media(world, capsys, ext):
    w = world
    p = w['run'] / ('note' + ext)
    p.write_bytes(b'x')
    assert C.main(args(w, ghi_chu_file=str(p))) == 2


def test_media_renamed_refused(world, capsys):
    w = world                                                                      # SA-10
    p = w['tmp'] / 'camera.json'
    p.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 20)
    base = ['--stra-a', str(w['sa']), 'them', '--loai', 'tai_lieu', '--nguon', str(p), '--ly-do', 'x']
    assert C.main(base) == 2 and 'media' in capsys.readouterr().out
    p.write_text('{"a": 1', encoding='utf-8')
    assert C.main(base) == 2 and 'not JSON' in capsys.readouterr().out
    box_ok(w)


def test_role_formats(world, capsys):
    """SA2-1: track.json must be JSON (a .txt is refused); SA2-5: text beginning with 'BM' is text."""
    w = world
    t = w['run'] / 'cam' / 'track.txt'
    t.write_text('không phải JSON', encoding='utf-8')
    assert C.main(args(w, tr=t)) == 2 and 'extension' in capsys.readouterr().out
    doc = w['tmp'] / 'gioi_han_pin.md'
    doc.write_text('BMS: điện áp tối thiểu 10,5 V\n', encoding='utf-8')
    assert C.main(['--stra-a', str(w['sa']), 'them', '--loai', 'tai_lieu', '--nguon', str(doc), '--ly-do', 'pin']) == 0


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


def test_luot_refuses_case_twins(world, capsys):
    w = world                                                                      # SA-4
    d = w['run'] / 'err'
    d.mkdir()
    p = d / 'LOG.csv'
    shutil.copy(w['rows'], p)
    assert C.main(args(w, rows=p, seg=False)) == 2 and 'ignore case' in capsys.readouterr().out
    assert not (w['hop'] / 'luot').exists() or os.listdir(w['hop'] / 'luot') == []


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
    box_ok(w)

def test_box_check_archive_tampered(world, capsys):
    """SA-17: the run copied again with a new score; the archived old score replaced by the current one."""
    w = world
    assert C.main(args(w)) == 0
    sc2 = json.loads(w['sc'].read_text(encoding='utf-8'))
    sc2['version'] = 'c7_v2'
    w['sc'].write_text(json.dumps(sc2), encoding='utf-8')
    set_row(w, gate_json_sha256=sha(w['sc']))
    assert C.main(args(w, thay=True)) == 0
    box_ok(w)
    cu = next(n for n in os.listdir(w['hop'] / 'luot') if '__cu_' in n)
    keep = (w['hop'] / 'luot' / cu / 'score.json').read_bytes()
    shutil.copy(w['hop'] / 'luot' / 'A-PID__A1__007' / 'score.json', w['hop'] / 'luot' / cu / 'score.json')
    _, bad = C.box_check(str(w['hop']))
    assert any('no older row' in x for x in bad)
    (w['hop'] / 'luot' / cu / 'score.json').write_bytes(keep)
    box_ok(w)
    os.remove(w['hop'] / 'luot' / cu / 'score.json')                              # SA-17: archived copy deleted
    _, bad = C.box_check(str(w['hop']))
    assert any('older row' in x and 'no file on the disk' in x for x in bad)


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
    assert C.main(args(w)) == 2                                                     # SA-23: exit 2, marker
    assert '#ket_qua,ma=2' in capsys.readouterr().out
    assert os.listdir(w['hop'] / 'luot') == [] and (w['hop'] / 'README.md').read_bytes() == old


def test_luot_readme_failure_rolls_back(world, capsys, monkeypatch):
    w = world                                                                      # SA-11
    assert C.main(args(w)) == 0
    before = sorted(os.listdir(w['hop'] / 'luot'))
    readme = (w['hop'] / 'README.md').read_bytes()

    def boom(hop, data):
        raise OSError('README locked (test)')
    monkeypatch.setattr(C, 'readme_write', boom)
    assert C.main(args(w, thay=True)) == 2
    assert sorted(os.listdir(w['hop'] / 'luot')) == before and (w['hop'] / 'README.md').read_bytes() == readme
    box_ok(w)


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
        C.readme_new(str(world['hop']), [('a', 'b | c', 'x', '0' * 64, 'd')])


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
    box_ok(w)
    s = w['tmp'] / 'sealed'
    s.mkdir()
    (s / 'x.md').write_text('x', encoding='utf-8')
    assert C.main(['--stra-a', str(w['sa']), 'them', '--loai', 'tai_lieu', '--nguon', str(s / 'x.md'),
                   '--ly-do', 'x']) == 2


@pytest.mark.parametrize('name', ['MIEN_A.csv', 'readme.md', 'Sweep.csv', 'x__cu_20261116-1200.md'])
def test_them_reserved_names_any_case(world, capsys, name):
    w = world                                                                      # SA-3
    p = w['tmp'] / name
    p.write_text('a,b\n1,2\n' if name.endswith('.csv') else 'x\n', encoding='utf-8')
    readme = (w['hop'] / 'README.md').read_bytes()
    assert C.main(['--stra-a', str(w['sa']), 'them', '--loai', 'tai_lieu', '--nguon', str(p), '--ly-do', 'x',
                   '--thay']) == 2
    assert 'reserved' in capsys.readouterr().out and (w['hop'] / 'README.md').read_bytes() == readme


def test_them_bad_reason_changes_nothing(world, capsys):
    w = world                                                                      # SA-11
    doc = w['tmp'] / 'gioi_han.md'
    doc.write_text('x\n', encoding='utf-8')
    files = sorted(os.listdir(w['hop']))
    assert C.main(['--stra-a', str(w['sa']), 'them', '--loai', 'tai_lieu', '--nguon', str(doc),
                   '--ly-do', 'a|b']) == 2
    assert sorted(os.listdir(w['hop'])) == files


# ── mien ───────────────────────────────────────────────────────────────────────────────────────────────────────────

SCOLS = ['controller', 'factor', 'value', 'v', 'mu', 'luoi', 'completed']
LCOLS = ['controller', 'factor', 'L', 'H', 'default', 'a', 'b', 'b_n', 'L_nguon', 'H_nguon']
PID_G = ['kp_xy', 'ki_xy', 'kp_wz', 'ki_wz', 'pos_kp', 'pos_ki', 'k_psi']
ADRC_G = ['omega_c_xy', 'omega_o_xy', 'omega_c_wz', 'omega_o_wz', 'pos_kp', 'pos_ki', 'k_psi']


KCOLS = ['dong', 'controller', 'factor', 'value', 'luoi', 'can', 'can_value', 'ket_qua', 'n_diem_ngoai', 'n_da_kiem',
         'n_chua_kiem', 'n_co', 'n_chua_ket_luan', 'n_khong_co']


def kiem_ok_rows(omega_hi=48.0):
    """S03 v8.4 kiem_ngoai_mien.csv of the fixture: every grid point outside [L, H] = [0.5, 2] (0 against L; 4, 8 or
    the omega_o top against H), 'không cờ'; one total per controller, 'đã kiểm đủ', with its counts (S84-5)."""
    rows, tot = [], []
    for c, gs in (('PID', PID_G), ('ADRC', ADRC_G)):
        n = 0
        for g in gs:
            om = g.startswith('omega_o')
            for x in [0.0, 0.5, 1.0, 2.0] + ([omega_hi] if om else [4.0, 8.0]):
                if 0.5 <= x <= 2.0:
                    continue
                rows.append(dict(dong='diem', controller=c, factor=g, value=repr(x),
                                 luoi='0' if x == 0 else ('noi' if (not om and x == 8.0) else 'log'),
                                 can='L' if x < 0.5 else 'H', can_value='0.5' if x < 0.5 else '2.0', ket_qua='không cờ'))
                n += 1
        tot.append(dict(dong='tong', controller=c, ket_qua='đã kiểm đủ', n_diem_ngoai=n, n_da_kiem=n, n_chua_kiem=0,
                        n_co=0, n_chua_ket_luan=0, n_khong_co=n))
    return rows + tot


def s03_run(tmp, name='2026-11-20_run03', meta=None, speeds=('0.2', '0.5', '0.8'), drop=None, extra=None,
            omega_hi=48.0, lev_edit=None, sw_edit=None, kiem_edit=None):
    run = tmp / '05_DATA' / 'results' / 'S03' / name
    run.mkdir(parents=True)
    sw, lv = [], []
    for c, gs in (('PID', PID_G), ('ADRC', ADRC_G)):
        for v in speeds:
            sw.append(dict(controller=c, factor='(default check)', value='', v=v, mu='0.5', luoi='mac_dinh',
                           completed='true'))
        for g in gs:
            om = g.startswith('omega_o')
            pts = [0.0, 0.5, 1.0, 2.0] + ([omega_hi] if om else [4.0, 8.0])
            for x in pts:
                for v in speeds:
                    if drop and drop(c, g, x, v):
                        continue
                    lu = '0' if x == 0 else ('noi' if (not om and x == 8.0) else 'log')
                    sw.append(dict(controller=c, factor=g, value=repr(x), v=v, mu='0.5', luoi=lu, completed='true'))
            lv.append(dict(controller=c, factor=g, L='0.5', H='2.0', default='1.0', a='0.5',
                           b=repr(omega_hi) if om else '4.0', b_n='' if om else '8.0', L_nguon='0',
                           H_nguon='trần ω_o' if om else 'điểm kế tiếp không hoàn thành'))
        for mu in ('0.4', '0.6'):                         # S84-4: S03 also runs the default gains at the low, high mu
            for v in speeds:
                sw.append(dict(controller=c, factor='(default)', value='', v=v, mu=mu, luoi='mac_dinh',
                               completed='true'))
    if extra:
        sw.append(extra)
    if lev_edit:
        lev_edit(lv)
    if sw_edit:
        sw_edit(sw)
    wcsv(run / 'sweep.csv', SCOLS, sw)
    wcsv(run / 'levels.csv', LCOLS, lv)
    kr = kiem_ok_rows(omega_hi)
    if kiem_edit:
        kr = kiem_edit(kr)
    if kr is not None:
        wcsv(run / 'kiem_ngoai_mien.csv', KCOLS, kr)
    m = dict(trial=False, date='2026-11-20', script=dict(commit='abc', dirty=False),
             firmware=dict(esp32=dict(dirty=False)))
    m.update(meta or {})
    (run / 'meta.yaml').write_text(yaml.safe_dump(m), encoding='utf-8')
    return run


def mien_args(w, run, out=None, *more):
    return ['mien', '--s03-run', str(run), '--tuning-log', str(w['tl0']), '--out', str(out or w['tmp'] / 'A'), *more]


@pytest.fixture
def w0(world):
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
    # S03 v8.4 / S05 v3.41: the original grid up to b; the extension point 8.0 (luoi noi) is not in the domain
    assert (d[('PID', 'kp_xy')]['lo'], d[('PID', 'kp_xy')]['hi'], d[('PID', 'kp_xy')]['nguon_hi']) == ('0.0', '4.0', 'b')
    assert d[('PID', 'kp_xy')]['n_diem'] == '5' and {r['nguon_hi'] for r in rows} == {'b', 'tran_omega_o'}
    assert d[('LADRC', 'omega_o_xy')]['hi'] == '48.0' and d[('LADRC', 'omega_o_xy')]['nguon_hi'] == 'tran_omega_o'
    js = json.load(open(w0['tmp'] / 'A' / 'mien_A.json', encoding='utf-8'))
    assert js['sha256']['sweep.csv'] == sha(run / 'sweep.csv') and js['s03_run'] == '05_DATA/results/S03/2026-11-20_run03'
    assert js['sha256']['kiem_ngoai_mien.csv'] == sha(run / 'kiem_ngoai_mien.csv') and 'S03 v8.4' in js['spec']
    assert C.main(mien_args(w0, run)) == 2                                   # no overwrite without --thay
    assert C.main(mien_args(w0, run, None, '--thay')) == 0
    assert len([n for n in os.listdir(w0['tmp'] / 'A') if '__cu_' in n]) == 2


def rename_gain(sw):
    for r in sw:
        if r['factor'] == 'kp_xy':
            r['factor'] = 'gain_la'


@pytest.mark.parametrize('kw, msg', [
    (dict(name='2026-11-10_run02'), 'history'),
    (dict(name='2026-11-10_RUN02'), 'history'),                                    # SA-6
    (dict(meta=dict(trial=True)), 'tool check'),
    (dict(meta=dict(tool_check=True)), 'tool check'),
    (dict(meta=dict(script=dict(commit='x', dirty=True))), 'dirty'),
    (dict(meta=dict(script=dict(commit='x'))), 'script.dirty'),                    # SA-6
    (dict(drop=lambda c, g, x, v: (c, g, x, v) == ('PID', 'kp_xy', 1.0, '0.5')), 'speeds'),
    (dict(drop=lambda c, g, x, v: v == '0.8'), 'speeds'),                          # SA-15
    (dict(speeds=('0.2', '0.5')), 'speeds'),                                       # SA-15: gone everywhere
    (dict(extra=dict(controller='PID', factor='extra_g', value='1.0', v='0.2', luoi='log', completed='true')),
     'need exactly'),
    (dict(sw_edit=rename_gain, lev_edit=lambda lv: lv[0].update(factor='gain_la')), 'need exactly'),   # SA-14
    (dict(sw_edit=lambda sw: sw[:] if False else [sw.remove(r) for r in list(sw) if r['controller'] == 'ADRC']),
     'controllers'),                                                                # SA-14
    (dict(omega_hi=60.0), '> 48'),
    (dict(lev_edit=lambda lv: lv[0].update(default='99')), 'default'),
    (dict(lev_edit=lambda lv: lv[0].update(H='50')), '[L, H]'),
    # S03 v8.4: hi = b, the extension is not the domain
    (dict(lev_edit=lambda lv: lv[1].update(b='8.0', b_n='')), 'is not b'),
    (dict(lev_edit=lambda lv: lv[0].update(H='8.0')), '[L, H]'),                    # a level on the extension
    (dict(sw_edit=lambda sw: [r.update(luoi='log') for r in sw if r['factor'] == 'kp_xy' and r['value'] == '8.0']),
     'is not b'),
    (dict(sw_edit=lambda sw: [r.update(luoi='noi') for r in sw if r['factor'] == 'kp_xy' and r['value'] == '4.0']),
     'is not b'),
    (dict(sw_edit=lambda sw: next(r for r in sw if r['factor'] == 'kp_xy' and r['value'] == '8.0').update(luoi='log')),
     'differs'),
    (dict(sw_edit=lambda sw: sw.extend(dict(controller='PID', factor='kp_xy', value='3.0', v=v, luoi='noi',
                                            completed='true') for v in ('0.2', '0.5', '0.8'))), 'not above b'),
    (dict(sw_edit=lambda sw: sw.extend(dict(controller='ADRC', factor='omega_o_xy', value='96.0', v=v, luoi='noi',
                                            completed='true') for v in ('0.2', '0.5', '0.8'))), 'never extended'),
    # S03 v8.4 out-of-range check
    (dict(kiem_edit=lambda kr: None), 'kiem_ngoai_mien.csv: no file'),
    (dict(kiem_edit=lambda kr: [dict(r, ket_qua='kiểm dở') if r['controller'] == 'ADRC' and r['dong'] == 'tong'
                                else r for r in kr]), 'checked in full'),
    (dict(kiem_edit=lambda kr: [dict(r, ket_qua='không kiểm') if r['dong'] == 'tong' else r for r in kr]),
     'checked in full'),
    (dict(kiem_edit=lambda kr: [dict(kr[0], ket_qua='cờ')] + kr[1:]), 'flagged'),
    (dict(kiem_edit=lambda kr: [r for r in kr if not (r['dong'] == 'tong' and r['controller'] == 'ADRC')]),
     'total rows for'),
    (dict(kiem_edit=lambda kr: kr + [dict(dong='tong', controller='PID', ket_qua='đã kiểm đủ')]), 'two total rows'),
    (dict(kiem_edit=lambda kr: [dict(kr[0], ket_qua='co')] + kr[1:]), 'not in'),
    (dict(kiem_edit=lambda kr: [dict(kr[0], dong='x')] + kr[1:]), 'is not diem or tong'),
    # S84-5: the point rows must be exactly the points outside [L, H] of the sweep, the totals must agree
    (dict(kiem_edit=lambda kr: [r for r in kr if r['dong'] == 'tong']), 'do not match'),
    (dict(kiem_edit=lambda kr: [r for r in kr if not (r['dong'] == 'diem' and r['controller'] == 'PID'
                                                     and r['factor'] == 'kp_xy' and r['can'] == 'L')]), 'do not match'),
    (dict(kiem_edit=lambda kr: [kr[0]] + kr), 'twice'),
    (dict(kiem_edit=lambda kr: [dict(kr[0], can='H')] + kr[1:]), 'wrong side'),
    (dict(kiem_edit=lambda kr: kr[:-1] + [dict(kr[-1], n_khong_co=int(kr[-1]['n_khong_co']) - 1)]), 'do not agree'),
    (dict(kiem_edit=lambda kr: kr[:-1] + [dict(kr[-1], n_chua_kiem=1)]), 'do not agree'),
    # S84-4: the default rows are complete per mu
    (dict(sw_edit=lambda sw: sw.remove(next(r for r in sw if r['factor'] == '(default)' and r['mu'] == '0.4'))),
     "at mu '0.4'"),
    (dict(sw_edit=lambda sw: sw.append(dict(controller='PID', factor='kp_xy', value='1', v='0.2', luoi='log',
                                            completed='true'))), 'once each'),       # SA-16
])
def test_mien_refuses(w0, capsys, kw, msg):
    run = s03_run(w0['tmp'], **kw)
    assert C.main(mien_args(w0, run)) == 2
    assert msg in capsys.readouterr().out
    assert not (w0['tmp'] / 'A' / 'mien_A.csv').exists()


def test_kiem_c13_item8_sees_a_changed_check_file(w0, capsys, monkeypatch):
    """S84-6: the check file is hashed in mien_A.json, so kiem-c13 item 8 fails when kiem_ngoai_mien.csv changes
    after the domain was made (here one blank line: the domain computed again is the same)."""
    run = s03_run(w0['tmp'])
    assert C.main(mien_args(w0, run)) == 0
    monkeypatch.setattr(C, '_tool_version', lambda cmd: '')
    real_git = C.git
    monkeypatch.setattr(C, 'git', lambda root, *a: '2026-11-19' if a[:2] == ('show', '-s') else real_git(root, *a))
    cmd = ['--stra-a', str(w0['sa']), 'kiem-c13', '--spec-commit', 'HEAD', '--ke-khai', str(w0['tmp'] / 'kk.yaml'),
           '--mien-a', str(w0['tmp'] / 'A' / 'mien_A.csv')]
    capsys.readouterr()
    C.main(cmd)
    line8 = next(x for x in capsys.readouterr().out.splitlines() if ' 8 ' in x)
    assert 'nguồn S03 khớp True' in line8 and 'tính lại khớp True' in line8
    k = run / 'kiem_ngoai_mien.csv'
    k.write_bytes(k.read_bytes() + b'\r\n')
    C.main(cmd)
    line8 = next(x for x in capsys.readouterr().out.splitlines() if ' 8 ' in x)
    assert line8.startswith('CHƯA') and 'nguồn S03 khớp False' in line8 and 'tính lại khớp True' in line8


def test_mien_default_alias(w0, capsys):
    """SA2-2: '(default)' rows are accepted like '(default check)'."""
    run = s03_run(w0['tmp'], sw_edit=lambda sw: [r.update(factor='(default)') for r in sw
                                                 if r['factor'] == '(default check)'])
    assert C.main(mien_args(w0, run)) == 0


def test_mien_leftover_temp_refused(w0, capsys):
    run = s03_run(w0['tmp'])
    (w0['tmp'] / 'A').mkdir()
    (w0['tmp'] / 'A' / 'mien_A.csv.tmp_mien').write_text('x', encoding='utf-8')     # SA-12
    assert C.main(mien_args(w0, run)) == 2 and 'left by a failed run' in capsys.readouterr().out
    assert (w0['tmp'] / 'A' / 'mien_A.csv.tmp_mien').exists() and not (w0['tmp'] / 'A' / 'mien_A.csv').exists()


def test_mien_refuses_sealed_link_and_old_columns(w0, capsys, tmp_path):
    run = s03_run(w0['tmp'])
    sealed = run / 'sealed'
    shutil.copytree(run, sealed, ignore=shutil.ignore_patterns('sealed'))
    assert C.main(mien_args(w0, sealed)) == 2 and 'fence' in capsys.readouterr().out
    other = tmp_path / 'x' / 'run04'                                               # SA-6: not under results/S03
    shutil.copytree(run, other, ignore=shutil.ignore_patterns('sealed'))
    assert C.main(mien_args(w0, other)) == 2 and 'S03/<run>' in capsys.readouterr().out
    os.remove(run / 'sweep.csv')                                                    # SA-5: hard link to sealed
    os.link(sealed / 'sweep.csv', run / 'sweep.csv')
    assert C.main(mien_args(w0, run)) == 2 and 'hard link' in capsys.readouterr().out
    os.remove(run / 'sweep.csv')
    shutil.copy(sealed / 'sweep.csv', run / 'sweep.csv')
    rows = list(csv.DictReader(open(run / 'levels.csv', encoding='utf-8')))
    wcsv(run / 'levels.csv', ['controller', 'factor', 'L', 'H', 'default'],
         [{k: r[k] for k in ('controller', 'factor', 'L', 'H', 'default')} for r in rows])
    assert C.main(mien_args(w0, run)) == 2 and 'v8.3' in capsys.readouterr().out


def test_mien_refuses_after_first_A_run(w0, capsys):
    run = s03_run(w0['tmp'])
    w0['tl0'] = w0['tl']                                                     # has an A-PID row
    assert C.main(mien_args(w0, run)) == 2 and 'locked before the first run' in capsys.readouterr().out
    bad = w0['tmp'] / 'tl_nocol.csv'                                         # SA-13: no campaign column
    wcsv(bad, ['run_uid'], [])
    w0['tl0'] = bad
    assert C.main(mien_args(w0, run)) == 2 and 'no column' in capsys.readouterr().out


def test_them_mien_A(w0, capsys):
    run = s03_run(w0['tmp'])
    assert C.main(mien_args(w0, run)) == 0
    p = w0['tmp'] / 'A' / 'mien_A.csv'
    base = ['--stra-a', str(w0['sa']), 'them', '--loai', 'mien_A', '--nguon', str(p), '--ly-do', 'miền A']
    assert C.main(base) == 2 and '--tuning-log' in capsys.readouterr().out
    base += ['--tuning-log', str(w0['tl0'])]
    t = p.read_bytes()
    p.write_bytes(t.replace(b'4.0', b'5.0', 1))
    assert C.main(base) == 2 and 'differs' in capsys.readouterr().out
    p.write_bytes(t)
    assert C.main(base) == 0
    assert sha(w0['hop'] / 'mien_A.csv') == sha(p)
    base[base.index(str(w0['tl0']))] = str(w0['tl'])                         # SA-13: after the first A run
    assert C.main(base + ['--thay']) == 2 and 'locked' in capsys.readouterr().out


def test_them_s03_quet_both_or_none(w0, capsys):
    run = s03_run(w0['tmp'])
    (run / 'levels.csv').write_bytes(b'\x89PNG\r\n')                         # SA-11: second file bad
    files = sorted(os.listdir(w0['hop']))
    assert C.main(['--stra-a', str(w0['sa']), 'them', '--loai', 's03_quet', '--nguon', str(run), '--ly-do', 'x']) == 2
    assert sorted(os.listdir(w0['hop'])) == files


# ── ke-khai ────────────────────────────────────────────────────────────────────────────────────────────────────────

def fence_files(sa, monkeypatch, drop=None, bad_line=False):
    (sa / '_codex_test').mkdir()
    lines = ['codex: x']
    for k in C.RAO_IDS:
        if k == drop:
            continue
        res = 'chay ok' if k in C.RAO_RUNS else ('200' if bad_line and k == '7' else 'Access is denied')
        lines.append(f'{k} bai                => {res}')
    p = sa / '_codex_test' / 'ket_qua_2026-11-14.txt'
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (sa / C.RAO_CLAUDE).write_text('7/7', encoding='utf-8')
    monkeypatch.setattr(C, 'SHA_RAO', {C.RAO_CLAUDE: sha(sa / C.RAO_CLAUDE), C.RAO_CODEX: sha(p)})
    (sa / 'NHAT_KY').mkdir()
    (sa / 'CODEX').mkdir()
    (sa / 'NHAT_KY' / '2026-11-20.md').write_text('buổi 1\n', encoding='utf-8')


def test_ke_khai(world, capsys, monkeypatch):
    w = world
    sa = w['sa']
    fence_files(sa, monkeypatch)
    out0 = w['tmp'] / 'kk' / 'hop_thu_A_0.yaml'
    out0.parent.mkdir()
    assert C.main(['--stra-a', str(sa), 'ke-khai', '--out', str(out0)]) == 2       # SA-18: --truoc or --lan-dau
    assert '--lan-dau' in capsys.readouterr().out and not out0.exists()
    assert C.main(['--stra-a', str(sa), 'ke-khai', '--out', str(out0), '--lan-dau']) == 2   # lock SHAs not there
    d = yaml.safe_load(open(out0, encoding='utf-8'))
    assert any('config.toml' in x for x in d['loi']) and d['nhiem_thong_tin']['co'] is False
    assert C.main(['--stra-a', str(sa), 'ke-khai', '--out', str(out0.parent / 'hop_thu_A_x.yaml'), '--lan-dau']) == 2
    assert 'already' in capsys.readouterr().out
    (sa / 'NHAT_KY' / '2026-11-20.md').write_text('buổi 1\nNHIEM: lỡ thấy x\n', encoding='utf-8')
    (w['hop'] / 'la.txt').write_text('x', encoding='utf-8')
    out2 = w['tmp'] / 'kk' / 'hop_thu_A_2.yaml'
    C.main(['--stra-a', str(sa), 'ke-khai', '--out', str(out2), '--truoc', str(out0)])
    d2 = yaml.safe_load(open(out2, encoding='utf-8'))
    assert d2['nhiem_thong_tin']['co'] is True and any('strange file' in x for x in d2['loi'])
    assert not any('append-only' in x for x in d2['loi'])
    (sa / 'NHAT_KY' / '2026-11-20.md').write_text('sửa đầu\n', encoding='utf-8')
    out3 = w['tmp'] / 'kk' / 'hop_thu_A_3.yaml'
    C.main(['--stra-a', str(sa), 'ke-khai', '--out', str(out3), '--truoc', str(out2)])
    assert any('append-only' in x for x in yaml.safe_load(open(out3, encoding='utf-8'))['loi'])
    empty = w['tmp'] / 'kk' / 'rong.yaml'
    empty.write_text('a: 1\n', encoding='utf-8')
    assert C.main(['--stra-a', str(sa), 'ke-khai', '--out', str(w['tmp'] / 'kk' / 'hop_thu_A_4.yaml'), '--truoc',
                   str(empty)]) == 2 and 'not a declaration' in capsys.readouterr().out
    assert C.main(['--stra-a', str(sa), 'ke-khai', '--out', str(w['tmp'] / 'k1.yaml'), '--lan-dau']) == 2
    assert 'hop_thu_A_' in capsys.readouterr().out                                 # SA-18: one naming, one place


def test_declaration_problems(world, monkeypatch):
    """SA-20, SA-22, SA2-4: a declaration is stale when NHAT_KY changed after it; bad Codex calls; bad types."""
    w = world
    sa = w['sa']
    fence_files(sa, monkeypatch)
    out = w['tmp'] / 'hop_thu_A_1.yaml'
    C.main(['--stra-a', str(sa), 'ke-khai', '--out', str(out), '--lan-dau'])
    kk = C.read_ke_khai(str(out))
    kk['loi'] = []
    assert any('STRA_A sessions' in x for x in C.declaration_problems(kk, str(sa)))
    kk['tro_ly_claude']['phien'] = ['']
    assert any('STRA_A sessions' in x for x in C.declaration_problems(kk, str(sa)))
    kk['tro_ly_claude']['phien'] = ['local_x,claude-sonnet,medium']
    assert C.declaration_problems(kk, str(sa)) == []
    (sa / 'NHAT_KY' / '2026-11-20.md').write_text('buổi 1\nthêm\n', encoding='utf-8')
    assert any('NHAT_KY' in x for x in C.declaration_problems(kk, str(sa)))
    kk['tro_ly_codex']['lan_goi'] = [dict(log='log_1.txt', model='gpt-6.1-sol', effort='xhigh')]
    assert any('Codex calls' in x for x in C.declaration_problems(kk, str(sa)))
    bad = w['tmp'] / 'bad.yaml'
    bad.write_text('loi: []\nhop_thu: null\nnhat_ky: {}\ntro_ly_codex: {lan_goi: []}\n', encoding='utf-8')
    with pytest.raises(C.Refused, match='not a declaration'):
        C.read_ke_khai(str(bad))


@pytest.mark.parametrize('kw, msg', [(dict(bad_line=True), 'test 7'), (dict(drop='4c'), 'differ from')])
def test_rao_wrong(world, monkeypatch, kw, msg):
    fence_files(world['sa'], monkeypatch, **kw)                                    # SA-19
    _, bad = C.rao_check(str(world['sa']))
    assert any(msg in x for x in bad)


def test_rao_pinned(world, monkeypatch):
    fence_files(world['sa'], monkeypatch)
    (world['sa'] / C.RAO_CLAUDE).write_text('', encoding='utf-8')
    _, bad = C.rao_check(str(world['sa']))
    assert any('approved result' in x for x in bad)


def test_box_check(world):
    (world['hop'] / 'spec.md').write_text('đổi\n', encoding='utf-8')
    _, bad = C.box_check(str(world['hop']))
    assert any('differs from its latest row' in x for x in bad)
    os.remove(world['hop'] / 'spec.md')                                            # SA-17: reverse direction
    (world['hop'] / '.dang_chep_x').mkdir()
    _, bad = C.box_check(str(world['hop']))
    assert any('no file on the disk' in x for x in bad) and any('temp folder' in x for x in bad)


def test_round3_fixes(world, monkeypatch, capsys):
    """SA3-1 element types; SA3-2 version only after the SHA; SA-18 newest baseline; SA-17 one file per row;
    SA-22 the mailbox itself checked against a declaration."""
    w = world
    sa = w['sa']
    fence_files(sa, monkeypatch)
    bad = w['tmp'] / 'b.yaml'
    bad.write_text('loi: []\nhop_thu: {readme_sha256: x}\nnhat_ky: {}\ntro_ly_codex: {lan_goi: [null]}\n',
                   encoding='utf-8')
    with pytest.raises(C.Refused):
        C.read_ke_khai(str(bad))
    exe = w['tmp'] / 'codex.exe'
    exe.write_bytes(b'gia')
    hc = C.load_hoi_codex()
    hc.CODEX_EXE = str(exe)
    monkeypatch.setattr(C, 'load_hoi_codex', lambda: hc)
    monkeypatch.setattr(C, '_tool_version', lambda cmd: pytest.fail('ran a program with a wrong SHA'))
    assert C.codex_cli_version() == 'SHA khác bản đã thử rào'
    monkeypatch.setattr(C, '_tool_version', lambda cmd: '')
    kk = w['tmp'] / 'kk'
    kk.mkdir()
    a0, a1, a2 = (str(kk / f'hop_thu_A_{i}.yaml') for i in range(3))
    C.main(['--stra-a', str(sa), 'ke-khai', '--out', a0, '--lan-dau'])
    import time
    time.sleep(1.1)
    C.main(['--stra-a', str(sa), 'ke-khai', '--out', a1, '--truoc', a0])
    capsys.readouterr()
    assert C.main(['--stra-a', str(sa), 'ke-khai', '--out', a2, '--truoc', a0]) == 2
    assert 'newer' in capsys.readouterr().out
    assert C.main(args(w)) == 0 and C.main(args(w, thay=True)) == 0         # same bytes twice
    cu = next(n for n in os.listdir(w['hop'] / 'luot') if '__cu_' in n)
    os.remove(w['hop'] / 'luot' / cu / 'log.csv')
    _, bb = C.box_check(str(w['hop']))
    assert any('log.csv' in x and 'no file on the disk' in x for x in bb)
    d = C.read_ke_khai(a1)
    (w['hop'] / 'spec.md').write_text('đổi\n', encoding='utf-8')
    assert any(x.startswith('mailbox:') for x in C.declaration_problems(d, str(sa)))


def test_round4_fixes(world, monkeypatch, capsys):
    """SA-18 chain (two declarations in the same second); SA-20 call entries compared with the real logs;
    SA4-1 log name typed."""
    w = world
    sa = w['sa']
    fence_files(sa, monkeypatch)
    monkeypatch.setattr(C, '_tool_version', lambda cmd: '')
    kk = w['tmp'] / 'kk'
    kk.mkdir()
    a0, a1, a2 = (str(kk / f'hop_thu_A_{i}.yaml') for i in range(3))
    C.main(['--stra-a', str(sa), 'ke-khai', '--out', a0, '--lan-dau'])
    C.main(['--stra-a', str(sa), 'ke-khai', '--out', a1, '--truoc', a0])
    capsys.readouterr()
    assert C.main(['--stra-a', str(sa), 'ke-khai', '--out', a2, '--truoc', a0]) == 2
    out = capsys.readouterr().out
    assert 'follows' in out or 'newer' in out
    (sa / 'CODEX' / 'log_20261120_1.txt').write_text(
        'x\n--------\nworkdir: D:\\STRA_A\nmodel: gpt-5\nreasoning effort: high\n--------\n', encoding='utf-8')
    d = C.read_ke_khai(a1)
    d['loi'], d['tro_ly_claude']['phien'] = [], ['s,m,e']
    d['tro_ly_codex']['lan_goi'] = [dict(log='log_20261120_1.txt', model='gpt-6.1-sol', effort='high')]
    d['nhat_ky'], _ = C.nhat_ky(str(sa), None)
    assert any('differ from the logs' in x for x in C.declaration_problems(d, str(sa)))
    bad = w['tmp'] / 'b.yaml'
    bad.write_text('loi: []\nhop_thu: {readme_sha256: x}\nnhat_ky: {}\ntro_ly_claude: {phien: []}\n'
                   'tro_ly_codex: {lan_goi: [{log: [], model: m, effort: e}]}\n', encoding='utf-8')
    with pytest.raises(C.Refused):
        C.read_ke_khai(str(bad))


def test_baseline_name_case(world, monkeypatch, capsys):
    """SA-18 (round 5): the predecessor named with another case is the same file on Windows."""
    w = world
    sa = w['sa']
    fence_files(sa, monkeypatch)
    monkeypatch.setattr(C, '_tool_version', lambda cmd: '')
    kk = w['tmp'] / 'kk'
    kk.mkdir()
    a0, a1, a2 = (str(kk / f'hop_thu_A_{i}.yaml') for i in range(3))
    C.main(['--stra-a', str(sa), 'ke-khai', '--out', a0, '--lan-dau'])
    C.main(['--stra-a', str(sa), 'ke-khai', '--out', a1, '--truoc', a0])
    d1 = yaml.safe_load(open(a1, encoding='utf-8'))
    d1['luc'] = yaml.safe_load(open(a0, encoding='utf-8'))['luc']        # same second
    with open(a1, 'w', encoding='utf-8') as f:
        yaml.safe_dump(d1, f, allow_unicode=True, sort_keys=False)
    capsys.readouterr()
    assert C.main(['--stra-a', str(sa), 'ke-khai', '--out', a2, '--truoc', str(kk / 'HOP_THU_A_0.YAML')]) == 2
    assert 'follows' in capsys.readouterr().out
