"""Test kiem_sha: sha CRLF/LF, tim_cap theo dang, ung_vien, CLI."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import kiem_sha as KS  # noqa: E402

H_A = 'a' * 64
H_B = 'b' * 64
H_C = 'c' * 64


def sha(data):
    return hashlib.sha256(data).hexdigest()


# ---------- sha_cac_dang / kiem ----------

def test_file_lf(tmp_path):
    f = tmp_path / 'a.txt'
    data = b'line1\nline2\n'
    f.write_bytes(data)
    d = KS.sha_cac_dang(f)
    assert d['tho'] == sha(data) and d['lf'] == d['tho']
    assert d['crlf'] == sha(data.replace(b'\n', b'\r\n'))
    assert KS.kiem(f, sha(data)) == 'tho'
    assert KS.kiem(f, sha(data.replace(b'\n', b'\r\n'))) == 'crlf'


def test_file_crlf(tmp_path):
    f = tmp_path / 'a.txt'
    data = b'a\r\nb\r\n'
    f.write_bytes(data)
    d = KS.sha_cac_dang(f)
    assert d['tho'] == sha(data) and d['crlf'] == d['tho']
    assert d['lf'] == sha(b'a\nb\n')
    assert KS.kiem(f, sha(b'a\nb\n')) == 'lf'


def test_file_hon_hop(tmp_path):
    f = tmp_path / 'a.txt'
    f.write_bytes(b'a\r\nb\n')
    d = KS.sha_cac_dang(f)
    assert d['tho'] == sha(b'a\r\nb\n')
    assert d['tho'] != d['lf'] and d['tho'] != d['crlf']
    # CRLF co san khong bi nhan doi
    assert d['crlf'] == sha(b'a\r\nb\r\n')
    assert d['crlf'] != sha(b'a\r\r\nb\r\n')
    assert KS.kiem(f, sha(b'a\nb\n')) == 'lf'
    assert KS.kiem(f, sha(b'a\r\nb\r\n')) == 'crlf'


def test_cr_don_giu_nguyen(tmp_path):
    f = tmp_path / 'a.txt'
    f.write_bytes(b'a\rb\n')
    d = KS.sha_cac_dang(f)
    assert d['tho'] == d['lf'] == sha(b'a\rb\n')
    assert d['crlf'] == sha(b'a\rb\r\n')


def test_khong_xuong_dong_va_rong(tmp_path):
    f1 = tmp_path / 'a.txt'
    f1.write_bytes(b'abc')
    d = KS.sha_cac_dang(f1)
    assert d['tho'] == d['lf'] == d['crlf'] == sha(b'abc')
    assert KS.kiem(f1, sha(b'abc')) == 'tho'
    f2 = tmp_path / 'b.txt'
    f2.write_bytes(b'')
    d2 = KS.sha_cac_dang(f2)
    assert d2['tho'] == d2['lf'] == d2['crlf'] == sha(b'')
    assert KS.kiem(f2, sha(b'')) == 'tho'


def test_byte_nul_la_nhi_phan(tmp_path):
    f = tmp_path / 'a.txt'
    data = b'a\x00b\n'
    f.write_bytes(data)
    d = KS.sha_cac_dang(f)
    assert set(d) == {'tho'}
    assert KS.kiem(f, sha(data)) == 'tho'
    assert KS.kiem(f, sha(b'ab\n')) == 'khong_khop'


@pytest.mark.parametrize('ext', ['.png', '.npz', '.mov', '.MOV', '.mp4', '.pdf'])
def test_duoi_nhi_phan(tmp_path, ext):
    f = tmp_path / ('a' + ext)
    data = b'x\r\ny\n'
    f.write_bytes(data)
    d = KS.sha_cac_dang(f)
    assert set(d) == {'tho'}
    assert KS.kiem(f, sha(data)) == 'tho'
    assert KS.kiem(f, sha(b'x\ny\n')) == 'khong_khop'


def test_sha_viet_hoa_khoang_trang(tmp_path):
    f = tmp_path / 'a.txt'
    f.write_bytes(b'abc\n')
    assert KS.kiem(f, '  ' + sha(b'abc\n').upper() + '  ') == 'tho'


def test_sha_sai_va_khong_hex(tmp_path):
    f = tmp_path / 'a.txt'
    f.write_bytes(b'abc\n')
    assert KS.kiem(f, '0' * 64) == 'khong_khop'
    assert KS.kiem(f, 'khong-phai-hex') == 'khong_khop'


def test_thieu_file_va_thu_muc(tmp_path):
    assert KS.kiem(tmp_path / 'khong_co.txt', H_A) == 'thieu_file'
    d = tmp_path / 'mot_thu_muc'
    d.mkdir()
    assert KS.kiem(d, H_A) == 'thieu_file'


def test_cache_vo_hieu_khi_doi_noi_dung(tmp_path):
    f = tmp_path / 'a.txt'
    f.write_bytes(b'noidung1')
    assert KS.sha_cac_dang(f)['tho'] == sha(b'noidung1')
    f.write_bytes(b'noidung-khac-dai-hon')
    assert KS.sha_cac_dang(f)['tho'] == sha(b'noidung-khac-dai-hon')


# ---------- tim_cap ----------

def test_dang_khoi(tmp_path):
    for kk, dang in (('duong_dan', 'khoi_duong_dan'),
                     ('path', 'khoi_path'),
                     ('file', 'khoi_file')):
        cap, kh, rong = KS.tim_cap({kk: 'data/a.csv', 'sha256': H_A})
        assert len(cap) == 1 and not kh and rong == 0
        c = cap[0]
        assert c['khoa'] == 'sha256' and c['dang'] == dang
        assert c['duong_dan'] == 'data/a.csv' and c['sha'] == H_A


def test_dang_ten_sha256():
    cap, kh, rong = KS.tim_cap({'video': 'clips/v.mp4', 'video_sha256': H_A})
    assert len(cap) == 1 and cap[0]['dang'] == 'ten_sha256'
    assert cap[0]['khoa'] == 'video_sha256'
    assert cap[0]['duong_dan'] == 'clips/v.mp4'


def test_dang_ten_file_va_ten_path():
    cap, _, _ = KS.tim_cap({'video_file': 'v.mp4', 'video_sha256': H_A})
    assert cap[0]['dang'] == 'ten_file' and cap[0]['duong_dan'] == 'v.mp4'
    cap, _, _ = KS.tim_cap({'video_path': 'v.mp4', 'video_sha256': H_A})
    assert cap[0]['dang'] == 'ten_path'


def test_ten_file_bi_chan_khi_co_sha_rieng():
    obj = {'video_file': 'v.mp4', 'video_sha256': H_A,
           'video_file_sha256': H_B}
    cap, kh, _ = KS.tim_cap(obj)
    assert not [c for c in cap if c['khoa'] == 'video_sha256']
    e = [c for c in cap if c['khoa'] == 'video_file_sha256']
    assert len(e) == 1 and e[0]['dang'] == 'ten_sha256'
    kh_k = {x['khoa']: x['ly_do'] for x in kh}
    assert kh_k.get('video_sha256') == 'khong_co_duong_dan'


def test_dang_hau_to():
    cap, kh, _ = KS.tim_cap({'file_a': 'data/a.csv', 'sha256_a': H_A})
    assert len(cap) == 1 and cap[0]['dang'] == 'sha256_hau_to'
    assert cap[0]['duong_dan'] == 'data/a.csv'


def test_sha256_file_rieng():
    obj = {'file': 'data/a.csv', 'sha256': H_A, 'sha256_file': H_B}
    cap, kh, _ = KS.tim_cap(obj)
    assert len(cap) == 1
    assert cap[0]['khoa'] == 'sha256_file'
    assert cap[0]['duong_dan'] == 'data/a.csv'
    assert cap[0]['sha'] == H_B
    assert cap[0]['dang'] == 'sha256_hau_to'
    assert len(kh) == 1
    assert kh[0]['khoa'] == 'sha256'
    assert kh[0]['ly_do'] == 'co_sha256_file_rieng'


def test_sha256_duong_dan_rieng():
    obj = {'duong_dan': 'data/a.csv', 'sha256': H_A,
           'sha256_duong_dan': H_B}
    cap, kh, _ = KS.tim_cap(obj)
    assert len(cap) == 1
    assert cap[0]['khoa'] == 'sha256_duong_dan'
    assert cap[0]['duong_dan'] == 'data/a.csv'
    assert cap[0]['sha'] == H_B
    assert cap[0]['dang'] == 'sha256_hau_to'
    assert len(kh) == 1
    assert kh[0]['khoa'] == 'sha256'
    assert kh[0]['ly_do'] == 'co_sha256_file_rieng'


def test_dang_danh_sach():
    obj = {'frames': ['f1.png', 'f2.png'], 'frames_sha256': [H_A, H_B]}
    cap, kh, _ = KS.tim_cap(obj)
    assert len(cap) == 2 and not kh
    assert cap[0]['khoa'] == 'frames_sha256[0]'
    assert cap[0]['duong_dan'] == 'f1.png' and cap[0]['sha'] == H_A
    assert cap[1]['khoa'] == 'frames_sha256[1]'
    assert cap[1]['dang'] == 'danh_sach' and cap[1]['sha'] == H_B


def test_manifest_duoi_khoa_sha256():
    cap, kh, _ = KS.tim_cap({'sha256': {'a.csv': H_A, 'sub/b.csv': H_B}})
    assert len(cap) == 2 and not kh
    assert cap[0]['khoa'] == 'sha256["a.csv"]'
    assert cap[0]['duong_dan'] == 'a.csv' and cap[0]['sha'] == H_A
    assert cap[1]['khoa'] == 'sha256["sub/b.csv"]'
    assert all(c['dang'] == 'manifest' for c in cap)


def test_manifest_thuong():
    obj = {'anh': {'x.png': H_A, 'y.png': H_B}, 'khac': 1}
    cap, kh, _ = KS.tim_cap(obj)
    assert len(cap) == 2
    assert cap[0]['khoa'] == 'anh["x.png"]' and cap[0]['dang'] == 'manifest'


def test_khong_co_duong_dan():
    _, kh, _ = KS.tim_cap({'sha256': H_A})
    assert len(kh) == 1 and kh[0]['ly_do'] == 'khong_co_duong_dan'
    assert kh[0]['khoa'] == 'sha256'
    _, kh, _ = KS.tim_cap({'clip': 'IMG_3620', 'clip_sha256': H_A})
    assert kh[0]['ly_do'] == 'khong_co_duong_dan'
    _, kh, _ = KS.tim_cap({'frames_out_sha256': H_A})
    assert kh[0]['ly_do'] == 'khong_co_duong_dan'
    _, kh, _ = KS.tim_cap({'script': {'commit': 'abc', 'sha256': H_A}})
    assert kh[0]['ly_do'] == 'khong_co_duong_dan'
    assert kh[0]['khoa'] == 'script.sha256'


def test_nhieu_duong_dan():
    obj = {'duong_dan': 'a.csv', 'file': 'b.csv', 'sha256': H_A}
    cap, kh, _ = KS.tim_cap(obj)
    assert not cap and kh[0]['ly_do'] == 'nhieu_duong_dan'


def test_sha_khong_hop_le():
    _, kh, _ = KS.tim_cap({'duong_dan': 'a.csv', 'sha256': 'zz'})
    assert kh[0]['ly_do'] == 'sha_khong_hop_le'


def test_danh_sach_khong_ghep():
    obj = {'names': ['x', 'y'], 'names_sha256': [H_A, H_B]}
    cap, kh, _ = KS.tim_cap(obj)
    assert not cap and kh[0]['ly_do'] == 'danh_sach_khong_ghep'


def test_manifest_khoa_khong_phai_file():
    _, kh, _ = KS.tim_cap({'sha256': {'esp32': H_A, 'frames': H_B}})
    assert len(kh) == 2
    assert all(x['ly_do'] == 'manifest_khoa_khong_phai_file' for x in kh)


def test_sha_none_dem_rong():
    cap, kh, rong = KS.tim_cap(
        {'a_sha256': None, 'b': 1, 'c': {'d_sha256': None}})
    assert rong == 2 and not cap and not kh


def test_khoa_chinh_xac():
    obj = {'nguon': [{'duong_dan': 'x.csv', 'sha256': H_A},
                     {'duong_dan': 'y.csv', 'sha256': H_B}]}
    cap, _, _ = KS.tim_cap(obj)
    assert cap[0]['khoa'] == 'nguon[0].sha256'
    assert cap[1]['khoa'] == 'nguon[1].sha256'
    obj = {'session': ['p1.png', 'p2.png'], 'session_sha256': [H_A, H_B]}
    cap, _, _ = KS.tim_cap(obj)
    assert cap[1]['khoa'] == 'session_sha256[1]'


# ---------- ung_vien / kiem_cap ----------

def test_ung_vien_tuong_doi(tmp_path):
    # C3-23 vong 2: duong co thu muc chi giai qua goc luot gan nhat
    jdir = tmp_path / '2026-11-20_run01'
    (jdir / 'data').mkdir(parents=True)
    f = jdir / 'data' / 'a.csv'
    f.write_bytes(b'1,2\n')
    out = KS.ung_vien('data/a.csv', jdir, tmp_path)
    assert out == [('thu_muc_luot', f)]
    kq, cach, thuc = KS.kiem_cap('data/a.csv', sha(b'1,2\n'), jdir, tmp_path)
    assert (kq, cach) == ('tho', 'thu_muc_luot')
    assert os.path.isfile(thuc)


def test_ung_vien_goc_repo(tmp_path):
    # C3-23 vong 2: khong con ung vien goc_repo cho duong co thu muc
    repo = tmp_path / 'repo'
    (repo / 'd').mkdir(parents=True)
    (repo / 'd' / 'a.csv').write_bytes(b'x\n')
    jdir = tmp_path / 'j'
    jdir.mkdir()
    kq, cach, thuc = KS.kiem_cap('d/a.csv', sha(b'x\n'), jdir, repo)
    assert (kq, cach, thuc) == ('khong_co_goc_luot', '', '')


def test_json_sai_repo_dung(tmp_path):
    repo = tmp_path / 'repo'
    jdir = repo / 'j'
    jdir.mkdir(parents=True)
    (jdir / 'a.csv').write_bytes(b'sai\n')
    (repo / 'a.csv').write_bytes(b'dung\n')
    kq, cach, _ = KS.kiem_cap('a.csv', sha(b'dung\n'), jdir, repo)
    # C3-23: phan giai khoa theo thu_muc_json (phuong an Codex), khong thu goc_repo
    assert (kq, cach) == ('khong_khop', 'thu_muc_json')


def test_tuyet_doi(tmp_path):
    f = tmp_path / 'a.csv'
    f.write_bytes(b'z\n')
    out = KS.ung_vien(str(f), tmp_path, tmp_path)
    assert out[0][0] == 'tuyet_doi'
    kq, cach, _ = KS.kiem_cap(str(f), sha(b'z\n'), tmp_path, tmp_path)
    assert (kq, cach) == ('tho', 'tuyet_doi')


def test_tuyet_doi_doi_goc(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'data').mkdir(parents=True)
    f = repo / 'data' / 'a.csv'
    f.write_bytes(b'q\n')
    for s in ('X:/khac/repo/data/a.csv', 'X:\\khac\\repo\\data\\a.csv'):
        cachs = [c for c, _ in KS.ung_vien(s, tmp_path, repo)]
        assert 'tuyet_doi' in cachs and 'tuyet_doi_doi_goc' in cachs
        kq, cach, thuc = KS.kiem_cap(s, sha(b'q\n'), tmp_path, repo)
        assert (kq, cach) == ('tho', 'tuyet_doi_doi_goc')
        assert os.path.normpath(thuc) == os.path.normpath(str(f))


def test_bo_qua_sealed(tmp_path, monkeypatch):
    jdir = tmp_path / 'j'
    jdir.mkdir()

    def cam(p, s):
        raise AssertionError('khong duoc doc sealed')

    monkeypatch.setattr(KS, 'kiem', cam)
    kq, cach, thuc = KS.kiem_cap('sealed/a.csv', H_A, jdir, tmp_path)
    assert (kq, cach, thuc) == ('bo_qua_sealed', '', '')


# ---------- CLI ----------

def _lam_repo(tmp_path):
    repo = tmp_path / 'repo'
    (repo / '.git').mkdir(parents=True)
    goc = repo / 'so_lieu'
    goc.mkdir()
    return repo, goc


def _doc_csv(path):
    lines = [l for l in path.read_text(encoding='utf-8').splitlines()
             if not l.startswith('#')]
    return list(csv.DictReader(lines))


def test_tim_goc_repo(tmp_path):
    repo = tmp_path / 'repo'
    (repo / '.git').mkdir(parents=True)
    sub = repo / 'a' / 'b'
    sub.mkdir(parents=True)
    assert KS.tim_goc_repo(sub) == repo.resolve()


def test_cli_ok(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    out = tmp_path / 'out'
    (goc / 'data').mkdir()
    (goc / 'data' / 'a.txt').write_bytes(b'x\r\ny\n')
    (goc / 'b.txt').write_bytes(b'bb')
    (goc / 'r1.json').write_text(json.dumps(
        {'a': {'duong_dan': 'data/a.txt', 'sha256': sha(b'x\ny\n')}}),
        encoding='utf-8')
    (goc / 'r2.json').write_text(json.dumps(
        {'file': 'b.txt', 'sha256': sha(b'bb')}), encoding='utf-8')
    # C3-23 vong 2: duong co thu muc giai qua --goc-luot
    ma = KS.main(['--goc', str(goc), '--goc-luot', str(goc),
                  '--ra', str(out)])
    assert ma == 0
    lines = (out / 'bang.csv').read_text(encoding='utf-8').splitlines()
    assert lines[0] == ('file_ghi,khoa,duong_dan,sha_ghi,ket_qua,cach,'
                        'duong_dan_thuc,dang,ghi_chu')
    assert lines[-1] == '#ket_qua,ma=0'
    rows = {r['khoa']: r for r in csv.DictReader(lines[:-1])}
    assert rows['a.sha256']['ket_qua'] == 'lf'
    assert rows['a.sha256']['cach'] == 'goc_luot'
    assert rows['a.sha256']['dang'] == 'khoi_duong_dan'
    assert rows['sha256']['ket_qua'] == 'tho'
    assert rows['sha256']['ghi_chu'] == 'ten_tran'
    tt = json.loads((out / 'tom_tat.json').read_text(encoding='utf-8'))
    assert tt['ma'] == 0 and tt['so_json'] == 2 and tt['lf'] == 1


def test_cli_thieu_file(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    out = tmp_path / 'out'
    (goc / 'r.json').write_text(json.dumps(
        {'duong_dan': 'khong/co.txt', 'sha256': H_A}), encoding='utf-8')
    # C3-23 vong 2: duong co thu muc giai qua --goc-luot, dich van thieu
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--goc-luot', str(goc), '--ra', str(out)])
    assert ma == 1
    lines = (out / 'bang.csv').read_text(encoding='utf-8').splitlines()
    assert lines[-1] == '#ket_qua,ma=1'
    rows = list(csv.DictReader(lines[:-1]))
    assert rows[0]['ket_qua'] == 'thieu_file'


def test_cli_khong_khop(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    out = tmp_path / 'out'
    (goc / 'c.txt').write_bytes(b'abc')
    (goc / 'r.json').write_text(json.dumps(
        {'file': 'c.txt', 'sha256': H_A}), encoding='utf-8')
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 1
    rows = _doc_csv(out / 'bang.csv')
    assert rows[0]['ket_qua'] == 'khong_khop'


def test_cli_sealed_khong_doc(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    out = tmp_path / 'out'
    (goc / 'sealed').mkdir()
    (goc / 'sealed' / 'x.json').write_text('{ json hong', encoding='utf-8')
    (goc / 'ok.txt').write_bytes(b'ok')
    (goc / 'r.json').write_text(json.dumps(
        {'duong_dan': 'sealed/t.txt', 'sha256': H_A}), encoding='utf-8')
    (goc / 'r2.json').write_text(json.dumps(
        {'file': 'ok.txt', 'sha256': sha(b'ok')}), encoding='utf-8')
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 0
    kh = _doc_csv(out / 'khong_hieu.csv')
    assert any(e['ly_do'] == 'tro_vao_sealed_khong_doc' for e in kh)
    assert not any('sealed' in e['file_ghi'] for e in kh)
    rows = _doc_csv(out / 'bang.csv')
    assert not any('sealed' in r['file_ghi'] for r in rows)


def test_cli_json_loi(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    out = tmp_path / 'out'
    (goc / 'bad.json').write_text('{ loi roi', encoding='utf-8')
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    # C3-22: json_loi tinh la chua xac minh nen ma 3 thay cho 0
    assert ma == 3
    kh = _doc_csv(out / 'khong_hieu.csv')
    assert any(e['ly_do'] == 'json_loi' and e['file_ghi'] == 'bad.json'
               for e in kh)
    tt = json.loads((out / 'tom_tat.json').read_text(encoding='utf-8'))
    assert tt['so_chua_xac_minh'] == 1


def test_cli_ra_trong_repo(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    (goc / 'r.json').write_text('{}', encoding='utf-8')
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(repo / 'out')])
    assert ma == 2
    assert not (repo / 'out').exists()
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(goc / 'out')])
    assert ma == 2
    assert not (goc / 'out').exists()


def test_cli_goc_khong_ton_tai(tmp_path):
    ma = KS.main(['--goc', str(tmp_path / 'khong_co'),
                  '--ra', str(tmp_path / 'o')])
    assert ma == 2


# ---------- C3-22 ----------

def test_c3_22_manifest_sha256_hash_sai():
    cap, kh, _ = KS.tim_cap({'sha256': {'a.csv': 'bad'}})
    assert not cap
    assert len(kh) == 1
    assert kh[0]['khoa'] == 'sha256["a.csv"]'
    assert kh[0]['ly_do'] == 'sha_khong_hop_le'


def test_c3_22_manifest_thuong_mot_hash_sai():
    cap, kh, _ = KS.tim_cap({'anh': {'x.png': H_A, 'y.png': 'bad'}})
    assert len(cap) == 1
    assert cap[0]['duong_dan'] == 'x.png'
    assert cap[0]['dang'] == 'manifest'
    assert len(kh) == 1
    assert kh[0]['khoa'] == 'anh["y.png"]'
    assert kh[0]['ly_do'] == 'sha_khong_hop_le'


def test_c3_22_manifest_none_va_kieu_la():
    # C3-22 vong 2: null o truong bat buoc la sha_rong_bat_buoc, khong dem rong
    cap, kh, rong = KS.tim_cap({'sha256': {'a.csv': None, 'b.csv': 5}})
    assert not cap
    assert rong == 0
    ly = {e['khoa']: e['ly_do'] for e in kh}
    assert ly == {'sha256["a.csv"]': 'sha_rong_bat_buoc',
                  'sha256["b.csv"]': 'kieu_khong_ho_tro'}


def test_c3_22_cli_manifest_hash_sai(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    out = tmp_path / 'out'
    (goc / 'a.csv').write_bytes(b'dung\n')
    (goc / 'r.json').write_text('{"sha256": {"a.csv": "bad"}}',
                               encoding='utf-8')
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 3
    lines = (out / 'bang.csv').read_text(encoding='utf-8').splitlines()
    assert lines[-1] == '#ket_qua,ma=3'
    tt = json.loads((out / 'tom_tat.json').read_text(encoding='utf-8'))
    assert tt['so_chua_xac_minh'] == 1 and tt['ma'] == 3


def test_c3_22_cli_uu_tien_ma_1(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    out = tmp_path / 'out'
    (goc / 'c.txt').write_bytes(b'abc')
    (goc / 'r.json').write_text(json.dumps(
        {'file': 'c.txt', 'sha256': H_A}), encoding='utf-8')
    (goc / 'bad.json').write_text('{ loi roi', encoding='utf-8')
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 1


def test_c3_22_cli_chi_thong_tin_ma_0(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    out = tmp_path / 'out'
    (goc / 'b.txt').write_bytes(b'bb')
    (goc / 'r.json').write_text(json.dumps(
        {'file': 'b.txt', 'sha256': sha(b'bb')}), encoding='utf-8')
    (goc / 'r2.json').write_text(json.dumps({'sha256': H_A}),
                                encoding='utf-8')
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 0
    tt = json.loads((out / 'tom_tat.json').read_text(encoding='utf-8'))
    assert tt['so_chua_xac_minh'] == 0


def test_c3_22_cli_manifest_khoa_khong_phai_file(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    out = tmp_path / 'out'
    (goc / 'b.txt').write_bytes(b'bb')
    (goc / 'r.json').write_text(json.dumps(
        {'file': 'b.txt', 'sha256': sha(b'bb')}), encoding='utf-8')
    (goc / 'r2.json').write_text(json.dumps(
        {'sha256': {'esp32': H_A}}), encoding='utf-8')
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 3


# ---------- C3-23 ----------

def test_c3_23_codex_ten_tran(tmp_path):
    repo = tmp_path / 'repo'
    jdir = repo / 'results' / 'run'
    jdir.mkdir(parents=True)
    (jdir / 'a.csv').write_bytes(b'sai\n')
    (repo / 'a.csv').write_bytes(b'dung\n')
    kq, cach, thuc = KS.kiem_cap('a.csv', sha(b'dung\n'), jdir, repo)
    assert (kq, cach) == ('khong_khop', 'thu_muc_json')
    assert os.path.normpath(thuc) == os.path.normpath(str(jdir / 'a.csv'))


def test_c3_23_codex_co_thu_muc(tmp_path):
    repo = tmp_path / 'repo'
    jdir = repo / 'results' / 'run'
    (jdir / 'd').mkdir(parents=True)
    (jdir / 'd' / 'a.csv').write_bytes(b'sai\n')
    (repo / 'd').mkdir(parents=True)
    (repo / 'd' / 'a.csv').write_bytes(b'dung\n')
    # C3-23 vong 2: khong goc luot thi khong giai duoc duong co thu muc
    kq, cach, thuc = KS.kiem_cap('d/a.csv', sha(b'dung\n'), jdir, repo)
    assert (kq, cach, thuc) == ('khong_co_goc_luot', '', '')
    kq, cach, thuc = KS.kiem_cap('d/a.csv', sha(b'dung\n'), jdir, repo,
                                 goc_luot=jdir)
    assert (kq, cach) == ('khong_khop', 'goc_luot')
    assert os.path.normpath(thuc) == os.path.normpath(
        str(jdir / 'd' / 'a.csv'))


def test_c3_23_cli_codex(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    rund = goc / 'run'
    rund.mkdir()
    (rund / 'a.csv').write_bytes(b'sai\n')
    (repo / 'a.csv').write_bytes(b'dung\n')
    (rund / 'r.json').write_text(json.dumps(
        {'sha256': {'a.csv': sha(b'dung\n')}}), encoding='utf-8')
    out = tmp_path / 'out'
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 1
    rows = _doc_csv(out / 'bang.csv')
    assert rows[0]['ket_qua'] == 'khong_khop'
    assert rows[0]['cach'] == 'thu_muc_json'


def test_c3_23_tuyet_doi_ton_tai_thang(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'data').mkdir(parents=True)
    (repo / 'data' / 'a.csv').write_bytes(b'dung\n')
    khac = tmp_path / 'khac' / 'repo' / 'data'
    khac.mkdir(parents=True)
    f_sai = khac / 'a.csv'
    f_sai.write_bytes(b'sai\n')
    kq, cach, thuc = KS.kiem_cap(str(f_sai), sha(b'dung\n'),
                                 tmp_path, repo)
    assert (kq, cach) == ('khong_khop', 'tuyet_doi')
    assert os.path.normpath(thuc) == os.path.normpath(str(f_sai))


def test_c3_23_ten_tran_khong_tim_goc_repo(tmp_path):
    repo = tmp_path / 'repo'
    jdir = repo / 'j'
    jdir.mkdir(parents=True)
    (repo / 'a.csv').write_bytes(b'dung\n')
    kq, cach, thuc = KS.kiem_cap('a.csv', sha(b'dung\n'), jdir, repo)
    assert (kq, cach, thuc) == ('thieu_file', '', '')


def test_c3_23_thu_muc_luot(tmp_path):
    repo = tmp_path / 'repo'
    run = repo / 'so_lieu' / '2026-11-20_run02'
    jdir = run / 'sub'
    jdir.mkdir(parents=True)
    f = run / 'F2' / 'a.jsonl'
    f.parent.mkdir(parents=True)
    f.write_bytes(b'x\n')
    # C3-23 vong 2: chi con ung vien thu_muc_luot
    out = KS.ung_vien('F2/a.jsonl', jdir, repo)
    assert [c for c, _ in out] == ['thu_muc_luot']
    kq, cach, thuc = KS.kiem_cap('F2/a.jsonl', sha(b'x\n'), jdir, repo)
    assert (kq, cach) == ('tho', 'thu_muc_luot')
    assert os.path.normpath(thuc) == os.path.normpath(str(f))


def test_c3_23_goc_repo_truoc_thu_muc_luot(tmp_path):
    # C3-23 vong 2: goc_repo khong con la ung vien nen dich thieu_file
    repo = tmp_path / 'repo'
    run = repo / 'so_lieu' / '2026-11-20_run02'
    jdir = run / 'sub'
    jdir.mkdir(parents=True)
    f = repo / 'F2' / 'a.jsonl'
    f.parent.mkdir(parents=True)
    f.write_bytes(b'x\n')
    kq, cach, thuc = KS.kiem_cap('F2/a.jsonl', sha(b'x\n'), jdir, repo)
    assert (kq, cach, thuc) == ('thieu_file', '', '')


def test_c3_23_mo_ho(tmp_path):
    # C3-23 vong 2: chi mot ung vien (thu_muc_luot) nen khong con mo ho
    repo = tmp_path / 'repo'
    run = repo / 'so_lieu' / '2026-11-20_run02'
    jdir = run / 'sub'
    jdir.mkdir(parents=True)
    (repo / 'F2').mkdir(parents=True)
    (repo / 'F2' / 'a.jsonl').write_bytes(b'x\n')
    (run / 'F2').mkdir(parents=True)
    (run / 'F2' / 'a.jsonl').write_bytes(b'x\n')
    kq, cach, thuc = KS.kiem_cap('F2/a.jsonl', H_A, jdir, repo)
    assert (kq, cach) == ('khong_khop', 'thu_muc_luot')


def test_c3_23_mo_ho_cli(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    run = goc / '2026-11-20_run02'
    jdir = run / 'sub'
    jdir.mkdir(parents=True)
    (repo / 'F2').mkdir(parents=True)
    (repo / 'F2' / 'a.jsonl').write_bytes(b'x\n')
    (run / 'F2').mkdir(parents=True)
    (run / 'F2' / 'a.jsonl').write_bytes(b'x\n')
    (jdir / 'k.json').write_text(json.dumps(
        {'sha256': {'F2/a.jsonl': H_A}}), encoding='utf-8')
    out = tmp_path / 'out'
    # C3-23 vong 2: thu_muc_luot la ung vien duy nhat, sha sai -> ma 1
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    kh = _doc_csv(out / 'khong_hieu.csv')
    assert not any(e['ly_do'] == 'mo_ho_phan_giai' for e in kh)
    assert ma == 1


def test_c3_23_thu_muc_luot_trung_thu_muc_json(tmp_path):
    repo = tmp_path / 'repo'
    run = repo / 'so_lieu' / '2026-11-20_run02'
    run.mkdir(parents=True)
    kq, cach, thuc = KS.kiem_cap('F2/a.jsonl', H_A, run, repo)
    assert (kq, cach, thuc) == ('thieu_file', '', '')


def test_c3_23_khong_co_thu_muc_luot(tmp_path):
    repo = tmp_path / 'repo'
    jdir = repo / 'j'
    jdir.mkdir(parents=True)
    # C3-23 vong 2: khong goc luot thi khong co ung vien nao
    out = KS.ung_vien('F2/a.jsonl', jdir, repo)
    assert out == []


# ---------- C3-24 ----------

def test_c3_24_cli_goc_la_sealed(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    sealed = goc / 'sealed'
    sealed.mkdir()
    (sealed / 'a.txt').write_bytes(b'ok')
    (sealed / 'r.json').write_text(json.dumps(
        {'file': 'a.txt', 'sha256': sha(b'ok')}), encoding='utf-8')
    out = tmp_path / 'out'
    ma = KS.main(['--goc', str(sealed), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 2
    assert not out.exists()
    st2 = goc / 'sealed_test' / 'sub'
    st2.mkdir(parents=True)
    (st2 / 'r.json').write_text('{}', encoding='utf-8')
    ma = KS.main(['--goc', str(st2), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 2
    assert not out.exists()


def test_c3_24_quet_goc_sealed_khong_mo_json(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    sealed = repo / 'sealed'
    sealed.mkdir(parents=True)
    (sealed / 'x.json').write_text('{ hong', encoding='utf-8')
    read_goc = Path.read_text

    def chan(self, *a, **kw):
        if any(p.lower() in ('sealed', 'sealed_test')
               for p in self.parts):
            raise AssertionError('doc file trong sealed')
        return read_goc(self, *a, **kw)

    monkeypatch.setattr(Path, 'read_text', chan)
    rows, kh, tt = KS.quet(sealed, repo)
    assert rows == []
    assert not any(e['ly_do'] == 'json_loi' for e in kh)
    assert any(e['ly_do'] == 'json_trong_sealed_khong_doc' for e in kh)


def test_c3_24_alias_thu_muc(tmp_path):
    if os.name != 'nt':
        pytest.skip('chi Windows')
    import _winapi
    repo, goc = _lam_repo(tmp_path)
    sealed = repo / 'sealed'
    sealed.mkdir()
    (sealed / 'x.json').write_text('{ hong', encoding='utf-8')
    try:
        _winapi.CreateJunction(str(sealed), str(goc / 'alias'))
    except OSError as e:
        pytest.skip(f'khong tao duoc junction: {e}')
    out = tmp_path / 'out'
    KS.main(['--goc', str(goc), '--goc-repo', str(repo),
             '--ra', str(out)])
    kh = _doc_csv(out / 'khong_hieu.csv')
    assert not any(e['ly_do'] == 'json_loi' for e in kh)
    assert not any(e['file_ghi'].startswith('alias') for e in kh)


def test_c3_24_alias_file(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    sealed = repo / 'sealed'
    sealed.mkdir()
    (sealed / 'x.json').write_text('{ hong', encoding='utf-8')
    try:
        os.symlink(str(sealed / 'x.json'), str(goc / 'y.json'))
    except OSError as e:
        pytest.skip(f'khong tao duoc symlink: {e}')
    out = tmp_path / 'out'
    KS.main(['--goc', str(goc), '--goc-repo', str(repo),
             '--ra', str(out)])
    kh = _doc_csv(out / 'khong_hieu.csv')
    e = [e for e in kh if e['file_ghi'] == 'y.json']
    assert len(e) == 1
    assert e[0]['ly_do'] == 'json_trong_sealed_khong_doc'
    assert not any(e['ly_do'] == 'json_loi' for e in kh)


# ---------- C3-22 vong 2 ----------

def test_c3_22_v2_ban_do_hash_sai():
    cap, kh, _ = KS.tim_cap({'anh': {'x.png': 'bad'}})
    assert not cap
    assert len(kh) == 1
    assert kh[0]['khoa'] == 'anh["x.png"]'
    assert kh[0]['ly_do'] == 'sha_khong_hop_le'


def test_c3_22_v2_cli_ban_do_hash_sai(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    out = tmp_path / 'out'
    (goc / 'x.png').write_bytes(b'png')
    (goc / 'r.json').write_text('{"anh": {"x.png": "bad"}}',
                               encoding='utf-8')
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 3
    kh = _doc_csv(out / 'khong_hieu.csv')
    e = [e for e in kh if e['khoa'] == 'anh["x.png"]']
    assert len(e) == 1 and e[0]['ly_do'] == 'sha_khong_hop_le'
    tt = json.loads((out / 'tom_tat.json').read_text(encoding='utf-8'))
    assert tt['so_chua_xac_minh'] == 1
    lines = (out / 'bang.csv').read_text(encoding='utf-8').splitlines()
    assert lines[-1] == '#ket_qua,ma=3'


def test_c3_22_v2_cli_manifest_null(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    out = tmp_path / 'out'
    (goc / 'a.csv').write_bytes(b'x\n')
    (goc / 'r.json').write_text('{"sha256": {"a.csv": null}}',
                               encoding='utf-8')
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 3
    kh = _doc_csv(out / 'khong_hieu.csv')
    e = [e for e in kh if e['khoa'] == 'sha256["a.csv"]']
    assert len(e) == 1 and e[0]['ly_do'] == 'sha_rong_bat_buoc'
    tt = json.loads((out / 'tom_tat.json').read_text(encoding='utf-8'))
    assert tt['so_rong'] == 0


def test_c3_22_v2_cli_sha256_null_co_duong_dan(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    out = tmp_path / 'out'
    (goc / 'b.txt').write_bytes(b'bb')
    (goc / 'r.json').write_text('{"file": "b.txt", "sha256": null}',
                               encoding='utf-8')
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 3
    kh = _doc_csv(out / 'khong_hieu.csv')
    e = [e for e in kh if e['khoa'] == 'sha256']
    assert len(e) == 1 and e[0]['ly_do'] == 'sha_rong_bat_buoc'
    cap, kh2, _ = KS.tim_cap({'video': 'v.mp4', 'video_sha256': None})
    assert not cap
    assert len(kh2) == 1
    assert kh2[0]['khoa'] == 'video_sha256'
    assert kh2[0]['ly_do'] == 'sha_rong_bat_buoc'


def test_c3_22_v2_cli_null_khong_duong_dan_ma_0(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    out = tmp_path / 'out'
    (goc / 'b.txt').write_bytes(b'bb')
    (goc / 'r.json').write_text(json.dumps(
        {'file': 'b.txt', 'sha256': sha(b'bb'), 'a_sha256': None}),
        encoding='utf-8')
    (goc / 'r2.json').write_text(json.dumps({'sha256': H_A}),
                                encoding='utf-8')
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 0
    tt = json.loads((out / 'tom_tat.json').read_text(encoding='utf-8'))
    assert tt['so_rong'] == 1
    kh = _doc_csv(out / 'khong_hieu.csv')
    assert any(e['ly_do'] == 'khong_co_duong_dan' for e in kh)


def test_c3_22_v2_ban_do_hash_hon_hop():
    cap, kh, _ = KS.tim_cap({'anh': {'x.png': None, 'y.png': 'bad'}})
    assert not cap
    ly = {e['khoa']: e['ly_do'] for e in kh}
    assert ly == {'anh["x.png"]': 'sha_rong_bat_buoc',
                  'anh["y.png"]': 'sha_khong_hop_le'}


def test_c3_22_v2_ban_do_hash_bien_do_dai():
    cap, kh, _ = KS.tim_cap(
        {'m': {'a.csv': 'a' * 63, 'b.csv': 'b' * 65, 'c.csv': 'C' * 64}})
    assert len(cap) == 1 and cap[0]['duong_dan'] == 'c.csv'
    ly = {e['khoa']: e['ly_do'] for e in kh}
    assert ly == {'m["a.csv"]': 'sha_khong_hop_le',
                  'm["b.csv"]': 'sha_khong_hop_le'}


def test_c3_22_v2_canh_ban_do_duong_dan():
    cap, kh, rong = KS.tim_cap({'doi_ten': {'a.csv': 'b/c.csv'}})
    assert not cap and not kh and rong == 0
    # ban do toan null khong phai ban do hash: khong kh, khong rong
    cap, kh, rong = KS.tim_cap({'outputs': {'a.csv': None}})
    assert not cap and not kh and rong == 0


def test_c3_22_v2_cli_ban_do_duong_dan_ma_0(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    out = tmp_path / 'out'
    (goc / 'b.txt').write_bytes(b'bb')
    (goc / 'r.json').write_text(json.dumps(
        {'doi_ten': {'a.csv': 'b/c.csv'},
         'file': 'b.txt', 'sha256': sha(b'bb')}), encoding='utf-8')
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 0


def test_c3_22_v2_docstring_ghi_ma_0():
    assert 'ma 0 khong phu hash khong kem duong dan' in KS.__doc__.lower()


# ---------- C3-23 vong 2 ----------

def test_c3_23_v2_jsonl_ben_canh_json(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    run = goc / '2026-11-20_run02'
    jdir = run / 'sub'
    (jdir / 'F2').mkdir(parents=True)
    (jdir / 'F2' / 'a.jsonl').write_bytes(b'cu\n')
    (run / 'F2').mkdir(parents=True)
    thuc_tep = run / 'F2' / 'a.jsonl'
    thuc_tep.write_bytes(b'moi\n')
    (jdir / 'k.json').write_text(json.dumps(
        {'sha256': {'F2/a.jsonl': sha(b'cu\n')}}), encoding='utf-8')
    out = tmp_path / 'out'
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 1
    rows = _doc_csv(out / 'bang.csv')
    assert rows[0]['ket_qua'] == 'khong_khop'
    assert rows[0]['cach'] == 'thu_muc_luot'
    assert os.path.normpath(rows[0]['duong_dan_thuc']) == os.path.normpath(
        str(thuc_tep))


def test_c3_23_v2_duong_dan_cham(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    run = goc / '2026-11-20_run02'
    jdir = run / 'sub'
    jdir.mkdir(parents=True)
    (jdir / 'a.csv').write_bytes(b'cu\n')
    (run / 'a.csv').write_bytes(b'moi\n')
    (jdir / 'k.json').write_text(json.dumps(
        {'file': './a.csv', 'sha256': sha(b'cu\n')}), encoding='utf-8')
    out = tmp_path / 'out'
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 1
    rows = _doc_csv(out / 'bang.csv')
    assert rows[0]['ket_qua'] == 'khong_khop'
    assert rows[0]['cach'] == 'thu_muc_luot'


def test_c3_23_v2_khong_thu_goc_repo(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    run = goc / '2026-11-20_run02'
    jdir = run / 'sub'
    jdir.mkdir(parents=True)
    (repo / 'F2').mkdir(parents=True)
    (repo / 'F2' / 'a.jsonl').write_bytes(b'x\n')
    (jdir / 'k.json').write_text(json.dumps(
        {'sha256': {'F2/a.jsonl': sha(b'x\n')}}), encoding='utf-8')
    out = tmp_path / 'out'
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 1
    rows = _doc_csv(out / 'bang.csv')
    assert rows[0]['ket_qua'] == 'thieu_file'
    assert rows[0]['duong_dan_thuc'] == ''


def test_c3_23_v2_goc_luot_tham_so(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    xd = goc / 'x'
    xd.mkdir()
    luot = tmp_path / '2026-11-20_run09'
    (luot / 'F2').mkdir(parents=True)
    (luot / 'F2' / 'a.jsonl').write_bytes(b'x\n')
    (xd / 'k.json').write_text(json.dumps(
        {'sha256': {'F2/a.jsonl': sha(b'x\n')}}), encoding='utf-8')
    out = tmp_path / 'out'
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--goc-luot', str(luot), '--ra', str(out)])
    assert ma == 0
    rows = _doc_csv(out / 'bang.csv')
    assert rows[0]['ket_qua'] == 'tho'
    assert rows[0]['cach'] == 'goc_luot'
    (xd / 'F2').mkdir()
    (xd / 'F2' / 'a.jsonl').write_bytes(b'cu\n')
    (luot / 'F2' / 'a.jsonl').write_bytes(b'moi\n')
    out2 = tmp_path / 'out2'
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--goc-luot', str(luot), '--ra', str(out2)])
    assert ma == 1
    rows = _doc_csv(out2 / 'bang.csv')
    assert rows[0]['ket_qua'] == 'khong_khop'
    assert rows[0]['cach'] == 'goc_luot'


def test_c3_23_v2_goc_luot_loi(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    out = tmp_path / 'out'
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--goc-luot', str(tmp_path / 'khong_co'),
                  '--ra', str(out)])
    assert ma == 2
    assert not out.exists()
    sealed = goc / 'sealed'
    sealed.mkdir()
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--goc-luot', str(sealed), '--ra', str(out)])
    assert ma == 2
    assert not out.exists()


def test_c3_23_v2_cli_khong_co_goc_luot(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    xd = goc / 'x'
    xd.mkdir()
    (xd / 'k.json').write_text(json.dumps(
        {'sha256': {'F2/a.jsonl': H_A}}), encoding='utf-8')
    out = tmp_path / 'out'
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 3
    kh = _doc_csv(out / 'khong_hieu.csv')
    e = [e for e in kh if e['ly_do'] == 'khong_co_goc_luot']
    assert len(e) == 1
    assert e[0]['khoa'] == 'sha256["F2/a.jsonl"]'
    assert e[0]['gia_tri'] == 'F2/a.jsonl'


def test_c3_23_v2_ten_tran_bo_qua_goc_luot(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    luot = tmp_path / 'luot'
    luot.mkdir()
    (luot / 'a.csv').write_bytes(b'x\n')
    (goc / 'k.json').write_text(json.dumps(
        {'file': 'a.csv', 'sha256': sha(b'x\n')}), encoding='utf-8')
    out = tmp_path / 'out'
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--goc-luot', str(luot), '--ra', str(out)])
    assert ma == 1
    rows = _doc_csv(out / 'bang.csv')
    assert rows[0]['ket_qua'] == 'thieu_file'


def test_c3_23_v2_gach_cheo_nguoc(tmp_path):
    repo = tmp_path / 'repo'
    run = repo / 'so_lieu' / '2026-11-20_run02'
    jdir = run / 'sub'
    jdir.mkdir(parents=True)
    f = run / 'F2' / 'a.jsonl'
    f.parent.mkdir(parents=True)
    f.write_bytes(b'x\n')
    kq, cach, thuc = KS.kiem_cap('F2\\a.jsonl', sha(b'x\n'), jdir, repo)
    assert (kq, cach) == ('tho', 'thu_muc_luot')
    assert os.path.normpath(thuc) == os.path.normpath(str(f))


def test_c3_23_v2_luot_long_nhau(tmp_path):
    repo = tmp_path / 'repo'
    run2 = repo / 'so_lieu' / '2026-11-20_run02'
    run3 = run2 / 'x' / '2026-11-21_run03'
    jdir = run3 / 'sub'
    jdir.mkdir(parents=True)
    f = run3 / 'F2' / 'a.jsonl'
    f.parent.mkdir(parents=True)
    f.write_bytes(b'x\n')
    (run2 / 'F2').mkdir(parents=True)
    (run2 / 'F2' / 'a.jsonl').write_bytes(b'khac\n')
    kq, cach, thuc = KS.kiem_cap('F2/a.jsonl', sha(b'x\n'), jdir, repo)
    assert (kq, cach) == ('tho', 'thu_muc_luot')
    assert os.path.normpath(thuc) == os.path.normpath(str(f))


def test_c3_23_v2_sealed_khong_can_goc(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    jdir = repo / 'j'
    jdir.mkdir(parents=True)

    def cam(p, s):
        raise AssertionError('khong duoc doc sealed')

    monkeypatch.setattr(KS, 'kiem', cam)
    kq, cach, thuc = KS.kiem_cap('sealed/t.txt', H_A, jdir, repo)
    assert (kq, cach, thuc) == ('bo_qua_sealed', '', '')


# ---------- C3-30 ----------

def _hard_link(src, dst):
    try:
        os.link(src, dst)
    except OSError as e:
        pytest.skip(f'khong tao duoc hard link: {e}')


def test_c3_30_json_hard_link(tmp_path, monkeypatch):
    repo, goc = _lam_repo(tmp_path)
    sealed = repo / 'sealed'
    sealed.mkdir()
    (sealed / 'x.json').write_text('{ hong', encoding='utf-8')
    _hard_link(sealed / 'x.json', goc / 'y.json')
    read_goc = Path.read_text

    def chan(self, *a, **kw):
        if os.stat(self).st_nlink > 1:
            raise AssertionError('doc file hard link')
        return read_goc(self, *a, **kw)

    monkeypatch.setattr(Path, 'read_text', chan)
    out = tmp_path / 'out'
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 3
    kh = _doc_csv(out / 'khong_hieu.csv')
    e = [e for e in kh if e['file_ghi'] == 'y.json']
    assert len(e) == 1 and e[0]['ly_do'] == 'hard_link'
    assert not any(e['ly_do'] == 'json_loi' for e in kh)


def test_c3_30_file_hard_link(tmp_path, monkeypatch):
    repo, goc = _lam_repo(tmp_path)
    run = goc / '2026-11-20_run02'
    run.mkdir()
    data = b'x\n'
    f = run / 'a.txt'
    f.write_bytes(data)
    _hard_link(f, tmp_path / 'khac.txt')
    (run / 'k.json').write_text(json.dumps(
        {'file': 'a.txt', 'sha256': sha(data)}), encoding='utf-8')

    def cam(p):
        raise AssertionError('khong duoc bam file hard link')

    monkeypatch.setattr(KS, 'sha_cac_dang', cam)
    out = tmp_path / 'out'
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 3
    kh = _doc_csv(out / 'khong_hieu.csv')
    e = [e for e in kh if e['ly_do'] == 'hard_link']
    assert len(e) == 1 and e[0]['khoa'] == 'sha256'
    rows = _doc_csv(out / 'bang.csv')
    assert not rows


def test_c3_30_file_hard_link_co_thu_muc(tmp_path, monkeypatch):
    repo, goc = _lam_repo(tmp_path)
    run = goc / '2026-11-20_run02'
    (run / 'F2').mkdir(parents=True)
    data = b'x\n'
    f = run / 'F2' / 'a.jsonl'
    f.write_bytes(data)
    _hard_link(f, tmp_path / 'khac2.jsonl')
    (run / 'k.json').write_text(json.dumps(
        {'sha256': {'F2/a.jsonl': sha(data)}}), encoding='utf-8')

    def cam(p):
        raise AssertionError('khong duoc bam file hard link')

    monkeypatch.setattr(KS, 'sha_cac_dang', cam)
    out = tmp_path / 'out'
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 3
    kh = _doc_csv(out / 'khong_hieu.csv')
    e = [e for e in kh if e['ly_do'] == 'hard_link']
    assert len(e) == 1 and e[0]['khoa'] == 'sha256["F2/a.jsonl"]'


def test_c3_30_bo_hard_link_thi_ma_0(tmp_path):
    repo, goc = _lam_repo(tmp_path)
    run = goc / '2026-11-20_run02'
    run.mkdir()
    data = b'x\n'
    f = run / 'a.txt'
    f.write_bytes(data)
    khac = tmp_path / 'khac.txt'
    _hard_link(f, khac)
    (run / 'k.json').write_text(json.dumps(
        {'file': 'a.txt', 'sha256': sha(data)}), encoding='utf-8')
    os.unlink(khac)
    out = tmp_path / 'out'
    ma = KS.main(['--goc', str(goc), '--goc-repo', str(repo),
                  '--ra', str(out)])
    assert ma == 0


def test_c3_30_kiem_hard_link(tmp_path):
    f = tmp_path / 'a.txt'
    f.write_bytes(b'x\n')
    _hard_link(f, tmp_path / 'b.txt')
    assert KS.kiem(f, sha(b'x\n')) == 'hard_link'
