"""Kiem sha256 ghi trong JSON, chiu sai khac CRLF/LF.

Repo du lieu dung autocrlf: sha ghi theo ban tren dia, checkout khac co the
doi dau xuong dong. Moi file co toi da 3 dang sha: tho (nguyen byte), lf, crlf.
File nhi phan (theo duoi mo rong hoac chua byte NUL) chi co 'tho'.

Ma thoat: 0 dat, 1 khong_khop/thieu_file, 2 loi tham so, 3 chua xac minh.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

BIN_EXT = frozenset({
    '.npz', '.npy', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff',
    '.mov', '.mp4', '.avi', '.mkv', '.wav', '.mp3', '.pdf', '.zip', '.gz',
    '.7z', '.tar', '.bin', '.elf', '.pkl', '.pickle', '.h5', '.hdf5', '.mat',
    '.xlsx', '.xls', '.docx', '.pptx', '.dll', '.so', '.exe', '.parquet',
    '.feather',
})

_HEX64 = re.compile(r'[0-9a-fA-F]{64}')
_DUOI_SHA = re.compile(r'(.+)_sha256$', re.IGNORECASE)
_DAU_SHA = re.compile(r'sha256_(.+)$', re.IGNORECASE)
_TUYET_DOI = re.compile(r'^[A-Za-z]:[\\/]')
_SEALED = frozenset({'sealed', 'sealed_test'})
_TEN_LUOT = re.compile(r'\d{4}-\d{2}-\d{2}_run\d+')
_CHUA_XAC_MINH = frozenset({
    'json_loi', 'sha_khong_hop_le', 'manifest_khoa_khong_phai_file',
    'danh_sach_khong_ghep', 'nhieu_duong_dan', 'kieu_khong_ho_tro',
    'mo_ho_phan_giai',
})
_BO_NHO = {}


def _hex64(v):
    return isinstance(v, str) and _HEX64.fullmatch(v) is not None


def la_duong_dan(s):
    if not isinstance(s, str) or s == '' or '\n' in s or '\r' in s:
        return False
    return ('/' in s or '\\' in s
            or re.search(r'\.[A-Za-z0-9]{1,6}$', s) is not None)


def sha_cac_dang(path):
    p = Path(path).resolve()
    st = p.stat()
    chia_khoa = (str(p), st.st_size, st.st_mtime_ns)
    cu = _BO_NHO.get(chia_khoa)
    if cu is not None:
        return dict(cu)
    if p.suffix.lower() in BIN_EXT:
        bam = hashlib.sha256()
        with open(p, 'rb') as f:
            for mieng in iter(lambda: f.read(1 << 20), b''):
                bam.update(mieng)
        out = {'tho': bam.hexdigest()}
    else:
        data = p.read_bytes()
        if b'\x00' in data:
            out = {'tho': hashlib.sha256(data).hexdigest()}
        else:
            lf = data.replace(b'\r\n', b'\n')
            crlf = lf.replace(b'\n', b'\r\n')
            out = {
                'tho': hashlib.sha256(data).hexdigest(),
                'lf': hashlib.sha256(lf).hexdigest(),
                'crlf': hashlib.sha256(crlf).hexdigest(),
            }
    _BO_NHO[chia_khoa] = out
    return dict(out)


def kiem(path, sha_ghi):
    p = Path(path)
    if not p.is_file():
        return 'thieu_file'
    dang = sha_cac_dang(p)
    s = sha_ghi.strip().lower()
    for k in ('tho', 'lf', 'crlf'):
        if dang.get(k) == s:
            return k
    return 'khong_khop'


def _con(path, k):
    return f'{path}.{k}' if path else str(k)


def _chi_so(path, i):
    return f'{path}[{i}]'


def _manifest_con(path, k):
    return f'{path}["{k}"]'


def _la_manifest(d):
    if not d:
        return False
    return all(la_duong_dan(k) and _hex64(v) for k, v in d.items())


def _chon_duong_dan(cands, khoa, sha, cap, kh):
    # cands: [(gia_tri, dang)] da loc la_duong_dan
    rieng = []
    for gia_tri, dang in cands:
        if gia_tri not in [g for g, _ in rieng]:
            rieng.append((gia_tri, dang))
    if len(rieng) == 1:
        cap.append({'khoa': khoa, 'duong_dan': rieng[0][0],
                    'sha': sha, 'dang': rieng[0][1]})
    elif not rieng:
        kh.append({'khoa': khoa, 'gia_tri': sha[:200],
                   'ly_do': 'khong_co_duong_dan'})
    else:
        kh.append({'khoa': khoa, 'gia_tri': sha[:200],
                   'ly_do': 'nhieu_duong_dan'})


def _xu_sha(k, v, D, path, cap, kh, st):
    khoa = _con(path, k)
    if v is None:
        st[0] += 1
        return
    if isinstance(v, str):
        if not _hex64(v):
            kh.append({'khoa': khoa, 'gia_tri': v[:200],
                       'ly_do': 'sha_khong_hop_le'})
            return
        cands = []
        if k.lower() == 'sha256':
            thuong = {kk.lower() for kk in D if isinstance(kk, str)}
            if any(kk in thuong for kk in
                   ('sha256_file', 'sha256_duong_dan', 'sha256_path')):
                kh.append({'khoa': khoa, 'gia_tri': v[:200],
                           'ly_do': 'co_sha256_file_rieng'})
                return
            for kk, dang in (('duong_dan', 'khoi_duong_dan'),
                             ('path', 'khoi_path'),
                             ('file', 'khoi_file')):
                if kk in D and la_duong_dan(D[kk]):
                    cands.append((D[kk], dang))
        else:
            m = _DUOI_SHA.match(k)
            if m:
                b = m.group(1)
                thuong = {kk.lower() for kk in D if isinstance(kk, str)}
                if b in D and la_duong_dan(D[b]):
                    cands.append((D[b], 'ten_sha256'))
                bf = b + '_file'
                if (bf in D and (b + '_file_sha256').lower() not in thuong
                        and la_duong_dan(D[bf])):
                    cands.append((D[bf], 'ten_file'))
                bp = b + '_path'
                if (bp in D and (b + '_path_sha256').lower() not in thuong
                        and la_duong_dan(D[bp])):
                    cands.append((D[bp], 'ten_path'))
            else:
                m2 = _DAU_SHA.match(k)
                if m2:
                    x = m2.group(1)
                    if (x.lower() in ('file', 'duong_dan', 'path')
                            and x in D and la_duong_dan(D[x])):
                        cands.append((D[x], 'sha256_hau_to'))
                    for kk in ('file_' + x, 'duong_dan_' + x, 'path_' + x):
                        if kk in D and la_duong_dan(D[kk]):
                            cands.append((D[kk], 'sha256_hau_to'))
        _chon_duong_dan(cands, khoa, v, cap, kh)
        return
    if isinstance(v, list):
        m = _DUOI_SHA.match(k)
        ok = False
        if m and m.group(1) in D:
            base = D[m.group(1)]
            if (isinstance(base, list) and len(base) == len(v)
                    and all(la_duong_dan(x) for x in base)
                    and all(_hex64(x) for x in v)):
                for i, (dd, s) in enumerate(zip(base, v)):
                    cap.append({'khoa': _chi_so(khoa, i), 'duong_dan': dd,
                                'sha': s, 'dang': 'danh_sach'})
                ok = True
        if not ok:
            kh.append({'khoa': khoa, 'gia_tri': str(v)[:200],
                       'ly_do': 'danh_sach_khong_ghep'})
        return
    if isinstance(v, dict):
        for k2, v2 in v.items():
            sub = _manifest_con(khoa, k2)
            if _hex64(v2):
                if la_duong_dan(k2):
                    cap.append({'khoa': sub, 'duong_dan': k2,
                                'sha': v2, 'dang': 'manifest'})
                else:
                    kh.append({'khoa': sub, 'gia_tri': str(k2)[:200],
                               'ly_do': 'manifest_khoa_khong_phai_file'})
            elif v2 is None:
                st[0] += 1
            elif isinstance(v2, str):
                kh.append({'khoa': sub, 'gia_tri': v2[:200],
                           'ly_do': 'sha_khong_hop_le'})
            elif isinstance(v2, (dict, list)):
                _duyet(v2, sub, cap, kh, st)
            else:
                kh.append({'khoa': sub, 'gia_tri': str(v2)[:200],
                           'ly_do': 'kieu_khong_ho_tro'})
        return
    kh.append({'khoa': khoa, 'gia_tri': str(v)[:200],
               'ly_do': 'kieu_khong_ho_tro'})


def _duyet(v, path, cap, kh, st):
    if isinstance(v, dict):
        if _la_manifest(v):
            for k2, v2 in v.items():
                cap.append({'khoa': _manifest_con(path, k2),
                            'duong_dan': k2, 'sha': v2, 'dang': 'manifest'})
            return
        if any(isinstance(k, str) and 'sha256' not in k.lower()
               and la_duong_dan(k) and _hex64(v2)
               for k, v2 in v.items()):
            for k, v2 in v.items():
                if (isinstance(k, str) and 'sha256' not in k.lower()
                        and la_duong_dan(k) and _hex64(v2)):
                    cap.append({'khoa': _manifest_con(path, k),
                                'duong_dan': k, 'sha': v2,
                                'dang': 'manifest'})
                elif (isinstance(k, str) and 'sha256' not in k.lower()
                      and la_duong_dan(k) and isinstance(v2, str)):
                    kh.append({'khoa': _manifest_con(path, k),
                               'gia_tri': v2[:200],
                               'ly_do': 'sha_khong_hop_le'})
                elif isinstance(k, str) and 'sha256' in k.lower():
                    _xu_sha(k, v2, v, path, cap, kh, st)
                else:
                    _duyet(v2, _con(path, k), cap, kh, st)
            return
        for k, v2 in v.items():
            if isinstance(k, str) and 'sha256' in k.lower():
                _xu_sha(k, v2, v, path, cap, kh, st)
            else:
                _duyet(v2, _con(path, k), cap, kh, st)
    elif isinstance(v, list):
        for i, item in enumerate(v):
            _duyet(item, _chi_so(path, i), cap, kh, st)


def tim_cap(obj):
    cap = []
    kh = []
    st = [0]
    _duyet(obj, '', cap, kh, st)
    return cap, kh, st[0]


def ung_vien(duong_dan, thu_muc_json, goc_repo):
    s = str(duong_dan)
    out = []
    if _TUYET_DOI.match(s) or s.startswith('/') or s.startswith('\\'):
        out.append(('tuyet_doi', Path(s)))
        if goc_repo is not None:
            ten = Path(goc_repo).name.lower()
            tach = [x for x in re.split(r'[\\/]', s) if x]
            cuoi = -1
            for i, x in enumerate(tach):
                if x.lower() == ten:
                    cuoi = i
            if 0 <= cuoi < len(tach) - 1:
                out.append(('tuyet_doi_doi_goc',
                            Path(goc_repo, *tach[cuoi + 1:])))
    else:
        rel = s.replace('\\', '/')
        out.append(('thu_muc_json', Path(thu_muc_json) / rel))
        if '/' not in rel:
            return out
        if goc_repo is not None:
            out.append(('goc_repo', Path(goc_repo) / rel))
        tmj = Path(thu_muc_json)
        for d in (tmj, *tmj.parents):
            if _TEN_LUOT.fullmatch(d.name):
                out.append(('thu_muc_luot', d / rel))
                break
    return out


def _la_sealed(p):
    try:
        parts = Path(p).resolve().parts
    except Exception:
        parts = Path(p).parts
    return any(x.lower() in _SEALED for x in parts)


def kiem_cap(duong_dan, sha, thu_muc_json, goc_repo):
    ds = ung_vien(duong_dan, thu_muc_json, goc_repo)
    if not ds:
        return 'thieu_file', '', ''
    c0, p0 = ds[0]
    if _la_sealed(p0):
        return 'bo_qua_sealed', '', ''
    if p0.is_file():
        return kiem(p0, sha), c0, str(p0)
    giu = []
    da_co = {p0.resolve()}
    for cach, p in ds[1:]:
        if _la_sealed(p):
            continue
        pr = p.resolve()
        if pr in da_co:
            continue
        if p.is_file():
            giu.append((cach, p))
            da_co.add(pr)
    if len(giu) == 1:
        cach, p = giu[0]
        return kiem(p, sha), cach, str(p)
    if len(giu) >= 2:
        return ('mo_ho_phan_giai', ';'.join(c for c, _ in giu),
                ';'.join(str(p) for _, p in giu))
    return 'thieu_file', '', ''


def tim_goc_repo(start):
    p = Path(start).resolve()
    for d in (p, *p.parents):
        if (d / '.git').exists():
            return d
    return None


def quet(goc, goc_repo):
    goc = Path(goc)
    ds_json = []
    for dirpath, dirnames, filenames in os.walk(goc):
        dirnames[:] = [d for d in dirnames
                       if d.lower() not in _SEALED
                       and not _la_sealed(Path(dirpath) / d)]
        for fn in filenames:
            if fn.lower().endswith('.json'):
                ds_json.append(Path(dirpath) / fn)
    ds_json.sort()
    rows = []
    kh_rows = []
    tong_rong = 0
    for pj in ds_json:
        file_ghi = pj.relative_to(goc).as_posix()
        if _la_sealed(pj):
            kh_rows.append({'file_ghi': file_ghi, 'khoa': '',
                            'gia_tri': '',
                            'ly_do': 'json_trong_sealed_khong_doc'})
            continue
        try:
            obj = json.loads(pj.read_text(encoding='utf-8-sig'))
        except Exception:
            kh_rows.append({'file_ghi': file_ghi, 'khoa': '',
                            'gia_tri': '', 'ly_do': 'json_loi'})
            continue
        cap, kh, rong = tim_cap(obj)
        tong_rong += rong
        for e in kh:
            kh_rows.append({'file_ghi': file_ghi, 'khoa': e['khoa'],
                            'gia_tri': e['gia_tri'], 'ly_do': e['ly_do']})
        for c in cap:
            kq, cach, thuc = kiem_cap(c['duong_dan'], c['sha'],
                                      pj.parent, goc_repo)
            if kq == 'bo_qua_sealed':
                kh_rows.append({'file_ghi': file_ghi, 'khoa': c['khoa'],
                                'gia_tri': c['duong_dan'],
                                'ly_do': 'tro_vao_sealed_khong_doc'})
                continue
            if kq == 'mo_ho_phan_giai':
                kh_rows.append({'file_ghi': file_ghi, 'khoa': c['khoa'],
                                'gia_tri': c['duong_dan'],
                                'ly_do': 'mo_ho_phan_giai'})
                continue
            dd = str(c['duong_dan'])
            ghi_chu = ('ten_tran' if ('/' not in dd and '\\' not in dd)
                       else '')
            rows.append({'file_ghi': file_ghi, 'khoa': c['khoa'],
                         'duong_dan': c['duong_dan'], 'sha_ghi': c['sha'],
                         'ket_qua': kq, 'cach': cach,
                         'duong_dan_thuc': thuc, 'dang': c['dang'],
                         'ghi_chu': ghi_chu})
    tom_tat = {'so_json': len(ds_json), 'so_khong_hieu': len(kh_rows),
               'so_rong': tong_rong,
               'so_chua_xac_minh': sum(1 for e in kh_rows
                                       if e['ly_do'] in _CHUA_XAC_MINH)}
    for kq in ('tho', 'lf', 'crlf', 'khong_khop', 'thieu_file'):
        tom_tat[kq] = sum(1 for r in rows if r['ket_qua'] == kq)
    return rows, kh_rows, tom_tat


def _trong(child, parent):
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _in(s):
    print(str(s).encode('ascii', 'backslashreplace').decode())


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Kiem sha256 trong JSON, chiu sai khac CRLF/LF.')
    ap.add_argument('--goc', required=True)
    ap.add_argument('--goc-repo', default=None)
    ap.add_argument('--ra', default=None)
    a = ap.parse_args(argv)
    goc = Path(a.goc)
    if not goc.is_dir():
        _in('loi: --goc khong phai thu muc')
        return 2
    goc = goc.resolve()
    if _la_sealed(goc):
        _in('loi: --goc nam trong sealed/sealed_test')
        return 2
    goc_repo = (Path(a.goc_repo).resolve() if a.goc_repo
                else tim_goc_repo(goc))
    if a.ra:
        ra = Path(a.ra).resolve()
    else:
        ra = Path(__file__).resolve().parents[1] / 'ket_qua_sha'
    if _trong(ra, goc) or (goc_repo is not None and _trong(ra, goc_repo)):
        _in('loi: --ra nam trong --goc hoac goc_repo')
        return 2
    rows, kh_rows, tom_tat = quet(goc, goc_repo)
    if any(r['ket_qua'] in ('khong_khop', 'thieu_file') for r in rows):
        ma = 1
    elif tom_tat['so_chua_xac_minh'] > 0:
        ma = 3
    else:
        ma = 0
    ra.mkdir(parents=True, exist_ok=True)
    cot = ['file_ghi', 'khoa', 'duong_dan', 'sha_ghi', 'ket_qua', 'cach',
           'duong_dan_thuc', 'dang', 'ghi_chu']
    with open(ra / 'bang.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cot)
        for r in rows:
            w.writerow([r[c] for c in cot])
        f.write(f'#ket_qua,ma={ma}\n')
    with open(ra / 'khong_hieu.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['file_ghi', 'khoa', 'gia_tri', 'ly_do'])
        for e in kh_rows:
            w.writerow([e['file_ghi'], e['khoa'], e['gia_tri'], e['ly_do']])
    tt = dict(tom_tat)
    tt['ma'] = ma
    tt['goc'] = str(goc)
    tt['goc_repo'] = str(goc_repo) if goc_repo is not None else None
    with open(ra / 'tom_tat.json', 'w', encoding='utf-8') as f:
        json.dump(tt, f, ensure_ascii=True, indent=2)
    for kq in ('tho', 'lf', 'crlf', 'khong_khop', 'thieu_file'):
        _in(f'{kq}: {tom_tat[kq]}')
    _in(f"so_khong_hieu: {tom_tat['so_khong_hieu']}")
    _in(f"so_rong: {tom_tat['so_rong']}")
    _in(f"chua_xac_minh: {tom_tat['so_chua_xac_minh']}")
    for r in rows:
        if r['ket_qua'] in ('khong_khop', 'thieu_file'):
            _in(f"{r['file_ghi']} | {r['khoa']} | {r['duong_dan']}")
    for e in kh_rows:
        if e['ly_do'] in _CHUA_XAC_MINH:
            _in(f"{e['file_ghi']} | {e['khoa']} | {e['ly_do']}")
    _in(f'ma={ma}')
    return ma


if __name__ == '__main__':
    sys.exit(main())
