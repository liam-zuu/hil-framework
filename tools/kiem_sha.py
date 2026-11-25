"""Kiem sha256 ghi trong JSON, chiu sai khac CRLF/LF.

Repo du lieu dung autocrlf: sha ghi theo ban tren dia, checkout khac co the
doi dau xuong dong. Moi file co toi da 3 dang sha: tho (nguyen byte), lf, crlf.
File nhi phan (theo duoi mo rong hoac chua byte NUL) chi co 'tho'.

Ma thoat: 0 dat, 1 khong_khop/thieu_file, 2 loi tham so, 3 chua xac minh.
Ma 0 khong phu hash khong kem duong dan (khong_co_duong_dan chi la thong
tin).

Truong bat buoc = hash gan voi duong dan (khoa manifest/ban do la duong dan,
hoac truong duong dan cung khoi); null o truong bat buoc ->
sha_rong_bat_buoc (ma 3). Null o hash khong kem duong dan -> so_rong
(thong tin).

Ban do outputs la bang SHA theo schema (C3-22). Ban do hash suy tu khoa:
dict co it nhat mot khoa duong dan
mang gia tri hex64 hay null la ban do; nhan dien truoc, roi moi muc khoa
duong dan duoc xet rieng: hex64 -> kiem; null -> sha_rong_bat_buoc; so,
bool, object, list, chuoi khong phai hex64 -> sha_khong_hop_le (ma 3).
Trong ban do khai bao duoi khoa sha256, khoa duong dan mang object/list
-> kieu_khong_ho_tro (ma 3), khong duyet tiep.

Doc file (C3-34): JSON va file tham chieu mo truoc, st_nlink kiem bang
os.fstat tren descriptor da mo roi doc tu chinh descriptor do; loi mo/doc
(file bien mat giua luc liet ke va luc kiem) ghi loi, khong vang: JSON ->
khong_doc_duoc (ma 3); file tham chieu -> thieu_file (ma 1) neu khong con,
khong_doc_duoc (ma 3) voi loi OSError khac.

Phan giai duong tuong doi: ten tran -> chi thu muc chua JSON; duong co thu
muc (ke ca './a.csv', 'F2\\a.jsonl') -> chi goc luot: --goc-luot neu khai
bao, khong thi thu muc to tien gan nhat cua thu muc JSON (ke ca chinh no)
co ten khop YYYY-MM-DD_runNN; khong co goc luot -> khong_co_goc_luot
(ma 3); dich khong ton tai -> thieu_file, khong thu goc repo. Ngoai le
duy nhat (CODE 23/11): duong bat dau bang '05_DATA/' la duong tinh tu goc
repo, chi mot dich goc_repo/<duong> (khong co goc repo -> khong_co_goc_luot,
ma 3); thu muc luot khong bao gio chua thu muc con 05_DATA. Gioi han cua
luat '05_DATA/': duong luon tinh tu goc repo; manifest nao dung
'05_DATA/x' de chi '<run>/05_DATA/x' se bi doc khac (tro goc_repo/05_DATA/x).
Quet 23/11: khong thu muc luot nao co thu muc con 05_DATA. Duong tuyet
doi giu luat cu. File co st_nlink > 1 (JSON hay file tham chieu) khong doc:
hard_link (ma 3).
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
    'mo_ho_phan_giai', 'sha_rong_bat_buoc', 'khong_co_goc_luot',
    'hard_link', 'khong_doc_duoc',
})
_BO_NHO = {}


def _hex64(v):
    return isinstance(v, str) and _HEX64.fullmatch(v) is not None


def la_duong_dan(s):
    if not isinstance(s, str) or s == '' or '\n' in s or '\r' in s:
        return False
    return ('/' in s or '\\' in s
            or re.search(r'\.[A-Za-z0-9]{1,6}$', s) is not None)


def sha_cac_dang(path, f=None):
    # f: file nhi phan da mo (C3-34: stat va doc tren cung descriptor)
    p = Path(path).resolve()
    if f is None:
        with open(p, 'rb') as f2:
            return sha_cac_dang(p, f2)
    st = os.fstat(f.fileno())
    chia_khoa = (str(p), st.st_size, st.st_mtime_ns)
    cu = _BO_NHO.get(chia_khoa)
    if cu is not None:
        return dict(cu)
    if p.suffix.lower() in BIN_EXT:
        bam = hashlib.sha256()
        for mieng in iter(lambda: f.read(1 << 20), b''):
            bam.update(mieng)
        out = {'tho': bam.hexdigest()}
    else:
        data = f.read()
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
    try:
        if not p.is_file():
            return 'thieu_file'
        with open(p, 'rb') as f:
            if os.fstat(f.fileno()).st_nlink > 1:
                return 'hard_link'
            dang = sha_cac_dang(p, f)
    except FileNotFoundError:
        return 'thieu_file'
    except OSError:
        return 'khong_doc_duoc'
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


def _la_khoa_duong_dan(k):
    return isinstance(k, str) and 'sha256' not in k.lower() and la_duong_dan(k)


def _la_ban_do_hash(d):
    # C3-22 vong 3: ban do = co khoa duong dan mang hex64 hay null; hoac
    # (luat cu) moi khoa la duong dan, gia tri null/chuoi khong phai duong dan
    if not d:
        return False
    co_hash = False
    tat_ca = True
    co_str = False
    for k, v in d.items():
        if not _la_khoa_duong_dan(k):
            tat_ca = False
            continue
        if v is None or _hex64(v):
            co_hash = True
        if v is None:
            continue
        if isinstance(v, str) and not la_duong_dan(v):
            co_str = True
        else:
            tat_ca = False
    return co_hash or (tat_ca and co_str)


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


def _tim_duong_dan(k, D):
    # cands: [(gia_tri, dang)] da loc la_duong_dan; rieng=True khi khoa
    # 'sha256' co sha256_file/sha256_duong_dan/sha256_path di kem
    cands = []
    rieng = False
    if k.lower() == 'sha256':
        thuong = {kk.lower() for kk in D if isinstance(kk, str)}
        if any(kk in thuong for kk in
               ('sha256_file', 'sha256_duong_dan', 'sha256_path')):
            rieng = True
        else:
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
    return cands, rieng


def _xu_sha(k, v, D, path, cap, kh, st):
    khoa = _con(path, k)
    if v is None:
        cands, rieng = _tim_duong_dan(k, D)
        if rieng or not cands:
            st[0] += 1
        else:
            kh.append({'khoa': khoa, 'gia_tri': 'null',
                       'ly_do': 'sha_rong_bat_buoc'})
        return
    if isinstance(v, str):
        if not _hex64(v):
            kh.append({'khoa': khoa, 'gia_tri': v[:200],
                       'ly_do': 'sha_khong_hop_le'})
            return
        cands, rieng = _tim_duong_dan(k, D)
        if rieng:
            kh.append({'khoa': khoa, 'gia_tri': v[:200],
                       'ly_do': 'co_sha256_file_rieng'})
            return
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
                if la_duong_dan(k2):
                    kh.append({'khoa': sub, 'gia_tri': 'null',
                               'ly_do': 'sha_rong_bat_buoc'})
                else:
                    st[0] += 1
            elif isinstance(v2, str):
                kh.append({'khoa': sub, 'gia_tri': v2[:200],
                           'ly_do': 'sha_khong_hop_le'})
            elif isinstance(v2, (dict, list)) and la_duong_dan(k2):
                # C3-22 vong 3: khoa duong dan mang object/list khong duyet
                kh.append({'khoa': sub, 'gia_tri': str(v2)[:200],
                           'ly_do': 'kieu_khong_ho_tro'})
            elif isinstance(v2, (dict, list)):
                _duyet(v2, sub, cap, kh, st)
            else:
                kh.append({'khoa': sub, 'gia_tri': str(v2)[:200],
                           'ly_do': 'kieu_khong_ho_tro'})
        return
    kh.append({'khoa': khoa, 'gia_tri': str(v)[:200],
               'ly_do': 'kieu_khong_ho_tro'})


def _duyet(v, path, cap, kh, st, *, ban_do_sha=False):
    if isinstance(v, dict):
        if _la_manifest(v):
            for k2, v2 in v.items():
                cap.append({'khoa': _manifest_con(path, k2),
                            'duong_dan': k2, 'sha': v2, 'dang': 'manifest'})
            return
        ban_do = ban_do_sha or _la_ban_do_hash(v)
        for k, v2 in v.items():
            if ban_do and (_la_khoa_duong_dan(k) or (ban_do_sha and la_duong_dan(k))):
                # C3-22: bao tung gia tri sai kieu, khong duyet tiep vao muc do.
                sub = _manifest_con(path, k)
                if _hex64(v2):
                    cap.append({'khoa': sub, 'duong_dan': k, 'sha': v2,
                                'dang': 'manifest'})
                elif v2 is None:
                    kh.append({'khoa': sub, 'gia_tri': 'null',
                               'ly_do': 'sha_rong_bat_buoc'})
                else:
                    kh.append({'khoa': sub, 'gia_tri': str(v2)[:200],
                               'ly_do': 'sha_khong_hop_le'})
            elif isinstance(k, str) and 'sha256' in k.lower():
                _xu_sha(k, v2, v, path, cap, kh, st)
            else:
                _duyet(v2, _con(path, k), cap, kh, st, ban_do_sha=k == 'outputs')
    elif isinstance(v, list):
        for i, item in enumerate(v):
            _duyet(item, _chi_so(path, i), cap, kh, st)


def tim_cap(obj):
    cap = []
    kh = []
    st = [0]
    _duyet(obj, '', cap, kh, st)
    return cap, kh, st[0]


def ung_vien(duong_dan, thu_muc_json, goc_repo, goc_luot=None):
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
        if '/' in rel and rel.split('/', 1)[0].lower() == '05_data':
            # duong tinh tu goc repo (vd. khoa_u_h_chung.json cua C8B): mot dich
            # duy nhat la goc_repo/<duong>; khong thu goc luot hay thu muc JSON
            if goc_repo is not None:
                out.append(('goc_repo_05_data', Path(goc_repo) / rel))
            return out
        if '/' not in rel:
            out.append(('thu_muc_json', Path(thu_muc_json) / rel))
        elif goc_luot is not None:
            out.append(('goc_luot', Path(goc_luot) / rel))
        else:
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


def kiem_cap(duong_dan, sha, thu_muc_json, goc_repo, goc_luot=None):
    s = str(duong_dan)
    tuong_doi = not (_TUYET_DOI.match(s) or s.startswith('/')
                     or s.startswith('\\'))
    if tuong_doi and any(x.lower() in _SEALED
                         for x in re.split(r'[\\/]', s)):
        return 'bo_qua_sealed', '', ''
    ds = ung_vien(s, thu_muc_json, goc_repo, goc_luot)
    if not ds:
        if tuong_doi and '/' in s.replace('\\', '/'):
            return 'khong_co_goc_luot', '', ''
        return 'thieu_file', '', ''
    cach, p = ds[0]
    if _la_sealed(p):
        return 'bo_qua_sealed', '', ''
    giu = []
    try:
        if p.is_file():
            return kiem(p, sha), cach, str(p)
        da_co = {p.resolve()}
        for cach, p in ds[1:]:
            if _la_sealed(p):
                continue
            pr = p.resolve()
            if pr in da_co:
                continue
            if p.is_file():
                giu.append((cach, p))
                da_co.add(pr)
    except OSError:
        return 'khong_doc_duoc', cach, str(p)
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


def quet(goc, goc_repo, goc_luot=None):
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
            with open(pj, 'rb') as f:
                data = (None if os.fstat(f.fileno()).st_nlink > 1
                        else f.read())
        except OSError:
            kh_rows.append({'file_ghi': file_ghi, 'khoa': '',
                            'gia_tri': '', 'ly_do': 'khong_doc_duoc'})
            continue
        if data is None:
            kh_rows.append({'file_ghi': file_ghi, 'khoa': '',
                            'gia_tri': '', 'ly_do': 'hard_link'})
            continue
        try:
            obj = json.loads(data.decode('utf-8-sig'))
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
                                      pj.parent, goc_repo, goc_luot)
            if kq == 'bo_qua_sealed':
                kh_rows.append({'file_ghi': file_ghi, 'khoa': c['khoa'],
                                'gia_tri': c['duong_dan'],
                                'ly_do': 'tro_vao_sealed_khong_doc'})
                continue
            if kq in ('mo_ho_phan_giai', 'hard_link', 'khong_co_goc_luot',
                      'khong_doc_duoc'):
                kh_rows.append({'file_ghi': file_ghi, 'khoa': c['khoa'],
                                'gia_tri': c['duong_dan'],
                                'ly_do': kq})
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
    ap.add_argument('--goc-luot', default=None)
    ap.add_argument('--ra', '--out', dest='ra', default=None)
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
    goc_luot = None
    if a.goc_luot:
        goc_luot = Path(a.goc_luot)
        if not goc_luot.is_dir():
            _in('loi: --goc-luot khong phai thu muc')
            return 2
        goc_luot = goc_luot.resolve()
        if _la_sealed(goc_luot):
            _in('loi: --goc-luot nam trong sealed/sealed_test')
            return 2
    if a.ra:
        ra = Path(a.ra).resolve()
    else:
        ra = Path(__file__).resolve().parents[1] / 'ket_qua_sha'
    if _trong(ra, goc) or (goc_repo is not None and _trong(ra, goc_repo)):
        _in('loi: --ra nam trong --goc hoac goc_repo')
        return 2
    rows, kh_rows, tom_tat = quet(goc, goc_repo, goc_luot)
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
    tt['goc_luot'] = str(goc_luot) if goc_luot is not None else None
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
