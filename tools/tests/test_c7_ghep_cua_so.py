"""Tests of tools/c7_ghep_cua_so.py (join of the per-run cua_so_v1.jsonl of a C8 session, CODE 20/11 2.4) on
SYNTHETIC files (not a measurement)."""
import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c7_ghep_cua_so as g  # noqa: E402

SES = 'e' * 64
CLIPS = [('IMG_1', 'a' * 64, '2026-11-20T05:00:00'), ('IMG_2', 'b' * 64, '2026-11-20T05:10:00'),
         ('IMG_3', 'c' * 64, '2026-11-20T05:20:00'), ('IMG_4', 'd' * 64, '2026-11-20T05:30:00')]
THU_TU = [('IMG_0', 'H0a', ''), ('IMG_1', 'HC_goc', 'HC_goc_1'), ('IMG_2', 'C8-P-F02-1', ''),
          ('IMG_3', 'C8-P-K1-1_khoa', ''), ('IMG_4', 'C8-P-K1-1', '')]


def win(lid, clip, loai, t, cho='goc', ses=SES, n=3):
    return dict(schema='cua_so_v1', luot_id=lid, cho=cho, loai=loai, clip_sha256=clip, session_sha256=ses,
                khung=[dict(id=int(t * 60) + k, pts_s=t + k / 60) for k in range(n)])


def write_jsonl(p, rows, nl='\n'):
    with open(p, 'w', encoding='utf-8', newline='') as f:
        for r in rows:
            f.write(json.dumps(r) + nl)
    return str(p)


@pytest.fixture
def buoi(tmp_path):
    tt = tmp_path / 'thu_tu_clip.csv'
    with open(tt, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['clip', 'buoc', 'thu_id', 'gio_pc_bam_quay', 'gio_pc_dung_quay', 'created_utc', 'log_pi',
                    'ghi_chu'])
        for c, b, t in THU_TU:
            w.writerow([c, b, t, '', '', '', '', ''])
    man = tmp_path / 'manifest_usb.csv'
    with open(man, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n', 'src_folder', 'src_name', 'bytes', 'sha256', 'created_utc'])
        for k, (c, s, cu) in enumerate([('IMG_0', 'f' * 64, '2026-11-20T04:50:00')] + CLIPS, start=1):
            w.writerow([k, 'x', c, 1, s, cu])
    hc = write_jsonl(tmp_path / 'hc.jsonl', [win('HC_goc_1', 'a' * 64, 'ref_12b', 1.0),
                                             win('HC_goc_1', 'a' * 64, 'dung_12b', 5.0),
                                             win('HC_goc_1', 'a' * 64, 'E_dau', 9.0),
                                             win('HC_goc_1', 'a' * 64, 'E_cuoi', 20.0)], nl='\r\n')
    f02 = write_jsonl(tmp_path / 'f02.jsonl', [win('C8-P-F02-1', 'b' * 64, 'pre', 1.0),
                                               win('C8-P-F02-1', 'b' * 64, 'post', 30.0)])
    k1 = write_jsonl(tmp_path / 'k1.jsonl', [win('C8-P-K1-1', 'd' * 64, 'pre', 1.0, cho='K1'),
                                             win('C8-P-K1-1', 'd' * 64, 'post', 25.0, cho='K1')])
    return dict(tt=str(tt), man=str(man), hc=hc, f02=f02, k1=k1, d=tmp_path)


def run(b, files, out='ra/cua_so_v1.jsonl', bo=()):
    argv = ['--thu-tu', b['tt'], '--manifest', b['man'], '--cua-so', *files, '--out', str(b['d'] / out)]
    for x in bo:
        argv += ['--bo', x]
    return g.main(argv)


def test_ghep_theo_thu_tu_ghi_hinh_giu_nguyen_dong(buoi, capsys):
    assert run(buoi, [buoi['k1'], buoi['f02'], buoi['hc']]) == 0          # files given out of order
    out = buoi['d'] / 'ra' / 'cua_so_v1.jsonl'
    raw = out.read_bytes()
    exp = b''.join(open(p, 'rb').read().replace(b'\r\n', b'\n') for p in (buoi['hc'], buoi['f02'], buoi['k1']))
    assert raw == exp                                                    # rows copied as they are, recording order
    rows = [json.loads(x) for x in raw.decode().splitlines()]
    assert [r['luot_id'] for r in rows] == ['HC_goc_1'] * 4 + ['C8-P-F02-1'] * 2 + ['C8-P-K1-1'] * 2
    n = json.load(open(str(out) + '.nguon.json', encoding='utf-8'))
    assert n['schema'] == 'cua_so_ghep_v1'
    assert [s['thu_tu_dong'] for s in n['nguon']] == [3, 4, 6]
    assert n['nguon'][0]['sha256'] == g.sha256_file(buoi['hc']) and n['nguon'][0]['so_dong'] == 4
    assert n['ra']['sha256'] == g.sha256_file(out) and n['ra']['so_dong'] == 8
    assert n['thu_tu_clip']['sha256'] == g.sha256_file(buoi['tt'])
    assert capsys.readouterr().out.strip().endswith('#ket_qua,ma=0')


def test_thieu_file_luot_bi_tu_choi_tru_khi_bo(buoi):
    assert run(buoi, [buoi['hc'], buoi['f02']]) == 2
    assert not (buoi['d'] / 'ra' / 'cua_so_v1.jsonl').exists()
    assert run(buoi, [buoi['hc'], buoi['f02']], bo=['C8-P-K1-1=log hong']) == 0
    n = json.load(open(str(buoi['d'] / 'ra' / 'cua_so_v1.jsonl') + '.nguon.json', encoding='utf-8'))
    assert n['bo'] == [dict(khoa='C8-P-K1-1', ly_do='log hong')]


@pytest.mark.parametrize('bo', ['C8-P-K1-1_khoa=x', 'C8-P-F02-1=x', 'C8-P-K1-1', 'C8-P-K1-1='])
def test_bo_sai_bi_tu_choi(buoi, bo):
    files = [buoi['hc'], buoi['f02']] + ([buoi['k1']] if bo.startswith('C8-P-F02') else [])
    assert run(buoi, files, bo=[bo]) == 2


def test_hai_file_cung_clip_hay_luot_trung(buoi):
    f02b = write_jsonl(buoi['d'] / 'f02b.jsonl', [win('C8-P-F02-1', 'b' * 64, 'pre', 2.0)])
    assert run(buoi, [buoi['hc'], buoi['f02'], f02b, buoi['k1']]) == 2


def test_luot_id_khac_khoa_hay_thieu(buoi):
    bad = write_jsonl(buoi['d'] / 'bad.jsonl', [win('O1-F02', 'b' * 64, 'pre', 1.0)])
    assert run(buoi, [buoi['hc'], bad, buoi['k1']]) == 2
    r = win('x', 'b' * 64, 'pre', 1.0)
    r['luot_id'] = None
    bad2 = write_jsonl(buoi['d'] / 'bad2.jsonl', [r])
    assert run(buoi, [buoi['hc'], bad2, buoi['k1']]) == 2


def test_clip_khong_phai_luot_hay_nhieu_clip(buoi):
    khoa = write_jsonl(buoi['d'] / 'khoa.jsonl', [win('C8-P-K1-1_khoa', 'c' * 64, 'pre', 1.0)])
    assert run(buoi, [buoi['hc'], buoi['f02'], buoi['k1'], khoa]) == 2
    mix = write_jsonl(buoi['d'] / 'mix.jsonl', [win('C8-P-F02-1', 'b' * 64, 'pre', 1.0),
                                                win('C8-P-F02-1', 'd' * 64, 'post', 30.0)])
    assert run(buoi, [buoi['hc'], mix, buoi['k1']]) == 2


def test_thu_tu_trong_clip_sai_bi_tu_choi(buoi):
    rev = write_jsonl(buoi['d'] / 'rev.jsonl', [win('C8-P-F02-1', 'b' * 64, 'post', 30.0),
                                                win('C8-P-F02-1', 'b' * 64, 'pre', 1.0)])
    assert run(buoi, [buoi['hc'], rev, buoi['k1']]) == 2


def test_created_utc_nguoc_bi_tu_choi(buoi):
    rows = list(csv.reader(open(buoi['man'], encoding='utf-8')))
    for r in rows:
        if r[2] == 'IMG_4':
            r[5] = '2026-11-20T04:00:00'
    with open(buoi['man'], 'w', encoding='utf-8', newline='') as f:
        csv.writer(f).writerows(rows)
    assert run(buoi, [buoi['hc'], buoi['f02'], buoi['k1']]) == 2


def test_nhieu_session_hay_thieu_session(buoi):
    k1b = write_jsonl(buoi['d'] / 'k1b.jsonl', [win('C8-P-K1-1', 'd' * 64, 'pre', 1.0, cho='K1', ses='9' * 64)])
    assert run(buoi, [buoi['hc'], buoi['f02'], k1b]) == 2
    r = win('C8-P-K1-1', 'd' * 64, 'pre', 1.0, cho='K1')
    r['session_sha256'] = None
    k1c = write_jsonl(buoi['d'] / 'k1c.jsonl', [r])
    assert run(buoi, [buoi['hc'], buoi['f02'], k1c]) == 2


def test_cua_so_rong_khong_tinh_thu_tu(buoi):
    post = win('C8-P-F02-1', 'b' * 64, 'post', 30.0)
    post['khung'] = []                                                   # CT4-3: clip shorter than the post window
    f = write_jsonl(buoi['d'] / 'f02e.jsonl', [win('C8-P-F02-1', 'b' * 64, 'pre', 1.0), post])
    assert run(buoi, [buoi['hc'], f, buoi['k1']]) == 0


def test_out_da_co_va_schema_khac(buoi):
    assert run(buoi, [buoi['hc'], buoi['f02'], buoi['k1']]) == 0
    assert run(buoi, [buoi['hc'], buoi['f02'], buoi['k1']]) == 2         # never overwrites
    r = win('C8-P-F02-1', 'b' * 64, 'pre', 1.0)
    r['schema'] = 'cua_so_v0'
    f = write_jsonl(buoi['d'] / 's.jsonl', [r])
    assert run(buoi, [buoi['hc'], f, buoi['k1']], out='ra2/x.jsonl') == 2


# ---- several camera segments (--doan; CODE 20/11 khuya chain A, C8 session 2 had three)
def _ses(b, name):
    p = b['d'] / f'{name}.json'
    p.write_text(json.dumps(dict(kind='c7_session', ten=name)), encoding='utf-8')
    return str(p), g.sha256_file(p)


def _hai_doan(b, ses_k1=None):
    pa, sa = _ses(b, 'sesA')
    pb, sb = _ses(b, 'sesB')
    hc = write_jsonl(b['d'] / 'hcA.jsonl', [win('HC_goc_1', 'a' * 64, 'ref_12b', 1.0, ses=sa)])
    f02 = write_jsonl(b['d'] / 'f02A.jsonl', [win('C8-P-F02-1', 'b' * 64, 'pre', 1.0, ses=sa),
                                              win('C8-P-F02-1', 'b' * 64, 'post', 30.0, ses=sa)])
    k1 = write_jsonl(b['d'] / 'k1B.jsonl', [win('C8-P-K1-1', 'd' * 64, 'pre', 1.0, cho='K1', ses=ses_k1 or sb)])
    return pa, sa, pb, sb, [hc, f02, k1]


def run_doan(b, files, doan, out='ra/cua_so_v1.jsonl'):
    argv = ['--thu-tu', b['tt'], '--manifest', b['man'], '--cua-so', *files, '--out', str(b['d'] / out)]
    for x in doan:
        argv += ['--doan', x]
    return g.main(argv)


def test_hai_doan_moi_file_dung_session_doan_minh(buoi):
    pa, sa, pb, sb, files = _hai_doan(buoi)
    # segment 2 opened at the IMG_3 row (C8-P-K1-1_khoa); the K1 run below it is of session B
    assert run_doan(buoi, files, [f'doan2={pb}@C8-P-K1-1_khoa', f'doan1={pa}@H0a']) == 0
    n = json.load(open(str(buoi['d'] / 'ra' / 'cua_so_v1.jsonl') + '.nguon.json', encoding='utf-8'))
    assert [(d['ten'], d['session']['sha256'], d['buoc_dau'], d['thu_tu_dong']) for d in n['doan']] == \
        [('doan1', sa, 'H0a', 2), ('doan2', sb, 'C8-P-K1-1_khoa', 5)]
    assert [(s['luot_id'], s['doan'], s['session_sha256']) for s in n['nguon']] == \
        [('HC_goc_1', 'doan1', sa), ('C8-P-F02-1', 'doan1', sa), ('C8-P-K1-1', 'doan2', sb)]


def test_hai_doan_session_sai_doan_bi_tu_choi(buoi):
    pa, sa, pb, sb, files = _hai_doan(buoi, ses_k1='0' * 64)
    assert run_doan(buoi, files, [f'doan1={pa}@H0a', f'doan2={pb}@C8-P-K1-1_khoa']) == 2
    pa, sa, pb, sb, files = _hai_doan(buoi, ses_k1=None)
    # the K1 run is of session B, but segment 2 is not declared: it falls in segment 1 (session A)
    assert run_doan(buoi, files, [f'doan1={pa}@H0a'], out='ra2/x.jsonl') == 2
    assert not (buoi['d'] / 'ra2' / 'x.jsonl').exists()


def test_khong_doan_van_tu_choi_nhieu_session(buoi):
    pa, sa, pb, sb, files = _hai_doan(buoi)
    assert run(buoi, files) == 2


@pytest.mark.parametrize('kieu', ['buoc_khong_co', 'ten_trung', 'session_trung', 'cung_dong', 'thieu_at',
                                  'luot_tren_doan_dau'])
def test_doan_sai_bi_tu_choi(buoi, kieu):
    pa, sa, pb, sb, files = _hai_doan(buoi)
    d1, d2 = f'doan1={pa}@H0a', f'doan2={pb}@C8-P-K1-1_khoa'
    doan = {'buoc_khong_co': [d1, f'doan2={pb}@H9z'], 'ten_trung': [d1, f'doan1={pb}@C8-P-K1-1_khoa'],
            'session_trung': [d1, f'doan2={pa}@C8-P-K1-1_khoa'], 'cung_dong': [d1, f'doan2={pb}@H0a'],
            'thieu_at': [d1, f'doan2={pb}'], 'luot_tren_doan_dau': [f'doan1={pa}@C8-P-F02-1', d2]}[kieu]
    assert run_doan(buoi, files, doan) == 2
