"""tools/s06_runs.py (spec S06 V5, CB3): the file only grows, the open row comes first, the result comes from
#ket_qua, cause and evidence are typed by the person, a run closes only with every required tool (C2-8)."""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import s06_runs as sr  # noqa: E402


def opened(f, run='r1', test='T2', **kw):
    args = ['open', '--file', f, '--run', run, '--test', test, '--commit-pi', 'a1', '--commit-h7', 'b2',
            '--commit-esp32', 'c3', '--cap', kw.get('cap', 'st-link'), '--nguon', kw.get('nguon', 'usb')]
    return sr.main(args)


def result(f, tmp_path, text, run='r1', tool='s06_t1_check'):
    p = tmp_path / 'out.txt'
    p.write_text(text, encoding='utf-8')
    return sr.main(['result', '--file', f, '--run', run, '--tool', tool, '--from', str(p)])


def all_tools(f, tmp_path, test, run='r1', text='#ket_qua,ma=0\n', skip=()):
    for t in sr.required_tools()[test]:
        if t not in skip:
            assert result(f, tmp_path, text, run, t) == 0


def status(f, capsys):
    code = sr.main(['status', '--file', f])
    return code, capsys.readouterr().out


def rows(f):
    with open(f, newline='', encoding='utf-8') as h:
        return list(csv.DictReader(h))


@pytest.fixture
def f(tmp_path):
    return str(tmp_path / 'S06_runs.csv')


def test_open_row_before_any_number(f):
    assert opened(f) == 0
    r = rows(f)
    assert len(r) == 1 and r[0]['loai_dong'] == 'mo' and r[0]['ma'] == ''
    assert (r[0]['lan'], r[0]['phep_thu'], r[0]['commit_pi'], r[0]['commit_h7'], r[0]['commit_esp32']) == \
           ('r1', 'T2', 'a1', 'b2', 'c3')
    assert (r[0]['cap'], r[0]['nguon']) == ('st-link', 'usb') and r[0]['ghi_luc']


def test_only_appends(f, tmp_path):
    opened(f)
    with open(f, 'rb') as h:
        before = h.read()
    assert result(f, tmp_path, 'x\n#ket_qua,ma=0\n') == 0
    assert sr.main(['cause', '--file', f, '--run', 'r1', '--nhom', 'mat_dien', '--bang-chung', 'ảnh, log']) == 0
    with open(f, 'rb') as h:
        assert h.read().startswith(before)
    assert [r['loai_dong'] for r in rows(f)] == ['mo', 'ket_qua', 'nguyen_nhan']


@pytest.mark.parametrize('text,code,why', [('#ket_qua,ma=0\n', '0', ''), ('a\n#ket_qua,ma=1\n\n', '1', ''),
                                           ('#ket_qua,ma=3,ly_do=chua_ket_luan\n', '3', 'chua_ket_luan'),
                                           ('#ket_qua,ma=3,ly_do=khong_du_dieu_kien\n', '3', 'khong_du_dieu_kien'),
                                           ('#ket_qua,ma=3,ly_do=thieu_bang_chung', '3', 'thieu_bang_chung')])
def test_result_from_last_line(f, tmp_path, text, code, why):
    opened(f)
    assert result(f, tmp_path, text) == 0
    r = rows(f)[-1]
    assert (r['ma'], r['ly_do'], r['cong_cu'], r['phep_thu']) == (code, why, 's06_t1_check', 'T2')


@pytest.mark.parametrize('text', ['', 'lỗi đầu vào: x\n', '#ket_qua,ma=0\nthêm dòng sau\n', '#ket_qua,ma=2\n',
                                  '#ket_qua,ma=3\n', '#ket_qua,ma=3,ly_do=khac\n', '#ket_qua,ma=0,ly_do=chua_ket_luan\n',
                                  '#ket_qua,ma=x\n'])
def test_no_valid_result_writes_nothing(f, tmp_path, text):
    """CB1: mã 2 không in #ket_qua, không phải kết quả lượt và không đóng lượt."""
    opened(f)
    n = len(rows(f))
    assert result(f, tmp_path, text) == 2
    assert len(rows(f)) == n


def test_result_and_cause_need_an_open_row(f, tmp_path):
    assert result(f, tmp_path, '#ket_qua,ma=0\n') == 2
    assert sr.main(['cause', '--file', f, '--run', 'r1', '--nhom', 'mat_dien', '--bang-chung', 'x']) == 2
    assert not os.path.exists(f)


def test_same_run_twice_refused(f):
    assert opened(f) == 0
    assert opened(f) == 2
    assert len(rows(f)) == 1


def test_cause_needs_evidence(f):
    opened(f)
    assert sr.main(['cause', '--file', f, '--run', 'r1', '--nhom', 'cap_thao_tac', '--bang-chung', '  ']) == 2
    assert len(rows(f)) == 1


def test_tool_never_classifies(f, tmp_path):
    """W5: mã 1 hay mã 3 không tự sinh nguyên nhân hay quyết định thay lượt."""
    opened(f)
    result(f, tmp_path, '#ket_qua,ma=1\n')
    r = rows(f)
    assert len(r) == 2 and r[1]['nhom_nguyen_nhan'] == '' and r[1]['bang_chung'] == ''


def test_status_unclosed_run(f, tmp_path, capsys):
    opened(f, 'r1')
    opened(f, 'r2', cap='khong', nguon='buck')
    code, out = status(f, capsys)
    assert code == 1 and 'r1' in out and 'CHƯA ĐÓNG' in out and '2 lượt, 2 chưa đóng' in out
    assert 'chưa lập S06_summary.csv, còn lượt mở: r1, r2' in out                 # CB3 (7)
    all_tools(f, tmp_path, 'T2', 'r1', '#ket_qua,ma=1\n')
    all_tools(f, tmp_path, 'T2', 'r2')
    code, out = status(f, capsys)
    assert code == 0 and '2 lượt, 0 chưa đóng' in out and 'chưa lập' not in out


def test_required_tools_are_the_spec_table():
    # spec 3.1 CB3 (Liam 30/10 "Ghi vào spec"); T1, T10 share the T2 run
    t2 = ['s06_t2_check.py', 's06_t1_check.py', 's06_h7_serial.py', 's06_load_activity.py', 's06_esp32_check.py']
    assert sr.required_tools() == {
        'T1': t2, 'T2': t2, 'T10': t2, 'T3b': ['t3b_score.py', 's06_esp32_check.py'],
        'T4': ['s06_t4_values.py', 's06_esp32_check.py'], 'T5': ['s06_t5_score.py', 's06_esp32_check.py'],
        'T6': ['s06_t6_check.py', 's06_esp32_check.py'],
        'CB5': ['s06_t2_check.py', 's06_esp32_check.py', 's06_frames_decode.py', 's06_enc_alive.py']}


def test_every_required_tool_closes_the_run(f, tmp_path, capsys):
    # CB3 (6): one result is not enough, also not a result of the wrong tool
    opened(f, test='T3b')
    assert result(f, tmp_path, '#ket_qua,ma=0\n', tool='t3b_score') == 0
    assert result(f, tmp_path, '#ket_qua,ma=0\n', tool='s06_t1_check') == 0
    code, out = status(f, capsys)
    assert code == 1 and 'thiếu kết quả s06_esp32_check' in out and 'thiếu kết quả t3b' not in out
    assert result(f, tmp_path, '#ket_qua,ma=0\n', tool='s06_esp32_check.py') == 0  # with or without .py
    assert status(f, capsys)[0] == 0


@pytest.mark.parametrize('test', ['T3', 'T7', 'T8', 'T9', 'T11'])
def test_test_without_a_list_stays_open(f, tmp_path, capsys, test):
    opened(f, test=test)
    result(f, tmp_path, '#ket_qua,ma=0\n')
    code, out = status(f, capsys)
    assert code == 1 and 'chưa có danh sách công cụ bắt buộc' in out


@pytest.mark.parametrize('line, closed', [('#ket_qua,ma=0\n', True), ('#ket_qua,ma=1\n', True),
                                          ('#ket_qua,ma=3,ly_do=chua_ket_luan\n', True),
                                          ('#ket_qua,ma=3,ly_do=khong_du_dieu_kien\n', False),
                                          ('#ket_qua,ma=3,ly_do=thieu_bang_chung\n', False)])
def test_close_rule_by_code(f, tmp_path, capsys, line, closed):
    # CB3 (6): mã 0, 1 and 3 (a) close; 3 (b), (c) only with the cause and evidence typed by the person
    opened(f, test='T6')
    assert result(f, tmp_path, '#ket_qua,ma=0\n', tool='s06_esp32_check') == 0
    assert result(f, tmp_path, line, tool='s06_t6_check') == 0
    code, out = status(f, capsys)
    assert (code == 0) == closed
    assert ('chưa ghi nhóm nguyên nhân' in out) == (not closed)
    assert sr.main(['cause', '--file', f, '--run', 'r1', '--nhom', 'chua_xac_dinh', '--bang-chung', 'không có']) == 0
    assert status(f, capsys)[0] == 0


def test_cause_may_come_before_the_result(f, tmp_path, capsys):
    # W5: the cause is written when it happens, before the data are processed
    opened(f, test='T6')
    assert sr.main(['cause', '--file', f, '--run', 'r1', '--nhom', 'mat_dien', '--bang-chung', 'log nguồn']) == 0
    all_tools(f, tmp_path, 'T6', text='#ket_qua,ma=3,ly_do=thieu_bang_chung\n')
    assert status(f, capsys)[0] == 0


def test_cause_of_another_run_does_not_count(f, tmp_path, capsys):
    opened(f, 'r1', test='T6')
    opened(f, 'r2', test='T6')
    assert sr.main(['cause', '--file', f, '--run', 'r2', '--nhom', 'mat_dien', '--bang-chung', 'x']) == 0
    all_tools(f, tmp_path, 'T6', 'r1', '#ket_qua,ma=3,ly_do=thieu_bang_chung\n')
    all_tools(f, tmp_path, 'T6', 'r2', '#ket_qua,ma=3,ly_do=thieu_bang_chung\n')
    code, out = status(f, capsys)
    assert code == 1 and 'chưa lập S06_summary.csv, còn lượt mở: r1' in out and 'r2' not in out.splitlines()[-1]


def test_latest_result_of_a_tool_counts(f, tmp_path, capsys):
    # a tool scored again: the file keeps both rows, the last one decides
    opened(f, test='T6')
    all_tools(f, tmp_path, 'T6', text='#ket_qua,ma=3,ly_do=khong_du_dieu_kien\n')
    assert status(f, capsys)[0] == 1
    assert result(f, tmp_path, '#ket_qua,ma=0\n', tool='s06_t6_check.py') == 0
    assert result(f, tmp_path, '#ket_qua,ma=0\n', tool='s06_esp32_check') == 0
    assert status(f, capsys)[0] == 0
    assert len(rows(f)) == 5
    assert result(f, tmp_path, '#ket_qua,ma=3,ly_do=thieu_bang_chung\n', tool='s06_t6_check') == 0
    assert status(f, capsys)[0] == 1


@pytest.mark.parametrize('text', ['{ khong phai json', '{"dung_chung": {"T1": "T2"}}'])
def test_bad_required_file_is_not_a_status(f, tmp_path, text):
    # mã 2, not mã 1 "a run is open"
    opened(f)
    p = tmp_path / 'req.json'
    p.write_text(text, encoding='utf-8')
    assert sr.main(['status', '--file', f, '--required', str(p)]) == 2
    assert sr.main(['status', '--file', f, '--required', str(tmp_path / 'thieu.json')]) == 2


def test_other_header_refused(f):
    with open(f, 'w', newline='') as h:
        h.write('a,b\n1,2\n')
    assert opened(f) == 2
    with open(f) as h:
        assert h.read() == 'a,b\n1,2\n'


def test_comma_and_quote_in_evidence_stay_one_row(f):
    opened(f)
    assert sr.main(['cause', '--file', f, '--run', 'r1', '--nhom', 'sai_quy_trinh',
                    '--bang-chung', 'nạp nhầm "bản", log H7, ảnh']) == 0
    r = rows(f)
    assert len(r) == 2 and r[1]['bang_chung'] == 'nạp nhầm "bản", log H7, ảnh'


def test_tools_end_with_the_line_this_reads(tmp_path):
    """Khớp với dòng các script chấm in (s06_stream.report_result)."""
    import s06_stream as stm
    import contextlib
    import io
    for code, why in ((0, ()), (1, ()), (3, ['khong_du_dieu_kien'])):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stm.report_result(code, why)
        assert sr.last_result(buf.getvalue())[0] == code
