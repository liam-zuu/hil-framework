"""Tests of tools/gain_table.py. Run: python -m pytest tools/tests/test_gain_table.py -q

The two tests that read the real firmware table are skipped when the thesis firmware tree is not
checked out next to this repo."""
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import gain_send as gs  # noqa: E402
import gain_table as gt  # noqa: E402

FW = 'D:/CLAUDE_THESIS/03_CODE/AGV_FIRMWARE/src/controller/gain_set.c'
needs_fw = pytest.mark.skipif(not os.path.isfile(FW), reason='firmware tree not checked out')

PID = dict(kp_xy=1.2, ki_xy=0.4, kp_wz=1.0, ki_wz=0.05, pos_kp=6, pos_ki=3, k_psi=1)
ADRC = dict(omega_c_xy=8, omega_o_xy=30, omega_c_wz=8, omega_o_wz=30, pos_kp=6, pos_ki=3, k_psi=1)

# expected table contents, written out here by hand (not derived from gt)
EXPECT_LO = {'PID': (0.0,) * 7, 'ADRC': (0.5, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0)}
EXPECT_HI = {'PID': (41.0, 1500.0, 4.6, 230.0, 200.0, 5500.0, 50.0),
             'ADRC': (200.0, 100.0, 200.0, 100.0, 200.0, 40000.0, 50.0)}

PC_FW = '''
static const float LO_PID[GAIN_N]  = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
static const float HI_PID[GAIN_N]  = { 41.0f, 1500.0f, 4.6f, 230.0f, 200.0f, 5500.0f, 50.0f };
static const float LO_ADRC[GAIN_N] = { 0.5f, 1.0f, 0.5f, 1.0f, 0.0f, 0.0f, 0.0f };
static const float HI_ADRC[GAIN_N] = { 200.0f, 100.0f, 200.0f, 100.0f, 200.0f, 40000.0f, 50.0f };
'''

FW_DIFF = ['HI_PID[0] kp_xy: PC 41, firmware 20',
           'HI_PID[1] ki_xy: PC 1500, firmware 50',
           'HI_PID[2] kp_wz: PC 4.5999999, firmware 20',
           'HI_PID[3] ki_wz: PC 230, firmware 50',
           'HI_PID[4] pos_kp: PC 200, firmware 30',
           'HI_PID[5] pos_ki: PC 5500, firmware 30',
           'HI_PID[6] k_psi: PC 50, firmware 10',
           'HI_ADRC[0] omega_c_xy: PC 200, firmware 50',
           'HI_ADRC[1] omega_o_xy: PC 100, firmware 49',
           'HI_ADRC[2] omega_c_wz: PC 200, firmware 50',
           'HI_ADRC[3] omega_o_wz: PC 100, firmware 49',
           'HI_ADRC[4] pos_kp: PC 200, firmware 30',
           'HI_ADRC[5] pos_ki: PC 40000, firmware 30',
           'HI_ADRC[6] k_psi: PC 50, firmware 10']

GOOD_FW = '''
static const float LO_PID[GAIN_N]  = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
static const float HI_PID[GAIN_N]  = { 20.0f, 50.0f, 20.0f, 50.0f, 30.0f, 30.0f, 10.0f };
static const float LO_ADRC[GAIN_N] = { 0.5f, 1.0f, 0.5f, 1.0f, 0.0f, 0.0f, 0.0f };
static const float HI_ADRC[GAIN_N] = { 50.0f, 49.0f, 50.0f, 49.0f, 30.0f, 30.0f, 10.0f };
'''


def ulp_up(x):
    """Next float32 above x (x positive), via the bits."""
    return struct.unpack('<f', struct.pack('<I', struct.unpack('<I', struct.pack('<f', x))[0] + 1))[0]


def f32_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def test_names_aliases_version():
    assert gt.NAMES == gs.NAMES and gt.ALIASES == gs.ALIASES
    assert gt.TABLE_VERSION == '2026-11-22 chot chung cau 1 + vong 6 a.7 (Liam 21/11)'


def test_table_values_and_float32_bits():
    assert gt.LO == EXPECT_LO and gt.HI == EXPECT_HI
    pc = gt.pc_arrays()
    for arr in gt.FW_ARRAYS:
        ctrl = 'PID' if arr.endswith('PID') else 'ADRC'
        exp = EXPECT_LO[ctrl] if arr.startswith('LO') else EXPECT_HI[ctrl]
        assert pc[arr] == [gt._f32(x) for x in exp]
    assert f32_bits(pc['HI_PID'][2]) == 0x40933333        # kp_wz = f32(4.6)


def test_base_sets_are_clean():
    assert gt.check('PID', dict(PID)) == ([], [])
    assert gt.check('ADRC', dict(ADRC)) == ([], [])


@pytest.mark.parametrize('ctrl,base', [('PID', PID), ('ADRC', ADRC)])
def test_bounds_themselves_are_allowed(ctrl, base):
    # S03 v8.6 2.1: the wz triangle is hard like xy, so the wz omega pair is treated like the xy pair
    for i, n in enumerate(gt.NAMES[ctrl]):
        for bound in (EXPECT_LO[ctrl][i], EXPECT_HI[ctrl][i]):
            gains = dict(base)
            for ax in ('xy', 'wz'):
                if ctrl == 'ADRC' and n == f'omega_c_{ax}':
                    gains[f'omega_o_{ax}'] = 1           # keep the {ax} triangle small
                if ctrl == 'ADRC' and n == f'omega_o_{ax}':
                    gains[f'omega_c_{ax}'] = 0.5
            gains[n] = bound
            v, w = gt.check(ctrl, gains)
            if ctrl == 'ADRC' and n.startswith('omega_') and bound == EXPECT_HI[ctrl][i]:
                # the ADRC omega upper bounds can never pass: the axis triangle fires first
                ax = n.rsplit('_', 1)[1]
                assert len(v) == 1 and v[0].startswith(f'omega_c_{ax} + 2*omega_o_{ax}'), (ctrl, n, v)
            else:
                assert v == [], (ctrl, n, bound, v)


def test_pid_kp_wz_bound_is_the_f32():
    assert gt.check('PID', dict(PID, kp_wz=4.6)) == ([], [])


@pytest.mark.parametrize('ctrl,base', [('PID', PID), ('ADRC', ADRC)])
def test_one_ulp_above_upper_is_a_violation(ctrl, base):
    # S03 v8.6 2.1: all four ADRC omega gains trip their axis triangle, not just the xy pair
    omegas = ('omega_c_xy', 'omega_o_xy', 'omega_c_wz', 'omega_o_wz')
    for i, n in enumerate(gt.NAMES[ctrl]):
        v, w = gt.check(ctrl, dict(base, **{n: ulp_up(EXPECT_HI[ctrl][i])}))
        assert any(n in x and 'upper bound' in x for x in v), (ctrl, n, v)
        if not (ctrl == 'ADRC' and n in omegas):
            assert len(v) == 1, (ctrl, n, v)     # the ADRC omega gains also trip the triangle


def test_below_lower_is_a_violation():
    v, w = gt.check('ADRC', dict(ADRC, omega_c_xy=0.49))
    assert any('omega_c_xy' in x and 'lower bound' in x for x in v)
    v, w = gt.check('PID', dict(PID, kp_xy=-1e-6))
    assert any('kp_xy' in x and 'lower bound' in x for x in v)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf'), 1e100])
def test_not_finite_is_a_violation(bad):
    v, w = gt.check('PID', dict(PID, kp_xy=bad))
    assert v and all('not finite' in x for x in v) and any('kp_xy' in x for x in v)


def test_xy_triangle():
    assert gt.check('ADRC', dict(ADRC, omega_c_xy=99.9, omega_o_xy=50)) == ([], [])
    v, w = gt.check('ADRC', dict(ADRC, omega_c_xy=100, omega_o_xy=50))
    assert len(v) == 1 and v[0].startswith('omega_c_xy + 2*omega_o_xy')


def test_xy_sum_is_binary64_on_float32_values():
    # S03 v8.6 2.1: the sum is binary64 on the float32 values; float32 addition of
    # 0x42c7ffff + 2*50 would round to exactly 200.0 (tie to even), binary64 gives
    # 199.99999237060547 < 200, so the set passes
    assert f32_bits(99.99999237060547) == 0x42c7ffff
    assert gt.check('ADRC', dict(ADRC, omega_c_xy=99.99999237060547, omega_o_xy=50)) == ([], [])


def test_wz_sum_is_binary64_on_float32_values():
    # S03 v8.6 2.1: same binary64 sum on the wz axis
    assert f32_bits(99.99999237060547) == 0x42c7ffff
    assert gt.check('ADRC', dict(ADRC, omega_c_wz=99.99999237060547, omega_o_wz=50)) == ([], [])


@pytest.mark.parametrize('ax', ['xy', 'wz'])
def test_triangle_float32_rounding_crosses_the_threshold(ax):
    def chk(c, o):
        gains = dict(ADRC)
        gains[f'omega_c_{ax}'], gains[f'omega_o_{ax}'] = c, o
        return gt.check('ADRC', gains)

    prefix = f'omega_c_{ax} + 2*omega_o_{ax}'
    v, w = chk(99.999999, 50)                # float32 of 99.999999 is 100.0: sum exactly 200
    assert len(v) == 1 and v[0].startswith(prefix)
    v, w = chk(100, 49.9999999)              # float32 of 49.9999999 is 50.0: sum exactly 200
    assert len(v) == 1 and v[0].startswith(prefix)
    assert chk(103.99999, 48) == ([], [])    # float32 is 103.99999237060547: sum 199.99999... < 200
    v, w = chk(104, 48)                      # the binary64 sum is exactly 200
    assert len(v) == 1 and v[0].startswith(prefix)


def test_wedge_quotient_is_float32():
    # f32(1/5.5) is 0x3e3a2e8c; a pos_kp equal to that float32 still warns (kp <= q)
    assert f32_bits(0.1818181872367859) == 0x3e3a2e8c
    v, w = gt.check('ADRC', dict(ADRC, pos_ki=1, pos_kp=0.1818181872367859))
    assert v == [] and any('pos_kp' in x for x in w)


def test_wz_sum_is_hard():
    # S03 v8.6 2.1: the wz triangle violates, it does not warn
    v, w = gt.check('ADRC', dict(ADRC, omega_c_wz=100, omega_o_wz=50))
    assert len(v) == 1 and v[0].startswith('omega_c_wz + 2*omega_o_wz') and w == []
    assert gt.check('ADRC', dict(ADRC, omega_c_wz=99.9, omega_o_wz=50)) == ([], [])


def test_wedge_warns_only():
    v, w = gt.check('PID', dict(PID, pos_ki=160, pos_kp=10))
    assert v == [] and any('pos_kp' in x for x in w)
    assert gt.check('PID', dict(PID, pos_ki=160, pos_kp=10.5)) == ([], [])
    assert gt.check('PID', dict(PID, pos_ki=0, pos_kp=0)) == ([], [])
    v, w = gt.check('ADRC', dict(ADRC, pos_ki=55, pos_kp=10))
    assert v == [] and any('pos_kp' in x for x in w)
    assert gt.check('ADRC', dict(ADRC, pos_ki=55, pos_kp=10.5)) == ([], [])


def test_aliases():
    aliased = dict(ADRC)
    for a, n in gt.ALIASES.items():
        aliased[a] = aliased.pop(n)
    assert gt.check('ADRC', aliased) == gt.check('ADRC', dict(ADRC))
    with pytest.raises(ValueError, match='twice'):
        gt.check('ADRC', dict(ADRC, wc_xy=9))


@pytest.mark.parametrize('ctrl', ['pid', 'LADRC', None])
def test_bad_ctrl(ctrl):
    with pytest.raises(ValueError, match='PID or ADRC'):
        gt.check(ctrl, dict(PID))


def test_missing_and_extra_gains():
    with pytest.raises(ValueError, match='missing k_psi'):
        gt.check('PID', {k: v for k, v in PID.items() if k != 'k_psi'})
    with pytest.raises(ValueError, match='not for PID'):
        gt.check('PID', dict(PID, omega_c_xy=3))
    with pytest.raises(ValueError, match='object'):
        gt.check('PID', [])
    with pytest.raises(ValueError, match='not a string'):
        gt.check('PID', {**PID, 1: 0})


def test_upper_lower_by_name():
    for c in gt.NAMES:
        for i, n in enumerate(gt.NAMES[c]):
            assert gt.upper(c, n) == EXPECT_HI[c][i]
            assert gt.lower(c, n) == EXPECT_LO[c][i]
            assert gt._f32(gt.upper(c, n)) == gt.pc_arrays()['HI_' + c][i]
    assert gt.upper('PID', 'kp_wz') == 4.6          # the table number as written, binary64
    assert gt.upper('ADRC', 'wc_xy') == 200.0       # aliases accepted
    assert gt.lower('ADRC', 'wo_wz') == 1.0
    with pytest.raises(ValueError, match='PID or ADRC'):
        gt.upper('pid', 'kp_xy')
    with pytest.raises(ValueError):
        gt.upper('PID', 'omega_c_xy')


def test_classify_parts():
    assert gt.classify('PID', dict(PID)) == gt.Verdict(True, True, (), ())
    assert gt.classify('ADRC', dict(ADRC)) == gt.Verdict(True, True, (), ())
    v = gt.classify('PID', dict(PID, kp_xy=ulp_up(41.0)))
    assert v.trong_bang is False and v.trong_tam_giac is True and v.vi_pham_tam_giac == ()
    assert len(v.vi_pham_bang) == 1 and 'kp_xy' in v.vi_pham_bang[0] and 'upper bound' in v.vi_pham_bang[0]
    assert gt.classify('PID', dict(PID, kp_xy=41.0000001)).trong_bang is True     # float32 is 41.0
    assert gt.classify('PID', dict(PID, kp_wz=4.60000001)).trong_bang is True     # float32 is f32(4.6)
    assert gt.classify('PID', dict(PID, kp_wz=4.6000002)).trong_bang is False     # float32 above f32(4.6)
    # 0 is in the table exactly where LO is 0
    assert gt.classify('PID', dict(PID, ki_xy=0)).trong_bang is True
    assert gt.classify('ADRC', dict(ADRC, pos_kp=0, pos_ki=0, k_psi=0)).trong_bang is True
    assert gt.classify('ADRC', dict(ADRC, omega_c_xy=0)).trong_bang is False      # LO is 0.5
    assert gt.classify('ADRC', dict(ADRC, omega_o_wz=1.0)).trong_bang is True
    assert gt.classify('ADRC', dict(ADRC, omega_o_wz=0.9999999403953552)).trong_bang is False
    v = gt.classify('ADRC', dict(ADRC, omega_c_wz=150, omega_o_wz=48))
    assert (v.trong_bang, v.trong_tam_giac) == (True, False) and v.vi_pham_bang == ()
    assert len(v.vi_pham_tam_giac) == 1 and v.vi_pham_tam_giac[0].startswith('omega_c_wz + 2*omega_o_wz')
    v = gt.classify('ADRC', dict(ADRC, omega_c_xy=ulp_up(200.0)))
    assert (v.trong_bang, v.trong_tam_giac) == (False, False)
    v = gt.classify('PID', dict(PID, kp_xy=float('nan')))
    assert (v.trong_bang, v.trong_tam_giac) == (False, True)
    v = gt.classify('ADRC', dict(ADRC, omega_c_xy=float('nan')))
    assert (v.trong_bang, v.trong_tam_giac) == (False, False) and v.vi_pham_tam_giac == ()
    with pytest.raises(ValueError):
        gt.classify('PID', {k: x for k, x in PID.items() if k != 'k_psi'})


def test_check_is_classify_plus_wedge():
    sets = [('PID', dict(PID)),
            ('ADRC', dict(ADRC)),
            ('PID', dict(PID, kp_xy=41.01)),
            ('ADRC', dict(ADRC, omega_c_xy=100, omega_o_xy=50)),
            ('ADRC', dict(ADRC, omega_c_wz=100, omega_o_wz=50)),
            ('ADRC', dict(ADRC, omega_c_xy=100, omega_o_xy=50, omega_c_wz=100, omega_o_wz=50)),
            ('ADRC', dict(ADRC, pos_ki=55, pos_kp=10))]
    for c, g in sets:
        v = gt.classify(c, g)
        assert gt.check(c, g)[0] == list(v.vi_pham_bang) + list(v.vi_pham_tam_giac)
    wedge = sets[-1]
    assert gt.check(*wedge)[1] != []                       # the wedge warning
    assert gt.classify(*wedge) == gt.Verdict(True, True, (), ())


@needs_fw
def test_read_firmware_table():
    assert gt.read_firmware_table(FW) == {
        'LO_PID': [0.0] * 7,
        'HI_PID': [20.0, 50.0, 20.0, 50.0, 30.0, 30.0, 10.0],
        'LO_ADRC': [0.5, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0],
        'HI_ADRC': [50.0, 49.0, 50.0, 49.0, 30.0, 30.0, 10.0]}


def test_read_firmware_table_bad(tmp_path):
    f = tmp_path / 'gain_set.c'
    f.write_text(GOOD_FW.replace('HI_ADRC', 'HI_ADRC2'))                    # one array missing
    with pytest.raises(ValueError, match='HI_ADRC'):
        gt.read_firmware_table(str(f))
    f.write_text(GOOD_FW.replace('30.0f, 30.0f, 10.0f };\nstatic const float LO_ADRC',
                                 '30.0f, 10.0f };\nstatic const float LO_ADRC'))   # HI_PID has 6 items
    with pytest.raises(ValueError, match='7'):
        gt.read_firmware_table(str(f))
    f.write_text('// static const float HI_PID[7] = { 9,9,9,9,9,9,9 };\n'
                 '/* static const float LO_PID[7] = { 8,8,8,8,8,8,8 }; */\n' + GOOD_FW)
    t = gt.read_firmware_table(str(f))                                      # fake arrays in comments ignored
    assert t['HI_PID'] == [20.0, 50.0, 20.0, 50.0, 30.0, 30.0, 10.0] and t['LO_PID'] == [0.0] * 7


def test_read_firmware_table_static_and_commas(tmp_path):
    f = tmp_path / 'gain_set.c'
    f.write_text(PC_FW.replace('static const float HI_PID', 'static/**/const float HI_PID'))
    assert gt.read_firmware_table(str(f))['HI_PID'] == [gt._f32(x) for x in EXPECT_HI['PID']]
    f.write_text(PC_FW.replace('50.0f };', '50.0f, };'))          # one trailing comma is fine
    t = gt.read_firmware_table(str(f))
    assert t['HI_PID'] == [gt._f32(x) for x in EXPECT_HI['PID']]
    for bad in ('{ 41.0f,, 1500.0f', '{ , 41.0f'):               # empty item not at the end
        f.write_text(PC_FW.replace('{ 41.0f', bad, 1))
        with pytest.raises(ValueError):
            gt.read_firmware_table(str(f))
    f.write_text(PC_FW.replace('50.0f };', '50.0f,, };'))        # two trailing commas
    with pytest.raises(ValueError):
        gt.read_firmware_table(str(f))


LITERAL_OK = [('41.000001907348633f', 41.000003814697266, 0x42240001),   # rounds past the double
              ('41.000001907348633', 41.0, 0x42240000),                  # no f = double constant
              ('2E+1F', 20.0, 0x41a00000),
              ('.5f', 0.5, 0x3f000000),
              ('5.f', 5.0, 0x40a00000),
              ('041.0f', 41.0, 0x42240000),                              # leading zero, still decimal
              ('41', 41.0, 0x42240000),
              ('0', 0.0, 0x00000000),
              ('-41.0f', -41.0, 0xc2240000)]
LITERAL_BAD = ['041', '00', '0x29', '0x1p5f', '41.0L', '41.0l', '41u', '41U', '41f', 'x', '1..0f']


@pytest.mark.parametrize('lit,val,bits', LITERAL_OK)
def test_firmware_literal_accepted(tmp_path, lit, val, bits):
    f = tmp_path / 'gain_set.c'
    f.write_text(PC_FW.replace('41.0f', lit, 1))
    got = gt.read_firmware_table(str(f))['HI_PID'][0]
    assert got == val and f32_bits(got) == bits


@pytest.mark.parametrize('lit', LITERAL_BAD)
def test_firmware_literal_rejected(tmp_path, lit):
    # e.g. 041 in place of HI_PID[0] must raise (binary 33), not be read as 41 and report equal
    f = tmp_path / 'gain_set.c'
    f.write_text(PC_FW.replace('41.0f', lit, 1))
    with pytest.raises(ValueError):
        gt.read_firmware_table(str(f))


@needs_fw
def test_diff_with_firmware():
    assert gt.diff_with_firmware(FW) == FW_DIFF


def test_diff_empty_when_equal(tmp_path):
    f = tmp_path / 'gain_set.c'
    f.write_text(PC_FW)
    assert gt.diff_with_firmware(str(f)) == []
    f.write_text(PC_FW.replace('41.0f', '40.0f', 1))
    assert gt.diff_with_firmware(str(f)) == ['HI_PID[0] kp_xy: PC 41, firmware 40']
    f.write_text(GOOD_FW)
    assert gt.diff_with_firmware(str(f)) == FW_DIFF
