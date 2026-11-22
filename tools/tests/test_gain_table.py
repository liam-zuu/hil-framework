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

GOOD_FW = '''
static const float LO_PID[GAIN_N]  = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
static const float HI_PID[GAIN_N]  = { 20.0f, 50.0f, 20.0f, 50.0f, 30.0f, 30.0f, 10.0f };
static const float LO_ADRC[GAIN_N] = { 0.5f, 1.0f, 0.5f, 1.0f, 0.0f, 0.0f, 0.0f };
static const float HI_ADRC[GAIN_N] = { 50.0f, 49.0f, 50.0f, 49.0f, 30.0f, 30.0f, 10.0f };
'''


def ulp_up(x):
    """Next float32 above x (x positive), via the bits."""
    return struct.unpack('<f', struct.pack('<I', struct.unpack('<I', struct.pack('<f', x))[0] + 1))[0]


def test_names_aliases_version():
    assert gt.NAMES == gs.NAMES and gt.ALIASES == gs.ALIASES
    assert gt.TABLE_VERSION == '2026-11-22 chot chung cau 1 + vong 6 a.7 (Liam 21/11)'


def test_base_sets_are_clean():
    assert gt.check('PID', dict(PID)) == ([], [])
    assert gt.check('ADRC', dict(ADRC)) == ([], [])


@pytest.mark.parametrize('ctrl,base', [('PID', PID), ('ADRC', ADRC)])
def test_bounds_themselves_are_allowed(ctrl, base):
    for i, n in enumerate(gt.NAMES[ctrl]):
        for bound in (gt.LO[ctrl][i], gt.HI[ctrl][i]):
            gains = dict(base)
            if ctrl == 'ADRC' and n == 'omega_c_xy':
                gains['omega_o_xy'] = 1          # keep the xy triangle small
            if ctrl == 'ADRC' and n == 'omega_o_xy':
                gains['omega_c_xy'] = 0.5
            gains[n] = bound
            v, w = gt.check(ctrl, gains)
            if ctrl == 'ADRC' and n in ('omega_c_xy', 'omega_o_xy') and bound == gt.HI[ctrl][i]:
                # the ADRC xy upper bounds can never pass: the triangle fires first
                assert len(v) == 1 and v[0].startswith('omega_c_xy + 2*omega_o_xy')
            else:
                assert v == [], (ctrl, n, bound, v)


def test_pid_kp_wz_bound_is_the_f32():
    assert gt.check('PID', dict(PID, kp_wz=4.6)) == ([], [])


@pytest.mark.parametrize('ctrl,base', [('PID', PID), ('ADRC', ADRC)])
def test_one_ulp_above_upper_is_a_violation(ctrl, base):
    xy = ('omega_c_xy', 'omega_o_xy')
    for i, n in enumerate(gt.NAMES[ctrl]):
        v, w = gt.check(ctrl, dict(base, **{n: ulp_up(gt.HI[ctrl][i])}))
        assert any(n in x and 'upper bound' in x for x in v), (ctrl, n, v)
        if not (ctrl == 'ADRC' and n in xy):
            assert len(v) == 1, (ctrl, n, v)     # the ADRC xy pair also trips the triangle


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


def test_wz_sum_warns_only():
    v, w = gt.check('ADRC', dict(ADRC, omega_c_wz=100, omega_o_wz=50))
    assert v == [] and any('omega_c_wz' in x for x in w)
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


@needs_fw
def test_diff_with_firmware():
    assert gt.diff_with_firmware(FW) == [
        'HI_PID[0] kp_xy: PC 41, firmware 20',
        'HI_PID[1] ki_xy: PC 1500, firmware 50',
        f'HI_PID[2] kp_wz: PC {gt._f32(4.6):.9g}, firmware 20',
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


def test_diff_empty_when_equal(tmp_path):
    text = ''.join(f'static const float {a}[GAIN_N] = {{'
                   + ', '.join(f'{v:.9g}f' for v in gt.pc_arrays()[a]) + '};\n'
                   for a in gt.FW_ARRAYS)
    f = tmp_path / 'gain_set.c'
    f.write_text(text)
    assert gt.diff_with_firmware(str(f)) == []
