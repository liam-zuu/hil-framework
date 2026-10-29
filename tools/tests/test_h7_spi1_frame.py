"""H7 SPI1 handler (stm32h7/Core/Src/spi1_handler.c) on the PC: the unchanged C file
against a fake HAL; the harness plays frames by calling SPI1_Callback() as the
interrupt (stm32h7/pc_tests/spi1/spi1_sim_main.c).

Fix 29/10/2026 (S06 #3, Codex #23 on the HIL data flow): main loop reads state +
traj of one frame under masked interrupts, never writes the tx buffer the DMA
holds, and a VALID|NEW setpoint is latched for the relay. The old file
(fix/g2-check bbf3fb6) runs the checks its API allows, to show the test sees the
problem there. Not covered: an interrupt inside one struct copy (only the
masking is checked).

S06 (29/10/2026): MISO v4 (checksums, CYCCNT stamp, frame count, state age) and
the H7 log records carried one per MISO frame (s06_log.c).
Run: python -m pytest tools/tests  (needs gcc and git on PATH)"""
import os
import shutil
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
H7 = os.path.join(ROOT, 'stm32h7')
SIM = os.path.join(H7, 'pc_tests', 'spi1')
OLD_REV = 'bbf3fb6'          # fix/g2-check, spi1_handler.c before the fix
pytestmark = pytest.mark.skipif(shutil.which('gcc') is None, reason='gcc not on PATH')


def build(tmp, src_c, src_h, old=False):
    """Copy next to the fake HAL header: a quoted #include looks in the including file's folder first."""
    for f in (os.path.join(H7, 'Core', 'Inc', 'hil_protocol.h'), os.path.join(SIM, 'stm32h7xx_hal.h'),
              os.path.join(SIM, 'spi1_sim_main.c'), os.path.join(H7, 'Core', 'Inc', 's06_log.h'),
              os.path.join(H7, 'Core', 'Src', 's06_log.c')):
        shutil.copy(f, tmp)
    shutil.copy(src_c, os.path.join(tmp, 'spi1_handler.c'))
    shutil.copy(src_h, os.path.join(tmp, 'spi1_handler.h'))
    exe = os.path.join(tmp, 'spi1_sim.exe')
    cmd = ['gcc', '-std=gnu11', '-O2', '-Wall', '-Werror', '-I', str(tmp)]
    if old:
        cmd.append('-DOLD_API')
    cmd += [os.path.join(tmp, 'spi1_sim_main.c'), os.path.join(tmp, 'spi1_handler.c'),
            os.path.join(tmp, 's06_log.c'), '-o', exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


def run(exe):
    r = subprocess.run([exe], capture_output=True, text=True)
    res = {}
    for line in r.stdout.splitlines():
        status, _, rest = line.partition(' ')
        res[rest.split(':')[0]] = status
    return r.returncode, res, r.stdout


def git_show(path, dst):
    src = subprocess.run(['git', 'show', f'{OLD_REV}:{path}'], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    with open(dst, 'w', newline='\n') as f:
        f.write(src)
    return dst


@pytest.fixture(scope='module')
def new_result(tmp_path_factory):
    d = tmp_path_factory.mktemp('spi1_new')
    return run(build(d, os.path.join(H7, 'Core', 'Src', 'spi1_handler.c'),
                     os.path.join(H7, 'Core', 'Inc', 'spi1_handler.h')))


@pytest.fixture(scope='module')
def old_result(tmp_path_factory):
    d = tmp_path_factory.mktemp('spi1_old')
    c = git_show('stm32h7/Core/Src/spi1_handler.c', os.path.join(d, 'old_spi1_handler.c'))
    h = git_show('stm32h7/Core/Inc/spi1_handler.h', os.path.join(d, 'old_spi1_handler.h'))
    return run(build(d, c, h, old=True))


NEW_CHECKS = ['pair_same_frame', 'getframe_masked', 'getframe_clears_ready', 'unread_counted',
              'tx_untouched_while_armed', 'reply_lag_two_frames', 'setoutput_restores_primask',
              'new_not_lost', 'new_taken_once', 'new_latest_wins', 'new_dropped_at_link_lost',
              'new_counters', 'take_restores_primask', 'nested_mask_kept',
              # S06 (29/10): MISO v4 fields, checksums, H7 log records in the MISO
              'crc_check_value', 'crc_table_equals_ref', 'miso_crc_valid', 'mosi_crc_of_answered_frame',
              'rx_cyc_of_answered_frame', 'rx_nf_of_answered_frame', 'age_field', 'miso_crc_catches_flip',
              'imu_raw_same_frame', 'log_each_record_once_in_order', 'log_payload_intact',
              'log_full_drops_counted', 'log_drop_leaves_index_gap', 'log_restores_primask']


@pytest.mark.parametrize('name', NEW_CHECKS)
def test_new_file(new_result, name):
    _, res, out = new_result
    assert res.get(name) == 'PASS', out


def test_new_file_exit_code(new_result):
    code, _, out = new_result
    assert code == 0, out


@pytest.mark.parametrize('name', ['pair_same_frame', 'tx_untouched_while_armed'])
def test_old_file_shows_the_problem(old_result, name):
    _, res, out = old_result
    assert res.get(name) == 'FAIL', out
