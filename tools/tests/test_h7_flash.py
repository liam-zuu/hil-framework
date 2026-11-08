"""tools/h7_flash.py: the pure parsing functions (no CubeIDE, no ST-Link, no serial port).
Text fixtures are shortened copies of the logs of the first command-line flash, 09/11
(CLOSED_LOOP/2026-11-09_run04/flash/ in the thesis repo)."""
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import h7_flash as hf  # noqa: E402

BUILD_OK = """00:34:00 **** Clean-only build of configuration Debug for project stm32h7 ****
make -j12 clean
arm-none-eabi-gcc "../Core/Src/main.c" -mcpu=cortex-m7 -O0 -c -o "Core/Src/main.o"
Finished building target: stm32h7.elf
arm-none-eabi-size  stm32h7.elf
   text	   data	    bss	    dec	    hex	filename
  99016	    488	  60032	 159536	  26f30	stm32h7.elf
Finished building: default.size.stdout

00:34:09 Build Finished. 0 errors, 0 warnings. (took 7s.214ms)

Headless build completed successfully
"""

FLASH_OK = """Board       : NUCLEO-H753ZI
Erasing memory corresponding to segment 0:
File download complete
Verifying...
Download verified successfully
MCU Reset
Software reset is performed
"""


def test_build_ok():
    b = hf.parse_build_log(BUILD_OK)
    assert b['ok'] and b['errors'] == 0 and b['warnings'] == 0 and b['cleaned']
    assert b['size'] == dict(text=99016, data=488, bss=60032, dec=159536)


def test_build_errors_not_ok():
    t = BUILD_OK.replace('0 errors, 0 warnings', '2 errors, 1 warning')
    b = hf.parse_build_log(t)
    assert not b['ok'] and b['errors'] == 2 and b['warnings'] == 1


def test_build_without_headless_success_line_not_ok():
    assert not hf.parse_build_log(BUILD_OK.replace('Headless build completed successfully', ''))['ok']


def test_build_without_size_not_ok():
    t = '\n'.join(l for l in BUILD_OK.splitlines() if 'stm32h7.elf' not in l or 'arm-none' in l or 'Finished' in l)
    assert hf.parse_build_log(t)['size'] is None and not hf.parse_build_log(t)['ok']


def test_flash_ok():
    f = hf.parse_flash_log(FLASH_OK)
    assert f['ok'] and f['reset'] and f['board'] == 'NUCLEO-H753ZI' and f['erased']


def test_flash_verify_failed():
    f = hf.parse_flash_log(FLASH_OK.replace('Download verified successfully', 'Error: data mismatch'))
    assert not f['ok']


def test_cubeide_running():
    gui = '"stm32cubeide.exe","1234","Console","1","500,000 K"\n"explorer.exe","1","Console","1","1 K"'
    headless = '"stm32cubeidec.exe","999","Console","1","100 K"'
    assert hf.cubeide_running(gui)
    assert not hf.cubeide_running(headless)
    assert not hf.cubeide_running('')


def _lines(t0=100.0):
    """old firmware lines, then the flash at t0, then a boot like run04."""
    old = [(t0 - 1.0, 'Loop: max=389.3 us prn_max=389.3 us over500us=0 over1ms=0 passes=17294')]
    new = [(t0 + 2.5, ''), (t0 + 2.5, '--- H753 HIL boot ---'), (t0 + 2.5, 'pclk1=120000000  pclk2=120000000'),
           (t0 + 2.5, 'Init done, entering main loop'), (t0 + 3.1, 'W1: omega=  0.00 dir= 0'),
           (t0 + 3.1, 'Loop: max=134.7 us prn_max=0.0 us over500us=0 over1ms=0 passes=21857')]
    return old + new


def test_boot_check_ok():
    b = hf.boot_check(_lines(), 100.0)
    assert b['ok'] and b['boot_t'] == 102.5 and b['loop_t'] == 103.1
    assert 'pclk1=120000000  pclk2=120000000' in b['boot_lines']
    assert b['first_loop'].startswith('Loop: max=134.7')


def test_boot_check_ignores_lines_before_flash():
    """A Loop: line of the old firmware must not count."""
    lines = [x for x in _lines() if 'Init done' not in x[1]]
    assert not hf.boot_check(lines, 100.0)['ok']


def test_boot_check_needs_boot_before_loop():
    lines = [(101.0, 'Loop: max=1.0 us'), (102.0, 'Init done, entering main loop')]
    assert not hf.boot_check(lines, 100.0)['ok']


def test_boot_check_old_boot_before_flash_does_not_count():
    lines = [(90.0, '--- H753 HIL boot ---'), (90.1, 'Init done, entering main loop'), (101.0, 'Loop: max=1 us')]
    assert not hf.boot_check(lines, 100.0)['ok']
