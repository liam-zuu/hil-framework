"""H7 UART relay packet (stm32h7/Core/Src/uart_relay.c, UartRelay_Pack) built on the PC
with the host gcc, compared byte for byte with rpi5/pc_sender.py, the Wi-Fi sender the
ESP32 traj_receiver.c already accepts (same parser, XOR over bytes 0..12).
Until 28/10/2026 the H7 XOR-ed bytes 1..12 only, so every packet differed by 0xBB.
Run: python -m pytest tools/tests  (needs gcc on PATH, as fwsim)"""
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'rpi5'))
import pc_sender  # noqa: E402

H7 = os.path.join(ROOT, 'stm32h7')
pytestmark = pytest.mark.skipif(shutil.which('gcc') is None, reason='gcc not on PATH')

CASES = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (-0.25, 0.1, 0.6), (0.123456, -0.654321, -1.5)]


@pytest.fixture(scope='module')
def relay_pack(tmp_path_factory):
    """Copy the unchanged H7 files next to the usart.h stub: a quoted #include looks in the
    including file's own folder first, so Core/Inc/usart.h (needs the HAL) would win otherwise."""
    d = tmp_path_factory.mktemp('relay')
    for src in (os.path.join(H7, 'Core', 'Src', 'uart_relay.c'), os.path.join(H7, 'Core', 'Inc', 'uart_relay.h'),
                os.path.join(H7, 'Core', 'Inc', 'hil_protocol.h'), os.path.join(H7, 'pc_tests', 'usart.h'),
                os.path.join(H7, 'pc_tests', 'relay_pack_main.c')):
        shutil.copy(src, d)
    exe = str(d / 'relay_pack.exe')
    cmd = ['gcc', '-std=c11', '-Wall', '-Werror', '-I', str(d),
           str(d / 'relay_pack_main.c'), str(d / 'uart_relay.c'), '-o', exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


def h7_packet(exe, vx, vy, wz):
    out = subprocess.run([exe, repr(vx), repr(vy), repr(wz)], check=True, capture_output=True, text=True)
    return bytes.fromhex(out.stdout.strip())


@pytest.mark.parametrize('sp', CASES)
def test_h7_packet_equals_wifi_packet(relay_pack, sp):
    assert h7_packet(relay_pack, *sp) == pc_sender.pack_packet(*sp)


@pytest.mark.parametrize('sp', CASES)
def test_esp32_parser_rule(relay_pack, sp):
    """traj_receiver.c _parse_packet: header 0xBB and XOR of bytes 0..12 == byte 13."""
    pkt = h7_packet(relay_pack, *sp)
    chk = 0
    for b in pkt[:13]:
        chk ^= b
    assert len(pkt) == 14 and pkt[0] == 0xBB and chk == pkt[13]
