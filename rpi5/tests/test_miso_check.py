"""A3 (Liam 09/11, CODE: "ok tôi đồng ý hết"; STATUS_CODE L50, L61 b): the Pi loop counts MISO frames
with a bad CRC or all 0 during the run and reports them at the end (#miso line, manifest)."""
import binascii

from params import PARAMS  # noqa: F401  (rpi5 path set by conftest)
from plant_interface import MockPlant
from hil_node import HILConfig, HILNode, MISO_SKIP_FRAMES
from protocol import FRAME_SIZE, OFS_MISO_CRC, MISO_OK, MISO_EMPTY, MISO_CRC_BAD, miso_check


def _good(seq=0):
    """A MISO v4 frame with a valid CRC (bytes 8–9 LE over 0–7 + 10–39), as the H7 sends it."""
    b = bytearray(FRAME_SIZE)
    b[0], b[3], b[16], b[17], b[25] = 0x12, 0x34, 1, seq & 0xFF, 0x5A
    crc = binascii.crc_hqx(bytes(b[:OFS_MISO_CRC] + b[OFS_MISO_CRC + 2:]), 0xFFFF)
    b[OFS_MISO_CRC], b[OFS_MISO_CRC + 1] = crc & 0xFF, crc >> 8
    return bytes(b)


def test_miso_check_good_empty_bad():
    g = _good(7)
    assert miso_check(g) == MISO_OK
    assert miso_check(bytes(FRAME_SIZE)) == MISO_EMPTY
    bad = bytearray(g)
    bad[30] ^= 0x01                                   # one bit flipped outside the CRC field
    assert miso_check(bytes(bad)) == MISO_CRC_BAD
    swapped = bytearray(g)                            # CRC written big-endian is not the H7 layout
    swapped[OFS_MISO_CRC], swapped[OFS_MISO_CRC + 1] = g[OFS_MISO_CRC + 1], g[OFS_MISO_CRC]
    assert g[OFS_MISO_CRC] == g[OFS_MISO_CRC + 1] or miso_check(bytes(swapped)) == MISO_CRC_BAD
    assert miso_check(list(g)) == MISO_OK             # spidev returns a list on the hardware path


class _Responder:
    """Fake H7 for the simulate SPI path: frame k (0-based call count) from a plan, else a good frame."""

    def __init__(self, plan):
        self.plan, self.k = plan, 0

    def __call__(self, mosi):
        k, self.k = self.k, self.k + 1
        kind = self.plan.get(k, 'good')
        if kind == 'empty':
            return bytes(FRAME_SIZE)
        if kind == 'bad':
            b = bytearray(_good(k))
            b[20] ^= 0xFF
            return bytes(b)
        return _good(k)


def _run(plan, steps=20):
    node = HILNode(MockPlant(), config=HILConfig(simulate_hw=True, max_steps=steps, period_us=1_000),
                   responder=_Responder(plan))
    node.run()
    return node


def test_node_counts_bad_and_empty_from_frame_2():
    assert MISO_SKIP_FRAMES == 2
    # frames 0, 1: leftover of the previous link (not counted); 5 empty; 7, 9 bad CRC
    node = _run({0: 'empty', 1: 'bad', 5: 'empty', 7: 'bad', 9: 'bad'})
    s = node.miso_summary()
    assert s == dict(checked=18, crc_bad=2, empty=1, first_crc_bad=7, first_empty=5)


def test_node_clean_run_reports_zero():
    s = _run({}).miso_summary()
    assert s == dict(checked=18, crc_bad=0, empty=0, first_crc_bad=None, first_empty=None)


def test_shutdown_logs_miso_line(caplog):
    caplog.set_level('INFO', logger='hil.node')
    _run({3: 'empty', 4: 'bad'})
    lines = [r.getMessage() for r in caplog.records]
    assert '#miso,checked=18,crc_bad=1,empty=1,first_crc_bad=4,first_empty=3' in lines
    assert any(r.levelname == 'WARNING' and r.getMessage().startswith('MISO:') for r in caplog.records)


def test_frame_log_manifest_has_miso_check(tmp_path):
    import json
    import numpy as np
    node = HILNode(MockPlant(), config=HILConfig(simulate_hw=True, max_steps=10, period_us=1_000,
                                                 frame_log=str(tmp_path / 'f.npz')),
                   responder=_Responder({6: 'bad'}))
    node.run()
    m = json.loads(str(np.load(tmp_path / 'f.npz')['manifest']))
    assert m['miso_check'] == dict(checked=8, crc_bad=1, empty=0, first_crc_bad=6, first_empty=None)
