"""Fault injection batch 2, iBUS loss, Pi side (design 2026-11-15_Thiet_ke_bom_loi_dot2_v1.md sections 4.2, 5;
Liam 15/11 Q2 "Chi ban HIL", 16/11 "Lan luot ca 3"): the node sends byte 20 bit 4 in the window only, the
true frame keeps it clear, and run_hil.py refuses ibus_loss where it would mean nothing (bench without the H7
part, --swa off)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import fault_schedule as fs
from hil_node import HILConfig, HILNode
from mecanum_plant import MecanumPlant
from protocol import CMD_IBUS_OFF, CMD_SWA_ON, FRAME_SIZE, OFS_FAULT

HEAD = fs.MAGIC + '\nid,fault,target,start_ms,end_ms,params\n'
DUTY = bytes([0x10, 0x00] * 4) + bytes(FRAME_SIZE - 8)


def schedule(body, run_ms=400):
    rows = fs.parse(HEAD + body)
    fs.check(rows, run_ms, 'plant', 'traction')
    return fs.Schedule(tuple(rows), 'x.csv', 'f' * 64)


def open_loop(tmp_path, body, steps=400, swa='on'):
    plant = MecanumPlant(dt=0.001, seed=0)
    plant.reset([0, 0, 0, 0.2, 0, 0.8, 10, 12, 14, 16])
    p = str(tmp_path / 'f.npz')
    cfg = HILConfig(simulate_hw=True, max_steps=steps, period_us=1_000, frame_log=p, faults=schedule(body),
                    swa_mode=swa)
    node = HILNode(plant, config=cfg)
    node._spi.transfer = lambda mosi: DUTY
    node.run()
    return np.load(p), node


def test_node_sends_bit4_in_the_window_only(tmp_path):
    f, node = open_loop(tmp_path, 'I1,ibus_loss,ibus,100,250,\n')
    sent, true = f['mosi'], f['mosi_true']
    bit = (sent[:, OFS_FAULT] & CMD_IBUS_OFF) != 0
    assert np.flatnonzero(bit).tolist() == list(range(100, 250))
    assert not np.any(true[:, OFS_FAULT] & CMD_IBUS_OFF)                # the truth never has it
    assert np.all(sent[:, OFS_FAULT] & CMD_SWA_ON) and np.all(true[:, OFS_FAULT] & CMD_SWA_ON)
    other = np.ones(FRAME_SIZE, dtype=bool)
    other[OFS_FAULT] = False
    assert np.array_equal(sent[:, other], true[:, other])               # no other byte changes
    assert np.array_equal(sent[:, OFS_FAULT] & ~np.uint8(CMD_IBUS_OFF), true[:, OFS_FAULT])
    man = json.loads(str(f['manifest']))
    assert man['faults']['events'] == [[100, 'I1', 1, 'on'], [250, 'I1', 0, 'off']]
    assert man['plant'] == 'MecanumPlant' and type(node.plant) is MecanumPlant   # no plant row: no FaultPlant


# ── run_hil.py --faults ─────────────────────────────────────────────────────
def _args(tmp_path, body, **kw):
    csv_ = tmp_path / 's.csv'
    csv_.write_text(HEAD + body)
    a = dict(faults=str(csv_), frame_log='f.npz', step_log='s.csv', mock=False, steps=1000, ground=None, mu=None)
    a.update(kw)
    return SimpleNamespace(**a)


def _cfg(swa):
    c = HILConfig(simulate_hw=True)
    c.swa_mode = swa
    return c


@pytest.mark.parametrize('mock,swa,h7,msg', [
    (False, 'phase', False, 'H7 part'),             # bench, H7 part not merged: the bit would change nothing
    (False, 'on', False, 'H7 part'),
    (True, 'off', False, 'swa off'),                # motors never on: nothing to lose
    (False, 'off', True, 'swa off'),
])
def test_run_hil_refuses_ibus_loss(tmp_path, monkeypatch, mock, swa, h7, msg):
    import run_hil
    monkeypatch.setattr(fs, 'IBUS_LOSS_H7', h7)
    with pytest.raises(SystemExit) as e:
        run_hil.load_faults(_args(tmp_path, 'I1,ibus_loss,ibus,1,9,\n', mock=mock), 0, _cfg(swa), False)
    assert msg in str(e.value)


@pytest.mark.parametrize('mock,swa,h7', [(True, 'phase', False), (True, 'on', False), (False, 'phase', True)])
def test_run_hil_accepts_ibus_loss(tmp_path, monkeypatch, mock, swa, h7):
    import run_hil
    monkeypatch.setattr(fs, 'IBUS_LOSS_H7', h7)
    s = run_hil.load_faults(_args(tmp_path, 'I1,ibus_loss,ibus,1,9,\nE1,enc_loss,FL,1,9,\n', mock=mock), 0,
                            _cfg(swa), False)
    assert [r.fault for r in s.rows] == ['ibus_loss', 'enc_loss']


def test_run_hil_other_rows_unchanged_with_swa_off(tmp_path):
    import run_hil
    s = run_hil.load_faults(_args(tmp_path, 'E1,enc_loss,FL,1,9,\n'), 0, _cfg('off'), False)   # bench, no ibus
    assert [r.fault for r in s.rows] == ['enc_loss']
