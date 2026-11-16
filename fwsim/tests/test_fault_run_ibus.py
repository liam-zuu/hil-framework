"""Fault injection batch 2 in fwsim: ibus_loss is refused, by the API and by the CLI, before any firmware runs
(design 2026-11-15 section 6; Liam 15/11 Q2 "Chi ban HIL": fwsim does not run the RC / iBUS logic of main.c).
Needs no fwsim.dll: the refusal comes before the firmware is used."""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
import fault_run                                      # noqa: E402
import fault_schedule as fs                           # noqa: E402
from params import MecanumParams                      # noqa: E402

HEAD = fs.MAGIC + '\nid,fault,target,start_ms,end_ms,params\n'


def test_refuse_bench_only():
    fault_run.refuse_bench_only(fs.parse(HEAD + 'E1,enc_loss,FL,1,9,\nL1,imu_loss,imu,1,9,\n'))
    with pytest.raises(fs.ScheduleError, match='bench only'):
        fault_run.refuse_bench_only(fs.parse(HEAD + 'E1,enc_loss,FL,1,9,\nI1,ibus_loss,ibus,3,5,\n'))


def test_api_refuses_before_the_firmware_and_leaves_no_wrapper():
    rows = fs.parse(HEAD + 'I1,ibus_loss,ibus,100,200,\n')
    with pytest.raises(fs.ScheduleError, match='bench only'):
        fault_run.FwsimFaults(fs.Schedule(tuple(rows))).run(None, lambda t: (0.0, 0.0, 0.0), 1.0, MecanumParams(),
                                                            imu=dict(rate_hz=20.0, noise='replay', noise_seed=0))
    assert fault_run.fwsim.plant_step is fault_run._REAL_PLANT_STEP


def test_cli_refuses(tmp_path):
    p = tmp_path / 's.csv'
    p.write_text(HEAD + 'I1,ibus_loss,ibus,3000,4000,\n')
    r = subprocess.run([sys.executable, os.path.join(HERE, '..', 'fault_run.py'), '--task', 'A-W1', '--mode', 'PID',
                        '--faults', str(p), '--out', str(tmp_path / 'out')], capture_output=True, text=True,
                       timeout=300)
    assert r.returncode != 0 and 'bench only' in r.stderr
    assert not (tmp_path / 'out').exists()
