# fwsim: firmware-in-the-loop on the PC

Runs the ESP32 firmware's own C code (`AGV_MECANUM`: `encoder.c`, `motor.c`,
`kinematics.c`, `pid.c`, `adrc.c`, `vel_ctrl.c`, compiled unchanged) against the
RPi5 plant (`rpi5/plant_step.py`, DC motor model with S01 parameters). Used for
DOE/GA tuning of the firmware gains before HIL (decision 21/10/2026, STATUS.md).

```bash
python fwsim/build_fwsim.py                      # gcc (Strawberry) -> fwsim/build/fwsim.dll
python -m pytest fwsim/tests -q
python fwsim/fwsim.py --traj steps --mode ADRC   # one run; --out <dir> saves log.csv, metrics.json, meta.yaml
python fwsim/tune_ga.py --mode ADRC --pop 24 --gen 20 --out <dir>
python fwsim/s02_groupA_sim.py --controller ADRC --out <dir>   # spec S02 group A, ESP32 log format
```

Plant and IMU options, the same in `fwsim.py`, `tune_ga.py` and `s02_groupA_sim.py`
(28/10): the firmware gets BNO085 reports by default (QD 23; `--imu-rate 20|100`,
`--no-imu` only to re-run pre-28/10 runs); the plant ground model is `params.py`'s
default, slip method A `traction` (`--ground rigid`, `--mu <mu_hd>`).

Rebuild the DLL after every firmware change. `fwsim_build.json` records the
firmware commit and dirty state; every run copies it into `meta.yaml`.

| Modelled as on the bench | Not modelled |
|---|---|
| plant 1 kHz, firmware tick 100 Hz, `encoder_get_rpm` over 10 ms | BNO085 report latency, IMU noise (until K1) |
| IMU reports from `rpi5/imu_model.py` into the firmware's `imu_data_t` (fusion, heading loop) | |
| 1320 counts/rev, PCNT reset at ±limit like ESP-IDF 5.5.3 (`--pcnt-mode wrap16` to compare) | RC, ultrasonic interlock, modes |
| `motor_set` → MCPWM compare → duty per GPIO, `rpwm − lpwm` as H7 reads it, plant `motor_polarity` | SPI/UART latency (`--io-delay-ms` adds a pure delay) |
| setpoint held at 50 Hz (trajectory publisher) | H7 timing jitter |

Glue repeated from `main.c` (needs FreeRTOS, so not compiled): the
encoder_task → control_task order inside one tick and the `dt_actual` guard;
see `fw_sim.c`.

Fault injection batch 1 (L82): `fault_run.py` runs the same schedule file as
`rpi5/run_hil.py --faults` (`rpi5/fault_schedule.py`: enc_loss, imu_loss,
imu_delay, gyro_scale; windows in ms from t = 0) by wrapping `fwsim.plant_step`,
the IMU noise object, the IMU hook and `fw.tick` while it runs, so `fwsim.py`
(S04 blob lock) is unchanged. Compare a fault run only with a `--baseline` run
(same wrapper, no row, same noise setting): with IMU noise the IMU values go
through `firmware_view`. An imu_delay / gyro_scale row needs the IMU noise on
(`--imu-noise replay`, the default); with `--imu-noise off` it is refused, since
it would change the IMU path of the whole run against its baseline.

```
python fwsim/fault_run.py --task A-W1 --mode PID --baseline --out <dir>
python fwsim/fault_run.py --task A-W1 --mode PID --faults <schedule.csv> --out <dir>
```
