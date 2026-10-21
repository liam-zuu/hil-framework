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
```

Rebuild the DLL after every firmware change. `fwsim_build.json` records the
firmware commit and dirty state; every run copies it into `meta.yaml`.

| Modelled as on the bench | Not modelled |
|---|---|
| plant 1 kHz, firmware tick 100 Hz, `encoder_get_rpm` over 10 ms | IMU (firmware only logs it) |
| 1320 counts/rev, PCNT reset at ±limit like ESP-IDF 5.5.3 (`--pcnt-mode wrap16` to compare) | RC, ultrasonic interlock, modes |
| `motor_set` → MCPWM compare → duty per GPIO, `rpwm − lpwm` as H7 reads it, plant `motor_polarity` | SPI/UART latency (`--io-delay-ms` adds a pure delay) |
| setpoint held at 50 Hz (trajectory publisher) | H7 timing jitter |

Glue repeated from `main.c` (needs FreeRTOS, so not compiled): the
encoder_task → control_task order inside one tick and the `dt_actual` guard;
see `fw_sim.c`.
