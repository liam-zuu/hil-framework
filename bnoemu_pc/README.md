# bnoemu_pc: G1 test of the BNO085 emulator on the PC

G1 milestone of IMU option A (spec `docs/2026-10-22_Spec_G0_Gia_lap_BNO085.md`, §8):
the **real SH-2 library of the ESP32 firmware** (`AGV_MECANUM`,
`components/esp32_BNO08x/SH2/`: `sh2.c`, `shtp.c`, `sh2_SensorValue.c`,
`sh2_util.c`, compiled unchanged) talks to the emulator core
`stm32h7/Core/Src/bno_emu.c`, the same file the H7 will run at G2.

```
python bnoemu_pc/build_bnoemu.py        # --fw <AGV_FIRMWARE> if not ../AGV_FIRMWARE
python -m pytest bnoemu_pc/tests
bnoemu_pc/build/bnoemu_harness.exe run 10000 600     # one scenario, key=value output
```

## What the harness models

- Fake HAL copied from `BNO08xSH2HAL.cpp`: read = wait INT, CS low, 4-byte
  header, `len` more bytes, CS high; write = wait INT, packet on MOSI, MISO
  ignored. INT wait 500 ms, then RST pulse (100 ms low, 100 ms wait) and the
  transaction fails. An INT falling edge sets one event bit, cleared by the wait.
- Init as `BNO08x::init_sh2_HAL` + `imu_init`: RST pulse, `sh2_open`,
  `sh2_getProdIds`, sensor callback, Set Feature for 0x01, 0x02, 0x08 with a
  30 ms service period after each. A reset-complete event after init
  re-enables the reports (`re_enable_reports`).
- Plant samples at 1 kHz with `imu_seq` (frozen plant = same seq).

## Limits (do not read more into a PASS)

- **Virtual time, one thread.** The driver's init call and its service task
  run in one loop; SPI is 3 MHz plus 20 µs per transaction (estimate). Real
  ESP32 task timing, ISR latency and SPI DMA come only at G2.
- The emulator's boot delay (20 ms) and INT retry gap (1 ms) are proposals;
  the real chip has not been recorded yet (G0 v1 §7, QĐ 26).
- Product IDs are emulator values (part numbers 90000000–90000003), not a
  real BNO085, so logs show which IMU answered.

## Scenarios and pass criteria ([ĐỀ XUẤT], G0 v1 §8)

| Scenario | Pass when |
|---|---|
| `boot <period> 20` | 20 boots in a row: reset complete in `sh2_open`, 4 product IDs, all 3 reports within 1 s, no host INT timeout |
| `run <period> 600` | 10 virtual minutes: mean report period within 1 %, raw values on the wire equal the emulator's, quantization ≤ 0.5 LSB (Q8/Q9/Q14, after saturation), no lost report, no host INT timeout |
| `collision <period> 10` | host writes while a packet waits: every Set Feature answered, reports keep their period |
| `rst_mid <period>` | RST pulse while running: one reset event, all reports back within 1 s |
| `silent <period>` | host does not answer INT for 50 ms: emulator releases INT and retries, reports resume |
| `stale <period>` | plant frozen: no report after 3 periods (+1 in flight), reports resume when the plant moves |
