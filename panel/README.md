# panel/ — HIL box panel, level M2

ESP32-S3 N16R8 + ST7796 3.5" SPI (480×320) + 4 buttons (NAV, SELECT, START, STOP).
The panel only draws and forwards buttons; the RPi5 daemon owns the run (QĐ 17, QĐ 22).

| Path | What |
|---|---|
| `lib/panel_proto/` | panel side of the protocol v6 in plain C99 (mirror of `rpi5/panel_proto.py`) |
| `lib/panel_app/` | buttons → commands, screen choice, result generation, plain C (tested on the PC) |
| `src/` | Arduino firmware: `main.cpp` (link task every 2 ms + drawing loop), `ui.cpp` (screens), `buttons.cpp`, `pins.h` (checked against the WeAct ESP32-S3-A schematic, not yet run) |
| `test_host/` | PC tests: C vs Python golden bytes, the two-end Bench scenarios with the C client, seeded side-by-side runs, the fake Pi |
| `tools/fake_pi.py` | bench "Pi" over a serial port with a FAKE run model |

```
python -m pytest panel/test_host -q            # from the repo root; needs gcc on PATH
cd panel && pio run -e panel_s3_bench          # protocol on the "UART" USB-C port (fake Pi)
cd panel && pio run -e panel_s3                # protocol on UART1 to the RPi5
```

Not done here: the daemon side on the RPi5 (`hil_daemon`, stage E) and S06 T9 (stage F).
Wiring and bench steps (Vietnamese): `00_OVERVIEW/Setup_Guides/2026-11-04_Huong_dan_Panel_M2_noi_day_nap.md`
in the thesis repo.
