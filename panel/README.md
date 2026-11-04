# panel/ — HIL box panel, level M2

ESP32-S3 N16R8 + ST7796 3.5" SPI (480×320) + 4 buttons (NAV, SELECT, START, STOP).
The panel only draws and forwards buttons; the RPi5 daemon owns the run (QĐ 17, QĐ 22).

| Path | What |
|---|---|
| `lib/panel_proto/` | panel side of the protocol v6 in plain C99 (mirror of `rpi5/panel_proto.py`) |
| `src/` | Arduino firmware: `main.cpp` (link, buttons → commands), `ui.cpp` (screens), `buttons.cpp`, `pins.h` (proposal, not confirmed) |
| `test_host/` | PC tests: C vs Python golden bytes, the two-end Bench scenarios with the C client, seeded side-by-side runs, the fake Pi |
| `tools/fake_pi.py` | bench "Pi" over a serial port with a FAKE run model |

```
python -m pytest panel/test_host -q            # from the repo root; needs gcc on PATH
cd panel && pio run -e panel_s3_usb            # protocol on USB (fake Pi)
cd panel && pio run -e panel_s3                # protocol on UART1 to the RPi5
```

Not done here: the daemon side on the RPi5 (`hil_daemon`, stage E) and S06 T9 (stage F).
Wiring and bench steps (Vietnamese): `00_OVERVIEW/Setup_Guides/2026-11-04_Huong_dan_Panel_M2_noi_day_nap.md`
in the thesis repo.
