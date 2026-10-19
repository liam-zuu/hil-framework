# hil-framework

Khung Hardware-in-the-Loop chi phí thấp cho AGV Mecanum (luận văn thạc sĩ IE, HCMUT).

```
RPi5 (plant 1 kHz) ──SPI1──► STM32H753 (bộ chuyển tín hiệu) ──encoder──► ESP32 (controller under test)
        ▲                       │  EncoderGen, PWMCapture, UART relay        │  PID / ADRC
        └──────── torque ◄──────┘◄──────────────── PWM ──────────────────────┘
```

Firmware ESP32 nằm ở repo riêng `liam-zuu/AGV_MECANUM`.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `matlab/SIMU_M6_V3/` | Mô phỏng MATLAB bản cuối: `scripts/` (chạy `run_simulation.m`), `scenarios/` (14 scenario + `results/`), `validate/`, `docs/` |
| `matlab/archive/` | Các bản M1–M6_V2 và `progress_summary/` |
| `rpi5/` | Plant chạy thời gian thực trên RPi5 (Python): `run_hil.py`, `plant_step.py`, `protocol.py`, `trajectory_pub.py`, `pc_sender.py`, `tests/`, `docs/` |
| `stm32h7/` | Project STM32CubeIDE cho NUCLEO-H753ZI: `Core/Src` gồm `spi1_handler`, `encoder_gen`, `pwm_capture`, `uart_relay` |
| `tools/` | Logger và vẽ log qua WiFi UDP: `agv_logger.py`, `plot_log.py`, `fake_esp32.py` |
| `docs/` | `Pin_out.xlsx` (sơ đồ chân H7 / ESP32 / RPi5) |

## Chạy

```bash
# RPi5: chạy thử không cần phần cứng
python rpi5/run_hil.py --mock --steps 5000
# RPi5: phần cứng thật, quỹ đạo tròn 60 s
sudo python rpi5/run_hil.py --traj circle --speed 0.3 --radius 0.5 --duration 60
# Test RPi5 (chạy trong thư mục rpi5/ vì test import trực tiếp params, protocol...)
cd rpi5 && python -m pytest tests
```

- MATLAB: mở `matlab/SIMU_M6_V3/scripts/`, chạy `run_simulation.m`.
- STM32H7: mở STM32CubeIDE, chọn workspace là thư mục gốc repo này, import `stm32h7/`. Thư mục `stm32h7/Debug/` do IDE tự sinh khi build, không đưa lên git.
- Logger: xem `tools/README.txt`. Chạy `agv_logger.py` trước, rồi mới gửi dữ liệu.

## Trạng thái (19/10/2026)

- **18/06:** vòng kín 3 board chạy được: RPi5 → H7 SPI1 → EncoderGen → ESP32 → PID → PWM → H7 PWMCapture.
- **19/06:** H7 thêm UART relay, mở rộng frame SPI1 lên 36 byte (thêm `vx_ref`, `vy_ref`, `wz_ref`, `traj_flags`), gửi gói `0xBB` 14 byte sang ESP32. Chưa test trên phần cứng.

## Việc cần làm tiếp

1. **Cập nhật `rpi5/protocol.py` lên frame 36 byte** cho khớp `stm32h7/Core/Inc/hil_protocol.h`. Hiện `protocol.py` vẫn dùng `FRAME_SIZE = 24`, nên RPi5 và H7 đang lệch nhau.
2. Xem lại `rpi5/trajectory_pub.py`. File này gửi gói UART 18 byte thẳng sang ESP32, trong khi đường mới là RPi5 → SPI → H7 → UART 14 byte.
3. Nối lại 8 dây PWM ESP32 → H7 (bảng mapping trong session summary 18/06), verify cả 8 kênh.
4. Verify trajectory loop qua UART, rồi tune PID/ADRC trên HIL.
