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

1. ~~Cập nhật `rpi5/protocol.py` lên frame 36 byte~~ (xong 19/10). RPi5 gửi setpoint vận tốc thân xe `(vx, vy, wz)` trong frame SPI, H7 relay sang ESP32. `tests/test_protocol.py` so hằng số với `hil_protocol.h` để phát hiện lệch.
2. **UART relay không chặn (nhánh `h7-uart-fifo`, chưa build trên board):** `UartRelay_Send()` ghi 14 byte thẳng vào FIFO 16 byte của USART2 thay vì chờ `HAL_UART_Transmit()` (trước đây chặn khoảng 1,2 ms). Relay chỉ chạy khi có đủ cả `VALID` và `NEW`. Đã biên dịch thử bằng STM32CubeCLT, chưa nạp. Cách kiểm tra:
   1. CubeIDE: Build, rồi nạp. Thư mục `Debug/` sẽ tự sinh lại.
   2. Mở cổng ST-Link VCP (USART3, 115200): mỗi 0,5 s có dòng `Relay: sent=… dropped=…`.
   3. Chạy `sudo python run_hil.py --traj circle --speed 0.3 --radius 0.5 --duration 30`: `sent` tăng khoảng 25 sau mỗi lần in, `dropped` giữ 0, `traj wz` khoảng 0.60.
   4. Logic analyzer: PD5 (USART2 TX) có cụm 14 byte mỗi 20 ms. PE1 (toggle mỗi frame SPI) đều đặn, không có khoảng hở 1,2 ms. Đây là bằng chứng relay không chặn vòng chính.
   5. Đạt thì gộp nhánh `h7-uart-fifo` vào `main`.
3. Nối lại 8 dây PWM ESP32 → H7 (bảng mapping trong session summary 18/06), verify cả 8 kênh.
4. Verify trajectory loop: chạy `sudo python run_hil.py --traj circle --speed 0.3 --radius 0.5 --duration 30`, xem ESP32 có nhận gói 0xBB qua UART không.
5. Sau khi link ổn định, tăng `--spi-speed` lên 2–4 MHz. Frame 36 byte ở 1 MHz mất 288 µs trong chu kỳ 1 ms.
6. Tune PID/ADRC trên HIL.
