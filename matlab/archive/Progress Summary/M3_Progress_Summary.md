# M3 — Signal Conditioning (Complete)

---

## Current Status
- Active milestone: M3 (complete)
- Completed: M1 (1.1→1.9), M2 (2.1→2.5), M3 (3.1→3.7)
- Blocked: none
- Next: M4 — Controller

---

## M3 — Đã làm

### Files thay đổi từ M2 → M3 (5 replaced + 2 updated + 1 new)

#### 1. `nucleoh7/spi_interface.m` — THAY HOÀN TOÀN
- Stub passthrough → Fixed-point N-bit quantization
- Mô hình: float → clamp to ±range → round to 2^N levels → reconstruct
- N = params.spi.float_bits (16-bit)
- Uplink (torque): 4 channels, shared full-scale range ±1.0 N·m
- Downlink (states): 10 channels, mỗi state có full-scale range riêng
- LSB torque ≈ 3.05e-5 N·m, LSB state tùy range (1.53e-4 → 1.22e-3)

#### 2. `nucleoh7/encoder_pulse_gen.m` — THAY HOÀN TOÀN
- Stub round() → Fractional accumulator + quantization + noise
- Dùng MATLAB `persistent` variable giữ phần lẻ giữa timesteps
- Critical ở low speed: ω=0.5 rad/s, dt=0.001 → 0.08 counts/step → không có accumulator thì mất hết
- Noise: Gaussian σ = params.enc_noise_sigma (0.02 counts)
- Output: integer counts (signed)
- Cần `clear encoder_pulse_gen` giữa các lần chạy simulation

#### 3. `nucleoh7/imu_packet_enc.m` — THAY HOÀN TOÀN
- Stub passthrough → ADC quantization + UART packet + XOR checksum
- ADC: 16-bit, accel range ±4g (39.24 m/s²), gyro range ±2000 deg/s (34.91 rad/s)
- Packet struct: header (0xAA), accel_raw (int32), gyro_raw (int32), scale factors, checksum
- Checksum = XOR of 6 payload codes → imu_reader verify được corruption
- LSB accel ≈ 1.20e-3 m/s², LSB gyro ≈ 1.07e-3 rad/s

#### 4. `nucleoh7/pwm_capture.m` — THAY HOÀN TOÀN
- Stub linear → Deadband + PWM resolution + capture jitter
- Step 1: Gaussian jitter (σ = 0.001 fraction of duty cycle)
- Step 2: Quantize to 10-bit (1024 levels)
- Step 3: Deadband |pwm| < 0.02 → tau = 0
- Step 4: Linear map pwm × tau_max

#### 5. `nucleoh7/gpio_sync.m` — THAY HOÀN TOÀN
- Stub always-true → Timing jitter model
- Mỗi cluster có independent Gaussian jitter (σ = 5μs)
- Sync fail nếu max jitter > timeout (50μs)
- 100% pass rate ở nominal (10σ margin)

#### 6. `esp32/imu_reader.m` — UPDATED
- Cập nhật để decode packet format mới từ imu_packet_enc
- Thêm header check (0xAA) + XOR checksum verification
- Decode: float = int32_code × scale_factor
- Return valid=false nếu header sai hoặc checksum mismatch

#### 7. `scripts/params_mecanum.m` — BỔ SUNG FIELDS
Thêm params mới cho M3:
- `params.spi.tau_range = 1.0` (full-scale torque range)
- `params.spi.state_ranges = [5;5;pi;3;3;10;40;40;40;40]` (per-state range)
- `params.imu_adc_bits = 16`
- `params.imu_accel_range = 4*9.81` (±4g)
- `params.imu_gyro_range = 2000*pi/180` (±2000 deg/s)
- `params.pwm_jitter_sigma = 0.001`
- `params.sync_jitter_us = 5`
- `params.sync_timeout_us = 50`

#### 8. `scripts/test_m3_signal_conditioning.m` — MỚI
- 7 test groups, 15 test cases
- Cover: SPI quantization, encoder accumulator, IMU encode/decode, PWM deadband, GPIO sync, full pipeline round-trip, encoder SNR

#### 9. `scripts/run_simulation.m` — SỬA NHỎ
- Thêm `clear encoder_pulse_gen` ở đầu file để reset persistent accumulator

---

## Test results

### Unit tests (test_m3_signal_conditioning.m): 15/15 PASS

| Test | Kết quả |
|------|---------|
| SPI uplink quantization | Error ≤ 0.5 LSB (max 1.39e-5, LSB=3.05e-5) |
| SPI downlink 10 states | Tất cả within ±0.5 LSB |
| SPI clipping | Values clamped đúng |
| Encoder constant speed | Total counts within ±3 of expected |
| Encoder zero speed | Total = [0,0,0,0] |
| Encoder low speed accumulator | Expected ~8.1, got [8,8,8,8] |
| IMU round-trip | Accel/gyro error ≤ 0.5 LSB |
| IMU checksum corruption | Detected (valid=false) |
| IMU ADC clipping | Clamped to ±39.2 m/s² |
| PWM linear + deadband | Exact match (zero jitter test) |
| PWM deadband edge | 0.019→0, 0.021→nonzero |
| PWM saturation | Clamped to ±0.50 N·m |
| GPIO sync normal | 100% pass (10000 trials) |
| GPIO cluster not done | sync_ok=false |
| Full pipeline round-trip | Packet valid, data flows correctly |

### Encoder SNR (1000 steps, statistical):

| Wheel speed | Mean | Std | SNR |
|-------------|------|-----|-----|
| ω=5 rad/s | 4.99 | 2.39 | 6.4 dB |
| ω=10 rad/s | 10.00 | 2.97 | 10.6 dB |
| ω=20 rad/s | 20.00 | 2.69 | 17.4 dB |
| ω=30 rad/s | 30.00 | 1.93 | 23.8 dB |

→ Quantization noise dominant ở low speed. M4 encoder_reader sẽ cần filtering.

### End-to-end (run_simulation.m): PASS

| Metric | M2 | M3 | Note |
|--------|----|----|------|
| RMS position error | ~1178 mm | ~1091 mm | Khác do random seed, cùng order |
| NaN/Inf | 0 | 0 | |
| Sync failures | 0/10000 | 0/10000 | |
| Trajectory shape | Circle, oversized | Circle, oversized | Expected — P-only controller saturate |

Trajectory gần như không đổi so với M2 vì controller P-only saturate 100% thời gian (tau_cmd >> tau_max). Signal conditioning effects sẽ visible khi M4 implement full PID/ADRC hoạt động trong vùng linear.

---

## Key Design Decisions (tham khảo cho M4+)

1. **Persistent accumulator trong encoder_pulse_gen**: Giải quyết vấn đề low-speed quantization. Trade-off: cần `clear` giữa các simulation runs.
2. **Per-state SPI range**: Mỗi state có full-scale riêng (position ±5m, wheel speed ±40 rad/s) → tối ưu resolution cho từng channel.
3. **XOR checksum**: Đơn giản, detect single-bit errors. Đủ cho simulation. Real UART sẽ dùng CRC.
4. **PWM deadband = 0.02**: ~2% duty cycle. Realistic cho H-bridge driver. Controller M4 cần aware deadband này.
5. **imu_reader updated sớm**: Tuy là module M4 (ESP32), nhưng packet format thay đổi nên update luôn để run_simulation không break. Logic vẫn stub (chưa filtering/outlier rejection).

---

## Còn lại stub (chờ M4)

### ESP32 (M4 — Controller):
- encoder_reader.m — basic decode, chưa filtering (SNR thấp ở low speed)
- imu_reader.m — decode + checksum OK, chưa filtering/outlier rejection
- pid_controller.m — P-only, chưa full PID (Ki, Kd)
- adrc_controller.m — P-only placeholder, chưa ESO
- pwm_output.m — basic saturation, chưa deadband compensation
- slip_detector.m — always false (optional, M6)
