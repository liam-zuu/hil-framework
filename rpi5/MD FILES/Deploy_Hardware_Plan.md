# Hardware Deploy Plan — HIL Mecanum AGV

Tài liệu này tóm tắt toàn bộ quá trình chuyển đổi code MATLAB sang firmware/code thật
cho 3 board trong kiến trúc HIL, theo thứ tự deploy hợp lý.

---

## Tổng quan kiến trúc

```
RPi5  (Linux + Python)          H7  (STM32, C)          ESP32  (C++)
─────────────────────           ──────────────           ────────────
plant_step.py                   spi_interface.c          encoder_reader.cpp
imu_model.py          ←─SPI─→  encoder_pulse_gen.c  ←─  imu_reader.cpp
state_manager.py                imu_packet_enc.c     ─→  pid_controller.cpp
                                pwm_capture.c            adrc_controller.cpp
                                gpio_sync.c              pwm_output.cpp
                                                         pose_estimator.cpp
                                                         position_controller.cpp
```

**Luồng data mỗi 1ms (1kHz):**
```
ESP32 tính τ_cmd
  → PWM → H7 capture → τ
  → H7 gửi τ lên RPi5 qua SPI
  → RPi5 chạy plant_step → state x[k]
  → RPi5 gửi x[k] xuống H7 qua SPI
  → H7 gen encoder pulses + IMU packet → ESP32
  → ESP32 đọc sensor → tính τ tiếp theo
```

---

## Module 1 — RPi5 (Plant Model)

### Ngôn ngữ: Python 3 + NumPy
**Lý do:** Gần MATLAB nhất, port line-by-line ít rủi ro, NumPy đủ nhanh cho 1kHz.

### Files MATLAB cần port → Python

| MATLAB | Python | Trạng thái |
|--------|--------|-----------|
| `params_mecanum.m` | `params_mecanum.py` | ✅ Hoàn thành |
| `plant_step.m` | `plant_step.py` | ✅ Hoàn thành |
| `imu_model.m` | `imu_model.py` | ✅ Hoàn thành |
| `state_manager.m` | `state_manager.py` | ✅ Hoàn thành |
| _(không có MATLAB tương đương)_ | `main_loop.py` | ✅ Hoàn thành |

### Lưu ý port quan trọng

- **Indexing:** MATLAB 1-based → Python 0-based. `x(7:10)` → `x[6:10]`, `x(3)` → `x[2]`
- **Matrix multiply:** MATLAB `A * B` (ma trận) → Python `A @ B`
- **Column vector:** MATLAB `[1;2;3]` → Python `np.array([1,2,3])` (1D, không cần reshape)
- **`persistent` variable trong MATLAB** → class attribute hoặc closure trong Python
- **`state_manager`** từ switch/case function → class `StateManager` với method `update()`

### Verification: 24/24 tests PASS

Chạy `python3 test_plant_python.py` — toàn bộ reference values lấy từ
`Plant_Validation_Summary.md` (49/49 MATLAB tests), không cần MATLAB runtime.

| Test group | Nội dung | Pass |
|-----------|---------|------|
| Group 1 (8) | M_eff symmetric/PD/eigenvalues, H_fwd×H_inv=I, kinematics spot-check | 8/8 |
| Group 2 (7) | Zero torque, forward dynamics, omega clamp, SS omega=25 rad/s | 7/7 |
| Group 3 (3) | IMU init, az≈g, no NaN | 3/3 |
| Group 4 (6) | StateManager init/update/history | 6/6 |

---

### Giai đoạn 1A — Port code (PC, không cần hardware) ✅ DONE

**Việc đã làm:**
- Port 4 file `.m` → `.py` với full docstring và inline comments
- Viết `test_plant_python.py` với 24 test cases, tất cả PASS

**Cách chạy kiểm tra:**
```bash
python3 test_plant_python.py
# Expected: 24/24 PASS
```

---

### Giai đoạn 1B — Real-time timing loop (cần RPi5) ⏳ PENDING — chờ mang Pi về

**Mục tiêu:** Xác nhận loop 1kHz chạy được trên Pi thật với jitter < 500µs.

**File:** `main_loop.py` — đã viết đầy đủ, gồm:
- `SCHED_FIFO` priority 99 (cần `sudo`)
- CPU affinity pin vào core 3
- Timing loop dùng `time.monotonic()` — không drift
- `JitterMonitor` tự in report mỗi 1 giây
- SPI stub: `spi_receive_tau()` trả τ cố định, `spi_send_state()` là `pass`

**Cách chạy trên Pi:**
```bash
# Bước 1: Copy files lên Pi
scp params_mecanum.py plant_step.py imu_model.py \
    state_manager.py main_loop.py pi@<IP_PI>:~/plant/

# Bước 2: Cài NumPy (nếu chưa có)
sudo apt install python3-numpy -y

# Bước 3: Verify port đúng
cd ~/plant && python3 test_plant_python.py

# Bước 4: Chạy real-time loop (cần sudo cho SCHED_FIFO)
sudo python3 main_loop.py --tau 0.05 --duration 5

# Bước 5: Đọc jitter report
# ✓ max_ever < 500µs  → đủ tốt, chuyển sang 1C
# ⚠ max_ever < 2ms   → chấp nhận, thêm isolcpus
# ✗ max_ever > 2ms   → cần tối ưu (xem bên dưới)
```

**Nếu jitter > 2ms — tối ưu theo thứ tự:**
```
1. Thêm vào /boot/firmware/cmdline.txt:
       isolcpus=3 rcu_nocbs=3 nohz_full=3
   → reboot

2. Tắt WiFi:  sudo rfkill block wifi
3. Tắt BT:    sudo rfkill block bluetooth
4. Verify chạy với sudo (SCHED_FIFO cần CAP_SYS_NICE)
```

---

### Giai đoạn 1C — SPI thật với H7 ⏳ PENDING — chờ có H7

**Mục tiêu:** Thay SPI stub bằng `spidev` thật.

**Chỉ cần sửa 2 hàm trong `main_loop.py`:**

```python
# Thêm import
import spidev

# Khởi tạo (ngoài loop)
spi = spidev.SpiDev()
spi.open(0, 0)           # bus=0, device=0 (CE0)
spi.max_speed_hz = 1_000_000
spi.mode = 0

# Thay spi_receive_tau():
def spi_receive_tau(stub_tau):
    raw = spi.xfer2([0x00] * 16)        # 4 float × 4 bytes
    tau = np.frombuffer(bytes(raw), dtype=np.float32).astype(float)
    return np.clip(tau, -params['tau_max'], params['tau_max'])

# Thay spi_send_state():
def spi_send_state(x, accel, gyro):
    payload = x.astype(np.float32).tobytes()   # 10 × 4 = 40 bytes
    spi.xfer2(list(payload))
```

**Lưu ý:** Format bytes (float32 vs fixed-point, byte order) phải khớp với
H7 firmware — xác nhận sau khi có H7 code.

---

## Module 2 — Nucleo H7 (Signal Conditioning)

### Ngôn ngữ: C (STM32 HAL, CubeMX)
**Lý do:** Timing hardware thật (SPI DMA, timer interrupt, PWM) cần kiểm soát
register-level. STM32 HAL là standard framework cho STM32.

### Files MATLAB cần port → C

| MATLAB | C file (dự kiến) | Trạng thái |
|--------|-----------------|-----------|
| `spi_interface.m` | `spi_interface.c/.h` | ⏳ Chưa làm |
| `encoder_pulse_gen.m` | `encoder_pulse_gen.c/.h` | ⏳ Chưa làm |
| `imu_packet_enc.m` | `imu_packet_enc.c/.h` | ⏳ Chưa làm |
| `pwm_capture.m` | `pwm_capture.c/.h` | ⏳ Chưa làm |
| `gpio_sync.m` | `gpio_sync.c/.h` | ⏳ Chưa làm |

### Vai trò của H7 trong hệ thống

H7 là **clock master** và **signal converter**:
- Nhận state `x[k]` từ RPi5 qua SPI → gen encoder pulses + IMU UART packet → ESP32
- Nhận PWM từ ESP32 → convert sang τ → gửi lên RPi5 qua SPI
- Trigger timing mỗi 1ms (H7 là nguồn clock, RPi5 respond)

### Mapping MATLAB logic → STM32 HAL

#### `spi_interface.c`
```
MATLAB: fixed-point quantization (float → int16 → float)
→ C: HAL_SPI_TransmitReceive_DMA()
     Pack: float → int16_t với scale factor
     Unpack: int16_t → float
     DMA để không block CPU trong transfer
```

#### `encoder_pulse_gen.c`
```
MATLAB: fractional accumulator + round() + noise
→ C: TIM_OC (Output Compare) để gen quadrature pulses
     Accumulator: static float variable (persistent)
     Frequency = omega × PPR / (2π) Hz
     Quadrature: 2 channels, 90° phase shift
```

#### `imu_packet_enc.c`
```
MATLAB: struct với header 0xAA + int32 ADC codes + XOR checksum
→ C: uint8_t packet[N] array
     HAL_UART_Transmit() hoặc DMA
     Packet format phải khớp CHÍNH XÁC với imu_reader.cpp (ESP32)
```

#### `pwm_capture.c`
```
MATLAB: Gaussian jitter + 10-bit quantize + deadband
→ C: TIM_IC (Input Capture) đo duty cycle
     Deadband: if (duty < DEADBAND_FRAC) tau = 0
     Linear map: tau = duty × tau_max
```

#### `gpio_sync.c`
```
MATLAB: Gaussian jitter model (simulation only)
→ C: Hardware GPIO interrupt hoặc timer sync
     Thực tế: H7 trigger mỗi 1ms bằng TIM interrupt
     Không cần jitter simulation — timing là hardware thật
```

### Thứ tự implement H7

```
Bước 1: CubeMX setup
  → Cấu hình SPI1 (slave, kết nối RPi5 — RPi5 là master)
  → Cấu hình GPIO_READY (output, báo hiệu cho RPi5 khi có data mới)
  → Cấu hình TIM2 (1kHz interrupt — H7 vẫn là clock master nội bộ)
  → Cấu hình TIM3/4 (PWM output cho encoder pulses)
  → Cấu hình USART1 (UART tới ESP32, IMU packet)
  → Cấu hình TIM5 (Input Capture cho PWM từ ESP32)

Bước 2: Main loop trong TIM2 interrupt (1kHz)
  → Capture PWM từ ESP32 → tính τ
  → Đặt τ vào SPI TX buffer
  → Kéo GPIO_READY HIGH → RPi5 initiate SPI transfer
  → SPI interrupt: nhận x[k] từ RPi5, gửi τ đi cùng lúc (full-duplex)
  → Update encoder pulse frequency từ x[k] (wheel speeds)
  → Gửi IMU packet qua UART

Bước 3: Test từng module
  → SPI: scope kiểm tra timing, logic analyzer verify data
  → Encoder: scope kiểm tra quadrature waveform, đếm pulses
  → UART: terminal verify IMU packet format

Bước 4: Integration test H7 + RPi5
  → RPi5 chạy main_loop.py với SPI thật
  → H7 gửi τ=0 → verify RPi5 nhận đúng → state = zeros
  → H7 gửi τ=0.05 → verify state tăng đúng theo plant model
```

### Lưu ý quan trọng cho H7

**SPI protocol với RPi5: RPi5 master, H7 slave ✅ Đã chốt**
- RPi5 là SPI master (chủ động initiate transaction) — dùng `spidev` standard
- H7 là SPI slave (respond khi được trigger)
- Lý do: Linux `spidev` driver chỉ support master. RPi5/RP1 có hardware SPI slave
  (SPI4, GPIO 8-11) nhưng kernel driver chưa support ổn định (experimental patch
  trên rpi-6.18.y, chưa có overlay). Dùng RPi5 master là path ít rủi ro nhất.
- Timing: RPi5 SCHED_FIFO jitter ~200µs không ảnh hưởng physics vì `plant_step`
  dùng dt=0.001 cố định trong toán — jitter chỉ ảnh hưởng *thời điểm* gọi SPI,
  không ảnh hưởng kết quả tính toán. H7 gen encoder bằng hardware TIM riêng,
  không phụ thuộc khi nào RPi5 trigger SPI.
- H7 dùng GPIO_READY pin báo hiệu "có data mới" → RPi5 nhận GPIO interrupt → initiate transfer
- Mỗi transaction full-duplex: RPi5 gửi x[k] (40 bytes = 10×float32),
  nhận τ (16 bytes = 4×float32) cùng lúc

**Encoder pulse format:**
- Quadrature encoder: 2 kênh A và B, lệch pha 90°
- PPR = 1024 → 4096 counts/rev (quadrature)
- Frequency = ω × 1024 / (2π) pulses/second
- Ở ω = 34.56 rad/s (max): f = 5632 Hz → TIM period = 177µs

**IMU packet format (phải khớp với ESP32):**
```c
typedef struct {
    uint8_t  header;        // 0xAA
    int32_t  accel_x_raw;   // ADC code
    int32_t  accel_y_raw;
    int32_t  accel_z_raw;
    int32_t  gyro_x_raw;
    int32_t  gyro_y_raw;
    int32_t  gyro_z_raw;
    float    accel_scale;   // m/s² per count
    float    gyro_scale;    // rad/s per count
    uint8_t  checksum;      // XOR của 6 int32 codes
} ImuPacket_t;
```

---

## Module 3 — ESP32 (Controller)

### Ngôn ngữ: C++ (ESP-IDF) ✅ Đã chốt
**Lý do:** FreeRTOS có sẵn với `xTaskCreatePinnedToCore()` để pin control loop vào
core 1, hardware timer chính xác, SPI/UART đều có DMA driver chuẩn không bị
Arduino abstraction che khuất. ESP-IDF hỗ trợ host-based unit test — compile và
chạy controller logic trên PC mà không cần board.

### Files MATLAB cần port → C++

| MATLAB | C++ file (dự kiến) | Trạng thái |
|--------|--------------------|-----------|
| `encoder_reader.m` | `encoder_reader.cpp/.h` | ⏳ Chưa làm |
| `imu_reader.m` | `imu_reader.cpp/.h` | ⏳ Chưa làm |
| `pid_controller.m` | `pid_controller.cpp/.h` | ⏳ Chưa làm |
| `adrc_controller.m` | `adrc_controller.cpp/.h` | ⏳ Chưa làm |
| `pwm_output.m` | `pwm_output.cpp/.h` | ⏳ Chưa làm |
| `pose_estimator.m` | `pose_estimator.cpp/.h` | ⏳ Chưa làm |
| `position_controller.m` | `position_controller.cpp/.h` | ⏳ Chưa làm |
| `slip_detector.m` | `slip_detector.cpp/.h` | ⏳ Chưa làm (optional M6) |

### Mapping MATLAB logic → C++

#### `encoder_reader.cpp`
```
MATLAB: counts/step → raw omega → IIR low-pass filter
        alpha = dt/(tau_f + dt) = 0.001/(0.005+0.001) ≈ 0.167

→ C++: class EncoderReader {
    float omega_prev = 0;
    float alpha;
    float decode(int counts, float dt) {
        float omega_raw = counts * 2*PI / (PPR * dt);
        float omega_filt = alpha*omega_raw + (1-alpha)*omega_prev;
        omega_prev = omega_filt;
        return omega_filt;
    }
}
Lưu ý: `persistent` trong MATLAB → class member variable trong C++
```

#### `pid_controller.cpp`
```
MATLAB: full PID + conditional integration anti-windup
        tau_tent = P + Ki*int_new + D
        nếu saturate VÀ error cùng chiều → freeze integral

→ C++: class PidController {
    float integral = 0, error_prev = 0;
    float compute(float ref, float meas, float dt) {
        float err = ref - meas;
        float P = Kp * err;
        float D = Kd * (err - error_prev) / dt;
        float int_new = integral + err * dt;
        float tau_tent = P + Ki*int_new + D;
        // Anti-windup: chỉ update integral khi không saturate
        if (fabsf(tau_tent) < tau_max || err*integral < 0)
            integral = int_new;
        integral = constrain(integral, -tau_max/Ki, tau_max/Ki);
        error_prev = err;
        return constrain(P + Ki*integral + D, -tau_max, tau_max);
    }
}
```

#### `adrc_controller.cpp`
```
MATLAB: 2nd-order ESO (z1=omega, z2=disturbance) + PD control
        z1 += dt*(b0*u_prev + z2 - beta1*e_eso)
        z2 += dt*(-beta2*e_eso)
        tau = (kp*(ref-z1) - z2) / b0

→ C++: class AdrcController {
    float z1=0, z2=0, u_prev=0;
    float compute(float ref, float omega_meas, float dt) {
        float e_eso = z1 - omega_meas;
        z1 += dt*(b0*u_prev + z2 - beta1*e_eso);
        z2 += dt*(-beta2*e_eso);
        z2 = constrain(z2, -z2_max, z2_max);  // anti-windup
        float tau = (kp*(ref - z1) - z2) / b0;
        tau = constrain(tau, -tau_max, tau_max);
        u_prev = tau;  // PHẢI clamp trước khi lưu
        return tau;
    }
}
Lưu ý: u_prev PHẢI dùng giá trị SAU clamp (bug quan trọng từ M5)
```

#### `position_controller.cpp`
```
MATLAB: PI outer loop, error trong world frame → rotate sang body frame
        int_pos += pos_err * dt  (với anti-windup)
        v_cmd = Kp*pos_err + Ki*int_pos

→ C++: Tương tự PID nhưng input là (x,y,theta) error,
        output là (vx_cmd, vy_cmd, wz_cmd) body frame
        Cần hàm rotation: rotate error bằng -theta_current
```

### Thứ tự implement ESP32

```
Bước 1: Setup project (Arduino IDE hoặc PlatformIO)
  → Board: ESP32 Dev Module
  → Tạo structure: src/main.cpp + include/*.h

Bước 2: Port từng class, compile offline
  → encoder_reader.h/cpp  → unit test với giá trị giả
  → imu_reader.h/cpp      → verify packet decode đúng format H7
  → pid_controller.h/cpp  → step response test (không cần plant)
  → adrc_controller.h/cpp → ESO convergence test
  → pwm_output.h/cpp      → deadband compensation verify
  → pose_estimator.h/cpp  → kinematics verify
  → position_controller.h/cpp → PI verify

Bước 3: main.cpp — control loop 1kHz
  → Dùng ESP32 hardware timer (timerAlarmAttach) hoặc FreeRTOS task
  → Thứ tự: đọc encoder → đọc IMU → controller → PWM output

Bước 4: Flash và test với H7
  → Không cần RPi5 trước: H7 gen encoder/IMU giả → ESP32 nhận
  → Verify τ_cmd output hợp lý
```

### Lưu ý quan trọng cho ESP32

**Real-time trên ESP32:**
- ESP32 có 2 core. Pin control loop vào core 1, WiFi/BT vào core 0
- Dùng `xTaskCreatePinnedToCore()` nếu dùng FreeRTOS
- Hardware timer interrupt cho 1kHz: `hw_timer_t* timer = timerBegin(0, 80, true)`

**Gains dùng lại từ MATLAB (params_mecanum.m):**
```cpp
// PID inner loop
const float Kp = 0.04f, Ki = 0.5f, Kd = 0.0004f;

// ADRC
const float b0 = 1.0f/0.00247f;  // ≈ 404.86
const float beta1 = 200.0f, beta2 = 10000.0f, kp = 20.0f;

// Outer loop
const float Kp_pos = 6.0f, Ki_pos = 3.0f;
const float Kp_theta = 8.0f, Ki_theta = 6.0f;
```

**IMU packet decode phải khớp CHÍNH XÁC với H7 encode:**
```cpp
// Packet structure (phải match ImuPacket_t trong H7)
struct ImuPacket {
    uint8_t  header;         // expect 0xAA
    int32_t  accel_raw[3];
    int32_t  gyro_raw[3];
    float    accel_scale;
    float    gyro_scale;
    uint8_t  checksum;       // XOR verify
};
```

---

## Thứ tự Integration Test (khi có hardware)

```
Phase 1: RPi5 standalone
  → python3 test_plant_python.py          (24/24 PASS đã xác nhận)
  → sudo python3 main_loop.py --duration 5 (verify jitter < 500µs)

Phase 2: H7 standalone
  → Logic analyzer verify SPI timing
  → Oscilloscope verify encoder quadrature waveform
  → Terminal verify IMU UART packet format

Phase 3: RPi5 + H7
  → H7 gửi τ=0 → RPi5 state = zeros ✓
  → H7 gửi τ=[0.05,0.05,0.05,0.05] → verify vx tăng đúng
  → So sánh với MATLAB simulation golden values

Phase 4: ESP32 standalone (với H7 giả lập encoder/IMU)
  → Flash ESP32, verify τ_cmd output không NaN/Inf
  → PID step response: omega ref 10 rad/s → settle trong ~0.36s

Phase 5: Cả 3 board — full loop
  → Trajectory thẳng → compare tracking error với MATLAB (~8.9mm)
  → Circle → compare (~6.4mm ADRC, ~7.0mm PID)
  → Nếu error lớn hơn nhiều → chạy diagnose_error_sources logic
     để tách nguồn error (controller/signal/odometry)
```

---

## File inventory hiện tại

### ✅ Đã có (Python, RPi5)
- `params_mecanum.py`
- `plant_step.py`
- `imu_model.py`
- `state_manager.py`
- `main_loop.py`
- `test_plant_python.py`

### ⏳ Chưa làm (C, H7)
- `spi_interface.c/.h`
- `encoder_pulse_gen.c/.h`
- `imu_packet_enc.c/.h`
- `pwm_capture.c/.h`
- `gpio_sync.c/.h`
- `main_h7.c` (STM32 main + TIM interrupt)

### ⏳ Chưa làm (C++, ESP32)
- `encoder_reader.cpp/.h`
- `imu_reader.cpp/.h`
- `pid_controller.cpp/.h`
- `adrc_controller.cpp/.h`
- `pwm_output.cpp/.h`
- `pose_estimator.cpp/.h`
- `position_controller.cpp/.h`
- `slip_detector.cpp/.h`
- `main.cpp` (ESP32 control loop)

---

## Quyết định đã chốt

| Quyết định | Chốt | Lý do |
|-----------|------|-------|
| ESP32 framework | **ESP-IDF** | FreeRTOS core pinning, DMA, host-based unit test |
| H7 SPI mode với RPi5 | **RPi5 master, H7 slave** | `spidev` Linux chỉ support master. RPi5/RP1 có hardware SPI slave nhưng kernel driver chưa ổn định (experimental rpi-6.18.y). RPi5 master đủ tốt vì SCHED_FIFO jitter ~200µs không ảnh hưởng physics. |
| Test ESP32 offline | **g++ cho controller logic; ESP-IDF host test cho hardware interface** | Controller logic (PID, ADRC, pose) là toán thuần túy, không phụ thuộc FreeRTOS hay hardware driver — compile g++ thẳng trên PC. Hardware interface (encoder GPIO, UART, LEDC) test trên board thật. |

### Chi tiết quyết định SPI

RPi5/RP1 hardware có SPI slave ở SPI4 (GPIO 8-11), nhưng Linux kernel driver
hiện tại không support target/slave mode. Có patch thử nghiệm trên branch
`rpi-6.18.y` nhưng chưa có device tree overlay và chưa stable.

Quan trọng hơn: **RPi5 làm master không phải vấn đề về timing.** `plant_step`
dùng `dt=0.001` cố định trong toán học — jitter ~200µs của SCHED_FIFO chỉ ảnh
hưởng *thời điểm* bắt đầu SPI transaction, không ảnh hưởng kết quả tính toán.
H7 gen encoder pulses bằng hardware TIM riêng, hoàn toàn độc lập với khi nào
RPi5 trigger SPI. Kiến trúc master/slave chỉ quyết định ai giữ SPI clock,
không quyết định ai là "clock master" của hệ thống HIL.

### Chi tiết quyết định test ESP32 offline

```
Lớp 1 — Controller logic (compile bằng g++ trên PC):
  pid_controller.cpp      → step response test, anti-windup verify
  adrc_controller.cpp     → ESO convergence test, u_prev clamp verify
  pose_estimator.cpp      → kinematics round-trip verify
  position_controller.cpp → PI integral verify

Lớp 2 — Hardware interface (test trên board thật):
  encoder_reader.cpp      → cần GPIO quadrature thật
  imu_reader.cpp          → cần UART packet thật từ H7
  pwm_output.cpp          → cần oscilloscope verify duty cycle
```
