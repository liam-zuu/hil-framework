# HIL Hardware Deployment Plan

**Mục đích:** Tài liệu căn cứ cho toàn bộ quá trình triển khai MATLAB simulation → phần cứng thật.  
**Cập nhật lần cuối:** 2026-04-21

---

## 1. Bức tranh tổng thể

### 1.1 Bản chất của HIL đúng nghĩa

HIL (Hardware-in-the-Loop) **thay thế phần cứng cơ học** (motor, bánh xe, mặt sàn), không thay thế controller logic hay reference trajectory. Controller chạy trong HIL **y hệt** như khi deploy thật — nó không biết mình đang được test.

```
Robot thật:   Mission PC → ESP32 → PWM → Motor → Encoder → ESP32
HIL:          Mission PC → ESP32 → PWM → H7 → RPi5 plant → Encoder → ESP32
                                              ↑
                                    HIL thay thế phần này
```

### 1.2 Vai trò từng board

| Board | Vai trò trong HIL | Tương đương MATLAB |
|-------|------------------|--------------------|
| RPi5 | Plant simulator + Mission computer | `rpi5/` cluster + `trajectory_generator.m` |
| STM32 Nucleo H7 | Signal conditioning (SPI slave) | `nucleoh7/` cluster |
| ESP32 | Controller (PID / ADRC) | `esp32/` cluster |

### 1.3 Luồng dữ liệu mỗi timestep (1ms)

```
┌─────────────────────────────────────────────────────────────────┐
│  RPi5 (Process 1 — Plant, 1kHz)                                 │
│    1. PrecisionTimer.wait_next_tick()   ← đợi đúng 1ms deadline │
│    2. gpio.pulse()                      ← báo H7 chuẩn bị       │
│    3. spi.transfer(encode(state))       ← gửi x[k], nhận τ[k]   │
│    4. state = plant.step(τ)             ← tính x[k+1]            │
└─────────────────────────────────────────────────────────────────┘
         ↕ SPI (1MHz, full-duplex)
┌─────────────────────────────────────────────────────────────────┐
│  H7 (SPI Slave, interrupt-driven)                               │
│    1. EXTI interrupt từ GPIO pulse      ← biết data sắp đến     │
│    2. DMA nhận state frame từ RPi5      ← x[k] vào buffer       │
│    3. Decode state → generate encoder pulses → gửi ESP32        │
│    4. Đọc PWM từ ESP32 → encode → gửi lên RPi5 qua SPI         │
└─────────────────────────────────────────────────────────────────┘
         ↕ Encoder pulses (timer output compare)
         ↕ PWM capture (timer input capture)
         ↕ UART (IMU packet)
┌─────────────────────────────────────────────────────────────────┐
│  ESP32 (FreeRTOS, 1kHz control task pinned core 1)             │
│    1. Đọc encoder → estimate wheel speeds                       │
│    2. Đọc UART IMU packet → estimate heading                    │
│    3. Nhận trajectory reference từ RPi5 qua UART riêng (50Hz)  │
│    4. PID hoặc ADRC → tính τ_cmd                                │
│    5. Output PWM → H7 capture                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 RPi5 có hai vai độc lập

```
Process 1 — Plant (1kHz, core 0, SCHED_FIFO):
    SPI ↔ H7 ↔ plant_step()

Process 2 — Mission computer (50Hz, core 2, normal):
    trajectory.get_reference(t) → UART → ESP32
```

Hai process **không share state, không block nhau**. Đây là kiến trúc standard: navigation stack và control loop luôn tách biệt.

---

## 2. RPi5 — Chi tiết triển khai

### 2.1 Những gì đã có (MATLAB → Python)

| MATLAB (simulation) | Python (hardware) | Ghi chú |
|--------------------|--------------------|---------|
| `plant_step.m` | `plant_interface.py` + wrapper | Logic vật lý giữ nguyên, thêm interface |
| `imu_model.m` | Tích hợp trong plant wrapper | IMU data gửi qua H7, không trực tiếp |
| `state_manager.m` | Biến Python thông thường | Không cần file riêng |
| `spi_interface.m` (encode/decode) | `protocol.py` | **Phải khớp chính xác byte-for-byte** |
| Không có | `spi_master.py` | Mới — giao tiếp kernel `/dev/spidev0.0` |
| Không có | `gpio_sync.py` | Mới — sync pulse với H7 |
| Không có | `timing.py` | Mới — precision 1ms loop |
| Không có | `hil_node.py` | Mới — orchestrator |

### 2.2 Những gì còn thiếu

- **Trajectory publisher (Process 2):** Module generate reference và gửi xuống ESP32 qua UART. Đây là vai "Mission computer" của RPi5.
- **Wire plant model thật:** `make_plant()` trong `run_hil.py` hiện dùng `MockPlant`. Cần thay bằng wrapper của `plant_step.py` có sẵn.
- **Multiprocessing:** Tách plant loop và trajectory publisher thành 2 process riêng với CPU pinning.

### 2.3 Cách wire plant model thật

Plant model Python đã có (24/24 tests pass) cần được wrap trong class kế thừa `PlantInterface`:

```python
from plant_interface import PlantInterface
from your_plant_module import PlantStep, load_params  # file đã có

class MecanumPlant(PlantInterface):
    def __init__(self, dt=0.001):
        self._params = load_params()
        self._dt = dt
        self._state = [0.0] * 10

    def step(self, torques):
        # Gọi đúng hàm plant_step Python có sẵn
        self._state = PlantStep(self._state, torques, self._params, self._dt)
        return list(self._state)

    def get_state(self):
        return list(self._state)

    def reset(self, state=None):
        self._state = state if state is not None else [0.0] * 10
```

### 2.4 Protocol encoding — điểm quan trọng nhất

`protocol.py` phải khớp **chính xác** với `spi_interface.m` (M3). Công thức:

```
int16 = round(clamp(value, ±range) / range × 32767)
```

| Direction | Nội dung | Bytes |
|-----------|----------|-------|
| Downlink MOSI (RPi5 → H7) | 10 states × int16 big-endian | 20 bytes + 4 padding = 24 bytes |
| Uplink MISO (H7 → RPi5) | 4 torques × int16 big-endian | 8 bytes + 16 padding = 24 bytes |

State ranges phải khớp `params.spi.state_ranges` trong MATLAB:
```
[x: ±5m, y: ±5m, θ: ±π, vx: ±3, vy: ±3, wz: ±10, ω1-4: ±40 rad/s]
```

Nếu sai 1 bit ở đây → H7 decode sai state → encoder pulses sai → controller nhận tín hiệu sai → mọi thứ sai mà không có error message nào.

### 2.5 Timing — sleep + busy-wait hybrid

```
|←── sleep ──────────────────→|←── busy-wait →|←── work ──→|
t=0                         t=deadline-200μs  t=deadline   t=deadline+work
```

Không dùng `time.sleep(0.001)` thuần vì Linux OS có thể wake muộn 500μs. Không dùng busy-wait thuần vì đốt 100% CPU. Hybrid: sleep đến 200μs trước deadline, busy-wait phần còn lại.

### 2.6 GPIO — lưu ý RPi5

RPi5 dùng chip RP1 (không phải BCM2835 như RPi4). Do đó:
- Dùng **`lgpio`**, không phải `RPi.GPIO`
- GPIO device là `/dev/gpiochip4`, không phải `/dev/gpiochip0`
- Pin numbering vẫn là BCM

### 2.7 Checklist trước khi cắm H7

```
□ python3 -m pytest tests/ → tất cả pass
□ python3 rpi5/run_hil.py --mock --steps 5000 → không overrun
□ python3 -c "from timing import measure_jitter; measure_jitter(2000)"
  → max jitter < 100μs
□ Short MOSI-MISO (pin 19-21)
□ python3 rpi5/loopback_test.py → 4/4 pass
□ Bỏ short, cắm H7
```

---

## 3. STM32 Nucleo H7 — Chi tiết triển khai

### 3.1 Vai trò

H7 là **signal conditioning layer** — không có logic điều khiển, không tính toán vật lý. Nó nhận state thô từ RPi5 và chuyển thành tín hiệu phần cứng mà ESP32 có thể đọc. Chiều ngược lại: nhận PWM từ ESP32, chuyển thành số và gửi lên RPi5.

### 3.2 Những gì cần implement (từ MATLAB sang C)

| MATLAB module | C implementation | Peripheral STM32 |
|---------------|-----------------|------------------|
| `spi_interface.m` | SPI slave DMA | SPI1 slave + DMA2 |
| `encoder_pulse_gen.m` | Timer output compare | TIM1/TIM8 (4 channels) |
| `imu_packet_enc.m` | UART TX DMA | USART3 |
| `pwm_capture.m` | Timer input capture | TIM2/TIM3 |
| `gpio_sync.m` | EXTI interrupt | EXTI line trên PA0 |

### 3.3 Kiến trúc firmware H7

**Approach: HAL cho init, LL cho hot loop**

```
CubeMX generate:
  HAL_SPI_Init()        ← boilerplate clock, DMA, NVIC
  HAL_TIM_Init()        ← timer config
  HAL_UART_Init()       ← UART config
  HAL_GPIO_Init()       ← EXTI config

Viết tay (LL):
  EXTI interrupt handler    ← nhận sync pulse từ RPi5
  SPI DMA complete callback ← decode state, trigger encoder gen
  PWM capture callback      ← encode torque, trigger SPI uplink
```

### 3.4 Luồng xử lý trong H7 (interrupt-driven, không có main loop logic)

```
1. RPi5 bắn GPIO pulse
   → EXTI interrupt fires
   → Enable SPI slave DMA receive (chuẩn bị nhận state frame)

2. SPI DMA complete (24 bytes nhận xong)
   → Decode 10 states từ int16 fixed-point
   → Generate encoder pulses cho 4 wheels (TIM output compare)
   → Pack IMU data → UART DMA TX

3. ESP32 gửi PWM lên
   → TIM input capture fires
   → Measure duty cycle → decode torque
   → Pack torque vào SPI TX buffer
   → SPI DMA transmit (sẵn sàng cho transaction tiếp theo)
```

Không có `while(1)` logic. Toàn bộ là interrupt/DMA callback. Đây là lý do chọn LL cho hot loop — HAL có overhead wrapper không cần thiết trong interrupt context.

### 3.5 Encoder pulse generation — điểm kỹ thuật quan trọng

MATLAB `encoder_pulse_gen.m` dùng fractional accumulator để handle low-speed case:

```
counts_exact = omega × PPR / (2π) × dt
accumulator += counts_exact
pulses_this_step = floor(accumulator)
accumulator -= pulses_this_step
```

C implementation phải làm y hệt — dùng `float` accumulator per wheel, `static` variable trong callback. Không dùng integer division đơn giản vì ở tốc độ thấp (ω=0.5 rad/s) chỉ có 0.08 counts/step, integer division luôn ra 0.

### 3.6 SPI mode và wiring

```
Mode: SPI Mode 0 (CPOL=0, CPHA=0) — khớp với RPi5 spidev default
Speed: 1MHz ban đầu, tăng lên 4MHz sau khi validate
Frame: 24 bytes full-duplex

Wiring:
  RPi5 GPIO8  (CE0,  pin 24) → H7 NSS   (active low)
  RPi5 GPIO11 (SCLK, pin 23) → H7 SCK
  RPi5 GPIO10 (MOSI, pin 19) → H7 MOSI
  RPi5 GPIO9  (MISO, pin 21) → H7 MISO
  RPi5 GPIO17 (pin 11)       → H7 EXTI pin (sync)
  RPi5 GND    (pin 9)        → H7 GND  ← bắt buộc, common ground
```

### 3.7 Checklist H7 trước khi cắm ESP32

```
□ CubeMX project generate OK, build không lỗi
□ SPI loopback test với RPi5 (H7 echo lại state frame)
□ Logic analyzer: verify encoder pulse frequency khớp với ω
□ Logic analyzer: verify UART IMU packet format khớp protocol
□ Logic analyzer: verify PWM capture decode đúng
□ Jitter SPI transaction < 50μs
```

---

## 4. ESP32 — Chi tiết triển khai

### 4.1 Vai trò

ESP32 là **controller node** — chạy PID hoặc ADRC, đọc sensor từ H7, gửi torque về H7, nhận trajectory reference từ RPi5.

### 4.2 Những gì cần implement (từ MATLAB sang C)

| MATLAB module | C implementation | Ghi chú |
|---------------|-----------------|---------|
| `encoder_reader.m` | IIR filter trên encoder count | Phải clear filter state khi reset |
| `imu_reader.m` | UART receive + XOR checksum | Packet format từ H7 |
| `pid_controller.m` | PID + conditional anti-windup | 4 instances (4 wheels) |
| `adrc_controller.m` | 2nd-order ESO + control law | 4 instances |
| `pwm_output.m` | PWM gen + deadband compensation | MCPWM peripheral |
| `pose_estimator.m` | Dead reckoning odometry | Integrate mỗi step |
| `position_controller.m` | PI outer loop | Dùng pose estimate |
| `slip_detector.m` | Kinematic consistency check | Optional, monitoring only |
| Không có | Trajectory receiver | UART từ RPi5 Process 2 |

### 4.3 Kiến trúc firmware ESP32 — FreeRTOS

**Lý do chọn ESP-IDF (không phải Arduino):** Core pinning và SCHED_FIFO priority cần thiết cho real-time control. Arduino không expose FreeRTOS API ở mức này.

```
Core 0 — System tasks (FreeRTOS default):
  WiFi stack, BLE (nếu dùng), housekeeping

Core 1 — Control task (pinned, priority cao nhất):
  1kHz control loop:
    encoder_reader()
    imu_reader()
    pose_estimator()
    position_controller()    ← outer loop
    pid_controller()         ← inner loop (hoặc adrc)
    pwm_output()

Core 1 — Trajectory receiver task (priority thấp hơn control):
    UART receive từ RPi5
    Cập nhật shared reference (mutex-protected)
```

### 4.4 Trajectory reference — interface với RPi5

RPi5 Process 2 gửi reference packet xuống ESP32 qua UART mỗi 20ms:

```
Packet format (tối giản):
  Header:    0xBB (1 byte)
  x_ref:     float32 (4 bytes)
  y_ref:     float32 (4 bytes)
  theta_ref: float32 (4 bytes)
  vx_ref:    float32 (4 bytes)
  checksum:  XOR of payload bytes (1 byte)
  Total: 18 bytes
```

ESP32 hold giá trị reference cũ nếu không nhận được packet mới trong 100ms — đây là fail-safe cơ bản.

### 4.5 Chuyển ADRC từ MATLAB sang C — điểm cần chú ý

**2nd-order ESO state phải được khởi tạo đúng:**

```c
// MATLAB dùng persistent variable → C dùng static struct
typedef struct {
    float z1;      // estimated omega
    float z2;      // estimated disturbance
    float u_prev;  // clamped! không phải tau_cmd raw
} ESO_State;

static ESO_State eso[4];  // 4 wheels
```

`u_prev` phải là giá trị đã clamp tại `±tau_max`, không phải `tau_cmd` raw — đây là bug đã fix trong M5.1. Nếu không clamp, ESO diverge.

**z2 cũng phải clamp:**

```c
float z2_max = TAU_MAX * B0;
eso[i].z2 = fmaxf(-z2_max, fminf(z2_max, eso[i].z2));
```

### 4.6 Encoder IIR filter — không dùng floating point nếu có thể

MATLAB dùng:
```
alpha = dt / (tau_f + dt) ≈ 0.167
omega_filt = alpha × omega_raw + (1 - alpha) × omega_prev
```

C trên ESP32: float32 đủ nhanh (FPU có sẵn trên ESP32-S3). Không cần fixed-point. Nhưng phải **clear filter state khi reset** — đây là điểm fragile trong MATLAB (`clear encoder_reader`), C thì dùng `memset(&filter_state, 0, sizeof(filter_state))`.

### 4.7 Checklist ESP32 trước khi tích hợp với H7

```
□ Build ESP-IDF project không warning
□ Test offline với g++ (logic chỉ, không cần board)
□ Test encoder reader với pulse generator giả (function gen hoặc RPi5 GPIO)
□ Test UART receive trajectory packet từ RPi5 Process 2
□ Test PID/ADRC với hardcoded sensor input → verify tau bounded ±tau_max
□ Không có task starvation: control task luôn finish trong < 0.5ms
```

---

## 5. Integration Testing — Thứ tự

```
Phase 1: RPi5 standalone (đã làm một phần)
  □ Plant model mock run 5000 steps, không overrun
  □ SPI loopback test (MOSI-MISO shorted) pass
  □ Trajectory publisher process chạy song song, không ảnh hưởng jitter

Phase 2: RPi5 ↔ H7
  □ H7 echo lại state frame → RPi5 decode đúng
  □ H7 generate encoder pulses → verify bằng logic analyzer
  □ H7 forward IMU packet → verify format

Phase 3: H7 ↔ ESP32
  □ ESP32 đọc encoder từ H7 → omega estimate bounded và smooth
  □ ESP32 đọc UART IMU từ H7 → checksum pass
  □ ESP32 gửi PWM → H7 capture đúng duty cycle

Phase 4: Full loop (RPi5 ↔ H7 ↔ ESP32)
  □ Closed-loop với trajectory đơn giản (line)
  □ Verify state không diverge
  □ Verify jitter toàn hệ thống < 200μs
  □ So sánh state trajectory với MATLAB simulation cùng điều kiện
```

---

## 6. Nguyên tắc chuyển từ MATLAB sang C/Python

### 6.1 Những gì giữ nguyên

- Phương trình vật lý (dynamics, kinematics)
- Hệ số, gains (Kp, Ki, beta1, beta2, b0...)
- Thứ tự tính toán trong mỗi step
- Fixed-point encoding formula (protocol)

### 6.2 Những gì phải thay đổi

| MATLAB | Python/C tương đương | Lý do |
|--------|---------------------|-------|
| `persistent` variable | `static` (C) / instance variable (Python) | Scope và lifetime |
| `clear function_name` | `memset` / `reset()` method | Reset state giữa runs |
| `rng(seed)` | `srand(seed)` / `random.seed()` | Reproducibility |
| Matrix ops (`*`, `\`) | Loop tường minh hoặc BLAS | Không có MATLAB engine |
| `mod(θ+π, 2π) - π` | `atan2(sin(θ), cos(θ))` | Portable heading wrap |

### 6.3 Điểm dễ sai nhất

1. **Persistent state không được clear:** Trong MATLAB có `clear encoder_pulse_gen`. Trong C/Python phải gọi reset function tường minh. Nếu quên, filter state từ run trước carry over → kết quả sai, không có warning.

2. **Byte order:** MATLAB `int16` mặc định little-endian trên x86. Protocol dùng big-endian. Phải explicit `">h"` trong Python, `__builtin_bswap16()` trong C.

3. **Integer division:** C `int / int = int`. Encoder count chia PPR phải cast sang float trước: `(float)count / PPR`.

4. **Float precision:** MATLAB double (64-bit). C/Python có thể mix float32/float64. ESO và PID nên dùng float32 (đủ cho 16-bit sensor data, tiết kiệm memory trên ESP32).

---

## 7. Thư mục project sau khi hoàn thành

```
HIL_HARDWARE/
├── rpi5/
│   ├── protocol.py           ← SPI encode/decode (done)
│   ├── spi_master.py         ← spidev wrapper (done)
│   ├── gpio_sync.py          ← sync pulse (done)
│   ├── timing.py             ← precision timer (done)
│   ├── plant_interface.py    ← abstract + MockPlant (done)
│   ├── hil_node.py           ← orchestrator Process 1 (done)
│   ├── trajectory_pub.py     ← mission computer Process 2 (TODO)
│   ├── mecanum_plant.py      ← wire plant_step.py vào đây (TODO)
│   └── run_hil.py            ← entry point (done, cần update make_plant)
│
├── nucleoh7/                 (TODO — sau khi mua board)
│   ├── Core/Src/
│   │   ├── spi_slave.c       ← SPI slave DMA
│   │   ├── encoder_gen.c     ← timer output compare
│   │   ├── imu_packet.c      ← UART TX
│   │   ├── pwm_capture.c     ← timer input capture
│   │   └── gpio_sync.c       ← EXTI handler
│   └── CubeMX project files
│
├── esp32/                    (TODO — sau khi H7 stable)
│   ├── main/
│   │   ├── encoder_reader.c
│   │   ├── imu_reader.c
│   │   ├── pid_controller.c
│   │   ├── adrc_controller.c
│   │   ├── pwm_output.c
│   │   ├── pose_estimator.c
│   │   ├── position_controller.c
│   │   ├── trajectory_rx.c   ← nhận reference từ RPi5
│   │   └── main.c            ← FreeRTOS task setup
│   └── sdkconfig
│
└── tests/
    ├── test_protocol.py      (done)
    ├── test_hil_node.py      (done)
    ├── test_timing.py        (done)
    └── integration/          (TODO — Phase 2-4)
```

---

## 8. Câu hỏi hội đồng và cách trả lời

**"Controller không biết nó đang chạy trong HIL hay robot thật?"**
> Đúng. Controller nhận encoder và IMU qua đúng hardware interface (GPIO pulses, UART). RPi5 gửi trajectory reference qua UART giống mission computer thật. Controller không có cách phân biệt.

**"RPi5 vừa chạy plant 1kHz vừa gửi trajectory có kịp không?"**
> Hai process độc lập trên hai core riêng. Plant loop pinned core 0 với SCHED_FIFO — không bị preempt. Trajectory publisher chạy core 2 ở 50Hz — tần số thấp, không ảnh hưởng timing plant.

**"Tại sao không dùng Simscape thay vì tự viết plant?"**
> Plant model được validate 5 tầng độc lập (49/49 tests), bao gồm cross-check với ode45 và literature comparison. Simscape phụ thuộc license. Plant tự viết cho phép deploy trực tiếp lên RPi5 Python mà không cần toolbox.

**"ADRC có ưu thế gì so với PID trong HIL context?"**
> ESO estimate disturbance online — encoder quantization coarse (PPR 256) và wheel slip đều được absorb vào z2. PID degradation exponential khi giảm PPR, ADRC gần như không đổi (M6 results: PID 32mm vs ADRC 5.5mm ở PPR 256).
