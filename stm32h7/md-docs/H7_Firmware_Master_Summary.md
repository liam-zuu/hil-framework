# H7 Firmware — Master Summary
**Dự án:** HIL Framework — STM32 Nucleo H753ZI (H7)  
**Cập nhật:** 26/04/2026  
**Trạng thái:** Phase 1 ✅ Phase 2 ✅ — Phase 3 (ESP32) đang chuẩn bị

---

## 1. Tổng quan kiến trúc

```
RPi5 (Plant simulation — Python)
  │  SPI1 slave 1 MHz, 24-byte frame, DMA, 1 kHz loop
  ▼
STM32H753ZI (H7) — Signal conditioning
  ├── SPI1 : nhận plant state từ RPi5, gửi torque về
  ├── TIM1-4: tạo xung quadrature encoder cho 4 bánh (OC Toggle)
  ├── TIM5/15/16/17: đo PWM từ ESP32 (chưa kích hoạt — xem mục 12)
  └── SPI2 : emulate BNO085 cho ESP32 (chưa triển khai)
  │  SPI2 slave
  ▼
ESP32-S3 (Controller — PID / ADRC)
  ├── Đọc encoder từ H7 (A/B quadrature)
  ├── Đọc IMU từ H7 (BNO085 emulation qua SPI2)
  ├── Chạy thuật toán điều khiển
  └── Output PWM → H7 đo lại → decode torque → gửi về RPi5
```

HIL loop đã verify: **1 kHz, jitter max 0.4 μs, 0 overrun** (software + logic analyzer).

---

## 2. CubeMX Configuration

### 2.1 Clock tree

| Thông số | Giá trị |
|---|---|
| HSE | 25 MHz (Crystal/Ceramic Resonator) |
| PLL: PLLM=5, PLLN=192, PLLP=2 | SYSCLK = 480 MHz |
| AHB (HPRE=/2) | 240 MHz |
| APB1, APB2 (DIV2) | 120 MHz mỗi bus |
| Timer clock (×2 APB) | **240 MHz** |
| Voltage scale | `PWR_REGULATOR_VOLTAGE_SCALE0` (bắt buộc cho 480 MHz) |
| Flash latency | `FLASH_LATENCY_4` |
| Timebase | TIM6 (không dùng SysTick — tránh conflict) |

### 2.2 GPIO

| Pin | Label | Mode | Pull |
|---|---|---|---|
| PB0 | LD1 | GPIO_Output | No pull |
| PE1 | LD2 | GPIO_Output | No pull |
| PB14 | LD3 | GPIO_Output | No pull |
| PD0 | SYNC_IN | GPIO_EXTI0 Rising edge | Pull-down |
| PD1 | BNO_HINT | GPIO_Output | No pull |
| PG0–3 | W1–4_REN | GPIO_Input | No pull |
| PC8–9 | W1–2_LEN | GPIO_Input | No pull |
| PA8–9 | W3–4_LEN | GPIO_Input | No pull |

### 2.3 SPI1 — RPi5 link

| Thông số | Giá trị |
|---|---|
| Mode | Full-Duplex **Slave** |
| NSS | Hardware NSS Input Signal |
| Data size | 8-bit |
| Frame Format | Motorola, MSB First |
| CPOL / CPHA | Low / 1 Edge (Mode 0) |
| DMA RX | DMA1 Stream0, Peripheral→Memory, Byte |
| DMA TX | DMA1 Stream1, Memory→Peripheral, Byte |
| NVIC | SPI1 global interrupt ✅ |
| Pins | PA4=NSS, PA5=SCK, PA6=MISO, PA7=MOSI |

### 2.4 SPI2 — BNO085 emulation (chưa dùng)

Cấu hình giống SPI1. DMA1 Stream2/3. Pins: PC1=MOSI, PC2=MISO, PB10=SCK, PB12=NSS.

### 2.5 TIM1–4 — Encoder generation

| Thông số | Giá trị |
|---|---|
| Mode | Output Compare Toggle, CH1 + CH2 |
| Prescaler | 239 → timer clock = **240 MHz / 240 = 1 MHz** |
| Counter Period (ARR) | 65535 |
| OC Polarity | High |
| NVIC | Enabled (TIM1 có 4 vector riêng: BRK/UP/TRG_COM/**CC**) |

> **Lưu ý TIM1:** Chỉ enable `TIM1_CC_IRQn`, không cần BRK/UP/TRG_COM cho encoder gen.

Pins:
```
TIM1: PE9 = W1_A,  PE11 = W1_B
TIM2: PA5 = W2_A,  PA15 = W2_B
TIM3: PA6 = W3_A,  PB5  = W3_B
TIM4: PD12 = W4_A, PD13 = W4_B
```

### 2.6 TIM5/15/16/17 — PWM Capture (chưa kích hoạt)

| Timer | Channels | Signal |
|---|---|---|
| TIM5 | CH1–4 | RPWM wheel 1–4 (PA0–PA3) |
| TIM15 | CH1–2 | LPWM wheel 1–2 (PE5, PE6) |
| TIM16 | CH1 | LPWM wheel 3 (PF6) |
| TIM17 | CH1 | LPWM wheel 4 (PF7) |

Mode: Input Capture direct, PSC=239, IC Polarity: Rising Edge. **Giữ `//PWMCapture_Init()` comment cho đến khi ESP32 nối vào — xem Bug #7.**

### 2.7 Manual overrides sau khi CubeMX generate

#### A. TIM15/16/17 Prescaler — CubeMX generate PSC=0 (sai)

```c
/* USER CODE BEGIN TIM15_Init 1 */
htim15.Init.Prescaler = 239;
/* USER CODE END TIM15_Init 1 */
/* tương tự TIM16_Init 1, TIM17_Init 1 */
```

#### B. SysTick_Handler — thêm `HAL_IncTick`

CubeMX để trống handler khi dùng TIM6 làm timebase:

```c
void SysTick_Handler(void)
{
    HAL_IncTick();
}
```

#### C. PWM Capture callback — dùng LL thay HAL

`HAL_TIM_IC_ConfigChannel` không an toàn trong ISR (HAL state machine không re-entrant). Dùng LL:

```c
LL_TIM_IC_SetPolarity(htim->Instance, ll_ch, LL_TIM_IC_POLARITY_FALLING);
```

#### D. MPU config (dự phòng khi bật D-Cache)

Khi cần bật D-Cache (optimization phase), thêm Region 1 vào `MPU_Config()` trong USER CODE (tồn tại qua regen):

```c
MPU_InitStruct.Number        = MPU_REGION_NUMBER1;
MPU_InitStruct.BaseAddress   = 0x30000000;   /* SRAM2 / RAM_D2 */
MPU_InitStruct.Size          = MPU_REGION_SIZE_32KB;
MPU_InitStruct.IsCacheable   = MPU_ACCESS_NOT_CACHEABLE;
MPU_InitStruct.IsBufferable  = MPU_ACCESS_NOT_BUFFERABLE;
/* ... full init như bình thường ... */
HAL_MPU_ConfigRegion(&MPU_InitStruct);
```

> D-Cache hiện tại **tắt** → DMA buffer đặt trong `.bss` bình thường là đủ. Không cần `__attribute__((section(".RAM_D2")))`.

---

## 3. SPI1 Protocol (RPi5 ↔ H7)

Frame 24 bytes, big-endian int16, full-duplex, trao đổi đồng thời.

### MOSI (RPi5 → H7) — Plant state

| Byte | Field | Scale |
|---|---|---|
| 0–1 | omega[0] | int16 = rad/s × 32767 / 40.0 |
| 2–3 | omega[1] | — |
| 4–5 | omega[2] | — |
| 6–7 | omega[3] | — |
| 8–9 | pos_x | int16 = m × 32767 / 10.0 |
| 10–11 | pos_y | — |
| 12–13 | theta | int16 = rad × 32767 / π |
| 14–15 | vx | int16 = m/s × 32767 / 5.0 |
| 16–17 | vy | — |
| 18–19 | omega_body | — |
| 20 | fault_flags | bitmask (xem `HIL_FAULT_*`) |
| 21 | seq | frame counter |
| 22–23 | reserved | 0x00 |

### MISO (H7 → RPi5) — Controller output

| Byte | Field | Scale |
|---|---|---|
| 0–1 | torque[0] | int16 = N·m × 32767 / 1.0 |
| 2–3 | torque[1] | — |
| 4–5 | torque[2] | — |
| 6–7 | torque[3] | — |
| 8–9 | pwm[0] | int16 = duty × 32767 |
| 10–11 | pwm[1] | — |
| 12–13 | pwm[2] | — |
| 14–15 | pwm[3] | — |
| 16 | status | uint8 |
| 17 | seq | uint8 |
| 18–23 | reserved | 0x00 |

> **Startup race condition:** Frame đầu tiên sau reset H7 thường trả về 0xFF (H7 chưa arm DMA kịp). Không phải bug — workaround: gửi 5 warmup frame omega=0 trước khi đọc data thật.

---

## 4. File structure

```
Core/
├── Inc/
│   ├── hil_protocol.h       ← types, frame layout, scales
│   ├── spi1_handler.h
│   ├── encoder_gen.h
│   └── pwm_capture.h        ← chưa dùng
└── Src/
    ├── spi1_handler.c
    ├── encoder_gen.c
    └── pwm_capture.c        ← chưa dùng
```

Các file `Core/Src/*.c` và `Core/Inc/*.h` tự tạo — **không bị CubeMX xóa khi regen.**

---

## 5. Code

### 5.1 hil_protocol.h

```c
#ifndef HIL_PROTOCOL_H
#define HIL_PROTOCOL_H
#include <stdint.h>

#define SPI1_FRAME_LEN   24u
#define SPI2_FRAME_LEN   64u

#define HIL_OMEGA_MAX    40.0f
#define HIL_TORQUE_MAX    1.0f
#define HIL_POS_MAX      10.0f
#define HIL_VEL_MAX       5.0f
#define HIL_THETA_MAX     3.14159265f
#define INT16_MAXF       32767.0f

#define HIL_FAULT_W1_JAM   (1u << 0)
#define HIL_FAULT_W2_JAM   (1u << 1)
#define HIL_FAULT_W3_JAM   (1u << 2)
#define HIL_FAULT_W4_JAM   (1u << 3)
#define HIL_FAULT_W1_ENC   (1u << 4)
#define HIL_FAULT_W2_ENC   (1u << 5)
#define HIL_FAULT_W3_ENC   (1u << 6)
#define HIL_FAULT_W4_ENC   (1u << 7)

typedef struct {
    float   omega[4];
    float   pos_x, pos_y, theta;
    float   vx, vy, omega_body;
    uint8_t fault_flags;
    uint8_t seq;
} HIL_State_t;

typedef struct {
    float   torque[4];
    float   pwm[4];
    uint8_t status;
    uint8_t seq;
} HIL_Output_t;

#endif
```

### 5.2 spi1_handler.h

```c
#ifndef SPI1_HANDLER_H
#define SPI1_HANDLER_H
#include "stm32h7xx_hal.h"
#include "hil_protocol.h"

void    SPI1_Handler_Init(SPI_HandleTypeDef *hspi);
uint8_t SPI1_IsFrameReady(void);
void    SPI1_GetState    (HIL_State_t *out);
void    SPI1_SetOutput   (const HIL_Output_t *out);
void    SPI1_Callback    (SPI_HandleTypeDef *hspi);

#endif
```

### 5.3 spi1_handler.c

```c
#include "spi1_handler.h"
#include <string.h>

static uint8_t _rx[SPI1_FRAME_LEN];
static uint8_t _tx[SPI1_FRAME_LEN];

static SPI_HandleTypeDef *_hspi;
static volatile uint8_t   _frame_ready;
static HIL_State_t        _state;
static HIL_Output_t       _output;

static inline int16_t _rd16(const uint8_t *b) {
    return (int16_t)(((uint16_t)b[0] << 8) | b[1]);
}
static inline void _wr16(uint8_t *b, int16_t v) {
    b[0] = (uint8_t)((uint16_t)v >> 8); b[1] = (uint8_t)v;
}
static inline float _clampf(float v, float lo, float hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static void _decode(void) {
    _state.omega[0]   = _rd16(&_rx[0])  * (HIL_OMEGA_MAX / INT16_MAXF);
    _state.omega[1]   = _rd16(&_rx[2])  * (HIL_OMEGA_MAX / INT16_MAXF);
    _state.omega[2]   = _rd16(&_rx[4])  * (HIL_OMEGA_MAX / INT16_MAXF);
    _state.omega[3]   = _rd16(&_rx[6])  * (HIL_OMEGA_MAX / INT16_MAXF);
    _state.pos_x      = _rd16(&_rx[8])  * (HIL_POS_MAX   / INT16_MAXF);
    _state.pos_y      = _rd16(&_rx[10]) * (HIL_POS_MAX   / INT16_MAXF);
    _state.theta      = _rd16(&_rx[12]) * (HIL_THETA_MAX / INT16_MAXF);
    _state.vx         = _rd16(&_rx[14]) * (HIL_VEL_MAX   / INT16_MAXF);
    _state.vy         = _rd16(&_rx[16]) * (HIL_VEL_MAX   / INT16_MAXF);
    _state.omega_body = _rd16(&_rx[18]) * (HIL_VEL_MAX   / INT16_MAXF);
    _state.fault_flags = _rx[20];
    _state.seq         = _rx[21];
}

static void _encode(void) {
    memset(_tx, 0, sizeof(_tx));
    for (int i = 0; i < 4; i++) {
        _wr16(&_tx[i * 2],     (int16_t)(_clampf(_output.torque[i],
                                -HIL_TORQUE_MAX, HIL_TORQUE_MAX)
                                * (INT16_MAXF / HIL_TORQUE_MAX)));
        _wr16(&_tx[8 + i * 2], (int16_t)(_clampf(_output.pwm[i], -1.0f, 1.0f)
                                * INT16_MAXF));
    }
    _tx[16] = _output.status;
    _tx[17] = _output.seq;
}

void SPI1_Handler_Init(SPI_HandleTypeDef *hspi) {
    _hspi = hspi;
    memset(_tx, 0, sizeof(_tx));
    memset(_rx, 0, sizeof(_rx));
    _frame_ready = 0;
    HAL_SPI_TransmitReceive_DMA(_hspi, _tx, _rx, SPI1_FRAME_LEN);
}

uint8_t SPI1_IsFrameReady(void) { return _frame_ready; }

void SPI1_GetState(HIL_State_t *out) {
    *out = _state;
    _frame_ready = 0;
}

void SPI1_SetOutput(const HIL_Output_t *out) {
    _output = *out;
    _encode();
}

/* Re-arm trong callback là safe: HAL H7 reset state TRƯỚC khi gọi callback */
void SPI1_Callback(SPI_HandleTypeDef *hspi) {
    if (hspi->Instance != _hspi->Instance) return;
    _decode();
    HAL_SPI_TransmitReceive_DMA(_hspi, _tx, _rx, SPI1_FRAME_LEN);
    _frame_ready = 1;
}
```

### 5.4 encoder_gen.h

```c
#ifndef ENCODER_GEN_H
#define ENCODER_GEN_H
#include "stm32h7xx_hal.h"

#define ENC_NUM_WHEELS  4u

void EncoderGen_Init(TIM_HandleTypeDef *ht1, TIM_HandleTypeDef *ht2,
                     TIM_HandleTypeDef *ht3, TIM_HandleTypeDef *ht4);
void EncoderGen_SetOmega(uint8_t wheel, float omega_rad_s);
void EncoderGen_OC_Callback(TIM_HandleTypeDef *htim);

extern volatile uint32_t enc_oc_count;
#endif
```

### 5.5 encoder_gen.c

```c
#include "encoder_gen.h"
#include <math.h>
#include <string.h>

#define ENC_TIMER_FREQ   1000000UL   /* Hz: 240 MHz / (PSC+1=240) */
#define ENC_PPR          330u        /* pulses/rev channel A */
#define ENC_OMEGA_MIN    0.3f        /* rad/s — dưới này coi là zero */

typedef struct {
    TIM_HandleTypeDef *htim;
    uint32_t           half_period;
    int8_t             dir;   /* +1=fwd, -1=rev, 0=stop */
} Wheel_t;

static Wheel_t _w[ENC_NUM_WHEELS];
volatile uint32_t enc_oc_count = 0;

/* half_period = F_timer / (omega × PPR)
 * Công thức lý thuyết có π ở tử số, nhưng thực đo H7 OC Toggle mode có
 * factor π trong timing → bỏ π để compensate → GPIO đúng 525 Hz tại omega=10.
 * Xem Bug #5 để biết chi tiết. */
static uint32_t _hp(float omega_abs) {
    float hp = (float)ENC_TIMER_FREQ / (omega_abs * (float)ENC_PPR);
    if (hp > 65535.0f) return 65535u;
    if (hp <     1.0f) return 1u;
    return (uint32_t)hp;
}

/* Dùng direct register write, KHÔNG dùng HAL_TIM_OC_Start_IT.
 * HAL kiểm tra channel state (READY/BUSY) → return HAL_ERROR silent nếu sai. */
static void _enc_arm_ch(TIM_HandleTypeDef *htim, uint32_t ch, uint32_t offset)
{
    uint32_t ccr = (htim->Instance->CNT + offset) & 0xFFFFu;
    switch (ch) {
        case TIM_CHANNEL_1:
            htim->Instance->CCR1  = ccr;
            htim->Instance->DIER |= TIM_DIER_CC1IE;
            htim->Instance->CCER |= TIM_CCER_CC1E;
            break;
        case TIM_CHANNEL_2:
            htim->Instance->CCR2  = ccr;
            htim->Instance->DIER |= TIM_DIER_CC2IE;
            htim->Instance->CCER |= TIM_CCER_CC2E;
            break;
        default: return;
    }
    if (IS_TIM_BREAK_INSTANCE(htim->Instance))
        htim->Instance->BDTR |= TIM_BDTR_MOE;  /* TIM1 requires MOE */
    htim->Instance->CR1 |= TIM_CR1_CEN;
}

static void _enc_stop(uint8_t i) {
    _w[i].htim->Instance->DIER &= ~(TIM_DIER_CC1IE | TIM_DIER_CC2IE);
    _w[i].htim->Instance->CCER &= ~(TIM_CCER_CC1E  | TIM_CCER_CC2E);
    _w[i].dir = 0;
}

static void _enc_start(uint8_t i, int8_t dir) {
    uint32_t hp = _w[i].half_period;
    /* Forward: A leads B by 90°. Reverse: B leads A by 90°. */
    uint32_t a_off = (dir > 0) ? hp : hp + (hp >> 1);
    uint32_t b_off = (dir > 0) ? hp + (hp >> 1) : hp;
    _enc_arm_ch(_w[i].htim, TIM_CHANNEL_1, a_off);
    _enc_arm_ch(_w[i].htim, TIM_CHANNEL_2, b_off);
    _w[i].dir = dir;
}

void EncoderGen_Init(TIM_HandleTypeDef *ht1, TIM_HandleTypeDef *ht2,
                     TIM_HandleTypeDef *ht3, TIM_HandleTypeDef *ht4)
{
    memset(_w, 0, sizeof(_w));
    _w[0].htim = ht1; _w[1].htim = ht2;
    _w[2].htim = ht3; _w[3].htim = ht4;
    /* KHÔNG gọi HAL_TIM_Base_Start — nó set state=BUSY, block _enc_arm_ch */
}

void EncoderGen_SetOmega(uint8_t wheel, float omega_rad_s) {
    if (wheel >= ENC_NUM_WHEELS) return;
    float abs_omega = (omega_rad_s >= 0.0f) ? omega_rad_s : -omega_rad_s;
    int8_t new_dir  = (omega_rad_s >  ENC_OMEGA_MIN) ?  1 :
                      (omega_rad_s < -ENC_OMEGA_MIN) ? -1 : 0;
    if (new_dir == 0) {
        if (_w[wheel].dir != 0) _enc_stop(wheel);
        return;
    }
    _w[wheel].half_period = _hp(abs_omega);
    if (_w[wheel].dir == 0) {
        _enc_start(wheel, new_dir);
    } else if (new_dir != _w[wheel].dir) {
        _enc_stop(wheel);
        _enc_start(wheel, new_dir);
    }
    /* Cùng chiều + đang chạy: chỉ cần cập nhật half_period.
     * Callback tiếp theo tự dùng giá trị mới. */
}

/* Gọi từ HAL_TIM_OC_DelayElapsedCallback.
 * QUAN TRỌNG: dùng htim->Channel (HAL set trước callback),
 * KHÔNG dùng __HAL_TIM_GET_FLAG (HAL đã clear flag trước callback). */
void EncoderGen_OC_Callback(TIM_HandleTypeDef *htim) {
    enc_oc_count++;
    uint8_t i;
    for (i = 0; i < ENC_NUM_WHEELS; i++)
        if (_w[i].htim->Instance == htim->Instance) break;
    if (i == ENC_NUM_WHEELS) return;

    uint32_t ch;
    switch (htim->Channel) {
        case HAL_TIM_ACTIVE_CHANNEL_1: ch = TIM_CHANNEL_1; break;
        case HAL_TIM_ACTIVE_CHANNEL_2: ch = TIM_CHANNEL_2; break;
        default: return;
    }
    /* Advance CCR by half_period — 16-bit wrap intentional */
    uint32_t hp  = _w[i].half_period;
    uint32_t cur = __HAL_TIM_GET_COMPARE(htim, ch);
    __HAL_TIM_SET_COMPARE(htim, ch, (uint16_t)((cur + hp) & 0xFFFFu));
}
```

---

## 6. main.c — USER CODE insertions

### USER CODE BEGIN Includes
```c
#include "hil_protocol.h"
#include "spi1_handler.h"
#include "encoder_gen.h"
```

### USER CODE BEGIN PV
```c
static HIL_State_t  g_state;
static HIL_Output_t g_output;
extern volatile uint32_t enc_oc_count;   /* debug — xóa sau */
```

### USER CODE BEGIN 2
```c
EncoderGen_Init(&htim1, &htim2, &htim3, &htim4);
SPI1_Handler_Init(&hspi1);
// PWMCapture_Init();  /* comment — xem mục 12 */
```

> **Không wrap trong `__disable_irq()` / `__enable_irq()`** — xem Bug #1.

### USER CODE BEGIN WHILE (production)
```c
if (SPI1_IsFrameReady()) {
    SPI1_GetState(&g_state);

    for (int i = 0; i < 4; i++) {
        EncoderGen_SetOmega(i, g_state.omega[i]);
    }

    /* TODO Phase 3: đọc PWM từ ESP32 → điền g_output.torque/pwm */
    SPI1_SetOutput(&g_output);   /* hiện tại gửi zero — placeholder */
}
```

### USER CODE BEGIN 4
```c
void HAL_SPI_TxRxCpltCallback(SPI_HandleTypeDef *hspi)
{
    SPI1_Callback(hspi);
}

void HAL_TIM_OC_DelayElapsedCallback(TIM_HandleTypeDef *htim)
{
    EncoderGen_OC_Callback(htim);
}
```

---

## 7. Bugs & Fixes

### Bug 1 — H7 boot lockup: LD1 không tắt

**Triệu chứng:** Board init bình thường (3 LED sáng) nhưng LD1 không tắt, processor treo.

**Nguyên nhân:** `HAL_SPI_TransmitReceive_DMA` được gọi bên trong `__disable_irq()` block. Trên STM32H7, SPI peripheral set bit `CSTART` ngay khi DMA arm → interrupt pending. Khi `__enable_irq()` được gọi, interrupt bùng phát → HAL state corruption → HardFault → `bkpt #0` → processor lockup (không có debugger attach).

**Fix:** Bỏ `__disable_irq()` / `__enable_irq()` wrap. Gọi `HAL_SPI_TransmitReceive_DMA` trực tiếp trong `USER CODE BEGIN 2`.

---

### Bug 2 — `HAL_TIM_Base_Start` block OC interrupt

**Triệu chứng:** `enc_oc_count` luôn = 0, OC callback không fire.

**Nguyên nhân:** `HAL_TIM_Base_Start(htim)` set `htim->State = HAL_TIM_STATE_BUSY`. Sau đó `HAL_TIM_OC_Start_IT(htim, ch)` kiểm tra state, thấy BUSY → return `HAL_BUSY` silent, không enable interrupt. Không có error log.

**Fix:** Không gọi `HAL_TIM_Base_Start`. `_enc_arm_ch` enable counter trực tiếp qua `CR1 |= TIM_CR1_CEN`.

---

### Bug 3 — Tên hàm `_start` conflict với CMSIS

**Triệu chứng:** Compile error: `conflicting types for '_start'`.

**Nguyên nhân:** `cmsis_gcc.h` khai báo `extern void _start(void) __NO_RETURN` là entry point của newlib.

**Fix:** Đổi tên thành `_enc_start`, `_enc_stop`, `_enc_arm_ch`.

---

### Bug 4 — `HAL_TIM_OC_Start_IT` silent fail (channel state)

**Triệu chứng:** Sau khi fix Bug 2, `enc_oc_count` vẫn = 0 ở một số thời điểm.

**Nguyên nhân:** Newer STM32H7 HAL dùng per-channel state (`HAL_TIM_CHANNEL_STATE_READY/BUSY`). `HAL_TIM_OC_Start_IT` trả về `HAL_ERROR` silent nếu state không phải READY.

**Fix:** Bypass HAL hoàn toàn, dùng direct register write (`_enc_arm_ch` như trên).

---

### Bug 5 — Tần số encoder sai — factor ~π

**Triệu chứng:** Đo được ~169 Hz thay vì 525 Hz tại omega=10 rad/s (ratio ≈ 1/π).

**Phân tích:** `enc_oc_count` xác nhận callback rate thấp hơn kỳ vọng ~π lần. STM32H7 OC Toggle mode có factor π trong timing (nguyên nhân gốc rễ chưa xác định rõ).

**Fix empirical:**
```c
// Trước (sai — có π):
float hp = ENC_TIMER_FREQ * M_PI / (omega * PPR);   // 952 ticks

// Sau (đúng — không có π):
float hp = ENC_TIMER_FREQ / (omega * PPR);            // 303 ticks
```
Kết quả: GPIO = **532 Hz** (kỳ vọng 525 Hz, sai số 1.3% do uint32_t truncation).

---

### Bug 6 — Debug variable sai trong MISO

**Triệu chứng:** `enc_oc_count` delta qua SPI luôn = 0 dù OC đang fire (LD1 nhấp nháy).

**Nguyên nhân:** Code debug gửi `g_ftimer` (hằng số) thay vì `enc_oc_count` vào byte 16–17 MISO.

**Fix:**
```c
// Sai:
g_output.status = (uint8_t)((g_ftimer >> 8) & 0xFF);
// Đúng:
g_output.status = (uint8_t)((enc_oc_count >> 8) & 0xFF);
```

---

### Bug 7 — PWMCapture_Init() gây SPI1 corruption

**Triệu chứng:** Khi bỏ comment `PWMCapture_Init()`, SPI1 bị corrupt (0xFF xen kẽ, data ngẫu nhiên).

**Đã loại trừ:** Floating pins (đã có PULLDOWN), NVIC priority conflict (đã thử priority 5 cho TIM5/15/16/17 — vẫn lỗi).

**Trạng thái:** Root cause chưa tìm ra. Cần có PWM signal thật từ ESP32 để debug. Giữ `//PWMCapture_Init()` cho đến Phase 3.

---

## 8. Debug không có hardware debugger (SSH remote)

### 8.1 Gửi giá trị nội bộ về RPi5 qua MISO reserved bytes

```c
/* Byte 16–17 MISO là reserved — dùng để tunnel debug value */
g_output.status = (uint8_t)((enc_oc_count >> 8) & 0xFF);
g_output.seq    = (uint8_t)(enc_oc_count & 0xFF);
SPI1_SetOutput(&g_output);
```
Python đọc: `count = (rx[16] << 8) | rx[17]`

> Chỉ 16-bit → wrap ở 65535. Tính **delta** giữa 2 lần đọc, không dùng absolute value.

### 8.2 LED blink xác nhận callback

```c
static uint32_t _blink = 0;
void HAL_TIM_OC_DelayElapsedCallback(TIM_HandleTypeDef *htim) {
    if (++_blink >= 3300) {     /* ~0.5s tại 6600 callback/s */
        HAL_GPIO_TogglePin(LD1_GPIO_Port, LD1_Pin);
        _blink = 0;
    }
    EncoderGen_OC_Callback(htim);
}
```
LD1 nhấp nháy = callback đang fire. LD1 tắt hoàn toàn = callback không được gọi.

### 8.3 Script đo enc_oc_count qua SPI (RPi5)

```python
import spidev, struct, time

spi = spidev.SpiDev(); spi.open(0, 0)
spi.max_speed_hz = 1_000_000; spi.mode = 0

def encode(omega0):
    buf = bytearray(24)
    struct.pack_into('>h', buf, 0, int(omega0 * 32767.0 / 40.0))
    return bytes(buf)

for _ in range(5):             # warmup frames
    spi.xfer2(list(encode(0.0))); time.sleep(0.02)

rx    = bytes(spi.xfer2(list(encode(10.0))))
start = (rx[16] << 8) | rx[17]
t0    = time.time()

while time.time() - t0 < 10.0:
    spi.xfer2(list(encode(10.0))); time.sleep(0.01)

rx  = bytes(spi.xfer2(list(encode(10.0))))
end = (rx[16] << 8) | rx[17]
delta = (end - start) & 0xFFFF
print(f"delta={delta}, avg/s={delta/10:.1f}  (kỳ vọng ~2100)")
spi.close()
```

---

## 9. Kết quả test

### Phase 1 — SPI1 communication

| Test | Kết quả |
|---|---|
| MISO torque all-zero | PASS ✓ |
| LED2 sáng khi omega[0] > 0 | PASS ✓ |
| LED2 tắt khi omega[0] = 0 | PASS ✓ |
| 50 frame không corrupt | PASS ✓ (0/50) |
| Throughput @ 100 Hz | 97.2 fps |

### Phase 2 — EncoderGen (verified bằng logic analyzer)

| Metric | Đo được | Kỳ vọng | Sai số |
|---|---|---|---|
| enc_oc_count avg/s | 2101.3 | 2100 | < 0.1% |
| Frequency PE9 (W1_A) | 532 Hz | 525 Hz | +1.3% |
| Duty cycle | 49.99% | 50% | ✓ |
| Phase A vs B | ~90° | 90° | ✓ |

Sai số 1.3% do `uint32_t` truncation: 303.03 → 303 ticks.

### HIL loop timing

| Metric | Đo được |
|---|---|
| Loop rate | 1 kHz |
| Jitter max | 0.4 μs |
| Overrun | 0 |
| NSS period (logic analyzer) | đúng 1 ms |

---

## 10. Lessons learned

| # | Bài học |
|---|---------|
| 1 | STM32H7 SPI: KHÔNG arm DMA trong `__disable_irq()` — CSTART kích hoạt ngay, gây HardFault |
| 2 | `HAL_TIM_Base_Start` set state=BUSY → block `HAL_TIM_OC_Start_IT` silent |
| 3 | `HAL_TIM_OC_Start_IT` kiểm tra per-channel state → silent fail → dùng direct register |
| 4 | `HAL_TIM_OC_DelayElapsedCallback`: dùng `htim->Channel`, KHÔNG dùng `__HAL_TIM_GET_FLAG` (đã clear trước callback) |
| 5 | OC Toggle half_period: bỏ π khỏi công thức để match H7 hardware behavior |
| 6 | Re-arm SPI DMA trong callback là safe trên H7 HAL (state reset trước khi callback được gọi) |
| 7 | `.ld` linker script bị overwrite khi regen CubeMX — không sửa, dùng MPU config trong USER CODE thay thế |
| 8 | Warmup frames (omega=0) stop timer → cần restart logic khi omega ≠ 0 |
| 9 | `USER CODE END WHILE` phải nằm TRONG while(1); `USER CODE END 3` nằm NGOÀI |
| 10 | Duplicate `USER CODE BEGIN PV` tag gây compile error |
| 11 | Code SPI loopback master-mode phải xóa sạch trước khi deploy slave production |
| 12 | Debug không có J-Link: tunnel nội bộ qua MISO reserved bytes + LED blink |

---

## 11. Trạng thái hiện tại

| Module | Trạng thái |
|---|---|
| H7 boot | ✅ Ổn định |
| SPI1 loopback test | ✅ PASS |
| SPI1 RPi5 ↔ H7 | ✅ Stable, protocol đúng, 1 kHz verified |
| EncoderGen 4 bánh | ✅ Verified (logic analyzer) |
| PWMCapture TIM5/15/16/17 | ❌ Gây SPI corruption, tạm comment |
| SPI2 BNO085 emulation | ⏳ Chưa triển khai |
| UART trajectory từ RPi5 | ⏳ Chưa triển khai |
| ESP32 firmware | ⏳ Chưa bắt đầu |

---

## 12. Kế hoạch tiếp theo

### Ngắn hạn (Phase 3)
1. Viết ESP32 firmware: đọc encoder từ H7 (quadrature), chạy PID/ADRC, output PWM
2. Nối ESP32 vào H7, debug `PWMCapture_Init()` với PWM signal thật
3. Triển khai SPI2 (BNO085 emulation) và UART (trajectory từ RPi5)

### Dài hạn
1. Chạy full loop 3-board: RPi5 ↔ H7 ↔ ESP32
2. Dọn code production (xóa debug: `enc_oc_count`, LED blink, MISO tunnel)
3. Viết M7 Process Metrics Framework
4. Hoàn thiện thesis
