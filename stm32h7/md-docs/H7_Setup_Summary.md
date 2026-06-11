# Nucleo H753ZI — CubeMX Setup & Code Changes Summary

## 1. Clock Tree
- **HSE:** Crystal/Ceramic Resonator (25MHz trên Nucleo-144)
- **PLL:** PLLM=5, PLLN=192, PLLP=2 → SYSCLK = 480MHz
- **Dividers:** HPRE=/2 → AHB=240MHz, APB1=/2=120MHz, APB2=/2=120MHz
- **Voltage scale:** PWR_REGULATOR_VOLTAGE_SCALE0 (bắt buộc cho 480MHz)
- **Flash latency:** FLASH_LATENCY_4 (bắt buộc cho 480MHz)

## 2. System Core
| Peripheral | Setting |
|---|---|
| SYS → Debug | Serial Wire |
| SYS → Timebase | TIM6 (không dùng SysTick) |
| RCC → HSE | Crystal/Ceramic Resonator |
| RCC → LSE | Disable |

## 3. GPIO
| Pin | Label | Mode | Pull |
|---|---|---|---|
| PB0 | LD1 | GPIO_Output | No pull |
| PE1 | LD2 | GPIO_Output | No pull |
| PB14 | LD3 | GPIO_Output | No pull |
| PD0 | SYNC_IN | GPIO_EXTI0 Rising edge | Pull-down |
| PD1 | BNO_HINT | GPIO_Output | No pull |
| PG0-3 | W1-4_REN | GPIO_Input | No pull |
| PC8-9 | W1-2_LEN | GPIO_Input | No pull |
| PA8-9 | W3-4_LEN | GPIO_Input | No pull |

## 4. SPI1 — RPi5 Link
| Parameter | Value |
|---|---|
| Mode | Full-Duplex Slave |
| Hardware NSS | Hardware NSS Input Signal |
| Data Size | 8 Bits |
| Frame Format | Motorola |
| First Bit | MSB First |
| CPOL | Low |
| CPHA | 1 Edge |
| DMA RX | DMA1 Stream0, Peripheral→Memory, Byte |
| DMA TX | DMA1 Stream1, Memory→Peripheral, Byte |
| NVIC | SPI1 global interrupt ✅ |

**Pins:** PA4=NSS, PA5=SCK, PA6=MISO, PA7=MOSI

## 5. SPI2 — BNO085 Emulate
Cấu hình y hệt SPI1:
| Parameter | Value |
|---|---|
| Mode | Full-Duplex Slave |
| Hardware NSS | Hardware NSS Input Signal |
| Data Size | 8 Bits |
| CPOL | Low |
| CPHA | 1 Edge |
| DMA RX | DMA1 Stream2, Byte |
| DMA TX | DMA1 Stream3, Byte |
| NVIC | SPI2 global interrupt ✅ |

**Pins:** PC1=MOSI, PC2=MISO, PB10=SCK, PB12=NSS

## 6. Timers — Encoder Generation (TIM1-4)
Cấu hình giống nhau cho TIM1, TIM2, TIM3, TIM4:
| Parameter | Value |
|---|---|
| CH1 | Output Compare CH1 — Toggle on match |
| CH2 | Output Compare CH2 — Toggle on match |
| Prescaler | 239 → timer clock = 2MHz |
| Counter Period | 65535 |
| OC Pulse | 1000 (initial) |
| OC Polarity | High |
| NVIC | TIMx global interrupt ✅ |

**Pins:**
```
TIM1: PE9=CH1 (W1_A), PE11=CH2 (W1_B)
TIM2: PA5=CH1 (W2_A), PA15=CH2 (W2_B)  ← verify lại
TIM3: PA6=CH1 (W3_A), PB5=CH2 (W3_B)   ← verify lại
TIM4: PD12=CH1 (W4_A), PD13=CH2 (W4_B) ← verify lại
```

## 7. Timers — PWM Capture (TIM5, TIM15, TIM16, TIM17)
| Timer | Channels | Role |
|---|---|---|
| TIM5 | CH1-4 | RPWM wheel 1-4 |
| TIM15 | CH1-2 | LPWM wheel 1-2 |
| TIM16 | CH1 | LPWM wheel 3 |
| TIM17 | CH1 | LPWM wheel 4 |

Cấu hình cho tất cả:
| Parameter | Value |
|---|---|
| Mode | Input Capture direct mode |
| Prescaler | **239** (phải set thủ công trong USER CODE cho TIM15/16/17 vì CubeMX không generate đúng) |
| Counter Period | 65535 |
| IC Polarity | Rising Edge |
| IC Selection | Direct |
| NVIC | TIMx global interrupt ✅ (TIM5 phải tick thủ công) |

**Pins:**
```
TIM5:  PA0=CH1, PA1=CH2, PA2=CH3, PA3=CH4
TIM15: PE5=CH1, PE6=CH2
TIM16: PF6=CH1
TIM17: PF7=CH1
```

---

## Những thay đổi so với CubeMX default

### A. TIM15/16/17 Prescaler — sửa thủ công trong USER CODE
CubeMX generate `Prescaler = 0` cho TIM15/16/17. Phải sửa thủ công:
```c
/* USER CODE BEGIN TIM15_Init 1 */
htim15.Init.Prescaler = 239;
/* USER CODE END TIM15_Init 1 */

/* USER CODE BEGIN TIM16_Init 1 */
htim16.Init.Prescaler = 239;
/* USER CODE END TIM16_Init 1 */

/* USER CODE BEGIN TIM17_Init 1 */
htim17.Init.Prescaler = 239;
/* USER CODE END TIM17_Init 1 */
```

### B. MPU Config — thêm Region 1 cho DMA buffer
CubeMX chỉ generate Region 0 (default). Phải thêm Region 1 trong `MPU_Config()`:
```c
/* Region 1 — RAM_D2: non-cacheable cho DMA buffer */
MPU_InitStruct.Number = MPU_REGION_NUMBER1;
MPU_InitStruct.BaseAddress = 0x30000000;
MPU_InitStruct.Size = MPU_REGION_SIZE_32KB;
MPU_InitStruct.SubRegionDisable = 0x00;
MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL1;
MPU_InitStruct.AccessPermission = MPU_REGION_FULL_ACCESS;
MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
MPU_InitStruct.IsShareable = MPU_ACCESS_NOT_SHAREABLE;
MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;
HAL_MPU_ConfigRegion(&MPU_InitStruct);
```

### C. SPI DMA Buffer — đặt ở RAM_D2
Buffer mặc định đặt ở DTCM (default stack/heap) — DMA không access được. Phải dùng:
```c
__attribute__((section(".RAM_D2"))) uint8_t spi1_rx[24];
__attribute__((section(".RAM_D2"))) uint8_t spi1_tx[24];
__attribute__((section(".RAM_D2"))) uint8_t spi2_rx[64];
__attribute__((section(".RAM_D2"))) uint8_t spi2_tx[64];
```

### D. TIM16/17 — cần HAL_TIM_Base_Start trước IC
CubeMX generate `HAL_TIM_Base_Init` cho TIM16/17 nhưng không start base. Phải thêm:
```c
HAL_TIM_Base_Start(&htim16);
HAL_TIM_IC_Start_IT(&htim16, TIM_CHANNEL_1);
HAL_TIM_Base_Start(&htim17);
HAL_TIM_IC_Start_IT(&htim17, TIM_CHANNEL_1);
```

### E. PWM Capture Callback — dùng LL thay HAL
`HAL_TIM_IC_ConfigChannel` không an toàn trong ISR (HAL state machine không re-entrant).
Thay bằng `LL_TIM_IC_SetPolarity()` — 1 register write, ISR-safe:
```c
// Thay vì HAL_TIM_IC_ConfigChannel + HAL_TIM_IC_Start_IT:
LL_TIM_IC_SetPolarity(htim->Instance, c->ll_ch, LL_TIM_IC_POLARITY_FALLING);
```

### F. SysTick_Handler — thêm HAL_IncTick
CubeMX để trống SysTick_Handler khi dùng TIM6 làm timebase. Thêm vào:
```c
void SysTick_Handler(void)
{
    HAL_IncTick();
}
```

---

## File structure tự tạo (không bị CubeMX xóa)
```
Core/Inc/spi1_handler.h
Core/Inc/encoder_gen.h
Core/Inc/pwm_capture.h
Core/Src/spi1_handler.c
Core/Src/encoder_gen.c
Core/Src/pwm_capture.c
```
