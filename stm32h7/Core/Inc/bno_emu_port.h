#ifndef BNO_EMU_PORT_H
#define BNO_EMU_PORT_H

#include "stm32h7xx_hal.h"
#include "bno_emu.h"

/*
 * BNO085 emulator on the H7 (G2): thin layer between the plain-C core
 * bno_emu.c and the hardware. Pins (G0 v1 §5, .ioc):
 *   SPI4 slave mode 3, hardware NSS: PE12 SCK, PE13 MISO, PE14 MOSI, PE4 NSS
 *   PG2  INT (output, active low)        PG3  RST (input, EXTI3, pull-up)
 *   PE4  also EXTI4 (both edges, set up here by hand, not in CubeMX) so the
 *        H7 knows where each transaction starts and ends.
 * TIM6 interrupt at 10 kHz runs the core (poll, new plant sample, INT pin).
 * TIM6, EXTI3, EXTI4, SPI4 and its DMA all have NVIC priority 6, so these
 * handlers never preempt each other and the core needs no locking.
 *
 * Call order:
 *   1. BnoPort_Init(&hspi4, &htim6)        — in USER CODE BEGIN 2
 *   2. BnoPort_Tick()                      — from HAL_TIM_PeriodElapsedCallback (TIM6)
 *   3. BnoPort_ExtiCallback(pin)           — from HAL_GPIO_EXTI_Callback (PG3)
 *   4. BnoPort_SpiError(hspi)              — from HAL_SPI_ErrorCallback
 *   5. BnoPort_GetStats(&st)               — main loop, debug print
 * EXTI4_IRQHandler (PE4) is defined in bno_emu_port.c.
 */

typedef struct {
    bno_emu_stats_t core;          /* copy of the core counters              */
    uint8_t  state;                /* bno_emu_state_t                        */
    uint32_t period_us[BNO_EMU_N_SENSORS];  /* report periods the host set   */
    uint32_t reports[BNO_EMU_N_SENSORS];    /* reports built per sensor      */
    uint32_t xfers;                /* CS low..high transactions              */
    uint32_t arm_mismatch;         /* MISO armed != what the core offered    */
    uint32_t missed_edge;          /* CS edge handled late (both edges in one ISR) */
    uint32_t arm_fail;             /* HAL_SPI_TransmitReceive_DMA not OK     */
    uint32_t spi_errors;           /* HAL_SPI_ErrorCallback on SPI4          */
    uint32_t max_clocked;          /* longest transaction in bytes           */
    uint32_t samples;              /* plant samples given to the core        */
} BnoPort_Stats_t;

void BnoPort_Init        (SPI_HandleTypeDef *hspi, TIM_HandleTypeDef *htim);
void BnoPort_Tick        (void);
void BnoPort_ExtiCallback(uint16_t pin);
void BnoPort_SpiError    (SPI_HandleTypeDef *hspi);
void BnoPort_GetStats    (BnoPort_Stats_t *out);

#endif /* BNO_EMU_PORT_H */
