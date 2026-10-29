#ifndef S06_LOAD_H
#define S06_LOAD_H

#include "stm32h7xx_hal.h"

/*
 * S06 T10 (R22): H7 load. Every IRQ handler is wrapped with S06_IRQ_ENTER() /
 * S06_IRQ_EXIT(id) (first and last statement of the handler, same scope).
 * Exclusive time = handler time minus the time of handlers that preempted it:
 * each handler zeroes a shared "nested" accumulator on entry, subtracts what
 * the preempting handlers added to it on exit, then hands its own total to
 * the handler it preempted. The few bookkeeping instructions run with
 * interrupts masked (a measurement cost, spec §4.1; calibrated at boot by
 * S06Load_CalibrateOverhead). Interrupt entry/exit (stacking, tail-chaining)
 * is not measured (spec T10: reported as "not measured").
 */

typedef enum {
    S06_IRQ_SYSTICK = 0, S06_IRQ_EXTI3, S06_IRQ_EXTI4,
    S06_IRQ_DMA1_S0, S06_IRQ_DMA1_S1, S06_IRQ_DMA1_S2, S06_IRQ_DMA1_S3,
    S06_IRQ_TIM1, S06_IRQ_TIM2, S06_IRQ_TIM3, S06_IRQ_TIM4,
    S06_IRQ_SPI1, S06_IRQ_TIM8_13, S06_IRQ_TIM5, S06_IRQ_TIM6, S06_IRQ_SPI4,
    S06_IRQ_TIM15, S06_IRQ_TIM16, S06_IRQ_TIM17, S06_IRQ_USART3,
    S06_IRQ_N
} S06_IrqId_t;

extern volatile uint32_t s06_nest_cyc;
void s06_irq_account(S06_IrqId_t id, uint32_t excl_cyc);

#define S06_IRQ_ENTER()                                   \
    uint32_t _s06_pm = __get_PRIMASK();                   \
    __disable_irq();                                      \
    uint32_t _s06_t0 = DWT->CYCCNT;                       \
    uint32_t _s06_saved = s06_nest_cyc;                   \
    s06_nest_cyc = 0;                                     \
    __set_PRIMASK(_s06_pm)

#define S06_IRQ_EXIT(id)                                  \
    do {                                                  \
        __disable_irq();                                  \
        uint32_t _s06_dt = DWT->CYCCNT - _s06_t0;         \
        s06_irq_account((id), _s06_dt - s06_nest_cyc);    \
        s06_nest_cyc = _s06_saved + _s06_dt;              \
        __set_PRIMASK(_s06_pm);                           \
    } while (0)

/* Main loop: one idle pass (nothing to do) */
void     S06Load_CountIdle(void);
uint32_t S06Load_IdleCount(void);
/* Boot, interrupts masked: cycles of one ENTER/EXIT pair around an empty
 * body (spec T10 "handler with / without DWT"). */
void     S06Load_CalibrateOverhead(void);
/* Idle-pass calibration (W10): main.c runs the real loop for a short time
 * with every interrupt masked before the main run and reports passes and
 * cycles here. */
void     S06Load_SetIdleCal(uint32_t n_pass, uint32_t cycles);
/* Main loop, every 10 s: prints and restarts the window. */
void     S06Load_Report(void);

#endif /* S06_LOAD_H */
