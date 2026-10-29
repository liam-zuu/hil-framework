/* s06_load.c — S06 T10 H7 load: exclusive time per IRQ handler, idle passes
 * of the main loop (see s06_load.h). Printed every 10 s from the main loop. */
#include "s06_load.h"
#include <stdio.h>
#include <string.h>

typedef struct {
    uint64_t sum;          /* exclusive cycles in the window */
    uint32_t n;            /* calls                          */
    uint32_t max;          /* largest exclusive time, cycles */
} _irq_stat_t;

volatile uint32_t s06_nest_cyc;
static _irq_stat_t      _st[S06_IRQ_N];
static volatile uint32_t _idle;
static uint32_t _win_ms0, _win_idle0;
static uint32_t _ovh_cyc;                   /* one ENTER/EXIT pair, empty body */
static uint32_t _cal_n, _cal_cyc;           /* idle calibration */

static const char *const _name[S06_IRQ_N] = {
    "SysTick", "EXTI3", "EXTI4", "DMA1_S0", "DMA1_S1", "DMA1_S2", "DMA1_S3",
    "TIM1", "TIM2", "TIM3", "TIM4", "SPI1", "TIM8_13", "TIM5", "TIM6", "SPI4",
    "TIM15", "TIM16", "TIM17", "USART3",
};

/* Called with interrupts masked (inside S06_IRQ_EXIT) */
void s06_irq_account(S06_IrqId_t id, uint32_t excl_cyc)
{
    _irq_stat_t *s = &_st[id];
    s->sum += excl_cyc;
    s->n++;
    if (excl_cyc > s->max) s->max = excl_cyc;
}

#ifdef S06_PC_TEST
/* PC harness only (stm32h7/pc_tests/s06_load) */
uint32_t s06_test_get(S06_IrqId_t id, uint64_t *sum, uint32_t *n, uint32_t *max)
{
    *sum = _st[id].sum; *n = _st[id].n; *max = _st[id].max;
    return 0;
}
#endif

void S06Load_CountIdle(void) { _idle++; }
uint32_t S06Load_IdleCount(void) { return _idle; }

void S06Load_CalibrateOverhead(void)
{
    uint32_t pm = __get_PRIMASK();
    __disable_irq();
    uint32_t t0 = DWT->CYCCNT;
    for (int i = 0; i < 1000; i++) {
        S06_IRQ_ENTER();
        __NOP();
        S06_IRQ_EXIT(S06_IRQ_SYSTICK);
    }
    _ovh_cyc = (DWT->CYCCNT - t0) / 1000u;
    memset(&_st[S06_IRQ_SYSTICK], 0, sizeof _st[0]);
    __set_PRIMASK(pm);
}

void S06Load_SetIdleCal(uint32_t n_pass, uint32_t cycles)
{
    _cal_n = n_pass;
    _cal_cyc = cycles;
}

void S06Load_Report(void)
{
    static _irq_stat_t snap[S06_IRQ_N];
    uint32_t pm = __get_PRIMASK();
    __disable_irq();
    uint32_t now_ms = HAL_GetTick();
    uint32_t idle = _idle;
    memcpy(snap, _st, sizeof snap);
    memset(_st, 0, sizeof _st);
    __set_PRIMASK(pm);

    /* window from the HAL ms tick: 10 s is longer than one CYCCNT wrap (8.9 s) */
    uint32_t win_ms = now_ms - _win_ms0;
    uint32_t n_idle = idle - _win_idle0;
    _win_ms0 = now_ms;
    _win_idle0 = idle;
    const uint32_t cpu = SystemCoreClock / 1000000u;
    uint64_t isr_sum = 0;
    for (int i = 0; i < S06_IRQ_N; i++) isr_sum += snap[i].sum;
    /* idle time estimate: passes x calibrated cycles per pass */
    uint64_t idle_cyc = (_cal_n > 0u) ? (uint64_t)n_idle * _cal_cyc / _cal_n : 0u;
    printf("T10: win_ms=%lu isr_us=%lu idle_pass=%lu idle_us_est=%lu cal=%lu/%lu ovh_cyc=%lu\r\n",
           (unsigned long)win_ms, (unsigned long)(isr_sum / cpu), (unsigned long)n_idle,
           (unsigned long)(idle_cyc / cpu), (unsigned long)_cal_n, (unsigned long)_cal_cyc,
           (unsigned long)_ovh_cyc);
    for (int i = 0; i < S06_IRQ_N; i++) {
        if (snap[i].n == 0u) continue;
        printf("T10 %-7s n=%lu excl_us=%lu max_us=%lu.%02lu\r\n", _name[i],
               (unsigned long)snap[i].n, (unsigned long)(snap[i].sum / cpu),
               (unsigned long)(snap[i].max / cpu), (unsigned long)((snap[i].max % cpu) * 100u / cpu));
    }
}
