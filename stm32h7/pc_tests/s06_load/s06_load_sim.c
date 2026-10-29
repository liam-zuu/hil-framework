/* PC harness for stm32h7/Core/Src/s06_load.c (tools/tests/test_h7_s06_load.py).
 * Nested interrupts are played as nested C calls; the cycle counter is set by
 * hand at each entry and exit, so exclusive times are known exactly.
 * Prints "PASS <name>" / "FAIL <name>: <why>" and the report lines; exit 1 on
 * any failure. */
#include <stdio.h>
#include <string.h>
#include "stm32h7xx_hal.h"
#include "s06_load.h"

DWT_Type fake_dwt;
uint32_t SystemCoreClock = 480000000u;
uint32_t fake_tick_ms;
int fake_irq_disabled;

static int _fails;
static void check(int ok, const char *name, const char *why)
{
    if (ok) printf("PASS %s\n", name);
    else { printf("FAIL %s: %s\n", name, why); _fails++; }
}

/* handler bodies: set CYCCNT at entry, run nested ones, set CYCCNT at exit */
static void handler(S06_IrqId_t id, uint32_t t_in, uint32_t t_out, void (*nested)(void))
{
    fake_dwt.CYCCNT = t_in;
    S06_IRQ_ENTER();
    if (nested) nested();
    fake_dwt.CYCCNT = t_out;
    S06_IRQ_EXIT(id);
}

static void c_level(void) { handler(S06_IRQ_SPI1, 160, 170, NULL); }              /* 10 */
static void b_level(void) { handler(S06_IRQ_TIM5, 150, 190, c_level); }           /* 40 total, 30 excl */
static void b_after(void) { handler(S06_IRQ_EXTI4, 200, 260, NULL); }             /* 60 */
static void a_two(void) { b_level(); b_after(); }

extern uint32_t s06_test_get(S06_IrqId_t id, uint64_t *sum, uint32_t *n, uint32_t *max);

int main(void)
{
    s06_nest_cyc = 12345u;               /* garbage left by the main loop: must not matter */
    /* TIM6 preempted by TIM5, which is preempted by SPI1; then EXTI4 */
    handler(S06_IRQ_TIM6, 100, 300, a_two);
    uint64_t sum; uint32_t n, max;
    s06_test_get(S06_IRQ_TIM6, &sum, &n, &max);
    check(sum == 200u - 40u - 60u && n == 1, "outer_exclusive", "TIM6 excl != 200 - 40 - 60");
    s06_test_get(S06_IRQ_TIM5, &sum, &n, &max);
    check(sum == 30u, "middle_exclusive", "TIM5 excl != 40 - 10");
    s06_test_get(S06_IRQ_SPI1, &sum, &n, &max);
    check(sum == 10u, "inner_exclusive", "SPI1 excl != 10");
    s06_test_get(S06_IRQ_EXTI4, &sum, &n, &max);
    check(sum == 60u, "sibling_exclusive", "EXTI4 excl != 60");
    check(fake_irq_disabled == 0, "primask_restored", "PRIMASK left set");

    /* CYCCNT wraps inside a handler */
    handler(S06_IRQ_TIM1, 0xFFFFFFF0u, 0x10u, NULL);
    s06_test_get(S06_IRQ_TIM1, &sum, &n, &max);
    check(sum == 0x20u && max == 0x20u, "wrap_inside_handler", "wrong time across CYCCNT wrap");

    /* max and count over several calls */
    handler(S06_IRQ_TIM1, 1000, 1500, NULL);
    handler(S06_IRQ_TIM1, 2000, 2100, NULL);
    s06_test_get(S06_IRQ_TIM1, &sum, &n, &max);
    check(n == 3 && sum == 0x20u + 500u + 100u && max == 500u, "count_sum_max", "n/sum/max wrong");

    /* idle passes and the report window restart */
    for (int i = 0; i < 7; i++) S06Load_CountIdle();
    S06Load_SetIdleCal(1000u, 48000u);
    fake_tick_ms = 10000u;
    S06Load_Report();
    s06_test_get(S06_IRQ_TIM1, &sum, &n, &max);
    check(n == 0 && sum == 0 && max == 0, "report_restarts_window", "stats not cleared by the report");
    check(S06Load_IdleCount() == 7u, "idle_counted", "idle passes not counted");
    check(fake_irq_disabled == 0, "report_restores_primask", "PRIMASK left set");
    return _fails ? 1 : 0;
}
