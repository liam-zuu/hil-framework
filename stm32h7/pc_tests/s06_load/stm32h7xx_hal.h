/* PC stand-in for the H7 HAL header, only what s06_load.c uses
 * (tools/tests/test_h7_s06_load.py). The harness sets the cycle counter by
 * hand before each S06_IRQ_ENTER / S06_IRQ_EXIT to play nested interrupts. */
#ifndef FAKE_STM32H7XX_HAL_H
#define FAKE_STM32H7XX_HAL_H

#include <stdint.h>

typedef struct { volatile uint32_t CYCCNT; } DWT_Type;
extern DWT_Type fake_dwt;
#define DWT (&fake_dwt)

extern uint32_t SystemCoreClock;
extern uint32_t fake_tick_ms;
static inline uint32_t HAL_GetTick(void) { return fake_tick_ms; }

extern int fake_irq_disabled;
static inline uint32_t __get_PRIMASK(void) { return (uint32_t)fake_irq_disabled; }
static inline void __disable_irq(void) { fake_irq_disabled = 1; }
static inline void __enable_irq(void) { fake_irq_disabled = 0; }
static inline void __set_PRIMASK(uint32_t m) { fake_irq_disabled = (int)m; }
static inline void __NOP(void) { }

#endif
