/* PC stand-in for the H7 HAL header, only what encoder_gen.c uses (tools/tests/
 * test_h7_encoder_gen.py). Register bit positions follow the STM32H7 TIM layout;
 * SR keeps its rc_w0 meaning through encgen_sim_main.c (it ANDs what was written). */
#ifndef FAKE_STM32H7XX_HAL_H
#define FAKE_STM32H7XX_HAL_H

#include <stdint.h>

typedef struct {
    volatile uint32_t CR1, DIER, SR, EGR, CCMR1, CCER, CNT, CCR1, CCR2, BDTR;
} TIM_TypeDef;

typedef enum {
    HAL_TIM_ACTIVE_CHANNEL_CLEARED = 0,
    HAL_TIM_ACTIVE_CHANNEL_1 = 1,
    HAL_TIM_ACTIVE_CHANNEL_2 = 2
} HAL_TIM_ActiveChannel;

typedef struct {
    TIM_TypeDef *Instance;
    HAL_TIM_ActiveChannel Channel;
} TIM_HandleTypeDef;

#define TIM_CHANNEL_1     0x0u
#define TIM_CHANNEL_2     0x4u
#define TIM_CR1_CEN       (1u << 0)
#define TIM_DIER_CC1IE    (1u << 1)
#define TIM_DIER_CC2IE    (1u << 2)
#define TIM_SR_CC1IF      (1u << 1)
#define TIM_SR_CC2IF      (1u << 2)
#define TIM_CCER_CC1E     (1u << 0)
#define TIM_CCER_CC2E     (1u << 4)
#define TIM_CCMR1_OC1M_0  (1u << 4)
#define TIM_CCMR1_OC1M_1  (1u << 5)
#define TIM_CCMR1_OC1M_2  (1u << 6)
#define TIM_CCMR1_OC1M    (TIM_CCMR1_OC1M_0 | TIM_CCMR1_OC1M_1 | TIM_CCMR1_OC1M_2 | (1u << 16))
#define TIM_CCMR1_OC2M_0  (1u << 12)
#define TIM_CCMR1_OC2M_1  (1u << 13)
#define TIM_CCMR1_OC2M_2  (1u << 14)
#define TIM_CCMR1_OC2M    (TIM_CCMR1_OC2M_0 | TIM_CCMR1_OC2M_1 | TIM_CCMR1_OC2M_2 | (1u << 24))
#define TIM_BDTR_MOE      (1u << 15)
#define IS_TIM_BREAK_INSTANCE(t) (0)

#define __HAL_TIM_GET_COMPARE(h, ch) ((ch) == TIM_CHANNEL_1 ? (h)->Instance->CCR1 : (h)->Instance->CCR2)
#define __HAL_TIM_SET_COMPARE(h, ch, v) \
    do { if ((ch) == TIM_CHANNEL_1) (h)->Instance->CCR1 = (v); else (h)->Instance->CCR2 = (v); } while (0)

extern int fake_irq_disabled;
extern long fake_irq_sections;
static inline uint32_t __get_PRIMASK(void) { return (uint32_t)fake_irq_disabled; }
static inline void __disable_irq(void) { fake_irq_disabled = 1; fake_irq_sections++; }
static inline void __set_PRIMASK(uint32_t m) { fake_irq_disabled = (int)m; }

#endif
