#include "pwm_capture.h"
#include <string.h>
#include <stdio.h>

typedef struct {
    TIM_HandleTypeDef *htim;
    uint32_t           hal_ch;
    uint32_t           ccer_bit;   /* CCxE bit in CCER */
    uint32_t           ccer_pol;   /* CCxP bit in CCER — polarity toggle */
    uint32_t           dier_bit;   /* CCxIE bit in DIER */
    uint32_t           t_rise;
    uint32_t           period;
    uint32_t           width;
    uint8_t            state;      /* 0=WAIT_RISE, 1=WAIT_FALL */
    uint32_t           last_tick;
    float              duty;
    GPIO_TypeDef      *port;       /* input pin, read when no edge arrives */
    uint16_t           pin;
} PwmCh_t;

#define N_CHANNELS 8u
static PwmCh_t _ch[N_CHANNELS];
static volatile uint32_t _isr_count;   /* capture interrupts handled (debug) */

/* Arm the LPWM capture channels of TIM15 (2 channels) or TIM16/TIM17 (1 channel):
 * input filter 0xF, rising edge, counter on, stale flags cleared while the
 * channels are off, then interrupts enabled. NVIC stays off until main.c. */
static void _arm_lpwm_timer(TIM_HandleTypeDef *ht, uint8_t n_ch)
{
    TIM_TypeDef *t = ht->Instance;
    uint32_t ccer = TIM_CCER_CC1E;
    uint32_t pol  = TIM_CCER_CC1P | TIM_CCER_CC1NP;
    uint32_t dier = TIM_DIER_CC1IE;
    uint32_t sr   = TIM_SR_CC1IF | TIM_SR_CC1OF;

    t->DIER &= ~(TIM_DIER_CC1IE | TIM_DIER_CC2IE);
    t->CCER &= ~(TIM_CCER_CC1E | TIM_CCER_CC2E);
    t->CCMR1 = (t->CCMR1 & ~TIM_CCMR1_IC1F) | (0xFu << TIM_CCMR1_IC1F_Pos);
    if (n_ch == 2u) {
        t->CCMR1 = (t->CCMR1 & ~TIM_CCMR1_IC2F) | (0xFu << TIM_CCMR1_IC2F_Pos);
        ccer |= TIM_CCER_CC2E;
        pol  |= TIM_CCER_CC2P | TIM_CCER_CC2NP;
        dier |= TIM_DIER_CC2IE;
        sr   |= TIM_SR_CC2IF | TIM_SR_CC2OF;
    }
    t->CCER &= ~pol;            /* rising edge */
    t->CR1  |= TIM_CR1_CEN;
    t->SR    = ~sr;             /* rc_w0: clear capture and overcapture flags */
    t->CCER |= ccer;
    t->SR    = ~sr;
    t->DIER |= dier;
}

static uint32_t _hal_active_to_ch(HAL_TIM_ActiveChannel ac)
{
    switch (ac) {
        case HAL_TIM_ACTIVE_CHANNEL_1: return TIM_CHANNEL_1;
        case HAL_TIM_ACTIVE_CHANNEL_2: return TIM_CHANNEL_2;
        case HAL_TIM_ACTIVE_CHANNEL_3: return TIM_CHANNEL_3;
        case HAL_TIM_ACTIVE_CHANNEL_4: return TIM_CHANNEL_4;
        default:                        return 0xFFFFFFFFu;
    }
}

/*
 * Set polarity, re-enable CCER and DIER — direct register writes, ISR-safe.
 * falling=1 → CCxP=1 (falling edge), falling=0 → CCxP=0 (rising edge).
 * No LL dependency.
 */
static void _arm(PwmCh_t *c, uint8_t falling)
{
    if (falling)
        c->htim->Instance->CCER |=  c->ccer_pol;
    else
        c->htim->Instance->CCER &= ~c->ccer_pol;
    c->htim->Instance->CCER |= c->ccer_bit;
    c->htim->Instance->DIER |= c->dier_bit;
}

void PWMCapture_Init(TIM_HandleTypeDef *ht5,
                     TIM_HandleTypeDef *ht15,
                     TIM_HandleTypeDef *ht16,
                     TIM_HandleTypeDef *ht17)
{
    memset(_ch, 0, sizeof(_ch));
    printf("PWMCapture: assign channels\r\n");

    _ch[0].htim = ht5;  _ch[0].hal_ch = TIM_CHANNEL_1;
    _ch[1].htim = ht5;  _ch[1].hal_ch = TIM_CHANNEL_2;
    _ch[2].htim = ht5;  _ch[2].hal_ch = TIM_CHANNEL_3;
    _ch[3].htim = ht5;  _ch[3].hal_ch = TIM_CHANNEL_4;
    _ch[4].htim = ht15; _ch[4].hal_ch = TIM_CHANNEL_1;
    _ch[5].htim = ht15; _ch[5].hal_ch = TIM_CHANNEL_2;
    _ch[6].htim = ht16; _ch[6].hal_ch = TIM_CHANNEL_1;
    _ch[7].htim = ht17; _ch[7].hal_ch = TIM_CHANNEL_1;

    /* Input pins (tim.c, pwm_capture.h) */
    _ch[0].port = GPIOA; _ch[0].pin = GPIO_PIN_0;
    _ch[1].port = GPIOA; _ch[1].pin = GPIO_PIN_1;
    _ch[2].port = GPIOA; _ch[2].pin = GPIO_PIN_2;
    _ch[3].port = GPIOA; _ch[3].pin = GPIO_PIN_3;
    _ch[4].port = GPIOE; _ch[4].pin = GPIO_PIN_5;
    _ch[5].port = GPIOE; _ch[5].pin = GPIO_PIN_6;
    _ch[6].port = GPIOF; _ch[6].pin = GPIO_PIN_6;
    _ch[7].port = GPIOF; _ch[7].pin = GPIO_PIN_7;

    printf("PWMCapture: resolve bits\r\n");

    /* Resolve ccer_bit, ccer_pol, dier_bit */
    for (uint8_t i = 0; i < N_CHANNELS; i++) {
        _ch[i].state = 0;
        switch (_ch[i].hal_ch) {
            case TIM_CHANNEL_1:
                _ch[i].ccer_bit = TIM_CCER_CC1E;
                _ch[i].ccer_pol = TIM_CCER_CC1P;
                _ch[i].dier_bit = TIM_DIER_CC1IE;
                break;
            case TIM_CHANNEL_2:
                _ch[i].ccer_bit = TIM_CCER_CC2E;
                _ch[i].ccer_pol = TIM_CCER_CC2P;
                _ch[i].dier_bit = TIM_DIER_CC2IE;
                break;
            case TIM_CHANNEL_3:
                _ch[i].ccer_bit = TIM_CCER_CC3E;
                _ch[i].ccer_pol = TIM_CCER_CC3P;
                _ch[i].dier_bit = TIM_DIER_CC3IE;
                break;
            case TIM_CHANNEL_4:
                _ch[i].ccer_bit = TIM_CCER_CC4E;
                _ch[i].ccer_pol = TIM_CCER_CC4P;
                _ch[i].dier_bit = TIM_DIER_CC4IE;
                break;
            default: break;
        }
    }

    /* ── TIM5 CH1-4 ────────────────────────────────────────────────────────
     * Init order: CCER → CEN → disable CC2E to clear stuck CC2IF → SR=0
     *             → re-enable CC2E → DIER (NVIC: main.c, after this function)
     * Input filter 0xF on all 4 channels, set here with the channels off:
     * tim.c (CubeMX) leaves ICFilter = 0, and the 18/06 fix for the PA1
     * interrupt storm was never committed (26/10). */
    ht5->Instance->CCER = 0;
    ht5->Instance->CCMR1 = (ht5->Instance->CCMR1 & ~(TIM_CCMR1_IC1F | TIM_CCMR1_IC2F))
                         | (0xFu << TIM_CCMR1_IC1F_Pos) | (0xFu << TIM_CCMR1_IC2F_Pos);
    ht5->Instance->CCMR2 = (ht5->Instance->CCMR2 & ~(TIM_CCMR2_IC3F | TIM_CCMR2_IC4F))
                         | (0xFu << TIM_CCMR2_IC3F_Pos) | (0xFu << TIM_CCMR2_IC4F_Pos);
    ht5->Instance->CCER  = (TIM_CCER_CC1E | TIM_CCER_CC2E |
                             TIM_CCER_CC3E | TIM_CCER_CC4E);
    ht5->Instance->CR1  |= TIM_CR1_CEN;
    ht5->Instance->CCER &= ~TIM_CCER_CC2E;  /* disable CC2E tạm để clear CC2IF */
    ht5->Instance->SR    = 0;
    ht5->Instance->CCER |= TIM_CCER_CC2E;   /* re-enable CC2E */
    ht5->Instance->DIER  = (TIM_DIER_CC1IE | TIM_DIER_CC2IE |
                             TIM_DIER_CC3IE | TIM_DIER_CC4IE);

    printf("PWMCapture: TIM5 armed (filter 0xF, NVIC still off)\r\n");

    /* TIM15 CH1-2, TIM16 CH1, TIM17 CH1 (LPWM wheels 1-4).
     * Same treatment as TIM5 (H7 hang after "resolve bits", 25/10):
     *   input filter 0xF (CubeMX left ICFilter = 0, floating/noisy pins storm),
     *   rising edge, start counter, drop stale flags with CCxE off, then enable
     *   interrupts. main.c enables these NVIC lines only after this function. */
    _arm_lpwm_timer(ht15, 2u);
    printf("PWMCapture: TIM15 armed\r\n");
    _arm_lpwm_timer(ht16, 1u);
    _arm_lpwm_timer(ht17, 1u);
    printf("PWMCapture: TIM16/17 armed\r\n");
}

/* Number of capture interrupts handled since boot (debug: interrupt load). */
uint32_t PWMCapture_GetIsrCount(void)
{
    return _isr_count;
}

/* Duty of one channel. Without an edge for PWM_TIMEOUT_MS the pin is held at a
 * constant level (0 % or 100 %): return that level instead of 0 (bug H7-PWM-LEVEL,
 * 22/10). The ESP32 holds the inactive BTS7960 input HIGH (motor.c motor_set). */
static float _chan_duty(const PwmCh_t *c, uint32_t now)
{
    if ((now - c->last_tick) < PWM_TIMEOUT_MS)
        return c->duty;
    return (HAL_GPIO_ReadPin(c->port, c->pin) == GPIO_PIN_SET) ? 1.0f : 0.0f;
}

void PWMCapture_GetDuty(float *duty_out)
{
    uint32_t now = HAL_GetTick();
    for (uint8_t w = 0; w < PWM_NUM_WHEELS; w++) {
        float rpwm = _chan_duty(&_ch[w], now);
        float lpwm = _chan_duty(&_ch[w + 4], now);
        duty_out[w] = rpwm - lpwm;
    }
}

void PWMCapture_IC_Callback(TIM_HandleTypeDef *htim)
{
//	printf("IC!\r\n");  // thêm tạm

    _isr_count++;
    uint32_t trig_ch = _hal_active_to_ch(htim->Channel);
    if (trig_ch == 0xFFFFFFFFu) return;

    PwmCh_t *c = NULL;
    for (uint8_t i = 0; i < N_CHANNELS; i++) {
        if (_ch[i].htim->Instance == htim->Instance &&
            _ch[i].hal_ch == trig_ch) {
            c = &_ch[i];
            break;
        }
    }
    if (!c) return;

    uint32_t cap = HAL_TIM_ReadCapturedValue(htim, trig_ch);
    c->last_tick = HAL_GetTick();

    if (c->state == 0) {
        /* WAIT_RISE: got rising edge */
        if (c->t_rise != 0)
            c->period = (uint16_t)(cap - (uint16_t)c->t_rise);
        c->t_rise = cap;
        c->state  = 1;
        _arm(c, 1);  /* switch to falling */
    } else {
        /* WAIT_FALL: got falling edge */
        c->width = (uint16_t)(cap - (uint16_t)c->t_rise);
        c->state = 0;
        _arm(c, 0);  /* switch to rising */
        if (c->period > 0) {
            float d = (float)c->width / (float)c->period;
            if (d > 1.0f) d = 1.0f;
            if (d < 0.0f) d = 0.0f;
            c->duty = c->duty * 0.75f + d * 0.25f;
        }
    }
}
