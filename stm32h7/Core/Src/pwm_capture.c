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
     *             → re-enable CC2E → DIER → NVIC enable
     * ICFilter=0xF in CubeMX required to suppress PA1 floating noise. */
    ht5->Instance->CCER = 0;
    ht5->Instance->CCER  = (TIM_CCER_CC1E | TIM_CCER_CC2E |
                             TIM_CCER_CC3E | TIM_CCER_CC4E);
    ht5->Instance->CR1  |= TIM_CR1_CEN;
    ht5->Instance->CCER &= ~TIM_CCER_CC2E;  /* disable CC2E tạm để clear CC2IF */
    ht5->Instance->SR    = 0;
    ht5->Instance->CCER |= TIM_CCER_CC2E;   /* re-enable CC2E */
    ht5->Instance->DIER  = (TIM_DIER_CC1IE | TIM_DIER_CC2IE |
                             TIM_DIER_CC3IE | TIM_DIER_CC4IE);
    NVIC->ISER[TIM5_IRQn >> 5] = (1u << (TIM5_IRQn & 0x1F));

    /* TIM15 CH1-2 (LPWM wheels 1-2): was never started (bug H7-TIM15-ARM, 22/10).
     * Rising edge first, start counter, drop stale flags, then enable interrupts.
     * main.c enables the TIM15 NVIC line after this function. */
    ht15->Instance->CCER &= ~(TIM_CCER_CC1P | TIM_CCER_CC2P);
    ht15->Instance->CCER |=  (TIM_CCER_CC1E | TIM_CCER_CC2E);
    ht15->Instance->CR1  |=  TIM_CR1_CEN;
    ht15->Instance->SR    =  0;
    ht15->Instance->DIER |=  (TIM_DIER_CC1IE | TIM_DIER_CC2IE);

    /* TIM16 CH1 */
    ht16->Instance->CCER &= ~TIM_CCER_CC1P;
    ht16->Instance->CCER |=  TIM_CCER_CC1E;
    ht16->Instance->CR1  |= TIM_CR1_CEN;
    ht16->Instance->DIER |=  TIM_DIER_CC1IE;

    /* TIM17 CH1 */
    ht17->Instance->CCER &= ~TIM_CCER_CC1P;
    ht17->Instance->CCER |=  TIM_CCER_CC1E;
    ht17->Instance->CR1  |= TIM_CR1_CEN;
    ht17->Instance->DIER |=  TIM_DIER_CC1IE;

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
