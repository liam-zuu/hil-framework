///*
// * pwm_capture.c — RPWM/LPWM duty cycle measurement via TIM Input Capture
// *
// * Hardware:
// *   TIM5  CH1-4 → RPWM wheels 1-4  (PA0, PA1, PA2, PA3)
// *   TIM15 CH1-2 → LPWM wheels 1-2  (PE5, PE6)
// *   TIM16 CH1   → LPWM wheel 3     (PF6)
// *   TIM17 CH1   → LPWM wheel 4     (PF7)
// *
// * _arm() sets CCER + DIER on every call — ISR-safe direct register writes.
// * LL_TIM_IC_SetPolarity toggles edge polarity for rising→falling→rising cycle.
// */
//
//#include "pwm_capture.h"
//#include <string.h>
//
//typedef struct {
//    TIM_HandleTypeDef *htim;
//    uint32_t           hal_ch;
//    uint32_t           ll_ch;
//    uint32_t           ccer_bit;   /* CCxE bit in CCER */
//    uint32_t           dier_bit;   /* CCxIE bit in DIER */
//    uint32_t           t_rise;
//    uint32_t           period;
//    uint32_t           width;
//    uint8_t            state;      /* 0=WAIT_RISE, 1=WAIT_FALL */
//    uint32_t           last_tick;
//    float              duty;
//} PwmCh_t;
//
//#define N_CHANNELS 8u
//static PwmCh_t _ch[N_CHANNELS];
//
//static uint32_t _ll_ch(uint32_t hal_ch)
//{
//    switch (hal_ch) {
//        case TIM_CHANNEL_1: return LL_TIM_CHANNEL_CH1;
//        case TIM_CHANNEL_2: return LL_TIM_CHANNEL_CH2;
//        case TIM_CHANNEL_3: return LL_TIM_CHANNEL_CH3;
//        case TIM_CHANNEL_4: return LL_TIM_CHANNEL_CH4;
//        default:            return LL_TIM_CHANNEL_CH1;
//    }
//}
//
//static uint32_t _hal_active_to_ch(HAL_TIM_ActiveChannel ac)
//{
//    switch (ac) {
//        case HAL_TIM_ACTIVE_CHANNEL_1: return TIM_CHANNEL_1;
//        case HAL_TIM_ACTIVE_CHANNEL_2: return TIM_CHANNEL_2;
//        case HAL_TIM_ACTIVE_CHANNEL_3: return TIM_CHANNEL_3;
//        case HAL_TIM_ACTIVE_CHANNEL_4: return TIM_CHANNEL_4;
//        default:                        return 0xFFFFFFFFu;
//    }
//}
//
///* Set polarity, re-enable CCER and DIER — all direct register, ISR-safe */
//static void _arm(PwmCh_t *c, uint32_t polarity)
//{
//    LL_TIM_IC_SetPolarity(c->htim->Instance, c->ll_ch, polarity);
//    c->htim->Instance->CCER |= c->ccer_bit;
//    c->htim->Instance->DIER |= c->dier_bit;
//}
//
//void PWMCapture_Init(TIM_HandleTypeDef *ht5,
//                     TIM_HandleTypeDef *ht15,
//                     TIM_HandleTypeDef *ht16,
//                     TIM_HandleTypeDef *ht17)
//{
//    memset(_ch, 0, sizeof(_ch));
//
//    _ch[0].htim = ht5;  _ch[0].hal_ch = TIM_CHANNEL_1;
//    _ch[1].htim = ht5;  _ch[1].hal_ch = TIM_CHANNEL_2;
//    _ch[2].htim = ht5;  _ch[2].hal_ch = TIM_CHANNEL_3;
//    _ch[3].htim = ht5;  _ch[3].hal_ch = TIM_CHANNEL_4;
//    _ch[4].htim = ht15; _ch[4].hal_ch = TIM_CHANNEL_1;
//    _ch[5].htim = ht15; _ch[5].hal_ch = TIM_CHANNEL_2;
//    _ch[6].htim = ht16; _ch[6].hal_ch = TIM_CHANNEL_1;
//    _ch[7].htim = ht17; _ch[7].hal_ch = TIM_CHANNEL_1;
//
//    /* Resolve ll_ch, ccer_bit, dier_bit */
//    for (uint8_t i = 0; i < N_CHANNELS; i++) {
//        _ch[i].ll_ch = _ll_ch(_ch[i].hal_ch);
//        _ch[i].state = 0;
//        switch (_ch[i].hal_ch) {
//            case TIM_CHANNEL_1:
//                _ch[i].ccer_bit = TIM_CCER_CC1E;
//                _ch[i].dier_bit = TIM_DIER_CC1IE;
//                break;
//            case TIM_CHANNEL_2:
//                _ch[i].ccer_bit = TIM_CCER_CC2E;
//                _ch[i].dier_bit = TIM_DIER_CC2IE;
//                break;
//            case TIM_CHANNEL_3:
//                _ch[i].ccer_bit = TIM_CCER_CC3E;
//                _ch[i].dier_bit = TIM_DIER_CC3IE;
//                break;
//            case TIM_CHANNEL_4:
//                _ch[i].ccer_bit = TIM_CCER_CC4E;
//                _ch[i].dier_bit = TIM_DIER_CC4IE;
//                break;
//            default: break;
//        }
//    }
//
//    /* Enable all channels: CCER first, then DIER, then start counter */
//    /* TIM5 CH1-4 */
//    ht5->Instance->CCER |= (TIM_CCER_CC1E | TIM_CCER_CC2E |
//                             TIM_CCER_CC3E | TIM_CCER_CC4E);
//    ht5->Instance->DIER |= (TIM_DIER_CC1IE | TIM_DIER_CC2IE |
//                             TIM_DIER_CC3IE | TIM_DIER_CC4IE);
//    ht5->Instance->CR1  |= TIM_CR1_CEN;
//
//    /* TIM15 CH1-2 */
//    ht15->Instance->CCER |= (TIM_CCER_CC1E | TIM_CCER_CC2E);
//    ht15->Instance->DIER |= (TIM_DIER_CC1IE | TIM_DIER_CC2IE);
//    ht15->Instance->CR1  |= TIM_CR1_CEN;
//
//    /* TIM16 CH1 */
//    ht16->Instance->CCER |= TIM_CCER_CC1E;
//    ht16->Instance->DIER |= TIM_DIER_CC1IE;
//    ht16->Instance->CR1  |= TIM_CR1_CEN;
//
//    /* TIM17 CH1 */
//    ht17->Instance->CCER |= TIM_CCER_CC1E;
//    ht17->Instance->DIER |= TIM_DIER_CC1IE;
//    ht17->Instance->CR1  |= TIM_CR1_CEN;
//}
//
//void PWMCapture_GetDuty(float *duty_out)
//{
//    uint32_t now = HAL_GetTick();
//    for (uint8_t w = 0; w < PWM_NUM_WHEELS; w++) {
//        float rpwm = 0.0f;
//        float lpwm = 0.0f;
//        if ((now - _ch[w].last_tick) < PWM_TIMEOUT_MS)
//            rpwm = _ch[w].duty;
//        if ((now - _ch[w + 4].last_tick) < PWM_TIMEOUT_MS)
//            lpwm = _ch[w + 4].duty;
//        duty_out[w] = rpwm - lpwm;
//    }
//}
//
//void PWMCapture_IC_Callback(TIM_HandleTypeDef *htim)
//{
//    uint32_t trig_ch = _hal_active_to_ch(htim->Channel);
//    if (trig_ch == 0xFFFFFFFFu) return;
//
//    PwmCh_t *c = NULL;
//    for (uint8_t i = 0; i < N_CHANNELS; i++) {
//        if (_ch[i].htim->Instance == htim->Instance &&
//            _ch[i].hal_ch == trig_ch) {
//            c = &_ch[i];
//            break;
//        }
//    }
//    if (!c) return;
//
//    uint32_t cap = HAL_TIM_ReadCapturedValue(htim, trig_ch);
//    c->last_tick = HAL_GetTick();
//
//    if (c->state == 0) {
//        /* WAIT_RISE: got rising edge */
//        if (c->t_rise != 0)
//            c->period = (uint16_t)(cap - (uint16_t)c->t_rise);
//        c->t_rise = cap;
//        c->state  = 1;
//        _arm(c, LL_TIM_IC_POLARITY_FALLING);
//    } else {
//        /* WAIT_FALL: got falling edge */
//        c->width = (uint16_t)(cap - (uint16_t)c->t_rise);
//        c->state = 0;
//        _arm(c, LL_TIM_IC_POLARITY_RISING);
//        if (c->period > 0) {
//            float d = (float)c->width / (float)c->period;
//            if (d > 1.0f) d = 1.0f;
//            if (d < 0.0f) d = 0.0f;
//            c->duty = c->duty * 0.75f + d * 0.25f;
//        }
//    }
//}



/*
 * pwm_capture.c — RPWM/LPWM duty cycle measurement via TIM Input Capture
 *
 * Hardware:
 *   TIM5  CH1-4 → RPWM wheels 1-4  (PA0, PA1, PA2, PA3)
 *   TIM15 CH1-2 → LPWM wheels 1-2  (PE5, PE6)
 *   TIM16 CH1   → LPWM wheel 3     (PF6)
 *   TIM17 CH1   → LPWM wheel 4     (PF7)
 *
 * _arm() sets CCER + DIER on every call — ISR-safe direct register writes.
 * Polarity toggle uses direct CCER CCxP bit (no LL dependency).
 */

#include "pwm_capture.h"
#include <string.h>

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

    _ch[0].htim = ht5;  _ch[0].hal_ch = TIM_CHANNEL_1;
    _ch[1].htim = ht5;  _ch[1].hal_ch = TIM_CHANNEL_2;
    _ch[2].htim = ht5;  _ch[2].hal_ch = TIM_CHANNEL_3;
    _ch[3].htim = ht5;  _ch[3].hal_ch = TIM_CHANNEL_4;
    _ch[4].htim = ht15; _ch[4].hal_ch = TIM_CHANNEL_1;
    _ch[5].htim = ht15; _ch[5].hal_ch = TIM_CHANNEL_2;
    _ch[6].htim = ht16; _ch[6].hal_ch = TIM_CHANNEL_1;
    _ch[7].htim = ht17; _ch[7].hal_ch = TIM_CHANNEL_1;

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

    /* Enable all channels: CCER (rising=CCxP clear) first, then DIER, then CEN */

    /* TIM5 CH1-4 */
    ht5->Instance->CCER &= ~(TIM_CCER_CC1P | TIM_CCER_CC2P |
                              TIM_CCER_CC3P | TIM_CCER_CC4P);  /* rising edge */
    ht5->Instance->CCER |=  (TIM_CCER_CC1E | TIM_CCER_CC2E |
                              TIM_CCER_CC3E | TIM_CCER_CC4E);
    ht5->Instance->DIER |=  (TIM_DIER_CC1IE | TIM_DIER_CC2IE |
                              TIM_DIER_CC3IE | TIM_DIER_CC4IE);
    ht5->Instance->CR1  |= TIM_CR1_CEN;

    /* TIM15 CH1-2 */
    ht15->Instance->CCER &= ~(TIM_CCER_CC1P | TIM_CCER_CC2P);
    ht15->Instance->CCER |=  (TIM_CCER_CC1E | TIM_CCER_CC2E);
    ht15->Instance->DIER |=  (TIM_DIER_CC1IE | TIM_DIER_CC2IE);
    ht15->Instance->CR1  |= TIM_CR1_CEN;

    /* TIM16 CH1 */
    ht16->Instance->CCER &= ~TIM_CCER_CC1P;
    ht16->Instance->CCER |=  TIM_CCER_CC1E;
    ht16->Instance->DIER |=  TIM_DIER_CC1IE;
    ht16->Instance->CR1  |= TIM_CR1_CEN;

    /* TIM17 CH1 */
    ht17->Instance->CCER &= ~TIM_CCER_CC1P;
    ht17->Instance->CCER |=  TIM_CCER_CC1E;
    ht17->Instance->DIER |=  TIM_DIER_CC1IE;
    ht17->Instance->CR1  |= TIM_CR1_CEN;
}

void PWMCapture_GetDuty(float *duty_out)
{
    uint32_t now = HAL_GetTick();
    for (uint8_t w = 0; w < PWM_NUM_WHEELS; w++) {
        float rpwm = 0.0f;
        float lpwm = 0.0f;
        if ((now - _ch[w].last_tick) < PWM_TIMEOUT_MS)
            rpwm = _ch[w].duty;
        if ((now - _ch[w + 4].last_tick) < PWM_TIMEOUT_MS)
            lpwm = _ch[w + 4].duty;
        duty_out[w] = rpwm - lpwm;
    }
}

void PWMCapture_IC_Callback(TIM_HandleTypeDef *htim)
{
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
