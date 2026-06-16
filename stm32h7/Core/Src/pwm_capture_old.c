/*
 * pwm_capture.c — RPWM/LPWM duty cycle measurement via TIM Input Capture
 *
 * State machine per channel (2-phase, rising→falling→rising):
 *   WAIT_RISE: arm rising edge → capture t_rise, arm falling
 *   WAIT_FALL: arm falling edge → compute width, arm rising,
 *              update duty on every complete rising→falling→rising cycle
 *
 * Polarity is toggled with LL_TIM_IC_SetPolarity() — one register write,
 * ISR-safe.  HAL_TIM_IC_ConfigChannel is NOT called in ISR context.
 */

#include "pwm_capture.h"
#include <string.h>

/* ── Per-channel descriptor ─────────────────────────────────────────────── */
typedef struct {
    TIM_HandleTypeDef *htim;
    uint32_t           hal_ch;     /* TIM_CHANNEL_x                      */
    uint32_t           ll_ch;      /* LL_TIM_CHANNEL_CHx                 */
    uint32_t           t_rise;     /* last rising-edge timestamp [ticks] */
    uint32_t           t_fall;     /* last falling-edge timestamp        */
    uint32_t           period;     /* ticks between consecutive rises    */
    uint32_t           width;      /* ticks of high pulse                */
    uint8_t            state;      /* 0=WAIT_RISE, 1=WAIT_FALL           */
    uint32_t           last_tick;  /* HAL_GetTick() at last capture      */
    float              duty;       /* filtered duty [0..1]               */
} PwmCh_t;

/* ── Channel table
 *   Index layout:
 *     0..3 = RPWM wheel 1-4   (TIM5 CH1-4)
 *     4..5 = LPWM wheel 1-2   (TIM15 CH1-2)
 *     6    = LPWM wheel 3     (TIM16 CH1)
 *     7    = LPWM wheel 4     (TIM17 CH1)
 * ─────────────────────────────────────────────────────────────────────────*/
#define N_CHANNELS 8u

static PwmCh_t _ch[N_CHANNELS];

/* ── LL channel map (HAL TIM_CHANNEL_x → LL_TIM_CHANNEL_CHx) ────────────── */
static uint32_t _ll_ch(uint32_t hal_ch)
{
    switch (hal_ch) {
        case TIM_CHANNEL_1: return LL_TIM_CHANNEL_CH1;
        case TIM_CHANNEL_2: return LL_TIM_CHANNEL_CH2;
        case TIM_CHANNEL_3: return LL_TIM_CHANNEL_CH3;
        case TIM_CHANNEL_4: return LL_TIM_CHANNEL_CH4;
        default:            return LL_TIM_CHANNEL_CH1;
    }
}

/* ── HAL active channel → TIM_CHANNEL_x ────────────────────────────────── */
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

/* ── Arm one IC channel ─────────────────────────────────────────────────── */
static void _arm(PwmCh_t *c, uint32_t polarity)
{
    /* LL polarity toggle — single register write, ISR-safe */
    LL_TIM_IC_SetPolarity(c->htim->Instance, c->ll_ch, polarity);
    HAL_TIM_IC_Start_IT(c->htim, c->hal_ch);
}

/* ── Public API ─────────────────────────────────────────────────────────── */

void PWMCapture_Init(TIM_HandleTypeDef *ht5,
                     TIM_HandleTypeDef *ht15,
                     TIM_HandleTypeDef *ht16,
                     TIM_HandleTypeDef *ht17)
{
    memset(_ch, 0, sizeof(_ch));

    /* RPWM: TIM5 CH1-4 */
    _ch[0].htim = ht5;  _ch[0].hal_ch = TIM_CHANNEL_1;
    _ch[1].htim = ht5;  _ch[1].hal_ch = TIM_CHANNEL_2;
    _ch[2].htim = ht5;  _ch[2].hal_ch = TIM_CHANNEL_3;
    _ch[3].htim = ht5;  _ch[3].hal_ch = TIM_CHANNEL_4;

    /* LPWM: TIM15 CH1-2 */
    _ch[4].htim = ht15; _ch[4].hal_ch = TIM_CHANNEL_1;
    _ch[5].htim = ht15; _ch[5].hal_ch = TIM_CHANNEL_2;

    /* LPWM: TIM16 CH1, TIM17 CH1 */
    _ch[6].htim = ht16; _ch[6].hal_ch = TIM_CHANNEL_1;
    _ch[7].htim = ht17; _ch[7].hal_ch = TIM_CHANNEL_1;

    /* Resolve LL channels */
    for (uint8_t i = 0; i < N_CHANNELS; i++) {
        _ch[i].ll_ch = _ll_ch(_ch[i].hal_ch);
        _ch[i].state = 0; /* WAIT_RISE */
    }

    /* Start TIM16/17 base counters via direct register write.
     * HAL_TIM_Base_Start() sets HAL state=BUSY which silently blocks
     * HAL_TIM_IC_Start_IT() called below — bypass with CR1 direct write. */
    ht16->Instance->CR1 |= TIM_CR1_CEN;
    ht17->Instance->CR1 |= TIM_CR1_CEN;

    /* Arm all channels for rising edge */
    for (uint8_t i = 0; i < N_CHANNELS; i++) {
        _arm(&_ch[i], LL_TIM_IC_POLARITY_RISING);
    }
}

void PWMCapture_GetDuty(float *duty_out)
{
    uint32_t now = HAL_GetTick();
    for (uint8_t w = 0; w < PWM_NUM_WHEELS; w++) {
        float rpwm = 0.0f;
        float lpwm = 0.0f;

        /* Timeout check */
        if ((now - _ch[w].last_tick) < PWM_TIMEOUT_MS) {
            rpwm = _ch[w].duty;
        }
        if ((now - _ch[w + 4].last_tick) < PWM_TIMEOUT_MS) {
            lpwm = _ch[w + 4].duty;
        }

        /* Net signed duty: positive = forward, negative = reverse */
        duty_out[w] = rpwm - lpwm;
    }
}

void PWMCapture_IC_Callback(TIM_HandleTypeDef *htim)
{
    /* Find channel */
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
        /* ── WAIT_RISE: got rising edge ─────────────────────────────────── */
        if (c->t_rise != 0) {
            /* Compute period (wraps correctly for 16-bit counter) */
            c->period = (uint16_t)(cap - (uint16_t)c->t_rise);
        }
        c->t_rise = cap;
        c->state  = 1;
        _arm(c, LL_TIM_IC_POLARITY_FALLING);

    } else {
        /* ── WAIT_FALL: got falling edge ────────────────────────────────── */
        c->width = (uint16_t)(cap - (uint16_t)c->t_rise);
        c->state = 0;
        _arm(c, LL_TIM_IC_POLARITY_RISING);

        /* Update duty when we have a valid period */
        if (c->period > 0) {
            float d = (float)c->width / (float)c->period;
            if (d > 1.0f) d = 1.0f;
            if (d < 0.0f) d = 0.0f;
            /* Simple low-pass: α = 0.25 */
            c->duty = c->duty * 0.75f + d * 0.25f;
        }
    }
}