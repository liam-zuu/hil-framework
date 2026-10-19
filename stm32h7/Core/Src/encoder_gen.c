#include "encoder_gen.h"
#include <math.h>
#include <string.h>

#define ENC_OMEGA_MIN 0.3f  /* rad/s — below this magnitude, treat as stopped */

/* ── Per-wheel state ────────────────────────────────────────────────────── */
typedef struct {
    TIM_HandleTypeDef *htim;
    uint32_t           half_period;   /* ticks between consecutive toggles  */
    int8_t             dir;           /* +1 fwd, -1 rev, 0 stopped           */
    float              omega_cmd;     /* last commanded omega [rad/s]        */

    /* CH1 period measurement */
    uint32_t           ch1_last_cnt;  /* CNT value at previous CH1 callback  */
    uint32_t           ch1_period;    /* measured ticks between CH1 callbacks*/
    uint8_t            ch1_has_last;  /* 0 until first CH1 callback seen     */
} Wheel_t;

static Wheel_t _w[ENC_NUM_WHEELS];

/* ── half_period from |omega| ───────────────────────────────────────────── */
static uint32_t _half_period(float omega_abs)
{
    /* half_period = ENC_TIMER_FREQ / (omega_abs * ENC_PPR)
     * π factor empirically removed — H7 OC Toggle timing behaviour. */
    float hp = (float)ENC_TIMER_FREQ /
               (omega_abs * (float)ENC_PPR);
    if (hp > 65535.0f) return 65535u;
    if (hp <     1.0f) return 1u;
    return (uint32_t)hp;
}

/* ── Direct-register arm: set CCRx, enable CCxE + DIER CCxIE, start CEN ── */
static void _arm_ch(TIM_HandleTypeDef *htim, uint32_t ch, uint32_t offset)
{
    uint32_t ccr = (htim->Instance->CNT + offset) & 0xFFFFu;

    switch (ch) {
        case TIM_CHANNEL_1:
            htim->Instance->CCR1  = ccr;
            htim->Instance->CCER |= TIM_CCER_CC1E;
            htim->Instance->DIER |= TIM_DIER_CC1IE;
            break;
        case TIM_CHANNEL_2:
            htim->Instance->CCR2  = ccr;
            htim->Instance->CCER |= TIM_CCER_CC2E;
            htim->Instance->DIER |= TIM_DIER_CC2IE;
            break;
        default:
            return;
    }

    /* Advanced timers (TIM1) need MOE to actually drive the output pins */
    if (IS_TIM_BREAK_INSTANCE(htim->Instance)) {
        htim->Instance->BDTR |= TIM_BDTR_MOE;
    }

    htim->Instance->CR1 |= TIM_CR1_CEN;
}

/* ── Disable both channels, mark stopped ────────────────────────────────── */
static void _stop(uint8_t i)
{
    _w[i].htim->Instance->DIER &= ~(TIM_DIER_CC1IE | TIM_DIER_CC2IE);
    _w[i].htim->Instance->CCER &= ~(TIM_CCER_CC1E  | TIM_CCER_CC2E);
    _w[i].dir          = 0;
    _w[i].ch1_has_last = 0;
}

/* ── Arm both channels with the 90 deg phase offset for the given dir ──── */
static void _start_wheel(uint8_t i, int8_t dir)
{
    uint32_t hp = _w[i].half_period;

    /*
     * Forward (dir>0): A leads B by 90 deg.
     *   A fires at cnt+hp,            B fires at cnt+hp + hp/2
     * Reverse (dir<0): B leads A by 90 deg.
     *   B fires at cnt+hp,            A fires at cnt+hp + hp/2
     */
    uint32_t a_off = (dir > 0) ? hp            : hp + (hp >> 1);
    uint32_t b_off = (dir > 0) ? hp + (hp >> 1) : hp;

    _arm_ch(_w[i].htim, TIM_CHANNEL_1, a_off);
    _arm_ch(_w[i].htim, TIM_CHANNEL_2, b_off);

    _w[i].dir          = dir;
    _w[i].ch1_has_last = 0;  /* discard stale period across restart */
}

/* ── Public API ─────────────────────────────────────────────────────────── */

void EncoderGen_Init(TIM_HandleTypeDef *ht1,
                     TIM_HandleTypeDef *ht2,
                     TIM_HandleTypeDef *ht3,
                     TIM_HandleTypeDef *ht4)
{
    memset(_w, 0, sizeof(_w));
    _w[0].htim = ht1;
    _w[1].htim = ht2;
    _w[2].htim = ht3;
    _w[3].htim = ht4;

    /* No HAL_TIM_Base_Start / OC_Start_IT — counters started lazily by
     * _arm_ch() via CR1 |= CEN. */
}

void EncoderGen_SetOmega(uint8_t wheel, float omega_rad_s)
{
    if (wheel >= ENC_NUM_WHEELS) return;

    _w[wheel].omega_cmd = omega_rad_s;

    float abs_omega = fabsf(omega_rad_s);
    int8_t new_dir  = (omega_rad_s >  ENC_OMEGA_MIN) ?  1 :
                      (omega_rad_s < -ENC_OMEGA_MIN) ? -1 : 0;

    if (new_dir == 0) {
        if (_w[wheel].dir != 0) _stop(wheel);
        return;
    }

    _w[wheel].half_period = _half_period(abs_omega);

    if (_w[wheel].dir == 0) {
        _start_wheel(wheel, new_dir);
    } else if (new_dir != _w[wheel].dir) {
        _stop(wheel);
        _start_wheel(wheel, new_dir);
    }
    /* Same direction, already running: half_period updated above and
     * picked up automatically on the next OC callback — no restart. */
}

void EncoderGen_OC_Callback(TIM_HandleTypeDef *htim)
{
    /* Find wheel */
    uint8_t i;
    for (i = 0; i < ENC_NUM_WHEELS; i++) {
        if (_w[i].htim->Instance == htim->Instance) break;
    }
    if (i == ENC_NUM_WHEELS) return;

    /* Map HAL active channel → TIM_CHANNEL_x */
    uint32_t ch;
    switch (htim->Channel) {
        case HAL_TIM_ACTIVE_CHANNEL_1: ch = TIM_CHANNEL_1; break;
        case HAL_TIM_ACTIVE_CHANNEL_2: ch = TIM_CHANNEL_2; break;
        default: return;
    }

    /* CH1 period measurement */
    if (ch == TIM_CHANNEL_1) {
        uint32_t now = htim->Instance->CNT;
        if (_w[i].ch1_has_last) {
            _w[i].ch1_period = (uint16_t)(now - (uint16_t)_w[i].ch1_last_cnt);
        }
        _w[i].ch1_last_cnt  = now;
        _w[i].ch1_has_last  = 1;
    }

    /* Advance compare register by half_period */
    uint32_t hp  = _w[i].half_period;
    uint32_t cur = __HAL_TIM_GET_COMPARE(htim, ch);
    __HAL_TIM_SET_COMPARE(htim, ch, (uint16_t)((cur + hp) & 0xFFFFu));
}

void EncoderGen_GetDebug(EncoderGenDebug_t *dbg)
{
    for (uint8_t i = 0; i < ENC_NUM_WHEELS; i++) {
        dbg[i].omega_cmd   = _w[i].omega_cmd;
        dbg[i].dir         = _w[i].dir;
        dbg[i].half_period = _w[i].half_period;
        dbg[i].ch1_period  = _w[i].ch1_period;

        /* measured_freq: two CH1 callbacks = one full pulse period
         * freq = ENC_TIMER_FREQ / (ch1_period * 2)
         * No π factor — empirically removed for H7 OC Toggle mode. */
        dbg[i].measured_freq = (_w[i].ch1_period > 0)
            ? (float)ENC_TIMER_FREQ / ((float)_w[i].ch1_period * 2.0f)
            : 0.0f;

        dbg[i].expected_freq = fabsf(_w[i].omega_cmd) / (2.0f * (float)M_PI)
                              * (float)ENC_PPR;
    }
}
