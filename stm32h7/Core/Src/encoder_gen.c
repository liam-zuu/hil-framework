/*
 * encoder_gen.c — Quadrature encoder pulse generation
 *
 * Timer frequency derivation:
 *   SYSCLK = 480 MHz, AHB = 240 MHz, APB1 = APB2 = 120 MHz
 *   Timer clock (APB prescaler > 1) = 2 × 120 MHz = 240 MHz
 *   PSC = 239  →  timer tick = 240 MHz / 240 = 1 MHz  ← ENC_TIMER_FREQ
 *
 *   (Previous session used 2 MHz — incorrect.  1 MHz is right for PSC=239
 *    with APB clock 120 MHz.  This halves computed half_period vs. before.)
 *
 * Frequency derivation:
 *   pulse_freq  = |omega| / (2π) × PPR
 *   half_period = ENC_TIMER_FREQ / (2 × pulse_freq)
 *               = ENC_TIMER_FREQ × π / (|omega| × PPR)
 *
 * Known issue from previous session: measured ~84.7 Hz instead of 525 Hz
 * at omega=10 rad/s (factor ≈ 2π).  Fixing ENC_TIMER_FREQ 2M→1M halves
 * half_period; if the issue was overcounting ticks, this may resolve it.
 * Debug path: read enc_oc_count in debugger after 1 s.
 *   If count ≈ 2×525×4 = 4200  → callbacks correct, check GPIO/OC advance
 *   If count ≈ 4200/2π ≈ 668   → callbacks too slow, OC advance wrong
 */

#include "encoder_gen.h"
#include <math.h>
#include <string.h>

/* ── Constants ──────────────────────────────────────────────────────────── */
#define ENC_TIMER_FREQ   1000000UL  /* Hz */
#define ENC_PPR          330u       /* pulses/rev on channel A */
#define ENC_OMEGA_MIN    0.3f       /* rad/s — below this treated as zero */

/* ── Per-wheel state ────────────────────────────────────────────────────── */
typedef struct {
    TIM_HandleTypeDef *htim;
    uint32_t           half_period;  /* ticks between OC toggle events */
    int8_t             dir;          /* +1 = fwd, -1 = rev, 0 = stopped */
} Wheel_t;

static Wheel_t _w[ENC_NUM_WHEELS];

/* Debug counter: exported so debugger watch window can read it directly */
volatile uint32_t enc_oc_count = 0;

/* ── Internal helpers ───────────────────────────────────────────────────── */

static uint32_t _hp(float omega_abs)
{
    /* half_period = FREQ × π / (omega × PPR) */
    float hp = (float)ENC_TIMER_FREQ /
               (omega_abs * (float)ENC_PPR);
    if (hp > 65535.0f) return 65535u;
    if (hp <     1.0f) return 1u;
    return (uint32_t)hp;
}

/* Direct register helpers — bypass HAL state machine entirely.
 * HAL_TIM_OC_Start_IT checks channel state (READY/BUSY) and silently
 * returns HAL_ERROR if state is wrong. Direct register access has no
 * such restriction. */

static void _enc_arm_ch(TIM_HandleTypeDef *htim, uint32_t ch, uint32_t offset)
{
    uint32_t ccr = (htim->Instance->CNT + offset) & 0xFFFFu;

    switch (ch) {
        case TIM_CHANNEL_1:
            htim->Instance->CCR1  = ccr;
            htim->Instance->DIER |= TIM_DIER_CC1IE;
            htim->Instance->CCER |= TIM_CCER_CC1E;
            break;
        case TIM_CHANNEL_2:
            htim->Instance->CCR2  = ccr;
            htim->Instance->DIER |= TIM_DIER_CC2IE;
            htim->Instance->CCER |= TIM_CCER_CC2E;
            break;
        default: return;
    }

    /* Advanced timers (TIM1) need MOE bit to drive output pins */
    if (IS_TIM_BREAK_INSTANCE(htim->Instance)) {
        htim->Instance->BDTR |= TIM_BDTR_MOE;
    }

    /* Start counter if not already running */
    htim->Instance->CR1 |= TIM_CR1_CEN;
}

static void _enc_stop(uint8_t i)
{
    /* Disable CC interrupts and outputs for both channels */
    _w[i].htim->Instance->DIER &= ~(TIM_DIER_CC1IE | TIM_DIER_CC2IE);
    _w[i].htim->Instance->CCER &= ~(TIM_CCER_CC1E  | TIM_CCER_CC2E);
    _w[i].dir = 0;
}

static void _enc_start(uint8_t i, int8_t dir)
{
    uint32_t hp = _w[i].half_period;
    /*
     * Forward (dir>0): A leads B by 90°.
     *   A fires at cnt+hp, B fires at cnt+hp + hp/2
     * Reverse (dir<0): B leads A by 90°.
     *   B fires at cnt+hp, A fires at cnt+hp + hp/2
     * This sets the initial phase at startup.
     * Subsequent OC callbacks advance each channel independently by hp,
     * so the 90° offset is preserved automatically.
     */
    uint32_t a_off = (dir > 0) ? hp          : hp + (hp >> 1);
    uint32_t b_off = (dir > 0) ? hp + (hp >> 1) : hp;
    _enc_arm_ch(_w[i].htim, TIM_CHANNEL_1, a_off);
    _enc_arm_ch(_w[i].htim, TIM_CHANNEL_2, b_off);
    _w[i].dir = dir;
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

    /* HAL_TIM_Base_Start NOT called here — it sets state=BUSY which blocks
     * HAL_TIM_OC_Start_IT. The OC start call enables the counter itself. */
}

void EncoderGen_SetOmega(uint8_t wheel, float omega_rad_s)
{
    if (wheel >= ENC_NUM_WHEELS) return;

    float abs_omega = (omega_rad_s >= 0.0f) ? omega_rad_s : -omega_rad_s;
    int8_t new_dir  = (omega_rad_s >  ENC_OMEGA_MIN) ?  1 :
                      (omega_rad_s < -ENC_OMEGA_MIN) ? -1 : 0;

    if (new_dir == 0) {
        /* Stop */
        if (_w[wheel].dir != 0) _enc_stop(wheel);
        return;
    }

    _w[wheel].half_period = _hp(abs_omega);

    if (_w[wheel].dir == 0) {
        /* Was stopped: start fresh */
        _enc_start(wheel, new_dir);
    } else if (new_dir != _w[wheel].dir) {
        /* Direction change: stop and restart with new phase */
        _enc_stop(wheel);
        _enc_start(wheel, new_dir);
    }
    /*
     * If same direction and already running: only half_period changed.
     * The new value is picked up automatically on the next OC callback,
     * so no restart is needed.
     */
}

/*
 * Call from HAL_TIM_OC_DelayElapsedCallback.
 * htim->Channel is set by HAL before the callback — do NOT use GET_FLAG.
 */
void EncoderGen_OC_Callback(TIM_HandleTypeDef *htim)
{
    enc_oc_count++;

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

    /*
     * Advance compare register by half_period.
     * This schedules the next toggle without drifting relative to the timer.
     * 16-bit wraparound is intentional and correct.
     */
    uint32_t hp  = _w[i].half_period;
    uint32_t cur = __HAL_TIM_GET_COMPARE(htim, ch);
    __HAL_TIM_SET_COMPARE(htim, ch, (uint16_t)((cur + hp) & 0xFFFFu));
}
