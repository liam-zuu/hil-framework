#include "encoder_gen.h"
#include <math.h>
#include <string.h>

#define ENC_OMEGA_MIN 0.3f  /* rad/s — below this magnitude, treat as stopped */

/* Phase sign per wheel (FL, FR, RL, RR), so that the ESP32 decodes a forward
 * wheel (omega > 0) as a positive count on all four wheels, as on the real
 * AGV (S01 _thu4, 21/10: positive PWM -> positive speed on every wheel).
 * AGV_MECANUM encoder.c swaps A/B for FL and RL (enc_cfg), and its PCNT setup
 * counts "A leads B" as negative, so the right wheels need B leading A for a
 * forward wheel. Measured on the bench 26/10 before this table: FR(+20) and
 * RR(-20) came out with the wrong sign, RL(-10) right (ENC_PI run02). */
static const int8_t ENC_DIR_SIGN[ENC_NUM_WHEELS] = { +1, -1, +1, -1 };

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
    /* half_period = ENC_TIMER_FREQ / (2 * f_pulse)
     *             = ENC_TIMER_FREQ * pi / (omega_abs * ENC_PPR)   (encoder_gen.h)
     * The pi was removed on 17-18/06 because the debug print was wrong, not
     * because of the output; measured on board 26/10 (05_DATA/results/ENC_PI/
     * 2026-10-26_run01): without pi the ESP32 counted pi times too many edges. */
    float hp = ENC_TICKS_PER_S * (float)M_PI /     /* ENC_CLOCK_CAL: encoder_gen.h */
               (omega_abs * (float)ENC_PPR);
    if (hp > 65535.0f) return 65535u;
    if (hp <     1.0f) return 1u;
    return (uint32_t)(hp + 0.5f);
}

/* OCxM field values (4 bits, OCxM[3] is a separate bit on H7) */
#define OCM_FORCE_INACTIVE_1  (TIM_CCMR1_OC1M_2)
#define OCM_FORCE_INACTIVE_2  (TIM_CCMR1_OC2M_2)
#define OCM_TOGGLE_1          (TIM_CCMR1_OC1M_1 | TIM_CCMR1_OC1M_0)
#define OCM_TOGGLE_2          (TIM_CCMR1_OC2M_1 | TIM_CCMR1_OC2M_0)

/* Drive A and B low, then back to toggle mode. In toggle mode a pin keeps the
 * level it had when the wheel stopped, so without this the 90 deg lead/lag of
 * A and B, i.e. the direction the ESP32 sees, depended on the previous run
 * (signs flipped between two runs with the same command on 26/10). */
static void _force_low_then_toggle(TIM_TypeDef *t)
{
    uint32_t m = t->CCMR1 & ~(TIM_CCMR1_OC1M | TIM_CCMR1_OC2M);
    t->CCMR1 = m | OCM_FORCE_INACTIVE_1 | OCM_FORCE_INACTIVE_2;
    t->CCMR1 = m | OCM_TOGGLE_1 | OCM_TOGGLE_2;
}

/* ── Stop both channels, pins held at their level, mark stopped ───────── */
/* 29/10 (Liam: "Sửa giữ mức"): before, _stop cleared CCxE, so the timer no
 * longer drove A and B; with GPIO_NOPULL the pins floated while the wheel was
 * stopped and the ESP32 could count noise. Now both channels go to frozen
 * mode (OCxM = 0000: a compare match no longer changes OCxREF) with the output
 * still enabled, so each pin stays driven at the level of its last edge. */
static void _stop(uint8_t i)
{
    TIM_TypeDef *t = _w[i].htim->Instance;
    t->DIER  &= ~(TIM_DIER_CC1IE | TIM_DIER_CC2IE);
    t->CCMR1 &= ~(TIM_CCMR1_OC1M | TIM_CCMR1_OC2M);   /* frozen */
    _w[i].dir          = 0;
    _w[i].ch1_has_last = 0;
}

/* ── Arm both channels with the 90 deg phase offset for the given dir ──── */
static void _start_wheel(uint8_t i, int8_t dir)
{
    TIM_TypeDef *t = _w[i].htim->Instance;
    uint32_t hp = _w[i].half_period;

    /*
     * Both pins start low (forced), then the first toggle of each is a
     * rising edge:
     * phase = dir * ENC_DIR_SIGN[i]
     * phase > 0: A leads B by 90 deg.
     *   A rises at cnt+hp,            B rises at cnt+hp + hp/2
     * phase < 0: B leads A by 90 deg.
     *   B rises at cnt+hp,            A rises at cnt+hp + hp/2
     * The compare values are written before toggle mode is restored, so an
     * old CCRx cannot toggle a pin in between.
     */
    int8_t   phase = (int8_t)(dir * ENC_DIR_SIGN[i]);   /* +1: A leads B */
    uint32_t a_off = (phase > 0) ? hp            : hp + (hp >> 1);
    uint32_t b_off = (phase > 0) ? hp + (hp >> 1) : hp;
    uint32_t cnt   = t->CNT;

    t->CCR1 = (cnt + a_off) & 0xFFFFu;
    t->CCR2 = (cnt + b_off) & 0xFFFFu;
    _force_low_then_toggle(t);
    t->SR    = ~(TIM_SR_CC1IF | TIM_SR_CC2IF);   /* rc_w0: drop stale flags */
    t->CCER |= TIM_CCER_CC1E  | TIM_CCER_CC2E;
    t->DIER |= TIM_DIER_CC1IE | TIM_DIER_CC2IE;

    /* Advanced timers (TIM1) need MOE to actually drive the output pins */
    if (IS_TIM_BREAK_INSTANCE(t)) {
        t->BDTR |= TIM_BDTR_MOE;
    }
    t->CR1 |= TIM_CR1_CEN;

    _w[i].dir          = dir;
    _w[i].ch1_has_last = 0;  /* discard stale period across restart */
}

/* ── Same direction, new half period: re-time both pending edges ───────── */
#define ENC_RETIME_MARGIN 3u   /* ticks: a compare closer than this is left to fire */

static void _retime(uint8_t i, uint32_t hp)
{
    /*
     * Before 28/10 only half_period changed and each callback added it to its
     * own channel (bug H7-ENC-RETIME, model 05_DATA/results/ENC_SCHED_MODEL/
     * 2026-10-28_run01). Two faults: (1) after a speed-up the next edge still
     * came one OLD half period later (up to ~4.8 ms, S06 review #2); (2) the
     * A/B offset stayed half the OLD half period, so after a speed-up of about
     * x2 in one frame, or a start from rest, B fell more than one new half
     * period behind A and the ESP32 decoded the wheel backwards.
     *
     * Now, with interrupts off (the encoder timers run at priority 0):
     *  1. a compare that already toggled its pin but whose callback has not
     *     run is advanced here by the old half period, as the callback would;
     *  2. the channel whose edge comes first (N) keeps its edge, or moves it to
     *     now + hp if that is sooner;
     *  3. the other channel follows N by hp / 2.
     * Edge order, so direction, is kept: the next edge is N's either way.
     */
    TIM_TypeDef *t = _w[i].htim->Instance;
    uint32_t primask = __get_PRIMASK();
    __disable_irq();

    uint32_t hp_old = _w[i].half_period;
    uint32_t now    = t->CNT;          /* read before SR: a match at or before now is flagged */
    uint32_t sr     = t->SR & (t->DIER & (TIM_DIER_CC1IE | TIM_DIER_CC2IE));
    uint32_t ccr1   = t->CCR1;
    uint32_t ccr2   = t->CCR2;
    if (sr & TIM_SR_CC1IF) ccr1 = (ccr1 + hp_old) & 0xFFFFu;
    if (sr & TIM_SR_CC2IF) ccr2 = (ccr2 + hp_old) & 0xFFFFu;

    uint32_t d1 = (ccr1 - now) & 0xFFFFu;
    uint32_t d2 = (ccr2 - now) & 0xFFFFu;
    int a_first = (d1 != d2) ? (d1 < d2)
                             : ((int8_t)(_w[i].dir * ENC_DIR_SIGN[i]) > 0);   /* tie: the leader */
    uint32_t dn  = a_first ? d1 : d2;
    if (dn > hp && hp >= ENC_RETIME_MARGIN) dn = hp;
    uint32_t n_at = (now + dn) & 0xFFFFu;
    uint32_t f_at = (n_at + (hp >> 1)) & 0xFFFFu;

    t->CCR1 = a_first ? n_at : f_at;
    t->CCR2 = a_first ? f_at : n_at;
    if (sr) t->SR = ~sr;               /* rc_w0: drop only the flags handled here */
    _w[i].half_period  = hp;
    _w[i].ch1_has_last = 0;            /* the next CH1 interval is not one half period */

    __set_PRIMASK(primask);
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
     * _start_wheel() via CR1 |= CEN. Outputs driven low from boot (29/10):
     * before, they were enabled only at the first start, so A and B floated
     * until the first nonzero omega. */
    for (int i = 0; i < ENC_NUM_WHEELS; i++) {
        TIM_TypeDef *t = _w[i].htim->Instance;
        uint32_t m = t->CCMR1 & ~(TIM_CCMR1_OC1M | TIM_CCMR1_OC2M);
        t->CCMR1 = m | OCM_FORCE_INACTIVE_1 | OCM_FORCE_INACTIVE_2;
        t->CCER |= TIM_CCER_CC1E | TIM_CCER_CC2E;
        if (IS_TIM_BREAK_INSTANCE(t)) t->BDTR |= TIM_BDTR_MOE;
    }
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

    uint32_t hp = _half_period(abs_omega);

    if (_w[wheel].dir == 0) {
        _w[wheel].half_period = hp;
        _start_wheel(wheel, new_dir);
    } else if (new_dir != _w[wheel].dir) {
        _w[wheel].half_period = hp;
        _stop(wheel);
        _start_wheel(wheel, new_dir);
    } else if (hp != _w[wheel].half_period) {
        _retime(wheel, hp);         /* same direction, new speed */
    }
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

        /* measured_freq: each CH1 callback is one toggle, so ch1_period (time
         * between two callbacks) is half a pulse period:
         *   freq = ENC_TICKS_PER_S / (2 * ch1_period)   (real Hz, ENC_CLOCK_CAL)
         * It comes from the same timer that makes the pulses, so it only
         * checks the scheduling, not the formula: an independent check needs
         * the ESP32 count (env enc_count) or a logic analyzer. */
        dbg[i].measured_freq = (_w[i].ch1_period > 0)
            ? ENC_TICKS_PER_S / ((float)_w[i].ch1_period * 2.0f)   /* real Hz */
            : 0.0f;

        dbg[i].expected_freq = fabsf(_w[i].omega_cmd) / (2.0f * (float)M_PI)
                              * (float)ENC_PPR;
    }
}
