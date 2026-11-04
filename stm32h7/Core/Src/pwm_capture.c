#include "pwm_capture.h"
#include "pwm_edges.h"
#include <string.h>
#include <stdio.h>

/* HAL side of the DMA capture (see pwm_capture.h); the decoding is in
 * pwm_edges.c, tested on the PC (pwmedges_pc). Codex Sol round 1 (04/11,
 * CLAUDE_THESIS 03_CODE/docs/review/2026-11-04_Review_H7_PWM_DMA_vong1.md):
 * fixes M1-M4, M6, M7, M10, M12 (Liam: "Duyệt sửa cả 9"). */

#define LAP_MS          5u   /* below the 6.5 ms counter wrap (pwm_edges.h) */
#define STALE_MS        5u
#define MAX_PERIOD    750u   /* ticks, 75 us = 1.5 x the 20 kHz period (M2) */
#define VERIFY_MS      10u   /* while known: full window set this often (M1) */
#define WIN_STEP_TICKS 70u   /* window starts 7 us apart, absolute on TIM5 CNT (M10) */
#define PIN_B_TICKS    17u   /* pin reads at 0, 1.7 us, 3.5 us in a window (M1) */
#define PIN_C_TICKS    35u   /* 3.5 us: > input filter (about 1.07 us) + DMA, also for the last channel read */
#define PROBE_TICKS   600u   /* after a lap: look 60 us (> one 50 us period) for a new capture (M3) */
#define N_WIN           3u

typedef struct {
    TIM_HandleTypeDef *htim;
    DMA_HandleTypeDef *hdma;
    uint32_t           ch;         /* TIM_CHANNEL_x */
    uint32_t           ccxe;       /* CCxE in CCER */
    uint32_t           ccxde;      /* CCxDE in DIER */
    uint32_t           ccxie;      /* CCxIE in DIER (kept off) */
    uint32_t           ccxif;      /* CCxIF in SR */
    uint32_t           ccxof;      /* CCxOF in SR */
    GPIO_TypeDef      *port;
    uint16_t           pin;
    uint8_t            active;     /* DMA stream started and still enabled (M6) */
    uint32_t           dma_err;    /* stream found disabled while active */
    PwmEdges_t         dec;
    float              duty;
} PwmCh_t;

static PwmCh_t _ch[PWM_NUM_CHANNELS];
/* Rings in .bss = RAM_D1 (AXI SRAM, reachable by DMA2; D-cache is off) */
static uint32_t _ring[PWM_NUM_CHANNELS][PWM_RING_LEN];
static TIM_TypeDef *_tbase;        /* 10 MHz counter used to time the windows */
static uint32_t _svc;              /* service counter, anchor "different service" test */

static volatile uint32_t *_ccr(TIM_TypeDef *t, uint32_t ch)
{
    switch (ch) {
        case TIM_CHANNEL_1: return &t->CCR1;
        case TIM_CHANNEL_2: return &t->CCR2;
        case TIM_CHANNEL_3: return &t->CCR3;
        default:            return &t->CCR4;
    }
}

static void _set(PwmCh_t *c, TIM_HandleTypeDef *ht, uint32_t ch,
                 GPIO_TypeDef *port, uint16_t pin)
{
    static const uint32_t ccxe[4]  = { TIM_CCER_CC1E,  TIM_CCER_CC2E,  TIM_CCER_CC3E,  TIM_CCER_CC4E };
    static const uint32_t ccxde[4] = { TIM_DIER_CC1DE, TIM_DIER_CC2DE, TIM_DIER_CC3DE, TIM_DIER_CC4DE };
    static const uint32_t ccxie[4] = { TIM_DIER_CC1IE, TIM_DIER_CC2IE, TIM_DIER_CC3IE, TIM_DIER_CC4IE };
    static const uint32_t ccxif[4] = { TIM_SR_CC1IF,   TIM_SR_CC2IF,   TIM_SR_CC3IF,   TIM_SR_CC4IF };
    static const uint32_t ccxof[4] = { TIM_SR_CC1OF,   TIM_SR_CC2OF,   TIM_SR_CC3OF,   TIM_SR_CC4OF };
    static const uint16_t dma_id[4] = { TIM_DMA_ID_CC1, TIM_DMA_ID_CC2, TIM_DMA_ID_CC3, TIM_DMA_ID_CC4 };
    uint32_t k = ch / 4u;          /* TIM_CHANNEL_1..4 = 0x0, 0x4, 0x8, 0xC */
    c->htim  = ht;
    c->ch    = ch;
    c->hdma  = ht->hdma[dma_id[k]];   /* linked by CubeMX in HAL_TIM_*_MspInit */
    c->ccxe  = ccxe[k];
    c->ccxde = ccxde[k];
    c->ccxie = ccxie[k];
    c->ccxif = ccxif[k];
    c->ccxof = ccxof[k];
    c->port  = port;
    c->pin   = pin;
}

/* Ring write index from the DMA counter (NDTR counts down, reloads in circular mode) */
static uint32_t _wr(const PwmCh_t *c)
{
    uint32_t left = ((DMA_Stream_TypeDef *)c->hdma->Instance)->NDTR;
    return (PWM_RING_LEN - left) % PWM_RING_LEN;
}

static int _pin(const PwmCh_t *c)
{
    return (c->port->IDR & c->pin) != 0u;
}

static uint16_t _now_ticks(void)
{
    return (uint16_t)_tbase->CNT;
}

/* Wait until 'ticks' after t0 on the 10 MHz base */
static void _wait_until(uint16_t t0, uint32_t ticks)
{
    while ((uint16_t)(_now_ticks() - t0) < ticks) { }
}

void PWMCapture_Init(TIM_HandleTypeDef *ht5,
                     TIM_HandleTypeDef *ht8,
                     TIM_HandleTypeDef *ht15,
                     TIM_HandleTypeDef *ht16,
                     TIM_HandleTypeDef *ht17)
{
    const PwmEdgesCfg_t cfg = { PWM_RING_LEN, LAP_MS, STALE_MS, PWM_TIMEOUT_MS, MAX_PERIOD, VERIFY_MS };
    unsigned n_active = 0;

    memset(_ch, 0, sizeof(_ch));
    memset(_ring, 0, sizeof(_ring));
    _set(&_ch[0], ht5,  TIM_CHANNEL_1, GPIOA, GPIO_PIN_0);
    _set(&_ch[1], ht8,  TIM_CHANNEL_3, GPIOC, GPIO_PIN_8);
    _set(&_ch[2], ht5,  TIM_CHANNEL_3, GPIOA, GPIO_PIN_2);
    _set(&_ch[3], ht5,  TIM_CHANNEL_4, GPIOA, GPIO_PIN_3);
    _set(&_ch[4], ht15, TIM_CHANNEL_1, GPIOE, GPIO_PIN_5);
    _set(&_ch[5], ht8,  TIM_CHANNEL_4, GPIOC, GPIO_PIN_9);
    _set(&_ch[6], ht16, TIM_CHANNEL_1, GPIOF, GPIO_PIN_6);
    _set(&_ch[7], ht17, TIM_CHANNEL_1, GPIOF, GPIO_PIN_7);
    _tbase = ht5->Instance;

    /* Per channel: capture off, interrupt off, DMA stream on, stale flags
     * cleared, DMA request on, capture on. Polarity (both edges), filter
     * 0xF and the input mapping come from MX_TIMx_Init (CubeMX). A channel
     * whose stream does not start stays inactive: never read, duty = pin
     * level, counted in the boot line (M6). */
    for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) {
        PwmCh_t *c = &_ch[i];
        TIM_TypeDef *t = c->htim->Instance;
        pwm_edges_init(&c->dec, &cfg, HAL_GetTick());
        t->CCER &= ~c->ccxe;
        t->DIER &= ~(c->ccxie | c->ccxde);
        if (c->hdma == NULL ||
            HAL_DMA_Start(c->hdma, (uint32_t)_ccr(t, c->ch), (uint32_t)_ring[i], PWM_RING_LEN) != HAL_OK) {
            printf("PWMCapture: DMA start FAILED on channel %u\r\n", (unsigned)i);
            continue;
        }
        t->SR = ~(c->ccxif | c->ccxof);   /* rc_w0 */
        t->DIER |= c->ccxde;
        t->CCER |= c->ccxe;
        c->active = 1u;
        n_active++;
    }
    ht5->Instance->CR1  |= TIM_CR1_CEN;
    ht8->Instance->CR1  |= TIM_CR1_CEN;
    ht15->Instance->CR1 |= TIM_CR1_CEN;
    ht16->Instance->CR1 |= TIM_CR1_CEN;
    ht17->Instance->CR1 |= TIM_CR1_CEN;
    printf("PWMCapture: DMA capture on %u of 8 channels (both edges, ring %u), no capture interrupt\r\n",
           n_active, (unsigned)PWM_RING_LEN);
}

void PWMCapture_Service(void)
{
    uint32_t now = HAL_GetTick();
    uint32_t wr1[N_WIN][PWM_NUM_CHANNELS], wr2[N_WIN][PWM_NUM_CHANNELS];
    uint8_t  pa[N_WIN][PWM_NUM_CHANNELS], pb[N_WIN][PWM_NUM_CHANNELS], pc[N_WIN][PWM_NUM_CHANNELS];
    uint32_t nwin = 1u;

    _svc++;
    for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) {
        PwmCh_t *c = &_ch[i];
        if (!c->active) continue;
        if (!(((DMA_Stream_TypeDef *)c->hdma->Instance)->CR & DMA_SxCR_EN)) {
            c->active = 0u;            /* stream stopped (transfer error): stop reading it (M6) */
            c->dma_err++;
            continue;
        }
        uint32_t w = pwm_edges_windows(&c->dec, now);
        if (w > nwin) nwin = w;
    }

    /* Windows first, at absolute times on the 10 MHz base (M10):
     * wr1, pin a (0), pin b (1.7 us), pin c (3.5 us), wr2 */
    uint16_t t0 = _now_ticks();
    for (uint32_t k = 0; k < nwin; k++) {
        uint32_t base = k * WIN_STEP_TICKS;
        _wait_until(t0, base);
        for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) if (_ch[i].active) wr1[k][i] = _wr(&_ch[i]);
        for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) if (_ch[i].active) pa[k][i]  = (uint8_t)_pin(&_ch[i]);
        _wait_until(t0, base + PIN_B_TICKS);
        for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) if (_ch[i].active) pb[k][i]  = (uint8_t)_pin(&_ch[i]);
        _wait_until(t0, base + PIN_C_TICKS);
        for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) if (_ch[i].active) pc[k][i]  = (uint8_t)_pin(&_ch[i]);
        for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) if (_ch[i].active) wr2[k][i] = _wr(&_ch[i]);
    }

    uint8_t probe = 0u;
    for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) {
        PwmCh_t *c = &_ch[i];
        if (!c->active) { c->duty = (float)_pin(c); continue; }
        TIM_TypeDef *t = c->htim->Instance;
        if (t->SR & c->ccxof) {
            /* Lost capture (M4): clear first, then take the write index, so
             * every entry that may follow the loss is dropped; no anchor from
             * windows taken before the loss. */
            t->SR = ~c->ccxof;         /* rc_w0: clears only CCxOF */
            pwm_edges_process(&c->dec, _ring[i], _wr(c), 1, now);
        } else {
            for (uint32_t k = 0; k < nwin; k++) {
                /* read only up to wr1: those entries are >= 3 us old (M7) */
                pwm_edges_process(&c->dec, _ring[i], wr1[k][i], 0, now);
                pwm_edges_anchor(&c->dec, _svc, now, wr1[k][i], wr2[k][i], pa[k][i], pb[k][i], pc[k][i]);
            }
        }
        if (c->dec.lap_pending) probe |= (uint8_t)(1u << i);
        c->duty = pwm_edges_duty(&c->dec, now, pc[nwin - 1u][i]);
    }

    /* After a lap (M3): a channel counts as active only if a new capture
     * arrives within one PWM period or more; otherwise its last edge keeps
     * its old age and the 50 ms timeout runs as before. */
    if (probe) {
        uint16_t p0 = _now_ticks();
        uint8_t left = probe;
        while (left && (uint16_t)(_now_ticks() - p0) < PROBE_TICKS) {
            for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) {
                if (!(left & (1u << i))) continue;
                if (_wr(&_ch[i]) != _ch[i].dec.rd) {
                    pwm_edges_mark_active(&_ch[i].dec, now);
                    left &= (uint8_t)~(1u << i);
                }
            }
        }
        for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++)
            if (probe & (1u << i)) _ch[i].duty = pwm_edges_duty(&_ch[i].dec, now, _pin(&_ch[i]));
    }
}

void PWMCapture_GetDuty(float *duty_out)
{
    for (uint8_t w = 0; w < PWM_NUM_WHEELS; w++)
        duty_out[w] = _ch[w].duty - _ch[w + 4].duty;
}

void PWMCapture_GetStats(PWMCapture_Stats_t *out)
{
    for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) {
        out->edges[i]      = _ch[i].dec.n_edges;
        out->lost[i]       = _ch[i].dec.n_lost;
        out->lap[i]        = _ch[i].dec.n_lap;
        out->parity_fix[i] = _ch[i].dec.n_parity_fix;
        out->bad_period[i] = _ch[i].dec.n_bad_period;
        out->dma_err[i]    = _ch[i].dma_err;
        out->active[i]     = _ch[i].active;
        out->known[i]      = _ch[i].dec.known;
    }
}
