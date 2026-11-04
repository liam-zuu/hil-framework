#include "pwm_capture.h"
#include "pwm_edges.h"
#include <string.h>
#include <stdio.h>

/* HAL side of the DMA capture (see pwm_capture.h); the decoding is in
 * pwm_edges.c, tested on the PC (pwmedges_pc). */

#define LAP_MS        5u     /* below the 6.5 ms counter wrap (pwm_edges.h) */
#define STALE_MS      5u
#define WIN_TICKS    30u     /* anchor window, 3 us at 10 MHz: > input filter (about 1.07 us) + DMA */
#define TRY_GAP_TICKS 40u    /* pause between two windows: windows start 7 us apart */
#define N_TRY         3u     /* windows per service while an edge type is unknown */

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
    PwmEdges_t         dec;
    float              duty;
} PwmCh_t;

static PwmCh_t _ch[PWM_NUM_CHANNELS];
/* Rings in .bss = RAM_D1 (AXI SRAM, reachable by DMA2; D-cache is off) */
static uint32_t _ring[PWM_NUM_CHANNELS][PWM_RING_LEN];
static TIM_TypeDef *_tbase;        /* 10 MHz counter used to time the windows */

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

static void _wait_ticks(uint32_t n)
{
    uint16_t t0 = (uint16_t)_tbase->CNT;
    while ((uint16_t)((uint16_t)_tbase->CNT - t0) < n) { }
}

void PWMCapture_Init(TIM_HandleTypeDef *ht5,
                     TIM_HandleTypeDef *ht8,
                     TIM_HandleTypeDef *ht15,
                     TIM_HandleTypeDef *ht16,
                     TIM_HandleTypeDef *ht17)
{
    const PwmEdgesCfg_t cfg = { PWM_RING_LEN, LAP_MS, STALE_MS, PWM_TIMEOUT_MS };

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
     * 0xF and the input mapping come from MX_TIMx_Init (CubeMX). */
    for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) {
        PwmCh_t *c = &_ch[i];
        TIM_TypeDef *t = c->htim->Instance;
        pwm_edges_init(&c->dec, &cfg);
        t->CCER &= ~c->ccxe;
        t->DIER &= ~(c->ccxie | c->ccxde);
        if (c->hdma == NULL ||
            HAL_DMA_Start(c->hdma, (uint32_t)_ccr(t, c->ch), (uint32_t)_ring[i], PWM_RING_LEN) != HAL_OK) {
            printf("PWMCapture: DMA start failed on channel %u\r\n", (unsigned)i);
            continue;
        }
        t->SR = ~(c->ccxif | c->ccxof);   /* rc_w0 */
        t->DIER |= c->ccxde;
        t->CCER |= c->ccxe;
    }
    ht5->Instance->CR1  |= TIM_CR1_CEN;
    ht8->Instance->CR1  |= TIM_CR1_CEN;
    ht15->Instance->CR1 |= TIM_CR1_CEN;
    ht16->Instance->CR1 |= TIM_CR1_CEN;
    ht17->Instance->CR1 |= TIM_CR1_CEN;
    printf("PWMCapture: DMA capture on 8 channels (both edges, ring %u), no capture interrupt\r\n",
           (unsigned)PWM_RING_LEN);
}

void PWMCapture_Service(void)
{
    uint32_t now = HAL_GetTick();
    uint32_t wr1[PWM_NUM_CHANNELS], wr2[PWM_NUM_CHANNELS];
    int      p1[PWM_NUM_CHANNELS], p2[PWM_NUM_CHANNELS];
    uint8_t  todo = 0xFFu;                 /* channels still in this service */

    for (uint32_t k = 0; k < N_TRY && todo; k++) {
        if (k) _wait_ticks(TRY_GAP_TICKS);
        /* window: wr1, pin, 3 us, pin, wr2 (order matters, pwm_edges.h) */
        for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) if (todo & (1u << i)) wr1[i] = _wr(&_ch[i]);
        for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) if (todo & (1u << i)) p1[i]  = _pin(&_ch[i]);
        _wait_ticks(WIN_TICKS);
        for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) if (todo & (1u << i)) p2[i]  = _pin(&_ch[i]);
        for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) if (todo & (1u << i)) wr2[i] = _wr(&_ch[i]);

        for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) {
            if (!(todo & (1u << i))) continue;
            PwmCh_t *c = &_ch[i];
            TIM_TypeDef *t = c->htim->Instance;
            int lost = (t->SR & c->ccxof) != 0u;
            if (lost) t->SR = ~c->ccxof;   /* rc_w0: clears only CCxOF */
            pwm_edges_process(&c->dec, _ring[i], wr2[i], lost, now);
            pwm_edges_anchor(&c->dec, wr1[i], wr2[i], p1[i], p2[i]);
            c->duty = pwm_edges_duty(&c->dec, now, p2[i]);
            if (c->dec.known) todo &= (uint8_t)~(1u << i);
        }
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
        out->known[i]      = _ch[i].dec.known;
    }
}
