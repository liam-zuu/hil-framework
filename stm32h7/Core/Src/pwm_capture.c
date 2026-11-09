#include "pwm_capture.h"
#include "pwm_edges.h"
#include <string.h>
#include <stdio.h>

/* 05/11 (Liam: "Duyet, viet di"): this file only is built at -O2, the rest of
 * the project stays at the Debug -O0. Reason: the anchor window was 14.7 us
 * at best at -O0 with the I-cache on (PWM_CAPTURE/2026-11-05_run03), M23
 * needs below about 7 us. Safe for the DMA rings: this file never reads
 * _ring itself (only passes its address to HAL_DMA_Start and pwm_edges.c,
 * another unit), and every register access goes through the CMSIS volatile
 * types. */
#pragma GCC optimize ("O2")

/* HAL side of the DMA capture (see pwm_capture.h); the decoding is in
 * pwm_edges.c, tested on the PC (pwmedges_pc). Codex Sol round 1 (04/11,
 * CLAUDE_THESIS 03_CODE/docs/review/2026-11-04_Review_H7_PWM_DMA_vong1.md):
 * fixes M1-M4, M6, M7, M10, M12 (Liam: "Duyệt sửa cả 9"). Round 2
 * (..._vong2.md, Liam: "mở vòng 3, sửa và gọi codex ở phiên mới"): M14 window
 * start moves per service + unanchored counter, M15-M17 probe after every lap,
 * M18 gaps counted from the end of the previous read group, M22 a stream that
 * does not start counts in dma_err. Round 3 (..._vong3.md, Liam: "sửa M24 và
 * M26 luôn, M13 và M23 theo khuyến nghị"): M24 last index check after the
 * probe loop; M23 accepted, window and service time maxima for the board. */

#define LAP_MS          5u   /* below the 6.5 ms counter wrap (pwm_edges.h) */
#define STALE_MS        5u
#define MAX_PERIOD    750u   /* ticks, 75 us = 1.5 x the 20 kHz period (M2) */
#define VERIFY_MS      10u   /* while known: full window set this often (M1) */
#define UNANCHORED_MS  20u   /* edges but edge type unknown this long: counted (M14) */
#define WIN_STEP_TICKS 100u  /* window starts 10 us apart, from one TIM5 CNT reading (M10) */
#define WIN_SHIFT_TICKS 25u  /* 3 windows: start + (service mod 4) x 2.5 us, breaks a phase lock (M14) */
#define PIN_GAP_TICKS   12u  /* pin b >= 1.2 us after the END of the pin a reads, pin c after pin b:
                                one glitch the input filter drops (< 1.07 us) covers one read only (M1, M18) */
#define WR2_GAP_TICKS   35u  /* wr2 >= 3.5 us after the END of the pin a reads: > input filter + DMA (M7, M18) */
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
    uint32_t           dma_err;    /* stream did not start at init, or found disabled while active (M6, M22) */
    PwmEdges_t         dec;
    float              duty;
} PwmCh_t;

static PwmCh_t _ch[PWM_NUM_CHANNELS];
/* Rings in .bss = RAM_D1 (AXI SRAM, reachable by DMA2; D-cache is off) */
static uint32_t _ring[PWM_NUM_CHANNELS][PWM_RING_LEN];
static TIM_TypeDef *_tbase;        /* 10 MHz counter used to time the windows */
static uint32_t _svc;              /* service counter, anchor "different service" test */
/* Largest window (wr1 group start to wr2 group end) and service time in
 * 10 MHz ticks since the last PWMCapture_TakeTiming(): the window shift (M14)
 * breaks a phase lock only while a window is shorter than about 3/4 of the
 * 10 us step, i.e. about 7 us (Codex M23; Liam: measure on the board). */
static uint16_t _win_max, _svc_max;
/* DIAG 05/11 (bản chẩn đoán, không dùng đo): win was 65-107 us on every 2 s
 * line on the board (PWM_CAPTURE/2026-11-05_run01, ENC_PI/2026-11-05_run03).
 * Count every window by length, < 7 us / 7-20 us / 20-50 us / >= 50 us, and
 * keep the shortest, to tell "a few windows preempted" from "every window
 * long". Read and reset by PWMCapture_TakeWinHist(). */
#define WIN_EDGE_7US   70u
#define WIN_EDGE_20US 200u
#define WIN_EDGE_50US 500u
static uint32_t _win_hist[4];
static uint16_t _win_min = 0xFFFFu;

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
    const PwmEdgesCfg_t cfg = { PWM_RING_LEN, LAP_MS, STALE_MS, PWM_TIMEOUT_MS, MAX_PERIOD, VERIFY_MS,
                                UNANCHORED_MS };
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
     * level, counted in the boot line (M6) and in dma_err (M22). */
    for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) {
        PwmCh_t *c = &_ch[i];
        TIM_TypeDef *t = c->htim->Instance;
        pwm_edges_init(&c->dec, &cfg, HAL_GetTick());
        t->CCER &= ~c->ccxe;
        t->DIER &= ~(c->ccxie | c->ccxde);
        if (c->hdma == NULL ||
            HAL_DMA_Start(c->hdma, (uint32_t)_ccr(t, c->ch), (uint32_t)_ring[i], PWM_RING_LEN) != HAL_OK) {
            printf("PWMCapture: DMA start FAILED on channel %u\r\n", (unsigned)i);
            c->dma_err++;
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
    uint16_t ts = _now_ticks();        /* service start, for the timing maxima (M23) */
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

    /* Windows first (M10): window k starts at shift + k x 10 us after one
     * reading of the 10 MHz base (at once if that time has passed). Inside a
     * window each wait is counted from the END of the read group it depends
     * on, so a preemption or a late start only makes a gap longer (M18):
     *   wr1 | pin a | >= 1.2 us | pin b | >= 1.2 us | pin c | wr2 >= 3.5 us after pin a.
     * Per channel the read of one group comes after the end of the group
     * before it, so these are lower bounds for every channel. */
    /* 05/11: register addresses of the active channels taken once, so each
     * read group is a tight loop of register reads (no call, no pointer
     * chain per channel). The DMA counters are kept raw (NDTR) inside the
     * window and turned into ring indices after it, same values as _wr(). */
    volatile uint32_t *ndtr[PWM_NUM_CHANNELS];
    volatile uint32_t *idr[PWM_NUM_CHANNELS];
    uint32_t msk[PWM_NUM_CHANNELS];
    uint8_t  act[PWM_NUM_CHANNELS];
    uint32_t na = 0u;
    for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) {
        if (!_ch[i].active) continue;
        ndtr[na] = &((DMA_Stream_TypeDef *)_ch[i].hdma->Instance)->NDTR;
        idr[na]  = &_ch[i].port->IDR;
        msk[na]  = _ch[i].pin;
        act[na]  = i;
        na++;
    }
    uint32_t n1[N_WIN][PWM_NUM_CHANNELS], n2[N_WIN][PWM_NUM_CHANNELS];

    uint16_t t0 = _now_ticks();
    uint32_t shift = (nwin == N_WIN) ? (_svc & 3u) * WIN_SHIFT_TICKS : 0u;   /* M14 */
    for (uint32_t k = 0; k < nwin; k++) {
        _wait_until(t0, shift + k * WIN_STEP_TICKS);
        uint16_t tw = _now_ticks();
        for (uint32_t j = 0; j < na; j++) n1[k][j] = *ndtr[j];
        for (uint32_t j = 0; j < na; j++) pa[k][act[j]] = (uint8_t)((*idr[j] & msk[j]) != 0u);
        uint16_t ta = _now_ticks();
        _wait_until(ta, PIN_GAP_TICKS);
        for (uint32_t j = 0; j < na; j++) pb[k][act[j]] = (uint8_t)((*idr[j] & msk[j]) != 0u);
        uint16_t tb = _now_ticks();
        _wait_until(tb, PIN_GAP_TICKS);
        for (uint32_t j = 0; j < na; j++) pc[k][act[j]] = (uint8_t)((*idr[j] & msk[j]) != 0u);
        _wait_until(ta, WR2_GAP_TICKS);
        for (uint32_t j = 0; j < na; j++) n2[k][j] = *ndtr[j];
        uint16_t span = (uint16_t)(_now_ticks() - tw);   /* wr1 group start to wr2 group end (M23) */
        if (span > _win_max) _win_max = span;
        if (span < _win_min) _win_min = span;                    /* DIAG 05/11 */
        _win_hist[(span < WIN_EDGE_7US) ? 0 : (span < WIN_EDGE_20US) ? 1 : (span < WIN_EDGE_50US) ? 2 : 3]++;
    }
    /* Raw DMA counters to ring write indices, as _wr() does */
    for (uint32_t k = 0; k < nwin; k++)
        for (uint32_t j = 0; j < na; j++) {
            wr1[k][act[j]] = (PWM_RING_LEN - n1[k][j]) % PWM_RING_LEN;
            wr2[k][act[j]] = (PWM_RING_LEN - n2[k][j]) % PWM_RING_LEN;
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

    /* After every lap (M3, M15): a channel counts as active only if a new
     * capture arrives within one PWM period or more (the write index moves
     * away from rd; this also sees an entry written after wr1 of the last
     * window); otherwise its last edge keeps its old age and the 50 ms
     * timeout runs as before. Either way the probe ends here (M16). */
    if (probe) {
        uint16_t p0 = _now_ticks();
        uint8_t left = probe;
        while (left && (uint16_t)(_now_ticks() - p0) < PROBE_TICKS) {
            for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) {
                if (!(left & (1u << i))) continue;
                if (_wr(&_ch[i]) != _ch[i].dec.rd) {
                    pwm_edges_probe_done(&_ch[i].dec, now, 1);
                    left &= (uint8_t)~(1u << i);
                }
            }
        }
        for (uint8_t i = 0; i < PWM_NUM_CHANNELS; i++) {
            if (!(probe & (1u << i))) continue;
            /* One last look after the loop: a preemption past the deadline
             * may have ended the loop before captures that came meanwhile
             * were seen (Codex M24). */
            if (left & (1u << i))
                pwm_edges_probe_done(&_ch[i].dec, now, _wr(&_ch[i]) != _ch[i].dec.rd);
            _ch[i].duty = pwm_edges_duty(&_ch[i].dec, now, _pin(&_ch[i]));
        }
    }

    uint16_t svc_ticks = (uint16_t)(_now_ticks() - ts);
    if (svc_ticks > _svc_max) _svc_max = svc_ticks;
}

void PWMCapture_TakeTiming(uint32_t *win_max_ticks, uint32_t *svc_max_ticks)
{
    *win_max_ticks = _win_max;
    *svc_max_ticks = _svc_max;
    _win_max = 0u;
    _svc_max = 0u;
}

/* DIAG 05/11: window counts by length and shortest window (10 MHz ticks)
 * since the last call; resets them. Main loop only, like TakeTiming. */
void PWMCapture_TakeWinHist(uint32_t hist[4], uint32_t *win_min_ticks)
{
    for (int i = 0; i < 4; i++) { hist[i] = _win_hist[i]; _win_hist[i] = 0u; }
    *win_min_ticks = (_win_min == 0xFFFFu) ? 0u : _win_min;
    _win_min = 0xFFFFu;
}

/* Sign (07/11, bug HIL-PWM-WIRE-SIGN, Liam: "đổi dây lại như cũ luôn từ bây
 * giờ, sửa code ngay đi"): the bench wires channels 0-3 to the ESP32 pins
 * named *_LPWM in motor.h (GPIO 3, 41, 9, 21) and channels 4-7 to *_RPWM
 * (GPIO 8, 40, 46, 47), see pwm_capture.h. On the car the *_RPWM pin drives
 * the BTS7960 RPWM input, so the driver-equivalent duty is channel w+4 minus
 * channel w. This is the input the plant expects (rpi5/params.py
 * motor_polarity from S01, fwsim/fw_sim.c duty = RPWM pin - LPWM pin). The
 * old sign (channel w minus w+4) fed the plant the opposite duty and the
 * first closed loop ran away (CLOSED_LOOP/2026-11-07_run01). */
void PWMCapture_GetDuty(float *duty_out)
{
    for (uint8_t w = 0; w < PWM_NUM_WHEELS; w++)
        duty_out[w] = _ch[w + 4].duty - _ch[w].duty;
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
        out->unanchored[i] = _ch[i].dec.n_unanchored;
        out->active[i]     = _ch[i].active;
        out->known[i]      = _ch[i].dec.known;
    }
}
