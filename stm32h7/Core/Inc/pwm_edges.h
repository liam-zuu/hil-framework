#ifndef PWM_EDGES_H
#define PWM_EDGES_H

/*
 * pwm_edges — duty of one PWM input from a ring of edge times (plain C, no HAL)
 *
 * The timer captures BOTH edges (CCxP = CCxNP = 1) and a circular DMA copies
 * every capture value into a ring in RAM. No capture interrupt, so the reading
 * does not depend on interrupt latency (bug H7-PWM-RISE-MISS, 04/11: with the
 * old polarity toggle in the ISR, TIM16 missed the rising edge after the 15 us
 * low phase of RL step 3; 05_DATA/results/PWM_CAPTURE/2026-11-04_run01).
 * Plan: CLAUDE_THESIS 03_CODE/docs/2026-11-04_Ke_hoach_PWMCapture_DMA.md
 *
 * The ring holds times only, not the edge type. Edge types alternate (the input
 * filter drops a short pulse as a pair), so one known level fixes all of them:
 * pwm_edges_anchor() takes the pin level read while no capture arrived in a
 * window longer than the input filter delay plus the DMA latency, and that
 * level is the level after the last edge in the ring.
 *
 * Duty per period = high time / period (rising to rising), then the same
 * exponential filter as before, duty = 0.75 duty + 0.25 d (kept on purpose,
 * Fable F-ND #8, Liam: "10 MHz, chung CubeMX (Khuyến nghị)").
 * No edge for timeout_ms: the output is the pin level (bug H7-PWM-LEVEL).
 *
 * Use from the main loop, per channel:
 *   wr1 = write index; pin1 = pin; wait >= 3 us; pin2 = pin; wr2 = write index;
 *   lost = overcapture flag (read and clear);
 *   pwm_edges_process(e, ring, wr2, lost, now_ms);
 *   pwm_edges_anchor(e, wr1, wr2, pin1, pin2);
 *   duty = pwm_edges_duty(e, now_ms, pin2);
 * Capture values are taken mod 65536 (16-bit counters; TIM5 runs with ARR 65535).
 */

#include <stdint.h>

typedef struct {
    uint32_t ring_len;     /* entries in the DMA ring */
    uint32_t lap_ms;       /* service gap from which the ring may have lapped: drop and re-anchor */
    uint32_t stale_ms;     /* no edge this long: forget the open period (counter wraps in 6.5 ms) */
    uint32_t timeout_ms;   /* no edge this long: output the pin level */
} PwmEdgesCfg_t;

typedef struct {
    PwmEdgesCfg_t cfg;
    uint32_t rd;           /* next ring index to read */
    uint8_t  started;      /* process() called at least once */
    uint8_t  any_edge;     /* at least one edge since init */
    uint8_t  known;        /* 'level' is the real level after the last edge read */
    uint8_t  level;        /* level after the last edge read (1 = high); a guess while !known */
    uint8_t  have_rise;    /* t_rise valid (and level known when it was taken) */
    uint8_t  have_fall;    /* t_fall valid, after t_rise */
    uint16_t t_rise;
    uint16_t t_fall;
    uint32_t last_svc_ms;
    uint32_t last_edge_ms; /* service that read the last edge */
    uint32_t edge_svc_lo;  /* service before that one: the last edge came after it */
    float    duty;
    /* debug counters since init */
    uint32_t n_edges;      /* ring entries read */
    uint32_t n_periods;    /* periods that updated duty */
    uint32_t n_lost;       /* lost flag from the hardware (overcapture) */
    uint32_t n_lap;        /* service gap >= lap_ms */
    uint32_t n_parity_fix; /* anchor disagreed with the counted edge type */
    uint32_t n_stale;      /* open period dropped after stale_ms without edges */
} PwmEdges_t;

void  pwm_edges_init(PwmEdges_t *e, const PwmEdgesCfg_t *cfg);

/* Read ring entries [rd, wr) (indices mod ring_len). lost != 0: the timer
 * reported a capture overwritten before the DMA read it. */
void  pwm_edges_process(PwmEdges_t *e, const volatile uint32_t *ring, uint32_t wr,
                        int lost, uint32_t now_ms);

/* wr1 read before pin1, wr2 read at least (filter delay + DMA latency) after
 * pin1, pin2 read just before wr2. Call after process(e, ring, wr2, ...). */
void  pwm_edges_anchor(PwmEdges_t *e, uint32_t wr1, uint32_t wr2, int pin1, int pin2);

/* Filtered duty [0..1], or the pin level (0 or 1) without an edge for timeout_ms */
float pwm_edges_duty(const PwmEdges_t *e, uint32_t now_ms, int pin);

#endif /* PWM_EDGES_H */
