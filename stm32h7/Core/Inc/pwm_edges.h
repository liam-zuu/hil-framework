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
 * pwm_edges_anchor() takes the pin level read three times while no capture
 * arrived in a window longer than the input filter delay plus the DMA latency;
 * that level is the level after the last edge read. The edge type counts as
 * known only after anchors of two different services agree (one anchor can be
 * fooled by short glitches on the read points, Codex M1); while known, an anchor
 * that disagrees drops it back to unknown and counts parity_fix.
 *
 * Duty per period = high time / period (rising to rising), then the same
 * exponential filter as before, duty = 0.75 duty + 0.25 d (kept on purpose,
 * Fable F-ND #8, Liam: "10 MHz, chung CubeMX (Khuyến nghị)"). A period longer
 * than max_period is not a PWM period (a gap between two bursts, or a pulse
 * the filter dropped): it is skipped and counted (Codex M2).
 * No edge for timeout_ms: the output is the pin level (bug H7-PWM-LEVEL).
 * Edges keep coming but the edge type stays unknown for unanchored_ms: the
 * duty is held at its last value, so this is counted (n_unanchored, once per
 * run of unknown, the run ends when the edge type becomes known; Codex M14).
 *
 * Use from the main loop, per service (svc = a counter that changes every service):
 *   n = pwm_edges_windows(e, now_ms)   (largest over the channels sharing the windows)
 *   take n windows on one time base (pwm_capture.c): wr1, pin a, pin b, pin c, wr2.
 *     Each read group waits a minimum time counted from the END of the group
 *     it depends on (pin b >= 1.2 us after pin a, pin c >= 1.2 us after pin b,
 *     wr2 >= 3.5 us after pin a), so a late or preempted read cannot squeeze a
 *     window (Codex M18). While 3 windows are taken, their start moves by
 *     0, 2.5, 5, 7.5 us from one service to the next, so a PWM pattern locked
 *     to the service cannot keep an edge in every window (Codex M14);
 *   lost = overcapture flag; if set: clear it, read the write index again (wr3),
 *          pwm_edges_process(e, ring, wr3, 1, now_ms) and no anchor this service;
 *   else per window k: pwm_edges_process(e, ring, wr1[k], 0, now_ms);
 *                      pwm_edges_anchor(e, svc, now_ms, wr1[k], wr2[k], a, b, c);
 *   if e->lap_pending: watch the write index for one PWM period or more, then
 *          pwm_edges_probe_done(e, now_ms, seen) whatever the result (Codex M16);
 *   duty = pwm_edges_duty(e, now_ms, pin);
 * Only entries up to wr1 (taken before the window) are read, so the newest
 * entries are at least the window length old (Codex M7).
 * After a service gap of lap_ms or more, neither the ring entries nor a
 * pending overcapture flag say when the input last moved, so only the probe
 * does (Codex M3, M15, M17). An input that stopped during the gap and shows no
 * capture in the probe reads its pin level at once, up to timeout_ms earlier
 * than without the gap: the pin does hold that level (M17 part two, accepted).
 * Capture values are taken mod 65536 (16-bit counters; TIM5 runs with ARR 65535).
 */

#include <stdint.h>

typedef struct {
    uint32_t ring_len;     /* entries in the DMA ring */
    uint32_t lap_ms;       /* service gap from which the ring may have lapped: drop and re-anchor */
    uint32_t stale_ms;     /* no edge this long: forget the open period (counter wraps in 6.5 ms) */
    uint32_t timeout_ms;   /* no edge this long: output the pin level */
    uint32_t max_period;   /* ticks; a longer rising-to-rising time is not a PWM period */
    uint32_t verify_ms;    /* while known: ask for the full window set this often */
    uint32_t unanchored_ms;/* edges but edge type unknown this long: count n_unanchored */
} PwmEdgesCfg_t;

typedef struct {
    PwmEdgesCfg_t cfg;
    uint32_t rd;           /* next ring index to read */
    uint8_t  any_edge;     /* at least one edge since init */
    uint8_t  known;        /* 'level' is the real level after the last edge read */
    uint8_t  level;        /* level after the last edge read (1 = high); the candidate while !known */
    uint8_t  cand_valid;   /* !known: 'level' was set by an anchor of service cand_svc */
    uint8_t  have_rise;    /* t_rise valid (and level known when it was taken) */
    uint8_t  have_fall;    /* t_fall valid, after t_rise */
    uint8_t  lap_pending;  /* this service dropped the ring after a gap: probe for activity */
    uint8_t  una_run;      /* 0 none, 1 edges read while unknown since una_start_ms, 2 counted; 0 again once known */
    uint16_t t_rise;
    uint16_t t_fall;
    uint32_t cand_svc;
    uint32_t last_svc_ms;
    uint32_t last_edge_ms; /* service that read the last edge */
    uint32_t edge_svc_lo;  /* service before that one: the last edge came after it */
    uint32_t verified_ms;  /* last agreeing anchor while known */
    uint32_t una_start_ms; /* first service of the current run of unknown with edges */
    float    duty;
    /* debug counters since init */
    uint32_t n_edges;      /* ring entries read */
    uint32_t n_periods;    /* periods that updated duty */
    uint32_t n_lost;       /* lost flag from the hardware (overcapture) */
    uint32_t n_lap;        /* service gap >= lap_ms */
    uint32_t n_parity_fix; /* anchor disagreed with the known edge type */
    uint32_t n_stale;      /* open period dropped after stale_ms without edges */
    uint32_t n_bad_period; /* rising-to-rising longer than max_period, skipped */
    uint32_t n_cand_retry; /* !known: two anchors disagreed, candidate replaced */
    uint32_t n_unanchored; /* runs of unknown with edges reaching unanchored_ms */
} PwmEdges_t;

/* now_ms: time the capture was started (a gap before the first service counts as a lap) */
void  pwm_edges_init(PwmEdges_t *e, const PwmEdgesCfg_t *cfg, uint32_t now_ms);

/* Read ring entries [rd, wr) (indices mod ring_len). lost != 0: the timer
 * reported a capture overwritten before the DMA read it. */
void  pwm_edges_process(PwmEdges_t *e, const volatile uint32_t *ring, uint32_t wr,
                        int lost, uint32_t now_ms);

/* One anchor window (see the use above). Call after process(e, ring, wr1, ...). */
void  pwm_edges_anchor(PwmEdges_t *e, uint32_t svc, uint32_t now_ms,
                       uint32_t wr1, uint32_t wr2, int pin_a, int pin_b, int pin_c);

/* End of the probe after a lap (call it whatever the result). seen != 0: a
 * capture arrived after the drop, so the input is active now; else the last
 * edge keeps its old age. */
void  pwm_edges_probe_done(PwmEdges_t *e, uint32_t now_ms, int seen);

/* Anchor windows this channel wants in this service: 3 while the edge type is
 * unknown or a check is due, else 1 */
uint32_t pwm_edges_windows(const PwmEdges_t *e, uint32_t now_ms);

/* Filtered duty [0..1], or the pin level (0 or 1) without an edge for timeout_ms */
float pwm_edges_duty(const PwmEdges_t *e, uint32_t now_ms, int pin);

#endif /* PWM_EDGES_H */
