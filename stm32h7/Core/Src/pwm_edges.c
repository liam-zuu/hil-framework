#include "pwm_edges.h"
#include <string.h>

/* See pwm_edges.h. Plain C: built for the H7 and for the PC test
 * (pwmedges_pc/harness.c) from this same file. */

static void _drop_open_period(PwmEdges_t *e)
{
    e->have_rise = 0u;
    e->have_fall = 0u;
}

static void _unknown(PwmEdges_t *e)
{
    e->known = 0u;
    e->cand_valid = 0u;
    _drop_open_period(e);
}

void pwm_edges_init(PwmEdges_t *e, const PwmEdgesCfg_t *cfg, uint32_t now_ms)
{
    memset(e, 0, sizeof(*e));
    e->cfg = *cfg;
    e->last_svc_ms = now_ms;     /* capture started: a long wait before the first service is a lap (M12) */
}

void pwm_edges_process(PwmEdges_t *e, const volatile uint32_t *ring, uint32_t wr,
                       int lost, uint32_t now_ms)
{
    const uint32_t len = e->cfg.ring_len;
    if (wr >= len) wr %= len;

    /* Gap since the previous call. A gap of lap_ms or more may hide a full
     * lap of the ring (rd == wr although entries arrived) and, with lap_ms
     * below the 6.5 ms counter wrap, also bounds the time between two entries
     * read in one call. */
    const uint32_t prev_svc = e->last_svc_ms;
    int lap = (uint32_t)(now_ms - e->last_svc_ms) >= e->cfg.lap_ms;
    e->last_svc_ms = now_ms;

    if (lap || lost) {
        if (lap)  e->n_lap++;
        if (lost) e->n_lost++;
        /* Edge types can no longer be counted through: skip to the write
         * index and wait for new anchors. After a lap the entries are of
         * unknown age, so they do not count as activity (M3); an overcapture
         * means a capture just happened. */
        if (lost) {
            e->any_edge = 1u;
            e->last_edge_ms = now_ms;
            e->edge_svc_lo = prev_svc;
            e->lap_pending = 0u;
        } else if (e->rd != wr) {
            e->lap_pending = 1u;
        }
        e->rd = wr;
        _unknown(e);
        return;
    }

    if (e->rd == wr) return;
    e->lap_pending = 0u;

    /* The last edge read before was seen at the service after edge_svc_lo,
     * so it is at most (now_ms - edge_svc_lo) + 1 ms old: from stale_ms on,
     * the next difference could pass the 16-bit counter wrap. */
    if ((e->have_rise || e->have_fall) &&
        (uint32_t)(now_ms - e->edge_svc_lo) >= e->cfg.stale_ms) {
        e->n_stale++;
        _drop_open_period(e);
    }

    while (e->rd != wr) {
        uint16_t t = (uint16_t)ring[e->rd];
        e->rd = (e->rd + 1u == len) ? 0u : e->rd + 1u;
        e->n_edges++;
        e->level ^= 1u;
        if (!e->known) continue;

        if (e->level) {                       /* rising edge */
            if (e->have_rise && e->have_fall) {
                uint16_t period = (uint16_t)(t - e->t_rise);
                uint16_t width  = (uint16_t)(e->t_fall - e->t_rise);
                if (period > e->cfg.max_period) {
                    e->n_bad_period++;        /* gap or dropped pulse, not a PWM period (M2) */
                } else if (period > 0u) {
                    float d = (float)width / (float)period;
                    if (d > 1.0f) d = 1.0f;
                    e->duty = e->duty * 0.75f + d * 0.25f;
                    e->n_periods++;
                }
            }
            e->t_rise = t;
            e->have_rise = 1u;
            e->have_fall = 0u;
        } else if (e->have_rise) {            /* falling edge after a rise */
            e->t_fall = t;
            e->have_fall = 1u;
        }
    }
    e->any_edge = 1u;
    e->last_edge_ms = now_ms;
    e->edge_svc_lo = prev_svc;
}

void pwm_edges_anchor(PwmEdges_t *e, uint32_t svc, uint32_t now_ms,
                      uint32_t wr1, uint32_t wr2, int pin_a, int pin_b, int pin_c)
{
    if (wr1 != wr2) return;                   /* a capture came in the window */
    int a = pin_a != 0, b = pin_b != 0, c = pin_c != 0;
    if (a != b || b != c) return;             /* pin moved: edge not captured yet, or a glitch */
    if (e->rd != wr2 % e->cfg.ring_len) return;   /* ring not read up to the window */

    uint8_t pin = (uint8_t)a;
    if (e->known) {
        if (e->level == pin) {
            e->verified_ms = now_ms;
        } else {                              /* disagree: start over, two anchors again */
            e->n_parity_fix++;
            _unknown(e);
            e->level = pin;
            e->cand_valid = 1u;
            e->cand_svc = svc;
        }
        return;
    }
    if (!e->cand_valid) {                     /* first anchor: candidate only */
        e->level = pin;
        e->cand_valid = 1u;
        e->cand_svc = svc;
        _drop_open_period(e);
    } else if (svc == e->cand_svc) {          /* same service: keep the latest reading */
        e->level = pin;
    } else if (e->level == pin) {             /* second service agrees: known */
        e->known = 1u;
        e->verified_ms = now_ms;
        _drop_open_period(e);
    } else {                                  /* second service disagrees: new candidate */
        e->n_cand_retry++;
        e->level = pin;
        e->cand_svc = svc;
    }
}

void pwm_edges_mark_active(PwmEdges_t *e, uint32_t now_ms)
{
    e->any_edge = 1u;
    e->last_edge_ms = now_ms;
    e->edge_svc_lo = now_ms;
    e->lap_pending = 0u;
}

uint32_t pwm_edges_windows(const PwmEdges_t *e, uint32_t now_ms)
{
    if (!e->known || (uint32_t)(now_ms - e->verified_ms) >= e->cfg.verify_ms) return 3u;
    return 1u;
}

float pwm_edges_duty(const PwmEdges_t *e, uint32_t now_ms, int pin)
{
    if (!e->any_edge || (uint32_t)(now_ms - e->last_edge_ms) >= e->cfg.timeout_ms)
        return (pin != 0) ? 1.0f : 0.0f;
    return e->duty;
}
