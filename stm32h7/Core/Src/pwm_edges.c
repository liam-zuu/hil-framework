#include "pwm_edges.h"
#include <string.h>

/* See pwm_edges.h. Plain C: built for the H7 and for the PC test
 * (pwmedges_pc/harness.c) from this same file. */

static void _drop_open_period(PwmEdges_t *e)
{
    e->have_rise = 0u;
    e->have_fall = 0u;
}

void pwm_edges_init(PwmEdges_t *e, const PwmEdgesCfg_t *cfg)
{
    memset(e, 0, sizeof(*e));
    e->cfg = *cfg;
}

void pwm_edges_process(PwmEdges_t *e, const volatile uint32_t *ring, uint32_t wr,
                       int lost, uint32_t now_ms)
{
    const uint32_t len = e->cfg.ring_len;
    if (wr >= len) wr %= len;

    /* Gap since the previous service. A gap of lap_ms or more may hide a full
     * lap of the ring (rd == wr although entries arrived) and, with lap_ms
     * below the 6.5 ms counter wrap, also bounds the time between two entries
     * read in one call. */
    const uint32_t prev_svc = e->started ? e->last_svc_ms : now_ms;
    int lap = e->started && (uint32_t)(now_ms - e->last_svc_ms) >= e->cfg.lap_ms;
    e->started = 1u;
    e->last_svc_ms = now_ms;

    if (lap || lost) {
        if (lap)  e->n_lap++;
        if (lost) e->n_lost++;
        /* Edge types can no longer be counted through: skip to the write
         * index and wait for the next anchor. */
        if (e->rd != wr || lost) {
            e->any_edge = 1u;
            e->last_edge_ms = now_ms;
        }
        e->rd = wr;
        e->known = 0u;
        _drop_open_period(e);
        return;
    }

    if (e->rd == wr) return;

    /* The last edge read before was seen at the service after prev_edge_svc,
     * so it is at most (now_ms - prev_edge_svc) + 1 ms old: from stale_ms on,
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
                if (period > 0u) {
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

void pwm_edges_anchor(PwmEdges_t *e, uint32_t wr1, uint32_t wr2, int pin1, int pin2)
{
    if (wr1 != wr2) return;                   /* a capture came in the window */
    if ((pin1 != 0) != (pin2 != 0)) return;   /* pin moved: edge not captured yet, or a glitch */
    if (e->rd != wr2 % e->cfg.ring_len) return;   /* ring not read up to wr2 */

    uint8_t pin = (pin1 != 0) ? 1u : 0u;
    if (!e->known) {
        e->known = 1u;
        e->level = pin;
        _drop_open_period(e);
    } else if (e->level != pin) {
        e->n_parity_fix++;
        e->level = pin;
        _drop_open_period(e);
    }
}

float pwm_edges_duty(const PwmEdges_t *e, uint32_t now_ms, int pin)
{
    if (!e->any_edge || (uint32_t)(now_ms - e->last_edge_ms) >= e->cfg.timeout_ms)
        return (pin != 0) ? 1.0f : 0.0f;
    return e->duty;
}
