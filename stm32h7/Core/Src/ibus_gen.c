/* ibus_gen.c — FlySky iBUS frame generator (see ibus_gen.h) */

#include "ibus_gen.h"
#include <string.h>

/* Orders the plain pub copy against the volatile pub_ver writes. ISR and
 * main loop run on the same core, so a compiler barrier is enough. */
#define _BARRIER()  __asm volatile ("" ::: "memory")

void ibus_gen_build_frame(uint8_t out[IBUS_GEN_FRAME_LEN], const uint16_t ch[IBUS_GEN_NUM_CH])
{
    out[0] = IBUS_GEN_HEADER_0;
    out[1] = IBUS_GEN_HEADER_1;
    for (uint32_t i = 0; i < IBUS_GEN_NUM_CH; i++) {
        out[2u + 2u * i] = (uint8_t)(ch[i] & 0xFFu);
        out[3u + 2u * i] = (uint8_t)(ch[i] >> 8);
    }
    uint16_t sum = 0xFFFFu;
    for (uint32_t i = 0; i < IBUS_GEN_FRAME_LEN - 2u; i++)
        sum = (uint16_t)(sum - out[i]);
    out[30] = (uint8_t)(sum & 0xFFu);
    out[31] = (uint8_t)(sum >> 8);
}

void ibus_gen_init(ibus_gen_t *g, const ibus_gen_io_t *io, uint32_t period_ticks)
{
    memset(g, 0, sizeof(*g));
    g->io = *io;
    g->period_ticks = period_ticks < IBUS_GEN_MIN_PERIOD_TICKS ? IBUS_GEN_MIN_PERIOD_TICKS
                                                               : period_ticks;
    uint16_t ch[IBUS_GEN_NUM_CH];
    for (uint32_t i = 0; i < IBUS_GEN_NUM_CH; i++) ch[i] = IBUS_GEN_VAL_MID;
    ch[IBUS_GEN_CH_SWC] = IBUS_GEN_VAL_SWC;
    ch[IBUS_GEN_CH_SWA] = IBUS_GEN_VAL_SWA;
    ibus_gen_build_frame(g->frame, ch);
    g->pos = IBUS_GEN_FRAME_LEN;
}

void ibus_gen_set_enabled(ibus_gen_t *g, int on)
{
    g->enable_req = on ? 1u : 0u;
}

static void _publish(ibus_gen_t *g)
{
    g->acc.window = ++g->win_no;
    g->acc.total_frames = g->total_frames;
    g->acc.enabled = g->enabled;
    g->pub_ver++;
    _BARRIER();
    g->pub = g->acc;
    _BARRIER();
    g->pub_ver++;
    memset(&g->acc, 0, sizeof(g->acc));
}

void ibus_gen_tick(ibus_gen_t *g)
{
    g->tick++;

    uint8_t req = g->enable_req;
    if (req != g->enabled) {
        g->enabled = req;
        if (req) g->acc.enables++;
        else     g->have_last = 0;     /* the gap while off is not a P */
    }

    /* A frame being written always finishes, also after a disable, so the
     * DUT never sees a cut frame from us. */
    int due = g->enabled &&
              (!g->have_last || (uint32_t)(g->tick - g->last_start) >= g->period_ticks);
    if (due && g->pos < IBUS_GEN_FRAME_LEN) {
        g->acc.late_ticks++;
    } else if (due) {
        if (g->have_last) {
            uint32_t p = g->tick - g->last_start;
            if (g->acc.p_count == 0u || p < g->acc.p_min_ticks) g->acc.p_min_ticks = p;
            if (p > g->acc.p_max_ticks) g->acc.p_max_ticks = p;
            g->acc.p_count++;
        }
        g->last_start = g->tick;
        g->have_last = 1u;
        g->pos = 0u;
        g->acc.frames++;
        g->total_frames++;
    }

    uint32_t n = 0;
    while (g->pos < IBUS_GEN_FRAME_LEN && n < IBUS_GEN_MAX_PUSH && g->io.tx_room()) {
        g->io.tx_put(g->frame[g->pos++]);
        n++;
    }

    if ((uint32_t)(g->tick - g->win_start) >= IBUS_GEN_WINDOW_TICKS) {
        g->win_start = g->tick;
        _publish(g);
    }
}

int ibus_gen_read_window(const ibus_gen_t *g, ibus_gen_window_t *out)
{
    for (int tries = 0; tries < 4; tries++) {
        uint32_t v1 = g->pub_ver;
        _BARRIER();
        *out = g->pub;
        _BARRIER();
        uint32_t v2 = g->pub_ver;
        if (v1 == v2 && (v1 & 1u) == 0u) return 1;
    }
    return 0;
}
