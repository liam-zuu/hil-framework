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

void ibus_gen_init(ibus_gen_t *g, const ibus_gen_io_t *io, uint32_t period_us, uint32_t now_us)
{
    memset(g, 0, sizeof(*g));
    g->io = *io;
    g->period_us = period_us < IBUS_GEN_MIN_PERIOD_US ? IBUS_GEN_MIN_PERIOD_US : period_us;
    uint16_t ch[IBUS_GEN_NUM_CH];
    for (uint32_t i = 0; i < IBUS_GEN_NUM_CH; i++) ch[i] = IBUS_GEN_VAL_MID;
    ch[IBUS_GEN_CH_SWC] = IBUS_GEN_VAL_SWC;
    ch[IBUS_GEN_CH_SWA] = IBUS_GEN_VAL_SWA;
    ibus_gen_build_frame(g->frame, ch);
    ch[IBUS_GEN_CH_SWA] = IBUS_GEN_VAL_SWA_OFF;
    ibus_gen_build_frame(g->frame_off, ch);
    g->cur = g->frame_off;
    g->pos = IBUS_GEN_FRAME_LEN;
    g->win_start_us = now_us;
}

static void _publish(ibus_gen_t *g)
{
    g->acc.window = ++g->win_no;
    g->acc.total_frames = g->total_frames;
    g->acc.enabled = g->on;
    g->acc.swa = g->last_swa;
    g->pub_ver++;
    _BARRIER();
    g->pub = g->acc;
    _BARRIER();
    g->pub_ver++;
    memset(&g->acc, 0, sizeof(g->acc));
}

static void _push(ibus_gen_t *g)
{
    uint32_t n = 0;
    while (g->pos < IBUS_GEN_FRAME_LEN && n < IBUS_GEN_MAX_PUSH && g->io.tx_room()) {
        g->io.tx_put(g->cur[g->pos++]);
        n++;
    }
}

void ibus_gen_tick(ibus_gen_t *g, uint32_t now_us, int on, int swa)
{
    if (g->have_tick) {
        uint32_t gap = now_us - g->last_tick_us;
        if (gap > g->acc.tick_gap_max_us) g->acc.tick_gap_max_us = gap;
    }
    g->have_tick = 1u;
    g->last_tick_us = now_us;

    uint8_t want = on ? 1u : 0u;
    if (want != g->on) {
        g->on = want;
        if (want) g->acc.enables++;
        else      g->p_valid = 0u;     /* the gap while off is not a P */
    }

    /* A frame being written always finishes, also after on drops, so the
     * DUT never sees a cut frame from us. */
    _push(g);

    int due = g->on && g->pos >= IBUS_GEN_FRAME_LEN &&
              (!g->have_start || (uint32_t)(now_us - g->last_start_us) >= g->period_us);
    if (due && !g->io.tx_idle()) {
        g->acc.late_ticks++;
    } else if (due) {
        if (g->p_valid) {
            uint32_t p = now_us - g->last_start_us;
            if (g->acc.p_count == 0u || p < g->acc.p_min_us) g->acc.p_min_us = p;
            if (p > g->acc.p_max_us) g->acc.p_max_us = p;
            g->acc.p_count++;
        }
        g->last_start_us = now_us;
        g->have_start = 1u;
        g->p_valid = 1u;
        /* I8: SWA is picked here, at the start of a frame, never inside one */
        uint8_t s = swa ? 1u : 0u;
        if (g->have_swa && s != g->last_swa) g->acc.swa_changes++;
        g->have_swa = 1u;
        g->last_swa = s;
        g->cur = s ? g->frame : g->frame_off;
        if (s) g->acc.swa_frames++;
        g->pos = 0u;
        g->acc.frames++;
        g->total_frames++;
        _push(g);                      /* line idle: the header leaves now */
    }

    if ((uint32_t)(now_us - g->win_start_us) >= IBUS_GEN_WINDOW_US) {
        g->win_start_us = now_us;
        _publish(g);
    }
}

void ibus_swa_filter_reset(ibus_swa_filter_t *f)
{
    f->out = 0u;
    f->cand = 0u;
    f->cnt = 0u;
}

int ibus_swa_filter_step(ibus_swa_filter_t *f, int bit)
{
    uint8_t b = bit ? 1u : 0u;
    if (f->cnt > 0u && b == f->cand) {
        if (f->cnt < 0xFFu) f->cnt++;
    } else {
        f->cand = b;
        f->cnt = 1u;
    }
    if (f->cnt >= IBUS_SWA_FILTER_N) f->out = f->cand;
    return f->out;
}

int ibus_gen_last_swa(const ibus_gen_t *g)
{
    return g->last_swa;          /* one byte: a single read, no tearing */
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
