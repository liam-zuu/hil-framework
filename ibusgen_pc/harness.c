/* harness.c — PC test of the H7 iBUS generator (stm32h7/Core/Src/ibus_gen.c,
 * the same file the H7 runs). Models the UART TX side as events: a 16-byte
 * FIFO in front of the shift register, 115200 8N1 (86.806 us per byte); a
 * byte leaves the FIFO only when the shift register is free and the line
 * is not held. TC (tx_idle) = FIFO empty, last byte out, line not held.
 * The TIM6 tick is called every tick_us; the generator gets that time.
 * Prints one JSON object per run on stdout.
 *
 *   harness frame
 *   harness run <ticks> [events...]
 *     off@T on@T     generator gate off / on from tick T
 *     swa0@T swa1@T  SWA asked off / on from tick T (I8; up to 8 of each;
 *                    default on from tick 1, so the older tests keep SWA 2000)
 *     block@T:N      line held from tick T for N ticks (no byte starts)
 *     stall@T:N      TIM6 interrupt not served for ticks T..T+N-1
 *     period=US      period passed to ibus_gen_init
 *     tickus=X       real time between ticks (default 100)
 */
#include "ibus_gen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BYTE_US   (1e6 * 10.0 / 115200.0)
#define FIFO_LEN  16
#define MAX_BYTES 2000000

static double   g_now;                  /* us, time of the current tick */
static double   g_start[MAX_BYTES];     /* wire start time of each byte */
static uint8_t  g_val[MAX_BYTES];
static long     g_n;                    /* bytes written by the generator */
static long     g_sent;                 /* bytes already on the wire (g_start valid) */
static double   g_shift_free;           /* shift register free from */
static double   g_b0 = -1, g_b1 = -1;   /* line held in [b0, b1) */

static int _held(double t) { return t >= g_b0 && t < g_b1; }

/* Move FIFO bytes onto the wire up to g_now */
static void _advance(void)
{
    while (g_sent < g_n) {
        double t = g_shift_free;
        if (_held(t)) t = g_b1;
        if (t > g_now) break;
        g_start[g_sent++] = t;
        g_shift_free = t + BYTE_US;
    }
}

static int _tx_room(void) { _advance(); return (g_n - g_sent) < FIFO_LEN; }

static int _tx_idle(void)
{
    _advance();
    return g_n == g_sent && g_shift_free <= g_now && !_held(g_now);
}

static void _tx_put(uint8_t b)
{
    if (g_n - g_sent >= FIFO_LEN) { fprintf(stderr, "FIFO overflow\n"); exit(3); }
    if (g_n == g_sent && g_shift_free < g_now) g_shift_free = g_now;   /* idle line */
    g_val[g_n++] = b;
    _advance();
}

static void _print_frame(const uint8_t *f)
{
    printf("\"frame\":[");
    for (int i = 0; i < (int)IBUS_GEN_FRAME_LEN; i++) printf(i ? ",%u" : "%u", f[i]);
    printf("]");
}

int main(int argc, char **argv)
{
    static ibus_gen_t g;
    ibus_gen_io_t io = { _tx_room, _tx_idle, _tx_put };
    if (argc >= 2 && strcmp(argv[1], "frame") == 0) {
        ibus_gen_init(&g, &io, IBUS_GEN_PERIOD_US, 0);
        printf("{");
        _print_frame(g.frame);
        printf(",\"frame_off\":[");
        for (int i = 0; i < (int)IBUS_GEN_FRAME_LEN; i++) printf(i ? ",%u" : "%u", g.frame_off[i]);
        printf("],\"last_swa\":%d", ibus_gen_last_swa(&g));
        printf(",\"period_us\":%u,\"min_period_us\":%u}\n",
               (unsigned)g.period_us, (unsigned)IBUS_GEN_MIN_PERIOD_US);
        return 0;
    }
    if (argc < 3 || strcmp(argv[1], "run") != 0) {
        fprintf(stderr, "usage: harness frame | harness run <ticks> [events]\n");
        return 2;
    }
    long ticks = atol(argv[2]);
    long off_at = -1, on_at = -1, blk_at = -1, blk_n = 0, st_at = -1, st_n = 0;
    long swa_at[16]; int swa_val[16]; int n_swa = 0;
    unsigned period = IBUS_GEN_PERIOD_US;
    double tick_us = 100.0;
    for (int i = 3; i < argc; i++) {
        long t_ev;
        if (sscanf(argv[i], "swa0@%ld", &t_ev) == 1 && n_swa < 16) { swa_at[n_swa] = t_ev; swa_val[n_swa++] = 0; continue; }
        if (sscanf(argv[i], "swa1@%ld", &t_ev) == 1 && n_swa < 16) { swa_at[n_swa] = t_ev; swa_val[n_swa++] = 1; continue; }
        if (sscanf(argv[i], "off@%ld", &off_at) == 1) continue;
        if (sscanf(argv[i], "on@%ld", &on_at) == 1) continue;
        if (sscanf(argv[i], "block@%ld:%ld", &blk_at, &blk_n) == 2) continue;
        if (sscanf(argv[i], "stall@%ld:%ld", &st_at, &st_n) == 2) continue;
        if (sscanf(argv[i], "period=%u", &period) == 1) continue;
        if (sscanf(argv[i], "tickus=%lf", &tick_us) == 1) continue;
        fprintf(stderr, "bad event %s\n", argv[i]);
        return 2;
    }
    if (blk_at >= 0) { g_b0 = blk_at * tick_us; g_b1 = (blk_at + blk_n) * tick_us; }
    ibus_gen_init(&g, &io, period, 0);

    printf("{\"period_us\":%u,\"windows\":[", (unsigned)g.period_us);
    uint32_t last_win = 0;
    int first = 1, on = 1, swa = 1;
    for (long t = 1; t <= ticks; t++) {
        if (t == off_at) on = 0;
        if (t == on_at)  on = 1;
        for (int k = 0; k < n_swa; k++)
            if (t == swa_at[k]) swa = swa_val[k];
        if (st_at >= 0 && t >= st_at && t < st_at + st_n) continue;   /* not served */
        g_now = t * tick_us;
        ibus_gen_tick(&g, (uint32_t)(g_now + 0.5), on, swa);
        if (g_n >= MAX_BYTES - 64) { fprintf(stderr, "byte buffer full\n"); return 2; }
        ibus_gen_window_t w;
        if (ibus_gen_read_window(&g, &w) && w.window != last_win) {
            last_win = w.window;
            printf("%s{\"window\":%u,\"frames\":%u,\"p_count\":%u,\"p_min\":%u,\"p_max\":%u,"
                   "\"late\":%u,\"tick_gap_max\":%u,\"enables\":%u,\"total\":%u,\"enabled\":%u,"
                   "\"swa_frames\":%u,\"swa_changes\":%u,\"swa\":%u}",
                   first ? "" : ",", (unsigned)w.window, (unsigned)w.frames,
                   (unsigned)w.p_count, (unsigned)w.p_min_us, (unsigned)w.p_max_us,
                   (unsigned)w.late_ticks, (unsigned)w.tick_gap_max_us, (unsigned)w.enables,
                   (unsigned)w.total_frames, (unsigned)w.enabled,
                   (unsigned)w.swa_frames, (unsigned)w.swa_changes, (unsigned)w.swa);
            first = 0;
        }
    }
    g_now = 1e18;                       /* let the FIFO drain */
    _advance();
    printf("],\"total_frames\":%u,\"last_swa\":%d,\"bytes\":%ld,\"t_us\":[",
           (unsigned)g.total_frames, ibus_gen_last_swa(&g), g_n);
    for (long i = 0; i < g_sent; i++) printf(i ? ",%.3f" : "%.3f", g_start[i]);
    printf("],\"val\":[");
    for (long i = 0; i < g_n; i++) printf(i ? ",%u" : "%u", g_val[i]);
    printf("]}\n");
    return 0;
}
