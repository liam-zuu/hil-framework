/* harness.c — PC test of the H7 iBUS generator (stm32h7/Core/Src/ibus_gen.c,
 * the same file the H7 runs). Models the UART TX side: a 16-byte FIFO in
 * front of the shift register, 115200 8N1 (86.806 us per byte), the TIM6
 * tick every 100 us. Prints one JSON object per run on stdout.
 *
 *   harness frame
 *   harness run <ticks> [events...]
 *     events: off@T on@T block@T:N (no byte leaves the FIFO for N ticks)
 *             period=P (ticks, passed to ibus_gen_init)
 */
#include "ibus_gen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BYTE_US   (1e6 * 10.0 / 115200.0)
#define TICK_US   100.0
#define FIFO_LEN  16
#define MAX_BYTES 2000000

static double   g_now;                  /* us, time of the current tick */
static double   g_start[MAX_BYTES];     /* wire start time of each byte */
static uint8_t  g_val[MAX_BYTES];
static long     g_n;                    /* bytes written */
static double   g_block_until;          /* shift register held until */

/* Bytes not yet on the wire at time g_now sit in the FIFO */
static int _in_fifo(void)
{
    int k = 0;
    for (long i = g_n - 1; i >= 0 && g_start[i] > g_now; i--) k++;
    return k;
}

static int _tx_room(void) { return _in_fifo() < FIFO_LEN; }

static void _tx_put(uint8_t b)
{
    double t = g_now;
    if (g_n > 0 && g_start[g_n - 1] + BYTE_US > t) t = g_start[g_n - 1] + BYTE_US;
    if (t < g_block_until) t = g_block_until;
    g_start[g_n] = t;
    g_val[g_n] = b;
    g_n++;
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
    ibus_gen_io_t io = { _tx_room, _tx_put };
    if (argc >= 2 && strcmp(argv[1], "frame") == 0) {
        ibus_gen_init(&g, &io, IBUS_GEN_PERIOD_TICKS);
        printf("{");
        _print_frame(g.frame);
        printf(",\"period_ticks\":%u,\"min_period_ticks\":%u}\n",
               (unsigned)g.period_ticks, (unsigned)IBUS_GEN_MIN_PERIOD_TICKS);
        return 0;
    }
    if (argc < 3 || strcmp(argv[1], "run") != 0) {
        fprintf(stderr, "usage: harness frame | harness run <ticks> [events]\n");
        return 2;
    }
    long ticks = atol(argv[2]);
    long off_at = -1, on_at = -1, blk_at = -1, blk_n = 0;
    unsigned period = IBUS_GEN_PERIOD_TICKS;
    for (int i = 3; i < argc; i++) {
        if (sscanf(argv[i], "off@%ld", &off_at) == 1) continue;
        if (sscanf(argv[i], "on@%ld", &on_at) == 1) continue;
        if (sscanf(argv[i], "block@%ld:%ld", &blk_at, &blk_n) == 2) continue;
        if (sscanf(argv[i], "period=%u", &period) == 1) continue;
        fprintf(stderr, "bad event %s\n", argv[i]);
        return 2;
    }
    ibus_gen_init(&g, &io, period);
    ibus_gen_set_enabled(&g, 1);

    printf("{\"period_ticks\":%u,\"windows\":[", (unsigned)g.period_ticks);
    uint32_t last_win = 0;
    int first = 1;
    for (long t = 1; t <= ticks; t++) {
        g_now = t * TICK_US;
        if (t == off_at) ibus_gen_set_enabled(&g, 0);
        if (t == on_at)  ibus_gen_set_enabled(&g, 1);
        if (t == blk_at) g_block_until = g_now + blk_n * TICK_US;
        ibus_gen_tick(&g);
        if (g_n >= MAX_BYTES - 64) { fprintf(stderr, "byte buffer full\n"); return 2; }
        ibus_gen_window_t w;
        if (ibus_gen_read_window(&g, &w) && w.window != last_win) {
            last_win = w.window;
            printf("%s{\"window\":%u,\"frames\":%u,\"p_count\":%u,\"p_min\":%u,\"p_max\":%u,"
                   "\"late\":%u,\"enables\":%u,\"total\":%u,\"enabled\":%u}",
                   first ? "" : ",", (unsigned)w.window, (unsigned)w.frames,
                   (unsigned)w.p_count, (unsigned)w.p_min_ticks, (unsigned)w.p_max_ticks,
                   (unsigned)w.late_ticks, (unsigned)w.enables, (unsigned)w.total_frames,
                   (unsigned)w.enabled);
            first = 0;
        }
    }
    /* Wire: byte values and start times (us) */
    printf("],\"total_frames\":%u,\"bytes\":%ld,\"t_us\":[", (unsigned)g.total_frames, g_n);
    for (long i = 0; i < g_n; i++) printf(i ? ",%.3f" : "%.3f", g_start[i]);
    printf("],\"val\":[");
    for (long i = 0; i < g_n; i++) printf(i ? ",%u" : "%u", g_val[i]);
    printf("]}\n");
    return 0;
}
