/* harness.c — PC test of the H7 PWM edge decoder (stm32h7/Core/Src/pwm_edges.c,
 * the same file the H7 runs). Models one input channel:
 *   - the ESP32 MCPWM output: a free-running period grid (high from the period
 *     start, low from duty * T), duty taken at each period start (compare
 *     update on TEZ), held level for duty 0 or 1;
 *   - the timer input filter: the filtered level follows the pin after the pin
 *     stays at the new level for filt_ns (shorter pulses are dropped as pairs);
 *   - the capture: counter at 10 MHz (100 ns per tick) mod 65536;
 *   - the DMA: each capture lands in the ring dma_ns later;
 *   - the main loop service every svc_us, as pwm_capture.c does it:
 *     wr1, pin1, (3 us), pin2, wr2, lost flag, process, anchor, duty.
 * Prints one JSON object: samples [t_ms, duty] per service and the counters.
 *
 *   harness [key=value ...]
 *     seg=D:MS      duty D for MS ms (repeatable, in order; 0 / 1 = held level)
 *     freq=HZ       PWM frequency (20000)
 *     phase_ns=N    period grid offset (0)
 *     filt_ns=N     input filter delay (1067: 8 samples at 240 MHz / 32)
 *     dma_ns=N      capture to ring (300)
 *     ring=N        ring entries (1024)
 *     svc_us=N      service period (1000); svc_off_ns=N first service (123457)
 *     lap=MS stale=MS timeout=MS   decoder config (5, 5, 50)
 *     stall=MS:LEN  no service in [MS, MS + LEN) ms (repeatable)
 *     lose=K        capture K (0-based) overwritten, overcapture flag set
 *     silent=K      capture K missing, no flag (not expected in hardware)
 *     glitch=US:NS  pin inverted for NS ns at US us (repeatable)
 *     jitter_ns=N   each PWM edge moved by a pseudo-random amount in [-N, N] ns
 *     n_try=N try_ns=N   anchor windows per service while unknown (3, 7000)
 */
#include "pwm_edges.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_EDGES  4000000
#define MAX_SEG    64
#define MAX_EV     64

typedef long long ns_t;

static double g_seg_d[MAX_SEG];
static ns_t   g_seg_end[MAX_SEG];       /* end time of each segment */
static int    g_nseg;
static ns_t   g_T, g_phase, g_filt = 1067, g_dma = 300;
static ns_t   g_glitch_t[MAX_EV], g_glitch_w[MAX_EV];
static int    g_nglitch;

static ns_t   g_raw_t[MAX_EDGES];       /* pin edges (before glitches) */
static int    g_raw_l[MAX_EDGES];
static long   g_nraw;
static int    g_level0;                 /* pin level at t = 0 */
static ns_t   g_jitter;
static unsigned long long g_rng = 88172645463325252ULL;

static ns_t _jit(void)
{
    if (g_jitter <= 0) return 0;
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (ns_t)(g_rng % (unsigned long long)(2 * g_jitter + 1)) - g_jitter;
}

/* Pin level from the edge list (what IDR reads), after _build_raw */
static int _pin_at(ns_t t)
{
    long lo = 0, hi = g_nraw;            /* first edge with time > t */
    while (lo < hi) { long m = (lo + hi) / 2; if (g_raw_t[m] <= t) lo = m + 1; else hi = m; }
    return lo ? g_raw_l[lo - 1] : g_level0;
}

static double _duty_at(ns_t t)
{
    for (int i = 0; i < g_nseg; i++)
        if (t < g_seg_end[i]) return g_seg_d[i];
    return g_seg_d[g_nseg - 1];
}

static void _add_raw(ns_t t, int l)
{
    int prev = g_nraw ? g_raw_l[g_nraw - 1] : g_level0;
    if (l == prev) return;
    if (g_nraw >= MAX_EDGES) { fprintf(stderr, "too many edges\n"); exit(3); }
    g_raw_t[g_nraw] = t;
    g_raw_l[g_nraw] = l;
    g_nraw++;
}

/* Pin level from the PWM grid only */
static int _pwm_level(ns_t t)
{
    ns_t k = (t - g_phase) >= 0 ? (t - g_phase) / g_T : ((t - g_phase) - g_T + 1) / g_T;
    ns_t s = g_phase + k * g_T;
    double d = _duty_at(s);
    if (d >= 1.0) return 1;
    if (d <= 0.0) return 0;
    return (t - s) < (ns_t)(d * (double)g_T + 0.5);   /* same rounding as _build_raw */
}

/* Pin level including glitches (what IDR reads) */
static int _pin(ns_t t)
{
    int l = _pwm_level(t);
    for (int i = 0; i < g_nglitch; i++)
        if (t >= g_glitch_t[i] && t < g_glitch_t[i] + g_glitch_w[i]) l ^= 1;
    return l;
}

/* All pin edges (PWM grid + glitches) up to t_end, time ordered */
static void _build_raw(ns_t t_end)
{
    g_level0 = _pin(0);
    ns_t k0 = -((g_phase / g_T) + 1);
    /* candidate times: period starts, duty ends, glitch borders; merge by
     * walking the grid and inserting glitch borders */
    ns_t *cand = malloc(sizeof(ns_t) * (size_t)(2 * (t_end / g_T + 4) + 2 * MAX_EV));
    long nc = 0;
    for (ns_t k = k0; ; k++) {
        ns_t s = g_phase + k * g_T;
        if (s > t_end) break;
        double d = _duty_at(s);
        if (s > 0) cand[nc++] = s;
        if (d > 0.0 && d < 1.0) {
            ns_t f = s + (ns_t)(d * (double)g_T + 0.5);
            if (f > 0 && f <= t_end) cand[nc++] = f;
        }
    }
    for (int i = 0; i < g_nglitch; i++) {
        cand[nc++] = g_glitch_t[i];
        cand[nc++] = g_glitch_t[i] + g_glitch_w[i];
    }
    /* sort (mostly sorted: insertion from the tail) */
    for (long i = 1; i < nc; i++) {
        ns_t v = cand[i]; long j = i - 1;
        while (j >= 0 && cand[j] > v) { cand[j + 1] = cand[j]; j--; }
        cand[j + 1] = v;
    }
    for (long i = 0; i < nc; i++) {
        int is_glitch = 0;
        for (int q = 0; q < g_nglitch; q++)
            if (cand[i] == g_glitch_t[q] || cand[i] == g_glitch_t[q] + g_glitch_w[q]) is_glitch = 1;
        long before = g_nraw;
        _add_raw(cand[i], _pin(cand[i]));
        if (g_nraw > before && !is_glitch) g_raw_t[g_nraw - 1] += _jit();
    }
    free(cand);
}

/* Captures after the input filter */
static ns_t  g_cap_t[MAX_EDGES];        /* capture time */
static long  g_ncap;

static void _filter(void)
{
    int fl = g_level0;
    for (long i = 0; i < g_nraw; i++) {
        int l = g_raw_l[i];
        ns_t next = (i + 1 < g_nraw) ? g_raw_t[i + 1] : (ns_t)1 << 62;
        if (l != fl && next - g_raw_t[i] >= g_filt) {
            g_cap_t[g_ncap++] = g_raw_t[i] + g_filt;
            fl = l;
        }
    }
}

int main(int argc, char **argv)
{
    double freq = 20000.0;
    long ring_len = 1024, svc_us = 1000;
    ns_t svc_off = 123457;
    PwmEdgesCfg_t cfg = { 0, 5, 5, 50 };
    ns_t stall_a[MAX_EV], stall_b[MAX_EV]; int nstall = 0;
    long lose[MAX_EV]; int nlose = 0;
    long silent[MAX_EV]; int nsilent = 0;
    ns_t t_end = 0;
    int  n_try = 3;
    ns_t try_ns = 7000;

    for (int i = 1; i < argc; i++) {
        char *a = argv[i];
        if (!strncmp(a, "seg=", 4)) {
            double d, ms;
            if (sscanf(a + 4, "%lf:%lf", &d, &ms) != 2 || g_nseg >= MAX_SEG) return 2;
            t_end += (ns_t)(ms * 1e6);
            g_seg_d[g_nseg] = d; g_seg_end[g_nseg] = t_end; g_nseg++;
        }
        else if (!strncmp(a, "freq=", 5))       freq = atof(a + 5);
        else if (!strncmp(a, "phase_ns=", 9))   g_phase = atoll(a + 9);
        else if (!strncmp(a, "filt_ns=", 8))    g_filt = atoll(a + 8);
        else if (!strncmp(a, "dma_ns=", 7))     g_dma = atoll(a + 7);
        else if (!strncmp(a, "ring=", 5))       ring_len = atol(a + 5);
        else if (!strncmp(a, "svc_us=", 7))     svc_us = atol(a + 7);
        else if (!strncmp(a, "svc_off_ns=", 11)) svc_off = atoll(a + 11);
        else if (!strncmp(a, "lap=", 4))        cfg.lap_ms = (uint32_t)atol(a + 4);
        else if (!strncmp(a, "stale=", 6))      cfg.stale_ms = (uint32_t)atol(a + 6);
        else if (!strncmp(a, "timeout=", 8))    cfg.timeout_ms = (uint32_t)atol(a + 8);
        else if (!strncmp(a, "stall=", 6)) {
            double ms, len;
            if (sscanf(a + 6, "%lf:%lf", &ms, &len) != 2 || nstall >= MAX_EV) return 2;
            stall_a[nstall] = (ns_t)(ms * 1e6); stall_b[nstall] = (ns_t)((ms + len) * 1e6); nstall++;
        }
        else if (!strncmp(a, "lose=", 5) && nlose < MAX_EV)       lose[nlose++] = atol(a + 5);
        else if (!strncmp(a, "silent=", 7) && nsilent < MAX_EV)   silent[nsilent++] = atol(a + 7);
        else if (!strncmp(a, "jitter_ns=", 10)) g_jitter = atoll(a + 10);
        else if (!strncmp(a, "n_try=", 6))      n_try = atoi(a + 6);
        else if (!strncmp(a, "try_ns=", 7))     try_ns = atoll(a + 7);
        else if (!strncmp(a, "glitch=", 7)) {
            double us, w;
            if (sscanf(a + 7, "%lf:%lf", &us, &w) != 2 || g_nglitch >= MAX_EV) return 2;
            g_glitch_t[g_nglitch] = (ns_t)(us * 1e3); g_glitch_w[g_nglitch] = (ns_t)w; g_nglitch++;
        }
        else { fprintf(stderr, "bad arg %s\n", a); return 2; }
    }
    if (g_nseg == 0) { fprintf(stderr, "no seg\n"); return 2; }
    g_T = (ns_t)(1e9 / freq + 0.5);
    cfg.ring_len = (uint32_t)ring_len;

    _build_raw(t_end);
    _filter();

    uint32_t *ring = calloc((size_t)ring_len, sizeof(uint32_t));
    PwmEdges_t e;
    pwm_edges_init(&e, &cfg);

    long ci = 0;            /* next capture to deliver */
    long written = 0;       /* entries written into the ring */
    int  ovc = 0;           /* overcapture flag (hardware, sticky until read) */

    printf("{\"samples\": [");
    int first = 1;
    for (ns_t t = svc_off; t < t_end; t += (ns_t)svc_us * 1000) {
        int stalled = 0;
        for (int s = 0; s < nstall; s++) if (t >= stall_a[s] && t < stall_b[s]) stalled = 1;
        if (stalled) continue;

        /* deliver the ring up to time tt */
        #define DELIVER(tt) \
            while (ci < g_ncap && g_cap_t[ci] + g_dma <= (tt)) { \
                int skip = 0; \
                for (int q = 0; q < nlose; q++)   if (lose[q] == ci)   { skip = 1; ovc = 1; } \
                for (int q = 0; q < nsilent; q++) if (silent[q] == ci) { skip = 1; } \
                if (!skip) { ring[written % ring_len] = (uint32_t)((g_cap_t[ci] / 100) & 0xFFFF); written++; } \
                ci++; \
            }

        uint32_t now_ms = (uint32_t)(t / 1000000);
        int pin2 = 0;
        /* Up to n_try anchor windows, try_ns apart, while the edge type is
         * unknown (one window when known), as pwm_capture.c does. */
        for (int k = 0; k < n_try; k++) {
            ns_t ta = t + (ns_t)k * try_ns;
            DELIVER(ta);
            uint32_t wr1 = (uint32_t)(written % ring_len);
            int pin1 = _pin_at(ta + 50);
            ns_t t2 = ta + 50 + 3000;
            pin2 = _pin_at(t2);
            DELIVER(t2 + 50);
            uint32_t wr2 = (uint32_t)(written % ring_len);
            int lost = ovc; ovc = 0;
            pwm_edges_process(&e, ring, wr2, lost, now_ms);
            pwm_edges_anchor(&e, wr1, wr2, pin1, pin2);
            if (e.known) break;
        }
        float d = pwm_edges_duty(&e, now_ms, pin2);
        printf("%s[%.4f,%.5f]", first ? "" : ",", (double)t / 1e6, (double)d);
        first = 0;
    }
    printf("], \"captures\": %ld, \"raw_edges\": %ld, \"n_edges\": %lu, \"n_periods\": %lu, "
           "\"n_lost\": %lu, \"n_lap\": %lu, \"n_parity_fix\": %lu, \"n_stale\": %lu}\n",
           g_ncap, g_nraw, (unsigned long)e.n_edges, (unsigned long)e.n_periods,
           (unsigned long)e.n_lost, (unsigned long)e.n_lap, (unsigned long)e.n_parity_fix,
           (unsigned long)e.n_stale);
    free(ring);
    return 0;
}
