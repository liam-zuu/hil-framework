/* harness.c — PC test of the H7 PWM edge decoder (stm32h7/Core/Src/pwm_edges.c,
 * the same file the H7 runs). Models one input channel:
 *   - the ESP32 MCPWM output: a free-running period grid (high from the period
 *     start, low from duty * T), duty taken at each period start (compare
 *     update on TEZ), held level for duty 0 or 1;
 *   - the timer input filter (ICF = 0xF, Codex M11, M19): the pin is sampled
 *     at fs_hz (240 MHz / 32 = 7.5 MHz); a counter of consecutive samples that
 *     differ from the filtered level, reset by a sample equal to it; the
 *     filtered level changes (capture) at the fn-th such sample (fn = 8). A
 *     pin change between two samples is not seen at all;
 *   - the capture: counter at 10 MHz (100 ns per tick) mod 65536;
 *   - the DMA: each capture lands in the ring dma_ns later;
 *   - the main loop service every svc_us, as pwm_capture.c does it:
 *     pwm_edges_windows() windows, window k from shift + k x 10 us after the
 *     service start (shift = (service mod 4) x 2.5 us while 3 windows, Codex
 *     M14), each: wr1, pin a, then pin b >= 1.2 us after the end of pin a,
 *     pin c >= 1.2 us after the end of pin b, wr2 >= 3.5 us after the end of
 *     pin a (Codex M18). Each read group of the 8 channels takes grp_ns; this
 *     channel is read at the start of each group. pre_ns adds random
 *     preemptions before read groups. Then the overcapture flag; then process
 *     up to wr1 + anchor per window; after a lap a 60 us probe that, like the
 *     firmware, compares the write index with rd (Codex M20).
 * Prints one JSON object: samples [t_ms, duty] per service and the counters.
 *
 *   harness [key=value ...]
 *     seg=D:MS      duty D for MS ms (repeatable, in order; 0 / 1 = held level)
 *     alt=D1:D2:MS  duty D1 on even, D2 on odd periods for MS ms (a segment)
 *     freq=HZ       PWM frequency (20000)
 *     phase_ns=N    period grid offset (0)
 *     fs_hz=F fn=N fphase_ns=P   input filter sampling (7.5e6, 8, 0)
 *     dma_ns=N      capture to ring (300)
 *     grp_ns=N      time of one read group of 8 channels (500)
 *     pre_ns=N      random preemption up to N ns before a read group (0 = off)
 *     win_shift=0   no window shift (test of the lock-out case)
 *     ring=N        ring entries (1024)
 *     svc_us=N      service period (1000); svc_off_ns=N first service (123457)
 *     lap=MS stale=MS timeout=MS verify=MS una=MS   decoder config (5, 5, 50, 10, 20)
 *     max_win=N     cap on windows per service (3; test of the lock-out case)
 *     stall=MS:LEN  no service in [MS, MS + LEN) ms (repeatable)
 *     lose=K        capture K (0-based) overwritten, overcapture flag set
 *     silent=K      capture K missing, no flag (not expected in hardware)
 *     glitch=US:NS  pin inverted for NS ns at US us (repeatable)
 *     jitter_ns=N   each PWM edge moved by a pseudo-random amount in [-N, N] ns
 *     dump_caps=N   also print the first N capture times (ns)
 *     dump_reads=N  also print the pin read times (ns) of the first N services
 */
#include "pwm_edges.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_EDGES  4000000
#define MAX_SEG    64
#define MAX_EV     64
#define MAX_DUMP   400
#define N_WIN      3

typedef long long ns_t;

static double g_seg_d[MAX_SEG], g_seg_d2[MAX_SEG];
static int    g_seg_alt[MAX_SEG];
static ns_t   g_seg_end[MAX_SEG];       /* end time of each segment */
static int    g_nseg;
static ns_t   g_T, g_phase, g_dma = 300;
static double g_fs = 7.5e6, g_fphase = 0.0;
static int    g_fn = 8;
static ns_t   g_glitch_t[MAX_EV], g_glitch_w[MAX_EV];
static int    g_nglitch;

static ns_t   g_raw_t[MAX_EDGES];       /* pin edges (with glitches) */
static int    g_raw_l[MAX_EDGES];
static long   g_nraw;
static int    g_level0;                 /* pin level at t = 0 */
static ns_t   g_jitter;
static unsigned long long g_rng = 88172645463325252ULL;
static ns_t   g_pre;
static unsigned long long g_rng2 = 0x9E3779B97F4A7C15ULL;

static ns_t _jit(void)
{
    if (g_jitter <= 0) return 0;
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (ns_t)(g_rng % (unsigned long long)(2 * g_jitter + 1)) - g_jitter;
}

/* Preemption before a read group: one group in four, up to g_pre ns */
static ns_t _pre(void)
{
    if (g_pre <= 0) return 0;
    g_rng2 ^= g_rng2 << 13; g_rng2 ^= g_rng2 >> 7; g_rng2 ^= g_rng2 << 17;
    if ((g_rng2 & 3u) != 0u) return 0;
    return (ns_t)((g_rng2 >> 8) % (unsigned long long)(g_pre + 1));
}

/* Pin level from the edge list (what IDR reads), after _build_raw */
static int _pin_at(ns_t t)
{
    long lo = 0, hi = g_nraw;            /* first edge with time > t */
    while (lo < hi) { long m = (lo + hi) / 2; if (g_raw_t[m] <= t) lo = m + 1; else hi = m; }
    return lo ? g_raw_l[lo - 1] : g_level0;
}

static ns_t _period_index(ns_t t)
{
    return (t - g_phase) >= 0 ? (t - g_phase) / g_T : ((t - g_phase) - g_T + 1) / g_T;
}

/* Duty of the period that starts at s */
static double _duty_at(ns_t s)
{
    for (int i = 0; i < g_nseg; i++)
        if (s < g_seg_end[i])
            return (g_seg_alt[i] && (_period_index(s) & 1)) ? g_seg_d2[i] : g_seg_d[i];
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
    ns_t s = g_phase + _period_index(t) * g_T;
    double d = _duty_at(s);
    if (d >= 1.0) return 1;
    if (d <= 0.0) return 0;
    return (t - s) < (ns_t)(d * (double)g_T + 0.5);   /* same rounding as _build_raw */
}

/* Pin level including glitches */
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

/* Input filter sample by sample (M19): a sample that differs from the
 * filtered level counts, a sample equal to it resets the count; the fn-th
 * count changes the filtered level. Stretches where the sampled pin equals
 * the filtered level are skipped up to the next raw edge. */
static void _filter(void)
{
    const double ts = 1e9 / g_fs;
    int fl = g_level0, lvl = g_level0, cnt = 0;
    long idx = 0;                                   /* raw edges at or before the sample */
    long long n = (long long)ceil((0.0 - g_fphase) / ts);
    for (;;) {
        double tn = g_fphase + (double)n * ts;
        while (idx < g_nraw && (double)g_raw_t[idx] <= tn) { lvl = g_raw_l[idx]; idx++; }
        if (lvl == fl) {
            cnt = 0;
            if (idx >= g_nraw) break;
            long long nn = (long long)ceil(((double)g_raw_t[idx] - g_fphase) / ts);
            n = (nn > n) ? nn : n + 1;
            continue;
        }
        if (++cnt >= g_fn) {
            if (g_ncap >= MAX_EDGES) { fprintf(stderr, "too many captures\n"); exit(3); }
            g_cap_t[g_ncap++] = (ns_t)tn;
            fl = lvl;
            cnt = 0;
        }
        n++;
    }
}

int main(int argc, char **argv)
{
    double freq = 20000.0;
    long ring_len = 1024, svc_us = 1000;
    ns_t svc_off = 123457, grp = 500;
    PwmEdgesCfg_t cfg = { 0, 5, 5, 50, 750, 10, 20 };
    ns_t stall_a[MAX_EV], stall_b[MAX_EV]; int nstall = 0;
    long lose[MAX_EV]; int nlose = 0;
    long silent[MAX_EV]; int nsilent = 0;
    ns_t t_end = 0;
    uint32_t max_win = N_WIN;
    int win_shift = 1, dump_caps = 0, dump_reads = 0;

    for (int i = 1; i < argc; i++) {
        char *a = argv[i];
        if (!strncmp(a, "seg=", 4)) {
            double d, ms;
            if (sscanf(a + 4, "%lf:%lf", &d, &ms) != 2 || g_nseg >= MAX_SEG) return 2;
            t_end += (ns_t)(ms * 1e6);
            g_seg_d[g_nseg] = d; g_seg_alt[g_nseg] = 0; g_seg_end[g_nseg] = t_end; g_nseg++;
        }
        else if (!strncmp(a, "alt=", 4)) {
            double d1, d2, ms;
            if (sscanf(a + 4, "%lf:%lf:%lf", &d1, &d2, &ms) != 3 || g_nseg >= MAX_SEG) return 2;
            t_end += (ns_t)(ms * 1e6);
            g_seg_d[g_nseg] = d1; g_seg_d2[g_nseg] = d2; g_seg_alt[g_nseg] = 1;
            g_seg_end[g_nseg] = t_end; g_nseg++;
        }
        else if (!strncmp(a, "freq=", 5))       freq = atof(a + 5);
        else if (!strncmp(a, "phase_ns=", 9))   g_phase = atoll(a + 9);
        else if (!strncmp(a, "fs_hz=", 6))      g_fs = atof(a + 6);
        else if (!strncmp(a, "fn=", 3))         g_fn = atoi(a + 3);
        else if (!strncmp(a, "fphase_ns=", 10)) g_fphase = atof(a + 10);
        else if (!strncmp(a, "dma_ns=", 7))     g_dma = atoll(a + 7);
        else if (!strncmp(a, "grp_ns=", 7))     grp = atoll(a + 7);
        else if (!strncmp(a, "pre_ns=", 7))     g_pre = atoll(a + 7);
        else if (!strncmp(a, "win_shift=", 10)) win_shift = atoi(a + 10);
        else if (!strncmp(a, "ring=", 5))       ring_len = atol(a + 5);
        else if (!strncmp(a, "svc_us=", 7))     svc_us = atol(a + 7);
        else if (!strncmp(a, "svc_off_ns=", 11)) svc_off = atoll(a + 11);
        else if (!strncmp(a, "lap=", 4))        cfg.lap_ms = (uint32_t)atol(a + 4);
        else if (!strncmp(a, "stale=", 6))      cfg.stale_ms = (uint32_t)atol(a + 6);
        else if (!strncmp(a, "timeout=", 8))    cfg.timeout_ms = (uint32_t)atol(a + 8);
        else if (!strncmp(a, "max_period=", 11)) cfg.max_period = (uint32_t)atol(a + 11);
        else if (!strncmp(a, "verify=", 7))     cfg.verify_ms = (uint32_t)atol(a + 7);
        else if (!strncmp(a, "una=", 4))        cfg.unanchored_ms = (uint32_t)atol(a + 4);
        else if (!strncmp(a, "max_win=", 8))    max_win = (uint32_t)atol(a + 8);
        else if (!strncmp(a, "dump_caps=", 10)) dump_caps = atoi(a + 10);
        else if (!strncmp(a, "dump_reads=", 11)) dump_reads = atoi(a + 11);
        else if (!strncmp(a, "stall=", 6)) {
            double ms, len;
            if (sscanf(a + 6, "%lf:%lf", &ms, &len) != 2 || nstall >= MAX_EV) return 2;
            stall_a[nstall] = (ns_t)(ms * 1e6); stall_b[nstall] = (ns_t)((ms + len) * 1e6); nstall++;
        }
        else if (!strncmp(a, "lose=", 5) && nlose < MAX_EV)       lose[nlose++] = atol(a + 5);
        else if (!strncmp(a, "silent=", 7) && nsilent < MAX_EV)   silent[nsilent++] = atol(a + 7);
        else if (!strncmp(a, "jitter_ns=", 10)) g_jitter = atoll(a + 10);
        else if (!strncmp(a, "glitch=", 7)) {
            double us, w;
            if (sscanf(a + 7, "%lf:%lf", &us, &w) != 2 || g_nglitch >= MAX_EV) return 2;
            g_glitch_t[g_nglitch] = (ns_t)(us * 1e3); g_glitch_w[g_nglitch] = (ns_t)w; g_nglitch++;
        }
        else { fprintf(stderr, "bad arg %s\n", a); return 2; }
    }
    if (g_nseg == 0) { fprintf(stderr, "no seg\n"); return 2; }
    if (max_win > N_WIN) max_win = N_WIN;
    if (dump_reads > MAX_DUMP) dump_reads = MAX_DUMP;
    g_T = (ns_t)(1e9 / freq + 0.5);
    cfg.ring_len = (uint32_t)ring_len;

    _build_raw(t_end);
    _filter();

    uint32_t *ring = calloc((size_t)ring_len, sizeof(uint32_t));
    PwmEdges_t e;
    pwm_edges_init(&e, &cfg, 0);            /* capture started at t = 0 */

    long ci = 0;            /* next capture to deliver */
    long written = 0;       /* entries written into the ring */
    int  ovc = 0;           /* overcapture flag (hardware, sticky until read) */
    uint32_t svc = 0;
    unsigned long n_probe = 0;
    static ns_t reads[MAX_DUMP][1 + 3 * N_WIN];
    static int  nreads[MAX_DUMP];
    int ndump = 0;

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
        uint32_t nwin = pwm_edges_windows(&e, now_ms);
        if (nwin > max_win) nwin = max_win;
        uint32_t w1[N_WIN], w2[N_WIN];
        int pa[N_WIN], pb[N_WIN], pc[N_WIN];
        svc++;
        ns_t shift = (win_shift && nwin == N_WIN) ? (ns_t)(svc & 3u) * 2500 : 0;
        ns_t cur = t;
        if (ndump < dump_reads) { reads[ndump][0] = t; nreads[ndump] = (int)nwin; }
        for (uint32_t k = 0; k < nwin; k++) {
            ns_t ts0 = t + shift + (ns_t)k * 10000;
            if (cur < ts0) cur = ts0;
            cur += _pre();
            DELIVER(cur); w1[k] = (uint32_t)(written % ring_len); cur += grp;
            cur += _pre();
            ns_t tpa = cur; pa[k] = _pin_at(cur); cur += grp;
            ns_t ta = cur;                              /* end of the pin a group */
            if (cur < ta + 1200) cur = ta + 1200;
            cur += _pre();
            ns_t tpb = cur; pb[k] = _pin_at(cur); cur += grp;
            ns_t tb = cur;
            if (cur < tb + 1200) cur = tb + 1200;
            cur += _pre();
            ns_t tpc = cur; pc[k] = _pin_at(cur); cur += grp;
            if (cur < ta + 3500) cur = ta + 3500;
            cur += _pre();
            DELIVER(cur); w2[k] = (uint32_t)(written % ring_len); cur += grp;
            if (ndump < dump_reads) {
                reads[ndump][1 + 3 * k] = tpa; reads[ndump][2 + 3 * k] = tpb; reads[ndump][3 + 3 * k] = tpc;
            }
        }
        if (ndump < dump_reads) ndump++;
        ns_t tsr = cur;
        DELIVER(tsr);
        if (ovc) {
            ovc = 0;
            pwm_edges_process(&e, ring, (uint32_t)(written % ring_len), 1, now_ms);
        } else {
            for (uint32_t k = 0; k < nwin; k++) {
                pwm_edges_process(&e, ring, w1[k], 0, now_ms);
                pwm_edges_anchor(&e, svc, now_ms, w1[k], w2[k], pa[k], pb[k], pc[k]);
            }
        }
        int pin_d = pc[nwin - 1];
        if (e.lap_pending) {
            /* firmware probe: the write index moves away from rd within 60 us */
            n_probe++;
            DELIVER(tsr + 60000);
            int seen = (uint32_t)(written % ring_len) != e.rd;
            pwm_edges_probe_done(&e, now_ms, seen);
            pin_d = _pin_at(tsr + 60000);
        }
        float d = pwm_edges_duty(&e, now_ms, pin_d);
        printf("%s[%.4f,%.5f]", first ? "" : ",", (double)t / 1e6, (double)d);
        first = 0;
    }
    printf("], \"captures\": %ld, \"raw_edges\": %ld, \"n_edges\": %lu, \"n_periods\": %lu, "
           "\"n_lost\": %lu, \"n_lap\": %lu, \"n_parity_fix\": %lu, \"n_stale\": %lu, "
           "\"n_bad_period\": %lu, \"n_cand_retry\": %lu, \"n_unanchored\": %lu, \"n_probe\": %lu, "
           "\"known\": %u",
           g_ncap, g_nraw, (unsigned long)e.n_edges, (unsigned long)e.n_periods,
           (unsigned long)e.n_lost, (unsigned long)e.n_lap, (unsigned long)e.n_parity_fix,
           (unsigned long)e.n_stale, (unsigned long)e.n_bad_period, (unsigned long)e.n_cand_retry,
           (unsigned long)e.n_unanchored, n_probe, (unsigned)e.known);
    if (dump_caps > 0) {
        printf(", \"caps\": [");
        for (long i = 0; i < g_ncap && i < dump_caps; i++) printf("%s%lld", i ? "," : "", g_cap_t[i]);
        printf("]");
    }
    if (dump_reads > 0) {
        printf(", \"reads\": [");
        for (int s = 0; s < ndump; s++) {
            printf("%s[%lld", s ? "," : "", reads[s][0]);
            for (int j = 0; j < 3 * nreads[s]; j++) printf(",%lld", reads[s][1 + j]);
            printf("]");
        }
        printf("]");
    }
    printf("}\n");
    free(ring);
    return 0;
}
