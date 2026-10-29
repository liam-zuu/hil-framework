/*
 * encgen_t3b_main.c: S06 T3b sensitivity harness (spec S06 v3.4 T3b, Z23).
 *
 * Runs the H7 encoder_gen.c unchanged on the PC (fake 16-bit timers of
 * stm32h7xx_hal.h here, random ISR latency 0..LAT_MAX ticks as in
 * encgen_sim_main.c) on ONE real-time base shared by the frames and the
 * reads (Z23: before, the omega command used tick * 1e-6 while the timer
 * ticked at ENC_TICKS_PER_S):
 *   - one timer tick lasts 1 / ENC_TICKS_PER_S real seconds, i.e. the fake
 *     timer runs exactly as fast as encoder_gen.c assumes the real one does;
 *   - frame k is applied (EncoderGen_SetOmega for the 4 wheels) at its
 *     t_apply_ns from the input file (the Pi send time t_send_ns is only
 *     carried for the scorer);
 *   - the "ESP32" reads the 4 x4-decoded counts every read_period_ns at
 *     read_phase_ns + n * read_period_ns (= t_before); the counts are sampled
 *     at a random instant in [t_before, t_before + read_span_ns] (= t_after).
 *
 * Input (text): first line "<n_frames>", then per frame
 *   <t_send_ns> <t_apply_ns> <w0> <w1> <w2> <w3>      (omega as framed, rad/s)
 * Output (stdout):
 *   R <t_before_ns> <t_after_ns> <c0..c3> <rv0..rv3>   one read
 *       c: decoded count (positive = the way omega says, as the ESP32 with
 *          the wheel signs of encoder_gen.c); rv: cumulative decoded steps
 *          against the applied omega (|omega| > 0.3), the per-edge truth,
 *          without the edges of pins forced low at a start
 *   C <wheel> <S|P> <t_ns> <cnt>                       _start_wheel / _stop
 *       call, inferred from the dir change seen by EncoderGen_GetDebug
 *       (SetOmega: 0 -> x start, x -> 0 stop, sign flip stop + start);
 *       cnt = timer CNT just before SetOmega (phase of _start_wheel, Y5)
 *
 *   encgen_t3b <frames.txt> <seed> <read_phase_ns> <read_span_ns> [read_period_ns]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "encoder_gen.h"

int  fake_irq_disabled = 0;
long fake_irq_sections = 0;

#define LAT_MAX 6
static TIM_TypeDef       T[4];
static TIM_HandleTypeDef H[4];
static const int SIGN[4] = { +1, -1, +1, -1 };       /* encoder_gen.c ENC_DIR_SIGN */

static uint32_t flags[4];
static long     pend_at[4][2];
static int      pin[4][2];
static int      prev_state[4];
static long     count[4], rev[4];
static double   omega[4];

/* x4 quadrature: Gray order 00 -> 10 -> 11 -> 01 is A leading B */
static int quad_index(int a, int b) { return a ? (b ? 2 : 1) : (b ? 3 : 0); }

/* forced: the pins forced low by a start (declared behaviour, not counted in rv) */
static void edge(int w, int forced)
{
    int s = quad_index(pin[w][0], pin[w][1]);
    int d = (s - prev_state[w] + 4) % 4;
    prev_state[w] = s;
    int step = (d == 1) ? 1 : (d == 3) ? -1 : 0;   /* d == 2: lost step, counted as 0 */
    step *= SIGN[w];
    count[w] += step;
    if (!forced && step != 0 && fabs(omega[w]) > 0.3 && (step > 0) != (omega[w] > 0)) rev[w]++;
}

static void run_isr(int w)
{
    T[w].SR = flags[w];
    uint32_t act = flags[w] & T[w].DIER;
    if (act & TIM_SR_CC1IF) {
        flags[w] &= ~TIM_SR_CC1IF; T[w].SR = flags[w];
        H[w].Channel = HAL_TIM_ACTIVE_CHANNEL_1;
        EncoderGen_OC_Callback(&H[w]);
        flags[w] &= T[w].SR;
    }
    if (act & TIM_SR_CC2IF) {
        flags[w] &= ~TIM_SR_CC2IF; T[w].SR = flags[w];
        H[w].Channel = HAL_TIM_ACTIVE_CHANNEL_2;
        EncoderGen_OC_Callback(&H[w]);
        flags[w] &= T[w].SR;
    }
}

static double urand(void) { return (double)rand() / ((double)RAND_MAX + 1.0); }

int main(int argc, char **argv)
{
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "usage: encgen_t3b <frames.txt> <seed> <read_phase_ns> <read_span_ns> [read_period_ns]\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 2; }
    long nf = 0;
    if (fscanf(f, "%ld", &nf) != 1 || nf <= 0) { fprintf(stderr, "bad header\n"); return 2; }
    double *t_apply = malloc(sizeof(double) * nf);
    float (*wf)[4] = malloc(sizeof(float[4]) * nf);
    for (long k = 0; k < nf; k++) {
        double ts;
        if (fscanf(f, "%lf %lf %f %f %f %f", &ts, &t_apply[k], &wf[k][0], &wf[k][1], &wf[k][2], &wf[k][3]) != 6) {
            fprintf(stderr, "bad frame %ld\n", k);
            return 2;
        }
        if (k > 0 && t_apply[k] < t_apply[k - 1]) { fprintf(stderr, "frames not in apply order at %ld\n", k); return 2; }
    }
    fclose(f);
    srand((unsigned)atoi(argv[2]));
    const double read_phase = atof(argv[3]), read_span = atof(argv[4]);
    const double read_period = argc == 6 ? atof(argv[5]) : 1e6;
    const double tick_ns = 1e9 / (double)ENC_TICKS_PER_S;
    const double t_end = t_apply[nf - 1] + 2.0 * read_period;

    for (int w = 0; w < 4; w++) {
        H[w].Instance = &T[w];
        T[w].CNT = (uint32_t)(rand() & 0xFFFF);
        pend_at[w][0] = pend_at[w][1] = -1;
    }
    EncoderGen_Init(&H[0], &H[1], &H[2], &H[3]);

    long k = 0;                                   /* next frame to apply */
    double rd_before = read_phase;                /* next read */
    double rd_sample = rd_before + urand() * read_span;
    EncoderGenDebug_t dbg[4], d1[4];
    for (long tick = 0;; tick++) {
        double t = tick * tick_ns;
        if (t > t_end) break;
        /* reads due at or before this tick: the counts as they are now */
        while (rd_sample <= t) {
            printf("R %.0f %.0f %ld %ld %ld %ld %ld %ld %ld %ld\n", rd_before, rd_before + read_span,
                   count[0], count[1], count[2], count[3], rev[0], rev[1], rev[2], rev[3]);
            rd_before += read_period;
            rd_sample = rd_before + urand() * read_span;
        }
        /* main loop: frames due */
        while (k < nf && t_apply[k] <= t) {
            EncoderGen_GetDebug(dbg);
            for (int w = 0; w < 4; w++) {
                int8_t dir0 = dbg[w].dir;
                uint32_t cnt0 = T[w].CNT;
                omega[w] = wf[k][w];
                T[w].SR = flags[w];
                EncoderGen_SetOmega((uint8_t)w, wf[k][w]);
                flags[w] &= T[w].SR;
                EncoderGen_GetDebug(d1);
                if (dir0 != 0 && d1[w].dir != dir0)
                    printf("C %d P %.0f %u\n", w, t, (unsigned)cnt0);
                if (d1[w].dir != 0 && d1[w].dir != dir0) {
                    printf("C %d S %.0f %u\n", w, t, (unsigned)cnt0);
                    for (int c = 0; c < 2; c++)          /* start: pins forced low */
                        if (pin[w][c]) { pin[w][c] = 0; edge(w, 1); }
                }
            }
            k++;
        }
        /* timers */
        for (int w = 0; w < 4; w++) {
            if (!(T[w].CR1 & TIM_CR1_CEN)) continue;
            T[w].CNT = (T[w].CNT + 1) & 0xFFFFu;
            uint32_t ccr[2] = { T[w].CCR1, T[w].CCR2 };
            uint32_t en[2]  = { TIM_CCER_CC1E, TIM_CCER_CC2E };
            uint32_t tog[2] = { TIM_CCMR1_OC1M_1 | TIM_CCMR1_OC1M_0, TIM_CCMR1_OC2M_1 | TIM_CCMR1_OC2M_0 };
            uint32_t msk[2] = { TIM_CCMR1_OC1M, TIM_CCMR1_OC2M };
            uint32_t fl[2]  = { TIM_SR_CC1IF, TIM_SR_CC2IF };
            for (int c = 0; c < 2; c++) {
                if (T[w].CNT == ccr[c] && (T[w].CCER & en[c]) && (T[w].CCMR1 & msk[c]) == tog[c]) {
                    pin[w][c] ^= 1;
                    edge(w, 0);
                    flags[w] |= fl[c];
                    if (pend_at[w][c] < 0) pend_at[w][c] = tick + rand() % (LAT_MAX + 1);
                }
            }
            int due = 0;
            for (int c = 0; c < 2; c++)
                if (pend_at[w][c] >= 0 && tick >= pend_at[w][c]) due = 1;
            if (due && !fake_irq_disabled) {
                run_isr(w);
                for (int c = 0; c < 2; c++)
                    if (!(flags[w] & fl[c])) pend_at[w][c] = -1;
            }
            for (int c = 0; c < 2; c++)          /* flag dropped by SetOmega (retime) */
                if (!(flags[w] & fl[c])) pend_at[w][c] = -1;
        }
    }
    free(t_apply);
    free(wf);
    return 0;
}
