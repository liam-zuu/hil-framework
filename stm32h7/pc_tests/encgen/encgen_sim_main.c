/*
 * encgen_sim_main.c: runs the H7 encoder_gen.c unchanged on the PC against four
 * fake 16-bit timers (stm32h7xx_hal.h here), one tick = 1 timer count (1 us).
 *
 * Per tick: each running timer counts; a compare match in toggle mode flips the
 * pin and sets CCxIF; the "ISR" runs a random 0..LAT_MAX ticks later unless
 * interrupts are off (as HAL_TIM_IRQHandler: clear the flag, set Channel, call
 * EncoderGen_OC_Callback). Once per 1 ms frame, at a random phase, the "main
 * loop" calls EncoderGen_SetOmega for every wheel. A start (stopped -> running
 * or a direction change) forces both pins low, as _force_low_then_toggle does.
 *
 * The pins are decoded x4 like the ESP32 PCNT, with the per-wheel phase sign of
 * encoder_gen.c, so a positive count means the wheel turns the way omega says.
 * Output, one line per wheel:
 *   W <i> <count> <ideal> <reverse_steps> <max_wait_excess> <irq_sections>
 *   ideal: integral of omega * 4 * PPR / (2 pi) while |omega| > 0.3 rad/s (H7 stops below)
 *   reverse_steps: decoded steps against the sign of the commanded omega
 *   max_wait_excess: after a speed-up, (first edge - SetOmega time) - new half period [ticks]
 *
 *   encgen_sim <scenario> <seed>
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
static const int WSIGN[4] = { +1, -1, +1, -1 };      /* wheel command sign in the scenarios */

static uint32_t flags[4];
static long     pend_at[4][2];
static int      pin[4][2];
static int      prev_state[4];
static long     count[4], rev[4];
static double   ideal[4];
static long     wait_from[4];
static uint32_t wait_hp[4];
static long     wait_excess_max[4];

static double scenario(const char *s, double t)
{
    if (!strcmp(s, "step_up"))      return t < 0.1 ? 2.0 : 39.2;
    if (!strcmp(s, "ratio2_5"))     return t < 0.1 ? 5.0 : 12.5;
    if (!strcmp(s, "startup_slow")) return 10.0 * (1.0 - exp(-t / 0.4));
    if (!strcmp(s, "startup_fast")) return 30.0 * (1.0 - exp(-t / 0.05));
    if (!strcmp(s, "ramp_down"))    return 39.2 - 37.2 * fmin(1.0, fmax(0.0, (t - 0.1) / 0.3));
    if (!strcmp(s, "sine"))         return 10.0 + 8.0 * sin(2.0 * M_PI * t);
    if (!strcmp(s, "reverse"))      return t < 0.3 ? 10.0 : -10.0;
    if (!strcmp(s, "stop_start"))   return (t >= 0.2 && t < 0.3) ? 0.0 : 10.0;
    fprintf(stderr, "unknown scenario %s\n", s);
    exit(2);
}

static double duration(const char *s)
{
    if (!strcmp(s, "sine")) return 3.0;
    if (!strncmp(s, "startup", 7)) return 1.5;
    return 0.6;
}

/* x4 quadrature: Gray order 00 -> 10 -> 11 -> 01 is A leading B */
static int quad_index(int a, int b) { return a ? (b ? 2 : 1) : (b ? 3 : 0); }

static void edge(int w, long tick, double omega)
{
    int s = quad_index(pin[w][0], pin[w][1]);
    int d = (s - prev_state[w] + 4) % 4;
    prev_state[w] = s;
    int step = (d == 1) ? 1 : (d == 3) ? -1 : 0;   /* d == 2: lost step, counted as 0 */
    step *= SIGN[w];
    count[w] += step;
    if (step != 0 && fabs(omega) > 0.3 && (step > 0) != (omega > 0)) rev[w]++;
    if (wait_from[w] >= 0) {
        long ex = (tick - wait_from[w]) - (long)wait_hp[w];
        if (ex > wait_excess_max[w]) wait_excess_max[w] = ex;
        wait_from[w] = -1;
    }
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

int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "usage: encgen_sim <scenario> <seed>\n"); return 2; }
    const char *sc = argv[1];
    srand((unsigned)atoi(argv[2]));
    for (int w = 0; w < 4; w++) {
        H[w].Instance = &T[w];
        T[w].CNT = (uint32_t)(rand() & 0xFFFF);
        wait_from[w] = -1;
        pend_at[w][0] = pend_at[w][1] = -1;
    }
    EncoderGen_Init(&H[0], &H[1], &H[2], &H[3]);

    /* 29/10 (fix/encgen-stop-hold): outputs driven from boot and held at their
     * level after a stop (frozen mode, output still enabled) */
    if (!strcmp(sc, "regs")) {
        const uint32_t en = TIM_CCER_CC1E | TIM_CCER_CC2E, oc = TIM_CCMR1_OC1M | TIM_CCMR1_OC2M;
        const uint32_t ie = TIM_DIER_CC1IE | TIM_DIER_CC2IE;
        int ok = 1;
        for (int w = 0; w < 4; w++) {
            if ((T[w].CCER & en) != en) { printf("REGS FAIL init: wheel %d output not enabled\n", w); ok = 0; }
            if ((T[w].CCMR1 & oc) != (TIM_CCMR1_OC1M_2 | TIM_CCMR1_OC2M_2)) { printf("REGS FAIL init: wheel %d not forced low\n", w); ok = 0; }
            EncoderGen_SetOmega((uint8_t)w, 10.0f);
            EncoderGen_SetOmega((uint8_t)w, 0.0f);
            if ((T[w].CCER & en) != en) { printf("REGS FAIL stop: wheel %d output disabled (pin floats)\n", w); ok = 0; }
            if ((T[w].CCMR1 & oc) != 0u) { printf("REGS FAIL stop: wheel %d not frozen\n", w); ok = 0; }
            if (T[w].DIER & ie) { printf("REGS FAIL stop: wheel %d compare interrupts still on\n", w); ok = 0; }
        }
        if (ok) printf("REGS ok\n");
        return ok ? 0 : 1;
    }

    long n = (long)(duration(sc) * 1e6);
    long next_frame = rand() % 300;
    double omega[4] = { 0 };
    EncoderGenDebug_t dbg[4];
    for (long tick = 0; tick < n; tick++) {
        /* main loop: one frame per ms */
        if (tick == next_frame) {
            EncoderGen_GetDebug(dbg);
            for (int w = 0; w < 4; w++) {
                int8_t dir0 = dbg[w].dir;
                uint32_t hp0 = dbg[w].half_period;
                omega[w] = WSIGN[w] * scenario(sc, tick * 1e-6);
                T[w].SR = flags[w];
                EncoderGen_SetOmega((uint8_t)w, (float)omega[w]);
                flags[w] &= T[w].SR;
                EncoderGenDebug_t d1[4];
                EncoderGen_GetDebug(d1);
                if (d1[w].dir != 0 && d1[w].dir != dir0) {     /* start: pins forced low */
                    for (int c = 0; c < 2; c++) {
                        if (pin[w][c]) { pin[w][c] = 0; edge(w, tick, omega[w]); }
                    }
                }
                if (d1[w].dir != 0 && d1[w].dir == dir0 && d1[w].half_period < hp0) {
                    wait_from[w] = tick;
                    wait_hp[w] = d1[w].half_period;
                }
            }
            next_frame = (tick / 1000 + 1) * 1000 + rand() % 300;
        }
        /* timers */
        for (int w = 0; w < 4; w++) {
            if (fabs(omega[w]) > 0.3)
                ideal[w] += fabs(omega[w]) * 4.0 * ENC_PPR / (2.0 * M_PI) / ENC_TICKS_PER_S
                            * (omega[w] > 0 ? 1 : -1);
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
                    edge(w, tick, omega[w]);
                    flags[w] |= fl[c];
                    if (pend_at[w][c] < 0) pend_at[w][c] = tick + rand() % (LAT_MAX + 1);
                }
            }
            /* ISR when due and interrupts on */
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
    for (int w = 0; w < 4; w++)
        printf("W %d %ld %.1f %ld %ld %ld\n", w, count[w], ideal[w], rev[w], wait_excess_max[w], fake_irq_sections);
    return 0;
}
