#ifndef ENCODER_GEN_H
#define ENCODER_GEN_H

#include "stm32h7xx_hal.h"

/*
 * EncoderGen — Quadrature encoder pulse generation via TIM Output Compare Toggle
 *
 * Hardware (CubeMX-generated, OC Toggle CH1/CH2):
 *   TIM1 → Wheel 1 (CH1=PE9,  CH2=PE11)
 *   TIM2 → Wheel 2 (CH1=PA15, CH2=PB3)   (PA0/PA1 are PWMCapture inputs, tim.c)
 *   TIM3 → Wheel 3 (CH1=PC6,  CH2=PC7)
 *   TIM4 → Wheel 4 (CH1=PD12, CH2=PD13)
 *
 * Timer clock: HCLK=480MHz, AHB=240MHz, APB1=120MHz.
 * APB prescaler > 1 → timer clock = 2 × APB = 240MHz.
 * PSC = 239 → tick = 240MHz / 240 = 1MHz = ENC_TIMER_FREQ.
 *
 * ── Frequency derivation ──────────────────────────────────────────────────
 * Channel A produces ENC_PPR rising edges per revolution.
 * Pulse frequency on A:
 *   f_pulse = |omega| / (2*pi) * ENC_PPR        [Hz]
 *
 * OC Toggle: 1 toggle event = half a period (one edge, either rising or
 * falling). 2 toggles = 1 full pulse period on that pin.
 *   half_period_ticks = ENC_TIMER_FREQ / (2 * f_pulse)
 *                     = ENC_TIMER_FREQ * pi / (|omega| * ENC_PPR)
 *
 * ── Quadrature phase ─────────────────────────────────────────────────────
 * Forward (omega > 0): A leads B by 90 deg  → B is delayed by half_period/2
 * Reverse (omega < 0): B leads A by 90 deg  → A is delayed by half_period/2
 *
 * Each channel toggles independently every half_period_ticks once started;
 * the initial 90 deg offset between A and B encodes direction and is
 * preserved as both channels advance by the same half_period_ticks on every
 * callback; when the speed changes, EncoderGen_SetOmega re-times both
 * channels to the new half period (before 28/10 it did not, bug H7-ENC-RETIME).
 * At every start both pins are forced low first, so the first
 * edge of each is rising and the lead/lag does not depend on the levels
 * left by the previous run.
 *
 * ── Debug / verification ─────────────────────────────────────────────────
 * For each wheel, the time between two consecutive CH1 (channel A)
 * callbacks is measured in timer ticks. One callback = one toggle, so
 *
 *   ch1_period_ticks ≈ half_period_ticks
 *   measured_freq_hz = ENC_TIMER_FREQ / (2 * ch1_period_ticks)
 *   expected_freq_hz = |omega| / (2*pi) * ENC_PPR
 *
 * measured_freq_hz comes from the same timer that makes the pulses: it checks
 * the scheduling, not the formula. The independent check is the ESP32 count
 * (env enc_count) or a logic analyzer (26/10: the version without pi gave
 * measured = pi x expected, and the ESP32 counted the same).
 */

#define ENC_NUM_WHEELS  4u
#define ENC_TIMER_FREQ  1000000UL  /* Hz, tick rate after PSC=239        */

/* Clock correction (27/10, Liam approved in the CODE session, bug H7-CLK-HSE):
 * the H7 clock runs about 0.64 % fast against two independent references
 * (RPi5 frame rate and the ESP32 crystal; 05_DATA/results/H7_CLOCK/
 * 2026-10-27_run01). The CubeMX clock setup is correct (8 MHz HSE bypass,
 * FRACN = 0), so the 8 MHz from the ST-LINK into PH0 is suspected. One real
 * second is therefore ENC_TIMER_FREQ * ENC_CLOCK_CAL timer ticks. Re-measure
 * with the 'Link period' line (main.c) after changing board or ST-LINK. */
#define ENC_CLOCK_CAL   1.0064f
#define ENC_TICKS_PER_S ((float)ENC_TIMER_FREQ * ENC_CLOCK_CAL)
#define ENC_PPR         330u       /* pulses/rev on channel A            */

typedef struct {
    float    omega_cmd;       /* last commanded omega [rad/s]            */
    int8_t   dir;             /* +1 fwd, -1 rev, 0 stopped                */
    uint32_t half_period;     /* current half-period [ticks]              */
    uint32_t ch1_period;      /* measured period between CH1 toggles [ticks] */
    float    measured_freq;   /* ENC_TIMER_FREQ / (2 * ch1_period) [Hz]   */
    float    expected_freq;   /* |omega|/(2*pi) * ENC_PPR [Hz]            */
} EncoderGenDebug_t;

/* Initialise — pass timer handles in wheel order 1-4.
 * Does NOT call HAL_TIM_Base_Start/OC_Start_IT: those set HAL state=BUSY
 * and block the direct-register arming used by EncoderGen_SetOmega(). */
void EncoderGen_Init(TIM_HandleTypeDef *ht1,
                     TIM_HandleTypeDef *ht2,
                     TIM_HandleTypeDef *ht3,
                     TIM_HandleTypeDef *ht4);

/* Set target angular velocity [rad/s].
 * omega = 0 (within ENC_OMEGA_MIN) stops the wheel (no toggling).
 * Direction change stops and restarts with the new 90 deg phase offset.
 * Same direction, new speed: both pending edges are re-timed (encoder_gen.c
 * _retime, 28/10): the next edge comes within one NEW half period and B stays
 * half a new half period from A, so the direction the ESP32 decodes holds.
 * Called from the main loop; disables interrupts for a few hundred cycles. */
void EncoderGen_SetOmega(uint8_t wheel, float omega_rad_s);

/* Call from HAL_TIM_OC_DelayElapsedCallback in main.c.
 * htim->Channel is set by HAL before the callback. */
void EncoderGen_OC_Callback(TIM_HandleTypeDef *htim);

/* Fill dbg[ENC_NUM_WHEELS] with current per-wheel debug/verification data. */
void EncoderGen_GetDebug(EncoderGenDebug_t *dbg);

#endif /* ENCODER_GEN_H */
