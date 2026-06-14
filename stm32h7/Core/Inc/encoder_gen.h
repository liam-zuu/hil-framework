#ifndef ENCODER_GEN_H
#define ENCODER_GEN_H

#include "stm32h7xx_hal.h"

/*
 * EncoderGen — Quadrature encoder pulse generation via TIM Output Compare Toggle
 *
 * Hardware (CubeMX-generated, OC Toggle CH1/CH2):
 *   TIM1 → Wheel 1 (CH1=PE9,  CH2=PE11)
 *   TIM2 → Wheel 2 (CH1=PA0,  CH2=PA1)
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
 * preserved automatically as both channels advance by the same
 * half_period_ticks on every callback.
 *
 * ── Debug / verification ─────────────────────────────────────────────────
 * For each wheel, the time between two consecutive CH1 (channel A)
 * callbacks is measured in timer ticks. Two consecutive callbacks on the
 * same channel = one toggle pair = one full pulse period.
 *
 *   measured_period_ticks ≈ 2 * half_period_ticks   (if logic is correct)
 *   measured_freq_hz      = ENC_TIMER_FREQ / measured_period_ticks
 *   expected_freq_hz      = |omega| / (2*pi) * ENC_PPR
 *
 * Compare measured_freq_hz vs expected_freq_hz to check for the 2*pi-class
 * discrepancy seen previously. EncoderGen_GetDebug() exposes both per wheel.
 */

#define ENC_NUM_WHEELS  4u
#define ENC_TIMER_FREQ  1000000UL  /* Hz, tick rate after PSC=239        */
#define ENC_PPR         330u       /* pulses/rev on channel A            */

typedef struct {
    float    omega_cmd;       /* last commanded omega [rad/s]            */
    int8_t   dir;             /* +1 fwd, -1 rev, 0 stopped                */
    uint32_t half_period;     /* current half-period [ticks]              */
    uint32_t ch1_period;      /* measured period between CH1 toggles [ticks] */
    float    measured_freq;   /* ENC_TIMER_FREQ / ch1_period [Hz]         */
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
 * Same direction: only half_period is updated, picked up on next callback. */
void EncoderGen_SetOmega(uint8_t wheel, float omega_rad_s);

/* Call from HAL_TIM_OC_DelayElapsedCallback in main.c.
 * htim->Channel is set by HAL before the callback. */
void EncoderGen_OC_Callback(TIM_HandleTypeDef *htim);

/* Fill dbg[ENC_NUM_WHEELS] with current per-wheel debug/verification data. */
void EncoderGen_GetDebug(EncoderGenDebug_t *dbg);

#endif /* ENCODER_GEN_H */
