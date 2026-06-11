#ifndef ENCODER_GEN_H
#define ENCODER_GEN_H

#include "stm32h7xx_hal.h"

/*
 * EncoderGen — Quadrature encoder pulse generation via TIM Output Compare Toggle
 *
 * Hardware:
 *   TIM1 → Wheel 1 (CH1=PE9/W1_A, CH2=PE11/W1_B)
 *   TIM2 → Wheel 2 (CH1=W2_A,     CH2=W2_B)
 *   TIM3 → Wheel 3 (CH1=W3_A,     CH2=W3_B)
 *   TIM4 → Wheel 4 (CH1=W4_A,     CH2=W4_B)
 *
 * Timer clock: APB_CLK × 2 / (PSC+1) = 240 MHz / 240 = 1 MHz
 * PPR: 330 pulses per revolution (channel A rising edges)
 *
 * Quadrature direction:
 *   Forward (omega > 0): A leads B by 90° (B starts half_period/2 after A)
 *   Reverse (omega < 0): B leads A by 90° (A starts half_period/2 after B)
 *
 * Debug: read enc_oc_count via debugger after ~1 s.
 *   Expected ≈ 2 × pulse_freq × N_wheels_running
 *   (2 OC events per pulse period, per active wheel)
 */

#define ENC_NUM_WHEELS  4u

/* Initialise — pass timer handles in wheel order 1-4 */
void EncoderGen_Init(TIM_HandleTypeDef *ht1,
                     TIM_HandleTypeDef *ht2,
                     TIM_HandleTypeDef *ht3,
                     TIM_HandleTypeDef *ht4);

/* Set target angular velocity.  Thread-safe: only updates half_period field.
 * Direction change causes stop+restart.  omega = 0 stops the wheel. */
void EncoderGen_SetOmega(uint8_t wheel, float omega_rad_s);

/* Call from HAL_TIM_OC_DelayElapsedCallback in main.c */
void EncoderGen_OC_Callback(TIM_HandleTypeDef *htim);

/* Debug counter (extern so debugger/watch can read it) */
extern volatile uint32_t enc_oc_count;

#endif /* ENCODER_GEN_H */
