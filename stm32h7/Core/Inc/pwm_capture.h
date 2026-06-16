#ifndef PWM_CAPTURE_H
#define PWM_CAPTURE_H

#include "stm32h7xx_hal.h"
#include "stm32h7xx_ll_tim.h"

/*
 * PWMCapture — Measure RPWM/LPWM duty cycles from ESP32 via TIM Input Capture
 *
 * Hardware:
 *   TIM5  CH1-4 → RPWM wheels 1-4  (PA0, PA1, PA2, PA3)
 *   TIM15 CH1-2 → LPWM wheels 1-2  (PE5, PE6)
 *   TIM16 CH1   → LPWM wheel 3     (PF6)
 *   TIM17 CH1   → LPWM wheel 4     (PF7)
 *
 * Timer clock: 1 MHz (same as encoder gen, PSC=239)
 *
 * Duty cycle measurement (rising→falling→rising):
 *   Rising  edge: record t_rise
 *   Falling edge: pulse_width = t_fall - t_rise
 *   Next rising:  period = t_rise_new - t_rise_prev
 *   duty [0..1]  = pulse_width / period
 *
 * Polarity toggle uses LL (ISR-safe single register write).
 * HAL_TIM_IC_ConfigChannel is NOT used in ISR — HAL state machine is
 * not re-entrant.
 *
 * ⚠ WARNING: PWMCapture_Init() caused SPI1 corruption in prior testing.
 * Root cause unresolved. Enable only when ESP32 is connected and SPI1
 * behaviour can be verified.  Keep commented out in Phase 1 & 2.
 *
 * Torque conversion (RPWM/LPWM → signed duty [-1..1]):
 *   duty_net[w] = rpwm_duty[w] - lpwm_duty[w]
 *   torque[w]   = duty_net[w] × HIL_TORQUE_MAX
 */

#define PWM_NUM_WHEELS   4u
#define PWM_TIMEOUT_MS  50u   /* wheel considered stopped if no edge for this long */

/* Initialise all IC timers and arm first capture */
void  PWMCapture_Init(TIM_HandleTypeDef *ht5,
                      TIM_HandleTypeDef *ht15,
                      TIM_HandleTypeDef *ht16,
                      TIM_HandleTypeDef *ht17);

/* Returns net duty [-1..1] for each wheel.
 * Positive = RPWM dominates (forward), Negative = LPWM dominates (reverse).
 * Sets duty to 0 if no pulse seen within PWM_TIMEOUT_MS. */
void  PWMCapture_GetDuty(float *duty_out);  /* float[4] */

/* Call from HAL_TIM_IC_CaptureCallback in main.c */
void  PWMCapture_IC_Callback(TIM_HandleTypeDef *htim);

#endif /* PWM_CAPTURE_H */
