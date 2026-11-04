#ifndef PWM_CAPTURE_H
#define PWM_CAPTURE_H

#include "stm32h7xx_hal.h"

/*
 * PWMCapture — Measure RPWM/LPWM duty cycles from ESP32 via TIM Input Capture
 *
 * Hardware (channel index 0..7 = RPWM FL FR RL RR, LPWM FL FR RL RR):
 *   0 TIM5  CH1  PA0   RPWM wheel 1        4 TIM15 CH1  PE5   LPWM wheel 1
 *   1 TIM8  CH3  PC8   RPWM wheel 2        5 TIM8  CH4  PC9   LPWM wheel 2
 *   2 TIM5  CH3  PA2   RPWM wheel 3        6 TIM16 CH1  PF6   LPWM wheel 3
 *   3 TIM5  CH4  PA3   RPWM wheel 4        7 TIM17 CH1  PF7   LPWM wheel 4
 * PC8 and PC9 are the pins on ZIO CN8; the CN12 spots with the same names are
 * not these pins (Liam measured 04/11 with the internal pull-up: CN8 3.2 V,
 * CN12 0 V). Wheel 2 moved to TIM8 on 04/11 because TIM13 and TIM15 CH2 have
 * no DMA request (before: PF8 TIM13, PE6 TIM15 CH2; PA1 TIM5 CH2 before 27/10).
 *
 * Timer clock: 10 MHz (PSC=23 on 240 MHz timer clocks), 20 kHz PWM period =
 * 500 ticks, 16-bit counting (ARR 65535, also TIM5), wraps after 6.5 ms.
 *
 * Since 04/11 (bug H7-PWM-RISE-MISS, Liam: "duyệt chuyển sang DMA đi"): each
 * channel captures BOTH edges, input filter 0xF, and a circular DMA2 stream
 * copies each capture into a RAM ring; there is no capture interrupt. The old
 * scheme toggled the polarity in the capture ISR and missed an edge whenever
 * the ISR came later than the short phase of the period (RL step 3: 15 us low,
 * 05_DATA/results/PWM_CAPTURE/2026-11-04_run01). Decoding: pwm_edges.c.
 *
 * Torque conversion (RPWM/LPWM → signed duty [-1..1]):
 *   duty_net[w] = rpwm_duty[w] - lpwm_duty[w]
 */

#define PWM_NUM_WHEELS   4u
#define PWM_NUM_CHANNELS 8u
#define PWM_TIMEOUT_MS  50u   /* channel considered held if no edge for this long */
#define PWM_RING_LEN  1024u   /* captures per channel ring: 25.6 ms at 20 kHz, both edges */

/* Start the 8 DMA streams and the capture channels (after MX_DMA_Init and the
 * MX_TIMx_Init calls). The capture timers' NVIC lines stay off. */
void  PWMCapture_Init(TIM_HandleTypeDef *ht5,
                      TIM_HandleTypeDef *ht8,
                      TIM_HandleTypeDef *ht15,
                      TIM_HandleTypeDef *ht16,
                      TIM_HandleTypeDef *ht17);

/* Read new captures and update the duty of each channel. Call every pass of
 * the main loop (about 3 us, up to about 25 us while an edge type is still
 * unknown). */
void  PWMCapture_Service(void);

/* Net duty [-1..1] per wheel from the last PWMCapture_Service().
 * Positive = RPWM dominates (forward), Negative = LPWM dominates (reverse).
 * A channel with no edge within PWM_TIMEOUT_MS counts as its pin level
 * (1.0 if held HIGH, 0.0 if held LOW). */
void  PWMCapture_GetDuty(float *duty_out);  /* float[4] */

typedef struct {
    uint32_t edges[PWM_NUM_CHANNELS];       /* captures read since boot */
    uint32_t lost[PWM_NUM_CHANNELS];        /* overcapture flags seen */
    uint32_t lap[PWM_NUM_CHANNELS];         /* service gaps >= 5 ms (ring dropped) */
    uint32_t parity_fix[PWM_NUM_CHANNELS];  /* anchor disagreed with the counted edge type */
    uint8_t  known[PWM_NUM_CHANNELS];       /* edge type known now */
} PWMCapture_Stats_t;

/* Debug counters, channel order as above */
void  PWMCapture_GetStats(PWMCapture_Stats_t *out);

#endif /* PWM_CAPTURE_H */
