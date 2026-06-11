#ifndef HIL_PROTOCOL_H
#define HIL_PROTOCOL_H

#include <stdint.h>

/* ── Frame sizes ─────────────────────────────────────────────────────────── */
#define SPI1_FRAME_LEN   24u   /* MOSI: RPi5→H7 (plant state), MISO: H7→RPi5 (ctrl output) */
#define SPI2_FRAME_LEN   64u   /* MOSI: ESP32→H7 (BNO085 req), MISO: H7→ESP32 (IMU data)   */

/* ── Physical ranges — MUST match RPi5 protocol.py ──────────────────────── */
#define HIL_OMEGA_MAX    40.0f      /* rad/s, per wheel      */
#define HIL_TORQUE_MAX    1.0f      /* N·m,  per wheel       */
#define HIL_POS_MAX      10.0f      /* m                     */
#define HIL_VEL_MAX       5.0f      /* m/s (body)            */
#define HIL_THETA_MAX     3.14159265f /* rad                 */
#define INT16_MAXF       32767.0f

/* ────────────────────────────────────────────────────────────────────────────
 * MOSI layout  (RPi5 → H7),  24 bytes
 * ─────────────────────────────────────────────────────────────────────────
 *  [0-1]   omega[0]    int16 BE   rad/s × INT16_MAXF/HIL_OMEGA_MAX
 *  [2-3]   omega[1]    int16 BE
 *  [4-5]   omega[2]    int16 BE
 *  [6-7]   omega[3]    int16 BE
 *  [8-9]   pos_x       int16 BE   m × INT16_MAXF/HIL_POS_MAX
 *  [10-11] pos_y       int16 BE
 *  [12-13] theta       int16 BE   rad × INT16_MAXF/HIL_THETA_MAX
 *  [14-15] vx          int16 BE   m/s × INT16_MAXF/HIL_VEL_MAX
 *  [16-17] vy          int16 BE
 *  [18-19] omega_body  int16 BE   rad/s × INT16_MAXF/HIL_VEL_MAX
 *  [20]    fault_flags uint8
 *  [21]    seq         uint8   (frame counter, wraps)
 *  [22-23] reserved    0x00
 *
 * MISO layout  (H7 → RPi5),  24 bytes
 * ─────────────────────────────────────────────────────────────────────────
 *  [0-1]   torque[0]   int16 BE   N·m × INT16_MAXF/HIL_TORQUE_MAX
 *  [2-3]   torque[1]   int16 BE
 *  [4-5]   torque[2]   int16 BE
 *  [6-7]   torque[3]   int16 BE
 *  [8-9]   pwm[0]      int16 BE   duty [-1..1] × INT16_MAXF
 *  [10-11] pwm[1]      int16 BE
 *  [12-13] pwm[2]      int16 BE
 *  [14-15] pwm[3]      int16 BE
 *  [16]    status      uint8
 *  [17]    seq         uint8
 *  [18-23] reserved    0x00
 * ─────────────────────────────────────────────────────────────────────────*/

/* ── Fault flag bits (MOSI byte 20) ─────────────────────────────────────── */
#define HIL_FAULT_W1_JAM   (1u << 0)
#define HIL_FAULT_W2_JAM   (1u << 1)
#define HIL_FAULT_W3_JAM   (1u << 2)
#define HIL_FAULT_W4_JAM   (1u << 3)
#define HIL_FAULT_W1_ENC   (1u << 4)
#define HIL_FAULT_W2_ENC   (1u << 5)
#define HIL_FAULT_W3_ENC   (1u << 6)
#define HIL_FAULT_W4_ENC   (1u << 7)

/* ── Plant state (decoded from MOSI) ────────────────────────────────────── */
typedef struct {
    float   omega[4];      /* wheel angular velocity [rad/s] */
    float   pos_x;         /* body position x [m]            */
    float   pos_y;         /* body position y [m]            */
    float   theta;         /* body heading [rad]             */
    float   vx;            /* body velocity x [m/s]          */
    float   vy;            /* body velocity y [m/s]          */
    float   omega_body;    /* body angular velocity [rad/s]  */
    uint8_t fault_flags;
    uint8_t seq;
} HIL_State_t;

/* ── Controller output (encoded to MISO) ────────────────────────────────── */
typedef struct {
    float   torque[4];     /* wheel torque command [N·m]     */
    float   pwm[4];        /* raw PWM duty [-1..1]           */
    uint8_t status;
    uint8_t seq;
} HIL_Output_t;

#endif /* HIL_PROTOCOL_H */
