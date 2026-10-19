/* hil_protocol.h — HIL SPI frame definitions
 * MOSI: RPi5 → H7  (36 bytes: plant state + trajectory reference)
 * MISO: H7   → RPi5 (36 bytes: controller output + padding)
 *
 * History:
 *   v1: 24 bytes (plant state only)
 *   v2: 36 bytes (+ vx_ref, vy_ref, wz_ref, traj_flags)
 */

#ifndef HIL_PROTOCOL_H
#define HIL_PROTOCOL_H

#include <stdint.h>

/* ── Frame size ─────────────────────────────────────────────────────── */
#define SPI1_FRAME_LEN      36          /* bytes, both directions */

/* ── Scale factors (MOSI — plant state) ────────────────────────────── */
#define HIL_OMEGA_MAX       40.0f       /* rad/s  — wheel speed  */
#define HIL_POS_MAX         5.0f        /* m      — x, y         */
#define HIL_THETA_MAX       3.14159265f /* rad    — heading      */
#define HIL_VEL_MAX         3.0f        /* m/s    — vx, vy       */
#define HIL_OMEGA_BODY_MAX  10.0f       /* rad/s  — wz body      */

/* ── Scale factors (MOSI — trajectory reference) ────────────────────── */
#define HIL_REF_VEL_MAX     3.0f        /* m/s    — vx_ref, vy_ref */
#define HIL_REF_OMEGA_MAX   10.0f       /* rad/s  — wz_ref         */

/* ── Scale factors (MISO — controller output) ───────────────────────── */
#define HIL_TORQUE_MAX      1.0f        /* N·m */

/* ── Fixed-point helpers ─────────────────────────────────────────────── */
#define INT16_MAXF          32767.0f

/* ── MOSI layout (RPi5 → H7), 36 bytes ─────────────────────────────── */
/* Byte  0– 1  omega[0]     int16 × (HIL_OMEGA_MAX / INT16_MAXF)        */
/* Byte  2– 3  omega[1]                                                  */
/* Byte  4– 5  omega[2]                                                  */
/* Byte  6– 7  omega[3]                                                  */
/* Byte  8– 9  pos_x        int16 × (HIL_POS_MAX / INT16_MAXF)          */
/* Byte 10–11  pos_y                                                     */
/* Byte 12–13  theta        int16 × (HIL_THETA_MAX / INT16_MAXF)        */
/* Byte 14–15  vx           int16 × (HIL_VEL_MAX / INT16_MAXF)          */
/* Byte 16–17  vy                                                        */
/* Byte 18–19  omega_body   int16 × (HIL_OMEGA_BODY_MAX / INT16_MAXF)   */
/* Byte 20     fault_flags  uint8                                        */
/* Byte 21     seq          uint8                                        */
/* Byte 22–23  reserved     0x00                                         */
/* Byte 24–25  vx_ref       int16 × (HIL_REF_VEL_MAX / INT16_MAXF)  ←NEW */
/* Byte 26–27  vy_ref                                                ←NEW */
/* Byte 28–29  wz_ref       int16 × (HIL_REF_OMEGA_MAX / INT16_MAXF)←NEW */
/* Byte 30     traj_flags   uint8  (bit0=valid, bit1=new)            ←NEW */
/* Byte 31–35  reserved     0x00                                     ←NEW */

/* ── MISO layout (H7 → RPi5), 36 bytes ─────────────────────────────── */
/* Byte  0– 1  torque[0]    int16 × (HIL_TORQUE_MAX / INT16_MAXF)       */
/* Byte  2– 3  torque[1]                                                 */
/* Byte  4– 5  torque[2]                                                 */
/* Byte  6– 7  torque[3]                                                 */
/* Byte  8– 9  pwm[0]       int16 = duty × INT16_MAXF                   */
/* Byte 10–11  pwm[1]                                                    */
/* Byte 12–13  pwm[2]                                                    */
/* Byte 14–15  pwm[3]                                                    */
/* Byte 16     status       uint8                                        */
/* Byte 17     seq          uint8                                        */
/* Byte 18–35  reserved     0x00                                         */

/* ── Fault flags (MOSI byte 20) ─────────────────────────────────────── */
#define HIL_FAULT_PLANT     (1u << 0)
#define HIL_FAULT_OVERRUN   (1u << 1)

/* ── Traj flags (MOSI byte 30) ──────────────────────────────────────── */
#define HIL_TRAJ_VALID      (1u << 0)   /* setpoint is valid            */
#define HIL_TRAJ_NEW        (1u << 1)   /* new setpoint this frame      */

/* ── Trajectory relay packet (H7 → ESP32 via USART2) ───────────────── */
/* 14 bytes: identical format to WiFi UDP packet on ESP32               */
/* 0xBB | vx_ref(4B LE float) | vy_ref(4B LE float)                    */
/*       | wz_ref(4B LE float) | XOR checksum(1B)                       */
#define TRAJ_UART_HEADER    0xBB
#define TRAJ_UART_LEN       14

/* ── Struct types ────────────────────────────────────────────────────── */
typedef struct {
    float omega[4];
    float pos_x, pos_y, theta;
    float vx, vy, omega_body;
    uint8_t fault_flags;
    uint8_t seq;
} HIL_State_t;

typedef struct {
    float vx_ref, vy_ref, wz_ref;
    uint8_t traj_flags;
} HIL_TrajRef_t;

typedef struct {
    float torque[4];
    float pwm[4];
    uint8_t status;
    uint8_t seq;
} HIL_Output_t;

#endif /* HIL_PROTOCOL_H */
