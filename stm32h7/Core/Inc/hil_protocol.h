/* hil_protocol.h — HIL SPI frame definitions
 * MOSI: RPi5 → H7  (40 bytes: plant state + trajectory reference + IMU)
 * MISO: H7   → RPi5 (40 bytes: controller output + padding)
 *
 * History:
 *   v1: 24 bytes (plant state only)
 *   v2: 36 bytes (+ vx_ref, vy_ref, wz_ref, traj_flags)
 *   v3: 40 bytes (+ IMU values for the BNO085 emulator: gz, ax, ay, az, yaw,
 *       imu_seq; G0 v1 §5, Liam chose 40 bytes 22/10/2026)
 *   MISO v4 (S06, 29/10/2026, Liam chose "inside MISO SPI1"): the unused
 *       bytes 8-15 (pwm, never set by the H7) and 18-39 carry checksums,
 *       the H7 cycle stamp of the answered frame and one S06 log record
 *       per frame. MOSI unchanged except fault bit 2 (IMU loss injection).
 */

#ifndef HIL_PROTOCOL_H
#define HIL_PROTOCOL_H

#include <stdint.h>

/* ── Frame size ─────────────────────────────────────────────────────── */
#define SPI1_FRAME_LEN      40          /* bytes, both directions */

/* ── Scale factors (MOSI — plant state) ────────────────────────────── */
#define HIL_OMEGA_MAX       40.0f       /* rad/s  — wheel speed  */
#define HIL_POS_MAX         5.0f        /* m      — x, y         */
#define HIL_THETA_MAX       3.14159265f /* rad    — heading      */
#define HIL_VEL_MAX         3.0f        /* m/s    — vx, vy       */
#define HIL_OMEGA_BODY_MAX  10.0f       /* rad/s  — wz body      */

/* ── Scale factors (MOSI — trajectory reference) ────────────────────── */
#define HIL_REF_VEL_MAX     3.0f        /* m/s    — vx_ref, vy_ref */
#define HIL_REF_OMEGA_MAX   10.0f       /* rad/s  — wz_ref         */

/* ── Scale factors (MOSI — IMU values for the BNO085 emulator) ────────── */
#define HIL_IMU_GYRO_MAX    34.906585f  /* rad/s  — gz, ±2000 deg/s (params.py imu_gyro_range)  */
#define HIL_IMU_ACCEL_MAX   39.24f      /* m/s^2  — ax, ay, az, ±4 g (params.py imu_accel_range) */
#define HIL_IMU_YAW_MAX     3.14159265f /* rad    — yaw                                          */

/* ── Scale factors (MISO — controller output) ───────────────────────── */
#define HIL_TORQUE_MAX      1.0f        /* N·m */

/* ── Fixed-point helpers ─────────────────────────────────────────────── */
#define INT16_MAXF          32767.0f

/* ── MOSI layout (RPi5 → H7), 40 bytes ─────────────────────────────── */
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
/* Byte 22–23  gz_imu       int16 × (HIL_IMU_GYRO_MAX / INT16_MAXF)  v3 */
/* Byte 24–25  vx_ref       int16 × (HIL_REF_VEL_MAX / INT16_MAXF)   v2 */
/* Byte 26–27  vy_ref                                                 v2 */
/* Byte 28–29  wz_ref       int16 × (HIL_REF_OMEGA_MAX / INT16_MAXF) v2 */
/* Byte 30     traj_flags   uint8  (bit0=valid, bit1=new)             v2 */
/* Byte 31–32  ax_imu       int16 × (HIL_IMU_ACCEL_MAX / INT16_MAXF) v3 */
/* Byte 33–34  ay_imu                                                 v3 */
/* Byte 35–36  az_imu                                                 v3 */
/* Byte 37–38  yaw_imu      int16 × (HIL_IMU_YAW_MAX / INT16_MAXF)   v3 */
/* Byte 39     imu_seq      uint8, +1 for every new plant IMU sample  v3 */

/* ── MISO layout (H7 → RPi5), 40 bytes ─────────────────────────────── */
/* Byte  0– 1  torque[0]    int16 × (HIL_TORQUE_MAX / INT16_MAXF)       */
/* Byte  2– 3  torque[1]                                                 */
/* Byte  4– 5  torque[2]                                                 */
/* Byte  6– 7  torque[3]                                                 */
/* v4 (S06): bytes 8-15 and 18-39 below; multi-byte fields little-endian */
/* Byte  8– 9  miso_crc     CRC-16 of bytes 0-7 and 10-39 of this frame,  */
/*                          computed when the frame is staged (Z3)        */
/* Byte 10–11  mosi_crc     CRC-16 of the 40 raw MOSI bytes of the frame  */
/*                          answered, on the main-loop copy (W3, Y9)      */
/* Byte 12–15  rx_cyc       DWT CYCCNT in SPI1_Callback of that frame     */
/* Byte 16     status       uint8                                        */
/* Byte 17     seq          uint8, seq of the frame answered             */
/* Byte 18–19  rx_nf        low 16 bits of the H7 frame count of it      */
/* Byte 20–21  age_us       callback -> main loop applied it, us (sat.)  */
/* Byte 22     log_type     S06 log record type, 0 = none (s06_log.h)    */
/* Byte 23–24  log_idx      continuous index of the H7 log record        */
/* Byte 25–39  log_payload  15 bytes, by type                            */
/* CRC-16: CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no XOR) */
#define HIL_MISO_CRC        8
#define HIL_MISO_MOSI_CRC   10
#define HIL_MISO_RX_CYC     12
#define HIL_MISO_STATUS     16
#define HIL_MISO_SEQ        17
#define HIL_MISO_RX_NF      18
#define HIL_MISO_AGE        20
#define HIL_MISO_LOG        22
#define HIL_MISO_LOG_LEN    18          /* type + idx + 15-byte payload */

/* ── Fault flags (MOSI byte 20) ─────────────────────────────────────── */
#define HIL_FAULT_PLANT     (1u << 0)
#define HIL_FAULT_OVERRUN   (1u << 1)
#define HIL_FAULT_IMU_LOSS  (1u << 2)   /* S06 T8: emulator builds no report [ĐỀ XUẤT] */

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
    float gz;                   /* rad/s */
    float ax, ay, az;           /* m/s^2 */
    float yaw;                  /* rad   */
    uint8_t imu_seq;
    /* S06 (T1 Z3): raw MOSI bytes 22-23 and 31-38 (gz, ax, ay, az, yaw) of
     * the same frame, and the H7 frame count of that frame, for the
     * checksum and source frame in the report-build log */
    uint8_t  raw[10];
    uint32_t src_nf;
} HIL_Imu_t;

typedef struct {
    float torque[4];
    float pwm[4];               /* not sent since MISO v4 (bytes 8-15 reused) */
    uint8_t status;
    uint8_t seq;
} HIL_Output_t;

#endif /* HIL_PROTOCOL_H */
