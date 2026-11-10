/*
 * fwview.c: what the ESP32 firmware receives from one IMU sample on HIL (Codex round 1
 * IMN-2, 10/11/2026), for the check of rpi5/imu_model.firmware_view().
 *
 * stdin: one line per sample, the five int16 IMU fields of the MOSI frame as the Pi
 * writes them (protocol.encode_state): raw_gz raw_ax raw_ay raw_az raw_yaw.
 * stdout: one line per sample: gz ax ay az yaw_deg as imu.cpp stores them.
 *
 * Chain (only the two copied steps are not the target code itself):
 *   [copy]  spi1_handler.c: value = raw * (HIL_IMU_*_MAX / INT16_MAXF)
 *   [copy]  bno_emu_port.c _sample_from_imu: qk = sinf(yaw/2), qr = cosf(yaw/2), qi = qj = 0
 *   [code]  bno_emu.c build_report (to_fixed Q8 / Q9 / Q14), included below
 *   [code]  ESP32 SH-2 library sh2_decodeSensorEvent (sh2_SensorValue.c, unchanged)
 *   [copy]  BNO08xGlobalTypes.hpp bno08x_euler_angle_t = quat, times RAD_2_DEG
 * Limit: PC float maths (sinf, cosf, atan2f of this compiler's runtime), not the
 * H7 / ESP32 newlib; a last-bit difference can move a value by one Q14 step at a
 * rounding boundary.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "sh2.h"
#include "sh2_SensorValue.h"
#include "sh2_err.h"
#include "hil_protocol.h"
#include "../stm32h7/Core/Src/bno_emu.c"    /* static build_report(), to_fixed() */

int main(void)
{
    static bno_emu_t emu;
    bno_emu_cfg_t cfg;
    bno_emu_default_cfg(&cfg);
    bno_emu_init(&emu, &cfg, 0);
    const float rad2deg = (float)(180.0f / M_PI);

    int r[5];
    while (scanf("%d %d %d %d %d", &r[0], &r[1], &r[2], &r[3], &r[4]) == 5) {
        float gz  = (float)(int16_t)r[0] * (HIL_IMU_GYRO_MAX  / INT16_MAXF);
        float ax  = (float)(int16_t)r[1] * (HIL_IMU_ACCEL_MAX / INT16_MAXF);
        float ay  = (float)(int16_t)r[2] * (HIL_IMU_ACCEL_MAX / INT16_MAXF);
        float az  = (float)(int16_t)r[3] * (HIL_IMU_ACCEL_MAX / INT16_MAXF);
        float yaw = (float)(int16_t)r[4] * (HIL_IMU_YAW_MAX   / INT16_MAXF);

        bno_emu_sample_t s;
        memset(&s, 0, sizeof s);
        float h = 0.5f * yaw;
        s.ax = ax;  s.ay = ay;  s.az = az;
        s.gx = 0.0f; s.gy = 0.0f; s.gz = gz;
        s.qi = 0.0f; s.qj = 0.0f;
        s.qk = sinf(h);
        s.qr = cosf(h);
        emu.sample = s;

        float out[5] = { 0 };
        for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) {
            uint8_t rep[64];
            uint16_t n = build_report(&emu, i, rep, 0);
            sh2_SensorEvent_t ev;
            memset(&ev, 0, sizeof ev);
            ev.len = (uint8_t)n;
            ev.reportId = rep[0];
            memcpy(ev.report, rep, n);
            sh2_SensorValue_t v;
            if (sh2_decodeSensorEvent(&v, &ev) != SH2_OK) { printf("decode_error %u\n", i); return 2; }
            if (k_sensor_id[i] == BNO_EMU_ID_ACCEL) {
                out[1] = v.un.accelerometer.x; out[2] = v.un.accelerometer.y; out[3] = v.un.accelerometer.z;
            } else if (k_sensor_id[i] == BNO_EMU_ID_GYRO) {
                out[0] = v.un.gyroscope.z;
            } else {
                float qr = v.un.gameRotationVector.real, qi = v.un.gameRotationVector.i;
                float qj = v.un.gameRotationVector.j, qk = v.un.gameRotationVector.k;
                float z = atan2f(2.0f * (qr * qk + qi * qj), 1.0f - 2.0f * (qj * qj + qk * qk));
                out[4] = z * rad2deg;
            }
        }
        printf("%.17g %.17g %.17g %.17g %.17g\n", (double)out[0], (double)out[1], (double)out[2],
               (double)out[3], (double)out[4]);
    }
    return 0;
}
