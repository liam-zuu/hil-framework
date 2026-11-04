/* fw_sim.c: PC entry points around the ESP32 firmware (AGV_MECANUM).
 *
 * The DLL is built from the firmware's own encoder.c, motor.c, kinematics.c,
 * pid.c, adrc.c, imu_fusion.c, heading.c, vel_ctrl.c (unchanged) plus the shim in
 * shim/. Only the scheduling glue of main.c is repeated here, because main.c needs
 * FreeRTOS:
 *   encoder_task: encoder_get_rpm() and encoder_get_count() for every wheel,
 *   control_task: dt_actual from esp_timer, imu_get() copy, vel_ctrl_step(),
 *                 motor on: motor_set() x 4 + vel_ctrl_applied(); off: motor_stop_all().
 * Both tasks wake on the same 10 ms tick; encoder_task has the higher priority
 * so it runs first (main.c). Ultrasonic interlock, RC and mode logic are not
 * modelled: the setpoint goes straight in, as in MODE_AUTO with no obstacle.
 *
 * IMU: what imu.cpp's report callback writes into imu_data_t (value, host receive
 * time, counter per report; reset_n on SH2_RESET) is written here by
 * fwsim_imu_report() / fwsim_imu_reset(), called from Python at the report times.
 * Every tick passes a copy to vel_ctrl_step(), as control_task copies imu_get().
 */
#include "shim/fw_shim.h"
#include "esp_timer.h"
#include "motor.h"
#include "encoder.h"
#include "vel_ctrl.h"
#include <string.h>

/* Firmware with the position loop (AGV feat/pos-loop, O1 PA3, 04/11): optional, so older
 * firmware trees still build. */
#if __has_include("pos_ctrl.h")
#define FWSIM_HAS_POS 1
#else
#define FWSIM_HAS_POS 0
#endif
/* Firmware with the wheel-consistency slip flag (AGV feat/wheel-slip, 04/11): optional too. */
#if __has_include("wheel_slip.h")
#define FWSIM_HAS_WS 1
#else
#define FWSIM_HAS_WS 0
#endif

#define FWSIM_API __declspec(dllexport)
#define FWSIM_ABI 5

static const int rpwm_gpio[4] = { FL_RPWM, FR_RPWM, RL_RPWM, RR_RPWM };
static const int lpwm_gpio[4] = { FL_LPWM, FR_LPWM, RL_LPWM, RR_LPWM };
static int64_t t_prev;
static int32_t count_in[4];
static int32_t enc_count[4];        /* encoder_get_count() of the last tick */
static vel_ctrl_out_t last_vc;      /* last vel_ctrl_step() output (metric counters) */
static imu_data_t imu;              /* what imu.cpp keeps in g_data */
static int imu_on;                  /* 0: no imu_get() copy yet (vel_ctrl gets NULL) */
static int run_active = 1, wheels_lifted;
static int imu_fail;                /* imu_init() failed at boot: g_imu_fail in main.c */

FWSIM_API int fwsim_abi(void) { return FWSIM_ABI; }
/* After a tick: encoder_get_count() per wheel and the vel_ctrl metric counters
 * out[0] tick, out[1..4] dpwm_sum FL..RR, out[5..8] sat_ticks FL..RR (as logged by main.c). */
FWSIM_API void fwsim_last(int32_t counts[4], uint32_t out[9]) {
    out[0] = last_vc.tick;
    for (int i = 0; i < 4; i++) {
        counts[i] = enc_count[i];
        out[1 + i] = last_vc.dpwm_sum[i];
        out[5 + i] = last_vc.sat_ticks[i];
    }
}

FWSIM_API int  fwsim_ctrl_mode(void) { return vel_ctrl_mode(); }
FWSIM_API void fwsim_set_ctrl_mode(int mode) { vel_ctrl_set_mode(mode); }

/* Boot: what app_main + control_task do before the first control tick. */
FWSIM_API void fwsim_reset(int pcnt_mode) {
    fw_shim_reset();
    fw_pcnt_mode = pcnt_mode;
    for (int i = 0; i < 4; i++) count_in[i] = 0;
    memset(&imu, 0, sizeof(imu));
    imu_on = 0;
    run_active = 1;
    wheels_lifted = 0;
    imu_fail = 0;
    motor_init();
    encoder_init();
    vel_ctrl_init();
    motor_enable(false);
    t_prev = esp_timer_get_time();
    motor_enable(true);            /* HIL_BYPASS_MOTOR_ENABLE: swa_on = true on the first tick */
}

/* One IMU report at host receive time t_us, as imu.cpp's callback stores it.
 * kind 0 accel: v = ax, ay, az [m/s^2]; 1 gyro: v[2] = gz [rad/s]; 2 game RV: v[2] = yaw [deg]. */
FWSIM_API void fwsim_imu_report(int kind, int64_t t_us, const float v[3]) {
    imu_on = 1;
    imu.valid = true;
    if (kind == 0) {
        imu.accel_x = v[0]; imu.accel_y = v[1]; imu.accel_z = v[2];
        imu.accel_t_us = t_us; imu.accel_n++;
    } else if (kind == 1) {
        imu.gyro_x = v[0]; imu.gyro_y = v[1]; imu.gyro_z = v[2];
        imu.gyro_t_us = t_us; imu.gyro_n++;
    } else {
        imu.roll = v[0]; imu.pitch = v[1]; imu.yaw = v[2];
        imu.rv_t_us = t_us; imu.rv_n++;
    }
}

/* SH2_RESET from the chip at t_us (reset_n counter, spec controller-IMU v2 section 3). */
FWSIM_API void fwsim_imu_reset(int64_t t_us) {
    imu_on = 1;
    imu.reset_n++;
    imu.reset_t_us = t_us;
}

/* SWA / motor_enable() (fwsim_reset turns the driver on, as HIL_BYPASS_MOTOR_ENABLE).
 * After an IMU init failure main.c forces swa_on = false: the driver never turns on. */
FWSIM_API void fwsim_motor_enable(int on) { motor_enable(on != 0 && !imu_fail); }

/* imu_init() failed (spec controller-IMU v2.2 section 7): call right after fwsim_reset(),
 * before the first tick. main.c sets g_imu_fail and keeps the motors disabled for the run. */
FWSIM_API void fwsim_imu_init_fail(void) {
    imu_fail = 1;
    motor_enable(false);
}

/* run_active: MODE_AUTO with a VALID trajectory (heading loop runs); wheels_lifted: bias off. */
FWSIM_API void fwsim_set_run(int run, int lifted) { run_active = run; wheels_lifted = lifted; }

/* Thresholds of imu_fusion / heading by name; returns 0 if the name is unknown. */
FWSIM_API int fwsim_set_param(const char *name, double v) {
    imu_fusion_params_t *f = imu_fusion_params();
    heading_params_t *h = heading_params();
#define F(n) if (!strcmp(name, #n)) { f->n = (__typeof__(f->n))v; return 1; }
#define H(n) if (!strcmp(name, #n)) { h->n = (float)v; return 1; }
    F(t_age_us) F(t_settle_us) F(gyro_sat) F(n_back) F(sigma_gz) F(sigma_a) F(omega_min)
    F(k_r) F(q_wz) F(t_slip_us) F(t_still_us) F(b_max) F(g_ref)
    H(k_psi) H(wz_corr_max) H(e_psi_reset)
#undef F
#undef H
#if FWSIM_HAS_POS
    /* position loop (pos_ctrl.c, firmware with O1 PA3): pos_on, pos_kp, pos_ki, pos_v_corr_max */
    pos_params_t *q = pos_params();
    if (!strcmp(name, "pos_on"))         { q->on = v != 0.0; return 1; }
    if (!strcmp(name, "pos_kp"))         { q->kp = (float)v; return 1; }
    if (!strcmp(name, "pos_ki"))         { q->ki = (float)v; return 1; }
    if (!strcmp(name, "pos_v_corr_max")) { q->v_corr_max = (float)v; return 1; }
#endif
#if FWSIM_HAS_WS
    /* wheel slip flag (wheel_slip.c): ws_win_ticks, ws_n_min, ws_k_rel, ws_act_min, ws_t_hold_us */
    wheel_slip_params_t *w = wheel_slip_params();
    if (!strcmp(name, "ws_win_ticks")) { w->win_ticks = (uint16_t)v; return 1; }
    if (!strcmp(name, "ws_n_min"))     { w->n_min = (float)v; return 1; }
    if (!strcmp(name, "ws_k_rel"))     { w->k_rel = (float)v; return 1; }
    if (!strcmp(name, "ws_act_min"))   { w->act_min = (float)v; return 1; }
    if (!strcmp(name, "ws_t_hold_us")) { w->t_hold_us = (int64_t)v; return 1; }
#endif
    return 0;
}

/* Wheel slip flag of the last tick (fwsim_ws_cols() = 0 without wheel_slip.c):
 * out[0] res, 1 th, 2 eval, 3 flag, 4 on_count */
#define FWSIM_WS_COLS 5
FWSIM_API int fwsim_ws_cols(void) { return FWSIM_HAS_WS ? FWSIM_WS_COLS : 0; }
FWSIM_API void fwsim_ws_last(double out[FWSIM_WS_COLS]) {
#if FWSIM_HAS_WS
    const wheel_slip_out_t *s = &last_vc.wslip;
    double v[FWSIM_WS_COLS] = { s->res, s->th, s->eval, s->flag, s->on_count };
    memcpy(out, v, sizeof(v));
#else
    memset(out, 0, sizeof(double) * FWSIM_WS_COLS);
#endif
}

/* Position loop of the last tick (firmware with pos_ctrl.c; fwsim_pos_cols() = 0 without it):
 * out[0] on, 1 corr_vx, 2 corr_vy, 3 x_ref, 4 y_ref, 5 th_ref, 6 x, 7 y, 8 th, 9 ex_b, 10 ey_b, 11 th_src */
#define FWSIM_POS_COLS 12
FWSIM_API int fwsim_pos_cols(void) { return FWSIM_HAS_POS ? FWSIM_POS_COLS : 0; }
/* out: on, kp, ki, v_corr_max as the firmware holds them (read back after a run) */
FWSIM_API void fwsim_get_pos(float out[4]) {
#if FWSIM_HAS_POS
    const pos_params_t *q = pos_params();
    out[0] = q->on; out[1] = q->kp; out[2] = q->ki; out[3] = q->v_corr_max;
#else
    out[0] = out[1] = out[2] = out[3] = 0.0f;
#endif
}
FWSIM_API void fwsim_pos_last(double out[FWSIM_POS_COLS]) {
#if FWSIM_HAS_POS
    const pos_out_t *p = &last_vc.pos;
    double v[FWSIM_POS_COLS] = { p->on, p->corr_vx, p->corr_vy, p->x_ref, p->y_ref, p->th_ref,
                                 p->x, p->y, p->th, p->ex_b, p->ey_b, p->th_src };
    memcpy(out, v, sizeof(v));
#else
    memset(out, 0, sizeof(double) * FWSIM_POS_COLS);
#endif
}

/* IMU/heading columns of the last tick (log section 8 of the spec):
 * out[0] g_valid, 1 r_valid, 2 wz_src, 3 wz_meas, 4 gz, 5 yaw, 6 age_g_ms, 7 age_r_ms,
 * 8 bias_gz, 9 r, 10 r_th, 11 slip_flag, 12 slip_on_count, 13 src_switch_count,
 * 14 src_switch_dwz, 15 wz_cmd, 16 head_on, 17 psi_ref, 18 psi_imu, 19 e_psi,
 * 20 psi_reset_count, 21 head_toggle_count, 22 bias_updates, 23 slip_eval */
#define FWSIM_IMU_COLS 24
FWSIM_API int fwsim_imu_cols(void) { return FWSIM_IMU_COLS; }
FWSIM_API void fwsim_imu_last(double out[FWSIM_IMU_COLS]) {
    const imu_fusion_out_t *m = &last_vc.imu;
    const heading_out_t *h = &last_vc.head;
    double v[FWSIM_IMU_COLS] = {
        m->g_valid, m->r_valid, m->wz_src, m->wz_meas, m->gz, m->yaw, m->age_g_ms, m->age_r_ms,
        m->bias_gz, m->r, m->r_th, m->slip_flag, m->slip_on_count, m->src_switch_count,
        m->src_switch_dwz, last_vc.wz_cmd, h->on, h->psi_ref, h->psi_imu, h->e_psi,
        h->psi_reset_count, h->toggle_count, m->bias_updates, m->slip_eval };
    memcpy(out, v, sizeof(v));
}

/* axis 0 vx, 1 vy, 2 wz */
FWSIM_API void fwsim_set_pid(int axis, float kp, float ki, float kd, float int_limit, float out_limit) {
    pid_controller_t *c = vel_ctrl_pid();
    pid_t *p = axis == 0 ? &c->vx : axis == 1 ? &c->vy : &c->wz;
    p->kp = kp; p->ki = ki; p->kd = kd;
    p->integral_limit = int_limit; p->output_limit = out_limit;
}

FWSIM_API void fwsim_get_pid(int axis, float out[5]) {
    pid_controller_t *c = vel_ctrl_pid();
    pid_t *p = axis == 0 ? &c->vx : axis == 1 ? &c->vy : &c->wz;
    out[0] = p->kp; out[1] = p->ki; out[2] = p->kd; out[3] = p->integral_limit; out[4] = p->output_limit;
}

static adrc_axis_t *adrc_axis(int axis) {
    adrc_controller_t *c = vel_ctrl_adrc();
    return axis == 0 ? &c->vx : axis == 1 ? &c->vy : &c->wz;
}

FWSIM_API void fwsim_set_adrc(int axis, float omega_c, float omega_o, float b0, float out_limit) {
    adrc_set_params(adrc_axis(axis), omega_c, omega_o, b0, out_limit);
}

/* out: omega_c (= kp), omega_o (= beta1 / 2), b0, out_limit */
FWSIM_API void fwsim_get_adrc(int axis, float out[4]) {
    adrc_axis_t *a = adrc_axis(axis);
    out[0] = a->kp; out[1] = a->eso.beta1 / 2.0f; out[2] = a->eso.b0; out[3] = a->output_limit;
}

/* One 10 ms tick at time t_us.
 *   counts[4]  in:  total quadrature counts the H7 has generated (FL, FR, RL, RR)
 *   sp[3]      in:  body setpoint vx, vy, wz
 *   duty[4]    out: rpwm - lpwm per wheel, as H7 PWMCapture_GetDuty() reads it
 *   actual[3]  out: body velocity from the encoders (what the firmware sends back)
 *   pwm[4]     out: command given to motor_set()
 *   rpm[4]     out: encoder_get_rpm()
 * Returns the EN pin level (1 = driver enabled). */
FWSIM_API int fwsim_tick(int64_t t_us, const int32_t counts[4], const float sp[3],
                         double duty[4], float actual[3], int pwm[4], float rpm[4]) {
    fw_time_us = t_us;
    for (int i = 0; i < 4; i++) { fw_pcnt_add(i, counts[i] - count_in[i]); count_in[i] = counts[i]; }

    /* encoder_task */
    float enc_rpm[4];
    for (int i = 0; i < ENC_COUNT; i++) {
        enc_rpm[i] = encoder_get_rpm((enc_id_t)i);
        enc_count[i] = encoder_get_count((enc_id_t)i);
    }

    /* control_task */
    const float dt = 0.01f;
    float dt_actual = (t_us - t_prev) * 1e-6f;
    t_prev = t_us;
    if (dt_actual <= 0.0f || dt_actual > 0.1f) dt_actual = dt;

    int on = (int)fw_gpio_level(GPIO_EN);   /* motor_enable() state = g_motor_enabled */
    imu_data_t imu_copy = imu;              /* imu_get() */
    vel_ctrl_in_t in = {
        .sp = { .vx = sp[0], .vy = sp[1], .wz = sp[2] },
        .imu = imu_on ? &imu_copy : NULL,
        .now_us = t_us,
        .dt = dt_actual,
        .motor_on = on,
        .run_active = on && run_active,
        .wheels_lifted = wheels_lifted,
    };
    for (int i = 0; i < 4; i++) { in.enc_rpm[i] = enc_rpm[i]; in.enc_count[i] = enc_count[i]; }
    vel_ctrl_out_t vc = vel_ctrl_step(&in);
    if (on) {
        /* main.c: motor_set() then vel_ctrl_applied() with the PWM applied (no obstacle cutoff here) */
        for (int i = 0; i < 4; i++) motor_set((motor_id_t)i, vc.pwm[i]);
        vel_ctrl_applied(vc.pwm, &vc);
    } else {
        motor_stop_all();
    }
    last_vc = vc;

    for (int i = 0; i < 4; i++) {
        duty[i] = fw_gpio_duty(rpwm_gpio[i]) - fw_gpio_duty(lpwm_gpio[i]);
        pwm[i]  = vc.pwm[i];
        rpm[i]  = enc_rpm[i];
    }
    actual[0] = vc.actual.vx; actual[1] = vc.actual.vy; actual[2] = vc.actual.wz;
    return (int)fw_gpio_level(GPIO_EN);
}
