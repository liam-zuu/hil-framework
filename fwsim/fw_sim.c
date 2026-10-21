/* fw_sim.c: PC entry points around the ESP32 firmware (AGV_MECANUM).
 *
 * The DLL is built from the firmware's own encoder.c, motor.c, kinematics.c,
 * pid.c, adrc.c, vel_ctrl.c (unchanged) plus the shim in shim/. Only the scheduling
 * glue of main.c is repeated here, because main.c needs FreeRTOS:
 *   encoder_task: encoder_get_rpm() and encoder_get_count() for every wheel,
 *   control_task: dt_actual from esp_timer, vel_ctrl_step(), motor_set() x 4.
 * Both tasks wake on the same 10 ms tick; encoder_task has the higher priority
 * so it runs first (main.c). Ultrasonic interlock, RC and mode logic are not
 * modelled: the setpoint goes straight in, as in MODE_AUTO with no obstacle.
 */
#include "shim/fw_shim.h"
#include "esp_timer.h"
#include "motor.h"
#include "encoder.h"
#include "vel_ctrl.h"

#define FWSIM_API __declspec(dllexport)
#define FWSIM_ABI 2

static const int rpwm_gpio[4] = { FL_RPWM, FR_RPWM, RL_RPWM, RR_RPWM };
static const int lpwm_gpio[4] = { FL_LPWM, FR_LPWM, RL_LPWM, RR_LPWM };
static int64_t t_prev;
static int32_t count_in[4];

FWSIM_API int fwsim_abi(void) { return FWSIM_ABI; }
FWSIM_API int  fwsim_ctrl_mode(void) { return vel_ctrl_mode(); }
FWSIM_API void fwsim_set_ctrl_mode(int mode) { vel_ctrl_set_mode(mode); }

/* Boot: what app_main + control_task do before the first control tick. */
FWSIM_API void fwsim_reset(int pcnt_mode) {
    fw_shim_reset();
    fw_pcnt_mode = pcnt_mode;
    for (int i = 0; i < 4; i++) count_in[i] = 0;
    motor_init();
    encoder_init();
    vel_ctrl_init();
    motor_enable(false);
    t_prev = esp_timer_get_time();
    motor_enable(true);            /* HIL_BYPASS_MOTOR_ENABLE: swa_on = true on the first tick */
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
        (void)encoder_get_count((enc_id_t)i);
    }

    /* control_task */
    const float dt = 0.01f;
    float dt_actual = (t_us - t_prev) * 1e-6f;
    t_prev = t_us;
    if (dt_actual <= 0.0f || dt_actual > 0.1f) dt_actual = dt;

    agv_velocity_t vel = { .vx = sp[0], .vy = sp[1], .wz = sp[2] };
    vel_ctrl_out_t vc = vel_ctrl_step(vel, enc_rpm, dt_actual);
    for (int i = 0; i < 4; i++) motor_set((motor_id_t)i, vc.pwm[i]);

    for (int i = 0; i < 4; i++) {
        duty[i] = fw_gpio_duty(rpwm_gpio[i]) - fw_gpio_duty(lpwm_gpio[i]);
        pwm[i]  = vc.pwm[i];
        rpm[i]  = enc_rpm[i];
    }
    actual[0] = vc.actual.vx; actual[1] = vc.actual.vy; actual[2] = vc.actual.wz;
    return (int)fw_gpio_level(GPIO_EN);
}
