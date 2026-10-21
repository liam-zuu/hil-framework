/* fwsim shim implementations: the ESP-IDF calls used by motor.c and encoder.c,
 * backed by plain memory that fw_sim.c reads and writes. */
#include "esp_timer.h"
#include "driver/gpio.h"
#include "driver/pulse_cnt.h"
#include "driver/mcpwm_prelude.h"
#include "fw_shim.h"
#include <string.h>

/* ── time ─────────────────────────────────────────────────────────────── */
int64_t fw_time_us = 0;
int64_t esp_timer_get_time(void) { return fw_time_us; }

/* ── GPIO ─────────────────────────────────────────────────────────────── */
static uint32_t gpio_level[FW_GPIO_MAX];
esp_err_t gpio_config(const gpio_config_t *cfg) { (void)cfg; return 0; }
esp_err_t gpio_set_level(int n, uint32_t level) { if (n >= 0 && n < FW_GPIO_MAX) gpio_level[n] = level; return 0; }
uint32_t fw_gpio_level(int n) { return (n >= 0 && n < FW_GPIO_MAX) ? gpio_level[n] : 0; }

/* ── PCNT ─────────────────────────────────────────────────────────────── */
struct fw_pcnt_unit { int low, high, raw; };
struct fw_pcnt_chan { int dummy; };
static struct fw_pcnt_unit units[FW_PCNT_MAX];
static struct fw_pcnt_chan chans[2 * FW_PCNT_MAX];
static int n_units, n_chans;
int fw_pcnt_mode = FW_PCNT_IDF;

esp_err_t pcnt_new_unit(const pcnt_unit_config_t *cfg, pcnt_unit_handle_t *ret) {
    if (n_units >= FW_PCNT_MAX) return -1;
    struct fw_pcnt_unit *u = &units[n_units++];
    u->low = cfg->low_limit; u->high = cfg->high_limit; u->raw = 0;
    *ret = u; return 0;
}
esp_err_t pcnt_new_channel(pcnt_unit_handle_t u, const pcnt_chan_config_t *c, pcnt_channel_handle_t *ret) {
    (void)u; (void)c; *ret = &chans[n_chans++ % (2 * FW_PCNT_MAX)]; return 0;
}
esp_err_t pcnt_channel_set_edge_action(pcnt_channel_handle_t c, pcnt_channel_edge_action_t a, pcnt_channel_edge_action_t b) { (void)c; (void)a; (void)b; return 0; }
esp_err_t pcnt_channel_set_level_action(pcnt_channel_handle_t c, pcnt_channel_level_action_t a, pcnt_channel_level_action_t b) { (void)c; (void)a; (void)b; return 0; }
esp_err_t pcnt_unit_set_glitch_filter(pcnt_unit_handle_t u, const pcnt_glitch_filter_config_t *c) { (void)u; (void)c; return 0; }
esp_err_t pcnt_unit_enable(pcnt_unit_handle_t u) { (void)u; return 0; }
esp_err_t pcnt_unit_clear_count(pcnt_unit_handle_t u) { u->raw = 0; return 0; }
esp_err_t pcnt_unit_start(pcnt_unit_handle_t u) { (void)u; return 0; }
esp_err_t pcnt_unit_get_count(pcnt_unit_handle_t u, int *v) { *v = u->raw; return 0; }

/* Advance unit i by delta counts. FW_PCNT_IDF: the hardware counter is reset to
 * zero when it reaches high_limit or low_limit (ESP-IDF PCNT behaviour without
 * the accum_count flag). FW_PCNT_WRAP16: plain int16 wrap-around. */
void fw_pcnt_add(int i, int delta) {
    if (i < 0 || i >= n_units) return;
    struct fw_pcnt_unit *u = &units[i];
    if (fw_pcnt_mode == FW_PCNT_WRAP16) {
        u->raw = (int16_t)(uint16_t)(u->raw + delta);
        return;
    }
    u->raw += delta;
    while (u->raw >= u->high) u->raw -= u->high;
    while (u->raw <= u->low)  u->raw -= u->low;
}

/* ── MCPWM ────────────────────────────────────────────────────────────── */
struct fw_mcpwm_timer { uint32_t period; };
struct fw_mcpwm_oper  { struct fw_mcpwm_timer *timer; };
struct fw_mcpwm_cmpr  { struct fw_mcpwm_oper *oper; uint32_t value; };
struct fw_mcpwm_gen   {
    struct fw_mcpwm_oper *oper; int gpio;
    mcpwm_generator_action_t on_empty;
    struct fw_mcpwm_cmpr *cmpr; mcpwm_generator_action_t on_cmpr;
};
static struct fw_mcpwm_timer timers[FW_MCPWM_MAX];
static struct fw_mcpwm_oper  opers[FW_MCPWM_MAX];
static struct fw_mcpwm_cmpr  cmprs[2 * FW_MCPWM_MAX];
static struct fw_mcpwm_gen   gens[2 * FW_MCPWM_MAX];
static int n_timers, n_opers, n_cmprs, n_gens;

esp_err_t mcpwm_new_timer(const mcpwm_timer_config_t *cfg, mcpwm_timer_handle_t *ret) {
    if (n_timers >= FW_MCPWM_MAX) return -1;
    timers[n_timers].period = cfg->period_ticks; *ret = &timers[n_timers++]; return 0;
}
esp_err_t mcpwm_new_operator(const mcpwm_operator_config_t *cfg, mcpwm_oper_handle_t *ret) {
    (void)cfg; if (n_opers >= FW_MCPWM_MAX) return -1;
    opers[n_opers].timer = NULL; *ret = &opers[n_opers++]; return 0;
}
esp_err_t mcpwm_operator_connect_timer(mcpwm_oper_handle_t o, mcpwm_timer_handle_t t) { o->timer = t; return 0; }
esp_err_t mcpwm_new_comparator(mcpwm_oper_handle_t o, const mcpwm_comparator_config_t *cfg, mcpwm_cmpr_handle_t *ret) {
    (void)cfg; if (n_cmprs >= 2 * FW_MCPWM_MAX) return -1;
    cmprs[n_cmprs].oper = o; cmprs[n_cmprs].value = 0; *ret = &cmprs[n_cmprs++]; return 0;
}
esp_err_t mcpwm_new_generator(mcpwm_oper_handle_t o, const mcpwm_generator_config_t *cfg, mcpwm_gen_handle_t *ret) {
    if (n_gens >= 2 * FW_MCPWM_MAX) return -1;
    struct fw_mcpwm_gen *g = &gens[n_gens++];
    memset(g, 0, sizeof *g); g->oper = o; g->gpio = cfg->gen_gpio_num; *ret = g; return 0;
}
esp_err_t mcpwm_generator_set_action_on_timer_event(mcpwm_gen_handle_t g, mcpwm_gen_timer_event_action_t ev) {
    if (ev.event == MCPWM_TIMER_EVENT_EMPTY) g->on_empty = ev.action;
    return 0;
}
esp_err_t mcpwm_generator_set_action_on_compare_event(mcpwm_gen_handle_t g, mcpwm_gen_compare_event_action_t ev) {
    g->cmpr = ev.comparator; g->on_cmpr = ev.action; return 0;
}
esp_err_t mcpwm_comparator_set_compare_value(mcpwm_cmpr_handle_t c, uint32_t v) { c->value = v; return 0; }
esp_err_t mcpwm_timer_enable(mcpwm_timer_handle_t t) { (void)t; return 0; }
esp_err_t mcpwm_timer_start_stop(mcpwm_timer_handle_t t, mcpwm_timer_start_stop_cmd_t c) { (void)t; (void)c; return 0; }

/* Fraction of the PWM period the GPIO is HIGH (count-up timer). -1: GPIO not
 * driven by a generator, -2: action pattern this shim does not model. */
double fw_gpio_duty(int gpio) {
    for (int i = 0; i < n_gens; i++) {
        struct fw_mcpwm_gen *g = &gens[i];
        if (g->gpio != gpio) continue;
        if (!g->cmpr || !g->oper || !g->oper->timer) return -2.0;
        double period = g->oper->timer->period;
        double c = g->cmpr->value < period ? g->cmpr->value : period;
        if (g->on_empty == MCPWM_GEN_ACTION_HIGH && g->on_cmpr == MCPWM_GEN_ACTION_LOW) return c / period;
        if (g->on_empty == MCPWM_GEN_ACTION_LOW && g->on_cmpr == MCPWM_GEN_ACTION_HIGH) return 1.0 - c / period;
        return -2.0;
    }
    return -1.0;
}

void fw_shim_reset(void) {
    fw_time_us = 0;
    memset(gpio_level, 0, sizeof gpio_level);
    n_units = n_chans = n_timers = n_opers = n_cmprs = n_gens = 0;
}
