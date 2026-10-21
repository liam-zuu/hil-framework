/* fwsim shim: MCPWM objects record the compare value and the generator actions,
 * so fw_shim.c can compute the duty on every GPIO exactly as motor.c programs it. */
#pragma once
#include <stdint.h>
#include <stdbool.h>
typedef int esp_err_t;
typedef struct fw_mcpwm_timer *mcpwm_timer_handle_t;
typedef struct fw_mcpwm_oper  *mcpwm_oper_handle_t;
typedef struct fw_mcpwm_cmpr  *mcpwm_cmpr_handle_t;
typedef struct fw_mcpwm_gen   *mcpwm_gen_handle_t;
typedef enum { MCPWM_TIMER_CLK_SRC_DEFAULT = 0 } mcpwm_timer_clock_source_t;
typedef enum { MCPWM_TIMER_COUNT_MODE_PAUSE = 0, MCPWM_TIMER_COUNT_MODE_UP } mcpwm_timer_count_mode_t;
typedef enum { MCPWM_TIMER_DIRECTION_UP = 0, MCPWM_TIMER_DIRECTION_DOWN } mcpwm_timer_direction_t;
typedef enum { MCPWM_TIMER_EVENT_EMPTY = 0, MCPWM_TIMER_EVENT_FULL } mcpwm_timer_event_t;
typedef enum { MCPWM_GEN_ACTION_KEEP = 0, MCPWM_GEN_ACTION_LOW, MCPWM_GEN_ACTION_HIGH, MCPWM_GEN_ACTION_TOGGLE } mcpwm_generator_action_t;
typedef enum { MCPWM_TIMER_START_NO_STOP = 0 } mcpwm_timer_start_stop_cmd_t;
typedef struct {
    int group_id; mcpwm_timer_clock_source_t clk_src; uint32_t resolution_hz;
    mcpwm_timer_count_mode_t count_mode; uint32_t period_ticks;
} mcpwm_timer_config_t;
typedef struct { int group_id; } mcpwm_operator_config_t;
typedef struct { struct { uint32_t update_cmp_on_tez : 1; } flags; } mcpwm_comparator_config_t;
typedef struct { int gen_gpio_num; } mcpwm_generator_config_t;
typedef struct { mcpwm_timer_direction_t direction; mcpwm_timer_event_t event; mcpwm_generator_action_t action; } mcpwm_gen_timer_event_action_t;
typedef struct { mcpwm_timer_direction_t direction; mcpwm_cmpr_handle_t comparator; mcpwm_generator_action_t action; } mcpwm_gen_compare_event_action_t;
#define MCPWM_GEN_TIMER_EVENT_ACTION(dir, ev, act) ((mcpwm_gen_timer_event_action_t){ .direction = (dir), .event = (ev), .action = (act) })
#define MCPWM_GEN_COMPARE_EVENT_ACTION(dir, cmp, act) ((mcpwm_gen_compare_event_action_t){ .direction = (dir), .comparator = (cmp), .action = (act) })
esp_err_t mcpwm_new_timer(const mcpwm_timer_config_t *cfg, mcpwm_timer_handle_t *ret);
esp_err_t mcpwm_new_operator(const mcpwm_operator_config_t *cfg, mcpwm_oper_handle_t *ret);
esp_err_t mcpwm_operator_connect_timer(mcpwm_oper_handle_t oper, mcpwm_timer_handle_t timer);
esp_err_t mcpwm_new_comparator(mcpwm_oper_handle_t oper, const mcpwm_comparator_config_t *cfg, mcpwm_cmpr_handle_t *ret);
esp_err_t mcpwm_new_generator(mcpwm_oper_handle_t oper, const mcpwm_generator_config_t *cfg, mcpwm_gen_handle_t *ret);
esp_err_t mcpwm_generator_set_action_on_timer_event(mcpwm_gen_handle_t gen, mcpwm_gen_timer_event_action_t ev);
esp_err_t mcpwm_generator_set_action_on_compare_event(mcpwm_gen_handle_t gen, mcpwm_gen_compare_event_action_t ev);
esp_err_t mcpwm_comparator_set_compare_value(mcpwm_cmpr_handle_t cmpr, uint32_t value);
esp_err_t mcpwm_timer_enable(mcpwm_timer_handle_t timer);
esp_err_t mcpwm_timer_start_stop(mcpwm_timer_handle_t timer, mcpwm_timer_start_stop_cmd_t cmd);
