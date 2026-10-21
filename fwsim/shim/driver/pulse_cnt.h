/* fwsim shim: PCNT units return the simulated 16-bit hardware count (fw_shim.c). */
#pragma once
#include <stdint.h>
#include <stdbool.h>
typedef int esp_err_t;
typedef struct fw_pcnt_unit *pcnt_unit_handle_t;
typedef struct fw_pcnt_chan *pcnt_channel_handle_t;
typedef struct { int low_limit; int high_limit; int intr_priority; struct { uint32_t accum_count : 1; } flags; } pcnt_unit_config_t;
typedef struct { int edge_gpio_num; int level_gpio_num; } pcnt_chan_config_t;
typedef struct { uint32_t max_glitch_ns; } pcnt_glitch_filter_config_t;
typedef enum { PCNT_CHANNEL_EDGE_ACTION_HOLD = 0, PCNT_CHANNEL_EDGE_ACTION_INCREASE, PCNT_CHANNEL_EDGE_ACTION_DECREASE } pcnt_channel_edge_action_t;
typedef enum { PCNT_CHANNEL_LEVEL_ACTION_KEEP = 0, PCNT_CHANNEL_LEVEL_ACTION_INVERSE, PCNT_CHANNEL_LEVEL_ACTION_HOLD } pcnt_channel_level_action_t;
esp_err_t pcnt_new_unit(const pcnt_unit_config_t *cfg, pcnt_unit_handle_t *ret);
esp_err_t pcnt_new_channel(pcnt_unit_handle_t unit, const pcnt_chan_config_t *cfg, pcnt_channel_handle_t *ret);
esp_err_t pcnt_channel_set_edge_action(pcnt_channel_handle_t ch, pcnt_channel_edge_action_t pos, pcnt_channel_edge_action_t neg);
esp_err_t pcnt_channel_set_level_action(pcnt_channel_handle_t ch, pcnt_channel_level_action_t high, pcnt_channel_level_action_t low);
esp_err_t pcnt_unit_set_glitch_filter(pcnt_unit_handle_t unit, const pcnt_glitch_filter_config_t *cfg);
esp_err_t pcnt_unit_enable(pcnt_unit_handle_t unit);
esp_err_t pcnt_unit_clear_count(pcnt_unit_handle_t unit);
esp_err_t pcnt_unit_start(pcnt_unit_handle_t unit);
esp_err_t pcnt_unit_get_count(pcnt_unit_handle_t unit, int *value);
esp_err_t pcnt_unit_add_watch_point(pcnt_unit_handle_t unit, int watch_point);
