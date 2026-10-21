/* fwsim: access to the shim state from fw_sim.c */
#pragma once
#include <stdint.h>
#define FW_GPIO_MAX   64
#define FW_PCNT_MAX   4
#define FW_MCPWM_MAX  4
#define FW_PCNT_IDF    0
#define FW_PCNT_WRAP16 1
extern int64_t fw_time_us;
extern int     fw_pcnt_mode;
uint32_t fw_gpio_level(int gpio);
double   fw_gpio_duty(int gpio);
void     fw_pcnt_add(int unit, int delta);
void     fw_shim_reset(void);
