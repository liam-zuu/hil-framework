/* fwsim shim: esp_timer_get_time() returns the simulated time set by fw_sim.c. */
#pragma once
#include <stdint.h>
int64_t esp_timer_get_time(void);
