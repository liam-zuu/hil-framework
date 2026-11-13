/* buzzer.h - drives the passive, low-level-trigger buzzer on PIN_BUZZER (Liam 05/11):
 * square wave from LEDC while a note plays, pin held HIGH (module off) when silent. */
#pragma once
#include <stdint.h>

void buzzer_begin(void);
void buzzer_out(uint16_t hz);      /* 0 = silent; only writes when the value changes */
