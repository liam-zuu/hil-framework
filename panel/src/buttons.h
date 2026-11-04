/* buttons.h — 4 panel buttons, debounced; NAV held = step back with repeat (đề xuất panel §5). */
#pragma once
#include <stdint.h>

enum {
    BTN_EV_NAV      = 1 << 0,   /* NAV short press (on release): next item */
    BTN_EV_NAV_BACK = 1 << 1,   /* NAV held >= NAV_HOLD_MS, then every NAV_REPEAT_MS: previous item */
    BTN_EV_SELECT   = 1 << 2,
    BTN_EV_START    = 1 << 3,
    BTN_EV_STOP     = 1 << 4,
};

#define BTN_DEBOUNCE_MS   30u   /* [ĐỀ XUẤT] */
#define NAV_HOLD_MS      600u   /* [ĐỀ XUẤT] */
#define NAV_REPEAT_MS    250u   /* [ĐỀ XUẤT] */

void buttons_begin(void);
/* call often; returns the BTN_EV_* that happened since the last call */
uint8_t buttons_poll(uint32_t now_ms);
