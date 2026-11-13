/* buttons.h — 4 panel buttons, debounced; NAV held = step back with repeat (đề xuất panel §5). */
#pragma once
#include <stdint.h>

#include "panel_app.h"

/* NAV short press (on release): next item; NAV held >= NAV_HOLD_MS, then every NAV_REPEAT_MS:
 * previous item. Same bits as APP_EV_* (panel_app.h). */
#define BTN_EV_NAV      APP_EV_NAV
#define BTN_EV_NAV_BACK APP_EV_NAV_BACK
#define BTN_EV_SELECT   APP_EV_SELECT
#define BTN_EV_START    APP_EV_START
#define BTN_EV_STOP     APP_EV_STOP

#define BTN_DEBOUNCE_MS   30u   /* [ĐỀ XUẤT] */
#define NAV_HOLD_MS      600u   /* [ĐỀ XUẤT] */
#define NAV_REPEAT_MS    250u   /* [ĐỀ XUẤT] */

void buttons_begin(void);
/* call often; returns the BTN_EV_* that happened since the last call */
uint8_t buttons_poll(uint32_t now_ms);
