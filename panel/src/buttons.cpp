#include "buttons.h"

#include <Arduino.h>

#include "pins.h"

namespace {

struct Btn {
    uint8_t  pin;
    bool     stable;        /* debounced level: true = pressed */
    bool     raw;
    uint32_t t_raw;         /* when raw last changed */
    uint32_t t_press;       /* when stable became pressed */
    uint32_t t_repeat;      /* next NAV_BACK repeat */
    bool     long_done;     /* NAV: the hold already gave NAV_BACK */
};

Btn g_btn[4] = {
    {PIN_BTN_NAV, false, false, 0, 0, 0, false},
    {PIN_BTN_SELECT, false, false, 0, 0, 0, false},
    {PIN_BTN_START, false, false, 0, 0, 0, false},
    {PIN_BTN_STOP, false, false, 0, 0, 0, false},
};

}  // namespace

void buttons_begin(void)
{
    for (auto &b : g_btn)
        pinMode(b.pin, INPUT_PULLUP);
}

uint8_t buttons_poll(uint32_t now)
{
    uint8_t ev = 0;
    for (int i = 0; i < 4; i++) {
        Btn &b = g_btn[i];
        bool raw = digitalRead(b.pin) == LOW;
        if (raw != b.raw) {
            b.raw = raw;
            b.t_raw = now;
        }
        if (raw != b.stable && (uint32_t)(now - b.t_raw) >= BTN_DEBOUNCE_MS) {
            b.stable = raw;
            if (raw) {
                b.t_press = now;
                b.long_done = false;
                if (i == 1) ev |= BTN_EV_SELECT;
                if (i == 2) ev |= BTN_EV_START;
                if (i == 3) ev |= BTN_EV_STOP;
            } else if (i == 0 && !b.long_done) {
                ev |= BTN_EV_NAV;                       /* short NAV: on release */
            }
        }
        if (i == 0 && b.stable) {
            if (!b.long_done && (uint32_t)(now - b.t_press) >= NAV_HOLD_MS) {
                b.long_done = true;
                b.t_repeat = now + NAV_REPEAT_MS;
                ev |= BTN_EV_NAV_BACK;
            } else if (b.long_done && (int32_t)(now - b.t_repeat) >= 0) {
                b.t_repeat = now + NAV_REPEAT_MS;
                ev |= BTN_EV_NAV_BACK;
            }
        }
    }
    return ev;
}
