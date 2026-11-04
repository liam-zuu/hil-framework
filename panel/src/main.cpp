/*
 * Panel M2 of the HIL box (QĐ 17, QĐ 22): ESP32-S3 + ST7796 + 4 buttons.
 * The panel only draws and forwards buttons; the RPi5 daemon owns the run state.
 * Protocol v6 (lib/panel_proto, mirror of rpi5/panel_proto.py). Controller choice is way (c):
 * the panel only shows the controller running (INFO), it is changed by flashing an env.
 *
 * Link: env panel_s3 -> UART1 to the RPi5 (pins.h); env panel_s3_usb -> the USB port
 * (bench, panel/tools/fake_pi.py on the PC). Nothing else is written to the link port.
 */
#include <Arduino.h>

#include "buttons.h"
#include "panel_proto.h"
#include "pins.h"
#include "ui.h"

#ifndef PANEL_LINK_USB
#define PANEL_LINK_USB 0
#endif

#if PANEL_LINK_USB
#define LINK Serial          /* USB CDC (ARDUINO_USB_CDC_ON_BOOT=1) */
#else
#define LINK Serial1
#endif

#define REDRAW_MIN_MS   200u     /* at most 5 full redraws per second (push ~60 ms at 40 MHz) */
#define REDRAW_MAX_MS  1000u

static pp_client_t g_c;
static pp_parser_t g_parser;
static PanelView   g_v;

static uint16_t nonce_cb(void *)
{
    return (uint16_t)esp_random();
}

static void send_cb(void *, const uint8_t *frame, size_t len)
{
    LINK.write(frame, len);
}

static void frame_cb(void *, const uint8_t *frame, size_t len)
{
    pp_client_on_frame(&g_c, frame, len, millis());
}

static const char *cmd_name(uint8_t cmd)
{
    switch (cmd) {
    case PP_CMD_SELECT: return "SELECT";
    case PP_CMD_START:  return "START";
    case PP_CMD_STOP:   return "STOP";
    default:            return "CMD";
    }
}

static const char *res_name(uint8_t r)
{
    switch (r) {
    case PP_RES_OK:          return "OK";
    case PP_RES_BUSY:        return "BUSY";
    case PP_RES_BAD_PROFILE: return "BAD TEST";
    case PP_RES_NOT_IDLE:    return "nothing to stop";
    case PP_RES_UNKNOWN:     return "UNKNOWN";
    case PP_RES_FAILED:      return "FAILED (effect unknown)";
    default:                 return "?";
    }
}

static void drain_events(void)
{
    pp_event_t ev;
    while (pp_client_pop_event(&g_c, &ev)) {
        if (ev.what == PP_EV_DONE)
            snprintf(g_v.msg, sizeof g_v.msg, "%s: %s", cmd_name(ev.cmd), res_name(ev.result));
        else if (ev.what == PP_EV_UNVERIFIED)
            snprintf(g_v.msg, sizeof g_v.msg, "%s: not verified (may or may not have run)", cmd_name(ev.cmd));
        else
            snprintf(g_v.msg, sizeof g_v.msg, "%s refused: busy or no link, press again", cmd_name(ev.cmd));
    }
}

static void move_cursor(int d)
{
    int n = g_c.has_profile_count ? g_c.profile_count : 0;
    if (n <= 0)
        return;
    g_v.cursor = (uint8_t)(((int)g_v.cursor + d + n) % n);
}

static void start(uint8_t profile, uint32_t now)
{
    if (!ui_ready(g_v)) {
        snprintf(g_v.msg, sizeof g_v.msg, "NOT READY: START not sent");
        return;
    }
    pp_client_press(&g_c, PP_CMD_START, profile, now);
}

static void on_buttons(uint8_t ev, uint32_t now)
{
    switch (ui_screen(g_v)) {
    case SCR_NOLINK:
        if (ev & BTN_EV_STOP)
            pp_client_press(&g_c, PP_CMD_STOP, 0, now);     /* resent until acknowledged */
        break;
    case SCR_HOME:
        if (ev & BTN_EV_NAV) move_cursor(+1);
        if (ev & BTN_EV_NAV_BACK) move_cursor(-1);
        if (ev & BTN_EV_SELECT) pp_client_press(&g_c, PP_CMD_SELECT, g_v.cursor, now);
        if (ev & BTN_EV_START) start(g_v.cursor, now);
        break;
    case SCR_RUN:
        if (ev & BTN_EV_STOP) pp_client_press(&g_c, PP_CMD_STOP, 0, now);
        break;
    case SCR_RESULT:
        if (ev & (BTN_EV_STOP | BTN_EV_NAV | BTN_EV_NAV_BACK | BTN_EV_SELECT))
            g_v.result_dismissed = true;                    /* back to HOME, nothing sent */
        else if (ev & BTN_EV_START)
            start(g_c.st.profile, now);
        break;
    case SCR_FAULT:
        if (ev & BTN_EV_STOP) pp_client_press(&g_c, PP_CMD_STOP, 0, now);    /* acknowledge */
        break;
    }
}

static void leds(uint32_t now)
{
    Screen s = ui_screen(g_v);
    digitalWrite(PIN_LED_START, (s == SCR_HOME || s == SCR_RESULT) && ui_ready(g_v) ? HIGH : LOW);
    bool stop_on = s == SCR_RUN || s == SCR_FAULT || g_c.stop.active;
    digitalWrite(PIN_LED_STOP, stop_on ? HIGH : LOW);
    (void)now;
}

void setup()
{
#if PANEL_LINK_USB
    LINK.begin(PI_BAUD);
#else
    LINK.setRxBufferSize(2048);
    LINK.begin(PI_BAUD, SERIAL_8N1, PIN_PI_RX, PIN_PI_TX);
#endif
    pinMode(PIN_LED_START, OUTPUT);
    pinMode(PIN_LED_STOP, OUTPUT);
    buttons_begin();
    ui_begin();
    pp_parser_init(&g_parser, PP_ACCEPT_PANEL, PP_PARSER_STALE_MS);
    pp_client_init(&g_c, nonce_cb, nullptr, send_cb, nullptr);
    g_v.c = &g_c;
    snprintf(g_v.msg, sizeof g_v.msg, "panel M2, protocol v%d%s", PP_VERSION, PANEL_LINK_USB ? ", USB link" : "");
}

void loop()
{
    static uint32_t t_draw = 0, sig_drawn = 0;
    uint32_t now = millis();

    uint8_t buf[128];
    int n = LINK.available();
    if (n > 0) {
        n = LINK.readBytes(buf, n > (int)sizeof buf ? (int)sizeof buf : n);
        pp_parser_feed(&g_parser, buf, (size_t)n, now, frame_cb, nullptr);
    }
    pp_client_tick(&g_c, now);
    drain_events();

    g_v.now = now;
    if (g_c.has_st && PP_ST_IS_ACTIVE(g_c.st.run_state))
        g_v.result_dismissed = false;               /* a new run: its RESULT is shown */
    uint8_t ev = buttons_poll(now);
    if (ev) {
        on_buttons(ev, now);
        drain_events();
    }
    leds(now);

    uint32_t sig = ui_signature(g_v);
    uint32_t since = now - t_draw;
    if ((sig != sig_drawn && since >= REDRAW_MIN_MS) || since >= REDRAW_MAX_MS) {
        ui_draw(g_v);
        sig_drawn = sig;
        t_draw = millis();
    }
    delay(1);
}
