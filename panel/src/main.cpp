/*
 * Panel M2 of the HIL box (QĐ 17, QĐ 22): ESP32-S3 + ST7796 + 4 buttons.
 * The panel only draws and forwards buttons; the RPi5 daemon owns the run state.
 * Protocol v6 (lib/panel_proto, mirror of rpi5/panel_proto.py); buttons -> commands and the
 * screen choice in lib/panel_app (plain C, tested on the PC). Controller choice is way (c):
 * the panel only shows the controller running (INFO); it is changed by flashing an env.
 *
 * Two tasks (Codex round 1 P5, Liam approved 04/11):
 *  - link task, every LINK_PERIOD_MS: UART bytes -> parser -> client, client tick, buttons,
 *    app, LEDs. One now per pass (P4). It never draws, so a ~60 ms sprite push cannot cut a
 *    frame (parser silence timeout) or swallow a short press.
 *  - Arduino loop (drawing): copies client + app under the mutex, draws from the copy, then
 *    reports what is on the glass (app_drawn, P2).
 *
 * Link: env panel_s3 -> UART1 to the RPi5 (pins.h); env panel_s3_bench -> UART0, the "UART"
 * USB-C port of the WeAct board (CH343P), used for flashing too (bench, panel/tools/fake_pi.py
 * on the PC). Nothing else is written to the link port; the ROM boot text on UART0 is dropped
 * by the fake Pi's frame parser.
 */
#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

#include "buttons.h"
#include "panel_app.h"
#include "panel_proto.h"
#include "pins.h"
#include "ui.h"

#ifndef PANEL_LINK_BENCH
#define PANEL_LINK_BENCH 0
#endif

#if PANEL_LINK_BENCH
#define LINK Serial0         /* UART0 = GPIO43/44 = CH343P = "UART" port */
#else
#define LINK Serial1
#endif

#define LINK_PERIOD_MS     2u     /* link task pass */
#define LINK_RX_BUFFER  2048      /* both envs (P6) */
#define REDRAW_MIN_MS    200u     /* at most 5 full redraws per second (push ~60 ms at 40 MHz) */
#define REDRAW_MAX_MS   1000u

static pp_client_t       g_c;
static pp_parser_t       g_parser;
static app_t             g_app;
static SemaphoreHandle_t g_mx;

static uint16_t nonce_cb(void *)
{
    return (uint16_t)esp_random();
}

static void send_cb(void *, const uint8_t *frame, size_t len)
{
    LINK.write(frame, len);
}

static void frame_cb(void *ctx, const uint8_t *frame, size_t len)
{
    pp_client_on_frame(&g_c, frame, len, *(const uint32_t *)ctx);     /* the pass's now (P4) */
}

static void leds(uint32_t now)
{
    app_screen_t s = app_screen(&g_app, now);
    bool start_on = (s == APP_SCR_HOME || s == APP_SCR_RESULT) && app_ready(&g_app, now);
    bool stop_on = s == APP_SCR_RUN || s == APP_SCR_FAULT || g_c.stop.active;
    digitalWrite(PIN_LED_START, start_on ? HIGH : LOW);
    digitalWrite(PIN_LED_STOP, stop_on ? HIGH : LOW);
}

static void link_task(void *)
{
    uint8_t buf[256];
    TickType_t wake = xTaskGetTickCount();
    for (;;) {
        uint32_t now = millis();
        xSemaphoreTake(g_mx, portMAX_DELAY);
        int n;
        while ((n = LINK.available()) > 0) {
            size_t k = LINK.read(buf, n > (int)sizeof buf ? sizeof buf : (size_t)n);
            pp_parser_feed(&g_parser, buf, k, now, frame_cb, &now);
        }
        pp_client_tick(&g_c, now);
        app_update(&g_app, now);                /* events of the frames first (Q8) */
        uint8_t ev = buttons_poll(now);
        if (ev) {
            app_buttons(&g_app, ev, now);
            app_update(&g_app, now);            /* only what this press caused */
        }
        leds(now);
        xSemaphoreGive(g_mx);
        vTaskDelayUntil(&wake, pdMS_TO_TICKS(LINK_PERIOD_MS));     /* FreeRTOS tick 1 kHz */
    }
}

void setup()
{
    LINK.setRxBufferSize(LINK_RX_BUFFER);       /* before begin (P6) */
#if PANEL_LINK_BENCH
    LINK.begin(PI_BAUD);
#else
    LINK.begin(PI_BAUD, SERIAL_8N1, PIN_PI_RX, PIN_PI_TX);
#endif
    pinMode(PIN_LED_START, OUTPUT);
    pinMode(PIN_LED_STOP, OUTPUT);
    buttons_begin();
    const char *spr = ui_begin();
    pp_parser_init(&g_parser, PP_ACCEPT_PANEL, PP_PARSER_STALE_MS);
    pp_client_init(&g_c, nonce_cb, nullptr, send_cb, nullptr);
    app_init(&g_app, &g_c);
    snprintf(g_app.msg, sizeof g_app.msg, "v%d%s, %s", PP_VERSION, PANEL_LINK_BENCH ? " bench" : "", spr);
    g_mx = xSemaphoreCreateMutex();
    xTaskCreatePinnedToCore(link_task, "panel_link", 6144, nullptr, 3, nullptr, 0);
}

void loop()
{
    static uint32_t t_draw = 0, sig_drawn = 0;
    static pp_client_t cs;                      /* copies drawn from (static: ~6 KB) */
    static app_t as;

    xSemaphoreTake(g_mx, portMAX_DELAY);
    memcpy(&cs, &g_c, sizeof cs);
    as = g_app;
    uint32_t now = millis();                    /* after the copy: never older than it (Q5) */
    as.c = &cs;
    app_view_t view = app_view(&as, now);
    bool changes_view = !as.drawn_valid || !app_view_same(&view, &as.drawn);
    PanelView v{&as, now};
    uint32_t sig = ui_signature(v);
    uint32_t since = now - t_draw;
    bool draw = (sig != sig_drawn && (since >= REDRAW_MIN_MS || changes_view)) || since >= REDRAW_MAX_MS;
    if (draw && changes_view)
        app_drawing(&g_app);                    /* buttons other than STOP wait (Q3) */
    xSemaphoreGive(g_mx);
    if (draw) {
        ui_draw(v);
        sig_drawn = sig;
        t_draw = millis();
        xSemaphoreTake(g_mx, portMAX_DELAY);
        app_drawn(&g_app, &view);               /* what the glass shows now (P2, Q3) */
        xSemaphoreGive(g_mx);
    }
    delay(5);
}
