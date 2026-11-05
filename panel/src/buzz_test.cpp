/*
 * Buzzer tool of the panel, env panel_s3_buzz_test only (decision 21, Liam 05/11: "Duyệt,
 * làm bước 1"). Not the panel firmware: no link to the RPi5, no run screens. It finds the
 * frequency the buzzer is loudest at (Liam listens; a phone sound meter at a fixed distance
 * if wanted) and plays every pattern built on it. Modes and buttons: lib/panel_buzz/buzz_tool.h.
 * The number found becomes PB_F0_DEFAULT (step 3), Liam approves the patterns.
 * GPIO7 is set HIGH as in the panel firmware (ui_begin): the bench wiring of 05/11 may feed
 * the buzzer VCC from it.
 */
#include <Arduino.h>
#include <TFT_eSPI.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "buttons.h"
#include "buzz_tool.h"
#include "buzzer.h"
#include "pins.h"

static_assert(BT_EV_NAV == BTN_EV_NAV && BT_EV_NAV_BACK == BTN_EV_NAV_BACK && BT_EV_SELECT == BTN_EV_SELECT &&
                  BT_EV_START == BTN_EV_START && BT_EV_STOP == BTN_EV_STOP,
              "buzz_tool button bits must match buttons.h");

static TFT_eSPI g_tft;
static bt_t     g_bt;
static portMUX_TYPE g_mux = portMUX_INITIALIZER_UNLOCKED;

/* buttons + tone every 2 ms on core 0; the drawing (slow SPI) stays in loop() */
static void tool_task(void *)
{
    TickType_t wake = xTaskGetTickCount();
    for (;;) {
        uint32_t now = millis();
        uint8_t ev = buttons_poll(now);
        portENTER_CRITICAL(&g_mux);
        if (ev)
            bt_buttons(&g_bt, ev, now);
        uint16_t hz = bt_tone(&g_bt, now);
        portEXIT_CRITICAL(&g_mux);
        buzzer_out(hz);
        vTaskDelayUntil(&wake, pdMS_TO_TICKS(2));
    }
}

void setup()
{
    buzzer_begin();
    bt_init(&g_bt, millis());                   /* start-up tune right away */
    buttons_begin();
    xTaskCreatePinnedToCore(tool_task, "buzz_tool", 3072, nullptr, 3, nullptr, 0);
    pinMode(PIN_TFT_BL, OUTPUT);
    digitalWrite(PIN_TFT_BL, HIGH);
    g_tft.init();
    g_tft.setRotation(1);                       /* landscape 480x320 */
    g_tft.fillScreen(TFT_BLACK);
    g_tft.setTextColor(TFT_WHITE, TFT_BLACK);
    g_tft.setTextDatum(TC_DATUM);
    g_tft.drawString("Buzzer tool", 240, 10, 4);
    g_tft.setTextColor(TFT_DARKGREY, TFT_BLACK);
    char line[48];
    snprintf(line, sizeof line, "f0 in firmware now: %u Hz", (unsigned)PB_F0_DEFAULT);
    g_tft.drawString(line, 240, 44, 2);
}

void loop()
{
    static uint16_t d_hz = 0xFFFF;
    static const char *d_mode = nullptr, *d_pat = nullptr;
    static bool d_sound = true;

    portENTER_CRITICAL(&g_mux);
    uint16_t hz = g_bt.hz;
    const char *mode = bt_mode_name(&g_bt), *pat = bt_pattern_name(&g_bt), *hint = bt_hint(&g_bt);
    bool sound = g_bt.mode == BT_TONE || g_bt.mode == BT_SWEEP;
    portEXIT_CRITICAL(&g_mux);

    if (hz != d_hz || mode != d_mode || pat != d_pat || sound != d_sound) {
        char num[16];
        snprintf(num, sizeof num, "%u Hz", (unsigned)hz);
        g_tft.setTextDatum(MC_DATUM);
        g_tft.setTextColor(sound ? TFT_YELLOW : TFT_LIGHTGREY, TFT_BLACK);
        g_tft.setTextSize(3);
        g_tft.setTextPadding(460);
        g_tft.drawString(num, 240, 130, 4);
        g_tft.setTextSize(1);
        g_tft.setTextColor(TFT_CYAN, TFT_BLACK);
        g_tft.drawString(mode, 240, 205, 4);
        g_tft.setTextColor(TFT_WHITE, TFT_BLACK);
        g_tft.drawString(pat, 240, 240, 4);
        g_tft.setTextColor(TFT_DARKGREY, TFT_BLACK);
        g_tft.drawString(hint, 240, 290, 2);
        g_tft.setTextPadding(0);
        d_hz = hz;
        d_mode = mode;
        d_pat = pat;
        d_sound = sound;
    }
    delay(50);
}
