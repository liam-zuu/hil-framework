#include "ui.h"

#include <Arduino.h>
#include <TFT_eSPI.h>

#include "pins.h"

namespace {

TFT_eSPI     g_tft;
TFT_eSprite  g_spr(&g_tft);
TFT_eSPI    *g = &g_tft;      /* draw target: the full-screen sprite in PSRAM when it exists */
bool         g_spr_ok = false;

const int W = 480, H = 320, BAR = 36, FOOT = 290;

uint32_t fnv(uint32_t h, const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++)
        h = (h ^ b[i]) * 16777619u;
    return h;
}

const char *phase_name(uint8_t st)
{
    switch (st) {
    case PP_ST_IDLE:     return "IDLE";
    case PP_ST_PREPARE:  return "PREPARE";
    case PP_ST_ARM:      return "ARM";
    case PP_ST_RUN:      return "RUN";
    case PP_ST_STOPPING: return "STOPPING";
    case PP_ST_END:      return "END";
    case PP_ST_ERROR:    return "ERROR";
    default:             return "?";
    }
}

void quality_text(uint8_t q, char *out, size_t n)
{
    if (!q) {
        snprintf(out, n, "ok");
        return;
    }
    snprintf(out, n, "%s%s%s%s%s", (q & PP_Q_LOST) ? "LOST " : "", (q & PP_Q_SP_STALE) ? "SP-OLD " : "",
             (q & PP_Q_SPI_STALE) ? "SPI-OLD " : "", (q & PP_Q_DUT_UNCONF) ? "DUT? " : "",
             (q & PP_Q_DATA_INVALID) ? "INVALID" : "");
}

void ctrl_text(const pp_client_t *c, char *out, size_t n)
{
    if (!c->has_info || c->info.ctrl_src == PP_CTRL_SRC_NONE || !c->info.ctrl[0])
        snprintf(out, n, "CTRL -");
    else if (c->info.ctrl_src == PP_CTRL_SRC_FW)
        snprintf(out, n, "CTRL %s", c->info.ctrl);
    else
        snprintf(out, n, "CTRL %s (not verified)", c->info.ctrl);   /* from the load profile */
}

void dot(int x, int y, uint16_t color) { g->fillCircle(x, y, 7, color); }

void text(int x, int y, const char *s, uint8_t font, uint16_t fg, uint8_t size = 1)
{
    g->setTextFont(font);
    g->setTextSize(size);
    g->setTextColor(fg);
    g->drawString(s, x, y);
}

void health_bar(const PanelView &v)
{
    const pp_client_t *c = v.c;
    bool link = c->sid && pp_client_link(c, v.now);
    g->fillRect(0, 0, W, BAR, TFT_DARKGREY);
    text(8, 10, "RPi5", 2, TFT_WHITE);
    dot(56, 18, link ? TFT_GREEN : TFT_RED);
    text(84, 10, "H7", 2, TFT_WHITE);
    dot(112, 18, (link && c->has_st && (c->st.links & PP_L_H7)) ? TFT_GREEN : TFT_RED);
    text(140, 10, "ESP32", 2, TFT_WHITE);
    if (link && c->has_st && (c->st.links & PP_L_ESP32_KNOWN))
        dot(198, 18, (c->st.links & PP_L_ESP32) ? TFT_GREEN : TFT_RED);
    else
        text(192, 10, "-", 2, TFT_LIGHTGREY);       /* not known yet (no H7 status bit) */
    char s[40];
    if (link && c->has_info && (c->info.flags & PP_I_LOOP_VALID))
        snprintf(s, sizeof s, "LOOP %uus  ovr %u", c->info.loop_us, c->info.overruns);
    else
        snprintf(s, sizeof s, "LOOP -");
    text(240, 10, s, 2, TFT_WHITE);
}

void footer(const PanelView &v)
{
    g->drawFastHLine(0, FOOT, W, TFT_DARKGREY);
    text(8, FOOT + 8, v.msg, 2, TFT_LIGHTGREY);
}

void scr_nolink(const PanelView &v)
{
    text(40, 90, "NO LINK", 4, TFT_YELLOW, 2);
    text(40, 160, v.c->sid ? "no STATUS from the RPi5" : "waiting for the RPi5 daemon", 4, TFT_WHITE);
    if (v.c->stop.active)
        text(40, 210, "STOP pending (resending)", 4, TFT_RED);
}

void scr_home(const PanelView &v)
{
    const pp_client_t *c = v.c;
    char s[48];
    ctrl_text(c, s, sizeof s);
    text(8, 46, s, 4, TFT_CYAN);
    bool ready = ui_ready(v);
    text(330, 46, ready ? "READY" : "NOT READY", 4, ready ? TFT_GREEN : TFT_YELLOW);
    int n = c->has_profile_count ? c->profile_count : 0;
    if (!n) {
        text(8, 110, c->has_profile_count ? "no tests on the RPi5" : "loading test list...", 4, TFT_LIGHTGREY);
        return;
    }
    const int rows = 6, row_h = 30, y0 = 90;
    int first = v.cursor - rows / 2;
    if (first > n - rows) first = n - rows;
    if (first < 0) first = 0;
    for (int r = 0; r < rows && first + r < n; r++) {
        int i = first + r;
        int y = y0 + r * row_h;
        bool cur = i == v.cursor;
        if (cur)
            g->fillRect(0, y - 2, W, row_h, TFT_NAVY);
        bool sel = c->has_st && c->st.profile == i;
        snprintf(s, sizeof s, "%s %s%s", cur ? ">" : " ", ui_profile_name(c, i), sel ? "  *" : "");
        text(8, y, s, 4, cur ? TFT_WHITE : TFT_LIGHTGREY);
    }
}

void title_line(const PanelView &v)
{
    const pp_client_t *c = v.c;
    char s[64], ct[40];
    ctrl_text(c, ct, sizeof ct);
    snprintf(s, sizeof s, "%s  |  %s", c->st.profile == PP_PROFILE_NONE ? "-" : ui_profile_name(c, c->st.profile),
             ct + 5 /* skip "CTRL " */);
    text(8, 46, s, 4, TFT_CYAN);
}

void scr_run(const PanelView &v)
{
    const pp_client_t *c = v.c;
    char s[112], q[48];
    title_line(v);
    uint16_t col = c->st.run_state == PP_ST_STOPPING ? TFT_YELLOW : TFT_GREEN;
    text(8, 86, phase_name(c->st.run_state), 4, col, 2);
    snprintf(s, sizeof s, "t = %u.%u s", c->st.phase_ds / 10, c->st.phase_ds % 10);
    text(8, 150, s, 4, TFT_WHITE, 2);
    quality_text(c->st.quality, q, sizeof q);
    if (c->st.data_age_ms == 0xFFFF) {          /* unknown or saturated on the RPi5 */
        snprintf(s, sizeof s, "data age -   lost %u   %s", c->st.lost, q);
    } else {
        uint32_t age = c->st.data_age_ms + (uint32_t)(v.now - c->last_status_ms);
        snprintf(s, sizeof s, "data age %lu ms   lost %u   %s", (unsigned long)age, c->st.lost, q);
    }
    text(8, 214, s, 2, TFT_LIGHTGREY);
    const char *stop = c->stop_state == PP_STOP_REQUESTED ? "STOP requested" :
                       c->stop_state == PP_STOP_ACKED ? "STOP acknowledged, stopping" : nullptr;
    if (stop)
        text(8, 244, stop, 4, TFT_RED);
}

void scr_result(const PanelView &v)
{
    const pp_client_t *c = v.c;
    char s[112], q[48];
    title_line(v);
    const char *out = c->st.outcome == PP_OUT_COMPLETED ? "COMPLETED" :
                      c->st.outcome == PP_OUT_USER_STOP ? "USER STOP" :
                      c->st.outcome == PP_OUT_ERROR ? "ERROR" : "END";
    uint16_t col = c->st.outcome == PP_OUT_COMPLETED ? TFT_GREEN :
                   c->st.outcome == PP_OUT_USER_STOP ? TFT_YELLOW : TFT_RED;
    text(8, 82, out, 4, col, 2);
    if (c->has_info && (c->info.flags & PP_I_RESULT_VALID)) {
        for (int k = 0; k < PP_RESULT_LINES; k++)
            if (c->info.result[k][0])
                text(8, 144 + 30 * k, c->info.result[k], 4, TFT_WHITE);
    } else {
        text(8, 144, "no result lines", 4, TFT_LIGHTGREY);
    }
    quality_text(c->st.quality, q, sizeof q);
    snprintf(s, sizeof s, "lost %u   %s      STOP: back   START: run again", c->st.lost, q);
    text(8, 262, s, 2, TFT_LIGHTGREY);
}

void scr_fault(const PanelView &v)
{
    const pp_client_t *c = v.c;
    char s[48];
    g->fillRect(0, BAR, W, FOOT - BAR, TFT_MAROON);
    text(8, 50, "FAULT", 4, TFT_WHITE, 2);
    snprintf(s, sizeof s, "E%02u", c->st.error_code);
    text(8, 120, s, 4, TFT_YELLOW, 2);
    text(8, 184, (c->has_info && c->info.error_text[0]) ? c->info.error_text : "(no text from the RPi5)", 4, TFT_WHITE);
    text(8, 250, "STOP: acknowledge", 4, TFT_LIGHTGREY);
}

}  // namespace

const char *ui_profile_name(const pp_client_t *c, int i)
{
    if (i >= 0 && i < PP_PROFILES_STORED && c->profile_ok[i])
        return c->profile_name[i];
    return "?";
}

bool ui_ready(const PanelView &v)
{
    const pp_client_t *c = v.c;
    return c->sid && pp_client_link(c, v.now) && c->has_st && (c->st.links & PP_L_H7);
}

Screen ui_screen(const PanelView &v)
{
    const pp_client_t *c = v.c;
    if (!c->sid || !c->has_st || !pp_client_link(c, v.now))
        return SCR_NOLINK;
    if (c->st.run_state == PP_ST_ERROR)
        return SCR_FAULT;
    if (PP_ST_IS_ACTIVE(c->st.run_state))
        return SCR_RUN;
    if (c->st.run_state == PP_ST_END && !v.result_dismissed)
        return SCR_RESULT;
    return SCR_HOME;
}

void ui_begin(void)
{
    pinMode(PIN_TFT_BL, OUTPUT);
    digitalWrite(PIN_TFT_BL, HIGH);
    g_tft.init();
    g_tft.setRotation(1);                       /* landscape 480x320 */
    g_tft.fillScreen(TFT_BLACK);
    g_spr.setColorDepth(16);
    g_spr_ok = g_spr.createSprite(W, H) != nullptr;   /* ~300 KB: in PSRAM */
    g = g_spr_ok ? (TFT_eSPI *)&g_spr : &g_tft;
}

uint32_t ui_signature(const PanelView &v)
{
    const pp_client_t *c = v.c;
    uint32_t h = 2166136261u;
    Screen s = ui_screen(v);
    h = fnv(h, &s, sizeof s);
    h = fnv(h, &v.cursor, sizeof v.cursor);
    h = fnv(h, v.msg, strlen(v.msg));
    h = fnv(h, &c->stop_state, sizeof c->stop_state);
    h = fnv(h, &c->profiles_have, sizeof c->profiles_have);
    bool link = c->sid && pp_client_link(c, v.now);
    h = fnv(h, &link, sizeof link);
    /* field by field: struct padding is not hashed (data age alone is not worth a redraw) */
    if (c->has_st) {
        const pp_status_t &st = c->st;
        uint16_t f[8] = {st.run_state, st.phase_ds, st.profile, st.outcome, st.quality, st.lost,
                         st.links, st.error_code};
        h = fnv(h, f, sizeof f);
    }
    if (c->has_info) {
        const pp_info_t &in = c->info;
        uint16_t f[4] = {in.loop_us, in.overruns, in.flags, in.ctrl_src};
        h = fnv(h, f, sizeof f);
        h = fnv(h, in.ctrl, strlen(in.ctrl));
        for (int k = 0; k < PP_RESULT_LINES; k++)
            h = fnv(h, in.result[k], strlen(in.result[k]) + 1);
        h = fnv(h, in.error_text, strlen(in.error_text));
    }
    return h;
}

void ui_draw(const PanelView &v)
{
    g->fillScreen(TFT_BLACK);
    health_bar(v);
    switch (ui_screen(v)) {
    case SCR_NOLINK: scr_nolink(v); break;
    case SCR_HOME:   scr_home(v); break;
    case SCR_RUN:    scr_run(v); break;
    case SCR_RESULT: scr_result(v); break;
    case SCR_FAULT:  scr_fault(v); break;
    }
    footer(v);
    if (g_spr_ok)
        g_spr.pushSprite(0, 0);
}
