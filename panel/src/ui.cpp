/* ui.cpp - screens of panel M2, new look (Liam 05/11, PANEL session: "khá đẹp rồi đấy, làm
 * theo bản mới đi"; mock-up 03_CODE/docs/panel/2026-11-05_Mockup_Panel_M2_giao_dien_moi_v1.html
 * in the thesis repo): dark background, rounded cards and chips, smooth (anti-aliased) fonts.
 * Same data as before, protocol v6 only: nothing new is asked from the RPi5, the screen choice,
 * the buttons and the signature rules are unchanged (lib/panel_app).
 *
 * Fonts: src/fonts/ (one header per font), made by tools/make_vlw_font.py (DejaVu Sans). One smooth font is
 * loaded at a time (TFT_eSPI keeps a single gFont), so text() switches only when it changes.
 * Only the characters of make_vlw_font.py may be drawn (ASCII, µ, ·; ui_font_big: digits . - ? E s):
 * a missing one shows as an empty box.
 * Every smooth shape and every text gets the colour it is drawn on (bg), so nothing is read
 * back from the sprite.
 */
#include "ui.h"

#include <Arduino.h>
#include <TFT_eSPI.h>
#include <esp_heap_caps.h>

#include "fonts/ui_font_big.h"
#include "fonts/ui_font_body.h"
#include "fonts/ui_font_bold.h"
#include "fonts/ui_font_head.h"
#include "fonts/ui_font_small.h"
#include "pins.h"

namespace {

TFT_eSPI     g_tft;
TFT_eSprite  g_spr(&g_tft);
TFT_eSPI    *g = &g_tft;      /* draw target: the full-screen sprite in PSRAM when it exists */
bool         g_spr_ok = false;

const int W = 480, H = 320, FOOT = 290;

constexpr uint16_t rgb(uint32_t c)
{
    return (uint16_t)(((c >> 8) & 0xF800) | ((c >> 5) & 0x07E0) | ((c >> 3) & 0x001F));
}

/* palette (mock-up v1) */
const uint16_t BG = rgb(0x0E1116), CHIP = rgb(0x1A1F27), CARD = rgb(0x151A21), LINE = rgb(0x1F2530),
               RING = rgb(0x2A313C), DIV = rgb(0x232A35), TXT = rgb(0xE6E9EE), MUTED = rgb(0x8B95A5),
               DIM = rgb(0x5B6474), ROWTXT = rgb(0xAEB6C2), WHITE = rgb(0xFFFFFF),
               GREEN = rgb(0x22C55E), GREEN_T = rgb(0x86EFAC), GREEN_BG = rgb(0x14532D),
               GREEN_DEEP = rgb(0x052E16), GREEN_H = rgb(0x4ADE80), GREEN_PALE = rgb(0xBBF7D0),
               AMBER = rgb(0xF59E0B), AMBER_T = rgb(0xFCD34D), AMBER_BG = rgb(0x451A03),
               AMBER_DEEP = rgb(0x2B1403),
               RED = rgb(0xEF4444), RED_T = rgb(0xFCA5A5), RED_BG = rgb(0x450A0A), RED_DEEP = rgb(0x2A0A0A),
               RED_CHIP = rgb(0x7F1D1D), RED_PALE = rgb(0xFECACA), RED_TXT = rgb(0xFEE2E2),
               BLUE = rgb(0x3B82F6), BLUE_T = rgb(0x93C5FD), BLUE_BG = rgb(0x1E293B),
               ROW_SEL = rgb(0x172554), BLUE_CHIP = rgb(0x1E3A8A), BLUE_PALE = rgb(0xBFDBFE);

#define DOT "\xC2\xB7"        /* · */
#define MICRO "\xC2\xB5"      /* µ */

const uint8_t *g_font = nullptr;

/* ascent of a VLW array: 5th big-endian int32 of the header (make_vlw_font.py) */
int ascent_of(const uint8_t *f)
{
    return (f[16] << 24) | (f[17] << 16) | (f[18] << 8) | f[19];
}

void use_font(const uint8_t *f)
{
    if (f != g_font) {
        g->loadFont(f);
        g_font = f;
    }
}

int width_of(const char *s, const uint8_t *f)
{
    use_font(f);
    return g->textWidth(s);
}

/* text with its BASELINE at y (as in the mock-up); returns the x after it */
int text(int x, int y, const char *s, const uint8_t *f, uint16_t fg, uint16_t bg)
{
    use_font(f);
    g->setTextColor(fg, bg);
    g->setTextDatum(TL_DATUM);
    g->drawString(s, x, y - ascent_of(f));
    return x + g->textWidth(s);
}

int text_c(int cx, int y, const char *s, const uint8_t *f, uint16_t fg, uint16_t bg)
{
    return text(cx - width_of(s, f) / 2, y, s, f, fg, bg);
}

int text_r(int rx, int y, const char *s, const uint8_t *f, uint16_t fg, uint16_t bg)
{
    return text(rx - width_of(s, f), y, s, f, fg, bg);
}

void card(int x, int y, int w, int h, int r, uint16_t fill, uint16_t bg)
{
    g->fillSmoothRoundRect(x, y, w, h, r, fill, bg);
}

/* card with a 1 px border */
void card_line(int x, int y, int w, int h, int r, uint16_t fill, uint16_t line, uint16_t bg)
{
    g->fillSmoothRoundRect(x, y, w, h, r, line, bg);
    g->fillSmoothRoundRect(x + 1, y + 1, w - 2, h - 2, r - 1, fill, line);
}

/* pill with centred text; returns its width */
int pill(int x, int y, int h, const char *s, const uint8_t *f, uint16_t fill, uint16_t fg, uint16_t bg, int pad)
{
    int w = width_of(s, f) + 2 * pad;
    card(x, y, w, h, h / 2, fill, bg);
    text(x + pad, y + h / 2 + ascent_of(f) / 2, s, f, fg, fill);
    return w;
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
    size_t k = strlen(out);
    if (k && out[k - 1] == ' ')
        out[k - 1] = 0;
}

/* controller name and where it comes from (INFO ctrl_src) */
struct Ctrl {
    const char *name;           /* "-" when unknown */
    const char *state;          /* "not verified" / "from #FW" / "unknown" */
    uint16_t    state_col;
};

Ctrl ctrl_of(const pp_client_t *c)
{
    if (!c->has_info || c->info.ctrl_src == PP_CTRL_SRC_NONE || !c->info.ctrl[0])
        return {"-", "unknown", MUTED};
    if (c->info.ctrl_src == PP_CTRL_SRC_FW)
        return {c->info.ctrl, "from #FW", GREEN_T};
    return {c->info.ctrl, "not verified", AMBER};    /* from the load profile */
}

/* "name · ctrl · state" on one line, small font */
void run_line(int x, int y, const pp_client_t *c, bool with_name, uint16_t name_col, uint16_t muted, uint16_t bg)
{
    Ctrl k = ctrl_of(c);
    char s[48];
    if (with_name) {
        const char *p = c->st.profile == PP_PROFILE_NONE ? "-" : ui_profile_name(c, c->st.profile);
        snprintf(s, sizeof s, "%s " DOT " ", p);
        x = text(x, y, s, ui_font_small, name_col, bg);
    }
    snprintf(s, sizeof s, "%s " DOT " ", k.name);
    x = text(x, y, s, ui_font_small, muted, bg);
    text(x, y, k.state, ui_font_small, k.state_col, bg);
}

/* ---- health bar -------------------------------------------------------------------- */

enum Lamp { LAMP_OK, LAMP_BAD, LAMP_UNKNOWN };

int chip(int x, const char *label, Lamp l)
{
    int w = 24 + width_of(label, ui_font_small) + 10;
    card(x, 7, w, 24, 12, CHIP, BG);
    if (l == LAMP_UNKNOWN) {
        g->fillSmoothCircle(x + 14, 19, 5, DIM, CHIP);
        g->fillSmoothCircle(x + 14, 19, 3, CHIP, DIM);
    } else {
        g->fillSmoothCircle(x + 14, 19, 5, l == LAMP_OK ? GREEN : RED, CHIP);
    }
    text(x + 24, 24, label, ui_font_small, l == LAMP_UNKNOWN ? MUTED : TXT, CHIP);
    return w;
}

void health_bar(const PanelView &v)
{
    const pp_client_t *c = v.a->c;
    bool link = c->sid && pp_client_link(c, v.now);
    int x = 8;
    x += chip(x, "RPi5", link ? LAMP_OK : LAMP_BAD) + 6;
    x += chip(x, "H7", (link && c->has_st && (c->st.links & PP_L_H7)) ? LAMP_OK : LAMP_BAD) + 6;
    Lamp esp = LAMP_UNKNOWN;                    /* not known yet (no H7 status bit) */
    if (link && c->has_st && (c->st.links & PP_L_ESP32_KNOWN))
        esp = (c->st.links & PP_L_ESP32) ? LAMP_OK : LAMP_BAD;
    chip(x, "ESP32", esp);

    char a[24], b[24];
    if (link && c->has_info && (c->info.flags & PP_I_LOOP_VALID)) {
        snprintf(a, sizeof a, "%u " MICRO "s", c->info.loop_us);
        snprintf(b, sizeof b, " " DOT " ovr %u", c->info.overruns);
        int xr = 472 - width_of(b, ui_font_small);
        int xa = xr - width_of(a, ui_font_small);
        text_r(xa - 6, 24, "loop", ui_font_small, MUTED, BG);
        text(xa, 24, a, ui_font_small, TXT, BG);
        text(xr, 24, b, ui_font_small, MUTED, BG);
    } else {
        text_r(472, 24, "loop  -", ui_font_small, DIM, BG);
    }
}

void footer(const PanelView &v)
{
    g->drawFastHLine(0, FOOT, W, LINE);
    text(12, FOOT + 20, v.a->msg, ui_font_small, MUTED, BG);
}

/* ---- screens ----------------------------------------------------------------------- */

void scr_nolink(const PanelView &v)
{
    g->drawSmoothArc(240, 132, 37, 31, 0, 360, RING, BG, false);
    g->drawSmoothArc(240, 132, 37, 31, 180, 270, AMBER, BG, true);   /* 0 = 6 o'clock, clockwise */
    text_c(240, 208, "NO LINK", ui_font_head, AMBER, BG);
    text_c(240, 236, v.a->c->sid ? "no STATUS from the RPi5" : "waiting for the RPi5 daemon", ui_font_body,
           MUTED, BG);
    if (v.a->c->stop.active)
        text_c(240, 266, "STOP pending (resending)", ui_font_bold, RED_T, BG);
}

void scr_home(const PanelView &v)
{
    const pp_client_t *c = v.a->c;
    Ctrl k = ctrl_of(c);
    text(12, 62, "CONTROLLER", ui_font_small, MUTED, BG);
    int w = width_of(k.name, ui_font_bold) + 20;
    card(12, 68, w, 24, 6, BLUE_BG, BG);
    text(22, 86, k.name, ui_font_bold, BLUE_T, BLUE_BG);
    text(12 + w + 8, 85, k.state, ui_font_small, k.state_col, BG);

    bool ready = app_ready(v.a, v.now);
    const char *rs = ready ? "READY" : "NOT READY";
    uint16_t fill = ready ? GREEN_BG : AMBER_BG, dot = ready ? GREEN : AMBER, fg = ready ? GREEN_T : AMBER_T;
    int rw = 30 + width_of(rs, ui_font_bold) + 14;
    card(470 - rw, 62, rw, 30, 15, fill, BG);
    g->fillSmoothCircle(470 - rw + 18, 77, 5, dot, fill);
    text(470 - rw + 30, 83, rs, ui_font_bold, fg, fill);

    int n = c->has_profile_count ? c->profile_count : 0;
    if (!n) {
        text(12, 140, c->has_profile_count ? "no tests on the RPi5" : "loading test list...", ui_font_body,
             MUTED, BG);
        return;
    }
    const int rows = 4, row_h = 40, stride = 46, y0 = 106;
    int first = v.a->cursor - rows / 2;
    if (first > n - rows) first = n - rows;
    if (first < 0) first = 0;
    for (int r = 0; r < rows && first + r < n; r++) {
        int i = first + r;
        int y = y0 + r * stride;
        bool cur = i == v.a->cursor;
        bool sel = c->has_st && c->st.profile == i;
        uint16_t rowbg = cur ? ROW_SEL : CARD;
        if (cur) {
            card_line(12, y, 456, row_h, 8, ROW_SEL, BLUE, BG);
            card(12, y + 8, 4, 24, 2, BLUE, ROW_SEL);
            text(28, y + 26, ui_profile_name(c, i), ui_font_bold, WHITE, rowbg);
        } else {
            card(12, y, 456, row_h, 8, CARD, BG);
            text(28, y + 26, ui_profile_name(c, i), ui_font_body, ROWTXT, rowbg);
        }
        if (sel) {
            int bw = width_of("selected", ui_font_small) + 24;
            card(458 - bw, y + 9, bw, 22, 11, BLUE_CHIP, rowbg);
            text(458 - bw + 12, y + 24, "selected", ui_font_small, BLUE_PALE, BLUE_CHIP);
        }
    }
}

void title_line(const PanelView &v)
{
    const pp_client_t *c = v.a->c;
    const char *p = c->st.profile == PP_PROFILE_NONE ? "-" : ui_profile_name(c, c->st.profile);
    int x = text(12, 64, p, ui_font_bold, TXT, BG);
    run_line(x + 12, 64, c, false, TXT, MUTED, BG);
}

void stepper(uint8_t st)
{
    static const char *name[3] = {"PREPARE", "ARM", "RUN"};
    static const int x[3] = {12, 164, 316}, w[3] = {140, 140, 152};
    int cur = st == PP_ST_PREPARE ? 0 : st == PP_ST_ARM ? 1 : st == PP_ST_RUN ? 2 : -1;
    for (int k = 0; k < 3; k++) {
        card(x[k], 78, w[k], 8, 4, (cur >= 0 && k <= cur) ? GREEN : RING, BG);
        if (k == cur)
            text(x[k], 104, name[k], ui_font_bold, GREEN_T, BG);
        else
            text(x[k], 104, name[k], ui_font_small, MUTED, BG);
    }
}

void stat_card(int x, int w, const char *label, const char *value, uint16_t col)
{
    card(x, 200, w, 56, 8, CARD, BG);
    text(x + 12, 221, label, ui_font_small, MUTED, CARD);
    const uint8_t *f = width_of(value, ui_font_body) <= w - 24 ? ui_font_body : ui_font_small;
    text(x + 12, 245, value, f, col, CARD);
}

void scr_run(const PanelView &v)
{
    const pp_client_t *c = v.a->c;
    char s[48], q[48];
    title_line(v);
    if (c->st.run_state == PP_ST_STOPPING) {
        card(12, 78, 160, 34, 17, AMBER_BG, BG);
        g->fillSmoothCircle(32, 95, 6, AMBER, AMBER_BG);
        text(46, 101, "STOPPING", ui_font_bold, AMBER_T, AMBER_BG);
    } else {
        stepper(c->st.run_state);
    }
    snprintf(s, sizeof s, "%u.%u", c->st.phase_ds / 10, c->st.phase_ds % 10);
    int x = text(12, 176, s, ui_font_big, WHITE, BG);
    x = text(x + 6, 176, "s", ui_font_head, MUTED, BG);
    snprintf(s, sizeof s, "in %s", phase_name(c->st.run_state));
    text(x + 14, 174, s, ui_font_small, MUTED, BG);

    const char *stop = c->stop_state == PP_STOP_REQUESTED ? "STOP requested" :
                       c->stop_state == PP_STOP_ACKED ? "STOP acknowledged, stopping" : nullptr;
    if (stop) {
        card_line(12, 200, 456, 56, 8, RED_BG, RED, BG);
        text(28, 234, stop, ui_font_bold, RED_T, RED_BG);
        return;
    }
    if (c->st.data_age_ms == 0xFFFF) {          /* unknown or saturated on the RPi5 */
        snprintf(s, sizeof s, "-");
    } else {
        uint32_t age = c->st.data_age_ms + (uint32_t)(v.now - c->last_status_ms);
        snprintf(s, sizeof s, "%lu ms", (unsigned long)age);
    }
    /* the quality card is the widest: all five flags must fit in the small font (PC render) */
    stat_card(12, 104, "data age", s, TXT);
    snprintf(s, sizeof s, "%u", c->st.lost);
    stat_card(124, 72, "lost", s, c->st.lost ? AMBER_T : TXT);
    quality_text(c->st.quality, q, sizeof q);
    stat_card(204, 264, "quality", q, c->st.quality ? AMBER_T : GREEN_T);
}

void scr_result(const PanelView &v)
{
    const pp_client_t *c = v.a->c;
    char s[96], q[48];
    const char *out = c->st.outcome == PP_OUT_COMPLETED ? "COMPLETED" :
                      c->st.outcome == PP_OUT_USER_STOP ? "USER STOP" :
                      c->st.outcome == PP_OUT_ERROR ? "ERROR" : "END";
    uint16_t fill, line, word, soft, chipf, chipt;
    if (c->st.outcome == PP_OUT_COMPLETED) {
        fill = GREEN_DEEP; line = GREEN; word = GREEN_H; soft = GREEN_T; chipf = GREEN_BG; chipt = GREEN_PALE;
    } else if (c->st.outcome == PP_OUT_USER_STOP) {
        fill = AMBER_DEEP; line = AMBER; word = AMBER; soft = AMBER_T; chipf = AMBER_BG; chipt = AMBER_T;
    } else {
        fill = RED_DEEP; line = RED; word = RED_T; soft = RED_T; chipf = RED_CHIP; chipt = RED_PALE;
    }
    card_line(12, 44, 456, 62, 10, fill, line, BG);
    run_line(28, 72, c, true, soft, soft, fill);
    text(28, 98, out, ui_font_head, word, fill);
    uint16_t run = app_run(v.a);                /* R1-R5: lines only of this run */
    if (run)
        snprintf(s, sizeof s, "run #%u", run);
    else
        snprintf(s, sizeof s, "run #?");
    int cw = width_of(s, ui_font_body) + 28;
    card(456 - cw, 60, cw, 28, 14, chipf, fill);
    text(456 - cw + 14, 80, s, ui_font_body, chipt, chipf);

    card(12, 116, 456, 128, 10, CARD, BG);
    app_result_t r = app_result(v.a);
    if (r == APP_RES_FRESH) {
        for (int k = 0; k < PP_RESULT_LINES; k++) {
            if (c->info.result[k][0])
                text(28, 148 + 40 * k, c->info.result[k], ui_font_body, TXT, CARD);
            if (k < PP_RESULT_LINES - 1)
                g->drawFastHLine(28, 161 + 40 * k, 424, DIV);
        }
    } else {
        const char *m = c->st.outcome != PP_OUT_COMPLETED ? "no result lines" :
                        r == APP_RES_WAITING ? "waiting for result..." : "result of another run (see PC)";
        text(28, 186, m, ui_font_body, MUTED, CARD);
    }

    quality_text(c->st.quality, q, sizeof q);
    snprintf(s, sizeof s, "lost %u " DOT " quality %s", c->st.lost, q);
    int w1 = width_of("START  run again", ui_font_small) + 24;
    int w2 = width_of("STOP  back", ui_font_small) + 24;
    int chips_x = 468 - w1 - 8 - w2;
    text(12, 272, s, ui_font_small, c->st.quality ? AMBER_T : MUTED, BG);
    if (12 + width_of(s, ui_font_small) + 12 <= chips_x) {      /* the hints only if they fit */
        card(chips_x, 256, w2, 24, 12, CHIP, BG);
        text(chips_x + 12, 273, "STOP  back", ui_font_small, ROWTXT, CHIP);
        card(468 - w1, 256, w1, 24, 12, CHIP, BG);
        text(468 - w1 + 12, 273, "START  run again", ui_font_small, ROWTXT, CHIP);
    }
}

void scr_fault(const PanelView &v)
{
    const pp_client_t *c = v.a->c;
    char s[16];
    /* everything centred on the screen (Liam 05/11: "cái FAULT cho nó ra giữa màn cho đẹp nhé") */
    card_line(12, 44, 456, 236, 12, RED_DEEP, RED, BG);
    int pw = width_of("FAULT", ui_font_bold) + 32;
    pill(240 - pw / 2, 60, 28, "FAULT", ui_font_bold, RED_CHIP, RED_PALE, RED_DEEP, 16);
    snprintf(s, sizeof s, "E%02u", c->st.error_code);
    text_c(240, 156, s, ui_font_big, RED_T, RED_DEEP);
    text_c(240, 198, (c->has_info && c->info.error_text[0]) ? c->info.error_text : "(no text from the RPi5)",
           ui_font_body, RED_TXT, RED_DEEP);
    int w = width_of("STOP  acknowledge", ui_font_small) + 32;
    card_line(240 - w / 2, 230, w, 30, 15, RED_BG, RED_CHIP, RED_DEEP);
    text_c(240, 250, "STOP  acknowledge", ui_font_small, RED_PALE, RED_BG);
}

}  // namespace

const char *ui_profile_name(const pp_client_t *c, int i)
{
    if (i >= 0 && i < PP_PROFILES_STORED && c->profile_ok[i])
        return c->profile_name[i];
    return "?";
}

const char *ui_begin(void)
{
    pinMode(PIN_TFT_BL, OUTPUT);
    digitalWrite(PIN_TFT_BL, HIGH);
    g_tft.init();
    g_tft.setRotation(1);                       /* landscape 480x320 */
    g_tft.fillScreen(BG);
    g_spr.setColorDepth(16);
    size_t ps_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    g_spr_ok = g_spr.createSprite(W, H) != nullptr;   /* ~300 KB: meant for PSRAM */
    size_t ps_after = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    g = g_spr_ok ? (TFT_eSPI *)&g_spr : &g_tft;
    g_font = nullptr;
    if (!g_spr_ok)
        return "no sprite: direct drawing (flicker)";
    return ps_before - ps_after >= (size_t)W * H * 2 ? "sprite in PSRAM" : "sprite in internal RAM";
}

uint32_t ui_signature(const PanelView &v)
{
    const pp_client_t *c = v.a->c;
    uint32_t h = 2166136261u;
    auto fnv = [&h](const void *p, size_t n) {
        const uint8_t *b = (const uint8_t *)p;
        for (size_t i = 0; i < n; i++)
            h = (h ^ b[i]) * 16777619u;
    };
    app_screen_t s = app_screen(v.a, v.now);
    fnv(&s, sizeof s);
    fnv(&v.a->cursor, sizeof v.a->cursor);
    fnv(v.a->msg, strlen(v.a->msg));
    app_result_t r = app_result(v.a);
    fnv(&r, sizeof r);
    uint16_t run = app_run(v.a);
    fnv(&run, sizeof run);
    fnv(&c->sid, sizeof c->sid);
    fnv(&c->profiles_gen, sizeof c->profiles_gen);
    fnv(&c->stop_state, sizeof c->stop_state);
    fnv(&c->profiles_have, sizeof c->profiles_have);
    bool link = c->sid && pp_client_link(c, v.now);
    fnv(&link, sizeof link);
    /* field by field: struct padding is not hashed (data age alone is not worth a redraw) */
    if (c->has_st) {
        const pp_status_t &st = c->st;
        uint16_t f[9] = {st.run_state, st.phase_ds, st.profile, st.outcome, st.quality, st.lost,
                         st.links, st.error_code, st.run_no};
        fnv(f, sizeof f);
    }
    if (c->has_info) {
        const pp_info_t &in = c->info;
        uint16_t f[5] = {in.loop_us, in.overruns, in.flags, in.ctrl_src, in.result_run};
        fnv(f, sizeof f);
        fnv(in.ctrl, strlen(in.ctrl));
        for (int k = 0; k < PP_RESULT_LINES; k++)
            fnv(in.result[k], strlen(in.result[k]) + 1);
        fnv(in.error_text, strlen(in.error_text));
    }
    return h;
}

void ui_draw(const PanelView &v)
{
    /* fillSprite, not fillScreen: a TFT_eSprite keeps the TFT's rotation-0 _width/_height
     * (320 x 480; Sprite.cpp never sets them), so fillScreen on the 480 x 320 sprite cleared
     * only x < 320 and the right third kept old pixels (bench 05/11: black right part, an old
     * "loop" over "ovr"). */
    if (g_spr_ok)
        g_spr.fillSprite(BG);
    else
        g->fillScreen(BG);
    health_bar(v);
    switch (app_screen(v.a, v.now)) {
    case APP_SCR_NOLINK: scr_nolink(v); break;
    case APP_SCR_HOME:   scr_home(v); break;
    case APP_SCR_RUN:    scr_run(v); break;
    case APP_SCR_RESULT: scr_result(v); break;
    case APP_SCR_FAULT:  scr_fault(v); break;
    }
    footer(v);
    if (g_spr_ok)
        g_spr.pushSprite(0, 0);
}
