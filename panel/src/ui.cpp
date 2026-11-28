/* ui.cpp - screens of panel M2, new look (Liam 05/11, PANEL session: "khá đẹp rồi đấy, làm
 * theo bản mới đi"; mock-up 03_CODE/docs/panel/2026-11-05_Mockup_Panel_M2_giao_dien_moi_v1.html
 * in the thesis repo): dark background, rounded cards and chips, smooth (anti-aliased) fonts.
 * Screen selection and START eligibility come from lib/panel_app.
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
#include <algorithm>
#include <cfloat>

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
        snprintf(out, n, "OK");
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
    const char *state;          /* "Not verified" / "From #FW" / "Unknown" */
    uint16_t    state_col;
};

Ctrl ctrl_of(const pp_client_t *c)
{
    if (!c->has_info || c->info.ctrl_src == PP_CTRL_SRC_NONE || !c->info.ctrl[0])
        return {"-", "Unknown", MUTED};
    if (c->info.ctrl_src == PP_CTRL_SRC_FW)
        return {c->info.ctrl, "From #FW", GREEN_T};
    return {c->info.ctrl, "Not verified", AMBER};    /* from the load profile */
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
        snprintf(b, sizeof b, " " DOT " Ovr %u", c->info.overruns);
        int xr = 472 - width_of(b, ui_font_small);
        int xa = xr - width_of(a, ui_font_small);
        text_r(xa - 6, 24, "Loop", ui_font_small, MUTED, BG);
        text(xa, 24, a, ui_font_small, TXT, BG);
        text(xr, 24, b, ui_font_small, MUTED, BG);
    } else {
        text_r(472, 24, "Loop  -", ui_font_small, DIM, BG);
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
    if (app_proto_mismatch(v.a, v.now)) {
        char s[48];
        snprintf(s, sizeof s, "Protocol v%u, panel v%u: reflash", v.a->proto_foreign_ver, PP_VERSION);
        text_c(240, 236, s, ui_font_body, AMBER_T, BG);
    } else {
        text_c(240, 236, v.a->c->sid ? "No STATUS from the RPi5" : "Waiting for the RPi5 daemon",
               ui_font_body, MUTED, BG);
    }
    if (v.a->c->stop.active)
        text_c(240, 266, "STOP pending (resending)", ui_font_bold, RED_T, BG);
}

void firmware_line(const pp_client_t *c)
{
    const pp_extra_t &e = c->extra;
    bool has_sha = c->has_extra && e.dut_sha[0];
    char dut[16];
    snprintf(dut, sizeof dut, "DUT %.8s", has_sha ? e.dut_sha : "-");
    int x = text(12, 111, dut, ui_font_small, TXT, BG);
    if (has_sha) {
        const char *state = "";
        uint16_t col = MUTED;
        switch (e.sha_state) {
        case PP_SHA_OK:       state = "OK";    col = GREEN_T; break;
        case PP_SHA_MISMATCH: state = "WRONG"; col = AMBER;   break;
        case PP_SHA_DIRTY:    state = "DIRTY"; col = AMBER;   break;
        case PP_SHA_NONE:     break;
        default:              state = "?";     break;
        }
        if (state[0])
            x = text(x + 4, 111, state, ui_font_small, col, BG);
    }
    int h7_x = 468 - width_of("H7 -", ui_font_small);
    if (c->has_extra && e.dut_env[0] && x + 12 + width_of(e.dut_env, ui_font_small) + 12 <= h7_x)
        text(x + 12, 111, e.dut_env, ui_font_small, MUTED, BG);
    text(h7_x, 111, "H7 -", ui_font_small, MUTED, BG);
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

    int n = c->has_profile_count ? c->profile_count : 0;
    if (v.a->cursor < n && c->has_extra && v.a->cursor == c->extra.reset_index)
        text_r(468, 56, "START = reset DUT", ui_font_small, BLUE_T, BG);
    bool ready = app_start_allowed(v.a, v.a->cursor, v.now);
    const char *rs = ready ? "READY" : "NOT READY";
    uint16_t fill = ready ? GREEN_BG : AMBER_BG, dot = ready ? GREEN : AMBER, fg = ready ? GREEN_T : AMBER_T;
    int rw = 30 + width_of(rs, ui_font_bold) + 14;
    card(470 - rw, 62, rw, 30, 15, fill, BG);
    g->fillSmoothCircle(470 - rw + 18, 77, 5, dot, fill);
    text(470 - rw + 30, 83, rs, ui_font_bold, fg, fill);

    firmware_line(c);
    if (!n) {
        text(12, 140, c->has_profile_count ? "No tests on the RPi5" : "Loading test list...", ui_font_body,
             MUTED, BG);
        return;
    }
    const int rows = 4, row_h = 36, stride = 42, y0 = 121;
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
            card(12, y + 6, 4, 24, 2, BLUE, ROW_SEL);
            text(28, y + 24, ui_profile_name(c, i), ui_font_bold, WHITE, rowbg);
        } else {
            card(12, y, 456, row_h, 8, CARD, BG);
            text(28, y + 24, ui_profile_name(c, i), ui_font_body, ROWTXT, rowbg);
        }
        if (sel) {
            int bw = width_of("Selected", ui_font_small) + 24;
            card(458 - bw, y + 7, bw, 22, 11, BLUE_CHIP, rowbg);
            text(458 - bw + 12, y + 22, "Selected", ui_font_small, BLUE_PALE, BLUE_CHIP);
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

bool fault_active(const pp_client_t *c)
{
    return c->has_extra && (c->extra.flags & PP_X_FAULT_ACTIVE);
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
    snprintf(s, sizeof s, "In %s", phase_name(c->st.run_state));
    text(x + 14, 174, s, ui_font_small, MUTED, BG);

    if (fault_active(c)) {
        card_line(12, 262, 456, 24, 6, AMBER_BG, AMBER, BG);
        text_c(240, 280, c->extra.fault_text, ui_font_bold, AMBER_T, AMBER_BG);
    }

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
    stat_card(12, 104, "Data age", s, TXT);
    snprintf(s, sizeof s, "%u", c->st.lost);
    stat_card(124, 72, "Lost", s, c->st.lost ? AMBER_T : TXT);
    quality_text(c->st.quality, q, sizeof q);
    stat_card(204, 264, "Quality", q, c->st.quality ? AMBER_T : GREEN_T);
}

constexpr int PLOT_POINTS = 200;

/* pp_track_point() flattens received chunks. Keep the gaps for the pen-up moves. */
template<class Visit>
void visit_track(const pp_track_series_t &s, Visit visit)
{
    bool first = true;
    int seen = 0;
    for (int chunk = 0; chunk < s.count; chunk++) {
        if (!(s.received & (1u << chunk))) {
            first = true;
            continue;
        }
        for (int i = 0; i < s.n[chunk]; i++) {
            visit(s.points[chunk][i], first);
            first = false;
            if (++seen == PLOT_POINTS) return;
        }
    }
}

void result_plot(const pp_client_t *c, uint16_t run, int w)
{
    const int x = 12, y = 114, h = 116, pad = 10;
    const int cx = x + w / 2;
    card(x, y, w, h, 10, CARD, BG);
    if (!run || c->track_run_no != run) {
        text_c(cx, y + h / 2 + 5, "no trajectory", ui_font_small, MUTED, CARD);
        return;
    }

    const char *const labels[] = {"ref", "plant", "DUT"};
    const uint16_t colors[] = {BLUE_T, AMBER, GREEN_H};
    int xmin = INT16_MAX, xmax = INT16_MIN, ymin = INT16_MAX, ymax = INT16_MIN;
    int top = y + 28;
    for (int k = 0; k < PP_TRACK_SERIES; k++) {
        const pp_track_series_t &s = c->tracks[k];
        int lx = x + pad + k * (w - 2 * pad) / PP_TRACK_SERIES;
        g->fillRect(lx, y + 12, 9, 2, colors[k]);
        text(lx + 13, y + 18, labels[k], ui_font_small, TXT, CARD);
        if (!pp_track_complete(&s) || s.point_count > PLOT_POINTS) {
            text(lx, y + 33, "partial", ui_font_small, MUTED, CARD);
            top = y + 42;
        }
        visit_track(s, [&](const pp_point_t &p, bool) {
            xmin = std::min<int>(xmin, p.x_mm);
            xmax = std::max<int>(xmax, p.x_mm);
            ymin = std::min<int>(ymin, p.y_mm);
            ymax = std::max<int>(ymax, p.y_mm);
        });
    }
    const int bottom = y + h - pad;
    const float cy = (top + bottom) * 0.5f;
    if (xmin > xmax) {
        text_c(cx, (int)cy + 5, "no trajectory", ui_font_small, MUTED, CARD);
        return;
    }
    const float sx = xmax > xmin ? (float)(w - 2 * pad) / (xmax - xmin) : FLT_MAX;
    const float sy = ymax > ymin ? (float)(bottom - top) / (ymax - ymin) : FLT_MAX;
    const float scale = std::min(sx, sy);
    const float mid_x = (xmin + xmax) * 0.5f, mid_y = (ymin + ymax) * 0.5f;
    for (int k = 0; k < PP_TRACK_SERIES; k++) {
        int last_x = 0, last_y = 0;
        visit_track(c->tracks[k], [&](const pp_point_t &p, bool first) {
            int px = (int)(cx + (p.x_mm - mid_x) * scale + 0.5f);
            int py = (int)(cy - (p.y_mm - mid_y) * scale + 0.5f);
            if (first) g->drawPixel(px, py, colors[k]);
            else g->drawLine(last_x, last_y, px, py, colors[k]);
            last_x = px;
            last_y = py;
        });
    }
}

void result_check(const pp_client_t *c, uint16_t run)
{
    const pp_extra_t &e = c->extra;
    if (!run || !c->has_extra || !(e.flags & PP_X_CHECK_VALID) || e.check_run != run)
        return;
    char reason[48];
    const char *label;
    uint16_t color;
    switch (e.check_code) {
    case PP_CHK_CLEAN_NO_SPIFIX: label = "CLEAN (SPIfix not checked)"; color = AMBER_T; break;
    case PP_CHK_CLEAN:           label = "CLEAN"; color = GREEN_T; break;
    case PP_CHK_RUNNING:         label = "checking..."; color = MUTED; break;
    case PP_CHK_BAD:
        snprintf(reason, sizeof reason, "BAD: %s", e.check_text);
        label = reason; color = RED_T; break;
    case PP_CHK_INCONCLUSIVE:
        snprintf(reason, sizeof reason, "INCONCLUSIVE: %s", e.check_text);
        label = reason; color = AMBER_T; break;
    default: return;
    }
    text(12, 250, label, ui_font_small, color, BG);
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
    uint16_t run = app_view(v.a, v.now).result_run;
    if (run)
        snprintf(s, sizeof s, "Run #%u", run);
    else
        snprintf(s, sizeof s, "Run #?");
    int cw = width_of(s, ui_font_body) + 28;
    card(456 - cw, 60, cw, 28, 14, chipf, fill);
    text(456 - cw + 14, 80, s, ui_font_body, chipt, chipf);

    app_result_t r = app_result(v.a);
    const char *message = c->st.outcome != PP_OUT_COMPLETED ? "No result lines" :
                          r == APP_RES_WAITING ? "Waiting for result..." : "Result of another run (see PC)";
    const int lines = r == APP_RES_FRESH ? PP_RESULT_LINES : 1;
    const char *rows[PP_RESULT_LINES] = {message};
    int inner = 184;
    for (int k = 0; k < lines; k++) {
        if (r == APP_RES_FRESH) rows[k] = c->info.result[k];
        inner = std::max(inner, width_of(rows[k], ui_font_small));
    }
    int info_x = 468 - inner - 20;
    result_plot(c, run, info_x - 8 - 12);
    card(info_x, 114, inner + 20, 116, 10, CARD, BG);
    for (int k = 0; k < lines; k++) {
        const uint8_t *font = width_of(rows[k], ui_font_body) <= inner ? ui_font_body : ui_font_small;
        text(info_x + 10, lines == 1 ? 178 : 140 + 37 * k, rows[k], font,
             r == APP_RES_FRESH ? TXT : MUTED, CARD);
        if (k + 1 < lines) g->drawFastHLine(info_x + 10, 151 + 37 * k, inner, DIV);
    }
    result_check(c, run);

    quality_text(c->st.quality, q, sizeof q);
    snprintf(s, sizeof s, "Lost %u " DOT " Quality %s", c->st.lost, q);
    int w1 = width_of("START  Run again", ui_font_small) + 24;
    int w2 = width_of("STOP  Back", ui_font_small) + 24;
    int chips_x = 468 - w1 - 8 - w2;
    text(12, 272, s, ui_font_small, c->st.quality ? AMBER_T : MUTED, BG);
    if (12 + width_of(s, ui_font_small) + 12 <= chips_x) {      /* the hints only if they fit */
        card(chips_x, 256, w2, 24, 12, CHIP, BG);
        text(chips_x + 12, 273, "STOP  Back", ui_font_small, ROWTXT, CHIP);
        card(468 - w1, 256, w1, 24, 12, CHIP, BG);
        text(468 - w1 + 12, 273, "START  Run again", ui_font_small, ROWTXT, CHIP);
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
    text_c(240, 198, (c->has_info && c->info.error_text[0]) ? c->info.error_text : "(No text from the RPi5)",
           ui_font_body, RED_TXT, RED_DEEP);
    int w = width_of("STOP  Acknowledge", ui_font_small) + 32;
    card_line(240 - w / 2, 230, w, 30, 15, RED_BG, RED_CHIP, RED_DEEP);
    text_c(240, 250, "STOP  Acknowledge", ui_font_small, RED_PALE, RED_BG);
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
    if (s == APP_SCR_HOME) {
        bool ready = app_start_allowed(v.a, v.a->cursor, v.now);
        fnv(&ready, sizeof ready);
        if (c->has_extra) {
            const pp_extra_t &e = c->extra;
            fnv(&e.reset_index, sizeof e.reset_index);
            fnv(&e.sha_state, sizeof e.sha_state);
            fnv(e.dut_sha, strlen(e.dut_sha) + 1);
            fnv(e.dut_env, strlen(e.dut_env) + 1);
        }
    } else if (s == APP_SCR_RUN && fault_active(c)) {
        fnv(c->extra.fault_text, strlen(c->extra.fault_text) + 1);
    } else if (s == APP_SCR_RESULT) {
        fnv(&c->track_run_no, sizeof c->track_run_no);
        for (const pp_track_series_t &series : c->tracks) {
            fnv(&series.count, sizeof series.count);
            fnv(&series.received, sizeof series.received);
            fnv(&series.point_count, sizeof series.point_count);
            if (run && run == c->track_run_no) {
                /* A repeated chunk can replace its coordinates at the same point count. */
                fnv(series.n, series.count);
                for (int i = 0; i < series.count; i++)
                    fnv(series.points[i], series.n[i] * sizeof(pp_point_t));
            }
        }
        fnv(&c->has_extra, sizeof c->has_extra);
        if (c->has_extra) {
            const pp_extra_t &e = c->extra;
            uint8_t valid = e.flags & PP_X_CHECK_VALID;
            fnv(&valid, sizeof valid);
            fnv(&e.check_run, sizeof e.check_run);
            fnv(&e.check_code, sizeof e.check_code);
            fnv(e.check_text, strlen(e.check_text) + 1);
        }
    } else if (s == APP_SCR_NOLINK && app_proto_mismatch(v.a, v.now)) {
        fnv(&v.a->proto_foreign_ver, sizeof v.a->proto_foreign_ver);
    }
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
