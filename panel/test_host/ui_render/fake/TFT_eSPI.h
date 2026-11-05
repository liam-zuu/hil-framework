/* TFT_eSPI.h - PC stand-in for the few TFT_eSPI 2.5.43 calls of src/ui.cpp, drawing into a
 * 480x320 RGB565 buffer so the real ui.cpp can be rendered to PNG on the PC
 * (test_host/ui_render/render_screens.py). Not the library:
 *  - smooth text follows the library's rules (Extensions/Smooth_font.cpp: glyph top at
 *    cursor_y + ascent - dY, xAdvance, space = (ascent + descent) / 4, missing glyph = box,
 *    textWidth, alphaBlend of TFT_eSPI.cpp) on the same VLW arrays, so text sizes and places
 *    are those of the board;
 *  - smooth shapes (round rects, circles, arcs) are drawn by 4x4 supersampling, close to but
 *    not bit-exact with the library's anti-aliasing;
 *  - only TL_DATUM; the sprite is a plain buffer pushed to the screen buffer.
 */
#pragma once
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define TL_DATUM 0

class TFT_eSPI {
public:
    static const int WMAX = 480, HMAX = 320;
    uint16_t buf[HMAX][WMAX];
    int w = WMAX, h = HMAX;

    TFT_eSPI() { memset(buf, 0, sizeof buf); }
    static int &missing_glyphs() { static int n = 0; return n; }
    virtual ~TFT_eSPI() {}
    void init() {}
    void setRotation(int) {}
    int width() const { return w; }
    int height() const { return h; }

    static uint16_t alphaBlend(uint8_t alpha, uint16_t fgc, uint16_t bgc)
    {
        uint32_t rxb = bgc & 0xF81F;
        rxb += ((fgc & 0xF81F) - rxb) * (alpha >> 2) >> 6;
        uint32_t xgx = bgc & 0x07E0;
        xgx += ((fgc & 0x07E0) - xgx) * alpha >> 8;
        return (uint16_t)((rxb & 0xF81F) | (xgx & 0x07E0));
    }

    void drawPixel(int x, int y, uint16_t c)
    {
        if (x >= 0 && y >= 0 && x < w && y < h)
            buf[y][x] = c;
    }
    void fillRect(int x, int y, int ww, int hh, uint16_t c)
    {
        for (int j = y; j < y + hh; j++)
            for (int i = x; i < x + ww; i++)
                drawPixel(i, j, c);
    }
    void fillScreen(uint16_t c) { fillRect(0, 0, w, h, c); }
    void drawFastHLine(int x, int y, int ww, uint16_t c) { fillRect(x, y, ww, 1, c); }
    void drawRect(int x, int y, int ww, int hh, uint16_t c)
    {
        fillRect(x, y, ww, 1, c); fillRect(x, y + hh - 1, ww, 1, c);
        fillRect(x, y, 1, hh, c); fillRect(x + ww - 1, y, 1, hh, c);
    }

    /* ---- smooth shapes: coverage by 4x4 samples, blended on the given bg ---- */
    template <class F> void cover(int x0, int y0, int x1, int y1, uint16_t fg, uint16_t bg, F inside)
    {
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++) {
                int n = 0;
                for (int sy = 0; sy < 4; sy++)
                    for (int sx = 0; sx < 4; sx++)
                        n += inside(x + (sx + 0.5) / 4.0, y + (sy + 0.5) / 4.0) ? 1 : 0;
                if (n == 16) drawPixel(x, y, fg);
                else if (n) drawPixel(x, y, alphaBlend((uint8_t)(n * 255 / 16), fg, bg));
            }
    }
    void fillSmoothRoundRect(int x, int y, int ww, int hh, int r, uint32_t color, uint32_t bg = 0x00FFFFFF)
    {
        if (r > ww / 2) r = ww / 2;
        if (r > hh / 2) r = hh / 2;
        double l = x, t = y, rr = x + ww, b = y + hh;
        cover(x, y, x + ww - 1, y + hh - 1, (uint16_t)color, (uint16_t)bg, [&](double px, double py) {
            double cx = px < l + r ? l + r : px > rr - r ? rr - r : px;
            double cy = py < t + r ? t + r : py > b - r ? b - r : py;
            double dx = px - cx, dy = py - cy;
            return dx * dx + dy * dy <= (double)r * r;
        });
    }
    void fillSmoothCircle(int x, int y, int r, uint32_t color, uint32_t bg = 0x00FFFFFF)
    {
        double cx = x + 0.5, cy = y + 0.5;
        cover(x - r - 1, y - r - 1, x + r + 1, y + r + 1, (uint16_t)color, (uint16_t)bg,
              [&](double px, double py) { return (px - cx) * (px - cx) + (py - cy) * (py - cy) <= (r + 0.5) * (r + 0.5); });
    }
    /* angle 0 = 6 o'clock, clockwise (TFT_eSPI.h) */
    void drawSmoothArc(int x, int y, int r, int ir, uint32_t a0, uint32_t a1, uint32_t fg, uint32_t bg, bool = false)
    {
        double cx = x + 0.5, cy = y + 0.5;
        cover(x - r - 1, y - r - 1, x + r + 1, y + r + 1, (uint16_t)fg, (uint16_t)bg, [&](double px, double py) {
            double dx = px - cx, dy = py - cy, d2 = dx * dx + dy * dy;
            if (d2 > (r + 0.5) * (r + 0.5) || d2 < (ir - 0.5) * (ir - 0.5)) return false;
            if (a0 == 0 && a1 >= 360) return true;
            double a = atan2(-dx, dy) * 180.0 / 3.14159265358979323846;
            if (a < 0) a += 360;
            return a0 <= a1 ? (a >= a0 && a <= a1) : (a >= a0 || a <= a1);
        });
    }

    /* ---- smooth font (VLW array), as Extensions/Smooth_font.cpp ---- */
    struct Glyph { uint16_t code; uint8_t h, w, adv; int16_t dy; int8_t dx; uint32_t bmp; };
    const uint8_t *font = nullptr;
    Glyph glyph[256];
    int gcount = 0, ascent = 0, descent = 0, space = 0;
    uint16_t fgc = 0xFFFF, bgc = 0;
    int cursor_x = 0, cursor_y = 0;

    static uint32_t be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3]; }
    void loadFont(const uint8_t *a)
    {
        font = a;
        gcount = (int)be32(a);
        ascent = (int)be32(a + 16);
        descent = (int)be32(a + 20);
        space = (ascent + descent) / 4;
        uint32_t bmp = 24 + gcount * 28;
        for (int i = 0; i < gcount && i < 256; i++) {
            const uint8_t *m = a + 24 + i * 28;
            Glyph &g = glyph[i];
            g.code = (uint16_t)be32(m); g.h = (uint8_t)be32(m + 4); g.w = (uint8_t)be32(m + 8);
            g.adv = (uint8_t)be32(m + 12); g.dy = (int16_t)be32(m + 16); g.dx = (int8_t)be32(m + 20);
            g.bmp = bmp;
            bmp += g.w * g.h;
        }
    }
    void unloadFont() { font = nullptr; }
    const Glyph *find(uint16_t code) const
    {
        for (int i = 0; i < gcount; i++)
            if (glyph[i].code == code) return &glyph[i];
        return nullptr;
    }
    static uint16_t utf8(const char *&s)
    {
        uint8_t c = (uint8_t)*s++;
        if ((c & 0xE0) == 0xC0 && (*s & 0xC0) == 0x80) return (uint16_t)(((c & 0x1F) << 6) | ((uint8_t)*s++ & 0x3F));
        return c;
    }
    void setTextColor(uint16_t fg, uint16_t bg, bool = false) { fgc = fg; bgc = bg; }
    void setTextDatum(uint8_t) {}
    int16_t textWidth(const char *s)
    {
        int32_t wdt = 0;
        while (*s) {
            uint16_t u = utf8(s);
            if (u == 0x20) { wdt += space; continue; }
            const Glyph *g = find(u);
            if (g) {
                if (wdt == 0 && g->dx < 0) wdt -= g->dx;
                if (*s) wdt += g->adv;
                else wdt += g->dx + g->w;
            } else wdt += space + 1;
        }
        return (int16_t)wdt;
    }
    void drawGlyph(uint16_t code)
    {
        if (code == 0x20) { cursor_x += space; return; }
        const Glyph *g = find(code);
        if (!g) {
            missing_glyphs()++;                                  /* counted for the test */
            drawRect(cursor_x, cursor_y, space, ascent, fgc);    /* maxAscent == ascent */
            cursor_x += space + 1;
            return;
        }
        if (cursor_x == 0) cursor_x -= g->dx;
        int cy = cursor_y + ascent - g->dy, cx = cursor_x + g->dx;
        for (int y = 0; y < g->h; y++)
            for (int x = 0; x < g->w; x++) {
                uint8_t p = font[g->bmp + x + g->w * y];
                if (p == 0xFF) drawPixel(x + cx, y + cy, fgc);
                else if (p) drawPixel(x + cx, y + cy, alphaBlend(p, fgc, bgc));
            }
        cursor_x += g->adv;
    }
    int16_t drawString(const char *s, int x, int y)
    {
        cursor_x = x; cursor_y = y;
        while (*s) drawGlyph(utf8(s));
        return (int16_t)(cursor_x - x);
    }
};

class TFT_eSprite : public TFT_eSPI {
public:
    TFT_eSPI *parent;
    explicit TFT_eSprite(TFT_eSPI *p) : parent(p) {}
    void setColorDepth(int) {}
    void *createSprite(int ww, int hh) { w = ww; h = hh; return buf; }
    void pushSprite(int, int) { memcpy(parent->buf, buf, sizeof buf); }
};
