/* buzz_tool.c - see buzz_tool.h. */
#include "buzz_tool.h"

#include <string.h>

static const pb_event_t PREVIEW_LIST[] = {PB_BOOT, PB_START, PB_DONE, PB_STOPPED, PB_FAULT, PB_LINK_LOST};
#define PREVIEW_N ((uint8_t)(sizeof PREVIEW_LIST / sizeof PREVIEW_LIST[0]))

void bt_init(bt_t *t, uint32_t now)
{
    memset(t, 0, sizeof *t);
    t->mode = BT_QUIET;
    t->hz = PB_F0_DEFAULT;
    pb_init(&t->pb);
    pb_play(&t->pb, PB_BOOT, now);
}

static void nudge(bt_t *t, int d)
{
    int hz = (int)t->hz + d;
    if (hz < (int)PB_F0_MIN) hz = PB_F0_MIN;
    if (hz > (int)PB_F0_MAX) hz = PB_F0_MAX;
    t->hz = (uint16_t)hz;
}

/* sweep frequency at now, 0 = past the end */
static uint16_t sweep_hz(const bt_t *t, uint32_t now)
{
    uint32_t f = BT_SWEEP_LO + (uint32_t)(now - t->t0) / BT_SWEEP_MS * BT_SWEEP_STEP;
    return f > BT_SWEEP_HI ? 0 : (uint16_t)f;
}

static void quiet(bt_t *t)
{
    t->mode = BT_QUIET;
    pb_play(&t->pb, PB_NONE, 0);
}

/* play hz + d without a break (the start-up tune or the preview stops) */
static void tone_at(bt_t *t, int d)
{
    pb_play(&t->pb, PB_NONE, 0);
    nudge(t, d);
    t->mode = BT_TONE;
}

static void sweep(bt_t *t, uint32_t now)
{
    pb_play(&t->pb, PB_NONE, 0);
    t->mode = BT_SWEEP;
    t->t0 = now;
    t->hz = BT_SWEEP_LO;
}

static void preview(bt_t *t, uint32_t now)
{
    t->mode = BT_PREVIEW;
    pb_set_f0(&t->pb, t->hz);
    t->k = 0;
    t->gap = false;
    pb_play(&t->pb, PREVIEW_LIST[0], now);
}

/* STOP: SWEEP keeps the frequency heard (fine-tune it there); the rest goes quiet */
static void stop(bt_t *t, uint32_t now)
{
    if (t->mode == BT_SWEEP) {
        uint16_t f = sweep_hz(t, now);
        t->hz = f ? f : BT_SWEEP_HI;
        t->mode = BT_TONE;
    } else {
        quiet(t);
    }
}

void bt_buttons(bt_t *t, uint8_t ev, uint32_t now)
{
    bool down = ev & (BT_EV_NAV | BT_EV_NAV_BACK), up = ev & BT_EV_SELECT;
    int d = (up ? (int)BT_STEP_HZ : 0) - (down ? (int)BT_STEP_HZ : 0);
    switch (t->mode) {
    case BT_QUIET:
        if (ev & BT_EV_START)
            preview(t, now);
        else if (down || up)
            tone_at(t, d);
        break;
    case BT_TONE:
        if (ev & BT_EV_STOP)
            stop(t, now);
        else if (ev & BT_EV_START)
            sweep(t, now);
        else if (down || up)
            tone_at(t, d);
        break;
    case BT_SWEEP:
        if (ev & BT_EV_STOP)
            stop(t, now);
        else if (ev & BT_EV_START)
            sweep(t, now);
        break;
    case BT_PREVIEW:
        if (ev & BT_EV_STOP)
            stop(t, now);
        break;
    }
}

void bt_key(bt_t *t, char c, uint32_t now)
{
    if (c >= '0' && c <= '9') {
        if (t->ndig < 5) {
            t->num = t->num * 10u + (uint32_t)(c - '0');
            t->ndig++;
        }
        return;
    }
    if ((c == '\r' || c == '\n') && t->ndig) {
        uint32_t f = t->num;
        t->num = 0;
        t->ndig = 0;
        t->hz = (uint16_t)(f > PB_F0_MAX ? PB_F0_MAX : f);
        tone_at(t, 0);                            /* nudge(0) clamps below PB_F0_MIN */
        return;
    }
    t->num = 0;
    t->ndig = 0;
    switch (c) {
    case 's': case 'S':           sweep(t, now); break;
    case 'x': case 'X':           stop(t, now); break;
    case '+': case '=':           tone_at(t, (int)BT_STEP_HZ); break;
    case '-': case '_':           tone_at(t, -(int)BT_STEP_HZ); break;
    case 'p': case 'P':           preview(t, now); break;
    default:                      break;
    }
}

uint16_t bt_tone(bt_t *t, uint32_t now)
{
    switch (t->mode) {
    case BT_TONE:
        return t->hz;
    case BT_SWEEP: {
        uint16_t f = sweep_hz(t, now);
        if (!f) {
            quiet(t);
            return 0;
        }
        t->hz = f;
        return f;
    }
    case BT_PREVIEW:
        if (t->gap) {
            if ((uint32_t)(now - t->t0) < BT_GAP_MS)
                return 0;
            t->gap = false;
            pb_play(&t->pb, PREVIEW_LIST[t->k], now);
        }
        {
            uint16_t f = pb_tone(&t->pb, now);
            if (!pb_busy(&t->pb)) {
                t->k++;
                if (t->k >= PREVIEW_N) {
                    quiet(t);
                } else {
                    t->gap = true;
                    t->t0 = now;
                }
            }
            return f;
        }
    case BT_QUIET:
    default:
        return pb_tone(&t->pb, now);              /* the start-up tune, then nothing */
    }
}

const char *bt_mode_name(const bt_t *t)
{
    switch (t->mode) {
    case BT_TONE:    return "Tone";
    case BT_SWEEP:   return "Sweep";
    case BT_PREVIEW: return "Preview";
    default:         return "Quiet";
    }
}

const char *bt_pattern_name(const bt_t *t)
{
    if (t->mode != BT_PREVIEW || t->k >= PREVIEW_N)
        return "";
    return pb_name(PREVIEW_LIST[t->k]);
}

const char *bt_hint(const bt_t *t)
{
    switch (t->mode) {
    case BT_TONE:    return "NAV/SEL: -/+50 Hz  START: sweep  STOP: quiet";
    case BT_SWEEP:   return "STOP: keep this Hz  START: sweep again";
    case BT_PREVIEW: return "STOP: quiet";
    default:         return "NAV/SEL: -/+50 Hz tone  START: preview";
    }
}
