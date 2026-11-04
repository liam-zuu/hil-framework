/* panel_app.c — see panel_app.h. */
#include "panel_app.h"

#include <stdio.h>
#include <string.h>

static int profile_n(const app_t *a)
{
    return a->c->has_profile_count ? a->c->profile_count : 0;
}

bool app_profile_ok(const app_t *a, int i)
{
    return i >= 0 && i < profile_n(a) && i < PP_PROFILES_STORED && a->c->profile_ok[i];
}

bool app_ready(const app_t *a, uint32_t now)
{
    const pp_client_t *c = a->c;
    return c->sid && pp_client_link(c, now) && c->has_st && (c->st.links & PP_L_H7);
}

app_screen_t app_screen(const app_t *a, uint32_t now)
{
    const pp_client_t *c = a->c;
    if (!c->sid || !c->has_st || !pp_client_link(c, now))
        return APP_SCR_NOLINK;
    if (c->st.run_state == PP_ST_ERROR)
        return APP_SCR_FAULT;
    if (PP_ST_IS_ACTIVE(c->st.run_state))
        return APP_SCR_RUN;
    if (c->st.run_state == PP_ST_END && a->dismissed_gen != a->gen)
        return APP_SCR_RESULT;
    return APP_SCR_HOME;
}

app_result_t app_result(const app_t *a)
{
    const pp_client_t *c = a->c;
    if (c->info_cleared_at <= a->info_count_at_gen)
        return APP_RES_NOT_VERIFIED;            /* no cleared INFO since this generation (Q1) */
    if (c->has_info && (c->info.flags & PP_I_RESULT_VALID))
        return APP_RES_FRESH;                   /* the newest INFO came after the clearing */
    return APP_RES_WAITING;
}

bool app_result_fresh(const app_t *a)
{
    return app_result(a) == APP_RES_FRESH;
}

static void new_gen(app_t *a)
{
    a->gen++;
    a->info_count_at_gen = a->c->info_count;
    a->have_end = false;            /* the next END seen belongs to this generation */
}

void app_init(app_t *a, pp_client_t *c)
{
    memset(a, 0, sizeof *a);
    a->c = c;
    a->gen = 1;
    a->drawn.result_profile = PP_PROFILE_NONE;
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

void app_update(app_t *a, uint32_t now)
{
    pp_client_t *c = a->c;
    pp_event_t ev;
    while (pp_client_pop_event(c, &ev)) {
        if (ev.what == PP_EV_DONE) {
            snprintf(a->msg, sizeof a->msg, "%s: %s", cmd_name(ev.cmd), res_name(ev.result));
            /* a START answered "did not start" gives back the closed state it replaced (Q2);
             * FAILED means "effect unknown": the run may exist, keep the new generation */
            if (ev.cmd == PP_CMD_START && ev.result != PP_RES_OK && ev.result != PP_RES_FAILED &&
                a->gen == a->start_gen && a->start_prev_dismissed)
                a->dismissed_gen = a->gen;
        } else if (ev.what == PP_EV_UNVERIFIED) {
            snprintf(a->msg, sizeof a->msg, "%s: not verified (may or may not have run)", cmd_name(ev.cmd));
        } else {
            snprintf(a->msg, sizeof a->msg, "%s refused: busy or no link, press again", cmd_name(ev.cmd));
        }
    }
    if (c->sid != a->last_sid) {                /* new session (or none) */
        a->last_sid = c->sid;
        new_gen(a);
    }
    bool link = c->sid && pp_client_link(c, now);
    bool active = link && c->has_st && PP_ST_IS_ACTIVE(c->st.run_state);
    if (active && !a->was_active)
        new_gen(a);                             /* a run became active */
    a->was_active = active;
    if (link && c->has_st && c->st.run_state == PP_ST_END && c->st.phase_ds != 0xFFFF) {
        uint32_t end_start = c->last_status_ms - (uint32_t)c->st.phase_ds * 100u;   /* phase_ds: 0.1 s */
        if (a->have_end && (int32_t)(end_start - a->end_start_ms) > (int32_t)APP_END_JUMP_MS)
            new_gen(a);                         /* another END than the one seen (Q2) */
        a->end_start_ms = end_start;
        a->have_end = true;
    }
    int n = profile_n(a);                       /* P3: keep the cursor in the list */
    if (n <= 0)
        a->cursor = 0;
    else if (a->cursor >= n)
        a->cursor = (uint8_t)(n - 1);
}

app_view_t app_view(const app_t *a, uint32_t now)
{
    app_view_t v;
    v.screen = app_screen(a, now);
    v.cursor = a->cursor;
    v.sid = a->c->sid;
    v.profiles_gen = a->c->profiles_gen;
    v.result_profile = (v.screen == APP_SCR_RESULT && a->c->has_st) ? a->c->st.profile : PP_PROFILE_NONE;
    return v;
}

bool app_view_same(const app_view_t *x, const app_view_t *y)
{
    return x->screen == y->screen && x->cursor == y->cursor && x->sid == y->sid &&
           x->profiles_gen == y->profiles_gen && x->result_profile == y->result_profile;
}

void app_drawing(app_t *a)
{
    a->drawn_valid = false;
}

void app_drawn(app_t *a, const app_view_t *v)
{
    a->drawn = *v;
    a->drawn_valid = true;
}

static void start(app_t *a, int profile, uint32_t now)
{
    if (!app_ready(a, now)) {
        snprintf(a->msg, sizeof a->msg, "NOT READY: START not sent");
        return;
    }
    if (!app_profile_ok(a, profile)) {
        snprintf(a->msg, sizeof a->msg, "test list not complete: START not sent");
        return;
    }
    bool was_dismissed = a->dismissed_gen == a->gen;
    pp_client_press(a->c, PP_CMD_START, (uint16_t)profile, now);
    if (a->c->pending.active && a->c->pending.cmd == PP_CMD_START) {   /* it went out (Q2) */
        new_gen(a);
        a->start_gen = a->gen;
        a->start_prev_dismissed = was_dismissed;
    }
}

void app_buttons(app_t *a, uint8_t ev, uint32_t now)
{
    app_screen_t s = app_screen(a, now);
    if (ev & APP_EV_STOP) {                     /* P1: always, first, nothing else this read */
        pp_client_press(a->c, PP_CMD_STOP, 0, now);
        if (s == APP_SCR_RESULT || (a->drawn_valid && a->drawn.screen == APP_SCR_RESULT))
            a->dismissed_gen = a->gen;
        return;
    }
    if (!ev)
        return;
    app_view_t v = app_view(a, now);
    if (!a->drawn_valid || a->drawn.screen != v.screen || a->drawn.sid != v.sid ||
        a->drawn.profiles_gen != v.profiles_gen || a->drawn.result_profile != v.result_profile) {
        snprintf(a->msg, sizeof a->msg, "screen changing: press again");   /* P2, Q3 */
        return;
    }
    int n = profile_n(a);
    switch (s) {
    case APP_SCR_HOME:
        if ((ev & APP_EV_NAV) && n > 0)
            a->cursor = (uint8_t)((a->cursor + 1) % n);
        if ((ev & APP_EV_NAV_BACK) && n > 0)
            a->cursor = (uint8_t)((a->cursor + n - 1) % n);
        if (ev & (APP_EV_SELECT | APP_EV_START)) {
            if (a->cursor != a->drawn.cursor) {         /* P2: the cursor moved, not drawn yet */
                snprintf(a->msg, sizeof a->msg, "cursor moved: press again");
            } else if (ev & APP_EV_SELECT) {
                if (app_profile_ok(a, a->cursor))
                    pp_client_press(a->c, PP_CMD_SELECT, a->cursor, now);
                else
                    snprintf(a->msg, sizeof a->msg, "test list not complete: SELECT not sent");
            } else {
                start(a, a->cursor, now);
            }
        }
        break;
    case APP_SCR_RESULT:
        if (ev & (APP_EV_NAV | APP_EV_NAV_BACK | APP_EV_SELECT))
            a->dismissed_gen = a->gen;          /* back to HOME, nothing sent */
        else if (ev & APP_EV_START)
            start(a, a->drawn.result_profile, now);     /* the test shown, again (Q3) */
        break;
    default:                                    /* NO LINK, RUN, FAULT: only STOP does something */
        break;
    }
}
