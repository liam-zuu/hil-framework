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

bool app_result_fresh(const app_t *a)
{
    const pp_client_t *c = a->c;
    return c->has_info && (c->info.flags & PP_I_RESULT_VALID) && c->info_count > a->info_count_at_gen;
}

static void new_gen(app_t *a)
{
    a->gen++;
    a->info_count_at_gen = a->c->info_count;
}

void app_init(app_t *a, pp_client_t *c)
{
    memset(a, 0, sizeof *a);
    a->c = c;
    a->gen = 1;
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
            if (ev.cmd == PP_CMD_START && ev.result == PP_RES_OK)
                new_gen(a);                     /* a new run was accepted */
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
    if (a->was_link && !link)
        new_gen(a);                             /* what comes after the gap is not trusted */
    a->was_link = link;
    bool active = link && c->has_st && PP_ST_IS_ACTIVE(c->st.run_state);
    if (active && !a->was_active)
        new_gen(a);                             /* a run became active */
    a->was_active = active;
    int n = profile_n(a);                       /* P3: keep the cursor in the list */
    if (n <= 0)
        a->cursor = 0;
    else if (a->cursor >= n)
        a->cursor = (uint8_t)(n - 1);
}

void app_drawn(app_t *a, app_screen_t s, uint8_t cursor)
{
    a->drawn_valid = true;
    a->drawn_screen = s;
    a->drawn_cursor = cursor;
}

static void start(app_t *a, int profile, uint32_t now)
{
    if (!app_ready(a, now))
        snprintf(a->msg, sizeof a->msg, "NOT READY: START not sent");
    else if (!app_profile_ok(a, profile))
        snprintf(a->msg, sizeof a->msg, "test list not complete: START not sent");
    else
        pp_client_press(a->c, PP_CMD_START, (uint16_t)profile, now);
}

void app_buttons(app_t *a, uint8_t ev, uint32_t now)
{
    app_screen_t s = app_screen(a, now);
    if (ev & APP_EV_STOP) {                     /* P1: always, first, nothing else this read */
        pp_client_press(a->c, PP_CMD_STOP, 0, now);
        if (s == APP_SCR_RESULT || (a->drawn_valid && a->drawn_screen == APP_SCR_RESULT))
            a->dismissed_gen = a->gen;
        return;
    }
    if (!ev)
        return;
    if (!a->drawn_valid || a->drawn_screen != s) {      /* P2: the screen shown is not the model */
        snprintf(a->msg, sizeof a->msg, "screen changing: press again");
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
            if (a->cursor != a->drawn_cursor) {         /* P2: the cursor moved, not drawn yet */
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
            start(a, a->c->st.profile, now);    /* run the same test again */
        break;
    default:                                    /* NO LINK, RUN, FAULT: only STOP does something */
        break;
    }
}
