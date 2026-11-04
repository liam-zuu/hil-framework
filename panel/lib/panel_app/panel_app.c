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

uint16_t app_run(const app_t *a)
{
    return a->c->has_st ? a->c->st.run_no : 0;
}

/* the RESULT of the END now is closed by the user */
static bool closed(const app_t *a)
{
    return a->c->has_st && a->closed_valid && a->closed_run == a->c->st.run_no;
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
    if (c->st.run_state == PP_ST_END && !closed(a))
        return APP_SCR_RESULT;
    return APP_SCR_HOME;
}

app_result_t app_result(const app_t *a)
{
    const pp_client_t *c = a->c;
    if (!c->has_st || !c->has_info || !(c->info.flags & PP_I_RESULT_VALID) || !c->info.result_run)
        return APP_RES_WAITING;                 /* no lines of any run */
    if (c->info.result_run == c->st.run_no)
        return APP_RES_FRESH;                   /* the lines of this run */
    return APP_RES_NOT_VERIFIED;                /* lines of another run: never shown here */
}

bool app_result_fresh(const app_t *a)
{
    return app_result(a) == APP_RES_FRESH;
}

static void close_result(app_t *a)
{
    if (!a->c->has_st)
        return;
    a->closed_run = a->c->st.run_no;
    a->closed_valid = true;
}

void app_init(app_t *a, pp_client_t *c)
{
    memset(a, 0, sizeof *a);
    a->c = c;
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
    (void)now;
    pp_client_t *c = a->c;
    pp_event_t ev;
    while (pp_client_pop_event(c, &ev)) {
        if (ev.what == PP_EV_DONE)
            snprintf(a->msg, sizeof a->msg, "%s: %s", cmd_name(ev.cmd), res_name(ev.result));
        else if (ev.what == PP_EV_UNVERIFIED)
            snprintf(a->msg, sizeof a->msg, "%s: not verified (may or may not have run)", cmd_name(ev.cmd));
        else
            snprintf(a->msg, sizeof a->msg, "%s refused: busy or no link, press again", cmd_name(ev.cmd));
    }
    if (c->has_daemon_boot && (!a->has_daemon_boot || c->daemon_boot != a->daemon_boot)) {
        a->has_daemon_boot = true;              /* another daemon: its runs count from 1 again */
        a->daemon_boot = c->daemon_boot;
        a->closed_valid = false;
    }
    if (c->has_st && a->closed_valid && c->st.run_no != a->closed_run)
        a->closed_valid = false;                /* another run seen: the closed one is over (S4) */
    int n = profile_n(a);                       /* P3: keep the cursor in the list */
    if (n <= 0)
        a->cursor = 0;
    else if (a->cursor >= n)
        a->cursor = (uint8_t)(n - 1);
}

app_view_t app_view(const app_t *a, uint32_t now)
{
    app_view_t v;
    memset(&v, 0, sizeof v);
    v.screen = app_screen(a, now);
    v.cursor = a->cursor;
    v.sid = a->c->sid;
    v.profiles_gen = a->c->profiles_gen;
    bool result = v.screen == APP_SCR_RESULT;
    v.result_profile = (result && a->c->has_st) ? a->c->st.profile : PP_PROFILE_NONE;
    v.result_run = result ? app_run(a) : 0;
    return v;
}

bool app_view_same(const app_view_t *x, const app_view_t *y)
{
    return x->screen == y->screen && x->cursor == y->cursor && x->sid == y->sid &&
           x->profiles_gen == y->profiles_gen && x->result_profile == y->result_profile &&
           x->result_run == y->result_run;
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
    pp_client_press(a->c, PP_CMD_START, (uint16_t)profile, now);   /* nothing to begin here (R3) */
}

void app_buttons(app_t *a, uint8_t ev, uint32_t now)
{
    app_screen_t s = app_screen(a, now);
    if (ev & APP_EV_STOP) {                     /* P1: always, first, nothing else this read */
        pp_client_press(a->c, PP_CMD_STOP, 0, now);
        if (s == APP_SCR_RESULT || (a->drawn_valid && a->drawn.screen == APP_SCR_RESULT))
            close_result(a);
        return;
    }
    if (!ev)
        return;
    app_view_t v = app_view(a, now);
    if (!a->drawn_valid || a->drawn.screen != v.screen || a->drawn.sid != v.sid ||
        a->drawn.profiles_gen != v.profiles_gen || a->drawn.result_profile != v.result_profile ||
        a->drawn.result_run != v.result_run) {
        snprintf(a->msg, sizeof a->msg, "screen changing: press again");   /* P2, Q3, R5 */
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
            close_result(a);                    /* back to HOME, nothing sent */
        else if (ev & APP_EV_START)
            start(a, a->drawn.result_profile, now);     /* the test shown, again (Q3) */
        break;
    default:                                    /* NO LINK, RUN, FAULT: only STOP does something */
        break;
    }
}
