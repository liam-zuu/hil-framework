/*
 * panel_app.h — what the panel does with buttons and what screen it shows, as plain C (no
 * Arduino), so it runs in the firmware and in the PC tests (panel/test_host).
 *
 * Codex round 1 (Liam approved 04/11: "duyệt sửa cả 16") and round 2 (Liam 04/11: "sửa xong
 * mở vòng 3"):
 *  P1      STOP always goes to the protocol client, on every screen, before anything else in
 *          the same button read; on RESULT it also closes the screen.
 *  P2, Q3  every other button is read against what the glass SHOWS (app_drawn: screen,
 *          cursor, session, list version, test of the RESULT shown); while a redraw that
 *          changes any of these is going on (app_drawing), only STOP is taken.
 *  P3, Q4  the cursor is kept inside the current list; SELECT/START/run-again only for a test
 *          whose name has arrived in the current list.
 *  P7, Q1  result lines are shown only when, after the current result generation began, the
 *          panel saw an INFO with the result cleared (contract v6: no result while a run is
 *          active) and then an INFO with a result. Otherwise "waiting" or "not verified".
 *  P8, Q2  a generation begins when the panel sends START (a failed START that is answered
 *          gives the closed state back), when a run is seen active, when the END phase start
 *          estimated from phase_ds jumps forward (a new END), and on a new session. A link gap
 *          alone does not begin one. "RESULT closed" belongs to one generation.
 */
#ifndef PANEL_APP_H
#define PANEL_APP_H

#include <stdbool.h>
#include <stdint.h>

#include "panel_proto.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    APP_EV_NAV      = 1 << 0,   /* NAV short press: next test */
    APP_EV_NAV_BACK = 1 << 1,   /* NAV held: previous test (repeats) */
    APP_EV_SELECT   = 1 << 2,
    APP_EV_START    = 1 << 3,
    APP_EV_STOP     = 1 << 4,
};

typedef enum { APP_SCR_NOLINK, APP_SCR_HOME, APP_SCR_RUN, APP_SCR_RESULT, APP_SCR_FAULT } app_screen_t;
typedef enum { APP_RES_NOT_VERIFIED, APP_RES_WAITING, APP_RES_FRESH } app_result_t;

#define APP_END_JUMP_MS 400u     /* END phase start moved forward by more: another END [ĐỀ XUẤT] */

/* what the glass shows: the inputs a button is read against */
typedef struct {
    app_screen_t screen;
    uint8_t  cursor;
    uint32_t sid;
    uint32_t profiles_gen;
    uint8_t  result_profile;     /* test of the RESULT shown (PP_PROFILE_NONE elsewhere) */
} app_view_t;

typedef struct {
    pp_client_t *c;
    uint8_t  cursor;                 /* test index under the cursor (HOME) */
    bool     drawn_valid;            /* false before the first drawing and during app_drawing */
    app_view_t drawn;
    uint32_t gen;                    /* result generation, starts at 1 */
    uint32_t info_count_at_gen;      /* c->info_count when the generation began */
    uint32_t dismissed_gen;          /* RESULT of this generation closed by the user; 0 = none */
    uint32_t start_gen;              /* generation begun by the START the panel sent last */
    bool     start_prev_dismissed;   /* the closed state that START replaced */
    bool     was_active;
    bool     have_end;
    uint32_t end_start_ms;           /* estimated start of the END phase seen last */
    uint32_t last_sid;
    char     msg[56];                /* feedback line */
} app_t;

void app_init(app_t *a, pp_client_t *c);
/* after the UART bytes and pp_client_tick of this pass, with the same now */
void app_update(app_t *a, uint32_t now);
app_screen_t app_screen(const app_t *a, uint32_t now);
bool app_ready(const app_t *a, uint32_t now);       /* START allowed: link + H7 answering */
bool app_profile_ok(const app_t *a, int i);         /* test i is in the list and its name arrived */
app_result_t app_result(const app_t *a);
bool app_result_fresh(const app_t *a);              /* app_result() == APP_RES_FRESH */
void app_buttons(app_t *a, uint8_t ev, uint32_t now);
/* the view a drawing would put on the glass, from the (copied) state */
app_view_t app_view(const app_t *a, uint32_t now);
bool app_view_same(const app_view_t *x, const app_view_t *y);
/* drawing side: app_drawing() before a redraw that changes the view, app_drawn() after it */
void app_drawing(app_t *a);
void app_drawn(app_t *a, const app_view_t *v);

#ifdef __cplusplus
}
#endif
#endif
