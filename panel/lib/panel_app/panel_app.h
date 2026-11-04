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
 *
 * Codex round 3 R1-R5 (Liam 04/11, PANEL session: "(a) Sửa gốc + R3"): the result generation
 * rules of P7/P8/Q1/Q2 (START, active run, END phase start from phase_ds, cleared INFO) are
 * gone. The daemon now numbers the runs in INFO (run_no, result_run), and the app keys
 * everything on the run:
 *  - the run of the END shown is info.run_no, known only while the newest INFO is current
 *    (pp_client_info_current: no run start and no STATUS gap seen since it arrived);
 *  - result lines are shown only when result_run == that run (APP_RES_FRESH); a known run
 *    without its result is "waiting"; while the run is not known, "waiting" too;
 *  - "RESULT closed" belongs to one run number. Closed while the run is not known: it is
 *    given to the run that becomes known next, unless a run start or a STATUS gap is seen
 *    first (then nothing is hidden). While the run is not known and no run start was seen
 *    since the last known run, the closed state of that run still holds (a link gap alone
 *    does not reopen a closed RESULT);
 *  - the view holds the run of the RESULT shown (R5); a new daemon (daemon_boot) forgets the
 *    closed state (run numbers start again);
 *  - R3: START no longer needs to know whether the press went out (nothing is begun on it).
 * Limit: a run whose every active STATUS is lost without a STATUS gap longer than
 * PP_INFO_GAP_MS, and whose every INFO is lost too, is not seen until an INFO of it arrives
 * (at most ~1 s later on a working link); until then the RESULT of the run before is shown,
 * under its own run number.
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

/* what the glass shows: the inputs a button is read against */
typedef struct {
    app_screen_t screen;
    uint8_t  cursor;
    uint32_t sid;
    uint32_t profiles_gen;
    uint8_t  result_profile;     /* test of the RESULT shown (PP_PROFILE_NONE elsewhere) */
    uint16_t result_run;         /* run of the RESULT shown; 0 = not known or not RESULT (R5) */
} app_view_t;

typedef struct {
    pp_client_t *c;
    uint8_t  cursor;                 /* test index under the cursor (HOME) */
    bool     drawn_valid;            /* false before the first drawing and during app_drawing */
    app_view_t drawn;
    bool     has_daemon_boot;  uint16_t daemon_boot;   /* run numbers belong to this daemon */
    uint16_t last_run;               /* run known last (0 = none) */
    uint32_t last_run_starts;        /* c->run_starts when last_run was known last */
    bool     closed_valid;  uint16_t closed_run;       /* RESULT of this run closed by the user */
    bool     closed_unknown;         /* closed while the run was not known */
    uint32_t closed_unknown_starts, closed_unknown_gaps;   /* c->run_starts, c->gaps then */
    char     msg[56];                /* feedback line */
} app_t;

void app_init(app_t *a, pp_client_t *c);
/* after the UART bytes and pp_client_tick of this pass, with the same now */
void app_update(app_t *a, uint32_t now);
app_screen_t app_screen(const app_t *a, uint32_t now);
bool app_ready(const app_t *a, uint32_t now);       /* START allowed: link + H7 answering */
bool app_profile_ok(const app_t *a, int i);         /* test i is in the list and its name arrived */
app_result_t app_result(const app_t *a);
uint16_t app_run(const app_t *a);                   /* run of the END shown; 0 = not known */
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
