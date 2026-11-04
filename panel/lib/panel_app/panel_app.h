/*
 * panel_app.h — what the panel does with buttons and what screen it shows, as plain C (no
 * Arduino), so it runs in the firmware and in the PC tests (panel/test_host). Fixes of Codex
 * round 1 (Liam approved 04/11: "duyệt sửa cả 16"):
 *  P1  STOP always goes to the protocol client, on every screen, before anything else in the
 *      same button read; on RESULT it also closes the screen.
 *  P2  every other button is read against what the screen SHOWS (app_drawn), not the model:
 *      if the screen or the cursor changed since the last drawing, the press is not used.
 *  P3  the cursor is kept inside the current test list; SELECT/START/run-again only for a test
 *      whose name has arrived.
 *  P7  result lines are shown only from an INFO received after the current result generation
 *      began (new run seen, START acknowledged, link lost, new session).
 *  P8  "RESULT closed" belongs to one generation, not to the whole uptime.
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

typedef struct {
    pp_client_t *c;
    uint8_t  cursor;                 /* test index under the cursor (HOME) */
    bool     drawn_valid;            /* what the screen shows now (P2) */
    app_screen_t drawn_screen;
    uint8_t  drawn_cursor;
    uint32_t gen;                    /* result generation (P7, P8), starts at 1 */
    uint32_t info_count_at_gen;      /* c->info_count when the generation began */
    uint32_t dismissed_gen;          /* RESULT of this generation closed by the user; 0 = none */
    bool     was_active, was_link;
    uint32_t last_sid;
    char     msg[56];                /* feedback line */
} app_t;

void app_init(app_t *a, pp_client_t *c);
/* after the UART bytes and pp_client_tick of this pass, with the same now */
void app_update(app_t *a, uint32_t now);
app_screen_t app_screen(const app_t *a, uint32_t now);
bool app_ready(const app_t *a, uint32_t now);       /* START allowed: link + H7 answering */
bool app_profile_ok(const app_t *a, int i);         /* test i is in the list and its name arrived */
bool app_result_fresh(const app_t *a);              /* INFO result lines belong to this generation */
void app_buttons(app_t *a, uint8_t ev, uint32_t now);
/* the drawing side reports what it has just put on the glass */
void app_drawn(app_t *a, app_screen_t s, uint8_t cursor);

#ifdef __cplusplus
}
#endif
#endif
