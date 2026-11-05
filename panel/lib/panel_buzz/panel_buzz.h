/* panel_buzz.h - buzzer of panel M3, part one (Liam 05/11, PANEL session: "Làm còi trước").
 * Plain C, tested on the PC (test_host/test_panel_buzz.py); the pin driving is in
 * src/buzzer.cpp. The buzzer is passive with a low-level trigger module (Liam 05/11:
 * "còi của tôi là passive, low level trigger"): it needs a square wave, and the pin must
 * stay HIGH when silent.
 *
 * What beeps (Claude's proposal, [ĐỀ XUẤT], change with Liam): only changes the RPi5 reports,
 * nothing the panel guesses, and nothing for the state already there when the link comes up
 * (boot, new session, link back):
 *  - PB_START      a new run number in an active state (run from the panel or from the PC)
 *  - PB_DONE       the run ended COMPLETED
 *  - PB_STOPPED    the run ended USER STOP
 *  - PB_FAULT      the run went to ERROR (or ended with outcome ERROR)
 *  - PB_LINK_LOST  the link to the RPi5 was up and is gone
 * A new event replaces the pattern still playing.
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { PB_NONE, PB_START, PB_DONE, PB_STOPPED, PB_FAULT, PB_LINK_LOST } pb_event_t;

typedef struct {
    uint16_t hz;               /* 0 = silent */
    uint16_t ms;               /* 0 = end of the pattern */
} pb_note_t;

typedef struct {
    bool     init;
    uint32_t sid;
    bool     link;
    uint8_t  state;
    uint16_t run_no;
    const pb_note_t *pat;      /* pattern playing, NULL = none */
    uint8_t  idx;
    uint32_t t_note;           /* start of note idx */
} pb_t;

void pb_init(pb_t *b);

/* Call on every link pass with what the client knows now. link = session up and STATUS
 * fresh (app_ready without the H7 part: sid && has_st && pp_client_link). Returns the event
 * found (and starts its pattern) or PB_NONE. */
pb_event_t pb_watch(pb_t *b, uint32_t sid, bool link, uint8_t run_state, uint16_t run_no,
                    uint8_t outcome, uint32_t now);

/* frequency to play now, 0 = silent */
uint16_t pb_tone(pb_t *b, uint32_t now);

/* the pattern of an event (for tests and for the wiring guide) */
const pb_note_t *pb_pattern(pb_event_t e);

#ifdef __cplusplus
}
#endif
