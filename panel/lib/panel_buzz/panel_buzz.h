/* panel_buzz.h - buzzer of panel M3, part one (Liam 05/11, PANEL session: "Làm còi trước").
 * Plain C, tested on the PC (test_host/test_panel_buzz.py); the pin driving is in
 * src/buzzer.cpp. The buzzer is passive with a low-level trigger module (Liam 05/11:
 * "còi của tôi là passive, low level trigger"): it needs a square wave, and the pin must
 * stay HIGH when silent.
 *
 * Every note is a fraction of one base frequency f0 (Liam 05/11, decision 21: "Duyệt, làm
 * bước 1"): a passive buzzer is loudest near its resonance, so f0 is meant to be the
 * frequency Liam finds loudest with the buzzer tool (env panel_s3_buzz_test, buzz_tool.h);
 * the patterns differ by rhythm and by going up or down, like the AGV buzzer. Until that
 * is measured f0 = PB_F0_DEFAULT (2700 Hz, the AGV value, not measured on this buzzer).
 *
 * What beeps (Claude's proposal, [ĐỀ XUẤT], change with Liam): only changes the RPi5 reports,
 * nothing the panel guesses, and nothing for the state already there when the link comes up
 * (boot, new session, link back):
 *  - PB_START      a new run number in an active state (run from the panel or from the PC)
 *  - PB_DONE       the run ended COMPLETED
 *  - PB_STOPPED    the run ended USER STOP
 *  - PB_FAULT      the run went to ERROR (or ended with outcome ERROR)
 *  - PB_LINK_LOST  the link to the RPi5 was up and is gone
 * plus PB_BOOT, the start-up tune, played by the firmware right at power-up (pb_play), not
 * by pb_watch. A new event replaces the pattern still playing (the start-up tune too).
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PB_F0_DEFAULT
#define PB_F0_DEFAULT 2700u        /* Hz; [ĐỀ XUẤT] until measured (step 2 of decision 21) */
#endif
#define PB_F0_MIN      500u        /* pb_set_f0 clamps to this range */
#define PB_F0_MAX     5000u

typedef enum { PB_NONE, PB_START, PB_DONE, PB_STOPPED, PB_FAULT, PB_LINK_LOST, PB_BOOT } pb_event_t;

typedef struct {
    uint16_t pm;               /* frequency in per mille of f0; 0 = silent */
    uint16_t ms;               /* 0 = end of the pattern */
} pb_note_t;

typedef struct {
    bool     init;
    uint32_t sid;
    bool     link;
    uint8_t  state;
    uint16_t run_no;
    uint16_t f0;               /* Hz */
    const pb_note_t *pat;      /* pattern playing, NULL = none */
    uint8_t  idx;
    uint32_t t_note;           /* start of note idx */
} pb_t;

/* f0 = PB_F0_DEFAULT, nothing playing */
void pb_init(pb_t *b);
void pb_set_f0(pb_t *b, uint16_t hz);

/* start the pattern of e now (replaces the one playing); PB_NONE stops it */
void pb_play(pb_t *b, pb_event_t e, uint32_t now);

/* true while a pattern plays (until pb_tone has passed its end) */
bool pb_busy(const pb_t *b);

/* Call on every link pass with what the client knows now. link = session up and STATUS
 * fresh (app_ready without the H7 part: sid && has_st && pp_client_link). Returns the event
 * found (and starts its pattern) or PB_NONE. Does not stop the pattern playing when it
 * finds nothing (the start-up tune keeps playing through the first STATUS). */
pb_event_t pb_watch(pb_t *b, uint32_t sid, bool link, uint8_t run_state, uint16_t run_no,
                    uint8_t outcome, uint32_t now);

/* frequency to play now, 0 = silent; round(f0 * pm / 1000) */
uint16_t pb_tone(pb_t *b, uint32_t now);

/* the pattern of an event (for tests and for the wiring guide) */
const pb_note_t *pb_pattern(pb_event_t e);
const char *pb_name(pb_event_t e);

#ifdef __cplusplus
}
#endif
