/* buzz_tool.h - logic of the buzzer tool, env panel_s3_buzz_test (decision 21, Liam 05/11:
 * "Duyệt, làm bước 1"): find the frequency the panel buzzer is loudest at (f0 of
 * panel_buzz.h), then listen to every pattern built on it. Plain C, tested on the PC
 * (test_host/test_panel_buzz.py); the screen and the pins are in src/buzz_test.cpp.
 *
 * Modes and buttons (the 4 panel buttons; bits as APP_EV_* of panel_app.h):
 *  QUIET    (after the start-up tune) silent.
 *           NAV / SELECT: -/+ BT_STEP_HZ and play the tone (TONE); START: PREVIEW at hz.
 *  TONE     plays hz without a break. NAV (also held, repeating) -BT_STEP_HZ, SELECT
 *           +BT_STEP_HZ; START: SWEEP; STOP: QUIET.
 *  SWEEP    BT_SWEEP_LO .. BT_SWEEP_HI by BT_SWEEP_STEP, BT_SWEEP_MS each, then QUIET.
 *           STOP: TONE at the frequency playing (fine-tune there); START: sweep again.
 *  PREVIEW  every pattern (pb_pattern) with f0 = hz, BT_GAP_MS between, then QUIET.
 *           STOP: QUIET.
 *
 * Keyboard on the serial port, for the bench without buttons (Liam 05/11: "tôi chưa nối nút
 * mà", then "thêm vào cách B đi"), bt_key, one character at a time, any mode:
 *  s          sweep from BT_SWEEP_LO
 *  x          stop: SWEEP -> TONE at the frequency playing; TONE, PREVIEW -> QUIET
 *  + (or =)   +BT_STEP_HZ, TONE        -  -BT_STEP_HZ, TONE
 *  digits + Enter (CR or LF)  play that frequency (clamped PB_F0_MIN..PB_F0_MAX), TONE
 *  p          PREVIEW at hz
 * Any other character drops the digits typed so far.
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>

#include "panel_buzz.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BT_EV_NAV       (1u << 0)      /* = APP_EV_NAV */
#define BT_EV_NAV_BACK  (1u << 1)      /* = APP_EV_NAV_BACK (NAV held, repeating) */
#define BT_EV_SELECT    (1u << 2)
#define BT_EV_START     (1u << 3)
#define BT_EV_STOP      (1u << 4)

#define BT_STEP_HZ      50u
#define BT_SWEEP_LO   1000u
#define BT_SWEEP_HI   4000u
#define BT_SWEEP_STEP  100u
#define BT_SWEEP_MS    500u
#define BT_GAP_MS     1200u

typedef enum { BT_QUIET, BT_TONE, BT_SWEEP, BT_PREVIEW } bt_mode_t;

typedef struct {
    bt_mode_t  mode;
    uint16_t   hz;            /* tone / sweep frequency, and f0 of the preview */
    uint32_t   t0;            /* SWEEP: start of the step; PREVIEW: end of the last pattern */
    uint8_t    k;             /* PREVIEW: index in the list */
    bool       gap;           /* PREVIEW: between two patterns */
    pb_t       pb;            /* start-up tune and preview patterns */
    uint32_t   num;           /* bt_key: digits typed so far */
    uint8_t    ndig;          /* how many (0 = none) */
} bt_t;

/* hz = PB_F0_DEFAULT, QUIET, the start-up tune playing */
void bt_init(bt_t *t, uint32_t now);
void bt_buttons(bt_t *t, uint8_t ev, uint32_t now);
void bt_key(bt_t *t, char c, uint32_t now);
/* frequency to play now (0 = silent); moves the sweep and the preview on */
uint16_t bt_tone(bt_t *t, uint32_t now);
/* what the screen says: mode name, pattern name in PREVIEW, button hints */
const char *bt_mode_name(const bt_t *t);
const char *bt_pattern_name(const bt_t *t);
const char *bt_hint(const bt_t *t);

#ifdef __cplusplus
}
#endif
