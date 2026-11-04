/*
 * panel_proto.h — panel side of the RPi5 daemon <-> panel UART protocol, version 6.
 *
 * Mirrors rpi5/panel_proto.py byte for byte (frame layouts, CRC, StreamParser) and rule for
 * rule (PanelClient). The Python file is the reference: its docstrings hold the contract;
 * this C code is checked against it by panel/test_host (golden bytes + the two-end
 * scenarios of rpi5/tests/test_panel_proto*.py run with this client).
 *
 * Plain C99, no Arduino / ESP-IDF calls, so the same file builds on the PC for the tests.
 * Frames are built in byte buffers (never a C struct sent as is). Times are uint32_t ms of a
 * free-running clock; only differences are used; calls into one client must come in time
 * order (never an older now after a newer one).
 *
 * Known, deliberate differences from the Python reference (Codex round 1):
 *  - text fields: a byte >= 0x80 becomes '?' (Python decodes it as U+FFFD). The contract says
 *    ASCII and the daemon encodes with errors='replace', so a conforming daemon never sends one
 *    (P11);
 *  - events (what the screen reports) wait in a ring of PP_EVENTS: the caller pops them after
 *    every call; if more pile up, the oldest are dropped and counted in ev_dropped (P18);
 *  - the link is marked expired in tick once NO_LINK_MS has passed, so a millis() wrap
 *    (49.7 days) cannot bring an old STATUS back to life (P9).
 */
#ifndef PANEL_PROTO_H
#define PANEL_PROTO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PP_VERSION        6

#define PP_START_CMD      0xCC
#define PP_START_STATUS   0xDD
#define PP_START_PROFILE  0xDE
#define PP_START_INFO     0xDF
#define PP_LEN_CMD        13
#define PP_LEN_STATUS     30
#define PP_LEN_PROFILE    28
#define PP_LEN_INFO       106
#define PP_LEN_MAX        PP_LEN_INFO
#define PP_NAME_LEN       18
#define PP_CTRL_LEN       8
#define PP_RESULT_LINES   3
#define PP_RESULT_LEN     20
#define PP_ERROR_TEXT_LEN 24

#define PP_PROFILE_MAX    254
#define PP_PROFILE_NONE   0xFF
#define PP_SEQ_END        0xFFFFu

/* timing [ĐỀ XUẤT] values of panel_proto.py */
#define PP_NO_LINK_MS       500u
#define PP_RESEND_MS        200u
#define PP_HELLO_RESEND_MS  200u
#define PP_CMD_MAX_AGE_MS   3000u
#define PP_PROFILE_WAIT_MS  1000u
#define PP_PARSER_STALE_MS  50u

/* commands */
#define PP_CMD_HELLO     0x01
#define PP_CMD_PROFILES  0x02
#define PP_CMD_SYNC      0x03
#define PP_CMD_SELECT    0x10
#define PP_CMD_START     0x20
#define PP_CMD_STOP      0x21

/* results */
#define PP_RES_NONE        0
#define PP_RES_OK          1
#define PP_RES_BUSY        2
#define PP_RES_BAD_PROFILE 3
#define PP_RES_NOT_IDLE    4
#define PP_RES_UNKNOWN     5
#define PP_RES_FAILED      6

/* run states */
enum { PP_ST_IDLE, PP_ST_PREPARE, PP_ST_ARM, PP_ST_RUN, PP_ST_STOPPING, PP_ST_END, PP_ST_ERROR };
#define PP_ST_IS_ACTIVE(s) ((s) == PP_ST_PREPARE || (s) == PP_ST_ARM || (s) == PP_ST_RUN || (s) == PP_ST_STOPPING)

/* outcomes */
enum { PP_OUT_NONE, PP_OUT_COMPLETED, PP_OUT_USER_STOP, PP_OUT_ERROR };

/* quality bits */
#define PP_Q_LOST         (1u << 0)
#define PP_Q_SP_STALE     (1u << 1)
#define PP_Q_SPI_STALE    (1u << 2)
#define PP_Q_DUT_UNCONF   (1u << 3)
#define PP_Q_DATA_INVALID (1u << 4)

/* link bits */
#define PP_L_RPI        (1u << 0)
#define PP_L_H7         (1u << 1)
#define PP_L_ESP32      (1u << 2)
#define PP_L_ESP32_KNOWN (1u << 3)

/* INFO flags and controller source (v6) */
#define PP_I_LOOP_VALID   (1u << 0)
#define PP_I_RESULT_VALID (1u << 1)
enum { PP_CTRL_SRC_NONE, PP_CTRL_SRC_PROFILE, PP_CTRL_SRC_FW };

typedef struct {
    uint32_t sid;
    uint16_t daemon_boot, panel_boot, last_cmd_seq;
    uint8_t  last_result;
    uint16_t next_seq;
    uint8_t  run_state;
    uint16_t phase_ds;
    uint8_t  profile, outcome, quality;
    uint16_t lost, data_age_ms;
    uint8_t  links, error_code, last_cmd;
} pp_status_t;

typedef struct {
    uint32_t sid;
    uint8_t  index, count;
    char     name[PP_NAME_LEN + 1];
} pp_profile_t;

typedef struct {
    uint32_t sid;
    uint16_t loop_us, overruns;
    uint8_t  flags, ctrl_src;
    char     ctrl[PP_CTRL_LEN + 1];
    char     result[PP_RESULT_LINES][PP_RESULT_LEN + 1];
    char     error_text[PP_ERROR_TEXT_LEN + 1];
} pp_info_t;

uint16_t pp_crc16(const uint8_t *data, size_t n);
/* out must hold PP_LEN_CMD bytes; returns PP_LEN_CMD */
size_t pp_encode_cmd(uint8_t *out, uint32_t sid, uint16_t seq, uint8_t cmd, uint16_t arg);
/* decoders take a complete frame already checked by the parser (start, version, CRC) */
void pp_decode_status(const uint8_t *f, pp_status_t *s);
void pp_decode_profile(const uint8_t *f, pp_profile_t *p);
void pp_decode_info(const uint8_t *f, pp_info_t *i);

/* ── byte stream -> frames (StreamParser) ── */
typedef void (*pp_frame_fn)(void *ctx, const uint8_t *frame, size_t len);

typedef struct {
    uint8_t  buf[PP_LEN_MAX];
    size_t   len;
    bool     have_t;
    uint32_t t_last_ms;        /* arrival time of the newest buffered byte */
    uint32_t stale_ms;
    uint8_t  accept;           /* PP_ACCEPT_* bits */
    uint32_t bad_frames, skipped_bytes, stale_drops;
} pp_parser_t;

#define PP_ACCEPT_CMD     (1u << 0)
#define PP_ACCEPT_STATUS  (1u << 1)
#define PP_ACCEPT_PROFILE (1u << 2)
#define PP_ACCEPT_INFO    (1u << 3)
#define PP_ACCEPT_PANEL   (PP_ACCEPT_STATUS | PP_ACCEPT_PROFILE | PP_ACCEPT_INFO)

void pp_parser_init(pp_parser_t *p, uint8_t accept, uint32_t stale_ms);
/* One call = one read from the UART (the stale check is done once per call, before the new
 * bytes, as StreamParser.feed). Every complete good frame is passed to fn. */
void pp_parser_feed(pp_parser_t *p, const uint8_t *data, size_t n, uint32_t now_ms,
                    pp_frame_fn fn, void *ctx);

/* ── panel state machine (PanelClient) ── */
typedef uint16_t (*pp_nonce_fn)(void *ctx);
typedef void (*pp_send_fn)(void *ctx, const uint8_t *frame, size_t len);

enum { PP_STOP_NONE, PP_STOP_REQUESTED, PP_STOP_ACKED, PP_STOP_CONFIRMED };
enum { PP_EV_DONE = 1, PP_EV_UNVERIFIED, PP_EV_REFUSED };

typedef struct {
    uint8_t what;      /* PP_EV_* */
    uint8_t cmd;
    bool    has_result;
    uint8_t result;
} pp_event_t;

#define PP_EVENTS 32
#define PP_PROFILES_STORED 255

typedef struct {
    bool     active;
    uint16_t seq;
    uint8_t  cmd;
    uint16_t arg;
    uint32_t t_first, t_sent;
} pp_pending_t;

typedef struct {
    bool     active;
    bool     has_seq;
    uint16_t seq;
    bool     sent;
    uint32_t t_sent;
} pp_stop_t;

typedef struct {
    pp_nonce_fn nonce;  void *nonce_ctx;
    pp_send_fn  send;   void *send_ctx;

    uint16_t boot;
    uint32_t sid;                       /* 0 = no session */
    bool     has_daemon_boot;  uint16_t daemon_boot;
    bool     has_next_seq;     uint16_t next_seq;
    bool     has_status_t;     uint32_t last_status_ms;
    pp_pending_t pending;               /* SELECT/START/SYNC in flight */
    bool     has_stale;        uint16_t stale;
    pp_stop_t stop;
    uint8_t  stop_state;                /* PP_STOP_* */
    uint32_t stop_failures;
    bool     fence;
    bool     has_t_session;    uint32_t t_session;
    bool     has_t_hello;      uint32_t t_hello;

    bool     has_profile_count; uint8_t profile_count;
    uint16_t profiles_have;             /* number of distinct indices received */
    bool     profile_ok[PP_PROFILES_STORED];
    char     profile_name[PP_PROFILES_STORED][PP_NAME_LEN + 1];

    bool     has_info;  pp_info_t info;     /* newest INFO of the session */
    uint32_t info_count;                    /* INFO frames of a session taken so far (never reset) */
    bool     has_st;    pp_status_t st;     /* newest STATUS of the session (for the screen) */

    pp_event_t events[PP_EVENTS];       /* what the screen reports, oldest first */
    uint8_t  ev_head, ev_count;
    uint32_t ev_dropped;
} pp_client_t;

void pp_client_init(pp_client_t *c, pp_nonce_fn nonce, void *nonce_ctx, pp_send_fn send, void *send_ctx);
bool pp_client_link(const pp_client_t *c, uint32_t now_ms);
/* user presses SELECT / START / STOP (NAV only moves the cursor and is never sent) */
void pp_client_press(pp_client_t *c, uint8_t cmd, uint16_t arg, uint32_t now_ms);
void pp_client_tick(pp_client_t *c, uint32_t now_ms);
/* a checked frame from the parser (STATUS / PROFILE / INFO) */
void pp_client_on_frame(pp_client_t *c, const uint8_t *frame, size_t len, uint32_t now_ms);
/* pops the oldest event; false when none */
bool pp_client_pop_event(pp_client_t *c, pp_event_t *ev);

#ifdef __cplusplus
}
#endif
#endif
