/*
 * panel_proto.c — see panel_proto.h. Every rule below is a line-by-line mirror of
 * rpi5/panel_proto.py (PanelClient, StreamParser); comments name the Python rule.
 */
#include "panel_proto.h"

#include <string.h>

/* ── bytes ── */

uint16_t pp_crc16(const uint8_t *data, size_t n)
{
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < n; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; b++)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
    return crc;
}

static void put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put32(uint8_t *p, uint32_t v) { put16(p, (uint16_t)v); put16(p + 2, (uint16_t)(v >> 16)); }
static uint16_t get16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t get32(const uint8_t *p) { return (uint32_t)get16(p) | ((uint32_t)get16(p + 2) << 16); }

size_t pp_encode_cmd(uint8_t *out, uint32_t sid, uint16_t seq, uint8_t cmd, uint16_t arg)
{
    out[0] = PP_START_CMD;
    out[1] = PP_VERSION;
    put32(out + 2, sid);
    put16(out + 6, seq);
    out[8] = cmd;
    put16(out + 9, arg);
    put16(out + 11, pp_crc16(out, 11));
    return PP_LEN_CMD;
}

void pp_decode_status(const uint8_t *f, pp_status_t *s)
{
    s->sid = get32(f + 2);
    s->daemon_boot = get16(f + 6);
    s->panel_boot = get16(f + 8);
    s->last_cmd_seq = get16(f + 10);
    s->last_result = f[12];
    s->next_seq = get16(f + 13);
    s->run_state = f[15];
    s->phase_ds = get16(f + 16);
    s->profile = f[18];
    s->outcome = f[19];
    s->quality = f[20];
    s->lost = get16(f + 21);
    s->data_age_ms = get16(f + 23);
    s->links = f[25];
    s->error_code = f[26];
    s->last_cmd = f[27];
    s->run_no = get16(f + 28);
}

/* ASCII field, NUL padded -> C string cut at the first NUL (Python: split(b'\x00')[0]).
 * Bytes >= 0x80 become '?' (Python decodes them with errors='replace'). */
static void text(char *dst, const uint8_t *src, size_t n)
{
    size_t i = 0;
    for (; i < n && src[i]; i++)
        dst[i] = (src[i] < 0x80) ? (char)src[i] : '?';
    dst[i] = '\0';
}

void pp_decode_profile(const uint8_t *f, pp_profile_t *p)
{
    p->sid = get32(f + 2);
    p->index = f[6];
    p->count = f[7];
    text(p->name, f + 8, PP_NAME_LEN);
}

void pp_decode_info(const uint8_t *f, pp_info_t *i)
{
    i->sid = get32(f + 2);
    i->loop_us = get16(f + 6);
    i->overruns = get16(f + 8);
    i->flags = f[10];
    i->ctrl_src = f[11];
    text(i->ctrl, f + 12, PP_CTRL_LEN);
    for (int k = 0; k < PP_RESULT_LINES; k++)
        text(i->result[k], f + 20 + k * PP_RESULT_LEN, PP_RESULT_LEN);
    i->result_run = get16(f + 80);
    text(i->error_text, f + 82, PP_ERROR_TEXT_LEN);
}

/* ── StreamParser ── */

static size_t frame_len(uint8_t start)
{
    switch (start) {
    case PP_START_CMD:     return PP_LEN_CMD;
    case PP_START_STATUS:  return PP_LEN_STATUS;
    case PP_START_PROFILE: return PP_LEN_PROFILE;
    case PP_START_INFO:    return PP_LEN_INFO;
    default:               return 0;
    }
}

static bool accepted(const pp_parser_t *p, uint8_t start)
{
    switch (start) {        /* start bytes are 0xCC, 0xDD, 0xDE, 0xDF: not contiguous */
    case PP_START_CMD:     return (p->accept & PP_ACCEPT_CMD) != 0;
    case PP_START_STATUS:  return (p->accept & PP_ACCEPT_STATUS) != 0;
    case PP_START_PROFILE: return (p->accept & PP_ACCEPT_PROFILE) != 0;
    case PP_START_INFO:    return (p->accept & PP_ACCEPT_INFO) != 0;
    default:               return false;
    }
}

void pp_parser_init(pp_parser_t *p, uint8_t accept, uint32_t stale_ms)
{
    memset(p, 0, sizeof *p);
    p->accept = accept;
    p->stale_ms = stale_ms;
}

static void drop(pp_parser_t *p, size_t k)
{
    memmove(p->buf, p->buf + k, p->len - k);
    p->len -= k;
}

/* the while loop of StreamParser.feed over what is buffered */
static void scan(pp_parser_t *p, pp_frame_fn fn, void *ctx)
{
    while (p->len) {
        if (!accepted(p, p->buf[0])) {
            drop(p, 1);
            p->skipped_bytes++;
            continue;
        }
        if (p->len >= 2 && p->buf[1] != PP_VERSION) {
            drop(p, 1);
            p->bad_frames++;
            continue;
        }
        size_t n = frame_len(p->buf[0]);
        if (p->len < n)
            break;
        if (pp_crc16(p->buf, n - 2) != get16(p->buf + n - 2)) {
            p->bad_frames++;
            drop(p, 1);
            continue;
        }
        uint8_t frame[PP_LEN_MAX];
        memcpy(frame, p->buf, n);
        drop(p, n);
        if (fn)
            fn(ctx, frame, n);
    }
}

void pp_parser_feed(pp_parser_t *p, const uint8_t *data, size_t n, uint32_t now_ms,
                    pp_frame_fn fn, void *ctx)
{
    if (p->len && p->have_t && (uint32_t)(now_ms - p->t_last_ms) > p->stale_ms) {
        p->stale_drops++;               /* a cut frame (panel reset, lost bytes) */
        p->len = 0;
    }
    if (n) {
        p->t_last_ms = now_ms;
        p->have_t = true;
    }
    /* byte by byte: the buffer never holds more than one frame, and the decisions are the
     * same as scanning the whole read at once (they depend only on the buffered prefix) */
    for (size_t i = 0; i < n; i++) {
        p->buf[p->len++] = data[i];
        scan(p, fn, ctx);
    }
}

/* ── PanelClient ── */

static void event(pp_client_t *c, uint8_t what, uint8_t cmd, bool has_result, uint8_t result)
{
    if (c->ev_count == PP_EVENTS) {     /* full: the oldest goes */
        c->ev_head = (uint8_t)((c->ev_head + 1) % PP_EVENTS);
        c->ev_count--;
        c->ev_dropped++;
    }
    pp_event_t *e = &c->events[(c->ev_head + c->ev_count) % PP_EVENTS];
    e->what = what;
    e->cmd = cmd;
    e->has_result = has_result;
    e->result = has_result ? result : 0;
    c->ev_count++;
}

bool pp_client_pop_event(pp_client_t *c, pp_event_t *ev)
{
    if (!c->ev_count)
        return false;
    *ev = c->events[c->ev_head];
    c->ev_head = (uint8_t)((c->ev_head + 1) % PP_EVENTS);
    c->ev_count--;
    return true;
}

static void send_cmd(pp_client_t *c, uint32_t sid, uint16_t seq, uint8_t cmd, uint16_t arg)
{
    uint8_t f[PP_LEN_CMD];
    pp_encode_cmd(f, sid, seq, cmd, arg);
    if (c->send)
        c->send(c->send_ctx, f, sizeof f);
}

/* _new_boot: a non-zero nonce different from the current boot */
static uint16_t new_boot(pp_client_t *c)
{
    for (;;) {
        uint16_t b = c->nonce(c->nonce_ctx);
        if (b && b != c->boot)
            return b;
    }
}

static void clear_profiles(pp_client_t *c)
{
    c->profiles_gen++;
    c->has_profile_count = false;
    c->profile_count = 0;
    c->profiles_have = 0;
    memset(c->profile_ok, 0, sizeof c->profile_ok);
}

void pp_client_init(pp_client_t *c, pp_nonce_fn nonce, void *nonce_ctx, pp_send_fn send, void *send_ctx)
{
    memset(c, 0, sizeof *c);
    c->nonce = nonce;
    c->nonce_ctx = nonce_ctx;
    c->send = send;
    c->send_ctx = send_ctx;
    c->boot = new_boot(c);
    c->stop_state = PP_STOP_NONE;
}

bool pp_client_link(const pp_client_t *c, uint32_t now_ms)
{
    return c->has_status_t && (uint32_t)(now_ms - c->last_status_ms) < PP_NO_LINK_MS;
}

/* _give_up */
static void give_up(pp_client_t *c, const pp_pending_t *p)
{
    event(c, PP_EV_UNVERIFIED, p->cmd, false, 0);
    if (!c->has_next_seq || c->next_seq <= p->seq) {
        if (!c->has_stale || p->seq > c->stale)
            c->stale = p->seq;
        c->has_stale = true;
    }
}

void pp_client_tick(pp_client_t *c, uint32_t now_ms)
{
    if (c->has_status_t && (uint32_t)(now_ms - c->last_status_ms) >= PP_NO_LINK_MS)
        c->has_status_t = false;                /* expired: only a new STATUS opens it (P9) */
    if (!c->sid) {
        if (!c->has_t_hello || (uint32_t)(now_ms - c->t_hello) >= PP_HELLO_RESEND_MS) {
            c->t_hello = now_ms;
            c->has_t_hello = true;
            send_cmd(c, 0, 0, PP_CMD_HELLO, c->boot);
        }
        return;
    }
    if (c->stop.active && c->has_next_seq) {
        if (!c->stop.has_seq) {
            c->stop.seq = c->has_stale ? (uint16_t)(c->stale + 1) : c->next_seq;
            c->stop.has_seq = true;
        }
        if (!c->stop.sent || (uint32_t)(now_ms - c->stop.t_sent) >= PP_RESEND_MS) {
            c->stop.t_sent = now_ms;
            c->stop.sent = true;
            send_cmd(c, c->sid, c->stop.seq, PP_CMD_STOP, 0);
        }
    }
    if (c->pending.active && c->pending.cmd != PP_CMD_SYNC &&
        (uint32_t)(now_ms - c->pending.t_first) >= PP_CMD_MAX_AGE_MS) {
        pp_pending_t p = c->pending;
        c->pending.active = false;
        give_up(c, &p);
    }
    bool link = pp_client_link(c, now_ms);
    if (!c->pending.active && !c->stop.active && link && (c->has_stale || c->fence)) {
        uint16_t seq = c->has_stale ? (uint16_t)(c->stale + 1) : c->next_seq;
        c->pending = (pp_pending_t){ true, seq, PP_CMD_SYNC, 0, now_ms, now_ms };
        send_cmd(c, c->sid, seq, PP_CMD_SYNC, 0);
    } else if (c->pending.active && link && (uint32_t)(now_ms - c->pending.t_sent) >= PP_RESEND_MS) {
        c->pending.t_sent = now_ms;
        send_cmd(c, c->sid, c->pending.seq, c->pending.cmd, c->pending.arg);
    }
    bool incomplete = !c->has_profile_count || c->profiles_have < c->profile_count;
    if (incomplete && c->has_t_session && (uint32_t)(now_ms - c->t_session) >= PP_PROFILE_WAIT_MS) {
        c->t_session = now_ms;
        send_cmd(c, c->sid, 0, PP_CMD_PROFILES, 0);
    }
}

void pp_client_press(pp_client_t *c, uint8_t cmd, uint16_t arg, uint32_t now_ms)
{
    if (cmd == PP_CMD_STOP) {
        if (c->stop.active) {                   /* one STOP in flight already */
            pp_client_tick(c, now_ms);
            return;
        }
        bool has_seq = false;
        uint16_t seq = 0;
        if (c->pending.active && c->pending.cmd == PP_CMD_SYNC) {
            seq = c->pending.seq;               /* STOP takes the SYNC's place */
            has_seq = true;
        } else if (c->pending.active) {
            pp_pending_t p = c->pending;
            give_up(c, &p);
            seq = (uint16_t)(p.seq + 1);
            has_seq = true;
        }
        c->pending.active = false;
        c->stop = (pp_stop_t){ true, has_seq, seq, false, 0 };
        c->stop_state = PP_STOP_REQUESTED;
        c->stop_failures = 0;
        pp_client_tick(c, now_ms);
        return;
    }
    if (!c->sid || !c->has_next_seq || !pp_client_link(c, now_ms) || c->pending.active ||
        c->stop.active || c->has_stale || c->fence || c->next_seq >= PP_SEQ_END - 1) {
        event(c, PP_EV_REFUSED, cmd, false, 0);     /* busy / no link: the user retries */
        return;
    }
    if (cmd == PP_CMD_START)
        c->stop_state = PP_STOP_NONE;               /* V5: a new run, no stop yet */
    c->pending = (pp_pending_t){ true, c->next_seq, cmd, arg, now_ms, now_ms };
    send_cmd(c, c->sid, c->next_seq, cmd, arg);
}

/* _lose_session */
static void lose_session(pp_client_t *c, uint32_t now_ms)
{
    if (c->pending.active && c->pending.cmd != PP_CMD_SYNC)
        event(c, PP_EV_UNVERIFIED, c->pending.cmd, false, 0);
    c->pending.active = false;
    c->has_stale = false;                       /* frames of the old sid are dropped */
    if (c->stop.active)
        c->stop = (pp_stop_t){ true, false, 0, false, 0 };    /* sent again in the new session */
    else if (c->stop_state == PP_STOP_CONFIRMED)
        c->stop_state = PP_STOP_NONE;           /* 'acked' kept: confirmed in the new session (W2) */
    c->fence = false;
    c->boot = new_boot(c);
    c->sid = 0;
    c->has_next_seq = false;
    c->has_daemon_boot = false;
    clear_profiles(c);
    c->has_t_session = false;
    c->has_info = false;
    c->has_st = false;
    c->has_t_hello = false;
    pp_client_tick(c, now_ms);
}

static void on_status(pp_client_t *c, const pp_status_t *st, uint32_t now_ms)
{
    if (!c->sid) {
        if (st->sid && st->panel_boot == c->boot) {
            c->sid = st->sid;
            c->daemon_boot = st->daemon_boot;
            c->has_daemon_boot = true;
            c->t_session = now_ms;
            c->has_t_session = true;
            c->fence = true;
        } else {
            return;
        }
    } else if (!c->has_daemon_boot || st->daemon_boot != c->daemon_boot || st->sid != c->sid ||
               st->panel_boot != c->boot) {
        lose_session(c, now_ms);
        return;
    }
    c->last_status_ms = now_ms;
    c->has_status_t = true;
    c->next_seq = st->next_seq;
    c->has_next_seq = true;
    c->st = *st;
    c->has_st = true;
    bool acked = st->last_result != PP_RES_NONE;
#define IS_ACK(seq_, cmd_) (acked && st->last_cmd_seq == (seq_) && st->last_cmd == (cmd_))

    if (c->pending.active) {
        pp_pending_t p = c->pending;
        if (IS_ACK(p.seq, p.cmd)) {
            if (p.cmd != PP_CMD_SYNC)
                event(c, PP_EV_DONE, p.cmd, true, st->last_result);
            c->pending.active = false;
        } else if (st->next_seq > p.seq) {
            if (p.cmd != PP_CMD_SYNC)
                event(c, PP_EV_UNVERIFIED, p.cmd, false, 0);
            c->pending.active = false;
        }
        if (!c->pending.active && p.cmd == PP_CMD_SYNC)
            c->fence = false;                   /* older frames up to this seq are behind us */
    }
    if (c->has_stale && st->next_seq > c->stale)
        c->has_stale = false;                   /* the old frame can no longer run */
    if (c->stop.active && c->stop.has_seq) {
        if (IS_ACK(c->stop.seq, PP_CMD_STOP)) {
            if (st->last_result == PP_RES_OK || st->last_result == PP_RES_NOT_IDLE) {
                c->stop.active = false;
                c->stop_state = PP_STOP_ACKED;
                c->fence = false;               /* the STOP took the fence's place */
            } else if (c->stop.sent && (uint32_t)(now_ms - c->stop.t_sent) < PP_RESEND_MS) {
                /* same ACK seen again: one failure only */
            } else {
                c->stop_failures++;             /* resent by tick (daemon runs it again) */
            }
        } else if (st->next_seq > c->stop.seq) {
            c->stop = (pp_stop_t){ true, true, st->next_seq, false, 0 };  /* passed without ACK */
        }
    }
#undef IS_ACK
    if (c->stop_state == PP_STOP_ACKED && !PP_ST_IS_ACTIVE(st->run_state))
        c->stop_state = PP_STOP_CONFIRMED;
    else if (c->stop_state == PP_STOP_CONFIRMED && PP_ST_IS_ACTIVE(st->run_state))
        c->stop_state = PP_STOP_NONE;           /* V5: a new run is active */
    if (st->next_seq >= PP_SEQ_END ||
        (st->next_seq >= PP_SEQ_END - 1 && !c->stop.active && !c->pending.active && !c->has_stale)) {
        lose_session(c, now_ms);                /* after the ACK of the last seq */
        return;
    }
    pp_client_tick(c, now_ms);
}

void pp_client_on_frame(pp_client_t *c, const uint8_t *frame, size_t len, uint32_t now_ms)
{
    if (len < 2)
        return;
    if (frame[0] == PP_START_STATUS && len == PP_LEN_STATUS) {
        pp_status_t st;
        pp_decode_status(frame, &st);
        on_status(c, &st, now_ms);
    } else if (frame[0] == PP_START_PROFILE && len == PP_LEN_PROFILE) {
        pp_profile_t pr;
        pp_decode_profile(frame, &pr);
        if (c->sid && pr.sid == c->sid) {
            if (c->has_profile_count && pr.count != c->profile_count)
                clear_profiles(c);              /* another list: old names do not belong (Q4) */
            if (!c->has_profile_count || pr.count != c->profile_count)
                c->profiles_gen++;
            c->profile_count = pr.count;
            c->has_profile_count = true;
            if (pr.index < pr.count) {          /* an empty list is one PROFILE with count 0 */
                if (!c->profile_ok[pr.index]) {
                    c->profile_ok[pr.index] = true;
                    c->profiles_have++;
                    c->profiles_gen++;
                } else if (strcmp(c->profile_name[pr.index], pr.name) != 0) {
                    c->profiles_gen++;
                }
                memcpy(c->profile_name[pr.index], pr.name, sizeof pr.name);
            }
        }
    } else if (frame[0] == PP_START_INFO && len == PP_LEN_INFO) {
        pp_info_t in;
        pp_decode_info(frame, &in);
        if (c->sid && in.sid == c->sid) {
            c->info = in;
            c->has_info = true;
            c->info_count++;
        }
    }
}
