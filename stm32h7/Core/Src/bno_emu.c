/*
 * bno_emu.c: BNO085 emulator core. See bno_emu.h and
 * docs/2026-10-22_Spec_G0_Gia_lap_BNO085.md (G0 v1). Plain C, no HAL.
 *
 * Sources for the byte layouts (ESP32 firmware AGV_MECANUM, read only):
 *   components/esp32_BNO08x/SH2/shtp.c  (SHTP header, advertisement TLVs)
 *   components/esp32_BNO08x/SH2/sh2.c   (control reports 0xF8/0xFC/0xFB,
 *                                        report lengths tag 0x81)
 *   components/esp32_BNO08x/SH2/sh2_SensorValue.c (Q8/Q9/Q14 sensor data)
 */
#include "bno_emu.h"
#include <string.h>

/* SHTP channels (advertised below, datasheet p. 22) */
#define CH_COMMAND      0u
#define CH_EXECUTABLE   1u
#define CH_CONTROL      2u
#define CH_INPUT        3u
#define CH_WAKE         4u
#define CH_GYRO_RV      5u

/* SHTP advertisement tags (shtp.h) */
#define TAG_GUID                        1u
#define TAG_MAX_CARGO_PLUS_HEADER_WRITE 2u
#define TAG_MAX_CARGO_PLUS_HEADER_READ  3u
#define TAG_MAX_TRANSFER_WRITE          4u
#define TAG_MAX_TRANSFER_READ           5u
#define TAG_NORMAL_CHANNEL              6u
#define TAG_WAKE_CHANNEL                7u
#define TAG_APP_NAME                    8u
#define TAG_CHANNEL_NAME                9u
#define TAG_SH2_REPORT_LENGTHS          0x81u

/* SH-2 control reports (sh2.c) */
#define RPT_COMMAND_REQ      0xF2u
#define RPT_COMMAND_RESP     0xF1u
#define RPT_PROD_ID_REQ      0xF9u
#define RPT_PROD_ID_RESP     0xF8u
#define RPT_SET_FEATURE      0xFDu
#define RPT_GET_FEATURE_REQ  0xFEu
#define RPT_GET_FEATURE_RESP 0xFCu
#define RPT_BASE_TIMESTAMP   0xFBu
#define RPT_TIMESTAMP_REBASE 0xFAu
#define RPT_FLUSH_COMPLETED  0xEFu

#define LEN_PROD_ID_RESP     16u
#define LEN_FEATURE          17u
#define LEN_COMMAND_REQ      12u

/* INT must stay high this long between two packets, so the host sees an edge */
#define INT_MIN_HIGH_US      10u

/* Product ID entries: emulator values, deliberately not real BNO085 part
 * numbers, so a log shows at once that the IMU is the H7 emulator. */
#define EMU_PART_NUMBER_BASE 90000000u

static const uint8_t k_sensor_id[BNO_EMU_N_SENSORS]  = { BNO_EMU_ID_ACCEL, BNO_EMU_ID_GYRO, BNO_EMU_ID_GAME_RV };
static const uint8_t k_sensor_len[BNO_EMU_N_SENSORS] = { 10u, 10u, 12u };

static const uint8_t k_zero[8] = { 0 };

/* ── Small helpers ──────────────────────────────────────────────────────── */

static bool time_ge(uint32_t a, uint32_t b) { return (int32_t)(a - b) >= 0; }

static void put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static uint32_t get32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* Saturate to +-range, scale by 2^q, round to nearest, saturate to int16. */
static int16_t to_fixed(float x, unsigned q, float range)
{
    if (x >  range) x =  range;
    if (x < -range) x = -range;
    float v = x * (float)(1u << q);
    int32_t r = (int32_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
    if (r >  32767) r =  32767;
    if (r < -32768) r = -32768;
    return (int16_t)r;
}

int bno_emu_sensor_index(uint8_t report_id)
{
    for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++)
        if (k_sensor_id[i] == report_id) return (int)i;
    return -1;
}

/* ── Outgoing queue ─────────────────────────────────────────────────────── */

static bool q_push(bno_emu_t *e, uint8_t chan, const uint8_t *payload, uint16_t plen)
{
    if (e->q_count >= BNO_EMU_QUEUE_LEN || plen + 4u > BNO_EMU_PKT_MAX) {
        e->st.q_drops++;
        return false;
    }
    bno_emu_pkt_t *p = &e->q[(e->q_head + e->q_count) % BNO_EMU_QUEUE_LEN];
    p->rmask = 0;
    p->len = (uint16_t)(plen + 4u);
    put16(p->buf, p->len);
    p->buf[2] = chan;
    p->buf[3] = e->chan_seq[chan]++;
    memcpy(p->buf + 4, payload, plen);
    memset(p->buf + p->len, 0, 4);          /* host clocks 4 bytes past the packet */
    e->q_count++;
    return true;
}

static void emit(bno_emu_t *e, const bno_emu_event_t *ev)
{
    if (e->hook) e->hook(e->hook_ctx, ev);
}

static void q_pop(bno_emu_t *e)
{
    if (e->q_count == 0) return;
    e->q_head = (uint8_t)((e->q_head + 1u) % BNO_EMU_QUEUE_LEN);
    e->q_count--;
}

static bool q_has_chan(const bno_emu_t *e, uint8_t chan)
{
    for (unsigned i = 0; i < e->q_count; i++)
        if (e->q[(e->q_head + i) % BNO_EMU_QUEUE_LEN].buf[2] == chan) return true;
    return false;
}

/* ── Packets the chip sends ─────────────────────────────────────────────── */

static void tlv(uint8_t *buf, uint16_t *n, uint8_t tag, uint8_t len, const void *val)
{
    buf[(*n)++] = tag;
    buf[(*n)++] = len;
    memcpy(buf + *n, val, len);
    *n = (uint16_t)(*n + len);
}

static void tlv_u32(uint8_t *buf, uint16_t *n, uint8_t tag, uint32_t v)
{
    uint8_t b[4]; put32(b, v); tlv(buf, n, tag, 4, b);
}

static void tlv_u16(uint8_t *buf, uint16_t *n, uint8_t tag, uint16_t v)
{
    uint8_t b[2]; put16(b, v); tlv(buf, n, tag, 2, b);
}

static void tlv_u8(uint8_t *buf, uint16_t *n, uint8_t tag, uint8_t v) { tlv(buf, n, tag, 1, &v); }

static void tlv_str(uint8_t *buf, uint16_t *n, uint8_t tag, const char *s)
{
    tlv(buf, n, tag, (uint8_t)(strlen(s) + 1u), s);     /* host strcpy()s: keep the NUL */
}

/* Minimal advertisement (G0 v1 §3.1): the apps and channels the host looks up
 * by name, transfer sizes, and the length of every report ID sent. */
static void send_advertisement(bno_emu_t *e)
{
    static const uint8_t lengths[] = {
        RPT_COMMAND_RESP, 16, RPT_PROD_ID_RESP, LEN_PROD_ID_RESP, RPT_GET_FEATURE_RESP, LEN_FEATURE,
        RPT_BASE_TIMESTAMP, 5, RPT_TIMESTAMP_REBASE, 5, RPT_FLUSH_COMPLETED, 2,
        BNO_EMU_ID_ACCEL, 10, BNO_EMU_ID_GYRO, 10, BNO_EMU_ID_GAME_RV, 12,
    };
    uint8_t a[BNO_EMU_PKT_MAX - 4u];
    uint16_t n = 0;

    a[n++] = 0x00;                                      /* advertisement response */
    tlv_u32(a, &n, TAG_GUID, 0);                        /* SHTP itself */
    tlv_u16(a, &n, TAG_MAX_CARGO_PLUS_HEADER_WRITE, 256);
    tlv_u16(a, &n, TAG_MAX_CARGO_PLUS_HEADER_READ, BNO_EMU_PKT_MAX);
    tlv_u16(a, &n, TAG_MAX_TRANSFER_WRITE, 256);
    tlv_u16(a, &n, TAG_MAX_TRANSFER_READ, BNO_EMU_PKT_MAX);
    tlv_u8(a, &n, TAG_NORMAL_CHANNEL, CH_COMMAND);
    tlv_str(a, &n, TAG_CHANNEL_NAME, "command");

    tlv_u32(a, &n, TAG_GUID, 1);
    tlv_str(a, &n, TAG_APP_NAME, "executable");
    tlv_u8(a, &n, TAG_NORMAL_CHANNEL, CH_EXECUTABLE);
    tlv_str(a, &n, TAG_CHANNEL_NAME, "device");

    tlv_u32(a, &n, TAG_GUID, 2);
    tlv_str(a, &n, TAG_APP_NAME, "sensorhub");
    tlv_u8(a, &n, TAG_NORMAL_CHANNEL, CH_CONTROL);
    tlv_str(a, &n, TAG_CHANNEL_NAME, "control");
    tlv_u8(a, &n, TAG_NORMAL_CHANNEL, CH_INPUT);
    tlv_str(a, &n, TAG_CHANNEL_NAME, "inputNormal");
    tlv_u8(a, &n, TAG_WAKE_CHANNEL, CH_WAKE);
    tlv_str(a, &n, TAG_CHANNEL_NAME, "inputWake");
    tlv_u8(a, &n, TAG_NORMAL_CHANNEL, CH_GYRO_RV);
    tlv_str(a, &n, TAG_CHANNEL_NAME, "inputGyroRv");
    tlv(a, &n, TAG_SH2_REPORT_LENGTHS, sizeof(lengths), lengths);

    q_push(e, CH_COMMAND, a, n);
    e->st.adverts++;
}

static void send_reset_complete(bno_emu_t *e)
{
    const uint8_t rc = 0x01;
    q_push(e, CH_EXECUTABLE, &rc, 1);
}

static void send_prod_ids(bno_emu_t *e)
{
    uint8_t r[4u * LEN_PROD_ID_RESP];
    memset(r, 0, sizeof(r));
    for (unsigned i = 0; i < 4u; i++) {
        uint8_t *p = r + i * LEN_PROD_ID_RESP;
        p[0] = RPT_PROD_ID_RESP;
        p[1] = (i == 0u) ? 1u : 0u;                     /* reset cause: power on */
        put32(p + 4, EMU_PART_NUMBER_BASE + i);         /* sw part number */
        put32(p + 8, 1u);                               /* sw build number */
    }
    q_push(e, CH_CONTROL, r, sizeof(r));
    e->st.prod_id_reqs++;
}

static void send_feature_resp(bno_emu_t *e, uint8_t sensor_id, const uint8_t *req)
{
    uint8_t r[LEN_FEATURE];
    int idx = bno_emu_sensor_index(sensor_id);
    if (req != 0) {
        memcpy(r, req, LEN_FEATURE);                    /* echo the Set Feature fields */
    } else {
        memset(r, 0, sizeof(r));
        put32(r + 5, (idx >= 0) ? e->period_us[idx] : 0u);
    }
    r[0] = RPT_GET_FEATURE_RESP;
    r[1] = sensor_id;
    q_push(e, CH_CONTROL, r, sizeof(r));
}

static uint16_t build_report(bno_emu_t *e, unsigned i, uint8_t *out, uint32_t now_us)
{
    const bno_emu_sample_t *s = &e->sample;
    bno_emu_last_t *l = &e->last[i];
    float in[4] = { 0 };
    unsigned nv = 3;

    switch (k_sensor_id[i]) {
    case BNO_EMU_ID_ACCEL:
        in[0] = s->ax; in[1] = s->ay; in[2] = s->az;
        for (unsigned k = 0; k < 3; k++) l->raw[k] = to_fixed(in[k], 8, e->cfg.accel_range);
        break;
    case BNO_EMU_ID_GYRO:
        in[0] = s->gx; in[1] = s->gy; in[2] = s->gz;
        for (unsigned k = 0; k < 3; k++) l->raw[k] = to_fixed(in[k], 9, e->cfg.gyro_range);
        break;
    default:                                            /* game rotation vector */
        in[0] = s->qi; in[1] = s->qj; in[2] = s->qk; in[3] = s->qr;
        nv = 4;
        for (unsigned k = 0; k < 4; k++) l->raw[k] = to_fixed(in[k], 14, 1.9999f);
        break;
    }

    out[0] = k_sensor_id[i];
    out[1] = e->rpt_seq[i];
    out[2] = 0x03;                                      /* accuracy: high; delay = 0 */
    out[3] = 0x00;
    for (unsigned k = 0; k < nv; k++) put16(out + 4 + 2u * k, (uint16_t)l->raw[k]);

    l->report_seq = e->rpt_seq[i];
    l->sample_seq = s->seq;
    memcpy(l->in, in, sizeof(in));
    l->t_us = now_us;
    l->count++;
    e->rpt_seq[i]++;
    return k_sensor_len[i];
}

/* ── State changes ──────────────────────────────────────────────────────── */

static void clear_chip(bno_emu_t *e)
{
    e->q_head = e->q_count = 0;
    memset(e->chan_seq, 0, sizeof(e->chan_seq));
    memset(e->period_us, 0, sizeof(e->period_us));
    memset(e->rpt_seq, 0, sizeof(e->rpt_seq));
    memset(e->pend, 0, sizeof(e->pend));
    e->int_low = false;
    e->in_xfer = false;
    e->xfer_offered = false;
    e->xfer_len = 0;
}

static void start_boot(bno_emu_t *e, uint32_t now_us)
{
    clear_chip(e);
    e->state = BNO_EMU_BOOTING;
    e->t_state_us = now_us;
    e->t_int_us = now_us;
    e->t_int_allowed_us = now_us;
}

static void handle_control(bno_emu_t *e, const uint8_t *p, uint16_t n, uint32_t now_us)
{
    uint16_t c = 0;
    while (c < n) {
        switch (p[c]) {
        case RPT_PROD_ID_REQ:
            send_prod_ids(e);
            c = (uint16_t)(c + 2u);
            break;
        case RPT_SET_FEATURE: {
            if (c + LEN_FEATURE > n) return;
            int idx = bno_emu_sensor_index(p[c + 1]);
            uint32_t interval = get32(p + c + 5);
            if (idx >= 0 && e->period_us[idx] != interval) {   /* same period: keep cadence */
                e->period_us[idx] = interval;
                e->next_due_us[idx] = now_us + interval;
            }
            send_feature_resp(e, p[c + 1], p + c);
            e->st.set_features++;
            c = (uint16_t)(c + LEN_FEATURE);
            break;
        }
        case RPT_GET_FEATURE_REQ:
            if (c + 2u > n) return;
            send_feature_resp(e, p[c + 1], 0);
            e->st.get_features++;
            c = (uint16_t)(c + 2u);
            break;
        case RPT_COMMAND_REQ:
            e->st.unknown_cmds++;                       /* not needed by the host (G0 v1 §3) */
            c = (uint16_t)(c + LEN_COMMAND_REQ);
            break;
        default:
            e->st.unknown_cmds++;
            return;
        }
    }
}

static void handle_host_packet(bno_emu_t *e, const uint8_t *p, uint16_t n, uint32_t now_us)
{
    uint16_t len = (uint16_t)((p[0] | (p[1] << 8)) & 0x7FFF);
    if (len > n) len = n;
    if (len < 4u) return;
    const uint8_t *pl = p + 4;
    uint16_t plen = (uint16_t)(len - 4u);

    switch (p[2]) {
    case CH_COMMAND:
        if (plen >= 1u && pl[0] == 0x00) send_advertisement(e);      /* advertise request */
        else e->st.unknown_cmds++;
        break;
    case CH_EXECUTABLE:
        if (plen >= 1u && pl[0] == 0x01) {                           /* soft reset */
            e->st.resets++;
            start_boot(e, now_us);
        }
        break;                                                       /* on / sleep: ignored */
    case CH_CONTROL:
        handle_control(e, pl, plen, now_us);
        break;
    default:
        e->st.unknown_cmds++;
        break;
    }
}

/* ── Public API ─────────────────────────────────────────────────────────── */

void bno_emu_default_cfg(bno_emu_cfg_t *cfg)
{
    cfg->boot_delay_us       = 20000u;   /* [ĐỀ XUẤT] real chip value not measured yet */
    cfg->int_timeout_us      = 10000u;   /* datasheet p. 18-19: about 10 ms */
    cfg->int_retry_gap_us    = 1000u;    /* [ĐỀ XUẤT] */
    cfg->empty_int_period_us = 10000u;   /* G0 v1 §4.2 */
    cfg->stale_periods       = 3u;       /* G0 v1 §3.2 */
    cfg->accel_range         = 8.0f * 9.80665f;      /* +-8 g, to check against the datasheet */
    cfg->gyro_range          = 2000.0f * 3.14159265f / 180.0f;   /* +-2000 deg/s */
}

void bno_emu_init(bno_emu_t *e, const bno_emu_cfg_t *cfg, uint32_t now_us)
{
    memset(e, 0, sizeof(*e));
    e->cfg = *cfg;
    start_boot(e, now_us);                  /* power on = reset released at now_us */
}

void bno_emu_set_reset(bno_emu_t *e, bool rst_low, uint32_t now_us)
{
    if (rst_low) {
        if (e->state != BNO_EMU_HELD) e->st.resets++;
        clear_chip(e);
        e->state = BNO_EMU_HELD;
        e->t_state_us = now_us;
    } else if (e->state == BNO_EMU_HELD) {
        start_boot(e, now_us);
    }
}

void bno_emu_set_sample(bno_emu_t *e, const bno_emu_sample_t *s, uint32_t now_us)
{
    if (!e->have_sample || s->seq != e->sample.seq) e->t_seq_change_us = now_us;
    e->sample = *s;
    e->have_sample = true;
    e->t_sample_us = now_us;
}

void bno_emu_poll(bno_emu_t *e, uint32_t now_us)
{
    if (e->state == BNO_EMU_HELD) return;

    if (e->state == BNO_EMU_BOOTING) {
        if (!time_ge(now_us, e->t_state_us + e->cfg.boot_delay_us)) return;
        send_advertisement(e);
        send_reset_complete(e);
        e->state = BNO_EMU_RUNNING;
        e->t_state_us = now_us;
        bno_emu_event_t ev = { .type = BNO_EV_RESET_DONE, .now_us = now_us };
        emit(e, &ev);
    }

    /* Report schedule: each sensor keeps its own cadence (next_due += period). */
    bool due[BNO_EMU_N_SENSORS] = { false };
    bool any_due = false;
    uint32_t pmin = 0;
    for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) {
        uint32_t p = e->period_us[i];
        if (p == 0u) continue;
        if (pmin == 0u || p < pmin) pmin = p;
        if (time_ge(now_us, e->next_due_us[i])) {
            due[i] = true;
            any_due = true;
            e->next_due_us[i] += p;
            if (time_ge(now_us, e->next_due_us[i])) {   /* more than one period behind */
                e->next_due_us[i] = now_us + p;
                e->st.late_resyncs++;
            }
        }
    }
    if (any_due) {
        bool stale = !e->have_sample ||
                     time_ge(now_us, e->t_seq_change_us + e->cfg.stale_periods * pmin);
        if (e->hold) {
            e->st.hold_skips++;                         /* S06 T8: injected IMU loss */
            memset(e->pend, 0, sizeof(e->pend));
        } else if (stale) {
            e->st.stale_skips++;                        /* like a dead chip: send nothing */
            memset(e->pend, 0, sizeof(e->pend));
        } else {
            for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) {
                if (!due[i]) continue;
                if (e->pend[i]) e->st.input_overruns++; /* previous one never left */
                e->pend[i] = true;
            }
        }
    }
    /* One input packet in flight at a time; reports that fall due meanwhile
     * wait and go out together in the next packet (the sensors are enabled
     * at different times, so their schedules are out of phase). */
    bool any_pend = false;
    for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) any_pend |= e->pend[i];
    if (any_pend && !q_has_chan(e, CH_INPUT)) {
        uint8_t pkt[5u + 10u + 10u + 12u];
        uint16_t n = 0;
        uint8_t mask = 0, seqs[BNO_EMU_N_SENSORS] = { 0 };
        pkt[n++] = RPT_BASE_TIMESTAMP;                  /* age of the plant sample, 100 us units */
        put32(pkt + n, (now_us - e->t_sample_us) / 100u);
        n = (uint16_t)(n + 4u);
        for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) {
            if (!e->pend[i]) continue;
            seqs[i] = e->rpt_seq[i];
            mask |= (uint8_t)(1u << i);
            n = (uint16_t)(n + build_report(e, i, pkt + n, now_us));
            e->pend[i] = false;
        }
        bool queued = q_push(e, CH_INPUT, pkt, n);
        if (queued) {
            bno_emu_pkt_t *p = &e->q[(e->q_head + e->q_count - 1u) % BNO_EMU_QUEUE_LEN];
            p->rmask = mask;
            memcpy(p->rseq, seqs, sizeof seqs);
        }
        for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) {
            if (!(mask & (1u << i))) continue;
            bno_emu_event_t ev = { .type = BNO_EV_REPORT, .sensor = (uint8_t)i, .rpt_seq = seqs[i],
                                   .flags = queued ? BNO_EV_QUEUED : BNO_EV_QFULL,
                                   .sample = &e->sample, .now_us = now_us };
            emit(e, &ev);
        }
    }

    /* INT (G0 v1 §4): packet waiting -> low; nothing for empty_int_period -> low
     * with length 0; no answer within int_timeout -> release and retry. */
    if (e->in_xfer) return;
    if (e->int_low) {
        if (time_ge(now_us, e->t_int_us + e->cfg.int_timeout_us)) {
            e->int_low = false;
            e->t_int_us = now_us;
            e->t_int_allowed_us = now_us + e->cfg.int_retry_gap_us;
            e->st.int_timeouts++;
        }
    } else if (time_ge(now_us, e->t_int_allowed_us)) {
        if (e->q_count > 0u ||
            (e->state == BNO_EMU_RUNNING && time_ge(now_us, e->t_int_us + e->cfg.empty_int_period_us))) {
            e->int_low = true;
            e->t_int_us = now_us;
        }
    }
}

bool bno_emu_int_low(const bno_emu_t *e) { return e->int_low; }

const uint8_t *bno_emu_peek_tx(const bno_emu_t *e, uint16_t *tx_len)
{
    if ((e->state != BNO_EMU_HELD) && (e->q_count > 0u)) {
        const bno_emu_pkt_t *p = &e->q[e->q_head];
        *tx_len = (uint16_t)(p->len + 4u);
        return p->buf;
    }
    *tx_len = 4u;                           /* header with length 0 */
    return k_zero;
}

const uint8_t *bno_emu_cs_fall(bno_emu_t *e, uint32_t now_us, uint16_t *tx_len)
{
    e->in_xfer = true;
    if (e->int_low) {                       /* chip releases INT when it sees CS low */
        e->int_low = false;
        e->t_int_us = now_us;
    }
    /* Remember what was offered: poll() may queue new packets while CS is low. */
    const uint8_t *tx = bno_emu_peek_tx(e, tx_len);
    e->xfer_offered = (tx != k_zero);
    e->xfer_len = e->xfer_offered ? (uint16_t)(*tx_len - 4u) : 0u;
    return tx;
}

void bno_emu_cs_rise(bno_emu_t *e, const uint8_t *mosi, uint16_t n_clocked, uint32_t now_us)
{
    bool offered = e->xfer_offered && (e->q_count > 0u);
    uint16_t offered_len = e->xfer_len;

    e->in_xfer = false;
    e->xfer_offered = false;
    e->t_int_allowed_us = now_us + INT_MIN_HIGH_US;
    if (e->state == BNO_EMU_HELD) return;

    bool write = (n_clocked >= 4u) && ((mosi[0] | mosi[1] | mosi[2] | mosi[3]) != 0u);
    if (write) {
        e->st.writes++;
        if (offered) e->st.collisions++;    /* packet stays queued, sent at next INT */
        handle_host_packet(e, mosi, n_clocked, now_us);
    } else {
        e->st.reads++;
        if (!offered) e->st.empty_reads++;
        else if (n_clocked >= offered_len) {
            const bno_emu_pkt_t *p = &e->q[e->q_head];
            if (p->rmask) {                     /* S06 Y14: the host has this report packet */
                bno_emu_event_t ev = { .type = BNO_EV_PKT_READ, .mask = p->rmask, .now_us = now_us };
                memcpy(ev.seqs, p->rseq, sizeof ev.seqs);
                emit(e, &ev);
            }
            q_pop(e);
        }
    }
}

void bno_emu_set_hook(bno_emu_t *e, bno_emu_hook_t fn, void *ctx)
{
    e->hook = fn;
    e->hook_ctx = ctx;
}

void bno_emu_set_hold(bno_emu_t *e, bool hold, uint32_t now_us)
{
    if (hold == e->hold) return;
    e->hold = hold;
    if (hold) memset(e->pend, 0, sizeof(e->pend));   /* nothing due before it is built later */
    bno_emu_event_t ev = { .type = BNO_EV_HOLD, .flags = hold ? 1u : 0u, .now_us = now_us };
    emit(e, &ev);
}
