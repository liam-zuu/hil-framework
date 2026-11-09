/*
 * harness.c: G1 test of the BNO085 emulator on the PC (G0 v1 §8).
 *
 * The real SH-2 library of the ESP32 firmware (sh2.c, shtp.c,
 * sh2_SensorValue.c, sh2_util.c from AGV_MECANUM, compiled unchanged) talks to
 * bno_emu.c through a fake HAL that copies BNO08xSH2HAL.cpp:
 *   read : wait INT -> CS low -> 4-byte header -> len -> len more bytes -> CS high
 *   write: wait INT -> CS low -> packet on MOSI (MISO ignored) -> CS high
 *   INT wait 500 ms, then the host pulses RST (hardware reset) and fails the
 *   transaction; INT edge = one event bit, cleared by the wait (BNO08x.cpp).
 * Time is virtual (64-bit microseconds); SPI at 3 MHz plus a fixed overhead.
 * One thread plays both the driver's init call and its service task, so the
 * 30 ms delay after each Set Feature (BNO08xRpt.cpp) runs the service loop.
 *
 *   harness <scenario> [period_us] [seconds]
 *   scenarios: boot, run, collision, rst_mid, silent, stale
 * Environment BNOEMU_H7_PORT=1 (G2, 28/10): the emulator is driven the way the
 * H7 layer stm32h7/Core/Src/bno_emu_port.c drives it: core poll, new plant
 * sample and INT pin only on a 100 us tick (TIM6); MISO comes from a DMA buffer
 * armed ahead from bno_emu_peek_tx() while CS and INT are high (not from the
 * cs_fall() return); INT goes low only for what is armed; after CS rise the
 * buffer is armed again. The bytes clocked are what the host shifted.
 * Prints key=value lines and RESULT=PASS|FAIL; exit code 0 on PASS.
 */
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sh2.h"
#include "sh2_err.h"
#include "sh2_hal.h"
#include "sh2_SensorValue.h"
#include "bno_emu.h"

#define STEP_US           20u        /* emulator poll / host idle step          */
#define PLANT_PERIOD_US   1000u      /* plant (RPi5 frame) rate                 */
#define HINT_TIMEOUT_US   500000u    /* sdkconfig HOST_INT_TIMEOUT              */
#define SPI_HZ            3000000.0  /* imu.cpp sclk_speed                      */
#define XFER_OVERHEAD_US  20u        /* per SPI transaction (estimate)          */
#define RST_LOW_US        100000u    /* BNO08x::toggle_reset                    */
#define RST_WAIT_US       100000u
#define SET_FEATURE_WAIT  30000u     /* BNO08xRpt::rpt_enable vTaskDelay(30 ms) */

static const uint8_t k_ids[BNO_EMU_N_SENSORS] = { SH2_ACCELEROMETER, SH2_GYROSCOPE_CALIBRATED, SH2_GAME_ROTATION_VECTOR };
static const unsigned k_q[BNO_EMU_N_SENSORS] = { 8, 9, 14 };
static const unsigned k_nv[BNO_EMU_N_SENSORS] = { 3, 3, 4 };

/* ── Virtual world ──────────────────────────────────────────────────────── */

static uint64_t now;
static bno_emu_t emu;
static bool prev_int, hint_flag;
static uint64_t next_plant;
static bool plant_on = true;
static uint8_t plant_seq;
static uint32_t int_edges;

static const uint8_t *zeros_port(void) { static const uint8_t z[(BNO_EMU_PKT_MAX + 4u + 16u)] = { 0 }; return z; }

/* ── H7 port model (BNOEMU_H7_PORT=1), mirrors bno_emu_port.c ─────────── */
#define PORT_TICK_US  100u
#define PORT_ARM_LEN  (BNO_EMU_PKT_MAX + 4u + 16u)
static bool port_mode;
static uint64_t port_next_tick;
static uint8_t port_tx[PORT_ARM_LEN];
static bool port_armed, port_in_xfer, port_int_pin, port_have_imu;   /* port_int_pin: PG2 low */
static const uint8_t *port_armed_src;
static uint16_t port_armed_len;
static uint8_t port_last_seq;
static bno_emu_sample_t port_frame;      /* latest plant sample in the SPI1 frame */
static uint32_t port_mismatch, port_rearms, port_ticks;

static void port_rearm(void)
{
    uint16_t len;
    const uint8_t *src = bno_emu_peek_tx(&emu, &len);
    if (len > PORT_ARM_LEN) len = PORT_ARM_LEN;
    memcpy(port_tx, src, len);
    memset(port_tx + len, 0, PORT_ARM_LEN - len);
    port_armed = true;
    port_armed_src = src;
    port_armed_len = len;
    port_rearms++;
}

static void port_tick(void)
{
    port_ticks++;
    if (port_have_imu ? (port_frame.seq != port_last_seq) : (port_frame.seq != 0u)) {
        bno_emu_set_sample(&emu, &port_frame, (uint32_t)now);
        port_have_imu = true;
        port_last_seq = port_frame.seq;
    }
    bno_emu_poll(&emu, (uint32_t)now);
    if (!port_in_xfer && !port_int_pin) {
        uint16_t len;
        const uint8_t *src = bno_emu_peek_tx(&emu, &len);
        if (!port_armed || src != port_armed_src || len != port_armed_len) port_rearm();
    }
    bool want = bno_emu_int_low(&emu) && !port_in_xfer;
    if (want && !port_int_pin) {
        if (port_armed) port_int_pin = true;
    } else if (!want && port_int_pin) {
        port_int_pin = false;
    }
}

/* CS fall: core bookkeeping; MISO is the armed buffer whatever the core says. */
static const uint8_t *port_cs_fall(void)
{
    uint16_t txl;
    const uint8_t *tx = bno_emu_cs_fall(&emu, (uint32_t)now, &txl);
    port_in_xfer = true;
    if (!port_armed || tx != port_armed_src || txl != port_armed_len) port_mismatch++;
    port_int_pin = false;
    return port_armed ? port_tx : zeros_port();
}

static void port_cs_rise(const uint8_t *mosi, uint16_t n)
{
    port_armed = false;
    port_in_xfer = false;
    bno_emu_cs_rise(&emu, mosi, n, (uint32_t)now);
    port_rearm();
}

static void port_reset(bool low)
{
    bno_emu_set_reset(&emu, low, (uint32_t)now);
    port_armed_src = NULL;
    if (low) port_int_pin = false;
}

static jmp_buf abort_jmp;
static bool guard_on;
static uint64_t guard_until;

static void sync_int(void)
{
    bool lo = port_mode ? port_int_pin : bno_emu_int_low(&emu);
    if (lo && !prev_int) { hint_flag = true; int_edges++; }
    prev_int = lo;
}

/* Plant signal: smooth motion plus a short gyro burst above the chip range
 * (t in 7.0..7.2 s of every 20 s) to exercise saturation. */
static void plant_sample(bno_emu_sample_t *s, double t)
{
    const double g = 9.80665, tw = 2.0 * 3.14159265358979;
    s->ax = (float)(0.8 * sin(tw * 0.5 * t));
    s->ay = (float)(0.6 * cos(tw * 0.3 * t));
    s->az = (float)(g + 0.05 * sin(tw * 2.0 * t));
    s->gx = (float)(0.02 * sin(tw * 1.0 * t));
    s->gy = (float)(-0.03 * cos(tw * 0.7 * t));
    s->gz = (float)(1.5 * sin(tw * 0.25 * t));
    if (fmod(t, 20.0) >= 7.0 && fmod(t, 20.0) < 7.2) s->gz = 40.0f;
    double yaw = 2.5 * sin(tw * 0.05 * t);
    s->qi = 0.0f; s->qj = 0.0f;
    s->qk = (float)sin(yaw / 2.0);
    s->qr = (float)cos(yaw / 2.0);
}

static void plant_tick(void)
{
    while (now >= next_plant) {
        next_plant += PLANT_PERIOD_US;
        static bno_emu_sample_t s;
        if (plant_on) {
            plant_seq++;
            plant_sample(&s, (double)now * 1e-6);
        }
        s.seq = plant_seq;                  /* frozen plant: same seq every frame */
        if (port_mode) port_frame = s;      /* the H7 tick picks it up */
        else bno_emu_set_sample(&emu, &s, (uint32_t)now);
    }
}

static void advance(uint64_t us)
{
    uint64_t end = now + us;
    while (now < end) {
        uint64_t step = (end - now < STEP_US) ? (end - now) : STEP_US;
        now += step;
        plant_tick();
        if (port_mode) {
            while (now >= port_next_tick) {
                port_next_tick += PORT_TICK_US;
                port_tick();
            }
        } else {
            bno_emu_poll(&emu, (uint32_t)now);
        }
        sync_int();
        if (guard_on && now >= guard_until) longjmp(abort_jmp, 1);
    }
}

static uint64_t xfer_us(unsigned nbytes) { return XFER_OVERHEAD_US + (uint64_t)(nbytes * 8.0 * 1e6 / SPI_HZ + 0.5); }

/* ── Fake HAL (BNO08xSH2HAL.cpp) ────────────────────────────────────────── */

static uint32_t hard_resets;          /* host timeouts that pulsed RST */
static uint8_t zeros[512];

static void toggle_reset(void)
{
    if (port_mode) port_reset(true); else bno_emu_set_reset(&emu, true, (uint32_t)now);
    sync_int();
    advance(RST_LOW_US);
    if (port_mode) port_reset(false); else bno_emu_set_reset(&emu, false, (uint32_t)now);
    sync_int();
    advance(RST_WAIT_US);
}

static bool wait_hint(void)
{
    uint64_t t0 = now;
    while (!hint_flag) {
        if (now - t0 >= HINT_TIMEOUT_US) {
            hard_resets++;
            toggle_reset();
            return false;
        }
        advance(STEP_US);
    }
    hint_flag = false;
    return true;
}

static int hal_open(sh2_Hal_t *self) { (void)self; wait_hint(); return 0; }
static void hal_close(sh2_Hal_t *self) { (void)self; }

static int hal_read(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len, uint32_t *t_us)
{
    (void)self; (void)t_us;               /* the real HAL leaves t_us at 0 */
    if (!wait_hint()) return 0;

    uint16_t txl = 0;
    uint8_t miso[BNO_EMU_PKT_MAX + 8];
    memset(miso, 0, sizeof(miso));
    if (port_mode) {
        memcpy(miso, port_cs_fall(), sizeof(miso));                /* armed DMA buffer */
        sync_int();
    } else {
        const uint8_t *tx = bno_emu_cs_fall(&emu, (uint32_t)now, &txl);
        memcpy(miso, tx, txl > sizeof(miso) ? sizeof(miso) : txl); /* DMA latches the buffer */
    }

    memcpy(pBuffer, miso, 4);
    advance(xfer_us(4));
    uint16_t plen = (uint16_t)((miso[0] | (miso[1] << 8)) & 0x7FFF);
    if (plen == 0 || plen > len) {
        if (port_mode) port_cs_rise(zeros, 4); else bno_emu_cs_rise(&emu, zeros, 4, (uint32_t)now);
        sync_int();
        return 0;
    }
    for (unsigned i = 0; i < plen; i++)
        pBuffer[4 + i] = (4u + i < sizeof(miso)) ? miso[4 + i] : 0;
    advance(xfer_us(plen));
    if (port_mode) port_cs_rise(zeros, (uint16_t)(4u + plen));
    else bno_emu_cs_rise(&emu, zeros, (uint16_t)(4u + plen), (uint32_t)now);
    sync_int();
    return plen;
}

static int hal_write(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len)
{
    (void)self;
    if (!wait_hint()) return 0;
    uint16_t txl;
    if (port_mode) { (void)port_cs_fall(); sync_int(); }
    else (void)bno_emu_cs_fall(&emu, (uint32_t)now, &txl);
    advance(xfer_us(len));
    if (port_mode) port_cs_rise(pBuffer, (uint16_t)len);
    else bno_emu_cs_rise(&emu, pBuffer, (uint16_t)len, (uint32_t)now);
    sync_int();
    return (int)len;
}

static uint32_t hal_time(sh2_Hal_t *self)
{
    (void)self;
    if (guard_on && now >= guard_until) longjmp(abort_jmp, 1);
    return (uint32_t)now;
}

static sh2_Hal_t hal = { hal_open, hal_close, hal_read, hal_write, hal_time };

/* ── Host side: callbacks and statistics ────────────────────────────────── */

typedef struct {
    uint32_t n;
    uint64_t t_first, t_last;
    double   sum_dt, min_dt, max_dt;
    uint32_t n_dt;
    uint32_t raw_mismatch, seq_mismatch, decode_mismatch;
    double   max_qerr_lsb;
} rx_stat_t;

static rx_stat_t rx[BNO_EMU_N_SENSORS];
static uint32_t reset_events, feature_resps;
static bool initialized, reenable_pending, sh2_is_open;
static uint32_t period_req;

/* S06 event hook of the core (29/10): every report the host library hands to
 * the sensor callback must have been built (REPORT, queued) and then read as
 * part of a packet (PKT_READ); rpt_seq of each sensor goes up by one per built
 * report within a reset; no report is built during IMU loss injection. */
enum { HK_NONE = 0, HK_QUEUED, HK_READ, HK_RECEIVED };
static uint8_t  hk_state[BNO_EMU_N_SENSORS][256];
static int      hk_last_seq[BNO_EMU_N_SENSORS] = { -1, -1, -1 };
static uint32_t hk_reset_done, hk_hold_on, hk_hold_off, hk_rpt_in_hold, hk_qfull;
static uint32_t hk_built, hk_read, hk_received, hk_seq_gap, hk_read_not_built, hk_rx_not_read;
static bool     hk_in_hold;

static void hook(void *ctx, const bno_emu_event_t *ev)
{
    (void)ctx;
    switch (ev->type) {
    case BNO_EV_REPORT:
        if (hk_in_hold) hk_rpt_in_hold++;
        if (hk_last_seq[ev->sensor] >= 0 && ev->rpt_seq != (uint8_t)(hk_last_seq[ev->sensor] + 1)) hk_seq_gap++;
        hk_last_seq[ev->sensor] = ev->rpt_seq;
        if (ev->flags & BNO_EV_QFULL) { hk_qfull++; hk_state[ev->sensor][ev->rpt_seq] = HK_NONE; }
        else hk_state[ev->sensor][ev->rpt_seq] = HK_QUEUED;
        hk_built++;
        break;
    case BNO_EV_PKT_READ:
        for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) {
            if (!(ev->mask & (1u << i))) continue;
            if (hk_state[i][ev->seqs[i]] != HK_QUEUED) hk_read_not_built++;
            hk_state[i][ev->seqs[i]] = HK_READ;
            hk_read++;
        }
        break;
    case BNO_EV_RESET_DONE:
        hk_reset_done++;
        for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) hk_last_seq[i] = -1;   /* rpt_seq restarts at 0 */
        break;
    case BNO_EV_HOLD:
        if (ev->flags) hk_hold_on++; else hk_hold_off++;
        break;
    }
}

static void rx_clear(void) { memset(rx, 0, sizeof(rx)); }

static void hal_cb(void *cookie, sh2_AsyncEvent_t *ev)
{
    (void)cookie;
    if (ev->eventId == SH2_RESET) {
        reset_events++;
        if (initialized) reenable_pending = true;     /* BNO08x::re_enable_reports */
    } else if (ev->eventId == SH2_GET_FEATURE_RESP) {
        feature_resps++;
    }
}

static float clampf(float x, float r) { return x > r ? r : (x < -r ? -r : x); }

static void sensor_cb(void *cookie, sh2_SensorEvent_t *ev)
{
    (void)cookie;
    int i = bno_emu_sensor_index(ev->reportId);
    if (i < 0) return;
    rx_stat_t *r = &rx[i];
    const bno_emu_last_t *l = &emu.last[i];

    if (hk_state[i][ev->report[1]] != HK_READ) hk_rx_not_read++;
    else { hk_state[i][ev->report[1]] = HK_RECEIVED; hk_received++; }

    if (r->n > 0) {
        double dt = (double)(now - r->t_last);
        if (r->n_dt == 0 || dt < r->min_dt) r->min_dt = dt;
        if (r->n_dt == 0 || dt > r->max_dt) r->max_dt = dt;
        r->sum_dt += dt;
        r->n_dt++;
    } else {
        r->t_first = now;
    }
    r->t_last = now;
    r->n++;

    if (ev->report[1] != l->report_seq) { r->seq_mismatch++; return; }

    sh2_SensorValue_t v;
    sh2_decodeSensorEvent(&v, ev);
    float dec[4];
    switch (i) {
    case 0: dec[0] = v.un.accelerometer.x; dec[1] = v.un.accelerometer.y; dec[2] = v.un.accelerometer.z; break;
    case 1: dec[0] = v.un.gyroscope.x; dec[1] = v.un.gyroscope.y; dec[2] = v.un.gyroscope.z; break;
    default: dec[0] = v.un.gameRotationVector.i; dec[1] = v.un.gameRotationVector.j;
             dec[2] = v.un.gameRotationVector.k; dec[3] = v.un.gameRotationVector.real; break;
    }
    float range = (i == 0) ? emu.cfg.accel_range : (i == 1) ? emu.cfg.gyro_range : 1.9999f;
    double lsb = 1.0 / (double)(1u << k_q[i]);
    for (unsigned k = 0; k < k_nv[i]; k++) {
        int16_t raw = (int16_t)(ev->report[4 + 2 * k] | (ev->report[5 + 2 * k] << 8));
        if (raw != l->raw[k]) r->raw_mismatch++;
        if (fabs(dec[k] - raw * lsb) > 1e-6 * (1.0 + fabs(dec[k]))) r->decode_mismatch++;
        double err = fabs(raw * lsb - (double)clampf(l->in[k], range)) / lsb;
        if (err > r->max_qerr_lsb) r->max_qerr_lsb = err;
    }
}

static void enable_reports(uint32_t period);

/* The driver's service task: on an INT edge call sh2_service(); after a
 * reset-complete event re-enable the reports. */
static void service_for(uint64_t us)
{
    uint64_t end = now + us;
    while (now < end) {
        if (reenable_pending) {
            reenable_pending = false;
            enable_reports(period_req);
        } else if (hint_flag) {
            sh2_service();
        } else {
            advance(STEP_US);
        }
    }
}

static void enable_reports(uint32_t period)
{
    for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) {
        sh2_SensorConfig_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.reportInterval_us = period;
        sh2_setSensorConfig(k_ids[i], &cfg);
        service_for(SET_FEATURE_WAIT);
    }
}

typedef struct {
    bool aborted;
    int open_rc, prodid_rc;
    uint8_t prod_entries;
    uint32_t reset_events_in_open;
} init_res_t;

/* BNO08x::init_sh2_HAL + imu_init (enable 3 reports). Aborts after 3 s of
 * virtual time (sh2_getProdIds has no timeout of its own, G0 v1 R1). */
static init_res_t host_init(uint32_t period)
{
    init_res_t res;
    memset(&res, 0, sizeof(res));
    period_req = period;
    initialized = false;
    if (sh2_is_open) { sh2_close(); sh2_is_open = false; }

    guard_on = true;
    guard_until = now + 3000000u;
    if (setjmp(abort_jmp)) {
        guard_on = false;
        res.aborted = true;
        return res;
    }
    toggle_reset();
    uint32_t ev0 = reset_events;
    res.open_rc = sh2_open(&hal, hal_cb, NULL);
    sh2_is_open = (res.open_rc == SH2_OK);
    res.reset_events_in_open = reset_events - ev0;

    sh2_ProductIds_t ids;
    memset(&ids, 0, sizeof(ids));
    res.prodid_rc = sh2_getProdIds(&ids);
    res.prod_entries = ids.numEntries;
    sh2_setSensorCallback(sensor_cb, NULL);
    initialized = true;
    enable_reports(period);
    guard_on = false;
    return res;
}

/* ── Reporting ──────────────────────────────────────────────────────────── */

static bool pass = true;

static void fail(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    printf("FAIL: ");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    pass = false;
}

static void print_rx(const char *tag, uint32_t period)
{
    for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) {
        rx_stat_t *r = &rx[i];
        double mean = r->n_dt ? r->sum_dt / r->n_dt : 0.0;
        printf("%s.id%02X.n=%u\n", tag, k_ids[i], r->n);
        printf("%s.id%02X.mean_dt_us=%.3f\n", tag, k_ids[i], mean);
        printf("%s.id%02X.min_dt_us=%.0f\n", tag, k_ids[i], r->min_dt);
        printf("%s.id%02X.max_dt_us=%.0f\n", tag, k_ids[i], r->max_dt);
        printf("%s.id%02X.period_err_pct=%.4f\n", tag, k_ids[i], period ? 100.0 * (mean - period) / period : 0.0);
        printf("%s.id%02X.raw_mismatch=%u\n", tag, k_ids[i], r->raw_mismatch);
        printf("%s.id%02X.seq_mismatch=%u\n", tag, k_ids[i], r->seq_mismatch);
        printf("%s.id%02X.decode_mismatch=%u\n", tag, k_ids[i], r->decode_mismatch);
        printf("%s.id%02X.max_qerr_lsb=%.4f\n", tag, k_ids[i], r->max_qerr_lsb);
    }
}

static void print_emu(void)
{
    const bno_emu_stats_t *s = &emu.st;
    printf("emu.reads=%u\nemu.empty_reads=%u\nemu.writes=%u\nemu.collisions=%u\n", s->reads, s->empty_reads, s->writes, s->collisions);
    printf("emu.int_timeouts=%u\nemu.resets=%u\nemu.input_overruns=%u\nemu.late_resyncs=%u\n", s->int_timeouts, s->resets, s->input_overruns, s->late_resyncs);
    printf("emu.stale_skips=%u\nemu.unknown_cmds=%u\nemu.prod_id_reqs=%u\nemu.set_features=%u\n", s->stale_skips, s->unknown_cmds, s->prod_id_reqs, s->set_features);
    printf("emu.get_features=%u\nemu.adverts=%u\nemu.q_drops=%u\n", s->get_features, s->adverts, s->q_drops);
    printf("host.hard_resets=%u\nhost.reset_events=%u\nhost.feature_resps=%u\nhost.int_edges=%u\n", hard_resets, reset_events, feature_resps, int_edges);
    printf("t_end_s=%.3f\n", (double)now * 1e-6);
    if (port_mode)
        printf("port.ticks=%u\nport.rearms=%u\nport.mismatch=%u\n", port_ticks, port_rearms, port_mismatch);
}

static bool check_init(const init_res_t *r, const char *tag)
{
    printf("%s.aborted=%d\n%s.open_rc=%d\n%s.reset_events_in_open=%u\n%s.prodid_rc=%d\n%s.prod_entries=%u\n",
           tag, r->aborted, tag, r->open_rc, tag, r->reset_events_in_open, tag, r->prodid_rc, tag, r->prod_entries);
    bool ok = !r->aborted && r->open_rc == SH2_OK && r->reset_events_in_open >= 1 &&
              r->prodid_rc == SH2_OK && r->prod_entries == 4;
    if (!ok) fail("%s: init failed", tag);
    return ok;
}

/* Data checks common to every scenario (transport exact, quantization). */
static void check_data(void)
{
    for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) {
        if (rx[i].raw_mismatch || rx[i].seq_mismatch || rx[i].decode_mismatch)
            fail("id%02X: %u raw / %u seq / %u decode mismatches", k_ids[i], rx[i].raw_mismatch, rx[i].seq_mismatch, rx[i].decode_mismatch);
        if (rx[i].max_qerr_lsb > 0.5 + 1e-3)
            fail("id%02X: quantization error %.4f LSB > 0.5", k_ids[i], rx[i].max_qerr_lsb);
    }
}

static void check_period(uint32_t period, double seconds)
{
    for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) {
        double mean = rx[i].n_dt ? rx[i].sum_dt / rx[i].n_dt : 0.0;
        double err = period ? fabs(mean - period) / period : 1.0;
        double expected = seconds * 1e6 / period;
        if (err > 0.01) fail("id%02X: period error %.3f %% > 1 %%", k_ids[i], 100.0 * err);
        if (rx[i].n + 2 < expected) fail("id%02X: %u reports, expected about %.0f", k_ids[i], rx[i].n, expected);
    }
}

/* ── Scenarios ──────────────────────────────────────────────────────────── */

static void sc_boot(uint32_t period, int n_boots)
{
    int ok_boots = 0;
    for (int b = 0; b < n_boots; b++) {
        char tag[32];
        snprintf(tag, sizeof(tag), "boot%02d", b);
        uint32_t hr0 = hard_resets;
        init_res_t r = host_init(period);
        bool ok = check_init(&r, tag);
        rx_clear();
        service_for(1000000u);
        for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++)
            if (rx[i].n == 0) { fail("%s: no report id%02X within 1 s", tag, k_ids[i]); ok = false; }
        if (hard_resets != hr0) { fail("%s: %u host INT timeouts", tag, hard_resets - hr0); ok = false; }
        check_data();
        ok_boots += ok;
    }
    printf("boots_ok=%d\nboots=%d\n", ok_boots, n_boots);
}

static void sc_run(uint32_t period, double seconds)
{
    init_res_t r = host_init(period);
    if (!check_init(&r, "init")) return;
    uint32_t hr0 = hard_resets, ov0 = emu.st.input_overruns;
    rx_clear();
    service_for((uint64_t)(seconds * 1e6));
    print_rx("run", period);
    check_data();
    check_period(period, seconds);
    if (hard_resets != hr0) fail("%u host INT timeouts during the run", hard_resets - hr0);
    if (emu.st.input_overruns != ov0) fail("%u reports lost (due again before sent)", emu.st.input_overruns - ov0);
}

/* Host writes while the emulator has a packet waiting (G0 v1 §4.3). */
static void sc_collision(uint32_t period, double seconds)
{
    init_res_t r = host_init(period);
    if (!check_init(&r, "init")) return;
    uint32_t col0 = emu.st.collisions, fr0 = feature_resps, sf0 = emu.st.set_features, hr0 = hard_resets;
    rx_clear();
    uint64_t end = now + (uint64_t)(seconds * 1e6);
    uint32_t n_cmd = 0;
    while (now < end) {
        service_for(7000u);                         /* odd spacing: hits pending packets */
        sh2_SensorConfig_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.reportInterval_us = period;
        sh2_setSensorConfig(SH2_GYROSCOPE_CALIBRATED, &cfg);
        n_cmd++;
    }
    service_for(100000u);
    print_rx("collision", period);
    printf("collision.commands=%u\ncollision.collisions=%u\ncollision.set_features_seen=%u\ncollision.feature_resps=%u\n",
           n_cmd, emu.st.collisions - col0, emu.st.set_features - sf0, feature_resps - fr0);
    if (emu.st.collisions == col0) fail("no collision happened, scenario did not test anything");
    if (emu.st.set_features - sf0 != n_cmd) fail("emulator saw %u of %u Set Feature commands", emu.st.set_features - sf0, n_cmd);
    if (feature_resps - fr0 != n_cmd) fail("host got %u of %u feature responses", feature_resps - fr0, n_cmd);
    if (hard_resets != hr0) fail("%u host INT timeouts", hard_resets - hr0);
    check_data();
    check_period(period, seconds);
}

/* RST pulse while running (G0 v1 §7 S4): chip reboots, host re-enables. */
static void sc_rst_mid(uint32_t period)
{
    init_res_t r = host_init(period);
    if (!check_init(&r, "init")) return;
    service_for(5000000u);
    uint32_t ev0 = reset_events, hr0 = hard_resets;
    toggle_reset();                                 /* RST pulse from the host side */
    uint64_t t_rel = now;
    rx_clear();
    service_for(5000000u);
    double resume_ms = 0.0;
    for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) {
        double ms = rx[i].n ? (double)(rx[i].t_first - t_rel) * 1e-3 : 1e9;
        printf("rst_mid.id%02X.resume_ms=%.1f\n", k_ids[i], ms);
        if (ms > resume_ms) resume_ms = ms;
    }
    print_rx("rst_mid", period);
    if (reset_events - ev0 != 1) fail("expected 1 reset event, got %u", reset_events - ev0);
    if (resume_ms > 1000.0) fail("reports resumed after %.1f ms > 1000 ms", resume_ms);
    if (hard_resets != hr0) fail("%u host INT timeouts", hard_resets - hr0);
    check_data();
}

/* Host does not answer INT for 50 ms (G0 v1 §4.4). */
static void sc_silent(uint32_t period)
{
    init_res_t r = host_init(period);
    if (!check_init(&r, "init")) return;
    service_for(3000000u);
    uint32_t to0 = emu.st.int_timeouts, hr0 = hard_resets;
    advance(50000u);                                /* host task blocked */
    uint64_t t_back = now;
    rx_clear();
    service_for(3000000u);
    uint32_t to = emu.st.int_timeouts - to0;
    double resume_ms = 0.0;
    for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) {
        double ms = rx[i].n ? (double)(rx[i].t_first - t_back) * 1e-3 : 1e9;
        if (ms > resume_ms) resume_ms = ms;
    }
    printf("silent.int_timeouts=%u\nsilent.resume_ms=%.1f\n", to, resume_ms);
    print_rx("silent", period);
    if (to == 0) fail("emulator never released INT while the host was silent");
    if (resume_ms > 3.0 * period * 1e-3 + 1.0) fail("reports resumed after %.1f ms", resume_ms);
    if (hard_resets != hr0) fail("%u host INT timeouts", hard_resets - hr0);
    check_data();
}

/* Plant frozen (imu_seq stops changing): reports stop, then resume (G0 v1 §3.2). */
static void sc_stale(uint32_t period)
{
    init_res_t r = host_init(period);
    if (!check_init(&r, "init")) return;
    service_for(3000000u);
    plant_on = false;
    uint64_t t_freeze = now;
    uint32_t n0[BNO_EMU_N_SENSORS], late[BNO_EMU_N_SENSORS];
    for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) n0[i] = rx[i].n;
    /* stale after 3 periods; allow one more period for the packet in flight */
    service_for((uint64_t)period * (emu.cfg.stale_periods + 1u));
    for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) late[i] = rx[i].n;
    service_for(2000000u);
    uint32_t during = 0;
    for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) during += rx[i].n - late[i];
    plant_on = true;
    uint64_t t_back = now;
    rx_clear();
    service_for(3000000u);
    double resume_ms = 0.0;
    for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) {
        double ms = rx[i].n ? (double)(rx[i].t_first - t_back) * 1e-3 : 1e9;
        if (ms > resume_ms) resume_ms = ms;
        printf("stale.id%02X.reports_after_freeze_grace=%u\n", k_ids[i], late[i] - n0[i]);
    }
    printf("stale.reports_while_frozen=%u\nstale.resume_ms=%.1f\nstale.freeze_s=%.3f\n", during, resume_ms, (double)(t_back - t_freeze) * 1e-6);
    if (during != 0) fail("%u reports while the plant was frozen", during);
    if (emu.st.stale_skips == 0) fail("emulator never skipped a stale report");
    if (resume_ms > 2.0 * period * 1e-3 + 1.0) fail("reports resumed after %.1f ms", resume_ms);
    check_data();
}

/* S06 T8 (V3-1): IMU loss injected for 1.5 s in the middle of a run */
static void sc_hold(uint32_t period)
{
    init_res_t r = host_init(period);
    if (!check_init(&r, "init")) return;
    service_for(2000000u);
    uint32_t built0 = hk_built, skips0 = emu.st.hold_skips;
    rx_clear();
    hk_in_hold = true;
    bno_emu_set_hold(&emu, true, (uint32_t)now);
    service_for(1500000u);
    uint32_t rx_hold = rx[0].n + rx[1].n + rx[2].n;
    hk_in_hold = false;
    bno_emu_set_hold(&emu, false, (uint32_t)now);
    rx_clear();
    service_for(2000000u);
    uint32_t rx_after = rx[0].n + rx[1].n + rx[2].n;
    printf("hold_rx_during=%u\nhold_rx_after=%u\nhold_skips=%u\n", rx_hold, rx_after, emu.st.hold_skips - skips0);
    if (hk_hold_on != 1 || hk_hold_off != 1) fail("HOLD events on=%u off=%u (want 1, 1)", hk_hold_on, hk_hold_off);
    if (hk_rpt_in_hold) fail("%u reports built during the hold", hk_rpt_in_hold);
    if (emu.st.hold_skips == skips0) fail("hold_skips did not grow");
    /* at most what was already queued at hold start reaches the host */
    if (rx_hold > 3) fail("%u reports reached the host during the hold", rx_hold);
    if (rx_after < (uint32_t)(3 * 2000000u / period) * 9 / 10) fail("only %u reports in 2 s after the hold", rx_after);
    if (hk_built == built0) fail("no report built after the hold");
}

int main(int argc, char **argv)
{
    const char *sc = argc > 1 ? argv[1] : "run";
    uint32_t period = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 50000u;
    double seconds = argc > 3 ? atof(argv[3]) : 600.0;

    const char *pm = getenv("BNOEMU_H7_PORT");
    port_mode = pm && pm[0] == '1';

    bno_emu_cfg_t cfg;
    bno_emu_default_cfg(&cfg);
    bno_emu_init(&emu, &cfg, 0);
    bno_emu_set_hook(&emu, hook, NULL);
    if (port_mode) { port_next_tick = PORT_TICK_US; port_rearm(); }
    printf("scenario=%s\nperiod_us=%u\nh7_port=%d\n", sc, period, (int)port_mode);

    if (strcmp(sc, "boot") == 0)           sc_boot(period, argc > 3 ? atoi(argv[3]) : 20);
    else if (strcmp(sc, "run") == 0)       sc_run(period, seconds);
    else if (strcmp(sc, "collision") == 0) sc_collision(period, argc > 3 ? seconds : 10.0);
    else if (strcmp(sc, "rst_mid") == 0)   sc_rst_mid(period);
    else if (strcmp(sc, "silent") == 0)    sc_silent(period);
    else if (strcmp(sc, "stale") == 0)     sc_stale(period);
    else if (strcmp(sc, "hold") == 0)      sc_hold(period);
    else { printf("unknown scenario\n"); return 2; }

    /* S06 hook checks, every scenario */
    printf("hook_built=%u\nhook_qfull=%u\nhook_read=%u\nhook_received=%u\nhook_reset_done=%u\n"
           "host_reset_events=%u\nhook_seq_gap=%u\nhook_read_not_built=%u\nhook_rx_not_read=%u\n",
           hk_built, hk_qfull, hk_read, hk_received, hk_reset_done, reset_events, hk_seq_gap,
           hk_read_not_built, hk_rx_not_read);
    if (hk_seq_gap) fail("rpt_seq not +1 per built report within a reset (%u)", hk_seq_gap);
    if (hk_read_not_built) fail("%u reports read by the host without a REPORT event", hk_read_not_built);
    if (hk_rx_not_read) fail("%u reports at the host callback without a PKT_READ event", hk_rx_not_read);
    if (hk_built == 0) fail("no REPORT event");

    print_emu();
    printf("RESULT=%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
