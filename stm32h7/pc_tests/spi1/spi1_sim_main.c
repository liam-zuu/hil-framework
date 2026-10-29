/* PC harness for stm32h7/Core/Src/spi1_handler.c (tools/tests/test_h7_spi1_frame.py).
 *
 * One "frame" = the harness writes the 40 MOSI bytes into the rx buffer the DMA
 * was armed with, checks the tx buffer was not changed since it was armed (a
 * real SPI slave DMA pulls tx bytes into the TX FIFO as soon as it is armed and
 * while the frame is clocked out), then calls SPI1_Callback() as the interrupt.
 * Main-loop steps run between frames, or (OLD_API only) with a frame between
 * two calls, which is where an interrupt can land on the board.
 * Not covered: an interrupt in the middle of one C struct copy (cannot be
 * forced on the PC); for that the test checks the copy runs with interrupts
 * masked (fake PRIMASK) and that PRIMASK is restored.
 *
 * Prints one line per check: "PASS <name>" or "FAIL <name>: <why>"; exit 1 if
 * any check failed. Build with -DOLD_API for the file before 29/10 (only the
 * checks that the old API can run). */
#include <stdio.h>
#include <string.h>
#include "stm32h7xx_hal.h"
#include "hil_protocol.h"
#include "spi1_handler.h"
#ifndef OLD_API
#include "s06_log.h"
#endif

GPIO_TypeDef fake_gpioa;
DWT_Type     fake_dwt;
int  fake_irq_disabled;
long fake_irq_sections;

static SPI_TypeDef       _spi_inst;
static SPI_HandleTypeDef _hspi = { &_spi_inst, 0 };
static uint8_t *_armed_tx, *_armed_rx;
static uint8_t  _armed_copy[SPI1_FRAME_LEN];   /* tx bytes at arm time */
static uint8_t  _sent[SPI1_FRAME_LEN];         /* tx bytes of the last frame */
static int      _tx_changed_after_arm;
static int      _fails;
/* S06: MOSI bytes and CYCCNT of each frame k (k < 256), set in frame() */
static uint8_t  _mosi_of[256][SPI1_FRAME_LEN];
static uint32_t _cyc_of[256];

HAL_StatusTypeDef HAL_SPI_TransmitReceive_DMA(SPI_HandleTypeDef *h, uint8_t *tx, uint8_t *rx, uint16_t n)
{
    (void)h; (void)n;
    _armed_tx = tx;
    _armed_rx = rx;
    memcpy(_armed_copy, tx, SPI1_FRAME_LEN);
    return HAL_OK;
}
HAL_StatusTypeDef HAL_SPI_Abort(SPI_HandleTypeDef *h) { (void)h; return HAL_OK; }
GPIO_PinState HAL_GPIO_ReadPin(GPIO_TypeDef *p, uint16_t pin) { (void)p; (void)pin; return GPIO_PIN_SET; }
uint32_t HAL_GetTick(void) { return 0; }

static void check(int ok, const char *name, const char *why)
{
    if (ok) printf("PASS %s\n", name);
    else { printf("FAIL %s: %s\n", name, why); _fails++; }
}

static void wr16(uint8_t *p, int16_t v) { p[0] = (uint8_t)((uint16_t)v >> 8); p[1] = (uint8_t)v; }

/* MOSI frame whose every field is derived from k, so a mix of two frames shows */
static void frame(int k, int is_new)
{
    uint8_t m[SPI1_FRAME_LEN];
    memset(m, 0, sizeof m);
    for (int i = 0; i < 4; i++) wr16(&m[i * 2], (int16_t)(100 * k + i));
    m[21] = (uint8_t)k;                          /* state.seq */
    wr16(&m[24], (int16_t)(1000 + k));           /* vx_ref */
    wr16(&m[26], (int16_t)(2000 + k));           /* vy_ref */
    wr16(&m[28], (int16_t)(3000 + k));           /* wz_ref */
    m[30] = (uint8_t)(HIL_TRAJ_VALID | (is_new ? HIL_TRAJ_NEW : 0));
    m[39] = (uint8_t)k;
    wr16(&m[22], (int16_t)(7 * k));              /* gz  */
    for (int i = 0; i < 4; i++) wr16(&m[31 + 2 * i], (int16_t)(11 * k + i));   /* ax..yaw */
    if (memcmp(_armed_tx, _armed_copy, SPI1_FRAME_LEN) != 0) _tx_changed_after_arm++;
    memcpy(_sent, _armed_copy, SPI1_FRAME_LEN);
    memcpy(_armed_rx, m, SPI1_FRAME_LEN);
    fake_dwt.CYCCNT += 480000u + (uint32_t)k;    /* 1 ms at 480 MHz, distinct per frame */
    memcpy(_mosi_of[k & 0xFF], m, SPI1_FRAME_LEN);
    _cyc_of[k & 0xFF] = fake_dwt.CYCCNT;
    SPI1_Callback(&_hspi);
}

static int seq_of_state(const HIL_State_t *s) { return s->seq; }
static int k_of_traj(const HIL_TrajRef_t *t)  /* vx_ref → k */
{
    float u = t->vx_ref * (INT16_MAXF / HIL_REF_VEL_MAX);
    return (int)(u + (u >= 0 ? 0.5f : -0.5f)) - 1000;
}

/* Main-loop reply for a frame: torque[0] carries the frame seq too */
static void reply(int seq)
{
    HIL_Output_t o;
    memset(&o, 0, sizeof o);
    o.torque[0] = (float)seq / 1000.0f;
    o.seq = (uint8_t)seq;
    SPI1_SetOutput(&o);
}

#ifndef OLD_API
/* Reference CRC-16/CCITT-FALSE, bit by bit (independent of s06_log.c) */
static uint16_t ref_crc(const uint8_t *d, int n)
{
    uint16_t c = 0xFFFF;
    for (int i = 0; i < n; i++) {
        c ^= (uint16_t)(d[i] << 8);
        for (int b = 0; b < 8; b++) c = (c & 0x8000) ? (uint16_t)((c << 1) ^ 0x1021) : (uint16_t)(c << 1);
    }
    return c;
}
static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t le32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static int miso_crc_ok(const uint8_t *b)
{
    uint8_t t[SPI1_FRAME_LEN];
    memcpy(t, b, sizeof t);
    t[8] = t[9] = 0;
    uint8_t u[SPI1_FRAME_LEN - 2];
    memcpy(u, t, 8);
    memcpy(u + 8, t + 10, SPI1_FRAME_LEN - 10);
    return ref_crc(u, SPI1_FRAME_LEN - 2) == le16(b + 8);
}
/* Main loop as in main.c: one frame, checksum of its raw copy, reply */
static void main_step(void)
{
    HIL_State_t s; HIL_TrajRef_t t; SPI1_FrameInfo_t fi; SPI1_MisoInfo_t mi;
    HIL_Output_t o;
    SPI1_GetFrameEx(&s, &t, &fi);
    mi.mosi_crc = S06_Crc16(0xFFFF, fi.raw, SPI1_FRAME_LEN);
    mi.rx_cyc = fi.rx_cyc;
    mi.rx_nf = fi.rx_nf;
    mi.age_us = 7;
    memset(&o, 0, sizeof o);
    o.seq = s.seq;
    SPI1_SetOutputEx(&o, &mi);
}
#endif

int main(void)
{
#ifndef OLD_API
    S06Log_Init();
#endif
    SPI1_Handler_Init(&_hspi);

#ifdef OLD_API
    /* Interrupt between the two reads of the old main loop */
    {
        HIL_State_t s; HIL_TrajRef_t t;
        frame(1, 0);
        SPI1_GetState(&s);
        frame(2, 0);
        SPI1_GetTrajRef(&t);
        check(seq_of_state(&s) == k_of_traj(&t), "pair_same_frame", "state and traj from different frames");
    }
    /* Reply written while the DMA holds the buffer */
    {
        _tx_changed_after_arm = 0;
        for (int k = 3; k < 13; k++) {
            HIL_State_t s; HIL_TrajRef_t t;
            frame(k, 0);
            SPI1_GetState(&s);
            SPI1_GetTrajRef(&t);
            reply(s.seq);
        }
        frame(13, 0);
        check(_tx_changed_after_arm == 0, "tx_untouched_while_armed", "main loop wrote the armed tx buffer");
    }
#else
    /* 1. state + traj of one frame, even with a frame between two main-loop polls */
    {
        HIL_State_t s; HIL_TrajRef_t t;
        frame(1, 0);
        frame(2, 0);
        long before = fake_irq_sections;
        SPI1_GetFrame(&s, &t);
        check(seq_of_state(&s) == 2 && k_of_traj(&t) == 2, "pair_same_frame", "state/traj not both from frame 2");
        check(fake_irq_sections > before && fake_irq_disabled == 0, "getframe_masked",
              "copy not under masked interrupts, or PRIMASK not restored");
        check(!SPI1_IsFrameReady(), "getframe_clears_ready", "frame_ready still set");
        SPI1_Stats_t st; SPI1_GetStats(&st);
        check(st.unread == 1, "unread_counted", "frame 1 overwritten unread but unread != 1");
    }
    /* 2. tx buffer never written while armed; reply to frame N goes out in frame N+2 */
    {
        _tx_changed_after_arm = 0;
        int lag_ok = 1;
        for (int k = 10; k < 30; k++) {
            HIL_State_t s; HIL_TrajRef_t t;
            frame(k, 0);
            if (k >= 12 && _sent[17] != (uint8_t)(k - 2)) lag_ok = 0;   /* MISO seq byte */
            SPI1_GetFrame(&s, &t);
            reply(s.seq);
        }
        check(_tx_changed_after_arm == 0, "tx_untouched_while_armed", "main loop wrote the armed tx buffer");
        check(lag_ok, "reply_lag_two_frames", "MISO seq of frame k is not k-2");
        int before_mask = fake_irq_disabled;
        reply(99);
        check(before_mask == 0 && fake_irq_disabled == 0, "setoutput_restores_primask", "PRIMASK left set");
    }
    /* 3. a NEW setpoint the main loop did not see is kept for the relay */
    {
        HIL_State_t s; HIL_TrajRef_t t, r; uint8_t seq = 0;
        SPI1_Stats_t a, b;
        SPI1_GetStats(&a);
        frame(40, 1);            /* NEW */
        frame(41, 0);
        frame(42, 0);            /* main loop only now: latest frame has no NEW */
        SPI1_GetFrame(&s, &t);
        int got = SPI1_TakeNewTraj(&r, &seq);
        check(got && seq == 40 && k_of_traj(&r) == 40 && (r.traj_flags & HIL_TRAJ_NEW),
              "new_not_lost", "NEW of frame 40 not returned");
        check(!SPI1_TakeNewTraj(&r, &seq), "new_taken_once", "same setpoint returned twice");
        frame(43, 1);
        frame(44, 1);            /* replaces 43 before the main loop takes it */
        got = SPI1_TakeNewTraj(&r, &seq);
        check(got && seq == 44 && k_of_traj(&r) == 44, "new_latest_wins", "pending not replaced by the newer NEW");
        frame(45, 1);
        SPI1_DropNewTraj();      /* link lost */
        check(!SPI1_TakeNewTraj(&r, &seq), "new_dropped_at_link_lost", "pending survived DropNewTraj");
        SPI1_GetStats(&b);
        check(b.new_seen - a.new_seen == 4 && b.new_taken - a.new_taken == 2 &&
              b.new_replaced - a.new_replaced == 1 && b.new_dropped - a.new_dropped == 1,
              "new_counters", "seen != taken + replaced + dropped");
        check(fake_irq_disabled == 0, "take_restores_primask", "PRIMASK left set");
    }
    /* 4. called with interrupts already masked: PRIMASK stays masked */
    {
        HIL_State_t s; HIL_TrajRef_t t, r;
        frame(50, 0);
        __disable_irq();
        SPI1_GetFrame(&s, &t);
        SPI1_TakeNewTraj(&r, NULL);
        check(fake_irq_disabled == 1, "nested_mask_kept", "a call re-enabled interrupts masked by its caller");
        __enable_irq();
    }
    /* 5. S06 CRC: table version equals the bit-by-bit reference; check value */
    {
        const uint8_t s9[] = "123456789";
        uint8_t r[64];
        for (int i = 0; i < 64; i++) r[i] = (uint8_t)(i * 37 + 5);
        check(S06_Crc16(0xFFFF, s9, 9) == 0x29B1 && ref_crc(s9, 9) == 0x29B1, "crc_check_value", "CRC(123456789) != 0x29B1");
        check(S06_Crc16(0xFFFF, r, 64) == ref_crc(r, 64), "crc_table_equals_ref", "table CRC differs from reference");
    }
    /* 6. MISO v4: frame k+2 carries checksum of frame k's raw MOSI, its
     *    stamp and count; MISO checksum valid on every frame sent */
    {
        int ok_mosi = 1, ok_cyc = 1, ok_nf = 1, ok_crc = 1, ok_age = 1;
        SPI1_Stats_t st0; SPI1_GetStats(&st0);
        frame(60, 0); main_step();
        frame(61, 0); main_step();
        for (int k = 62; k < 90; k++) {
            frame(k, 0);                         /* _sent = MISO clocked out in frame k */
            const int a = k - 2;                 /* frame it answers */
            if (!miso_crc_ok(_sent)) ok_crc = 0;
            if (_sent[17] != (uint8_t)a) ok_mosi = 0;
            if (le16(_sent + 10) != ref_crc(_mosi_of[a], SPI1_FRAME_LEN)) ok_mosi = 0;
            if (le32(_sent + 12) != _cyc_of[a]) ok_cyc = 0;
            if (le16(_sent + 18) != (uint16_t)(st0.frames + (uint32_t)(a - 60) + 1u)) ok_nf = 0;
            if (le16(_sent + 20) != 7) ok_age = 0;
            main_step();
        }
        check(ok_crc, "miso_crc_valid", "MISO checksum does not match its bytes");
        check(ok_mosi, "mosi_crc_of_answered_frame", "mosi_crc/seq is not that of frame k-2");
        check(ok_cyc, "rx_cyc_of_answered_frame", "rx_cyc is not the CYCCNT of frame k-2");
        check(ok_nf, "rx_nf_of_answered_frame", "rx_nf is not the frame count of frame k-2");
        check(ok_age, "age_field", "age_us not carried");
        /* one byte flipped -> checksum catches it */
        uint8_t bad[SPI1_FRAME_LEN];
        memcpy(bad, _sent, sizeof bad);
        bad[30] ^= 0x10;
        check(!miso_crc_ok(bad), "miso_crc_catches_flip", "flipped byte not detected");
    }
    /* 7. IMU copy for the report log: raw bytes and source frame of the
     *    same frame as the floats */
    {
        HIL_Imu_t im;
        SPI1_Stats_t st;
        frame(91, 0);
        SPI1_GetStats(&st);
        SPI1_GetImu(&im);
        uint8_t want[10] = { _mosi_of[91][22], _mosi_of[91][23] };
        memcpy(want + 2, &_mosi_of[91][31], 8);
        check(memcmp(im.raw, want, 10) == 0 && im.imu_seq == 91 && im.src_nf == st.frames,
              "imu_raw_same_frame", "IMU raw bytes / source frame not from frame 91");
        main_step();
    }
    /* 8. log records: each one leaves exactly once, in order, whether the
     *    main loop replies 0, 1 or 2 times between two frames */
    {
        enum { N = 60 };
        uint8_t pl[S06_LOG_PAYLOAD];
        uint16_t first = 0;
        int have_first = 0, ok_order = 1, ok_payload = 1;
        int n_seen = 0, last = -1;
        for (int i = 0; i < N; i++) {
            memset(pl, 0, sizeof pl);
            pl[0] = (uint8_t)i; pl[14] = (uint8_t)(0xA0 ^ i);
            S06Log_Push(S06_REC_TICK, pl, S06_LOG_PAYLOAD);
        }
        for (int k = 100; k < 100 + 3 * N && n_seen < N; k++) {
            frame(k & 0xFF, 0);
            if (_sent[22] == S06_REC_TICK) {
                uint16_t idx = le16(_sent + 23);
                if (!have_first) { first = idx; have_first = 1; }
                int rel = (uint16_t)(idx - first);
                if (rel != last) {                 /* new record (repeats keep the index) */
                    if (rel != last + 1) ok_order = 0;
                    if (_sent[25] != (uint8_t)rel || _sent[39] != (uint8_t)(0xA0 ^ rel)) ok_payload = 0;
                    last = rel;
                    n_seen++;
                }
            }
            int calls = (k % 3);                   /* 0, 1 or 2 replies before the next frame */
            for (int c = 0; c < calls; c++) main_step();
        }
        check(n_seen == N && ok_order, "log_each_record_once_in_order", "a record was lost, skipped or reordered");
        check(ok_payload, "log_payload_intact", "record payload changed on the way");
    }
    /* 9. ring full: dropped records still use an index (gap), drop count */
    {
        S06_Rec_t r;
        while (S06Log_Pop(&r)) { }
        uint32_t d0 = S06Log_Dropped();
        uint8_t pl[1] = { 0 };
        S06Log_Push(S06_REC_TICK, pl, 1);
        S06Log_Pop(&r);
        uint16_t i0 = r.idx;
        for (unsigned i = 0; i < S06_LOG_SLOTS + 10u; i++) S06Log_Push(S06_REC_TICK, pl, 1);
        int n = 0; uint16_t last_idx = 0;
        while (S06Log_Pop(&r)) { last_idx = r.idx; n++; }
        S06Log_Push(S06_REC_TICK, pl, 1);
        S06Log_Pop(&r);
        check(n == (int)S06_LOG_SLOTS && S06Log_Dropped() - d0 == 10u, "log_full_drops_counted",
              "ring did not keep 512 and count 10 drops");
        check(last_idx == (uint16_t)(i0 + S06_LOG_SLOTS) && r.idx == (uint16_t)(i0 + S06_LOG_SLOTS + 11u),
              "log_drop_leaves_index_gap", "dropped records did not consume an index");
        check(fake_irq_disabled == 0, "log_restores_primask", "PRIMASK left set");
    }
#endif
    return _fails ? 1 : 0;
}
