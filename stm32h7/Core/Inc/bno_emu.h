/*
 * bno_emu.h: BNO085 emulator core (IMU option A, G0 v1 spec
 * docs/2026-10-22_Spec_G0_Gia_lap_BNO085.md).
 *
 * Plain C, no HAL calls: the same code runs on the PC (G1, linked against the
 * real sh2.c/shtp.c of the ESP32 firmware through a fake HAL) and on the H7
 * (G2, behind a thin SPI4/EXTI layer).
 *
 * The emulator does not compute IMU values. The plant (RPi5) sends them; this
 * core only packs them into SHTP/SH-2 the way the ESP32 driver reads a real
 * BNO085: advertisement + reset complete after reset, product IDs, Set/Get
 * Feature, then input reports (0x01 accelerometer, 0x02 calibrated gyro,
 * 0x08 game rotation vector) at the period the host asked for.
 *
 * Transaction model (one SPI transaction = CS low .. CS high):
 *   bno_emu_cs_fall()  returns the bytes to shift out on MISO (pending packet
 *                      plus 4 zero bytes, or a 4-byte header of length 0);
 *   bno_emu_cs_rise()  gets the bytes the host shifted in on MOSI. A non-zero
 *                      4-byte header means the host wrote a packet; all zeros
 *                      means it was a read.
 * INT (active low): bno_emu_int_low(). The caller turns its falling edges into
 * the host interrupt. Times are a free-running microsecond counter (wraps).
 */
#ifndef BNO_EMU_H
#define BNO_EMU_H

#include <stdbool.h>
#include <stdint.h>

#define BNO_EMU_PKT_MAX     300u   /* largest packet incl. 4-byte header       */
#define BNO_EMU_QUEUE_LEN   8u     /* outgoing packets waiting for the host     */
#define BNO_EMU_N_SENSORS   3u     /* accelerometer, calibrated gyro, game RV   */

/* Sensor report IDs (SH-2) */
#define BNO_EMU_ID_ACCEL    0x01u
#define BNO_EMU_ID_GYRO     0x02u
#define BNO_EMU_ID_GAME_RV  0x08u

/* One IMU sample from the plant. seq changes every time the plant has a new
 * sample; if it stops changing the emulator stops reporting (G0 v1 §3.2). */
typedef struct {
    float   ax, ay, az;         /* m/s^2 */
    float   gx, gy, gz;         /* rad/s */
    float   qi, qj, qk, qr;     /* game rotation vector, unit quaternion */
    uint8_t seq;
    /* S06 log only, not used by the core: H7 frame count and checksum of the
     * raw frame bytes the sample came from (bno_emu_port.c) */
    uint32_t src_nf;
    uint16_t src_crc;
} bno_emu_sample_t;

typedef struct {
    uint32_t boot_delay_us;        /* RST released -> first INT                */
    uint32_t int_timeout_us;       /* host did not answer INT: release, retry  */
    uint32_t int_retry_gap_us;     /* INT high time before a retry             */
    uint32_t empty_int_period_us;  /* INT with length 0 when nothing to send   */
    uint32_t stale_periods;        /* sample seq unchanged this many report
                                      periods -> stop reporting                */
    float    accel_range;          /* m/s^2, saturate (chip range)             */
    float    gyro_range;           /* rad/s, saturate (chip range)             */
} bno_emu_cfg_t;

/* Last report the emulator built for a sensor (for tests and debug output). */
typedef struct {
    uint8_t  report_seq;
    uint8_t  sample_seq;
    int16_t  raw[4];               /* fixed-point values as sent              */
    float    in[4];                /* plant values they came from             */
    uint32_t t_us;
    uint32_t count;
} bno_emu_last_t;

typedef struct {
    uint32_t reads, empty_reads, writes, collisions;
    uint32_t int_timeouts, resets, late_resyncs, stale_skips;
    uint32_t input_overruns;       /* report due again before the last one was sent */
    uint32_t unknown_cmds, prod_id_reqs, set_features, get_features, adverts;
    uint32_t q_drops;              /* queue full, packet not sent             */
    uint32_t hold_skips;           /* S06: reports due during IMU loss injection */
} bno_emu_stats_t;

typedef struct {
    uint16_t len;                               /* SHTP length incl. header */
    uint8_t  buf[BNO_EMU_PKT_MAX + 4u];         /* +4 zero pad (G0 v1 §2)   */
    uint8_t  rmask;                             /* S06: sensors with a report in it */
    uint8_t  rseq[3];                           /* S06: their report sequence numbers */
} bno_emu_pkt_t;

/* S06 event hook (spec S06 v3.4 T1 Z3, V3-1, V3-2, Y14). The core calls it, if
 * set, at the moment each event happens; it does not change the behaviour. */
typedef enum {
    BNO_EV_REPORT = 1,      /* a report was built: sensor, rpt_seq, flags, sample */
    BNO_EV_PKT_READ,        /* host clocked out a whole packet with reports: mask, seqs */
    BNO_EV_RESET_DONE,      /* reset-complete queued (end of boot)        */
    BNO_EV_HOLD,            /* IMU loss injection: flags 1 = on, 0 = off  */
} bno_emu_ev_type_t;

#define BNO_EV_QUEUED  0x01u        /* REPORT: its packet entered the queue    */
#define BNO_EV_QFULL   0x02u        /* REPORT: its packet was dropped, queue full */

typedef struct {
    bno_emu_ev_type_t type;
    uint8_t  sensor;                /* REPORT: 0 accel, 1 gyro, 2 game RV       */
    uint8_t  rpt_seq;               /* REPORT                                   */
    uint8_t  flags;
    uint8_t  mask;                  /* PKT_READ: bit i = sensor i in the packet  */
    uint8_t  seqs[3];               /* PKT_READ: rpt_seq of sensor i            */
    const bno_emu_sample_t *sample; /* REPORT: sample the values came from      */
    uint32_t now_us;
} bno_emu_event_t;

typedef void (*bno_emu_hook_t)(void *ctx, const bno_emu_event_t *ev);

typedef enum { BNO_EMU_HELD = 0, BNO_EMU_BOOTING, BNO_EMU_RUNNING } bno_emu_state_t;

typedef struct {
    bno_emu_cfg_t    cfg;
    bno_emu_state_t  state;
    uint32_t         t_state_us;

    /* Outgoing queue */
    bno_emu_pkt_t    q[BNO_EMU_QUEUE_LEN];
    uint8_t          q_head, q_count;
    uint8_t          chan_seq[6];               /* SHTP sequence per channel */

    /* INT and transaction state */
    bool             int_low;
    bool             in_xfer;
    bool             xfer_offered;              /* head packet put on MISO at CS fall */
    uint16_t         xfer_len;
    uint32_t         t_int_us;                  /* last INT edge (low or high) */
    uint32_t         t_int_allowed_us;          /* no new INT before this      */

    /* Sensors */
    uint32_t         period_us[BNO_EMU_N_SENSORS];  /* 0 = disabled          */
    uint32_t         next_due_us[BNO_EMU_N_SENSORS];
    uint8_t          rpt_seq[BNO_EMU_N_SENSORS];
    bool             pend[BNO_EMU_N_SENSORS];   /* due, waiting for the input channel */
    bno_emu_last_t   last[BNO_EMU_N_SENSORS];

    /* Plant sample */
    bno_emu_sample_t sample;
    bool             have_sample;
    uint32_t         t_sample_us;               /* sample arrived              */
    uint32_t         t_seq_change_us;           /* sample seq last changed     */

    bno_emu_stats_t  st;

    /* S06 */
    bool             hold;                      /* IMU loss injection: build no report */
    bno_emu_hook_t   hook;
    void            *hook_ctx;
} bno_emu_t;

void bno_emu_default_cfg(bno_emu_cfg_t *cfg);
void bno_emu_init(bno_emu_t *e, const bno_emu_cfg_t *cfg, uint32_t now_us);

/* RST pin: low = held in reset. */
void bno_emu_set_reset(bno_emu_t *e, bool rst_low, uint32_t now_us);
/* New plant sample (call at the plant rate, e.g. 1 kHz). */
void bno_emu_set_sample(bno_emu_t *e, const bno_emu_sample_t *s, uint32_t now_us);
/* Housekeeping: boot, report schedule, INT. Call often (every 10-100 us). */
void bno_emu_poll(bno_emu_t *e, uint32_t now_us);
bool bno_emu_int_low(const bno_emu_t *e);

/* What bno_emu_cs_fall() would put on MISO now, without changing any state.
 * The H7 copies it into its SPI DMA buffer before INT goes low, because the
 * slave must have its first byte ready when the host pulls CS low. */
const uint8_t *bno_emu_peek_tx(const bno_emu_t *e, uint16_t *tx_len);
const uint8_t *bno_emu_cs_fall(bno_emu_t *e, uint32_t now_us, uint16_t *tx_len);
void bno_emu_cs_rise(bno_emu_t *e, const uint8_t *mosi, uint16_t n_clocked, uint32_t now_us);

/* S06: event hook (after bno_emu_init, which clears it) and IMU loss
 * injection. While held the report schedule keeps running but no report is
 * built (rpt_seq does not advance); the host can still read and write. Not
 * cleared by a chip reset (it is the plant's fault, not the chip's). */
void bno_emu_set_hook(bno_emu_t *e, bno_emu_hook_t fn, void *ctx);
void bno_emu_set_hold(bno_emu_t *e, bool hold, uint32_t now_us);

/* Index 0..2 for report ID 0x01, 0x02, 0x08; -1 otherwise. */
int bno_emu_sensor_index(uint8_t report_id);

#endif /* BNO_EMU_H */
