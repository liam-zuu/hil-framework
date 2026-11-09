/* ibus_gen.h — FlySky iBUS frame generator for the DUT's RC input (QD 6, bac 1)
 *
 * The H7 stands in for the RC receiver so the vehicle binary runs unchanged
 * on the HIL bench: SWA (channel 7) = 2000 enables the motors, SWC (channel
 * 5) = 1000 keeps AUTO, the other channels 1500 (03_CODE/docs/
 * 2026-10-31_Danh_gia_Mot_binary.md, Q1; spec S06 T12).
 *
 * I8 (Liam 09/11, "Duyet ca 5"; thesis repo 03_CODE/docs/
 * 2026-11-09_De_xuat_I8_bit_SWA_va_reset_panel.md): the RPi5 flips SWA like
 * a person on the RC switch. Two frames are built at init, SWA 2000 and SWA
 * 1000 (everything else equal); the tick picks one only when a new frame
 * starts, so a frame on the wire never mixes the two. SWA off still sends
 * frames (the DUT has RC, motors off); link down sends none (the DUT loses
 * RC), as before.
 *
 * Frame (32 bytes): 0x20 0x40, 14 channels as uint16 little-endian, then
 * checksum = 0xFFFF - sum(bytes 0..29), little-endian. The ESP32 parser
 * (AGV_MECANUM src/ibus/ibus.c) reads the first 10 channels.
 *
 * Timing: ibus_gen_tick() runs from the TIM6 10 kHz interrupt and gets the
 * time in microseconds from the port, so a late or skipped interrupt shows
 * in the statistics instead of being counted as one tick (Codex I4). A frame
 * starts when the generator is on, at least period_us after the previous
 * start (kept across off/on, Codex I2) and only when the UART line is idle,
 * so the first byte goes on the wire at the recorded start time (Codex I3).
 * Its bytes go into the UART TX FIFO as room appears, so the frame leaves
 * the pin as one burst (about 2.8 ms at 115200). The period must stay at or
 * above about 3.7 ms (2.8 ms burst + 0.87 ms idle): shorter, and the ESP32
 * UART driver waits for 120 bytes before handing them over, which times
 * out the 20 ms read and turns the motors off.
 *
 * This file has no HAL calls; ibus_gen_port.c drives UART4 (PD1). The same
 * core is compiled on the PC by ibusgen_pc/.
 */

#ifndef IBUS_GEN_H
#define IBUS_GEN_H

#include <stdint.h>

#define IBUS_GEN_FRAME_LEN         32u
#define IBUS_GEN_NUM_CH            14u
#define IBUS_GEN_HEADER_0          0x20u
#define IBUS_GEN_HEADER_1          0x40u

#define IBUS_GEN_PERIOD_US         7000u     /* like a real receiver */
#define IBUS_GEN_MIN_PERIOD_US     3700u     /* floor, see above */
#define IBUS_GEN_WINDOW_US         10000000u /* 10 s summary window */
#define IBUS_GEN_MAX_PUSH          16u       /* bytes per tick at most */

/* Channel values for an automatic run (index = channel - 1) */
#define IBUS_GEN_CH_SWC            4u        /* channel 5 */
#define IBUS_GEN_CH_SWA            6u        /* channel 7 */
#define IBUS_GEN_VAL_SWC           1000u     /* not MANUAL (> 1700 is) */
#define IBUS_GEN_VAL_SWA           2000u     /* motors on (> 1500) */
#define IBUS_GEN_VAL_SWA_OFF       1000u     /* motors off (I8) */
#define IBUS_GEN_VAL_MID           1500u

typedef struct {
    int  (*tx_room)(void);        /* nonzero: the TX FIFO takes one more byte */
    int  (*tx_idle)(void);        /* nonzero: FIFO empty and last byte shifted out */
    void (*tx_put)(uint8_t b);
} ibus_gen_io_t;

/* One 10 s window, published by the tick. P = microseconds between the
 * starts of two frames in a row within one on period; the first frame
 * after an off period has no P. late_ticks counts ticks where the next
 * frame was due but the line was still busy. tick_gap_max_us is the longest
 * time between two calls of ibus_gen_tick (TIM6 service, nominal 100 us). */
typedef struct {
    uint32_t window;              /* 1, 2, ...; 0 = no window closed yet */
    uint32_t frames;
    uint32_t p_count;
    uint32_t p_min_us;            /* 0 when p_count = 0 */
    uint32_t p_max_us;
    uint32_t late_ticks;
    uint32_t tick_gap_max_us;
    uint32_t enables;             /* off -> on changes in the window */
    uint32_t total_frames;        /* since init */
    uint32_t enabled;             /* state at the end of the window */
    uint32_t swa_frames;          /* frames started with SWA 2000 in the window (I8) */
    uint32_t swa_changes;         /* frame starts whose SWA differs from the frame before (I8) */
    uint32_t swa;                 /* SWA of the last frame started, 1 = 2000 (end of window) */
} ibus_gen_window_t;

typedef struct {
    ibus_gen_io_t io;
    uint32_t period_us;
    uint8_t  frame[IBUS_GEN_FRAME_LEN];      /* SWA 2000 */
    uint8_t  frame_off[IBUS_GEN_FRAME_LEN];  /* SWA 1000 (I8) */
    const uint8_t *cur;           /* frame being written (picked at its start) */
    uint8_t  pos;                 /* bytes of the current frame written; FRAME_LEN = idle */
    volatile uint8_t last_swa;    /* SWA of the last frame started; 0 before any (I8) */
    uint8_t  have_swa;            /* last_swa comes from a started frame */
    uint8_t  on;
    uint8_t  have_start;          /* a frame was ever started (spacing) */
    uint8_t  p_valid;             /* previous start is in the current on period */
    uint8_t  have_tick;
    uint32_t last_start_us;
    uint32_t last_tick_us;
    uint32_t win_start_us;
    uint32_t win_no;
    uint32_t total_frames;
    ibus_gen_window_t acc;        /* window being filled */
    volatile uint32_t pub_ver;    /* odd while pub is written */
    ibus_gen_window_t pub;
} ibus_gen_t;

void ibus_gen_build_frame(uint8_t out[IBUS_GEN_FRAME_LEN], const uint16_t ch[IBUS_GEN_NUM_CH]);
/* period_us below IBUS_GEN_MIN_PERIOD_US is raised to it */
void ibus_gen_init(ibus_gen_t *g, const ibus_gen_io_t *io, uint32_t period_us, uint32_t now_us);
/* TIM6 interrupt only. on: frames may start (link up); a frame being
 * written always finishes. swa: SWA of the next frame to start (nonzero =
 * 2000); a frame already being written keeps its value (I8). Intervals use
 * unsigned differences of now_us. */
void ibus_gen_tick(ibus_gen_t *g, uint32_t now_us, int on, int swa);
/* SWA of the last frame started (1 = 2000), 0 before any; any context */
int  ibus_gen_last_swa(const ibus_gen_t *g);

/* I8-3 (Codex vong 1, Liam 09/11 "Duyet sua, vong 2"): MOSI has no checksum the
 * H7 can check, so one corrupted frame must not flip SWA. The main loop feeds
 * the bit of every applied frame; the output changes only after
 * IBUS_SWA_FILTER_N frames in a row agree (3 ms at 1 kHz), both directions.
 * Reset (output 0) at link lost and before the first frame after a link
 * timeout seen by the TIM6 tick (I8-R2-1: only a real timeout, not a short
 * gap between frames). */
#define IBUS_SWA_FILTER_N          3u
typedef struct {
    uint8_t out;                  /* filtered SWA, 0 after reset */
    uint8_t cand;                 /* value of the current run of equal bits */
    uint8_t cnt;                  /* length of that run (saturates) */
} ibus_swa_filter_t;
void ibus_swa_filter_reset(ibus_swa_filter_t *f);
int  ibus_swa_filter_step(ibus_swa_filter_t *f, int bit);
/* Copy of the last closed window; returns 0 if the tick kept rewriting it */
int  ibus_gen_read_window(const ibus_gen_t *g, ibus_gen_window_t *out);

#endif /* IBUS_GEN_H */
