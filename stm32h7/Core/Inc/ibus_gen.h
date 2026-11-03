/* ibus_gen.h — FlySky iBUS frame generator for the DUT's RC input (QD 6, bac 1)
 *
 * The H7 stands in for the RC receiver so the vehicle binary runs unchanged
 * on the HIL bench: SWA (channel 7) = 2000 enables the motors, SWC (channel
 * 5) = 1000 keeps AUTO, the other channels 1500 (03_CODE/docs/
 * 2026-10-31_Danh_gia_Mot_binary.md, Q1; spec S06 T12).
 *
 * Frame (32 bytes): 0x20 0x40, 14 channels as uint16 little-endian, then
 * checksum = 0xFFFF - sum(bytes 0..29), little-endian. The ESP32 parser
 * (AGV_MECANUM src/ibus/ibus.c) reads the first 10 channels.
 *
 * Timing: ibus_gen_tick() runs from the TIM6 10 kHz interrupt. A frame
 * starts every period_ticks (70 = 7 ms, like a real receiver) and its bytes
 * go into the UART TX FIFO as room appears, so the frame leaves the pin as
 * one burst (about 2.8 ms at 115200). The period must stay at or above
 * about 3.7 ms (2.8 ms burst + 0.87 ms idle): shorter, and the ESP32 UART
 * driver waits for 120 bytes before handing them over, which times out the
 * 20 ms read and turns the motors off.
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

#define IBUS_GEN_TICK_HZ           10000u    /* TIM6 rate */
#define IBUS_GEN_PERIOD_TICKS      70u       /* 7 ms */
#define IBUS_GEN_MIN_PERIOD_TICKS  37u       /* 3.7 ms floor, see above */
#define IBUS_GEN_WINDOW_TICKS      100000u   /* 10 s summary window */
#define IBUS_GEN_MAX_PUSH          16u       /* bytes per tick at most */

/* Channel values for an automatic run (index = channel - 1) */
#define IBUS_GEN_CH_SWC            4u        /* channel 5 */
#define IBUS_GEN_CH_SWA            6u        /* channel 7 */
#define IBUS_GEN_VAL_SWC           1000u     /* not MANUAL (> 1700 is) */
#define IBUS_GEN_VAL_SWA           2000u     /* motors on (> 1500) */
#define IBUS_GEN_VAL_MID           1500u

typedef struct {
    int  (*tx_room)(void);        /* nonzero: the TX FIFO takes one more byte */
    void (*tx_put)(uint8_t b);
} ibus_gen_io_t;

/* One 10 s window, published by the tick. P = ticks between the starts of
 * two frames in a row (0.1 ms each, H7 clock); the first frame after an
 * enable has no P. late_ticks counts ticks where the next frame was due but
 * the previous one was still waiting for FIFO room. */
typedef struct {
    uint32_t window;              /* 1, 2, ...; 0 = no window closed yet */
    uint32_t frames;
    uint32_t p_count;
    uint32_t p_min_ticks;         /* 0 when p_count = 0 */
    uint32_t p_max_ticks;
    uint32_t late_ticks;
    uint32_t enables;             /* off -> on changes in the window */
    uint32_t total_frames;        /* since init */
    uint32_t enabled;             /* state at the end of the window */
} ibus_gen_window_t;

typedef struct {
    ibus_gen_io_t io;
    uint32_t period_ticks;
    uint8_t  frame[IBUS_GEN_FRAME_LEN];
    uint8_t  pos;                 /* bytes of the current frame written; FRAME_LEN = idle */
    volatile uint8_t enable_req;  /* written by the main loop */
    uint8_t  enabled;
    uint8_t  have_last;
    uint32_t tick;
    uint32_t last_start;
    uint32_t win_start;
    uint32_t win_no;
    uint32_t total_frames;
    ibus_gen_window_t acc;        /* window being filled */
    volatile uint32_t pub_ver;    /* odd while pub is written */
    ibus_gen_window_t pub;
} ibus_gen_t;

void ibus_gen_build_frame(uint8_t out[IBUS_GEN_FRAME_LEN], const uint16_t ch[IBUS_GEN_NUM_CH]);
/* period_ticks below IBUS_GEN_MIN_PERIOD_TICKS is raised to it */
void ibus_gen_init(ibus_gen_t *g, const ibus_gen_io_t *io, uint32_t period_ticks);
void ibus_gen_set_enabled(ibus_gen_t *g, int on);   /* main loop */
void ibus_gen_tick(ibus_gen_t *g);                  /* TIM6 interrupt only */
/* Copy of the last closed window; returns 0 if the tick kept rewriting it */
int  ibus_gen_read_window(const ibus_gen_t *g, ibus_gen_window_t *out);

#endif /* IBUS_GEN_H */
