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

GPIO_TypeDef fake_gpioa;
int  fake_irq_disabled;
long fake_irq_sections;

static SPI_TypeDef       _spi_inst;
static SPI_HandleTypeDef _hspi = { &_spi_inst, 0 };
static uint8_t *_armed_tx, *_armed_rx;
static uint8_t  _armed_copy[SPI1_FRAME_LEN];   /* tx bytes at arm time */
static uint8_t  _sent[SPI1_FRAME_LEN];         /* tx bytes of the last frame */
static int      _tx_changed_after_arm;
static int      _fails;

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
    if (memcmp(_armed_tx, _armed_copy, SPI1_FRAME_LEN) != 0) _tx_changed_after_arm++;
    memcpy(_sent, _armed_copy, SPI1_FRAME_LEN);
    memcpy(_armed_rx, m, SPI1_FRAME_LEN);
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

int main(void)
{
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
#endif
    return _fails ? 1 : 0;
}
