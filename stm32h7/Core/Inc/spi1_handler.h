#ifndef SPI1_HANDLER_H
#define SPI1_HANDLER_H

#include "stm32h7xx_hal.h"
#include "hil_protocol.h"

/*
 * SPI1 Handler — RPi5 ↔ H7 link (slave, DMA, 40-byte frames v3)
 *
 * Call order:
 *   1. SPI1_Handler_Init(&hspi1)           — in USER CODE BEGIN 2
 *   2. SPI1_IsFrameReady()                 — poll in while(1)
 *   3. SPI1_GetFrame(&state, &traj)        — state + traj of one frame,
 *                                            clears the ready flag
 *   4. SPI1_TakeNewTraj(&traj, &seq)       — setpoint to relay, if one came
 *   5. SPI1_SetOutput(&output)             — staged, sent from the frame
 *                                            after next (see spi1_handler.c)
 *   6. SPI1_Callback(&hspi1)               — from HAL_SPI_TxRxCpltCallback
 *
 * (29/10: SPI1_GetState() + SPI1_GetTrajRef() replaced by SPI1_GetFrame(),
 *  which copies both from the same frame with interrupts masked.)
 *
 * S06 (29/10, MISO v4): SPI1_GetFrameEx() also returns the raw MOSI bytes,
 * the CYCCNT stamp and the frame count of that frame; SPI1_SetOutputEx()
 * sends the MOSI checksum, stamp, count and state age of the frame it
 * answers, plus one S06 log record (s06_log.h) and the MISO checksum.
 */

typedef struct {
    uint8_t  raw[SPI1_FRAME_LEN];  /* MOSI bytes as received                 */
    uint32_t rx_cyc;               /* DWT CYCCNT in SPI1_Callback of it       */
    uint32_t rx_nf;                /* H7 frame count of it (first frame = 1)  */
} SPI1_FrameInfo_t;

typedef struct {
    uint16_t mosi_crc;             /* S06_Crc16 over SPI1_FrameInfo_t.raw     */
    uint32_t rx_cyc;
    uint32_t rx_nf;
    uint16_t age_us;               /* callback -> applied, saturated          */
} SPI1_MisoInfo_t;

void    SPI1_Handler_Init(SPI_HandleTypeDef *hspi);
uint8_t SPI1_IsFrameReady(void);
void    SPI1_GetFrame    (HIL_State_t *state, HIL_TrajRef_t *traj); /* clears frame_ready */
void    SPI1_GetFrameEx  (HIL_State_t *state, HIL_TrajRef_t *traj, SPI1_FrameInfo_t *info);
uint8_t SPI1_TakeNewTraj (HIL_TrajRef_t *out, uint8_t *seq);       /* 1 = got one */
void    SPI1_DropNewTraj (void);                 /* link lost: forget pending */
void    SPI1_GetImu      (HIL_Imu_t *out);       /* IMU values, v3 frame */
void    SPI1_SetOutput   (const HIL_Output_t *out);
void    SPI1_SetOutputEx (const HIL_Output_t *out, const SPI1_MisoInfo_t *info);
void    SPI1_Callback    (SPI_HandleTypeDef *hspi);

/* Error recovery and link statistics (27/10):
 *   7. SPI1_ErrorCallback(&hspi1)          — from HAL_SPI_ErrorCallback
 *   8. SPI1_Service()                      — poll in while(1); restarts the
 *                                            DMA after an error or a request,
 *                                            only while NSS (PA4) is high
 *   9. SPI1_RequestRestart()               — e.g. from the link watchdog
 */
typedef struct {
    uint32_t frames;       /* completed 40-byte frames since boot          */
    uint32_t errors;       /* HAL_SPI_ErrorCallback calls since boot       */
    uint32_t rearms;       /* DMA restarts done by SPI1_Service()          */
    uint32_t last_error;   /* hspi->ErrorCode at the last error            */
    uint32_t last_frame_ms;/* HAL_GetTick() in the callback of the last frame */
    uint32_t unread;       /* frames decoded before the main loop read the last */
    uint32_t new_seen;     /* VALID|NEW frames decoded                      */
    uint32_t new_replaced; /* pending setpoint replaced by a newer one      */
    uint32_t new_taken;    /* setpoints taken by the main loop for the relay */
    uint32_t new_dropped;  /* pending setpoints forgotten at link lost      */
} SPI1_Stats_t;

void    SPI1_ErrorCallback  (SPI_HandleTypeDef *hspi);
void    SPI1_Service        (void);
void    SPI1_RequestRestart (void);
void    SPI1_GetStats       (SPI1_Stats_t *out);
uint32_t SPI1_GetFrameCount (void);              /* frames received so far (S06 log) */
/* HAL_GetTick() in the callback of the last frame; one 32-bit read, safe
 * from any interrupt (iBUS link check in the TIM6 tick, 03/11) */
uint32_t SPI1_GetLastFrameMs(void);

/* DIAG 08/11 (bug hunt "SPI link lost" in closed loop, CLOSED_LOOP/2026-11-08
 * run02/04/05: no callback for 50 ms, err=0, every restart HAL_OK).
 * Diagnostic build only (branch fix/spi1-link-diag), not for thesis numbers.
 *
 * SPI1_DiagSnapshot(): SPI1 and DMA1 S0/S1 registers, HAL states, NSS level,
 * the DMA RX buffer, frame count and time since the last callback, copied
 * with interrupts masked (some 60 reads, well under 1 us).
 * SPI1_DiagTakeCb(): callback statistics since the previous call (gap between
 * callbacks, callback duration, NSS at entry and right after the re-arm, MOSI
 * seq jumps, failed re-arms), then cleared.
 * SPI1_DiagRestartStats(): restarts done by SPI1_Service() and how many
 * returned != HAL_OK (before, a failed restart was not counted). */
typedef struct {
    uint32_t since_cb_us;          /* CYCCNT now - CYCCNT at the last callback */
    uint32_t frames;               /* completed frames since boot              */
    uint32_t sr, cr1, cr2, cfg1, ier;          /* SPI1 registers               */
    uint32_t rx_cr, rx_ndtr, tx_cr, tx_ndtr;   /* DMA1 Stream0 (RX), Stream1 (TX) */
    uint32_t dma1_lisr;            /* bits 0-5 stream 0, 6-11 stream 1        */
    uint32_t hspi_err, rx_err, tx_err;         /* HAL ErrorCode                */
    uint8_t  hspi_state, rx_state, tx_state;   /* HAL State                    */
    uint8_t  nss;                  /* PA4 level, 1 = high (no frame)           */
    uint8_t  rearm_pending;
    uint8_t  rx[SPI1_FRAME_LEN];   /* DMA RX buffer as it is now               */
} SPI1_Diag_t;

typedef struct {
    uint32_t n;                    /* callbacks in the window                  */
    uint32_t gap_hist[5];          /* gap <0.9, 0.9-1.1, 1.1-2, 2-10, >=10 ms  */
    uint32_t gap_max_us, gap_min_us;
    uint32_t dur_min_us, dur_max_us;           /* callback entry -> exit       */
    uint32_t nss_low_entry;        /* NSS low at callback entry                */
    uint32_t nss_low_rearm;        /* NSS low right after the re-arm           */
    uint32_t seq_jump;             /* MOSI seq != previous + 1 (first frame after a restart excluded) */
    uint32_t rearm_fail;           /* re-arm in the callback != HAL_OK         */
} SPI1_DiagCb_t;

void SPI1_DiagSnapshot    (SPI1_Diag_t *d);
void SPI1_DiagTakeCb      (SPI1_DiagCb_t *out);
void SPI1_DiagRestartStats(uint32_t *done, uint32_t *failed);

#endif /* SPI1_HANDLER_H */
