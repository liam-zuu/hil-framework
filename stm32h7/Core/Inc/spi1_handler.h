#ifndef SPI1_HANDLER_H
#define SPI1_HANDLER_H

#include "stm32h7xx_hal.h"
#include "hil_protocol.h"

/*
 * SPI1 Handler — RPi5 ↔ H7 link (slave, DMA, 36-byte frames v2)
 *
 * Call order:
 *   1. SPI1_Handler_Init(&hspi1)           — in USER CODE BEGIN 2
 *   2. SPI1_IsFrameReady()                 — poll in while(1)
 *   3. SPI1_GetState(&state)               — read plant state
 *   4. SPI1_GetTrajRef(&traj)              — read traj ref, clears ready flag
 *   5. SPI1_SetOutput(&output)             — update MISO before next frame
 *   6. SPI1_Callback(&hspi1)               — from HAL_SPI_TxRxCpltCallback
 *
 * NOTE: SPI1_GetTrajRef() MUST be called after SPI1_GetState() —
 *       it clears _frame_ready.
 */

void    SPI1_Handler_Init(SPI_HandleTypeDef *hspi);
uint8_t SPI1_IsFrameReady(void);
void    SPI1_GetState    (HIL_State_t *out);
void    SPI1_GetTrajRef  (HIL_TrajRef_t *out);   /* clears frame_ready */
void    SPI1_SetOutput   (const HIL_Output_t *out);
void    SPI1_Callback    (SPI_HandleTypeDef *hspi);

/* Error recovery and link statistics (27/10):
 *   7. SPI1_ErrorCallback(&hspi1)          — from HAL_SPI_ErrorCallback
 *   8. SPI1_Service()                      — poll in while(1); restarts the
 *                                            DMA after an error or a request,
 *                                            only while NSS (PA4) is high
 *   9. SPI1_RequestRestart()               — e.g. from the link watchdog
 */
typedef struct {
    uint32_t frames;       /* completed 36-byte frames since boot          */
    uint32_t errors;       /* HAL_SPI_ErrorCallback calls since boot       */
    uint32_t rearms;       /* DMA restarts done by SPI1_Service()          */
    uint32_t last_error;   /* hspi->ErrorCode at the last error            */
} SPI1_Stats_t;

void    SPI1_ErrorCallback  (SPI_HandleTypeDef *hspi);
void    SPI1_Service        (void);
void    SPI1_RequestRestart (void);
void    SPI1_GetStats       (SPI1_Stats_t *out);

#endif /* SPI1_HANDLER_H */
