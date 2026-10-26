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
void    SPI1_GetImu      (HIL_Imu_t *out);       /* IMU values, v3 frame */
void    SPI1_SetOutput   (const HIL_Output_t *out);
void    SPI1_Callback    (SPI_HandleTypeDef *hspi);

#endif /* SPI1_HANDLER_H */
