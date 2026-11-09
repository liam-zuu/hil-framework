/* ibus_gen_port.h — iBUS generator on UART4 TX (PD1 -> ESP32 GPIO18)
 * UART4 115200 8N1 comes from CubeMX (MX_UART4_Init); this file turns its
 * TX FIFO on and feeds it from the TIM6 tick. Without UART4 in CubeMX the
 * port stays off and every call is a no-op.
 */

#ifndef IBUS_GEN_PORT_H
#define IBUS_GEN_PORT_H

#include "main.h"
#include "ibus_gen.h"

/* Same age as the main-loop SPI1 watchdog (SPI_LINK_TIMEOUT_MS), but
 * checked in the TIM6 tick against the SPI1 callback time, so a blocking
 * debug print in the main loop cannot keep iBUS running (Codex I1). */
#define IBUS_LINK_TIMEOUT_MS  50u

/* Returns 1 when UART4 is set up (UE and TE on) and the generator runs */
int  IbusPort_Init(USART_TypeDef *u);
void IbusPort_Tick(void);                 /* TIM6 10 kHz callback */
/* Main loop gate: 1 from the first SPI1 frame of a link-up period, 0 at
 * link lost. Frames go out only while the gate is 1 and the last SPI1
 * frame is at most IBUS_LINK_TIMEOUT_MS old. */
void IbusPort_SetEnabled(int on);
/* I8: SWA of the frames that start from now on (main loop, from MOSI byte
 * 20 bit 3 of every applied frame; 0 at link lost). 0 at boot. */
void IbusPort_SetSwa(int on);
/* I8-R2-1: 1 once after the TIM6 tick saw the link timed out (last SPI1
 * frame older than IBUS_LINK_TIMEOUT_MS), then 0 until the next timeout.
 * Main loop only: called before applying a frame, a 1 restarts the SWA
 * filter. Stays 0 when UART4 is not set up (the tick does not run). */
int  IbusPort_TakeTimeout(void);
/* SWA of the last iBUS frame started (1 = 2000); 0 before any or when
 * UART4 is not set up. For MISO status bit 0 (HIL_STATUS_SWA). */
int  IbusPort_LastSwa(void);
int  IbusPort_ReadWindow(ibus_gen_window_t *out);

#endif /* IBUS_GEN_PORT_H */
