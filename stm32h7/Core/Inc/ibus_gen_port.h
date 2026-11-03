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
int  IbusPort_ReadWindow(ibus_gen_window_t *out);

#endif /* IBUS_GEN_PORT_H */
