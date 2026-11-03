/* ibus_gen_port.h — iBUS generator on UART4 TX (PD1 -> ESP32 GPIO18)
 * UART4 115200 8N1 comes from CubeMX (MX_UART4_Init); this file turns its
 * TX FIFO on and feeds it from the TIM6 tick. Without UART4 in CubeMX the
 * port stays off and every call is a no-op.
 */

#ifndef IBUS_GEN_PORT_H
#define IBUS_GEN_PORT_H

#include "main.h"
#include "ibus_gen.h"

/* Returns 1 when UART4 is set up (UE and TE on) and the generator runs */
int  IbusPort_Init(USART_TypeDef *u);
void IbusPort_Tick(void);                 /* TIM6 10 kHz callback */
void IbusPort_SetEnabled(int on);         /* main loop: SPI1 link up/down */
int  IbusPort_ReadWindow(ibus_gen_window_t *out);

#endif /* IBUS_GEN_PORT_H */
