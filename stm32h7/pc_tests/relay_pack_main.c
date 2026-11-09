/* PC harness: print the 14-byte relay packet UartRelay_Pack() builds for
 * vx vy wz given on the command line, as hex. Built and run by
 * tools/tests/test_h7_uart_relay.py together with the unchanged
 * stm32h7/Core/Src/uart_relay.c and the usart.h stub in this folder. */
#include <stdio.h>
#include <stdlib.h>
#include "uart_relay.h"

HAL_StatusTypeDef HAL_UARTEx_EnableFifoMode(UART_HandleTypeDef *huart) { (void)huart; return HAL_OK; }

int main(int argc, char **argv)
{
    if (argc != 4) { fprintf(stderr, "usage: relay_pack vx vy wz\n"); return 2; }
    HIL_TrajRef_t t = { (float)atof(argv[1]), (float)atof(argv[2]), (float)atof(argv[3]),
                        HIL_TRAJ_VALID | HIL_TRAJ_NEW };
    uint8_t pkt[TRAJ_UART_LEN];
    UartRelay_Pack(&t, pkt);
    for (int i = 0; i < TRAJ_UART_LEN; i++) printf("%02x", pkt[i]);
    printf("\n");
    return 0;
}
