/* uart_relay.h — Trajectory relay: SPI frame → USART2 → ESP32
 * Pins: PD5 = USART2_TX, PD6 = USART2_RX (RX unused, kept for symmetry)
 * Baud: 115200, 8N1
 * Packet: 14 bytes — identical to WiFi UDP format on ESP32 side
 *   0xBB | vx_ref(4B LE float) | vy_ref(4B LE float)
 *         | wz_ref(4B LE float) | XOR checksum(1B)
 */

#ifndef UART_RELAY_H
#define UART_RELAY_H

#include "hil_protocol.h"
#include "usart.h"          /* CubeMX-generated, provides huart2 */

void UartRelay_Init(UART_HandleTypeDef *huart);
void UartRelay_Send(const HIL_TrajRef_t *traj);

#endif /* UART_RELAY_H */
