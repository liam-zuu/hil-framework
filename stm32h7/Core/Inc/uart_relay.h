/* uart_relay.h — Trajectory relay: SPI frame → USART2 → ESP32
 * Pins: PD5 = USART2_TX, PD6 = USART2_RX (RX unused, kept for symmetry)
 * Baud: 115200, 8N1, TX FIFO enabled (non-blocking send)
 * Packet: 14 bytes — identical to WiFi UDP format on ESP32 side
 *   0xBB | vx_ref(4B LE float) | vy_ref(4B LE float)
 *         | wz_ref(4B LE float) | XOR checksum(1B)
 * Sent only when traj_flags has HIL_TRAJ_VALID and HIL_TRAJ_NEW.
 */

#ifndef UART_RELAY_H
#define UART_RELAY_H

#include <stdint.h>
#include "hil_protocol.h"
#include "usart.h"          /* CubeMX-generated, provides huart2 */

void UartRelay_Init(UART_HandleTypeDef *huart);
void UartRelay_Send(const HIL_TrajRef_t *traj);
void UartRelay_GetStats(uint32_t *sent, uint32_t *dropped);

#endif /* UART_RELAY_H */
