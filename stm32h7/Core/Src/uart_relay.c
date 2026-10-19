/* uart_relay.c — Trajectory relay: SPI frame → USART2 → ESP32
 *
 * Called from main loop whenever SPI1_IsFrameReady() and
 * traj_flags & HIL_TRAJ_VALID.
 *
 * Packet format (14 bytes, little-endian floats):
 *   [0]      0xBB  header
 *   [1..4]   vx_ref  float32 LE
 *   [5..8]   vy_ref  float32 LE
 *   [9..12]  wz_ref  float32 LE
 *   [13]     XOR checksum of bytes [1..12]
 *
 * Uses blocking HAL_UART_Transmit — 14 bytes @ 115200 = ~1.2ms.
 * At 50Hz relay rate (20ms period) this is acceptable (6% CPU).
 * Switch to DMA if 1kHz relay is ever needed.
 */

#include "uart_relay.h"
#include <string.h>
#include <stdint.h>

/* ── Private ─────────────────────────────────────────────────────────── */
static UART_HandleTypeDef *_huart;

/* Pack float as 4-byte little-endian into buf */
static void _pack_float_le(uint8_t *buf, float f)
{
    /* memcpy avoids strict-aliasing UB */
    memcpy(buf, &f, 4);
    /* STM32 is little-endian natively — no byte swap needed */
}

/* ── Public API ──────────────────────────────────────────────────────── */
void UartRelay_Init(UART_HandleTypeDef *huart)
{
    _huart = huart;
}

void UartRelay_Send(const HIL_TrajRef_t *traj)
{
    if (_huart == NULL) return;

    /* Only relay if setpoint is valid */
    if (!(traj->traj_flags & HIL_TRAJ_VALID)) return;

    uint8_t pkt[TRAJ_UART_LEN];

    pkt[0] = TRAJ_UART_HEADER;             /* 0xBB */
    _pack_float_le(&pkt[1], traj->vx_ref);
    _pack_float_le(&pkt[5], traj->vy_ref);
    _pack_float_le(&pkt[9], traj->wz_ref);

    /* XOR checksum over payload bytes [1..12] */
    uint8_t cs = 0;
    for (int i = 1; i < TRAJ_UART_LEN - 1; i++) {
        cs ^= pkt[i];
    }
    pkt[13] = cs;

    /* Blocking transmit — timeout 5ms (generous for 14 bytes @ 115200) */
    HAL_UART_Transmit(_huart, pkt, TRAJ_UART_LEN, 5);
}
