/* uart_relay.c — Trajectory relay: SPI frame → USART2 → ESP32
 *
 * Called from main loop whenever SPI1_IsFrameReady(). Sends only when
 * traj_flags has both HIL_TRAJ_VALID and HIL_TRAJ_NEW set, i.e. once per
 * new setpoint from the RPi5 (~50 Hz), not on every 1 kHz SPI frame.
 *
 * Packet format (14 bytes, little-endian floats):
 *   [0]      0xBB  header
 *   [1..4]   vx_ref  float32 LE
 *   [5..8]   vy_ref  float32 LE
 *   [9..12]  wz_ref  float32 LE
 *   [13]     XOR checksum of bytes [0..12] (header included), as the ESP32
 *            parser (AGV_MECANUM traj_receiver.c) and rpi5/pc_sender.py compute it.
 *            Until 28/10/2026 this file XOR-ed [1..12] only: every relayed
 *            packet differed by 0xBB and the ESP32 dropped it (found by reading
 *            the code; the relay had not been checked on the board yet).
 *
 * Non-blocking: the 14 bytes are written straight into the 16-byte USART
 * TX FIFO and the hardware shifts them out (~1.2 ms @ 115200) while the
 * main loop keeps running. The old HAL_UART_Transmit() version blocked the
 * main loop for those 1.2 ms, longer than the 1 ms HIL period.
 */

#include "uart_relay.h"
#include <string.h>
#include <stdint.h>

/* ── Private ─────────────────────────────────────────────────────────── */
static UART_HandleTypeDef *_huart;
static volatile uint32_t   _n_sent;
static volatile uint32_t   _n_dropped;

#define _TRAJ_SEND_MASK  (HIL_TRAJ_VALID | HIL_TRAJ_NEW)

/* Pack float as 4-byte little-endian into buf */
static void _pack_float_le(uint8_t *buf, float f)
{
    /* memcpy avoids strict-aliasing UB */
    memcpy(buf, &f, 4);
    /* STM32 is little-endian natively — no byte swap needed */
}

/* ── Public API ──────────────────────────────────────────────────────── */
void UartRelay_Pack(const HIL_TrajRef_t *traj, uint8_t pkt[TRAJ_UART_LEN])
{
    pkt[0] = TRAJ_UART_HEADER;             /* 0xBB */
    _pack_float_le(&pkt[1], traj->vx_ref);
    _pack_float_le(&pkt[5], traj->vy_ref);
    _pack_float_le(&pkt[9], traj->wz_ref);

    /* XOR checksum over bytes [0..12], header included (ESP32 traj_receiver.c) */
    uint8_t cs = 0;
    for (int i = 0; i < TRAJ_UART_LEN - 1; i++) {
        cs ^= pkt[i];
    }
    pkt[TRAJ_UART_LEN - 1] = cs;
}

void UartRelay_Init(UART_HandleTypeDef *huart)
{
    _huart     = huart;
    _n_sent    = 0;
    _n_dropped = 0;

    /* usart.c (CubeMX) calls HAL_UARTEx_DisableFifoMode(). Re-enable the
     * FIFO here, in a user file, so regenerating CubeMX code keeps it. */
    if (HAL_UARTEx_EnableFifoMode(_huart) != HAL_OK) {
        _huart = NULL;   /* relay disabled; UartRelay_Send() becomes a no-op */
    }
}

void UartRelay_Send(const HIL_TrajRef_t *traj)
{
    if (_huart == NULL) return;

    /* Only relay a valid setpoint that is new in this frame */
    if ((traj->traj_flags & _TRAJ_SEND_MASK) != _TRAJ_SEND_MASK) return;

    USART_TypeDef *u = _huart->Instance;

    /* Previous packet still in the FIFO → drop this one rather than wait.
     * At 50 Hz (20 ms period) the 1.2 ms packet is long gone, so this only
     * triggers if the RPi5 flags NEW far too often. */
    if ((u->ISR & USART_ISR_TXFE) == 0u) {
        _n_dropped++;
        return;
    }

    uint8_t pkt[TRAJ_UART_LEN];
    UartRelay_Pack(traj, pkt);

    /* FIFO is empty and 16 deep, so all 14 writes are accepted immediately */
    for (int i = 0; i < TRAJ_UART_LEN; i++) {
        u->TDR = pkt[i];
    }
    _n_sent++;
}

void UartRelay_GetStats(uint32_t *sent, uint32_t *dropped)
{
    if (sent)    *sent    = _n_sent;
    if (dropped) *dropped = _n_dropped;
}
