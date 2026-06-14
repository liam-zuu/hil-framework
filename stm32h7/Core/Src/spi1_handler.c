#include "spi1_handler.h"
#include <string.h>

/* ── DMA buffers — plain .bss, D-Cache OFF so no MPU/flush needed ────────── */
static uint8_t _rx[SPI1_FRAME_LEN];
static uint8_t _tx[SPI1_FRAME_LEN];

/* ── Internal state ─────────────────────────────────────────────────────── */
static SPI_HandleTypeDef *_hspi;
static volatile uint8_t   _frame_ready;
static HIL_State_t        _state;
static HIL_Output_t       _output;

/* ── int16 big-endian helpers ───────────────────────────────────────────── */
static inline int16_t _rd16(const uint8_t *b)
{
    return (int16_t)(((uint16_t)b[0] << 8) | b[1]);
}

static inline void _wr16(uint8_t *b, int16_t v)
{
    b[0] = (uint8_t)((uint16_t)v >> 8);
    b[1] = (uint8_t)v;
}

/* ── Clamp helpers ──────────────────────────────────────────────────────── */
static inline float _clampf(float v, float lo, float hi)
{
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

/* ── Decode MOSI → _state ───────────────────────────────────────────────── */
static void _decode(void)
{
    const float sc_omega = HIL_OMEGA_MAX / INT16_MAXF;
    const float sc_pos   = HIL_POS_MAX   / INT16_MAXF;
    const float sc_theta = HIL_THETA_MAX / INT16_MAXF;
    const float sc_vel   = HIL_VEL_MAX   / INT16_MAXF;

    _state.omega[0]   = _rd16(&_rx[0])  * sc_omega;
    _state.omega[1]   = _rd16(&_rx[2])  * sc_omega;
    _state.omega[2]   = _rd16(&_rx[4])  * sc_omega;
    _state.omega[3]   = _rd16(&_rx[6])  * sc_omega;
    _state.pos_x      = _rd16(&_rx[8])  * sc_pos;
    _state.pos_y      = _rd16(&_rx[10]) * sc_pos;
    _state.theta      = _rd16(&_rx[12]) * sc_theta;
    _state.vx         = _rd16(&_rx[14]) * sc_vel;
    _state.vy         = _rd16(&_rx[16]) * sc_vel;
    _state.omega_body = _rd16(&_rx[18]) * sc_vel;
    _state.fault_flags = _rx[20];
    _state.seq         = _rx[21];
}

/* ── Encode _output → MISO ──────────────────────────────────────────────── */
static void _encode(void)
{
    const float sc_torque = INT16_MAXF / HIL_TORQUE_MAX;
    const float sc_pwm    = INT16_MAXF;

    memset(_tx, 0, sizeof(_tx));
    for (int i = 0; i < 4; i++) {
        float t = _clampf(_output.torque[i], -HIL_TORQUE_MAX, HIL_TORQUE_MAX);
        float p = _clampf(_output.pwm[i],    -1.0f,           1.0f);
        _wr16(&_tx[i * 2],      (int16_t)(t * sc_torque));
        _wr16(&_tx[8 + i * 2], (int16_t)(p * sc_pwm));
    }
    _tx[16] = _output.status;
    _tx[17] = _output.seq;
}

/* ── Public API ─────────────────────────────────────────────────────────── */

void SPI1_Handler_Init(SPI_HandleTypeDef *hspi)
{
    _hspi = hspi;
    memset(_tx, 0, sizeof(_tx));
    memset(_rx, 0, sizeof(_rx));
    _frame_ready = 0;
    HAL_SPI_TransmitReceive_DMA(_hspi, _tx, _rx, SPI1_FRAME_LEN);
}

uint8_t SPI1_IsFrameReady(void)
{
    return _frame_ready;
}

void SPI1_GetState(HIL_State_t *out)
{
    *out = _state;
    _frame_ready = 0;
}

void SPI1_SetOutput(const HIL_Output_t *out)
{
    _output = *out;
    _encode();
    /* _tx is live for the NEXT DMA transfer (already re-armed in callback) */
}

/*
 * Call this from HAL_SPI_TxRxCpltCallback in main.c:
 *   void HAL_SPI_TxRxCpltCallback(SPI_HandleTypeDef *hspi) {
 *       SPI1_Callback(hspi);
 *   }
 *
 * Re-arming inside the callback is safe on H7: HAL resets internal state
 * before invoking the callback, so no re-entrancy issue.
 */
void SPI1_Callback(SPI_HandleTypeDef *hspi)
{
    if (hspi->Instance != _hspi->Instance) return;
    _decode();
    HAL_SPI_TransmitReceive_DMA(_hspi, _tx, _rx, SPI1_FRAME_LEN);
    _frame_ready = 1;
}
