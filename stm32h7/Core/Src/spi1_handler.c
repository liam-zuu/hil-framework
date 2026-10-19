/* spi1_handler.c — SPI1 slave DMA handler (RPi5 ↔ H7)
 * Frame: 36 bytes full-duplex (v2 — includes trajectory reference)
 * Re-arm inside HAL_SPI_TxRxCpltCallback — safe on H7 HAL.
 */

#include "spi1_handler.h"
#include "hil_protocol.h"
#include <string.h>

/* ── Private state ───────────────────────────────────────────────────── */
static SPI_HandleTypeDef *_hspi;
static uint8_t _tx[SPI1_FRAME_LEN];
static uint8_t _rx[SPI1_FRAME_LEN];

static volatile uint8_t _frame_ready = 0;
static HIL_State_t   _state;
static HIL_TrajRef_t _traj;
static HIL_Output_t  _output;

/* ── Helpers ─────────────────────────────────────────────────────────── */
static inline int16_t _rd16(const uint8_t *p)
{
    return (int16_t)((uint16_t)p[0] << 8 | p[1]);  /* big-endian */
}

static inline void _wr16(uint8_t *p, int16_t v)
{
    p[0] = (uint8_t)(v >> 8);
    p[1] = (uint8_t)(v);
}

static inline float _clampf(float v, float lo, float hi)
{
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

/* ── Decode MOSI (RPi5 → H7) ────────────────────────────────────────── */
static void _decode(void)
{
    /* Plant state (bytes 0–21) */
    _state.omega[0]   = _rd16(&_rx[0])  * (HIL_OMEGA_MAX      / INT16_MAXF);
    _state.omega[1]   = _rd16(&_rx[2])  * (HIL_OMEGA_MAX      / INT16_MAXF);
    _state.omega[2]   = _rd16(&_rx[4])  * (HIL_OMEGA_MAX      / INT16_MAXF);
    _state.omega[3]   = _rd16(&_rx[6])  * (HIL_OMEGA_MAX      / INT16_MAXF);
    _state.pos_x      = _rd16(&_rx[8])  * (HIL_POS_MAX        / INT16_MAXF);
    _state.pos_y      = _rd16(&_rx[10]) * (HIL_POS_MAX        / INT16_MAXF);
    _state.theta      = _rd16(&_rx[12]) * (HIL_THETA_MAX      / INT16_MAXF);
    _state.vx         = _rd16(&_rx[14]) * (HIL_VEL_MAX        / INT16_MAXF);
    _state.vy         = _rd16(&_rx[16]) * (HIL_VEL_MAX        / INT16_MAXF);
    _state.omega_body = _rd16(&_rx[18]) * (HIL_OMEGA_BODY_MAX / INT16_MAXF);
    _state.fault_flags = _rx[20];
    _state.seq         = _rx[21];
    /* bytes 22–23: reserved */

    /* Trajectory reference (bytes 24–30) — NEW in v2 */
    _traj.vx_ref      = _rd16(&_rx[24]) * (HIL_REF_VEL_MAX   / INT16_MAXF);
    _traj.vy_ref      = _rd16(&_rx[26]) * (HIL_REF_VEL_MAX   / INT16_MAXF);
    _traj.wz_ref      = _rd16(&_rx[28]) * (HIL_REF_OMEGA_MAX  / INT16_MAXF);
    _traj.traj_flags   = _rx[30];
    /* bytes 31–35: reserved */
}

/* ── Encode MISO (H7 → RPi5) ────────────────────────────────────────── */
static void _encode(void)
{
    memset(_tx, 0, sizeof(_tx));
    for (int i = 0; i < 4; i++) {
        _wr16(&_tx[i * 2],
              (int16_t)(_clampf(_output.torque[i], -HIL_TORQUE_MAX, HIL_TORQUE_MAX)
                        * (INT16_MAXF / HIL_TORQUE_MAX)));
        _wr16(&_tx[8 + i * 2],
              (int16_t)(_clampf(_output.pwm[i], -1.0f, 1.0f) * INT16_MAXF));
    }
    _tx[16] = _output.status;
    _tx[17] = _output.seq;
    /* bytes 18–35: remain 0x00 */
}

/* ── Public API ──────────────────────────────────────────────────────── */
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
}

void SPI1_GetTrajRef(HIL_TrajRef_t *out)
{
    *out = _traj;
    _frame_ready = 0;   /* clear after both state + traj consumed */
}

void SPI1_SetOutput(const HIL_Output_t *out)
{
    _output = *out;
    _encode();
}

/* Re-arm inside callback — safe: HAL H7 resets state BEFORE calling callback */
void SPI1_Callback(SPI_HandleTypeDef *hspi)
{
    if (hspi->Instance != _hspi->Instance) return;
    _decode();
    HAL_SPI_TransmitReceive_DMA(_hspi, _tx, _rx, SPI1_FRAME_LEN);
    _frame_ready = 1;
}
