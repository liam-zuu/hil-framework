/* spi1_handler.c — SPI1 slave DMA handler (RPi5 ↔ H7)
 * Frame: 40 bytes full-duplex (v3 — trajectory reference + IMU values)
 * Re-arm inside HAL_SPI_TxRxCpltCallback — safe on H7 HAL.
 *
 * Frame consistency (29/10, S06 #3 and Codex #23 on the HIL data flow):
 *  - the main loop reads state + traj of ONE frame with SPI1_GetFrame(), copied
 *    with interrupts masked (before, SPI1_GetState() and SPI1_GetTrajRef() were
 *    two unlocked copies, so a frame interrupt in between or in the middle of a
 *    copy could mix two frames);
 *  - the main loop never writes the buffer the DMA is sending: SPI1_SetOutput()
 *    stages the MISO bytes, the callback copies them into _tx while the DMA is
 *    idle (before, _encode() memset and rewrote _tx after the DMA was re-armed,
 *    so the TX FIFO could already hold old or zero bytes). Cost: MISO now
 *    always carries the output staged after the PREVIOUS callback, i.e. the
 *    reply to frame N goes out in frame N+2 (seq says which frame it answers);
 *  - a setpoint flagged VALID|NEW is latched in the callback until the main
 *    loop takes it for the relay (SPI1_TakeNewTraj), so a NEW frame the main
 *    loop did not see (debug printing longer than 1 ms) is not lost silently.
 *    A newer NEW replaces a pending one (the DUT should follow the latest
 *    setpoint); replaced and dropped ones are counted.
 *
 * S06 (29/10, MISO v4, hil_protocol.h; design 2026-10-29_Thiet_ke_H7_S06.md):
 *  - the callback stamps each frame with DWT CYCCNT (read directly, not through
 *    a shared us clock: this runs at priority 0) and its frame count, and keeps
 *    the raw MOSI bytes with the decoded copy;
 *  - SPI1_GetFrameEx() hands the main loop the raw bytes, stamp and count of the
 *    same frame, so the MOSI checksum is computed on the copy actually used;
 *  - SPI1_SetOutputEx() puts that checksum, stamp, count and state age into the
 *    MISO, adds one S06 log record and the MISO checksum of the staged bytes.
 *    A log record leaves the ring only after the frame carrying the previous
 *    one went to the DMA; a frame sent twice repeats its record (same index).
 */

#include "spi1_handler.h"
#include "hil_protocol.h"
#include "s06_log.h"
#include <string.h>

/* ── Private state ───────────────────────────────────────────────────── */
static SPI_HandleTypeDef *_hspi;
static uint8_t _tx[SPI1_FRAME_LEN];       /* buffer handed to the DMA      */
static uint8_t _tx_next[SPI1_FRAME_LEN];  /* staged by SPI1_SetOutput()    */
static volatile uint8_t _tx_next_valid;
static uint8_t _rx[SPI1_FRAME_LEN];
static uint8_t _rx_raw[SPI1_FRAME_LEN];   /* raw MOSI bytes of _state/_traj/_imu */
static uint32_t _rx_cyc;                  /* CYCCNT in the callback of that frame */
static uint32_t _rx_nf;                   /* _n_frames counting that frame        */

static volatile uint8_t _frame_ready = 0;
/* S06 log record carried by the staged MISO frame */
static S06_Rec_t _slot;
static uint8_t   _slot_valid;
static volatile uint32_t _n_unread;       /* frames decoded over an unread one */

/* Setpoint waiting for the UART relay (Codex #23) */
static HIL_TrajRef_t     _new_traj;
static uint8_t           _new_seq;        /* state.seq of the frame that carried it */
static volatile uint8_t  _new_pending;
static volatile uint32_t _n_new;          /* VALID|NEW frames decoded       */
static volatile uint32_t _n_new_replaced; /* pending one replaced by a newer */
static uint32_t          _n_new_taken;
static uint32_t          _n_new_dropped;  /* cleared by SPI1_DropNewTraj()  */

/* Link statistics and error recovery (27/10). Before, a SPI1 error stopped
 * the DMA for good (no error callback) and the H7 ignored the RPi5 until a
 * reset. The error callback only flags a restart; SPI1_Service() does it
 * from the main loop while NSS (PA4) is high, i.e. between two RPi5 frames,
 * so the new DMA transfer starts on a frame boundary. */
#define _NSS_PORT  GPIOA
#define _NSS_PIN   GPIO_PIN_4
static volatile uint32_t _n_frames;
static volatile uint32_t _n_errors;
static volatile uint32_t _last_error;
static volatile uint32_t _last_frame_ms;   /* tick at the last completed frame (clock check) */
static volatile uint8_t  _rearm_pending;
static uint32_t          _n_rearms;

static HIL_State_t   _state;
static HIL_TrajRef_t _traj;
static HIL_Imu_t     _imu;
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
    _imu.gz           = _rd16(&_rx[22]) * (HIL_IMU_GYRO_MAX   / INT16_MAXF);

    /* Trajectory reference (bytes 24–30) — NEW in v2 */
    _traj.vx_ref      = _rd16(&_rx[24]) * (HIL_REF_VEL_MAX   / INT16_MAXF);
    _traj.vy_ref      = _rd16(&_rx[26]) * (HIL_REF_VEL_MAX   / INT16_MAXF);
    _traj.wz_ref      = _rd16(&_rx[28]) * (HIL_REF_OMEGA_MAX  / INT16_MAXF);
    _traj.traj_flags   = _rx[30];

    /* IMU values for the BNO085 emulator (bytes 22–23, 31–39) — v3 */
    _imu.ax           = _rd16(&_rx[31]) * (HIL_IMU_ACCEL_MAX  / INT16_MAXF);
    _imu.ay           = _rd16(&_rx[33]) * (HIL_IMU_ACCEL_MAX  / INT16_MAXF);
    _imu.az           = _rd16(&_rx[35]) * (HIL_IMU_ACCEL_MAX  / INT16_MAXF);
    _imu.yaw          = _rd16(&_rx[37]) * (HIL_IMU_YAW_MAX    / INT16_MAXF);
    _imu.imu_seq      = _rx[39];
    /* S06: raw gz (22-23) and ax..yaw (31-38) for the report-build checksum */
    _imu.raw[0] = _rx[22];
    _imu.raw[1] = _rx[23];
    memcpy(&_imu.raw[2], &_rx[31], 8);
    memcpy(_rx_raw, _rx, SPI1_FRAME_LEN);
}

/* ── Encode MISO (H7 → RPi5), v4 ────────────────────────────────────── */
static void _encode(uint8_t *buf, const SPI1_MisoInfo_t *info)
{
    memset(buf, 0, SPI1_FRAME_LEN);
    for (int i = 0; i < 4; i++) {
        _wr16(&buf[i * 2],
              (int16_t)(_clampf(_output.torque[i], -HIL_TORQUE_MAX, HIL_TORQUE_MAX)
                        * (INT16_MAXF / HIL_TORQUE_MAX)));
    }
    if (info) {
        s06_put16(&buf[HIL_MISO_MOSI_CRC], info->mosi_crc);
        s06_put32(&buf[HIL_MISO_RX_CYC], info->rx_cyc);
        s06_put16(&buf[HIL_MISO_RX_NF], (uint16_t)info->rx_nf);
        s06_put16(&buf[HIL_MISO_AGE], info->age_us);
    }
    buf[HIL_MISO_STATUS] = _output.status;
    buf[HIL_MISO_SEQ]    = _output.seq;
    if (_slot_valid) {
        buf[HIL_MISO_LOG] = _slot.type;
        s06_put16(&buf[HIL_MISO_LOG + 1], _slot.idx);
        memcpy(&buf[HIL_MISO_LOG + 3], _slot.payload, S06_LOG_PAYLOAD);
    }
    uint16_t crc = S06_Crc16(0xFFFFu, buf, HIL_MISO_CRC);
    crc = S06_Crc16(crc, &buf[HIL_MISO_CRC + 2], SPI1_FRAME_LEN - (HIL_MISO_CRC + 2));
    s06_put16(&buf[HIL_MISO_CRC], crc);
}

/* Staged MISO bytes → DMA buffer. Only while the DMA is idle: in the frame
 * callback before the re-arm, or in SPI1_Service() after the abort. */
static void _take_staged_tx(void)
{
    if (_tx_next_valid) {
        memcpy(_tx, _tx_next, SPI1_FRAME_LEN);
        _tx_next_valid = 0;
    }
}

/* VALID|NEW in the frame just decoded → latch it for the relay */
static void _latch_new_traj(void)
{
    const uint8_t mask = HIL_TRAJ_VALID | HIL_TRAJ_NEW;
    if ((_traj.traj_flags & mask) != mask) return;
    if (_new_pending) _n_new_replaced++;
    _new_traj    = _traj;
    _new_seq     = _state.seq;
    _new_pending = 1;
    _n_new++;
}

/* ── Public API ──────────────────────────────────────────────────────── */
void SPI1_Handler_Init(SPI_HandleTypeDef *hspi)
{
    _hspi = hspi;
    memset(_tx, 0, sizeof(_tx));
    memset(_tx_next, 0, sizeof(_tx_next));
    _tx_next_valid = 0;
    memset(_rx, 0, sizeof(_rx));
    _frame_ready = 0;
    _new_pending = 0;
    HAL_SPI_TransmitReceive_DMA(_hspi, _tx, _rx, SPI1_FRAME_LEN);
}

uint8_t SPI1_IsFrameReady(void)
{
    return _frame_ready;
}

/* State and traj of the same (latest) frame; clears frame_ready. With info:
 * also the raw bytes, CYCCNT stamp and frame count of that frame (S06). */
void SPI1_GetFrameEx(HIL_State_t *state, HIL_TrajRef_t *traj, SPI1_FrameInfo_t *info)
{
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    *state = _state;
    *traj  = _traj;
    if (info) {
        memcpy(info->raw, _rx_raw, SPI1_FRAME_LEN);
        info->rx_cyc = _rx_cyc;
        info->rx_nf  = _rx_nf;
    }
    _frame_ready = 0;
    __set_PRIMASK(primask);
}

void SPI1_GetFrame(HIL_State_t *state, HIL_TrajRef_t *traj)
{
    SPI1_GetFrameEx(state, traj, NULL);
}

/* Latched VALID|NEW setpoint, if any: 1 and a copy (plus the seq of the frame
 * that carried it), 0 if none is waiting */
uint8_t SPI1_TakeNewTraj(HIL_TrajRef_t *out, uint8_t *seq)
{
    uint8_t got = 0;
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    if (_new_pending) {
        *out = _new_traj;
        if (seq) *seq = _new_seq;
        _new_pending = 0;
        _n_new_taken++;
        got = 1;
    }
    __set_PRIMASK(primask);
    return got;
}

/* Forget a pending setpoint (link lost: do not relay an old one later) */
void SPI1_DropNewTraj(void)
{
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    if (_new_pending) {
        _new_pending = 0;
        _n_new_dropped++;
    }
    __set_PRIMASK(primask);
}

/* IMU values of the last frame (does not touch frame_ready). Copied with
 * interrupts masked: the BNO085 emulator reads it from the TIM6 interrupt,
 * which SPI1 (priority 0) can preempt in the middle of the copy. */
void SPI1_GetImu(HIL_Imu_t *out)
{
    __disable_irq();
    *out = _imu;
    __enable_irq();
}

/* Stage the reply; the DMA buffer is only written while the DMA is idle */
void SPI1_SetOutputEx(const HIL_Output_t *out, const SPI1_MisoInfo_t *info)
{
    uint8_t buf[SPI1_FRAME_LEN];
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    uint8_t taken = !_tx_next_valid;   /* the frame staged last went to the DMA */
    __set_PRIMASK(primask);
    /* Next log record only once the frame with the current one was taken;
     * if the callback takes it right after this check, the record goes out
     * twice (same index), never lost. */
    if (taken || !_slot_valid)
        _slot_valid = S06Log_Pop(&_slot);
    _output = *out;
    _encode(buf, info);
    primask = __get_PRIMASK();
    __disable_irq();
    memcpy(_tx_next, buf, SPI1_FRAME_LEN);
    _tx_next_valid = 1;
    __set_PRIMASK(primask);
}

void SPI1_SetOutput(const HIL_Output_t *out)
{
    SPI1_SetOutputEx(out, NULL);
}

/* Re-arm inside callback — safe: HAL H7 resets state BEFORE calling callback */
void SPI1_Callback(SPI_HandleTypeDef *hspi)
{
    if (hspi->Instance != _hspi->Instance) return;
    uint32_t cyc = DWT->CYCCNT;        /* S06 L7-4: raw stamp, priority 0 */
    _decode();
    _rx_cyc = cyc;
    _rx_nf  = _n_frames + 1u;          /* this frame, counted below */
    _imu.src_nf = _rx_nf;
    _latch_new_traj();
    _take_staged_tx();
    if (HAL_SPI_TransmitReceive_DMA(_hspi, _tx, _rx, SPI1_FRAME_LEN) != HAL_OK)
        _rearm_pending = 1;
    _last_frame_ms = HAL_GetTick();
    _n_frames++;
    if (_frame_ready) _n_unread++;
    _frame_ready = 1;
}

void SPI1_ErrorCallback(SPI_HandleTypeDef *hspi)
{
    if (hspi->Instance != _hspi->Instance) return;
    _n_errors++;
    _last_error = hspi->ErrorCode;
    _rearm_pending = 1;
}

void SPI1_RequestRestart(void)
{
    _rearm_pending = 1;
}

void SPI1_Service(void)
{
    if (!_rearm_pending) return;
    if (HAL_GPIO_ReadPin(_NSS_PORT, _NSS_PIN) != GPIO_PIN_SET) return;  /* frame in progress */

    HAL_SPI_Abort(_hspi);
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    _take_staged_tx();
    __set_PRIMASK(primask);
    if (HAL_SPI_TransmitReceive_DMA(_hspi, _tx, _rx, SPI1_FRAME_LEN) == HAL_OK) {
        _rearm_pending = 0;
        _n_rearms++;
        uint8_t p[6];                  /* S06: cuts the H7 -> RPi5 clock fit */
        s06_put32(p, DWT->CYCCNT);
        s06_put16(p + 4, (uint16_t)_n_rearms);
        S06Log_Push(S06_REC_REARM, p, sizeof p);
    }
}

void SPI1_GetStats(SPI1_Stats_t *out)
{
    out->errors     = _n_errors;
    out->rearms     = _n_rearms;
    out->last_error = _last_error;
    /* frames and last_frame_ms are read with interrupts masked (a few
     * cycles) so both belong to the same frame */
    __disable_irq();
    out->frames        = _n_frames;
    out->last_frame_ms = _last_frame_ms;
    out->unread        = _n_unread;
    out->new_seen      = _n_new;
    out->new_replaced  = _n_new_replaced;
    out->new_taken     = _n_new_taken;
    out->new_dropped   = _n_new_dropped;
    __enable_irq();
}
