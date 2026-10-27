/* bno_emu_port.c — BNO085 emulator on SPI4 + INT/RST pins (G2)
 * The core (bno_emu.c) decides what a real BNO085 would do; this file only
 * moves bytes and pin levels. Plan Liam approved 27/10: hardware NSS + EXTI4
 * on PE4 (Q1), TIM6 10 kHz tick (Q2), plant IMU values without noise (Q3).
 *
 * MISO must be ready before the host pulls CS low, so the SPI4 DMA is armed
 * ahead with a copy of what the core would send (bno_emu_peek_tx), padded
 * with zeros. Arming happens only while CS is high and INT is high: the host
 * starts a transaction only after INT goes low. At CS rise the number of
 * bytes clocked is read from the RX DMA counter, the transfer is aborted and
 * the DMA armed again for the next transaction.
 */

#include "bno_emu_port.h"
#include "spi1_handler.h"
#include "encoder_gen.h"        /* ENC_CLOCK_CAL */
#include <math.h>
#include <string.h>

#define _INT_PORT  GPIOG
#define _INT_PIN   GPIO_PIN_2
#define _RST_PORT  GPIOG
#define _RST_PIN   GPIO_PIN_3
#define _NSS_PORT  GPIOE
#define _NSS_PIN   GPIO_PIN_4

/* DMA length armed for every transaction: longest packet + 4 zero bytes
 * + margin, so a transaction never reaches the end of the DMA (no EOT). */
#define _ARM_LEN   (BNO_EMU_PKT_MAX + 4u + 16u)

/* ── Private state ───────────────────────────────────────────────────── */
static SPI_HandleTypeDef *_hspi;
static bno_emu_t _emu;
static uint8_t _tx[_ARM_LEN];
static uint8_t _rx[_ARM_LEN];

static uint8_t        _armed;          /* DMA running with _tx            */
static const uint8_t *_armed_src;      /* core buffer _tx was copied from */
static uint16_t       _armed_len;
static uint8_t        _in_xfer;        /* CS low seen, rise not yet       */
static uint8_t        _int_low;        /* level driven on PG2             */
static uint8_t        _have_imu;
static uint8_t        _last_imu_seq;

static uint32_t _n_xfers, _n_arm_mismatch, _n_missed_edge, _n_arm_fail;
static uint32_t _n_spi_errors, _max_clocked, _n_samples;

/* Microsecond time base from the DWT cycle counter, corrected like
 * EncoderGen: the H7 clock runs 0.64 % fast (bug H7-CLK-HSE), so one real
 * millisecond is SystemCoreClock / 1000 * ENC_CLOCK_CAL cycles. Only called
 * at NVIC priority 6 (and in init), so no locking. */
static uint32_t _cyc_last;
static uint64_t _cyc_total;
static uint32_t _cyc_per_ms;

static uint32_t _now_us(void)
{
    uint32_t c = DWT->CYCCNT;
    _cyc_total += (uint32_t)(c - _cyc_last);
    _cyc_last = c;
    return (uint32_t)((_cyc_total * 1000u) / _cyc_per_ms);
}

/* ── Helpers ─────────────────────────────────────────────────────────── */
static void _set_int(uint8_t low)
{
    _int_low = low;
    HAL_GPIO_WritePin(_INT_PORT, _INT_PIN, low ? GPIO_PIN_RESET : GPIO_PIN_SET);
}

static uint8_t _cs_high(void)
{
    return HAL_GPIO_ReadPin(_NSS_PORT, _NSS_PIN) == GPIO_PIN_SET;
}

/* Stop the running transfer (if any) and arm the DMA with what the core
 * would send now. Only with CS high. */
static void _rearm(void)
{
    HAL_SPI_Abort(_hspi);
    _armed = 0;
    _armed_src = NULL;

    uint16_t len;
    const uint8_t *src = bno_emu_peek_tx(&_emu, &len);
    if (len > _ARM_LEN) len = _ARM_LEN;
    memcpy(_tx, src, len);
    memset(&_tx[len], 0, _ARM_LEN - len);
    if (HAL_SPI_TransmitReceive_DMA(_hspi, _tx, _rx, _ARM_LEN) == HAL_OK) {
        _armed = 1;
        _armed_src = src;
        _armed_len = len;
    } else {
        _n_arm_fail++;
    }
}

/* Planar vehicle: roll and pitch rates 0, game rotation vector from yaw
 * only (G2 Q3, Liam 27/10). Noise and bias come at G3. */
static void _sample_from_imu(const HIL_Imu_t *m, bno_emu_sample_t *s)
{
    float h = 0.5f * m->yaw;
    s->ax = m->ax;  s->ay = m->ay;  s->az = m->az;
    s->gx = 0.0f;   s->gy = 0.0f;   s->gz = m->gz;
    s->qi = 0.0f;   s->qj = 0.0f;
    s->qk = sinf(h);
    s->qr = cosf(h);
    s->seq = m->imu_seq;
}

static void _cs_fall(uint32_t now)
{
    uint16_t len;
    const uint8_t *tx = bno_emu_cs_fall(&_emu, now, &len);
    _in_xfer = 1;
    _n_xfers++;
    /* Can differ when a packet was queued after INT went low for an empty
     * read: the host then reads length 0 and the core keeps the packet. */
    if (!_armed || tx != _armed_src || len != _armed_len) _n_arm_mismatch++;
    _set_int(0);                         /* chip releases INT at CS low */
}

static void _cs_rise(uint32_t now)
{
    uint32_t n = 0;
    if (_armed) {
        /* let the RX DMA take the last byte out of the SPI FIFO */
        for (int i = 0; i < 100 && (_hspi->Instance->SR & SPI_SR_RXP); i++) { }
        n = _ARM_LEN - __HAL_DMA_GET_COUNTER(_hspi->hdmarx);
    }
    HAL_SPI_Abort(_hspi);
    _armed = 0;
    if (n > _max_clocked) _max_clocked = n;
    _in_xfer = 0;
    bno_emu_cs_rise(&_emu, _rx, (uint16_t)n, now);
    _rearm();
}

/* ── Interrupt entry points (all NVIC priority 6) ────────────────────── */
void EXTI4_IRQHandler(void)
{
    __HAL_GPIO_EXTI_CLEAR_IT(_NSS_PIN);
    uint32_t now = _now_us();
    uint8_t high = _cs_high();

    if (!_in_xfer) {
        _cs_fall(now);
        if (high) {                      /* CS already back high: both edges */
            _n_missed_edge++;
            _cs_rise(now);
        }
    } else if (high) {
        _cs_rise(now);
    } else {                             /* rise and next fall both missed */
        _n_missed_edge++;
        _cs_rise(now);
        _cs_fall(now);
    }
}

void BnoPort_Tick(void)
{
    uint32_t now = _now_us();

    HIL_Imu_t imu;
    SPI1_GetImu(&imu);
    /* imu_seq changes with every RPi5 frame; the zero frame before the first
     * RPi5 frame is not a sample */
    if (_have_imu ? (imu.imu_seq != _last_imu_seq) : (imu.imu_seq != 0u)) {
        bno_emu_sample_t s;
        _sample_from_imu(&imu, &s);
        bno_emu_set_sample(&_emu, &s, now);
        _have_imu = 1;
        _last_imu_seq = imu.imu_seq;
        _n_samples++;
    }

    bno_emu_poll(&_emu, now);

    if (!_in_xfer && !_int_low && _cs_high()) {
        uint16_t len;
        const uint8_t *src = bno_emu_peek_tx(&_emu, &len);
        if (!_armed || src != _armed_src || len != _armed_len) _rearm();
    }

    uint8_t want_low = bno_emu_int_low(&_emu) && !_in_xfer;
    if (want_low && !_int_low) {
        if (_armed) _set_int(1);         /* announce only what is armed */
    } else if (!want_low && _int_low) {
        _set_int(0);
    }
}

void BnoPort_ExtiCallback(uint16_t pin)
{
    if (pin != _RST_PIN) return;
    uint8_t low = HAL_GPIO_ReadPin(_RST_PORT, _RST_PIN) == GPIO_PIN_RESET;
    bno_emu_set_reset(&_emu, low, _now_us());
    _armed_src = NULL;                   /* queue cleared: tick arms again */
    if (low) _set_int(0);
}

void BnoPort_SpiError(SPI_HandleTypeDef *hspi)
{
    if (_hspi == NULL || hspi->Instance != _hspi->Instance) return;
    _n_spi_errors++;
    _armed = 0;                          /* HAL stopped the DMA */
    _armed_src = NULL;
}

/* ── Public API ──────────────────────────────────────────────────────── */
void BnoPort_Init(SPI_HandleTypeDef *hspi, TIM_HandleTypeDef *htim)
{
    _hspi = hspi;

    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = 0xC5ACCE55u;              /* unlock DWT (Cortex-M7) */
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    _cyc_last = DWT->CYCCNT;
    _cyc_total = 0;
    _cyc_per_ms = (uint32_t)((float)SystemCoreClock / 1000.0f * ENC_CLOCK_CAL + 0.5f);

    bno_emu_cfg_t cfg;
    bno_emu_default_cfg(&cfg);
    uint32_t now = _now_us();
    bno_emu_init(&_emu, &cfg, now);
    if (HAL_GPIO_ReadPin(_RST_PORT, _RST_PIN) == GPIO_PIN_RESET)
        bno_emu_set_reset(&_emu, true, now);

    _set_int(0);
    _rearm();

    /* EXTI4 on PE4, both edges. PE4 stays SPI4_NSS (alternate function);
     * the EXTI input still sees the pin level. */
    __HAL_RCC_SYSCFG_CLK_ENABLE();
    SYSCFG->EXTICR[1] = (SYSCFG->EXTICR[1] & ~SYSCFG_EXTICR2_EXTI4_Msk)
                        | (4u << SYSCFG_EXTICR2_EXTI4_Pos);     /* port E */
    EXTI->RTSR1 |= EXTI_RTSR1_TR4;
    EXTI->FTSR1 |= EXTI_FTSR1_TR4;
    __HAL_GPIO_EXTI_CLEAR_IT(_NSS_PIN);
    EXTI_D1->IMR1 |= EXTI_IMR1_IM4;
    HAL_NVIC_SetPriority(EXTI4_IRQn, 6, 0);
    HAL_NVIC_ClearPendingIRQ(EXTI4_IRQn);
    HAL_NVIC_EnableIRQ(EXTI4_IRQn);

    HAL_TIM_Base_Start_IT(htim);
}

void BnoPort_GetStats(BnoPort_Stats_t *out)
{
    /* copied with interrupts masked (a few hundred cycles) so the numbers
     * belong to the same moment */
    __disable_irq();
    out->core  = _emu.st;
    out->state = (uint8_t)_emu.state;
    for (unsigned i = 0; i < BNO_EMU_N_SENSORS; i++) {
        out->period_us[i] = _emu.period_us[i];
        out->reports[i]   = _emu.last[i].count;
    }
    out->xfers        = _n_xfers;
    out->arm_mismatch = _n_arm_mismatch;
    out->missed_edge  = _n_missed_edge;
    out->arm_fail     = _n_arm_fail;
    out->spi_errors   = _n_spi_errors;
    out->max_clocked  = _max_clocked;
    out->samples      = _n_samples;
    __enable_irq();
}
