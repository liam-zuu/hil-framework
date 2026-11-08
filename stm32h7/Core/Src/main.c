/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "dma.h"
#include "spi.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "hil_protocol.h"
#include "spi1_handler.h"
#include "encoder_gen.h"
#include "pwm_capture.h"
#include "uart_relay.h"         /* +++ NEW */
#include "bno_emu_port.h"
#include "s06_log.h"
#include "s06_load.h"
#include "ibus_gen_port.h"
#include <stdio.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* SPI1 link watchdog (27/10): no frame from the RPi5 for this long -> all
 * wheel omegas set to 0 so EncoderGen stops instead of holding the last
 * speed. The ESP32 drops its setpoint to 0 on its own after 200 ms without
 * a relay packet (AGV_MECANUM traj_receiver.c, TRAJ_TIMEOUT_MS). */
#define SPI_LINK_TIMEOUT_MS  50u
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
static HIL_State_t   g_state;
static HIL_TrajRef_t g_traj;   /* +++ NEW */
static HIL_Output_t  g_output;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/* Debug print (S06 V4, W16): blocking during init, so a boot hang still shows
 * the last line; from the main loop on, through a 4 KB ring sent by the
 * USART3 TXE interrupt, so the main loop never waits for the UART (before:
 * HAL_UART_Transmit with 100 ms timeout, about 11 lines every 0.5 s at
 * 115200 baud). Ring full: characters dropped and counted (S06log line). */
#define _DBG_TX_LEN 4096u                       /* power of two */
static uint8_t           _dbg_tx[_DBG_TX_LEN];
static volatile uint32_t _dbg_h, _dbg_t;
static volatile uint32_t _dbg_drop;
static uint8_t           _dbg_async;

int _write(int file, char *ptr, int len)
{
    (void)file;
    if (!_dbg_async) {
        HAL_UART_Transmit(&huart3, (uint8_t *)ptr, (uint16_t)len, 100);
        return len;
    }
    for (int i = 0; i < len; i++) {             /* main loop only: one producer */
        uint32_t n = (_dbg_h + 1u) & (_DBG_TX_LEN - 1u);
        if (n == _dbg_t) { _dbg_drop += (uint32_t)(len - i); break; }
        _dbg_tx[_dbg_h] = (uint8_t)ptr[i];
        _dbg_h = n;
    }
    SET_BIT(USART3->CR1, USART_CR1_TXEIE_TXFNFIE);
    return len;
}

/* Not in stm32h7xx_it.c: the USART3 interrupt is off in the .ioc. If it is
 * ever enabled there, CubeMX generates a second handler: remove this one. */
void USART3_IRQHandler(void)
{
    S06_IRQ_ENTER();
    if (READ_BIT(USART3->ISR, USART_ISR_TXE_TXFNF)) {
        if (_dbg_t != _dbg_h) {
            USART3->TDR = _dbg_tx[_dbg_t];
            _dbg_t = (_dbg_t + 1u) & (_DBG_TX_LEN - 1u);
        } else {
            CLEAR_BIT(USART3->CR1, USART_CR1_TXEIE_TXFNFIE);
            if (_dbg_t != _dbg_h)               /* main loop wrote meanwhile */
                SET_BIT(USART3->CR1, USART_CR1_TXEIE_TXFNFIE);
        }
    }
    S06_IRQ_EXIT(S06_IRQ_USART3);
}

void HAL_SPI_TxRxCpltCallback(SPI_HandleTypeDef *hspi)
{
    SPI1_Callback(hspi);
}

void HAL_SPI_ErrorCallback(SPI_HandleTypeDef *hspi)
{
    SPI1_ErrorCallback(hspi);
    BnoPort_SpiError(hspi);
}

/* BNO085 emulator (G2) and iBUS generator: TIM6 10 kHz tick; RST pin PG3 (EXTI3) */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM6) {
        IbusPort_Tick();
        BnoPort_Tick();
    }
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    BnoPort_ExtiCallback(GPIO_Pin);
}

void HAL_TIM_OC_DelayElapsedCallback(TIM_HandleTypeDef *htim)
{
    EncoderGen_OC_Callback(htim);
}

/* DIAG 08/11 (fix/spi1-link-diag): SPI link lost in closed loop. While the
 * link is down after having been up: SPI1 restart every SPI_DIAG_RESTART_MS
 * (before: one restart request at the up -> down transition only, so a
 * restart that returned HAL_OK but never completed left the link dead).
 * At the loss: snapshot A, then NSS (PA4) polled for SPI_DIAG_POLL_US with
 * falling edges and low time counted (Pi still clocking or not), snapshot B,
 * then the restart. Lines: "SPIdiag A/B", "SPIdiag rx", "SPIdiag nss",
 * "SPIdiag down" (every 0.5 s while down), "SPIcb" (every 0.5 s). */
#define SPI_DIAG_RESTART_MS  200u
#define SPI_DIAG_POLL_US     3000u

static void _print_spi_diag(const char *tag, const SPI1_Diag_t *d)
{
    printf("SPIdiag %s: dt_cb=%luus fr=%lu sr=%08lx cr1=%08lx cr2=%08lx cfg1=%08lx ier=%08lx"
           " rxs=%08lx/%lu txs=%08lx/%lu lisr=%03lx hs=%u/%lx dr=%u/%lx dt=%u/%lx nss=%u pend=%u\r\n",
           tag, (unsigned long)d->since_cb_us, (unsigned long)d->frames,
           (unsigned long)d->sr, (unsigned long)d->cr1, (unsigned long)d->cr2,
           (unsigned long)d->cfg1, (unsigned long)d->ier,
           (unsigned long)d->rx_cr, (unsigned long)d->rx_ndtr,
           (unsigned long)d->tx_cr, (unsigned long)d->tx_ndtr,
           (unsigned long)d->dma1_lisr,
           (unsigned)d->hspi_state, (unsigned long)d->hspi_err,
           (unsigned)d->rx_state, (unsigned long)d->rx_err,
           (unsigned)d->tx_state, (unsigned long)d->tx_err,
           (unsigned)d->nss, (unsigned)d->rearm_pending);
}

static void _print_spi_rx(const SPI1_Diag_t *d)
{
    char hex[2 * SPI1_FRAME_LEN + 1];
    static const char hx[] = "0123456789abcdef";
    for (int i = 0; i < SPI1_FRAME_LEN; i++) {
        hex[2 * i]     = hx[d->rx[i] >> 4];
        hex[2 * i + 1] = hx[d->rx[i] & 0xFu];
    }
    hex[2 * SPI1_FRAME_LEN] = '\0';
    printf("SPIdiag rx: %s\r\n", hex);
}

#ifdef ENCODER_GEN_HAS_EVENT_HOOK
/* S06 T3b (Y5, V2-6): one log record per _start_wheel / _stop call: wheel u8,
 * timer CNT at the call u16, calls of that kind on that wheel u32, CYCCNT u32,
 * H7 frame count u16 (little-endian). The Pi compares the calls with the
 * scheduled starts and stops and reports the start phases. Needs the
 * EncoderGen with the event hook (feat/encgen-events); without it no record. */
static void _s06_enc_event(uint8_t wheel, uint8_t ev, uint16_t cnt, uint32_t n)
{
    uint8_t p[13];
    p[0] = wheel;
    s06_put16(p + 1, cnt);
    s06_put32(p + 3, n);
    s06_put32(p + 7, DWT->CYCCNT);
    s06_put16(p + 11, (uint16_t)SPI1_GetFrameCount());
    S06Log_Push(ev == ENC_EV_START ? S06_REC_ENC_START : S06_REC_ENC_STOP, p, sizeof p);
}
#endif
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  /* Instruction cache on (05/11, Liam: "Duyet bat I-cache"): code runs from
   * flash at 480 MHz; without the I-cache the PWMCapture read window was
   * 19.2 us at best (PWM_CAPTURE/2026-11-05_run02), M23 needs below about
   * 7 us. D-cache stays OFF: the DMA rings and SPI buffers in AXI SRAM rely
   * on it (pwm_capture.c, spi1_handler.c). To do in CubeMX later: Cortex-M7
   * > CPU ICache = Enabled, then drop this line. */
  SCB_EnableICache();
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_DMA_Init();
  MX_GPIO_Init();
  MX_USART3_UART_Init();
  MX_TIM1_Init();
  MX_TIM2_Init();
  MX_TIM3_Init();
  MX_TIM4_Init();
  MX_SPI1_Init();
  MX_TIM5_Init();
  MX_TIM15_Init();
  MX_TIM16_Init();
  MX_TIM17_Init();
  MX_USART2_UART_Init();
  MX_SPI4_Init();
  MX_TIM6_Init();
  MX_UART4_Init();
  MX_TIM8_Init();
  /* USER CODE BEGIN 2 */

  /* Capture timers: no interrupt since 04/11 (DMA capture, pwm_capture.h).
   * CubeMX MspInit still enables these NVIC lines; keep them off. */
  HAL_NVIC_DisableIRQ(TIM5_IRQn);
  HAL_NVIC_DisableIRQ(TIM15_IRQn);
  HAL_NVIC_DisableIRQ(TIM16_IRQn);
  HAL_NVIC_DisableIRQ(TIM17_IRQn);

  setvbuf(stdout, NULL, _IONBF, 0);

  printf("\r\n--- H753 HIL boot ---\r\n");
  printf("DIAG build fix/spi1-link-diag (08/11): SPIdiag/SPIcb lines, SPI1 restart every %u ms while down; not for thesis numbers\r\n",
         (unsigned)SPI_DIAG_RESTART_MS);
  printf("pclk1=%lu  pclk2=%lu\r\n",
         HAL_RCC_GetPCLK1Freq(), HAL_RCC_GetPCLK2Freq());

  /* S06: cycle counter and log ring before the first SPI1 frame can be
   * stamped (BnoPort_Init no longer zeroes CYCCNT) */
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->LAR = 0xC5ACCE55u;              /* unlock DWT (Cortex-M7), as in BnoPort_Init */
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  S06Log_Init();

  EncoderGen_Init(&htim1, &htim2, &htim3, &htim4);
#ifdef ENCODER_GEN_HAS_EVENT_HOOK
  EncoderGen_SetEventHook(_s06_enc_event);
#endif
  SPI1_Handler_Init(&hspi1);
  UartRelay_Init(&huart2);     /* +++ NEW */

  PWMCapture_Init(&htim5, &htim8, &htim15, &htim16, &htim17);
  printf("BNO: emulator on SPI4 (INT PG2, RST PG3), TIM6 tick\r\n");
  BnoPort_Init(&hspi4, &htim6);
  if (IbusPort_Init(UART4))
    printf("IBUS: UART4 TX PD1, frame every %u us while the SPI1 link is up (stop after %u ms without a frame)\r\n",
           (unsigned)IBUS_GEN_PERIOD_US, (unsigned)IBUS_LINK_TIMEOUT_MS);
  else
    printf("IBUS: UART4 not set up (add it in CubeMX), iBUS off\r\n");
  printf("Init done, entering main loop\r\n");
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  static uint32_t _dbg_last = 0;
  static uint32_t _last_frame_ms = 0;
  static uint8_t  _link_up = 0;        /* 0 until the first frame */
  /* Clock check (27/10): first-frame tick and frame count of the current
   * link-up period. The RPi5 sends one frame per 1 ms of its own clock, so
   * N frames span N-1 Pi ms; compare with the H7 ms printed at link lost. */
  static uint32_t _up_first_ms = 0;
  static uint32_t _up_first_frames = 0;
  static uint32_t _n_link_lost = 0;
  static uint32_t _diag_restart_ms = 0;      /* DIAG 08/11 */
  static uint32_t _diag_down_restarts = 0;   /* DIAG 08/11: restarts requested while down */
  /* S06 T1 (V4): frames applied by this loop, frames received but never
   * applied (overwritten before the loop read them, from the frame count),
   * largest state age callback -> applied */
  static uint32_t _n_applied = 0, _n_skipped = 0, _age_max_us = 0;
  static uint32_t _last_applied_nf = 0;
  const uint32_t cyc_per_us = SystemCoreClock / 1000000u;
  /* S06 T10: cost of the S06_IRQ_ENTER/EXIT pair; then (W10) the cost of one
   * idle pass, measured on this very loop with every interrupt masked for
   * 100 ms (no frame, no tick, so every pass is idle). Debug print turns
   * non-blocking after that. */
  static uint32_t _t10_last = 0;
  S06Load_CalibrateOverhead();
  uint8_t  _cal_on = 1;
  _dbg_last = _t10_last = HAL_GetTick();
  __disable_irq();
  uint32_t _cal_t0 = DWT->CYCCNT;
  uint32_t _cal_idle0 = S06Load_IdleCount();

  while (1)
  {
    uint8_t _busy = 0;
    PWMCapture_Service();      /* read new PWM captures every pass (04/11, DMA) */
    if (_cal_on && (DWT->CYCCNT - _cal_t0) >= SystemCoreClock / 10u) {
        S06Load_SetIdleCal(S06Load_IdleCount() - _cal_idle0, DWT->CYCCNT - _cal_t0);
        _cal_on = 0;
        HAL_NVIC_SetPriority(USART3_IRQn, 10, 0);   /* below every other handler */
        HAL_NVIC_EnableIRQ(USART3_IRQn);
        _dbg_async = 1;
        __enable_irq();
        _busy = 1;
    }
    if (SPI1_IsFrameReady()) {
        _busy = 1;
        _last_frame_ms = HAL_GetTick();
        if (!_link_up) {
            SPI1_Stats_t s0;
            SPI1_GetStats(&s0);
            _link_up = 1;
            IbusPort_SetEnabled(1);
            _up_first_ms = s0.last_frame_ms;
            _up_first_frames = s0.frames;
            printf("SPI link up at t=%lu ms\r\n", (unsigned long)_up_first_ms);
        }
        SPI1_FrameInfo_t fi;
        SPI1_GetFrameEx(&g_state, &g_traj, &fi);   /* one frame; clears frame_ready */
        for (int i = 0; i < ENC_NUM_WHEELS; i++)
            EncoderGen_SetOmega(i, g_state.omega[i]);
        /* S06: state age at the moment the omegas were applied, and the
         * checksum of the raw bytes of this very copy (W3, Y9) */
        uint32_t age_us = (DWT->CYCCNT - fi.rx_cyc) / cyc_per_us;
        SPI1_MisoInfo_t mi;
        mi.mosi_crc = S06_Crc16(0xFFFFu, fi.raw, SPI1_FRAME_LEN);
        mi.rx_cyc   = fi.rx_cyc;
        mi.rx_nf    = fi.rx_nf;
        mi.age_us   = (age_us > 0xFFFFu) ? 0xFFFFu : (uint16_t)age_us;
        if (_n_applied > 0u && fi.rx_nf > _last_applied_nf + 1u)
            _n_skipped += fi.rx_nf - _last_applied_nf - 1u;
        _last_applied_nf = fi.rx_nf;
        _n_applied++;
        if (age_us > _age_max_us) _age_max_us = age_us;

        /* Relay the latched VALID|NEW setpoint (also one from a frame this
         * loop did not see), not just the flags of the latest frame */
        HIL_TrajRef_t relay_traj;
        if (SPI1_TakeNewTraj(&relay_traj, NULL))
            UartRelay_Send(&relay_traj);

        /* Read PWM duty → pack into output torque for RPi5 */
        float duty[4];
        PWMCapture_GetDuty(duty);
        for (int i = 0; i < 4; i++)
            g_output.torque[i] = duty[i];

        g_output.seq = g_state.seq;
        SPI1_SetOutputEx(&g_output, &mi);

        HAL_GPIO_TogglePin(GPIOE, GPIO_PIN_1);
    } else if (_link_up && (HAL_GetTick() - _last_frame_ms) > SPI_LINK_TIMEOUT_MS) {
        _busy = 1;
        _link_up = 0;
        _n_link_lost++;
        IbusPort_SetEnabled(0);  /* DUT loses RC -> motors off (S06 T12 (d)) */
        for (int i = 0; i < ENC_NUM_WHEELS; i++)
            EncoderGen_SetOmega(i, 0.0f);
        SPI1_DropNewTraj();      /* no old setpoint relayed after reconnect */
        /* DIAG 08/11: state before the restart (main loop blocked about
         * SPI_DIAG_POLL_US here; encoders already stopped) */
        {
            SPI1_Diag_t da, db;
            SPI1_DiagSnapshot(&da);
            uint32_t t0 = DWT->CYCCNT, lim = SPI_DIAG_POLL_US * cyc_per_us;
            uint32_t tp = t0, falls = 0, low_cyc = 0;
            uint32_t prev = GPIOA->IDR & GPIO_PIN_4;
            while ((uint32_t)(DWT->CYCCNT - t0) < lim) {
                uint32_t now = DWT->CYCCNT;
                uint32_t lv  = GPIOA->IDR & GPIO_PIN_4;
                if (!lv) low_cyc += now - tp;
                if (prev && !lv) falls++;
                prev = lv;
                tp = now;
            }
            SPI1_DiagSnapshot(&db);
            _print_spi_diag("A", &da);
            _print_spi_rx(&da);
            _print_spi_diag("B", &db);
            printf("SPIdiag nss: %lu falling edges, low %lu us in %lu us between A and B (Pi at 1 kHz, 1 MHz: about 3 and 960)\r\n",
                   (unsigned long)falls, (unsigned long)(low_cyc / cyc_per_us),
                   (unsigned long)SPI_DIAG_POLL_US);
        }
        _diag_restart_ms = HAL_GetTick();
        SPI1_RequestRestart();   /* covers a DMA stuck without an error */
        SPI1_Stats_t s1;
        SPI1_GetStats(&s1);
        printf("SPI link lost (no frame for %lu ms): encoders stopped, iBUS stopped, SPI1 restart\r\n",
               (unsigned long)SPI_LINK_TIMEOUT_MS);
        /* Clock check: H7 ms between the first and the last frame of this
         * period vs frame intervals (= Pi ms at 1 kHz) */
        printf("Link period: %lu frames, H7 %lu ms from first to last frame (Pi: %lu ms at 1 kHz)\r\n",
               (unsigned long)(s1.frames - _up_first_frames + 1u),
               (unsigned long)(s1.last_frame_ms - _up_first_ms),
               (unsigned long)(s1.frames - _up_first_frames));
    } else if (!_link_up && _n_link_lost > 0u
               && (HAL_GetTick() - _diag_restart_ms) >= SPI_DIAG_RESTART_MS) {
        /* DIAG 08/11: keep restarting while down (also when the Pi stopped:
         * harmless, the restart only runs while NSS is high) */
        _busy = 1;
        _diag_restart_ms = HAL_GetTick();
        _diag_down_restarts++;
        SPI1_RequestRestart();
    }
    SPI1_Service();

    if ((HAL_GetTick() - _dbg_last) >= 500) {
        _busy = 1;
        _dbg_last = HAL_GetTick();

        EncoderGenDebug_t dbg[ENC_NUM_WHEELS];
        EncoderGen_GetDebug(dbg);
        for (int i = 0; i < ENC_NUM_WHEELS; i++) {
            printf("W%d: omega=%6.2f dir=%2d hp=%5lu meas=%7.2fHz exp=%7.2fHz\r\n",
                   i + 1,
                   (double)dbg[i].omega_cmd, (int)dbg[i].dir,
                   (unsigned long)dbg[i].half_period,
                   (double)dbg[i].measured_freq,
                   (double)dbg[i].expected_freq);
        }

        float duty[4];
        PWMCapture_GetDuty(duty);
        printf("PWM: %.2f %.2f %.2f %.2f\r\n",
               (double)duty[0], (double)duty[1],
               (double)duty[2], (double)duty[3]);

        /* DMA capture counters (04/11), channel order RPWM FL FR RL RR,
         * LPWM FL FR RL RR: e/s = captures per second (about 40000 on a
         * channel with 20 kHz PWM, 0 when held), lost = overcapture flags,
         * lap = service gaps >= 5 ms (ring dropped, FL channel), fix = edge
         * type corrected by the pin anchor, bad = periods > 75 us skipped,
         * una = runs of edges with the edge type unknown >= 20 ms (duty held),
         * known = edge type known, act = DMA stream running, dmaerr = stream
         * did not start or found stopped, win / svc = largest anchor window
         * and largest PWMCapture_Service() time in us over the last 2 s
         * (Codex M23, Liam: measure on the board; window above about 7 us:
         * revisit the window step).
         * Printed every 2 s, not every 0.5 s: the line is about 200 bytes and
         * printf blocks the main loop about 17 ms at 115200 baud (Codex M21). */
        static uint32_t _edges_last[PWM_NUM_CHANNELS];
        static uint32_t _pwm_dbg_ms = 0;
        uint32_t pwm_el = HAL_GetTick() - _pwm_dbg_ms;
        if (pwm_el >= 2000u) {
            _pwm_dbg_ms += pwm_el;
            PWMCapture_Stats_t ps;
            PWMCapture_GetStats(&ps);
            char pl[320];
            int n = snprintf(pl, sizeof(pl), "PWMdma: e/s=");
            for (int i = 0; i < (int)PWM_NUM_CHANNELS && n < (int)sizeof(pl); i++) {
                n += snprintf(pl + n, sizeof(pl) - (size_t)n, "%s%lu", i ? "," : "",
                              (unsigned long)((uint64_t)(ps.edges[i] - _edges_last[i]) * 1000u / pwm_el));
                _edges_last[i] = ps.edges[i];
            }
            for (int i = 0; i < (int)PWM_NUM_CHANNELS && n < (int)sizeof(pl); i++)
                n += snprintf(pl + n, sizeof(pl) - (size_t)n, "%s%lu", i ? "," : " lost=", (unsigned long)ps.lost[i]);
            if (n < (int)sizeof(pl))
                n += snprintf(pl + n, sizeof(pl) - (size_t)n, " lap=%lu", (unsigned long)ps.lap[0]);
            for (int i = 0; i < (int)PWM_NUM_CHANNELS && n < (int)sizeof(pl); i++)
                n += snprintf(pl + n, sizeof(pl) - (size_t)n, "%s%lu", i ? "," : " fix=", (unsigned long)ps.parity_fix[i]);
            for (int i = 0; i < (int)PWM_NUM_CHANNELS && n < (int)sizeof(pl); i++)
                n += snprintf(pl + n, sizeof(pl) - (size_t)n, "%s%lu", i ? "," : " bad=", (unsigned long)ps.bad_period[i]);
            for (int i = 0; i < (int)PWM_NUM_CHANNELS && n < (int)sizeof(pl); i++)
                n += snprintf(pl + n, sizeof(pl) - (size_t)n, "%s%lu", i ? "," : " una=", (unsigned long)ps.unanchored[i]);
            for (int i = 0; i < (int)PWM_NUM_CHANNELS && n < (int)sizeof(pl); i++)
                n += snprintf(pl + n, sizeof(pl) - (size_t)n, "%s%u", i ? "" : " known=", (unsigned)ps.known[i]);
            for (int i = 0; i < (int)PWM_NUM_CHANNELS && n < (int)sizeof(pl); i++)
                n += snprintf(pl + n, sizeof(pl) - (size_t)n, "%s%u", i ? "" : " act=", (unsigned)ps.active[i]);
            for (int i = 0; i < (int)PWM_NUM_CHANNELS && n < (int)sizeof(pl); i++)
                n += snprintf(pl + n, sizeof(pl) - (size_t)n, "%s%lu", i ? "," : " dmaerr=", (unsigned long)ps.dma_err[i]);
            uint32_t win_t, svc_t;
            PWMCapture_TakeTiming(&win_t, &svc_t);
            if (n < (int)sizeof(pl))
                n += snprintf(pl + n, sizeof(pl) - (size_t)n, " win=%lu.%lu svc=%lu.%lu",
                              (unsigned long)(win_t / 10u), (unsigned long)(win_t % 10u),
                              (unsigned long)(svc_t / 10u), (unsigned long)(svc_t % 10u));
            /* DIAG 05/11: windows by length (<7 / 7-20 / 20-50 / >=50 us) and shortest window */
            uint32_t wh[4], wmin_t;
            PWMCapture_TakeWinHist(wh, &wmin_t);
            if (n < (int)sizeof(pl))
                n += snprintf(pl + n, sizeof(pl) - (size_t)n, " wh=%lu/%lu/%lu/%lu wmin=%lu.%lu",
                              (unsigned long)wh[0], (unsigned long)wh[1], (unsigned long)wh[2], (unsigned long)wh[3],
                              (unsigned long)(wmin_t / 10u), (unsigned long)(wmin_t % 10u));
            printf("%s\r\n", pl);
        }
        uint32_t relay_sent, relay_dropped;
        UartRelay_GetStats(&relay_sent, &relay_dropped);
        printf("Relay: sent=%lu dropped=%lu  traj vx=%.2f vy=%.2f wz=%.2f\r\n",
               (unsigned long)relay_sent, (unsigned long)relay_dropped,
               (double)g_traj.vx_ref, (double)g_traj.vy_ref, (double)g_traj.wz_ref);
        static uint32_t _frames_last = 0;
        SPI1_Stats_t ss;
        SPI1_GetStats(&ss);
        printf("Link: %s lost=%lu  SPI frames=%lu/s err=%lu rearm=%lu last_err=0x%lx\r\n",
               _link_up ? "up" : "down", (unsigned long)_n_link_lost,
               (unsigned long)((ss.frames - _frames_last) * 2u),
               (unsigned long)ss.errors, (unsigned long)ss.rearms,
               (unsigned long)ss.last_error);
        /* DIAG 08/11: callbacks in the last 0.5 s; restarts; state when down */
        {
            SPI1_DiagCb_t cb;
            SPI1_DiagTakeCb(&cb);
            uint32_t r_done, r_fail;
            SPI1_DiagRestartStats(&r_done, &r_fail);
            printf("SPIcb: n=%lu gap<0.9/0.9-1.1/1.1-2/2-10/>=10ms=%lu/%lu/%lu/%lu/%lu"
                   " gap=%lu..%luus dur=%lu..%luus nss_low entry=%lu rearm=%lu seqj=%lu"
                   " rearm_fail=%lu restart=%lu fail=%lu down_req=%lu\r\n",
                   (unsigned long)cb.n,
                   (unsigned long)cb.gap_hist[0], (unsigned long)cb.gap_hist[1],
                   (unsigned long)cb.gap_hist[2], (unsigned long)cb.gap_hist[3],
                   (unsigned long)cb.gap_hist[4],
                   (unsigned long)(cb.n > 1u ? cb.gap_min_us : 0u), (unsigned long)cb.gap_max_us,
                   (unsigned long)(cb.n ? cb.dur_min_us : 0u), (unsigned long)cb.dur_max_us,
                   (unsigned long)cb.nss_low_entry, (unsigned long)cb.nss_low_rearm,
                   (unsigned long)cb.seq_jump, (unsigned long)cb.rearm_fail,
                   (unsigned long)r_done, (unsigned long)r_fail,
                   (unsigned long)_diag_down_restarts);
            if (!_link_up && _n_link_lost > 0u) {
                SPI1_Diag_t dd;
                SPI1_DiagSnapshot(&dd);
                _print_spi_diag("down", &dd);
            }
        }
        /* Frames the main loop did not read, and the NEW setpoint path:
         * seen = taken + replaced + dropped (+1 if one is pending) */
        printf("Frame: unread=%lu  new seen=%lu taken=%lu repl=%lu drop=%lu"
               "  applied=%lu skipped=%lu age_max=%luus\r\n",
               (unsigned long)ss.unread, (unsigned long)ss.new_seen,
               (unsigned long)ss.new_taken, (unsigned long)ss.new_replaced,
               (unsigned long)ss.new_dropped,
               (unsigned long)_n_applied, (unsigned long)_n_skipped,
               (unsigned long)_age_max_us);
        printf("S06log: pushed=%lu dropped=%lu  print_drop=%lu\r\n",
               (unsigned long)S06Log_Pushed(), (unsigned long)S06Log_Dropped(),
               (unsigned long)_dbg_drop);
        _frames_last = ss.frames;
        /* BNO085 emulator (G2): host init (adv, prod id, set features),
         * transactions and reports per sensor (accel / gyro / game RV) */
        BnoPort_Stats_t bs;
        BnoPort_GetStats(&bs);
        printf("BNO: st=%u rst=%lu adv=%lu pid=%lu sf=%lu gf=%lu rd=%lu empty=%lu wr=%lu col=%lu tmo=%lu\r\n",
               (unsigned)bs.state, (unsigned long)bs.core.resets,
               (unsigned long)bs.core.adverts, (unsigned long)bs.core.prod_id_reqs,
               (unsigned long)bs.core.set_features, (unsigned long)bs.core.get_features,
               (unsigned long)bs.core.reads, (unsigned long)bs.core.empty_reads,
               (unsigned long)bs.core.writes, (unsigned long)bs.core.collisions,
               (unsigned long)bs.core.int_timeouts);
        printf("BNO: rpt=%lu/%lu/%lu per=%lu/%lu/%lu us ovr=%lu stale=%lu drop=%lu unk=%lu"
               " | xfer=%lu mism=%lu miss=%lu armf=%lu err=%lu max=%lu smp=%lu\r\n",
               (unsigned long)bs.reports[0], (unsigned long)bs.reports[1],
               (unsigned long)bs.reports[2],
               (unsigned long)bs.period_us[0], (unsigned long)bs.period_us[1],
               (unsigned long)bs.period_us[2],
               (unsigned long)bs.core.input_overruns, (unsigned long)bs.core.stale_skips,
               (unsigned long)bs.core.q_drops, (unsigned long)bs.core.unknown_cmds,
               (unsigned long)bs.xfers, (unsigned long)bs.arm_mismatch,
               (unsigned long)bs.missed_edge, (unsigned long)bs.arm_fail,
               (unsigned long)bs.spi_errors, (unsigned long)bs.max_clocked,
               (unsigned long)bs.samples);
        /* iBUS 10 s summary (S06 T12 (b)): P = frame start to frame start
         * in us (DWT, real time), printed in ms with 0.01 ms; tgap = longest
         * time between two TIM6 ticks; once per closed window */
        static uint32_t _ibus_win_last = 0;
        ibus_gen_window_t iw;
        if (IbusPort_ReadWindow(&iw) && iw.window != _ibus_win_last) {
            _ibus_win_last = iw.window;
            printf("IBUS: win=%lu on=%lu frames=%lu P=%lu.%02lu..%lu.%02lu ms n=%lu late=%lu tgap=%lu us en=%lu total=%lu\r\n",
                   (unsigned long)iw.window, (unsigned long)iw.enabled,
                   (unsigned long)iw.frames,
                   (unsigned long)(iw.p_min_us / 1000u), (unsigned long)((iw.p_min_us % 1000u) / 10u),
                   (unsigned long)(iw.p_max_us / 1000u), (unsigned long)((iw.p_max_us % 1000u) / 10u),
                   (unsigned long)iw.p_count, (unsigned long)iw.late_ticks,
                   (unsigned long)iw.tick_gap_max_us,
                   (unsigned long)iw.enables, (unsigned long)iw.total_frames);
        }
        printf("---\r\n");

        HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_0);
    }
    if ((HAL_GetTick() - _t10_last) >= 10000u) {   /* S06 T10, every 10 s */
        _busy = 1;
        _t10_last = HAL_GetTick();
        S06Load_Report();
    }
    if (!_busy) S06Load_CountIdle();   /* W10: idle pass (an SPI1 rearm counts as idle) */
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 120;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 5;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
/* HAL_TIM_IC_CaptureCallback removed 04/11: PWM capture uses DMA, no capture interrupt */
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  __disable_irq();
  while (1) {}
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
