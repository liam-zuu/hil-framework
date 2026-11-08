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
/* While the link is down after having been up: SPI1 restart requested every
 * this many ms (08/11, bug HIL-SPI1-SCK-GLITCH). Before, one request at the
 * up -> down transition only, so a restart that returned HAL_OK but never
 * completed left the link dead until a reset (CLOSED_LOOP/2026-11-08_run02,
 * 01:48). Harmless when the Pi has stopped: the restart runs only while NSS
 * is high. */
#define SPI_DOWN_RESTART_MS  200u
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

/* Debug block, one line per main-loop pass (08/11, STATUS_CODE L51). Before,
 * the whole 0.5 s block (and the PWMdma line every 2 s, the T10 report every
 * 10 s) was formatted in ONE pass: the loop missed 1-5 SPI1 frames each time
 * (reply in frame N+3..N+7 instead of N+2, frames never applied:
 * CLOSED_LOOP/2026-11-08_run08). Now every counter is copied once when the
 * block is due (_dbg_snap_t), then each pass prints at most one line (the
 * PWMdma line is built in sections, one per pass) and only when no SPI1 frame
 * is waiting. Lines and their order are unchanged, plus the "Loop:" line. */
typedef struct {
    EncoderGenDebug_t  enc[ENC_NUM_WHEELS];
    float              duty[4];
    uint8_t            pwm_line;                 /* PWMdma line in this block */
    uint32_t           pwm_es[PWM_NUM_CHANNELS];
    PWMCapture_Stats_t ps;
    uint32_t           win_t, svc_t, wh[4], wmin_t;
    uint32_t           relay_sent, relay_dropped;
    HIL_TrajRef_t      traj;
    uint8_t            link_up;
    uint32_t           n_link_lost;
    SPI1_Stats_t       ss;
    uint32_t           frames_per_s;
    uint32_t           svc_min, svc_max, n_down_restarts, cyc_per_us;
    uint32_t           n_applied, n_skipped, age_max_us;
    uint32_t           s06_pushed, s06_dropped, dbg_drop;
    BnoPort_Stats_t    bs;
    uint8_t            ibus_new;
    ibus_gen_window_t  iw;
    /* main-loop passes since the last block (L39): longest pass, longest
     * pass that did debug work (copy or one line), passes over 500 us and
     * over 1 ms, passes; DWT cycles, preemption by interrupts included */
    uint32_t           lp_max, lp_prn_max, lp_n500, lp_n1ms, lp_n;
} _dbg_snap_t;

enum {
    DS_W1 = 0, DS_PWM = 4, DS_DMA0 = 5, DS_DMA_LAST = 13,   /* PWMdma: 9 sections */
    DS_RELAY, DS_LINK, DS_SPIFIX, DS_LOOP, DS_FRAME, DS_S06LOG, DS_BNO1, DS_BNO2,
    DS_IBUS, DS_END
};

static _dbg_snap_t _ds;
static int         _ds_step = -1;                /* -1: nothing to print */
static char        _ds_pl[320];
static int         _ds_pn;
static uint32_t    _lp_max, _lp_prn_max, _lp_n500, _lp_n1ms, _lp_n;

#define _PL(...) do { if (_ds_pn < (int)sizeof(_ds_pl))                                 \
        _ds_pn += snprintf(_ds_pl + _ds_pn, sizeof(_ds_pl) - (size_t)_ds_pn, __VA_ARGS__); \
    } while (0)

/* us with one decimal from DWT cycles */
#define _US10(c) ((unsigned long)((uint64_t)(c) * 10u / _ds.cyc_per_us))

static void _dbg_print_next(void)
{
    const _dbg_snap_t *d = &_ds;
    int s = _ds_step;
    if (s >= DS_W1 && s < DS_PWM) {
        int i = s - DS_W1;
        printf("W%d: omega=%6.2f dir=%2d hp=%5lu meas=%7.2fHz exp=%7.2fHz\r\n",
               i + 1,
               (double)d->enc[i].omega_cmd, (int)d->enc[i].dir,
               (unsigned long)d->enc[i].half_period,
               (double)d->enc[i].measured_freq,
               (double)d->enc[i].expected_freq);
    } else if (s == DS_PWM) {
        printf("PWM: %.2f %.2f %.2f %.2f\r\n",
               (double)d->duty[0], (double)d->duty[1],
               (double)d->duty[2], (double)d->duty[3]);
        if (!d->pwm_line) s = DS_DMA_LAST;       /* no PWMdma line this block */
    } else if (s >= DS_DMA0 && s <= DS_DMA_LAST) {
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
         * revisit the window step). Printed every 2 s (Codex M21). */
        switch (s - DS_DMA0) {
        case 0:
            _ds_pn = 0;
            _PL("PWMdma: e/s=");
            for (int i = 0; i < (int)PWM_NUM_CHANNELS; i++) _PL("%s%lu", i ? "," : "", (unsigned long)d->pwm_es[i]);
            break;
        case 1: for (int i = 0; i < (int)PWM_NUM_CHANNELS; i++) _PL("%s%lu", i ? "," : " lost=", (unsigned long)d->ps.lost[i]); break;
        case 2:
            _PL(" lap=%lu", (unsigned long)d->ps.lap[0]);
            for (int i = 0; i < (int)PWM_NUM_CHANNELS; i++) _PL("%s%lu", i ? "," : " fix=", (unsigned long)d->ps.parity_fix[i]);
            break;
        case 3: for (int i = 0; i < (int)PWM_NUM_CHANNELS; i++) _PL("%s%lu", i ? "," : " bad=", (unsigned long)d->ps.bad_period[i]); break;
        case 4: for (int i = 0; i < (int)PWM_NUM_CHANNELS; i++) _PL("%s%lu", i ? "," : " una=", (unsigned long)d->ps.unanchored[i]); break;
        case 5: for (int i = 0; i < (int)PWM_NUM_CHANNELS; i++) _PL("%s%u", i ? "" : " known=", (unsigned)d->ps.known[i]); break;
        case 6: for (int i = 0; i < (int)PWM_NUM_CHANNELS; i++) _PL("%s%u", i ? "" : " act=", (unsigned)d->ps.active[i]); break;
        case 7: for (int i = 0; i < (int)PWM_NUM_CHANNELS; i++) _PL("%s%lu", i ? "," : " dmaerr=", (unsigned long)d->ps.dma_err[i]); break;
        default:
            _PL(" win=%lu.%lu svc=%lu.%lu",
                (unsigned long)(d->win_t / 10u), (unsigned long)(d->win_t % 10u),
                (unsigned long)(d->svc_t / 10u), (unsigned long)(d->svc_t % 10u));
            /* DIAG 05/11: windows by length (<7 / 7-20 / 20-50 / >=50 us) and shortest window */
            _PL(" wh=%lu/%lu/%lu/%lu wmin=%lu.%lu",
                (unsigned long)d->wh[0], (unsigned long)d->wh[1], (unsigned long)d->wh[2], (unsigned long)d->wh[3],
                (unsigned long)(d->wmin_t / 10u), (unsigned long)(d->wmin_t % 10u));
            printf("%s\r\n", _ds_pl);
            break;
        }
    } else if (s == DS_RELAY) {
        printf("Relay: sent=%lu dropped=%lu  traj vx=%.2f vy=%.2f wz=%.2f\r\n",
               (unsigned long)d->relay_sent, (unsigned long)d->relay_dropped,
               (double)d->traj.vx_ref, (double)d->traj.vy_ref, (double)d->traj.wz_ref);
    } else if (s == DS_LINK) {
        printf("Link: %s lost=%lu  SPI frames=%lu/s err=%lu rearm=%lu last_err=0x%lx\r\n",
               d->link_up ? "up" : "down", (unsigned long)d->n_link_lost,
               (unsigned long)d->frames_per_s,
               (unsigned long)d->ss.errors, (unsigned long)d->ss.rearms,
               (unsigned long)d->ss.last_error);
    } else if (s == DS_SPIFIX) {
        /* 08/11 (bug HIL-SPI1-SCK-GLITCH): SPI1 found not armed (each one is
         * a lost frame; a run with stuck > 0 is not a main-number run),
         * failed restarts, restarts requested while down, and the shortest /
         * longest SPI1_Service() call over the last 0.5 s (L39) */
        printf("SPIfix: stuck=%lu stuck_err=0x%lx restart_fail=%lu down_req=%lu svc=%lu..%lu cyc (%lu cyc/us)\r\n",
               (unsigned long)d->ss.stuck, (unsigned long)d->ss.stuck_err,
               (unsigned long)d->ss.restart_fail, (unsigned long)d->n_down_restarts,
               (unsigned long)d->svc_min, (unsigned long)d->svc_max,
               (unsigned long)d->cyc_per_us);
    } else if (s == DS_LOOP) {
        /* 08/11 (L51, L39): main-loop pass times since the last block; a pass
         * over 1 ms can miss an SPI1 frame (one frame per ms) */
        printf("Loop: max=%lu.%lu us prn_max=%lu.%lu us over500us=%lu over1ms=%lu passes=%lu\r\n",
               _US10(d->lp_max) / 10u, _US10(d->lp_max) % 10u,
               _US10(d->lp_prn_max) / 10u, _US10(d->lp_prn_max) % 10u,
               (unsigned long)d->lp_n500, (unsigned long)d->lp_n1ms, (unsigned long)d->lp_n);
    } else if (s == DS_FRAME) {
        /* Frames the main loop did not read, and the NEW setpoint path:
         * seen = taken + replaced + dropped (+1 if one is pending) */
        printf("Frame: unread=%lu  new seen=%lu taken=%lu repl=%lu drop=%lu"
               "  applied=%lu skipped=%lu age_max=%luus\r\n",
               (unsigned long)d->ss.unread, (unsigned long)d->ss.new_seen,
               (unsigned long)d->ss.new_taken, (unsigned long)d->ss.new_replaced,
               (unsigned long)d->ss.new_dropped,
               (unsigned long)d->n_applied, (unsigned long)d->n_skipped,
               (unsigned long)d->age_max_us);
    } else if (s == DS_S06LOG) {
        printf("S06log: pushed=%lu dropped=%lu  print_drop=%lu\r\n",
               (unsigned long)d->s06_pushed, (unsigned long)d->s06_dropped,
               (unsigned long)d->dbg_drop);
    } else if (s == DS_BNO1) {
        /* BNO085 emulator (G2): host init (adv, prod id, set features),
         * transactions and reports per sensor (accel / gyro / game RV) */
        const BnoPort_Stats_t *bs = &d->bs;
        printf("BNO: st=%u rst=%lu adv=%lu pid=%lu sf=%lu gf=%lu rd=%lu empty=%lu wr=%lu col=%lu tmo=%lu\r\n",
               (unsigned)bs->state, (unsigned long)bs->core.resets,
               (unsigned long)bs->core.adverts, (unsigned long)bs->core.prod_id_reqs,
               (unsigned long)bs->core.set_features, (unsigned long)bs->core.get_features,
               (unsigned long)bs->core.reads, (unsigned long)bs->core.empty_reads,
               (unsigned long)bs->core.writes, (unsigned long)bs->core.collisions,
               (unsigned long)bs->core.int_timeouts);
    } else if (s == DS_BNO2) {
        const BnoPort_Stats_t *bs = &d->bs;
        printf("BNO: rpt=%lu/%lu/%lu per=%lu/%lu/%lu us ovr=%lu stale=%lu drop=%lu unk=%lu"
               " | xfer=%lu mism=%lu miss=%lu armf=%lu err=%lu max=%lu smp=%lu\r\n",
               (unsigned long)bs->reports[0], (unsigned long)bs->reports[1],
               (unsigned long)bs->reports[2],
               (unsigned long)bs->period_us[0], (unsigned long)bs->period_us[1],
               (unsigned long)bs->period_us[2],
               (unsigned long)bs->core.input_overruns, (unsigned long)bs->core.stale_skips,
               (unsigned long)bs->core.q_drops, (unsigned long)bs->core.unknown_cmds,
               (unsigned long)bs->xfers, (unsigned long)bs->arm_mismatch,
               (unsigned long)bs->missed_edge, (unsigned long)bs->arm_fail,
               (unsigned long)bs->spi_errors, (unsigned long)bs->max_clocked,
               (unsigned long)bs->samples);
    } else if (s == DS_IBUS) {
        /* iBUS 10 s summary (S06 T12 (b)): P = frame start to frame start
         * in us (DWT, real time), printed in ms with 0.01 ms; tgap = longest
         * time between two TIM6 ticks; once per closed window */
        if (d->ibus_new) {
            const ibus_gen_window_t *iw = &d->iw;
            printf("IBUS: win=%lu on=%lu frames=%lu P=%lu.%02lu..%lu.%02lu ms n=%lu late=%lu tgap=%lu us en=%lu total=%lu\r\n",
                   (unsigned long)iw->window, (unsigned long)iw->enabled,
                   (unsigned long)iw->frames,
                   (unsigned long)(iw->p_min_us / 1000u), (unsigned long)((iw->p_min_us % 1000u) / 10u),
                   (unsigned long)(iw->p_max_us / 1000u), (unsigned long)((iw->p_max_us % 1000u) / 10u),
                   (unsigned long)iw->p_count, (unsigned long)iw->late_ticks,
                   (unsigned long)iw->tick_gap_max_us,
                   (unsigned long)iw->enables, (unsigned long)iw->total_frames);
        } else {
            s = DS_END;                          /* no IBUS line: "---" in this pass */
            printf("---\r\n");
        }
    } else {
        printf("---\r\n");
    }
    _ds_step = (s >= DS_END) ? -1 : s + 1;
}
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
  static uint32_t _down_restart_ms = 0;
  static uint32_t _n_down_restarts = 0;   /* restarts requested while down */
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
  static uint8_t  _t10_pend = 0;       /* T10 window taken, lines not all printed */
  /* debug block state, kept across blocks (were statics inside the block) */
  static uint32_t _edges_last[PWM_NUM_CHANNELS];
  static uint32_t _pwm_dbg_ms = 0;
  static uint32_t _frames_last = 0;
  static uint32_t _ibus_win_last = 0;
  S06Load_CalibrateOverhead();
  uint8_t  _cal_on = 1;
  _dbg_last = _t10_last = HAL_GetTick();
  __disable_irq();
  uint32_t _cal_t0 = DWT->CYCCNT;
  uint32_t _cal_idle0 = S06Load_IdleCount();

  while (1)
  {
    uint32_t _pass_t0 = DWT->CYCCNT;   /* L51: pass time, "Loop:" line */
    uint8_t _busy = 0;
    uint8_t _prn = 0;                  /* this pass did debug work */
    PWMCapture_Service();     /* read new PWM captures every pass (04/11, DMA) */
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
        SPI1_RequestRestart();   /* covers a DMA stuck without an error */
        _down_restart_ms = HAL_GetTick();
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
               && (HAL_GetTick() - _down_restart_ms) >= SPI_DOWN_RESTART_MS) {
        _busy = 1;
        _down_restart_ms = HAL_GetTick();
        _n_down_restarts++;
        SPI1_RequestRestart();
    }
    SPI1_Service();

    /* Debug block (08/11, L51): copy every counter once when due and the
     * printer is idle, then print one line per pass (_dbg_print_next) */
    if (_ds_step < 0 && !_t10_pend && (HAL_GetTick() - _dbg_last) >= 500) {
        _busy = 1;
        _prn = 1;
        _dbg_last = HAL_GetTick();
        _dbg_snap_t *d = &_ds;
        EncoderGen_GetDebug(d->enc);
        PWMCapture_GetDuty(d->duty);
        uint32_t pwm_el = HAL_GetTick() - _pwm_dbg_ms;
        d->pwm_line = (pwm_el >= 2000u);
        if (d->pwm_line) {
            _pwm_dbg_ms += pwm_el;
            PWMCapture_GetStats(&d->ps);
            for (int i = 0; i < (int)PWM_NUM_CHANNELS; i++) {
                d->pwm_es[i] = (uint32_t)((uint64_t)(d->ps.edges[i] - _edges_last[i]) * 1000u / pwm_el);
                _edges_last[i] = d->ps.edges[i];
            }
            PWMCapture_TakeTiming(&d->win_t, &d->svc_t);
            PWMCapture_TakeWinHist(d->wh, &d->wmin_t);
        }
        UartRelay_GetStats(&d->relay_sent, &d->relay_dropped);
        d->traj = g_traj;
        SPI1_GetStats(&d->ss);
        d->frames_per_s = (d->ss.frames - _frames_last) * 2u;
        _frames_last = d->ss.frames;
        d->link_up = _link_up;
        d->n_link_lost = _n_link_lost;
        SPI1_TakeServiceCycles(&d->svc_min, &d->svc_max);
        d->n_down_restarts = _n_down_restarts;
        d->cyc_per_us = cyc_per_us;
        d->n_applied = _n_applied;
        d->n_skipped = _n_skipped;
        d->age_max_us = _age_max_us;
        d->s06_pushed = S06Log_Pushed();
        d->s06_dropped = S06Log_Dropped();
        d->dbg_drop = _dbg_drop;
        BnoPort_GetStats(&d->bs);
        d->ibus_new = (uint8_t)(IbusPort_ReadWindow(&d->iw) && d->iw.window != _ibus_win_last);
        if (d->ibus_new) _ibus_win_last = d->iw.window;
        d->lp_max = _lp_max;   d->lp_prn_max = _lp_prn_max;
        d->lp_n500 = _lp_n500; d->lp_n1ms = _lp_n1ms; d->lp_n = _lp_n;
        _lp_max = _lp_prn_max = _lp_n500 = _lp_n1ms = _lp_n = 0u;
        _ds_step = DS_W1;

        HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_0);
    }
    if ((HAL_GetTick() - _t10_last) >= 10000u) {   /* S06 T10, every 10 s */
        _busy = 1;
        _prn = 1;
        _t10_last = HAL_GetTick();
        S06Load_Snapshot();      /* window taken on time, lines printed below */
        _t10_pend = 1;
    }
    /* At most one debug line per pass, and none while an SPI1 frame waits
     * (next pass handles the frame first). T10 lines once the block is out. */
    if (!_prn && !SPI1_IsFrameReady()) {
        if (_ds_step >= 0) {
            _busy = 1;
            _prn = 1;
            _dbg_print_next();
        } else if (_t10_pend) {
            _busy = 1;
            _prn = 1;
            if (!S06Load_PrintNext()) _t10_pend = 0;
        }
    }
    if (!_busy) S06Load_CountIdle();   /* W10: idle pass (an SPI1 rearm counts as idle) */
    {                                  /* L51: pass time (interrupts included) */
        uint32_t dt = DWT->CYCCNT - _pass_t0;
        _lp_n++;
        if (dt > _lp_max) _lp_max = dt;
        if (_prn && dt > _lp_prn_max) _lp_prn_max = dt;
        if (dt > 500u * cyc_per_us) _lp_n500++;
        if (dt > 1000u * cyc_per_us) _lp_n1ms++;
    }
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
