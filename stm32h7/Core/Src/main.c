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
int _write(int file, char *ptr, int len)
{
    HAL_UART_Transmit(&huart3, (uint8_t *)ptr, (uint16_t)len, 100);
    return len;
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

/* BNO085 emulator (G2): TIM6 10 kHz tick and RST pin PG3 (EXTI3) */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM6) BnoPort_Tick();
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    BnoPort_ExtiCallback(GPIO_Pin);
}

void HAL_TIM_OC_DelayElapsedCallback(TIM_HandleTypeDef *htim)
{
    EncoderGen_OC_Callback(htim);
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

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
  MX_TIM13_Init();
  MX_TIM6_Init();
  /* USER CODE BEGIN 2 */

  HAL_NVIC_DisableIRQ(TIM5_IRQn);
  HAL_NVIC_DisableIRQ(TIM15_IRQn);
  HAL_NVIC_DisableIRQ(TIM16_IRQn);
  HAL_NVIC_DisableIRQ(TIM17_IRQn);
  HAL_NVIC_DisableIRQ(TIM8_UP_TIM13_IRQn);   /* TIM13 = wheel 2 RPWM (PF8) */

  setvbuf(stdout, NULL, _IONBF, 0);

  printf("\r\n--- H753 HIL boot ---\r\n");
  printf("pclk1=%lu  pclk2=%lu\r\n",
         HAL_RCC_GetPCLK1Freq(), HAL_RCC_GetPCLK2Freq());

  EncoderGen_Init(&htim1, &htim2, &htim3, &htim4);
  SPI1_Handler_Init(&hspi1);
  UartRelay_Init(&huart2);     /* +++ NEW */

  /* TIM5/13/15/16/17 priority lower than SPI1 DMA (priority 0) — Bug #7 fix */
  HAL_NVIC_SetPriority(TIM5_IRQn,  5, 0);
  HAL_NVIC_SetPriority(TIM8_UP_TIM13_IRQn, 5, 0);
  HAL_NVIC_SetPriority(TIM15_IRQn, 5, 0);
  HAL_NVIC_SetPriority(TIM16_IRQn, 5, 0);
  HAL_NVIC_SetPriority(TIM17_IRQn, 5, 0);
  /* NVIC lines stay off until PWMCapture_Init has cleared stale flags (25/10:
   * enabling them here first could hang the boot after "resolve bits"). */

  PWMCapture_Init(&htim5, &htim13, &htim15, &htim16, &htim17);

  /* One line before each enable: if the boot stops right after a line, that
   * timer's interrupt is the one that never lets main run (26/10). */
  printf("NVIC: enabling TIM5\r\n");
  HAL_NVIC_EnableIRQ(TIM5_IRQn);
  printf("NVIC: enabling TIM13\r\n");
  HAL_NVIC_EnableIRQ(TIM8_UP_TIM13_IRQn);
  printf("NVIC: enabling TIM15/16/17\r\n");
  HAL_NVIC_EnableIRQ(TIM15_IRQn);
  HAL_NVIC_EnableIRQ(TIM16_IRQn);
  HAL_NVIC_EnableIRQ(TIM17_IRQn);
  printf("BNO: emulator on SPI4 (INT PG2, RST PG3), TIM6 tick\r\n");
  BnoPort_Init(&hspi4, &htim6);
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

  while (1)
  {
    if (SPI1_IsFrameReady()) {
        _last_frame_ms = HAL_GetTick();
        if (!_link_up) {
            SPI1_Stats_t s0;
            SPI1_GetStats(&s0);
            _link_up = 1;
            _up_first_ms = s0.last_frame_ms;
            _up_first_frames = s0.frames;
            printf("SPI link up at t=%lu ms\r\n", (unsigned long)_up_first_ms);
        }
        SPI1_GetFrame(&g_state, &g_traj);   /* one frame; clears frame_ready */

        for (int i = 0; i < ENC_NUM_WHEELS; i++)
            EncoderGen_SetOmega(i, g_state.omega[i]);

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
        SPI1_SetOutput(&g_output);

        HAL_GPIO_TogglePin(GPIOE, GPIO_PIN_1);
    } else if (_link_up && (HAL_GetTick() - _last_frame_ms) > SPI_LINK_TIMEOUT_MS) {
        _link_up = 0;
        _n_link_lost++;
        for (int i = 0; i < ENC_NUM_WHEELS; i++)
            EncoderGen_SetOmega(i, 0.0f);
        SPI1_DropNewTraj();      /* no old setpoint relayed after reconnect */
        SPI1_RequestRestart();   /* covers a DMA stuck without an error */
        SPI1_Stats_t s1;
        SPI1_GetStats(&s1);
        printf("SPI link lost (no frame for %lu ms): encoders stopped, SPI1 restart\r\n",
               (unsigned long)SPI_LINK_TIMEOUT_MS);
        /* Clock check: H7 ms between the first and the last frame of this
         * period vs frame intervals (= Pi ms at 1 kHz) */
        printf("Link period: %lu frames, H7 %lu ms from first to last frame (Pi: %lu ms at 1 kHz)\r\n",
               (unsigned long)(s1.frames - _up_first_frames + 1u),
               (unsigned long)(s1.last_frame_ms - _up_first_ms),
               (unsigned long)(s1.frames - _up_first_frames));
    }
    SPI1_Service();

    if ((HAL_GetTick() - _dbg_last) >= 500) {
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

        /* Capture interrupt rate (debug, 25/10): expected about 2 x 20 kHz per
         * channel that has PWM edges, 0 when all PWM pins are idle. */
        static uint32_t _isr_last = 0;
        uint32_t isr_now = PWMCapture_GetIsrCount();
        printf("ISR: %lu/s\r\n", (unsigned long)((isr_now - _isr_last) * 2u));
        _isr_last = isr_now;
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
        /* Frames the main loop did not read, and the NEW setpoint path:
         * seen = taken + replaced + dropped (+1 if one is pending) */
        printf("Frame: unread=%lu  new seen=%lu taken=%lu repl=%lu drop=%lu\r\n",
               (unsigned long)ss.unread, (unsigned long)ss.new_seen,
               (unsigned long)ss.new_taken, (unsigned long)ss.new_replaced,
               (unsigned long)ss.new_dropped);
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
        printf("---\r\n");

        HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_0);
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
void HAL_TIM_IC_CaptureCallback(TIM_HandleTypeDef *htim)
{
    PWMCapture_IC_Callback(htim);
}
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
