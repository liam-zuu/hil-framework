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
#include <stdio.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

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
  /* USER CODE BEGIN 2 */

  HAL_NVIC_DisableIRQ(TIM5_IRQn);
  HAL_NVIC_DisableIRQ(TIM15_IRQn);
  HAL_NVIC_DisableIRQ(TIM16_IRQn);
  HAL_NVIC_DisableIRQ(TIM17_IRQn);

  setvbuf(stdout, NULL, _IONBF, 0);

  printf("\r\n--- H753 HIL boot ---\r\n");
  printf("pclk1=%lu  pclk2=%lu\r\n",
         HAL_RCC_GetPCLK1Freq(), HAL_RCC_GetPCLK2Freq());

  EncoderGen_Init(&htim1, &htim2, &htim3, &htim4);
  SPI1_Handler_Init(&hspi1);
  UartRelay_Init(&huart2);     /* +++ NEW */

  /* TIM5/15/16/17 priority lower than SPI1 DMA (priority 0) — Bug #7 fix */
  HAL_NVIC_SetPriority(TIM5_IRQn,  5, 0);
  HAL_NVIC_SetPriority(TIM15_IRQn, 5, 0);
  HAL_NVIC_SetPriority(TIM16_IRQn, 5, 0);
  HAL_NVIC_SetPriority(TIM17_IRQn, 5, 0);
  HAL_NVIC_EnableIRQ(TIM5_IRQn);
  HAL_NVIC_EnableIRQ(TIM15_IRQn);
  HAL_NVIC_EnableIRQ(TIM16_IRQn);
  HAL_NVIC_EnableIRQ(TIM17_IRQn);

  PWMCapture_Init(&htim5, &htim15, &htim16, &htim17);

  HAL_NVIC_EnableIRQ(TIM5_IRQn);
  HAL_NVIC_EnableIRQ(TIM15_IRQn);
  HAL_NVIC_EnableIRQ(TIM16_IRQn);
  HAL_NVIC_EnableIRQ(TIM17_IRQn);
  printf("Init done, entering main loop\r\n");
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  static uint32_t _dbg_last = 0;

  while (1)
  {
    if (SPI1_IsFrameReady()) {
        SPI1_GetState(&g_state);
        SPI1_GetTrajRef(&g_traj);   /* +++ NEW — clears frame_ready */

        for (int i = 0; i < ENC_NUM_WHEELS; i++)
            EncoderGen_SetOmega(i, g_state.omega[i]);

        UartRelay_Send(&g_traj);    /* +++ NEW — relay traj → ESP32 */

        /* Read PWM duty → pack into output torque for RPi5 */
        float duty[4];
        PWMCapture_GetDuty(duty);
        for (int i = 0; i < 4; i++)
            g_output.torque[i] = duty[i];

        g_output.seq = g_state.seq;
        SPI1_SetOutput(&g_output);

        HAL_GPIO_TogglePin(GPIOE, GPIO_PIN_1);
    }

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
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
