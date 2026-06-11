/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32h7xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

void HAL_TIM_MspPostInit(TIM_HandleTypeDef *htim);

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define LD1_Pin GPIO_PIN_0
#define LD1_GPIO_Port GPIOB
#define LD3_Pin GPIO_PIN_14
#define LD3_GPIO_Port GPIOB
#define W1_REN_Pin GPIO_PIN_4
#define W1_REN_GPIO_Port GPIOG
#define W2_REN_Pin GPIO_PIN_5
#define W2_REN_GPIO_Port GPIOG
#define W3_REN_Pin GPIO_PIN_6
#define W3_REN_GPIO_Port GPIOG
#define W4_REN_Pin GPIO_PIN_7
#define W4_REN_GPIO_Port GPIOG
#define W1_LEN_Pin GPIO_PIN_8
#define W1_LEN_GPIO_Port GPIOC
#define W2_LEN_Pin GPIO_PIN_9
#define W2_LEN_GPIO_Port GPIOC
#define W3_LEN_Pin GPIO_PIN_8
#define W3_LEN_GPIO_Port GPIOA
#define W4_LEN_Pin GPIO_PIN_9
#define W4_LEN_GPIO_Port GPIOA
#define SYNC_IN_Pin GPIO_PIN_0
#define SYNC_IN_GPIO_Port GPIOD
#define SYNC_IN_EXTI_IRQn EXTI0_IRQn
#define BNO_HINT_Pin GPIO_PIN_1
#define BNO_HINT_GPIO_Port GPIOD
#define LD2_Pin GPIO_PIN_1
#define LD2_GPIO_Port GPIOE

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
