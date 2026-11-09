/* PC stub of the CubeMX usart.h, only what uart_relay.c needs, so that
 * stm32h7/Core/Src/uart_relay.c compiles unchanged with the host gcc
 * (tools/tests/test_h7_uart_relay.py). Not used by the firmware build. */
#ifndef USART_H_PC_STUB
#define USART_H_PC_STUB

#include <stdint.h>

typedef struct { volatile uint32_t ISR; volatile uint32_t TDR; } USART_TypeDef;
typedef struct { USART_TypeDef *Instance; } UART_HandleTypeDef;
typedef enum { HAL_OK = 0, HAL_ERROR = 1 } HAL_StatusTypeDef;
#define USART_ISR_TXFE (1u << 23)

HAL_StatusTypeDef HAL_UARTEx_EnableFifoMode(UART_HandleTypeDef *huart);

#endif
