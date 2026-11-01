/* PC stand-in for the H7 HAL header, only what spi1_handler.c uses
 * (tools/tests/test_h7_spi1_frame.py). The DMA is not simulated: the harness
 * records what HAL_SPI_TransmitReceive_DMA was armed with and plays the frame
 * by writing the rx buffer and calling SPI1_Callback(), i.e. the interrupt. */
#ifndef FAKE_STM32H7XX_HAL_H
#define FAKE_STM32H7XX_HAL_H

#include <stdint.h>

typedef enum { HAL_OK = 0, HAL_ERROR = 1 } HAL_StatusTypeDef;
typedef enum { GPIO_PIN_RESET = 0, GPIO_PIN_SET = 1 } GPIO_PinState;

typedef struct { int dummy; } SPI_TypeDef;
typedef struct { int dummy; } GPIO_TypeDef;

typedef struct {
    SPI_TypeDef *Instance;
    uint32_t ErrorCode;
} SPI_HandleTypeDef;

extern GPIO_TypeDef fake_gpioa;
#define GPIOA       (&fake_gpioa)
#define GPIO_PIN_4  (1u << 4)

HAL_StatusTypeDef HAL_SPI_TransmitReceive_DMA(SPI_HandleTypeDef *h, uint8_t *tx, uint8_t *rx, uint16_t n);
HAL_StatusTypeDef HAL_SPI_Abort(SPI_HandleTypeDef *h);
GPIO_PinState     HAL_GPIO_ReadPin(GPIO_TypeDef *port, uint16_t pin);
uint32_t          HAL_GetTick(void);

extern int  fake_irq_disabled;
extern long fake_irq_sections;
static inline uint32_t __get_PRIMASK(void) { return (uint32_t)fake_irq_disabled; }
static inline void __disable_irq(void) { fake_irq_disabled = 1; fake_irq_sections++; }
static inline void __enable_irq(void) { fake_irq_disabled = 0; }
static inline void __set_PRIMASK(uint32_t m) { fake_irq_disabled = (int)m; }

#endif
