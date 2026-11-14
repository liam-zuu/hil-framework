/* PC stand-in for the H7 HAL header, only what spi1_handler.c uses
 * (tools/tests/test_h7_spi1_frame.py). The DMA is not simulated: the harness
 * records what HAL_SPI_TransmitReceive_DMA was armed with and plays the frame
 * by writing the rx buffer and calling SPI1_Callback(), i.e. the interrupt. */
#ifndef FAKE_STM32H7XX_HAL_H
#define FAKE_STM32H7XX_HAL_H

#include <stdint.h>

typedef enum { HAL_OK = 0, HAL_ERROR = 1 } HAL_StatusTypeDef;
typedef enum { GPIO_PIN_RESET = 0, GPIO_PIN_SET = 1 } GPIO_PinState;

/* SPI CR1 SPE and the HAL handle State (08/11, bug HIL-SPI1-SCK-GLITCH: the
 * armed check in SPI1_Service reads both). The harness DMA start sets
 * State BUSY_TX_RX and SPE, the abort clears them, as the HAL does. */
typedef struct { volatile uint32_t CR1; } SPI_TypeDef;
typedef struct { int dummy; } GPIO_TypeDef;
#define SPI_CR1_SPE  (1u << 0)
typedef enum { HAL_SPI_STATE_RESET = 0, HAL_SPI_STATE_READY = 1, HAL_SPI_STATE_BUSY = 2,
               HAL_SPI_STATE_BUSY_TX = 3, HAL_SPI_STATE_BUSY_RX = 4,
               HAL_SPI_STATE_BUSY_TX_RX = 5 } HAL_SPI_StateTypeDef;

typedef struct {
    SPI_TypeDef *Instance;
    uint32_t ErrorCode;
    HAL_SPI_StateTypeDef State;   /* after ErrorCode: the { inst, 0 } init stays valid */
} SPI_HandleTypeDef;

extern GPIO_TypeDef fake_gpioa;
#define GPIOA       (&fake_gpioa)
#define GPIO_PIN_4  (1u << 4)

HAL_StatusTypeDef HAL_SPI_TransmitReceive_DMA(SPI_HandleTypeDef *h, uint8_t *tx, uint8_t *rx, uint16_t n);
HAL_StatusTypeDef HAL_SPI_Abort(SPI_HandleTypeDef *h);
GPIO_PinState     HAL_GPIO_ReadPin(GPIO_TypeDef *port, uint16_t pin);
uint32_t          HAL_GetTick(void);

/* S06: cycle counter (the harness sets CYCCNT before each frame) */
typedef struct { volatile uint32_t CYCCNT; } DWT_Type;
extern DWT_Type fake_dwt;
#define DWT (&fake_dwt)

extern int  fake_irq_disabled;
extern long fake_irq_sections;
/* RS-4 P1 (Codex RS4-R10-3): an interrupt pending while masked runs when the
 * mask is lifted. The harness may arm fake_irq_hook to fire at the n-th
 * unmask from now (fake_irq_hook_in = n): one-shot, cleared before it runs. */
extern void (*fake_irq_hook)(void);
extern int  fake_irq_hook_in;
static inline void fake_unmasked(void)
{
    if (fake_irq_hook && --fake_irq_hook_in <= 0) {
        void (*h)(void) = fake_irq_hook;
        fake_irq_hook = 0;
        h();
    }
}
static inline uint32_t __get_PRIMASK(void) { return (uint32_t)fake_irq_disabled; }
static inline void __disable_irq(void) { fake_irq_disabled = 1; fake_irq_sections++; }
static inline void __enable_irq(void) { fake_irq_disabled = 0; fake_unmasked(); }
static inline void __set_PRIMASK(uint32_t m) { fake_irq_disabled = (int)m; if (!m) fake_unmasked(); }

#endif
