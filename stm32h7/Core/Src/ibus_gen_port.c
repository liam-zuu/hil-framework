/* ibus_gen_port.c — iBUS generator on UART4 TX (see ibus_gen_port.h)
 * The core (ibus_gen.c) decides when a frame starts; this file only moves
 * bytes into the UART4 TX FIFO (16 deep), which shifts them out as one
 * burst. Frames go out only while the SPI1 link is up, so the DUT loses
 * RC and turns its motors off when the RPi5 stops (spec S06 T12 (d)).
 */

#include "ibus_gen_port.h"

static USART_TypeDef *volatile _u;     /* NULL until init succeeded */
static ibus_gen_t _gen;

static int _tx_room(void)
{
    return (_u->ISR & USART_ISR_TXE_TXFNF) != 0u;
}

static void _tx_put(uint8_t b)
{
    _u->TDR = b;
}

int IbusPort_Init(USART_TypeDef *u)
{
    /* UART4 not in CubeMX yet: its clock is off. Check the clock before
     * touching the registers, then that MX_UART4_Init turned UE and TE on. */
    if (u != UART4 || !__HAL_RCC_UART4_IS_CLK_ENABLED())
        return 0;
    if ((u->CR1 & (USART_CR1_UE | USART_CR1_TE)) != (USART_CR1_UE | USART_CR1_TE))
        return 0;

    /* CubeMX leaves the FIFO off; FIFOEN can only change while UE = 0.
     * Done here, in a user file, so regenerating CubeMX code keeps it
     * (same as UartRelay_Init). Nothing has been sent yet. */
    u->CR1 &= ~USART_CR1_UE;
    u->CR1 |= USART_CR1_FIFOEN;
    u->CR1 |= USART_CR1_UE;

    static const ibus_gen_io_t io = { _tx_room, _tx_put };
    ibus_gen_init(&_gen, &io, IBUS_GEN_PERIOD_TICKS);
    _u = u;                            /* last: the TIM6 tick may already run */
    return 1;
}

void IbusPort_Tick(void)
{
    if (_u != NULL) ibus_gen_tick(&_gen);
}

void IbusPort_SetEnabled(int on)
{
    if (_u != NULL) ibus_gen_set_enabled(&_gen, on);
}

int IbusPort_ReadWindow(ibus_gen_window_t *out)
{
    return _u != NULL && ibus_gen_read_window(&_gen, out);
}
