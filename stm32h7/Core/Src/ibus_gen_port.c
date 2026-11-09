/* ibus_gen_port.c — iBUS generator on UART4 TX (see ibus_gen_port.h)
 * The core (ibus_gen.c) decides when a frame starts; this file only moves
 * bytes into the UART4 TX FIFO (16 deep), which shifts them out as one
 * burst. Frames go out only while the SPI1 link is up, so the DUT loses
 * RC and turns its motors off when the RPi5 stops (spec S06 T12 (d)).
 */

#include "ibus_gen_port.h"
#include "spi1_handler.h"
#include "encoder_gen.h"        /* ENC_CLOCK_CAL */

#define _BARRIER()  __asm volatile ("" ::: "memory")

static USART_TypeDef *volatile _u;     /* NULL until init succeeded */
static volatile uint8_t _gate;
static volatile uint8_t _swa;          /* I8: SWA asked by the RPi5 (MOSI byte 20 bit 3) */
static volatile uint8_t _tmo;          /* I8-R2-1: link timeout seen by the tick, latched until taken */
static ibus_gen_t _gen;

static int _tx_room(void)
{
    return (_u->ISR & USART_ISR_TXE_TXFNF) != 0u;
}

static int _tx_idle(void)
{
    return (_u->ISR & USART_ISR_TC) != 0u;    /* FIFO empty, last stop bit sent */
}

static void _tx_put(uint8_t b)
{
    _u->TDR = b;
}

/* Microseconds from the DWT cycle counter, corrected like EncoderGen and
 * the BNO085 emulator: one real millisecond is SystemCoreClock / 1000 *
 * ENC_CLOCK_CAL cycles. ENC_CLOCK_CAL is 1.0 since 09/11 (HSE 5 MHz from the
 * ST-LINK MCO HSE/5, bug H7-CLK-HSE fixed at the source); it was 1.0064 while
 * the H7 ran 0.64 % fast on the ST-LINK HSI/2 default.
 * Called only from the TIM6 tick (and init), so no locking. */
static uint32_t _cyc_last;
static uint64_t _cyc_total;
static uint32_t _cyc_per_ms;

static uint32_t _now_us(void)
{
    uint32_t c = DWT->CYCCNT;
    _cyc_total += (uint32_t)(c - _cyc_last);
    _cyc_last = c;
    uint64_t ms = _cyc_total / _cyc_per_ms;
    uint64_t rem = _cyc_total % _cyc_per_ms;
    return (uint32_t)(ms * 1000u + (rem * 1000u) / _cyc_per_ms);
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

    /* DWT is already running (BnoPort_Init); enabling again is harmless */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    _cyc_last = DWT->CYCCNT;
    _cyc_total = 0;
    _cyc_per_ms = (uint32_t)((float)SystemCoreClock / 1000.0f * ENC_CLOCK_CAL + 0.5f);

    static const ibus_gen_io_t io = { _tx_room, _tx_idle, _tx_put };
    ibus_gen_init(&_gen, &io, IBUS_GEN_PERIOD_US, _now_us());
    _BARRIER();                        /* _gen complete before it is published (Codex I5) */
    _u = u;                            /* last: the TIM6 tick may already run */
    return 1;
}

void IbusPort_Tick(void)
{
    if (_u == NULL) return;
    _BARRIER();                        /* no _gen access before the _u check */
    /* Read the frame time before the clock: the SPI1 callback (higher
     * priority) may update it in between, which only makes the age larger. */
    uint32_t last_ms = SPI1_GetLastFrameMs();
    uint32_t age_ms = HAL_GetTick() - last_ms;
    int on = _gate && age_ms <= IBUS_LINK_TIMEOUT_MS;
    /* I8-4: a link timeout seen here drops the SWA command at once, so a frame
     * that arrives before the main loop ran its link-lost branch cannot start
     * an iBUS frame with the SWA of the old link period. The main loop sets it
     * again from the new frames, after its filter. I8-R2-1: the same timeout
     * is latched in _tmo, so the main loop restarts its filter before the
     * first frame after it (IbusPort_TakeTimeout); a short gap without a
     * timeout no longer touches the filter. */
    if (age_ms > IBUS_LINK_TIMEOUT_MS) {
        _swa = 0u;
        _tmo = 1u;
    }
    ibus_gen_tick(&_gen, _now_us(), on, _swa);
}

void IbusPort_SetEnabled(int on)
{
    _gate = on ? 1u : 0u;
}

void IbusPort_SetSwa(int on)
{
    _swa = on ? 1u : 0u;
}

int IbusPort_TakeTimeout(void)
{
    /* Main loop only. The tick (higher priority) can set _tmo again between
     * the read and the clear: that set is for the same timeout, already
     * reported by this call, so losing it is harmless. */
    if (!_tmo) return 0;
    _tmo = 0u;
    return 1;
}

int IbusPort_LastSwa(void)
{
    return _u != NULL && ibus_gen_last_swa(&_gen);
}

int IbusPort_ReadWindow(ibus_gen_window_t *out)
{
    return _u != NULL && ibus_gen_read_window(&_gen, out);
}
