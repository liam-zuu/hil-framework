/* s06_log.c — S06 measurement log ring of the H7 (see s06_log.h) */
#include "s06_log.h"
#include "stm32h7xx_hal.h"
#include <string.h>

static S06_Rec_t         _ring[S06_LOG_SLOTS];
static volatile uint32_t _head;      /* next slot to write (pushed records kept) */
static volatile uint32_t _tail;      /* next slot to read                         */
static volatile uint32_t _pushed;    /* every push, kept or dropped               */
static volatile uint32_t _dropped;
static volatile uint16_t _next_idx;

static void _crc_init(void);

void S06Log_Init(void)
{
    _crc_init();
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    _head = _tail = 0;
    _pushed = _dropped = 0;
    _next_idx = 0;
    __set_PRIMASK(primask);
}

void S06Log_Push(uint8_t type, const uint8_t *payload, uint8_t len)
{
    if (len > S06_LOG_PAYLOAD) len = S06_LOG_PAYLOAD;
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    uint16_t idx = _next_idx++;
    _pushed++;
    if (_head - _tail >= S06_LOG_SLOTS) {
        _dropped++;                      /* the index gap tells the RPi5 */
    } else {
        S06_Rec_t *r = &_ring[_head & (S06_LOG_SLOTS - 1u)];
        r->type = type;
        r->idx = idx;
        memcpy(r->payload, payload, len);
        memset(r->payload + len, 0, S06_LOG_PAYLOAD - len);
        _head++;
    }
    __set_PRIMASK(primask);
}

uint8_t S06Log_Pop(S06_Rec_t *out)
{
    uint8_t got = 0;
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    if (_tail != _head) {
        *out = _ring[_tail & (S06_LOG_SLOTS - 1u)];
        _tail++;
        got = 1;
    }
    __set_PRIMASK(primask);
    return got;
}

uint32_t S06Log_Dropped(void) { return _dropped; }
uint32_t S06Log_Pushed(void)  { return _pushed; }

/* Table-driven: about 80 bytes per frame in the main loop (MOSI + MISO), so
 * the bit loop (8 steps per byte) would cost several us per frame. */
static uint16_t _crc_tab[256];
static uint8_t  _crc_tab_ok;

static void _crc_init(void)
{
    for (unsigned i = 0; i < 256u; i++) {
        uint16_t c = (uint16_t)(i << 8);
        for (int b = 0; b < 8; b++)
            c = (c & 0x8000u) ? (uint16_t)((c << 1) ^ 0x1021u) : (uint16_t)(c << 1);
        _crc_tab[i] = c;
    }
    _crc_tab_ok = 1;
}

uint16_t S06_Crc16(uint16_t crc, const uint8_t *data, uint32_t len)
{
    if (!_crc_tab_ok) _crc_init();   /* S06Log_Init builds it at boot */
    while (len--)
        crc = (uint16_t)((crc << 8) ^ _crc_tab[(uint8_t)((crc >> 8) ^ *data++)]);
    return crc;
}
