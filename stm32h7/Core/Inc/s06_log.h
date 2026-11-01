#ifndef S06_LOG_H
#define S06_LOG_H

#include <stdint.h>

/*
 * S06 measurement log of the H7 (spec S06 v3.4 §4.3 rule 6; design
 * 03_CODE/docs/2026-10-29_Thiet_ke_H7_S06.md in the thesis repo).
 *
 * Records are pushed from interrupts (TIM6, EXTI3, EXTI4 at priority 6,
 * SPI1 at priority 0) and from the main loop, and leave the H7 one per MISO
 * frame (spi1_handler.c, bytes 22-39). Every record gets the next value of a
 * continuous 16-bit index, also when the ring is full and the record is
 * dropped, so the RPi5 sees a loss as a gap; the drop count rides in the
 * TICK record.
 *
 * Multi-byte payload fields are little-endian.
 */

#define S06_LOG_PAYLOAD   15u
#define S06_LOG_SLOTS     512u      /* power of two */

/* Record types (MISO byte 22). 0 = no record in this frame. */
#define S06_REC_NONE        0x00u
#define S06_REC_REPORT      0x10u   /* + sensor index: 0x10 accel, 0x11 gyro, 0x12 game RV */
#define S06_REC_PKT_READ    0x20u   /* host finished reading an input-report packet (Y14) */
#define S06_REC_RESET_DONE  0x30u   /* reset-complete sent, epoch + 1 (V3-2, V4-1)        */
#define S06_REC_HOLD_ON     0x31u   /* IMU loss injection starts (V3-1)                   */
#define S06_REC_HOLD_OFF    0x32u   /* IMU loss injection ends                            */
#define S06_REC_RST_LOW     0x33u   /* RST pin (PG3) falling edge                         */
#define S06_REC_RST_HIGH    0x34u   /* RST pin rising edge                                */
#define S06_REC_ENC_START   0x35u   /* EncoderGen _start_wheel (T3b Y5, V2-6)            */
#define S06_REC_ENC_STOP    0x36u   /* EncoderGen _stop                                   */
#define S06_REC_TICK        0x40u   /* CYCCNT + 64-bit cycle total + drop count (L7-4)    */
#define S06_REC_REARM       0x50u   /* SPI1 DMA restarted                                 */

/* REPORT flags */
#define S06_RPT_QUEUED      0x01u   /* packet with this report entered the output queue  */
#define S06_RPT_QFULL       0x02u   /* packet dropped: output queue full (bno_emu.c q_push) */

typedef struct {
    uint8_t  type;
    uint16_t idx;
    uint8_t  payload[S06_LOG_PAYLOAD];
} S06_Rec_t;

void     S06Log_Init(void);
/* Push one record (payload shorter than 15 bytes is zero-padded). Safe from
 * any interrupt priority: the ring update runs with interrupts masked. */
void     S06Log_Push(uint8_t type, const uint8_t *payload, uint8_t len);
/* Oldest record into *out: 1, or 0 when the ring is empty. Main loop only. */
uint8_t  S06Log_Pop(S06_Rec_t *out);
uint32_t S06Log_Dropped(void);       /* records lost because the ring was full */
uint32_t S06Log_Pushed(void);        /* records pushed (kept + dropped)        */

/* CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final XOR.
 * Same as Python binascii.crc_hqx(data, 0xFFFF); "123456789" -> 0x29B1. */
uint16_t S06_Crc16(uint16_t crc, const uint8_t *data, uint32_t len);

/* Little-endian writers for record payloads and MISO fields */
static inline void s06_put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static inline void s06_put32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

#endif /* S06_LOG_H */
