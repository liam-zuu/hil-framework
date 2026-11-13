/* buzzer.cpp - see buzzer.h. Arduino-ESP32 core 2.0.17 LEDC (esp32-hal-ledc.c):
 *  - ledcWriteTone(ch, hz) sets the timer to hz, 10-bit, duty 0x1FF (50 %);
 *  - ledcWrite(ch, 1023) at 10 bits is turned into "full on" by the core = pin constantly
 *    HIGH = the low-level-trigger module off. ledcWriteTone(ch, 0) would write duty 0 = pin
 *    LOW = module on with no wave (current, no sound), so it is never used.
 * The pin is set HIGH as a GPIO before LEDC takes it, so the buzzer does not click at boot. */
#include "buzzer.h"

#include <Arduino.h>

#include "pins.h"

#define BUZZER_CH       6          /* LEDC channel 6 (timer 3); nothing else here uses LEDC */
#define BUZZER_BITS     10
#define BUZZER_SILENT   ((1u << BUZZER_BITS) - 1u)

static uint16_t g_hz = 0;

void buzzer_begin(void)
{
    pinMode(PIN_BUZZER, OUTPUT);
    digitalWrite(PIN_BUZZER, HIGH);
    ledcSetup(BUZZER_CH, 2000, BUZZER_BITS);
    ledcWrite(BUZZER_CH, BUZZER_SILENT);
    ledcAttachPin(PIN_BUZZER, BUZZER_CH);
    ledcWrite(BUZZER_CH, BUZZER_SILENT);
    g_hz = 0;
}

void buzzer_out(uint16_t hz)
{
    if (hz == g_hz)
        return;
    g_hz = hz;
    if (hz)
        ledcWriteTone(BUZZER_CH, hz);
    else
        ledcWrite(BUZZER_CH, BUZZER_SILENT);
}
