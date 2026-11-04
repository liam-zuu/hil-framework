/*
 * pins.h — panel board pins (ESP32-S3 N16R8, WeAct board; Liam 04/11, PANEL session).
 *
 * [ĐỀ XUẤT] Claude's proposal, NOT checked on the board and NOT yet confirmed by Liam.
 * Rules used: no strapping pins (0, 3, 45, 46); 35-37 belong to the octal PSRAM; 19/20 are
 * the USB pins; 43/44 left for UART0 (boot log); 48 left free (some S3 boards put an RGB LED
 * there). Change the numbers here and in platformio.ini (TFT_* flags) together.
 */
#pragma once

/* ST7796 (SPI, 480x320 landscape); TFT_eSPI takes its pins from the TFT_* build flags in
 * platformio.ini; repeated here only for the wiring guide */
#define PIN_TFT_SCLK   12
#define PIN_TFT_MOSI   11
#define PIN_TFT_CS     10
#define PIN_TFT_DC      9
#define PIN_TFT_RST     8
#define PIN_TFT_BL      7      /* backlight: through the module's LED pin; HIGH = on */

/* buttons: to GND when pressed, internal pull-up (+ 10-100 nF to GND per button) */
#define PIN_BTN_NAV     4
#define PIN_BTN_SELECT  5
#define PIN_BTN_START   6
#define PIN_BTN_STOP   15

/* LEDs of the illuminated START / STOP buttons: HIGH = on (through a resistor) */
#define PIN_LED_START  16
#define PIN_LED_STOP   17

/* passive buzzer: reserved, not driven in M2 */
#define PIN_BUZZER     18

/* UART1 to the RPi5 (Pi pin 8 GPIO14 TX -> panel RX, Pi pin 10 GPIO15 RX <- panel TX, GND) */
#define PIN_PI_TX       1      /* panel TX -> Pi RX (pin 10) */
#define PIN_PI_RX       2      /* panel RX <- Pi TX (pin 8) */
#define PI_BAUD    115200
