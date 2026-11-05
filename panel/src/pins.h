/*
 * pins.h — panel board pins (ESP32-S3 N16R8, WeAct board; Liam 04/11, PANEL session).
 *
 * [ĐỀ XUẤT] Claude's proposal, not yet run on the board. Checked 04/11 against the WeAct
 * ESP32-S3-A schematic (WeAct_ESP32_S3_A_Sch.pdf): every pin below is on a header and has
 * nothing else on the board. Avoided: 0 (BOOT), 3, 46 (strapping), 45 (strapping + KEY
 * button with 10k pull-up), 48 (WS2812 RGB LED), 35-37 (octal PSRAM of N16R8), 19/20 ("USB"
 * port), 43/44 (UART0 = CH343P = "UART" port, used by env panel_s3_bench).
 * Change the numbers here and in platformio.ini (TFT_* flags) together.
 */
#pragma once

/* ST7796 (SPI, 480x320 landscape); TFT_eSPI takes its pins from the TFT_* build flags in
 * platformio.ini; repeated here only for the wiring guide */
#define PIN_TFT_SCLK   12
#define PIN_TFT_MOSI   11
#define PIN_TFT_CS     10
#define PIN_TFT_DC      9
#define PIN_TFT_RST     8
#define PIN_TFT_BL      7      /* driven HIGH; the wiring guide ties the module's LED pin to
                                  3V3 instead (backlight current not taken from a GPIO) */

/* buttons: to GND when pressed, internal pull-up (+ 10-100 nF to GND per button) */
#define PIN_BTN_NAV     4
#define PIN_BTN_SELECT  5
#define PIN_BTN_START   6
#define PIN_BTN_STOP   15

/* LEDs of the illuminated START / STOP buttons: HIGH = on (through a resistor) */
#define PIN_LED_START  16
#define PIN_LED_STOP   17

/* passive buzzer on a low-level-trigger module (Liam 05/11): I/O LOW = module on; the pin
 * carries a square wave while a note plays and stays HIGH when silent (src/buzzer.cpp) */
#define PIN_BUZZER     18

/* UART1 to the RPi5 (Pi pin 8 GPIO14 TX -> panel RX, Pi pin 10 GPIO15 RX <- panel TX, GND) */
#define PIN_PI_TX       1      /* panel TX -> Pi RX (pin 10) */
#define PIN_PI_RX       2      /* panel RX <- Pi TX (pin 8) */
#define PI_BAUD    115200
