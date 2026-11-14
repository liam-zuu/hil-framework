/* Arduino.h - PC stand-in for the calls of src/ui.cpp (test_host/ui_render). */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define OUTPUT 1
#define HIGH 1
static inline void pinMode(int, int) {}
static inline void digitalWrite(int, int) {}
