/* esp_heap_caps.h - PC stand-in (test_host/ui_render): pretends the sprite went to PSRAM. */
#pragma once
#include <stddef.h>
#define MALLOC_CAP_SPIRAM 1
static inline size_t heap_caps_get_free_size(int)
{
    static size_t free_ps = 8u << 20;
    size_t v = free_ps;
    free_ps -= 480u * 320u * 2u;
    return v;
}
