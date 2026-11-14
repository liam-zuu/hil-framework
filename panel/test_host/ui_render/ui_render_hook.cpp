/* ui_render_hook.cpp - builds src/ui.cpp unchanged in this translation unit and hands the
 * screen buffer to render_main.cpp (g_tft is in ui.cpp's anonymous namespace). */
#include "../../src/ui.cpp"

TFT_eSPI *ui_render_screen(void)
{
    return &g_tft;
}
