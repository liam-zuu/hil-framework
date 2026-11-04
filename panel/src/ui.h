/* ui.h — screens of panel M2: health bar + NO LINK / HOME / RUN / RESULT / FAULT
 * (spec panel 20/06 §3, đề xuất panel §2 M2; texts ASCII, English like the thesis). */
#pragma once
#include <stdint.h>

#include "panel_proto.h"

enum Screen { SCR_NOLINK, SCR_HOME, SCR_RUN, SCR_RESULT, SCR_FAULT };

struct PanelView {
    const pp_client_t *c;
    uint32_t now;
    uint8_t  cursor;            /* profile index under the cursor (HOME) */
    bool     result_dismissed;  /* STOP/NAV on RESULT went back to HOME */
    char     msg[48];           /* feedback of the last command */
};

Screen ui_screen(const PanelView &v);
bool ui_ready(const PanelView &v);              /* START allowed: link + H7 answering */
const char *ui_profile_name(const pp_client_t *c, int i);

void ui_begin(void);
/* hash of what is on screen: redraw only when it changes (or once a second) */
uint32_t ui_signature(const PanelView &v);
void ui_draw(const PanelView &v);
