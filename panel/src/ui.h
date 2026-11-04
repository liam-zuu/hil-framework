/* ui.h — screens of panel M2: health bar + NO LINK / HOME / RUN / RESULT / FAULT
 * (spec panel 20/06 §3, đề xuất panel §2 M2; texts ASCII, English like the thesis).
 * Draws from a COPY of the client and the app taken under the mutex (main.cpp), never from
 * the live ones the link task changes. */
#pragma once
#include <stdint.h>

#include "panel_app.h"
#include "panel_proto.h"

struct PanelView {
    const app_t *a;             /* copy; a->c points at the copied client */
    uint32_t now;
};

const char *ui_profile_name(const pp_client_t *c, int i);

/* returns a short line about where the full-screen sprite lives (PSRAM / internal / none) */
const char *ui_begin(void);
/* hash of what is on screen: redraw only when it changes (or once a second) */
uint32_t ui_signature(const PanelView &v);
void ui_draw(const PanelView &v);
