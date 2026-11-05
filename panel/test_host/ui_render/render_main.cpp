/* render_main.cpp - draws every screen of src/ui.cpp on the PC (fake TFT_eSPI in fake/) and
 * writes one binary PPM per case into the folder given as argv[1].
 * The client and app states are set by hand (as the link task would leave them); the
 * screen choice, result rules and texts are those of lib/panel_app and src/ui.cpp.
 * Numbers are made up for the picture (FAKE = what panel/tools/fake_pi.py sends). */
#include <stdio.h>
#include <string.h>

#include "TFT_eSPI.h"
#include "panel_app.h"
#include "panel_proto.h"
#include "ui.h"

extern TFT_eSPI *ui_render_screen(void);   /* ui_render_hook.cpp */

static uint16_t nonce_cb(void *) { return 1; }
static void send_cb(void *, const uint8_t *, size_t) {}

static pp_client_t C;
static app_t A;
static const uint32_t NOW = 100000;

static void base(void)
{
    pp_client_init(&C, nonce_cb, nullptr, send_cb, nullptr);
    app_init(&A, &C);
    C.sid = 0x1234;
    C.has_status_t = true;
    C.last_status_ms = NOW - 20;
    C.has_st = true;
    memset(&C.st, 0, sizeof C.st);
    C.st.links = PP_L_RPI | PP_L_H7;
    C.st.profile = 1;
    C.st.run_no = 3;
    C.st.data_age_ms = 20;
    C.has_info = true;
    memset(&C.info, 0, sizeof C.info);
    C.info.flags = PP_I_LOOP_VALID;
    C.info.loop_us = 27;
    C.info.ctrl_src = PP_CTRL_SRC_PROFILE;
    strcpy(C.info.ctrl, "PID");
    C.has_profile_count = true;
    C.profile_count = 3;
    const char *names[] = {"FAKE-1", "FAKE-2", "FAKE-3"};
    for (int i = 0; i < 3; i++) {
        C.profile_ok[i] = true;
        strcpy(C.profile_name[i], names[i]);
    }
    C.profiles_have = 3;
    A.cursor = 1;
    A.msg[0] = 0;
}

static void shot(const char *dir, const char *name)
{
    PanelView v{&A, NOW};
    ui_draw(v);
    TFT_eSPI *t = ui_render_screen();
    char path[512];
    snprintf(path, sizeof path, "%s/%s.ppm", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return; }
    fprintf(f, "P6\n480 320\n255\n");
    for (int y = 0; y < 320; y++)
        for (int x = 0; x < 480; x++) {
            uint16_t c = t->buf[y][x];
            unsigned char px[3] = {(unsigned char)(((c >> 11) & 0x1F) * 255 / 31),
                                   (unsigned char)(((c >> 5) & 0x3F) * 255 / 63),
                                   (unsigned char)((c & 0x1F) * 255 / 31)};
            fwrite(px, 1, 3, f);
        }
    fclose(f);
    printf("%s\n", name);
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : ".";
    printf("%s\n", ui_begin());

    base(); C.sid = 0; C.has_st = false; C.has_status_t = false;
    shot(dir, "01_nolink_waiting");

    base(); C.last_status_ms = NOW - 2000; C.stop.active = true;
    shot(dir, "02_nolink_lost_stop_pending");

    base(); strcpy(A.msg, "SELECT: OK");
    shot(dir, "03_home_ready");

    base();
    C.st.links = PP_L_RPI;                              /* no fresh H7 answers: NOT READY */
    C.info.ctrl_src = PP_CTRL_SRC_NONE; C.info.ctrl[0] = 0;
    C.info.flags = 0;
    C.profile_count = 6;
    const char *more[] = {"A-K1 step 0.3", "A-K2 steps", "A-L2 line 0.5 m/s", "A-F1 figure-8 slow",
                          "A-F2 figure-8 fast", "A-S1 strafe 0.3"};
    for (int i = 0; i < 6; i++) { C.profile_ok[i] = true; strcpy(C.profile_name[i], more[i]); }
    C.profiles_have = 6; A.cursor = 4; C.st.profile = 2;
    strcpy(A.msg, "NOT READY: START not sent");
    shot(dir, "04_home_not_ready_long_list");

    base(); C.st.run_state = PP_ST_RUN; C.st.phase_ds = 63; C.st.data_age_ms = 23;
    strcpy(A.msg, "START: OK");
    shot(dir, "05_run");

    base(); C.st.run_state = PP_ST_PREPARE; C.st.phase_ds = 3; C.st.lost = 12;
    C.st.quality = PP_Q_LOST | PP_Q_SP_STALE | PP_Q_SPI_STALE | PP_Q_DUT_UNCONF | PP_Q_DATA_INVALID;
    C.st.data_age_ms = 0xFFFF; C.info.ctrl_src = PP_CTRL_SRC_FW; strcpy(C.info.ctrl, "LADRC");
    shot(dir, "06_prepare_all_quality_flags");

    base(); C.st.run_state = PP_ST_STOPPING; C.st.phase_ds = 2; C.stop_state = PP_STOP_ACKED;
    strcpy(A.msg, "STOP: OK");
    shot(dir, "07_stopping");

    base(); C.st.run_state = PP_ST_END; C.st.outcome = PP_OUT_COMPLETED;
    C.info.flags |= PP_I_RESULT_VALID; C.info.result_run = 3;
    strcpy(C.info.result[0], "FAKE e_vx 0.042 m/s");
    strcpy(C.info.result[1], "FAKE e_wz 0.17 rad/s");
    strcpy(C.info.result[2], "FAKE lost 0");
    shot(dir, "08_result_completed");

    base(); C.st.run_state = PP_ST_END; C.st.outcome = PP_OUT_COMPLETED;
    shot(dir, "09_result_waiting");

    base(); C.st.run_state = PP_ST_END; C.st.outcome = PP_OUT_USER_STOP;
    shot(dir, "10_result_user_stop");

    base(); C.st.run_state = PP_ST_ERROR; C.st.error_code = 1;
    strcpy(C.info.error_text, "FAKE fault (bench)");
    shot(dir, "11_fault");

    base(); C.st.run_state = PP_ST_ERROR; C.st.error_code = 12;
    C.st.links = PP_L_RPI | PP_L_H7 | PP_L_ESP32_KNOWN;  /* ESP32 known and down (future H7 bit) */
    strcpy(A.msg, "STOP: FAILED (effect unknown)");
    shot(dir, "12_fault_no_text_esp32_down");
    printf("missing_glyphs %d\n", TFT_eSPI::missing_glyphs());
    return 0;
}
