/* render_main.cpp - draws every screen of src/ui.cpp on the PC (fake TFT_eSPI in fake/) and
 * writes one binary PPM per case into the folder given as argv[1].
 * The client and app states are set by hand (as the link task would leave them); the
 * screen choice, result rules and texts are those of lib/panel_app and src/ui.cpp.
 * Numbers are made up for the picture (FAKE = what panel/tools/fake_pi.py sends). */
#include <assert.h>
#include <math.h>
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

static void figure_eight(uint16_t run, bool partial)
{
    const int samples = 161;  /* 8 s at 20 Hz, including both endpoints */
    const int chunks = (samples + PP_TRACK_POINTS_MAX - 1) / PP_TRACK_POINTS_MAX;
    const double pi = acos(-1.0);
    for (int k = 0; k < PP_TRACK_SERIES; k++) {
        for (int chunk = 0; chunk < chunks; chunk++) {
            if (partial && k == PP_TR_PLANT && chunk == 4) continue;
            pp_point_input_t points[PP_TRACK_POINTS_MAX];
            int n = 0;
            for (int i = chunk * PP_TRACK_POINTS_MAX; i < samples && n < PP_TRACK_POINTS_MAX; i++) {
                double t = i / 20.0, a = 2 * pi * t / 8;
                double x = (k == PP_TR_DUT ? 790 : 850) * sin(a);
                double y = (k == PP_TR_DUT ? 220 : 260) * sin(2 * a);
                if (k == PP_TR_PLANT && t >= 4 && t <= 6)
                    y += 220 * sin(pi * (t - 4) / 2);
                points[n++] = {(int32_t)lround(x), (int32_t)lround(y)};
            }
            uint8_t frame[PP_LEN_TRACK];
            size_t len = pp_encode_track(frame, C.sid, run, k, chunk, chunks, n, points);
            assert(len == sizeof frame);
            pp_client_on_frame(&C, frame, len, NOW);
        }
    }
}

/* Render optional wire frames on top of a waiting RESULT for the host tests. */
static int probe_result(const char *dir, const char *frames)
{
    base();
    C.st.run_state = PP_ST_END;
    C.st.outcome = PP_OUT_COMPLETED;
    FILE *f = fopen(frames, "rb");
    if (!f) { perror(frames); return 1; }
    pp_parser_t parser;
    pp_parser_init(&parser, PP_ACCEPT_PANEL, PP_PARSER_STALE_MS);
    uint8_t buf[PP_LEN_MAX];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) != 0)
        pp_parser_feed(&parser, buf, n, NOW, [](void *, const uint8_t *frame, size_t len) {
            pp_client_on_frame(&C, frame, len, NOW);
        }, nullptr);
    fclose(f);
    TFT_eSPI::on_text = [](const char *s) { printf("text %s\n", s); };
    shot(dir, "probe");
    printf("signature %u\n", (unsigned)ui_signature({&A, NOW}));
    return 0;
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : ".";
    PanelView v{&A, NOW};
    printf("%s\n", ui_begin());
    if (argc == 3) return probe_result(dir, argv[2]);

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

    base();
    C.has_extra = true;
    C.extra.sid = C.sid;
    C.extra.reset_index = 3;
    C.profile_count = C.profiles_have = 4;
    C.profile_ok[3] = true; strcpy(C.profile_name[3], "RESET DUT");
    C.extra.sha_state = PP_SHA_OK;
    strcpy(C.extra.dut_sha, "2da73b9e45c1");
    strcpy(C.extra.dut_env, "PID"); C.info.ctrl_src = PP_CTRL_SRC_FW;
    shot(dir, "13_home_sha_ok");

    C.extra.sha_state = PP_SHA_MISMATCH;
    C.st.links = PP_L_RPI;
    A.cursor = C.extra.reset_index;
    shot(dir, "14_home_sha_wrong_cursor_reset");

    base();
    C.st.run_state = PP_ST_RUN; C.st.phase_ds = 59;
    strcpy(C.profile_name[C.st.profile], "DEMO-ENC");
    C.has_extra = true; C.extra.sid = C.sid; C.extra.reset_index = PP_PROFILE_NONE;
    C.extra.flags = PP_X_FAULT_ACTIVE;
    strcpy(C.extra.fault_text, "ENC LOSS FL 2.1s left");
    shot(dir, "15_run_fault_banner");
    uint32_t fault_sig = ui_signature(v);
    C.extra.flags = 0;
    assert(ui_signature(v) != fault_sig);

    base(); C.sid = 0; C.has_st = false; C.has_status_t = false;
    A.proto_foreign_n = 1; A.proto_foreign_ver = 6;
    app_update(&A, NOW);
    shot(dir, "16_nolink_proto_mismatch");
    assert(ui_signature(v) != ui_signature({&A, NOW + PP_NO_LINK_MS}));

    base(); C.st.run_state = PP_ST_END; C.st.outcome = PP_OUT_COMPLETED;
    strcpy(C.profile_name[C.st.profile], "O1-F05");
    C.info.ctrl_src = PP_CTRL_SRC_FW;
    C.info.flags |= PP_I_RESULT_VALID; C.info.result_run = C.st.run_no;
    strcpy(C.info.result[0], "completed #3");
    strcpy(C.info.result[1], "e_max 220 e_end 0");
    strcpy(C.info.result[2], "p99 27 max 38us");
    C.has_extra = true; C.extra.flags = PP_X_CHECK_VALID;
    C.extra.check_run = C.st.run_no; C.extra.check_code = PP_CHK_CLEAN_NO_SPIFIX;
    figure_eight(C.st.run_no, false);
    shot(dir, "17_result_track_clean");

    C.extra.check_code = PP_CHK_BAD;
    strcpy(C.extra.check_text, "pos_reset +1");
    shot(dir, "18_result_track_bad");

    memset(C.tracks, 0, sizeof C.tracks);
    figure_eight(C.st.run_no, true);
    C.extra.check_code = PP_CHK_RUNNING;
    shot(dir, "19_result_track_partial");

    C.st.run_no++;
    C.extra.check_code = PP_CHK_CLEAN;
    shot(dir, "20_result_other_run_track");

    printf("missing_glyphs %d\n", TFT_eSPI::missing_glyphs());

    /* every frame must start from a clean screen (bench 05/11: the right third kept old
     * pixels): NO LINK drawn after RUN must equal NO LINK drawn after FAULT */
    static uint16_t after_run[320][480];
    base(); C.st.run_state = PP_ST_RUN; C.st.phase_ds = 63; ui_draw(v);
    base(); C.sid = 0; C.has_st = false; C.has_status_t = false; ui_draw(v);
    memcpy(after_run, ui_render_screen()->buf, sizeof after_run);
    base(); C.st.run_state = PP_ST_ERROR; C.st.error_code = 1; ui_draw(v);
    base(); C.sid = 0; C.has_st = false; C.has_status_t = false; ui_draw(v);
    int stale = 0;
    for (int y = 0; y < 320; y++)
        for (int x = 0; x < 480; x++)
            stale += after_run[y][x] != ui_render_screen()->buf[y][x];
    printf("stale_pixels %d\n", stale);
    return 0;
}
