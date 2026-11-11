/* Host test of AGV firmware src/controller/gain_set.c (L90, gain over WiFi, Liam 12/11).
 * Built by test_gain_set.py with the fwsim firmware sources + shim (no ESP-IDF).
 * Prints one line per check: "PASS <name>" or "FAIL <name>: <detail>"; exit code = number of FAILs.
 * Checks: packet parsing, safety ranges, controller match, that apply writes the same fields as
 * fwsim (fw_sim.c fwsim_set_pid / fwsim_set_adrc, tune_ga.gains_for) and keeps the fixed ones,
 * read-back and %.9g round trip, failed requests change nothing. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "gain_set.h"
#include "vel_ctrl.h"
#include "pid.h"
#include "adrc.h"
#include "heading.h"
#include "pos_ctrl.h"

static int n_fail;

static void check(int ok, const char *name, const char *detail) {
    if (ok) printf("PASS %s\n", name);
    else { printf("FAIL %s: %s\n", name, detail); n_fail++; }
}

static int parse(const char *s, gain_set_t *g) { return gain_set_parse(s, strlen(s), g); }

static const char *PID_OK =
    "#gain_set,v=1,id=7,ctrl=PID,kp_xy=1.25,ki_xy=0.5,kp_wz=2,ki_wz=0.75,pos_kp=4,pos_ki=1.5,k_psi=0.8";
static const char *ADRC_OK =
    "#gain_set,v=1,id=9,ctrl=ADRC,omega_c_xy=10,omega_o_xy=35,omega_c_wz=12,omega_o_wz=40,pos_kp=5,pos_ki=2,k_psi=1.2";

static void expect_err(const char *name, const char *pkt, int want) {
    gain_set_t g;
    int r = parse(pkt, &g);
    if (r == GAIN_OK) r = gain_set_check_range(&g);
    char d[64];
    snprintf(d, sizeof d, "got %s, want %s", gain_err_str(r), gain_err_str(want));
    check(r == want, name, d);
}

int main(void) {
    gain_set_t g;
    char d[256];

    /* ── defaults after init: id 0, values of pid.c / heading.h / pos_ctrl.h ── */
    vel_ctrl_init();
    vel_ctrl_set_mode(CTRL_PID);
    gain_set_current(&g);
    check(g.id == 0 && g.ctrl == CTRL_PID && g.v[0] == 1.5f && g.v[1] == 0.05f && g.v[2] == 1.0f &&
          g.v[3] == 0.05f && g.v[4] == 6.0f && g.v[5] == 3.0f && g.v[6] == 1.0f,
          "defaults_pid", "current() after init differs from pid.c / pos_ctrl.h / heading.h defaults");
    char line[320];
    gain_set_format(line, sizeof line, &g, gain_set_applied_n());
    check(strcmp(line, ",id=0,applied_n=0,ctrl=PID,kp_xy=1.5,ki_xy=0.0500000007,kp_wz=1,ki_wz=0.0500000007,"
                       "pos_kp=6,pos_ki=3,k_psi=1") == 0, "format_defaults", line);

    /* ── parse ── */
    int r = parse(PID_OK, &g);
    check(r == GAIN_OK && g.id == 7 && g.ctrl == CTRL_PID && g.v[0] == 1.25f && g.v[3] == 0.75f && g.v[6] == 0.8f,
          "parse_pid", gain_err_str(r));
    /* order of the gain pairs is free; CR/LF at the end is accepted */
    r = parse("#gain_set,ctrl=PID,k_psi=0.8,pos_ki=1.5,pos_kp=4,ki_wz=0.75,kp_wz=2,ki_xy=0.5,kp_xy=1.25,id=7,v=1\r\n", &g);
    check(r == GAIN_OK && g.v[0] == 1.25f && g.v[6] == 0.8f, "parse_any_order_crlf", gain_err_str(r));
    r = parse(ADRC_OK, &g);
    check(r == GAIN_OK && g.ctrl == CTRL_ADRC && g.v[1] == 35.0f, "parse_adrc", gain_err_str(r));

    expect_err("err_header",        "#gain_sett,v=1,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_v2",            "#gain_set,v=2,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_no_v",          "#gain_set,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_id0",           "#gain_set,v=1,id=0,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_id_big",        "#gain_set,v=1,id=2147483648,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_id_sign",       "#gain_set,v=1,id=+7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_id_twice",      "#gain_set,v=1,id=7,id=8,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_ctrl",          "#gain_set,v=1,id=7,ctrl=pid,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_missing_gain",  "#gain_set,v=1,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0", GAIN_ERR_FORMAT);
    expect_err("err_dup_gain",      "#gain_set,v=1,id=7,ctrl=PID,kp_xy=1,kp_xy=1,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_extra_gain",    "#gain_set,v=1,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1,kd_xy=0", GAIN_ERR_FORMAT);
    expect_err("err_adrc_names_pid","#gain_set,v=1,id=7,ctrl=PID,omega_c_xy=10,omega_o_xy=35,omega_c_wz=12,omega_o_wz=40,pos_kp=5,pos_ki=2,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_value_junk",    "#gain_set,v=1,id=7,ctrl=PID,kp_xy=1.5x,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_value_empty",   "#gain_set,v=1,id=7,ctrl=PID,kp_xy=,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_empty_field",   "#gain_set,v=1,id=7,ctrl=PID,,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_no_equal",      "#gain_set,v=1,id=7,ctrl=PID,kp_xy,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_space",         "#gain_set,v=1,id=7,ctrl=PID,kp_xy=1 ,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);

    expect_err("range_nan",         "#gain_set,v=1,id=7,ctrl=PID,kp_xy=nan,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_RANGE);
    expect_err("range_inf",         "#gain_set,v=1,id=7,ctrl=PID,kp_xy=inf,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_RANGE);
    expect_err("range_negative",    "#gain_set,v=1,id=7,ctrl=PID,kp_xy=1,ki_xy=-0.1,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_RANGE);
    expect_err("range_pid_kp_max",  "#gain_set,v=1,id=7,ctrl=PID,kp_xy=20.5,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_RANGE);
    expect_err("range_pid_at_max",  "#gain_set,v=1,id=7,ctrl=PID,kp_xy=20,ki_xy=50,kp_wz=20,ki_wz=50,pos_kp=30,pos_ki=30,k_psi=10", GAIN_OK);
    expect_err("range_k_psi_max",   "#gain_set,v=1,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=10.5", GAIN_ERR_RANGE);
    expect_err("range_adrc_wo_49",  "#gain_set,v=1,id=9,ctrl=ADRC,omega_c_xy=10,omega_o_xy=49,omega_c_wz=12,omega_o_wz=40,pos_kp=5,pos_ki=2,k_psi=1", GAIN_OK);
    expect_err("range_adrc_wo_50",  "#gain_set,v=1,id=9,ctrl=ADRC,omega_c_xy=10,omega_o_xy=49.5,omega_c_wz=12,omega_o_wz=40,pos_kp=5,pos_ki=2,k_psi=1", GAIN_ERR_RANGE);
    expect_err("range_adrc_wc_low", "#gain_set,v=1,id=9,ctrl=ADRC,omega_c_xy=0.4,omega_o_xy=35,omega_c_wz=12,omega_o_wz=40,pos_kp=5,pos_ki=2,k_psi=1", GAIN_ERR_RANGE);

    char longpkt[400];
    memset(longpkt, '0', sizeof longpkt - 1);
    longpkt[sizeof longpkt - 1] = '\0';
    memcpy(longpkt, PID_OK, strlen(PID_OK));
    expect_err("err_too_long", longpkt, GAIN_ERR_FORMAT);
    char nul[200];
    size_t L = strlen(PID_OK);
    memcpy(nul, PID_OK, L);
    nul[20] = '\0';
    check(gain_set_parse(nul, L, &g) == GAIN_ERR_FORMAT, "err_embedded_nul", "packet with a 0 byte accepted");

    /* ── apply PID: same fields as fwsim_set_pid with tune_ga.gains_for, fixed ones kept ── */
    pid_controller_t *pc = vel_ctrl_pid();
    pc->vx.integral = 0.3f; pc->wz.integral = 0.2f;
    parse(PID_OK, &g);
    r = gain_set_apply(&g);
    check(r == GAIN_OK, "apply_pid_ok", gain_err_str(r));
    check(pc->vx.kp == 1.25f && pc->vy.kp == 1.25f && pc->vx.ki == 0.5f && pc->vy.ki == 0.5f &&
          pc->wz.kp == 2.0f && pc->wz.ki == 0.75f, "apply_pid_values", "PID gains not as sent");
    check(pc->vx.kd == 0.0f && pc->vy.kd == 0.0f && pc->wz.kd == 0.0f &&
          pc->vx.integral_limit == 0.5f && pc->vx.output_limit == 1.5f &&
          pc->vy.integral_limit == 0.5f && pc->vy.output_limit == 1.5f &&
          pc->wz.integral_limit == 1.0f && pc->wz.output_limit == 2.0f,
          "apply_pid_fixed_as_gains_for", "kd or limits changed (tune_ga.gains_for: kd 0, 0.5/1.5, 1.0/2.0)");
    check(pc->vx.integral == 0.0f && pc->wz.integral == 0.0f, "apply_resets_integral", "integral kept");
    check(pos_params()->kp == 4.0f && pos_params()->ki == 1.5f && heading_params()->k_psi == 0.8f,
          "apply_outer", "pos_kp / pos_ki / k_psi not as sent");
    gain_set_t cur;
    gain_set_current(&cur);
    check(cur.id == 7 && gain_set_applied_n() == 1 && memcmp(cur.v, g.v, sizeof g.v) == 0,
          "readback_pid", "current() differs from the applied set");

    /* %.9g round trip: what the firmware reports parses back to the same bits */
    gain_set_format(line, sizeof line, &cur, gain_set_applied_n());
    char pkt[400];
    const char *p = strstr(line, ",ctrl=");
    snprintf(pkt, sizeof pkt, "#gain_set,v=1,id=7%s", p);
    gain_set_t back;
    r = parse(pkt, &back);
    check(r == GAIN_OK && memcmp(back.v, cur.v, sizeof cur.v) == 0, "roundtrip_9g", pkt);

    /* awkward float values round trip too */
    const char *odd = "#gain_set,v=1,id=8,ctrl=PID,kp_xy=0.1,ki_xy=3.33333325,kp_wz=1e-07,ki_wz=19.9999981,"
                      "pos_kp=0.300000012,pos_ki=7.1,k_psi=2.71828175";
    parse(odd, &g);
    gain_set_apply(&g);
    gain_set_current(&cur);
    gain_set_format(line, sizeof line, &cur, 0);
    snprintf(pkt, sizeof pkt, "#gain_set,v=1,id=8%s", strstr(line, ",ctrl="));
    r = parse(pkt, &back);
    check(r == GAIN_OK && memcmp(back.v, g.v, sizeof g.v) == 0, "roundtrip_odd_values", pkt);

    /* ── a rejected request changes nothing ── */
    gain_set_t before;
    gain_set_current(&before);
    uint32_t n_before = gain_set_applied_n();
    parse(ADRC_OK, &g);
    r = gain_set_apply(&g);                               /* controller is PID */
    check(r == GAIN_ERR_CTRL, "apply_ctrl_mismatch", gain_err_str(r));
    g = (gain_set_t){ .id = 11, .ctrl = CTRL_PID, .v = { 1, 0, 1, 0, 1, 0, NAN } };
    r = gain_set_apply(&g);
    check(r == GAIN_ERR_RANGE, "apply_range_rejected", gain_err_str(r));
    gain_set_current(&cur);
    check(memcmp(&cur, &before, sizeof cur) == 0 && gain_set_applied_n() == n_before,
          "rejected_changes_nothing", "gains or counters changed by a rejected request");

    /* ── apply ADRC: adrc_set_params, b0 and out_limit kept ── */
    vel_ctrl_set_mode(CTRL_ADRC);
    adrc_controller_t *ac = vel_ctrl_adrc();
    float b0_xy = ac->vx.eso.b0, b0_wz = ac->wz.eso.b0, lim_xy = ac->vx.output_limit, lim_wz = ac->wz.output_limit;
    ac->vx.eso.z2 = 0.7f; ac->vx.eso.initialized = true;
    parse(ADRC_OK, &g);
    r = gain_set_apply(&g);
    check(r == GAIN_OK, "apply_adrc_ok", gain_err_str(r));
    check(ac->vx.kp == 10.0f && ac->vy.kp == 10.0f && ac->wz.kp == 12.0f &&
          ac->vx.eso.beta1 == 70.0f && ac->vx.eso.beta2 == 1225.0f && ac->vy.eso.beta1 == 70.0f &&
          ac->wz.eso.beta1 == 80.0f && ac->wz.eso.beta2 == 1600.0f, "apply_adrc_values", "LADRC gains not as sent");
    check(ac->vx.eso.b0 == b0_xy && ac->vy.eso.b0 == b0_xy && ac->wz.eso.b0 == b0_wz &&
          ac->vx.output_limit == lim_xy && ac->wz.output_limit == lim_wz && b0_xy == 5.85f && b0_wz == 10.3f,
          "apply_adrc_fixed", "b0 or out_limit changed");
    check(ac->vx.eso.z2 == 0.0f && !ac->vx.eso.initialized, "apply_resets_eso", "ESO state kept");
    gain_set_current(&cur);
    snprintf(d, sizeof d, "id %lu ctrl %d", (unsigned long)cur.id, cur.ctrl);
    check(cur.id == 9 && cur.ctrl == CTRL_ADRC && memcmp(cur.v, g.v, sizeof g.v) == 0, "readback_adrc", d);

    printf("%d FAIL\n", n_fail);
    return n_fail;
}
