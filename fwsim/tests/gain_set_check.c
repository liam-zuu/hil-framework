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

static uint32_t last_tx;
static int parse_n(const char *s, size_t n, gain_set_t *g) {
    int kind = 0;
    int r = gain_msg_parse(s, n, &kind, g, &last_tx);
    return (r == GAIN_OK && kind != GAIN_MSG_SET) ? -99 : r;
}
static int parse(const char *s, gain_set_t *g) { return parse_n(s, strlen(s), g); }

/* fwsim entry points (fw_sim.c, linked into this test): the path the GA gains take on the PC */
void fwsim_set_pid(int axis, float kp, float ki, float kd, float int_limit, float out_limit);
void fwsim_set_adrc(int axis, float omega_c, float omega_o, float b0, float out_limit);
void fwsim_get_adrc(int axis, float out[4]);
int  fwsim_set_param(const char *name, double v);

static const char *PID_OK =
    "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=1.25,ki_xy=0.5,kp_wz=2,ki_wz=0.75,pos_kp=4,pos_ki=1.5,k_psi=0.8";
static const char *ADRC_OK =
    "#gain_set,v=1,tx=5,id=9,ctrl=ADRC,omega_c_xy=10,omega_o_xy=35,omega_c_wz=12,omega_o_wz=40,pos_kp=5,pos_ki=2,k_psi=1.2";

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
    r = parse("#gain_set,tx=3,ctrl=PID,k_psi=0.8,pos_ki=1.5,pos_kp=4,ki_wz=0.75,kp_wz=2,ki_xy=0.5,kp_xy=1.25,id=7,v=1\r\n", &g);
    check(r == GAIN_OK && g.v[0] == 1.25f && g.v[6] == 0.8f, "parse_any_order_crlf", gain_err_str(r));
    r = parse(ADRC_OK, &g);
    check(r == GAIN_OK && g.ctrl == CTRL_ADRC && g.v[1] == 35.0f, "parse_adrc", gain_err_str(r));

    expect_err("err_header",        "#gain_sett,v=1,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_v2",            "#gain_set,v=2,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_no_v",          "#gain_set,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_id0",           "#gain_set,v=1,tx=5,id=0,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_id_big",        "#gain_set,v=1,tx=5,id=2147483648,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_id_sign",       "#gain_set,v=1,tx=5,id=+7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_id_twice",      "#gain_set,v=1,tx=5,id=7,id=8,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_ctrl",          "#gain_set,v=1,tx=5,id=7,ctrl=pid,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_missing_gain",  "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0", GAIN_ERR_FORMAT);
    expect_err("err_dup_gain",      "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=1,kp_xy=1,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_extra_gain",    "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1,kd_xy=0", GAIN_ERR_FORMAT);
    expect_err("err_adrc_names_pid","#gain_set,v=1,tx=5,id=7,ctrl=PID,omega_c_xy=10,omega_o_xy=35,omega_c_wz=12,omega_o_wz=40,pos_kp=5,pos_ki=2,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_value_junk",    "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=1.5x,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_value_empty",   "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_empty_field",   "#gain_set,v=1,tx=5,id=7,ctrl=PID,,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_no_equal",      "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_space",         "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=1 ,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);

    expect_err("err_nan_word",      "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=nan,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_inf_word",      "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=inf,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("range_overflow",    "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=1e40,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_RANGE);
    expect_err("range_negative",    "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=1,ki_xy=-0.1,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_RANGE);
    expect_err("range_pid_kp_max",  "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=20.5,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_RANGE);
    expect_err("range_pid_at_max",  "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=20,ki_xy=50,kp_wz=20,ki_wz=50,pos_kp=30,pos_ki=30,k_psi=10", GAIN_OK);
    expect_err("range_k_psi_max",   "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=10.5", GAIN_ERR_RANGE);
    expect_err("range_adrc_wo_49",  "#gain_set,v=1,tx=5,id=9,ctrl=ADRC,omega_c_xy=10,omega_o_xy=49,omega_c_wz=12,omega_o_wz=40,pos_kp=5,pos_ki=2,k_psi=1", GAIN_OK);
    expect_err("range_adrc_wo_50",  "#gain_set,v=1,tx=5,id=9,ctrl=ADRC,omega_c_xy=10,omega_o_xy=49.5,omega_c_wz=12,omega_o_wz=40,pos_kp=5,pos_ki=2,k_psi=1", GAIN_ERR_RANGE);
    expect_err("range_adrc_wc_low", "#gain_set,v=1,tx=5,id=9,ctrl=ADRC,omega_c_xy=0.4,omega_o_xy=35,omega_c_wz=12,omega_o_wz=40,pos_kp=5,pos_ki=2,k_psi=1", GAIN_ERR_RANGE);

    char longpkt[400];
    memset(longpkt, '0', sizeof longpkt - 1);
    longpkt[sizeof longpkt - 1] = '\0';
    memcpy(longpkt, PID_OK, strlen(PID_OK));
    expect_err("err_too_long", longpkt, GAIN_ERR_FORMAT);
    char nul[200];
    size_t L = strlen(PID_OK);
    memcpy(nul, PID_OK, L);
    nul[20] = '\0';
    check(parse_n(nul, L, &g) == GAIN_ERR_FORMAT, "err_embedded_nul", "packet with a 0 byte accepted");

    /* ── round-1 fixes (Codex L90-7, L90-8, L90-10, L90-12) ── */
    r = parse(PID_OK, &g);
    check(r == GAIN_OK && last_tx == 5, "tx_returned", "tx not returned");
    expect_err("err_no_tx",        "#gain_set,v=1,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_tx0",          "#gain_set,v=1,tx=0,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("tx_max",           "#gain_set,v=1,tx=4294967295,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_OK);
    expect_err("err_tx_big",       "#gain_set,v=1,tx=4294967296,id=7,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_id_lead0",     "#gain_set,v=1,tx=5,id=07,ctrl=PID,kp_xy=1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_cr_in_value",  "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=\r1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_lf_in_value",  "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=\n1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_tab",          "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=\t1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_plus",         "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=+1,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_hex",          "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=0x1p0,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_trailing_dot", "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=1.,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_leading_dot",  "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=.5,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_exp_empty",    "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=1e,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("err_word_inf",     "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=INF,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    expect_err("exp_forms_ok",     "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=1e+1,ki_xy=1E-07,kp_wz=1.5e0,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_OK);
    expect_err("err_non_ascii",    "#gain_set,v=1,tx=5,id=7,ctrl=PID,kp_xy=1\xc3,ki_xy=0,kp_wz=1,ki_wz=0,pos_kp=1,pos_ki=0,k_psi=1", GAIN_ERR_FORMAT);
    char eol[400];
    snprintf(eol, sizeof eol, "%s\n", PID_OK);
    check(parse(eol, &g) == GAIN_OK, "one_lf_ok", eol);
    snprintf(eol, sizeof eol, "%s\r\n", PID_OK);
    check(parse(eol, &g) == GAIN_OK, "one_crlf_ok", eol);
    snprintf(eol, sizeof eol, "%s\n\n", PID_OK);
    check(parse(eol, &g) == GAIN_ERR_FORMAT, "err_two_lf", "two line ends accepted (Codex L90-7 truncation case)");
    snprintf(eol, sizeof eol, "%s\r", PID_OK);
    check(parse(eol, &g) == GAIN_ERR_FORMAT, "err_lone_cr", "lone CR accepted");
    char at_max[GAIN_MSG_MAX + 2];
    memset(at_max, '0', GAIN_MSG_MAX);
    at_max[GAIN_MSG_MAX] = '\0';
    memcpy(at_max, PID_OK, strlen(PID_OK));
    check(parse_n(at_max, GAIN_MSG_MAX, &g) == GAIN_ERR_FORMAT, "err_at_max_padding", "padded packet accepted");
    check(parse_n(at_max, GAIN_MSG_MAX + 1, &g) == GAIN_ERR_FORMAT, "err_over_max", "packet over GAIN_MSG_MAX accepted");
    int kind = 0;
    uint32_t tx = 0;
    r = gain_msg_parse("#gain_get,v=1,tx=9", 18, &kind, &g, &tx);
    check(r == GAIN_OK && kind == GAIN_MSG_GET && tx == 9, "get_ok", gain_err_str(r));
    r = gain_msg_parse("#gain_get,v=1,tx=9,id=3", 23, &kind, &g, &tx);
    check(r == GAIN_ERR_FORMAT && tx == 9, "err_get_extra_field", gain_err_str(r));
    r = gain_msg_parse("#gain_get,v=1", 13, &kind, &g, &tx);
    check(r == GAIN_ERR_FORMAT, "err_get_no_tx", gain_err_str(r));
    gain_set_t gd;
    gain_set_current(&gd);
    char small[40];
    check(gain_set_format(small, sizeof small, &gd, 0) == -1 && small[0] == '\0', "format_short_buffer",
          "short buffer did not return -1 with an empty string");

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
    snprintf(pkt, sizeof pkt, "#gain_set,v=1,tx=5,id=7%s", p);
    gain_set_t back;
    r = parse(pkt, &back);
    check(r == GAIN_OK && memcmp(back.v, cur.v, sizeof cur.v) == 0, "roundtrip_9g", pkt);

    /* awkward float values round trip too */
    const char *odd = "#gain_set,v=1,tx=5,id=8,ctrl=PID,kp_xy=0.1,ki_xy=3.33333325,kp_wz=1e-07,ki_wz=19.9999981,"
                      "pos_kp=0.300000012,pos_ki=7.1,k_psi=2.71828175";
    parse(odd, &g);
    gain_set_apply(&g);
    gain_set_current(&cur);
    gain_set_format(line, sizeof line, &cur, 0);
    snprintf(pkt, sizeof pkt, "#gain_set,v=1,tx=5,id=8%s", strstr(line, ",ctrl="));
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

    /* ── parity with fwsim (fw_sim.c + tune_ga.gains_for), Codex L90-14 ── */
    {
        float v[GAIN_N] = { 1.25f, 0.5f, 2.0f, 0.75f, 4.0f, 1.5f, 0.8f };
        vel_ctrl_init(); vel_ctrl_set_mode(CTRL_PID);
        /* tune_ga.gains_for PID: pid={0: (a, b, 0.0, 0.5, 1.5), 1: (a, b, 0.0, 0.5, 1.5), 2: (c, d, 0.0, 1.0, 2.0)} */
        fwsim_set_pid(0, v[0], v[1], 0.0f, 0.5f, 1.5f);
        fwsim_set_pid(1, v[0], v[1], 0.0f, 0.5f, 1.5f);
        fwsim_set_pid(2, v[2], v[3], 0.0f, 1.0f, 2.0f);
        fwsim_set_param("pos_kp", v[4]); fwsim_set_param("pos_ki", v[5]); fwsim_set_param("k_psi", v[6]);
        pid_controller_t a = *vel_ctrl_pid();
        float a_pos[2] = { pos_params()->kp, pos_params()->ki }, a_psi = heading_params()->k_psi;
        vel_ctrl_init(); vel_ctrl_set_mode(CTRL_PID);
        gain_set_t gp = { .id = 3, .ctrl = CTRL_PID };
        memcpy(gp.v, v, sizeof v);
        r = gain_set_apply(&gp);
        const pid_controller_t *b = vel_ctrl_pid();
        int same = r == GAIN_OK && pos_params()->kp == a_pos[0] && pos_params()->ki == a_pos[1] &&
                   heading_params()->k_psi == a_psi;
        const pid_t *pa[3] = { &a.vx, &a.vy, &a.wz }, *pb[3] = { &b->vx, &b->vy, &b->wz };
        for (int i = 0; i < 3; i++)
            same = same && pa[i]->kp == pb[i]->kp && pa[i]->ki == pb[i]->ki && pa[i]->kd == pb[i]->kd &&
                   pa[i]->integral_limit == pb[i]->integral_limit && pa[i]->output_limit == pb[i]->output_limit;
        check(same, "parity_fwsim_pid", "gain_set_apply and the fwsim path give different PID / outer fields");

        float w[GAIN_N] = { 10.0f, 35.0f, 12.0f, 40.0f, 5.0f, 2.0f, 1.2f };
        vel_ctrl_init(); vel_ctrl_set_mode(CTRL_ADRC);
        /* tune_ga.gains_for ADRC: b0, out_limit from fw.get_adrc after reset */
        for (int ax = 0; ax < 3; ax++) {
            float base[4];
            fwsim_get_adrc(ax, base);
            fwsim_set_adrc(ax, ax < 2 ? w[0] : w[2], ax < 2 ? w[1] : w[3], base[2], base[3]);
        }
        fwsim_set_param("pos_kp", w[4]); fwsim_set_param("pos_ki", w[5]); fwsim_set_param("k_psi", w[6]);
        adrc_controller_t aa = *vel_ctrl_adrc();
        vel_ctrl_init(); vel_ctrl_set_mode(CTRL_ADRC);
        gain_set_t ga = { .id = 4, .ctrl = CTRL_ADRC };
        memcpy(ga.v, w, sizeof w);
        r = gain_set_apply(&ga);
        const adrc_controller_t *ab = vel_ctrl_adrc();
        same = r == GAIN_OK && pos_params()->kp == w[4] && pos_params()->ki == w[5] && heading_params()->k_psi == w[6];
        const adrc_axis_t *xa[3] = { &aa.vx, &aa.vy, &aa.wz }, *xb[3] = { &ab->vx, &ab->vy, &ab->wz };
        for (int i = 0; i < 3; i++)
            same = same && xa[i]->kp == xb[i]->kp && xa[i]->eso.beta1 == xb[i]->eso.beta1 &&
                   xa[i]->eso.beta2 == xb[i]->eso.beta2 && xa[i]->eso.b0 == xb[i]->eso.b0 &&
                   xa[i]->output_limit == xb[i]->output_limit;
        check(same, "parity_fwsim_adrc", "gain_set_apply and the fwsim path give different LADRC / outer fields");
    }

    printf("%d FAIL\n", n_fail);
    return n_fail;
}
