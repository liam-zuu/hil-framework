/* L95 (Liam 12/11 "SWB + ghi log", 13/11 "Dau luot + khi doi + cuoi"): host checks of the firmware
 * module src/ultrasonic/us_state.c (SWB rule, per-run counters, line format). Prints PASS/FAIL lines;
 * exit code 1 on any FAIL. Built and run by test_us_state.py. */
#include <stdio.h>
#include <string.h>
#include "us_state.h"

static int n_fail = 0, n_pass = 0;

static void check(int ok, const char *what) {
    if (ok) { n_pass++; printf("PASS %s\n", what); }
    else    { n_fail++; printf("FAIL %s\n", what); }
}

static void check_str(const char *got, const char *want, const char *what) {
    int ok = strcmp(got, want) == 0;
    if (!ok) printf("  got : %s  want: %s", got, want);
    check(ok, what);
}

int main(void) {
    /* (a) SWB rule: < 1500 is ON (old MANUAL rule), no valid iBUS is ON (safe) */
    check(us_swb_enabled(false, 2000) == true, "no rc -> on (even with SWB high)");
    check(us_swb_enabled(false, 0) == true,    "no rc -> on");
    check(us_swb_enabled(true, 1000) == true,  "SWB 1000 -> on");
    check(us_swb_enabled(true, 1499) == true,  "SWB 1499 -> on");
    check(us_swb_enabled(true, 1500) == false, "SWB 1500 -> off (H7 sends 1500 on channel 9)");
    check(us_swb_enabled(true, 2000) == false, "SWB 2000 -> off");

    /* (b) run counters */
    us_run_t r;
    memset(&r, 0xAB, sizeof(r));
    us_run_begin(&r, true);
    check(r.on == 1 && r.changes == 0 && r.ticks == 0 && r.blk_ticks == 0 && r.cut_ticks == 0,
          "begin clears counters, on=1");
    check(us_run_tick(&r, true, false, false) == false, "same state -> no change");
    check(r.ticks == 1 && r.changes == 0, "tick counted");
    check(us_run_tick(&r, true, true, false) == false && r.blk_ticks == 1 && r.cut_ticks == 0,
          "blocked tick counted");
    check(us_run_tick(&r, true, true, true) == false && r.blk_ticks == 2 && r.cut_ticks == 1,
          "cut tick counted");
    check(us_run_tick(&r, false, false, false) == true && r.on == 0 && r.changes == 1,
          "off mid-run -> change");
    check(us_run_tick(&r, false, false, false) == false && r.changes == 1, "stays off -> no change");
    check(us_run_tick(&r, true, false, false) == true && r.on == 1 && r.changes == 2,
          "on again -> second change");
    check(r.ticks == 6, "six ticks");
    us_run_begin(&r, false);
    check(r.on == 0 && r.changes == 0 && r.ticks == 0, "new run starts clean, on=0");

    /* (c) line format */
    char buf[256];
    us_run_t s = { .on = 1 };
    int n = us_format_line(buf, sizeof(buf), US_EV_START, &s, 1234, 0);
    check_str(buf, "#us,ev=start,on=1,first_seq=1234,line_drop=0\n", "start line");
    check(n == (int)strlen(buf), "start length");
    s.on = 0;
    us_format_line(buf, sizeof(buf), US_EV_CHANGE, &s, 4294967295u, 7);
    check_str(buf, "#us,ev=change,on=0,first_seq=4294967295,line_drop=7\n", "change line, u32 max");
    us_run_t e = { .on = 1, .changes = 2, .blk_ticks = 30, .cut_ticks = 12, .ticks = 1500 };
    us_format_line(buf, sizeof(buf), US_EV_END, &e, 99, 1);
    check_str(buf, "#us_end,on=1,changes=2,blk_ticks=30,cut_ticks=12,ticks=1500,last_data_seq=99,line_drop=1\n",
              "end line");
    us_run_t big = { .on = 1, .changes = 4294967295u, .blk_ticks = 4294967295u,
                     .cut_ticks = 4294967295u, .ticks = 4294967295u };
    n = us_format_line(buf, sizeof(buf), US_EV_END, &big, 4294967295u, 4294967295u);
    check(n > 0 && n < (int)sizeof(buf) && n < 256, "longest end line fits the 1024-byte logger buffer");
    char small[10];
    n = us_format_line(small, sizeof(small), US_EV_START, &s, 1, 0);
    check(n >= (int)sizeof(small), "small buffer -> length >= size (logger drops the line)");
    check(us_format_line(buf, sizeof(buf), 7, &s, 1, 0) < 0, "unknown event -> error");
    check(us_format_line(NULL, 10, US_EV_START, &s, 1, 0) < 0, "NULL buffer -> error");
    check(us_format_line(buf, sizeof(buf), US_EV_START, NULL, 1, 0) < 0, "NULL run -> error");

    printf("%d pass, %d fail\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
