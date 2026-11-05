/* panel_buzz.c - see panel_buzz.h. */
#include "panel_buzz.h"

#include <stddef.h>
#include <string.h>

#include "panel_proto.h"

/* patterns [ĐỀ XUẤT] (decision 21, Liam 05/11), in per mille of f0; with f0 = 2700 Hz:
 *  BOOT       1800, 2025, 2400, 2700 (110 ms each), rest, 2400, 2700 held: ~1 s, going up
 *  START      one 150 ms beep (80 ms before: too short to sound loud)
 *  DONE       two beeps going up (2400 then 2700)
 *  STOPPED    one long 400 ms beep
 *  FAULT      three 300 ms beeps (at f0 now: 1500 Hz before was far from resonance)
 *  LINK_LOST  f0 then 2/3 f0, the falling shape Liam heard loudest ("lúc no link ... to, rõ") */
#define F1  1000u              /* f0 */
#define F89  889u              /* 8/9 */
#define F34  750u              /* 3/4 */
#define F23  667u              /* 2/3 */

static const pb_note_t P_BOOT[] = {{F23, 110}, {F34, 110}, {F89, 110}, {F1, 110}, {0, 70},
                                   {F89, 90}, {F1, 450}, {0, 0}};
static const pb_note_t P_START[] = {{F1, 150}, {0, 0}};
static const pb_note_t P_DONE[] = {{F89, 120}, {0, 60}, {F1, 250}, {0, 0}};
static const pb_note_t P_STOPPED[] = {{F1, 400}, {0, 0}};
static const pb_note_t P_FAULT[] = {{F1, 300}, {0, 150}, {F1, 300}, {0, 150}, {F1, 300}, {0, 0}};
static const pb_note_t P_LINK_LOST[] = {{F1, 150}, {F23, 250}, {0, 0}};

const pb_note_t *pb_pattern(pb_event_t e)
{
    switch (e) {
    case PB_START:     return P_START;
    case PB_DONE:      return P_DONE;
    case PB_STOPPED:   return P_STOPPED;
    case PB_FAULT:     return P_FAULT;
    case PB_LINK_LOST: return P_LINK_LOST;
    case PB_BOOT:      return P_BOOT;
    default:           return NULL;
    }
}

const char *pb_name(pb_event_t e)
{
    switch (e) {
    case PB_START:     return "Start";
    case PB_DONE:      return "Done";
    case PB_STOPPED:   return "User stop";
    case PB_FAULT:     return "Fault";
    case PB_LINK_LOST: return "Link lost";
    case PB_BOOT:      return "Boot";
    default:           return "";
    }
}

void pb_init(pb_t *b)
{
    memset(b, 0, sizeof *b);
    b->f0 = PB_F0_DEFAULT;
}

void pb_set_f0(pb_t *b, uint16_t hz)
{
    if (hz < PB_F0_MIN) hz = PB_F0_MIN;
    if (hz > PB_F0_MAX) hz = PB_F0_MAX;
    b->f0 = hz;
}

void pb_play(pb_t *b, pb_event_t e, uint32_t now)
{
    b->pat = pb_pattern(e);
    b->idx = 0;
    b->t_note = now;
}

bool pb_busy(const pb_t *b)
{
    return b->pat != NULL;
}

static void take(pb_t *b, uint32_t sid, bool link, uint8_t st, uint16_t rn)
{
    b->init = true;
    b->sid = sid;
    b->link = link;
    b->state = st;
    b->run_no = rn;
}

static pb_event_t end_event(uint8_t outcome)
{
    if (outcome == PP_OUT_COMPLETED) return PB_DONE;
    if (outcome == PP_OUT_USER_STOP) return PB_STOPPED;
    if (outcome == PP_OUT_ERROR) return PB_FAULT;
    return PB_NONE;
}

pb_event_t pb_watch(pb_t *b, uint32_t sid, bool link, uint8_t st, uint16_t rn, uint8_t outcome, uint32_t now)
{
    pb_event_t e = PB_NONE;
    if (!b->init || (link && b->link && sid != b->sid)) {
        take(b, sid, link, st, rn);              /* first look or another session: silent */
        return PB_NONE;
    }
    if (b->link && !link) {
        e = PB_LINK_LOST;
    } else if (link && !b->link) {
        e = PB_NONE;                             /* link back: the state found is not news */
    } else if (link) {
        if (rn != b->run_no) {                   /* another run */
            if (PP_ST_IS_ACTIVE(st)) e = PB_START;
            else if (st == PP_ST_END) e = end_event(outcome);
            else if (st == PP_ST_ERROR) e = PB_FAULT;
        } else if (st == PP_ST_ERROR && b->state != PP_ST_ERROR) {
            e = PB_FAULT;
        } else if (st == PP_ST_END && b->state != PP_ST_END) {
            e = end_event(outcome);
        }
    }
    take(b, sid, link, st, rn);
    if (e != PB_NONE)
        pb_play(b, e, now);
    return e;
}

uint16_t pb_tone(pb_t *b, uint32_t now)
{
    if (!b->pat)
        return 0;
    while (b->pat[b->idx].ms && (uint32_t)(now - b->t_note) >= b->pat[b->idx].ms) {
        b->t_note += b->pat[b->idx].ms;
        b->idx++;
    }
    if (!b->pat[b->idx].ms) {
        b->pat = NULL;
        return 0;
    }
    if (!b->pat[b->idx].pm)
        return 0;
    return (uint16_t)(((uint32_t)b->f0 * b->pat[b->idx].pm + 500u) / 1000u);
}
