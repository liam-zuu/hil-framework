/* panel_buzz.c - see panel_buzz.h. */
#include "panel_buzz.h"

#include <stddef.h>
#include <string.h>

#include "panel_proto.h"

/* patterns [ĐỀ XUẤT]: about 2-3 kHz suits small passive buzzers; the fault is longer and
 * lower so it reads as different without looking at the screen */
static const pb_note_t P_START[] = {{2700, 80}, {0, 0}};
static const pb_note_t P_DONE[] = {{2700, 90}, {0, 90}, {2700, 90}, {0, 0}};
static const pb_note_t P_STOPPED[] = {{2000, 250}, {0, 0}};
static const pb_note_t P_FAULT[] = {{1500, 300}, {0, 150}, {1500, 300}, {0, 150}, {1500, 300}, {0, 0}};
static const pb_note_t P_LINK_LOST[] = {{2000, 150}, {1200, 250}, {0, 0}};

const pb_note_t *pb_pattern(pb_event_t e)
{
    switch (e) {
    case PB_START:     return P_START;
    case PB_DONE:      return P_DONE;
    case PB_STOPPED:   return P_STOPPED;
    case PB_FAULT:     return P_FAULT;
    case PB_LINK_LOST: return P_LINK_LOST;
    default:           return NULL;
    }
}

void pb_init(pb_t *b)
{
    memset(b, 0, sizeof *b);
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
    if (e != PB_NONE) {
        b->pat = pb_pattern(e);
        b->idx = 0;
        b->t_note = now;
    }
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
    return b->pat[b->idx].hz;
}
