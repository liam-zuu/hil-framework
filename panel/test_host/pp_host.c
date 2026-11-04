/* pp_host.c — PC-only wrappers so Python (ctypes) can drive panel_proto.c in the tests.
 * Not built into the firmware. */
#include <stdlib.h>
#include <string.h>

#include "panel_proto.h"

pp_client_t *pp_host_client_new(pp_nonce_fn nonce, pp_send_fn send)
{
    pp_client_t *c = (pp_client_t *)calloc(1, sizeof *c);
    pp_client_init(c, nonce, NULL, send, NULL);
    return c;
}

void pp_host_client_free(pp_client_t *c) { free(c); }

uint32_t pp_host_sid(const pp_client_t *c) { return c->sid; }
uint16_t pp_host_boot(const pp_client_t *c) { return c->boot; }
int pp_host_next_seq(const pp_client_t *c) { return c->has_next_seq ? c->next_seq : -1; }
int pp_host_stale(const pp_client_t *c) { return c->has_stale ? c->stale : -1; }
int pp_host_fence(const pp_client_t *c) { return c->fence; }
int pp_host_stop_state(const pp_client_t *c) { return c->stop_state; }
uint32_t pp_host_stop_failures(const pp_client_t *c) { return c->stop_failures; }

/* pending: returns 0 when none, else 1 and fills seq/cmd/arg */
int pp_host_pending(const pp_client_t *c, uint16_t *seq, uint8_t *cmd, uint16_t *arg)
{
    if (!c->pending.active)
        return 0;
    *seq = c->pending.seq;
    *cmd = c->pending.cmd;
    *arg = c->pending.arg;
    return 1;
}

/* stop: 0 none; 1 active without seq; 2 active with seq (filled) */
int pp_host_stop(const pp_client_t *c, uint16_t *seq)
{
    if (!c->stop.active)
        return 0;
    if (!c->stop.has_seq)
        return 1;
    *seq = c->stop.seq;
    return 2;
}

/* profile i: 1 and the name when received in this session, else 0 */
int pp_host_profile(const pp_client_t *c, int i, char *name)
{
    if (i < 0 || i >= PP_PROFILES_STORED || !c->profile_ok[i])
        return 0;
    strcpy(name, c->profile_name[i]);
    return 1;
}

const pp_info_t *pp_host_info(const pp_client_t *c) { return c->has_info ? &c->info : NULL; }

pp_parser_t *pp_host_parser_new(uint8_t accept, uint32_t stale_ms)
{
    pp_parser_t *p = (pp_parser_t *)malloc(sizeof *p);
    pp_parser_init(p, accept, stale_ms);
    return p;
}

void pp_host_parser_free(pp_parser_t *p) { free(p); }

void pp_host_parser_counts(const pp_parser_t *p, uint32_t out[3])
{
    out[0] = p->bad_frames;
    out[1] = p->skipped_bytes;
    out[2] = p->stale_drops;
}
