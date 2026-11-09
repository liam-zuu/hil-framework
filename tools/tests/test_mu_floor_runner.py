"""Tests of tools/mu_floor_runner.py (no hardware). Run: python -m pytest tools/tests -q

FakeEsp32 below is a small UDP server on 127.0.0.1 that speaks protocol v2 of AGV_MECANUM
src/tests/test_mu_floor.c (PING/PONG, RAMP <id> with DUP, H, STOP, D lines with seq, DONE n=), with
switches to lose datagrams, so the runner's resend, heartbeat and missing-sample rules are tested.
It is a copy of the protocol, not of the firmware: the firmware itself is tested on the board."""
import csv
import os
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import mu_floor_runner as mf  # noqa: E402


def test_mu_from_force_is_kgf_over_kg():
    assert mf.mu_from_force(2.4, 4.0) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        mf.mu_from_force(1.0, 0.0)
    with pytest.raises(ValueError):
        mf.mu_from_force(-1.0, 4.0)


def test_summarize_levels_are_min_mean_max_of_good_trials():
    tr = [dict(surface='floor', direction='XP', mu=m) for m in (0.5, 0.6, 0.7)] + \
         [dict(surface='floor', direction='YP', mu=m) for m in (0.4, None)] + \
         [dict(surface='mica', direction='XP', mu=0.2)]
    s = {(r['surface'], r['direction']): r for r in mf.summarize(tr)}
    assert s[('floor', 'all')]['n'] == 4                               # the bad (None) trial is left out
    assert (s[('floor', 'all')]['mu_min'], s[('floor', 'all')]['mu_max']) == (0.4, 0.7)
    assert s[('floor', 'all')]['mu_mean'] == pytest.approx((0.5 + 0.6 + 0.7 + 0.4) / 4)
    assert s[('floor', 'YP')]['n'] == 1 and s[('mica', 'all')]['mu_mean'] == 0.2


def test_parse_lines():
    d = mf.parse_data('D,120,101,3,-3,-3,3,0.0123,-0.0040,9.8061,0.00012,24,12')
    assert d == dict(t_ms=120, pwm=101, c_fl=3, c_fr=-3, c_rl=-3, c_rr=3, ax=0.0123, ay=-0.004, az=9.8061,
                     gz=0.00012, imu_n=24, seq=12)
    assert mf.parse_data('D,120,101,3,-3,-3,3,0.0123,-0.0040,9.8061,0.00012,24') is None   # v1 line
    assert mf.parse_data('D,1,2,3') is None and mf.parse_data('EV START id=3 t_ms=0') is None
    assert mf.parse_kv('DONE reason=slip t_ms=8340 pwm=183 id=4 n=835 udp_err=0') == \
        dict(reason='slip', t_ms='8340', pwm='183', id='4', n='835', udp_err='0')
    assert mf.parse_kv('EV SLIP t_ms=8340 pwm=183 w=3.41') == dict(t_ms='8340', pwm='183', w='3.41')
    assert mf.parse_kv('OK RAMP id=7') == dict(id='7') and mf.parse_kv('DUP RAMP id=2 last=5') == dict(id='2', last='5')
    assert mf.parse_kv('PONG ip=192.168.1.9 imu=ok fw=abc dirty=0 env=mu_floor last_id=3')['last_id'] == '3'


def rows_of(seqs):
    return [dict(seq=s) for s in seqs]


def test_link_check_two_percent_rule():
    ev = dict(start={}, done=dict(reason='slip', n='100'))
    assert mf.link_check(rows_of(range(100)), ev)['link_ok']
    c = mf.link_check(rows_of(s for s in range(100) if s not in (5, 6)), ev)     # 2 of 100 = 2 %: still ok
    assert c['link_ok'] and c['n_missing'] == 2
    c = mf.link_check(rows_of(s for s in range(100) if s not in (5, 6, 7)), ev)  # 3 % > 2 %
    assert not c['link_ok'] and '3/100' in c['link_note']
    # the count comes from DONE n, so missing lines at the end are seen too
    assert mf.link_check(rows_of(range(97)), ev)['n_missing'] == 3


def test_link_check_needs_start_done_and_no_heartbeat_stop():
    assert 'no EV START' in mf.link_check(rows_of(range(10)), dict(done=dict(n='10')))['link_note']
    c = mf.link_check(rows_of(range(10)), dict(start={}, link='no line for 3 s'))
    assert not c['link_ok'] and 'no DONE (no line for 3 s)' in c['link_note'] and c['n_expected'] == 10
    c = mf.link_check(rows_of(range(10)), dict(start={}, done=dict(reason='heartbeat', n='10')))
    assert not c['link_ok'] and 'heartbeat' in c['link_note']


def test_ask_force_accepts_comma_and_blank_is_bad():
    seq = iter(['abc', '-1', '2,35'])
    assert mf.ask_force('', lambda p: next(seq)) == pytest.approx(2.35)
    assert mf.ask_force('', lambda p: '') is None


def args(out, **kw):
    a = dict(out=str(out), dirs=['XP'], reps=2, surface='floor', pwm0=100, rate=50, pwm_max=1000, w_slip=3.0,
             mass_kg=4.0, dry_run=True, gauge='test')
    a.update(kw)
    return mf.argparse.Namespace(**a)


def test_dry_run_writes_trials_summary_meta_and_resumes(tmp_path):
    yaml = pytest.importorskip('yaml')
    out = tmp_path / 'run'
    answers = iter(['', '2.4', '', ''] * 2)                            # Enter to start, F, ...; 2nd trial bad
    a = args(out)
    dev = mf.FakeDevice(mass_kg=4.0, mu=0.6)
    os.makedirs(out)
    mf.write_meta(a, dev, os.path.dirname(mf.__file__))
    trials, summ = mf.run(a, dev, inp=lambda p: next(answers))
    rows = list(csv.DictReader(open(out / 'trials.csv', encoding='utf-8')))
    assert [r['trial'] for r in rows] == ['floor_XP_01', 'floor_XP_02']
    assert rows[0]['reason'] == 'slip' and float(rows[0]['mu']) == pytest.approx(0.6) and rows[1]['mu'] == ''
    assert rows[0]['link_ok'] == 'True' and rows[0]['n_missing'] == '0'
    raw = list(csv.DictReader(open(out / 'raw' / 'floor_XP_01.csv', encoding='utf-8')))
    assert raw and list(raw[0]) == mf.RAW_COLS
    m = yaml.safe_load(open(out / 'meta.yaml', encoding='utf-8'))
    assert m['mass_kg'] == 4.0 and m['link']['kind'] == 'dry-run' and m['link_rules']['missing_max'] == 0.02
    # resume: only the bad trial is run again
    seen = []
    again = iter(['', '2.0'])
    mf.run(a, dev, inp=lambda p: (seen.append(p), next(again))[1])
    assert any('floor_XP_02' in p for p in seen) and not any('floor_XP_01' in p for p in seen)
    rows = list(csv.DictReader(open(out / 'trials.csv', encoding='utf-8')))
    assert len(rows) == 2 and float(rows[1]['mu']) == pytest.approx(0.5)
    s = {r['direction']: r for r in csv.DictReader(open(out / 'summary.csv', encoding='utf-8'))}
    assert int(s['all']['n']) == 2 and float(s['all']['mu_mean']) == pytest.approx(0.55)


def test_main_needs_one_link_unless_dry_run(tmp_path):
    with pytest.raises(SystemExit, match='exactly one of --udp'):
        mf.main(['--out', str(tmp_path / 'o'), '--mass-kg', '4'])
    with pytest.raises(SystemExit, match='exactly one of --udp'):
        mf.main(['--out', str(tmp_path / 'o'), '--mass-kg', '4', '--udp', 'auto', '--port', 'COM5'])


class Pong:
    def __init__(self, **kw):
        self.pong = dict(fw='1ffb0813a1b2', dirty='0', env='mu_floor', imu='ok', last_id='0')
        self.pong.update(kw)
        self.pong_line = 'PONG ' + ' '.join(f'{k}={v}' for k, v in self.pong.items())


def test_check_firmware():
    assert mf.check_firmware(Pong()) == ''
    assert 'protocol v2' in mf.check_firmware(type('P', (), dict(pong={}, pong_line='PONG'))())
    assert 'env' in mf.check_firmware(Pong(env='ladrc'))
    assert 'expect' in mf.check_firmware(Pong(), expect='b64095e')
    assert mf.check_firmware(Pong(), expect='1ffb081') == ''
    assert 'IMU' in mf.check_firmware(Pong(imu='fail'))


# ── fake ESP32 over UDP ──────────────────────────────────────────────────────

class FakeEsp32:
    """Protocol v2 on 127.0.0.1:<port>. n_lines D lines then DONE reason=max (1 ms apart).
    drop_ramp: how many RAMP datagrams to ignore; drop_seq: seq numbers not sent; drop_done: DONE
    not sent; silent_after: stop sending anything after this seq (the run goes on until heartbeat)."""

    def __init__(self, n_lines=50, drop_ramp=0, drop_seq=(), drop_done=False, silent_after=None, last_id=0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', 0))
        self.sock.settimeout(0.02)
        self.port = self.sock.getsockname()[1]
        self.n_lines, self.drop_ramp, self.drop_seq = n_lines, drop_ramp, set(drop_seq)
        self.drop_done, self.silent_after = drop_done, silent_after
        self.last_id, self.runs, self.got, self.peer = last_id, [], [], None
        self.quit = threading.Event()
        self.th = threading.Thread(target=self.loop, daemon=True)
        self.th.start()

    def close(self):
        self.quit.set()
        self.th.join()
        self.sock.close()

    def send(self, s, force=False):
        if self.peer and (force or not self.silent):
            self.sock.sendto((s + '\n').encode(), self.peer)

    def lines(self):
        try:
            data, addr = self.sock.recvfrom(2048)
        except (socket.timeout, ConnectionResetError):
            return []
        self.peer = addr
        ls = [s for s in data.decode().splitlines() if s]
        self.got += ls
        return ls

    def loop(self):
        self.silent = False
        while not self.quit.is_set():
            for ln in self.lines():
                if ln == 'PING':
                    self.send(f'PONG ip=127.0.0.1 imu=ok fw=1ffb0813a1b2 dirty=0 env=mu_floor last_id={self.last_id} '
                              'last_reason=none last_n=0', force=True)
                elif ln.startswith('RAMP '):
                    if self.drop_ramp > 0:
                        self.drop_ramp -= 1
                        continue
                    rid = int(ln.split()[1])
                    if rid <= self.last_id:
                        self.send(f'DUP RAMP id={rid} last={self.last_id}', force=True)
                        continue
                    self.last_id = rid
                    self.run(rid)

    def run(self, rid):
        self.runs.append(rid)
        self.send(f'OK RAMP id={rid}')
        self.send(f'EV START id={rid} t_ms=0')
        last_hb, reason, seq = time.time(), 'max', 0
        while seq < self.n_lines:
            for ln in self.lines():
                last_hb = time.time()
                if ln == 'STOP':
                    reason = 'stop'
            if reason == 'stop':
                break
            if time.time() - last_hb > 0.5:
                reason = 'heartbeat'
                break
            if self.silent_after is not None and seq > self.silent_after:
                self.silent = True
            if seq not in self.drop_seq:
                self.send(f'D,{seq * 10},{100 + seq},{seq},{-seq},{-seq},{seq},0.0,0.0,9.81,0.0,{seq // 5},{seq}')
            seq += 1
            time.sleep(0.001)
        if self.silent_after is not None:                 # silent link: wait for the heartbeat stop
            while time.time() - last_hb <= 0.5 and not self.quit.is_set():
                if self.lines():
                    last_hb = time.time()
            reason = 'heartbeat'
        if not self.drop_done:
            self.send(f'DONE reason={reason} t_ms={seq * 10} pwm={100 + seq} id={rid} n={seq} udp_err=0')
        self.silent = False


@pytest.fixture
def fake():
    made = []

    def make(**kw):
        f = FakeEsp32(**kw)
        made.append(f)
        return f
    yield make
    for f in made:
        f.close()


def test_udp_discovery_and_ramp_ids_follow_last_id(fake):
    esp = fake(last_id=6)
    dev = mf.UdpLink('auto', esp.port, bcast='127.0.0.1')
    assert dev.host == '127.0.0.1' and dev.pong['fw'] == '1ffb0813a1b2' and dev.next_id == 7
    rows, ev = dev.ramp('XP', 100, 10, 1000, 3.0)
    assert ev['id'] == 7 and ev['sends'] == 1 and ev['done']['n'] == '50' and len(rows) == 50
    assert mf.link_check(rows, ev)['link_ok']
    assert esp.got.count('H') >= 0 and esp.runs == [7]
    rows, ev = dev.ramp('YP', 100, 10, 1000, 3.0)
    assert ev['id'] == 8 and esp.runs == [7, 8]


def test_udp_resends_ramp_and_firmware_runs_it_once(fake):
    esp = fake(drop_ramp=1)
    dev = mf.UdpLink('127.0.0.1', esp.port)
    t0 = time.time()
    rows, ev = dev.ramp('XP', 100, 10, 1000, 3.0)
    assert ev['sends'] == 2 and time.time() - t0 >= mf.ACK_S * 0.9
    assert esp.runs == [1] and mf.link_check(rows, ev)['link_ok']


def test_udp_no_answer_marks_bad_and_sends_stop(fake):
    esp = fake(drop_ramp=99)
    dev = mf.UdpLink('127.0.0.1', esp.port)
    rows, ev = dev.ramp('XP', 100, 10, 1000, 3.0)
    assert ev['sends'] == mf.RAMP_SENDS and 'no answer' in ev['link'] and esp.runs == []
    time.sleep(0.1)
    assert esp.got.count('STOP') == 3 and 'H' in esp.got          # heartbeat ran from the first send
    assert not mf.link_check(rows, ev)['link_ok']


def test_udp_missing_lines_and_missing_done(fake):
    esp = fake(drop_seq=range(10, 12))                              # 2/50 = 4 % > 2 %
    dev = mf.UdpLink('127.0.0.1', esp.port)
    rows, ev = dev.ramp('XP', 100, 10, 1000, 3.0)
    c = mf.link_check(rows, ev)
    assert len(rows) == 48 and not c['link_ok'] and c['n_missing'] == 2
    esp2 = fake(drop_done=True)
    dev2 = mf.UdpLink('127.0.0.1', esp2.port)
    old = mf.LINE_TIMEOUT_S
    mf.LINE_TIMEOUT_S = 0.5
    try:
        rows, ev = dev2.ramp('XP', 100, 10, 1000, 3.0)
    finally:
        mf.LINE_TIMEOUT_S = old
    c = mf.link_check(rows, ev)
    assert 'done' not in ev and not c['link_ok'] and 'no DONE' in c['link_note']
    time.sleep(0.1)
    assert esp2.got.count('STOP') == 3


def test_udp_link_lost_mid_run_is_bad(fake):
    esp = fake(n_lines=200, silent_after=20)
    dev = mf.UdpLink('127.0.0.1', esp.port)
    old = mf.LINE_TIMEOUT_S
    mf.LINE_TIMEOUT_S = 0.4
    try:
        rows, ev = dev.ramp('XP', 100, 10, 1000, 3.0)
    finally:
        mf.LINE_TIMEOUT_S = old
    c = mf.link_check(rows, ev)
    assert not c['link_ok'] and len(rows) <= 22


def test_udp_stale_id_gets_dup_and_next_id_is_refreshed(fake):
    esp = fake(last_id=4)
    dev = mf.UdpLink('127.0.0.1', esp.port)
    dev.next_id = 3                                                 # as if the PC had an old id
    rows, ev = dev.ramp('XP', 100, 10, 1000, 3.0)
    assert ev['link'] == 'stale id' and esp.runs == [] and dev.next_id == 5
    assert not mf.link_check(rows, ev)['link_ok']
    rows, ev = dev.ramp('XP', 100, 10, 1000, 3.0)
    assert esp.runs == [5] and mf.link_check(rows, ev)['link_ok']


def test_run_over_udp_marks_link_bad_trial_without_asking_force(fake, tmp_path):
    esp = fake(drop_seq=range(0, 5))
    dev = mf.UdpLink('127.0.0.1', esp.port)
    asked = []
    a = args(tmp_path / 'r', reps=1, dry_run=False)
    mf.run(a, dev, inp=lambda p: (asked.append(p), '')[1])
    rows = list(csv.DictReader(open(tmp_path / 'r' / 'trials.csv', encoding='utf-8')))
    assert len(asked) == 1                                           # only the "Enter to start" prompt
    assert rows[0]['link_ok'] == 'False' and rows[0]['mu'] == '' and rows[0]['note'].startswith('link:')
    assert rows[0]['n_missing'] == '5' and rows[0]['run_id'] == '1'
