"""C8b v3 §5 in rq3_log: session premise and shifted-anchor bound of campaign car runs (CODE_GATE 18/11/2026).

Design CLAUDE_THESIS 03_CODE/docs/2026-11-18_Thiet_ke_rq3_u_moc_v4.md (Liam 18/11 at CODE_GATE, hop chon, nguyen van:
"Duyệt, code luôn (Recommended)"); the numbers in the test test names refer to its section 8. Every file here is FAKE
(made-up content, no measured number): the U files (u_moc_luot_v1, CODE_CAM), the premise files (tien_de_buoi_v1,
CODE_KIEMCHEO) and the C8 gate file with a camera.c8b lock (c8b_khoa_v4, CODE_C8B) follow the field names of the design.
"""
import csv
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import gate_c8  # noqa: E402
import rq3_log as rl  # noqa: E402

# CODE_GATE 19/11 (CODE's choice (a)): these tests build a gate_c8_v1 shifted-anchor lock, refused since the gate
# is gate_c8_v2; they wait for the campaign part (the one text below, grep CAMPAIGN_V2_TODO).
pytestmark = pytest.mark.skip(reason=rl.CAMPAIGN_V2_TODO)

LOCK_XY, LOCK_PSI = 0.010, 0.50          # FAKE lock numbers (m, deg, before the margin)
CP = dict(tag_offset=None, side_m=0.1445, h_m=0.20)


def sha(p) -> str:
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def wt(p, text: str) -> str:
    os.makedirs(os.path.dirname(str(p)), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(text)
    return str(p)


def wj(p, d) -> str:
    return wt(p, json.dumps(d))


class World:
    """A FAKE dataset: car runs of A-PID written straight into the tuning log, their H0 scorings, a C8 gate file."""

    def __init__(self, tmp, lock=None):
        self.tmp = tmp
        self.ds = str(tmp / 'ds')
        os.makedirs(self.ds)
        c = tmp / 'common'
        self.tap = wt(c / 'tap_moc.json', 'FAKE tap_moc')
        self.u_so = wt(c / 'u_so.json', 'FAKE u_so')
        self.ses = wt(c / 'session_Ha.json', 'FAKE session H_a')
        self.gate = self.gate_file(tmp / 'gate_c8.json', lock)
        self.rows, self.runs = [], {}

    def gate_file(self, p, lock=None, **md_over):
        g = gate_c8.fake_gate()
        md = dict(U_moc_xy_m=LOCK_XY, U_moc_psi_deg=LOCK_PSI, phuong_phap_duyet='FAKE', tien_de_buoi='dat',
                  nguon=dict(bo_lay_mau='blm_FAKE_v1', car_pose=dict(CP), tap_moc=sha(self.tap)))
        md.update(md_over)
        g['camera'] = {'c8b': {'moc_dich': md if lock is None else lock, 'khoa_sha256': 'a' * 64}}
        return wj(p, g)

    def save(self):
        rl._write_all(os.path.join(self.ds, rl.TUNING), rl.TUNING_COLS, self.rows)

    def run(self, it, session='S1', date='2026-11-20', outcome='dat', luoi=False, stage='screen', block='',
            so_h=True):
        uid = f'{session}:{it}'
        rd = self.tmp / 'runs' / uid.replace(':', '_')
        frames = wt(rd / 'frames.csv', f'FAKE frames {uid}')
        gio = None
        if luoi:                                       # RM2-5: the time file outside the results_dir
            gio = wj(self.tmp / 'gio' / f"{uid.replace(':', '_')}.json", {'sha256': {'frames': sha(frames)}})
            score = wj(rd / 'luoi' / 'luoi.json', {'gio_sha256': sha(gio), 'outcome': outcome})
        else:
            score = wj(rd / 'score.json', {'frames_sha256': sha(frames), 'outcome': outcome})
        wt(rd / 'lich.json', f'FAKE lich {uid}')
        wt(rd / 'dets.npz', f'FAKE dets {uid}')
        self.rows.append(dict(session_id=session, date=date, platform='HW', controller='PID', trajectory='O1-F05',
                              iter_no=str(it), run_uid=uid, campaign='A-PID', gain_set_id='A-PID/G1',
                              speed_mps='0.5', stage=stage, block_id=block, run_outcome=outcome, results_dir=str(rd),
                              gate_json_sha256=sha(score), gate_file_sha256=sha(self.gate),
                              wall_start=f'{date}T08:{it:02d}:00', wall_end=f'{date}T08:{it:02d}:30'))
        self.runs[uid] = dict(rd=rd, frames=frames, gio=gio, score=score)
        self.save()
        if so_h:                                       # RC1-2: a run is judged after its clip H comparison
            rl.append_row(os.path.join(self.ds, rl.SO_H), rl.SO_H_COLS,
                          dict(run_uid=uid, outcome_tam=outcome, outcome_sau=outcome, flags='', vuot='false',
                               D_H_mm='0.5', ghi_luc='2026-01-01T00:00:00'))
        return uid

    def u(self, uid, lan='H0', ok=True, xy=0.005, psi=0.20, frames=None, name=None, drop=(), **over):
        """A U file of scoring lan of run uid, under its results_dir."""
        run = self.runs[uid]
        rd = run['rd']
        fr = frames or run['frames']
        ng = [dict(vai_tro='frames', duong_dan=fr, sha256=sha(fr)),
              dict(vai_tro='lich', duong_dan='lich.json', sha256=sha(rd / 'lich.json')),
              dict(vai_tro='tap_moc', duong_dan=self.tap, sha256=sha(self.tap)),
              dict(vai_tro='u_so', duong_dan=self.u_so, sha256=sha(self.u_so)),
              dict(vai_tro='dets', duong_dan='dets.npz', sha256=sha(rd / 'dets.npz')),
              dict(vai_tro='session', duong_dan=self.ses, sha256=sha(self.ses))]
        if run['gio']:
            ng.append(dict(vai_tro='gio', duong_dan=run['gio'], sha256=sha(run['gio'])))
        ng = [e for e in ng if e['vai_tro'] not in drop]
        d = dict(format='u_moc_luot_v1', task='O1-F05', ok=ok, ly_do=[] if ok else ['khop thu loi (FAKE)'],
                 thieu=list(drop), sha256=dict(frames=sha(fr)), nguon=ng, U_moc_xy_m=xy if ok else None,
                 U_moc_psi_deg=psi if ok else None, bo_lay_mau='blm_FAKE_v1' if ok else None,
                 car_pose=dict(CP) if ok else None, n_khung=10, n_to_hop=200)
        d.update(over)
        return wj(rd / (name or f'u_moc_{lan}.json'), d)

    def td(self, session='S1', ngay='2026-11-20', kq='dat', luot=None, thieu=(), name=None, h1=False, **over):
        """A premise file of a session; luot: {id: frames path}, by default every run of the session."""
        if luot is None:
            luot = {f'L{k}': self.runs[uid]['frames'] for k, uid in enumerate(self.runs) if uid.startswith(session)}
        d = dict(format='tien_de_buoi_v1', session_id=session, buoi=session, ngay=ngay, ket_qua=kq, ly_do=[],
                 nguon=[dict(vai_tro='u_so', duong_dan=self.u_so, sha256=sha(self.u_so)),
                        dict(vai_tro='sessions', duong_dan=self.ses, sha256=sha(self.ses))],
                 sha256=dict(u_so=sha(self.u_so), sessions={'H_a': sha(self.ses)},
                             luot={k: (dict(frames=sha(p)) if isinstance(p, str) else
                                       dict(frames=sha(p[0]), lan_cham=p[1])) for k, p in luot.items()}),
                 luot=list(luot), luot_thieu_kiem=list(thieu), mau_thuan=[], tap_neo={}, h1_ho_tro=h1)
        d.update(over)
        return wj(self.tmp / 'td' / (name or f'{session}_{kq}.json'), d)

    def after(self):
        return {r['run_uid']: r for r in rl.apply_moc(rl.live_rows(rl.read_rows(os.path.join(self.ds, rl.TUNING))),
                                                        self.ds)}

    def moc(self):
        return {r['run_uid']: r for r in rl.apply_moc(rl.live_rows(rl.read_rows(os.path.join(self.ds, rl.TUNING))),
                                                      self.ds)}

    def drop_row(self, name, keep):
        p = os.path.join(self.ds, name)
        rows = [r for r in rl.read_rows(p) if keep(r)]
        rl._write_all(p, rl.U_MOC_COLS if name == rl.U_MOC else rl.TIEN_DE_COLS, rows)


@pytest.fixture
def w(tmp_path):
    return World(tmp_path)


def full(w, *uids, session='S1', **td):
    for uid in uids:
        rl.u_moc(w.ds, uid, 'H0', w.u(uid), w.gate)
    rl.tien_de(w.ds, session, w.td(session, **td))


# ── 1, 2: every condition, missing files ────────────────────────────────────────────────────────────────────────

def test_1_all_conditions_hold_keep_dat_and_khong_dat(w):
    a, b = w.run(1), w.run(2, outcome='khong_dat')
    full(w, a, b)
    got = w.after()
    assert (got[a]['run_outcome'], got[a]['moc_flags']) == ('dat', '')
    assert (got[b]['run_outcome'], got[b]['moc_flags']) == ('khong_dat', '')
    assert got[a]['U_moc_xy_m'] == '0.005'


def test_2_missing_files_never_keep_the_outcome(w):
    a = w.run(1)
    assert w.after()[a]['run_outcome'] == 'chua_ket_luan'
    assert w.after()[a]['moc_flags'] == 'cho_tien_de,cho_u_moc'
    full(w, a)
    assert w.after()[a]['run_outcome'] == 'dat'
    w.drop_row(rl.U_MOC, lambda r: False)                                  # RM1-1: deleting a row cannot raise it
    assert (w.after()[a]['run_outcome'], w.after()[a]['moc_flags']) == ('chua_ket_luan', 'cho_u_moc')
    w.drop_row(rl.TIEN_DE, lambda r: False)
    assert w.after()[a]['moc_flags'] == 'cho_tien_de,cho_u_moc'


def test_2_lost_and_inconclusive_runs_are_kept(w):
    a, b = w.run(1, outcome='hong_he_do'), w.run(2, outcome='chua_ket_luan')
    got = w.after()
    assert got[a]['run_outcome'] == 'hong_he_do' and got[b]['run_outcome'] == 'chua_ket_luan'
    assert 'moc_flags' not in got[a]


# ── session premise ─────────────────────────────────────────────────────────────────────────────────────────────

def test_premise_not_dat_lowers_every_run_null_schema_accepted(w):
    a, b = w.run(1), w.run(2)
    for uid in (a, b):
        rl.u_moc(w.ds, uid, 'H0', w.u(uid), w.gate)
    # RM2-3: the missing-data schema of S6 (sha256.luot, sha256.u_so null, no tung_khung)
    rl.tien_de(w.ds, 'S1', w.td(kq='thieu_du_lieu', sha256=dict(u_so=None, luot=None, sessions=None)))
    got = w.after()
    assert {got[a]['moc_flags'], got[b]['moc_flags']} == {'tien_de_thieu'}


def test_premise_khong_dat(w):
    a = w.run(1)
    full(w, a, kq='khong_dat')
    assert w.after()[a]['moc_flags'] == 'tien_de_khong_dat'


def test_5_premise_of_another_day(w):
    a = w.run(1, date='2026-11-21')
    full(w, a)                                           # premise ngay 2026-11-20
    assert w.after()[a]['moc_flags'] == 'tien_de_khac_ngay'


def test_5_a_run_outside_the_scope_lowers_the_whole_session(w):
    a = w.run(1)
    b = w.run(2)
    for uid in (a, b):
        rl.u_moc(w.ds, uid, 'H0', w.u(uid), w.gate)
    rl.tien_de(w.ds, 'S1', w.td(luot={'L0': w.runs[a]['frames']}))
    got = w.after()
    assert got[a]['moc_flags'] == got[b]['moc_flags'] == 'tien_de_pham_vi_thieu'


def test_5_luot_thieu_kiem_lowers_that_run_only(w):
    a, b = w.run(1), w.run(2)
    full(w, a, b, thieu=('L1',))
    got = w.after()
    assert (got[a]['run_outcome'], got[b]['moc_flags']) == ('dat', 'tien_de_luot_thieu')


def test_5_a_lost_run_with_a_scoring_counts_in_the_scope(w):
    a, c = w.run(1), w.run(2, outcome='hong_he_do')
    rl.u_moc(w.ds, a, 'H0', w.u(a), w.gate)
    rl.tien_de(w.ds, 'S1', w.td(luot={'L0': w.runs[a]['frames']}))
    assert w.after()[a]['moc_flags'] == 'tien_de_pham_vi_thieu'


def test_8d_scope_taken_from_the_log_deleting_a_u_row_does_not_shrink_it(w):
    a, b = w.run(1), w.run(2, luoi=True)
    for uid in (a, b):
        rl.u_moc(w.ds, uid, 'H0', w.u(uid), w.gate)
    rl.tien_de(w.ds, 'S1', w.td(luot={'L0': w.runs[a]['frames']}))
    assert w.after()[a]['moc_flags'] == 'tien_de_pham_vi_thieu'
    w.drop_row(rl.U_MOC, lambda r: r['run_uid'] != b)                    # RM3-1
    got = w.after()
    assert got[a]['moc_flags'] == 'tien_de_pham_vi_thieu'                # b cannot be resolved: still in the scope
    assert got[b]['moc_flags'] == 'khong_gan_duoc_lan_cham,tien_de_pham_vi_thieu,cho_u_moc'


def test_4_luoi_scoring_through_the_time_file_outside_the_results_dir(w):
    a, b = w.run(1), w.run(2, luoi=True)
    full(w, a, b)
    got = w.after()
    assert got[a]['run_outcome'] == got[b]['run_outcome'] == 'dat'


def test_4_luoi_scoring_needs_the_gio_source(w):
    b = w.run(1, luoi=True)
    with pytest.raises(rl.LogError, match='gio'):
        rl.u_moc(w.ds, b, 'H0', w.u(b, drop=('gio',)), w.gate)
    other = wj(w.tmp / 'gio' / 'other.json', {'sha256': {'frames': 'c' * 64}})
    with pytest.raises(rl.LogError, match='gio_sha256 differs'):
        ng = json.load(open(w.u(b)))['nguon']
        ng = [e if e['vai_tro'] != 'gio' else dict(e, duong_dan=other, sha256=sha(other)) for e in ng]
        rl.u_moc(w.ds, b, 'H0', w.u(b, name='u2.json', nguon=ng), w.gate)


def test_4_frames_sha256_and_time_file_must_agree(w):
    b = w.run(1, luoi=True)
    sc = json.load(open(w.runs[b]['score']))
    sc['frames_sha256'] = 'd' * 64
    with pytest.raises(rl.LogError, match='differ'):
        rl.frames_of(sc, ('gio', w.runs[b]['gio'], sha(w.runs[b]['gio'])), 'x')


# ── U of the run ────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('xy, psi, want', [(LOCK_XY, LOCK_PSI, ''), (0.0101, 0.2, 'u_moc_vuot'),
                                           (0.005, 0.51, 'u_moc_vuot'), (0.0, 0.0, '')])
def test_7_each_component_against_the_lock(w, xy, psi, want):
    a = w.run(1)
    rl.u_moc(w.ds, a, 'H0', w.u(a, xy=xy, psi=psi), w.gate)
    rl.tien_de(w.ds, 'S1', w.td())
    assert w.after()[a]['moc_flags'] == want


@pytest.mark.parametrize('bad', [float('nan'), -0.001, True, '0.005', None])
def test_7_invalid_u_numbers_refused(w, bad):
    a = w.run(1)
    with pytest.raises(rl.LogError):
        rl.u_moc(w.ds, a, 'H0', w.u(a, U_moc_xy_m=bad), w.gate)


def test_8_ok_false_null_branch(w):
    a = w.run(1)
    rl.u_moc(w.ds, a, 'H0', w.u(a, ok=False, drop=('u_so', 'session', 'dets')), w.gate)
    rl.tien_de(w.ds, 'S1', w.td())
    assert w.after()[a]['moc_flags'] == 'u_moc_khong_tinh'
    with pytest.raises(rl.LogError, match='ly_do'):
        rl.u_moc(w.ds, w.run(2), 'H0', w.u('S1:2', ok=False, ly_do=[]), w.gate)
    with pytest.raises(rl.LogError, match='sha256.frames'):
        rl.u_moc(w.ds, 'S1:2', 'H0', w.u('S1:2', ok=False, sha256={}), w.gate)
    with pytest.raises(rl.LogError, match='task'):
        rl.u_moc(w.ds, 'S1:2', 'H0', w.u('S1:2', ok=False, task=None), w.gate)


def test_8_ok_true_needs_every_source(w):
    a = w.run(1)
    with pytest.raises(rl.LogError, match='missing'):
        rl.u_moc(w.ds, a, 'H0', w.u(a, drop=('dets',)), w.gate)


@pytest.mark.parametrize('over, want', [
    (dict(car_pose=dict(CP, side_m=0.145)), 'u_moc_nguon_khac_khoa'),
    (dict(car_pose=dict(CP, tag_offset=[0.0, 0.0, 0.0])), ''),            # null = [0, 0, 0] (C8B 18/11)
    (dict(car_pose=dict(CP, tag_offset=[0.0, 0.001, 0.0])), 'u_moc_nguon_khac_khoa'),
    (dict(bo_lay_mau='blm_FAKE_v2'), 'u_moc_nguon_khac_khoa')])
def test_7_sources_against_the_lock(w, over, want):
    a = w.run(1)
    rl.u_moc(w.ds, a, 'H0', w.u(a, **over), w.gate)
    rl.tien_de(w.ds, 'S1', w.td())
    assert w.after()[a]['moc_flags'] == want


def test_7_another_tap_moc_file(w):
    a = w.run(1)
    tap2 = wt(w.tmp / 'common' / 'tap_moc_2.json', 'FAKE other tap_moc')
    ng = [e if e['vai_tro'] != 'tap_moc' else dict(e, duong_dan=tap2, sha256=sha(tap2))
          for e in json.load(open(w.u(a)))['nguon']]
    rl.u_moc(w.ds, a, 'H0', w.u(a, name='u2.json', nguon=ng), w.gate)
    rl.tien_de(w.ds, 'S1', w.td())
    assert w.after()[a]['moc_flags'] == 'u_moc_nguon_khac_khoa'


def test_8c_u_so_and_session_of_another_session(w):
    a, b = w.run(1), w.run(2)
    u2 = wt(w.tmp / 'common' / 'u_so_2.json', 'FAKE other u_so')
    ng = [e if e['vai_tro'] != 'u_so' else dict(e, duong_dan=u2, sha256=sha(u2)) for e in json.load(open(w.u(a)))['nguon']]
    rl.u_moc(w.ds, a, 'H0', w.u(a, name='u2.json', nguon=ng), w.gate)
    rl.u_moc(w.ds, b, 'H0', w.u(b), w.gate)
    # RM2-4: the session sha256 of b is a KEY of the mapping, not a value
    rl.tien_de(w.ds, 'S1', w.td(sha256=dict(u_so=sha(w.u_so), sessions={sha(w.ses): 'e' * 64},
                                            luot={f'L{k}': dict(frames=sha(w.runs[u]['frames']))
                                                  for k, u in enumerate((a, b))})))
    got = w.after()
    assert got[a]['moc_flags'] == got[b]['moc_flags'] == 'u_moc_khac_buoi'


def test_8c_sessions_must_be_a_mapping(w):
    w.run(1)
    with pytest.raises(rl.LogError, match='mapping'):
        rl.tien_de(w.ds, 'S1', w.td(sha256=dict(u_so=sha(w.u_so), sessions=[sha(w.ses)], luot={})))


# ── binding (RM1-2, RM1-4, RM1-5) ──────────────────────────────────────────────────────────────────────────────

def test_3_u_file_of_another_run_refused(w):
    a, b = w.run(1), w.run(2)
    # every source of the file exists and is unchanged, but its frames are those of run b (same task, same session)
    with pytest.raises(rl.LogError, match='not the frames of scoring H0'):
        rl.u_moc(w.ds, a, 'H0', w.u(a, frames=w.runs[b]['frames'], name='u_of_b.json'), w.gate)


def test_3_refusals_at_the_command(w):
    a = w.run(1)
    with pytest.raises(rl.LogError, match='results_dir'):
        rl.u_moc(w.ds, a, 'H0', wj(w.tmp / 'elsewhere.json', json.load(open(w.u(a)))), w.gate)
    with pytest.raises(rl.LogError, match='task'):
        rl.u_moc(w.ds, a, 'H0', w.u(a, task='O1-F02'), w.gate)
    with pytest.raises(rl.LogError, match='format'):
        rl.u_moc(w.ds, a, 'H0', w.u(a, format='u_moc_luot_v0'), w.gate)
    g2 = w.gate_file(w.tmp / 'gate2.json', U_moc_xy_m=0.02)
    with pytest.raises(rl.LogError, match='not the one of the run'):
        rl.u_moc(w.ds, a, 'H0', w.u(a), g2)
    with pytest.raises(rl.LogError, match='H1'):
        rl.u_moc(w.ds, a, 'H1', w.u(a), w.gate)
    rl.u_moc(w.ds, a, 'H0', w.u(a), w.gate)
    with pytest.raises(rl.LogError, match='already'):
        rl.u_moc(w.ds, a, 'H0', w.u(a), w.gate)


def test_3_gate_file_without_the_lock_refused(tmp_path):
    w = World(tmp_path, lock='x')
    w.gate = wj(tmp_path / 'g_old.json', dict(gate_c8.fake_gate()))      # camera {}: a lock before c8b_khoa_v4
    a = w.run(1)
    with pytest.raises(rl.LogError, match='moc_dich'):
        rl.u_moc(w.ds, a, 'H0', w.u(a), w.gate)


@pytest.mark.parametrize('what', ['lich', 'dets', 'tap_moc', 'u_so_td', 'u_file', 'td_file', 'gate'])
def test_6_a_changed_source_stops_the_apply(w, what):
    a = w.run(1)
    full(w, a)
    p = dict(lich=w.runs[a]['rd'] / 'lich.json', dets=w.runs[a]['rd'] / 'dets.npz', tap_moc=w.tap, u_so_td=w.u_so,
             u_file=w.runs[a]['rd'] / 'u_moc_H0.json', td_file=w.tmp / 'td' / 'S1_dat.json', gate=w.gate)[what]
    with open(p, 'a', encoding='utf-8') as f:
        f.write(' ')
    with pytest.raises(rl.LogError):
        w.after()


def test_lock_numbers_read_again(w):
    a = w.run(1)
    full(w, a)
    p = os.path.join(w.ds, rl.U_MOC)
    rows = rl.read_rows(p)
    rows[0]['khoa_xy_m'] = '0.5'
    rl._write_all(p, rl.U_MOC_COLS, rows)
    with pytest.raises(rl.LogError, match='lock numbers'):
        w.after()


@pytest.mark.parametrize('md_over', [dict(U_moc_xy_m=True), dict(U_moc_psi_deg=float('inf')),
                                     dict(U_moc_xy_m=-1.0), dict(nguon=dict(bo_lay_mau='', car_pose=CP, tap_moc='a' * 64)),
                                     dict(nguon=dict(bo_lay_mau='b', car_pose=dict(CP, tag_offset=[0, 0]),
                                                     tap_moc='a' * 64)),
                                     dict(nguon=dict(bo_lay_mau='b', car_pose=dict(CP, side_m=0), tap_moc='a' * 64))])
def test_moc_dich_of_refuses_bad_locks(w, md_over):
    g = gate_c8.read_gate_c8(w.gate_file(w.tmp / 'gbad.json', **md_over))
    with pytest.raises(gate_c8.GateFileError):
        gate_c8.moc_dich_of(g)


# ── H1 (RM2-1, RM3-2, RM4-1) ───────────────────────────────────────────────────────────────────────────────────

def h1_setup(w, uid, h1_in_luot=True, thieu=(), xy_h1=0.005):
    """so-h above eps_H, H0 and H1 judged dat (so-h row written directly: apply_moc is tested alone here)."""
    rd = w.runs[uid]['rd']
    f1 = wt(rd / 'h1' / 'frames.csv', f'FAKE frames H1 {uid}')
    sf = wj(rd / 'score_final.json', {'outcome': 'dat', 'calib_drift': {'h1_score': {'frames_sha256': sha(f1)},
                                                                         'vuot': True, 'thieu_H1': False}})
    rl.append_row(os.path.join(w.ds, rl.SO_H), rl.SO_H_COLS,
                  dict(run_uid=uid, outcome_tam='dat', outcome_sau='dat', flags='calib_drift', vuot='true',
                       D_H_mm='3', score_final_file=sf, score_final_sha256=sha(sf), ghi_luc='2026-11-20T10:00:00'))
    w.runs[uid]['f1'] = f1
    rl.u_moc(w.ds, uid, 'H0', w.u(uid), w.gate)
    rl.u_moc(w.ds, uid, 'H1', w.u(uid, lan='H1', frames=f1, xy=xy_h1), w.gate)
    luot = {f'L{k}': w.runs[u]['frames'] for k, u in enumerate(w.runs)}
    if h1_in_luot:
        luot[f"L{list(w.runs).index(uid)}@H1"] = (f1, 'H1')
    rl.tien_de(w.ds, 'S1', w.td(luot=luot, thieu=thieu, h1=h1_in_luot))


def test_8b_h1_covered_and_below_the_lock_keeps_dat(w):
    a, b = w.run(1, so_h=False), w.run(2)
    rl.u_moc(w.ds, b, 'H0', w.u(b), w.gate)
    h1_setup(w, a)
    got = w.moc()
    assert got[a]['run_outcome'] == 'dat' and got[b]['run_outcome'] == 'dat'


def test_8b_h1_above_the_lock(w):
    a = w.run(1, so_h=False)
    h1_setup(w, a, xy_h1=0.02)
    assert w.moc()[a]['moc_flags'] == 'u_moc_vuot'
    assert w.moc()[a]['U_moc_xy_m'] == '0.02'


def test_8b_h1_not_covered_lowers_that_run_only(w):
    a, b = w.run(1, so_h=False), w.run(2)
    rl.u_moc(w.ds, b, 'H0', w.u(b), w.gate)
    h1_setup(w, a, h1_in_luot=False)
    got = w.moc()
    assert (got[a]['moc_flags'], got[b]['run_outcome']) == ('tien_de_khong_phu_h1', 'dat')


def test_8b_h1_id_in_luot_thieu_kiem(w):
    a, b = w.run(1, so_h=False), w.run(2)
    rl.u_moc(w.ds, b, 'H0', w.u(b), w.gate)
    h1_setup(w, a, thieu=('L0@H1',))
    got = w.moc()
    assert (got[a]['moc_flags'], got[b]['run_outcome']) == ('tien_de_luot_thieu', 'dat')


def test_8b_h1_row_missing_and_so_h_row_removed(w):
    a = w.run(1, so_h=False)
    h1_setup(w, a)
    w.drop_row(rl.U_MOC, lambda r: r['lan'] != 'H1')
    assert w.moc()[a]['moc_flags'] == 'cho_u_moc'
    rl._write_all(os.path.join(w.ds, rl.SO_H), rl.SO_H_COLS, [])
    assert w.moc()[a]['moc_flags'] == 'cho_so_h'         # RC1-2: no so-h row, the scorings needed are unknown


def test_8b_h1_row_without_so_h_stops(w):
    a = w.run(1, so_h=False)
    h1_setup(w, a)
    rl._write_all(os.path.join(w.ds, rl.SO_H), rl.SO_H_COLS, [])
    with pytest.raises(rl.LogError, match='H1'):
        w.moc()


# ── summary, freeze, s05_final_table, begin ─────────────────────────────────────────────────────────────────────

def test_10_summary_counts_and_gate_file(w, monkeypatch):
    # the row checks of R134 / O1 v11.33 (boot answers, gate columns) are tested elsewhere; these FAKE rows lack them
    monkeypatch.setattr(rl, 'check_query_rows', lambda tun, ds: None)
    monkeypatch.setattr(rl, 'check_gate_rows', lambda tun, gate_file=None: None)
    monkeypatch.setattr(rl, 'apply_so_h', lambda tun, ds: tun)        # raw so-h rows (World); real ones: demo tests
    a, b = w.run(1), w.run(2)
    rl.u_moc(w.ds, a, 'H0', w.u(a, xy=0.02), w.gate)                  # lowered
    rl.tien_de(w.ds, 'S1', w.td())                                     # b waits for its U file
    row = {r['campaign']: r for r in rl.summary(w.ds)}['A-PID']
    assert (row['moc_ha_n'], row['moc_cho_n']) == ('1', '1')
    g2 = rl.read_gate_file(w.gate_file(w.tmp / 'gate2.json', U_moc_xy_m=0.02))
    with pytest.raises(rl.LogError, match='another gate file'):
        rl.summary(w.ds, gate_file=g2)


def test_11_freeze_confirm_block(w, monkeypatch):
    monkeypatch.setattr(rl, 'apply_so_h', lambda tun, ds: tun)        # so-h rows written directly below
    a = w.run(1, stage='confirm', block='B1')                         # its so-h row: ghi_luc 2026-01-01
    gain = wt(w.tmp / 'gain_A_PID_G1.json', '{"FAKE": 1}')
    with pytest.raises(rl.LogError, match='not all dat'):             # no U / premise files: chua_ket_luan
        rl.freeze(w.ds, 'A-PID', 'khoi_xac_nhan_dat', 'A-PID/G1', gain, at='2099-01-01T00:00:00')
    full(w, a)
    with pytest.raises(rl.LogError, match='u-moc / tien-de'):           # u-moc / tien-de rows written now
        rl.freeze(w.ds, 'A-PID', 'khoi_xac_nhan_dat', 'A-PID/G1', gain, at='2026-01-02T00:00:00')
    rl.freeze(w.ds, 'A-PID', 'khoi_xac_nhan_dat', 'A-PID/G1', gain, at='2099-01-01T00:00:00')


def test_12_final_table_uses_the_outcome_after_both(w, monkeypatch):
    import s05_final_table as s5
    w.run(1, stage='final', outcome='hong_he_do')
    seen = []

    def spy(tun, ds):
        seen.append(len(tun))
        raise rl.LogError('SPY')
    monkeypatch.setattr(rl, 'apply_after', spy)
    with pytest.raises(ValueError, match='SPY'):
        s5.load_runs(w.ds, 'score.json', 'esp32.csv', gate_file=w.gate)
    assert seen == [1]


def test_12_begin_does_not_use_the_rule(w, monkeypatch):
    monkeypatch.setattr(rl, 'apply_so_h', lambda tun, ds: tun)        # raw so-h rows (World)
    a = w.run(1)
    rl.u_moc(w.ds, a, 'H0', w.u(a, xy=0.02), w.gate)                  # a lowered by the rule
    run = rl.begin(w.ds, 'S1', 'A-PID', 'HW', 'PID', 'O1-F05', '0.5', 'screen', gain_set='A-PID/G1')
    assert run['row']['run_uid'] == 'S1:2'


def test_cli_u_moc_and_tien_de(w, capsys):
    a = w.run(1)
    assert rl.main(['--dataset', w.ds, 'u-moc', '--run-uid', a, '--file', w.u(a), '--gate-file', w.gate]) == 0
    assert rl.main(['--dataset', w.ds, 'tien-de', '--session', 'S1', '--file', w.td()]) == 0
    assert rl.main(['--dataset', w.ds, 'tien-de', '--session', 'S1', '--file', w.td()]) == 2
    with open(os.path.join(w.ds, rl.U_MOC), newline='', encoding='utf-8') as f:
        assert next(csv.reader(f)) == rl.U_MOC_COLS
    assert w.after()[a]['run_outcome'] == 'dat'


def test_s6_10_luot_is_the_list_of_ids(w):
    a = w.run(1)
    rl.u_moc(w.ds, a, 'H0', w.u(a), w.gate)
    good = json.load(open(w.td()))
    with pytest.raises(rl.LogError, match='list of run ids'):              # a mapping instead of the list
        rl.tien_de(w.ds, 'S1', wj(w.tmp / 'td' / 'x1.json', dict(good, luot={'L0': 1})))
    with pytest.raises(rl.LogError, match='every id of luot'):             # an id without frames
        rl.tien_de(w.ds, 'S1', wj(w.tmp / 'td' / 'x2.json', dict(json.load(open(w.td())), luot=['L0', 'L9'])))


def test_s6_10_h1_not_supported_lowers_every_run_needing_h1(w):
    """TD3-1 of CODE_KIEMCHEO: h1_ho_tro false, even with the H1 frames listed, every run needing H1 is lowered."""
    a, b = w.run(1, so_h=False), w.run(2)
    rl.u_moc(w.ds, b, 'H0', w.u(b), w.gate)
    h1_setup(w, a)
    p = os.path.join(w.ds, rl.TIEN_DE)
    td_file = rl.read_rows(p)[0]['file']
    d = json.load(open(td_file))
    rl._write_all(p, rl.TIEN_DE_COLS, [])
    rl.tien_de(w.ds, 'S1', wj(w.tmp / 'td' / 'no_h1.json', dict(d, h1_ho_tro=False)))
    got = w.moc()
    assert (got[a]['moc_flags'], got[b]['run_outcome']) == ('tien_de_khong_phu_h1', 'dat')


def test_s6_10_h1_frames_under_the_h0_id_do_not_cover_h1(w):
    """TD3-1: the H1 scoring with the same frames sha256 as H0 (one id in luot) is not an H1 coverage, h1_ho_tro or not."""
    a = w.run(1, so_h=False)
    rd, f0 = w.runs[a]['rd'], w.runs[a]['frames']
    sf = wj(rd / 'score_final.json', {'outcome': 'dat', 'calib_drift': {'h1_score': {'frames_sha256': sha(f0)}}})
    rl.append_row(os.path.join(w.ds, rl.SO_H), rl.SO_H_COLS,
                  dict(run_uid=a, outcome_tam='dat', outcome_sau='dat', flags='calib_drift', vuot='true',
                       D_H_mm='3', score_final_file=sf, score_final_sha256=sha(sf), ghi_luc='2026-11-20T10:00:00'))
    rl.u_moc(w.ds, a, 'H0', w.u(a), w.gate)
    rl.u_moc(w.ds, a, 'H1', w.u(a, lan='H1'), w.gate)
    rl.tien_de(w.ds, 'S1', w.td(h1=True))                       # luot = [L0] with the frames of H0 = H1
    assert w.moc()[a]['moc_flags'] == 'tien_de_khong_phu_h1'


def test_rc1_3_h1_record_needs_lan_cham_h1(w):
    """RC1-3: a second id with the H1 frames but no lan_cham is a record of H0, not a coverage of H1."""
    a = w.run(1, so_h=False)
    h1_setup(w, a, h1_in_luot=False)
    p = os.path.join(w.ds, rl.TIEN_DE)
    d = json.load(open(rl.read_rows(p)[0]['file']))
    rl._write_all(p, rl.TIEN_DE_COLS, [])
    d['luot'].append('X')
    d['sha256']['luot']['X'] = dict(frames=sha(w.runs[a]['f1']))       # no lan_cham: H0
    d['h1_ho_tro'] = True
    rl.tien_de(w.ds, 'S1', wj(w.tmp / 'td' / 'h1_no_lan.json', d))
    assert w.moc()[a]['moc_flags'] == 'tien_de_khong_phu_h1'


def test_rc1_4_gate_sha_of_every_u_row(w):
    a, b = w.run(1), w.run(2)
    full(w, a, b)
    p = os.path.join(w.ds, rl.U_MOC)
    rows = rl.read_rows(p)
    rows[1]['gate_file_sha256'] = 'b' * 64
    rl._write_all(p, rl.U_MOC_COLS, rows)
    with pytest.raises(rl.LogError, match='gate_file_sha256 of the row'):
        w.after()


# ── the demo: the whole chain with the real so-h (RC1-1, RC1-2, RC1-6) ─────────────────────────────────────────

@pytest.fixture
def demo(tmp_path):
    d = str(tmp_path / 'dry')
    rl.demo(d)
    return d


def _after(ds):
    return {r['run_uid']: r for r in rl.apply_after(rl.live_rows(rl.read_rows(os.path.join(ds, rl.TUNING))), ds)}


def _rows_of(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _drop(ds, name, cols, keep):
    rl._write_all(os.path.join(ds, name), cols, [r for r in rl.read_rows(os.path.join(ds, name)) if keep(r)])


def _final_dat(ds):
    return next(r for r in rl.read_rows(os.path.join(ds, rl.TUNING))
                if r['session_id'] == 'FAKE-F1' and r['run_outcome'] == 'dat')


def _table(ds, out):
    import s05_final_table as T
    os.makedirs(out)
    assert T.main(['table', '--dataset', ds, '--out', out, '--gate-file', os.path.join(ds, 'FAKE_gate_c8.json'), '--gate-file-fake']) == 0
    return {r['run_uid']: r for r in _rows_of(os.path.join(out, 'S05_final_runs.csv'))}


def test_demo_chain_keeps_every_judged_car_run(demo):
    got = _after(demo)
    car = [r for r in got.values() if r['platform'] == 'HW' and 'moc_flags' in r]
    # FAKE-A1:2 (khong_dat), the final dat run; FAKE-A1:3 is chua_ket_luan at so-h (split), not under the rule
    assert len(car) == 2 and all(r['moc_flags'] == '' and r['U_moc_xy_m'] == '0.004' for r in car)
    rows = {r['campaign']: r for r in _rows_of(os.path.join(demo, rl.SUMMARY))}
    assert all((r['moc_cho_n'], r['moc_ha_n']) == ('0', '0') for r in rows.values())


def test_demo_final_table_exports_the_outcome_flags_and_u(demo, tmp_path):
    uid = _final_dat(demo)['run_uid']
    t = _table(demo, str(tmp_path / 't1'))
    assert (t[uid]['outcome'], t[uid]['moc_flags'], t[uid]['U_moc_xy_m']) == ('dat', '', '0.004')
    _drop(demo, rl.TIEN_DE, rl.TIEN_DE_COLS, lambda r: r['session_id'] != 'FAKE-F1')
    t = _table(demo, str(tmp_path / 't2'))
    assert (t[uid]['outcome'], t[uid]['moc_flags']) == ('chua_ket_luan', 'cho_tien_de')


def test_demo_rc1_1_vuot_edited_by_hand_stops(demo):
    uid = _final_dat(demo)['run_uid']
    p = os.path.join(demo, rl.SO_H)
    rows = rl.read_rows(p)
    for r in rows:
        if r['run_uid'] == uid:
            r['vuot'] = 'true'
    rl._write_all(p, rl.SO_H_COLS, rows)
    with pytest.raises(rl.LogError, match='vuot'):
        _after(demo)


def test_demo_rc1_2_so_h_row_removed_waits(demo):
    uid = _final_dat(demo)['run_uid']
    _drop(demo, rl.SO_H, rl.SO_H_COLS, lambda r: r['run_uid'] != uid)
    got = _after(demo)[uid]
    assert (got['run_outcome'], got['moc_flags']) == ('chua_ket_luan', 'cho_so_h')
    row = {r['campaign']: r for r in rl.summary(demo, gate_file=rl.read_gate_file(
        os.path.join(demo, 'FAKE_gate_c8.json')))}['A-PID']
    assert row['moc_cho_n'] == '1'


def test_demo_real_so_h_above_eps_needs_and_checks_h1(demo):
    """so-h above eps_H with H0 and H1 both khong_dat (real so-h, checked by check_score_final): U of H1 needed;
    with it, the premise (h1_ho_tro false) does not cover H1, so the run is chua_ket_luan (S6-10b TD3-1)."""
    uid = 'FAKE-A1:2'
    r = next(x for x in rl.read_rows(os.path.join(demo, rl.TUNING)) if x['run_uid'] == uid)
    d = r['results_dir']
    gf = rl.read_gate_file(os.path.join(demo, 'FAKE_gate_c8.json'))
    f1 = rl.fake_frames(os.path.join(d, 'h1'), 'H1 ' + uid)
    h1 = dict(FAKE=True, outcome='khong_dat', frames_sha256=sha(f1), gate_v3=rl.floor_gate(rl.fake_gate(
        e_trk=0.5, gate_file=gf, level=r['trajectory'], stage='screen')))
    _drop(demo, rl.SO_H, rl.SO_H_COLS, lambda x: x['run_uid'] != uid)
    rl.so_h(demo, uid, rl.fake_score_final(d, 3.0, h1=h1, name='sf_h1.json'))
    got = _after(demo)[uid]
    assert (got['run_outcome'], got['moc_flags']) == ('chua_ket_luan', 'tien_de_khong_phu_h1,cho_u_moc')
    common = {k: os.path.join(demo, 'FAKE_moc', f'FAKE_{k}.json') for k in ('tap_moc', 'u_so', 'session')}
    rl.u_moc(demo, uid, 'H1', rl.fake_u_moc(d, r['trajectory'], common, lan='H1', frames=f1, xy=0.02),
             os.path.join(demo, 'FAKE_gate_c8.json'))
    got = _after(demo)[uid]
    assert got['moc_flags'] == 'tien_de_khong_phu_h1,u_moc_vuot' and got['U_moc_xy_m'] == '0.02'


# ── S05 v3.44 (O1 v11.40): caps and clip H hours (CODE 18/11) ─────────────────────────────────────────────────────

def _summary_rows(ds):
    gf = rl.read_gate_file(os.path.join(ds, 'FAKE_gate_c8.json'))
    return {r['campaign']: r for r in rl.summary(ds, gate_file=gf)}


def test_s05_h4_h5_lowered_runs_stay_in_the_caps(demo):
    """S05 v3.44 §4.3 H4-H5: a run made chua_ket_luan by the premise / anchor rule still counts in the car-run and
    hour caps (§4.1), like any chua_ket_luan run."""
    keep = ('N_HW', 'N_HW_tune', 'N_hong', 'T_operator_h', 'T_tune_no_hoc_h', 'cap_hit')
    before = {k: _summary_rows(demo)['A-PID'][k] for k in keep}
    _drop(demo, rl.TIEN_DE, rl.TIEN_DE_COLS, lambda r: r['session_id'] != 'FAKE-A1')
    after = _summary_rows(demo)['A-PID']
    assert after['moc_cho_n'] == '1'                                   # FAKE-A1:2 waits for its premise file
    assert {k: after[k] for k in keep} == before


def test_s05_h6_clip_h_hours_in_t_and_apart(demo):
    """S05 v3.44 §3.2 H6: an effort row of clip H filming (ghi_chu 'clip_h ...', the campaign code) is in T_operator
    and T_tune like any effort row, and given apart in T_clip_h_h; a B-chung row is split like the others."""
    b = _summary_rows(demo)
    rl.effort(demo, '2026-11-20', '07:00', '07:15', campaign='A-PID', ghi_chu='clip_h FAKE 4 clip H buoi FAKE-A1')
    rl.effort(demo, '2026-11-20', '07:15', '07:21', campaign='A-PID', ghi_chu='FAKE other work')
    rl.effort(demo, '2026-11-20', '12:00', '12:30', campaign='B-chung', ghi_chu='clip_h FAKE buoi chung')
    a = _summary_rows(demo)
    assert a['A-PID']['T_clip_h_h'] == '0.250' and b['A-PID']['T_clip_h_h'] == '0.000'
    for k in ('T_operator_h', 'T_tune_no_hoc_h'):
        assert abs(float(a['A-PID'][k]) - float(b['A-PID'][k]) - 0.35) < 1e-9, k
    assert a['B-chung']['T_clip_h_h'] == '0.500'
    assert float(a['B-PID']['T_clip_h_h']) + float(a['B-LADRC']['T_clip_h_h']) == 0.5


def test_rs1_1_clip_h_is_never_hoc(demo):
    """RS1-1: clip H filming counts in the hour cap, so a clip_h row cannot carry the hoc label."""
    with pytest.raises(rl.LogError, match='not hoc'):
        rl.effort(demo, '2026-11-20', '07:00', '07:15', campaign='A-PID', nhan='hoc', ghi_chu='clip_h FAKE')
    with pytest.raises(rl.LogError, match='not hoc'):
        rl.hoc_start(demo, 'A-PID', 'clip_h FAKE')
    p = os.path.join(demo, rl.EFFORT)
    rows = rl.read_rows(p)
    rows.append(dict(rows[0], chien_dich='A-PID', khoan='', nhan='hoc', ghi_chu=f'{rl.MANUAL} clip_h FAKE by hand'))
    rl._write_all(p, rl.EFFORT_COLS, rows)
    with pytest.raises(rl.LogError, match='cannot be hoc'):
        _summary_rows(demo)


def test_rs1_2_short_effort_rows_keep_their_time(demo):
    """RS1-2 (Liam 18/11 "Sửa luôn"): gio with 4 decimals (0.36 s); four 1-minute clip_h rows give the hours of one
    4-minute row within the quantum (0.18 s per row; RS3-1), a 5 s row no longer vanishes (0.01 h was 36 s per row)."""
    b = _summary_rows(demo)['A-PID']
    for m in range(4):
        r = rl.effort(demo, '2026-11-20', f'07:0{m}', f'07:0{m + 1}', campaign='A-PID', ghi_chu=f'clip_h FAKE {m}')
        assert r['gio'] == '0.0167'
    rl.effort(demo, '2026-11-20', '07:10:00', '07:10:05', campaign='A-PID', ghi_chu='clip_h FAKE 5 s')
    a = _summary_rows(demo)['A-PID']
    assert a['T_clip_h_h'] == '0.068'                                   # 4 x 0.0167 + 0.0014 h
    assert abs(float(a['T_tune_no_hoc_h']) - float(b['T_tune_no_hoc_h']) - 0.068) < 0.0015


def test_rs3_1_split_and_joined_rows_agree_within_the_quantum(tmp_path):
    """RS3-1: the same 4 minutes as four rows or as one row; the stored gio sums differ by at most 0.18 s per row."""
    d = str(tmp_path / 'e')
    os.makedirs(d)
    split = [float(rl.effort(d, '2026-11-20', f'07:0{m}', f'07:0{m + 1}', campaign='A-PID', ghi_chu='clip_h x')['gio'])
             for m in range(4)]
    joined = float(rl.effort(d, '2026-11-20', '08:00', '08:04', campaign='A-PID', ghi_chu='clip_h y')['gio'])
    assert abs(sum(split) - 4 / 60) * 3600 <= 0.18 * 4 + 1e-9
    assert abs(joined - 4 / 60) * 3600 <= 0.18 + 1e-9
    assert abs(sum(split) - joined) * 3600 <= 0.18 * 5 + 1e-9
