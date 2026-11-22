"""c8b_gio for K (no turn command): channels judged by the command of the task, U~ from F runs of the same part
(Liam 22/11 at CODE, "Can rong tu luot F (Recommended)"; Fable (ii)(b) b3)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c8b_gio as G  # noqa: E402
import test_c8b as T  # noqa: E402


def _straight(t):
    s = T._ramp(t, 3.0, 4.5) * (1 - T._ramp(t, 22.0, 23.5))
    return 0.5 * s * (1 + 0.3 * np.sin(0.7 * t)), 0.0 * t          # speed only, no turn (K)


@pytest.fixture(scope='module')
def k_run(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(T, 'motion', _straight)
    d = str(tmp_path_factory.mktemp('k'))
    fp, lp = T.make_run(d)
    mp.undo()
    return fp, lp


def _f_file(tmp_path, name, delta, ctrl='PID', task='O1-F05', ket='ok'):
    D = 100.0
    g = dict(format=G.FORMAT, task=task, ket_luan=ket, controller=ctrl, U_s=D + delta,
             kenh=dict(speed_cnt=dict(D_s=D)))
    p = tmp_path / f'{name}.json'
    p.write_text(json.dumps(g), encoding='utf-8')
    return str(p)


def test_kenh_theo_bai_by_command():
    assert G.kenh_theo_bai('O1-K1') == G.kenh_theo_bai('O1-K2') == ('speed_cnt',)
    assert G.kenh_theo_bai('O1-F05') == G.CHANNELS and G.kenh_theo_bai('O1-F08') == G.CHANNELS


def test_k_without_f_runs_is_undecided_not_lost(k_run):
    fp, lp = k_run
    out, _, _ = G.analyse(fp, lp, 'O1-K1')
    assert out['kenh_xet'] == ['speed_cnt'] and out['kenh']['speed_cnt']['ok']
    assert all(out['kenh'][c]['xet'] is False and 'khong_xet' in out['kenh'][c] for c in ('wz_cnt', 'gyro_rx', 'rv_rx'))
    assert out['ket_luan'] == 'chua_ket_luan' and out['ly_do_hong'] == [] and out['nodes_s'] == []
    assert any('no upper clock bound' in w for w in out['ly_do_chua'])
    assert out['dem_truot_kiem_gio'] is False and out['kiem_gio_d1']['trang_thai'] == 'khong_kiem'
    assert out['L_kenh'] == 'speed_cnt' and out['U_s'] is None


def test_k_with_f_runs_u_from_f(k_run, tmp_path):
    fp, lp = k_run
    f1, f2 = _f_file(tmp_path, 'a', 0.0265), _f_file(tmp_path, 'b', 0.0198, task='O1-F02')
    u = G.u_tu_luot_f([f1, f2], 'PID')
    assert u[0] == pytest.approx(0.0265) and len(u[1]) == 2
    out, _, _ = G.analyse(fp, lp, 'O1-K1', u_tu_f=u)
    L = out['kenh']['speed_cnt']['D_s']
    assert out['ok'] and out['ket_luan'] == 'ok' and out['L_kenh'] == 'speed_cnt' and out['U_kenh'] == 'uoc_luong_F'
    assert out['L_s'] == L and out['U_s'] == pytest.approx(L + 0.0265)
    assert out['half_s'] == pytest.approx(G.LE * 0.5 * 0.0265) and out['nodes_s']
    assert out['gyro_minus_rv_s'] is None and out['uoc_luong_gia_dinh']['delta_s'] == pytest.approx(0.0265)
    assert out['kiem_gio_d1']['trang_thai'] == 'dat' and out['dem_truot_kiem_gio'] is False


def test_f_task_unchanged(k_run):
    # an F task keeps the four channels judged: the same no-turn run as O1-F05 is lost on the turn channels
    fp, lp = k_run
    out, _, _ = G.analyse(fp, lp, 'O1-F05')
    assert out['kenh_xet'] == list(G.CHANNELS) and out['ket_luan'] == 'hong_he_do'
    assert any(w.startswith('wz_cnt') or w.startswith('gyro_rx') or w.startswith('rv_rx') for w in out['ly_do_hong'])


@pytest.mark.parametrize('bad', [dict(ket='hong_he_do'), dict(ctrl='ADRC'), dict(task='O1-K2')])
def test_u_tu_luot_f_refuses(tmp_path, bad):
    with pytest.raises(G.GioError):
        G.u_tu_luot_f([_f_file(tmp_path, 'x', 0.02, **bad)], 'PID')


def test_cli_u_tu_luot_f_only_for_k(base_run, tmp_path):
    _, fp, lp = base_run
    f1 = _f_file(tmp_path, 'a', 0.0265, ctrl=None)
    assert G.main(['--frames', fp, '--esp32', lp, '--task', 'O1-F05', '--u-tu-luot-F', f1,
                   '--out', str(tmp_path / 'g.json')]) == 2


def test_cli_k(k_run, tmp_path):
    fp, lp = k_run
    ctrl, _ = G.controller_of(lp)
    f1 = _f_file(tmp_path, 'a', 0.0265, ctrl=ctrl)
    out = tmp_path / 'g.json'
    assert G.main(['--frames', fp, '--esp32', lp, '--task', 'O1-K2', '--u-tu-luot-F', f1, '--out', str(out)]) == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    assert d['U_kenh'] == 'uoc_luong_F' and d['ket_luan'] == 'ok' and any(n['file'] == 'a.json' for n in d['nguon'])


base_run = T.base_run
