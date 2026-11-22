"""mo_hinh_con.py: spectral radius of the one-axis position sub-model of one gain set (report only).

Spec S04 v6.8 section 6, bullet [v6.8, cau 1] (06_SPECS/2026-11-05_Spec_S04_GA_chon_gain.md line 111):
before a gain set goes to HIL, the spectral radius of the sub-model of the whole set is written to
meta.yaml, and a value >= 1 is reported to Liam. It is not a constraint of the GA: the sub-model is
linear, without saturation and clamps.

The sub-model is the one of "V6" = 03_CODE/docs/review/2026-11-21_Fable_API_can_cu_nhom1/vong6/
ket_luan_da_kiem.md of the thesis repo, section a.1 (lines 24-32): one translational axis (x or y),
linear, discrete at the tick T = 10 ms, no saturation, loop delay 0. The wz axis and the heading loop are
not in it. The firmware lines quoted in the code are of AGV_FIRMWARE/src (controller/pos_ctrl.c,
vel_ctrl.c, pid.c, adrc.c, encoder/encoder.c), as read on 22/11/2026.

Plant (V6 a.1 line 29): first order, tau*dv/dt = -v + G*u, zero-order hold on u, so over one tick
    v[k+1]     = a*v[k] + G*(1 - a)*u[k]              a = exp(-T/tau)
    v_enc[k+1] = alpha*v[k] + G*(1 - alpha)*u[k]      alpha = tau*(1 - a)/T
v_enc is the mean velocity over the tick: the encoder speed is the count difference of one tick
(encoder.c:116-132), so the position sum x += v_enc*T is the true wheel position at the tick. V6 does
not write the formula of alpha; it follows from "mean over one tick" and gives the a = 0.94321,
alpha = 0.97133 printed in vong3/ket_luan_da_kiem.md line 66 (tau = 0.018671/0.10916 there).

State right after the control step of tick k, reference 0 (so the position error is -x):
    PID : x, v, i_pos, i_vel, u      5 states (V6 a.7 line 125)
    ADRC: x, v, i_pos, z1, z2, u     6 states
An integrator whose gain is exactly 0 does not act on the loop and is left out: pos_ki = 0 drops i_pos,
PID ki_xy = 0 drops i_vel, pos_kp = pos_ki = 0 drops x as well (what is left is the velocity loop,
V6 a.1 line 36 "bo vong vi tri"). Kept in, each would add an eigenvalue 1 that is not a mode of the
loop; V6 a.2 (pos_ki = 0) and a.4 (ki_xy = 0) print finite limits there, which needs them left out.
V6 does not state this rule (BAO_CAO.md, "Diem mo ho").

Gains: the 7 names of tools/gain_table.py NAMES (aliases as there), each rounded to float32, the value
the ESP32 gets (tools/gain_send.py); the matrix is then built in float64. Used: PID kp_xy, ki_xy,
pos_kp, pos_ki; ADRC omega_c_xy, omega_o_xy, pos_kp, pos_ki. The other three only have to be finite
numbers. Fixed as in the firmware: PID kd = 0 (pid.c:14), LADRC b0 = 5.85 (adrc.c:33).
The range table is not checked here (tools/gain_table.py check does that).

  python tools/mo_hinh_con.py --ctrl PID --gains "{\\"kp_xy\\": 1.5, \\"ki_xy\\": 0.05, \\"kp_wz\\": 1.0,
      \\"ki_wz\\": 0.05, \\"pos_kp\\": 6, \\"pos_ki\\": 3, \\"k_psi\\": 1}"
  python tools/mo_hinh_con.py --ctrl ADRC --gains path/to/gains.json [--G 1.127] [--tau 0.171]

--gains: the JSON text of an object (first non-blank character '{'), otherwise the path of a JSON
file. The object is flat {name: value} (then --ctrl is needed) or has the gain file shape of
tools/gain_send.py {"ctrl": "PID", "gains": {name: value}, ...}; --ctrl may be left out when the
object has "ctrl" and must agree with it when both are given. A JSON key given twice is refused.
Output, ASCII, one key=value per line:
    ctrl=PID
    G=1.0
    tau=0.171
    bien_trang_thai=x,v,i_pos,i_vel,u
    ban_kinh_pho=0.9998000399488338      repr of the float64, reads back to the same value
    trang_thai=on_dinh                   ban_kinh_pho < 1; khong_on_dinh when >= 1
Exit code 0 for both states (the state is reported, it is not a constraint); 1, with 'error: ...' on
stderr and no output line, when the arguments, the JSON or the gain set are refused.
"""
import argparse
import json
import math
import numbers
import sys

import numpy as np

if __package__:
    from . import gain_table                 # run as -m tools.mo_hinh_con (tools is a namespace package)
else:
    import gain_table                        # run as tools/mo_hinh_con.py

T_TICK = 0.01        # [s] control tick (V6 a.1 line 24)
TAU = 0.171          # [s] time constant of the x, y axis (V6 a.1 line 29; adrc.c:16)
G_MAC_DINH = 1.0     # static gain u -> v (V6 a.1 line 29 "G = 1 hoac 1,127"; the V6 tables a.3, a.4 are at G = 1)
B0_XY = 5.85         # [1/s] LADRC b0 of vx, vy, fixed in the firmware (V6 a.1 line 28; adrc.c:33)
TEN = {'PID': ('x', 'v', 'i_pos', 'i_vel', 'u'),
       'ADRC': ('x', 'v', 'i_pos', 'z1', 'z2', 'u')}
ON_DINH, KHONG_ON_DINH = 'on_dinh', 'khong_on_dinh'


def _dung(ctrl, g, G=G_MAC_DINH, tau=TAU, b0=B0_XY):
    """(A, names) with s[k+1] = A s[k], for plain float gains g: no float32 rounding, no checks.

    Every state of TEN[ctrl] is first a row vector over s[k]; one tick of the plant and of the
    firmware is then written out on these rows, in the order of the code. names are the states kept."""
    h = T_TICK / tau
    a = math.exp(-h)                             # V6 a.1 line 29: first order, zero-order hold
    alpha = -math.expm1(-h) / h                  # = tau*(1 - a)/T, mean of v over one tick for u = 0
    ten = TEN[ctrl]
    e = dict(zip(ten, np.eye(len(ten))))
    x, v, i_pos, u = e['x'], e['v'], e['i_pos'], e['u']

    # plant from tick k to k+1 with u[k] held; loop delay 0 (V6 a.1 lines 30-31)
    v_enc = alpha * v + G * (1.0 - alpha) * u    # encoder speed at k+1 = mean over the tick (encoder.c:116-132)
    v1 = a * v + G * (1.0 - a) * u               # velocity at k+1

    # position loop at tick k+1 (V6 a.1 line 26)
    x1 = x + T_TICK * v_enc                      # pos_ctrl.c:103  S.x += vx_enc*dt (theta = 0)
    err = -x1                                    # pos_ctrl.c:108-109  ex = x_ref - x, x_ref = 0
    i_pos1 = i_pos + T_TICK * err                # pos_ctrl.c:50  i_new = i_old + e*dt, before the output
    v_cmd = g['pos_kp'] * err + g['pos_ki'] * i_pos1    # pos_ctrl.c:51, 53; vel_ctrl.c:132 (v_ref = 0)

    if ctrl == 'PID':                            # V6 a.1 line 27
        ev = v_cmd - v_enc                       # pid.c:35  error = sp - actual
        i_vel1 = e['i_vel'] + T_TICK * ev        # pid.c:41  integral += error*dt, before the output
        u1 = v_cmd + g['kp_xy'] * ev + g['ki_xy'] * i_vel1    # vel_ctrl.c:168; pid.c:38, 44, 55 (kd = 0)
        hang = {'x': x1, 'v': v1, 'i_pos': i_pos1, 'i_vel': i_vel1, 'u': u1}
    else:                                        # V6 a.1 line 28
        beta1 = 2.0 * g['omega_o_xy']            # adrc.c:108
        beta2 = g['omega_o_xy'] * g['omega_o_xy']    # adrc.c:109
        eso = v_enc - e['z1']                    # adrc.c:50  err = actual - z1
        z1_1 = e['z1'] + T_TICK * (e['z2'] + beta1 * eso + b0 * u)    # adrc.c:53, 56; u = u_prev (adrc.c:77)
        z2_1 = e['z2'] + T_TICK * beta2 * eso    # adrc.c:54, 57
        u1 = (g['omega_c_xy'] * (v_cmd - z1_1) - z2_1) / b0    # vel_ctrl.c:164; adrc.c:80-81 (kp = omega_c, adrc.c:111)
        hang = {'x': x1, 'v': v1, 'i_pos': i_pos1, 'z1': z1_1, 'z2': z2_1, 'u': u1}

    bo = set()                                   # integrators with gain 0: no path to the loop
    if g['pos_ki'] == 0:
        bo.add('i_pos')
        if g['pos_kp'] == 0:
            bo.add('x')
    if ctrl == 'PID' and g['ki_xy'] == 0:
        bo.add('i_vel')
    giu = [i for i, n in enumerate(ten) if n not in bo]
    A = np.vstack([hang[n] for n in ten])
    assert not A[np.ix_(giu, [i for i in range(len(ten)) if i not in giu])].any()
    return A[np.ix_(giu, giu)], tuple(ten[i] for i in giu)


def chuan_hoa(ctrl, gains) -> dict:
    """The gain set as {name: float32 value} in NAMES order; ValueError when the set is malformed
    (controller, names, a value that is not a number) or a value is not finite as a float32."""
    g = gain_table._gains(ctrl, gains)
    for n, x in g.items():
        if not math.isfinite(x):
            raise ValueError(f'gain {n}={x!r} is not finite as a float32')
    return g


def _duong(ten, x) -> float:
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        raise ValueError(f'{ten}={x!r} is not a number')
    try:
        x = float(x)
    except OverflowError:
        raise ValueError(f'{ten}={x!r} does not fit a float')
    if not math.isfinite(x) or x <= 0.0:
        raise ValueError(f'{ten}={x!r} must be finite and > 0')
    return x


def ma_tran(ctrl, gains, *, G=G_MAC_DINH, tau=TAU):
    """(A, names) of the sub-model for one gain set (7 gains, rounded to float32); ValueError as chuan_hoa,
    or when G, tau are not finite numbers > 0."""
    return _dung(ctrl, chuan_hoa(ctrl, gains), _duong('G', G), _duong('tau', tau))


def _ban_kinh(A) -> float:
    """Largest |eigenvalue| of the float64 matrix A."""
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def ban_kinh_pho(ctrl, gains, *, G=G_MAC_DINH, tau=TAU) -> float:
    """Spectral radius (largest |eigenvalue|) of the sub-model matrix of one gain set."""
    A, _ = ma_tran(ctrl, gains, G=G, tau=tau)
    return _ban_kinh(A)


def trang_thai(rho) -> str:
    """'on_dinh' for rho < 1, 'khong_on_dinh' for rho >= 1 (S04 line 111: >= 1 is reported). The float64
    value is compared with 1 as it is, no tolerance band. ValueError when rho is not a finite number."""
    if isinstance(rho, bool) or not isinstance(rho, numbers.Real) or not math.isfinite(rho):
        raise ValueError(f'ban_kinh_pho={rho!r} is not a finite number')
    return KHONG_ON_DINH if rho >= 1.0 else ON_DINH


def _doc_gains(text):
    """(ctrl | None, gains dict) of one --gains argument: JSON text (first non-blank '{'), else a path.

    ValueError on broken JSON, a JSON key given twice, a top level that is not an object, or a
    "gains" that is not an object; OSError when the file cannot be read."""
    def no_dup(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError(f'JSON key {k!r} given twice')
            d[k] = v
        return d
    if text.lstrip().startswith('{'):
        d = json.loads(text, object_pairs_hook=no_dup)
    else:
        with open(text, encoding='utf-8') as f:
            d = json.load(f, object_pairs_hook=no_dup)
    if not isinstance(d, dict):
        raise ValueError('JSON object expected')
    if 'gains' in d:
        if not isinstance(d['gains'], dict):
            raise ValueError('"gains" must be an object')
        return d.get('ctrl'), d['gains']
    return None, d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ctrl', choices=sorted(gain_table.NAMES))
    ap.add_argument('--gains', required=True,
                    help='JSON text of the gain object, or the path of a JSON file')
    ap.add_argument('--G', type=float, default=G_MAC_DINH, help='static gain u -> v')
    ap.add_argument('--tau', type=float, default=TAU, help='time constant of the axis [s]')
    a = ap.parse_args(argv)
    try:
        f_ctrl, raw = _doc_gains(a.gains)
        if f_ctrl is not None and a.ctrl is not None and f_ctrl != a.ctrl:
            raise ValueError(f'--ctrl {a.ctrl} but the JSON says {f_ctrl}')
        ctrl = a.ctrl if a.ctrl is not None else f_ctrl
        if ctrl is None:
            raise ValueError('--ctrl PID|ADRC needed (or "ctrl" in the JSON)')
        A, ten = ma_tran(ctrl, raw, G=a.G, tau=a.tau)
        rho = _ban_kinh(A)
        tt = trang_thai(rho)
    except (ValueError, OSError) as e:
        print('error: ' + str(e).encode('ascii', 'backslashreplace').decode('ascii'), file=sys.stderr)
        return 1
    print(f'ctrl={ctrl}')
    print(f'G={float(a.G)!r}')
    print(f'tau={float(a.tau)!r}')
    print(f'bien_trang_thai={",".join(ten)}')
    print(f'ban_kinh_pho={rho!r}')
    print(f'trang_thai={tt}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
