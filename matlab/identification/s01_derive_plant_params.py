"""Derive per-wheel plant parameters from S01 run01 identification (direction-averaged)."""
import csv, statistics as st
P = {}
for r in csv.DictReader(open('identification/s01_params_per_wheel.csv')):
    P.setdefault(r['wheel'], {}).setdefault(r['quantity'], []).append(float(r['mean']))
runs = list(csv.DictReader(open('identification/s01_run_fits.csv')))
runsraw = list(csv.DictReader(open('runs.csv')))
V = st.mean([(float(x['v_batt_start']) + float(x['v_batt_end'])) / 2 for x in runsraw])
R = {'FL': 3.31, 'FR': 3.38, 'RL': 3.73, 'RR': 3.95}
out = []
for w in ['FL', 'FR', 'RL', 'RR']:
    K = st.mean(P[w]['K_Vs_per_rad']); J = st.mean(P[w]['J_kgm2']); b = st.mean(P[w]['b_Nms_per_rad'])
    tcj = st.mean([float(x['Tc_over_J']) for x in runs if x['wheel'] == w and x['scenario'] == 'coast' and x['Tc_over_J'] not in ('', 'NaN')])
    Tc = tcj * J
    u_int = st.mean(P[w]['deadband_pwm']) / 1000            # x-intercept of linear omega_ss(u)
    u0 = max(u_int - Tc * R[w] / (K * V), 0.0)               # voltage-equivalent offset (brush drop etc.)
    # breakaway: per direction, midpoint between highest level not turning and lowest turning (mean over reps)
    us = []
    for d in ('pos', 'neg'):
        lv = {}
        for x in runs:
            if x['wheel'] == w and x['scenario'] == 'step' and x['direction'] == d:
                lv.setdefault(int(x['pwm_level']), []).append(abs(float(x['w_ss'])))
        turn = sorted(l for l, v in lv.items() if st.mean(v) > 0.5)
        still = sorted(l for l, v in lv.items() if st.mean(v) <= 0.5 and l < turn[0])
        us.append((turn[0] + (still[-1] if still else 0)) / 2 / 1000)
    u_start = st.mean(us)
    Ts = K / R[w] * V * (u_start - u0)                       # static torque needed to break away at omega = 0
    out.append(dict(wheel=w, K=K, R=R[w], J=J, b=b, Tc=Tc, u0=u0, u_start=u_start, Ts=Ts, V_run=V,
                    stall_torque_full_duty=K / R[w] * V * (1 - u0)))
with open('identification/plant_params_from_S01.csv', 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(out[0].keys())); wr.writeheader()
    for o in out: wr.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in o.items()})
for o in out: print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in o.items()})
