"""Synthetic attempt folders for tools/c8_hc_a.py tests (CODE_HIEUCHUAN 19/11): an ESP32 log at 100 Hz in the
agv_logger column names, the sender files of rpi5/c8_hc_seq.py, and a c7_measure-like frames.csv at 60 fps, from the
real schedule (c8_hc_seq.schedule) with a known tag-axis angle a_true. Kinematics: firmware kinematics_inverse."""
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
import c8_hc_seq as hs  # noqa: E402

R, PPR, LXY = 0.0485, 1320, 0.24
START = {'goc': (0.0, 0.0, 45.0), 'K1': (1.0, 0.0, 90.0), 'K2': (-0.5, -0.5, 0.0)}
LOG_HDR = ['timestamp_us', 'fl_count', 'fr_count', 'rl_count', 'rr_count', 'sp_vx', 'sp_vy', 'sp_wz', 'yaw_rad',
           'run_active', 'motor_on', 'pos_reset_n', 'log_seq', 'r_valid', 'rv_n', 'age_r_ms', 'reset_n',
           'psi_reset_count', 'traj_age_ms', 'pos_x_ref', 'pos_y_ref']


def query_text(t0, boot='e75f708f'):
    return (f'#query_pc,t_start={t0:.6f},t_end={t0 + 0.5:.6f}\n'
            f'#gain_running,sha=5bde4c55bf3a,env=esp32-s3-devkitc-1,mac=aa:bb:cc:dd:ee:ff,boot={boot},id=0,'
            f'applied_n=0,ctrl=PID,kp_xy=1.5,ki_xy=0.05,kp_wz=1,ki_wz=0.05,pos_kp=6,pos_ki=3,k_psi=1\n#ket_qua,ma=0\n')


def write_summary(csv_path, status='closed'):
    """The agv_logger summary of a closed run (status, files.csv sha256), as agv_logger.main writes it."""
    h = hashlib.sha256(open(csv_path, 'rb').read()).hexdigest()
    with open(csv_path + '.summary.json', 'w', encoding='utf-8') as f:
        json.dump(dict(status=status, files=dict(csv=dict(file=os.path.basename(csv_path), sha256=h))), f)


def make(d, cho='K1', phan='all', a_true_deg=0.7, q=(0.004, -0.003), lag=0.15, D=50.0, seed=1, noise_p=0.0003,
         noise_psi_deg=0.22, yaw_bias=None, cam_bias=None, cut=None, second_edge=False, rv_freeze=None,
         wiggle=None, boot_after='e75f708f', rx_latency=0.003):
    """Write one attempt folder; returns dict of the truth. Options:
    yaw_bias(t) rad added to the IMU yaw; cam_bias(t) rad added to the camera heading; cut=(t_from, t_to) zeroes
    sp_vx (ultrasonic cut before integration: the reference loses the distance); rv_freeze=(t_from, t_to) holds yaw;
    wiggle=(t_from, t_to, amp_rad) real body rotation (half sine) not seen by a frozen RV when combined."""
    rng = np.random.default_rng(seed)
    os.makedirs(d)
    segs = hs.schedule(cho, phan)
    lich = hs.lich_csv(segs)
    with open(os.path.join(d, 'lich.csv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(lich)
    sha = hashlib.sha256(lich.encode()).hexdigest()
    with open(os.path.join(d, 'sender.txt'), 'w', encoding='utf-8') as f:
        f.write(f'#log_dong_dau,t_epoch=1\n#gain_query,when=truoc,exit=0\n'
                f'#seq_start,ver=c8_hc_seq_v3,cho={cho},phan={phan},thu_id={os.path.basename(d)},t_epoch=1,iso=x,'
                f'duration_s={hs.duration(segs):.3f},send_hz=50,tail_s=2.5,dst=x,lich_sha256={sha}\n'
                f'#seq_end,t_epoch=2,sent=1,planned=1,n_bo=0,gap_max_ms=20.0,complete=1,sched_first_s=0,sched_last_s=1\n'
                f'#gain_query,when=sau,exit=0\n#logger_dong,ma=0\n#ket_qua,ma=0\n')
    with open(os.path.join(d, 'gain_query_truoc.txt'), 'w', encoding='utf-8') as f:
        f.write(query_text(1000.0))
    with open(os.path.join(d, 'gain_query_sau.txt'), 'w', encoding='utf-8') as f:
        f.write(query_text(2000.0, boot_after))
    open(os.path.join(d, 'log.csv.events.txt'), 'w').close()
    t_s = 6.0
    T_sched = hs.duration(segs) + hs.TAIL_S
    t_end = t_s + T_sched + 0.2 + 6.0
    t = np.arange(0.0, t_end, 0.01)
    n = len(t)
    nk = hs.n_samples(segs)
    total = nk + int(math.ceil(hs.TAIL_S * hs.SEND_HZ - 1e-9))
    # packets k = 0..total-1 received at t_s + k/50 + latency; the firmware keeps the newest one, run_active while
    # its age < 200 ms (main.c 524-525, 611), traj_age_ms = tick time - receive time (main.c 529-541; HCC1-4)
    t_rx = t_s + np.arange(total) / hs.SEND_HZ + rx_latency
    last = np.searchsorted(t_rx, t, side='right') - 1                   # newest packet received by row i
    age = np.where(last >= 0, np.floor((t - t_rx[np.maximum(last, 0)]) * 1000.0 + 1e-9), -1.0)
    run = (last >= 0) & (age < 200.0)
    sp = np.zeros((n, 3))
    for i in range(n):
        if run[i] and last[i] < nk:
            sp[i] = hs.sample(segs, int(last[i]))
    if cut:
        m = (t >= cut[0] + t_s) & (t < cut[1] + t_s)
        sp[m, 0] = 0.0
    # firmware reference = integral of the received sp; the car follows it with a lag
    xr = np.cumsum(sp[:, 0] * 0.01)
    thr = np.cumsum(sp[:, 2] * 0.01)
    xa = np.interp(t - lag, t, xr, left=0.0)
    tha = np.interp(t - lag, t, thr, left=0.0)
    if wiggle:
        a0, a1, amp = wiggle
        m = (t >= a0 + t_s) & (t <= a1 + t_s)
        tha = tha + np.where(m, amp * np.sin(math.pi * (t - a0 - t_s) / (a1 - a0)), 0.0)
    wl = np.column_stack([(xa - LXY * tha) / R, (xa + LXY * tha) / R, (xa - LXY * tha) / R, (xa + LXY * tha) / R])
    counts = np.floor(wl * PPR / (2 * math.pi)).astype(np.int64)
    yaw0 = 1.234
    yaw = tha + yaw0 + (np.array([yaw_bias(x) for x in t]) if yaw_bias else 0.0)
    rep = (np.arange(n) // 5)
    yaw = yaw[rep * 5]                                         # RV at 20 Hz, held between reports
    rv_n = rep.astype(float)
    if rv_freeze:
        m = (t >= rv_freeze[0] + t_s) & (t <= rv_freeze[1] + t_s)
        i0 = np.flatnonzero(m)[0]
        yaw = np.where(m, yaw[i0], yaw)                       # RV keeps reporting the old value
    ra = run.astype(float)
    if second_edge:
        ra[(t >= t_s + 20.0) & (t < t_s + 20.05)] = 0.0
    pos_reset = np.where(np.arange(n) >= int(np.argmax(run)), 4.0, 3.0)     # +1 at the run_active edge
    with open(os.path.join(d, 'log.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(LOG_HDR)
        for i in range(n):
            w.writerow([int(round(t[i] * 1e6)) + 100000000, *counts[i].tolist(), f'{sp[i, 0]:.6f}', f'{sp[i, 1]:.6f}',
                        f'{sp[i, 2]:.6f}', f'{math.atan2(math.sin(yaw[i]), math.cos(yaw[i])):.6f}', int(ra[i]), 1,
                        int(pos_reset[i]), 1000 + i, 1, int(rv_n[i]), (i % 5) * 10, 3, 0, int(age[i]),
                        f'{xr[i]:.6f}', '0.000000'])
    write_summary(os.path.join(d, 'log.csv'))
    # camera
    x0, y0, h0 = START[cho]
    h0 = math.radians(h0)
    tc = np.arange(0.3, t_end - 0.3, 1 / 60.0) - D
    dxa = np.r_[0.0, np.diff(xa)]                              # the car moves along its own (turning) axis
    Xs, Ys = np.cumsum(np.cos(tha) * dxa), np.cumsum(np.sin(tha) * dxa)
    xb, yb = np.interp(tc + D, t, Xs), np.interp(tc + D, t, Ys)
    thb = np.interp(tc + D, t, tha)
    hd = h0 + thb
    bx = x0 + math.cos(h0) * xb - math.sin(h0) * yb
    by = y0 + math.sin(h0) * xb + math.cos(h0) * yb
    tx = bx + np.cos(hd) * q[0] - np.sin(hd) * q[1] + rng.normal(0, noise_p, len(tc))
    ty = by + np.sin(hd) * q[0] + np.cos(hd) * q[1] + rng.normal(0, noise_p, len(tc))
    ps = hd + math.radians(a_true_deg) + rng.normal(0, math.radians(noise_psi_deg), len(tc))
    if cam_bias:
        ps = ps + np.array([cam_bias(x + D - t_s) for x in tc])
    with open(os.path.join(d, 'frames.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 't_s', 'valid', 'car_seen', 'x', 'y', 'psi', 'tag_x', 'tag_y'])
        for i in range(len(tc)):
            p = math.atan2(math.sin(ps[i]), math.cos(ps[i]))
            w.writerow([i, f'{tc[i]:.6f}', 1, 1, f'{tx[i]:.6f}', f'{ty[i]:.6f}', f'{p:.6f}', f'{tx[i]:.6f}', f'{ty[i]:.6f}'])
    with open(os.path.join(d, 'track.json'), 'w') as f:
        json.dump(dict(tag_offset=[0.0, 0.0, 0.0]), f)
    return dict(a_true=math.radians(a_true_deg), t_s=t_s, D=D)
