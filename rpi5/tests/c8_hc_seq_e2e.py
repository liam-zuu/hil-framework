"""End-to-end run of rpi5/c8_hc_seq.run_attempt with the REAL tools/agv_logger.py on POSIX (HCC1-14, CODE_HIEUCHUAN
19/11): real monotonic clock, real UDP sockets on 127.0.0.1, the logger as a child in its own session stopped by
SIGINT, its summary checked; a fake ESP32 thread receives the trajectory packets on TRAJ_PORT and prints log lines
to the logger like the firmware: run_active = motor on && AUTO && last packet < 200 ms (main.c 524-525), sp_* = last
packet while fresh, traj_age_ms = tick time - receive time (main.c 529-541). Only the gain question is faked.

Runs from pytest on Linux (test_c8_hc_seq.test_e2e_real_logger_posix) or alone, without pytest (WSL, the Pi):
    python3 rpi5/tests/c8_hc_seq_e2e.py [--logger PATH_TO_agv_logger.py] [--ctrl-c]
Exit 0 when every check holds. The line format below is copied from tools/tests/test_agv_logger.py (DUT_HEADER,
DUT_FMT: AGV logger.c), not from agv_logger.py."""
import argparse
import json
import os
import signal
import socket
import sys
import tempfile
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import c8_hc_seq as hs  # noqa: E402
import pc_sender as ps  # noqa: E402

DUT_HEADER = (
    "timestamp_us,fl_count,fr_count,rl_count,rr_count,fl_rpm,fr_rpm,rl_rpm,rr_rpm,vx,vy,wz,"
    "sp_vx,sp_vy,sp_wz,pwm_fl,pwm_fr,pwm_rl,pwm_rr,ax,ay,az,gx,gy,gz,yaw,pitch,roll,"
    "tick,dpwm_fl,dpwm_fr,dpwm_rl,dpwm_rr,sat_fl,sat_fr,sat_rl,sat_rr,"
    "log_seq,log_qdrop,log_send_err,motor_on,imu_init,"
    "wz_meas,wz_cmd,gz_fused,yaw_rad,age_g_ms,age_r_ms,gyro_n,rv_n,reset_n,"
    "g_valid,r_valid,wz_src,src_switch_count,psi_ref,e_psi,head_on,psi_reset_count,"
    "slip_flag,slip_on_count,r,bias_gz,"
    "src_switch_dwz,bias_updates,slip_eval,"
    "pos_on,pos_x,pos_y,pos_th,pos_x_ref,pos_y_ref,pos_th_ref,pos_corr_vx,pos_corr_vy,"
    "pos_th_src,pos_in_pos,still,run_active,pos_reset_n,"
    "ws_res,ws_th,ws_flag,ws_on_count,ws_eval,"
    "traj_age_ms")
DUT_FMT = ("%lld," "%ld,%ld,%ld,%ld," "%.2f,%.2f,%.2f,%.2f," "%.3f,%.3f,%.3f," "%.3f,%.3f,%.3f,"
           "%d,%d,%d,%d," "%.3f,%.3f,%.3f," "%.3f,%.3f,%.3f," "%.2f,%.2f,%.2f,"
           "%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu," "%lu,%lu,%lu,%u,%d,"
           "%.4f,%.4f,%.4f,%.4f,%.1f,%.1f,%lu,%lu,%lu," "%u,%u,%u,%lu,%.4f,%.4f,%u,%lu,"
           "%u,%lu,%.4f,%.5f," "%.4f,%lu,%u,"
           "%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f," "%u,%u,%u,%u,%lu,"
           "%.0f,%.1f,%u,%lu,%u,"
           "%ld").split(',')
NAMES = DUT_HEADER.split(',')


def dut_line(seq, ts, **over):
    vals = {n: (c % 0.0 if 'f' in c else '0') for n, c in zip(NAMES, DUT_FMT)}
    vals['timestamp_us'], vals['log_seq'], vals['motor_on'] = str(ts), str(seq), '1'
    for k, v in over.items():
        conv = DUT_FMT[NAMES.index(k)]
        vals[k] = conv % v if 'f' in conv else str(int(v))
    return ','.join(vals[n] for n in NAMES)


class FakeEsp32(threading.Thread):
    def __init__(self, log_port, csv_path):
        super().__init__(daemon=True)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(('127.0.0.1', ps.TRAJ_PORT))
        self.rx.setblocking(False)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.log_port, self.csv_path = log_port, csv_path
        self.stop = threading.Event()
        self.n_rx, self.edges = 0, 0

    def run(self):
        t0 = time.monotonic()
        while not os.path.exists(self.csv_path) and not self.stop.is_set():   # logger opened its files
            time.sleep(0.02)
        time.sleep(0.2)
        self.tx.sendto((DUT_HEADER + '\n').encode(), ('127.0.0.1', self.log_port))
        last, t_rx, seq, prev_ra = (0.0, 0.0, 0.0), None, 0, 0
        nxt = time.monotonic()
        while not self.stop.is_set():
            while True:
                try:
                    pkt = self.rx.recv(64)
                except BlockingIOError:
                    break
                last, t_rx = ps_decode(pkt), time.monotonic()
                self.n_rx += 1
            now = time.monotonic()
            fresh = t_rx is not None and now - t_rx < 0.200
            ra = int(fresh)
            self.edges += int(ra == 1 and prev_ra == 0)
            prev_ra = ra
            sp = last if fresh else (0.0, 0.0, 0.0)
            age = -1 if t_rx is None else int((now - t_rx) * 1000)
            ln = dut_line(seq, int((now - t0) * 1e6) + 5_000_000, sp_vx=sp[0], sp_vy=sp[1], sp_wz=sp[2],
                          run_active=ra, traj_age_ms=age, r_valid=1, rv_n=seq // 5)
            self.tx.sendto((ln + '\n').encode(), ('127.0.0.1', self.log_port))
            seq += 1
            nxt += 0.010
            time.sleep(max(0.0, nxt - time.monotonic()))
        self.rx.close()
        self.tx.close()


def ps_decode(pkt):
    import struct
    return struct.unpack_from('<fff', pkt, 1)


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


def chay(tmp, logger=None, ctrl_c_after=None, cho='K2', phan='all'):
    """One attempt; returns (exit code, sender.txt text, summary dict or None, fake ESP32)."""
    d = os.path.join(tmp, f'HC_{cho}_1')
    port = free_port()
    esp = FakeEsp32(port, os.path.join(d, 'log.csv'))
    esp.start()
    hs.STILL_BEFORE_S = hs.STILL_AFTER_S = 1.0

    def query(ip, path):
        with open(path, 'x', encoding='utf-8') as f:
            f.write('#gain_running,sha=x,env=y,boot=b1,applied_n=0\n#ket_qua,ma=0\n')
        return 0

    def pgrep():
        txt = hs._pgrep_default()
        return '\n'.join(ln for ln in txt.splitlines() if 'pytest' not in ln and 'c8_hc_seq_e2e' not in ln)

    if ctrl_c_after:
        threading.Timer(ctrl_c_after, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    a = types.SimpleNamespace(ip='127.0.0.1', src_ip='127.0.0.1', cho=cho, phan=phan, thu_muc=d)
    try:
        code = hs.run_attempt(a, out=lambda s: None, query=query, pgrep=pgrep, logger=logger,
                              logger_extra=('--bind', '127.0.0.1', '--port', str(port)))
    finally:
        esp.stop.set()
        esp.join(2)
    with open(os.path.join(d, 'sender.txt'), encoding='utf-8') as f:
        txt = f.read()
    try:
        with open(os.path.join(d, 'log.csv.summary.json'), encoding='utf-8') as f:
            summ = json.load(f)
    except OSError:
        summ = None
    return code, txt, summ, esp


def kiem(code, txt, summ, esp, ctrl_c):
    errs = []
    if ctrl_c:
        if code != 3 or '#seq_huy,ly_do=ctrl_c' not in txt:
            errs.append(f'ctrl-c: ma={code}')
    else:
        if code != 0 or 'complete=1' not in txt:
            errs.append(f'ma={code}')
        if '#kiem_truoc_chay,k=' not in txt or 'ket_qua=dat' not in txt:
            errs.append('kiem_truoc_chay')
        if esp.edges != 1:
            errs.append(f'run_active edges {esp.edges}')
    if summ is None or summ.get('status') != 'closed':
        errs.append('summary')
    if 'khop=1' not in txt:
        errs.append('summary sha')
    gm = [ln for ln in txt.splitlines() if ln.startswith('#seq_end')]
    if not gm:
        errs.append('no #seq_end')
    else:
        g = float(gm[0].split('gap_max_ms=')[1].split(',')[0])
        if g > hs.GAP_ABORT_S * 1000:
            errs.append(f'gap_max_ms {g}')
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logger', default=None)
    ap.add_argument('--ctrl-c', action='store_true')
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        code, txt, summ, esp = chay(tmp, a.logger, 12.0 if a.ctrl_c else None)
        errs = kiem(code, txt, summ, esp, a.ctrl_c)
        print(txt)
        print(f'#e2e,ma={code},rx={esp.n_rx},edges={esp.edges},loi={"/".join(errs) or "-"}')
    return 1 if errs else 0


if __name__ == '__main__':
    sys.exit(main())
