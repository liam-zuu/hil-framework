#!/usr/bin/env python3
"""
s06_esp32_check.py — kiểm một lượt S06 từ log ESP32 env `s06_load` (thư mục do s06_logger.py ghi).

Căn cứ: spec S06 v3.4 §4.3 (đủ log), §5 (CSV đầu ra), T4 IMU (thông lượng, hở gyro) và luật
"đếm ở cổng SPI" (Liam 29/10, phiên THESIS): report ESP32 lấy ở cổng nhận SPI (dòng J); mỗi lượt
báo theo loại report ba số: ở cổng SPI, tới callback ứng dụng (dòng I), bị ghi đè (hiệu hai số).
Report không tới callback là hành vi DUT: báo riêng, không làm phần HIL không đạt.

Chỉ phần ESP32. Phần cần luồng H7/RPi5 (ghép giá trị với frame nguồn, T1, T3, T6 MISO) chưa có:
cột src_frame, value_src, err, ok của T4_imu_cb.csv để trống.

Lượt không hợp lệ (đo lại, không tính bộ HIL không đạt; Liam duyệt 29/10 ở phiên THESIS, "Theo khuyến
nghị", spec S06 T3b): số lần đọc count trong lượt ≠ thời lượng / 1 ms ± 1. Mục loại này có
category "valid"; mục khác "check".

Sửa theo Codex Sol vòng 1/2 (Liam duyệt 30/10 "Duyệt, sửa hết"):
  * S26: số lần đọc encoder của nguồn tính theo dải `eseq` (không theo số dòng PC nhận); mất dòng hoặc lỗi
    sản phẩm đã thấy luôn là "fail", "invalid" chỉ khi các mục điều kiện là mục duy nhất không đạt (Z19).
  * S7: reset hoàn tất ±1 s quanh #run_start thành mục điều kiện (không hợp lệ, đo lại); phía H7 ở
    tools/s06_frames_decode.py.
  * S27: hở gyro tính cả khoảng qua hai biên lượt (report ngay trước #run_start, ngay sau #run_end).
  * S30: relay_bad tăng trong lượt = không đạt; relay_ok và việc count encoder có đổi được báo (`load`).
  * S37: log_send_err_run lấy từ dòng #run_stats của ESP32 (mốc lúc START, s06_load b9b3984); firmware
    cũ thì lấy từ DATA đầu PC nhận và ghi chú.

Usage:
    python tools/s06_esp32_check.py <thư mục lượt> [<thư mục lượt> ...] [--out <thư mục kết quả>]
Mã thoát (một lượt): 0 đạt; 1 có mục không đạt; 2 thiếu #run_start/#run_end; 3 lượt không hợp lệ
(đo lại). Nhiều lượt: 0 khi mọi lượt đạt, ngược lại 1; in số lượt bị loại vì số lần đọc count.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

KINDS = {1: "accel", 2: "gyro", 8: "game_rv"}
DATA_GAP_US = 20_000          # §4.3 điều 3: quy ước 2 chu kỳ điều khiển
DURATION_TOL_US = 20_000      # §4.3 điều 3
T9_ACK_US = 120_000           # §4.3 V3-4: t_rx(#run_end) − t_rx(#run_start) ∈ [D; D + 120 ms]
WIN_US = 1_000_000            # T4: cửa sổ 1 s
WIN_LO, WIN_HI = 0.95, 1.05   # T4: số report mỗi cửa sổ ∈ [0,95; 1,05]·f (5 % là quy ước)
T_AGE_PERIODS = 2.5           # T4: hở gyro ≤ T_age = 2,5 chu kỳ report (imu_fusion.h 030a836)
ENC_COUNT_TOL = 1             # [ĐỀ XUẤT CODE 29/10] số lần đọc encoder = thời lượng / chu kỳ ± 1
RESET_NEAR_US = 1_000_000     # V4-1: thông báo reset trong ±1 s quanh #run_start
SEQ_MOD = 256                 # số thứ tự report SH-2 8 bit

# Vị trí cột trong DATA của logger.c khi không bắt được dòng tiêu đề (tiêu đề gửi một lần lúc khởi động)
DATA_COLS_DEFAULT = {"timestamp_us": 0, "log_seq": 37, "log_qdrop": 38, "log_send_err": 39}


# ─── Đọc file thô ────────────────────────────────────────────────────────────

def read_raw(path):
    """Danh sách (t_rx_us, dòng) từ file `<t_rx_us>\\t<dòng>`."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            t, sep, line = raw.partition("\t")
            if sep and t.isdigit():
                out.append((int(t), line))
    return out


def parse_text(line):
    """'#name,k=v,...' → (name, dict). Dòng #cfg dùng ';' trong trường thứ hai."""
    parts = line[1:].split(",")
    name, kv = parts[0], {}
    for p in parts[1:]:
        if name == "cfg" and ";" in p:
            for q in p.split(";"):
                k, _, v = q.partition("=")
                kv[k] = v
            continue
        k, sep, v = p.partition("=")
        if sep:
            kv[k] = v
    return name, kv


def parse_events(raw):
    """Tách luồng sự kiện theo loại dòng. Dòng hỏng đếm ở 'bad'."""
    ev = {"J": [], "I": [], "R": [], "P": [], "E": [], "S": [], "text": [], "bad": 0, "qseq": []}
    for t_rx, line in raw:
        try:
            if line.startswith("#"):
                name, kv = parse_text(line)
                q = int(kv["qseq"])
                ev["text"].append({"t_rx": t_rx, "name": name, "kv": kv, "qseq": q, "line": line})
                ev["qseq"].append(q)
                continue
            f = line.split(",")
            k = f[0]
            if k == "J" and len(f) == 7:
                r = dict(qseq=int(f[1]), t_us=int(f[2]), kind=int(f[3]), queued=int(f[4]),
                         seq=int(f[5]), epoch=int(f[6]))
            elif k == "I" and len(f) == 12:
                r = dict(qseq=int(f[1]), t_us=int(f[2]), kind=int(f[3]), applied=int(f[4]),
                         seq=int(f[5]), epoch=int(f[6]), n=int(f[7]), lib_n=int(f[8]),
                         v=(float(f[9]), float(f[10]), float(f[11])))
            elif k == "R" and len(f) == 4:
                r = dict(qseq=int(f[1]), t_us=int(f[2]), reset_count=int(f[3]))
            elif k == "P" and len(f) == 10:
                r = dict(qseq=int(f[1]), t_before=int(f[2]), t_after=int(f[3]), wheel=int(f[4]),
                         speed=int(f[5]), gpio_a=int(f[6]), cmp_a=int(f[7]), gpio_b=int(f[8]),
                         cmp_b=int(f[9]))
            elif k == "E" and len(f) == 8:
                ev["E"].append(dict(eseq=int(f[1]), t_before=int(f[2]), t_after=int(f[3]),
                                    c=tuple(int(x) for x in f[4:8])))
                continue
            elif k == "S" and len(f) in (15, 16):
                # 16 trường từ s06_load có temp_c (nhiệt độ chip ESP32-S3, T5, 30/10); bản cũ 15 trường
                r = dict(qseq=int(f[1]), t_us=int(f[2]), run=int(f[3]), ev_qdrop=int(f[4]),
                         enc_ovf=int(f[5]), imu_rx=int(f[6]), imu_cbq=int(f[7]), imu_mutex=int(f[8]),
                         gyro_gap_max_us=int(f[9]), relay_ok=int(f[10]), relay_bad=int(f[11]),
                         log_qdrop=int(f[12]), log_send_err=int(f[13]), ev_send_err=int(f[14]),
                         temp_c=float(f[15]) if len(f) == 16 else None)
            else:
                ev["bad"] += 1
                continue
        except (ValueError, KeyError, IndexError):
            ev["bad"] += 1
            continue
        r["t_rx"] = t_rx
        ev[k].append(r)
        ev["qseq"].append(r["qseq"])
    return ev


def parse_data(raw):
    """Dòng DATA (log_seq, timestamp, qdrop, send_err) và dòng '#' của logger.c (có seq=)."""
    cols = dict(DATA_COLS_DEFAULT)
    rows, logger_text, bad, header_seen = [], [], 0, False
    for t_rx, line in raw:
        if line.startswith("timestamp_us"):
            names = line.split(",")
            cols = {k: names.index(k) for k in DATA_COLS_DEFAULT if k in names}
            header_seen = len(cols) == len(DATA_COLS_DEFAULT)
            if not header_seen:
                cols = dict(DATA_COLS_DEFAULT)
            continue
        if line.startswith("#"):
            name, kv = parse_text(line)
            logger_text.append({"t_rx": t_rx, "name": name, "kv": kv})
            continue
        f = line.split(",")
        try:
            rows.append(dict(t_rx=t_rx, ts=int(f[cols["timestamp_us"]]), seq=int(f[cols["log_seq"]]),
                             qdrop=int(f[cols["log_qdrop"]]), send_err=int(f[cols["log_send_err"]])))
        except (ValueError, IndexError):
            bad += 1
    return rows, logger_text, bad, header_seen


# ─── Kiểm ────────────────────────────────────────────────────────────────────

def seq_stream(values, lo, hi):
    """Số kỳ vọng, nhận (không trùng), hở, trùng của chỉ số trong [lo; hi]."""
    inside = [v for v in values if lo <= v <= hi]
    uniq = set(inside)
    expected = hi - lo + 1 if hi >= lo else 0
    return dict(expected=expected, received=len(uniq), gaps=expected - len(uniq),
                dups=len(inside) - len(uniq))


def first_text(ev, name):
    for t in ev["text"]:
        if t["name"] == name:
            return t
    return None


def match_reports(J, I):
    """Ghép mỗi callback (I) với report ở cổng SPI (J) cùng (loại, epoch, seq), J mới nhất có
    t_us ≤ t_us của I. Trả về: tập chỉ số J đã tới callback; I không ghép được; I lặp report."""
    by_key = defaultdict(list)                 # (kind, epoch, seq) → [(t_us, chỉ số J)] theo thời gian
    for idx, j in sorted(enumerate(J), key=lambda x: x[1]["t_us"]):
        by_key[(j["kind"], j["epoch"], j["seq"])].append((j["t_us"], idx))
    delivered, unmatched, repeat = set(), [], []
    i_to_j = {}
    for k, i in enumerate(I):
        cands = by_key.get((i["kind"], i["epoch"], i["seq"]), [])
        best = None
        for t, idx in cands:
            if t <= i["t_us"]:
                best = idx
            else:
                break
        if best is None:
            unmatched.append(k)
            continue
        if best in delivered:
            repeat.append(k)
        delivered.add(best)
        i_to_j[k] = best
    return delivered, unmatched, repeat, i_to_j


def check_run(run_dir):
    ev = parse_events(read_raw(os.path.join(run_dir, "esp32_events.txt")))
    data, logger_text, data_bad, header_seen = parse_data(read_raw(os.path.join(run_dir, "esp32_data.txt")))
    res = {"run_dir": run_dir, "items": [], "notes": []}

    def item(name, value, limit, ok, category="check"):
        res["items"].append(dict(item=name, value=value, limit=limit, ok=ok, category=category))
        return ok

    rs, re_ = first_text(ev, "run_start"), first_text(ev, "run_end")
    cfg = first_text(ev, "cfg")
    item("run_start_and_run_end", f"run_start={'có' if rs else 'thiếu'}, run_end={'có' if re_ else 'thiếu'}",
         "có cả hai (§4.3 điều 1)", bool(rs and re_))
    if not (rs and re_):
        res["status"] = "insufficient"
        return res, ev, None
    t0, t1 = int(rs["kv"]["timestamp_us"]), int(re_["kv"]["timestamp_us"])
    dur_s, auto_end = int(rs["kv"]["duration_s"]), int(rs["kv"]["auto_end"])
    period_us = int(cfg["kv"]["imu_period_us"]) if cfg else None
    enc_period_us = int(cfg["kv"]["enc_period_us"]) if cfg else None
    res.update(t_start_us=t0, t_end_us=t1, duration_s=dur_s, auto_end=auto_end,
               profile=rs["kv"].get("profile"), fw=rs["kv"].get("fw"), dirty=rs["kv"].get("dirty"),
               env=rs["kv"].get("env"), cfg_hash=rs["kv"].get("cfg_hash"),
               end_reason=re_["kv"].get("reason"), imu_period_us=period_us)
    if not cfg:
        res["notes"].append("thiếu dòng #cfg: không biết chu kỳ IMU, encoder")
    # T4 (1): kết quả khởi tạo IMU trong #run_start (s06_load 4f32de2; dòng "BNO085 init OK" chỉ ra serial)
    ii = rs["kv"].get("imu_init")
    item("imu_init", f"mã {ii}, {rs['kv'].get('imu_init_ms')} ms" if ii is not None else "thiếu trường",
         "mã 0 = IMU_INIT_OK (T4 (1))", ii == "0")

    # §4.3 điều 2, 4: luồng DATA theo log_seq
    lo, hi = int(rs["kv"]["first_data_seq"]), int(re_["kv"]["last_data_seq"])
    d_in = [r for r in data if lo <= r["seq"] <= hi]
    lt_seq = [int(t["kv"]["seq"]) for t in logger_text if "seq" in t["kv"]]
    ds = seq_stream([r["seq"] for r in data] + lt_seq, lo, hi)
    ds.update(stream="esp32_data", overflow=int(re_["kv"].get("log_qdrop_run", -1)))
    ds["ok"] = ds["gaps"] == 0 and ds["dups"] == 0 and ds["overflow"] == 0
    # S37: the run figure comes from the ESP32 itself (#run_stats after #run_end, baseline at START,
    # s06_load b9b3984); older firmware: from the first DATA the PC got, which may be later than START
    rst = first_text(ev, "run_stats")
    if rst and "log_send_err_run" in rst["kv"]:
        send_err_d, src = int(rst["kv"]["log_send_err_run"]), "từ START tới #run_end (#run_stats)"
    else:
        send_err0 = min((r["send_err"] for r in d_in), default=None)
        send_err_d = int(re_["kv"]["log_send_err"]) - send_err0 if send_err0 is not None else None
        src = "từ DATA đầu PC nhận tới #run_end: không gồm lỗi gửi trước DATA đầu (firmware cũ, không có #run_stats)"
        res["notes"].append(f"log_send_err_run {src}")
    item("data_seq_range", f"{lo}..{hi}: nhận {ds['received']}/{ds['expected']}, hở {ds['gaps']}",
         "đủ, không hở (§4.3 điều 2)", ds["gaps"] == 0)
    item("data_seq_dups", ds["dups"], "0 (§4.3 điều 4)", ds["dups"] == 0)
    item("log_qdrop_run", ds["overflow"], "0 (§4.3 điều 4)", ds["overflow"] == 0)
    item("log_send_err_run", send_err_d, f"0 (§4.3 điều 4; {src})", send_err_d == 0)
    if not header_seen:
        res["notes"].append("không bắt được dòng tiêu đề DATA: dùng vị trí cột mặc định của logger.c")

    # §4.3 điều 3: phủ thời gian
    ts = sorted(r["ts"] for r in d_in)
    if ts:
        gap = max([b - a for a, b in zip(ts, ts[1:])], default=0)
        item("data_gap_max_us", gap, f"≤ {DATA_GAP_US}", gap <= DATA_GAP_US)
        item("data_first_after_start_us", ts[0] - t0, f"≤ {DATA_GAP_US}", ts[0] - t0 <= DATA_GAP_US)
        item("data_last_before_end_us", t1 - ts[-1], f"≤ {DATA_GAP_US}", t1 - ts[-1] <= DATA_GAP_US)
    else:
        item("data_in_run", 0, "> 0", False)
    if auto_end:
        dd = (t1 - t0) - dur_s * 1_000_000
        item("duration_err_us", dd, f"|.| ≤ {DURATION_TOL_US}", abs(dd) <= DURATION_TOL_US)
    else:
        rx = re_["t_rx"] - rs["t_rx"] - dur_s * 1_000_000
        item("t9_rx_duration_minus_D_us", rx, f"∈ [0; {T9_ACK_US}] (đồng hồ Pi, V3-4)", 0 <= rx <= T9_ACK_US)

    # §4.3 điều 6: luồng sự kiện (qseq), luồng đọc encoder (eseq)
    es = seq_stream(ev["qseq"], rs["qseq"], re_["qseq"])
    es.update(stream="esp32_events", overflow=int(re_["kv"].get("ev_qdrop_run", -1)))
    es["ok"] = es["gaps"] == 0 and es["dups"] == 0 and es["overflow"] == 0
    E = [e for e in ev["E"] if t0 - 1000 <= e["t_before"] <= t1 + 1000]
    enc = dict(stream="esp32_enc", expected=None, received=len({e["eseq"] for e in E}), gaps=None,
               dups=len(E) - len({e["eseq"] for e in E}), overflow=int(re_["kv"].get("enc_ovf_run", -1)))
    if E:
        eq = sorted(e["eseq"] for e in E)
        enc["gaps"] = (eq[-1] - eq[0] + 1) - len(set(eq))
        # S26: reads the SOURCE made = span of eseq (independent of lines lost on the way to the PC)
        enc["source_reads"] = eq[-1] - eq[0] + 1
    else:
        enc["source_reads"] = 0
    if enc_period_us:
        enc["expected"] = (t1 - t0) // enc_period_us
    enc["ok"] = enc["gaps"] == 0 and enc["dups"] == 0 and enc["overflow"] == 0
    item("enc_reads_vs_duration", f"{enc['source_reads']} lần đọc ở ESP32 (theo eseq), kỳ vọng {enc['expected']}",
         f"± {ENC_COUNT_TOL} (T3b; không đạt = nhịp đọc sai, lượt không hợp lệ, đo lại)",
         enc["expected"] is not None and abs(enc["source_reads"] - enc["expected"]) <= ENC_COUNT_TOL, "valid")
    if E:
        tb = sorted(e["t_before"] for e in E)
        res["enc_period_max_us"] = max([b - a for a, b in zip(tb, tb[1:])], default=0)
        res["enc_read_dur_max_us"] = max(e["t_after"] - e["t_before"] for e in E)
    res["streams"] = [ds, es, enc]
    for s in (es, enc):
        item(f"stream_{s['stream']}", f"nhận {s['received']}/{s['expected']}, hở {s['gaps']}, "
             f"trùng {s['dups']}, tràn {s['overflow']}", "đủ (§4.3 điều 6)", s["ok"])
    item("event_lines_bad", ev["bad"] + data_bad, "0", ev["bad"] + data_bad == 0)

    # T4 IMU ở cổng SPI (phần HIL) và tới callback (phần DUT)
    imu = imu_counts(ev, t0, t1, period_us)
    res["imu"] = imu
    for r in imu["kinds"]:
        item(f"imu_{r['report_type']}_windows", f"{r['windows_bad']}/{r['windows']} cửa sổ ngoài khoảng "
             f"(min {r['min_win']}, max {r['max_win']})", f"0 (∈ [{WIN_LO}; {WIN_HI}]·f, T4)",
             r["windows"] > 0 and r["windows_bad"] == 0)
    g = next((r for r in imu["kinds"] if r["report_type"] == "gyro"), None)
    if g:
        item("imu_gyro_gap_max_us", g["gap_max_us"], f"≤ T_age {g['t_age_us']} (gồm khoảng qua hai biên lượt, S27)",
             g["gap_max_us"] is not None and g["t_age_us"] is not None and g["gap_max_us"] <= g["t_age_us"])
        if not g["gap_edges_known"]:
            res["notes"].append("không có report gyro ngay trước #run_start hoặc ngay sau #run_end: khoảng hở qua "
                                "biên lượt không kiểm được, không khẳng định hở gyro 'suốt lượt'")
    res["resets_in_run"] = [r for r in ev["R"] if t0 <= r["t_us"] <= t1]
    res["resets_near_start"] = [r for r in ev["R"] if abs(r["t_us"] - t0) <= RESET_NEAR_US]
    # S7 / V4-1: a reset-complete within ±1 s of #run_start makes the run not valid (measure again);
    # the H7 side of the condition is checked by tools/s06_frames_decode.py
    item("resets_near_run_start", len(res["resets_near_start"]), "0 trong ±1 s quanh #run_start (V4-1)",
         not res["resets_near_start"], "valid")
    # S30: load condition of the run: the relay must not report bad frames; its good count and the
    # encoder movement are reported so the scoring tools can tell a run without the load
    S = [s for s in ev["S"] if t0 <= s["t_us"] <= t1]
    if len(S) >= 2:
        relay_ok_d, relay_bad_d = S[-1]["relay_ok"] - S[0]["relay_ok"], S[-1]["relay_bad"] - S[0]["relay_bad"]
        item("relay_bad_run", relay_bad_d, "0 (relay H7 → ESP32 không có frame sai)", relay_bad_d == 0)
    else:
        relay_ok_d = relay_bad_d = None
        item("relay_stats_in_run", len(S), "≥ 2 dòng S trong lượt (đọc bộ đếm relay)", False)
    moved = [any(E[i]["c"][w] != E[0]["c"][w] for i in range(len(E))) for w in range(4)] if E else [False] * 4
    res["load"] = dict(relay_ok_run=relay_ok_d, relay_bad_run=relay_bad_d, encoder_moved=moved)
    dut = imu["dut"]
    if any(r["n_overwritten"] for r in imu["kinds"]) or any(dut.values()):
        res["notes"].append("DUT: có report không tới callback hoặc bộ đếm bỏ > 0: thêm dòng bug log "
                            "bug_class DUT (luật 'đếm ở cổng SPI'); không làm phần HIL không đạt")
    # Hở gyro do firmware tính ở callback: chỉ báo kèm, thông tin DUT (spec T4 (2), Liam duyệt 29/10)
    res["dut_fw_gyro_gap_max_us"] = re_["kv"].get("gyro_gap_max_us")
    failed = [i for i in res["items"] if not i["ok"]]
    # S26 (Z19 order): lost data and product faults already seen stay a FAILURE; "not valid (measure
    # again)" only when the condition items are the only ones failed
    if any(i["category"] != "valid" for i in failed):
        res["status"] = "fail"
        if any(i["category"] == "valid" for i in failed):
            res["notes"].append("có mục không đạt và cả điều kiện lượt không hợp lệ: vẫn tính là không đạt (Z19)")
    elif failed:
        res["status"] = "invalid"
    else:
        res["status"] = "pass"
    return res, ev, imu


def imu_counts(ev, t0, t1, period_us):
    # Ghép trên cả luồng (callback đầu lượt có thể là report tới cổng SPI ngay trước #run_start,
    # callback cuối có thể chạy sau #run_end), rồi chỉ đếm report có t_us ở cổng SPI trong lượt.
    J_all, I = ev["J"], ev["I"]
    delivered, unmatched, repeat, i_to_j = match_reports(J_all, I)
    J = [j for j in J_all if t0 <= j["t_us"] <= t1]
    j_in_cb = {id(J_all[k]) for k in delivered}
    kinds = []
    f_hz = 1e6 / period_us if period_us else None
    n_win = (t1 - t0) // WIN_US
    for kind, name in KINDS.items():
        jk = [j for j in J if j["kind"] == kind]
        ts = sorted(j["t_us"] for j in jk)
        # S27: a gap crossing a run boundary counts too: add the report just before #run_start and the one
        # just after #run_end (the logger keeps lines around the run) to the series of gaps
        all_t = sorted(j["t_us"] for j in J_all if j["kind"] == kind)
        before = [t for t in all_t if t < t0][-1:]
        after = [t for t in all_t if t > t1][:1]
        ts_gap = before + ts + after
        edges_known = bool(before) and bool(after)
        wins = [0] * n_win
        for t in ts:
            w = (t - t0) // WIN_US
            if 0 <= w < n_win:
                wins[w] += 1
        bad = [c for c in wins if not (WIN_LO * f_hz <= c <= WIN_HI * f_hz)] if f_hz else wins
        n_cb = sum(1 for j in jk if id(j) in j_in_cb)
        ik = [k for k, i in enumerate(I) if i["kind"] == kind and k in i_to_j
              and t0 <= J_all[i_to_j[k]]["t_us"] <= t1]
        kinds.append(dict(
            report_type=name, n_spi=len(jk), n_spi_queue_drop=sum(1 for j in jk if not j["queued"]),
            n_cb=n_cb, n_overwritten=len(jk) - n_cb,
            cb_lines=len(ik), cb_repeat=sum(1 for k in repeat if k in ik),
            cb_not_applied=sum(1 for k in ik if not I[k]["applied"]),
            cb_unmatched=sum(1 for k in unmatched if I[k]["kind"] == kind and t0 <= I[k]["t_us"] <= t1),
            windows=n_win, windows_bad=len(bad), min_win=min(wins, default=None), max_win=max(wins, default=None),
            gap_max_us=max([b - a for a, b in zip(ts_gap, ts_gap[1:])], default=None),
            gap_edges_known=edges_known,
            t_age_us=int(T_AGE_PERIODS * period_us) if period_us else None))
    re_ = first_text(ev, "run_end")
    drops = (re_["kv"].get("imu_drop_run", "-1/-1/-1") if re_ else "-1/-1/-1").split("/")
    dut = dict(layer1_rx_queue=int(drops[0]), layer2_cb_queue=int(drops[1]), layer3_cb_mutex=int(drops[2]))
    cb_rows = []
    for k, i in enumerate(I):
        j = i_to_j.get(k)
        if j is None and not (t0 <= i["t_us"] <= t1):
            continue
        if j is not None and not (t0 <= J_all[j]["t_us"] <= t1):
            continue
        cb_rows.append(dict(report_type=KINDS.get(i["kind"], i["kind"]), sequence=i["seq"], epoch=i["epoch"],
                            t_us=i["t_us"], t_spi_us=J_all[j]["t_us"] if j is not None else "",
                            applied=i["applied"], n=i["n"], lib_n=i["lib_n"],
                            value_1=i["v"][0], value_2=i["v"][1], value_3=i["v"][2],
                            src_frame="", value_src="", err="", ok=""))
    return dict(kinds=kinds, dut=dut, cb_rows=cb_rows, cb_unmatched_total=len(unmatched))


# ─── Ghi kết quả ─────────────────────────────────────────────────────────────

def write_csv(path, rows, cols, idx=True):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow((["idx"] if idx else []) + cols)
        for n, r in enumerate(rows):
            w.writerow(([n] if idx else []) + [r.get(c, "") if isinstance(r, dict) else r[c] for c in cols])


def write_outputs(out_dir, res, ev, imu):
    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, "run_check.csv"), res["items"], ["item", "category", "value", "limit", "ok"])
    if "streams" in res:
        write_csv(os.path.join(out_dir, "streams_check.csv"), res["streams"],
                  ["stream", "expected", "received", "gaps", "dups", "overflow", "ok"])
    if imu:
        write_csv(os.path.join(out_dir, "T4_imu_counts.csv"), imu["kinds"],
                  ["report_type", "n_spi", "n_cb", "n_overwritten", "n_spi_queue_drop", "cb_lines",
                   "cb_repeat", "cb_not_applied", "cb_unmatched", "windows", "windows_bad", "min_win",
                   "max_win", "gap_max_us", "t_age_us"])
        write_csv(os.path.join(out_dir, "T4_imu_cb.csv"), imu["cb_rows"],
                  ["report_type", "sequence", "epoch", "t_us", "t_spi_us", "applied", "n", "lib_n",
                   "value_1", "value_2", "value_3", "src_frame", "value_src", "err", "ok"])
    t0, t1 = res.get("t_start_us"), res.get("t_end_us")
    if t0 is not None:
        E = [e for e in ev["E"] if t0 - 1000 <= e["t_before"] <= t1 + 1000]
        write_csv(os.path.join(out_dir, "enc_reads.csv"),
                  [dict(eseq=e["eseq"], t_read_before_us=e["t_before"], t_read_after_us=e["t_after"],
                        fl=e["c"][0], fr=e["c"][1], rl=e["c"][2], rr=e["c"][3]) for e in E],
                  ["eseq", "t_read_before_us", "t_read_after_us", "fl", "fr", "rl", "rr"])
        write_csv(os.path.join(out_dir, "pwm_writes.csv"), [p for p in ev["P"] if t0 <= p["t_before"] <= t1 + 20_000],
                  ["qseq", "t_before", "t_after", "wheel", "speed", "gpio_a", "cmp_a", "gpio_b", "cmp_b"])
    summ = {k: v for k, v in res.items() if k not in ("imu",)}
    if imu:
        summ["imu"] = {"kinds": imu["kinds"], "dut": imu["dut"], "cb_unmatched_total": imu["cb_unmatched_total"]}
    with open(os.path.join(out_dir, "s06_esp32_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1, default=str)


def print_report(res):
    print(f"Lượt: {res['run_dir']}")
    for i in res["items"]:
        print(f"  [{'ĐẠT' if i['ok'] else 'KHÔNG'}] {i['item']}: {i['value']}  ({i['limit']})")
    imu = res.get("imu")
    if imu:
        print("  IMU theo loại (ở cổng SPI / tới callback / bị ghi đè):")
        for r in imu["kinds"]:
            print(f"    {r['report_type']:8s} {r['n_spi']} / {r['n_cb']} / {r['n_overwritten']}"
                  f"  (bỏ tầng 1: {r['n_spi_queue_drop']}, callback lặp report: {r['cb_repeat']}, "
                  f"không áp: {r['cb_not_applied']})")
        print(f"  Bộ đếm bỏ trong lượt (DUT): {imu['dut']}")
        print(f"  Hở gyro firmware tính ở callback (DUT, không chấm): {res.get('dut_fw_gyro_gap_max_us')} µs")
    for n in res["notes"]:
        print(f"  Ghi chú: {n}")
    print(f"  => {res['status']}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Kiểm lượt S06 từ log ESP32 s06_load")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--out", default=None, help="thư mục kết quả, chỉ khi một lượt (mặc định <run_dir>/check)")
    a = p.parse_args(argv)
    if a.out and len(a.run_dirs) > 1:
        p.error("--out chỉ dùng với một lượt")
    status = []
    for d in a.run_dirs:
        res, ev, imu = check_run(d)
        write_outputs(a.out or os.path.join(d, "check"), res, ev, imu)
        print_report(res)
        status.append(res["status"])
    if len(status) == 1:
        return {"pass": 0, "fail": 1, "insufficient": 2, "invalid": 3}[status[0]]
    print(f"Tổng {len(status)} lượt: đạt {status.count('pass')}, không đạt {status.count('fail')}, "
          f"thiếu #run_start/#run_end {status.count('insufficient')}, "
          f"bị loại vì số lần đọc count (đo lại) {status.count('invalid')}")
    return 0 if all(s == "pass" for s in status) else 1


if __name__ == "__main__":
    sys.exit(main())
