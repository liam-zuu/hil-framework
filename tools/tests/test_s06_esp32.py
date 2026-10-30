"""Test PC cho tools/s06_esp32_check.py và tools/s06_logger.py (log ESP32 env s06_load).

Dữ liệu giả lập theo đúng định dạng dòng ở đầu src/s06/s06_load.c (AGV feat/s06-load e5a31c2) và
logger.c; mỗi ca biết trước chỗ mất / ghi đè để kiểm script đếm đúng.
"""

import os
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import s06_esp32_check as chk  # noqa: E402
import s06_logger as lg  # noqa: E402

T0 = 1_000_000
PERIOD = 10_000
FIRST_SEQ = 101


def make_run(tmp_path, dur_s=3, auto_end=1, drop_gyro=(), overwrite_gyro=(), drop_qseq=(), data_dup=False,
             no_run_end=False, pre_start_report=False, rx_end_delay_us=None, name="run",
             imu_init="imu_init=0,imu_init_ms=412", enc_short=0, s_temp=None, drop_e=(), resets=(),
             relay_bad_end=0, run_stats=None, enc_reads=True):
    """Ghi esp32_events.txt, esp32_data.txt cho một lượt; trả thư mục. drop_e: eseq của dòng E mất trên
    đường UDP (ESP32 vẫn đọc); enc_short: ESP32 đọc ít lần hơn (nhịp đọc sai); resets: t_us của dòng R;
    relay_bad_end: relay_bad của dòng S cuối; run_stats: log_send_err_run của dòng #run_stats (None = không
    có trường này, firmware trước S37); enc_reads: dòng #run_stats có enc_reads_run = số lần ESP32 đọc (S41;
    False = firmware trước S41)."""
    t1 = T0 + dur_s * 1_000_000
    ev = []                                   # (t_us, dòng không có qseq, là text?)
    n_rep = dur_s * 1_000_000 // PERIOD
    for kind, off in ((1, 3000), (2, 5000), (8, 7000)):
        for k in range(n_rep):
            if kind == 2 and k in drop_gyro:
                continue
            t = T0 + off + k * PERIOD
            seq = k % 256
            ev.append((t, f"J,{{q}},{t},{kind},1,{seq},1", False))
            if kind == 2 and k in overwrite_gyro:
                continue                      # callback của report này đọc report sau (ghi đè)
            if kind == 2 and k - 1 in overwrite_gyro:
                # callback chậm của report trước chạy sau khi report này tới: đọc report này
                ev.append((t + 200, f"I,{{q}},{t + 200},{kind},1,{seq},1,0,0,0.1,0.2,0.3", False))
            ev.append((t + 1500, f"I,{{q}},{t + 1500},{kind},1,{seq},1,0,0,0.1,0.2,0.3", False))
    if pre_start_report:
        ev.append((T0 - 500, "J,{q},%d,2,1,255,1" % (T0 - 500), False))
        ev.append((T0 + 1000, "I,{q},%d,2,1,255,1,0,0,0.1,0.2,0.3" % (T0 + 1000), False))
    for k in range(dur_s):
        rb = relay_bad_end if k == dur_s - 1 else 0
        ev.append((T0 + (k + 1) * 1_000_000 - 1, "S,{q}," + f"{T0 + (k + 1) * 1_000_000 - 1}"
                   + f",1,0,0,0,0,0,10000,{(k + 1) * 50},{rb},0,0,0" + ("" if s_temp is None else f",{s_temp}"),
                   False))
    for t in resets:
        ev.append((t, "R,{q},%d,1" % t, False))
    ev.append((T0, "#run_start,first_data_seq=%d,timestamp_us=%d,profile=heavy,duration_s=%d,auto_end=%d,"
               "fw=abc123,dirty=0,env=s06_load,cfg_hash=deadbeef" % (FIRST_SEQ, T0, dur_s, auto_end)
               + ("," + imu_init if imu_init else ""), True))
    ev.append((T0 + 1, "#cfg,profile=heavy;duration_us=%d;auto_end=%d;imu_period_us=%d;enc_period_us=1000;"
               "heavy_swap_us=15000000;t6_hold_us=1000000" % (dur_s * 1_000_000, auto_end, PERIOD), True))
    ev.append((T0 + 2, "P,{q},%d,%d,0,500,3,512,8,0" % (T0 + 2, T0 + 5), False))
    n_data = (t1 - T0 - 2000 - 1) // 10_000 + 1
    last_seq = FIRST_SEQ + n_data - 1
    if not no_run_end:
        ev.append((t1, "#run_end,last_data_seq=%d,log_qdrop=0,log_send_err=0,timestamp_us=%d,reason=duration,"
                   "elapsed_us=%d,log_qdrop_run=0,ev_qdrop_run=0,enc_ovf_run=0,imu_drop_run=0/0/0,"
                   "gyro_gap_max_us=10000" % (last_seq, t1, t1 - T0), True))
        kv = ([f"log_send_err_run={run_stats}"] if run_stats is not None else []) + \
             ([f"enc_reads_run={(t1 - T0) // 1000 - enc_short}"] if enc_reads else [])
        if kv:
            ev.append((t1 + 1, "#run_stats," + ",".join(kv), True))
    ev.sort(key=lambda x: x[0])
    lines, q = [], 0
    rx_base = 50_000_000
    for t, l, is_text in ev:
        q += 1
        if q in drop_qseq:
            continue
        line = (l + f",qseq={q}") if is_text else l.replace("{q}", str(q))
        t_rx = rx_base + (t - T0) + 3000
        if rx_end_delay_us is not None and line.startswith("#run_end"):
            t_rx = rx_base + 3000 + rx_end_delay_us
        lines.append((t_rx, line))
    eseq = 0
    for k in range((t1 - T0) // 1000 - enc_short):
        eseq += 1
        tb = T0 + 500 + k * 1000
        if eseq in drop_e:
            continue
        lines.append((rx_base + tb - T0 + 4000, f"E,{eseq},{tb},{tb + 12},{k},{-k},{k},{-k}"))
    data = []
    for n in range(n_data):
        f = ["0"] * 66
        f[0], f[37] = str(T0 + 2000 + n * 10_000), str(FIRST_SEQ + n)
        data.append((rx_base + n * 10_000, ",".join(f)))
        if data_dup and n == 5:
            data.append((rx_base + n * 10_000 + 1, ",".join(f)))
    d = tmp_path / name
    d.mkdir()
    (d / "esp32_events.txt").write_text("".join(f"{t}\t{l}\n" for t, l in lines), encoding="utf-8")
    (d / "esp32_data.txt").write_text("".join(f"{t}\t{l}\n" for t, l in data), encoding="utf-8")
    return str(d)


def items(res):
    return {i["item"]: i for i in res["items"]}


def kind(res, name):
    return next(r for r in res["imu"]["kinds"] if r["report_type"] == name)


def test_clean_run_passes_and_counts(tmp_path):
    res, ev, imu = chk.check_run(make_run(tmp_path))
    assert res["status"] == "pass", [i for i in res["items"] if not i["ok"]]
    for name in ("accel", "gyro", "game_rv"):
        r = kind(res, name)
        assert (r["n_spi"], r["n_cb"], r["n_overwritten"]) == (300, 300, 0)
        assert r["windows"] == 3 and r["windows_bad"] == 0 and r["gap_max_us"] == PERIOD
        assert r["cb_unmatched"] == 0 and r["cb_repeat"] == 0
    s = {x["stream"]: x for x in res["streams"]}
    assert s["esp32_enc"]["received"] == s["esp32_enc"]["expected"] == 3000
    assert s["esp32_data"]["received"] == s["esp32_data"]["expected"]


def test_seq_wraps_past_255_still_matches(tmp_path):
    # 300 report mỗi loại > 256: số thứ tự 8 bit quay vòng; mọi callback vẫn ghép đúng report của nó
    res, _, imu = chk.check_run(make_run(tmp_path))
    rows = [r for r in imu["cb_rows"] if r["report_type"] == "gyro"]
    assert len(rows) == 300
    assert all(r["t_us"] - r["t_spi_us"] == 1500 for r in rows)


def test_overwrite_counted_as_dut_not_hil_fail(tmp_path):
    res, _, _ = chk.check_run(make_run(tmp_path, overwrite_gyro=(50, 120)))
    g = kind(res, "gyro")
    assert (g["n_spi"], g["n_cb"], g["n_overwritten"]) == (300, 298, 2)
    assert g["cb_repeat"] == 2
    assert res["status"] == "pass"
    assert any("DUT" in n for n in res["notes"])


def test_missing_gyro_reports_fail_gap(tmp_path):
    res, _, _ = chk.check_run(make_run(tmp_path, drop_gyro=(100, 101)))
    it = items(res)
    assert it["imu_gyro_gap_max_us"]["value"] == 30_000
    assert not it["imu_gyro_gap_max_us"]["ok"]
    assert it["imu_gyro_windows"]["ok"]              # 98 report trong cửa sổ vẫn ∈ [95; 105]
    assert res["status"] == "fail"


def test_window_count_fail(tmp_path):
    res, _, _ = chk.check_run(make_run(tmp_path, drop_gyro=tuple(range(100, 110))))
    assert not items(res)["imu_gyro_windows"]["ok"]
    assert kind(res, "gyro")["min_win"] == 90


def test_udp_loss_in_event_stream_fails(tmp_path):
    res, _, _ = chk.check_run(make_run(tmp_path, drop_qseq=(400,)))
    s = {x["stream"]: x for x in res["streams"]}
    assert s["esp32_events"]["gaps"] == 1
    assert res["status"] == "fail"


def test_duplicate_data_seq_fails(tmp_path):
    res, _, _ = chk.check_run(make_run(tmp_path, data_dup=True))
    assert items(res)["data_seq_dups"]["value"] == 1
    assert res["status"] == "fail"


def test_missing_run_end_is_insufficient(tmp_path):
    d = make_run(tmp_path, no_run_end=True)
    res, _, _ = chk.check_run(d)
    assert res["status"] == "insufficient"
    assert chk.main([d]) == 2


def test_report_before_start_not_unmatched(tmp_path):
    # report tới cổng SPI ngay trước #run_start, callback chạy sau: không tính là callback lạc
    res, _, _ = chk.check_run(make_run(tmp_path, pre_start_report=True))
    assert all(r["cb_unmatched"] == 0 for r in res["imu"]["kinds"])
    assert kind(res, "gyro")["n_spi"] == 300


def test_t9_rule_on_pi_clock(tmp_path):
    ok = make_run(tmp_path, auto_end=0, rx_end_delay_us=3_000_000 + 80_000, name="a")
    res, _, _ = chk.check_run(ok)
    assert items(res)["t9_rx_duration_minus_D_us"]["ok"]
    late = make_run(tmp_path, auto_end=0, rx_end_delay_us=3_000_000 + 150_000, name="b")
    res, _, _ = chk.check_run(late)
    assert not items(res)["t9_rx_duration_minus_D_us"]["ok"]


def test_outputs_written(tmp_path):
    d = make_run(tmp_path)
    assert chk.main([d]) == 0
    out = os.path.join(d, "check")
    for f in ("run_check.csv", "streams_check.csv", "T4_imu_counts.csv", "T4_imu_cb.csv", "enc_reads.csv",
              "pwm_writes.csv", "s06_esp32_summary.json"):
        assert os.path.exists(os.path.join(out, f)), f
    head = open(os.path.join(out, "T4_imu_cb.csv"), encoding="utf-8").readline().strip()
    assert head.startswith("idx,report_type,sequence,epoch,t_us")


# ─── Bộ nhận: ESP32 giả trên localhost ──────────────────────────────────────

def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def fake_esp32(cmd_port, data_port, evt_port, wait_stop, stop_event):
    """Trả lời START như s06_load: #ack, #run_start, vài dòng, #run_end (tự hết hoặc khi nhận STOP)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", cmd_port))
    s.settimeout(0.1)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ev = ("127.0.0.1", evt_port)
    q = 0

    def send(line, port=ev):
        out.sendto((line + "\n").encode(), port)

    while not stop_event.is_set():
        try:
            msg = s.recv(200).decode()
        except socket.timeout:
            continue
        q += 1
        send(f"#ack,t_us=1,ok=1,why=ok,cmd={msg},qseq={q}")
        if msg.startswith("START"):
            q += 1
            send(f"#run_start,first_data_seq=1,timestamp_us=1000,profile=zero,duration_s=1,"
                 f"auto_end={0 if wait_stop else 1},fw=abc,dirty=0,env=s06_load,cfg_hash=1,qseq={q}")
            out.sendto(b"1500," + b"0," * 64 + b"0\n", ("127.0.0.1", data_port))
            send("E,1,2000,2010,0,0,0,0")
            if not wait_stop:
                q += 1
                send(f"#run_end,last_data_seq=1,log_qdrop=0,log_send_err=0,timestamp_us=2000,reason=duration,"
                     f"elapsed_us=1000,qseq={q}")
        elif msg.startswith("STOP"):
            q += 1
            send(f"#run_end,last_data_seq=1,log_qdrop=0,log_send_err=0,timestamp_us=2000,"
                 f"reason={msg.split()[1]},elapsed_us=1000,qseq={q}")
    s.close()
    out.close()


@pytest.mark.parametrize("no_auto_end", [False, True])
def test_logger_start_stop_with_fake_esp32(tmp_path, no_auto_end):
    cmd, data, evt = free_port(), free_port(), free_port()
    stop = threading.Event()
    th = threading.Thread(target=fake_esp32, args=(cmd, data, evt, no_auto_end, stop), daemon=True)
    th.start()
    out = str(tmp_path / "cap")
    argv = ["--esp32", "127.0.0.1", "--profile", "zero", "--duration", "1", "--out", out,
            "--listen-before", "0.1", "--linger", "0.2", "--bind", "127.0.0.1",
            "--cmd-port", str(cmd), "--data-port", str(data), "--evt-port", str(evt),
            "--temp-path", str(tmp_path / "temp")]
    (tmp_path / "temp").write_text("45123\n")
    if no_auto_end:
        argv.append("--no-auto-end")
    t = time.monotonic()
    try:
        assert lg.main(argv) == 0
    finally:
        stop.set()
        th.join(1)
    y = open(os.path.join(out, "capture.yaml"), encoding="utf-8").read()
    assert "result: ok" in y and "build_diff_vs_car" in y
    evl = open(os.path.join(out, "esp32_events.txt"), encoding="utf-8").read().splitlines()
    assert any("\t#run_start," in l for l in evl) and any("\t#run_end," in l for l in evl)
    assert all(l.split("\t", 1)[0].isdigit() for l in evl)
    assert len(open(os.path.join(out, "esp32_data.txt"), encoding="utf-8").read().splitlines()) == 1
    temps = open(os.path.join(out, "pi_temp.txt"), encoding="utf-8").read().splitlines()
    assert temps and all(l.split("\t")[1] == "45123" for l in temps)
    assert f"samples: {len(temps)}," in y and "error: \"\"" in y
    if no_auto_end:
        assert "STOP pi_stop" in y
        assert time.monotonic() - t >= 1.0          # Pi gửi STOP sau đúng thời lượng
    else:
        assert "STOP" not in y


def test_imu_init_field_checked(tmp_path):
    res, _, _ = chk.check_run(make_run(tmp_path, name="ok"))
    assert items(res)["imu_init"]["ok"] and items(res)["imu_init"]["value"] == "mã 0, 412 ms"
    res, _, _ = chk.check_run(make_run(tmp_path, name="err", imu_init="imu_init=3,imu_init_ms=2000"))
    assert not items(res)["imu_init"]["ok"] and res["status"] == "fail"
    res, _, _ = chk.check_run(make_run(tmp_path, name="old", imu_init=""))
    assert items(res)["imu_init"]["value"] == "thiếu trường" and res["status"] == "fail"


def test_enc_read_count_makes_run_invalid_not_fail(tmp_path):
    # đọc thiếu 1 lần: vẫn hợp lệ (± 1); thiếu 2 lần (eseq vẫn liên tục): lượt không hợp lệ, đo lại
    res, _, _ = chk.check_run(make_run(tmp_path, name="one", enc_short=1))
    assert res["status"] == "pass"
    d = make_run(tmp_path, name="two", enc_short=2)
    res, _, _ = chk.check_run(d)
    assert res["status"] == "invalid"
    assert items(res)["enc_reads_vs_duration"]["category"] == "valid"
    assert {x["stream"]: x for x in res["streams"]}["esp32_enc"]["ok"]
    assert chk.main([d]) == 3


def test_multi_run_counts_excluded(tmp_path, capsys):
    a = make_run(tmp_path, name="a")
    b = make_run(tmp_path, name="b", enc_short=5)
    c = make_run(tmp_path, name="c", drop_gyro=(100, 101))
    assert chk.main([a, b, c]) == 1
    out = capsys.readouterr().out
    assert "đạt 1, không đạt 1" in out and "bị loại vì số lần đọc count (đo lại) 1" in out


def test_firmware_gyro_gap_reported_not_scored(tmp_path):
    res, _, _ = chk.check_run(make_run(tmp_path))
    assert res["dut_fw_gyro_gap_max_us"] == "10000"
    assert not any("fw" in i["item"] and "gap" in i["item"] for i in res["items"])


@pytest.mark.parametrize("s_temp, want", [(None, None), ("41.5", 41.5), ("nan", "nan")])
def test_s_line_with_esp32_temperature(tmp_path, s_temp, want):
    """dòng S 16 trường (temp_c của s06_load, T5 30/10) vẫn là dòng đúng; bản 15 trường cũ vẫn đọc được"""
    res, ev, imu = chk.check_run(make_run(tmp_path, s_temp=s_temp))
    assert res["status"] == "pass" and ev["bad"] == 0 and len(ev["S"]) == 3
    got = ev["S"][0]["temp_c"]
    if want == "nan":
        assert got != got
    else:
        assert got == want


def test_logger_temperature_unreadable_is_recorded(tmp_path):
    cap = lg.Capture(str(tmp_path / "cap"), free_port(), free_port(), "127.0.0.1",
                     temp_path=str(tmp_path / "missing"))
    cap.poll(0.01)
    cap.close()
    assert cap.temp_n == 0 and cap.temp_error.startswith("FileNotFoundError")
    assert (tmp_path / "cap" / "pi_temp.txt").read_text() == ""


def test_s26_lost_lines_stay_a_failure(tmp_path):
    # two E lines lost on the way to the PC: the ESP32 made every read (count by eseq, not by lines
    # received): the run is not "not valid", the lost data is a failure (Z19)
    res, _, _ = chk.check_run(make_run(tmp_path, drop_e=(1000, 1001)))
    it = items(res)
    assert it["enc_reads_vs_duration"]["ok"] and not it["stream_esp32_enc"]["ok"]
    assert res["status"] == "fail"
    # the ESP32 itself read too few times: read rhythm wrong, run not valid (measure again)
    res, _, _ = chk.check_run(make_run(tmp_path, enc_short=5, name="short"))
    assert res["status"] == "invalid"
    # both: still a failure
    res, _, _ = chk.check_run(make_run(tmp_path, enc_short=5, drop_e=(1000,), name="both"))
    assert res["status"] == "fail"


def test_s41_head_tail_loss_is_a_failure(tmp_path):
    # S41 (Codex Sol round 2/2): E lines 1 and 2 lost: no hole between the eseq received, the eseq span is
    # 2 short of the duration; before, this read as a rhythm fault ("invalid"); the producer count proves
    # the ESP32 read every time: lost data, a failure
    res, _, _ = chk.check_run(make_run(tmp_path, drop_e=(1, 2)))
    it = items(res)
    assert not it["enc_lines_vs_producer"]["ok"] and it["enc_reads_vs_duration"]["ok"]
    assert res["status"] == "fail"
    n = 3000                                         # last two reads of a 3 s run lost
    res, _, _ = chk.check_run(make_run(tmp_path, drop_e=(n - 1, n), name="tail"))
    assert res["status"] == "fail"
    # firmware without enc_reads_run: the stream cannot be proven complete: neither pass nor "invalid"
    res, _, _ = chk.check_run(make_run(tmp_path, enc_reads=False, name="old"))
    assert res["status"] == "insufficient" and not items(res)["enc_reads_run_known"]["ok"]


def _with_run_ids(d, late_read=False):
    """Rewrite a make_run directory in the manifest format: run=1 on the marks, run field on E and P."""
    p = os.path.join(d, "esp32_events.txt")
    out = []
    for raw in open(p, encoding="utf-8").read().splitlines():
        t, _, l = raw.partition("\t")
        if l.startswith(("#run_start,", "#run_end,", "#run_stats,")):
            name, _, rest = l.partition(",")
            l = f"{name},run=1,{rest}"
        elif l.startswith(("E,", "P,")):
            l += ",1"
        out.append(f"{t}\t{l}")
    if late_read:                                    # the last read: counted before END, stamped 5 ms after it
        i = max(k for k, x in enumerate(out) if "\tE," in x)
        t, _, l = out[i].partition("\t")
        f = l.split(",")
        f[2], f[3] = str(T0 + 3_000_000 + 5000), str(T0 + 3_000_000 + 5012)
        out[i] = t + "\t" + ",".join(f)
    open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
    return d


def test_s58_run_id_pairs_marks_and_lines(tmp_path):
    res, _, _ = chk.check_run(_with_run_ids(make_run(tmp_path)))
    assert res["status"] == "pass"
    # S59: the read in flight at END carries the run field: it counts, it is not "lost"
    d = _with_run_ids(make_run(tmp_path, name="late", enc_short=-1), late_read=True)
    res, _, _ = chk.check_run(d)
    assert res["status"] == "pass", [i for i in res["items"] if not i["ok"]]
    # Codex S58: run 1 lost its first E line and its #run_stats; the #run_stats of run 2 says 2999
    d = make_run(tmp_path, name="two", drop_e=(1,), enc_reads=False)
    _with_run_ids(d)
    p = os.path.join(d, "esp32_events.txt")
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"99000000\t#run_stats,run=2,enc_reads_run=2999,qseq=999999\n")
    res, _, _ = chk.check_run(d)
    assert res["status"] != "pass" and not items(res)["enc_reads_run_known"]["ok"]


def test_s58_two_runs_in_one_directory(tmp_path):
    d = _with_run_ids(make_run(tmp_path))
    with open(os.path.join(d, "esp32_events.txt"), "a", encoding="utf-8") as f:
        f.write("99000000\t#run_start,run=2,first_data_seq=9,timestamp_us=99000000,profile=heavy,duration_s=3,"
                "auto_end=1,qseq=999999\n")
    res, _, _ = chk.check_run(d)
    assert res["status"] == "insufficient" and not items(res)["one_run_per_file"]["ok"]


def test_s7_reset_near_run_start_not_valid(tmp_path):
    res, _, _ = chk.check_run(make_run(tmp_path, resets=(T0 + 400_000,)))
    assert not items(res)["resets_near_run_start"]["ok"] and res["status"] == "invalid"
    res, _, _ = chk.check_run(make_run(tmp_path, resets=(T0 + 1_500_000,), name="later"))
    assert items(res)["resets_near_run_start"]["ok"] and res["status"] == "pass"


def test_s27_gyro_gap_across_run_start(tmp_path):
    # the last gyro before START at T0 - 500, the first four of the run lost: 45.5 ms without gyro;
    # every 1 s window still has 96 reports (in [95; 105])
    res, _, _ = chk.check_run(make_run(tmp_path, pre_start_report=True, drop_gyro=(0, 1, 2, 3)))
    g = kind(res, "gyro")
    assert g["windows_bad"] == 0 and g["gap_max_us"] == 45_500
    assert not items(res)["imu_gyro_gap_max_us"]["ok"] and res["status"] == "fail"


def test_s30_relay_bad_fails_and_load_reported(tmp_path):
    res, _, _ = chk.check_run(make_run(tmp_path, relay_bad_end=1))
    assert not items(res)["relay_bad_run"]["ok"] and res["status"] == "fail"
    res, _, _ = chk.check_run(make_run(tmp_path, name="ok"))
    assert res["load"]["relay_ok_run"] == 100 and res["load"]["encoder_moved"] == [True] * 4


def test_s37_send_err_from_run_stats(tmp_path):
    # the ESP32 counted 2 send errors from START (one before the first DATA the PC got): from #run_stats
    res, _, _ = chk.check_run(make_run(tmp_path, run_stats=2))
    it = items(res)["log_send_err_run"]
    assert it["value"] == 2 and not it["ok"] and "#run_stats" in it["limit"]
    res, _, _ = chk.check_run(make_run(tmp_path, name="old"))   # older firmware: fallback, noted
    assert items(res)["log_send_err_run"]["ok"] and any("firmware cũ" in n for n in res["notes"])
