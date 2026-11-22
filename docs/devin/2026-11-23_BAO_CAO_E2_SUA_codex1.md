# BAO_CAO — E2, sửa sau phản biện Codex vòng 1 (C3-14, C3-15, C3-16, C3-17)

Worktree `D:/HIL_WT/devin_e2`. Chỉ làm 4 ID ACCEPTED của bộ E2 (`tham_dinh_vong1.md` dòng 24–27). Không commit.

File sửa: `tools/hil_gioi_han.py`, `tools/s06_tre_vong.py`, `tools/tests/test_hil_gioi_han.py`, `tools/tests/test_s06_tre_vong.py`. File tạo: `ket_qua_e2_v2/` (lô mới), `BAO_CAO.md` này. `ket_qua_e2/` giữ nguyên.

## 1. ID → sửa ở đâu → test → kết quả

| ID | Sửa ở đâu (file:dòng) | Test | Kết quả |
|---|---|---|---|
| C3-14 | `tools/hil_gioi_han.py`: docstring mục (c) 41–61; `GAP_FACTOR`, `KIND_INVALID` 198–199; `longest_run(mask, cut=None)` 269; `tol_segments(kind, cut=None)` 314; `frame_validity` 329 (mới); `count_run` 484, khối C3-14 515–524 và cột mới 552–565; `esp_measure` dấu 620, tìm vào dải 637 (bỏ frame không valid thay vì chỉ frame rỗng) | `test_c3_14_empty_miso_not_duty0` 503, `test_c3_14_crc_and_gaps_cut_segments` 529, `test_c3_14_bad_crc_frame_not_first_in_band` 559 | Đạt |
| C3-15 | `tools/hil_gioi_han.py`: docstring (c-ESP32) 63–68, 97–106, 118–131; `A11_STAT`, `a11_cols` 201–206; `ROW_COLS`, `EVENT_COLS` 231–241; hàm `event` trong `esp_measure` 651–677; `a11_stats` 749 (mới) | `test_c3_15_single_frame_in_band` 585, `test_c3_15_censored_by_log_end` 614; đổi tên cột trong `test_normal_hold` 268, `test_motor_off_events` 415, `test_seqgap_event_skipped`, `test_wrong_sign_not_scored`, `test_motor_off_outside` | Đạt |
| C3-16 | `tools/s06_tre_vong.py`: docstring 46–86; `CSV_COLS`, `T3_SUMM_KEYS`, `SUMS`, `CHI_SO` 112–119; `bind_t3` 209, `bind_t6` 231, `status` 270 (mới); `main`: `--chot` 280, gọi ràng buộc 296–299, đếm chuyển mức bị cắt 321 và 340–342, khối trạng thái 376–420, cột `trang_thai`/`bat_dinh_ms` 436–440, `chot` 442–453, summary 454–470, mã thoát 478–480 | `test_c3_16_codex_counterexample` 233, `test_c3_16_u_boundary` 255, `test_c3_16_t6_evidence_gates` 267, `test_c3_16_binding_mismatch` 279, `test_c3_16_chot` 306; sửa `test_known_stats` 113, `test_summary_gates` 194 | Đạt |
| C3-17 | `tools/hil_gioi_han.py`: docstring (a) 21–30, lô 142–146; `score_window` 346 (mới); `count_run` khối C3-17 567–577; `batch` `--cham` 837–846; `main` `--score-json`, `--cham` 880–897 | `test_c3_17_score_window` 647, `test_c3_17_score_json_mismatch` 676, `test_c3_17_batch_cham` 686 | Đạt |

### Mỗi ID đã sửa gì

**C3-14** (frame MISO rỗng hay lỗi bị tính là duty 0):
- **Cột tương thích `dem_s1`.** Chỉ các cột trong `CMP_SAME` giữ đúng định nghĩa cũ để `so_khop_s1` còn so được: `n_duty0_any`, `n_sat_any`, `n_flat_any`, `max_flat_ms`, `n_all4_duty0`, `max_all4_0_ms`, `n_miso_empty_run`. Docstring ghi rõ các cột này không dùng làm bằng chứng.
- **Frame đủ bằng chứng.** Là frame có `protocol.miso_check == MISO_OK`, tức không rỗng và đúng CRC; cùng luật với `s06_t6_check`.
- **Khoảng mất mẫu.** Seq MOSI không tăng đúng 1, hoặc bước `t_send` lớn hơn 1,5 chu kỳ (ngưỡng 1,5 là [ĐỀ XUẤT]).
- **Mọi số còn lại là số bằng chứng.** Gồm `n_d0_*`, `n_sat_*`, các cột dung sai `*_tol_*`, `doan_phang/` và các cột mới `*_bc`. Các số này chỉ đếm frame valid; đoạn liền bị cắt tại frame không valid và tại khoảng mất mẫu.
- **Cột báo riêng:** `n_miso_crc_bad_run`, `n_mat_mau_run`, `n_khong_bc_run`.
- **A11 và kiểm dấu** bỏ cả frame sai CRC, không chỉ frame rỗng.

**C3-15** (A11 gắn "theo kịp" ngay ở frame đầu vào dải):
- **Đổi tên kết quả.** Nay gọi là thời gian tới lần đầu quan sát vào dải: `t_vao_dai_ms`, `vao_dai`, `k_vao_dai`. Các cột dòng đổi tương ứng: `n_follow_*` thành `n_vao_dai_*`, `hold_*` thành `t_vao_dai_*`.
- **Mỗi sự kiện ghi thêm:**
  - `moc_t0`: `udp_envelope` hay `udp_envelope+tick_dt_med`.
  - `bat_dinh_ms`: phần dư của đường khớp, cộng độ tản chu kỳ tick với sự kiện `tat_motor`.
  - `n_lien_vao_dai`: số frame valid liền nhau trong dải, tính từ lần vào đầu.
  - `n_khong_bc`: số frame không quan sát được trên đoạn đã quét.
  - `cat`: lý do bị cắt, `doi_lenh` hay `het_log`.
- **Cột dòng mới:** `n_cat_doi_lenh_*`, `n_cat_het_log_*`, `n_ev_mat_mau_*`.
- **Độ lệch một phía chưa đo.** Trễ UDP nhỏ nhất làm số đo ngắn hơn thật; docstring ghi điều này kèm ví dụ mốc trễ 4 ms. Dung sai vẫn là 0,0055.
- **Không làm phần T6.** Phần "nếu dùng cho chỉ báo T6 thì phải có mốc ghi compare và kiểm 100 frame liền" không làm: theo `tham_dinh` dòng 25, số này không được gọi là theo kịp theo nghĩa T6.

**C3-16** (`s06_tre_vong` trả mã 0 khi thiếu bằng chứng):
- **Ràng buộc CSV với summary; lệch thì mã 2.**
  - `T3_summary.summary` phải khớp `T3_result.csv` ở các số đếm `steps`, `measured`, `expired`, `expired_ii`, `extra_pulse`, `no_data`, `unsynced`, và `delay_max_us` trong ±1e-3 µs (CSV ghi 3 số lẻ).
  - `T6_summary.follow` phải khớp `T6_follow.csv` theo từng cặp mức: `n`, `not_settled`, `follow_ms_max`.
- **Cột `trang_thai` cho mỗi thành phần và mỗi tổng.** Một thành phần chỉ "đủ" khi đạt cả bốn điều:
  1. Không bậc T3 nào hết hạn hay không đo được (với (ii), không bậc nào thiếu (ii)).
  2. `u_us` ≤ 250 µs + 1e-9.
  3. Phía T6: `sync` là `manual`, `limit` = 0,0055, `run_frames_bad` = 0, `frame_log_faults` rỗng, `qseq_stream` ok và known.
  4. Không chuyển mức không phẳng nào bị cắt, và có mẫu.
- **Bất định:** cột `bat_dinh_ms` của tổng = (u_us + t_mark_span_us_max) / 1000.
- **Ghi rõ phương pháp.** Trung vị và p95 lấy trên tích Descartes của hai lượt, giả định độc lập; ghi ở `ghi_chu` và ở `summary.phuong_phap_tong`.
- **`--chot TP:CS`.** Chỉ thống kê đã chốt, và chỉ khi tổng đó "đủ", mới vào `summary.chot.dau_vao_2ms`.
- **Mã thoát 0** khi tổng cần dùng đều "đủ": tổng đã chốt nếu có `--chot`, cả S1 lẫn S2 nếu không.

**C3-17** (cửa sổ đếm dải 10–14,6 mm/s):
- **Cột cũ giữ làm số tương thích.** `n_*_win`, `win_kind`, `win_n` giữ nguyên định nghĩa `dem_s1` (cửa sổ danh định, hoặc 1 s cuối) và được ghi là tương thích.
- **Cột mới `*_ht`** (`win_ht_src`, `t_ht0_s`, `t_ht1_s`, `win_ht_n`, `n_vband_ht`, `n_wband_any_ht`, `n_wb_ht_*`) dùng mặt nạ của bộ chấm `hil_o1_score.score`:
  - frame k ≥ k0 của bộ chấm;
  - t = (k − t0_step) × chu kỳ;
  - T_cmd − 1e-9 ≤ t < t_deadline − 1e-9;
  - t ≤ t_end + 1e-9.
- **Nguồn mặt nạ.** Đọc từ `--score-json`, hoặc ở lô `--cham` chạy chính bộ chấm rồi lưu `cham/<run>__<ten>_score.json`. JSON không khớp log thì mã 2.
- **Không có kết quả chấm** (gồm lúc bỏ `--task`): ghi `khong_co_ket_qua_cham`, cột `*_ht` để rỗng. Số 1 s cuối chỉ còn ở cột mô tả `*_last1s`.

### Test hỏng trên code cũ, đạt trên code mới: đã kiểm thế nào

Trước khi sửa, tôi chép hai file gốc ra `_cu/` (xoá sau khi kiểm xong).

**(a) Chạy bộ test mới với module cũ.** Tôi nạp module cũ trước để pytest dùng nó.
- `hil_gioi_han`: `python -c "...sys.path[:0]=['.../_cu',...]; import hil_gioi_han ...; pytest.main([... '-k','c3_' ...])"` cho `8 failed`. Đó là đủ 8 test C3-14/15/17 của file này.
- `s06_tre_vong`: nạp bằng `importlib` từ `_cu/s06_tre_vong.py` cho `6 failed, 5 passed`. Phản ví dụ Codex hỏng đúng ở `assert code == 3` với mã cũ trả 0 (dòng 239). Biên u = 250,001 cũng trả 0 (dòng 260). `run_frames_bad=1` trả 0. Summary lệch trả 0 thay vì 2. `--chot` thì không có trong code cũ.

**(b) Chạy chính các phản ví dụ trên code cũ.** Vì vài test ở (a) hỏng sớm do thiếu tên cột, tôi chạy các phản ví dụ bằng script tạm `_cu/kiem_cu.py` để thấy hành vi cũ sai, không chỉ tên cột thiếu:

| Phản ví dụ | Code cũ | Code mới |
|---|---|---|
| C3-14: 60 frame rỗng, duty thật 0,5 | `n_d0_tol_fl` 60, một đoạn phẳng 60 ms | 0 |
| C3-14: 1 frame sai CRC, 3 khoảng mất mẫu | 1 đoạn 1 900 ms | 5 đoạn, dài nhất 500 ms |
| C3-14: frame sai CRC ở vị trí vào dải | nhận frame 1253 làm lần vào dải | lấy 1254, `n_khong_bc` = 1 |
| C3-15: chuỗi −0,2 → 0,004 → −0,2 | `followed True`, `hold_ms 7.276` | `vao_dai` True, `n_lien_vao_dai` = 1 |
| C3-17: chuỗi đúng phản ví dụ | `n_vband_win` 2000 | `n_vband_ht` 501 |

(Lệnh `python _cu/kiem_cu.py`.)

**(c) Trên code mới, cả hai file đạt** (mục 2).

## 2. Lệnh test và dòng tổng kết nguyên văn

```
python -m pytest -q tools/tests/test_hil_gioi_han.py tools/tests/test_s06_tre_vong.py
31 passed in 25.72s
```
(Trước khi sửa, cùng lệnh cho `18 passed in 29.53s`. Riêng `test_s06_tre_vong.py` sau sửa: `11 passed in 3.73s`.)

## 3. Chạy lại lô vào `ket_qua_e2_v2/` và khác biệt với `ket_qua_e2/`

```
python tools/hil_gioi_han.py --batch-from-s1 D:/CLAUDE_THESIS/05_DATA/results/HIL_GIOI_HAN/2026-11-22_run01/ket_qua_s1.csv --root D:/CLAUDE_THESIS/05_DATA/results/CLOSED_LOOP --out D:/HIL_WT/devin_e2/ket_qua_e2_v2 --cham
rows=158 errors=0 no_esp=10 mismatches=0
#ket_qua,ma=3,ly_do=thieu_bang_chung
```
Dòng tổng kết giống lần trước. Lệnh khác lần trước ở cờ `--cham`, cờ này chỉ thêm các cột `*_ht`.

Các số dưới lấy từ script so sánh tạm `_cu/so_sanh.py` và một lệnh `python -c` trên `ket_qua_e2_v2/`. Hai script tạm trong `_cu/` đã xoá cùng thư mục; chúng không thuộc phần giao.

**Định dạng.**
- 158 dòng; số cột từ 131 lên 167.
- 20 cột A11 cũ đổi tên; giá trị mới khớp từng dòng với cột cũ tương ứng (0 dòng lệch).
- `so_khop_s1.csv`: 0 dòng lệch, như lần trước.

**C3-14.**
- Trên cả lô: 18 frame rỗng (3 dòng), 44 frame sai CRC (4 dòng), 0 khoảng mất mẫu; tổng 62 frame không đủ bằng chứng.
- Số đổi ở 3 dòng `2026-11-08_run09` (w1, w2, w3), tính số cũ → số mới:
  - `n_duty0_any_bc` so với cột tương thích: 361 → 354, 353 → 348, 370 → 363.
  - `n_all4_duty0_bc`: 359 → 353, 353 → 348, 370 → 363.
  - `n_d0_tol_any` và `n_flat_tol_any`: 379 → 372, 383 → 378, 379 → 372.
- `n_d0_*`, `n_d0_tol_*`, `n_seg_tol_*`, `seg_med_ms_*` cũng đổi ở 3 dòng này. `doan_phang/`: 3 trên 158 file đổi.
- `max_flat_ms`, `max_all4_0_ms`, `max_flat_tol_ms` không đổi ở dòng nào.
- `corr_*` đổi ở 1 dòng (`2026-11-09_run15/b1_circ90`, 6 frame sai CRC; `corr_fl` −0,976524 → −0,976525).
- Sự kiện A11: 0 thay đổi về số sự kiện và giá trị; không sự kiện nào có frame không valid trong đoạn quét.

**C3-15** (cùng số sự kiện như lần trước).
- `lenh`: 590 sự kiện vào dải và 169 bị cắt, cả 169 do `doi_lenh`.
- `tat_motor`: 136 sự kiện, cả 136 bị cắt do `het_log`.
- Số mới về `n_lien_vao_dai`: trong 484 sự kiện "đổi thật" có vào dải, 142 sự kiện vào dải trước 10 ms. Trong 142 sự kiện đó, 9 sự kiện có `n_lien_vao_dai` = 1, 35 sự kiện dưới 10 frame, trung vị 10 frame. Nhóm trên 40 ms (322 sự kiện) có trung vị 2 842,5 frame.
- `bat_dinh_ms`: `lenh` 0,139 / 2,120 / 4,319 ms; `tat_motor` 2,252 / 6,814 / 9,280 ms (nhỏ nhất / trung vị / lớn nhất).

**C3-17.**
- 24 dòng O1 đều chấm được. Mốc t0: 18 dòng là `esp32_run_active_edge`, 6 dòng là `lifecycle_RUN`. Kết quả `completed`: 20 `true`, 3 `inconclusive`, 1 `false`.
- Tổng trên 24 dòng: `n_vband_win` 2 013 → `n_vband_ht` 1 917 (13 dòng khác); `win_n` 45 409 → `win_ht_n` 33 467; `n_wband_any_win` 8 962 → `n_wband_any_ht` 8 741 (17 dòng khác).
- 134 dòng không phải O1 ghi `khong_co_ket_qua_cham`.

## 4. Điểm mơ hồ

1. **Mặt nạ cửa sổ hoàn thành (C3-17).** Spec không có định nghĩa viết sẵn (`De_xuat_spec_..._v6.md:387`: "trong cửa sổ hoàn thành"). Tôi dùng [T_cmd; t_end] ∩ [.., t_deadline) với các mốc lấy từ kết quả của `hil_o1_score.score`, theo `evaluate_v7` (`fwsim/o1_feasibility.py:231–235, 302–305`). Chưa tính phần cửa sổ giữ t_d trước T_cmd.
2. **Chạy bộ chấm với `allow_pre_b81_15=True` và `esp_log` (C3-17).** Các log 10/11 và 11/11 có trước kiểm B81-15, nên không có cờ này bộ chấm từ chối. Có log ESP32 thì bộ chấm dùng cạnh run_active (18 dòng của 15/11). Cần Claude xác nhận đây là "kết quả chấm" được duyệt cho phép đếm này.
3. **Khoảng mất mẫu (C3-14).** Tôi định nghĩa là seq MOSI không tăng đúng 1, hoặc bước `t_send` lớn hơn 1,5 chu kỳ. Ngưỡng 1,5 là [ĐỀ XUẤT]. Trên lô thật cách này cho 0 chỗ cắt; bước `t_send` lớn nhất khoảng 1,07 ms (lệnh dò trên 158 npz).
4. **Cỡ mẫu tối thiểu cho median/p95 (C3-16).** Spec không có số này. Tôi đòi "đủ" theo nghĩa không có phần bị cắt, không đặt n tối thiểu. Một lượt T3/T6 đầy đủ mà rất ít bậc vẫn "đủ".
5. **Thống kê A12 nào dùng cho điều kiện ≥ 2 ms** chưa chốt (S1 hay S2; median, p95 hay max). Code chỉ nhận qua `--chot`; không có `--chot` thì `dau_vao_2ms` = None và mã 0 đòi cả S1 lẫn S2 "đủ".
6. **Cổng T6 trong C3-16** dùng các trường có sẵn trong `T6_summary.json` (`run_frames_bad`, `frame_log_faults`, `qseq_stream`, `sync`, `limit`). `u_duty` không được đòi vì là bất định duty, không phải thời gian. `s06_t6_check.LIMIT` vẫn là 0,006, nên mọi `T6_summary` hiện tại sẽ ra "thiếu", như trước.

## 5. Phần chưa làm và lý do

- **C3-15, phần chỉ báo T6** (mốc ghi compare, lúc rời mức cũ, 100 frame liền): không làm. Theo `tham_dinh_vong1.md` dòng 25, số A11 trên log cũ không gọi là theo kịp theo nghĩa T6, và log cũ không có mốc ghi compare.
- **Bất định chưa được đưa vào cột io.** `io_floor_*` / `io_ceil_*` của `s06_tre_vong` chưa tính lại theo `bat_dinh_ms`; bất định mới vào cột `trang_thai`/`bat_dinh_ms`. Đề nói "tính vào trạng thái kết quả", nên tôi không đổi nghĩa các cột io.
- **Chưa chạy `s06_tre_vong` trên dữ liệu S06 thật** vì chưa có lượt T3/T6.
