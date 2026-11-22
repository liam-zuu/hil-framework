# Báo cáo sửa sau phản biện Codex vòng 1: bộ E3 (chỉ C3-18)

Worktree `D:/HIL_WT/devin_e3`. Nguồn việc: `03_CODE/docs/review/2026-11-22_Codex_C3_dot1/tra_loi_vong1.md:57` (C3-18) và `tham_dinh_vong1.md:28` (Claude: ACCEPTED, P2: "chọn ω đặt cho mã int16 8187 (phần lẻ khoảng 0,550, gần 0,5 hơn) theo spec 'chỉnh ω đặt nếu cần' (CC1-10); lưu mã int16, ω giải mã, nửa chu kỳ trước/sau làm tròn; test đủ lần ghé, hai dấu, điểm chỉ báo"). Spec: `06_SPECS/2026-10-28_Spec_S06_Nghiem_thu_HIL.md:119` (T5, `[v3.5, câu 7 … chỉnh ω đặt nếu cần để phần lẻ gần 0,5 (CC1-10)]`). Không làm ID nào khác.

## 1. ID → sửa ở đâu → test → kết quả

| ID | Sửa ở đâu (file:dòng) | Test | Kết quả |
|---|---|---|---|
| C3-18 (chọn mã 8187) | `rpi5/t5_schedule.py:32-33` `OFF_GRID_POINTS`, `POINTS`: 9,995 → **9,9942** (ω đặt ở gần tâm mã 8187: 9,9942/40·32767 = 8186,9988, cách hai biên mã khoảng 0,5); docstring `:4-8` | `rpi5/tests/test_t5_schedule.py::test_c3_18_point_9995_moved_to_code_8187` (frame `encode_state` cả hai dấu cho ±8187, cả khi lệch ±1e-9; trong các mã 8186–8189 thì 8187 có phần lẻ gần 0,5 nhất: 0,5502 so với 0,4338 của 8188; timer 953 so với 952); `test_v35_eleven_points_both_directions` (sửa giá trị mong đợi, xem dưới) | hỏng trên code cũ, đạt trên code mới |
| C3-18 (lưu mã int16, ω giải mã, nửa chu kỳ trước/sau làm tròn) | `rpi5/t5_schedule.py:34-41` `OFF_GRID_LOCK` (4 điểm lệch lưới); `:116-153` `_f32`, `H7_OMEGA_LSB`, `H7_HP_NUM`, `h7_timer`, `h7_off_grid`, `check_off_grid_lock`; `:161` `main()` gọi `check_off_grid_lock()` trước khi ghi CSV (lệch thì `ValueError`, không ghi file) | `test_c3_18_lock_matches_frame_and_h7_both_signs` (đối chứng độc lập trong test bằng `numpy.float32` và byte frame `struct '>4h'`, cả hai dấu, 4 điểm); `test_c3_18_h7_timer_rounding_edges` (đúng 952,5 → 953, láng giềng float32 dưới → 952, kẹp 65535 và 1); `test_c3_18_wrong_lock_stops_schedule[old_code_8188 / timer_cut_no_half / point_not_locked]` (ca sai phải dừng, không ghi file) | hỏng trên code cũ (không có `OFF_GRID_LOCK`, `h7_timer`), đạt trên code mới |
| C3-18 (đủ 25 lần ghé, tăng/giảm, hai dấu, W15) | như trên (lịch sinh từ `POINTS`) | `test_c3_18_full_25_visits_each_pass` (danh sách viết tay đủ 25 lần ghé mỗi lượt, đúng thứ tự, đúng `visit`, `run_pass`) | hỏng trên code cũ (vị trí 10: 9,995 ≠ 9,9942), đạt trên code mới |
| C3-18 (bộ chấm: điểm mới được chấm, điểm chỉ báo không chấm H5; lịch còn 9,995 bị từ chối) | không sửa `tools/s06_t5_score.py` (đọc lịch khóa qua `t5.make_schedule`) | `tools/tests/test_s06_t5_score.py::test_c3_18_point_9_9942_scored_indicator_not` (±9,9942 có 5×4 cửa sổ, `omega_d` = 8187·40/32767, `f_cmd` đúng, đạt; 50 count lạc ở lượt chỉ báo: ghi `indicator_counts`, không vào `points()`, `pass` = 460, mã 0); `test_c3_18_schedule_with_old_9995_not_locked` (lịch còn 9,995 → S18 từ chối; lệch 1e-12 vẫn nhận) | hỏng trên code cũ, đạt trên code mới |
| C3-18 (`ket_qua_e3/` theo điểm mới) | `ket_qua_e3/kiem_bon_diem.py:37-40` (`NEW_POINTS` dùng 9,9942; `SPEC_REPLACED = {9.995: 9.9942}`), `:163-176` (nhóm mới `spec_goc`: điểm 9,995 của spec, cột `de_xuat_*` ghi 9,9942 / 8187 / 0,5502), `:213-215` (ghi chú C3-18 vào `nguon_cong_thuc.txt`); chạy lại → `bang.csv` 26 dòng, `nguon_cong_thuc.txt` | `ket_qua_e3/test_kiem_bon_diem.py::test_table_26_rows_new_points_inside_frac_band` (đổi từ 24 dòng), `test_new_points_match_t5_schedule_lock` (mới: bảng khớp `OFF_GRID_LOCK`), `test_frame_roundtrip_all_26_omega_set` (đổi tên) | đạt |

Giá trị khóa (từ `python ket_qua_e3/kiem_bon_diem.py`, `bang.csv:18-27`): 9,9942 → int16 8187, ω_d H7 9,99420166015625, nửa chu kỳ float32 952,5501708984375 (phần lẻ 0,5502), timer 953; điểm gốc 9,995 → 8188, 9,99542236328125, 952,433837890625 (0,4338), timer 952. Ba điểm kia giữ nguyên: 19,979 → 16366 / 476,50787353515625 / 477; 31,68 → 25951 / 300,5097351074219 / 301; 38,78 → 31768 / 245,48374938964844 / 245.

Biểu thức H7 dùng (đọc nguồn trong worktree): giải mã `_rd16(&_rx[0]) * (HIL_OMEGA_MAX / INT16_MAXF)` (`stm32h7/Core/Src/spi1_handler.c:116`); nửa chu kỳ `ENC_TICKS_PER_S * (float)M_PI / (omega_abs * (float)ENC_PPR)`, kẹp, `(uint32_t)(hp + 0.5f)` (`encoder_gen.c:45-49`); `ENC_TICKS_PER_S = (float)ENC_TIMER_FREQ * ENC_CLOCK_CAL`, `ENC_CLOCK_CAL 1.0f`, `ENC_PPR 330u` (`encoder_gen.h:57,75-77`). `t5_schedule` mô phỏng float32 bằng `struct` (mỗi phép làm tròn một lần từ kết quả binary64; với `*`, `/` hai toán hạng float32 thì làm tròn hai lần vẫn đúng vì 53 ≥ 2·24+2), không thêm phụ thuộc numpy; test đối chứng bằng `numpy.float32`.

Test cũ phải sửa vì kiểm đúng hành vi đề đổi: `rpi5/tests/test_t5_schedule.py::test_v35_eleven_points_both_directions` (mong đợi `OFF_GRID_POINTS` có 9,9942 thay 9,995); `ket_qua_e3/test_kiem_bon_diem.py` (mong đợi 8187/953/0,5502 cho điểm mới, bảng 26 dòng).

**Cách kiểm "hỏng trên code cũ, đạt trên code mới":** viết test mới trước, chạy trên code chưa sửa (chưa đụng `t5_schedule.py`, `kiem_bon_diem.py`), rồi mới sửa code và chạy lại.
- Trước sửa, `python -m pytest -q tests/test_t5_schedule.py` (thư mục `rpi5`): `8 failed, 9 passed in 2.05s` (hỏng: `test_v35_eleven_points_both_directions`, `test_c3_18_point_9995_moved_to_code_8187`, `test_c3_18_lock_matches_frame_and_h7_both_signs`, `test_c3_18_h7_timer_rounding_edges`, 3 ca `test_c3_18_wrong_lock_stops_schedule`, `test_c3_18_full_25_visits_each_pass`; lý do đúng phản ví dụ: 9,995 ≠ 9,9942, thiếu `OFF_GRID_LOCK`/`h7_timer`).
- Trước sửa, `python -m pytest -q tools/tests/test_s06_t5_score.py -k c3_18`: `2 failed, 22 deselected in 1.94s` (lịch 9,995 được nhận là lịch khóa; không có lượt ghé ±9,9942).

## 2. Lệnh test đã chạy sau sửa (dòng tổng kết nguyên văn)

- `python -m pytest -q tests/test_t5_schedule.py` (từ `rpi5`): `17 passed in 1.50s`
- `python -m pytest -q tools/tests/test_s06_t5_score.py`: `24 passed in 360.19s (0:06:00)`
- `python -m pytest -q ket_qua_e3/test_kiem_bon_diem.py`: `7 passed in 0.79s`
- `python ket_qua_e3/kiem_bon_diem.py`: `bang.csv: 26 rows; new points outside [0.4, 0.6]: 0`

## 3. Điểm mơ hồ

- Spec (`Spec_S06:119`, câu 7) nêu "chỉnh ω đặt nếu cần để phần lẻ gần 0,5" nhưng không cho ngưỡng "gần". Theo quyết định của Claude, chỉ điểm 9,995 được dời. Với cùng tiêu chí "mã lân cận có phần lẻ gần 0,5 hơn" (tính bằng `kiem_bon_diem.h7_half_period_float`), hai điểm khác cũng có mã lân cận tốt hơn: 31,68 (mã 25951, 0,5097) → mã 25952 cho 0,4981; 38,78 (mã 31768, 0,4837) → mã 31767 cho 0,4915, mã 31766 cho 0,4992. 19,979 (16366, 0,5079) đã là mã tốt nhất trong các mã ±2. **Không đổi** hai điểm 31,68 và 38,78 vì ngoài phạm vi việc được giao; Claude quyết có dời hay không.
- Cũng không có quy tắc chọn **ω đặt** trong một mã. Tôi chọn 9,9942 (thập phân ngắn, gần tâm mã 8187, nằm trong mã cả khi lệch ±1e-9). Giá trị ở tâm mã là 9,99420148…
- Giữ giả định cũ của bảng: tick timer đúng 1 MHz danh định (`ENC_CLOCK_CAL = 1.0`); chưa kiểm trên board.

## 4. Phần chưa làm và lý do

- Không đổi 31,68 và 38,78 (xem mục 3): ngoài phạm vi việc được giao.
- Chưa kiểm trên board (đề không cho dùng board). `OFF_GRID_LOCK` là phép tính trên nguồn H7 đọc trong worktree; chưa đối chiếu với bản H7 đã nạp.
