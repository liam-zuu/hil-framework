# BAO_CAO: sửa sau phản biện Codex vòng 1, bộ C3A (C3-3, C3-4, C3-5)

Worktree `D:/HIL_WT/devin_c3a`. Chưa commit. Chỉ làm ba ID trạng thái ACCEPTED của bộ C3a trong `tham_dinh_vong1.md` (dòng 13–15); C3-1, C3-2 (DEFERRED) không đụng. Số dòng dưới đây là của file sau khi sửa.

File đã sửa (4): `fwsim/s03_doe.py`, `fwsim/tests/test_s03_doe.py`, `tools/chep_luot_A.py`, `tools/tests/test_chep_luot_A.py`. File tạo: `BAO_CAO.md`. Không đụng `tools/gain_table.py`, `tools/gain_send.py`, test của chúng, `fwsim/s04_ga.py`, `fwsim/rsm_compare.py`.

## 1. Bảng ID → chỗ sửa → test → kết quả

| ID | Sửa ở đâu (file:hàm:dòng) | Test | Kết quả |
|---|---|---|---|
| C3-3 | `tools/chep_luot_A.py`: hằng `TRAN_RTOL` dòng 84; `mien_rows` dòng 893–899 (ngoại lệ khi nút gốc cuối đã được thay bằng `T`), docstring 803–804 | `tools/tests/test_chep_luot_A.py`: `c33_run` 731; `test_c3_3_mien_accepts_the_original_grid_ending_at_the_ceiling` 780 (5 ca); `test_c3_3_mien_still_refuses_an_original_grid_not_ending_at_b` 801 (7 ca). Phía S03 (không sửa code): `fwsim/tests/test_s03_doe.py` `test_c3_3_b_within_1e9_below_the_ceiling_is_the_end_of_the_original_grid` 2709 | Đạt. Trên code cũ: 3 ca nhận hỏng (đúng phản ví dụ), 9 ca còn lại đạt |
| C3-4 | `fwsim/s03_doe.py`: hàm mới `default_node` 404–419; `sweep_grid_cut` 422–431 (gọi `default_node` thay cho `default not in grid`); `main` dòng 1381 (`d_node`), 1387–1393 (`is_default`, `luoi`, mốc `y3_ref`), 1402 (`levels_from_sweep`); chữ `meta.yaml` dòng 1773; docstring đầu file 82–83 | `fwsim/tests/test_s03_doe.py`: `test_c3_4_default_on_a_log_point_merged_into_the_ceiling` 2728; `test_c3_4_default_added_beside_the_log_points_merged_into_the_ceiling` 2756; `test_c3_4_default_node_boundaries` 2778; `test_c3_4_default_another_float32_than_the_ceiling_stops` 2799; `test_c3_4_main_runs_the_default_at_the_ceiling` 2832 (qua `main()`) | Đạt. Trên code cũ: cả 5 hỏng |
| C3-5 | `fwsim/s03_doe.py`: `cut_grid` dòng 394 (`b_n` của `ω_o` = điểm lớn nhất của lưới), docstring 389–390; chữ `meta.yaml` dòng 1742, 1773, 1783; docstring đầu file 80–81. `tools/chep_luot_A.py`: `mien_rows` dòng 885 (`nguon_hi` = `b`) | `fwsim/tests/test_s03_doe.py`: `test_c3_5_omega_o_b_n_is_the_largest_grid_point` 2819 (3 ca); test cũ đã sửa 2179, 2437–2442. `tools/tests/test_chep_luot_A.py`: `test_c3_5_omega_o_nguon_hi_is_b` 810 (4 ca); test cũ đã sửa `test_mien_domain` 487, `test_mien_omega_o_b_n_equal_to_b_is_accepted` 686 | Đạt. Trên code cũ: xem mục 2 |

## 2. Từng ID: phản ví dụ, cách sửa, cách kiểm "hỏng trên code cũ, đạt trên code mới"

**Cách kiểm chung.** Viết test trước, chạy trên nguồn chưa sửa, rồi mới sửa nguồn và chạy lại đúng lệnh đó (không dùng git). Thứ tự theo đồng hồ máy 22/11: hai file test ghi lần cuối 23:16:25 và 23:16:49; lượt "đỏ" bắt đầu 23:16:57 và 23:17:12 (trường `timestamp` của junit XML); `s03_doe.py`, `chep_luot_A.py` ghi lần cuối 23:17:50 và 23:17:58; lượt "xanh" 23:18:04 và 23:18:16 (lệnh `python -c` đọc `os.path.getmtime` và junit XML).

| Lệnh (thêm `--tb=line`) | Nguồn chưa sửa | Nguồn đã sửa |
|---|---|---|
| `python -m pytest -q tools/tests/test_chep_luot_A.py -k "c3_ or test_mien_domain or omega_o_b_n_equal"` | `7 failed, 11 passed, 111 deselected in 7.09s` | `18 passed, 111 deselected in 6.40s` |
| `python -m pytest -q fwsim/tests/test_s03_doe.py -k "c3_ or uncut_gains or v83_main_writes"` | `11 failed, 5 passed, 207 deselected in 14.24s` | `16 passed, 207 deselected in 18.21s` |

Test hỏng trên nguồn chưa sửa (tên lấy từ junit XML):

- chep (7): `test_c3_3_mien_accepts_…[kw0]`, `[kw1]`, `[kw2]`; `test_c3_5_omega_o_nguon_hi_is_b[48.0-]`, `[48.0-48.0]`; `test_mien_domain`; `test_mien_omega_o_b_n_equal_to_b_is_accepted`.
- s03 (11): năm test `test_c3_4_*`; ba ca `test_c3_5_omega_o_b_n_is_the_largest_grid_point`; `test_v86_uncut_gains_keep_the_old_grid_bit_for_bit[ADRC-omega_o_xy]`, `[ADRC-omega_o_wz]`; `test_v83_main_writes_levels_sources_luoi_and_grid_meta`.

Test đạt cả trên nguồn chưa sửa là các ca canh (giữ hành vi cũ), không phải ca tái hiện: `test_c3_3_mien_accepts_…[kw3]`, `[kw4]`; bảy ca `test_c3_3_mien_still_refuses_…`; `test_c3_5_omega_o_nguon_hi_is_b[40.0-]`, `[40.0-40.0]`; `test_c3_3_b_within_1e9_…` (phía S03 vốn đúng); bốn ca `test_v86_uncut_gains_…` không phải `ω_o`.

### C3-3 (`chep_luot_A.py` `mien_rows`)

- Phản ví dụ của Codex, dựng lại bằng chính `s03_doe` (lệnh `python -c` trên code cũ): PID `pos_kp`, mặc định 6, cận tạm 0 … 199,9999999, `T` = 200. `sweep_grid_cut` cho lưới 13 điểm, điểm cuối 200.0, `b` = 199.9999999, `b_n` = 200.0, `n_bo` = 4, `noi` = `[]`, lưới gốc là cả lưới (điểm nối đầu tiên 357,53 > `T` bị bỏ; nút `b` thành `T` và giữ phần lưới gốc). `mien_rows` cũ đòi `g_hi == min(b, b_n)` = 199.9999999 nên từ chối.
- Sửa: khi `g_hi != min(b, b_n)`, nhận nếu đủ ba điều: `g_hi == b_n` (không còn điểm nối, `T` là điểm lớn nhất); mọi dòng `sweep.csv` của điểm `g_hi` có `tran` = `true`; `math.isclose(b, g_hi, rel_tol=1e-9)` (cùng hàm, cùng dung sai với `s03_doe.cut_at_ceiling`). Không thì từ chối bằng đúng thông báo cũ.
- Ca nhận (5): `b` = 199.9999999 (phản ví dụ); thêm `H` = 200.0 với `H_nguon` = `trần bảng chặn`; `b` = 199.9999998 (= 200·(1 − 10⁻⁹), đúng biên dung sai, `math.isclose` bao hàm biên); `b` = 200.0 (bằng đúng `T`); `b` = 200.0000001 (ngay trên `T`: S03 bỏ `b` và thêm `T` vào lưới gốc). Hai ca cuối vốn đạt theo luật cũ.
- Ca từ chối (7): `tran` = `false` ở điểm cuối; `sweep.csv` không có cột `tran`; `tran` chỉ `true` ở một phần số dòng của điểm; `b` = 199.9999997 (ngoài dung sai); `b` = 199.999999 (float32 của nó là 200.0 nhưng binary64 cách 5·10⁻⁹: phép so là binary64); `b` = 100.0; còn điểm nối 300.0 phía trên (`T` không phải `b_n`).

### C3-4 (`s03_doe.py`)

- Phản ví dụ của Codex chạy trên code cũ (lệnh `python -c`): `sweep_grid_cut('PID', 'kp_xy', 41·(1 − 5·10⁻¹⁰), 0, 41)` dừng với `PID kp_xy: the default 40.9999999795 is not a point of the grid cut at the ceiling 41.0 (S03 v8.6 3.1)`.
- Sửa: `default_node(grid, info, default)` trả điểm lưới đại diện cho mặc định: chính mặc định khi nó là điểm lưới; `T` khi luật (2) đã thay mặc định (cách `T` trong 10⁻⁹, không lớn hơn `T`) bằng `T` **và** hai giá trị cùng float32. Mặc định lớn hơn `T` (bị bỏ) hay khác float32 với `T`: `S03Stop` như trước. `main` dùng điểm đại diện cho luật đặt mức (`levels_from_sweep`), `is_default`, nhãn `luoi` và mốc `y3_reference`; cột `default` của `levels.csv` vẫn là giá trị firmware; `info` của lưới không thêm khóa nào.
- Ca trùng nút log (test 2728): mặc định là điểm log thứ 6; lưới sau cắt là 0, năm điểm log dưới nó, 41.0; `default_node` = 41.0; mức `(0, 41)` nguồn `trần bảng chặn`; chỉ lượt ở `T` hoàn thành thì `L = H = T`; lượt ở `T` không hoàn thành thì dừng ca (i); nút `W` = năm điểm log + `T` (6 nút), trùng từng bit với kỳ vọng viết từ spec (`_w_nodes_want`) và với lưới quét trừ điểm 0.
- Ca mặc định thêm riêng (test 2756): PID `kp_wz`, `T` = 4,6, mặc định 4,6·(1 − 5·10⁻¹⁰), cận tạm 0 … 100 (nên `b` = 100, mặc định không trùng điểm log nào). Lưới sau cắt là 0, năm điểm log dưới `T`, 4.6; nút `W` = năm điểm log + `T` (6 nút), trùng từng bit với kỳ vọng. `w_nodes` không phải sửa.
- Ca biên (test 2778): mặc định bằng đúng `T`; mặc định = `T`·(1 − 10⁻⁹) (đúng biên, được thay); `T`·(1 − 2·10⁻⁹) (ngoài dung sai: mặc định là điểm riêng, `T` thêm sau nó); mặc định 1,5 (xa `T`); `T`·(1 + 5·10⁻¹⁰) (lớn hơn `T`: bị bỏ, dừng).
- Float32 làm tròn qua ngưỡng (test 2799): `T` = (4 + 2⁻²²)·(1 + 2·10⁻¹⁰), mặc định = `T`·(1 − 5·10⁻¹⁰); float32 của `T` là 4 + 2⁻²¹, của mặc định là 4.0; hai giá trị cách nhau dưới 10⁻⁹ nhưng khác float32: `default_node` và `sweep_grid_cut` dừng với thông báo có chữ `float32`.
- Qua `main()` (test 2832, cần DLL): mặc định giả PID `kp_xy` = 41·(1 − 5·10⁻¹⁰); mã thoát 0, không có `[STOP]`; ba dòng `sweep.csv` ở `value` = 41.0 có `is_default` = `true`, `luoi` = `mac_dinh`, `tran` = `true`; các điểm khác nhỏ hơn mặc định, `is_default` = `false`; `levels.csv`: `L` 0.0, `H` 41.0, `H_nguon` `trần bảng chặn`, `b_n` 41.0, `default` = `repr` của mặc định firmware; thiết kế chính chạy.

### C3-5 (`s03_doe.py` `cut_grid`, `chep_luot_A.py` `mien_rows`)

- Code cũ (lệnh `python -c`): `sweep_grid_cut('ADRC', 'omega_o_xy', 30, 5, 48)` cho `b_n` = `None`, `b` = 48.0.
- Sửa S03: nhánh không cắt của `cut_grid` đặt `b_n = grid[-1]` khi `sweep_grid_n` cho `None` (chỉ `ω_o`). `sweep_grid_n` giữ nguyên (vẫn `None`; `o1_feas_search` và test v8.3 dùng nó). `levels.csv` và khối `grid` của `meta.yaml` nay ghi 48.0 cho `ω_o` (kiểm ở test `main` 2179).
- Sửa miền A: `nguon_hi` của `ω_o` luôn là `b` (bỏ `tran_omega_o`). Ô `b_n` rỗng của dữ liệu cũ vẫn đọc được; `b_n` khác `b` vẫn bị từ chối (ca có sẵn, `test_mien_refuses` thông báo `is not b`).
- Test: `ω_o` với `b` = 48 và `b_n` rỗng / `48.0`; `b` = 40 và `b_n` rỗng / `40.0` (đều `nguon_hi` = `b`); phía S03 thêm lưới `ω_o` có `b` = 30 < 48 (`b_n` = 30.0).

## 3. Lệnh test đã chạy và dòng tổng kết nguyên văn (sau mọi sửa)

DLL: dùng `fwsim/build/fwsim.dll` có sẵn, không build lại. Không có test nào bị bỏ qua (junit XML: `skipped` = 0 ở cả bốn file).

| Lệnh | Dòng tổng kết |
|---|---|
| `python -m pytest -q fwsim/tests/test_s03_doe.py` | `223 passed in 288.25s (0:04:48)` |
| `python -m pytest -q tools/tests/test_chep_luot_A.py` | `129 passed in 62.98s (0:01:02)` |
| `python -m pytest -q tools/tests/test_gain_table.py` | `54 passed in 1.43s` |
| `python -m pytest -q tools/tests/test_gain_send.py` | `31 passed in 9.50s` |

So với báo cáo việc trước (`docs/devin/2026-11-22_BAO_CAO_C3a.md` dòng 106–109: 214, 113, 54, 31): `test_s03_doe.py` thêm 9 (7 hàm test mới, trong đó một hàm có 3 ca: 6 + 3), `test_chep_luot_A.py` thêm 16 (3 hàm test mới: 5 + 7 + 4 ca).

### Test cũ đã sửa và lý do (đều do C3-5 đổi đúng hành vi mà test đó kiểm)

- `test_s03_doe.py` `test_v86_uncut_gains_keep_the_old_grid_bit_for_bit` (2437–2442): với `ω_o`, `b_n` nay là 48.0 thay cho `None`; các khóa khác của `info` vẫn so đủ với `sweep_grid_n`.
- `test_s03_doe.py` `test_v83_main_writes_levels_sources_luoi_and_grid_meta` (2179): `grid_meta['ADRC omega_o_xy']['b_n']` nay 48.0, thêm kiểm ô `b_n` của `levels.csv` là `48.0`.
- `test_chep_luot_A.py` `test_mien_domain` (496–497), `test_mien_omega_o_b_n_equal_to_b_is_accepted` (695): `nguon_hi` của `ω_o` nay là `b`.
- `test_chep_luot_A.py` `s03_run` (430, 461): thêm tham số `scols` để ghi `sweep.csv` có cột `tran`; mặc định như cũ.

## 4. Điểm mơ hồ

1. **C3-4, phạm vi "đại diện".** Codex chỉ ghi "đại diện của lượt mặc định trong luật đặt mức". Tôi dùng cùng điểm đại diện cho `is_default`, `luoi` (= `mac_dinh`, theo quy ước v8.3 "mặc định ưu tiên") và mốc `y3_reference`, vì lượt ở `T` gửi cùng float32 nên chính là lượt gain mặc định. Hệ quả: ở ca này cột `default` của `levels.csv` (giá trị firmware binary64) không trùng giá trị nào ở cột `value` của `sweep.csv`. Spec §3.1 (dòng 96) không nói về ca này.
2. **C3-4, điều kiện cùng float32.** Mặc định cách `T` dưới 10⁻⁹ nhưng khác float32 với `T` thì S03 vẫn dừng. Với bảng hiện tại nhánh này không tới được: cả 14 trần của `tools/gain_table.py` đều có float32 của `T`·(1 − 10⁻⁹) bằng float32 của `T` (lệnh `python -c` trên `gain_table.HI`).
3. **C3-4, ca kề không sửa.** Mặc định lớn hơn `T` trong 10⁻⁹ (cùng float32 với `T`, nên hợp lệ theo bảng float32) vẫn bị luật (1) bỏ và S03 dừng, như trước. Codex chỉ nêu ca "phép thay"; tôi không mở rộng. `test_c3_4_default_node_boundaries` ghi lại hành vi này.
4. **C3-4, nút `W` ở ca mặc định thêm riêng.** Spec §3.1b điều 2 (dòng 116) vừa cho cách dựng (11 điểm log + điểm nối, rồi luật (1), (2): `T` được thêm) vừa ghi "trừ điểm 0 và giá trị mặc định thêm vào". Ở ca này điểm `T` của lưới quét chính là điểm mặc định thêm riêng đã thành `T`. `w_nodes` (không sửa) theo cách dựng: `T` là nút cuối, `w` = `b_n` = `T`.
5. **C3-5, giá trị `b_n` của `ω_o`.** Tôi ghi theo định nghĩa spec "điểm lớn nhất của lưới sau khi cắt" (`grid[-1]`), bằng 48 với mặc định chiến dịch. Lưới `ω_o` có `b` < 48 (ví dụ mặc định 3, cận tạm 20) cho `b_n` = `b` = 30, không phải 48.
6. **C3-5, `mien_A.csv` cũ.** `them --loai mien_A` và `kiem-c13` so byte với miền tính lại (`chep_luot_A.py` dòng 662–663). Một `mien_A.csv` tạo trước lần sửa này có `tran_omega_o` sẽ lệch và phải chạy lại `mien`. Có file như vậy trong `05_DATA` hay không: chưa kiểm.
7. **C3-3, cột `tran`.** Ngoại lệ chỉ áp khi `sweep.csv` có cột `tran`; file trước v8.6 (không có cột) bị từ chối như cũ ở ca này. `mien_rows` không kiểm tính nhất quán chung của cột `tran` (đúng một điểm `true`, là điểm lớn nhất): ngoài ID.

## 5. Phần chưa làm và lý do

- **`fwsim/s04_ga.py` `read_domain` (dòng 210) vẫn từ chối `H > b`.** Ở ca C3-3, khi điểm `T` hoàn thành thì `H` = `T` > `b` và S04 sẽ từ chối `levels.csv` đó. Bảng thẩm định chỉ giao `chep_luot_A` cho C3-3 và việc trước cấm sửa `s04_ga.py`, nên không sửa; cần xử ở bộ C3b hay đợt 2.
- **Không chạy `test_s04_ga.py`, `test_rsm_compare.py`, `test_o1_feas_search.py`** (ngoài danh sách bắt buộc; chưa kiểm). Theo báo cáo việc trước (dòng 119–125), 18 test của chúng so `fwsim/s03_doe.py` với HEAD qua git nên hỏng chừng nào file chưa commit. Ba file test này chỉ gọi `sweep_grid_n` của `s03_doe` trong số các hàm lưới (grep), hàm đó không đổi.
- Không chạy lượt S03 thật, không chạy `kiem-c13`.
- C3-1, C3-2: DEFERRED theo bảng thẩm định, không làm.

## 6. Ngoài 4 file

- Vì không dùng git, bản gốc của 4 file, các file diff và junit XML của các lượt test được ghi tạm ở `%TEMP%/devin_c3a_r2/`; thư mục này đã xóa khi xong. Lượt "đỏ" tái hiện được bằng cách chạy hai lệnh ở mục 2 với hai file test mới trên `s03_doe.py`, `chep_luot_A.py` của HEAD.
- `.pytest_cache/` ở gốc worktree do pytest tạo (có `.gitignore` riêng), `fwsim/build/` giữ nguyên.
