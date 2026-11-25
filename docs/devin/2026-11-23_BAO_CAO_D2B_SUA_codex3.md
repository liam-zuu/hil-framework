# BAO_CAO: việc G-D2b, sửa hẹp sau Codex vòng 3 (C3-8/C3-47, C3-41…C3-46, C3-48 phía S04)

Worktree `D:/HIL_WT/devin3_d2b`, nhánh `fix/devin-d2b-codex3` (nền `23e5847`), 23/11/2026. Không commit, không chạy `git`.

**File sửa (8):** `fwsim/s04_ga.py`, `fwsim/rsm_compare.py`, `fwsim/o1_feas_search.py`, `tools/gain_send.py`, `fwsim/tests/test_s04_ga.py`, `fwsim/tests/test_rsm_compare.py`, `fwsim/tests/test_o1_feas_search.py`, `tools/tests/test_gain_send.py`. **File tạo (2):** báo cáo này và `BAO_CAO.md` ở gốc (cùng nội dung). Căn cứ "chỉ 8 file đó đổi": một lệnh `python` liệt kê file có giờ sửa sau lúc lưu bản cũ (bỏ `.git`, `__pycache__`, `.pytest_cache`) cho đúng 8 đường dẫn trên; `fwsim/s03_doe.py`, `fwsim/s03_sangloc.py`, `tools/mo_hinh_con.py`, `tools/gain_table.py`, `fwsim/build/fwsim.dll` có giờ sửa trước mốc đó.

**Kết luận.** Chín ID của đề đã sửa và có test đi qua entry point thật. Bốn file test bắt buộc đạt trên code mới (mục 2). 150 test mới `test_c3_*`: trên code mới 149 đạt, 1 bỏ qua (máy không cho tạo symlink); trên code cũ 131 hỏng, 18 đạt, 1 bỏ qua (18 test đạt là ca "phải được nhận", ca code cũ vốn đã từ chối, và 3 test C3-48 không kèm sửa code; liệt kê ở mục 2). Hai chỗ gọi `gain_send.py` ngoài danh sách file được sửa nay bị từ chối cho tới khi thêm cờ nguồn gain (mục 4). Có 12 điểm mơ hồ hay tự chọn cần Claude xem (mục 3).

Số dòng dưới đây là của file sau khi sửa (lấy bằng công cụ tìm trên file hiện tại).

## 1. Mỗi ID → chỗ sửa → test → kết quả

Cột "mới" và "cũ" là số test `test_c3_<id>_*` đạt/hỏng trên code sau sửa và trên code trước sửa (cùng file test mới; bản code cũ lưu ngoài worktree, chép đè tạm 4 file nguồn rồi trả lại, SHA-256 bốn file sau khi trả trùng bản mới).

| ID | Chỗ sửa (file: hàm: dòng) | Test (file: tên: dòng) | Mới | Cũ |
|---|---|---|---|---|
| C3-41 | `s04_ga.py`: `check_s03_ket_qua`: 1874; `check_s03_manifest`: 1897; gọi ở `main`: 1671–1672 (ngay sau `check_s03_sealed`, trước `check_s03_source`, trước mọi khóa); hằng `S03_MANIFEST`, `S03_KET_QUA`, `S03_DONE`: 179–181 | `test_s04_ga.py`: `test_c3_41_untouched_fixture_accepted` 2702, `test_c3_41_ket_qua_gate` 2710 (`{}`, `dang_chay`, không phải object, không phải JSON, thiếu file), `test_c3_41_every_controller_screened` 2732, `test_c3_41_manifest_catches_a_tampered_sealed_file` 2757 (bốn file), `..._tampered_runs_csv` 2767, `test_c3_41_manifest_gate` 2777, `..._listing_a_deleted_file` 2789, `..._trial_s03_without_so_dem_accepted` 2798. Mỗi ca chạy ở bốn đường: `--phase trial --tool-check`, `--phase trial` chính, `--phase main --tool-check`, `--phase main` chính (lock viết tay sau khi sửa file, để phép so fingerprint không là thứ từ chối) | 63 đạt | 52 hỏng, 11 đạt |
| C3-42 | `s04_ga.py`: `_domain_rows`: 333 (đúng một hàng mỗi gain); `_level`: 402; `_sweep_levels`: 411; `read_domain`: 430 (hữu hạn, thang, `[L, H]` trong `W`, `ω_o` ≤ 48, kết quả kiểm ngoài miền chỉ gắn khi cùng điểm) | `test_c3_42_codex_domain_outside_the_reference_box` 2832 (phản ví dụ Codex qua `read_domain` và qua `s4.main`), `test_c3_42_reference_box_edges` 2847, `..._non_finite_level_refused` 2877, `..._wrong_scale_refused` 2887, `..._exactly_one_row_per_gain` 2896, `..._omega_o_cap` 2930, `..._out_of_range_result_only_at_the_same_point` 2950 | 32 đạt | 29 hỏng, 3 đạt |
| C3-43 | `o1_feas_search.py`: `_s03_meta_line`: 859; `s03_unsealed_gate`: 872; `cmd_inbox`: 900 (cổng là câu lệnh đầu: 904; SHA vào manifest niêm phong: 922; `meta.yaml` công khai bỏ `s03_levels_sha256`, thêm `s03_unsealed: true`: 928); `sealed_manifest(sealed, extra=None)`: 491; cờ `--s03-unsealed`: 959 | `test_o1_feas_search.py`: `test_c3_43_inbox_needs_the_unsealed_flag` 636, `..._needs_the_finished_main_s03` 654, `..._needs_the_public_meta` 667; `test_inbox` 561 và `test_inbox_missing_sealed_levels` 592 sửa theo | 5 đạt | 5 hỏng |
| C3-44 | `s04_ga.py`: `s03_is_trial`: 201; `_s03_entry`: 215; `s03_path`: 232; `s03_read`: 238 (kiểm `os.fstat` trên descriptor đã mở); `s03_text`: 253; `s03_csv`: 259; mọi chỗ đọc file S03 đi qua đó: `_domain_rows` 333, `_du_phong_tran_exception` 374, `_sweep_levels` 411, `kiem_file` 483, `read_out_of_range` 491, `mu_rule` 601, `main` 1635, `fingerprint` 1855, `check_s03_ket_qua` 1874, `check_s03_manifest` 1897, `so_goc` 1923, `check_s03_trial` 1972. `rsm_compare.py`: `S03_GUARDED`: 99; `Inputs.check_links`: 565 (dòng 569); `check_mode`: 582 (dòng 588); `INTERPRETATIONS`: 196 | `test_s04_ga.py`: `test_c3_44_trial_must_be_a_bool` 2985 (`1`, `"true"`), `..._main_mode_trial_must_be_a_bool` 2997, `..._post_and_sens_after_trial_must_be_a_bool` 3005, `..._hard_link_alias_refused` 3023 (năm file), `..._hard_link_on_runs_csv_refused` 3040, `..._junction_on_sangloc_thu_refused` 3055, `..._symlink_on_sangloc_thu_refused` 3075 (bỏ qua), `..._opened_file_identity_rechecked` 3094. `test_rsm_compare.py`: `test_c3_44_rsm_refuses_a_hard_link_before_the_start_row` 1930 (`ket_qua.json`, `so_dem.json`), `..._refuses_a_junction_directory` 1947, `..._main_mode_refuses_a_non_bool_trial` 1966 | 12 đạt, 1 bỏ qua; RSM 4 đạt | 12 hỏng, 1 bỏ qua; RSM 4 hỏng |
| C3-45 | `s04_ga.py`: `POST_SOURCES` 2022, `PHASE_SOURCES` 2025, `LOCKED_SOURCES` 2027; `fingerprint`: 1855 (`scripts` băm `LOCKED_SOURCES`); `source_diff`: 2038; `post`: 1342 và `sens_after`: 1520 (gọi lại `check_s03_source`); `write_meta`: 2179 (thuyết minh: bỏ chữ `screen_u`, thêm dòng cho từng ID) | `test_c3_45_source_lock_covers_every_phase_source` 3115, `..._post_compares_the_changed_source` 3127, `..._tool_check_lock_only_warns` 3138, `..._post_compares_the_screening_script` 3146, `..._sens_after_compares_the_sources` 3154, `..._lock_fingerprint_difference_stops_post` 3166; thêm hai phép kiểm trong `test_tool_check_three_phases` 1070 (DLL thật: `scripts` của lock bằng `LOCKED_SOURCES`; `meta_PID.yaml` không còn `screen_u`) | 6 đạt | 5 hỏng, 1 đạt |
| C3-46 | `s04_ga.py`: `determinism_rows`: 1268; `post`: 1377 (mục `cho_liam` `determinism`, `notes['<ctrl> determinism']`); `CHO_LIAM_NOTE`: 1254 | `test_c3_46_determinism_all_equal` 3180, `..._mismatch_waits_for_liam` 3217 (thiếu một kết quả, thừa một kết quả, digest `None`, digest khác, thiếu hàng GA, hash GA rỗng, hash GA sai dạng, hàng GA thừa khóa), `..._determinism_rows_unit` 3234; `_fake_post` 1848 sửa (digest hợp lệ, chạy qua `s4.main --phase post`) | 10 đạt | 10 hỏng |
| C3-8, C3-47 | `s04_ga.py` `post`: 1326 (băm `selected.csv` một lần đọc), `write_notes` 1434, 1449, 1455 (`post_trang_thai`, `selected_sha256`, `selected_sha256_lf`); `POST_DONE`, `POST_RUNNING`: 1263–1264. `gain_send.py`: `s04_hand_over`: 310 (phần tử `cho_liam` hỏng: 342; `tool_check`: 347; `post_trang_thai`: 350; đối chiếu SHA; dòng `#ban_giao_s04` thêm `selected_sha256`: 403); `main`: 434 (cờ `--nguon-gain-tay`: 448; không cờ nào → mã 5: 492; bọc lỗi của `s04_hand_over` → mã 5: 538; dòng log `#nguon_gain,loai=tay`: 549) | `test_s04_ga.py` (từ `post` tới `gain_send`): `test_c3_47_post_notes_commit_selected_and_send` 3272, `..._send_refused_when_selected_changed_after_post` 3287, `..._send_refused_for_broken_cho_liam` 3311 (`[null]`, `[{}]`, thiếu `reasons`, `reasons` không phải list), `..._send_refused_when_post_not_done` 3328, `..._tool_check_post_never_handed_over` 3344, `..._determinism_difference_gates_the_send` 3353, `..._gain_send_needs_a_declared_source` 3364. `test_gain_send.py` (qua `gs.main` + FakeEsp32): `test_c3_8_gain_send_needs_a_declared_source` 554, `test_c3_47_s04_run_refusals_at_unit_level` 568, `test_c3_47_hand_over_exception_is_code_5` 609 | 11 + 3 đạt | 11 + 3 hỏng |
| C3-48 (phía S04) | Không sửa code: quy tắc dừng khi DLL thiếu kênh kẹp ở `main` giữ nguyên | `test_c3_48_kep_values_in_the_ga_evals` 3378 (DLL thật, `kep_cols` phải khác 0; PID ba `kep_frac_*` hữu hạn trong [0, 1], `kep_note` rỗng; LADRC trống; đọc lại qua CSV), `test_c3_48_hash_compat_with_and_without_kep` 3409 (chỉ phép băm, một lượt), `test_c3_48_same_dll_twice_bit_exact_including_kep` 3422 (chỉ tất định: hai lượt cùng tiến trình và một lượt tiến trình mới, so cả `log_digest`, gồm mục `kep`); siết nhánh PID của vòng kiểm kẹp trong `test_tool_check_three_phases` | 3 đạt | 3 đạt (test thêm, không kèm sửa code) |

### Tóm tắt hành vi mới (để đối chiếu)

- **C3-41.** Ở mọi lần khởi động thử và chính: `sealed/sangloc/ket_qua.json` phải có, là object JSON, `trang_thai` = `hoan_tat` (run chính: cả hai controller; `--tool-check`: controller đang chạy). Mọi file niêm phong mà fingerprint băm (bốn file của `S03_SEALED_LOCK`; `sealed/runs.csv` khi đã mở niêm phong) phải trùng SHA-256 ghi ở `sealed/manifest.json`; file vắng thì cũng phải vắng trong manifest.
- **C3-42.** Mỗi gain đúng một hàng; `L`, `H` hữu hạn; `scale` bằng `s03_doe.scale_of(L, H)`; với `sang_loc`: `a`, `b_n` hữu hạn, `0 < a ≤ L`, `H ≤ b_n`, `ω_o`: `H ≤ 48`; so đúng binary64, hai đầu tính là trong, không dung sai. `L_kiem`, `H_kiem` của miền `sang_loc` chỉ điền khi cận niêm phong bằng đúng mức quét cùng phía ở `levels.csv` công khai.
- **C3-43.** `inbox` cần `--s03-unsealed`, `meta.yaml` công khai của S03 có `status` = `ran to the end` và `trial` là đúng `false` (hay không có dòng); kiểm trước mọi thao tác dưới `sealed/` và trước khi tạo thư mục ra. SHA-256 mức niêm phong nằm ở `sealed/manifest.json` của inbox (khóa `s03_levels`).
- **C3-44.** `trial` chỉ nhận đúng bool; `1`, `"true"`… dừng ở mọi chế độ. Mọi file của run S03 mà `s04_ga.py` mở hay băm đi qua `s03_read`: từ chối trước khi mở nếu là link, junction, đường phân giải sang chỗ khác, không phải file thường hay có hơn một tên; sau khi mở kiểm lại bằng `os.fstat` (một tên, cùng file đã kiểm). RSM kiểm cả danh sách file mà `s4.fingerprint` băm trước dòng bắt đầu.
- **C3-45.** `fingerprint()['scripts']` băm hợp các nguồn của mọi pha (thêm `fwsim/s03_sangloc.py`, `tools/mo_hinh_con.py`); `post`, `sens_after` chạy lại phép so blob với commit S03.
- **C3-46.** Hậu kiểm tất định so đủ từng khóa; thiếu, thừa, trùng, hash không phải 64 chữ số hex, hay hai hash khác nhau → mục `cho_liam` `determinism`, `post` trả 2, `selected.csv` không đổi.
- **C3-8, C3-47.** `post` ghi `post_trang_thai` (`dang_chay` trong lúc chạy, `hoan_tat` ở lần ghi cuối) và SHA-256 của `selected.csv`; `gain_send --s04-run` từ chối (mã 5, dòng cuối `#ket_qua,ma=5`) khi run là `tool_check`, `post` chưa hoàn tất, `selected.csv` khác bản đã băm, hay `cho_liam` hỏng. Mỗi lần gửi phải có `--s04-run` hoặc `--nguon-gain-tay "<mô tả>"` (in dòng `#nguon_gain,loai=tay,mo_ta=…`); không cờ nào → mã 5; cả hai cờ hay mô tả rỗng → mã 1.

## 2. Lệnh test đã chạy và dòng tổng kết nguyên văn

Chạy ở gốc worktree, mỗi lần một lệnh, biến môi trường `PYTHONIOENCODING=utf-8`. DLL `fwsim/build/fwsim.dll` có sẵn (`fwsim_build.json`: esp32 commit `5bde4c5`, `dirty: False`; `kep_cols` = 7), không build lại. Không chạy GA thật (chỉ các lượt `--tool-check` cỡ nhỏ có sẵn của bộ test).

**Code sau sửa, cả file:**

| Lệnh | Dòng tổng kết |
|---|---|
| `python -m pytest -q -rs fwsim/tests/test_s04_ga.py` | `367 passed, 1 skipped in 1119.65s (0:18:39)` |
| `python -m pytest -q -rs fwsim/tests/test_rsm_compare.py` | `123 passed, 2 skipped in 1127.37s (0:18:47)` |
| `python -m pytest -q fwsim/tests/test_o1_feas_search.py` | `62 passed in 106.64s (0:01:46)` |
| `python -m pytest -q tools/tests/test_gain_send.py` | `42 passed in 5.50s` |

- Test bỏ qua ở `test_s04_ga.py`: `test_c3_44_symlink_on_sangloc_thu_refused` (dòng 3087, "os.symlink not permitted: [WinError 1314]"; lệnh `python -m pytest -q -rs fwsim/tests/test_s04_ga.py -k "c3_44 and (symlink or junction or hard_link or opened)"` → `8 passed, 1 skipped, 359 deselected in 1.66s`).
- Hai test bỏ qua ở `test_rsm_compare.py` là test có sẵn cần quyền tạo symlink (lệnh `python -m pytest -q -rs fwsim/tests/test_rsm_compare.py -k "links_are_refused or nested_directory_link or c3_"` → `7 passed, 2 skipped, 116 deselected in 9.60s`; dòng 1454 "no symlink permission on this machine", dòng 1808 "no symlink on this machine").
- Không test nào bỏ qua vì thiếu DLL.
- **Biến thể symlink chưa kiểm trên máy này** (không có quyền tạo symlink): nhánh từ chối link của `s03_path` mới được chạy qua test hard link và junction.
- Số test `test_c3_*` theo file (lệnh `pytest --collect-only -q -k c3_` qua `python -c`): `test_s04_ga.py` 138 (C3-41: 63, C3-42: 32, C3-44: 13, C3-45: 6, C3-46: 10, C3-8/47: 11, C3-48: 3); `test_rsm_compare.py` 4; `test_o1_feas_search.py` 5; `test_gain_send.py` 3. Tổng 150.

**Code trước sửa (4 file nguồn cũ, 4 file test mới), chỉ test mới:**

| Lệnh | Dòng tổng kết |
|---|---|
| `python -m pytest -q -rp fwsim/tests/test_s04_ga.py -k c3_` | `119 failed, 18 passed, 1 skipped, 230 deselected in 194.60s (0:03:14)` |
| `python -m pytest -q fwsim/tests/test_rsm_compare.py -k c3_` | `4 failed, 121 deselected in 10.21s` |
| `python -m pytest -q -rp fwsim/tests/test_o1_feas_search.py -k c3_` | `5 failed, 57 deselected in 4.92s` |
| `python -m pytest -q tools/tests/test_gain_send.py -k c3_` | `3 failed, 39 deselected in 1.15s` |

18 test đạt trên code cũ (theo `-rp`), đều không phải phản ví dụ:

- Ca phải được nhận ở cả hai bản: `test_c3_41_untouched_fixture_accepted` (4), `test_c3_41_every_controller_screened[tc_trial]`, `[tc_main]`, `test_c3_41_trial_s03_without_so_dem_accepted`, `test_c3_42_reference_box_edges[H_eq_bn-False]`, `[L_eq_a-False]`, `test_c3_42_omega_o_cap[at_cap-False]`.
- Ca code cũ vốn đã từ chối: `test_c3_41_ket_qua_gate[missing-mm_trial]`, `[missing-mm_main]` và `test_c3_41_manifest_listing_a_deleted_file[mm_trial]`, `[mm_main]` (run chính thiếu file niêm phong đã bị `check_s03_sealed` chặn); `test_c3_45_lock_fingerprint_difference_stops_post` (lock có `scripts` khác đã bị `check_fingerprint` chặn).
- Ba test C3-48 (test thêm theo đề, không kèm sửa code).

Sau lần chạy trên code cũ, bốn file nguồn mới được chép lại; một lệnh `python` in `True` cho cả bốn phép so SHA-256 với bản mới đã lưu. Bản code cũ và bản lưu tạm nằm ở thư mục tạm của hệ điều hành (`…/Temp/d2b_codex3/`), đã xóa khi kết thúc.

### Test cũ đã sửa và lý do

Lập từ phép so từng dòng (`difflib`) giữa bản cũ đã lưu và file hiện tại.

- `test_s04_ga.py`:
  - `_s03_dir`: file mức niêm phong thêm cột `a`, điền `a = L` (khi `L > 0`), `b = H`, `b_n = H`, `scale` lấy từ `s3.scale_of` (C3-42 nay kiểm); `ket_qua.json` mặc định có `trang_thai: hoan_tat` cho cả hai controller thay cho `{}`; ghi `sealed/manifest.json` (C3-41); thêm tham số `dll`, `ket_qua`, `manifest`; thêm `_reseal`.
  - `_fake_post`: dùng run S03 đủ của `_s03_dir` và fingerprint thật (không còn thay `s4.fingerprint`), digest hợp lệ và `log_sha256` khớp (C3-46: digest `None` nay là lý do chờ Liam), chạy qua `s4.main()` với `--phase post`; thêm `det_mod`, `evals_mod`.
  - `test_tool_check_three_phases`: thêm phép kiểm C3-45 và C3-48; danh sách `check` hợp lệ của `cho_liam` thêm `determinism`.
  - `test_sens_after_pre_checks_every_variant_before_any_simulation`: thay thêm `s4.check_s03_source` (test này thay `s4.fingerprint` bằng `{}`, mà `sens_after` nay gọi lại phép so blob).
- `test_rsm_compare.py`: `make_s03` như `_s03_dir` (cột `a`, `ket_qua.json` hợp lệ, `sealed/manifest.json`).
- `test_o1_feas_search.py`: `test_inbox`, `test_inbox_missing_sealed_levels`: thêm `meta.yaml` công khai của S03 và cờ `--s03-unsealed`; `test_inbox` đổi phép kiểm SHA (trước kiểm SHA có trong `meta.yaml` công khai, đúng hành vi C3-43 bỏ) sang kiểm SHA nằm trong `sealed/manifest.json` và không có ở `meta.yaml`, `console.txt`.
- `test_gain_send.py`: `args` tự thêm `--nguon-gain-tay 'test bench'` khi lệnh không có `--s04-run`, `--nguon-gain-tay`, `--query`; `_s04_run` ghi hồ sơ `post` hoàn tất (`tool_check` mặc định `False`, `post_trang_thai`, hai SHA); `test_s04_hand_over_cho_liam_blocks_and_overrides`: `tool_check=true` → `tool_check=false` trên dòng `#ban_giao_s04` (run `tool_check` nay bị từ chối).

## 3. Điểm mơ hồ và chỗ tự chọn

1. **C3-41, "controller dùng" hay "cả hai".** Đề ghi "cho controller dùng"; S03 v8.6 §3.1b điều 11 (spec dòng 144) ghi "ở cả hai controller". Code: run chính đòi cả hai, `--tool-check` chỉ đòi controller đang chạy (run S03 thử có thể chỉ có một controller).
2. **C3-41, manifest.** Manifest không vào fingerprint; `post`, `sens_after` không kiểm lại manifest (dựa vào fingerprint đã khóa). Không file công khai nào ghi SHA của chính `sealed/manifest.json`, nên manifest là gốc tin cậy. Phép đối chiếu áp cho mọi mục `sealed/…` của fingerprint, tức thêm `sealed/runs.csv` khi đã mở niêm phong (đề nêu bốn file). Chưa đối chiếu `nguon_mien` trong `ket_qua.json` với `levels_sangloc.csv`.
3. **C3-42, "thang hợp lệ".** Hiểu là `scale` phải bằng kết quả `s03_doe.scale_of(L, H)` (S03 §3.1 "Thang", dòng 108), áp cho cả `sang_loc` và `du_phong`. Phép "đúng một hàng mỗi gain" và "mức hữu hạn" cũng áp cho cả hai nguồn (đề chỉ nêu `sang_loc`); phép `W` chỉ áp cho `sang_loc`. Thêm điều kiện `a > 0` (S03 §3.1b điều 2: `W` không chứa 0). Không kiểm cột `b` của hàng `sang_loc`, không kiểm `L`, `H` là nút của `W`.
4. **C3-42, "cùng điểm".** Mức quét lấy từ `levels.csv` công khai (file kiểm ngoài miền không đủ khi một phía không có điểm). Không cùng điểm thì `L_kiem`/`H_kiem` để rỗng, cùng giá trị với trường hợp run S03 không có file kiểm. Phần nhãn trần `H = b_n = T` ở phía ghi thuộc `s03_sangloc.py` (chỉ đọc).
5. **C3-43, "mốc mở".** Spec không có file hay trường nào làm mốc mở niêm phong; S04 chỉ có cờ `--s03-unsealed`. Code đòi cờ đó, cộng `status` = `ran to the end` và `trial` = `false` ở `meta.yaml` công khai (như S04 đòi với run chính). Hai điều kiện sau là phần tự thêm; bỏ được nếu không đúng ý. `cmd_inbox` chưa kiểm link/hardlink cho file mức niêm phong (đề không nêu).
6. **C3-44, phạm vi kiểm.** Phép kiểm áp cho mọi file S03 mà `s04_ga.py` mở, kể cả file công khai (`meta.yaml`, `levels.csv`, `sweep.csv`). Phép so đường dùng đường tuyệt đối (như `kiem_file` cũ): thư mục S03 đi qua link, junction hay tên ngắn 8.3 bị từ chối. Trong `rsm_compare.py`, các lần đọc `sealed/runs.csv`, `design.csv`, `notes.json` của chính RSM vẫn là kiểm đường trước khi mở (`check_links`), chưa kiểm `os.fstat`; cờ `rsm_fake` vẫn nhận `1`/`"true"`.
7. **C3-44, khoảng hở thời gian.** `read_domain` chạy trước `fingerprint` trong `main` (thứ tự cũ, có test dựa vào); miền được đọc và băm ở hai lần đọc khác nhau.
8. **C3-45.** Phép so blob dùng hợp nguồn của mọi pha ở mọi pha, nên `tools/mo_hinh_con.py` lệch commit S03 thì run chính dừng ngay ở lượt thử (không đợi tới `post`). Thư mục S04 có `lock.yaml` tạo bằng code trước sửa không chạy tiếp được (`scripts` khác): cần thư mục mới.
9. **C3-46.** Hash so vẫn là hash trên `fwsim.BIT_EXACT_KEYS` (kênh `kep` không nằm trong `log_sha256` của `evals.csv`), nên hậu kiểm tất định của `post` không so kênh kẹp; test C3-48 so kênh kẹp trên cùng DLL. Thiếu hay thừa hàng là lý do `cho_liam`, không là ngoại lệ; cột của `determinism.csv` giữ nguyên.
10. **C3-8, mã thoát khi không khai nguồn.** Chọn mã 5 (từ chối bàn giao); cũng có thể coi là lỗi tham số (mã 1). Đường `--nguon-gain-tay` không kiểm gì thêm ngoài bảng chặn: gain chiến dịch vẫn gửi tay được nếu khai nguồn, và dòng log ghi lại.
11. **C3-8, so `selected.csv`.** Nhận khi SHA-256 của byte trùng `selected_sha256`, hoặc SHA sau khi đổi CRLF → LF trùng `selected_sha256_lf` (repo gốc dùng `core.autocrlf=true`). `post` ghi `notes.json` hai lần (`dang_chay`, rồi `hoan_tat` sau `phases.json`, `meta.yaml` và sau khi băm lại `selected.csv`); `selected.csv` đổi trong lúc `post` chạy thì `post` dừng, `notes.json` giữ `dang_chay`.
12. **C3-48, "chỉ báo hỏng".** Bảng thẩm định ghi "DLL chính thiếu kênh kẹp (PID) hay chỉ báo hỏng → dừng ở cả S03 và S04"; đề của việc này chỉ ghi "giữ quy tắc dừng khi DLL chính thiếu kênh kẹp". S04 hiện dừng khi DLL thiếu kênh; khi `fwsim.kep_indicator` báo lỗi ở một lượt (ví dụ `kep_lech`), lượt đó chỉ ghi `kep_note` và chạy tiếp (không đổi).

## 4. Phần chưa làm và lý do

- **Hai chỗ gọi `gain_send.py` chưa truyền cờ nguồn gain** (file ngoài danh sách được sửa): `tools/hil_bench_session.py` `gain_set` (dòng 332) và `rpi5/pc_sender.py` `gain_set_file` (dòng 217–232) gọi `gain_send.py --ip … --file …`. Sau sửa, hai lệnh này bị từ chối (mã 5, không gửi) cho tới khi thêm `--s04-run <run>` (gain chiến dịch) hay `--nguon-gain-tay "<mô tả>"`. `--query` không đổi.
- **Đọc file S03 của RSM qua `s03_read`** và **siết `rsm_fake`**: ngoài phần đề nêu (điểm mơ hồ 6).
- **Đóng khoảng hở giữa `read_domain` và `fingerprint`** (điểm mơ hồ 7): cần đổi thứ tự trong `main`, đề không nêu.
- Hàng `DEFERRED`, `REJECTED` của bảng thẩm định và các ID khác: không đụng.
