# BAO_CAO: sửa sau phản biện Codex vòng 1, bộ C3b (C3-6 … C3-11)

Worktree `D:/HIL_WT/devin_c3b`. Chưa commit. Không dùng `git`, không đọc `sealed/`, `sealed_test/`, không build lại DLL (`fwsim/build/fwsim_build.json`: `dll_sha256` `113da63a…`, firmware `5bde4c5`, `dirty` false).

File đã sửa (SHA-256 16 ký tự đầu, số dòng; lệnh `python -c` băm file trước và sau):

| File | Trước | Sau |
|---|---|---|
| `fwsim/s04_ga.py` | `0d30364e64d23dee`, 1572 dòng | `dd716768182daebb`, 1614 dòng |
| `fwsim/rsm_compare.py` | `a482b6e9f42a4375`, 1762 | `0db71953ff495198`, 1782 |
| `fwsim/tests/test_s04_ga.py` | `b774cbabf1acb329`, 1474 | `f77560fe99d72888`, 1819 |
| `fwsim/tests/test_rsm_compare.py` | `ae1e555e77d4eab1`, 1625 | `7a20156d5a1b60a8`, 1777 |

File mới: `BAO_CAO.md`. Không sửa `tools/*`, `fwsim/s03_doe.py` hay file nào khác. Số dòng dưới đây là của file sau khi sửa.

## 1. ID → sửa ở đâu → test → kết quả

"Cũ" = chạy test mới trên code trước khi sửa (cách kiểm ở mục 2). "Mới" = code sau sửa; mọi test ở bảng đều đạt trên code mới (mục 3).

| ID | Sửa ở đâu (file:hàm:dòng) | Test | Cũ |
|---|---|---|---|
| **C3-6** | `fwsim/s04_ga.py`: bỏ hằng `CMP_EPS`; hàm mới `mu_within` (363–368) gọi `s03_doe.not_above_by` hai chiều; `mu_rule` (371–443), khối Y2, Y3 ở 423–429; câu diễn giải `write_meta` (1596) | `test_s04_ga.py:461` `test_mu_rule_bounds_use_the_s03_exact_convention` (14 ca; điểm vi phạm chỉ ở tốc độ 0,5; bỏ tốc độ đó khỏi cổng thì `mid`) | 6 ca hỏng: Y3 4,4000000000002 và 3,5999999999998 so với 4,0; Y2 0,0400000000005 và 0,0199999999995 so với 0,03 (code cũ cho `mid`); NaN (code cũ cho `mid`); +Inf (code cũ ghi lý do `dY2`). 8 ca đạt: sáu ca `mid` (đúng biên 4,4; 3,6; 0,04; 0,02 và ngay dưới biên) và hai ca vượt 1e-9 |
| **C3-7** | `fwsim/s04_ga.py`: `TABLE_CEILING_SOURCE`, `EDGE_LIMIT_TABLE` (310–311); `edge_report` (328–346; phần thêm 341–343, 346); `write_meta` (1595) | `test_s04_ga.py:360` `test_v68_edge_report_table_ceiling_limit` (7 ca), `:376` `test_v68_selected_rows_carry_the_table_ceiling_limit` (qua `read_domain` + `summarise`), `:395` `test_v68_table_ceiling_texts_are_those_of_the_spec`; `test_rsm_compare.py:1587` `test_edge_limit_names_the_table_ceiling` (qua `candidates.csv`) | Ca 0, 3, 5, 6 hỏng ở phép so `edge_limit` (code cũ cho chuỗi rỗng hay chỉ có câu "hết lưới"); ca 1, 2, 4 đạt (không cờ, hay chỉ nguồn "hết lưới dưới": chuỗi không đổi); test `selected` và test RSM hỏng ở `edge_limit`; test chữ nguyên văn hỏng vì chưa có hằng (AttributeError) |
| **C3-8** | `fwsim/s04_ga.py`: `CHO_LIAM_NOTE` (942–946); `post` (949–1059): danh sách `cho_liam` (967), thêm mục `pre_hil` (975–978), thêm mục `three_mu` (1027–1028), khóa `cho_liam`, `cho_liam_note` trong `post_checks/notes.json` (1041), hai dòng in và `return 2 if cho_liam else 0` (1053–1059); `write_meta` (1608) | `test_s04_ga.py:1626` `_fake_post` (chạy `post` thật, không cần DLL), `:1673` `test_post_three_mu_findings_wait_for_liam` (7 ca a–g); `:964` `test_tool_check_three_phases` thêm khẳng định `cho_liam == []` (có DLL) | Cả 7 ca hỏng: a (`g4_state=khong_dat` ở μ 0,4), c (`g5_state=chua_ket_luan` ở μ 0,6), d (`track_err` ở μ 0,6): code cũ trả 0; b, e, f, g: `notes.json` cũ chưa có khóa `cho_liam` (KeyError) |
| **C3-9** | `fwsim/s04_ga.py`: `score` (593–599): quy đổi mọi dòng bằng `tv_duty_run_error` rồi mới tính cờ lỗi, điểm, cờ phạt, số đếm (598–599) | `test_s04_ga.py:1506` `test_score_fallback_uses_the_converted_row` (6 ca: thiếu khóa, `''`, NaN, +Inf, −Inf, chữ), `:1525` `test_tv_duty_error_then_successful_diagnosis_keeps_the_penalty` (4 ca: `''`, NaN, ±Inf; lần chẩn đoán thành công sau lần 1 lỗi) | 10 ca hỏng: code cũ cho `n_run_error` = 0, `n_pen_runs` = 0 ở đường dự phòng |
| **C3-10** | `fwsim/s04_ga.py`: `seed_error_rates(evals, ctrl, seeds=None)` (758–772; khởi tạo theo `seeds` 764, `rate=None` khi `n` = 0 ở 771); nơi gọi trong `main` (1320); `write_meta` (1606) | `test_s04_ga.py:1435` `test_seed_error_rate_reports_a_seed_without_runs`, `:1451` `test_seed_with_only_ngoai_bang_candidates_is_reported` (qua `run_seed`, miền nằm hẳn ngoài tam giác); `:964` thêm khẳng định khóa `seed_error_rates` = hai seed đã chạy | 2 test hỏng: hàm cũ không nhận danh sách seed (TypeError); test thứ hai cho thấy hàm cũ trả `{}` cho seed chỉ có `ngoai_bang` |
| **C3-11** (lề 199,99) | `fwsim/rsm_compare.py`: hàm mới `triangle_check` (1004–1011); `optimise(…, tri=None)` (1014–1058): tính tổng trên gain binary64 giải mã và so `<= TRI_MAX` không dung sai (1031–1032), đưa vào `ok` (1034), hai cột mới `tri_sums`, `tri_ok` của `starts.csv` (1040); `run_opt` truyền `triangle_check` (1387–1388); `INTERPRETATIONS` (116–118, 164–169). Phép bỏ theo float32 trước khi đếm M giữ nguyên (1393–1395) | `test_rsm_compare.py:699` `test_adrc_solver_result_over_the_margin_is_not_used` (4 ca qua `run_phase`: bộ giải giả lập trả tổng 199,9900000005), `:1258` `test_optimise_margin_check_is_inclusive_and_exact` (đúng 199,99 nhận; số kế tiếp, trục wz, NaN, 199,9900000005 loại), `:1287` `test_optimise_with_the_margin_check_never_uses_a_result_over_it` (SLSQP thật, chỉ kiểm bất biến) | 4 ca hỏng theo hành vi: `both`, `step2` (code cũ cho `x_RSM`, M = 1), `mixed` (code cũ giữ ứng viên có tổng 199,99000000049999), `corner` (code cũ cho trạng thái (ii)); hai test đơn vị hỏng vì chưa có tham số `tri`, hàm `triangle_check` |
| **C3-11** (`classify`) | `fwsim/s04_ga.py`: `gain_check` (445–468): dùng `gain_table.classify` khi mô-đun có (453–459), không có thì phân loại theo chuỗi của `check()` như trước (460–463); chữ ở `pre_hil_check` (930), `write_meta` (1587, 1606); `fwsim/rsm_compare.py` `INTERPRETATIONS` (170–174) | `test_s04_ga.py:1291` `_Verdict`, `:1294` `_fake_classify`, `:1313` `test_gain_check_uses_classify_when_the_table_has_it`, `:1352` `test_gain_check_without_classify_keeps_the_text_classification` | Test thứ nhất hỏng (code cũ không gọi `classify`); test thứ hai đạt (đường cũ giữ nguyên) |

Test cũ đã sửa và lý do:

- `test_rsm_compare.py:750` `test_adrc_candidate_outside_the_triangle_is_dropped_before_m` và `:785` `test_adrc_every_candidate_outside_the_triangle_is_state_ii`: hai test này dùng nghiệm giả ở góc (tổng 260) để kiểm phép bỏ theo float32. Sau C3-11 nghiệm đó bị loại sớm hơn, ở phép kiểm binary64. Với lề 0,01 không có điểm thật nào qua binary64 mà trượt float32, nên tôi cho bộ giải giả lập trả hai mã nằm trong lề (tổng 199,5 và 199,98) và thay `s4.gain_check` bằng bản giả báo vi phạm tam giác cho mã thứ hai. Các khẳng định về M, `n_ngoai_tam_giac`, trạng thái (ii) giữ nguyên. Hai test đạt trên cả code cũ và mới.
- `test_s04_ga.py:964` `test_tool_check_three_phases`: thêm hai khẳng định (C3-8, C3-10), không đổi gì khác.

## 2. Cách kiểm "hỏng trên code cũ, đạt trên code mới"

Trước khi sửa, tôi chép bốn file vào thư mục tạm của hệ thống (`%TEMP%/c3b_vong1_base/`, ngoài worktree; SHA ở cột "Trước" của bảng đầu). Mỗi lần kiểm là một lệnh `python -c`: chép `s04_ga.py`, `rsm_compare.py` cũ đè vào worktree, chạy `python -B -m pytest … -k <test mới>` trên file test mới, rồi chép lại bản mới trong `finally` và so SHA; cả năm lần đều in `restored: True`. Dòng tổng kết của các lần chạy trên code cũ:

- `test_s04_ga.py`, 43 ca test mới: `30 failed, 13 passed, 155 deselected in 35.99s`
- `test_rsm_compare.py`, 9 ca: `7 failed, 2 passed, 107 deselected in 99.67s`
- `test_s04_ga.py`, test `post` sau khi tôi bỏ phần dung thứ thiếu khóa `cho_liam`: `7 failed, 191 deselected in 11.61s`
- `test_s04_ga.py`, test C3-7 sau khi đổi sang chữ nguyên văn của spec, và test không có `classify`: `6 failed, 4 passed, 189 deselected in 9.61s`
- `test_rsm_compare.py`, test C3-7: `1 failed, 115 deselected in 54.23s`

Kết quả từng test ở cột "Cũ" của mục 1. Các ca đạt trên code cũ là ca giữ hành vi (đúng biên, ngay dưới biên, không cờ, đường không có `classify`, hai test RSM đã sửa). Năm test hỏng trên code cũ vì thiếu tên hay tham số mới, không phải vì một phép khẳng định về hành vi: hai test C3-10, hai test đơn vị của `optimise`, test chữ nguyên văn C3-7.

## 3. Lệnh test và dòng tổng kết nguyên văn

Thư mục chạy `D:/HIL_WT/devin_c3b`, biến môi trường `PYTHONIOENCODING=utf-8`, tùy chọn `-p no:cacheprovider` (không tạo `.pytest_cache` trong worktree). Chạy trên file cuối (SHA ở cột "Sau"); hai lệnh chạy cùng lúc ở hai cửa sổ nên lâu hơn bình thường.

- `python -m pytest -q -p no:cacheprovider -rs fwsim/tests/test_s04_ga.py`
  `199 passed in 1555.54s (0:25:55)`
- `python -m pytest -q -p no:cacheprovider -rs fwsim/tests/test_rsm_compare.py`
  `114 passed, 2 skipped in 1889.86s (0:31:29)`
  `SKIPPED [1] fwsim\tests\test_rsm_compare.py:1356: no symlink permission on this machine`
  `SKIPPED [1] fwsim\tests\test_rsm_compare.py:1708: no symlink on this machine`
  Hai test bỏ qua là test cũ, cần quyền tạo symlink (như lần trước).

Kiểm chéo với `gain_table.py` có `classify` (đọc file ở worktree `D:/HIL_WT/devin_c3a/tools/gain_table.py`, SHA-256 `6e7022b04b9ef29d…`, nạp vào `sys.modules` trước khi nhập `s04_ga`, chạy `pytest.main` trong cùng tiến trình; không sửa file nào):

- `fwsim/tests/test_s04_ga.py -k "gain_check or candidate or pre_hil or triangle or seed or classify or post_three_mu or v68"`: `30 passed, 169 deselected in 62.54s (0:01:02)`
- `fwsim/tests/test_rsm_compare.py -k "triangle or margin or to_gains or optimise or edge_limit_names"`: `17 passed, 99 deselected in 126.82s (0:02:06)`

Chưa kiểm: file đó có đúng là blob của commit `850de73` không (không dùng `git`; worktree C3a đang được sửa); hai bộ test đầy đủ với `gain_table.py` của C3a; các bộ test khác của repo; GA thật; `post` thật với ba mức μ (chỉ qua `_fake_post` và lượt `--tool-check` một mức μ).

## 4. `gain_table.classify`

`gain_check` hỏi `getattr(gain_table, 'classify', None)`. Có thì lấy `Verdict` (`trong_bang`, `trong_tam_giac`, `vi_pham_bang`, `vi_pham_tam_giac`): hai cờ quyết định, chuỗi chỉ để báo. Không có thì gọi `check()` và phân loại theo tiền tố `<tên gain>=` như trước. Ở worktree này `tools/gain_table.py` chưa có `classify`, nên đường đang chạy là đường cũ: tam giác trục wz vẫn chỉ là cảnh báo, tổng tính trên float32. Sau khi gộp C3a, cùng mã sẽ đi đường `classify` (hai trục đều chặn, tổng binary64 trên giá trị float32). `rsm_compare.py` không tự phân loại theo chuỗi; nó gọi `s4.gain_check`.

Lần kiểm chéo đầu với file của C3a bắt được một lỗi trong test của tôi: test "không có `classify`" xóa tên `classify` khỏi mô-đun, trong khi `check()` của C3a gọi `classify` bên trong (NameError). Tôi đổi test sang thay `s4.gain_table` bằng một đối tượng chỉ có `check` và `NAMES`.

## 5. Điểm mơ hồ

1. **Lề 199,99 chặt và nghiệm SLSQP thật (C3-11; spec S04 §12.3 dòng 235, 237).** Spec chỉ định dung sai ε = 1e−6 m cho ràng buộc `f1`, không định dung sai cho tổng `ω_c + 2·ω_o`. Phản ví dụ của Codex (199,9900000005, vượt 5e−10) buộc phép kiểm không dung sai. Tôi đo trên bài giả của test (`LEVELS_TRI`, `q1_tri`, `q2_tri`; SciPy 1.17.1, NumPy 2.4.1; lệnh `python -c` gọi `rc.optimise` có và không có `tri`):
   - 143 khởi đầu: bước 1 có 143 kết quả bộ giải báo thành công, chỉ 31 còn `converged`, 112 vượt lề (vượt từ 5,7e−14 tới 9,0e−10); bước 2 có 141 thành công, 42 còn `converged`, 101 dòng `tri_ok` = false (vượt từ 2,8e−14 tới 6,0e−8; gồm cả 2 dòng bộ giải không báo thành công). Vẫn ra 1 ứng viên, tổng đúng bằng 199,99; `f1*` 0,029059026271758593 (không kiểm: 0,0290590262717565).
   - 12 khởi đầu (mỗi 12 điểm lấy một): cả 12 kết quả bước 1 vượt lề (9,8e−12 tới 3,2e−10), trạng thái thành (i). Không kiểm thì ra 1 ứng viên, tổng vượt lề 1,25e−12.
   
   Tức là khi ràng buộc tam giác hoạt động, phần lớn nghiệm SLSQP nằm trên lề với sai số cỡ dung sai của bộ giải, và giá trị Codex nêu nằm trong dải sai số đó. Kết quả (có ứng viên hay trạng thái (i)) khi đó phụ thuộc việc có nghiệm nào rơi đúng vào hay dưới 199,99. Tôi làm đúng việc được giao (kiểm chặt) và không tự thêm dung sai hay phép chiếu.
2. **Trạng thái khi kết quả vượt lề (spec §12.4 dòng 249–250).** Tôi ghi kết quả vượt lề là không `converged` (cùng cách với RSM-15: kết quả phải kéo về hộp quá 1e−9). Vì vậy một bước không còn kết quả nào thì trạng thái là (i) "không hội tụ", dù bộ giải báo thành công. Spec không định trường hợp này; cách đọc kia là (ii) "có hội tụ nhưng M = 0". `starts.csv` giữ cả `success` và hai cột mới `tri_sums`, `tri_ok` để phân biệt.
3. **Y2, Y3 không hữu hạn ở luật μ (C3-6; spec §3 dòng 62).** `s03_doe.exact` không nhận NaN, Inf. Tôi coi điểm đó là không thỏa (lý do `<cột> not finite`), giống `s03_doe.comparable`. Code cũ cho NaN đi qua (mọi phép so với NaN đều sai), nên đây là đổi hành vi ở một đường ngoài phản ví dụ.
4. **Phạm vi của `cho_liam` (C3-8; spec §6 dòng 114, 115).** `cho_liam` gồm đúng những gì làm `post` trả 2: kiểm trước HIL (đã trả 2 từ trước) và phát hiện của phép kiểm ba μ. Kết quả "nhạy" của `b0`, R (dòng 114 cũng ghi "báo Liam trước khi chuyển sang HIL") vẫn chỉ nằm ở `notes[…]['sensitive']` và `post` vẫn trả 0 cho nó; `cho_liam_note` ghi rõ điều này. Nhánh `done=False` (lượt `--tool-check` trên S03 một mức μ) không thêm gì. `cho_liam` không được chép sang `meta.yaml`.
5. **Định dạng `edge_limit` (C3-7; spec §2 dòng 55).** Hai loại giới hạn nối bằng ` | `; câu "trần bảng chặn" không kèm kết quả kiểm ngoài miền (lưới đã cắt ở T). Các nguồn khác của cùng câu spec (`biên W`, `trần ω_o`, `sàng lọc`) chưa làm, không thuộc ID.
6. **Danh sách seed (C3-10; spec §3 dòng 67).** `main` truyền các seed thực chạy; ở lượt chính đó là danh sách đã khóa. Lượt thử không báo tỉ lệ lỗi seed (như trước).
7. **`score` (C3-9).** Cờ lỗi nay suy từ chính các dòng đã quy đổi. Nơi gọi truyền `error=False` kèm một dòng `run_error` rõ ràng thì nay nhận phạt đủ; code cũ ở ca đó lỗi KeyError khi dòng thiếu `tv_duty`. Không nơi gọi nào hiện có làm vậy.
8. **Định dạng đầu ra thêm:** `rsm/opt/starts.csv` thêm cột `tri_sums`, `tri_ok` (rỗng với PID); `post_checks/notes.json` thêm khóa `cho_liam`, `cho_liam_note`; `seed_error_rates` trong `meta_<controller>.yaml` có thể mang `rate: null`; `edge_limit` có thể mang câu thứ hai.

## 6. Phần chưa làm, không đồng ý

- Không bỏ ID nào. Không làm ID khác.
- C3-11: không bất đồng về việc kiểm lại, nhưng điểm 1 và 2 của mục 5 cần quyết định trước khi chạy RSM thật.
- Thư mục tạm `%TEMP%/c3b_vong1_base/` (bản gốc bốn file và bốn file diff) đã xóa sau khi viết báo cáo này.
