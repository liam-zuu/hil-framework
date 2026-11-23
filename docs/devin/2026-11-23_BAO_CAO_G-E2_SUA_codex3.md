# BAO_CAO — G-E2, sửa hẹp `tools/hil_gioi_han.py` sau Codex vòng 3 (C3-28)

Worktree `D:/HIL_WT/devin3_e2`, nhánh `fix/devin-e2-codex3` (từ bản gộp thử `23e5847`). Không commit.

Nguồn:
- Phát hiện: `D:/CLAUDE_THESIS/03_CODE/docs/review/2026-11-23_Codex_C3_vong3_dot2/tra_loi_vong3.md` dòng 15 (bảng phần A, C3-28) và dòng 30 (đoạn "E2 và nguồn bộ chấm").
- Cách sửa đã chốt: `tham_dinh_vong3.md` dòng 15 (C3-28, ACCEPTED, P2).

File sửa: `tools/hil_gioi_han.py`, `tools/tests/test_hil_gioi_han.py`. File tạo: báo cáo này và `BAO_CAO.md` ở gốc worktree. Không đụng ID khác, không đụng file khác.

## 1. ID → file:hàm:dòng → test → kết quả

Số dòng là của hai file sau khi sửa. Mọi test mới đi qua `hg.main([...])` (entry point CLI) và kiểm mã thoát.

| Điều của C3-28 | Sửa ở đâu (`tools/hil_gioi_han.py`) | Test (`tools/tests/test_hil_gioi_han.py`) | Code cũ | Code mới |
|---|---|---|---|---|
| Thêm `protocol` (rpi5) và `gate_c8` vào bộ chấm | `SCORER_MODULES` 230–231 (9 module); `scorer_blobs` 392 (mới, tính lại blob từ `sys.modules[m].__file__` mỗi lần gọi); `score_lock` 397, dòng 404 dùng `scorer_blobs()` | `test_c3_28_batch_real_scorer` 914 (bộ chấm thật, tập khóa 9 module, blob `protocol`, `gate_c8` dòng 931–932) | Hỏng: tập `scorer` chỉ có 7 module | Đạt |
| Đọc khóa: cờ bằng đúng `CHAM_FLAGS` (giá trị và kiểu bool) | `check_score_lock` 407, dòng 423–426 | `test_c3_28_same_npz_other_scoring_config` 1007, ca 5, 6 (`False`); `test_c3_28_score_json_fields` 970, ca 7 (`1`, int) | Hỏng (ca đầu tiên hỏng của test là i=1, xem dưới) | Đạt |
| Đọc khóa: `esp_log_sha256` trùng SHA-256 của `--esp-log` lượt này; không có log thì cả hai `None` | `check_score_lock` dòng 427–430; `count_run` 611 (thêm kwarg `esp_path=None`), gọi ở dòng 696; `process_one` 898 truyền `esp_path` | `test_c3_28_same_npz_other_scoring_config` ca 1 (khóa `None`, có `--esp-log` → 2), ca 2 (khóa SHA x, `--esp-log x` → 3), ca 3 (khóa SHA x, `--esp-log y` khác nội dung → 2), ca 4 (khóa SHA x, không `--esp-log` → 2) | Hỏng: mã 3 thay vì 2 ở i=1 | Đạt |
| Đọc khóa: blob mọi module `SCORER_MODULES` trùng blob tính lại trên bản đang chạy | `check_score_lock` dòng 431–437 | `test_c3_28_same_npz_other_scoring_config` ca 7–9 (blob `gate_c8`, `protocol`, `o1_feasibility` thay bằng 40 hex khác), ca 10–11 (khóa thiếu `protocol` / `gate_c8`, như khóa của bản trước) | Hỏng (test dừng ở i=1) | Đạt |
| `score_window`: `k0` là chỉ số nguyên trong `[0, n)` | `score_window` 441, dòng 456–458 | `test_c3_28_k0_index` 1090: −1, 12200 (= n), 12201, 100.5, 100.0, `True`, `'100'` → 2; 0 và 12199 (= n−1) → 3 | Hỏng: mã 3 thay vì 2 ở i=0 (`k0=-1`) | Đạt |
| `score_window`: `t_end` (khi có) trong `[T_cmd; t_deadline]` ±1e-9 | `score_window` dòng 472–474 | `test_c3_28_t_end_outside_window` 1061: JSON bộ chấm thật runA, `completed='true'`, `t_end = T_cmd − 0.5` → 2 (phản ví dụ Codex) và `t_deadline + 0.5` → 2; ca biên trên JSON dựng: `t_end` = 10.0 và 10.0 − 5e-10 → 3 (`win_ht_n` 1), 10.0 − 2e-9 → 2, 12.0 + 5e-10 → 3 (`win_ht_n` 2000), 12.0 + 2e-9 → 2, `completed='false'` `t_end` 9.0 → 2, 11.999 → 3 (`win_ht_n` 2000) | Hỏng: mã 3 thay vì 2 ở i=0 | Đạt |
| Ca đạt bằng bộ chấm thật | (không đổi code) fixture `real_cham` 887 chạy `--batch-from-s1 ... --cham` với `hil_o1_score.score` thật | `test_c3_28_batch_real_scorer` 914; `test_c3_28_score_json_locked_to_npz_and_task` 950 (ca mã 3); `test_c3_28_same_npz_other_scoring_config` ca 0 và ca 2 (mã 3, `win_ht_n`, `win_ht_src` bằng dòng lô) | — | Đạt |
| Docstring | module (a) 31–41 (bỏ "chỉ ghi, không đối chiếu"); `check_score_lock` 408–412; `score_window` 442–446 | — | — | — |

Định dạng `win_ht_src` và các cột CSV không đổi.

### Test cũ đã sửa và lý do

- `SCORER_MODULES` của test (842–843) nay là 9 module, như code.
- `_lock` (638) nay ghi blob thật, tính độc lập trong test bằng `_git_blob(sys.modules[m].__file__)`. Hằng `BLOB` giả (`0123…`) đã bỏ. Lý do: đề đổi đúng hành vi này, blob giả nay bị từ chối.
- `test_c3_28_score_json_fields` 970: dòng 1002–1004 so `win_ht_src` với blob thật của `hil_o1_score`, thay cho `BLOB`. Các ca và mã kỳ vọng giữ nguyên, kể cả ca `scorer={'hil_o1_score': 'abc'}` → 2.
- `test_c3_28_batch_real_scorer` 914: thêm kiểm blob `protocol`, `gate_c8` (931–932). Tập 9 module làm test này hỏng trên code cũ.
- Thêm `import copy` (dòng 11).
- Các test khác giữ nguyên.

## 2. Lệnh test và kết quả

DLL fwsim đã có sẵn ở `fwsim/build/fwsim.dll` (liệt kê `fwsim/build`: `fwsim.dll`, `fwsim_build.json`), không build lại. File test này không bỏ qua test nào.

Code mới (cwd `D:/HIL_WT/devin3_e2`):

```
python -m pytest -q tools/tests/test_hil_gioi_han.py
30 passed in 18.00s
```

Chạy lại lần hai, sau khi xóa bản cũ: `30 passed in 18.07s`.

Code cũ, để so:
- Trước khi sửa, `tools/hil_gioi_han.py` được chép ra `D:/HIL_WT/_e2c3_old/tools/`, ngoài worktree. Kèm theo là bản chép `hil_o1_score.py` không đổi và file test mới.
- Lệnh chạy trong `D:/HIL_WT/_e2c3_old` với `PYTHONPATH=D:/HIL_WT/devin3_e2/tools;D:/HIL_WT/devin3_e2/rpi5;D:/HIL_WT/devin3_e2/fwsim`. Thư mục đã xóa sau khi so.

```
python -m pytest -q -p no:cacheprovider tools/tests/test_hil_gioi_han.py
4 failed, 26 passed in 17.39s
```

Bốn test hỏng trên code cũ:
- `test_c3_28_batch_real_scorer`: `set(lk['scorer'])` thiếu `protocol`, `gate_c8` (khóa cũ có 7 module).
- `test_c3_28_same_npz_other_scoring_config`: `3 == 2` ở i=1 (code cũ nhận `--esp-log` không có trong khóa).
- `test_c3_28_t_end_outside_window`: `3 == 2` ở i=0 (code cũ nhận `t_end = T_cmd − 0.5`, `completed='true'`).
- `test_c3_28_k0_index`: `3 == 2` ở i=0 (code cũ nhận `k0 = −1`).

Không test cũ nào khác hỏng. Mỗi test dừng ở assert hỏng đầu tiên. Vì vậy với code cũ, chỉ ca đầu tiên hỏng của mỗi test được ghi nhận. Các ca sau trong cùng test chưa được chạy riêng trên code cũ: **chưa kiểm**.

## 3. Điểm mơ hồ

1. **`t_end` ngoài cửa sổ bị từ chối bất kể `completed`.**
   - Căn cứ: đề và `tham_dinh_vong3.md` dòng 15 ghi "`t_end` (khi có) nằm trong `[T_cmd; t_deadline]`", không giới hạn ở `completed='true'`.
   - Hệ quả: theo `fwsim/o1_feasibility.py`, khi chưa hoàn thành, `t_end` là frame cuối đã xét (dòng 180–181 và 305). Một log ngắn kết thúc trước `T_cmd` có `completed='inconclusive'` (`hil_o1_score.py` 145–151) và `t_end < T_cmd`. JSON như vậy nay cho mã 2. Ở lô, dòng đó thành dòng lỗi. Trước đây nó cho mặt nạ rỗng.
   - Nếu Claude muốn chỉ áp cho `completed='true'` thì chỉ cần sửa một điều kiện ở dòng 472.
2. **`k0` kiểu float nguyên (`100.0`) cũng bị từ chối.** Tôi đọc "chỉ số nguyên" là kiểu `int` của JSON, không nhận bool hay chuỗi. `hil_o1_score._json_safe` (348–359) luôn ghi `k0` thành `int`, nên JSON do `--cham` ghi không bị ảnh hưởng.
3. **SHA log ESP32 chỉ băm file CSV.** Bộ chấm còn đọc các file đi kèm `<log>.segments.csv`, `.events.txt`, `.raw.jsonl` (`hil_edge_map.py` 207, 249, 251). Các file đó không nằm trong khóa, vì đề chỉ nêu "SHA-256 của file log ESP32". Muốn khóa cả chúng thì cần thêm trường mới vào khóa: ngoài đề, chưa làm.
4. **Blob vẫn chưa đối chiếu với `git hash-object` thật.** Như vòng trước: cách tính theo `core.autocrlf=true`, nhưng **chưa kiểm** bằng git trên máy này, vì đề cấm gọi `git`. Phép so trong khóa là blob tính lại bằng cùng hàm trên bản đang chạy. Kết quả so vì vậy không phụ thuộc vào việc blob có trùng git hay không.
5. **Khóa có thêm module ngoài `SCORER_MODULES` thì vẫn đọc được.** Đề yêu cầu "mọi module trong `SCORER_MODULES` trùng"; khóa có thêm khóa phụ không bị từ chối.
6. **Ca biên "float32 làm tròn qua ngưỡng" của luật chung không áp dụng.** Các mốc ở đây là float64 trong JSON, không có phép float32. Ca biên dùng sai số 1e-9: lệch 5e-10 thì đạt, lệch 2e-9 thì từ chối.

## 4. Chưa làm

- Không chạy lô thật, đúng như đề. JSON `cham/*.json` của các lô cũ mang khóa 7 module. Đọc chúng qua `--score-json` nay cho mã 2, nên cần chạy lại `--cham`.
- Không đụng ID khác, cũng như hàng `DEFERRED` và `REJECTED`.
