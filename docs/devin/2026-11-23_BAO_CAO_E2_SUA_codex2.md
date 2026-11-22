# BAO_CAO — F-E2, sửa hẹp sau Codex vòng 2 (C3-15/C3-31, C3-17/C3-28)

Worktree `D:/HIL_WT/devin_fe2`, nhánh `fix/devin-e2-codex2` (từ `int/devin-c3 162b249`). Không commit.

Nguồn: `D:/CLAUDE_THESIS/03_CODE/docs/review/2026-11-22_Codex_C3_dot1/tra_loi_vong2.md` (C3-15 dòng 23, C3-17 dòng 25, C3-28 dòng 42, C3-31 dòng 48); cách sửa đã chốt: `tham_dinh_vong2.md` dòng 17 (C3-15, C3-31) và dòng 19 (C3-17, C3-28).

File sửa: `tools/hil_gioi_han.py`, `tools/tests/test_hil_gioi_han.py`. File tạo: báo cáo này và `BAO_CAO.md` ở gốc worktree. `tools/s06_tre_vong.py` không đụng (C3-16, C3-27 hoãn).

## 1. ID → file:hàm:dòng → test → kết quả

Số dòng là của `tools/hil_gioi_han.py` và `tools/tests/test_hil_gioi_han.py` sau khi sửa.

| ID | Sửa ở đâu (file:hàm:dòng) | Test (dòng) | Code cũ | Code mới |
|---|---|---|---|---|
| C3-15, C3-31: khoảng mất mẫu giao cửa sổ, kể cả vắt qua `t0` (và vắt qua `t_end`) | `hil_gioi_han.py`: `n_gaps_in` 466 (mới); `esp_measure.event` 755, phép đếm `n_khong_bc` 763–766; docstring (c-ESP32) 110–120 | `test_c3_31_gap_across_t0` 773 (phản ví dụ Codex, qua `main`); `test_c3_31_gap_across_t_end` 812 (qua `main`); `test_c3_31_gaps_in_bounds` 828 (biên) | Hỏng: `n_khong_bc` '0' thay vì '1' (cả hai ca CLI); `n_gaps_in` không có (AttributeError) | Đạt |
| C3-31: `duty_first` để trống khi frame đầu không valid | `esp_measure.event` 778–779 | `test_c3_31_invalid_first_frame` 800 (qua `main`) | Hỏng: `duty_first` '0.0' (frame CRC sai) | Đạt |
| C3-31: cột riêng "độ lệch UDP chưa đo" | `LECH_UDP` 224; `EVENT_COLS` 266–269 (cột `lech_udp` ngay sau `bat_dinh_ms`); `event` 777; docstring 116–120, 144–145 | `test_c3_31_gap_across_t0` (vị trí cột, giá trị `chua_do`, `bat_dinh_ms` vẫn = phần dư khớp), `test_c3_31_gap_across_t_end` | Hỏng (không có cột) | Đạt |
| C3-15/C3-31: căn cứ `GAP_FACTOR` | chú thích 215–219 | — (chỉ chú thích) | — | — |
| C3-17, C3-28: khoá kết quả chấm theo SHA-256 NPZ + task | `sha256_file` 374, `git_blob` 382, `score_lock` 390, `check_score_lock` 400 (mới); `count_run` 672; `batch` 949–952; `main` help `--score-json` 991; docstring (a) 31–39, lô 159–162 | `test_c3_28_score_json_locked_to_npz_and_task` 950 (phản ví dụ Codex: JSON của lượt khác cùng tên NPZ, cùng số frame; task khác), `test_c3_28_score_json_fields` 970 | Hỏng: mã 3 thay vì 2 (JSON runA với NPZ runB; JSON không khoá) | Đạt |
| C3-28: ghi bộ cờ chấm và phiên bản bộ chấm vào `win_ht_src` (và JSON) | `LOCK_KEY`, `CHAM_FLAGS`, `SCORER_MODULES` 226–229; `score_window` 460–462; `score_lock` 390 | `test_c3_28_batch_real_scorer` 916 (bộ chấm thật) | Hỏng: JSON không có `e2_khoa_cham` (KeyError) | Đạt |
| C3-28: `t_end` hữu hạn khi `completed='true'`; thiếu `frame_log`, mốc không hữu hạn → mã 2 | `score_window` 423: 433–452 | `test_c3_28_score_json_fields` 970 | Hỏng (mã 3) | Đạt |
| C3-28: test bằng bộ chấm thật, ca hoàn thành sớm và ca tới deadline | fixture `real_cham` 889 chạy `--batch-from-s1 ... --cham` qua `main` với `hil_o1_score.score` thật trên hai log O1-F08 dựng (`_fig8_npz` 854) | `test_c3_28_batch_real_scorer` 916 | Hỏng (như trên) | Đạt |

### Đã sửa gì

**C3-15, C3-31 (A11):**
- `n_khong_bc` = số frame không valid trong khoảng đã quét (như cũ) + `n_gaps_in(t_f, cut, t0, t_hi)`: số khoảng mất mẫu `(t_f[k-1]; t_f[k])` có `t_f[k] > t0` và `t_f[k-1] < t_hi`. `t_hi = t_f[k_vao_dai]` khi vào dải, `t_hi = t_end` khi bị cắt (censored).
- Phép đếm này tính cả khoảng vắt qua `t0`, ca của Codex: mẫu ở 1200 ms, mốc khoảng 1202,7 ms, mẫu kế ở 1210 ms đã vào dải. Nó cũng tính khoảng vắt qua `t_end` của ca bị cắt; code cũ bỏ sót ca này, vì dùng `cut[k_first+1:k_last]`.
- Khoảng kết thúc đúng tại `t0` thì mọi mẫu mất đều nằm trước `t0`, nên không tính. Khoảng bắt đầu đúng tại `t_hi` cũng không tính. Hai ca biên này cùng ca lệch 1e-9 có trong `test_c3_31_gaps_in_bounds`.
- `duty_first` để trống khi frame `k_first` không valid (rỗng hay sai CRC).
- Thêm cột `lech_udp` = `chua_do` ngay sau `bat_dinh_ms`. Cột này ghi độ lệch một phía do trễ UDP nhỏ nhất, chưa đo. `bat_dinh_ms` không đổi: vẫn là phần dư của đường khớp, cộng độ tản tick với `tat_motor`.
- Chú thích cạnh `GAP_FACTOR` ghi căn cứ: mọi ngưỡng trong (1,07; 2) chu kỳ cho cùng kết quả trên 158 log CLOSED_LOOP. Hằng này là quy ước.

**C3-17, C3-28 (cửa sổ hoàn thành):**
- Lô `--cham` thêm khối `e2_khoa_cham` vào JSON chấm, gồm:
  - `frame_log` và `frame_log_sha256`, băm trước khi chấm;
  - `task`;
  - `allow_final_set: true`, `allow_pre_b81_15: true`;
  - `esp_log`, `esp_log_sha256` (chỉ ghi, không đối chiếu);
  - `scorer`: blob git của `hil_o1_score`, `o1_gate_eval`, `o1_gate`, `o1_feasibility`, `o1_lanes`, `o1_tasks`, `hil_edge_map`. Blob tính trên nội dung đã đổi CRLF thành LF.
- Lời gọi bộ chấm dùng chính `CHAM_FLAGS`, nên cờ ghi ra và cờ dùng là một.
- `count_run` gọi `check_score_lock` trước khi dùng kết quả chấm, cả ở `--score-json` lẫn ở lô. Hàm này băm lại NPZ và đòi:
  - SHA-256 khớp;
  - `task` của khoá = `task` của bộ chấm = `--task` (hay task của dòng s1);
  - hai cờ có mặt, kiểu bool;
  - blob `hil_o1_score` là 40 ký tự hex.
  Sai một điều → `ValueError` → mã 2 (lô: dòng lỗi, cột `error`).
- `score_window` báo mã 2 khi: thiếu `frame_log`; `t0_step`, `T_cmd` hay `t_deadline` không hữu hạn; `t_end` có mặt mà không hữu hạn hay không phải số; `completed='true'` mà không có `t_end`. Riêng `completed='false'` không có `t_end` vẫn được đọc, cửa sổ tới deadline.
- `win_ht_src` có dạng `hil_o1_score:<t0_src>:completed=<..>:allow_final_set=1:allow_pre_b81_15=1:scorer_blob=<40 hex>`. Phần đầu giữ như cũ, các trường mới nối sau.

### Test cũ đã sửa và lý do

- `_score` (dòng 649) nay thêm `task='O1-F08'`. Khi truyền `npz`, nó thêm cả khối khoá của file đó (`_lock` 640).
- `test_c3_17_score_window` 668 và `test_c3_17_score_json_mismatch` 697 truyền `npz` cho `_score`. Lý do: đề đổi đúng hành vi này, JSON không khoá nay bị từ chối. Các ca và số kỳ vọng của hai test giữ nguyên.
- `test_c3_17_batch_cham` không đổi. Bộ chấm giả trả `_score()` có `task`; lô tự gắn khoá.
- Các test cũ khác giữ nguyên.

## 2. Lệnh test và kết quả

Code mới, lần chạy cuối, sau khi đã xóa bản cũ:

```
python -m pytest -q tools/tests/test_hil_gioi_han.py      (cwd D:/HIL_WT/devin_fe2)
27 passed in 76.09s (0:01:16)
```

Code cũ, để so. Bản `tools/hil_gioi_han.py` lúc chưa sửa được chép ra `D:/HIL_WT/_fe2_old/tools/` (ngoài worktree), kèm file test mới và bản chép `hil_o1_score.py` không đổi. Lệnh chạy trong `D:/HIL_WT/_fe2_old`, với `PYTHONPATH=D:/HIL_WT/devin_fe2/tools;D:/HIL_WT/devin_fe2/rpi5;D:/HIL_WT/devin_fe2/fwsim`. Thư mục đã xóa sau khi so.

```
python -m pytest -q -p no:cacheprovider tools/tests/test_hil_gioi_han.py
7 failed, 20 passed in 80.49s (0:01:20)
python -m pytest -q -p no:cacheprovider tools/tests/test_hil_gioi_han.py -k "c3_31 or c3_28"   (sau khi đổi thứ tự assert trong test_c3_31_gap_across_t0)
7 failed, 20 deselected in 56.89s
```

- Bảy test hỏng trên code cũ là đúng bảy test mới: `test_c3_31_gap_across_t0`, `test_c3_31_invalid_first_frame`, `test_c3_31_gap_across_t_end`, `test_c3_31_gaps_in_bounds`, `test_c3_28_batch_real_scorer`, `test_c3_28_score_json_locked_to_npz_and_task`, `test_c3_28_score_json_fields`.
- Hai mươi test cũ, kể cả ba test C3-17 đã sửa fixture, vẫn đạt trên code cũ.
- Chỗ hỏng trên code cũ:
  - `gap_across_t0`: `assert '0' == '1'` ở `n_khong_bc`;
  - `invalid_first_frame`: `duty_first` '0.0';
  - `gap_across_t_end`: `n_khong_bc` '0';
  - `gaps_in_bounds`: không có `n_gaps_in`;
  - `batch_real_scorer`: `KeyError 'e2_khoa_cham'`;
  - `locked_to_npz_and_task`: mã 3 ở ca `i=2` (JSON runA với NPZ runB);
  - `score_json_fields`: mã 3 ở ca `i=2` (JSON không khoá).

Các lần chạy trước đó trên code mới, ghi đủ:
- `1 failed, 26 passed`: oracle đếm frame của test sai một đơn vị vì `T_cmd` không nằm đúng lưới 1 ms. Đã sửa oracle sang `ceil`.
- Sau đó `27 passed` ba lần: 117,11 s, 76,32 s, 76,09 s. Giữa lần 2 và lần 3 chỉ rút ngắn một dòng chú thích.

Ghi chú về các số của test bộ chấm thật:
- Lấy từ lần chạy, không ghi trong file: log dựng có 12853 frame, `k0 = 400` (stdout của test).
- Ca A: `completed='true'`, `t_end < t_deadline − 0,5 s`, `win_ht_n` bằng oracle `round(t_end·1000) − ceil(T_cmd·1000 − 1e-6) + 1`, nhỏ hơn 2000.
- Ca B: `completed ≠ 'true'`, `t_deadline − 1 ms ≤ t_end < t_deadline`, `win_ht_n = 2000`.
- Giá trị số cụ thể của `t_end` không in ra: chưa ghi.

## 3. Điểm mơ hồ

1. **JSON thô của `hil_o1_score.py --out` nay bị từ chối ở `--score-json`, mã 2.**
   - Lý do: bộ chấm không ghi SHA của NPZ, mà đề không cho sửa `hil_o1_score.py`. Vì vậy nguồn JSON có khoá duy nhất là lô `--cham`, ghi ở `cham/<run>__<ten>_score.json`.
   - Hệ quả: muốn chạy `--score-json` cho một lượt lẻ thì phải có JSON từ lô.
   - Hướng sau, cần Claude chọn: (a) cho `hil_o1_score` ghi `frame_log_sha256`; hay (b) thêm `--cham` cho chế độ một lượt. Chưa làm vì ngoài file được phép sửa hay ngoài đề.
2. **"Phiên bản scorer (commit/blob)": chỉ ghi blob.**
   - Blob tính trên nội dung file, sau khi đổi CRLF thành LF; không ghi commit, vì đề cấm gọi `git`.
   - Blob tính theo cách của `git hash-object` khi `core.autocrlf=true`, nhưng **chưa kiểm** bằng `git hash-object` thật trên máy này.
   - Nếu file có `.gitattributes` đặc biệt (eol, filter) thì blob có thể khác blob trong git. Với các file `.py` này: chưa kiểm.
3. **`esp_log_sha256` chỉ ghi, không đối chiếu với `--esp-log` của lượt đơn.**
   - Mặt nạ chỉ dùng `t0_step` trong JSON, còn A11 dùng `--esp-log`; hai log có thể khác nhau mà không ảnh hưởng mặt nạ.
   - Đề chỉ yêu cầu khoá theo NPZ + task.
4. **Căn cứ của `GAP_FACTOR` lấy nguyên từ đề, chưa kiểm lại ở đây.** Các số là 158 log, bước 1,07 chu kỳ, mất một mẫu khoảng 2 chu kỳ. Tôi định chạy một script chỉ đọc `t_send_ns` của các log để kiểm, nhưng lệnh ghi script bị từ chối.
5. **Khoảng mất mẫu vắt qua `t_end` của ca bị cắt cũng được tính.** Tôi đọc "mọi khoảng mất mẫu giao cửa sổ quan sát" là gồm cả ca này. Nó làm `n_khong_bc` của sự kiện censored có thể tăng so với lô cũ, khi lô được chạy lại.
6. **Kết quả chấm sai khoá ở chế độ lô thành dòng lỗi (mã 2), không phải `loi_cham`.** `loi_cham` vẫn chỉ dùng khi chính bộ chấm ném lỗi; khi đó lô chạy tiếp như cũ.

## 4. Chưa làm

- Không chạy lô thật. Đề giao CODE chạy lại sau. Các cột mới (`lech_udp`, `win_ht_src` dài hơn) và phép đếm `n_khong_bc` mới sẽ làm đầu ra lô khác `HIL_GIOI_HAN/2026-11-23_run03`.
- Không đụng `tools/s06_tre_vong.py` (C3-16, C3-27 hoãn).
