# Báo cáo G-E5: sửa hẹp `fwsim/sens_tre_slip.py` sau Codex vòng 3 (C3-19/C3-29, C3-20)

Worktree `D:/HIL_WT/devin3_e5`, nhánh `fix/devin-e5-codex3` (nền `23e5847`). File sửa: `fwsim/sens_tre_slip.py`, `fwsim/tests/test_sens_tre_slip.py`. Chưa commit.

Nguồn: phát hiện `tra_loi_vong3.md` dòng 16 (C3-19, C3-29), dòng 17 (C3-20); cách sửa `tham_dinh_vong3.md` dòng 16, 17.

## 1. Bảng ID → chỗ sửa → test → kết quả

| ID | Chỗ sửa (file:hàm:dòng) | Test | Code cũ | Code mới |
|---|---|---|---|---|
| C3-19, C3-29: ngoại lệ worker chạy lại như `run_error` | `sens_tre_slip.py:_loi_row:214` (hàng thay thế, `status='loi_worker'`); `main.run_pool:612-618` (ngoại lệ giữ khóa, sinh hàng thay thế); `main:629-634` (`retry_jobs` gồm `LOI_STATUS`, bỏ hàng thay thế sau gộp → khóa thiếu); `merge_retries:336-356` (nhận `loi_worker`, giữ hàng `run_error` thật khi lần 2 chỉ có hàng thay thế); `main:638-641` (`loi_con`: chỉ lỗi chưa giải quyết vào `completeness`); hằng `LOI_STATUS:118` | (a) `test_c3_19_r3_exception_then_ok[X_SO, X_MOC]` (dòng 799); (b) `test_c3_19_worker_exception_recorded` (748, viết lại), `test_c3_19_worker_exception_reference` (778, sửa); thêm `test_c3_19_r3_exception_then_run_error` (824) | hỏng (81 lần gọi, khóa thiếu, không có lần 2) | đạt |
| C3-19: chờ worker có hạn | hằng `HAN_CHO_WORKER_S = 600.0:116` (căn cứ ghi ở dòng 111-115); `main.run_pool:590` (`it.next(timeout=HAN_CHO_WORKER_S)`), `:593-602` (`mp.TimeoutError` → khóa chưa về vào `loi_worker` với `ly_do='het_han_cho_worker'`, `pool.terminate()`, không chạy lại); `main:639-640` (lỗi hết hạn luôn làm `chưa kết luận`) | (c) `test_c3_19_r3_worker_os_exit` (964: `mp.Pool` thật, worker giả `os._exit(3)`, `main` chạy như script con); `test_c3_19_r3_timeout_in_process` (876), `test_c3_19_r3_timeout_in_rerun` (895), `test_c3_19_r3_wait_constant` (911) | (c) hỏng: script treo, bị giết ở hạn 90 s của `subprocess.run` (`TimeoutExpired`) | đạt |
| C3-20: định dạng `esp32.commit` | `COMMIT_RE:119` (`[0-9a-fA-F]{7,40}`, `fullmatch`); `dll_info:533-534` | `test_c3_20_r3_garbage_commit_main` (1127, qua `main`), `test_c3_20_r3_commit_format` (1153; 6/7/40/41 ký tự, chữ hoa, ký tự lạ, khoảng trắng, `\n`, `UNKNOWN`, rỗng, số nguyên) | hỏng (`garbage` → `đã xác minh`) | đạt |
| C3-20: commit có trong repo AGV | `--fw` `build_parser:433` (mặc định `DEFAULT_FW:110` = mặc định `--fw` của `build_fwsim.py:58`); `full_command:458`; `fw_readable:462`; `commit_in_repo:467` (`subprocess.run(['git','-C',fw,'cat-file','-e',f'{commit}^{{commit}}'])`, hạn `GIT_TIMEOUT_S`, dòng 120); `dll_info:535-543` (trường mới `fw_dir`, `esp32_commit_trong_repo`, cảnh báo); `main:571` | `test_c3_20_r3_commit_in_fw_repo[True/False]` (1165, qua `main`, stub), `test_c3_20_r3_fw_not_readable` (1187), `test_c3_20_r3_commit_in_repo_git_call` (1198, stub `subprocess.run`), `test_full_command` (146, thêm `--fw`) | hỏng | đạt |
| C3-20: docstring `build_verified` | `dll_info:480-491` | `test_c3_20_r3_build_verified_doc` (1220) | hỏng | đạt |

Dòng là dòng của bản sau sửa.

## 2. Lệnh test và kết quả nguyên văn

DLL: `fwsim/build/fwsim.dll` có sẵn ở nền (không build lại); 4 test cần DLL đã chạy (không có dòng skip).

- Trước sửa, test cũ trên code cũ: `python -m pytest -q fwsim/tests/test_sens_tre_slip.py` → `77 passed in 8.12s`.
- Sau sửa: `python -m pytest -q fwsim/tests/test_sens_tre_slip.py` → `102 passed in 13.31s`.
- Test mới trên code cũ (bản sao `fwsim/`, `rpi5/`, `tools/` trong `%TEMP%\e5_cu\tree`, `sens_tre_slip.py` cũ; đã xóa sau khi so):
  - nguyên bộ test mới: `68 failed, 34 passed in 20.07s` (hầu hết vì code cũ không nhận `--fw`, `SystemExit` của argparse).
  - để so đúng chỗ, bản sao test bỏ `--fw` ở hai lệnh `main` và cho `_no_git` `raising=False`: `22 failed, 80 passed in 97.27s (0:01:37)`. 22 ca hỏng: `test_full_command`; C3-19 `worker_exception_recorded`, `worker_exception_reference`, `r3_exception_then_ok[key0]`, `[key1]`, `r3_exception_then_run_error`, `r3_timeout_in_process`, `r3_timeout_in_rerun`, `r3_wait_constant`, `r3_worker_os_exit` (`subprocess.TimeoutExpired ... timed out after 90 seconds`); C3-20 `r3_garbage_commit_main`, 6 ca `r3_commit_format` có `ok=False` (`garbage`, `5bde4c`, 41 ký tự, `5bde4cg`, ` 5bde4c5`, `5bde4c5\n`), `r3_commit_in_fw_repo[True]`, `[False]`, `r3_fw_not_readable`, `r3_commit_in_repo_git_call`, `r3_build_verified_doc`. Các ca `r3_commit_format` có `ok=True` và `UNKNOWN`, rỗng, số nguyên đạt cả trên code cũ (hành vi cũ đã đúng ở đó).

## 3. Test cũ đã sửa và lý do

- `test_c3_19_worker_exception_recorded`, `test_c3_19_worker_exception_reference`: kiểm đúng hành vi đề đổi (vòng 2: ngoại lệ worker không chạy lại, `n_run_error_lan_1 == 0`, `loi_worker` chỉ lan 1). Nay: chạy lại một lần, `loi_worker` có lan 1 và lan 2, `chay_lai.csv` hai lần thử `status=loi_worker`.
- `test_full_command`: thêm `--fw` (tùy chọn mới đề yêu cầu).
- Đồ giả: `_FakePool.imap_unordered` trả `_FakeIter` có `next(timeout)` như `IMapIterator`; `_FakePool.terminate`; `_RaisingIter.next`; `_main_fakes` truyền `--fw` tới thư mục không có (không gọi git).

## 4. Hằng hạn chờ

`HAN_CHO_WORKER_S = 600.0` s, hạn chờ giữa hai kết quả liên tiếp của pool (`IMapIterator.next(timeout)`). Căn cứ: lượt dài nhất của `05_DATA/results/SENS_TRE_SLIP/2026-11-22_run01/bang.csv` là `t_chay_s` 27.25 s (PID, `io_delay_ms` 0, `slip_omega_s` 0.05, μ 0.473, v 0.2; 162 lượt; lệnh `python -c` đọc `bang.csv`, min 2.48 s, trung vị 4.90 s, max 27.25 s); 600 s ≈ 22 × 27.25 s, bội 20 làm tròn lên là **quy ước**. `GIT_TIMEOUT_S = 60` s cho `git cat-file` cũng là quy ước.

## 5. Điểm mơ hồ

1. Khóa hết hạn chờ (worker chết ở mã native) **không chạy lại**: đề ghi "quá hạn thì ... ghi vào `loi_worker` ..., pool bị hủy, kết luận `chưa kết luận`", nên tôi không đưa khóa này vào `retry_jobs` (chạy lại thì có thể thành `đủ cặp`, trái câu đó). Nếu muốn khóa hết hạn cũng được một lần chạy lại thì chỉ cần bỏ điều kiện `not r.get('_het_han')` ở `main:630`, rồi đổi `test_c3_19_r3_timeout_in_process`.
2. Hạn chờ áp cho khoảng giữa hai kết quả liên tiếp, không phải cho cả lượt. Khi hết hạn, mọi khóa chưa về đều được ghi `het_han_cho_worker`, kể cả khóa đã mất trước đó vì iterator ném lỗi (khóa không biết): khóa đó có thể có hai mục trong `loi_worker` (một mục `khoa=''`, một mục hết hạn).
3. `chay_lai.csv`: lần thử ném ngoại lệ hay hết hạn ghi `status=loi_worker` (giá trị mới của cột `status`, không thêm cột), `error` là chuỗi ngoại lệ hay `het_han_cho_worker: ...`.
4. `kiem_cap.n_loi_worker`, `khoa_loi_worker` nay chỉ đếm lỗi **chưa giải quyết** (ngoại lệ mà khóa không có hàng tốt, lỗi iterator, hết hạn). Ngoại lệ ở lần 1 mà lần 2 đạt vẫn có trong `meta.json loi_worker`, có trong `loi_chua_ro_nguyen_nhan`, nhưng không làm `chưa kết luận` ("như `run_error`"). `n_run_error_lan_1` giữ nghĩa cũ (chỉ đếm `status=run_error` ở lần 1).
5. `--fw` "đọc được" = `os.path.isdir` và `os.access(R_OK)`. Thư mục có mà không phải repo git, hay không có git trên máy, thì `commit_in_repo` trả False → `khớp SHA, thiếu nguồn ESP32` (chặt). Thư mục không có thì không kiểm repo (`esp32_commit_trong_repo` None), chỉ quy tắc định dạng quyết định. Mặc định `--fw` từ worktree `D:/HIL_WT/devin3_e5` là `D:/HIL_WT/AGV_FIRMWARE`, thư mục không có (đã kiểm), nên lượt chạy từ worktree này mà không có `--fw` thì không kiểm repo; từ repo chính thì mặc định là `D:/CLAUDE_THESIS/03_CODE/AGV_FIRMWARE` (có `.git`).
6. Nhận cả hex chữ hoa (`5BDE4C5`); git chấp nhận chữ hoa khi phân giải tên đối tượng (chưa kiểm bằng git thật).

## 6. Chưa làm

- Không chạy lưới thật (đề cấm). Không chạy `commit_in_repo` với git thật (đề cho stub).
- Không kiểm tiến trình worker mồ côi sau khi test (c) chạy trên code cũ bị giết ở hạn 90 s: tôi định chạy `tasklist` nhưng lệnh bị từ chối.
