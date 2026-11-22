# Báo cáo F-E5: sửa hẹp `fwsim/sens_tre_slip.py` sau Codex vòng 2 (C3-19/C3-29, C3-20)

Worktree `D:/HIL_WT/devin_fe5`, nhánh `fix/devin-e5-codex2` (nền `int/devin-c3 162b249`). Chưa commit.
Nguồn: `tra_loi_vong2.md` dòng 27, 28, 44; `tham_dinh_vong2.md` dòng 20, 21; spec S03 §8 điều kiện 12 (`2026-11-05_Spec_S03_DOE_chon_gain.md` dòng 274).

File sửa: `fwsim/sens_tre_slip.py`, `fwsim/tests/test_sens_tre_slip.py`. Không file nào khác.

## 1. ID → chỗ sửa → test

| ID | File:hàm:dòng | Sửa | Test (trong `fwsim/tests/test_sens_tre_slip.py`) | Code cũ | Code mới |
|---|---|---|---|---|---|
| C3-19 (ngoại lệ worker) | `sens_tre_slip.py:_job_safe:187-193` | bọc `_job` trong worker: ngoại lệ thành `{'_loi_worker', '_khoa'}`, gắn với khóa lưới | `test_c3_19_worker_exception_recorded` (707), `test_c3_19_worker_exception_reference` (726) | hỏng: `RuntimeError: boom` thoát khỏi `main`, không có `meta.json` | đạt |
| C3-19 (ngoại lệ của chính iterator) | `sens_tre_slip.py:main.run_pool:514-543` | đọc iterator bằng `next()` trong `try`; ngoại lệ ghi vào `loi_worker` (khóa rỗng khi không biết), đọc tiếp các kết quả còn lại; chặn vòng lặp khi số ngoại lệ > số job | `test_c3_19_iterator_exception_recorded` (733), `test_c3_19_rerun_exception` (746, ngoại lệ ở lần chạy lại, `lan` = 2) | hỏng: `ValueError` / `OSError` thoát khỏi `main` | đạt |
| C3-19 (hồ sơ) | `sens_tre_slip.py:main:548,553,579,597-603` | `run_pool(..., lan=2)` cho lần chạy lại; `completeness(rows, ctrls, loi_worker)`; `meta.json` thêm `loi_worker`; dòng WARNING thêm số cặp không so được và số ngoại lệ | `test_c3_19_no_exception_meta_empty` (767) | hỏng (`KeyError: 'loi_worker'`) | đạt |
| C3-29 | `sens_tre_slip.py:Y1_VALUES/GATE_VALUES/Y_FINITE:241-243`, `missing_quantities:246-258`, `completeness:261-302` | cặp chỉ tính khi cả hai lượt có Y1 (`completed` ∈ {true, false, inconclusive}), kết quả cổng (`gate_outcome` ∈ {dat, khong_dat, chua_ket_luan}), `e_perp_max` (Y2) và `tv_duty` (Y3) hữu hạn; thêm `n_cap_khong_so_duoc`, `cap_khong_so_duoc`, `khoa_thieu_dai_luong` (khóa + tên trường), `n_loi_worker`, `khoa_loi_worker`; `n_cap_du` trừ cả cặp không so được; có ngoại lệ worker hay cặp không so được thì `chưa kết luận` | `test_c3_29_main_y3_bad_in_ok_run` (597; qua `main`, đúng 81 khóa, một lượt `status='ok'` có `tv_duty` NaN, rỗng, None, ±inf, `'abc'`, float32 NaN), `test_c3_29_main_y3_nan_reference` (618), `test_c3_29_main_each_quantity` (636), `test_c3_29_comparable_edges` (653), `test_c3_29_missing_quantities_unit` (664) | hỏng: `kiem_cap.ket_luan` = `đủ cặp` (đúng phản ví dụ Codex) | đạt |
| C3-29 (cờ) | `sens_tre_slip.py:main:561-564` | `dung_lam_luot_xac_nhan` = không có lý do nào; `chưa kết luận` (gồm cặp không so được, ngoại lệ worker) đưa vào `ly_do_khong_dung` | các test C3-29 / C3-19 trên kiểm `dung_lam_luot_xac_nhan is False` | hỏng | đạt |
| C3-20 | `sens_tre_slip.py:dll_info:426-490` (phần mới 477-490) | tách hai trạng thái: `sidecar_khop_sha` (= `build_verified`, `dll_sha256` khớp) và `nguon_esp32_du` (sidecar khớp SHA còn có `esp32.commit` khác rỗng, khác `UNKNOWN`, và `esp32.dirty` kiểu bool); thêm `esp32_commit`; `nguon_build` = `đã xác minh` chỉ khi cả hai, `khớp SHA, thiếu nguồn ESP32` khi chỉ khớp SHA, `chưa xác minh` khi SHA thiếu/lệch; một WARNING khi khớp SHA mà thiếu nguồn ESP32. Dirty = true không chặn (không thêm luật "build không dirty") | `test_c3_20_main_sidecar_sha_only` (774; qua `main`, sidecar chỉ `{"dll_sha256": "<đúng>"}`), `test_c3_20_esp32_source` (804, 11 ca), `test_c3_20_esp32_commit_wrong_sha` (820), `test_c3_20_main_dirty_with_commit_confirms` (829), `test_c3_20_main_both_reasons` (839) | hỏng: `nguon_build` = `đã xác minh`, `dung_lam_luot_xac_nhan` = True | đạt |
| C3-20 (cờ) | `sens_tre_slip.py:main:561-564` | lượt xác nhận cần cả hai: khớp SHA mà thiếu nguồn ESP32 thêm lý do `nguon ESP32 chua du: esp32.commit, esp32.dirty (C3-20)`; SHA thiếu/lệch giữ đúng một lý do cũ `nguon build chua xac minh (C3-20)` | như trên; `test_c3_20_main_unverified_not_confirmation` (517, test cũ, không đổi) | — | đạt |

Docstring đầu file (dòng 57-65) và docstring `dll_info`, `completeness` ghi luật mới.

Đã kiểm thêm (không phải test): `multiprocessing.Pool.imap_unordered` của CPython 3.12.10 đọc tiếp sau một mục ném ngoại lệ. Lệnh `python -c "...p.imap_unordered(math.sqrt,[4,-1,9,16])..."` in `['2.0', '3.0', '4.0', "ValueError('math domain error')"]`. `_RaisingIter` trong test mô phỏng đúng hành vi này.

## 2. Lệnh test và dòng tổng kết nguyên văn

| Lần chạy | Lệnh | Tổng kết |
|---|---|---|
| Code cũ + test cũ (worktree, trước khi sửa) | `python -m pytest -q fwsim/tests/test_sens_tre_slip.py` | `29 passed in 25.41s` |
| Code cũ + test mới (bản chép `fwsim`, `rpi5`, `tools` vào `%TEMP%/fe5_cmp`, `sens_tre_slip.py` cũ lấy từ `%TEMP%/fe5_old`; cả hai thư mục đã xóa) | `python -m pytest -q -p no:cacheprovider fwsim/tests/test_sens_tre_slip.py` | `47 failed, 30 passed in 45.87s` |
| Code mới + test mới (worktree) | `python -m pytest -q fwsim/tests/test_sens_tre_slip.py` | `77 passed in 35.84s` |

Trên code cũ, 30 test đạt gồm 29 test cũ (kể cả ba test cũ đã chỉnh fixture, mục 4) và `test_c3_20_main_dirty_with_commit_confirms` (hành vi giữ nguyên: dirty không chặn). Lỗi trên code cũ của các test phản ví dụ: `AssertionError: assert 'đủ cặp' == 'chưa kết luận'` (C3-29), `RuntimeError: boom`, `ValueError: result not picklable`, `OSError: worker died` thoát khỏi `main` (C3-19), `AssertionError: assert 'đã xác minh' != 'đã xác minh'` (C3-20). Các test biên (`comparable_edges`, `esp32_source`, ...) hỏng trên code cũ vì chưa có trường mới (`KeyError`). Ba test cần DLL (`test_ref_equals_direct` ×2, `test_slip_omega_reaches_plant`, `test_io_delay_2ms`) đã chạy, không bị skip. Không chạy lưới thật.

## 3. Điểm mơ hồ

1. **"Có Y1, kết quả cổng"** (tham_dinh_vong2 dòng 20; spec S03 dòng 274: "không lượt nào đổi Y1 (`completed`) hay kết quả cổng O1 tổng (`gate_outcome`)"). Tôi chọn cách đọc hẹp: giá trị nằm trong tập mà evaluator / cổng sinh ra, kể cả `inconclusive` (`o1_feasibility.py:318`) và `chua_ket_luan` (`o1_gate.py:158-165`), vì phép so "có đổi không" vẫn làm được bằng so bằng. Nếu cần "đã kết luận" (chỉ true/false, dat/khong_dat) thì chỉ đổi hai hằng `Y1_VALUES`, `GATE_VALUES` (`sens_tre_slip.py:241-242`).
2. **Y4 (`sat_frac`) và `e_perp_max_cens`** không thuộc điều kiện của cặp (đề và §8.12 chỉ nêu Y1, cổng, Y2, Y3). NaN ở hai trường này vẫn cho `đủ cặp` (`test_c3_29_comparable_edges`). `tom_tat.csv` vẫn tính chúng như cũ.
3. **`status='track_err'`** không loại cặp (đề không nêu). Chỉ `run_error` loại cặp, như trước.
4. **Ngoại lệ worker không được chạy lại.** Spec chỉ nói "lượt `run_error` chạy lại một lần". Đề yêu cầu ghi `chưa kết luận`, nên chạy lại không đổi kết luận. Nếu `run_error` có lần chạy lại ném ngoại lệ, `chay_lai.csv` không có dòng nào cho khóa đó (`merge_retries` không đổi). Lần 1 nằm ở `bang.csv` (dòng `run_error`), lần 2 nằm ở `meta.json loi_worker` (`lan` = 2).
5. **Đủ nguồn ESP32** = `esp32.commit` là chuỗi khác rỗng, khác `UNKNOWN` (giá trị `build_fwsim.git_state` ghi khi lỗi git, `build_fwsim.py:36`), và `esp32.dirty` kiểu bool. Không kiểm commit có trong repo AGV hay không (đề cấm `git`). Không đòi `hil_framework`. Dirty = true vẫn đủ (đề: không thêm luật "build không dirty").
6. `nguon_build` có thêm giá trị mới `khớp SHA, thiếu nguồn ESP32`. Hai giá trị cũ giữ nguyên nghĩa với các ca cũ.
7. Worker bị giết hẳn (DLL crash làm chết tiến trình) thì `mp.Pool` có thể treo, không ném ngoại lệ. Lỗi này có từ trước và nằm ngoài đề.

## 4. Test cũ đã chỉnh và lý do

- `_fake_job` (dòng 278): `gate_outcome='pass'` → `'dat'`. `pass` không phải giá trị của `o1_gate.run_outcome`. Với C3-29, giá trị này làm mọi cặp giả thành không so được.
- `test_dll_info_clean` (189), `test_c3_20_sidecar_wrong_sha` (phần `info2`, 500): thêm `esp32.commit='5bde4c5'`. Theo C3-20, sidecar thiếu commit không còn là `đã xác minh` hay "sạch".
- `_main_fakes` (369): build `'match'` thêm commit; thêm chế độ `'shaonly'` và chế độ truyền dict.
- Assert cũ không bỏ cái nào.

## 5. Phần chưa làm

- Chưa chạy lưới thật (đề cấm).
- Chưa kiểm với `multiprocessing` thật qua `main`: worker spawn không nhận monkeypatch, và cần DLL. Hành vi iterator đã kiểm riêng ở mục 1.
