# Báo cáo sửa E5 sau phản biện Codex vòng 1 (C3-19, C3-20, C3-21)

Worktree `D:/HIL_WT/devin_c2`. Nguồn đề: `03_CODE/docs/review/2026-11-22_Codex_C3_dot1/tra_loi_vong1.md` dòng 59, 61, 63; `tham_dinh_vong1.md` dòng 29–31 (cả ba ACCEPTED). Spec: S03 v8.6 §5.2 (dòng 204–218), §8 điều kiện 12 (dòng 274). Chỉ sửa `fwsim/sens_tre_slip.py` và `fwsim/tests/test_sens_tre_slip.py`; không đụng ID khác. Không dùng git (theo luật đề). `BAO_CAO.md` của việc trước tôi giữ nguyên, không ghi đè.

## 1. Bảng ID → sửa → test → kết quả

| ID | Sửa ở đâu (file:dòng, bản mới) | Test | Code cũ | Code mới |
|---|---|---|---|---|
| C3-19 | `fwsim/sens_tre_slip.py`: `key_of`, `_ks` :218–224; `completeness` :227–255 (kiểm đủ khóa, không khóa trùng hay lạ, không còn `run_error`, đủ lượt mốc, đủ cặp so; thiếu → `ket_luan` = `chưa kết luận`); `merge_retries` :258–277 (lần chạy lại thay dòng lỗi, giữ cả hai lần, ghi `lỗi chưa rõ nguyên nhân`); `per_condition_rows` :280–300 và `summary_rows` :303–… dùng `.get`, không còn tra khóa trực tiếp; `main` :450–477 (`run_pool`; `run_error` chạy lại đúng một lần trong pool mới, :462–464; bỏ `assert` mốc; ghi `chay_lai.csv` :475); meta.json :490–493 (`kiem_cap`, `n_run_error_lan_1`, `loi_chua_ro_nguyen_nhan`, `chay_lai_csv`, `dung_lam_luot_xac_nhan`, `ly_do_khong_dung`); cảnh báo ASCII :510–517 | `test_c3_19_missing_pair_no_crash` (:390), `test_c3_19_missing_reference` (:409), `test_c3_19_run_error_rerun_ok` (:422), `test_c3_19_run_error_twice` (:449), `test_c3_19_completeness_dup_and_foreign` (:470) | 5/5 hỏng | 5/5 đạt |
| C3-20 | `dll_info` :379–426: cảnh báo khi sidecar thiếu `dll_sha256` (:417–420); `verified` chỉ khi `dll_sha256` của sidecar bằng SHA của DLL đang dùng (:421); không xác minh thì `esp32_dirty = None` (:422–423); trả thêm `build_verified`, `nguon_build` (`đã xác minh` / `chưa xác minh`). `main` ghi `nguon_build` (:490); build chưa xác minh → `dung_lam_luot_xac_nhan = false` kèm lý do (:476, :493) | `test_c3_20_sidecar_without_sha` (:482), `test_c3_20_sidecar_wrong_sha` (:493), `test_c3_20_main_unverified_not_confirmation[nosha, badsha]` (:508–518); sửa `test_dll_info_esp32_dirty` (:200) | 4/4 hỏng | 4/4 đạt |
| C3-21 | Bỏ bản sao `THRESH_C2`; `THRESH_SOURCE` :98; `thresholds()` :207–215 đọc `s03_doe.THRESH` lúc chạy, thiếu ngưỡng của một đáp ứng thì dừng; `per_condition_rows`, `summary_rows` nhận `thr` (mặc định `thresholds()`); `main` đọc ngưỡng trước khi chạy lượt nào (:435), ghi `thresholds` (giá trị đã dùng) và `thresholds_source` (:500) | `test_c3_21_threshold_from_s03[0.009-True, 0.008-False, 0.008000001-True, 0.007-False]` (:529), `test_c3_21_meta_records_thresholds` (:548), `test_c3_21_refuse_without_tv_threshold` (:559); sửa `test_main_ctrl_and_meta` (:302) | 6/6 hỏng | 6/6 đạt |

### Tái hiện đúng phản ví dụ của Codex
- **C3-19, thiếu cặp so nhưng đủ mốc:** pool giả bỏ khóa `('ADRC', 1, 0.05, 0.413, 0.2)`. Trên code cũ test hỏng ở `KeyError: ('ADRC', 1, 0.05, 0.413, 0.2)` tại `fwsim/sens_tre_slip.py:228` (`summary_rows`), tức sau khi đã ghi `bang.csv` và trước `meta.json`, đúng như Codex nêu. Trên code mới đủ file; `kiem_cap`: `n_cap_thieu` 1/72, `khoa_thieu` ghi đúng khóa đó, `ket_luan` = `chưa kết luận`.
- **C3-19, `run_error`:** trên code cũ `_job` chỉ được gọi 81 lần (`assert (81 == 82)`), tức không chạy lại; với lỗi hai lần thì `assert (1 == 2)`. Thiếu mốc: code cũ dừng ở `AssertionError: missing reference runs` (`sens_tre_slip.py:359`), không ghi gì.
- **C3-20:** sidecar `{"esp32":{"dirty":false}}` không có SHA, và sidecar có SHA sai: code cũ trả `esp32_dirty` = False (`assert False is None` ở :487, :498, :513).
- **C3-21:** khóa ngưỡng Y2 ở 0,008, chênh 0,009 m: code cũ trả `vuot_nguong` False (`assert (1 == 1 and False is True)` ở :540), cột `nguong` = 0,01 (`assert 0.01 == 0.008` ở :541).

### Cách kiểm "hỏng trên code cũ, đạt trên code mới"
1. Trước khi sửa, chép `fwsim/sens_tre_slip.py` ra bản tạm (`python -c "shutil.copy(...)"`).
2. Sau khi sửa và viết test: chép bản mới ra bản tạm, chép bản cũ vào chỗ của nó, chạy `python -m pytest -q --tb=line --show-capture=no -rf -p no:cacheprovider fwsim/tests/test_sens_tre_slip.py`. Kết quả nguyên văn: `16 failed, 13 passed in 22.30s`. 16 test hỏng gồm 15 test mới và `test_dll_info_esp32_dirty` đã sửa; 13 test cũ còn lại đạt.
3. Chép lại bản mới (`filecmp.cmp(..., shallow=False)` in `True`), chạy lại và được 29 đạt (mục 2). Bản tạm đã xóa.

Các test hỏng trên code cũ vì sai hành vi, trừ hai test:
- `test_c3_19_completeness_dup_and_foreign` hỏng vì code cũ không có hàm `completeness`. Code cũ không có phép kiểm khóa trùng hay khóa lạ, nên không có hành vi cũ nào để so.
- `test_c3_21_refuse_without_tv_threshold` hỏng vì code cũ không có hàm `thresholds`.

## 2. Lệnh test và dòng tổng kết

`python -m pytest -q -p no:cacheprovider fwsim/tests/test_sens_tre_slip.py` (cwd `D:/HIL_WT/devin_c2`, chạy sau khi đặt lại bản mới):

```
29 passed in 17.52s
```

Không có test bị bỏ qua: cả 3 test cần DLL (`test_ref_equals_direct` ×2, `test_slip_omega_reaches_plant`, `test_io_delay_2ms`) đều chạy và đạt. Tôi không build lại DLL.

### Test cũ đã sửa, và lý do
- `test_dll_info_esp32_dirty`: sidecar của test không có `dll_sha256`. Theo C3-20, sidecar như vậy không được dùng làm kết quả nữa, nên `esp32_dirty` thành None. Tôi thêm `dll_sha256` đúng SHA để test vẫn kiểm được đường "dirty = true". Đường không có SHA do test mới kiểm.
- `test_main_ctrl_and_meta`: thêm `monkeypatch.setitem(s03.THRESH, 'tv_duty', 0.4)`. Theo C3-21, `main` đọc ngưỡng từ `s03_doe.THRESH`, mà ở cây này `s03_doe.THRESH` chưa có `tv_duty` (xem mục 3).

## 3. Điểm mơ hồ

1. **Ngưỡng Y3 tuyệt đối 0,4 s⁻¹ chưa có trong `s03_doe.py` của worktree này.** `fwsim/s03_doe.py:133–134` vẫn là bản v8.5: `THRESH` chỉ có `e_perp_max`, `e_perp_max_cens`, `sat_frac`, còn `Y3_REL = 0.20` là ngưỡng tương đối. Spec S03 v8.6 §5.2 (dòng 211) và §11 mục 6 (b) (dòng 301) đổi sang 0,4 s⁻¹ tuyệt đối trong `s03_doe.py`. Worktree `D:/HIL_WT/devin_c3a` (tôi chỉ đọc) có `Y3_ABS = 0.4` và `THRESH[...'tv_duty': Y3_ABS]` ở dòng 156–158. Đề yêu cầu đọc ngưỡng "cùng nguồn đã khóa với S03", nên tôi không giữ bản sao riêng. `thresholds()` dừng (`SystemExit`, thông báo ASCII) khi `s03_doe.THRESH` thiếu ngưỡng của một đáp ứng. Hệ quả: chạy thật `sens_tre_slip.py` trên nhánh này một mình sẽ bị từ chối cho tới khi gộp với phần sửa `s03_doe.py` của C3a. Test đặt `tv_duty` bằng monkeypatch nên vẫn chạy được trước và sau khi gộp. Claude cần xác nhận cách này, hay muốn giữ nguồn dự phòng có ghi nhãn.
2. **Điều kiện để là "lượt xác nhận dùng trước S03 chính".** Tôi đặt `dung_lam_luot_xac_nhan = true` chỉ khi build đã xác minh (C3-20) **và** `kiem_cap.ket_luan` = `đủ cặp` (C3-19). Tôi không thêm điều kiện `esp32_dirty == false`, cũng không xét chênh vượt ngưỡng hay đổi Y1 (§8 điều 12, dòng 274, câu "báo Liam trước S03 chính"). Đó là phần đọc kết quả, không thuộc ba ID này. Nếu muốn coi build dirty cũng là không dùng được, đó là một điều kiện mới cần quyết.
3. **Chạy lại `run_error` "trong tiến trình mới".** §8 điều 12 chỉ nói "chạy lại một lần". Tôi chạy lại trong một `mp.Pool` mới, tức tiến trình mới, giống S04 §3 và `s03_doe.checked`. Lần chạy lại thành công thì số của nó vào `bang.csv` (spec: "dùng số của lần chạy lại"). Cả hai lần ghi vào `chay_lai.csv` (cột như `bang.csv`, thêm `attempt`). Spec nói ghi `meta.yaml`, còn script này ghi `meta.json` như trước; `meta.yaml` của thư mục lượt do Claude soạn.
4. **Ca biên float32:** không áp dụng cho ba ID này. Ngưỡng và hiệu đều so trên binary64 như code cũ. Ca bằng đúng ngưỡng (0,008, không vượt) và ca trên ngưỡng 1e-9 (vượt) đã có trong test.

## 4. Phần chưa làm và lý do

- Chưa chạy lượt thật nào bằng DLL với code mới, vì đề không yêu cầu. Test `main` dùng pool và `_job` giả.
- Chưa sửa `s03_doe.py` để thêm ngưỡng Y3 tuyệt đối: đó là việc của ID khác (C3a), ngoài phạm vi.
- Không commit (theo luật đề).
