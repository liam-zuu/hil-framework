# Báo cáo F-C3C: sửa hẹp test kênh kẹp sau Codex vòng 3 (C3-13, C3-35)

Worktree `D:/HIL_WT/devin3_c3c`, nhánh `fix/devin-c3c-codex3`. File sửa: `fwsim/tests/test_fwsim_kep.py`; file mới: `fwsim/tests/data/fw_sim_62e2fa5.c` (bản sao byte của `_nguon_cu/fw_sim_62e2fa5.c`). Không sửa `fw_sim.c`, `fwsim.py`, `build_fwsim.py`. `_nguon_cu/` giữ nguyên. Không commit.

Nguồn đề: `tra_loi_vong3.md` dòng 11 (C3-13), dòng 33 (C3-35); cách sửa đã chốt: `tham_dinh_vong3.md` dòng 13–14.

## 1. Bảng ID → chỗ sửa → test → kết quả

Số dòng là của file sau sửa.

| ID | Phản ví dụ Codex | Chỗ sửa (file:hàm:dòng) | Test | Trước sửa | Sau sửa |
|---|---|---|---|---|---|
| C3-13 (a) | Test so cặp bỏ qua khi thiếu nguồn cũ (file ngoài repo) | `test_fwsim_kep.py`: `NGUON_CU` (1044–1046, trỏ `fwsim/tests/data/`); `_doc_nguon_cu()` (1049–1059): thiếu file hay sai SHA-256 (bỏ CR) là assert hỏng, không skip; `test_dll_pair_with_the_source_before_kep` bỏ `pytest.skip` (1120) | `test_old_source_is_in_the_repo` (1156–1161); `test_old_source_check_rejects` (1164–1186, 3 ca `thieu`/`sai_sha`/`crlf`); `test_pair_fails_not_skips_without_the_old_source` (1189–1201) | BỎ QUA trên bản gộp (file ngoài repo); trên bản cũ chạy thử: 5 test mới HỎNG (`in_the_repo` sai thư mục, `not_skips` bắt được skip, `check_rejects` NameError) | ĐẠT (5 test) |
| C3-13 (b) | Chưa nhận diện hai phía: nguồn wrapper, cây firmware, toolchain, DLL thực nạp | `_dll_nap()` (1062–1076): `GetModuleFileNameW` trên handle ctypes → đường và SHA-256 của DLL thật sự được nạp; `_fw_src_sha()` (1079–1086): SHA-256 gộp trên các nguồn firmware theo danh sách `sources` của `fwsim_build.json`; `_tin_cap()` (1089–1112): msg đầy đủ; CLI `--pair` thêm `dll_cu_nap`, `dll_cu_nap_sha256`, `dll_moi_nap`, `dll_moi_nap_sha256` (1412–1423); assert đường/SHA DLL thực nạp từng dòng JSON và `file_sha256(_DLL) == build_info['dll_sha256']` (1134, 1147–1152); docstring module dòng 6–7 | `test_dll_pair_with_the_source_before_kep` (1115–1153), chạy CLI `--pair` bằng subprocess | Trước: msg chỉ có SHA hai DLL đường truyền vào + compiler/esp32/flags (không chứng minh DLL nào thật sự được nạp) | ĐẠT; dòng msg in ra ghi cả 4 nhóm nhận diện (mục 2) |
| C3-35 | Fake probe `real_works(dll) and len(probes) > 1` theo thứ tự: nếu ld từ chối `--wrap` lúc build, probe đầu đã là probe dự phòng và fake làm nó hỏng | `_bien_the()` (1220–1222): nhận diện DLL theo biến thể; `_gia_lap_probe()` (1225–1255): spy `build_variant`/`_gcc`/`_trace_works` trên module global, `wrap_ld=False` mô phỏng ld từ chối (RuntimeError trước khi gọi gcc), kết quả probe theo biến thể; `test_trace_fallback_builds_a_separate_dll` viết lại trên helper (1258–1275) | `test_trace_fallback_when_the_linker_rejects_wrap` (1278–1298); `test_trace_wrap_probe_ok` (1301–1311); `test_trace_fallback_fails_too` (1314–1327, 2 ca `ld_tu_choi`/`ca_hai`) | Trên fake cũ: 3 HỎNG — ca linker từ chối văng `RuntimeError: the trace records no pid_update calls` (probe đầu là dự phòng, fake theo thứ tự đánh hỏng), `wrap_probe_ok` bị ép dự phòng, `ca_hai` không ném | ĐẠT (5 test) |

Chi tiết:

- **`test_old_source_is_in_the_repo`:** `dirname(abspath(NGUON_CU))` (normcase) phải là `fwsim/tests/data`; `_doc_nguon_cu()` trả bytes (file 15967 byte, đo bằng `len(open(...,'rb').read())` khi chép).
- **`test_old_source_check_rejects`:** `thieu` → AssertionError khớp `'missing'`; `sai_sha` (bản sao + 1 byte `\n`) → AssertionError khớp SHA ghim `a93b6d09…`; `crlf` (bản sao `\n` → `\r\n`) → đạt và trả đúng bytes CRLF (SHA tính sau khi bỏ CR).
- **`test_pair_fails_not_skips_without_the_old_source`:** monkeypatch `NGUON_CU` tới đường không tồn tại rồi gọi trực tiếp `test_dll_pair_with_the_source_before_kep`: bắt riêng `pytest.skip.Exception` (BaseException) → `pytest.fail('skipped instead of failing')`; `AssertionError` phải chứa `'missing'`; không ngoại lệ nào → fail. Trên code cũ test này bắt được đúng skip của nhánh `pytest.skip(f'{NGUON_CU} not present…')`.
- **`test_trace_fallback_when_the_linker_rejects_wrap`:** `wrap_ld=False` → lệnh gcc đầu (có `-DTRACE_WRAP` và đối số `--wrap`) bị spy ném `RuntimeError("gcc failed (1): ld.exe: unrecognized option '--wrap=pid_update_vx'")` trước khi chạy gcc; kiểm: `built == [(X.dll, None), (X_du_phong.dll, False)]`, `cmds[0]` chứa `-DTRACE_WRAP` và `--wrap`, không `cmd` nào sau chứa chúng, `X.dll` không tồn tại, `probes == [('du_phong', X_du_phong.dll)]` (bản wrap không bao giờ được probe), fixture trả DLL dự phòng, `_WRAP_OK is False`, golden run trên nó trùng `glog` từng bit mọi khóa.
- **`test_trace_wrap_probe_ok`:** bình thường: `built == [(dll, None)]`, `probes == [('wrap', dll)]`, không file `_du_phong`, `_WRAP_OK is True`.
- **`test_trace_fallback_fails_too`:** `ld_tu_choi` (ld từ chối + probe dự phòng cũng hỏng) → `RuntimeError` khớp `^the trace records no pid_update calls$`, `probes == [('du_phong', …)]`; `ca_hai` (cả hai probe hỏng) → `RuntimeError` khớp `renamed symbols too`, `probes == [('wrap', …), ('du_phong', …)]`. Mỗi ca một tên DLL riêng (`fwsim_trace_hong_ld`, `fwsim_trace_hong_ca_hai`).

## 2. Dòng msg in ra (lệnh `-s -k source_before_kep`)

```
nguon cu D:\HIL_WT\devin3_c3c\fwsim\tests\data\fw_sim_62e2fa5.c sha256=a93b6d0928a83ea3c7301fd98a57ac9c67d2edfa88109f1deaa6310a9e3e4b4d sha256_sach_cr=a93b6d0928a83ea3c7301fd98a57ac9c67d2edfa88109f1deaa6310a9e3e4b4d; nguon moi D:\HIL_WT\devin3_c3c\fwsim\fw_sim.c sha256=780966fd5e49ce2ca41708d0c5557220d845750d1742773c280b633277742c9d luc_build=780966fd5e49ce2ca41708d0c5557220d845750d1742773c280b633277742c9d; firmware cu D:\CLAUDE_THESIS\03_CODE\AGV_FIRMWARE nguon_sha256=54123955fce970d295cb593d5c67866e117578cae0c23304a7b33596e0facff6; firmware moi commit=5bde4c5 dirty=False; gcc cu C:\Strawberry\c\bin\gcc.EXE | gcc.EXE (MinGW-W64 x86_64-ucrt-posix-seh, built by Brecht Sanders, r8) 13.2.0; gcc moi gcc (MinGW-W64 x86_64-ucrt-posix-seh, built by Brecht Sanders, r8) 13.2.0 flags -std=gnu11 -O2 -ffp-contract=off -Wall -Wno-unused-variable -shared; dll cu nap C:\Users\liam\AppData\Local\Temp\pytest-of-liam\pytest-7108\c3dll0\fwsim_62e2fa5.dll sha256=cc0dfd452db50497e1dd7fa2481c2e9294339aceda2b140b580e5782db6e9457; dll moi nap D:\HIL_WT\devin3_c3c\fwsim\build\fwsim.dll sha256=526934942dc8220713861d1770a6a606f0a28b7258129d518c021ca850de5d38 ghi_luc_build=526934942dc8220713861d1770a6a606f0a28b7258129d518c021ca850de5d38
```

SHA DLL mới `52693494…` trùng `dll_sha256` trong `fwsim/build/fwsim_build.json` (assert `file_sha256(_DLL) == build_info['dll_sha256']` đạt). SHA `fw_sim.c` mới `780966fd…` trùng `wrapper_sha256['fw_sim.c']` của json.

## 3. Lệnh test và dòng tổng kết nguyên văn

Thư mục chạy `D:/HIL_WT/devin3_c3c`, `PYTHONIOENCODING=utf-8`; các bản cũ chạy với `PYTHONPATH=D:/HIL_WT/devin3_c3c/fwsim`.

| Lần chạy | Lệnh | Tổng kết nguyên văn |
|---|---|---|
| File gốc trước sửa | `python -m pytest -q -p no:cacheprovider fwsim/tests/test_fwsim_kep.py -rs` | `54 passed in 48.83s` |
| Cũ C3-13 (file gốc + 3 test mới nối đuôi, không có `_doc_nguon_cu`) | `python -m pytest -q -p no:cacheprovider C:/Users/liam/AppData/Local/Temp/c3c3_tmp/test_fwsim_kep_cu13.py -k "in_the_repo or not_skips or check_rejects" -rs` | `5 failed, 54 deselected in 55.99s` |
| Cũ C3-35 (file mới, fake đổi về `real_works(dll) and len(probes) > 1`) | `python -m pytest -q -p no:cacheprovider C:/Users/liam/AppData/Local/Temp/c3c3_tmp/test_fwsim_kep_cu35.py -k "fallback or wrap_probe" -rs` | `3 failed, 2 passed, 58 deselected in 84.12s (0:01:24)` |
| Mới, nhóm C3-13 | `python -m pytest -q -p no:cacheprovider fwsim/tests/test_fwsim_kep.py -k "in_the_repo or not_skips or check_rejects" -rs` | `5 passed, 58 deselected in 2.12s` |
| Mới, nhóm C3-35 | `python -m pytest -q -p no:cacheprovider fwsim/tests/test_fwsim_kep.py -k "fallback or wrap_probe" -rs` | `5 passed, 58 deselected in 36.33s` |
| Mới, in msg | `python -m pytest -q -s -p no:cacheprovider fwsim/tests/test_fwsim_kep.py -k source_before_kep` | `1 passed, 62 deselected in 16.03s` |
| Mới, cả file | `python -m pytest -q -p no:cacheprovider fwsim/tests/test_fwsim_kep.py -rs` | `63 passed in 105.24s (0:01:45)` (0 bỏ qua) |
| Sau khi tách `sach_cr` khỏi f-string trong `_tin_cap` (dòng 1101–1102, cùng số dòng; tránh dấu `\` trong biểu thức f-string, chỉ hợp lệ từ Python 3.12) | `python -m pytest -q -s -p no:cacheprovider fwsim/tests/test_fwsim_kep.py -k "source_before_kep or in_the_repo or not_skips or check_rejects"` | `6 passed, 57 deselected in 11.17s` |
| Lần cuối, cả file | `python -m pytest -q -p no:cacheprovider fwsim/tests/test_fwsim_kep.py -rs` | `63 passed in 87.39s (0:01:27)` (0 bỏ qua) |

Chi tiết hỏng ở các bản cũ (file `.txt` trong `C:/Users/liam/AppData/Local/Temp/c3c3_tmp/`):

- `truoc_c3_13.txt`: `test_old_source_is_in_the_repo` hỏng `assert 'd:\\hil_wt\\...3c\\_nguon_cu' == 'd:\\hil_wt\\...\\tests\\data'`; `test_old_source_check_rejects` cả 3 ca `NameError: name '_doc_nguon_cu' is not defined`; `test_pair_fails_not_skips_without_the_old_source` hỏng `Failed: skipped instead of failing: ...khong_co_nguon_cu.c not present: the pair with the DLL of 62e2fa5 is not run`.
- `truoc_c3_35.txt`: `test_trace_fallback_when_the_linker_rejects_wrap` hỏng `RuntimeError: the trace records no pid_update calls` (probe đầu là DLL dự phòng, fake theo thứ tự `len(probes) > 1` làm nó hỏng); `test_trace_wrap_probe_ok` hỏng assert `built == [(dll, None)]` (probe đầu bị đánh hỏng nên fixture build dự phòng); `test_trace_fallback_fails_too[ca_hai]` `Failed: DID NOT RAISE RuntimeError` (probe thứ 2 được `len(probes) > 1` cho đạt bất kể cờ); `test_trace_fallback_builds_a_separate_dll` và `test_trace_fallback_fails_too[ld_tu_choi]` vẫn đạt — đúng bản chất lỗi theo thứ tự: trùng hợp đúng trong thứ tự 2 probe, sai khi probe đầu đã là dự phòng.

Bản cũ để so: `test_fwsim_kep_cu13.py` = file gốc + ba test mới nối đuôi nguyên văn (không `_doc_nguon_cu` nên `check_rejects` lỗi NameError — đúng dự kiến); `test_fwsim_kep_cu35.py` = file mới chỉ đổi dòng return của `fake_works` về `real_works(dll) and len(probes) > 1`. Đã xóa các bản `.py` sau khi chạy, giữ `.txt`.

## 4. Điểm mơ hồ

1. **DLL "lịch sử" không phải build từ toàn bộ cây 62e2fa5.** Không được dùng `git`, nên DLL cũ = `fw_sim.c` của 62e2fa5 + shim hiện tại + cây firmware hiện tại (`D:/CLAUDE_THESIS/03_CODE/AGV_FIRMWARE`), không phải checkout 62e2fa5 đầy đủ. Danh tính firmware phía cũ vì vậy là SHA-256 gộp trên nội dung các nguồn được biên dịch (`_fw_src_sha`, danh sách `sources` của `fwsim_build.json`), không phải mã commit; phía mới lấy `esp32.commit`/`dirty` từ `fwsim_build.json` (không chạy git trong test).
2. **`_nguon_cu/` để nguyên** tại gốc worktree; test không còn đọc nó (nguồn cũ giờ là `fwsim/tests/data/fw_sim_62e2fa5.c`, bản sao byte — đã kiểm SHA-256 thô bằng nhau và SHA bỏ CR = `a93b6d09…`).
3. **Ca "ld từ chối `--wrap`" được mô phỏng** bằng spy `_gcc` ném `RuntimeError` trước khi gọi gcc, vì `--wrap` hoạt động trên máy này (gcc 13.2.0 MinGW-W64); phản ví dụ không xảy ra tự nhiên. Tương tự ca "self-check wrap hỏng" ép `wrap_bat=False`.
4. **Tên biến thể của `test_trace_fallback_builds_a_separate_dll`** đổi `fwsim_trace_thu_du_phong` → `fwsim_trace_thu`: tên cũ kết thúc `_du_phong.dll` nên `_bien_the` xếp nhầm DLL wrap thành dự phòng (phát hiện khi chạy bản cũ C3-35 lần đầu: `probes` chỉ có `[('du_phong', X.dll)]`).
5. **`msg` của ca linker từ chối** không băm `built[0]` vì file không được tạo (`_dll_tin` chỉ dùng cho DLL tồn tại).
6. **gcc thực dùng cho build biến thể** là `C:\Strawberry\c\bin\gcc.EXE` (`shutil.which('gcc')`), chữ ký `gcc --version` dòng đầu trùng chuỗi compiler trong `fwsim_build.json`; msg ghi cả hai.
7. **`GetModuleFileNameW` trên Windows trả đường DLL đã ánh xạ**; assert so `normcase(abspath(...))` nên khác biệt chữ hoa/ký hiệu ổ không ảnh hưởng. Nền khác Windows dùng `lib._name` (chưa chạy được ở đây).

8. **SHA-256 của DLL cũ đổi theo từng lần build:** hai lần chạy cho `cc0dfd45…` (mục 2) và `b8638564…` (lần chạy `-s` sau sửa f-string), cùng nguồn, cùng gcc. Vậy SHA DLL cũ chỉ nhận diện file được nạp trong đúng lần chạy đó, không là hằng số để ghim; SHA DLL mới cố định vì là file `fwsim/build/fwsim.dll` (trùng `dll_sha256` của json). Nguyên nhân (dấu thời gian PE hay khác): chưa kiểm.

## 5. Phần chưa làm

- Chưa chạy test nào ngoài `fwsim/tests/test_fwsim_kep.py` (đề chỉ yêu cầu file này).
- Chưa so với shim/firmware của 62e2fa5 (xem điểm mơ hồ 1 — cần `git`, ngoài phạm vi được phép).
- `test_pair_fails_not_skips_without_the_old_source` bắt `pytest.skip.Exception` trước `AssertionError`: nếu trong tương lai `pytest.skip` đổi sang kế thừa `AssertionError` thì nhánh skip sẽ lọt vào assert `'missing'` — xác suất thấp, ghi nhận.
