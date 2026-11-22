# Báo cáo F-C3C: sửa hẹp test kênh kẹp sau Codex vòng 2 (C3-13, C3-33)

Worktree `D:/HIL_WT/devin_fc3c`, nhánh `fix/devin-c3c-codex2` (từ `int/devin-c3 162b249`). File sửa duy nhất: `fwsim/tests/test_fwsim_kep.py`. Không sửa `fw_sim.c`, `fwsim.py`, `build_fwsim.py`. Không commit.

Nguồn đề: `tra_loi_vong2.md` dòng 21 (C3-13), dòng 52 (C3-33); cách sửa đã chốt: `tham_dinh_vong2.md` dòng 16.

## 1. Bảng ID → chỗ sửa → test → kết quả

Số dòng là của file sau sửa.

| ID | Phản ví dụ Codex | Chỗ sửa (file:hàm:dòng) | Test | Trước sửa | Sau sửa |
|---|---|---|---|---|---|
| C3-33 (a) | DLL trace `--wrap` nạp được nhưng không bắt lời gọi: fixture build đè cùng đường dẫn DLL đang nạp | `test_fwsim_kep.py:variants.get:697–723` (dự phòng build ra `<tên>_du_phong.dll` ở dòng 708 và 715; dòng 719 `elif _WRAP_OK is None`); docstring fixture 682–687 | `test_trace_fallback_builds_a_separate_dll` (1099–1128) | HỎNG: `RuntimeError: gcc failed (1): ... ld.exe: cannot open output file ...\fwsim_trace_thu_du_phong.dll: Permission denied` | ĐẠT |
| C3-33 (b) | `_cap_runs` lấy phía DLL mới của ca S03 từ `_RUNS` chỉ khóa theo tên ca | `test_fwsim_kep.py:_cap_runs:982–993` (dòng 993: `s03_run(fw_moi, 0.8, s03_defaults(fw_moi))[0]`) | `test_cap_runs_simulates_both_sides` (1083–1096) | HỎNG: `AssertionError: assert {...} is not {...}` (log cũ trong cache lọt vào cặp `s03`) | ĐẠT |
| C3-13 / C3-33 (c) | Cặp mặc định chỉ là bật/tắt `kep` của mã hiện tại, chưa so với DLL lịch sử; chưa ghi SHA hai DLL | `NGUON_CU`, `NGUON_CU_SHA` (1045–1046); `_dll_tin` (996–998); CLI `--pair` thêm `dll_cu`, `dll_cu_sha256`, `dll_moi`, `dll_moi_sha256` (1210–1218); thông báo assert có SHA ở `test_dll_pair_without_and_with_kep` (1019–1041); docstring module dòng 6–7 | `test_dll_pair_with_the_source_before_kep` (1049–1081), chạy CLI thật `--pair` bằng subprocess | HỎNG: `KeyError: 'dll_cu_sha256'` (CLI cũ không ghi SHA) | ĐẠT |

Chi tiết từng test:

- **`test_trace_fallback_builds_a_separate_dll`:** đặt `_WRAP_OK = None`, bọc `build_variant` để ghi lại đường dẫn từng lần build, thay `_trace_works` bằng bản gọi phép thử thật (DLL được nạp và giữ nạp trong tiến trình) nhưng trả `False` ở lần thử đầu (giả lập `--wrap` link được mà không bắt lời gọi; trên máy này `--wrap` chạy được nên chỉ tái hiện được bằng cách ép như vậy). Kiểm: đúng hai lần build, lần 2 có `_WRAP_OK is False`, đường dẫn lần 2 khác lần 1, fixture trả DLL lần 2, cả hai file tồn tại; golden run trên DLL dự phòng (đổi tên ký hiệu) trùng log DLL sản xuất từng bit ở mọi khóa (kể cả `kep`).
- **`test_cap_runs_simulates_both_sides`:** đặt vào `_RUNS['v08_mac_dinh']` log của ca `v08_ki_xy_1500` (một log khác dưới cùng tên ca, như DLL có mặc định firmware khác sẽ để lại). Kiểm: log phía mới của cặp `s03` không phải log trong cache, `so_cap` cho `n_lech == 0`, digest và npz bằng; cache không bị đổi.
- **`test_dll_pair_with_the_source_before_kep`:** không có file nguồn thì `pytest.skip` kèm lý do (nêu đường dẫn, "the pair with the DLL of 62e2fa5 is not run"). Có thì: SHA-256 của file (bỏ CR) phải bằng `a93b6d0928a83ea3c7301fd98a57ac9c67d2edfa88109f1deaa6310a9e3e4b4d` (đầu `a93b6d09…` trùng giá trị CODE ghi ở `tham_dinh_vong2.md` dòng 16; giá trị đầy đủ tính bằng `python -c "...hashlib.sha256(b.replace(b'\r\n',b'\n'))..."` trên `_nguon_cu/fw_sim_62e2fa5.c`, file có 15967 byte, 0 CRLF). Build DLL từ nguồn đó, kiểm `kep_cols == 0` và không có `fwsim_kep_last`; chạy `python fwsim/tests/test_fwsim_kep.py --pair <dll cũ>`; kiểm bốn dòng JSON `golden`, `adrc`, `motor_tat`, `s03`: khóa cũ = `BIT_EXACT_KEYS`, khóa mới = khóa cũ + `kep`, `n_lech == 0`, `digest_win_bang`, `digest_khac_chi_kep`, `npz`, SHA trong JSON bằng SHA tính lại. Thông báo assert (và dòng in ra) gồm đường dẫn và SHA-256 hai DLL, compiler, commit ESP32, cờ build.

Dòng thông tin in ra (lệnh `-s` ở mục 2, file `C:/Users/liam/AppData/Local/Temp/fc3c_tmp/sau_sua_moi.txt`):

```
old C:\Users\liam\AppData\Local\Temp\pytest-of-liam\pytest-6931\c3dll0\fwsim_62e2fa5.dll sha256=2e3b48076e82a83f8b48a7dcb14a3efd77954a3b971e59a6b0a91c18b5cca962; new D:\HIL_WT\devin_fc3c\fwsim\build\fwsim.dll sha256=77c9a2791f2d04e65ba6b161cca5b5ea4f98a6fac460c7b673160000fb78c2fd; build gcc (MinGW-W64 x86_64-ucrt-posix-seh, built by Brecht Sanders, r8) 13.2.0 esp32 {'commit': '5bde4c5', 'branch': 'HEAD', 'dirty': False, 'modified': []} flags -std=gnu11 -O2 -ffp-contract=off -Wall -Wno-unused-variable -shared
```

SHA DLL mới `77c9a279…` trùng `dll_sha256` ở `fwsim/build/fwsim_build.json` dòng 30.

## 2. Lệnh test và dòng tổng kết nguyên văn

Thư mục chạy `D:/HIL_WT/devin_fc3c`.

| Lần chạy | Lệnh | Tổng kết nguyên văn |
|---|---|---|
| Trước sửa (file gốc) | `python -m pytest -q fwsim/tests/test_fwsim_kep.py --durations=15 -p no:cacheprovider` | `51 passed in 141.74s (0:02:21)` |
| Bản cũ để so (3 test mới trên code trước sửa) | `python -m pytest -q -p no:cacheprovider C:/Users/liam/AppData/Local/Temp/fc3c_tmp/test_fwsim_kep_cu.py -k "source_before_kep or both_sides or separate_dll" -rs` (PYTHONPATH=`D:/HIL_WT/devin_fc3c/fwsim`) | `3 failed, 51 deselected in 123.07s (0:02:03)` |
| Sau sửa, 3 test mới | `python -m pytest -q -s -p no:cacheprovider fwsim/tests/test_fwsim_kep.py -k "source_before_kep or both_sides or separate_dll"` | `3 passed, 51 deselected in 80.90s (0:01:20)` |
| Sau sửa, cả file | `python -m pytest -q -p no:cacheprovider fwsim/tests/test_fwsim_kep.py -rs` | `54 passed in 134.55s (0:02:14)` |
| Sau sửa, cả file (lần cuối, sau khi đổi chỗ một dòng assert trong `test_trace_fallback_builds_a_separate_dll`) | `python -m pytest -q -p no:cacheprovider fwsim/tests/test_fwsim_kep.py -rs` | `54 passed in 93.10s (0:01:33)` |

Bản cũ để so: bản sao của file test mới, chỉ trả ba chỗ (`variants.get`, `_cap_runs`, khối CLI `--pair`) về nguyên văn code trước sửa; đặt ở `C:/Users/liam/AppData/Local/Temp/fc3c_tmp/`, đã xóa sau khi chạy (giữ lại các file `.txt` đầu ra: `truoc_sua.txt`, `sau_sua.txt`, `sau_sua_moi.txt`, `diff.txt`). Lần chạy bản cũ làm trước khi đổi chỗ dòng `assert len(built) == 2` lên trước dòng tạo `msg`; đổi chỗ này không ảnh hưởng kết quả trên code cũ (test hỏng ngay ở lời gọi `variants(...)`).

## 3. Điểm mơ hồ

1. **DLL "lịch sử" không phải build từ toàn bộ cây 62e2fa5.** Không được dùng `git`, nên chỉ có `fw_sim.c` của 62e2fa5 (file CODE đặt sẵn). Nguồn này `#include "shim/fw_shim.h"` theo đường tương đối với chính file; build thẳng từ `_nguon_cu/` lỗi `fatal error: shim/fw_shim.h: No such file or directory`. Test chép `fwsim/shim/` hiện tại và file nguồn vào thư mục tạm rồi build. Vậy DLL cũ = `fw_sim.c` của 62e2fa5 + shim hiện tại + firmware hiện tại (5bde4c5) + cùng cờ, compiler. Diff `fw_sim.c` cũ → mới (python difflib) chỉ gồm phần `kep` (`#include <math.h>`, `FWSIM_HAS_KEP`, `kep_step`, `fwsim_kep_cols/last`, gọi `kep_step` sau `vel_ctrl_step`). Shim của 62e2fa5 có khác shim hiện tại hay không: chưa kiểm (cần `git`).
2. **Định dạng đầu ra CLI `--pair`** đổi: mỗi dòng JSON thêm 4 trường `dll_cu`, `dll_cu_sha256`, `dll_moi`, `dll_moi_sha256` (đặt sau các trường cũ). Đây là cách tôi hiểu "ghi SHA hai DLL" của đề và câu "ghi SHA của hai DLL và thông tin build" của Codex (`tra_loi_vong2.md` dòng 52). CLI chỉ thuộc file test.
3. **Sửa kèm trong fixture:** `else: _WRAP_OK = True` đổi thành `elif _WRAP_OK is None: _WRAP_OK = True` (dòng 719). Trước đó, khi build `--wrap` lỗi và bản dự phòng chạy được, cờ bị đặt lại `True`, nên lần build trace sau lại dùng `--wrap`. Lỗi này nằm trong cùng nhánh dự phòng của C3-33; test riêng cho nhánh lỗi build `--wrap` (đường `except RuntimeError`): chưa có.
4. **Test so cặp lịch sử trên code cũ** hỏng do thiếu trường SHA (`KeyError`), không do lệch kênh: trên code cũ, các assert so kênh của dòng `golden` vẫn đạt trước khi gặp `KeyError`. Đây là phép kiểm mới thêm (bằng chứng lịch sử trong test), không phải tái hiện lỗi so kênh.
5. Phản ví dụ "`--wrap` không bắt lời gọi" không xảy ra tự nhiên trên máy này (`--wrap` chạy được); test ép bằng cách cho phép thử đầu trả `False` sau khi DLL đã thật sự được nạp.

## 4. Phần chưa làm

- Chưa chạy test nào ngoài `fwsim/tests/test_fwsim_kep.py` (đề chỉ yêu cầu file này; file khác không bị sửa).
- Chưa so với shim, firmware của 62e2fa5 (xem điểm mơ hồ 1).
