# BAO_CAO: sửa sau phản biện Codex vòng 1, bộ C3c (C3-12, C3-13)

Worktree `D:/HIL_WT/devin_c3c`. Chưa commit. Firmware không sửa. Chỉ sửa ba file của bộ C3c: `fwsim/fw_sim.c`, `fwsim/fwsim.py`, `fwsim/tests/test_fwsim_kep.py`.
DLL cuối build từ AGV `5bde4c5` (`dirty=False`), gcc 13.2.0, sha256 DLL `0944f958…` (`fwsim/build/fwsim_build.json` trường `dll_sha256`, trùng sha256 tính lại của `fwsim/build/fwsim.dll`).

## 0. Tóm tắt

- **C3-12:** ở tick logic dừng đang giữ (`vc.still`), DLL không còn bỏ phép tự kiểm: trạng thái PID của firmware sau bước phải đúng là trạng thái reset, không thì `kep_lech` = 1. Thêm bộ test dùng **dấu vết độc lập** lấy ngay sau phép kẹp thật trong firmware (DLL chỉ dùng cho test, build lúc chạy test). Phản ví dụ của Codex tái hiện bằng firmware đột biến: code cũ báo chạm với `kep_lech` = 0, code mới báo `kep_lech` = 1.
- **Còn một ca DLL không tự thấy được:** nhánh giữ reset PID rồi bỏ qua bước cập nhật. Sau bước, trạng thái firmware giống hệt firmware thật, nên chỉ bộ test dấu vết bắt được. Muốn DLL tự bắt thì phải sửa `fwsim/build_fwsim.py` (ngoài ba file của bộ này), tôi không làm (mục 5, 6).
- **Firmware chiến dịch không đi nhánh Codex giả định:** `vel_ctrl.c` thật cập nhật và kẹp PID trước, reset sau, ở mọi tick giữ (mục 2.1).
- **C3-13:** định nghĩa kênh phải trùng từng bit là `fwsim.BIT_EXACT_KEYS`; thêm công tắc biên dịch `FWSIM_NO_KEP` để test dựng DLL không có bộ đếm; test cặp DLL trên các kênh đó; test mẫu số của `kep_indicator` với cửa sổ của chính bộ chấm. Phần mẫu số **không có lỗi ở code cũ**: test mới đạt cả trên code cũ.
- Ba file test đề yêu cầu đều đạt, không test nào bị bỏ qua (mục 4).

## 1. Bảng ID → sửa ở đâu → test → kết quả

Số dòng là của file cuối. Test ở `fwsim/tests/test_fwsim_kep.py`.

| ID | Sửa ở đâu | Test | Kết quả |
|---|---|---|---|
| C3-12 | `fwsim/fw_sim.c` `kep_step` dòng 283–303 (phép kiểm mới 296–298; chú thích 273–282). `fwsim/fwsim.py` chú thích `KEP_COLS` dòng 99–102, thông báo lỗi của `kep_indicator` dòng 472–473 | `test_mutant_hold_is_flagged` (943, hai đột biến); `test_mutant_reset_then_skip_is_seen_by_the_trace` (961); `test_trace_dll_gives_the_log_of_the_production_dll` (804); `test_trace_enter_hold_exit_still` (813); `test_trace_motor_off_on` (834); `test_trace_pid_adrc_pid` (872); `test_trace_gain_change_mid_run` (891); `test_trace_simulate_runs` (919) | Code mới: đạt cả. Code cũ: `test_mutant_hold_is_flagged` hỏng cả hai tham số; các test còn lại đạt |
| C3-13 | `fwsim/fwsim.py` `BIT_EXACT_KEYS` dòng 104–111. `fwsim/fw_sim.c` công tắc `FWSIM_NO_KEP` dòng 40–47, 63–67, 95–98, 267–270 | `test_bit_exact_keys_definition` (972); `test_dll_pair_without_and_with_kep` (1009); `test_denominator_is_the_evaluator_window` (1066, ba ca); lệnh `--pair` (dòng 1106–1114) | Code mới: đạt cả. Code cũ: hai test đầu hỏng; test mẫu số đạt (không có lỗi hành vi) |

Hàm phụ mới trong file test: `TRACE_C` (559), `build_variant` (611), `TraceSim` (648), fixture `variants` (681), `kep_theo_vet` (722), `boot_trace` (733), `codex_ticks` (751), `_MUTANTS` (771), `mutant_vel_ctrl` (783), `_cap_runs` (978), `so_cap` (991), `_eval_rows` (1031), `s03_motor_tat` (1041). Test cũ không sửa; file test chỉ đổi một dòng cũ (dòng cuối docstring đầu file, để thêm dòng `--pair`), còn lại là thêm (588 → 1 147 dòng; lệnh `python fwsim/build/c3_evidence/do_mau.py`).

## 2. C3-12

### 2.1. Thứ tự reset và kẹp trong firmware thật

Đọc `D:/CLAUDE_THESIS/03_CODE/AGV_FIRMWARE/src/controller/vel_ctrl.c` và `pid.c` (cây mà DLL build ra, commit `5bde4c5`):

- Motor tắt: `vel_ctrl_step` trả về sớm, PID không chạy (`vel_ctrl.c` dòng 146–150).
- Cạnh motor tắt → bật: `pid_reset` **trước** bước cập nhật (dòng 154–158).
- Bước cập nhật: `pid_update_vx/vy/wz` (dòng 168–170); phép kẹp nằm trong `pid_axis_update` (`pid.c` dòng 41–43).
- Logic dừng đang giữ (`g_still`): `pid_reset` **sau** bước cập nhật, ở mọi tick giữ kể cả tick vào (dòng 194–199).
- `vel_ctrl_set_mode` reset PID giữa hai tick (dòng 44–50). `pid_reset` đặt `integral` = 0, `prev_error` = 0, `initialized` = false (`pid.c` dòng 94–104).

Vậy firmware chiến dịch không đi nhánh "reset hoặc bỏ qua PID trước cập nhật" mà Codex nêu. Bộ test dấu vết dưới đây xác nhận điều này bằng chính mã chạy, không chỉ bằng đọc nguồn.

### 2.2. Sửa trong DLL

`kep_step` (`fw_sim.c` dòng 283–303). Ở tick `vc->still`, giá trị báo vẫn lấy từ bản chép (firmware đã reset nên không đọc lại được). Phép kiểm mới: trạng thái PID firmware giữ sau bước phải đúng là trạng thái reset ở cả ba trục (`integral` = 0, `prev_error` = 0, `initialized` = false, so từng bit); khác thì `kep[6]` (`kep_lech`) = 1. Tick khác không đổi: bản chép phải trùng tích phân firmware từng bit. `fwsim.kep_indicator` đã từ chối (`ValueError`) mọi log có `kep_lech` ≠ 0; chỉ đổi câu thông báo.

Phép kiểm này bắt được hai trong ba cách nhánh giữ có thể lệch (bảng 2.4). Nó không gây báo sai với firmware thật: 51 test đạt, trong đó có các lượt có tick giữ.

### 2.3. Dấu vết độc lập

Sau bước, firmware không để lại gì cho biết PID đã chạy trước hay sau lần reset. Vì vậy dấu vết phải lấy **bên trong bước**. Test tự build một DLL riêng vào thư mục tạm của pytest: cùng nguồn firmware, cùng `fw_sim.c`, cùng cờ biên dịch (đọc từ `fwsim/build/fwsim_build.json`), thêm một file C ngắn (`TRACE_C`) và cờ liên kết `-Wl,--wrap=pid_update_vx,--wrap=pid_update_vy,--wrap=pid_update_wz`. File đó chặn ba lời gọi `pid_update_*` của `vel_ctrl_step` trên PID của firmware, gọi hàm thật, rồi ghi số lần gọi, tích phân ngay sau phép kẹp thật và giới hạn. Bản chép của `kep_step` cũng gọi các hàm này trên bản sao; phép so con trỏ `c == vel_ctrl_pid()` loại nó ra.

- `kep_theo_vet` (722) đổi một dòng dấu vết thành dòng `KEP_COLS` phải có: không lời gọi → bảy số 0; ba lời gọi → cờ `|I| == giới hạn`, ba tích phân, `kep_lech` 0.
- DLL dấu vết không làm đổi kết quả: lượt `golden_run` qua nó trùng log của DLL chính từng bit ở mọi kênh, kể cả `kep` (`test_trace_dll_gives_the_log_of_the_production_dll`).
- Cờ `--wrap` chạy được với ld của MinGW trên máy này; đường lùi (biên dịch riêng `vel_ctrl.c` với tên hàm đổi) có trong `build_variant` nhưng chưa phải dùng.

### 2.4. Phản ví dụ của Codex và các ca yêu cầu

**Kịch bản** (`codex_ticks`, tick trực tiếp 62,5 ms, PID mặc định, `still_v` = 0,5): 7 tick sai số 1,0 cho `I_vx` = 0,4375; một tick 0,84 cho `I_vx` trong (0,4899; 0,4901), cờ 0; 8 tick lệnh 0, xe đứng, tích phân giữ nguyên; tick thứ 9 là **tick vào giữ**, bánh lùi 54 count (vx đo trong (−0,21; −0,19) m/s) nên tổng vượt 0,5. Sau đó một tick giữ đứng yên, một tick giữ có chuyển động, một tick ra.

**Firmware thật** (`test_trace_enter_hold_exit_still`): ở tick vào, dấu vết ghi ba lời gọi, `I_vx` sau kẹp = giới hạn = 0,5; DLL báo `kep_vx` = 1, `kep_I_vx` = 0,5, `kep_lech` = 0. Tick giữ có chuyển động: DLL báo đúng tích phân sau kẹp của dấu vết (khác 0) dù firmware reset ngay sau. Tick ra: 0,0625. Mọi tick: dòng `kep` bằng dòng suy từ dấu vết.

**Firmware đột biến** (bản chép `vel_ctrl.c` vá trong thư mục tạm, không đụng cây firmware):

| Đột biến | Nhánh giữ làm gì | Dấu vết ở tick vào | DLL cũ | DLL mới |
|---|---|---|---|---|
| `reset_truoc_cap_nhat` | reset trước cập nhật, không reset sau | PID chạy, không chạm | `kep_vx` = 1, `I` = 0,5, `kep_lech` = 0 | `kep_lech` = 1 |
| `bo_qua_khong_reset` | bỏ qua cập nhật, không reset | PID không chạy | như trên | `kep_lech` = 1 |
| `reset_roi_bo_qua` | bỏ qua cập nhật, vẫn reset sau | PID không chạy | như trên | `kep_lech` = 0 (không thấy được) |

Hai dòng đầu là `test_mutant_hold_is_flagged`: test khẳng định trước rằng dấu vết không có lần chạm thật và DLL đang báo chạm (1; 0,5), rồi đòi `kep_lech` = 1 và `kep_indicator` ném `ValueError` có chữ `kep_lech`. Dòng ba là `test_mutant_reset_then_skip_is_seen_by_the_trace`: chỉ khẳng định phép so với dấu vết phát hiện lệch; không khẳng định gì về `kep_lech`.

**Các ca Codex yêu cầu ghép thêm**, mỗi tick so với dấu vết:

| Ca | Test |
|---|---|
| Tick vào, giữ, ra khỏi `still` | `test_trace_enter_hold_exit_still`; `test_trace_simulate_runs` (a): lượt `simulate` 5 s chạy, dừng, chạy lại |
| Motor tắt–bật (đang chạm kẹp; đang giữ) | `test_trace_motor_off_on` |
| PID–ADRC–PID | `test_trace_pid_adrc_pid` |
| Đổi gain giữa lượt | `test_trace_gain_change_mid_run` |
| Lượt số 8 kiểu S03 có chạm kẹp (0,8 m/s, `ki_xy` = 1 500) | `test_trace_simulate_runs` (b) |

Ca biên trong `test_trace_gain_change_mid_run`: hạ giới hạn xuống dưới tích phân đang có (chạm ngay, `I` = 0,25); đặt giới hạn **bằng đúng** tích phân 0,375 với sai số 0 (chạm, không bị cắt); đặt giới hạn cao hơn một bậc float32 (không chạm). Ca sai phải từ chối: hai đột biến trên.

### 2.5. Đã kiểm "hỏng trên code cũ" thế nào

Trước khi sửa, chép ba file đang có (đúng bản đã commit) vào `fwsim/build/c3_old/`. Sau khi xong: chép `fw_sim.c` và `fwsim.py` cũ đè lên, giữ file test mới, build lại DLL, chạy `python -m pytest -q -rfEs fwsim/tests/test_fwsim_kep.py`. Kết quả (`fwsim/build/c3_evidence/kep_on_old_code.txt`): `4 failed, 47 passed in 57.92s`. Hai test C3-12 hỏng ở đúng dòng `assert kep[K['kep_lech']] == 1.0` với `assert 0.0 == 1.0`, sau khi hai khẳng định trước nó (dấu vết không chạm; DLL báo (1; 0,5)) đã đạt. Đó là đúng phản ví dụ: DLL cũ báo chạm với `kep_lech` = 0. Rồi chép lại file mới (so sha256 với bản lưu `fwsim/build/c3_new/`: trùng cả ba), build lại, chạy ba file test (mục 4).

Các test dấu vết trên firmware thật và test đột biến thứ ba đạt cả trên code cũ: với firmware thật, bản chép của code cũ vốn đúng.

## 3. C3-13

### 3.1. Kênh phải trùng từng bit

`fwsim.BIT_EXACT_KEYS` (`fwsim.py` dòng 104–111): 19 kênh, là mọi kênh của `simulate()` trừ `kep`, theo đúng thứ tự khóa. Định nghĩa: cùng firmware, trình biên dịch, cờ build, mã plant và tham số thì DLL có cổng `kep` và DLL không có cho các mảng này trùng kiểu, kích thước và từng byte; `kep` là khóa duy nhất thêm, đứng cuối. Hệ quả ghi trong chú thích: file npz, `s03_doe.log_digest`, cột `log.csv` và mọi mã băm dựng từ chúng khác nhau giữa hai DLL **đúng ở mục `kep`**; muốn so với kết quả của DLL cũ thì so trên các khóa này. Thân `simulate()` và thứ tự khóa không đổi. `test_bit_exact_keys_definition` buộc danh sách này bằng `OLD_KEYS` (danh sách mà mốc `GOLDEN` của vòng trước phủ).

### 3.2. Cặp DLL không có và có bộ đếm

- Công tắc `-DFWSIM_NO_KEP` trong `fw_sim.c`: khi bật, không dòng mã `kep` nào được biên dịch và DLL không có cổng `fwsim_kep_cols`, `fwsim_kep_last` (như DLL trước khi thêm bộ đếm; `fwsim.py` đi đường "DLL cũ"). Build mặc định không đổi.
- `test_dll_pair_without_and_with_kep`: build DLL `FWSIM_NO_KEP` vào thư mục tạm, so với DLL chính trên bốn lượt: `golden_run` (PID, có chạm kẹp và giữ đứng yên), ADRC 1 s, motor tắt 0,5 s, lượt số 8 kiểu S03 0,8 m/s gain mặc định. Mỗi lượt: khóa của log cũ = `BIT_EXACT_KEYS`; khóa log mới = cũ + `kep`; 0 kênh lệch; `s03_doe.log_digest` trên các khóa đó bằng nhau và hai bản đầy đủ chỉ khác mục `kep`; các mảng đọc lại từ `s03_doe.npz_bytes` trùng từng bit, `kep` là file thêm duy nhất. Riêng `golden_run`: `log.csv` của `s03_doe.save_log_csv` trùng từng ô chữ sau khi bỏ các cột `kep_*`.
- **Với nguồn thật trước khi có bộ đếm:** `D:/HIL_WT/devin_c3a/fwsim/fw_sim.c` (chỉ đọc; không có chữ `kep` nào; sha256 `caf930c3…149d695`) build bằng `build_variant` ra `fwsim/build/c3_old/fwsim_prekep.dll`, rồi `python fwsim/tests/test_fwsim_kep.py --pair fwsim/build/c3_old/fwsim_prekep.dll` so với DLL cuối (sha256 `0944f958…`). Cả bốn lượt: 0 kênh lệch trên 19 kênh, `digest_win_bang` true, `digest_khac_chi_kep` true, `npz` true (`fwsim/build/c3_evidence/pair_prekep_final.txt`).
- Mốc `GOLDEN` ghi từ DLL trước khi sửa ở vòng trước vẫn khớp: `test_old_channels_bit_for_bit_as_before_the_change` đạt, không bị bỏ qua.

### 3.3. Mẫu số của `kep_indicator` với cửa sổ của bộ chấm

`test_denominator_is_the_evaluator_window` lấy cửa sổ từ chính bộ chấm: `rows` mà `o1_feasibility.evaluate_v7` trả về là các khung nó đã chấm, tức n khung đầu của log. Test kiểm: thời gian của `rows` bằng n dấu thời gian đầu của log; `rows[-1]['t']` = `t_end`; biểu thức `t <= t_end + 1e-9` cho đúng mặt nạ đó; `sat_frac` của bộ chấm tính lại trên mặt nạ đó ra đúng số (bộ chấm tính `sat_frac` trên mặt nạ `m` của nó, `o1_feasibility.py` dòng 322–344). Rồi: `kep_mau` = số tick trong mặt nạ có `en` = 1; `kep_n` từng trục = tổng cờ trong mặt nạ; lùi `t_end` một tick thì `kep_mau` giảm đúng bằng `en` của tick ở `t_end` (tick ở `t_end` nằm trong cửa sổ).

Số đo ba ca (lệnh `python fwsim/build/c3_evidence/do_mau.py`):

| Ca | Số khung bộ chấm | Số tick log | `t_end` (s) | Hoàn thành | Tick motor tắt trong cửa sổ | `kep_mau` | `kep_n` vx / vy / wz |
|---|---|---|---|---|---|---|---|
| 0,8 m/s, mặc định | 1 096 | 1 195 | 10,96 | true | 0 | 1 096 | 0 / 0 / 0 |
| 0,8 m/s, `ki_xy` = 1 500 | 1 195 | 1 195 | 11,95 | false | 0 | 1 195 | 0 / 0 / 40 |
| 0,8 m/s, mặc định, motor tắt khi 3,0 ≤ t < 3,5 s | 1 096 | 1 195 | 10,96 | true | 50 | 1 046 | 0 / 0 / 0 |

- Ca một: 99 tick sau `t_end` nằm ngoài cửa sổ và ngoài mẫu số.
- Ca ba: 50 tick motor tắt nằm trong cửa sổ của bộ chấm nhưng không vào mẫu số (1 096 − 50 = 1 046); cả bảy cột `kep` bằng 0 ở các tick đó.
- Ca hai: log không có tick nào ở hay sau hạn chót (tick cuối 11,95 s, hạn chót 11,9533 s), nên cửa sổ là cả log. Test kiểm đúng điều đó thay cho "tick ở hạn chót nằm ngoài".

**Không tìm thấy lỗi ở phần này:** test đạt cả trên code cũ (không nằm trong 4 test hỏng ở mục 2.5). `kep_indicator` không sửa.

### 3.4. "Hỏng trên code cũ" của C3-13

Cùng lượt chạy ở mục 2.5: `test_bit_exact_keys_definition` hỏng vì `fwsim` chưa có `BIT_EXACT_KEYS` (`AttributeError`); `test_dll_pair_without_and_with_kep` hỏng ở `assert ref.kep_cols == 0` (`7 == 0`: `fw_sim.c` cũ bỏ qua `-DFWSIM_NO_KEP`). Hai test này hỏng vì code cũ **thiếu định nghĩa và công tắc**, không phải vì kênh cũ lệch: cặp DLL với nguồn thật trước khi có bộ đếm cho 0 kênh lệch (mục 3.2), và mốc `GOLDEN` đã khớp từ vòng trước.

## 4. Lệnh đã chạy và dòng tổng kết nguyên văn

Build (đường dẫn mặc định của `--fw` không có trên máy này):

```
python fwsim/build_fwsim.py --fw D:/CLAUDE_THESIS/03_CODE/AGV_FIRMWARE
Built D:\HIL_WT\devin_c3c\fwsim\build\fwsim.dll
  ESP32 firmware 5bde4c5 dirty=False
  gcc (MinGW-W64 x86_64-ucrt-posix-seh, built by Brecht Sanders, r8) 13.2.0
```

Không có cảnh báo biên dịch (`fwsim/build/c3_evidence/build_final.txt`, phần STDERR rỗng).

Lần chạy cuối, code mới, sau mọi sửa đổi (`-q -rfEs`; đầu ra lưu ở `fwsim/build/c3_evidence/final_*.txt`):

| Lệnh | Dòng tổng kết nguyên văn |
|---|---|
| `python -m pytest -q -rfEs fwsim/tests/test_fwsim_kep.py` | `51 passed in 77.11s (0:01:17)` |
| `python -m pytest -q -rfEs fwsim/tests/test_fwsim.py` | `11 passed in 13.13s` |
| `python -m pytest -q -rfEs fwsim/tests/test_fwsim_pos.py` | `28 passed in 37.17s` |

Không test nào hỏng hay bị bỏ qua. `test_fwsim_kep.py`: 37 test cũ + 14 test mới.

File test mới trên code cũ (`fw_sim.c`, `fwsim.py` của bản đã commit; `kep_on_old_code.txt`): `4 failed, 47 passed in 57.92s`. Bốn test hỏng:

```
FAILED fwsim/tests/test_fwsim_kep.py::test_mutant_hold_is_flagged[reset_truoc_cap_nhat]
FAILED fwsim/tests/test_fwsim_kep.py::test_mutant_hold_is_flagged[bo_qua_khong_reset]
FAILED fwsim/tests/test_fwsim_kep.py::test_bit_exact_keys_definition - Attrib...
FAILED fwsim/tests/test_fwsim_kep.py::test_dll_pair_without_and_with_kep - as...
```

## 5. Điểm mơ hồ

1. **C3-12, chọn cách nào trong hai cách Codex nêu.** Codex nêu "lấy cờ ngay sau phép kẹp thật" **hoặc** "kiểm bản chép bằng dấu vết độc lập"; bảng thẩm định ghi cách hai. Tôi làm cách hai. Dấu vết chỉ lấy được bên trong bước, mà trong ba file của bộ C3c không có cách nào chặn lời gọi giữa `vel_ctrl.c` và `pid.c` của DLL chính (cần cờ liên kết, tức sửa `fwsim/build_fwsim.py`). Nên dấu vết nằm ở DLL của test. Hệ quả: đột biến `reset_roi_bo_qua` chỉ bị bắt **khi chạy test** trên cây firmware đó, không bị bắt trong một lượt S03 hay S04. Cần chốt: chấp nhận mức này, hay cho sửa `build_fwsim.py` để DLL chính lấy cờ ngay sau phép kẹp thật (thêm một cờ `-Wl,--wrap=…` và khoảng 15 dòng ở `fw_sim.c`; cách này đã chạy được trong DLL của test).
2. **"DLL cũ" của C3-13.** Luật cấm `git`, nên test không lấy được `fw_sim.c` của `62e2fa5`. Test dùng DLL build từ cùng cây với `-DFWSIM_NO_KEP`. Phép so với nguồn thật trước khi có bộ đếm làm tay, lấy file của worktree `devin_c3a` (mục 3.2); tôi không xác nhận được commit của file đó, chỉ biết nó không có mã `kep` và sha256 của nó.
3. **Mã băm cả log ở `s03_doe.py`, `s04_ga.py`.** `log_digest`, `npz_bytes`, `save_log_csv` và `s04_ga.log_sha` chạy trên mọi khóa, nên với DLL có bộ đếm chúng khác kết quả của DLL cũ ở mục `kep`. Tôi chỉ ghi định nghĩa và test; hai file đó ngoài phạm vi, không sửa. Nếu có chỗ so mã băm cả log với lượt chạy bằng DLL cũ thì phải so trên `BIT_EXACT_KEYS`.
4. **Dòng Codex dẫn `fwsim/s03_doe.py:1063–1067`** trong worktree này là `source_snapshot`, không phải đường lưu log; tôi hiểu ý là `npz_bytes` / `save_log` (dòng 917–928) và test theo đó.

## 6. Phần chưa làm và lý do

- **Lấy cờ ngay sau phép kẹp thật trong DLL chính:** cần sửa `fwsim/build_fwsim.py`, ngoài ba file của bộ C3c (điểm mơ hồ 1).
- `s03_doe.py`, `s04_ga.py`, `fwsim/README.md` không sửa (ngoài phạm vi).
- **Chưa kiểm:** `test_s03_doe.py`, `test_s04_ga.py` và mọi test ngoài ba file đề nêu. `test_fwsim_kep.py` nay `import s03_doe` (dùng `log_digest`, `npz_bytes`, `save_log_csv`) và `build_fwsim` (dùng `FW_INCLUDES`).
- **Chưa kiểm:** build với cây firmware không có `VEL_CTRL_HAS_BLOCK`.
- **Điều kiện của test mới:** 10 trong 14 test mới (các test dấu vết, đột biến và cặp DLL) cần `gcc`, cây `AGV_FIRMWARE` và `fwsim/build/fwsim_build.json`; thiếu thì tự bỏ qua, có ghi lý do. Ba test đột biến bỏ qua nếu đoạn mã neo trong `vel_ctrl.c` đổi. Mỗi lượt chạy build 5 DLL biến thể vào thư mục tạm của pytest; thời gian file test tăng từ `17.95s` (báo cáo vòng trước, `docs/devin/2026-11-22_BAO_CAO_C3c.md` dòng 76) lên `77.11s`.
- **File tạm còn lại**, đều trong `fwsim/build/` (đã có trong `.gitignore`), xóa được: `c3_old/` (bản trước khi sửa và `fwsim_prekep.dll`), `c3_new/` (bản sao file cuối), `c3_evidence/` (đầu ra các lệnh trên và `do_mau.py`).
- Ngoài worktree tôi chỉ **đọc** `D:/HIL_WT/devin_c3a/fwsim/fw_sim.c`, các file review, spec và cây firmware; không sửa, không tạo file nào ở đó.
