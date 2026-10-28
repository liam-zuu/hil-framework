# Ghi chú thiết kế: cách A, mô hình trượt bánh–thân (QĐ 25)

*Phiên CODE `local_40c6a22c…`, 28/10/2026. **Trạng thái: Liam duyệt 28/10 ở phiên CODE, nguyên văn "duyết tất cả" (Q1–Q7 theo khuyến nghị: Q3 = (a), Q6 giữ ngưỡng như bảng mục 6, Q7 nhánh `plant-slip-a`).** Các dấu [ĐX] dưới đây nay là đã duyệt. Căn cứ: Outline V7.4 QĐ 25 (Liam chốt 27/10 khuya: làm cả A và C), hồ sơ `00_OVERVIEW/2026-10-27_Ho_so_QD25_Mo_hinh_truot.md`, spec controller–IMU v2.2 §1b.4, §6.1. Nhãn: [ĐO] đã đo, [TÍNH] tính từ code/tham số hiện có, [UL] ước lượng, [GĐ] giả định, [ĐX] đề xuất chờ duyệt.*

## 1. A phải làm được gì

- Thân xe có chuyển động riêng (khối lượng M, quán tính Iz), **không bị buộc theo tốc độ bánh**. Khi lực kéo vượt giới hạn bám, bánh quay khác thân (trượt).
- Hệ quả mong muốn: gyro giả lập (đọc wz của thân) và encoder (đọc ω bánh) **lệch nhau một cách tự nhiên** khi trượt, để thử cờ nghi trượt và để DOE/GA có μ làm yếu tố nhiễu (Ch. 3 dòng 262, QĐ 27).
- Phạm vi tối thiểu, không hơn: lực kéo mỗi bánh bão hòa ở giới hạn bám; μ quét được; chế độ kê bánh giữ nguyên.
- **Không làm:** dồn tải trước/sau khi tăng tốc, ma sát lăn của con lăn, biến dạng lốp, mặt sàn không đều, trượt ngẫu nhiên.

## 2. Plant hiện tại (đọc code `ctrl-imu` `877e23c`; `plant_step.py` giống hệt `g2-bno-h7`)

- Thân tính thẳng từ bánh: `v_body = H_fwd @ omega_new` (`rpi5/plant_step.py:86`). Thân luôn khớp bánh, nên `r_true` mà fwsim ghi (`fwsim/fwsim.py:284`) luôn bằng 0.
- Khối lượng thân dồn vào bánh qua `M_eff = H_fwdᵀ·diag(M, M, Iz)·H_fwd + J` (`rpi5/params.py:190`).
- Đã có `SlipParams` (`params.py:32-39`, mặc định tắt) và `_apply_slip()` (`plant_step.py:142`): chỉ **cắt mô-men bánh** ở μ·N·r và thêm trượt ngẫu nhiên. Thân vẫn theo bánh. Chưa từng kiểm bằng dữ liệu.
- Chế độ kê bánh (`wheels_lifted`) không có thân; S01 và S02 P1 đều là dữ liệu kê bánh.
- Bản thử chỉ để đo thời gian: `_slip_a_step()` trong `rpi5/plant_timing.py:41`. Ghi chú này chính thức hóa bản đó.

## 3. Mô hình đề xuất [ĐX]

### 3.1 Trạng thái

Giữ **vector 10 phần tử như cũ** `[x, y, θ, vx_w, vy_w, wz, ω1..ω4]`. Khác biệt duy nhất: `vx_w, vy_w, wz` (phần tử 3–5) nay là **trạng thái độc lập**, được tích phân, không còn tính lại từ ω. Vì vậy `protocol.py`, `hil_node.plant_to_frame_state`, `imu_model.py`, H7 và ESP32 **không đổi**.

### 3.2 Phương trình (mỗi bánh i = FL, FR, RL, RR)

1. Tốc độ tiếp xúc quy về bánh: `ω_c = H_inv · v_b`, với `v_b` là vận tốc thân trong hệ thân (`v_w` quay đi −θ). Đây đúng là hàng thứ i của động học nghịch hiện có.
2. Mô-men kéo ở mặt tiếp xúc (Coulomb làm trơn):
   `τ_c,i = C_i · sat((ω_i − ω_c,i) / ω_s)`, với `sat(z) = max(−1, min(1, z))`, `C_i = μ_hd,i · N · r`, `N = M·g/4`.
3. Bánh: `J_i · ω̇_i = τ_m,i − τ_f,i − τ_c,i`. `τ_m` là `motor_torque()`, `τ_f` là ma sát motor S01, **giữ nguyên** từ `_dc_wheel_step`. Điều kiện bánh đứng yên (ma sát tĩnh) xét theo `|τ_m − τ_c| ≤ T_s` thay cho `|τ_m| ≤ T_s`.
4. Thân, tích phân **trong hệ thế giới** (để khỏi phải viết số hạng Coriolis):
   `F_b = H_invᵀ · τ_c` (gồm Fx, Fy, Mz), rồi `a_w = R(θ)·[Fx, Fy]/M` và `ẇz = Mz/Iz`.
5. Vị trí: tích phân như hiện nay (góc điểm giữa).

**Kiểm năng lượng** [TÍNH]: công suất bánh mất `τ_c·ω`, thân nhận `F_b·v_b = τ_c·ω_c`. Phần tiêu tán `τ_c·(ω − ω_c)` luôn ≥ 0 vì `τ_c` cùng dấu với `ω − ω_c`. Vậy mô hình không tự sinh năng lượng.

**Vì sao `H_invᵀ`** [TÍNH]: con lăn 45° chỉ truyền lực theo phương vuông góc trục con lăn; dọc trục con lăn thì lăn tự do. Chiếu lực đó về trục bánh và về thân cho đúng hàng i của `H_inv` (cùng gốc với động học đang dùng).

### 3.3 Ý nghĩa của μ trong mô hình: dùng thẳng μ_hd

- Với giới hạn `C_i = μ_hd·N·r`, xe đẩy thẳng vào lực kế cho lực lớn nhất `F = Σ C_i / r = μ_hd·M·g`. Đẩy ngang cũng vậy (đối xứng). [TÍNH] Nói cách khác, **tham số của mô hình chính là số đo `μ_hd = F_max/(m·g)`** mà buổi đo lực kế (spec v2.2 §1b.4) cho ra; không cần quy đổi.
- Kiểm bằng mô phỏng thô (scratchpad, không phải code repo): thân giữ bằng lò xo cứng, duty tăng dần 0 → 1 trong 4 s. Kết quả `F_max/(M·g)` = 0,319 (thẳng) / 0,322 (ngang) khi μ_hd = 0,3; 0,614 / 0,624 khi μ_hd = 0,6. Phần dư 2–7 % là dao động của lò xo lúc bắt đầu trượt, không phải sai công thức. Khi đo thật, lực kế lấy đỉnh cũng sẽ có phần động này.
- **Không dùng `mu_static = 0,8` trong `SlipParams` cho A.** Số đó là hệ số ma sát vật liệu ("dry concrete"). Với con lăn 45° lý tưởng, μ_hd ≈ μ_vật liệu/√2 [TÍNH, GĐ con lăn lý tưởng]. Trước buổi đo, A dùng một giá trị [GĐ] ghi rõ nhãn; sau buổi đo thì thay bằng số đo `05_DATA/results/MU_FLOOR/`.

### 3.4 Bỏ trượt ngẫu nhiên

Trên đường plant `dc` có A, **không dùng `_apply_slip()`**: bỏ trượt tự phát (`prob_spontaneous`) và nhiễu μ (`noise_sigma`). Lý do: không có số liệu, làm mô hình không tất định, và lẫn với cách C. Nếu DOE cần μ khác nhau giữa các bánh thì dùng `μ_hd,i` riêng từng bánh, đặt tất định. Mô hình mô-men cũ (`motor_model = 'torque'`) giữ nguyên để tra số M6 cũ.

### 3.5 Chế độ kê bánh

Không đổi một dòng nào: kê bánh thì không có tiếp xúc. Phát lại S01 và S02 P1 phải **trùng từng bit** với plant hiện tại (mục 6, T-A1).

### 3.6 Khác biệt với plant hiện tại khi μ rất lớn (cần biết trước)

- Khi μ rất lớn, A tiến tới ràng buộc cứng `ω = H_inv·v`: 4 bánh chỉ còn 3 bậc tự do.
- Plant hiện tại dùng giả nghịch đảo `H_fwd`, nên vẫn để bánh có **một kiểu "giằng nhau"**: hai bánh trước quay một chiều, hai bánh sau quay chiều kia (mẫu `[−1, −1, +1, +1]` [TÍNH]), mà thân không động. Kiểu này chỉ bị cản bởi quán tính bánh J.
- Trên xe thật, kiểu giằng nhau buộc phải trượt. A cản nó bằng lực bám; plant hiện tại thì không.
- Kiểu này bị kích khi 4 motor khác nhau (K, R, u0 khác nhau theo S01) hoặc controller ra lệnh lệch. Vì vậy **"μ lớn cho lại plant hiện tại" chỉ đúng gần đúng**, cần một ngưỡng chấp nhận (T-A2), không đòi trùng tuyệt đối.

## 4. Tích phân số

- Phần tiếp xúc rất cứng. Hằng số thời gian nhanh nhất là `ω_s / (C · λ_max)`, với `λ_max` = trị riêng lớn nhất của `J⁻¹ + H_inv·diag(M, M, Iz)⁻¹·H_invᵀ` ≈ 3000 kg⁻¹m⁻² [TÍNH].
- Số bước con tối thiểu để Euler hiện ổn định, trong 1 ms [TÍNH, tham số hiện tại]:

  | μ_hd | ω_s = 0,05 rad/s | 0,1 | 0,2 |
  |---|---|---|---|
  | 0,3 | 5 | 3 | 2 |
  | 0,6 | 9 | 5 | 3 |

  Thực tế lấy gấp khoảng 2 lần số này để có biên (và để T-A7 đạt).
- **Phương án 1 [ĐX, khuyến nghị làm trước]:** Euler bán ẩn với `n_sub` bước con cố định, vòng trong viết bằng số thực Python (không numpy). Đơn giản, dễ kiểm. Thời gian bản thử numpy trên PC [ĐO 27/10, bản không có trong repo]: 126 / 245 / 489 µs ở 5 / 10 / 20 bước con.
- **Phương án 2 (dự phòng):** Euler ẩn một bước mỗi 1 ms. Mỗi tổ hợp bánh đang bám/đang trượt (16 tổ hợp) có một ma trận 7×7 tính sẵn; đoán tổ hợp, giải, kiểm lại, lặp tối đa vài lần. Ổn định với mọi độ cứng, nhưng code và test phức tạp hơn.
- **Luật chọn [ĐX]:** khi có số `plant_timing.py` trên Pi, chọn phương án 1 nếu
  `p99(tick hiện tại) + p99(A ở n_sub chọn) − p99(plant hiện tại) + SPI + biên 200 µs ≤ 1000 µs`.
  Không đạt thì làm phương án 2. Ngưỡng H2 của S06 (THESIS chưa khóa) thay biên 200 µs nếu chặt hơn.
- **fwsim cũng chậm theo:** mỗi lượt fwsim chạy cùng `plant_step`, nên GA tốn thêm thời gian PC. Đo khi làm; chưa biết [?].

## 5. Tham số mới và chỗ sửa

**Tham số [ĐX]** (trong `params.py`, cùng nhãn nguồn như các dòng khác):

| Tên | Mặc định | Nhãn |
|---|---|---|
| `ground_model` | `'rigid'` (plant hiện tại) hoặc `'traction'` (A) | xem câu hỏi Q3 |
| `mu_hd` | 0,5 cho cả 4 bánh, nhận mảng 4 phần tử | `[ASSUMPTION — chờ MU_FLOOR]` |
| `slip_omega_s` | 0,1 rad/s | `[ASSUMPTION — độ trơn số học]`; kiểm độ nhạy ở T-A7 |
| `slip_nsub` | chọn theo mục 4 | `[calculated]` |

**File sửa:** `rpi5/plant_step.py` (nhánh A), `rpi5/params.py`, `rpi5/mecanum_plant.py` (`set_slip` đổi nghĩa), `rpi5/plant_timing.py` (đo hàm thật thay bản thử), `rpi5/run_hil.py` và `fwsim/fwsim.py` (tùy chọn `--ground`, `--mu`; ghi vào `meta.yaml`), `fwsim/s02_groupA_sim.py` (ghi tham số A vào meta), test mới `rpi5/tests/test_slip_a.py`.

**Không sửa:** `protocol.py`, frame SPI, `imu_model.py` (cách C vẫn là lớp bơm ngoài plant, đúng như THESIS khuyến nghị), H7, ESP32.

**MATLAB:** M6 chỉ còn tham khảo (từ 21/10). Đề xuất không port A sang MATLAB, chỉ thêm các dòng tham số vào `params_mecanum.m` kèm ghi chú "chỉ Python dùng" để giữ quy ước hai file cùng commit (Q5).

## 6. Kiểm trước khi dùng A cho số liệu [ĐX: ngưỡng khóa trước khi chạy]

| # | Phép kiểm | Đạt khi |
|---|---|---|
| T-A1 | Kê bánh: phát lại S01 (`tools/s01_replay_plant.py`) và chạy lại S02 P1 | Trùng từng bit với plant hiện tại |
| T-A2 | μ_hd = 5 (gần như không trượt): nhóm A trên fwsim, PID và LADRC, so với `'rigid'` (μ lớn làm tiếp xúc cứng hơn, `n_sub` tăng theo mục 4, khoảng 40; chỉ là phép kiểm nên chấp nhận chậm) | Mỗi metric `s02_metrics` lệch ≤ 1× `qrms` của metric đó (mức sàn lượng tử); ghi biên độ kiểu giằng nhau |
| T-A3 | Lực kế ảo (như mục 3.3), đẩy thẳng và ngang, 3 giá trị μ_hd | Lực tĩnh khi duty giữ không đổi sau trượt: `F/(M·g) = μ_hd` ± 1 % |
| T-A4 | Năng lượng: thả trôi (EN = 0) từ vận tốc đầu bất kỳ | Tổng động năng không tăng ở bất kỳ bước nào |
| T-A5 | Bắt đầu trượt: lệnh tăng tốc thẳng vượt μ_hd·g | Bánh trượt; gia tốc thân ≈ μ_hd·g (± 5 %); accelerometer giả lập bão hòa ở mức đó |
| T-A6 | Tất định | Hai lần chạy cùng đầu vào trùng từng bit; A không dùng RNG |
| T-A7 | Hội tụ số: `n_sub` và `2·n_sub`; `ω_s` và `ω_s/2` | Metric nhóm A lệch ≤ 0,5× `qrms` |
| T-A8 | Cờ nghi trượt có thật: μ_hd thấp (ví dụ 0,15), quay tại chỗ tăng tốc mạnh trong fwsim | `r_true ≠ 0` khi trượt; ghi lại, chỉ mô tả |

Sau khi có số liệu xe:
- **MU_FLOOR:** thay μ_hd [GĐ] bằng số đo, chạy lại T-A3.
- **Ca X3 (mặt trơn):** phát lại lệnh của X3 trong fwsim với A và μ_hd mặt trơn, so `r` đo với mô phỏng, chỉ mô tả (QĐ 25, V7).
- **Có thể (không bắt buộc):** log sàn 10/06 có cả `wz` encoder và `gz` IMU (`05_DATA/AGV_real_logs/2026-06-10/`), cho biết độ lệch encoder–IMU khi xe chạy thường. Lưu ý: đó là firmware tháng 6 (còn lỗi PCNT), nên chỉ dùng tham khảo.

## 7. Giới hạn phải ghi trong luận văn

- A chưa kiểm được động lực sau khi trượt; chỉ kiểm được mức lực bắt đầu trượt (MU_FLOOR) và một ca mặt trơn mô tả (X3).
- Bỏ qua dồn tải khi tăng tốc (chưa có độ cao trọng tâm), ma sát lăn của con lăn, mặt sàn không đều.
- Kết quả quét μ trong DOE là **độ nhạy của controller với một mô hình trượt đã kiểm một phần**, không phải dự báo cho xe thật.

## 8. Nhánh, thứ tự, phụ thuộc

- **Nhánh [ĐX]:** `plant-slip-a` tách từ `ctrl-imu` `877e23c` (fwsim có IMU, `r_true` nằm ở đó), worktree `D:\HIL_WT\hil_slip-a`. Thư mục chính giữ nguyên cho buổi G2. Gộp sau khi `ctrl-imu` lên `main`.
- **Thứ tự:** (1) mô hình + T-A3–T-A6 (không cần số Pi); (2) T-A1, T-A2, T-A7 trên fwsim; (3) có số Pi thì chốt `n_sub` hoặc chuyển phương án 2, viết vòng trong không numpy; (4) cập nhật `plant_timing.py` để Liam đo lại A thật trên Pi.
- **Phải chờ:** số `plant_timing` trên Pi (bước 3); MU_FLOOR (thay μ_hd); X3 (mô tả).
- **Giờ [UL]:** 11–19 h như hồ sơ QĐ 25. Theo bài học 27/10, số giờ Claude tự ước lượng có thể lệch nhiều lần, chỉ để tham khảo.
- **Phải chờ A xong:** thí nghiệm chọn tần số IMU trên fwsim, DOE/GA, mọi lượt fwsim/HIL lấy số cho luận văn. Hạn: trước khóa cấu hình 15/11.

## 9. Câu hỏi cho Liam

- **Q1. Dạng mô hình:** lực kéo bão hòa (Coulomb làm trơn) theo mục 3, thân tích phân riêng. *Khuyến nghị: duyệt.*
- **Q2. Tham số μ:** dùng thẳng μ_hd (đúng số lực kế đo), bỏ trượt ngẫu nhiên của `_apply_slip` trên đường `dc`. *Khuyến nghị: duyệt.*
- **Q3. Mặc định sau khi A đạt T-A1–T-A7:** (a) `ground_model = 'traction'` là mặc định cho fwsim và HIL, mọi số mới dùng A; hoặc (b) giữ `'rigid'` là mặc định, bật A bằng cờ. *Khuyến nghị (a)*, vì Outline ghi số liệu cho luận văn phải chạy trên A; HIL G3 trước khi A đạt vẫn dùng `'rigid'`.
- **Q4. Tích phân:** làm phương án 1 trước, chọn lại theo luật mục 4 khi có số Pi. *Khuyến nghị: duyệt.*
- **Q5. MATLAB:** không port A, chỉ thêm dòng tham số vào `params_mecanum.m`. *Khuyến nghị: duyệt.*
- **Q6. Ngưỡng kiểm mục 6:** khóa như bảng, hay sửa con số nào.
- **Q7. Nhánh và worktree** như mục 8, và có commit ghi chú này lên nhánh đó không.

## 10. Kết quả (28/10/2026, chỉ trên PC, chưa chạy trên Pi)

- **Code:** nhánh `plant-slip-a`. `4ee47ed` mô hình (`ground_model = 'traction'`), `01a5bc3` `plant_timing.py`, `8b454f4` `fwsim/slip_a_check.py`, `b039b09` tùy chọn `--ground`/`--mu`, `ed096d4` sửa plant rigid (xem dưới). Sau đó đổi mặc định sang `'traction'` (Q3 a).
- **T-A1, T-A3–T-A6** (`rpi5/tests/test_slip_a.py`): đạt. Lực kế ảo cho F/(M·g) = μ_hd ± 1 %, cả đẩy thẳng lẫn đẩy ngang, ở μ_hd = 0,2 / 0,5 / 0,8. Khi cả 4 bánh đều trượt thì không còn gì cản thân quay; đó là đặc tính của mô hình, không phải lỗi.
- **T-A2, T-A7** (`05_DATA/results/SLIP_A/`):
  - run01: T-A7 đạt 128/128 dòng; T-A2 trượt 2/128 dòng (A-F2, PID, `e_vy` 1,34× qrms).
  - run02_diag: tìm ra nguyên nhân. Plant `'rigid'` thiếu số hạng hệ quay `M·wz × v_thân`, ghi vào bug log M7 là `HIL-PLANT-ROTFRAME`, sửa ở `ed096d4`.
  - run03: T-A2 và T-A7 đều đạt 128/128 dòng.
- **Thời gian trên PC [ĐO]:** một bước A mất 49 µs (μ 0,5, 8 bước con) và 60 µs (μ 0,8, 12 bước con); plant cũ mất 45–55 µs. Trên Pi chưa đo.
- **Thông tin cho §10 spec controller–IMU** (test logic `fwsim/tests/test_fwsim_imu.py` nay ghim vào `'rigid'`, Liam duyệt 28/10). Trên `'traction'`, hai ca cho số khác do phản ứng động của plant, không do trượt (`r_true` < 0,001 rad/s):
  - mất gyro rồi có lại: cú nhảy PWM khi chuyển về IMU là 36, giới hạn [ĐỀ XUẤT] là max(1,5 × 22, 20) = 33;
  - bơm cách C với s = 0,3: cờ trượt tắt lúc 3,5 s, giới hạn là ≤ 3,26 s.
  
  Xét lại hai ngưỡng này khi khóa §10.
- **Còn lại:**
  - đo trên Pi bằng `plant_timing.py` của nhánh này (cần push), rồi áp luật ở mục 4;
  - thay μ_hd [GĐ] bằng số đo MU_FLOOR, chạy lại T-A3;
  - ca X3 (mặt trơn);
  - T-A8 (cờ nghi trượt với μ thấp, chỉ mô tả).
