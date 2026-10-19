# Session Review Summary — Code Review & Strategic Clarification

**Ngày:** 2026-04-17  
**Nội dung:** Review toàn bộ codebase SIMU_M6_V3 + 14 scenarios, làm rõ định hướng thesis và các vấn đề kỹ thuật cần lưu ý.

---

## 1. Định hướng thesis — Clarification quan trọng nhất

**Thesis là Hướng A: HIL framework là contribution chính.**

Plant model HIL là một công cụ test độc lập với thuật toán điều khiển. Nó nhận torque τ vào, trả ra state x[k], và không biết τ đến từ PID, ADRC, MPC, hay bất kỳ thuật toán nào. Đây là đặc tính cốt lõi của HIL đúng nghĩa.

PID và ADRC trong project hiện tại đóng vai trò **case study để validate framework**, không phải contribution chính. Chúng được dùng để:
- Chứng minh HIL chạy được trong vòng lặp kín (closed-loop)
- Phát hiện các vấn đề plant model (quantization, timing, slip)
- Demonstrate rằng framework phản ánh đúng behavior khi có disturbance

**Không nên làm thêm controller khác.** Thêm MPC, fuzzy, hay LQR không tăng giá trị thesis Hướng A. Hai controller đã đủ để demonstrate framework — một classical (PID) và một modern disturbance-rejection (ADRC).

**Phần còn thiếu là framing và argument, không phải thêm experiment.**

---

## 2. Cấu trúc project — Điểm mạnh

- **Ba-cluster separation** (`rpi5/`, `nucleoh7/`, `esp32/`) khớp với ranh giới phần cứng thật. Ranh giới kiến trúc này được làm đúng và defensible.
- **`params_mecanum.m` là single source of truth.** Tất cả 16 modules nhận `params` struct, không hardcode giá trị.
- **Scenarios layer không xâm phạm code gốc.** 14 scenarios chạy trên 16 modules mà không sửa một dòng nào.
- **Plant validation 49/49 tests pass**, được tổ chức thành 5 tầng độc lập (kinematics → dynamics → integration → ode45 cross-check → literature comparison). Phần này có thể defend ngay với advisor.

---

## 3. Cấu trúc project — Vấn đề kỹ thuật cần biết

### 3.1 `scripts/` folder bị overloaded

Hiện tại chứa 20 files thuộc 5-6 loại mục đích khác nhau không được phân loại:

| Loại | Files |
|------|-------|
| Core config | `params_mecanum.m` |
| Core runners | `run_simulation.m`, `trajectory_generator.m`, `plot_results.m` |
| Experiment runners | `run_m5_comparison.m`, `run_single_scenario.m`, `run_m6_disturbance.m` |
| Diagnostic tools | `tune_gains.m`, `diagnose_error_sources.m`, `diagnose_remaining_error.m` |
| Unit tests | `test_m3_*.m`, `test_m4_*.m`, `test_m5_*.m`, `test_m6_*.m` |
| Validation | `validate_kinematics.m`, `validate_dynamics.m`, `validate_integrator.m`, `validate_cross_ode45.m`, `validate_literature_comparison.m` |
| Utility | `setfields.m` |

Nếu project tiếp tục, nên tách thành `scripts/core/`, `scripts/tests/`, `scripts/validation/`, `scripts/diagnostics/`.

### 3.2 Version proliferation — v1 và v2 song song

Hai trajectory generator tồn tại song song:
- `scripts/trajectory_generator.m` — dùng bởi `run_simulation.m`
- `scenarios/trajectory_generator_v2.m` — dùng bởi tất cả 14 scenarios

Hai scenario runners song song:
- `scripts/run_single_scenario.m`
- `scenarios/run_single_scenario_v2.m`

**Rủi ro:** Nếu sửa logic vật lý trong v1 mà quên cập nhật v2 (hoặc ngược lại), kết quả hai bộ sẽ không tương thích mà không có warning nào. Cần document rõ cái nào dùng cho mục đích gì, hoặc gộp lại.

**Sự khác biệt thực chất giữa v1 và v2:**
- v2 dùng struct `spec` thay vì tham số rời rạc
- v2 có **time warping** — giữ đúng hình dạng path (radius circle không bị méo) trong khi ramp vận tốc từ 0 lên full speed
- v2 có thêm trajectory types: zigzag, sinusoidal, racetrack, rounded_square

### 3.3 `persistent` variables — cần `clear` thủ công, không có warning

Ba modules dùng `persistent` variables:
- `encoder_pulse_gen.m` (fractional accumulator)
- `encoder_reader.m` (IIR filter state)
- `imu_reader.m` (outlier rejection + filter state)

**Vấn đề:** Nếu chạy simulation, dừng giữa chừng, rồi chạy lại mà không clear workspace, các persistent variables giữ state từ lần chạy trước. Kết quả simulation sẽ sai nhưng MATLAB không đưa ra bất kỳ error hay warning nào — hoàn toàn im lặng.

**Cách xử lý hiện tại:** `run_simulation.m` đã thêm các lệnh `clear encoder_pulse_gen`, `clear encoder_reader`, `clear imu_reader` ở đầu file. Tuy nhiên nếu ai gọi các modules này từ một script khác (test script mới, scenario mới), rất dễ quên và ra kết quả sai.

**Lưu ý cho tương lai:** Bất kỳ runner script mới nào cũng cần có 3 dòng clear này ở đầu. Đây là điểm fragile nhất của codebase hiện tại.

### 3.4 Double nesting trong ZIP

Cấu trúc thực tế trong SIMU_M6_V3.zip là `scenarios/scenarios/` (double nesting). Có thể là lỗi khi copy folder. Cần kiểm tra MATLAB path khi deploy để tránh nhầm lẫn.

---

## 4. Kết quả 14 scenarios — Cách đọc đúng

### 4.1 Kết quả tổng quan

```
Overall: ADRC wins=2, PID wins=2, ties=10 (out of 14)
Mean improvement: +5.3%
Max improvement: +75.9% (s14_industrial_nightmare)
```

### 4.2 Tại sao ADRC không thắng nhiều ở V3

Đây **không phải vấn đề về ADRC hay plant model**. Lý do nằm ở thiết kế thí nghiệm:

12/14 scenarios chạy ở **nominal conditions** — không có disturbance lớn, không có fault. Ở điều kiện này, ESO của ADRC liên tục estimate disturbance z2 dù không có disturbance thật, tạo ra một lượng nhỏ torque sai so với PID (gọi là ESO injection noise). PID ở nominal thì đơn giản hơn, ít moving parts, ít noise injection.

Có một **crossover point**: dưới ngưỡng disturbance nhất định, PID tốt hơn hoặc ngang; trên ngưỡng đó, ADRC vượt trội và advantage tăng nhanh. M6 test ở vùng trên crossover (PPR 256, wheel slip, worst case). V3 nominal test ở vùng dưới crossover. Đây là lý do M6 cho ADRC thắng 77%, V3 cho 14%.

Điều này **consistent với lý thuyết ADRC** — Han Jingqing (2009) và Gao Zhiqiang đều thừa nhận ADRC không được thiết kế để tối ưu nominal performance, mà cho uncertain và disturbed systems.

### 4.3 Cách present với advisor

Không nên show 14 scenarios một mình. Nên present theo cấu trúc:
1. **Plant validation (49 tests)** — establish credibility
2. **M6 disturbance analysis (12 conditions × N=5 trials)** — nơi ADRC advantage rõ và có statistical backing
3. **14 scenarios** — comprehensive trajectory benchmark để show framework chạy được trên nhiều loại đường

Nếu advisor hỏi tại sao ADRC không thắng hầu hết 14 scenarios, câu trả lời là: "nominal conditions, ADRC advantage chỉ xuất hiện khi có model uncertainty — đúng với lý thuyết, confirmed trong M6." Đây còn thể hiện hiểu sâu hơn là nếu ADRC thắng tất cả.

### 4.4 Một vấn đề cần giải thích

s03 (circle R=1m) cho 19mm SS error, s04 (circle R=2m) cho 71mm — trong khi M5.2 circle R=0.5m chỉ 7mm. Circle lớn hơn (curvature thấp hơn) lại cho error lớn hơn gần 10×. Nguyên nhân: dead reckoning drift tích lũy tuyến tính theo thời gian. s04 chạy T_sim=60s, dài gấp 6× so với M6 T_sim=10s, nên drift lớn hơn nhiều dù controller hoàn hảo. Cần explain rõ điều này khi present.

---

## 5. Gap còn thiếu cho thesis Hướng A

### 5.1 Chưa có baseline so sánh (M7 chưa làm)

Câu hỏi chưa được trả lời: "HIL này tiết kiệm được gì so với không dùng HIL?" Đây là M7 trong plan ban đầu — Process Metrics Framework. Nếu bỏ qua M7, thesis Hướng A thiếu argument quan trọng nhất.

### 5.2 Chưa có deployment story

HIL có giá trị thực tế khi nó giúp developer phát hiện vấn đề *trước* khi deploy lên robot thật. Data từ 14 scenarios (đặc biệt fault scenarios s11-s14) có thể được dùng để argue điều này — nhưng framing hiện tại chưa làm rõ connection đó.

### 5.3 Hướng đi tiếp theo được đề xuất

Không làm thêm simulation hay controller mới. Thay vào đó:

1. **M7** — Define và measure HIL value metrics dùng data đã có từ M1-M6
2. **Commissioning guide ngắn** — Nếu developer khác dùng framework này cho controller của họ, họ cần làm gì. Chứng minh framework có tính reusable.
3. **Viết thesis chapters** — Framing và argument là phần còn thiếu, không phải thêm experiment.

---

## 6. Tóm tắt một câu cho mỗi vấn đề

| Vấn đề | Tóm tắt |
|--------|---------|
| Định hướng thesis | HIL framework là contribution chính; PID/ADRC chỉ là case study validate |
| Không làm thêm controller | Thêm controller không tăng giá trị Hướng A |
| `scripts/` overloaded | 20 files, 6 loại, không subfolder — cần phân loại nếu project tiếp tục |
| Version proliferation | v1 và v2 song song, nguy cơ inconsistency khi sửa một mà quên cái kia |
| `persistent` variables | Không clear trước khi chạy lại → kết quả sai hoàn toàn, không có warning |
| Double nesting | `scenarios/scenarios/` trong ZIP — kiểm tra MATLAB path |
| 14 scenarios ADRC wins=2 | Nominal conditions, dưới crossover point — đúng lý thuyết, không phải bug |
| s03/s04 error lớn | Dead reckoning drift tích lũy theo T_sim dài, không phải plant issue |
| Gap M7 | Chưa có argument về HIL value — phần quan trọng nhất còn thiếu |
