# Đổi gain giữa mô phỏng (MATLAB M6 / plant HIL) và firmware ESP32

*Tạo 21/10/2026, phiên CODE. Tham số motor lấy từ spec S01 run01 (`05_DATA/results/S01/2026-10-20_run01/identification/`), giá trị trung bình hai chiều quay. R đo bằng đồng hồ, bất định khoảng ±15–25%, nên mọi hệ số có R đều mang bất định này.*

## 1. Hai cấu trúc điều khiển khác nhau (đọc trước)

| | Mô phỏng M6 (`esp32/*.m`) | Firmware ESP32 (`AGV_FIRMWARE/src/main.c`) |
|---|---|---|
| Vòng trong | PID hoặc LADRC **trên tốc độ từng bánh** ω [rad/s], xuất **mô-men** [N·m], rồi `pwm_output` đổi sang duty = τ/τ_max (τ_max = 0.5) | **Không có** vòng tốc độ bánh |
| Vòng ngoài | P/PI vị trí (từ tư thế ước lượng) | PI trên **vận tốc thân** (vx, vy, wz), cộng feedforward: `ctrl = v_ref + PID(v_ref, v_đo)` |
| Ra PWM | duty từ mô-men | động học nghịch → ω từng bánh → `pwm = ω / W_MAX · 1000`, W_MAX = 34.56 rad/s (`kinematics.c`), không có vùng chết |
| Đơn vị đo tốc độ | rad/s | RPM (`encoder_get_rpm`), đổi sang rad/s trước động học thuận |

Vì hai cấu trúc khác nhau, **không có phép đổi 1:1 cho mọi gain**. Phần 2 cho các phép đổi đơn vị đúng về vật lý. Phần 4 là đề xuất để bỏ hẳn việc phải đổi.

## 2. Đổi đơn vị

Mô hình motor (plant `motor_model = 'dc'`): `τ = (K/R)·(V·u_eff − K·ω)`, u = duty ∈ [−1, 1], `u_eff = sign(u)·max(|u| − u0, 0)`.

**Mô-men → duty** (tại một tốc độ ω, bỏ qua u0): `u = R·τ/(K·V) + K·ω/V`.
- Hệ số tăng ích **∂u/∂τ = R/(K·V)** [duty/(N·m)].
- `pwm_output.m` hiện dùng 1/τ_max = 2.0 duty/(N·m), tức gấp 1.8–2.3 lần giá trị thật (bảng dưới). Gain mô phỏng cũ vì vậy mạnh hơn khoảng 2 lần khi chạy trên motor thật.
- Số hạng `K·ω/V` là phần bù sức phản điện động: ở tốc độ cao cần thêm duty chỉ để giữ tốc độ.

**Tốc độ:** 1 rad/s = 60/(2π) ≈ 9.549 RPM. **PWM ESP32:** pwm = 1000 · duty.

| Bánh | K [V·s/rad] | R [Ω] | J [kg·m²] | R/(K·V) [duty/(N·m)] | K·V/R [N·m/duty, mô-men kẹt] | b0 duty [rad/s² mỗi duty] = K·V/(R·J) |
|---|---|---|---|---|---|---|
| FL | 0.304 | 3.31 | 0.00209 | 0.878 | 1.14 | 545 |
| FR | 0.322 | 3.38 | 0.00236 | 0.846 | 1.18 | 501 |
| RL | 0.309 | 3.73 | 0.00204 | 0.973 | 1.03 | 504 |
| RR | 0.288 | 3.95 | 0.00249 | 1.106 | 0.90 | 363 |

V = 12.4 V (trung bình pin trong S01 run01).

### PID tốc độ bánh: mô phỏng [N·m] → duty → PWM/RPM

```
Kp_duty = Kp_sim · R/(K·V)                [duty / (rad/s)]
Kp_esp  = Kp_duty · 1000 · 2π/60          [PWM / RPM]
Ki_esp  = Ki_sim · R/(K·V) · 1000 · 2π/60  [PWM / (RPM·s)]   (cùng dt tính bằng giây)
Kd_esp  = Kd_sim · R/(K·V) · 1000 · 2π/60  [PWM·s / RPM]
```

Ví dụ gain M6 hiện tại (Kp = 0.04, Ki = 0.5, Kd = 0.0004), bánh FL: Kp ≈ 3.68 PWM/RPM, Ki ≈ 46.0 PWM/(RPM·s), Kd ≈ 0.037 PWM·s/RPM. Các bánh khác khác tối đa khoảng 26% (RR).

### LADRC: b0

Trong mô phỏng `b0 = 1/J` [rad/s² mỗi N·m]. Nếu bộ điều khiển xuất duty thì `b0_duty = K·V/(R·J)` (cột cuối bảng). Sức phản điện động `−K²ω/(R·J)` khi đó là một phần "nhiễu tổng" mà ESO ước lượng. Có thể đưa nó vào mô hình danh định để ESO đỡ việc.

## 3. Feedforward PWM của firmware so với số đo

Firmware: `pwm = ω/34.56 · 1000`, tức duty = ω/34.56, không có vùng chết. Số đo S01 (`s01_params_per_wheel.csv`): ω_xác lập ≈ s·(|u| − u_int), với s = 36–43 rad/s mỗi duty và u_int = 0.045–0.115 tùy bánh và chiều. Map hiện tại thiếu duty ở tốc độ thấp (không bù vùng chết; bánh **không quay** khi |pwm| < 125–200) và gần đúng ở tốc độ cao. Map đo được: `duty = sign(ω)·(u_int + |ω|/s)` theo từng bánh và chiều. Lưu ý chiều: trong S01 run01, "pos" là PWM dương với `motor_dir = +1`, tức bánh quay lùi. Sau khi đặt `motor_dir = −1` (21/10), chiều tiến của bánh ứng với cột "neg" của run01.

## 4. Đề xuất (chưa làm, cần duyệt)

Để gain tune bằng GA (spec S04) chuyển thẳng sang xe mà không cần các phép đổi trên: thêm vào MATLAB một mô hình ESP32 **giống hệt firmware** (PI vận tốc thân + feedforward + map PWM), và cho GA tune đúng các tham số của firmware (kp/ki/kd cho vx, vy, wz; map feedforward), theo đơn vị của firmware.
