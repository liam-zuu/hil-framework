# Progress Update — M1 & M2 Complete

---

## Current Status
- Active milestone: M2 (complete)
- Completed: M1 (1.1→1.9), M2 (2.1→2.5)
- Blocked: none
- Next: M3 — Signal Conditioning

---

## M1 — Interface & Skeleton (Complete)

### Đã làm
- Tạo 19 files đúng layout: 14 modules + 4 scripts + 1 doc
- Định nghĩa state vector 10 states, sign convention, wheel numbering
- Định nghĩa I/O spec cho 14 modules (full signature + data types)
- Định nghĩa data flow giữa 3 clusters (ESP32 → H7 → RPi5)
- Ghi toàn bộ vào `docs/system_architecture.md`
- Viết `params_mecanum.m`, `trajectory_generator.m` (circle/line/square/figure8)
- Viết 14 stub functions với đúng signature và dummy output
- Viết `run_simulation.m` (main loop) + `plot_results.m`
- Chạy end-to-end verify data flow — pass

### Issues phát hiện và fix trong M1 review
1. **Iz = 0.05 bịa** → Fix: tính từ geometry = 0.0384 kg·m²
2. **PWM sign convention mâu thuẫn** → Fix: đồng bộ [-1, +1] signed convention
3. **Thiếu roller angle spec** → Fix: thêm wheel position table + X-config
4. **Thiếu forward kinematics** → Fix: thêm contract + verified Forward∘Inverse = Identity
5. **lx, ly comment sai** → Fix: longitudinal/lateral half-distance
6. **Thiếu gravity constant** → Fix: thêm params.g = 9.81 + IMU gravity note
7. **b_w, tau_max không có source** → Fix: flag [ASSUMPTION] rõ ràng

### Kinematics verification (5 test cases, all pass)
- Forward, strafe, rotation, diagonal, full combo
- Forward(Inverse(v)) = v, error < 1e-15

### MATLAB path issue
- `run('scripts/run_simulation.m')` khiến pwd = scripts/ → addpath sai
- Fix: dùng `mfilename('fullpath')` hoặc `restoredefaultpath` + `cd` trước khi chạy
- Khi mở project mới (M2, M3...) luôn chạy `restoredefaultpath` trước để tránh MATLAB load file cũ từ project khác

---

## M2 — Plant Model (Complete)

### Files thay đổi từ M1 → M2 (chỉ 3 files + 1 file bổ sung fields)

#### 1. `functions/rpi5/plant_step.m` — THAY HOÀN TOÀN
- Stub `x_new = x` → Full Lagrangian coupled dynamics
- Mô hình: M_eff * dω/dt = τ - b_w * ω (no-slip constraint)
- M_eff = (r/4)² * K' * diag(M,M,Iz) * K + J_w * I₄ (4×4 coupled inertia)
- Off-diagonal coupling ~29% of diagonal — wheels coupled qua body mass
- Body velocities = H_fwd * ω (forward kinematics, no-slip)
- Pose integration: semi-implicit Euler + midpoint rotation
- Theta normalization: mod(θ+π, 2π) - π

#### 2. `functions/rpi5/imu_model.m` — THAY HOÀN TOÀN
- Stub → Full IMU noise model
- True accel: dvx/dt - vy*wz (centripetal), dvy/dt + vx*wz, +g (gravity)
- True gyro: [0, 0, wz] (planar)
- Bias drift: random walk, bias(k) = bias(k-1) + √dt * drift * randn
- Measurement = true + bias + σ * randn

#### 3. `functions/rpi5/state_manager.m` — KHÔNG ĐỔI LOGIC
- Đã functional từ M1, chỉ xóa label "STUB"

#### 4. `scripts/params_mecanum.m` — BỔ SUNG FIELDS
Thêm vào cuối file (trước `end`):
- `params.g = 9.81`
- `params.H_fwd` (3×4 forward kinematics matrix)
- `params.H_inv` (4×3 inverse kinematics matrix)
- `params.M_eff` (4×4 effective inertia matrix)
- `params.M_eff_inv` (precomputed inverse)

### Open-loop test results (6/6 pass)
1. Forward: τ=[0.1,0.1,0.1,0.1] → xe đi thẳng +X ✓
2. CCW rotation: τ=[-0.1,0.1,-0.1,0.1] → quay CCW, x≈y≈0 ✓
3. Left strafe: τ=[-0.1,0.1,0.1,-0.1] → xe trượt ngang +Y ✓
4. Diagonal: τ=[0.05,0.15,0.15,0.05] → chéo forward+left ✓
5. Steady-state: ω saturate tại omega_max = 34.56 rad/s ✓
6. Friction decay: ω(5s) = 3.55 từ 10.0, khớp time constant J_eff/b_w = 4.82s ✓

### End-to-end test results (circle, P-only stub controller)
- Data flow: no NaN/Inf ✓
- Robot di chuyển, wheels quay ✓
- XY trajectory: hình tròn nhưng lớn hơn reference (R≈0.7 vs 0.5m)
- vx ≈ 1.25 m/s (quá nhanh, do P-only không đủ)
- RMS position error: ~1178mm (expected — P-only controller)
- Heading wrap [-π,π] hoạt động đúng
- Encoder quantization gây oscillation trong wheel speed estimates → torque oscillation
- Tất cả issues trên đều expected cho M2, sẽ fix ở M3 (signal conditioning) và M4 (full controller)

---

## Key Design Decisions (tham khảo cho M3+)

1. **Coupled dynamics**: Dùng Lagrangian M_eff, không phải 4 wheel độc lập. Đúng vật lý hơn.
2. **No-slip constraint**: Body velocities là hàm kinematics của wheel speeds, không phải states độc lập. Slip sẽ thêm ở M6.
3. **Semi-implicit Euler**: Tính velocity trước, pose sau. Stable tại dt=0.001.
4. **PWM signed [-1,+1]**: Motor 2 chiều, convention nhất quán từ pwm_output → pwm_capture → plant.
5. **Encoder quantization**: Tại tốc độ thấp, counts/timestep rất ít (1-2) → quantization error lớn. M3 sẽ address.
6. **imu_model dùng finite difference**: dvx/dt = (vx - vx_prev)/dt. Chấp nhận được vì dt nhỏ.

---

## Còn lại stub (chờ M3, M4)

### Nucleo H7 (M3 — Signal Conditioning):
- spi_interface.m — passthrough, chưa quantization
- encoder_pulse_gen.m — basic round(), chưa noise
- imu_packet_enc.m — passthrough, chưa encode thật
- pwm_capture.m — linear, chưa deadband
- gpio_sync.m — always true

### ESP32 (M4 — Controller):
- encoder_reader.m — basic decode, chưa filtering
- imu_reader.m — passthrough, chưa checksum verify
- pid_controller.m — P-only, chưa full PID
- adrc_controller.m — P-only placeholder, chưa ESO
- pwm_output.m — basic saturation, chưa deadband
- slip_detector.m — always false (optional, M6)
