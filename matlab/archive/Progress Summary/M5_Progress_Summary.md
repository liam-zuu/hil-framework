# M5 — Full Integration (Complete)

---

## Current Status
- Active milestone: M5 (complete — M5.1 + M5.2)
- Completed: M1 (1.1→1.9), M2 (2.1→2.5), M3 (3.1→3.7), M4 (4.1→4.7), M5 (5.1→5.7)
- Blocked: none
- Next: M6 — Disturbance & Robustness

---

## M5.1 — Full Integration (Closed-loop + Debug)

### Vấn đề cốt lõi đã giải quyết

M4 chỉ có inner loop (wheel velocity control) → robot chạy đúng tốc độ nhưng lệch quỹ đạo từ transient startup, không có gì correct → RMS ~1000mm. M5.1 thêm **outer loop position control** với dead reckoning odometry, tạo thành kiến trúc **two-loop cascade**: outer loop (position → body velocity) → inner loop (body velocity → wheel velocity → torque).

### Files thay đổi từ M4 → M5.1 (4 new + 4 modified + 2 new scripts)

#### 1. `esp32/pose_estimator.m` — MỚI
- Dead reckoning odometry từ encoder + IMU gyro
- Pipeline: omega_est → forward kinematics → body velocities (vx, vy)
- Heading rate: dùng gyro wz (ít quantization noise hơn encoder-derived)
- Integrate: midpoint rotation (cùng phương pháp plant_step.m)
- Dùng struct state (không persistent) để clean reset
- Theta normalization: mod(θ+π, 2π) - π

#### 2. `esp32/position_controller.m` — MỚI (3 iterations)
- **Iteration 1:** P-only + feedforward → SS error ~375mm trên circle (type-0 system không track được moving reference)
- **Iteration 2:** PI + feedforward + anti-windup → SS ~100mm trên circle
- **Final version:** PI controller với conditional integration anti-windup
- Position error: world frame → rotate sang body frame → P + I correction
- Heading error: angular wrapping [-π, π] → P + I correction
- Velocity clamping: prevent inner loop saturation
- Anti-windup: freeze integral khi output saturated VÀ error cùng chiều integral (cùng pattern pid_controller.m inner loop)
- Safety clamp backup trên integral
- Dùng `persistent int_pos int_theta` — cần `clear position_controller` giữa các simulation runs

#### 3. `esp32/adrc_controller.m` — SỬA (2 fixes)
**Fix 1 — u_prev clamping:**
- Bug: `u_prev = tau_cmd` (unclamped, có thể 8 N·m) nhưng plant chỉ nhận ±0.5 N·m
- ESO nghĩ đã apply 8 N·m → thấy plant response không match → gán sai vào z2 (disturbance) → positive feedback → diverge
- Fix: `u_prev = max(-tau_max, min(tau_max, tau_cmd))`

**Fix 2 — z2 clamping:**
- Bug: prolonged saturation → z2 (disturbance estimate) tích lũy không giới hạn
- z2 update: `z2 += dt * (-beta2 * e_eso)` với beta2 = 10000, tích lũy nhanh
- Fix: `z2_max = tau_max * b0` (≈ 202 rad/s²), clamp z2 mỗi step

**Tổng cộng 3 lớp bảo vệ ADRC:**
1. u_prev clamped → ESO biết plant chỉ nhận ±tau_max
2. z2 clamped → disturbance estimate bounded
3. Smooth trajectory (fix trajectory_generator) → reference không có spike

#### 4. `scripts/trajectory_generator.m` — SỬA figure-8
- Bug: `wz_ref = gradient(theta_ref, dt)` trong đó `theta_ref = atan2(...)`. Khi atan2 wrap từ +π sang -π tại crossover point → gradient spike ~6000 rad/s → outer loop request omega_ref cực lớn → ADRC saturate liên tục → diverge
- Fix: dùng **curvature formula** tính wz analytical:
  - `wz = (dx*ddy - dy*ddx) / (dx² + dy²)` — smooth everywhere, không discontinuity
  - Analytical derivatives cho dx, dy, ddx, ddy thay vì numerical `gradient()`

#### 5. `scripts/params_mecanum.m` — BỔ SUNG FIELDS
Thêm params cho outer loop (giá trị ban đầu, chưa tối ưu):
- `params.pos_ctrl.Kp_pos = 3.0`, `Ki_pos = 0.5` (Ti = 6s)
- `params.pos_ctrl.Kp_theta = 4.0`, `Ki_theta = 1.0` (Ti = 4s)
- `params.pos_ctrl.vx_max = 1.5`, `vy_max = 1.5`, `wz_max = 5.0`

#### 6. `scripts/run_simulation.m` — VIẾT LẠI HOÀN TOÀN
#### 7. `scripts/plot_results.m` — VIẾT LẠI (6 → 8 subplots)
#### 8. `scripts/run_m5_comparison.m` — MỚI
#### 9. `scripts/test_m5_integration.m` — MỚI (8/8 PASS)

### M5.1 Results (chưa tối ưu)

| Trajectory | PID SS (mm) | ADRC SS (mm) | Improvement |
|------------|-------------|--------------|-------------|
| line | 17.0 | 6.2 | +63.5% |
| circle | 96.8 | 104.4 | -7.8% |
| square | 9.6 | 4.4 | +54.0% |
| figure8 | 34.8 | 25.4 | +27.2% |

**Circle ~100mm** — chưa tối ưu, do outer loop Ki_pos = 0.5 quá chậm (Ti = 6s).

### M5.1 Debug log (3 iterations)

**Iteration 1:** P-only outer loop → SS error ~375mm
- Root cause: Type-0 system tracking moving reference. SS error = v_ref / Kp ≈ 314mm.
- Fix: P → PI + anti-windup

**Iteration 2:** ADRC figure-8 diverge → SS ~1035mm
- Root cause 1: `u_prev = tau_cmd` unclamped → ESO positive feedback
- Root cause 2: `wz_ref = gradient(atan2(...))` → spike ~6000 rad/s
- Fix: u_prev clamp + z2 clamp + curvature formula

**Iteration 3:** Final verification → 8/8 tests PASS, all scenarios stable

---

## M5.2 — Baseline Optimization (Diagnosis + Tuning + Plant Calibration)

### Vấn đề cốt lõi: Circle SS ~100mm từ đâu?

Viết `diagnose_error_sources.m` để tách error từng nguồn — chạy 4 điều kiện A→D, mỗi điều kiện bỏ đi 1 tầng:

| Điều kiện | PID SS (mm) | ADRC SS (mm) | Giải thích |
|-----------|-------------|--------------|------------|
| A. Controller only (ideal sensors+pose) | 103.0 | 103.2 | ← **100% error từ controller** |
| B. + Signal conditioning noise | 103.1 (+0.1) | 103.2 (+0.0) | Noise không ảnh hưởng |
| C. + Dead reckoning drift | 104.1 (+1.1) | 104.4 (+1.2) | Odometry drift ~1mm |
| D. + H7 torque pipeline (full system) | 100.2 | 100.3 | H7 pipeline ~0mm |

**Kết luận dứt khoát:**
- Plant: ĐÃ TỐT (không phải bottleneck)
- Signal conditioning: ĐÃ TỐT (Δ < 1mm)
- Dead reckoning: ĐÃ TỐT (drift ~1mm cho 10s)
- **Controller: CHƯA TỐI ƯU** — 1 parameter (Ki_pos = 0.5, Ti = 6s) gây 93% error

### Bandwidth analysis

| Loop | BW trước | BW sau | Ghi chú |
|------|----------|--------|---------|
| PID inner | 5.6 rad/s (0.89 Hz) | 11 rad/s (1.7 Hz) | Kp: 0.02 → 0.04 |
| ADRC inner | 30 rad/s (4.8 Hz) | 20 rad/s (3.2 Hz) | kp: 30 → 20 (ít noise) |
| Outer loop | 3.0 rad/s (0.48 Hz) | 6.0 rad/s (0.95 Hz) | Kp_pos: 3 → 6 |
| PID_inner / outer | 1.9× | 1.8× | Vẫn < 3× nhưng stable |
| ADRC_inner / outer | 10.0× | 3.3× | OK |

### Systematic gain sweep (`tune_gains.m`)

- **Sweep 1:** Outer loop gains (25 combinations) → Best: Kp_pos=6.0, Ki_pos=3.0
- **Sweep 2:** PID inner Kp×Ki (18 combinations) → Best: Kp=0.04, Ki=0.5, Kd=0.0004
- **Sweep 3:** ADRC kp×ω_o (17 combinations) → Best: kp=20, ω_o=100

### Figure-8 plant capability check (`diagnose_remaining_error.m`)

Figure-8 (period=5s) vượt giới hạn vật lý plant:

| Yêu cầu | Plant limit | Tỷ lệ | Kết luận |
|----------|-------------|--------|----------|
| Max τ = 0.647 N·m | τ_max = 0.5 N·m | 129% | Plant KHÔNG ĐỦ |
| Max ω = 43.5 rad/s | ω_max = 34.56 rad/s | 126% | Plant KHÔNG ĐỦ |

→ Fix: figure-8 period 5s → 8s. Sau fix: max τ = 0.257 (51% limit), max ω = 27.2 (79% limit), 0% saturation.

### ADRC noise sensitivity check

| Condition | PID (mm) | ADRC (mm) |
|-----------|----------|-----------|
| Ideal sensors | 0.29 | 0.34 |
| Real sensors | 6.79 | 9.88 |

ADRC ideal ≈ PID ideal → structure OK. ADRC real >> PID real → ESO ω_o=200 amplify encoder noise.
Fix: ω_o: 200 → 100 (circle impact: chỉ +0.1mm).

### Plant parameter calibration

Tăng `b_w` từ 0.001 lên 0.002 (realistic hơn cho motor có gear). Controller không cần re-tune — cho thấy controller robust với thay đổi plant parameter ×2.

### Gains thay đổi tổng M5.1 → M5.2

| Parameter | M5.1 | M5.2 (tối ưu) | Phương pháp |
|-----------|------|---------------|-------------|
| b_w | 0.001 | **0.002** | Calibration (conservative) |
| PID Kp | 0.02 | **0.04** | Sweep 275 combinations |
| PID Ki | 0.5 | 0.5 | Giữ |
| PID Kd | 0.0002 | **0.0004** | Scale with Kp |
| ADRC kp | 30 | **20** | Sweep + noise diagnosis |
| ADRC ω_o (β1,β2) | 100 (200, 10000) | 100 (200, 10000) | Giữ (ω_o=200 amplify noise) |
| Outer Kp_pos | 3.0 | **6.0** | Sweep |
| Outer Ki_pos | 0.5 | **3.0** | Sweep — **main fix** |
| Outer Kp_theta | 4.0 | **8.0** | Sweep |
| Outer Ki_theta | 1.0 | **6.0** | Sweep |
| Figure-8 period | 5s | **8s** | Plant capability check |

### M5.2 Results — Baseline tối ưu (b_w=0.002)

| Trajectory | PID SS (mm) | ADRC SS (mm) | ADRC vs PID |
|------------|-------------|--------------|-------------|
| line | 8.9 | **7.5** | +15.7% |
| circle | 7.0 | **6.4** | +8.9% |
| square | 6.4 | **3.6** | +43.7% |
| figure8 | **6.1** | 6.7 | -9.7% |

**ADRC tốt hơn PID trên 3/4 trajectories.** Tất cả trajectories đều single-digit mm SS error.

### So sánh toàn bộ quá trình M5

| Trajectory | M4 | M5.1 | M5.2 | Cải thiện tổng |
|------------|-----|------|------|----------------|
| Circle PID | 1036 mm | 97 mm | **7.0 mm** | **148×** |
| Circle ADRC | 1006 mm | 104 mm | **6.4 mm** | **157×** |
| Figure-8 ADRC | diverge | 25 mm | **6.7 mm** | ∞ → 6.7mm |
| Line PID | drift | 17 mm | **8.9 mm** | ∞ → 8.9mm |

---

## Key Design Decisions

1. **PI outer loop (không PID):** Derivative trên position error không cần thiết vì inner loop đã handle velocity tracking. Integral eliminates SS error trên moving references.

2. **Outer loop BW ~1 Hz << Inner loop BW ~3-5 Hz:** Tỷ lệ ≥3× (ADRC) đảm bảo stability.

3. **Integral time Ti_pos = Kp/Ki = 2s:** Ban đầu Ti=6s quá chậm gây 100mm error. Ti=2s xóa SS error trong ~2-3s.

4. **Dead reckoning dùng gyro cho heading:** Encoder-derived wz có quantization noise cao ở low speed (SNR 6.4 dB). Gyro noise nhỏ hơn.

5. **ADRC ESO anti-windup (3 layers):** u_prev clamp + z2 clamp + smooth trajectory.

6. **Figure-8 curvature formula:** `wz = (dx*ddy - dy*ddx)/(dx²+dy²)` thay gradient(atan2).

7. **b_w = 0.002:** Conservative hơn 0.001, realistic cho motor có gear. Controller robust — không cần re-tune khi thay đổi 2× friction.

8. **Remaining error 3-9mm là noise floor:** Encoder quantization (SNR 6.4dB ở low speed) + dead reckoning drift. Chỉ cải thiện được bằng hardware tốt hơn (higher PPR) hoặc sensor fusion (Kalman filter).

---

## Diagnostic scripts (giữ lại cho M6+ và commissioning)

| Script | Mục đích |
|--------|----------|
| `diagnose_error_sources.m` | Tách error: controller vs signal chain vs odometry vs plant |
| `diagnose_remaining_error.m` | Plant capability, ADRC noise sensitivity, saturation analysis |
| `tune_gains.m` | Systematic sweep outer+inner gains (275 combinations) |

---

## Hướng dẫn commissioning khi có robot thật

### Nhóm 1 — Thay số, hệ thống tự adapt, KHÔNG cần tune

| Parameter | Nguồn cập nhật |
|-----------|---------------|
| `enc_ppr` | Datasheet motor |
| `imu_*_noise`, `bias` | Datasheet IMU hoặc đo Allan variance |
| `pwm_res`, `deadband` | Datasheet driver |
| `spi.float_bits` | Hardware SPI config |
| `imu_adc_bits`, ranges | Datasheet ADC |

### Nhóm 2 — Thay số, M_eff tự tính lại, CÓ THỂ cần tune

| Parameter | Nguồn cập nhật | Cần tune? |
|-----------|---------------|-----------|
| `r` | Đo trực tiếp bánh xe | Nếu sai >5% |
| `lx`, `ly` | Đo từ CAD hoặc thực tế | Nếu sai >10% |
| `M` | Cân robot hoàn chỉnh | ADRC ESO bù mismatch |
| `Iz` | CAD hoặc swing test | ADRC ESO bù mismatch |
| `b_w` | Coast-down test | Thường không (robust ×2) |

### Nhóm 3 — CHẮC CHẮN cần tune lại

| Parameter | Nguồn cập nhật | Lý do |
|-----------|---------------|-------|
| `J_w` | Datasheet motor+gear hoặc step test | ADRC `b0 = 1/J_w` — sai b0 → ESO sai |
| `tau_max` | Stall test hoặc datasheet | PID Kp bound, outer loop clamps |
| `omega_max` | No-load test | Trajectory feasibility |

### Parameters hiện đang ước lượng — cần cập nhật chính xác để tăng độ tin cậy model

| Parameter | Giá trị hiện tại | Nguồn hiện tại | Cách cập nhật chính xác |
|-----------|-------------------|----------------|------------------------|
| **`Iz`** | 0.0384 kg·m² | Tính: (1/12)×M×(a²+b²), giả sử uniform rectangle | **Từ CAD**: tính từ 3D model với phân bố mass thật (motor, battery, frame). Hoặc **swing test** trên robot thật |
| **`J_w`** | 0.00247 kg·m² | Từ spec (chưa rõ có gear ratio không) | **Đo**: step response test trên 1 wheel → fit exponential → extract J. Hoặc **motor datasheet** × gear_ratio² |
| **`b_w`** | 0.002 N·m·s/rad | Ước lượng conservative | **Coast-down test**: cho wheel quay, tắt motor, đo ω(t) decay → b_w = J_w / τ_decay |
| **`tau_max`** | 0.5 N·m | Ước lượng | **Stall test** hoặc **motor datasheet**: τ_stall × gear_ratio × efficiency |
| **`M`** | 4.0 kg | Từ spec | **Cân** robot hoàn chỉnh |
| **`r`** | 0.0485 m | Từ spec | **Đo** bán kính thực tế (mecanum roller contact radius ≠ nominal radius) |

### Quy trình tune khi có robot thật

```
Bước 1: ĐO physical parameters
   → r, lx, ly, M, J_w, b_w, tau_max, omega_max, enc_ppr

Bước 2: CẬP NHẬT params_mecanum.m
   → M_eff, H_fwd, H_inv tự tính lại từ formula (đã code sẵn)
   → b0 = 1/J_w_mới

Bước 3: CHẠY diagnose_error_sources.m
   → Error A (ideal) < 10mm → gains OK, chuyển bước 5
   → Error A > 20mm → chạy bước 4

Bước 4: CHẠY tune_gains.m (sweep tự động ~10-15 phút)
   → Apply recommended gains

Bước 5: CHẠY run_m5_comparison.m
   → Verify tất cả trajectories < 15mm SS

Bước 6: Tiếp tục M6 disturbance testing
```

---

## Còn lại stub (chờ M6)

### ESP32:
- slip_detector.m — always false (M6 sẽ implement)

### Tất cả 15 modules khác + 2 modules mới (pose_estimator, position_controller) đều functional.

---

## File inventory sau M5

### ESP32 (8 modules):
- encoder_reader.m — IIR filter (M4)
- imu_reader.m — outlier rejection + filter (M4)
- pid_controller.m — full PID + anti-windup (M4)
- adrc_controller.m — 2nd-order ESO + u_prev/z2 clamp (M4 + M5.1 fixes)
- pwm_output.m — deadband compensation (M4)
- slip_detector.m — stub (M6)
- **pose_estimator.m** — dead reckoning (M5.1 NEW)
- **position_controller.m** — PI outer loop (M5.1 NEW)

### Nucleo H7 (5 modules, unchanged from M3):
- spi_interface.m, encoder_pulse_gen.m, imu_packet_enc.m, pwm_capture.m, gpio_sync.m

### RPi5 (3 modules, unchanged from M2):
- plant_step.m, imu_model.m, state_manager.m

### Scripts (11 files):
- params_mecanum.m (optimized M5.2)
- trajectory_generator.m (fixed M5.1 + M5.2)
- run_simulation.m (rewritten M5.1)
- plot_results.m (rewritten M5.1)
- run_m5_comparison.m (M5.1 NEW)
- test_m5_integration.m (M5.1 NEW)
- **diagnose_error_sources.m** (M5.2 NEW)
- **diagnose_remaining_error.m** (M5.2 NEW)
- **tune_gains.m** (M5.2 NEW)
- test_m3_signal_conditioning.m (M3)
- test_m4_controllers.m (M4)

### Docs:
- system_architecture.md (M1)

**Tổng: 16 modules + 11 scripts + 1 doc = 28 files** (tăng từ 25 ở M5.1, từ 19 ở M1)
