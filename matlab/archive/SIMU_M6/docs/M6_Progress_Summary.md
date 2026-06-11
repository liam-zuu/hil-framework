# M6 — Disturbance & Robustness (Complete)

---

## Current Status
- Active milestone: M6 (complete)
- Completed: M1 (1.1→1.9), M2 (2.1→2.5), M3 (3.1→3.7), M4 (4.1→4.7), M5 (5.1→5.7), M6 (6.1→6.6)
- Blocked: none
- Next: M7 — Process Metrics Framework

---

## M6 — Đã làm

### Files thay đổi / thêm mới từ M5 → M6

#### 1. `rpi5/plant_step.m` — THAY HOÀN TOÀN (M6.1)
- M5: no-slip dynamics only
- M6: Thêm optional wheel slip model, controlled bởi `params.slip.enabled`
- Backward compatible: khi `slip.enabled = false`, behavior giống hệt M5
- Physics:
  - `F_N = M*g/4` (normal force per wheel, equal weight distribution)
  - `tau_friction_max = mu_static * F_N * r` = 0.8 × 9.81 × 0.0485 = **0.381 N·m**
  - Khi `|tau| > tau_friction_max`: effective torque drops to kinetic friction level
  - `tau_eff = sign(tau) * mu_kinetic/mu_static * tau_friction_max * noise_factor`
  - Noise factor: Gaussian (σ=0.15), clamped [0.5, 1.5]
- Spontaneous slip: random probability per wheel per step (p=0.002)
  - Khi tau thấp hơn kinetic friction: giảm 30-70% torque randomly
- Critical insight: `tau_max = 0.5 N·m > tau_friction_max = 0.381 N·m`
  → slip CAN occur at high torque commands, especially during transients

#### 2. `esp32/slip_detector.m` — THAY HOÀN TOÀN (M6.1)
- Stub `always false` → Two-method detection:
  - **Method 1: Kinematic consistency** — 4 wheels, 3 DOF → overdetermined system.
    Compute body velocity from all wheels (least-squares via H_fwd), then predict
    each wheel's speed from body velocity (via H_inv). Deviation = slip ratio.
  - **Method 2: IMU cross-check** — compare encoder-derived wz vs IMU gyro wz.
    Large mismatch suggests at least one wheel slipping.
  - Combined: `slip_flag(i) = true` if `|slip_ratio(i)| > threshold` (15%)
    OR if IMU mismatch detected AND wheel has highest residual.

#### 3. `scripts/params_mecanum.m` — BỔ SUNG FIELDS
```
params.slip.enabled          = false
params.slip.mu_static        = 0.8     (dry concrete)
params.slip.mu_kinetic       = 0.5     (63% of static)
params.slip.prob_spontaneous = 0.002   (per wheel per step)
params.slip.noise_sigma      = 0.15    (kinetic friction variation)
params.slip.detect_threshold = 0.15    (15% deviation → flag)
params.slip.imu_wz_threshold = 0.5     (rad/s mismatch)
params.disturbance.enabled   = false
params.disturbance.type      = 'none'  ('step'|'ramp'|'random'|'combined')
params.disturbance.magnitude = 0.05    (N·m, 10% of tau_max)
params.disturbance.start_time = 3.0    (s)
params.disturbance.ramp_rate  = 0.02   (N·m/s)
params.disturbance.random_sigma = 0.03 (N·m)
```

#### 4. `scripts/run_single_scenario.m` — MỚI
- Reusable simulation function encapsulating run_simulation.m loop
- Input: controller type, trajectory type, params struct
- Output: metrics struct (rms_pos_ss, max_pos, rms_theta, sat_pct, slip_events, ...)
- Includes load disturbance injection AFTER H7 pipeline, BEFORE plant_step
- Supports: step, ramp, random, combined disturbance types
- Calls slip_detector in ESP32 loop

#### 5. `scripts/run_m6_disturbance.m` — MỚI
- Master comparison script: 12 conditions × 2 controllers × 4 trajectories = 96 scenarios
- Prints comprehensive tables and analysis

#### 6. `scripts/test_m6_disturbance.m` — MỚI
- 16 unit tests across 5 groups: slip model, slip detector, disturbance generation,
  end-to-end stability, ADRC ESO response
- **Result: 16/16 PASS**

---

## Test Results

### Unit tests: 16/16 PASS

| Test | Result |
|------|--------|
| 1.1 No slip when disabled | PASS |
| 1.2 Slip triggers at high torque | PASS (domega reduced) |
| 1.3 Slip output bounded (100 trials) | PASS (no NaN/Inf) |
| 1.4 Spontaneous slip at low torque | PASS |
| 2.1 No false detection (consistent speeds) | PASS (ratio=0.0000) |
| 2.2 Detect slip on inconsistent wheel | PASS |
| 2.3 IMU wz cross-check | PASS |
| 3.1 Step disturbance logic | PASS |
| 3.2 Ramp disturbance logic | PASS |
| 3.3 Random disturbance statistics | PASS (std=0.030) |
| 4.1 PID stable under wheel slip | PASS (SS=25mm) |
| 4.2 ADRC stable under wheel slip | PASS (SS=28mm) |
| 4.3 PID stable under combined disturbance | PASS (SS=28mm) |
| 4.4 ADRC stable under combined disturbance | PASS (SS=25mm) |
| 4.5 Worst case still bounded | PASS (PID=27mm, ADRC=27mm) |
| 5.1 ADRC step disturbance recovery | PASS |

---

## M6.5 — Comprehensive Comparison Table

### Steady-State RMS Position Error (mm) — T_sim = 5s

| # | Condition | PID line | PID circle | PID sq | PID f8 | PID avg | ADRC line | ADRC circle | ADRC sq | ADRC f8 | ADRC avg | ADRC imp |
|---|-----------|----------|------------|--------|--------|---------|-----------|-------------|---------|---------|----------|----------|
| 1 | Nominal | 2.3 | 27.2 | 6.6 | 5.4 | 10.4 | 2.4 | 27.6 | 3.8 | 6.4 | 10.1 | +3% |
| 2 | **Wheel slip** | 7.2 | 26.2 | 17.0 | 24.3 | **18.7** | 3.0 | 27.7 | 10.7 | 9.5 | **12.7** | **+32%** |
| 3 | Enc noise ×2.5 | 3.4 | 25.2 | 6.5 | 6.8 | 10.5 | 2.1 | 25.2 | 3.8 | 6.6 | 9.4 | +10% |
| 4 | Enc noise ×5 | 2.2 | 26.0 | 6.4 | 5.5 | 10.0 | 2.2 | 26.0 | 3.8 | 6.3 | 9.6 | +4% |
| 5 | Enc noise ×10 | 5.8 | 25.9 | 6.4 | 6.7 | 11.2 | 2.8 | 28.1 | 4.4 | 6.4 | 10.4 | +7% |
| 6 | IMU noise ×3 | 3.9 | 27.1 | 6.4 | 5.2 | 10.7 | 2.2 | 26.2 | 3.9 | 6.5 | 9.7 | +9% |
| 7 | IMU noise ×5 | 3.9 | 25.0 | 6.5 | 5.2 | 10.2 | 2.1 | 27.6 | 3.8 | 6.3 | 9.9 | +3% |
| 8 | Load: step | 2.1 | 26.0 | 6.6 | 5.8 | 10.1 | 2.2 | 28.9 | 3.9 | 6.3 | 10.3 | -2% |
| 9 | Load: ramp | 4.4 | 26.3 | 6.4 | 5.4 | 10.6 | 3.5 | 28.4 | 3.9 | 6.4 | 10.6 | +1% |
| 10 | Load: random | 4.3 | 28.4 | 6.6 | 5.3 | 11.2 | 2.3 | 24.9 | 4.0 | 6.3 | 9.4 | +16% |
| 11 | Load: combined | 2.3 | 25.0 | 6.5 | 5.2 | 9.8 | 2.2 | 25.5 | 3.8 | 6.7 | 9.6 | +2% |
| 12 | **Worst case** | 8.1 | 25.8 | 16.4 | 24.2 | **18.6** | 2.4 | 27.1 | 10.6 | 10.4 | **12.6** | **+32%** |

### Degradation from Nominal (×factor)

| Condition | PID line | PID sq | PID f8 | ADRC line | ADRC sq | ADRC f8 |
|-----------|----------|--------|--------|-----------|---------|---------|
| Nominal | 1.0× | 1.0× | 1.0× | 1.0× | 1.0× | 1.0× |
| Wheel slip | **3.1×** | **2.6×** | **4.5×** | 1.3× | 2.8× | **1.5×** |
| Enc noise ×10 | 2.5× | 1.0× | 1.2× | 1.2× | 1.2× | 1.0× |
| Load: combined | 1.0× | 1.0× | 1.0× | 0.9× | 1.0× | 1.0× |
| **Worst case** | **3.5×** | **2.5×** | **4.5×** | 1.0× | **2.8×** | **1.6×** |

> Circle excluded from degradation table: dominated by trajectory/settling dynamics (~25mm) 
> rather than controller/disturbance effects. Relative comparisons not meaningful.

---

## M6.6 — Analysis: When ADRC Outperforms PID

### Key Finding 1: Wheel slip is the differentiator

ADRC advantage is **strongest under wheel slip** (+32% improvement). This is the exact scenario ADRC is designed for: the ESO estimates the torque loss from slip as part of z2 (total disturbance) and the control law compensates automatically:

```
u0 = kp * (omega_ref - z1)     ← desired acceleration
tau = (u0 - z2) / b0           ← cancel estimated disturbance
```

When a wheel slips, effective torque drops → ω doesn't change as expected → ESO sees discrepancy → z2 increases → controller applies more torque → wheel recovers faster.

PID has no disturbance estimation — it relies solely on integral action, which is slow (Ti = Kp/Ki ≈ 0.08s inner loop but limited by anti-windup).

**Degradation pattern:**
- PID figure-8 under slip: 5.4 → 24.3mm (**4.5× worse**)
- ADRC figure-8 under slip: 6.4 → 9.5mm (**1.5× worse**)
- ADRC is **2.6× more robust** than PID against wheel slip on figure-8

### Key Finding 2: Sensor noise has minimal impact

Both controllers robust to encoder noise up to ×10 and IMU noise up to ×5. This is because:
1. Encoder reader IIR filter (τ=5ms) smooths quantization noise
2. IMU reader outlier rejection + IIR filter (τ=3ms) smooths IMU noise
3. Outer loop PI integrator naturally filters high-frequency noise
4. Signal conditioning (M3) provides adequate noise floor

ADRC slightly worse at high encoder noise on circle (+2mm at ×10) — ESO amplifies noise through beta2=10000 gain. But the effect is small.

### Key Finding 3: Load disturbance — both handle well

Step, ramp, random, and combined load disturbances (0.03-0.05 N·m, 6-10% of tau_max) cause minimal degradation for both controllers. The outer loop PI integral absorbs constant offsets, and inner loop anti-windup prevents saturation issues. ADRC is slightly better on random load (+16%) due to faster disturbance estimation.

### Key Finding 4: Worst case validates ADRC robustness

Under simultaneous slip + noise ×5 + combined load:
- PID average: 18.6mm (1.8× nominal)
- ADRC average: 12.6mm (1.2× nominal)
- **ADRC 32% better than PID under worst case**

The worst-case degradation is dominated by wheel slip (the noise and load effects are secondary).

### Summary: Conditions Where Each Controller Excels

| Condition | Better Controller | Margin | Reason |
|-----------|------------------|--------|--------|
| Nominal (no disturbance) | TIE | ~3% | Both well-tuned, noise floor dominated |
| Wheel slip | **ADRC** | **+32%** | ESO estimates traction loss as disturbance |
| High encoder noise | ADRC | +4-10% | ESO smooths estimation; PID derivative affected |
| High IMU noise | TIE | ~3-9% | IMU filter adequate for both |
| Step load | TIE | ~2% | Both PI (outer) + anti-windup handle DC offset |
| Ramp load | TIE | ~1% | Both integral action tracks ramp |
| Random load | **ADRC** | +16% | ESO faster than integral for random |
| Combined load | TIE | ~2% | Loads too small to differentiate |
| **Worst case** | **ADRC** | **+32%** | Cumulative advantage from slip + random |

### Conclusion for Thesis (Chapter 5)

1. **ADRC vượt trội khi có model uncertainty lớn** — wheel slip là ví dụ điển hình. ESO
   estimate disturbance online, không cần biết friction model trước.

2. **PID đủ tốt khi disturbance nhỏ và structured** — step/ramp load ≤10% tau_max được
   PI integral absorb tốt. Không cần ADRC overhead.

3. **Noise robustness tương đương** — cả hai đều robust với sensor noise nhờ signal
   conditioning pipeline (M3) và filter (M4). Bottleneck là hardware (encoder PPR,
   IMU quality), không phải controller.

4. **ADRC trade-off: complexity vs robustness** — ADRC cần tune thêm b0, ω_o, ω_c
   (3 params vs PID 3 params Kp, Ki, Kd). Nhưng ADRC tự adapt khi plant thay đổi,
   PID cần re-tune.

---

## File inventory sau M6

### ESP32 (8 modules):
- encoder_reader.m — IIR filter (M4)
- imu_reader.m — outlier rejection + filter (M4)
- pid_controller.m — full PID + anti-windup (M4)
- adrc_controller.m — 2nd-order ESO + clamping (M4 + M5.1 fixes)
- pwm_output.m — deadband compensation (M4)
- **slip_detector.m** — kinematic consistency + IMU cross-check (**M6 REPLACED**)
- pose_estimator.m — dead reckoning (M5.1)
- position_controller.m — PI outer loop (M5.1)

### RPi5 (3 modules):
- **plant_step.m** — dynamics + optional wheel slip (**M6 REPLACED**)
- imu_model.m — noise + bias drift (M2)
- state_manager.m — state storage (M1)

### Nucleo H7 (5 modules, unchanged from M3):
- spi_interface.m, encoder_pulse_gen.m, imu_packet_enc.m, pwm_capture.m, gpio_sync.m

### Scripts (14 files):
- params_mecanum.m (updated M6)
- trajectory_generator.m (M5.2)
- run_simulation.m (M5.1)
- plot_results.m (M5.1)
- run_m5_comparison.m (M5.1)
- **run_single_scenario.m** (M6 NEW)
- **run_m6_disturbance.m** (M6 NEW)
- **test_m6_disturbance.m** (M6 NEW)
- test_m5_integration.m (M5.1)
- test_m4_controllers.m (M4)
- test_m3_signal_conditioning.m (M3)
- diagnose_error_sources.m (M5.2)
- diagnose_remaining_error.m (M5.2)
- tune_gains.m (M5.2)

### Docs:
- system_architecture.md (M1)

**Tổng: 16 modules + 14 scripts + 1 doc = 31 files** (tăng từ 28 ở M5)
