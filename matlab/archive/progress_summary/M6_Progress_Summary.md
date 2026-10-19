# M6 — Disturbance & Robustness (Complete)

---

## Current Status
- Active milestone: M6 (complete — v3 final)
- Completed: M1 (1.1→1.9), M2 (2.1→2.5), M3 (3.1→3.7), M4 (4.1→4.7), M5 (5.1→5.7), M6 (6.1→6.6)
- Blocked: none
- Next: M7 — Process Metrics Framework
- Iterations: v1 (initial) → v2 (multi-trial) → v3 (redesigned conditions)

---

## M6 Overview

M6 tests the robustness of PID and ADRC controllers (tuned in M5) against three classes of disturbance:
1. **Wheel slip** — traction loss at wheel-ground interface (plant-level)
2. **Sensor degradation** — encoder PPR reduction, IMU noise increase (signal-level)
3. **Load torque disturbance** — external forces on wheels (step/ramp/random)

The controllers are NOT re-tuned — M6 evaluates how well M5.2 gains handle conditions they were not designed for.

---

## M6.1 — Files thay đổi / thêm mới từ M5

### 1. `rpi5/plant_step.m` — THAY HOÀN TOÀN

**Trước (M5):** No-slip dynamics only. Body velocities là hàm kinematics của wheel speeds.

**Sau (M6):** Thêm optional wheel slip model, controlled bởi `params.slip.enabled`. Backward compatible — khi `slip.enabled = false`, behavior giống hệt M5 (verified: Test 1.1, diff < 1e-10).

**Physics:**
- Normal force per wheel: `F_N = M*g/4 = 4×9.81/4 = 9.81 N` (equal weight distribution)
- Max static friction torque: `tau_friction_max = mu_static × F_N × r = 0.8 × 9.81 × 0.0485 = 0.381 N·m`
- Critical: `tau_max = 0.5 N·m > tau_friction_max = 0.381 N·m` → slip CAN occur at high torque

**Slip conditions (per wheel, per timestep):**
1. **Torque-induced:** `|tau_i| > tau_friction_max` → effective torque drops to kinetic level
2. **Spontaneous:** `rand < prob_spontaneous` (p=0.002) → random surface imperfections

**During slip:**
- `tau_eff = sign(tau) × mu_kinetic/mu_static × tau_friction_max × noise_factor`
- `noise_factor = 1 + 0.15×randn`, clamped [0.5, 1.5]
- Spontaneous slip at low torque: lose 30-70% of torque randomly

**Tại sao model này phù hợp cho thesis:**
- ADRC ESO sees torque loss as disturbance in z2 → compensates automatically
- PID chỉ có integral action → phản ứng chậm hơn
- Đúng lesson từ Rev_2: thêm slip SAU khi controller đã có IMU (M4), không phải trước

### 2. `esp32/slip_detector.m` — THAY HOÀN TOÀN

**Trước (M5):** Stub `always false`.

**Sau (M6):** Two-method detection:

**Method 1 — Kinematic consistency (overdetermined system):**
- Mecanum 4 wheels, 3 DOF → 1 redundant measurement
- Compute body velocity: `v_body = H_fwd × omega_est` (least-squares)
- Predict each wheel: `omega_predicted = H_inv × v_body`
- Slip ratio: `(omega_actual - omega_predicted) / max(|omega_actual|, 1.0)`
- Flag if `|slip_ratio| > 0.15` (15% threshold)

**Method 2 — IMU cross-check:**
- Compare encoder-derived wz (from H_fwd) vs IMU gyro wz
- Flag if mismatch > 0.5 rad/s AND no wheel already flagged

**Lưu ý:** Slip detector hiện tại chỉ DETECT, không CORRECT. Controller không dùng slip_flag để thay đổi behavior. ADRC ESO tự estimate disturbance mà không cần biết "đó là slip" — đây là ưu điểm cốt lõi của ADRC. Slip detector hữu ích cho monitoring/logging và có thể dùng cho future work (adaptive control).

### 3. `scripts/params_mecanum.m` — BỔ SUNG FIELDS

```matlab
%% --- Wheel Slip Model (M6) ---
params.slip.enabled          = false    % default OFF
params.slip.mu_static        = 0.8      % static friction (dry concrete)
params.slip.mu_kinetic       = 0.5      % kinetic friction (~63% of static)
params.slip.prob_spontaneous = 0.002    % probability per wheel per step
params.slip.noise_sigma      = 0.15     % kinetic friction variation (σ)
params.slip.detect_threshold = 0.15     % slip ratio detection threshold
params.slip.imu_wz_threshold = 0.5      % IMU yaw rate mismatch (rad/s)

%% --- Load Disturbance (M6) ---
params.disturbance.enabled      = false
params.disturbance.type         = 'none'   % 'step'|'ramp'|'random'|'combined'
params.disturbance.magnitude    = 0.05     % N·m
params.disturbance.start_time   = 3.0      % s
params.disturbance.ramp_rate    = 0.02     % N·m/s
params.disturbance.random_sigma = 0.03     % N·m
```

### 4. `scripts/run_single_scenario.m` — MỚI

Reusable simulation function encapsulating run_simulation.m loop:
- Input: `ctrl_type`, `traj_type`, `params`, `seed` (optional)
- Output: metrics struct (rms_pos_ss, max_pos, rms_theta, sat_pct, slip_events, ...)
- Load disturbance injected AFTER H7 pipeline, BEFORE plant_step (simulates external load)
- Calls slip_detector in ESP32 loop
- `seed` parameter: khi cung cấp, gọi `rng(seed)` đầu mỗi run → reproducible results

### 5. `scripts/run_m6_disturbance.m` — MỚI (3 iterations)

Master comparison script. Chi tiết iterations bên dưới.

### 6. `scripts/test_m6_disturbance.m` — MỚI

16 unit tests across 5 groups. Chi tiết bên dưới.

### 7. `scripts/setfields.m` — MỚI (v2+)

Utility function hỗ trợ set nested struct fields bằng dot notation:
```matlab
p = setfields(params, 'slip.enabled', true, 'enc_ppr', 256);
```

---

## Unit Tests — 16/16 PASS

### Group 1: Wheel Slip Model (plant_step.m)

| Test | Mô tả | Criteria | Result |
|------|--------|----------|--------|
| 1.1 | No slip when disabled | Chạy plant_step 2 lần: slip OFF vs ON (nhưng tau < friction limit, p_spont=0). Output phải identical (diff < 1e-10) | PASS |
| 1.2 | Slip triggers at high torque | tau=0.45 > friction limit 0.381. domega_slip < domega_noslip (less acceleration when slipping) | PASS |
| 1.3 | Bounded output | p_spont=0.5 (extreme), tau=±0.5, 100 trials. No NaN/Inf | PASS |
| 1.4 | Spontaneous slip at low torque | p_spont=1.0 (forced), tau=0.1 (below friction limit). Output phải khác no-slip (diff > 1e-6) | PASS |

### Group 2: Slip Detector

| Test | Mô tả | Criteria | Result |
|------|--------|----------|--------|
| 2.1 | No false detection | omega=[10,10,10,10] (consistent forward). slip_flag phải all false | PASS (ratio=0.0000) |
| 2.2 | Detect inconsistent wheel | omega=[10,10,10,40] (wheel 4 spinning 4×). Ít nhất 1 flag | PASS (4 flagged, ratio=0.75) |
| 2.3 | IMU cross-check | omega=[10,10,10,10] (wz_enc≈0) nhưng gyro wz=2.0. Mismatch detected | PASS |

### Group 3: Load Disturbance Logic

| Test | Mô tả | Criteria | Result |
|------|--------|----------|--------|
| 3.1 | Step disturbance | 0 trước start_time, magnitude sau start_time | PASS |
| 3.2 | Ramp disturbance | Linear tăng dần: 0.02×1s=0.02, 0.02×2s=0.04 | PASS |
| 3.3 | Random statistics | σ=0.03, 10000 samples, std ≈ 0.03 | PASS (0.030) |

### Group 4: End-to-End Stability

| Test | Điều kiện | Pass criteria | Actual |
|------|-----------|--------------|--------|
| 4.1 | PID + wheel slip | SS < 100mm, no NaN | PASS (25mm) |
| 4.2 | ADRC + wheel slip | SS < 100mm, no NaN | PASS (28mm) |
| 4.3 | PID + combined load | SS < 200mm, no NaN | PASS (28mm) |
| 4.4 | ADRC + combined load | SS < 200mm, no NaN | PASS (25mm) |
| 4.5 | Worst case (all) | SS < 500mm, no NaN | PASS (PID=27mm, ADRC=27mm) |

### Group 5: ADRC ESO Response

| Test | Mô tả | Result |
|------|--------|--------|
| 5.1 | Step disturbance recovery | PASS (ADRC=2.4mm, PID=2.0mm — both handle, no diverge) |

---

## Iteration History: v1 → v2 → v3

### v1 — Initial (single-trial, T_sim=5s)

**Approach:** 12 conditions × 2 controllers × 4 trajectories = 96 runs, 1 trial each.

**Conditions:**
- Nominal, Wheel slip
- Encoder noise sweep: σ = 0.02 (nom), 0.05 (×2.5), 0.10 (×5), 0.20 (×10)
- IMU noise: ×3, ×5
- Load disturbance: step 0.05 N·m, ramp 0.02 N·m/s, random σ=0.03, combined
- Worst case: slip + enc noise ×5 + combined load

**Issues discovered:**

1. **T_sim=5s quá ngắn cho circle:** Outer loop integral Ti=Kp/Ki=6/3=2s → cần ~3τ=6s để converge. Ở T_sim=5s, SS window chỉ 2.5-5s, integral chưa settle → circle SS ~25mm thay vì ~7mm (T_sim=10s). Data circle không đáng tin cậy.

2. **Octave `rms()` function missing:** Octave không có `rms()` built-in (thuộc signal package). Fix: thay bằng `sqrt(mean(x.^2))` trong run_single_scenario.m.

3. **Random variance artifacts trên line trajectory:** ADRC/line nominal cho 11.8mm (single trial), nhưng M5.2 cùng params = 7.5mm. Nguyên nhân: line 0.3m/s × 10s = 3m travel distance → dead reckoning drift tích lũy, phụ thuộc random seed. Single-trial không reliable.

**Kết luận v1:** Wheel slip pattern rõ ràng (+32% ADRC advantage), nhưng data nhiễu bởi random variance. Cần multi-trial averaging.

---

### v2 — Multi-trial (N=5 trials per scenario, T_sim=10s)

**Changes:**
- Mỗi scenario chạy 5 lần với seeds cố định [101, 202, 303, 404, 505]
- Report mean ± std thay vì single value
- run_single_scenario.m thêm parameter `seed`
- Thêm `setfields.m` utility function
- Variance analysis (CoV%) per condition

**Results (MATLAB, T_sim=10s, 480 runs, 6 min):**

Nominal: PID 5.8±mixed, ADRC 5.3±mixed → baseline confirmed.

**Issues discovered:**

1. **Encoder noise sweep GIỐNG HỆT Nominal:** Conditions 1, 3, 4 cho **đúng cùng con số** ở mọi trial:
   ```
   Nominal:        PID/line: [7.1 7.0 2.6 4.8 5.0]
   Enc noise x2.5: PID/line: [7.1 7.0 2.6 4.8 5.0]  ← identical
   Enc noise x5:   PID/line: [7.1 7.0 2.6 4.8 5.0]  ← identical
   Enc noise x10:  PID/line: [7.0 6.8 3.1 5.1 7.3]  ← barely different
   ```
   **Root cause:** `encoder_pulse_gen.m` line 52: `enc_counts = round(int_counts + noise)`. Ở ω=10 rad/s, counts/step ≈ 1.63. Noise sigma 0.02→0.10 cộng vào ~1.63 rồi round() → noise bị nuốt. Quantization noise dominant, additive noise invisible. Chỉ sigma 0.20 đôi khi flip integer → barely different. **3 conditions hoàn toàn vô nghĩa.**

2. **Load disturbance KHÔNG có effect:** Step 0.05 N·m (10% tau_max), ramp 0.02 N·m/s, random σ=0.03 N·m (6% tau_max). Tất cả cho kết quả ~5.3-5.9mm ≈ nominal 5.8mm. Outer loop PI integral absorb trivially. **4 conditions không tạo differentiation.**

3. **Wheel slip vẫn tốt:** PID figure8 = 22.0±0.6mm, ADRC = 5.6±0.2mm (+74%). Std rất nhỏ → reliable. Pattern consistent across trials.

**Kết luận v2:** 7/12 conditions vô nghĩa (encoder noise 3 + load disturbance 4). Cần redesign conditions.

---

### v3 — Redesigned conditions (FINAL)

**Changes from v2:**

| Aspect | v2 (broken) | v3 (fixed) | Lý do |
|--------|-------------|------------|-------|
| Encoder test | `enc_noise_sigma` sweep (0.05/0.10/0.20) | **PPR sweep** (512/256/128) | Additive noise bị round() nuốt. Giảm PPR = giảm counts/step = quantization coarser thực sự |
| Load step | 0.05 N·m (10% tau_max) | **0.15 N·m (30% tau_max)** | 10% bị PI absorb trivially |
| Load ramp | 0.02 N·m/s, cap 0.1 | **0.05 N·m/s, cap 0.25 (50% tau_max)** | Quá nhỏ, không thấy effect |
| Load random σ | 0.03 N·m (6%) | **0.10 N·m (20% tau_max)** | Cần đủ lớn để thấy transient |
| Worst case | slip + enc σ=0.10 + load cũ | **slip + PPR 256 + load mới** | Combine actual differentiators |

**PPR sweep — tại sao effective:**
```
At omega=10 rad/s, dt=0.001:
  PPR 1024: 1.63 counts/step (adequate)
  PPR  512: 0.81 counts/step (borderline)
  PPR  256: 0.41 counts/step (coarse → significant quantization error)
  PPR  128: 0.20 counts/step (extreme → most timesteps read 0 or 1)
```

Giảm PPR tạo quantization error thực sự mà không bị round() nuốt.

---

## M6.5 — Final Results (v3, MATLAB, T_sim=10s, N=5 trials)

### Comparison Table: Mean ± Std (mm)

| # | Condition | PID line | PID circle | PID sq | PID f8 | PID avg | ADRC line | ADRC circle | ADRC sq | ADRC f8 | ADRC avg | Impr |
|---|-----------|----------|------------|--------|--------|---------|-----------|-------------|---------|---------|----------|------|
| 1 | Nominal | 5.3±1.8 | 6.8±0.7 | 5.8±0.3 | 5.2±0.1 | **5.8** | 5.2±1.8 | 6.8±0.7 | 3.8±0.5 | 5.4±0.1 | **5.3** | +8.1% |
| 2 | **Wheel slip** | 7.9±1.3 | 7.5±0.8 | 14.4±0.2 | 22.0±0.6 | **12.9** | 5.2±1.6 | 7.4±0.7 | 9.5±0.3 | 5.6±0.2 | **6.9** | **+46.4%** |
| 3 | PPR 512 | 5.3±1.8 | 6.9±0.6 | 7.5±0.2 | 10.0±0.0 | **7.4** | 5.3±1.8 | 6.8±0.7 | 3.9±0.5 | 5.5±0.1 | **5.4** | +27.5% |
| 4 | **PPR 256** | 33.9±11.4 | 35.3±5.8 | 18.9±5.3 | 40.5±0.5 | **32.2** | 5.5±1.8 | 6.9±0.7 | 4.1±0.6 | 5.6±0.1 | **5.5** | **+82.9%** |
| 5 | **PPR 128** | 63.4±27.8 | 75.3±0.8 | 47.4±38.9 | 133.0±7.8 | **79.8** | 5.8±1.7 | 7.1±0.6 | 4.7±0.6 | 5.9±0.1 | **5.9** | **+92.7%** |
| 6 | IMU noise ×3 | 5.7±2.3 | 7.0±0.7 | 5.9±0.3 | 5.3±0.1 | **6.0** | 5.7±2.3 | 6.9±0.7 | 3.9±0.5 | 5.5±0.2 | **5.5** | +7.5% |
| 7 | IMU noise ×5 | 6.3±2.7 | 7.2±0.8 | 5.9±0.3 | 5.4±0.2 | **6.2** | 6.2±2.7 | 7.1±0.8 | 4.0±0.6 | 5.7±0.2 | **5.7** | +7.3% |
| 8 | Load: step 30% | 5.2±1.9 | 6.7±0.7 | 7.2±0.3 | 4.7±0.1 | **5.9** | 5.2±1.9 | 6.7±0.7 | 4.8±0.4 | 5.5±0.1 | **5.6** | +6.1% |
| 9 | Load: ramp 50% | 5.2±1.9 | 7.1±0.7 | 8.6±0.4 | 5.5±0.2 | **6.6** | 5.2±1.9 | 6.8±0.7 | 6.2±0.5 | 5.5±0.1 | **6.0** | +9.8% |
| 10 | Load: random 20% | 5.4±2.1 | 6.9±0.8 | 5.8±0.3 | 5.3±0.1 | **5.8** | 5.3±2.1 | 6.8±0.8 | 3.8±0.6 | 5.4±0.1 | **5.4** | +8.4% |
| 11 | Load: combined | 5.4±2.2 | 6.8±0.8 | 6.7±0.4 | 4.7±0.1 | **5.9** | 5.3±2.1 | 6.8±0.8 | 4.5±0.5 | 5.5±0.1 | **5.5** | +6.4% |
| 12 | **Worst case** | 38.5±3.0 | 82.3±1.4 | 53.5±7.8 | 172.7±2.6 | **86.8** | 5.3±1.4 | 7.6±0.6 | 8.6±0.4 | 7.3±0.2 | **7.2** | **+91.7%** |

### Degradation from Nominal (×factor)

| Condition | PID line | PID circle | PID sq | PID f8 | ADRC line | ADRC circle | ADRC sq | ADRC f8 |
|-----------|----------|------------|--------|--------|-----------|-------------|---------|---------|
| Nominal | 1.0× | 1.0× | 1.0× | 1.0× | 1.0× | 1.0× | 1.0× | 1.0× |
| Wheel slip | 1.5× | 1.1× | **2.5×** | **4.2×** | 1.0× | 1.1× | 2.5× | 1.0× |
| PPR 512 | 1.0× | 1.0× | 1.3× | **1.9×** | 1.0× | 1.0× | 1.0× | 1.0× |
| PPR 256 | **6.4×** | **5.2×** | **3.3×** | **7.8×** | 1.0× | 1.0× | 1.1× | 1.0× |
| PPR 128 | **12.0×** | **11.0×** | **8.2×** | **25.5×** | 1.1× | 1.1× | 1.2× | 1.1× |
| IMU ×5 | 1.2× | 1.0× | 1.0× | 1.0× | 1.2× | 1.0× | 1.0× | 1.0× |
| Load: ramp 50% | 1.0× | 1.0× | 1.5× | 1.0× | 1.0× | 1.0× | 1.6× | 1.0× |
| **Worst case** | **7.3×** | **12.0×** | **9.2×** | **33.1×** | 1.0× | 1.1× | **2.3×** | 1.3× |

### ADRC wins: 37/48 scenario pairs (77.1%)

### Variance analysis

| Condition | PID CoV | ADRC CoV |
|-----------|---------|----------|
| Nominal | 12.6% | 15.0% |
| Wheel slip | 5.6% | 9.9% |
| PPR 256 | 17.9% | 14.3% |
| PPR 128 | 23.6% | 13.0% |
| IMU ×5 | 16.2% | 18.7% |
| Worst case | 4.3% | 9.2% |

Tất cả CoV < 25% → data reliable. Đặc biệt wheel slip và worst case CoV < 10% cho PID → kết quả rất consistent.

---

## M6.6 — Analysis: When ADRC Outperforms PID

### Finding 1: ADRC immune to encoder quantization degradation

Đây là phát hiện **mạnh nhất và bất ngờ nhất** của M6:

| PPR | PID avg (mm) | ADRC avg (mm) | PID degradation | ADRC degradation |
|-----|-------------|---------------|-----------------|------------------|
| 1024 (nom) | 5.8 | 5.3 | 1.0× | 1.0× |
| 512 | 7.4 | 5.4 | 1.3× | 1.0× |
| 256 | **32.2** | 5.5 | **5.6×** | **1.0×** |
| 128 | **79.8** | 5.9 | **13.8×** | **1.1×** |

PID degradation **exponential** khi giảm PPR. ADRC gần như **bất biến**. Giải thích:

- PID dùng `error = omega_ref - omega_est` trực tiếp. Khi encoder coarse (PPR thấp), `omega_est` oscillate mạnh giữa 0 và full-count → error signal noisy → P và D terms oscillate → torque command oscillate → plant vibrate → position error tích lũy.
- ADRC ESO estimate ω qua `z1`, filtered bởi ESO bandwidth (ω_o=100). ESO hoạt động như observer bậc cao, tự smooth quantization noise. Disturbance estimate z2 absorb sai số. Control law dùng z1 (smooth) thay vì omega_est trực tiếp (noisy).

**Ý nghĩa cho thesis:** ADRC cho phép sử dụng encoder PPR thấp hơn (giảm cost) mà không mất tracking performance. Đây là ưu điểm thực tế cho industrial application.

### Finding 2: Wheel slip — ADRC ESO estimates traction loss

| Trajectory | PID (mm) | ADRC (mm) | ADRC impr | PID degrad |
|-----------|----------|-----------|-----------|------------|
| line | 7.9±1.3 | 5.2±1.6 | +33.9% | 1.5× |
| circle | 7.5±0.8 | 7.4±0.7 | +0.9% | 1.1× |
| square | 14.4±0.2 | 9.5±0.3 | +33.9% | 2.5× |
| figure8 | **22.0±0.6** | **5.6±0.2** | **+74.4%** | **4.2×** |

ESO mechanism:
```
Wheel slips → effective torque drops → ω doesn't respond as expected
→ ESO error: e_eso = z1 - omega_meas ≠ 0
→ z2 update: z2 += dt × (-beta2 × e_eso)   [beta2=10000]
→ z2 increases (estimates the traction loss)
→ Control law: tau = (kp×(ref-z1) - z2) / b0
→ Compensates by applying more torque
→ Wheel recovers faster
```

PID has no disturbance estimation — relies solely on integral action:
```
Wheel slips → omega drops → error increases
→ P-term reacts immediately (but proportional only)
→ I-term slowly accumulates → slow compensation
→ Position error during transient is large
→ Anti-windup may freeze integral if saturated → even slower
```

**Figure-8 worst:** Highest torque requirements + frequent direction changes → more slip events (6390 events/10s) + less time for integral to settle between slip events.

**Circle paradox (TIE):** Circle has constant speed and direction → torque near-constant → slip occurs mainly during initial transient → PI integral has time to settle → both controllers reach similar SS.

### Finding 3: Load disturbance — both controllers robust

Ngay cả ở 30-50% tau_max, effect chỉ ~1-2mm. Tại sao?

- **Outer loop PI integral absorbs DC offset:** Load step 0.15 N·m → inner loop sees velocity error → integral accumulates → compensates within ~Ti = Kp_inner/Ki_inner ≈ 0.08s. Outer loop then adjusts position.
- **Architecture design:** Two-loop cascade isolates position tracking from torque disturbance. Inner loop rejects disturbance trước khi nó affect position.
- **Kết luận:** Load disturbance ≤50% tau_max **không phải** differentiator giữa PID và ADRC trong architecture này. Đây là kết quả hợp lệ — cho thấy cascade PI design robust với load disturbance.

### Finding 4: Worst case — headline result cho thesis

**PID: 86.8mm (15.0× nominal). ADRC: 7.2mm (1.4× nominal).**

| Trajectory | PID (mm) | ADRC (mm) | ADRC tốt hơn |
|-----------|----------|-----------|--------------|
| line | 38.5±3.0 | 5.3±1.4 | +86.3% |
| circle | 82.3±1.4 | 7.6±0.6 | +90.8% |
| square | 53.5±7.8 | 8.6±0.4 | +84.0% |
| figure8 | **172.7±2.6** | **7.3±0.2** | **+95.8%** |

Worst case = slip + PPR 256 + combined load. ADRC xử lý **đồng thời** cả 3 loại disturbance mà PID collapse:
- Slip → ESO z2 estimate traction loss
- PPR coarse → ESO z1 smooth quantization noise
- Load → ESO z2 estimate load change

PID figure-8 worst: 172.7mm = **33× nominal**. ADRC: 7.3mm = **1.3× nominal**. ADRC robust gấp **23.7×** PID.

### Finding 5: IMU noise — marginal effect

IMU noise ×3 và ×5 cho degradation < 1mm cho cả hai controller. Signal conditioning pipeline (M3 IIR filter + M4 outlier rejection) đủ tốt. IMU chỉ dùng cho heading rate trong pose_estimator — encoder vẫn là primary sensor.

### Summary Table

| Condition | Winner | Margin | Mechanism |
|-----------|--------|--------|-----------|
| Nominal | TIE | +8% ADRC | Noise floor dominated, both well-tuned |
| Wheel slip | **ADRC** | **+46%** | ESO estimates traction loss online |
| PPR 512 | ADRC | +28% | ESO smooths quantization |
| PPR 256 | **ADRC** | **+83%** | PID collapses, ADRC immune |
| PPR 128 | **ADRC** | **+93%** | PID 80mm error, ADRC still 6mm |
| IMU ×3 | TIE | +8% ADRC | IMU filter adequate |
| IMU ×5 | TIE | +7% ADRC | IMU filter adequate |
| Load step 30% | TIE | +6% ADRC | PI integral absorbs DC offset |
| Load ramp 50% | ADRC | +10% | Ramp slightly challenges integral |
| Load random 20% | TIE | +8% ADRC | Both handle stochastic load |
| Load combined | TIE | +6% ADRC | Cascade design robust |
| **Worst case** | **ADRC** | **+92%** | PID 87mm, ADRC 7mm |

---

## Conclusions for Thesis (Chapter 5)

### 1. ADRC vượt trội khi có model uncertainty lớn
Wheel slip và encoder quantization coarse là hai điều kiện mà PID không thể handle nhưng ADRC gần như không bị ảnh hưởng. ESO estimate disturbance online mà không cần biết nguồn gốc (slip? friction? load? quantization?).

### 2. PID đủ tốt khi disturbance nhỏ và structured
Load disturbance ≤50% tau_max được cascade PI architecture absorb tốt. Nominal performance chỉ thua ADRC 8%. Nếu environment known và stable, PID đủ tốt với chi phí implementation thấp hơn.

### 3. Encoder PPR là bottleneck quan trọng cho PID
PID degradation exponential khi giảm PPR. Ở PPR 256 (common cho low-cost encoder), PID error tăng 5.6× trong khi ADRC không thay đổi. **ADRC cho phép dùng encoder rẻ hơn** — ý nghĩa thực tế cho industrial application.

### 4. Signal conditioning pipeline (M3-M4) effective cho cả hai
IIR filter (encoder τ=5ms, IMU τ=3ms) + outlier rejection + deadband compensation tạo noise floor ~5-6mm mà cả PID và ADRC đều đạt. Bottleneck là hardware quality, không phải controller.

### 5. ADRC robustness advantage compounds dưới multiple disturbances
Worst case (slip + PPR 256 + load): PID degrades 15× nominal, ADRC chỉ 1.4×. Ưu điểm ADRC không phải từ 1 disturbance mà từ **khả năng handle nhiều disturbance đồng thời** — ESO z2 estimate tổng hợp tất cả disturbance vào 1 scalar.

---

## File inventory sau M6

### ESP32 (8 modules):
- encoder_reader.m — IIR filter (M4)
- imu_reader.m — outlier rejection + filter (M4)
- pid_controller.m — full PID + anti-windup (M4)
- adrc_controller.m — 2nd-order ESO + u_prev/z2 clamp (M4 + M5.1 fixes)
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

### Scripts (15 files):
- params_mecanum.m (updated M6)
- trajectory_generator.m (M5.2)
- run_simulation.m (M5.1)
- plot_results.m (M5.1)
- run_m5_comparison.m (M5.1)
- **run_single_scenario.m** (M6 NEW)
- **run_m6_disturbance.m** (M6 NEW, v3 final)
- **test_m6_disturbance.m** (M6 NEW)
- **setfields.m** (M6 NEW, v2+)
- test_m5_integration.m (M5.1)
- test_m4_controllers.m (M4)
- test_m3_signal_conditioning.m (M3)
- diagnose_error_sources.m (M5.2)
- diagnose_remaining_error.m (M5.2)
- tune_gains.m (M5.2)

### Docs:
- system_architecture.md (M1)
- **M6_Progress_Summary.md** (M6)

**Tổng: 16 modules + 15 scripts + 2 docs = 33 files** (tăng từ 28 ở M5)
