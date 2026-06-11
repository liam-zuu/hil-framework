# HIL Simulation — System Architecture

---

## 1. State Vector Definition (Step 1.1)

### Full state vector x[k] ∈ ℝ¹⁰

| Index | Symbol | Description              | Unit    | Sign Convention               |
|-------|--------|--------------------------|---------|-------------------------------|
| 1     | x      | Position X (world)       | m       | +X = forward                  |
| 2     | y      | Position Y (world)       | m       | +Y = left                     |
| 3     | θ      | Heading (world)          | rad     | +θ = CCW from +X              |
| 4     | vx     | Body velocity X (body)   | m/s     | +vx = forward                 |
| 5     | vy     | Body velocity Y (body)   | m/s     | +vy = left                    |
| 6     | ωz     | Yaw rate (body)          | rad/s   | +ωz = CCW                     |
| 7     | ω1     | Wheel 1 speed (FL)       | rad/s   | + = forward drive             |
| 8     | ω2     | Wheel 2 speed (FR)       | rad/s   | + = forward drive             |
| 9     | ω3     | Wheel 3 speed (RL)       | rad/s   | + = forward drive             |
| 10    | ω4     | Wheel 4 speed (RR)       | rad/s   | + = forward drive             |

### Wheel numbering (top view)

```
    Front (+X)
  [1]---[2]     FL=1, FR=2
   |     |
  [3]---[4]     RL=3, RR=4
    Rear (-X)
```

### Wheel positions (body frame)

| Wheel | Position         | Roller angle |
|-------|------------------|--------------|
| 1 FL  | (+lx, +ly)       | +45°         |
| 2 FR  | (+lx, -ly)       | -45°         |
| 3 RL  | (-lx, +ly)       | -45°         |
| 4 RR  | (-lx, -ly)       | +45°         |

Roller angle convention: X-configuration (FL/RR same, FR/RL same).

### Mecanum kinematics contract

**Inverse kinematics** (body velocity → wheel speed) — used in `run_simulation.m`:

```
ω1 = (1/r)(vx - vy - L·ωz)     L = lx + ly
ω2 = (1/r)(vx + vy + L·ωz)
ω3 = (1/r)(vx + vy - L·ωz)
ω4 = (1/r)(vx - vy + L·ωz)
```

**Forward kinematics** (wheel speed → body velocity) — must match in `plant_step.m`:

```
vx = (r/4)(ω1 + ω2 + ω3 + ω4)
vy = (r/4)(-ω1 + ω2 + ω3 - ω4)
ωz = (r/(4L))(-ω1 + ω2 - ω3 + ω4)
```

Verification: Forward(Inverse(v)) = v. These two MUST be mathematically inverse.

### Motor model (simplification)

Direct torque input: τ → J_w·(dω/dt) + b_w·ω = τ.
No electrical dynamics (back-EMF, inductance). Adequate for control comparison.

### Auxiliary signals (NOT in state vector)

| Signal      | Size | Unit   | Source        |
|-------------|------|--------|---------------|
| τ_cmd       | 4×1  | N·m    | ESP32 output  |
| τ_applied   | 4×1  | N·m    | After PWM sat |
| IMU_raw     | 6×1  | varies | imu_model     |
| enc_counts  | 4×1  | pulses | encoder_pulse_gen |

---

## 2. Module I/O Specifications (Step 1.2)

### 2.1 RPi5 Cluster

#### plant_step.m
```
function x_new = plant_step(x, tau, params, dt)
% Input:
%   x      [10×1] current state vector
%   tau    [4×1]  applied torques (N·m), after saturation
%   params [struct] from params_mecanum.m
%   dt     [scalar] timestep (s)
% Output:
%   x_new  [10×1] next state vector
```

#### imu_model.m
```
function [accel, gyro, imu_state] = imu_model(x, x_prev, dt, imu_state, params)
% Input:
%   x        [10×1] current state
%   x_prev   [10×1] previous state (for accel calc)
%   dt       [scalar] timestep
%   imu_state [struct] bias states (persistent)
%   params   [struct] noise/bias parameters
% Output:
%   accel    [3×1] accelerometer readings (m/s²) — body frame
%   gyro     [3×1] gyroscope readings (rad/s) — body frame
%   imu_state [struct] updated bias states
```

#### state_manager.m
```
function sm = state_manager(action, sm, varargin)
% Input:
%   action   [string] 'init' | 'update' | 'get' | 'get_prev'
%   sm       [struct] state manager struct
%   varargin — for 'init': (x0, params) | for 'update': (x_new)
% Output:
%   sm       [struct] with fields:
%     .x       [10×1] current state
%     .x_prev  [10×1] previous state
%     .k       [scalar] current timestep index
%     .history [10×N] state history matrix
```

### 2.2 Nucleo H7 Cluster

#### spi_interface.m
```
function [tau_up, states_down] = spi_interface(action, data, params)
% Input:
%   action [string] 'uplink' | 'downlink'
%   data   — uplink: tau [4×1] | downlink: x [10×1]
%   params [struct]
% Output:
%   tau_up      [4×1]  torque received by RPi5 (uplink mode)
%   states_down [10×1] states received by H7 (downlink mode)
% Note: simulates SPI full-duplex pack/unpack, quantization
```

#### encoder_pulse_gen.m
```
function enc_counts = encoder_pulse_gen(omega, dt, params)
% Input:
%   omega  [4×1] wheel angular velocities (rad/s) from plant state
%   dt     [scalar] timestep
%   params [struct] PPR, noise σ
% Output:
%   enc_counts [4×1] encoder pulse counts (integer + noise)
```

#### imu_packet_enc.m
```
function packet = imu_packet_enc(accel, gyro, params)
% Input:
%   accel  [3×1] accelerometer data (m/s²)
%   gyro   [3×1] gyroscope data (rad/s)
%   params [struct] packet format, resolution
% Output:
%   packet [struct] UART-style packet with header, payload, checksum
```

#### pwm_capture.m
```
function tau = pwm_capture(pwm_signal, params)
% Input:
%   pwm_signal [4×1] signed PWM duty cycle [-1, +1] from ESP32
%   params     [struct] max torque, deadband
% Output:
%   tau        [4×1] reconstructed torque command (N·m), signed
```

#### gpio_sync.m
```
function sync_ok = gpio_sync(step_k, cluster_done, params)
% Input:
%   step_k       [scalar] current timestep
%   cluster_done [3×1 logical] flags: [esp32_done, h7_done, rpi5_done]
%   params       [struct] timing constraints
% Output:
%   sync_ok      [logical] true if timing satisfied
```

### 2.3 ESP32 Cluster

#### encoder_reader.m
```
function omega_est = encoder_reader(enc_counts, dt, params)
% Input:
%   enc_counts [4×1] encoder pulse counts from H7
%   dt         [scalar] timestep
%   params     [struct] PPR, filtering
% Output:
%   omega_est  [4×1] estimated wheel velocities (rad/s)
```

#### imu_reader.m
```
function [accel, gyro, valid] = imu_reader(packet, params)
% Input:
%   packet [struct] UART packet from H7
%   params [struct] expected format
% Output:
%   accel  [3×1] accelerometer readings (m/s²)
%   gyro   [3×1] gyroscope readings (rad/s)
%   valid  [logical] checksum pass/fail
```

#### slip_detector.m
```
function [slip_flag, slip_ratio] = slip_detector(omega_est, accel, gyro, params)
% Input:
%   omega_est [4×1] estimated wheel velocities
%   accel     [3×1] accelerometer readings
%   gyro      [3×1] gyroscope readings
%   params    [struct] detection thresholds
% Output:
%   slip_flag  [4×1 logical] per-wheel slip detected
%   slip_ratio [4×1] estimated slip ratio
```

#### pid_controller.m
```
function [tau_cmd, pid_state] = pid_controller(omega_ref, omega_est, pid_state, params)
% Input:
%   omega_ref  [4×1] reference wheel velocities (rad/s)
%   omega_est  [4×1] estimated wheel velocities (rad/s)
%   pid_state  [struct] integral/prev_error states
%   params     [struct] Kp, Ki, Kd, dt
% Output:
%   tau_cmd    [4×1] torque command (N·m)
%   pid_state  [struct] updated states
```

#### adrc_controller.m
```
function [tau_cmd, adrc_state] = adrc_controller(omega_ref, omega_est, imu_data, adrc_state, params)
% Input:
%   omega_ref   [4×1] reference wheel velocities (rad/s)
%   omega_est   [4×1] estimated wheel velocities (rad/s)
%   imu_data    [struct] .accel [3×1], .gyro [3×1]
%   adrc_state  [struct] ESO states
%   params      [struct] ESO gains, control law gains
% Output:
%   tau_cmd     [4×1] torque command (N·m)
%   adrc_state  [struct] updated ESO states
```

#### pwm_output.m
```
function pwm_signal = pwm_output(tau_cmd, params)
% Input:
%   tau_cmd    [4×1] torque command (N·m)
%   params     [struct] tau_max, deadband, PWM resolution
% Output:
%   pwm_signal [4×1] signed PWM duty cycle [-1, +1], saturated
```

---

## 3. Data Flow Between Clusters (Step 1.3)

### Per-timestep sequence (step k)

```
┌─────────────────────────────────────────────────────────────┐
│ STEP k                                                       │
│                                                              │
│  ┌──────────┐                                                │
│  │  ESP32   │ 1. encoder_reader(enc_counts[k-1])→ω_est      │
│  │          │ 2. imu_reader(packet[k-1])→accel,gyro          │
│  │          │ 3. controller(ω_ref,ω_est)→τ_cmd               │
│  │          │ 4. pwm_output(τ_cmd)→pwm_signal                │
│  └────┬─────┘                                                │
│       │ pwm_signal [4×1]                                     │
│       ▼                                                      │
│  ┌──────────┐                                                │
│  │   H7     │ 5. pwm_capture(pwm_signal)→τ                  │
│  │          │ 6. spi_interface('uplink',τ)→forward to RPi5   │
│  └────┬─────┘                                                │
│       │ τ [4×1] via SPI                                      │
│       ▼                                                      │
│  ┌──────────┐                                                │
│  │  RPi5    │ 7. plant_step(x[k-1],τ)→x[k]                  │
│  │          │ 8. imu_model(x[k],x[k-1])→accel,gyro          │
│  │          │ 9. state_manager('update',x[k])                │
│  └────┬─────┘                                                │
│       │ x[k] via SPI downlink                                │
│       ▼                                                      │
│  ┌──────────┐                                                │
│  │   H7     │ 10. spi_interface('downlink',x[k])             │
│  │          │ 11. encoder_pulse_gen(ω[k])→enc_counts[k]      │
│  │          │ 12. imu_packet_enc(accel,gyro)→packet[k]       │
│  └──────────┘     → ready for ESP32 at step k+1             │
│                                                              │
│  gpio_sync validates timing at each transition               │
└─────────────────────────────────────────────────────────────┘
```

### Inter-cluster signal summary

| From   | To   | Signal          | Protocol | Size/Range        |
|--------|------|-----------------|----------|-------------------|
| ESP32  | H7   | pwm_signal      | PWM      | 4×1 float [-1,+1] |
| H7     | RPi5 | τ (uplink)      | SPI      | 4×1 float (N·m)   |
| RPi5   | H7   | x[k] (downlink) | SPI      | 10×1 float        |
| H7     | ESP32| enc_counts      | GPIO/pulse| 4×1 int           |
| H7     | ESP32| imu_packet      | UART     | struct             |

### Trajectory → Wheel reference conversion

`trajectory_generator.m` outputs body-level references `[vx_ref, vy_ref, ωz_ref]`.
Conversion to `ω_ref [4×1]` uses the **inverse kinematics** defined in Section 1
(Mecanum kinematics contract), computed in `run_simulation.m`.

---

## 4. Timing & Simulation Parameters

| Parameter      | Value  | Note                        |
|----------------|--------|-----------------------------|
| dt             | 0.001  | 1 kHz simulation rate       |
| T_sim          | 10     | Default simulation time (s) |
| N_steps        | 10000  | T_sim / dt                  |
| Controller dt  | 0.001  | Same as sim dt (simplification) |
| SPI clock      | N/A    | Simulated as instant in sim |

---

## 5. Coordinate Frames

- **World frame {W}:** Fixed, right-hand. X forward, Y left, Z up.
- **Body frame {B}:** Attached to AGV center. X forward, Y left, Z up.
- **Rotation W→B:** Standard 2D rotation matrix R(θ).
- **Gravity:** g = [0; 0; -9.81] in world frame. IMU at rest measures [0; 0; +9.81] on accelerometer (reaction to gravity).

```
R(θ) = [cos(θ) -sin(θ);
        sin(θ)  cos(θ)]

World velocity = R(θ) * [vx; vy]  (body → world)
```

### IMU frame
IMU is assumed co-located with body center, aligned with body frame.
- accel[1:2] = body-frame acceleration (includes centripetal, Coriolis)
- accel[3] = ≈ +g when stationary (measures reaction to gravity)
- gyro[3] = ωz (yaw rate, same as state x(6))
- gyro[1:2] ≈ 0 for planar motion (roll/pitch rates)
