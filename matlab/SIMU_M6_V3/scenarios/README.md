# HIL Scenarios — PID vs ADRC Comparison

Bộ 14 test scenarios để so sánh phản ứng của hai controller (PID và ADRC) dưới các điều kiện khác nhau. Mỗi scenario tự chạy cả hai controller trên cùng reference trajectory và cùng random seed — đảm bảo so sánh công bằng.

## Cách cài đặt

Copy thư mục `scenarios/` vào thư mục gốc của project M6_v3 (cùng cấp với `scripts/`, `esp32/`, `nucleoh7/`, `rpi5/`). Cấu trúc sẽ là:

```
SIMU_M6/
├── scripts/            (v1 files, không đụng tới)
├── esp32/              (v1 files, không đụng tới)
├── nucleoh7/           (v1 files, không đụng tới)
├── rpi5/               (v1 files, không đụng tới)
├── results/            (v1 output)
├── docs/               (v1 docs)
└── scenarios/          ← COPY VÀO ĐÂY
    ├── s01_line_baseline.m
    ├── ...
    └── results/        (scenario outputs)
```

**Không có file nào trong `scripts/`, `esp32/`, `nucleoh7/`, `rpi5/` bị thay đổi.** Scenarios framework **tuân thủ** signatures và loop logic của v1 — chỉ thêm wrapper layer bên ngoài.

## Quick Start

```matlab
% Chạy 1 scenario
cd SIMU_M6/scenarios
s01_line_baseline

% Chạy tất cả 14 scenarios (~15-30 phút tùy máy)
run_all_scenarios

% Chỉ chạy các scenario không có fault
run_all_scenarios('tracking')

% Chỉ chạy fault scenarios
run_all_scenarios('faults')

% Bỏ qua các scenario chạy lâu (R=2m circle, racetrack)
run_all_scenarios('skip_slow')
```

Sau khi chạy, tất cả output nằm trong `scenarios/results/`:
- `<scenario>.png` — dashboard 4×2 panel so sánh PID vs ADRC
- `<scenario>.mat` — data thô để plot/phân tích thêm
- `<scenario>.txt` — báo cáo text ngắn (dễ copy)
- `summary_table.txt`, `summary_table.csv` — bảng tổng hợp
- `summary_bars.png` — bar chart tổng hợp
- `summary_report.pdf` — PDF gộp tất cả (MATLAB R2020a+)

## Danh sách scenarios

### Tracking (10 scenarios) — Controller performance trên đường sạch

Không có fault, không có disturbance. Test cơ bản để xác lập noise floor và xem controller xử lý các dạng trajectory khác nhau như thế nào.

| # | Scenario | Trajectory | Mục đích |
|---|----------|-----------|----------|
| 01 | `s01_line_baseline` | Đường thẳng, v=0.3 m/s | Baseline đơn giản nhất |
| 02 | `s02_circle_R05` | Vòng tròn R=0.5m, T=20s | Low curvature |
| 03 | `s03_circle_R10` | Vòng tròn R=1.0m, T=30s | Curvature trung bình (M5.2 baseline) |
| 04 | `s04_circle_R20` | Vòng tròn R=2.0m, T=60s | Low demand, long run — test drift |
| 05 | `s05_square` | Hình vuông cạnh 1m, v=0.3 | Corner handling — 4 góc vuông sắc |
| 06 | `s06_rounded_square` | Vuông bo góc R=0.2m | So với s05 — impact của smooth corner |
| 07 | `s07_figure8` | Figure-8 kích thước 1m, T=8s | High-demand, curvature biến đổi |
| 08 | `s08_zigzag` | Zigzag A=0.3m, λ=1m | Đảo chiều heading liên tục |
| 09 | `s09_sinusoidal` | Sóng sin A=0.5, f=0.2Hz | Curvature biến thiên trơn |
| 10 | `s10_racetrack` | Oval L=3m, W=1m | Mixed straight + arc |

### Fault scenarios (3 scenarios) — Test từng loại fault riêng

Thêm **một** loại fault duy nhất vào trajectory circle R=1m hoặc figure-8 để isolate impact của fault đó.

| # | Scenario | Fault | Điểm test |
|---|----------|-------|-----------|
| 11 | `s11_wheel_jam` | Bánh 1 tăng ma sát nhớt ×5 từ t=3.5s | Stuck bearing / wrapped debris. **ADRC lợi thế**: ESO estimate torque loss và bù trực tiếp vào z2, PID phải chờ integral tích lũy |
| 12 | `s12_encoder_dropout` | Bánh 2 đọc 0 trong khoảng t=[4.0, 4.5]s | Lỏng giắc encoder. Test fault tolerance — controller có panic không khi thấy 1 bánh báo dừng? |
| 13 | `s13_battery_fade` | tau_max giảm tuyến tính 0.5→0.3 N·m từ t=3s→t=10s | Pin sụt áp / driver quá nhiệt. Test **anti-windup** |

### Compound fault (1 scenario)

| # | Scenario | Mô tả |
|---|----------|-------|
| 14 | `s14_industrial_nightmare` | Fault onset tuần tự: mass bias → slip → battery fade → load disturbance. Cấu hình trong `nightmare_config.m` |

## Cách đọc dashboard

Mỗi scenario sinh file `<scenario>.png` với layout 4×2:

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ XY Trajectory│ Position err │ Heading err  │ Vel command  │
│ (ref + 2 ctrl)│ vs time      │ vs time      │ outer loop   │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Wheel speeds │ Torque cmd   │ Body veloc.  │ Metrics table│
│ (W1 và W2)   │ (W1)         │ (plant actual)│             │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

**Màu convention:** PID = xanh, ADRC = đỏ, Reference = đen đứt nét.

**Metrics table** (góc dưới phải) gồm:
- `RMS full` — RMS position error trên toàn bộ simulation (bao gồm startup transient)
- `RMS SS` — RMS position error ở steady-state (sau ramp + 2s buffer)
- `Peak err` — peak position error tức thời
- `t_peak` — thời điểm đạt peak
- `RMS θ` — RMS heading error
- `Settle` — thời gian từ khi hết ramp đến khi error < 10mm
- `Max τ` — torque command max
- `Sat %` — % thời gian controller saturated (|τ| > 98% tau_max)
- `Slip events` — số lần slip_detector flag

## Ghi chú thiết kế

### Composite ramp
Tất cả trajectories dùng **composite ramp**: hold 0.5s + ramp 2.5s + cruise. Tránh velocity step tại t=0 gây infinite acceleration reference (industrial standard practice cho AGV thật).

### Seed cố định
Mọi scenario dùng `seed = 42` — kết quả reproducible. Random noise giống nhau giữa PID và ADRC run → so sánh công bằng, không bị random variance.

### Fault injection architecture
Fault logic inline trong `run_single_scenario_v2.m` (local functions `inject_enc_dropout`, `inject_torque_faults`). Không đụng vào `plant_step.m`, `encoder_pulse_gen.m`, `pwm_capture.m` hoặc controller code của v1. Cho phép:
- Plant validation từ M6 vẫn pass
- Bật/tắt fault độc lập qua `params.fault.<type>.enabled`
- Compound fault = bật nhiều flag cùng lúc
- Tương thích với M6 load disturbance (`params.disturbance.*`) — cả hai có thể chạy song song

### Fault types hỗ trợ

```matlab
% Wheel jam: bánh xe bị kẹt thêm ma sát nhớt
params.fault.wheel_jam = struct( ...
    'enabled', true, 'wheel', 1, ...
    'b_extra', 5*params.b_w, 't_start', 3.0);

% Encoder dropout: bánh xe mất tín hiệu encoder trong khoảng time
params.fault.enc_dropout = struct( ...
    'enabled', true, 'wheel', 2, ...
    't_start', 4.0, 't_end', 4.5);

% Battery fade: tau_max giảm tuyến tính
params.fault.battery_fade = struct( ...
    'enabled', true, ...
    't_start', 3.0, 't_end', 10.0, ...
    'tau_max_nominal', 0.5, 'tau_max_final', 0.3);

% Mass bias: drag torque constant (simulates load onset)
params.fault.mass_bias = struct( ...
    'enabled', true, ...
    't_start', 4.0, 'tau_bias', 0.04);
```

### s14 Nightmare — limitation
Scenario s14 có một số simplification do framework không hỗ trợ mid-sim structural changes:
- **Mass change mid-sim** → simulate bằng constant load bias torque (M_eff precomputed, không thể thay đổi mid-sim)
- **Encoder PPR change mid-sim** → không implement (persistent state trong encoder_pulse_gen). Compensate bằng amplified load disturbance.
- **Wheel slip time gating** → plant slip model global enable, `t_start` chỉ là informational. Slip CÓ THỂ trigger trước t_start nếu torque vượt friction limit.

Giới hạn này được ghi chú trong comment của `s14_industrial_nightmare.m`. Kết quả vẫn có giá trị thesis vì mục tiêu là show compound degradation, không phải exact timeline.

## File structure

```
scenarios/
├── README.md                         ← file này
├── composite_ramp.m                  ← helper: 3-phase ramp profile
├── trajectory_generator_v2.m         ← 10 trajectory types, struct-spec
├── metrics_compute.m                 ← compute RMS, peak, settle, sat%
├── run_single_scenario_v2.m          ← main sim loop (matches v1 signatures)
├── run_scenario.m                    ← common runner (boilerplate)
├── scenario_setup_paths.m            ← path setup helper (adds rpi5/, nucleoh7/, esp32/)
├── plot_comparison.m                 ← 4×2 dashboard plotter
├── nightmare_config.m                ← config for s14
├── s01_line_baseline.m               ← each scenario: ~15 lines
├── s02_circle_R05.m
├── ... (s03-s13)
├── s14_industrial_nightmare.m        ← compound fault
├── run_all_scenarios.m               ← master runner
├── generate_summary_report.m         ← PDF/CSV/TXT summary
└── results/                          ← auto-generated outputs
    ├── s01_line_baseline.png/.mat/.txt
    ├── ...
    ├── all_results.mat
    ├── summary_table.txt
    ├── summary_table.csv
    ├── summary_bars.png
    └── summary_report.pdf
```

## Thêm scenario mới

Template:

```matlab
function sXX_my_scenario()
    scenario_setup_paths();
    params = params_mecanum();

    spec = struct();
    spec.type   = 'circle';   % hoặc 'line','square','figure8','zigzag',...
    spec.R      = 1.5;
    spec.period = 25;
    spec.t_hold = 0.5;
    spec.t_ramp = 2.5;

    % (Optional) Fault injection
    params.fault.wheel_jam = struct( ...
        'enabled', true, 'wheel', 1, ...
        'b_extra', 3*params.b_w, 't_start', 3.0);

    T_sim = 20;
    seed  = 42;
    run_scenario(mfilename, spec, params, T_sim, seed);
end
```

Không cần khai báo ở đâu khác — `run_all_scenarios.m` chỉ cần thêm tên file vào list.

## Blind test (self-written)

Framework hỗ trợ blind test (không biết controller nào đang chạy). Chỉ cần truyền function handle thay vì string `'pid'` / `'adrc'` vào `run_single_scenario_v2`. Implementation để bạn tự viết.

## Verification notes

Các signature sau đã được verify khớp với M6_v3 v1:

| Module | Signature sử dụng |
|--------|-------------------|
| `encoder_reader(enc_counts, dt, params)` | ✓ |
| `imu_reader(packet, params)` returns `[accel, gyro, valid]` | ✓ |
| `pose_estimator(omega_est, gyro_meas, pe_state, params)` | ✓ |
| `slip_detector(omega_est, accel, gyro, params)` | ✓ |
| `position_controller(pose_ref, pose_est, vel_ref, params)` | ✓ |
| `pid_controller(omega_ref, omega_est, pid_state, params)` | ✓ |
| `adrc_controller(omega_ref, omega_est, imu_data, adrc_state, params)` | ✓ |
| `pwm_output(tau_cmd, params)` | ✓ |
| `pwm_capture(pwm_signal, params)` | ✓ |
| `spi_interface(action, data, params)` (uplink + downlink) | ✓ |
| `plant_step(x, tau, params, dt)` | ✓ |
| `imu_model(x, x_prev, dt, imu_state, params)` | ✓ |
| `state_manager(action, sm, varargin)` | ✓ |
| `encoder_pulse_gen(omega, dt, params)` | ✓ |
| `imu_packet_enc(accel, gyro, params)` | ✓ |
| `gpio_sync(step_k, cluster_done, params)` | ✓ |
