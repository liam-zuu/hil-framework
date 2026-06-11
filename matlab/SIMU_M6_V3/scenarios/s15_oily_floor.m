function s15_oily_floor()
% S15 — Oily floor (grease puddle at 3m mark)
%
% Robot đi sinusoidal ~6m, gặp vũng nhớt tại x≈3m (t=12.5s).
% Trong vùng dầu mỡ: ma sát tĩnh giảm từ 0.8 (dry concrete) xuống 0.10
% (motor oil on concrete) → gần như mọi timestep đều xảy ra slip.
%
% Observable signature:
%   - slip_detector flags nhiều bánh cùng lúc (kinematic inconsistency)
%   - IMU wz không khớp với wz tính từ encoder (body không xoay như bánh)
%   - Position error spike sharp tại t=12.5s, controller phải struggle
%   - PID: integral windup — tích lũy error → gửi max torque → làm slip nặng hơn
%   - ADRC: ESO estimate traction loss vào z2 → giảm torque command → recover nhanh hơn
%
% Grease zone: t = [12.5s, 22.5s] (tương đương x ≈ [3m, 6m] trên trục forward)
% Recovery: T_sim = 26s → thấy thêm 3.5s sau khi ra khỏi vũng dầu
%
% Physical params (oily floor):
%   mu_static  = 0.10  (dry concrete: 0.80, wet: 0.45, oily: 0.10)
%   mu_kinetic = 0.05  (≈ 50% of static, consistent with oil lubrication)
%   prob_spontaneous = 0.85  (85% chance mỗi bánh xe slip mỗi timestep)
%   tau_friction_max = mu_static × (M×g/4) × r
%                    = 0.10 × (4×9.81/4) × 0.0485 ≈ 0.048 N·m
%   → Hầu hết torque commands (~0.05-0.15 N·m) đều vượt friction limit
%
% Note: Slip model hiện tại (M6) couple wheel và body qua M_eff Lagrangian.
% "Free spinning" hoàn toàn (encoder cao, body = 0) cần decoupled dynamics
% — documented là future work. Tuy nhiên kinematic inconsistency vẫn rõ
% qua slip_detector vì 4 bánh slip với noise_factor khác nhau.

    params = params_mecanum();

    %% Trajectory: sinusoidal, v=0.3 m/s, ~6m total forward distance
    % Wavelength = v/f = 0.3/0.15 = 2m → 3 sóng trên 6m
    spec = struct();
    spec.type      = 'sinusoidal';
    spec.amplitude = 0.40;   % m (biên độ ngang)
    spec.frequency = 0.15;   % Hz
    spec.v         = 0.30;   % m/s (tốc độ forward)
    spec.t_hold    = 0.50;   % s (hold trước khi bắt đầu)
    spec.t_ramp    = 2.00;   % s (ramp lên full speed)
    % Cruise bắt đầu tại t = t_hold + t_ramp = 2.5s
    % Khoảng cách forward sau t giây cruise: 0.3 × t
    % 3m → t_cruise = 10s → t_grease_start = 2.5 + 10 = 12.5s
    % 6m → t_cruise = 20s → t_grease_end   = 2.5 + 20 = 22.5s

    T_sim = 26;   % 3.5s recovery sau khi ra khỏi vũng dầu
    seed  = 42;

    %% Grease zone fault config
    slip_oily = struct(...
        'enabled',          true,  ...
        'mu_static',        0.10,  ...   % motor oil on concrete
        'mu_kinetic',       0.05,  ...
        'prob_spontaneous', 0.85,  ...   % near-constant slip
        'noise_sigma',      0.25,  ...   % chaotic kinetic friction variation
        'detect_threshold', 0.15,  ...   % same as nominal
        'imu_wz_threshold', 0.50);       % same as nominal

    params.fault.grease_zone = struct(...
        'enabled', true,        ...
        't_start', 12.5,        ...   % ≈ 3m vào cruise
        't_end',   22.5,        ...   % ≈ 6m (hết vùng dầu)
        'slip_oily', slip_oily);

    % Background slip vẫn giữ default OFF (params.slip.enabled = false)
    % Grease zone hook sẽ override params.slip chỉ trong [t_start, t_end]

    fprintf('\n=== S15: OILY FLOOR ===\n');
    fprintf('Trajectory : sinusoidal, A=%.2fm, f=%.2fHz, v=%.1fm/s\n', ...
            spec.amplitude, spec.frequency, spec.v);
    fprintf('Grease zone: t=[%.1f, %.1f]s (≈x=[3m, 6m])\n', ...
            params.fault.grease_zone.t_start, params.fault.grease_zone.t_end);
    fprintf('mu_static  : %.2f (oily floor, vs %.2f dry concrete)\n', ...
            slip_oily.mu_static, 0.80);
    fprintf('tau_friction_max ≈ %.3f N·m (plant tau_max = %.2f N·m)\n', ...
            slip_oily.mu_static * (params.M * 9.81 / 4) * params.r, ...
            params.tau_max);

    run_scenario(mfilename, spec, params, T_sim, seed);
end
