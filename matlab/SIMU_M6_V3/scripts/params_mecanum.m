function params = params_mecanum()
% PARAMS_MECANUM  All physical and simulation parameters for HIL simulation.
%
% Usage: params = params_mecanum();

    %% --- Simulation ---
    params.dt     = 0.001;    % Timestep (s) — 1 kHz
    params.T_sim  = 10;       % Default simulation duration (s)

    %% --- Mechanical (AGV Mecanum) ---
    params.r      = 0.0485;   % Wheel radius (m) [from spec]
    params.lx     = 0.12;     % Longitudinal half-distance: center→wheel along body X (m) [from spec]
    params.ly     = 0.12;     % Lateral half-distance: center→wheel along body Y (m) [from spec]
    params.M      = 4;        % Total mass (kg) [from spec]
    params.Iz     = (1/12) * 4 * ((2*0.12)^2 + (2*0.12)^2);  % Body yaw inertia (kg·m²) = 0.0384 [calculated: rectangular body]
    params.J_w    = 0.00247;  % Wheel inertia (kg·m²) [from spec]
    params.b_w    = 0.002;    % Wheel viscous friction (N·m·s/rad) [ASSUMPTION — not in spec]
    params.g      = 9.81;     % Gravity (m/s²)

    %% --- Motor / Actuator ---
    params.tau_max   = 0.5;   % Max motor torque (N·m) [ASSUMPTION — not in spec]
    params.omega_max = 34.56; % Max wheel speed (rad/s) [from spec]
    % Motor wiring polarity [FL FR RL RR]: sign of real wheel rotation for a positive
    % command as H7 reads it. S01 run01 (05_DATA/results/S01/2026-10-20_run01): every
    % wheel turns backward for +PWM with firmware motor_dir = +1. [measured]
    params.motor_polarity = [-1; -1; -1; -1];
    % Firmware per-wheel command sign (AGV_FIRMWARE src/motor/motor.c motor_dir),
    % applied in esp32/pwm_output.m. motor_dir .* motor_polarity = +1: wheels turn forward.
    params.motor_dir      = [-1; -1; -1; -1];

    %% --- DC motor model (spec S01 run01, 21/10/2026) ---
    % 'dc'    : plant command = PWM duty u in [-1,1];
    %           tau_m = K/R*(V*u_eff - K*omega), u_eff = sign(u)*max(|u|-u0,0);
    %           friction b*omega + Tc*sign(omega), static breakaway Ts.
    % 'torque': legacy M6 model (command = torque, viscous b_w only).
    % Per-wheel values [FL FR RL RR], direction-averaged, from
    % 05_DATA/results/S01/2026-10-20_run01/identification/plant_params_from_S01.csv
    % (matlab/identification/s01_derive_plant_params.py). R measured with a
    % multimeter at the driver terminals, about +-15-25 % uncertain; J, b scale with 1/R.
    params.motor_model  = 'dc';
    params.motor.K      = [0.30394; 0.32210; 0.30917; 0.28811];   % [V.s/rad] [measured]
    params.motor.R      = [3.31; 3.38; 3.73; 3.95];               % [ohm] [measured]
    params.motor.u0     = [0.05223; 0.03170; 0.03574; 0.05413];   % duty offset [measured]
    params.motor.Tc     = [0.02394; 0.02738; 0.02347; 0.02433];   % Coulomb [N.m] [measured]
    params.motor.Ts     = [0.11878; 0.14760; 0.10489; 0.08129];   % static breakaway [N.m] [measured]
    params.motor.b      = [0.00062; 0.00081; 0.00074; 0.00175];   % viscous [N.m.s/rad] [measured]
    params.J_wheels     = [0.00209; 0.00236; 0.00204; 0.00249];   % [kg.m^2] [measured]
    params.V_batt       = 12.4;    % [V] S01 run01 mean 12.435 V [measured]
    params.motor_enable = 1;       % 1 = driver enabled, 0 = EN low (free coast)
    params.wheels_lifted = false;  % true = wheels off the ground (S01 bench)
    params.pwm_res   = 1024;  % PWM resolution (10-bit)
    params.deadband  = 0.02;  % PWM deadband fraction [ASSUMPTION]

    %% --- Encoder ---
    params.enc_ppr     = 1320;  % Counts per wheel revolution, 4x decoding [from spec: GB37-520, 11 PPR x 30:1 = 330 pulses/ch/rev; x4 = 1320; matches H7 ENC_PPR=330 and ESP32 PPR=1320] (was 1024 until 2026-10-20)
    params.enc_noise_sigma = 0.02; % Encoder noise std dev (fraction of 1 pulse)
    params.enc_filter_tau  = 0.005; % Encoder reader low-pass filter time constant (s)

    %% --- IMU ---
    % Noise std per IMU sample at imu_rate_hz. [measured] BNO085 on the AGV, standing still,
    % 600 samples / 30 s, TEST_IMU test 6, 11/06/2026 (recorded only in
    % 00_OVERVIEW/Session_Logs/2026-06-11_Session_A_He_thong_hoa_code.md, no raw file yet).
    % To be replaced by the raw IMU noise run (tools/imu_capture.py, 05_DATA/results/IMU_NOISE/).
    % Before 21/10/2026: accel 0.05, gyro 0.005, updated every 1 ms (see use_torque_model.m).
    params.imu_accel_noise  = 0.01254; % Accelerometer noise std dev (m/s²) [measured, accel_z]
    params.imu_gyro_noise   = 0.00031; % Gyroscope noise std dev (rad/s) [measured]
    params.imu_rate_hz      = 20;      % Output rate (Hz): imu.cpp enables reports at 50 ms; held between samples [from firmware]
    params.imu_accel_bias0  = 0.01;    % Initial accel bias (m/s²)
    params.imu_gyro_bias0   = 0.001;   % Initial gyro bias (rad/s)
    params.imu_bias_drift   = 1e-5;    % Bias random walk coefficient
    params.imu_filter_tau   = 0.003;   % IMU reader low-pass filter time constant (s)
    params.imu_outlier_accel = 50;     % Outlier rejection threshold for accel (m/s²)
    params.imu_outlier_gyro  = 20;     % Outlier rejection threshold for gyro (rad/s)

    %% --- PID Controller ---
    % Tuned via systematic sweep (tune_gains.m)
    % Inner BW ≈ 11 rad/s (1.7 Hz), settling ~0.36s
    % At startup error 20 rad/s: τ_cmd = 0.8 N·m → brief saturation, OK with anti-windup
    params.pid.Kp = 0.04;
    params.pid.Ki = 0.5;
    params.pid.Kd = 0.0004;

    %% --- ADRC Controller ---
    % Tuned via systematic sweep + noise diagnosis
    % Controller BW ω_c = kp = 20 rad/s (3.2 Hz)
    % ESO BW ω_o = 100 rad/s (reduced from 200: ω_o=200 amplifies encoder noise
    %   on low-speed trajectories with only +0.1mm impact on circle)
    % ESO/controller ratio = 5× (balanced observation vs noise rejection)
    params.adrc.b0        = 1 / params.J_w;   % Control gain estimate (~405)
    params.adrc.eso_beta1 = 200;               % ESO gain 1 = 2*w_o
    params.adrc.eso_beta2 = 10000;             % ESO gain 2 = w_o^2
    params.adrc.kp        = 20;                % State feedback gain = w_c

    %% --- SPI Interface ---
    params.spi.float_bits = 16;   % Simulated quantization bits for fixed-point
    params.spi.tau_range   = 1.0;  % Full-scale range for torque channel (N·m)
    params.spi.state_ranges = [5; 5; pi; 3; 3; 10; ...
                               40; 40; 40; 40];  % Full-scale range per state

    %% --- IMU ADC (on H7, encoding for UART) ---
    params.imu_adc_bits   = 16;              % ADC resolution
    params.imu_accel_range = 4 * 9.81;       % ±4g full-scale (m/s²)
    params.imu_gyro_range  = 2000 * pi/180;  % ±2000 deg/s full-scale (rad/s)

    %% --- PWM Capture ---
    params.pwm_jitter_sigma = 0.001;  % PWM capture timing jitter (fraction of duty cycle)

    %% --- GPIO Sync ---
    params.sync_jitter_us   = 5;      % Timing jitter std dev (microseconds)
    params.sync_timeout_us  = 50;     % Sync timeout threshold (microseconds)

    %% --- Kinematics matrices ---
    % Forward kinematics: v_body = H_fwd * omega_wheel
    r_ = params.r; L_ = params.lx + params.ly;
    params.H_fwd = (r_/4) * [1  1  1  1;
                              -1  1  1 -1;
                              -1/L_  1/L_  -1/L_  1/L_];  % 3×4

    % Inverse kinematics: omega_wheel = H_inv * v_body
    params.H_inv = (1/r_) * [1 -1 -L_;
                              1  1  L_;
                              1  1 -L_;
                              1 -1  L_];  % 4×3

    %% --- Effective inertia (Lagrangian, no-slip) ---
    % M_eff = (r/4)^2 * K^T * diag(M,M,Iz) * K + J_w * I_4
    % K = H_fwd / (r/4) = [1 1 1 1; -1 1 1 -1; -1/L 1/L -1/L 1/L]
    K_ = [1  1  1  1;
         -1  1  1 -1;
         -1/L_  1/L_  -1/L_  1/L_];
    M_body = diag([params.M, params.M, params.Iz]);
    M_body_wheel = (r_/4)^2 * (K_' * M_body * K_);
    params.M_eff_torque = M_body_wheel + params.J_w * eye(4);          % legacy model
    params.M_eff_dc     = M_body_wheel + diag(params.J_wheels);       % DC model, on ground
    params.M_eff_lifted = diag(params.J_wheels);                      % DC model, wheels lifted
    params.M_eff_torque_inv = inv(params.M_eff_torque);
    params.M_eff_dc_inv     = inv(params.M_eff_dc);
    params.M_eff_lifted_inv = inv(params.M_eff_lifted);
    params.M_eff     = params.M_eff_torque;       % kept for code that reads M_eff (ADRC b0, scripts)
    params.M_eff_inv = params.M_eff_torque_inv;

    %% --- Position Controller (outer loop) ---
    % Tuned via systematic sweep (tune_gains.m)
    % Outer BW ~1.0 Hz (Kp_pos=6.0), Ti_pos = 2.0s, Ti_theta = 1.3s
    % BW ratios: ADRC_inner/outer = 3.3×, PID_inner/outer = 1.8×
    params.pos_ctrl.Kp_pos   = 6.0;    % Position proportional gain (1/s)
    params.pos_ctrl.Ki_pos   = 3.0;    % Position integral gain (1/s²)
    params.pos_ctrl.Kp_theta = 8.0;    % Heading proportional gain (1/s)
    params.pos_ctrl.Ki_theta = 6.0;    % Heading integral gain (1/s²)
    params.pos_ctrl.vx_max   = 1.5;    % Max forward velocity command (m/s)
    params.pos_ctrl.vy_max   = 1.5;    % Max lateral velocity command (m/s)
    params.pos_ctrl.wz_max   = 5.0;    % Max yaw rate command (rad/s)

    %% --- Wheel Slip Model (M6) ---
    % Physics: F_N = M*g/4 per wheel, tau_max_friction = mu * F_N * r
    % At nominal: tau_max_friction = 0.8 * (4*9.81/4) * 0.0485 = 0.381 N·m
    % Since tau_max = 0.5 N·m > 0.381 → slip CAN occur at high torque
    params.slip.enabled          = false;  % default OFF, M6 scripts enable explicitly
    params.slip.mu_static        = 0.8;    % static friction coefficient (dry concrete)
    params.slip.mu_kinetic       = 0.5;    % kinetic friction during slip (~63% of static)
    params.slip.prob_spontaneous = 0.002;  % probability of random slip per wheel per step
    params.slip.noise_sigma      = 0.15;   % stochastic variation of kinetic friction (σ)
    params.slip.detect_threshold = 0.15;   % slip ratio threshold for detection (15%)
    params.slip.imu_wz_threshold = 0.5;    % IMU yaw rate mismatch threshold (rad/s)

    %% --- Load Disturbance (M6) ---
    % External torque disturbance applied to wheels (step/ramp/random)
    % Applied in simulation loop, not inside plant_step
    params.disturbance.enabled = false;  % default OFF
    params.disturbance.type    = 'none'; % 'step' | 'ramp' | 'random' | 'combined'
    params.disturbance.magnitude = 0.05; % N·m (10% of tau_max)
    params.disturbance.start_time = 3.0; % when disturbance begins (s)
    params.disturbance.ramp_rate  = 0.02; % N·m/s for ramp disturbance
    params.disturbance.random_sigma = 0.03; % σ for random disturbance (N·m)

    %% --- State vector info ---
    params.n_states = 10;
    params.n_wheels = 4;
    params.state_names = {'x','y','theta','vx','vy','wz','w1','w2','w3','w4'};

end
