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
    params.b_w    = 0.001;    % Wheel viscous friction (N·m·s/rad) [ASSUMPTION — not in spec]
    params.g      = 9.81;     % Gravity (m/s²)

    %% --- Motor / Actuator ---
    params.tau_max   = 0.5;   % Max motor torque (N·m) [ASSUMPTION — not in spec]
    params.omega_max = 34.56; % Max wheel speed (rad/s) [from spec]
    params.pwm_res   = 1024;  % PWM resolution (10-bit)
    params.deadband  = 0.02;  % PWM deadband fraction [ASSUMPTION]

    %% --- Encoder ---
    params.enc_ppr     = 1024;  % Pulses per revolution
    params.enc_noise_sigma = 0.02; % Encoder noise std dev (fraction of 1 pulse)

    %% --- IMU ---
    params.imu_accel_noise  = 0.05;    % Accelerometer noise std dev (m/s²)
    params.imu_gyro_noise   = 0.005;   % Gyroscope noise std dev (rad/s)
    params.imu_accel_bias0  = 0.01;    % Initial accel bias (m/s²)
    params.imu_gyro_bias0   = 0.001;   % Initial gyro bias (rad/s)
    params.imu_bias_drift   = 1e-5;    % Bias random walk coefficient

    %% --- PID Controller ---
    params.pid.Kp = 0.5;
    params.pid.Ki = 2.0;
    params.pid.Kd = 0.01;

    %% --- ADRC Controller ---
    params.adrc.b0        = 1 / params.J_w;   % Control gain estimate
    params.adrc.eso_beta1 = 100;               % ESO gain 1
    params.adrc.eso_beta2 = 3000;              % ESO gain 2
    params.adrc.eso_beta3 = 30000;             % ESO gain 3
    params.adrc.kp        = 50;                % State feedback gain
    params.adrc.kd        = 10;                % Derivative feedback gain

    %% --- SPI Interface ---
    params.spi.float_bits = 16;   % Simulated quantization bits
    params.spi.scale_tau  = 1.0;  % Torque scaling factor
    params.spi.scale_state = 1.0; % State scaling factor

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
    params.M_eff = (r_/4)^2 * (K_' * M_body * K_) + params.J_w * eye(4);
    params.M_eff_inv = inv(params.M_eff);

    %% --- State vector info ---
    params.n_states = 10;
    params.n_wheels = 4;
    params.state_names = {'x','y','theta','vx','vy','wz','w1','w2','w3','w4'};

end
