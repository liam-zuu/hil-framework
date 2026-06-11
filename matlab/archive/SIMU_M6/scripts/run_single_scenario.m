function res = run_single_scenario(ctrl_type, traj_type, params)
% RUN_SINGLE_SCENARIO  Run one full closed-loop simulation and return metrics.
%
% Encapsulates the complete simulation loop from run_simulation.m into
% a callable function. Used by run_m6_disturbance.m to sweep across
% multiple conditions (slip, noise, disturbance) systematically.
%
% Input:
%   ctrl_type  [string] 'pid' or 'adrc'
%   traj_type  [string] 'line' | 'circle' | 'square' | 'figure8'
%   params     [struct] from params_mecanum (may have modified fields)
% Output:
%   res        [struct] with metrics:
%     .rms_pos_full   — RMS position error, full run (mm)
%     .rms_pos_ss     — RMS position error, steady-state last 50% (mm)
%     .max_pos        — max position error (mm)
%     .rms_theta      — RMS heading error (deg)
%     .rms_theta_ss   — RMS heading error, steady-state (deg)
%     .max_tau        — max |torque command| (N·m)
%     .sat_pct        — saturation percentage (%)
%     .sync_fails     — sync failure count
%     .slip_events    — total slip detections (if slip_detector active)
%     .mean_slip_ratio— mean |slip_ratio| when slip detected

    % Reset persistent states
    clear encoder_pulse_gen encoder_reader imu_reader position_controller;

    dt    = params.dt;
    T_sim = params.T_sim;
    N     = round(T_sim / dt);

    r = params.r;
    L = params.lx + params.ly;

    %% Generate trajectory
    traj = trajectory_generator(traj_type, T_sim, dt, params);

    %% Initialize state
    x0 = zeros(params.n_states, 1);
    x0(1) = traj.x_ref(1);
    x0(2) = traj.y_ref(1);
    x0(3) = traj.theta_ref(1);

    sm = state_manager('init', [], x0, params);

    pid_state  = [];
    adrc_state = [];
    imu_state  = [];
    pe_state   = [];
    pe_state.x = x0(1); pe_state.y = x0(2); pe_state.theta = x0(3);

    %% Initialize H7 outputs
    omega_init = x0(7:10);
    enc_counts = encoder_pulse_gen(omega_init, dt, params);
    [accel_init, gyro_init, imu_state] = imu_model(x0, x0, dt, imu_state, params);
    imu_packet = imu_packet_enc(accel_init, gyro_init, params);

    %% Logging
    pos_err_vec   = zeros(1, N);
    theta_err_vec = zeros(1, N);
    tau_cmd_log   = zeros(4, N);
    sync_fail_count = 0;
    slip_event_count = 0;
    slip_ratio_sum   = 0;
    slip_ratio_n     = 0;

    %% Main simulation loop
    for k = 1:N
        t_now = (k-1) * dt;

        %% ===== CLUSTER 1: ESP32 =====
        % 1. Read sensors
        omega_est = encoder_reader(enc_counts, dt, params);
        [accel_meas, gyro_meas, ~] = imu_reader(imu_packet, params);
        imu_data.accel = accel_meas;
        imu_data.gyro  = gyro_meas;

        % 2. Pose estimation
        [pose_est, pe_state] = pose_estimator(omega_est, gyro_meas, pe_state, params);

        % 3. Slip detection (M6)
        [slip_flag, slip_ratio] = slip_detector(omega_est, accel_meas, gyro_meas, params);
        if any(slip_flag)
            slip_event_count = slip_event_count + sum(slip_flag);
            slip_ratio_sum = slip_ratio_sum + sum(abs(slip_ratio(slip_flag)));
            slip_ratio_n = slip_ratio_n + sum(slip_flag);
        end

        % 4. Outer loop
        pose_ref = [traj.x_ref(k); traj.y_ref(k); traj.theta_ref(k)];
        vel_ref  = [traj.vx_ref(k); traj.vy_ref(k); traj.wz_ref(k)];
        vel_cmd = position_controller(pose_ref, pose_est, vel_ref, params);

        % 5. Inverse kinematics
        omega_ref = (1/r) * [vel_cmd(1) - vel_cmd(2) - L*vel_cmd(3);
                             vel_cmd(1) + vel_cmd(2) + L*vel_cmd(3);
                             vel_cmd(1) + vel_cmd(2) - L*vel_cmd(3);
                             vel_cmd(1) - vel_cmd(2) + L*vel_cmd(3)];

        % 6. Inner loop
        switch ctrl_type
            case 'pid'
                [tau_cmd, pid_state] = pid_controller(omega_ref, omega_est, pid_state, params);
            case 'adrc'
                [tau_cmd, adrc_state] = adrc_controller(omega_ref, omega_est, imu_data, adrc_state, params);
        end

        % 7. PWM output
        pwm_signal = pwm_output(tau_cmd, params);

        %% ===== CLUSTER 2: H7 (uplink) =====
        tau = pwm_capture(pwm_signal, params);
        [tau_up, ~] = spi_interface('uplink', tau, params);

        %% ===== Load torque disturbance (M6) =====
        % Applied AFTER H7 pipeline, BEFORE plant_step
        % This simulates external load on wheels (friction change, slope, payload shift)
        if isfield(params, 'disturbance') && params.disturbance.enabled
            tau_dist = compute_disturbance(t_now, k, params);
            tau_up = tau_up + tau_dist;
        end

        %% ===== CLUSTER 3: RPi5 =====
        x_cur = sm.x;
        x_new = plant_step(x_cur, tau_up, params, dt);
        [accel_sim, gyro_sim, imu_state] = imu_model(x_new, x_cur, dt, imu_state, params);
        sm = state_manager('update', sm, x_new);

        %% ===== CLUSTER 2: H7 (downlink) =====
        [~, states_h7] = spi_interface('downlink', x_new, params);
        omega_plant = states_h7(7:10);
        enc_counts = encoder_pulse_gen(omega_plant, dt, params);
        imu_packet = imu_packet_enc(accel_sim, gyro_sim, params);

        cluster_done = [true; true; true];
        sync_ok = gpio_sync(k, cluster_done, params);
        if ~sync_ok, sync_fail_count = sync_fail_count + 1; end

        %% Logging
        pos_err_vec(k) = sqrt((x_new(1) - traj.x_ref(k))^2 + ...
                               (x_new(2) - traj.y_ref(k))^2);
        e_th = x_new(3) - traj.theta_ref(k);
        theta_err_vec(k) = mod(e_th + pi, 2*pi) - pi;
        tau_cmd_log(:,k) = tau_cmd;
    end

    %% Compute metrics
    ss_start = round(N/2);  % steady-state = last 50%

    res.rms_pos_full   = sqrt(mean(pos_err_vec.^2)) * 1000;                    % mm
    res.rms_pos_ss     = sqrt(mean(pos_err_vec(ss_start:end).^2)) * 1000;      % mm
    res.max_pos        = max(pos_err_vec) * 1000;                     % mm
    res.rms_theta      = sqrt(mean(theta_err_vec.^2)) * 180/pi;                 % deg
    res.rms_theta_ss   = sqrt(mean(theta_err_vec(ss_start:end).^2)) * 180/pi;   % deg
    res.max_tau        = max(abs(tau_cmd_log(:)));                     % N·m
    res.sat_pct        = sum(abs(tau_cmd_log(:)) >= params.tau_max*0.99) / numel(tau_cmd_log) * 100;
    res.sync_fails     = sync_fail_count;
    res.slip_events    = slip_event_count;
    if slip_ratio_n > 0
        res.mean_slip_ratio = slip_ratio_sum / slip_ratio_n;
    else
        res.mean_slip_ratio = 0;
    end

end


%% ===== LOCAL FUNCTION: Compute disturbance torque =====
function tau_dist = compute_disturbance(t_now, k, params)
% COMPUTE_DISTURBANCE  Generate load torque disturbance based on type.
%
% Returns [4x1] disturbance torque vector (N·m) applied to all wheels.

    tau_dist = zeros(4, 1);
    cfg = params.disturbance;

    if t_now < cfg.start_time
        return;  % disturbance hasn't started yet
    end

    dt_since = t_now - cfg.start_time;

    switch cfg.type
        case 'step'
            % Sudden constant load on all wheels
            tau_dist = cfg.magnitude * ones(4, 1);

        case 'ramp'
            % Linearly increasing load
            tau_dist = min(cfg.magnitude, cfg.ramp_rate * dt_since) * ones(4, 1);

        case 'random'
            % Random white noise load
            tau_dist = cfg.random_sigma * randn(4, 1);

        case 'combined'
            % Step + ramp + random simultaneously
            step_part = cfg.magnitude * 0.5;
            ramp_part = min(cfg.magnitude * 0.3, cfg.ramp_rate * dt_since);
            rand_part = cfg.random_sigma * randn(4, 1);
            tau_dist = (step_part + ramp_part) * ones(4,1) + rand_part;

        otherwise
            tau_dist = zeros(4, 1);
    end

end
