function [metrics, log] = run_single_scenario_v2(ctrl_type, traj, params, seed)
% RUN_SINGLE_SCENARIO_V2  Scenario-framework wrapper around the M6_v3 sim loop.
%
% Differences from v1 (scripts/run_single_scenario.m):
%   - Accepts a pre-generated trajectory struct (from trajectory_generator_v2)
%     instead of generating internally via trajectory_generator(type, T, dt).
%     Lets scenarios define custom trajectories (R, period, zigzag, ...).
%   - Adds FAULT INJECTION HOOKS via params.fault.* (wheel_jam, enc_dropout,
%     battery_fade, mass_bias, grease_zone). Coexists with built-in
%     params.disturbance.* (M6 load disturbance is still honored).
%   - Returns full LOG struct (t, x, y, theta, omega, tau, slip_flag, ...)
%     in addition to metrics, so plot_comparison can render full timeseries.
%
% ADDED (s15 support): grease_zone fault hook
%   params.fault.grease_zone.enabled   = true
%   params.fault.grease_zone.t_start   = 12.5   % s
%   params.fault.grease_zone.t_end     = 22.5   % s
%   params.fault.grease_zone.slip_oily = struct with mu_static, mu_kinetic,
%                                        prob_spontaneous, noise_sigma, ...
%   Effect: overrides params.slip inside [t_start, t_end] for plant_step only.
%   Outside the zone, params.slip is restored to original values.
%
% Loop structure (matches v1 exactly except for fault hooks):
%   ESP32: encoder_reader → imu_reader → pose_estimator → slip_detector →
%          position_controller → inverse kinematics → pid/adrc → pwm_output
%   H7:    pwm_capture → spi_interface('uplink')
%   [Load disturbance + fault torque injection]
%   [Grease zone: override params.slip before plant_step]          ← NEW
%   RPi5:  plant_step → imu_model
%   H7:    spi_interface('downlink') → encoder_pulse_gen → imu_packet_enc
%   Sync:  gpio_sync
%
% Inputs:
%   ctrl_type : 'pid' | 'adrc'
%   traj      : struct from trajectory_generator_v2 with .t, .x_ref, ...
%   params    : from params_mecanum() with optional .fault.* overrides
%   seed      : (optional) RNG seed for reproducibility
%
% Outputs:
%   metrics : performance metrics (see metrics_compute.m)
%   log     : full simulation log for plotting

    if nargin >= 4 && ~isempty(seed)
        rng(seed);
    end

    % Reset persistent states
    clear encoder_pulse_gen encoder_reader imu_reader position_controller;

    dt    = params.dt;
    r     = params.r;
    L     = params.lx + params.ly;
    N     = length(traj.t);

    %% Initialize state (start at reference t=0 position)
    x0 = zeros(params.n_states, 1);
    x0(1) = traj.x_ref(1);
    x0(2) = traj.y_ref(1);
    x0(3) = traj.theta_ref(1);

    sm = state_manager('init', [], x0, params);

    pid_state  = [];
    adrc_state = [];
    imu_state  = [];
    pe_state   = struct('x', x0(1), 'y', x0(2), 'theta', x0(3));
    state_fault = struct('tau_max_current', params.tau_max);

    %% Initialize H7 outputs (first-step sensor readings)
    omega_init = x0(7:10);
    enc_counts = encoder_pulse_gen(omega_init, dt, params);
    [accel_init, gyro_init, imu_state] = imu_model(x0, x0, dt, imu_state, params);
    imu_packet = imu_packet_enc(accel_init, gyro_init, params);

    %% Logging
    log.t_log        = traj.t;
    log.x_log        = zeros(1, N);
    log.y_log        = zeros(1, N);
    log.theta_log    = zeros(1, N);
    log.vx_log       = zeros(1, N);
    log.vy_log       = zeros(1, N);
    log.wz_log       = zeros(1, N);
    log.omega_log    = zeros(4, N);   % actual wheel speeds from plant
    log.omega_est_log= zeros(4, N);   % estimated (after encoder + filter + dropout)
    log.tau_log      = zeros(4, N);   % torque applied to plant
    log.tau_cmd_log  = zeros(4, N);   % torque commanded by controller
    log.pwm_log      = zeros(4, N);
    log.pos_est_log  = zeros(3, N);   % from pose_estimator
    log.vel_cmd_log  = zeros(3, N);   % [vx_cmd; vy_cmd; wz_cmd] from position_controller
    log.slip_log     = zeros(4, N);
    log.tau_max_log  = params.tau_max * ones(1, N);
    log.grease_log   = false(1, N);   % NEW: grease zone active flag
    log.sync_fail_count = 0;
    log.slip_event_count = 0;

    %% Main simulation loop
    for k = 1:N
        t_now = traj.t(k);
        x_actual = sm.x;
        omega_actual = x_actual(7:10);

        %% ===== CLUSTER 1: ESP32 =====
        % 1. Read sensors (from previous step's H7 outputs)
        omega_est = encoder_reader(enc_counts, dt, params);
        [accel_meas, gyro_meas, ~] = imu_reader(imu_packet, params);

        % 1a. FAULT INJECTION: encoder dropout (before controller sees signal)
        omega_est = inject_enc_dropout(omega_est, t_now, params);

        % Assemble imu_data struct for adrc_controller
        imu_data.accel = accel_meas;
        imu_data.gyro  = gyro_meas;

        % 2. Pose estimation
        [pose_est, pe_state] = pose_estimator(omega_est, gyro_meas, pe_state, params);

        % 3. Slip detection (monitoring)
        [slip_flag, ~] = slip_detector(omega_est, accel_meas, gyro_meas, params);

        % 4. Outer loop: position → body velocity
        pose_ref = [traj.x_ref(k); traj.y_ref(k); traj.theta_ref(k)];
        vel_ref  = [traj.vx_ref(k); traj.vy_ref(k); traj.wz_ref(k)];
        vel_cmd  = position_controller(pose_ref, pose_est, vel_ref, params);

        % 5. Inverse kinematics (X-config mecanum)
        omega_ref = (1/r) * [vel_cmd(1) - vel_cmd(2) - L*vel_cmd(3);
                             vel_cmd(1) + vel_cmd(2) + L*vel_cmd(3);
                             vel_cmd(1) + vel_cmd(2) - L*vel_cmd(3);
                             vel_cmd(1) - vel_cmd(2) + L*vel_cmd(3)];

        % 6. Inner loop: wheel velocity → torque
        switch lower(ctrl_type)
            case 'pid'
                [tau_cmd, pid_state] = pid_controller(omega_ref, omega_est, pid_state, params);
            case 'adrc'
                [tau_cmd, adrc_state] = adrc_controller(omega_ref, omega_est, imu_data, adrc_state, params);
            otherwise
                error('Unknown controller: %s', ctrl_type);
        end

        % 7. PWM output (deadband compensation)
        pwm_signal = pwm_output(tau_cmd, params);

        %% ===== CLUSTER 2: H7 (uplink torque) =====
        tau = pwm_capture(pwm_signal, params);
        [tau_up, ~] = spi_interface('uplink', tau, params);

        %% ===== Load disturbance + FAULT injection (between H7 and plant) =====
        % M6-style disturbance (params.disturbance.*)
        if isfield(params, 'disturbance') && params.disturbance.enabled
            tau_dist = compute_disturbance(t_now, params);
            tau_up = tau_up + tau_dist;
        end

        % Torque-level fault injection (wheel_jam, mass_bias, battery_fade)
        [tau_up, state_fault] = inject_torque_faults(tau_up, omega_actual, t_now, params, state_fault);

        %% ===== GREASE ZONE: override slip params for plant_step only =====  % NEW
        p_plant = inject_grease_zone(params, t_now);                           % NEW
        log.grease_log(k) = p_plant.slip.enabled && ...                        % NEW
            isfield(params.fault,'grease_zone') && ...                          % NEW
            params.fault.grease_zone.enabled && ...                             % NEW
            t_now >= params.fault.grease_zone.t_start && ...                   % NEW
            t_now <  params.fault.grease_zone.t_end;                           % NEW

        %% ===== CLUSTER 3: RPi5 (plant + sensors) =====
        x_cur = sm.x;
        x_new = plant_step(x_cur, tau_up, p_plant, dt);                       % CHANGED: p_plant
        [accel_sim, gyro_sim, imu_state] = imu_model(x_new, x_cur, dt, imu_state, params);
        sm = state_manager('update', sm, x_new);

        %% ===== CLUSTER 2: H7 (downlink states) =====
        [~, states_h7] = spi_interface('downlink', x_new, params);
        omega_plant = states_h7(7:10);
        enc_counts = encoder_pulse_gen(omega_plant, dt, params);
        imu_packet = imu_packet_enc(accel_sim, gyro_sim, params);

        %% ===== GPIO Sync =====
        cluster_done = [true; true; true];
        sync_ok = gpio_sync(k, cluster_done, params);
        if ~sync_ok, log.sync_fail_count = log.sync_fail_count + 1; end

        %% ===== Log =====
        log.x_log(k)          = x_new(1);
        log.y_log(k)          = x_new(2);
        log.theta_log(k)      = x_new(3);
        log.vx_log(k)         = x_new(4);
        log.vy_log(k)         = x_new(5);
        log.wz_log(k)         = x_new(6);
        log.omega_log(:,k)    = x_new(7:10);
        log.omega_est_log(:,k)= omega_est;
        log.tau_log(:,k)      = tau_up;
        log.tau_cmd_log(:,k)  = tau_cmd;
        log.pwm_log(:,k)      = pwm_signal;
        log.pos_est_log(:,k)  = [pose_est(1); pose_est(2); pose_est(3)];
        log.vel_cmd_log(:,k)  = vel_cmd(:);
        log.slip_log(:,k)     = slip_flag(:);
        log.tau_max_log(k)    = state_fault.tau_max_current;
        if any(slip_flag)
            log.slip_event_count = log.slip_event_count + sum(slip_flag);
        end
    end

    %% Compute metrics
    metrics = metrics_compute(log, traj, params);
    metrics.ctrl_type = ctrl_type;
    metrics.traj_type = traj.spec.type;
end


% =====================================================================
% LOCAL: GREASE ZONE HOOK (NEW — added for s15 oily floor)
% =====================================================================

function p_out = inject_grease_zone(p_in, t)
% INJECT_GREASE_ZONE  Override params.slip inside the grease zone.
%
% Called every timestep BEFORE plant_step. When t is inside [t_start, t_end],
% replaces params.slip with slip_oily config (very low mu, high prob_spontaneous).
% Outside the zone, params.slip is unchanged (background level from params_mecanum).
%
% Design: modifies a LOCAL copy of params (p_out). The caller's params struct
% is never mutated — safe to call in a loop.
    p_out = p_in;
    if ~isfield(p_in, 'fault'),            return; end
    if ~isfield(p_in.fault, 'grease_zone'), return; end
    gz = p_in.fault.grease_zone;
    if ~isstruct(gz) || ~isfield(gz,'enabled') || ~gz.enabled, return; end
    if t >= gz.t_start && t < gz.t_end
        p_out.slip = gz.slip_oily;
    end
end


% =====================================================================
% LOCAL: FAULT INJECTION HELPERS (unchanged from original)
% =====================================================================

function omega_out = inject_enc_dropout(omega_in, t, params)
% Force omega_est of a specified wheel to 0 during dropout window
    omega_out = omega_in;
    if ~isfield(params, 'fault'), return; end
    if ~isfield(params.fault, 'enc_dropout'), return; end
    F = params.fault.enc_dropout;
    if ~isstruct(F) || ~isfield(F,'enabled') || ~F.enabled, return; end
    if t >= F.t_start && t <= F.t_end
        omega_out(F.wheel) = 0;
    end
end

function [tau_out, state_out] = inject_torque_faults(tau_in, omega_actual, t, params, state_in)
% Apply wheel_jam (extra viscous friction) + battery_fade (shrinking tau_max)
% + mass_bias (constant torque bias after t_start, simulates payload drag)
    tau_out = tau_in;
    state_out = state_in;

    if ~isfield(params, 'fault')
        return;
    end
    F = params.fault;

    % Wheel jam: add viscous friction to specific wheel
    if isfield(F,'wheel_jam') && isstruct(F.wheel_jam) && ...
       isfield(F.wheel_jam,'enabled') && F.wheel_jam.enabled
        if t >= F.wheel_jam.t_start
            w = F.wheel_jam.wheel;
            tau_jam = -F.wheel_jam.b_extra * omega_actual(w);
            tau_out(w) = tau_out(w) + tau_jam;
        end
    end

    % Mass bias: constant drag (simulates load onset)
    if isfield(F,'mass_bias') && isstruct(F.mass_bias) && ...
       isfield(F.mass_bias,'enabled') && F.mass_bias.enabled
        if t >= F.mass_bias.t_start
            tau_out = tau_out - F.mass_bias.tau_bias * ones(4,1);
        end
    end

    % Battery fade: shrink tau_max over [t_start, t_end]
    if isfield(F,'battery_fade') && isstruct(F.battery_fade) && ...
       isfield(F.battery_fade,'enabled') && F.battery_fade.enabled
        BF = F.battery_fade;
        if t < BF.t_start
            tau_max_now = BF.tau_max_nominal;
        elseif t > BF.t_end
            tau_max_now = BF.tau_max_final;
        else
            alpha = (t - BF.t_start) / max(BF.t_end - BF.t_start, 1e-6);
            tau_max_now = (1 - alpha)*BF.tau_max_nominal + alpha*BF.tau_max_final;
        end
        tau_out = max(-tau_max_now, min(tau_max_now, tau_out));
        state_out.tau_max_current = tau_max_now;
    end
end


% =====================================================================
% LOCAL: M6-style load disturbance (inlined from v1 for compatibility)
% =====================================================================

function tau_dist = compute_disturbance(t_now, params)
% Identical to the local function in scripts/run_single_scenario.m (v1)
    tau_dist = zeros(4, 1);
    cfg = params.disturbance;
    if t_now < cfg.start_time
        return;
    end
    dt_since = t_now - cfg.start_time;

    switch cfg.type
        case 'step'
            tau_dist = cfg.magnitude * ones(4, 1);
        case 'ramp'
            tau_dist = min(cfg.magnitude, cfg.ramp_rate * dt_since) * ones(4, 1);
        case 'random'
            tau_dist = cfg.random_sigma * randn(4, 1);
        case 'combined'
            step_part = cfg.magnitude * 0.5;
            ramp_part = min(cfg.magnitude * 0.3, cfg.ramp_rate * dt_since);
            rand_part = cfg.random_sigma * randn(4, 1);
            tau_dist  = (step_part + ramp_part) * ones(4,1) + rand_part;
        otherwise
            tau_dist = zeros(4, 1);
    end
end
