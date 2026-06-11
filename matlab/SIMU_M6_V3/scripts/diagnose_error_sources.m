%% DIAGNOSE_ERROR_SOURCES  Isolate where tracking error comes from.
%
% Runs 4 conditions on circle trajectory (worst case ~100mm):
%   A. Ideal: ground truth pose + ground truth sensors → controller limit only
%   B. Real sensors + ground truth pose → adds signal conditioning noise
%   C. Real sensors + dead reckoning pose → adds odometry drift
%   D. Full system (= current baseline)
%
% Difference between levels reveals error contribution of each subsystem.
% Also computes inner loop bandwidth and settling time.

clear; clc; close all;

params = params_mecanum();
dt = params.dt;
T_sim = params.T_sim;
N = round(T_sim / dt);
r = params.r;
L = params.lx + params.ly;

%% ===== PART 1: INNER LOOP BANDWIDTH ANALYSIS =====
fprintf('========================================\n');
fprintf('  PART 1: Inner Loop Bandwidth Analysis\n');
fprintf('========================================\n\n');

% Effective inertia (diagonal element of M_eff)
J_eff_diag = params.M_eff(1,1);
fprintf('M_eff diagonal (wheel 1): %.5f kg·m²\n', J_eff_diag);
fprintf('M_eff off-diagonal (1,2): %.5f kg·m² (coupling %.0f%%)\n', ...
    params.M_eff(1,2), abs(params.M_eff(1,2)/J_eff_diag)*100);

% PID inner loop: approximate as 1st-order with P gain
% Closed-loop pole: s = -(Kp + b_w) / J_eff
pid_bw = (params.pid.Kp + params.b_w) / J_eff_diag;
pid_settle = 4 / pid_bw;  % 4 time constants to ~2% settling
fprintf('\nPID inner loop:\n');
fprintf('  Kp = %.3f → BW = %.1f rad/s (%.2f Hz)\n', params.pid.Kp, pid_bw, pid_bw/(2*pi));
fprintf('  Settling time (4τ): %.2f s\n', pid_settle);
fprintf('  At startup error 20 rad/s: τ_cmd = %.2f N·m (τ_max=%.1f)\n', ...
    params.pid.Kp * 20, params.tau_max);

% ADRC inner loop
fprintf('\nADRC inner loop:\n');
fprintf('  kp = %.0f → BW ≈ %.0f rad/s (%.1f Hz)\n', ...
    params.adrc.kp, params.adrc.kp, params.adrc.kp/(2*pi));
fprintf('  ESO BW: ω_o = %.0f rad/s (%.1f Hz)\n', ...
    sqrt(params.adrc.eso_beta2), sqrt(params.adrc.eso_beta2)/(2*pi));
fprintf('  Settling time (4/kp): %.3f s\n', 4/params.adrc.kp);

% Outer loop
outer_bw = params.pos_ctrl.Kp_pos;
fprintf('\nOuter loop:\n');
fprintf('  Kp_pos = %.1f → BW ≈ %.1f rad/s (%.2f Hz)\n', ...
    params.pos_ctrl.Kp_pos, outer_bw, outer_bw/(2*pi));
fprintf('  Ti_pos = Kp/Ki = %.1f s\n', params.pos_ctrl.Kp_pos / params.pos_ctrl.Ki_pos);
fprintf('  Ti_theta = Kp/Ki = %.1f s\n', params.pos_ctrl.Kp_theta / params.pos_ctrl.Ki_theta);

% Bandwidth ratios
fprintf('\nBandwidth ratios (should be >3×):\n');
fprintf('  ADRC_inner / outer = %.0f / %.1f = %.1f×\n', ...
    params.adrc.kp, outer_bw, params.adrc.kp / outer_bw);
fprintf('  PID_inner / outer  = %.1f / %.1f = %.1f×\n', ...
    pid_bw, outer_bw, pid_bw / outer_bw);

% Circle trajectory demands
v_circ = 0.5 * 2*pi/5;  % R*omega
a_cent = v_circ^2 / 0.5;
omega_wheel_circ = v_circ / r;  % approximate
fprintf('\nCircle trajectory demands:\n');
fprintf('  v = %.3f m/s, ω_circ = %.3f rad/s\n', v_circ, 2*pi/5);
fprintf('  Centripetal accel = %.3f m/s²\n', a_cent);
fprintf('  Wheel speed ≈ %.1f rad/s (max = %.1f)\n', omega_wheel_circ, params.omega_max);
fprintf('  Startup acceleration at τ_max: %.0f rad/s² → reach %.0f rad/s in %.3f s\n', ...
    params.tau_max / J_eff_diag, omega_wheel_circ, omega_wheel_circ / (params.tau_max/J_eff_diag));

%% ===== PART 2: ERROR SOURCE ISOLATION =====
fprintf('\n========================================\n');
fprintf('  PART 2: Error Source Isolation (Circle)\n');
fprintf('========================================\n\n');

conditions = {'A_ideal', 'B_real_sensors', 'C_dead_reckoning', 'D_full_system'};
controllers = {'pid', 'adrc'};

results = struct();

for ci = 1:2
    ctrl = controllers{ci};
    for cond = 1:4
        cond_name = conditions{cond};

        % Reset all persistent states
        clear encoder_pulse_gen encoder_reader imu_reader position_controller;

        traj = trajectory_generator('circle', T_sim, dt, params);

        x0 = zeros(params.n_states, 1);
        x0(1) = traj.x_ref(1); x0(2) = traj.y_ref(1); x0(3) = traj.theta_ref(1);
        sm = state_manager('init', [], x0, params);

        pid_state = []; adrc_state = []; imu_state = [];
        pe_state = [];
        pe_state.x = x0(1); pe_state.y = x0(2); pe_state.theta = x0(3);

        omega_init = x0(7:10);
        enc_counts = encoder_pulse_gen(omega_init, dt, params);
        [a_i, g_i, imu_state] = imu_model(x0, x0, dt, imu_state, params);
        imu_packet = imu_packet_enc(a_i, g_i, params);

        pos_err = zeros(1, N);
        sat_count = 0;

        for k = 1:N
            x_cur = sm.x;

            %% Sensor reading — depends on condition
            if cond == 1  % A: ideal sensors (ground truth)
                omega_est = x_cur(7:10);
                gyro_meas = [0; 0; x_cur(6)];
            else  % B, C, D: real sensors through H7 pipeline
                omega_est = encoder_reader(enc_counts, dt, params);
                [accel_meas, gyro_meas, ~] = imu_reader(imu_packet, params);
            end

            %% Pose estimation — depends on condition
            if cond <= 2  % A, B: ground truth pose
                pose_est = [x_cur(1); x_cur(2); x_cur(3)];
            else  % C, D: dead reckoning
                [pose_est, pe_state] = pose_estimator(omega_est, gyro_meas, pe_state, params);
            end

            %% Outer loop (always active)
            pose_ref = [traj.x_ref(k); traj.y_ref(k); traj.theta_ref(k)];
            vel_ref = [traj.vx_ref(k); traj.vy_ref(k); traj.wz_ref(k)];
            vel_cmd = position_controller(pose_ref, pose_est, vel_ref, params);

            omega_ref = (1/r) * [vel_cmd(1)-vel_cmd(2)-L*vel_cmd(3);
                                 vel_cmd(1)+vel_cmd(2)+L*vel_cmd(3);
                                 vel_cmd(1)+vel_cmd(2)-L*vel_cmd(3);
                                 vel_cmd(1)-vel_cmd(2)+L*vel_cmd(3)];

            %% Inner loop
            imu_data.accel = zeros(3,1); imu_data.gyro = gyro_meas;
            switch ctrl
                case 'pid'
                    [tau_cmd, pid_state] = pid_controller(omega_ref, omega_est, pid_state, params);
                case 'adrc'
                    [tau_cmd, adrc_state] = adrc_controller(omega_ref, omega_est, imu_data, adrc_state, params);
            end

            sat_count = sat_count + sum(abs(tau_cmd) >= params.tau_max * 0.99);

            pwm_signal = pwm_output(tau_cmd, params);

            %% Plant — depends on condition
            if cond <= 3  % A, B, C: bypass H7 signal conditioning on torque path
                tau_applied = tau_cmd;
                tau_applied = max(-params.tau_max, min(params.tau_max, tau_applied));
            else  % D: full H7 pipeline
                tau = pwm_capture(pwm_signal, params);
                [tau_applied, ~] = spi_interface('uplink', tau, params);
            end

            x_new = plant_step(x_cur, tau_applied, params, dt);
            [accel_sim, gyro_sim, imu_state] = imu_model(x_new, x_cur, dt, imu_state, params);
            sm = state_manager('update', sm, x_new);

            % H7 downlink for sensor generation (needed for conditions B, C, D)
            if cond >= 2
                [~, states_h7] = spi_interface('downlink', x_new, params);
                enc_counts = encoder_pulse_gen(states_h7(7:10), dt, params);
                imu_packet = imu_packet_enc(accel_sim, gyro_sim, params);
            end

            pos_err(k) = sqrt((x_new(1)-traj.x_ref(k))^2 + (x_new(2)-traj.y_ref(k))^2);
        end

        ss_start = round(N/2);
        res.rms_full = rms(pos_err) * 1000;
        res.rms_ss = rms(pos_err(ss_start:end)) * 1000;
        res.max_err = max(pos_err) * 1000;
        res.sat_pct = sat_count / (4*N) * 100;

        results.(ctrl).(cond_name) = res;

        fprintf('%s / %s: RMS=%.1f mm, SS=%.1f mm, Max=%.1f mm, Sat=%.1f%%\n', ...
            upper(ctrl), cond_name, res.rms_full, res.rms_ss, res.max_err, res.sat_pct);
    end
    fprintf('\n');
end

%% ===== PART 3: ERROR DECOMPOSITION =====
fprintf('========================================\n');
fprintf('  PART 3: Error Decomposition (Circle SS)\n');
fprintf('========================================\n\n');

for ci = 1:2
    ctrl = controllers{ci};
    A = results.(ctrl).A_ideal.rms_ss;
    B = results.(ctrl).B_real_sensors.rms_ss;
    C = results.(ctrl).C_dead_reckoning.rms_ss;
    D = results.(ctrl).D_full_system.rms_ss;

    fprintf('%s:\n', upper(ctrl));
    fprintf('  A. Controller only (ideal sensors+pose):  %.1f mm  ← controller limit\n', A);
    fprintf('  B. + Signal conditioning noise:           %.1f mm  (Δ = +%.1f mm)\n', B, B-A);
    fprintf('  C. + Dead reckoning drift:                %.1f mm  (Δ = +%.1f mm)\n', C, C-B);
    fprintf('  D. + H7 torque pipeline:                  %.1f mm  (Δ = +%.1f mm)\n', D, D-C);
    fprintf('  ─────────────────────────────────\n');
    fprintf('  Controller accounts for: %.0f%% of total error\n', A/D*100);
    fprintf('  Signal chain accounts for: %.0f%%\n', (D-A)/D*100);
    fprintf('\n');
end

fprintf('CONCLUSION:\n');
pid_A = results.pid.A_ideal.rms_ss;
pid_D = results.pid.D_full_system.rms_ss;
adrc_A = results.adrc.A_ideal.rms_ss;
adrc_D = results.adrc.D_full_system.rms_ss;

if pid_A/pid_D > 0.7
    fprintf('  → Controller is the DOMINANT error source (>70%% of total)\n');
    fprintf('  → Plant is NOT the bottleneck\n');
    fprintf('  → Signal conditioning contributes minimally\n');
    fprintf('  → TUNE CONTROLLER GAINS before adding disturbance\n');
else
    fprintf('  → Error is distributed across subsystems\n');
    fprintf('  → Both controller and signal chain need attention\n');
end
