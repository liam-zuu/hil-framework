%% RUN_M5_COMPARISON  Compare PID vs ADRC across all trajectory types.
%
% Runs 8 scenarios (2 controllers × 4 trajectories), collects metrics,
% prints comparison table and generates summary plots.
%
% Step 5.7: So sánh tracking error PID vs ADRC, verify khớp kỳ vọng.

clear; clc; close all;

%% Configuration
controllers = {'pid', 'adrc'};
trajectories = {'line', 'circle', 'square', 'figure8'};
params = params_mecanum();
dt = params.dt;
T_sim = params.T_sim;
N = round(T_sim / dt);

r = params.r;
L = params.lx + params.ly;

%% Results storage
n_ctrl = length(controllers);
n_traj = length(trajectories);
results = struct();

%% Run all scenarios
fprintf('====================================================\n');
fprintf('  M5 COMPARISON: PID vs ADRC × 4 Trajectories\n');
fprintf('====================================================\n\n');

for ci = 1:n_ctrl
    for ti = 1:n_traj
        ctrl = controllers{ci};
        traj_type = trajectories{ti};

        fprintf('Running: %s / %s ... ', upper(ctrl), traj_type);

        % Reset persistent states
        clear encoder_pulse_gen encoder_reader imu_reader position_controller;

        % Generate trajectory
        traj = trajectory_generator(traj_type, T_sim, dt, params);

        % Initialize state
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

        % Initialize H7 outputs
        omega_init = x0(7:10);
        enc_counts = encoder_pulse_gen(omega_init, dt, params);
        [accel_init, gyro_init, imu_state] = imu_model(x0, x0, dt, imu_state, params);
        imu_packet = imu_packet_enc(accel_init, gyro_init, params);

        % Logging
        pos_err_vec = zeros(1, N);
        theta_err_vec = zeros(1, N);
        tau_cmd_log = zeros(4, N);
        sync_fail_count = 0;

        % --- Main loop ---
        for k = 1:N
            % ESP32: sensors
            omega_est = encoder_reader(enc_counts, dt, params);
            [accel_meas, gyro_meas, ~] = imu_reader(imu_packet, params);
            imu_data.accel = accel_meas;
            imu_data.gyro  = gyro_meas;

            % ESP32: pose estimation
            [pose_est, pe_state] = pose_estimator(omega_est, gyro_meas, pe_state, params);

            % ESP32: outer loop
            pose_ref = [traj.x_ref(k); traj.y_ref(k); traj.theta_ref(k)];
            vel_ref  = [traj.vx_ref(k); traj.vy_ref(k); traj.wz_ref(k)];
            vel_cmd = position_controller(pose_ref, pose_est, vel_ref, params);

            % ESP32: inverse kinematics
            omega_ref = (1/r) * [vel_cmd(1) - vel_cmd(2) - L*vel_cmd(3);
                                 vel_cmd(1) + vel_cmd(2) + L*vel_cmd(3);
                                 vel_cmd(1) + vel_cmd(2) - L*vel_cmd(3);
                                 vel_cmd(1) - vel_cmd(2) + L*vel_cmd(3)];

            % ESP32: inner loop
            switch ctrl
                case 'pid'
                    [tau_cmd, pid_state] = pid_controller(omega_ref, omega_est, pid_state, params);
                case 'adrc'
                    [tau_cmd, adrc_state] = adrc_controller(omega_ref, omega_est, imu_data, adrc_state, params);
            end

            pwm_signal = pwm_output(tau_cmd, params);

            % H7: uplink
            tau = pwm_capture(pwm_signal, params);
            [tau_up, ~] = spi_interface('uplink', tau, params);

            % RPi5: plant
            x_cur = sm.x;
            x_new = plant_step(x_cur, tau_up, params, dt);
            [accel_sim, gyro_sim, imu_state] = imu_model(x_new, x_cur, dt, imu_state, params);
            sm = state_manager('update', sm, x_new);

            % H7: downlink
            [~, states_h7] = spi_interface('downlink', x_new, params);
            omega_plant = states_h7(7:10);
            enc_counts = encoder_pulse_gen(omega_plant, dt, params);
            imu_packet = imu_packet_enc(accel_sim, gyro_sim, params);

            cluster_done = [true; true; true];
            sync_ok = gpio_sync(k, cluster_done, params);
            if ~sync_ok, sync_fail_count = sync_fail_count + 1; end

            % Log errors
            pos_err_vec(k) = sqrt((x_new(1) - traj.x_ref(k))^2 + ...
                                   (x_new(2) - traj.y_ref(k))^2);
            e_th = x_new(3) - traj.theta_ref(k);
            theta_err_vec(k) = mod(e_th + pi, 2*pi) - pi;
            tau_cmd_log(:,k) = tau_cmd;
        end

        % --- Metrics ---
        ss_start = round(N/2);  % steady-state = last 50%

        res.rms_pos_full = rms(pos_err_vec) * 1000;                   % mm
        res.rms_pos_ss   = rms(pos_err_vec(ss_start:end)) * 1000;     % mm
        res.max_pos      = max(pos_err_vec) * 1000;                    % mm
        res.rms_theta    = rms(theta_err_vec) * 180/pi;                % deg
        res.rms_theta_ss = rms(theta_err_vec(ss_start:end)) * 180/pi;  % deg
        res.max_tau      = max(abs(tau_cmd_log(:)));                    % N·m
        res.sat_pct      = sum(abs(tau_cmd_log(:)) >= params.tau_max*0.99) / numel(tau_cmd_log) * 100;
        res.sync_fails   = sync_fail_count;

        results.(ctrl).(traj_type) = res;

        fprintf('RMS=%.1fmm (SS=%.1fmm)\n', res.rms_pos_full, res.rms_pos_ss);
    end
end

%% ===== PRINT COMPARISON TABLE =====
fprintf('\n');
fprintf('===========================================================================\n');
fprintf('  M5 RESULTS SUMMARY — Position Tracking Error\n');
fprintf('===========================================================================\n');
fprintf('%-10s | %-6s | %10s | %10s | %10s | %8s | %6s\n', ...
        'Trajectory', 'Ctrl', 'RMS(mm)', 'SS_RMS(mm)', 'Max(mm)', 'θ_RMS(°)', 'Sat(%)');
fprintf('-----------+--------+------------+------------+------------+----------+-------\n');

for ti = 1:n_traj
    traj_type = trajectories{ti};
    for ci = 1:n_ctrl
        ctrl = controllers{ci};
        r_ = results.(ctrl).(traj_type);
        fprintf('%-10s | %-6s | %10.1f | %10.1f | %10.1f | %8.2f | %5.1f\n', ...
                traj_type, upper(ctrl), r_.rms_pos_full, r_.rms_pos_ss, ...
                r_.max_pos, r_.rms_theta, r_.sat_pct);
    end
    fprintf('-----------+--------+------------+------------+------------+----------+-------\n');
end

%% ===== ADRC IMPROVEMENT OVER PID =====
fprintf('\n');
fprintf('===========================================================================\n');
fprintf('  ADRC Improvement over PID (steady-state RMS)\n');
fprintf('===========================================================================\n');
fprintf('%-10s | %12s | %12s | %12s\n', 'Trajectory', 'PID_SS(mm)', 'ADRC_SS(mm)', 'Improvement');
fprintf('-----------+--------------+--------------+-------------\n');

for ti = 1:n_traj
    traj_type = trajectories{ti};
    pid_ss  = results.pid.(traj_type).rms_pos_ss;
    adrc_ss = results.adrc.(traj_type).rms_pos_ss;
    if pid_ss > 0
        imp = (pid_ss - adrc_ss) / pid_ss * 100;
        fprintf('%-10s | %12.1f | %12.1f | %+10.1f%%\n', ...
                traj_type, pid_ss, adrc_ss, imp);
    end
end
fprintf('-----------+--------------+--------------+-------------\n');

%% ===== SUMMARY PLOT =====
figure('Name', 'M5 Comparison Summary', 'Position', [100 100 1000 500]);

% Bar chart: RMS position error (steady-state) for each trajectory
subplot(1,2,1);
bar_data = zeros(n_traj, n_ctrl);
for ti = 1:n_traj
    for ci = 1:n_ctrl
        bar_data(ti, ci) = results.(controllers{ci}).(trajectories{ti}).rms_pos_ss;
    end
end
b = bar(bar_data);
b(1).FaceColor = [0.3 0.5 0.8]; b(2).FaceColor = [0.9 0.3 0.2];
set(gca, 'XTickLabel', trajectories);
ylabel('Steady-State RMS Position Error (mm)');
title('Position Tracking: PID vs ADRC');
legend('PID', 'ADRC', 'Location', 'northwest');
grid on;

% Bar chart: heading error
subplot(1,2,2);
bar_data_th = zeros(n_traj, n_ctrl);
for ti = 1:n_traj
    for ci = 1:n_ctrl
        bar_data_th(ti, ci) = results.(controllers{ci}).(trajectories{ti}).rms_theta_ss;
    end
end
b2 = bar(bar_data_th);
b2(1).FaceColor = [0.3 0.5 0.8]; b2(2).FaceColor = [0.9 0.3 0.2];
set(gca, 'XTickLabel', trajectories);
ylabel('Steady-State RMS Heading Error (deg)');
title('Heading Tracking: PID vs ADRC');
legend('PID', 'ADRC', 'Location', 'northwest');
grid on;

sgtitle('M5 Full Integration — PID vs ADRC Comparison', 'FontSize', 14);

fprintf('\nM5 comparison complete.\n');
