%% RUN_SIMULATION  Main HIL simulation loop — M5 Full Integration.
%
% Two-loop architecture:
%   Outer loop: position_controller (pose error → body velocity commands)
%   Inner loop: pid_controller / adrc_controller (wheel velocity tracking)
%
% Data flow each timestep k:
%   1. ESP32 reads sensors (encoder, IMU) — signals from step k-1
%   2. ESP32 estimates pose (dead reckoning from encoder + IMU)
%   3. ESP32 outer loop: position error → corrected body velocities
%   4. ESP32 inverse kinematics: body vel → omega_ref
%   5. ESP32 inner loop: omega_ref vs omega_est → tau_cmd
%   6. ESP32 → PWM output
%   7. H7 captures PWM → torque, forwards to RPi5
%   8. RPi5 plant step → new state
%   9. RPi5 IMU model → sensor signals
%  10. H7 generates encoder pulses + IMU packet → ready for step k+1
%
% Calls 3 clusters (ESP32 → H7 → RPi5) in correct order each timestep.

clear; clc; close all;
clear encoder_pulse_gen;  % Reset persistent accumulator
clear encoder_reader;     % Reset persistent filter state
clear imu_reader;         % Reset persistent filter state
clear position_controller; % Reset persistent integral state

%% Load parameters
params = params_mecanum();

%% Configuration
controller_type = 'pid';   % 'pid' or 'adrc'
traj_type       = 'circle'; % 'line' | 'circle' | 'square' | 'figure8'
T_sim           = params.T_sim;
dt              = params.dt;
N               = round(T_sim / dt);

%% Generate trajectory
traj = trajectory_generator(traj_type, T_sim, dt, params);

%% Inverse kinematics constants (for online computation)
r  = params.r;
lx = params.lx;
ly = params.ly;
L  = lx + ly;

%% Initialize state
x0 = zeros(params.n_states, 1);
x0(1) = traj.x_ref(1);
x0(2) = traj.y_ref(1);
x0(3) = traj.theta_ref(1);

%% Initialize state manager
sm = state_manager('init', [], x0, params);

%% Initialize controller and estimator states
pid_state  = [];
adrc_state = [];
imu_state  = [];
pe_state   = [];  % pose estimator state

% Initialize pose estimator at true initial position
pe_state.x     = x0(1);
pe_state.y     = x0(2);
pe_state.theta = x0(3);

%% Initialize H7 outputs (for step k=1, using initial conditions)
omega_init  = x0(7:10);
enc_counts  = encoder_pulse_gen(omega_init, dt, params);
[accel_init, gyro_init, imu_state] = imu_model(x0, x0, dt, imu_state, params);
imu_packet  = imu_packet_enc(accel_init, gyro_init, params);

%% Preallocate logging
log.tau_cmd     = zeros(4, N);
log.tau_applied = zeros(4, N);
log.omega_est   = zeros(4, N);
log.omega_ref   = zeros(4, N);
log.pwm         = zeros(4, N);
log.sync        = true(1, N);
log.pose_est    = zeros(3, N);   % estimated pose [x; y; theta]
log.vel_cmd     = zeros(3, N);   % corrected body velocity commands
log.pose_ref    = zeros(3, N);   % reference pose

%% Main simulation loop
fprintf('Running %s controller, %s trajectory, %d steps...\n', ...
        controller_type, traj_type, N);

for k = 1:N

    %% ===== CLUSTER 1: ESP32 =====

    % --- 1. Read sensors (from H7 signals of step k-1) ---
    omega_est = encoder_reader(enc_counts, dt, params);
    [accel_meas, gyro_meas, imu_valid] = imu_reader(imu_packet, params);
    imu_data.accel = accel_meas;
    imu_data.gyro  = gyro_meas;

    % --- 2. Estimate pose (dead reckoning) ---
    [pose_est, pe_state] = pose_estimator(omega_est, gyro_meas, pe_state, params);

    % --- 3. Outer loop: position controller ---
    pose_ref = [traj.x_ref(k); traj.y_ref(k); traj.theta_ref(k)];
    vel_ref  = [traj.vx_ref(k); traj.vy_ref(k); traj.wz_ref(k)];

    vel_cmd = position_controller(pose_ref, pose_est, vel_ref, params);

    % --- 4. Inverse kinematics: corrected body vel → omega_ref ---
    omega_ref = (1/r) * [vel_cmd(1) - vel_cmd(2) - L*vel_cmd(3);
                         vel_cmd(1) + vel_cmd(2) + L*vel_cmd(3);
                         vel_cmd(1) + vel_cmd(2) - L*vel_cmd(3);
                         vel_cmd(1) - vel_cmd(2) + L*vel_cmd(3)];

    % --- 5. Inner loop: wheel velocity control ---
    switch controller_type
        case 'pid'
            [tau_cmd, pid_state] = pid_controller(omega_ref, omega_est, pid_state, params);
        case 'adrc'
            [tau_cmd, adrc_state] = adrc_controller(omega_ref, omega_est, imu_data, adrc_state, params);
    end

    % --- 6. Output PWM ---
    pwm_signal = pwm_output(tau_cmd, params);

    %% ===== CLUSTER 2: H7 (uplink) =====
    % 7. Capture PWM → torque
    tau = pwm_capture(pwm_signal, params);

    % 8. Forward torque to RPi5 via SPI uplink
    [tau_up, ~] = spi_interface('uplink', tau, params);

    %% ===== CLUSTER 3: RPi5 =====
    % 9. Plant step
    x_cur = sm.x;
    x_new = plant_step(x_cur, tau_up, params, dt);

    % 10. IMU model
    [accel_sim, gyro_sim, imu_state] = imu_model(x_new, x_cur, dt, imu_state, params);

    % 11. Update state manager
    sm = state_manager('update', sm, x_new);

    %% ===== CLUSTER 2: H7 (downlink) =====
    % 12. Send states down from RPi5 to H7
    [~, states_h7] = spi_interface('downlink', x_new, params);

    % 13. Generate encoder pulses from plant wheel speeds
    omega_plant = states_h7(7:10);
    enc_counts = encoder_pulse_gen(omega_plant, dt, params);

    % 14. Pack IMU data
    imu_packet = imu_packet_enc(accel_sim, gyro_sim, params);

    % 15. Sync check
    cluster_done = [true; true; true];
    sync_ok = gpio_sync(k, cluster_done, params);

    %% ===== LOGGING =====
    log.tau_cmd(:,k)     = tau_cmd;
    log.tau_applied(:,k) = tau_up;
    log.omega_est(:,k)   = omega_est;
    log.omega_ref(:,k)   = omega_ref;
    log.pwm(:,k)         = pwm_signal;
    log.sync(k)          = sync_ok;
    log.pose_est(:,k)    = pose_est;
    log.vel_cmd(:,k)     = vel_cmd;
    log.pose_ref(:,k)    = pose_ref;
end

fprintf('Simulation complete.\n');

%% ===== RESULTS =====
% Extract state history
x_hist = sm.history(:, 1:sm.k);
t_hist = (0:sm.k-1) * dt;

% Position error
N_err = min(length(traj.x_ref), length(t_hist));
ex = x_hist(1,1:N_err) - traj.x_ref(1:N_err);
ey = x_hist(2,1:N_err) - traj.y_ref(1:N_err);
e_pos = sqrt(ex.^2 + ey.^2);
rms_err = rms(e_pos) * 1000;  % mm

% Steady-state error (last 50%)
ss_start = round(N_err/2);
rms_ss = rms(e_pos(ss_start:end)) * 1000;  % mm

% Quick summary
fprintf('\n--- Summary (%s, %s) ---\n', controller_type, traj_type);
fprintf('Final position: x=%.4f, y=%.4f, theta=%.4f\n', ...
        x_hist(1,end), x_hist(2,end), x_hist(3,end));
fprintf('RMS position error (full):       %.2f mm\n', rms_err);
fprintf('RMS position error (steady-state): %.2f mm\n', rms_ss);
fprintf('Max position error:              %.2f mm\n', max(e_pos)*1000);
fprintf('Sync failures: %d / %d\n', sum(~log.sync), N);

%% Plot results
plot_results;
