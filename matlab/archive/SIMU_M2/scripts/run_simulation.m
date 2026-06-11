%% RUN_SIMULATION  Main HIL simulation loop.
%
% Calls 3 clusters (ESP32 → H7 → RPi5) in correct order each timestep.
% See docs/system_architecture.md for full data flow.

clear; clc; close all;

%% Add paths
% addpath(genpath(fullfile(pwd, 'functions')));
% addpath(fullfile(pwd, 'scripts'));

%% Load parameters
params = params_mecanum();

%% Configuration
controller_type = 'pid';  % 'pid' or 'adrc'
traj_type       = 'circle';
T_sim           = params.T_sim;
dt              = params.dt;
N               = round(T_sim / dt);

%% Generate trajectory
traj = trajectory_generator(traj_type, T_sim, dt, params);

%% Inverse mecanum kinematics: body ref → wheel ref
r  = params.r;
lx = params.lx;
ly = params.ly;
L  = lx + ly;

omega_ref_all = zeros(4, length(traj.t));
for i = 1:length(traj.t)
    vx = traj.vx_ref(i);
    vy = traj.vy_ref(i);
    wz = traj.wz_ref(i);
    omega_ref_all(:,i) = (1/r) * [vx - vy - L*wz;
                                    vx + vy + L*wz;
                                    vx + vy - L*wz;
                                    vx - vy + L*wz];
end

%% Initialize state
x0 = zeros(params.n_states, 1);
x0(1) = traj.x_ref(1);
x0(2) = traj.y_ref(1);
x0(3) = traj.theta_ref(1);

%% Initialize state manager
sm = state_manager('init', [], x0, params);

%% Initialize controller states
pid_state  = [];
adrc_state = [];
imu_state  = [];

%% Initialize H7 outputs (for step k=1, using initial conditions)
omega_init  = x0(7:10);
enc_counts  = encoder_pulse_gen(omega_init, dt, params);
[accel_init, gyro_init, imu_state] = imu_model(x0, x0, dt, imu_state, params);
imu_packet  = imu_packet_enc(accel_init, gyro_init, params);

%% Preallocate logging
log.tau_cmd    = zeros(4, N);
log.tau_applied = zeros(4, N);
log.omega_est  = zeros(4, N);
log.omega_ref  = zeros(4, N);
log.pwm        = zeros(4, N);
log.sync       = true(1, N);

%% Main simulation loop
fprintf('Running %s controller, %s trajectory, %d steps...\n', ...
        controller_type, traj_type, N);

for k = 1:N
    % Current reference
    omega_ref = omega_ref_all(:, k);

    %% ===== CLUSTER 1: ESP32 =====
    % 1. Read sensors (from H7 signals of step k-1)
    omega_est = encoder_reader(enc_counts, dt, params);
    [accel_meas, gyro_meas, imu_valid] = imu_reader(imu_packet, params);
    imu_data.accel = accel_meas;
    imu_data.gyro  = gyro_meas;

    % 2. Compute control
    switch controller_type
        case 'pid'
            [tau_cmd, pid_state] = pid_controller(omega_ref, omega_est, pid_state, params);
        case 'adrc'
            [tau_cmd, adrc_state] = adrc_controller(omega_ref, omega_est, imu_data, adrc_state, params);
    end

    % 3. Output PWM
    pwm_signal = pwm_output(tau_cmd, params);

    %% ===== CLUSTER 2: H7 (uplink) =====
    % 4. Capture PWM → torque
    tau = pwm_capture(pwm_signal, params);

    % 5. Forward torque to RPi5 via SPI uplink
    [tau_up, ~] = spi_interface('uplink', tau, params);

    %% ===== CLUSTER 3: RPi5 =====
    % 6. Plant step
    x_cur = sm.x;
    x_new = plant_step(x_cur, tau_up, params, dt);

    % 7. IMU model
    [accel_sim, gyro_sim, imu_state] = imu_model(x_new, x_cur, dt, imu_state, params);

    % 8. Update state manager
    sm = state_manager('update', sm, x_new);

    %% ===== CLUSTER 2: H7 (downlink) =====
    % 9. Send states down from RPi5 to H7
    [~, states_h7] = spi_interface('downlink', x_new, params);

    % 10. Generate encoder pulses from plant wheel speeds
    omega_plant = states_h7(7:10);
    enc_counts = encoder_pulse_gen(omega_plant, dt, params);

    % 11. Pack IMU data
    imu_packet = imu_packet_enc(accel_sim, gyro_sim, params);

    % 12. Sync check
    cluster_done = [true; true; true]; % all done in sim
    sync_ok = gpio_sync(k, cluster_done, params);

    %% ===== LOGGING =====
    log.tau_cmd(:,k)     = tau_cmd;
    log.tau_applied(:,k) = tau_up;
    log.omega_est(:,k)   = omega_est;
    log.omega_ref(:,k)   = omega_ref;
    log.pwm(:,k)         = pwm_signal;
    log.sync(k)          = sync_ok;

    if k == 1
    fprintf('\n=== DEBUG step k=1 ===\n');
    fprintf('omega_ref: [%.2f, %.2f, %.2f, %.2f]\n', omega_ref);
    fprintf('enc_counts: [%d, %d, %d, %d]\n', enc_counts);
    fprintf('omega_est: [%.2f, %.2f, %.2f, %.2f]\n', omega_est);
    fprintf('tau_cmd: [%.4f, %.4f, %.4f, %.4f]\n', tau_cmd);
    fprintf('pwm: [%.4f, %.4f, %.4f, %.4f]\n', pwm_signal);
    fprintf('tau_applied: [%.4f, %.4f, %.4f, %.4f]\n', tau_up);
    fprintf('x_new(1:3): [%.4f, %.4f, %.4f]\n', x_new(1:3));
    fprintf('x_new(7:10): [%.2f, %.2f, %.2f, %.2f]\n', x_new(7:10));
    end
end

fprintf('Simulation complete.\n');

%% ===== RESULTS =====
% Extract state history
x_hist = sm.history(:, 1:sm.k);
t_hist = (0:sm.k-1) * dt;

% Quick summary
fprintf('\n--- Summary ---\n');
fprintf('Final position: x=%.4f, y=%.4f, theta=%.4f\n', ...
        x_hist(1,end), x_hist(2,end), x_hist(3,end));
fprintf('Sync failures: %d / %d\n', sum(~log.sync), N);

%% Plot results
plot_results;
