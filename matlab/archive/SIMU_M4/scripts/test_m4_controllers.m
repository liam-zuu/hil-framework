%% TEST_M4_CONTROLLERS  Open-loop unit tests for M4 modules.
%
% Feeds known sensor data to each module, verifies outputs are reasonable.
% Run BEFORE closed-loop (step 4.7) to catch bugs early.
%
% Tests:
%   1. encoder_reader: filter response to step input
%   2. imu_reader: outlier rejection + filter
%   3. pwm_output: deadband compensation
%   4. pid_controller: step response, anti-windup
%   5. adrc_controller: step response, ESO convergence
%   6. Pipeline: encoder → controller → pwm_output → pwm_capture round-trip

clear; clc; close all;
clear encoder_pulse_gen encoder_reader imu_reader;

params = params_mecanum();
dt = params.dt;
n_pass = 0;
n_fail = 0;

fprintf('=== M4 Controller Unit Tests ===\n\n');

%% ===== TEST 1: encoder_reader filter response =====
fprintf('--- Test 1: encoder_reader filter step response ---\n');
clear encoder_reader;

% Feed constant enc_counts for 50 steps (simulating ω=10 rad/s)
counts_per_rad = params.enc_ppr / (2*pi);
omega_true = 10.0;
enc_counts_const = round(omega_true * dt * counts_per_rad) * ones(4,1);

omega_hist = zeros(4, 50);
for k = 1:50
    omega_hist(:,k) = encoder_reader(enc_counts_const, dt, params);
end

% After 5*tau_f = 25ms = 25 steps, filter should be ~99% settled
omega_final = omega_hist(1, end);
omega_expected = enc_counts_const(1) / (dt * counts_per_rad);  % raw decode value
settle_error = abs(omega_final - omega_expected) / omega_expected;

if settle_error < 0.02  % within 2%
    fprintf('  PASS: Filter settles to %.2f rad/s (expected %.2f, error %.1f%%)\n', ...
        omega_final, omega_expected, settle_error*100);
    n_pass = n_pass + 1;
else
    fprintf('  FAIL: Filter at %.2f, expected %.2f (error %.1f%%)\n', ...
        omega_final, omega_expected, settle_error*100);
    n_fail = n_fail + 1;
end

% Check that initial value is attenuated (filter effect)
alpha = dt / (params.enc_filter_tau + dt);
if omega_hist(1,1) < omega_final * 0.5
    fprintf('  PASS: First sample attenuated (%.2f vs final %.2f) — filter working\n', ...
        omega_hist(1,1), omega_final);
    n_pass = n_pass + 1;
else
    fprintf('  FAIL: First sample not attenuated — filter may not be working\n');
    n_fail = n_fail + 1;
end

%% ===== TEST 2: imu_reader outlier rejection =====
fprintf('\n--- Test 2: imu_reader outlier rejection ---\n');
clear imu_reader;

% Create a valid packet
accel_true = [1.0; 0.5; 9.81];
gyro_true  = [0.0; 0.0; 0.3];
pkt_good = imu_packet_enc(accel_true, gyro_true, params);

% Feed 10 good packets to establish baseline
for k = 1:10
    [~, ~, ~] = imu_reader(pkt_good, params);
end

% Now feed a packet with spike (outlier) in accel
accel_spike = [100.0; 0.5; 9.81];  % 100 m/s² spike in ax
pkt_spike = imu_packet_enc(accel_spike, gyro_true, params);
[accel_out, ~, valid] = imu_reader(pkt_spike, params);

if valid && abs(accel_out(1)) < 50  % Should reject the 100 m/s² spike
    fprintf('  PASS: Accel spike (100 m/s²) rejected, output ax=%.2f\n', accel_out(1));
    n_pass = n_pass + 1;
else
    fprintf('  FAIL: Accel spike not rejected, output ax=%.2f\n', accel_out(1));
    n_fail = n_fail + 1;
end

% Test checksum corruption
pkt_bad = pkt_good;
pkt_bad.checksum = int32(999);
[accel_bad, ~, valid_bad] = imu_reader(pkt_bad, params);

if ~valid_bad
    fprintf('  PASS: Corrupted checksum detected (valid=false)\n');
    n_pass = n_pass + 1;
else
    fprintf('  FAIL: Corrupted checksum not detected\n');
    n_fail = n_fail + 1;
end

%% ===== TEST 3: pwm_output deadband compensation =====
fprintf('\n--- Test 3: pwm_output deadband compensation ---\n');

% Small torque command (would be in deadband without compensation)
tau_small = [0.005; -0.005; 0.0; 0.001];
pwm = pwm_output(tau_small, params);

% Check: nonzero commands should produce |pwm| >= deadband
if abs(pwm(1)) >= params.deadband && abs(pwm(2)) >= params.deadband
    fprintf('  PASS: Small torques (0.005 N·m) compensated above deadband (|pwm|=%.4f)\n', abs(pwm(1)));
    n_pass = n_pass + 1;
else
    fprintf('  FAIL: Small torques still in deadband (pwm=[%.4f, %.4f])\n', pwm(1), pwm(2));
    n_fail = n_fail + 1;
end

% Check: zero command stays zero
if pwm(3) == 0
    fprintf('  PASS: Zero torque → zero PWM\n');
    n_pass = n_pass + 1;
else
    fprintf('  FAIL: Zero torque → PWM=%.4f (should be 0)\n', pwm(3));
    n_fail = n_fail + 1;
end

% Check: full torque saturates to ±1
tau_full = [1.0; -1.0; 0.5; -0.5];
pwm_full = pwm_output(tau_full, params);
if abs(pwm_full(1)) == 1.0 && abs(pwm_full(2)) == 1.0
    fprintf('  PASS: Full torque saturates to ±1\n');
    n_pass = n_pass + 1;
else
    fprintf('  FAIL: Saturation incorrect (pwm=[%.4f, %.4f])\n', pwm_full(1), pwm_full(2));
    n_fail = n_fail + 1;
end

%% ===== TEST 4: pid_controller step response =====
fprintf('\n--- Test 4: pid_controller step response ---\n');

omega_ref = [10; 10; 10; 10];
omega_est = [0; 0; 0; 0];
pid_state = [];

% First step: large error → large tau
[tau1, pid_state] = pid_controller(omega_ref, omega_est, pid_state, params);

if all(tau1 > 0)
    fprintf('  PASS: Positive error → positive torque (tau=%.3f)\n', tau1(1));
    n_pass = n_pass + 1;
else
    fprintf('  FAIL: Wrong sign (tau=%.3f)\n', tau1(1));
    n_fail = n_fail + 1;
end

% Run 100 steps with error → check integral grows
for k = 1:100
    [tau_k, pid_state] = pid_controller(omega_ref, omega_est, pid_state, params);
end

% Integral should be clamped at tau_max/Ki = 0.25
int_max = params.tau_max / params.pid.Ki;
if all(abs(pid_state.integral) <= int_max + 1e-10)
    fprintf('  PASS: Anti-windup active, integral clamped at ±%.3f\n', int_max);
    n_pass = n_pass + 1;
else
    fprintf('  FAIL: Integral exceeded limit (%.3f > %.3f)\n', max(abs(pid_state.integral)), int_max);
    n_fail = n_fail + 1;
end

%% ===== TEST 5: adrc_controller ESO convergence =====
fprintf('\n--- Test 5: adrc_controller ESO convergence ---\n');

omega_ref_a  = [15; 15; 15; 15];
omega_meas_a = [15; 15; 15; 15];  % Already at setpoint
imu_data_a.accel = [0; 0; 9.81];
imu_data_a.gyro  = [0; 0; 0];
adrc_state = [];

% Run 1000 steps (1s) — ESO should converge (w_o=100 → settle ~50ms)
for k = 1:1000
    [tau_a, adrc_state] = adrc_controller(omega_ref_a, omega_meas_a, imu_data_a, adrc_state, params);
end

% z1 should track omega_meas
z1_error = max(abs(adrc_state.z1 - omega_meas_a));
if z1_error < 0.1
    fprintf('  PASS: ESO z1 converged to measurement (error=%.4f rad/s)\n', z1_error);
    n_pass = n_pass + 1;
else
    fprintf('  FAIL: ESO z1 not converged (error=%.4f rad/s)\n', z1_error);
    n_fail = n_fail + 1;
end

% Tau should be bounded (not diverging). In open-loop test without plant,
% ESO estimates fictitious disturbance, so tau may not be exactly zero.
% Check it's within reasonable range.
if max(abs(tau_a)) < 1.0
    fprintf('  PASS: At setpoint, tau bounded (max=%.4f N·m)\n', max(abs(tau_a)));
    n_pass = n_pass + 1;
else
    fprintf('  FAIL: At setpoint, tau diverging (max=%.4f N·m)\n', max(abs(tau_a)));
    n_fail = n_fail + 1;
end

% Step change: suddenly change ref → tau should increase (try to speed up)
tau_before = tau_a;
omega_ref_a2 = [20; 20; 20; 20];
[tau_step, adrc_state] = adrc_controller(omega_ref_a2, omega_meas_a, imu_data_a, adrc_state, params);

% tau should increase compared to before (more torque to reach higher ref)
if all(tau_step > tau_before - 0.01)
    fprintf('  PASS: Step in ref → tau increased (%.3f → %.3f N·m)\n', tau_before(1), tau_step(1));
    n_pass = n_pass + 1;
else
    fprintf('  FAIL: Wrong response to ref step (%.3f → %.3f)\n', tau_before(1), tau_step(1));
    n_fail = n_fail + 1;
end

%% ===== TEST 6: Pipeline round-trip =====
fprintf('\n--- Test 6: Pipeline encoder → PID → pwm_output → pwm_capture ---\n');
clear encoder_reader;

% Simulate: true ω=10, generate encoder counts, read, control, output, capture
omega_true_pipe = [10; 10; 10; 10];
clear encoder_pulse_gen;
enc_c = encoder_pulse_gen(omega_true_pipe, dt, params);

% Wait a few steps for filter to settle
for k = 1:30
    enc_c = encoder_pulse_gen(omega_true_pipe, dt, params);
    omega_e = encoder_reader(enc_c, dt, params);
end

% Now do one full pipeline pass
omega_ref_pipe = [12; 12; 12; 12];  % Slightly above current speed
pid_st = [];
for k = 1:10
    [tau_pipe, pid_st] = pid_controller(omega_ref_pipe, omega_e, pid_st, params);
end
pwm_pipe = pwm_output(tau_pipe, params);
tau_received = pwm_capture(pwm_pipe, params);

% tau_received should be positive (trying to speed up) and within tau_max
if all(tau_received > 0) && all(tau_received <= params.tau_max)
    fprintf('  PASS: Pipeline produces valid torque (%.4f N·m)\n', tau_received(1));
    n_pass = n_pass + 1;
else
    fprintf('  FAIL: Pipeline output invalid (tau=%.4f)\n', tau_received(1));
    n_fail = n_fail + 1;
end

%% ===== SUMMARY =====
fprintf('\n========================================\n');
fprintf('M4 Unit Tests: %d PASS / %d FAIL / %d total\n', n_pass, n_fail, n_pass + n_fail);
fprintf('========================================\n');

if n_fail == 0
    fprintf('All tests passed — ready for closed-loop (step 4.7)\n');
else
    fprintf('Fix failures before running closed-loop.\n');
end
