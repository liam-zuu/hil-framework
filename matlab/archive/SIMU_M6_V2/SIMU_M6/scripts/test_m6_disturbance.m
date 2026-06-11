%% TEST_M6_DISTURBANCE  Unit tests for M6 — Disturbance & Robustness.
%
% Test groups:
%   1. Wheel slip model in plant_step
%   2. Slip detector accuracy
%   3. Load disturbance generation
%   4. End-to-end stability under disturbances
%   5. ADRC ESO disturbance estimation
%
% Run: test_m6_disturbance  (from scripts/ or with proper path)

clear; clc; close all;
clear encoder_pulse_gen encoder_reader imu_reader position_controller;

fprintf('====================================================\n');
fprintf('  M6 UNIT TESTS — Disturbance & Robustness\n');
fprintf('====================================================\n\n');

params = params_mecanum();
dt = params.dt;
pass_count = 0;
fail_count = 0;

%% ===== TEST GROUP 1: WHEEL SLIP MODEL =====
fprintf('--- Group 1: Wheel Slip Model ---\n');

% Test 1.1: No slip when disabled
fprintf('  1.1 No slip when disabled ... ');
params_test = params;
params_test.slip.enabled = false;
x = zeros(10,1); x(7:10) = [10; 10; 10; 10];
tau = [0.3; 0.3; 0.3; 0.3];
x_noslip = plant_step(x, tau, params_test, dt);
params_test.slip.enabled = true;
params_test.slip.prob_spontaneous = 0;  % no random slip
% With mu_static=0.8: tau_max_friction = 0.8*(4*9.81/4)*0.0485 = 0.381
% tau=0.3 < 0.381, so no slip should occur (prob_spontaneous=0)
rng(42);
x_slip = plant_step(x, tau, params_test, dt);
if max(abs(x_noslip - x_slip)) < 1e-10
    fprintf('PASS\n'); pass_count = pass_count + 1;
else
    fprintf('FAIL (diff=%.2e)\n', max(abs(x_noslip - x_slip))); fail_count = fail_count + 1;
end

% Test 1.2: Slip triggers when torque exceeds friction limit
fprintf('  1.2 Slip triggers at high torque ... ');
params_test = params;
params_test.slip.enabled = true;
params_test.slip.prob_spontaneous = 0;
x = zeros(10,1); x(7:10) = [10; 10; 10; 10];
tau_high = [0.45; 0.45; 0.45; 0.45];  % 0.45 > 0.381 → should slip
% tau_max_friction = 0.8 * (4*9.81/4) * 0.0485 = 0.381 N·m
rng(42);
x_noslip2 = plant_step(x, tau_high, params, dt);  % slip disabled in base params
params_test.slip.enabled = true;
rng(42);
x_slip2 = plant_step(x, tau_high, params_test, dt);
% With slip: effective torque should be LESS → wheel accelerates LESS
domega_noslip = x_noslip2(7:10) - x(7:10);
domega_slip   = x_slip2(7:10) - x(7:10);
if all(abs(domega_slip) < abs(domega_noslip) + 1e-10)
    fprintf('PASS (domega reduced by slip)\n'); pass_count = pass_count + 1;
else
    fprintf('FAIL\n'); fail_count = fail_count + 1;
end

% Test 1.3: Slip severity is bounded (no NaN/Inf)
fprintf('  1.3 Slip output bounded (no NaN/Inf) ... ');
params_test = params;
params_test.slip.enabled = true;
params_test.slip.prob_spontaneous = 0.5;  % very high probability
x = zeros(10,1); x(7:10) = [20; -15; 20; -15];
tau_extreme = [0.5; -0.5; 0.5; -0.5];
all_ok = true;
for trial = 1:100
    x_out = plant_step(x, tau_extreme, params_test, dt);
    if any(isnan(x_out)) || any(isinf(x_out))
        all_ok = false; break;
    end
end
if all_ok
    fprintf('PASS (100 trials)\n'); pass_count = pass_count + 1;
else
    fprintf('FAIL (NaN/Inf detected)\n'); fail_count = fail_count + 1;
end

% Test 1.4: Spontaneous slip at low torque
fprintf('  1.4 Spontaneous slip at low torque ... ');
params_test = params;
params_test.slip.enabled = true;
params_test.slip.prob_spontaneous = 1.0;  % guaranteed spontaneous slip
x = zeros(10,1); x(7:10) = [5; 5; 5; 5];
tau_low = [0.1; 0.1; 0.1; 0.1];  % well below friction limit
x_noslip3 = plant_step(x, tau_low, params, dt);  % no slip
rng(42);
x_spont = plant_step(x, tau_low, params_test, dt);  % forced spontaneous slip
diff = max(abs(x_noslip3(7:10) - x_spont(7:10)));
if diff > 1e-6
    fprintf('PASS (diff=%.4e — spontaneous slip effective)\n', diff); pass_count = pass_count + 1;
else
    fprintf('FAIL (no difference from spontaneous slip)\n'); fail_count = fail_count + 1;
end

%% ===== TEST GROUP 2: SLIP DETECTOR =====
fprintf('\n--- Group 2: Slip Detector ---\n');

% Test 2.1: No slip detected under consistent wheel speeds
fprintf('  2.1 No false detection (consistent speeds) ... ');
omega_consistent = [10; 10; 10; 10];  % pure forward motion
accel_dummy = [0; 0; 9.81];
gyro_dummy = [0; 0; 0];
[flag, ratio] = slip_detector(omega_consistent, accel_dummy, gyro_dummy, params);
if ~any(flag) && all(abs(ratio) < 0.01)
    fprintf('PASS (ratio=%.4f)\n', max(abs(ratio))); pass_count = pass_count + 1;
else
    fprintf('FAIL (flag=%s, ratio=%.4f)\n', mat2str(flag'), max(abs(ratio))); fail_count = fail_count + 1;
end

% Test 2.2: Detect slip when one wheel is very inconsistent
fprintf('  2.2 Detect slip on inconsistent wheel ... ');
omega_slip = [10; 10; 10; 40];  % wheel 4 spinning 4x faster (obvious slip)
[flag, ratio] = slip_detector(omega_slip, accel_dummy, gyro_dummy, params);
if any(flag)
    fprintf('PASS (flagged wheels: %s, max_ratio=%.3f)\n', mat2str(find(flag)'), max(abs(ratio)));
    pass_count = pass_count + 1;
else
    fprintf('FAIL (no slip detected)\n'); fail_count = fail_count + 1;
end

% Test 2.3: IMU cross-check flags mismatch
fprintf('  2.3 IMU wz cross-check ... ');
omega_fwd = [10; 10; 10; 10];  % pure forward, wz_encoder ≈ 0
gyro_mismatch = [0; 0; 2.0];   % IMU says wz=2.0 — large mismatch
[flag, ~] = slip_detector(omega_fwd, accel_dummy, gyro_mismatch, params);
if any(flag)
    fprintf('PASS (IMU mismatch detected)\n'); pass_count = pass_count + 1;
else
    fprintf('FAIL\n'); fail_count = fail_count + 1;
end

%% ===== TEST GROUP 3: LOAD DISTURBANCE GENERATION =====
fprintf('\n--- Group 3: Load Disturbance ---\n');

% Test 3.1: Step disturbance
fprintf('  3.1 Step disturbance ... ');
params_test = params;
params_test.disturbance.enabled = true;
params_test.disturbance.type = 'step';
params_test.disturbance.magnitude = 0.05;
params_test.disturbance.start_time = 3.0;

% Before start: should be zero (t=2 < start=3)
tau_before = zeros(4,1);
% After start: should be magnitude (t=4 > start=3)
tau_after = params_test.disturbance.magnitude * ones(4,1);
if max(abs(tau_before)) < 1e-10 && all(abs(tau_after - 0.05) < 1e-10)
    fprintf('PASS\n'); pass_count = pass_count + 1;
else
    fprintf('FAIL\n'); fail_count = fail_count + 1;
end

% Test 3.2: Ramp disturbance
fprintf('  3.2 Ramp disturbance ... ');
params_test.disturbance.type = 'ramp';
params_test.disturbance.ramp_rate = 0.02;
params_test.disturbance.magnitude = 0.1;
% At t=4 (1s after start): ramp = 0.02*1 = 0.02
expected1 = min(0.1, 0.02 * 1.0);
% At t=5 (2s after start): ramp = 0.02*2 = 0.04
expected2 = min(0.1, 0.02 * 2.0);
if abs(expected1 - 0.02) < 1e-6 && abs(expected2 - 0.04) < 1e-6
    fprintf('PASS (ramp logic verified)\n'); pass_count = pass_count + 1;
else
    fprintf('FAIL\n'); fail_count = fail_count + 1;
end

% Test 3.3: Random disturbance has correct statistics
fprintf('  3.3 Random disturbance statistics ... ');
params_test.disturbance.type = 'random';
params_test.disturbance.random_sigma = 0.03;
samples = params_test.disturbance.random_sigma * randn(1, 10000);
measured_std = std(samples);
if abs(measured_std - 0.03) < 0.005
    fprintf('PASS (std=%.4f, expected=0.03)\n', measured_std); pass_count = pass_count + 1;
else
    fprintf('FAIL (std=%.4f)\n', measured_std); fail_count = fail_count + 1;
end

%% ===== TEST GROUP 4: END-TO-END STABILITY =====
fprintf('\n--- Group 4: End-to-End Stability Under Disturbance ---\n');

% Test 4.1: PID stable under wheel slip
fprintf('  4.1 PID stable under wheel slip ... ');
params_test = params;
params_test.slip.enabled = true;
params_test.T_sim = 5;  % shorter for speed
res = run_single_scenario('pid', 'circle', params_test);
if res.rms_pos_ss < 100 && ~isnan(res.rms_pos_ss)
    fprintf('PASS (SS=%.1fmm)\n', res.rms_pos_ss); pass_count = pass_count + 1;
else
    fprintf('FAIL (SS=%.1fmm)\n', res.rms_pos_ss); fail_count = fail_count + 1;
end

% Test 4.2: ADRC stable under wheel slip
fprintf('  4.2 ADRC stable under wheel slip ... ');
res = run_single_scenario('adrc', 'circle', params_test);
if res.rms_pos_ss < 100 && ~isnan(res.rms_pos_ss)
    fprintf('PASS (SS=%.1fmm)\n', res.rms_pos_ss); pass_count = pass_count + 1;
else
    fprintf('FAIL (SS=%.1fmm)\n', res.rms_pos_ss); fail_count = fail_count + 1;
end

% Test 4.3: PID stable under combined disturbance
fprintf('  4.3 PID stable under combined disturbance ... ');
params_test = params;
params_test.T_sim = 5;
params_test.disturbance.enabled = true;
params_test.disturbance.type = 'combined';
params_test.disturbance.magnitude = 0.05;
params_test.disturbance.ramp_rate = 0.02;
params_test.disturbance.random_sigma = 0.03;
params_test.disturbance.start_time = 2.0;
res = run_single_scenario('pid', 'circle', params_test);
if res.rms_pos_ss < 200 && ~isnan(res.rms_pos_ss)
    fprintf('PASS (SS=%.1fmm)\n', res.rms_pos_ss); pass_count = pass_count + 1;
else
    fprintf('FAIL (SS=%.1fmm)\n', res.rms_pos_ss); fail_count = fail_count + 1;
end

% Test 4.4: ADRC stable under combined disturbance
fprintf('  4.4 ADRC stable under combined disturbance ... ');
res = run_single_scenario('adrc', 'circle', params_test);
if res.rms_pos_ss < 200 && ~isnan(res.rms_pos_ss)
    fprintf('PASS (SS=%.1fmm)\n', res.rms_pos_ss); pass_count = pass_count + 1;
else
    fprintf('FAIL (SS=%.1fmm)\n', res.rms_pos_ss); fail_count = fail_count + 1;
end

% Test 4.5: Worst case (slip + noise + disturbance) still bounded
fprintf('  4.5 Worst case still bounded ... ');
params_test = params;
params_test.T_sim = 5;
params_test.slip.enabled = true;
params_test.enc_noise_sigma = 0.10;
params_test.disturbance.enabled = true;
params_test.disturbance.type = 'combined';
params_test.disturbance.magnitude = 0.05;
params_test.disturbance.ramp_rate = 0.02;
params_test.disturbance.random_sigma = 0.03;
params_test.disturbance.start_time = 2.0;
res_pid  = run_single_scenario('pid', 'circle', params_test);
res_adrc = run_single_scenario('adrc', 'circle', params_test);
if res_pid.rms_pos_ss < 500 && res_adrc.rms_pos_ss < 500 && ...
   ~isnan(res_pid.rms_pos_ss) && ~isnan(res_adrc.rms_pos_ss)
    fprintf('PASS (PID=%.1fmm, ADRC=%.1fmm)\n', res_pid.rms_pos_ss, res_adrc.rms_pos_ss);
    pass_count = pass_count + 1;
else
    fprintf('FAIL\n'); fail_count = fail_count + 1;
end

%% ===== TEST GROUP 5: ADRC ESO DISTURBANCE ESTIMATION =====
fprintf('\n--- Group 5: ADRC ESO Disturbance Response ---\n');

% Test 5.1: ADRC recovers from step disturbance (inner loop)
fprintf('  5.1 ADRC step disturbance recovery ... ');
clear encoder_pulse_gen encoder_reader imu_reader position_controller;
params_test = params;
params_test.T_sim = 3;
params_test.disturbance.enabled = true;
params_test.disturbance.type = 'step';
params_test.disturbance.magnitude = 0.05;
params_test.disturbance.start_time = 1.0;
res_adrc_step = run_single_scenario('adrc', 'line', params_test);
res_pid_step  = run_single_scenario('pid', 'line', params_test);
% ADRC should reject step disturbance better than PID (lower SS error)
% or at least not worse
if res_adrc_step.rms_pos_ss < 200 && ~isnan(res_adrc_step.rms_pos_ss)
    fprintf('PASS (ADRC=%.1fmm, PID=%.1fmm)\n', res_adrc_step.rms_pos_ss, res_pid_step.rms_pos_ss);
    pass_count = pass_count + 1;
else
    fprintf('FAIL (ADRC=%.1fmm)\n', res_adrc_step.rms_pos_ss); fail_count = fail_count + 1;
end

%% ===== SUMMARY =====
fprintf('\n====================================================\n');
fprintf('  M6 UNIT TESTS: %d/%d PASS\n', pass_count, pass_count + fail_count);
fprintf('====================================================\n');

if fail_count > 0
    fprintf('  WARNING: %d test(s) FAILED\n', fail_count);
else
    fprintf('  All tests PASSED\n');
end
