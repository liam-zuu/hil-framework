%% TEST_M3_SIGNAL_CONDITIONING
% Step 3.6: Verify all 5 nucleoh7 modules against ground truth.
%
% For each module, we feed known inputs and compare the output
% after signal conditioning with the ideal (ground truth) values.
% Metrics: max error, RMS error, SNR where applicable.

clear; clc; close all;
clear encoder_pulse_gen;  % reset persistent accumulator

%% Setup
% addpath('../nucleoh7');
% addpath('../esp32');
% addpath('../rpi5');
% addpath('../scripts');
params = params_mecanum();
dt = params.dt;

fprintf('========================================\n');
fprintf('  M3 Signal Conditioning — Test Suite\n');
fprintf('========================================\n\n');

pass_count = 0;
fail_count = 0;
total_tests = 0;

%% ============================================================
%  TEST 1: SPI Interface (spi_interface.m)
%  ============================================================
fprintf('--- TEST 1: SPI Interface ---\n');

% 1a: Uplink (torque)
tau_true = [0.1234; -0.2567; 0.0001; 0.4999];
[tau_spi, ~] = spi_interface('uplink', tau_true, params);
err_tau = abs(tau_spi - tau_true);

% Expected quantization step: 2*range / 2^bits
lsb_tau = 2 * params.spi.tau_range / 2^params.spi.float_bits;
fprintf('  Uplink torque:\n');
fprintf('    Max error:  %.2e  (LSB = %.2e)\n', max(err_tau), lsb_tau);

total_tests = total_tests + 1;
if max(err_tau) <= lsb_tau / 2 + 1e-15
    fprintf('    PASS: error within +/-0.5 LSB\n');
    pass_count = pass_count + 1;
else
    fprintf('    FAIL: error exceeds +/-0.5 LSB\n');
    fail_count = fail_count + 1;
end

% 1b: Downlink (state vector)
x_true = [1.234; -0.567; 1.5708; 0.5; -0.3; 2.0; ...
          10.0; -15.0; 20.0; -5.0];
[~, x_spi] = spi_interface('downlink', x_true, params);
err_state = abs(x_spi - x_true);

fprintf('  Downlink states:\n');
for i = 1:10
    lsb_i = 2 * params.spi.state_ranges(i) / 2^params.spi.float_bits;
    fprintf('    %5s: true=%.4f  spi=%.4f  err=%.2e  lsb=%.2e\n', ...
            params.state_names{i}, x_true(i), x_spi(i), err_state(i), lsb_i);
end

total_tests = total_tests + 1;
max_lsb_ratio = 0;
for i = 1:10
    lsb_i = 2 * params.spi.state_ranges(i) / 2^params.spi.float_bits;
    max_lsb_ratio = max(max_lsb_ratio, err_state(i) / lsb_i);
end
if max_lsb_ratio <= 0.5 + 1e-10
    fprintf('    PASS: all states within +/-0.5 LSB\n');
    pass_count = pass_count + 1;
else
    fprintf('    FAIL: some states exceed +/-0.5 LSB (ratio=%.4f)\n', max_lsb_ratio);
    fail_count = fail_count + 1;
end

% 1c: Clipping test
tau_clip = [2.0; -3.0; 0.0; 0.5];
[tau_clipped, ~] = spi_interface('uplink', tau_clip, params);
total_tests = total_tests + 1;
if all(abs(tau_clipped) <= params.spi.tau_range + 1e-10)
    fprintf('  Clipping: PASS (values clamped to +/-%.1f)\n', params.spi.tau_range);
    pass_count = pass_count + 1;
else
    fprintf('  Clipping: FAIL\n');
    fail_count = fail_count + 1;
end

fprintf('\n');

%% ============================================================
%  TEST 2: Encoder Pulse Gen (encoder_pulse_gen.m)
%  ============================================================
fprintf('--- TEST 2: Encoder Pulse Generator ---\n');

clear encoder_pulse_gen;

% 2a: Constant speed over 100 steps
omega_test = [10.0; -5.0; 20.0; 0.0];
N_enc = 100;
total_counts = zeros(4,1);
for k = 1:N_enc
    counts = encoder_pulse_gen(omega_test, dt, params);
    total_counts = total_counts + counts;
end

expected_counts = omega_test * N_enc * dt * params.enc_ppr / (2*pi);
err_enc = abs(total_counts - expected_counts);

fprintf('  Constant speed, %d steps:\n', N_enc);
for i = 1:4
    fprintf('    Wheel %d: omega=%.1f  expected=%.1f  actual=%d  err=%.2f\n', ...
            i, omega_test(i), expected_counts(i), total_counts(i), err_enc(i));
end

total_tests = total_tests + 1;
if all(err_enc < 3)
    fprintf('    PASS: total counts within +/-3 of expected\n');
    pass_count = pass_count + 1;
else
    fprintf('    FAIL: count error too large\n');
    fail_count = fail_count + 1;
end

% 2b: Zero speed
clear encoder_pulse_gen;
zero_counts = zeros(4,1);
for k = 1:100
    zero_counts = zero_counts + encoder_pulse_gen(zeros(4,1), dt, params);
end

total_tests = total_tests + 1;
if all(abs(zero_counts) < 5)
    fprintf('  Zero speed: PASS (total = [%d %d %d %d])\n', zero_counts);
    pass_count = pass_count + 1;
else
    fprintf('  Zero speed: FAIL (total = [%d %d %d %d])\n', zero_counts);
    fail_count = fail_count + 1;
end

% 2c: Low speed accumulator carry-over
clear encoder_pulse_gen;
omega_low = [0.5; 0.5; 0.5; 0.5];
low_counts = zeros(4,1);
for k = 1:100
    low_counts = low_counts + encoder_pulse_gen(omega_low, dt, params);
end
expected_low = 0.5 * 100 * dt * params.enc_ppr / (2*pi);

total_tests = total_tests + 1;
if all(abs(low_counts - expected_low) < 3)
    fprintf('  Low speed accumulator: PASS (expected~%.1f, got=[%d %d %d %d])\n', ...
            expected_low, low_counts);
    pass_count = pass_count + 1;
else
    fprintf('  Low speed accumulator: FAIL (expected~%.1f, got=[%d %d %d %d])\n', ...
            expected_low, low_counts);
    fail_count = fail_count + 1;
end

fprintf('\n');

%% ============================================================
%  TEST 3: IMU Packet Encode/Decode (imu_packet_enc + imu_reader)
%  ============================================================
fprintf('--- TEST 3: IMU Packet Encode/Decode ---\n');

% 3a: Round-trip
accel_true = [0.5; -1.2; 9.81];
gyro_true  = [0.001; -0.003; 1.5];

packet = imu_packet_enc(accel_true, gyro_true, params);
[accel_dec, gyro_dec, valid] = imu_reader(packet, params);

err_accel = abs(accel_dec - accel_true);
err_gyro  = abs(gyro_dec - gyro_true);

accel_lsb = 2 * params.imu_accel_range / 2^params.imu_adc_bits;
gyro_lsb  = 2 * params.imu_gyro_range  / 2^params.imu_adc_bits;

fprintf('  Round-trip encode/decode:\n');
fprintf('    Accel err: [%.2e, %.2e, %.2e]  (LSB=%.2e m/s^2)\n', err_accel, accel_lsb);
fprintf('    Gyro  err: [%.2e, %.2e, %.2e]  (LSB=%.2e rad/s)\n', err_gyro, gyro_lsb);

total_tests = total_tests + 1;
if valid && all(err_accel <= accel_lsb/2 + 1e-12) && all(err_gyro <= gyro_lsb/2 + 1e-12)
    fprintf('    PASS: valid packet, errors within +/-0.5 LSB\n');
    pass_count = pass_count + 1;
else
    fprintf('    FAIL: valid=%d, max_accel=%.2e, max_gyro=%.2e\n', ...
            valid, max(err_accel), max(err_gyro));
    fail_count = fail_count + 1;
end

% 3b: Checksum corruption detection
packet_bad = packet;
packet_bad.accel_raw(1) = packet_bad.accel_raw(1) + int32(1);
[~, ~, valid_bad] = imu_reader(packet_bad, params);

total_tests = total_tests + 1;
if ~valid_bad
    fprintf('  Checksum corruption: PASS (detected)\n');
    pass_count = pass_count + 1;
else
    fprintf('  Checksum corruption: FAIL (not detected)\n');
    fail_count = fail_count + 1;
end

% 3c: ADC clipping
accel_clip = [100; -100; 0];
packet_clip = imu_packet_enc(accel_clip, zeros(3,1), params);
[accel_clip_dec, ~, ~] = imu_reader(packet_clip, params);

total_tests = total_tests + 1;
if all(abs(accel_clip_dec) <= params.imu_accel_range + accel_lsb)
    fprintf('  ADC clipping: PASS (clamped to +/-%.1f m/s^2)\n', params.imu_accel_range);
    pass_count = pass_count + 1;
else
    fprintf('  ADC clipping: FAIL\n');
    fail_count = fail_count + 1;
end

fprintf('\n');

%% ============================================================
%  TEST 4: PWM Capture (pwm_capture.m)
%  ============================================================
fprintf('--- TEST 4: PWM Capture ---\n');

% 4a: Linear mapping (zero jitter for deterministic test)
params_nj = params;
params_nj.pwm_jitter_sigma = 0;

pwm_test = [0.5; -0.8; 0.01; 1.0];
tau_cap = pwm_capture(pwm_test, params_nj);

pwm_res = params.pwm_res;
expected_pwm = round(pwm_test * pwm_res) / pwm_res;
expected_pwm(3) = 0;  % 0.01 < deadband(0.02)
expected_tau = expected_pwm * params.tau_max;
err_pwm = abs(tau_cap - expected_tau);

fprintf('  No-jitter test:\n');
for i = 1:4
    fprintf('    Ch%d: pwm=%.3f  tau=%.4f  expected=%.4f  err=%.2e\n', ...
            i, pwm_test(i), tau_cap(i), expected_tau(i), err_pwm(i));
end

total_tests = total_tests + 1;
if all(err_pwm < 1e-10)
    fprintf('    PASS: linear mapping + deadband correct\n');
    pass_count = pass_count + 1;
else
    fprintf('    FAIL\n');
    fail_count = fail_count + 1;
end

% 4b: Deadband edge cases
pwm_db = [0.019; -0.019; 0.021; -0.021];
tau_db = pwm_capture(pwm_db, params_nj);

total_tests = total_tests + 1;
if tau_db(1) == 0 && tau_db(2) == 0 && tau_db(3) ~= 0 && tau_db(4) ~= 0
    fprintf('  Deadband edge: PASS (below->0, above->nonzero)\n');
    pass_count = pass_count + 1;
else
    fprintf('  Deadband edge: FAIL (tau=[%.4f %.4f %.4f %.4f])\n', tau_db);
    fail_count = fail_count + 1;
end

% 4c: Saturation
pwm_sat = [1.5; -1.5; 0; 0];
tau_sat = pwm_capture(pwm_sat, params_nj);

total_tests = total_tests + 1;
if abs(tau_sat(1)) <= params.tau_max + 1e-10 && abs(tau_sat(2)) <= params.tau_max + 1e-10
    fprintf('  Saturation: PASS (clamped to +/-%.2f Nm)\n', params.tau_max);
    pass_count = pass_count + 1;
else
    fprintf('  Saturation: FAIL\n');
    fail_count = fail_count + 1;
end

fprintf('\n');

%% ============================================================
%  TEST 5: GPIO Sync (gpio_sync.m)
%  ============================================================
fprintf('--- TEST 5: GPIO Sync ---\n');

N_sync = 10000;
sync_results = false(1, N_sync);
for k = 1:N_sync
    sync_results(k) = gpio_sync(k, [true; true; true], params);
end
fail_rate = sum(~sync_results) / N_sync;

fprintf('  Normal operation (%d trials):\n', N_sync);
fprintf('    Pass rate: %.2f%%\n', 100*(1-fail_rate));
fprintf('    Fail rate: %.4f%%\n', 100*fail_rate);

total_tests = total_tests + 1;
if (1-fail_rate) > 0.99
    fprintf('    PASS: >99%% sync success\n');
    pass_count = pass_count + 1;
else
    fprintf('    FAIL: too many sync failures\n');
    fail_count = fail_count + 1;
end

% Cluster not done
total_tests = total_tests + 1;
sync_fail = gpio_sync(1, [true; false; true], params);
if ~sync_fail
    fprintf('  Cluster not done: PASS (sync_ok=false)\n');
    pass_count = pass_count + 1;
else
    fprintf('  Cluster not done: FAIL (sync_ok=true)\n');
    fail_count = fail_count + 1;
end

fprintf('\n');

%% ============================================================
%  TEST 6: Full H7 Pipeline Round-trip
%  ============================================================
fprintf('--- TEST 6: Full H7 Pipeline Round-trip ---\n');

clear encoder_pulse_gen;

x_test = [0.5; 0.3; 0.785; 0.4; -0.2; 1.0; 12.0; -8.0; 15.0; 5.0];
x_prev = x_test * 0.99;
tau_in = [0.15; -0.10; 0.20; -0.05];

% Forward path (PWM -> H7 -> RPi5)
tau_captured = pwm_capture(tau_in / params.tau_max, params);
[tau_spi_up, ~] = spi_interface('uplink', tau_captured, params);

% Sensor path (RPi5 -> H7 -> ESP32)
[~, x_spi_down] = spi_interface('downlink', x_test, params);
omega_h7 = x_spi_down(7:10);
enc = encoder_pulse_gen(omega_h7, dt, params);

[accel_sim, gyro_sim, ~] = imu_model(x_test, x_prev, dt, [], params);
pkt = imu_packet_enc(accel_sim, gyro_sim, params);

% ESP32 decode
omega_decoded = encoder_reader(enc, dt, params);
[accel_decoded, gyro_decoded, pkt_valid] = imu_reader(pkt, params);

fprintf('  Torque pipeline:\n');
fprintf('    Input:   [%.4f, %.4f, %.4f, %.4f]\n', tau_in);
fprintf('    Output:  [%.4f, %.4f, %.4f, %.4f]\n', tau_spi_up);

fprintf('  Encoder pipeline:\n');
fprintf('    True omega: [%.2f, %.2f, %.2f, %.2f]\n', x_test(7:10));
fprintf('    Decoded:    [%.2f, %.2f, %.2f, %.2f]\n', omega_decoded);
fprintf('    Error:      [%.2f, %.2f, %.2f, %.2f] rad/s\n', ...
        abs(omega_decoded - x_test(7:10)));

fprintf('  IMU pipeline:\n');
fprintf('    Accel true: [%.4f, %.4f, %.4f]\n', accel_sim);
fprintf('    Accel dec:  [%.4f, %.4f, %.4f]\n', accel_decoded);
fprintf('    Packet valid: %d\n', pkt_valid);

total_tests = total_tests + 1;
if pkt_valid
    fprintf('    PASS: full pipeline functional\n');
    pass_count = pass_count + 1;
else
    fprintf('    FAIL: packet invalid\n');
    fail_count = fail_count + 1;
end

%% ============================================================
%  TEST 7: Encoder SNR over 1000 steps
%  ============================================================
fprintf('\n--- TEST 7: Encoder SNR (1000 steps) ---\n');

clear encoder_pulse_gen;
omega_snr = [5.0; 10.0; 20.0; 30.0];
N_snr = 1000;
omega_est_hist = zeros(4, N_snr);
for k = 1:N_snr
    enc_k = encoder_pulse_gen(omega_snr, dt, params);
    omega_est_hist(:,k) = encoder_reader(enc_k, dt, params);
end

for i = 1:4
    signal_power = omega_snr(i)^2;
    noise_power  = var(omega_est_hist(i,:) - omega_snr(i));
    snr_db = 10*log10(signal_power / max(noise_power, 1e-20));
    fprintf('  Wheel %d (omega=%.0f): mean=%.2f  std=%.2f  SNR=%.1f dB\n', ...
            i, omega_snr(i), mean(omega_est_hist(i,:)), ...
            std(omega_est_hist(i,:)), snr_db);
end
fprintf('  (INFO: quantization noise dominant at low speed)\n');

%% ============================================================
%  SUMMARY
%  ============================================================
fprintf('\n========================================\n');
fprintf('  RESULTS: %d/%d tests passed\n', pass_count, total_tests);
if fail_count > 0
    fprintf('  *** %d FAILURES ***\n', fail_count);
else
    fprintf('  ALL PASS\n');
end
fprintf('========================================\n');
