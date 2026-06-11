%% TEST_M5_INTEGRATION  Integration tests for M5 — Full Integration.
%
% Tests:
%   1. Pose estimator accuracy (short run, no drift)
%   2. Position controller direction correctness
%   3. Position controller zero-error passthrough
%   4. Outer + inner loop: line trajectory convergence (PID)
%   5. Outer + inner loop: line trajectory convergence (ADRC)
%   6. Circle trajectory: RMS improvement over M4
%   7. All trajectories produce valid data (no NaN/Inf)
%   8. Heading tracking convergence

clear; clc; close all;
fprintf('===== M5 Integration Tests =====\n\n');

params = params_mecanum();
dt = params.dt;
pass_count = 0;
fail_count = 0;

r = params.r;
L = params.lx + params.ly;

%% ===== TEST 1: Pose Estimator — static =====
fprintf('Test 1: Pose estimator (static) ... ');
pe_state = [];
pe_state.x = 1.0; pe_state.y = 2.0; pe_state.theta = 0.5;
omega_zero = zeros(4,1);
gyro_zero = [0; 0; 0];
[pose, pe_state] = pose_estimator(omega_zero, gyro_zero, pe_state, params);
if abs(pose(1) - 1.0) < 1e-10 && abs(pose(2) - 2.0) < 1e-10 && abs(pose(3) - 0.5) < 1e-6
    fprintf('PASS\n'); pass_count = pass_count + 1;
else
    fprintf('FAIL (pose=[%.4f, %.4f, %.4f])\n', pose); fail_count = fail_count + 1;
end

%% ===== TEST 2: Pose Estimator — forward motion =====
fprintf('Test 2: Pose estimator (forward 100 steps) ... ');
pe_state2 = [];
pe_state2.x = 0; pe_state2.y = 0; pe_state2.theta = 0;
% All wheels forward at 10 rad/s → vx = r*10 = 0.485 m/s
omega_fwd = 10 * ones(4,1);
gyro_fwd = [0; 0; 0];
for i = 1:100
    [pose2, pe_state2] = pose_estimator(omega_fwd, gyro_fwd, pe_state2, params);
end
expected_x = params.r * 10 * 100 * dt;  % 0.0485 * 10 * 0.1 = 0.0485 m
if abs(pose2(1) - expected_x) < 0.001 && abs(pose2(2)) < 1e-6
    fprintf('PASS (x=%.4f, expected=%.4f)\n', pose2(1), expected_x);
    pass_count = pass_count + 1;
else
    fprintf('FAIL (x=%.4f, expected=%.4f, y=%.6f)\n', pose2(1), expected_x, pose2(2));
    fail_count = fail_count + 1;
end

%% ===== TEST 3: Position Controller — zero error passthrough =====
fprintf('Test 3: Position controller (zero error → feedforward only) ... ');
clear position_controller;  % Reset integral
pose_ref = [1; 2; 0.5];
pose_est = [1; 2; 0.5];
vel_ref  = [0.3; 0; 1.0];
vel_cmd = position_controller(pose_ref, pose_est, vel_ref, params);
if abs(vel_cmd(1) - 0.3) < 1e-10 && abs(vel_cmd(2)) < 1e-10 && abs(vel_cmd(3) - 1.0) < 1e-10
    fprintf('PASS\n'); pass_count = pass_count + 1;
else
    fprintf('FAIL (vel_cmd=[%.4f, %.4f, %.4f])\n', vel_cmd); fail_count = fail_count + 1;
end

%% ===== TEST 4: Position Controller — correction direction =====
fprintf('Test 4: Position controller (error → correction direction) ... ');
clear position_controller;  % Reset integral
% Robot at origin, reference at (0.1, 0, 0), theta=0 → should correct +vx
pose_ref4 = [0.1; 0; 0];
pose_est4 = [0; 0; 0];
vel_ref4  = [0; 0; 0];
vel_cmd4 = position_controller(pose_ref4, pose_est4, vel_ref4, params);
% Should get positive vx correction
if vel_cmd4(1) > 0 && abs(vel_cmd4(2)) < 1e-10 && abs(vel_cmd4(3)) < 1e-10
    fprintf('PASS (vx_cmd=%.4f)\n', vel_cmd4(1)); pass_count = pass_count + 1;
else
    fprintf('FAIL (vel_cmd=[%.4f, %.4f, %.4f])\n', vel_cmd4); fail_count = fail_count + 1;
end

%% ===== TEST 5: Position Controller — heading correction =====
fprintf('Test 5: Position controller (heading error → wz correction) ... ');
clear position_controller;  % Reset integral
pose_ref5 = [0; 0; 0.3];
pose_est5 = [0; 0; 0];
vel_ref5  = [0; 0; 0];
vel_cmd5 = position_controller(pose_ref5, pose_est5, vel_ref5, params);
if vel_cmd5(3) > 0  % positive wz to correct CCW
    fprintf('PASS (wz_cmd=%.4f)\n', vel_cmd5(3)); pass_count = pass_count + 1;
else
    fprintf('FAIL (wz_cmd=%.4f)\n', vel_cmd5(3)); fail_count = fail_count + 1;
end

%% ===== TEST 6: Line trajectory — PID convergence =====
fprintf('Test 6: Line trajectory, PID, 5s — convergence ... ');
clear encoder_pulse_gen encoder_reader imu_reader position_controller;
T_test = 5; N_test = round(T_test/dt);
traj = trajectory_generator('line', T_test, dt, params);

x0 = zeros(10,1);
x0(1) = traj.x_ref(1); x0(2) = traj.y_ref(1); x0(3) = traj.theta_ref(1);
sm = state_manager('init', [], x0, params);
pid_state = []; imu_state = []; pe_state = [];
pe_state.x = x0(1); pe_state.y = x0(2); pe_state.theta = x0(3);

enc_counts = encoder_pulse_gen(x0(7:10), dt, params);
[a_i, g_i, imu_state] = imu_model(x0, x0, dt, imu_state, params);
imu_packet = imu_packet_enc(a_i, g_i, params);

pos_err = zeros(1, N_test);
for k = 1:N_test
    omega_est = encoder_reader(enc_counts, dt, params);
    [am, gm, ~] = imu_reader(imu_packet, params);
    imu_data.accel = am; imu_data.gyro = gm;

    [pose_est, pe_state] = pose_estimator(omega_est, gm, pe_state, params);
    pose_ref = [traj.x_ref(k); traj.y_ref(k); traj.theta_ref(k)];
    vel_ref = [traj.vx_ref(k); traj.vy_ref(k); traj.wz_ref(k)];
    vel_cmd = position_controller(pose_ref, pose_est, vel_ref, params);

    omega_ref = (1/r)*[vel_cmd(1)-vel_cmd(2)-L*vel_cmd(3);
                       vel_cmd(1)+vel_cmd(2)+L*vel_cmd(3);
                       vel_cmd(1)+vel_cmd(2)-L*vel_cmd(3);
                       vel_cmd(1)-vel_cmd(2)+L*vel_cmd(3)];

    [tau_cmd, pid_state] = pid_controller(omega_ref, omega_est, pid_state, params);
    pwm_signal = pwm_output(tau_cmd, params);
    tau = pwm_capture(pwm_signal, params);
    [tau_up, ~] = spi_interface('uplink', tau, params);
    x_cur = sm.x;
    x_new = plant_step(x_cur, tau_up, params, dt);
    [as, gs, imu_state] = imu_model(x_new, x_cur, dt, imu_state, params);
    sm = state_manager('update', sm, x_new);
    [~, sh] = spi_interface('downlink', x_new, params);
    enc_counts = encoder_pulse_gen(sh(7:10), dt, params);
    imu_packet = imu_packet_enc(as, gs, params);

    pos_err(k) = sqrt((x_new(1)-traj.x_ref(k))^2 + (x_new(2)-traj.y_ref(k))^2);
end

rms_ss = rms(pos_err(round(N_test/2):end)) * 1000;
if rms_ss < 100  % should be well under 100mm for line
    fprintf('PASS (SS RMS=%.1f mm)\n', rms_ss); pass_count = pass_count + 1;
else
    fprintf('FAIL (SS RMS=%.1f mm, expected <100mm)\n', rms_ss); fail_count = fail_count + 1;
end

%% ===== TEST 7: Line trajectory — ADRC convergence =====
fprintf('Test 7: Line trajectory, ADRC, 5s — convergence ... ');
clear encoder_pulse_gen encoder_reader imu_reader position_controller;
traj = trajectory_generator('line', T_test, dt, params);

x0 = zeros(10,1);
x0(1) = traj.x_ref(1); x0(2) = traj.y_ref(1); x0(3) = traj.theta_ref(1);
sm = state_manager('init', [], x0, params);
adrc_state = []; imu_state = []; pe_state = [];
pe_state.x = x0(1); pe_state.y = x0(2); pe_state.theta = x0(3);

enc_counts = encoder_pulse_gen(x0(7:10), dt, params);
[a_i, g_i, imu_state] = imu_model(x0, x0, dt, imu_state, params);
imu_packet = imu_packet_enc(a_i, g_i, params);

pos_err = zeros(1, N_test);
for k = 1:N_test
    omega_est = encoder_reader(enc_counts, dt, params);
    [am, gm, ~] = imu_reader(imu_packet, params);
    imu_data.accel = am; imu_data.gyro = gm;

    [pose_est, pe_state] = pose_estimator(omega_est, gm, pe_state, params);
    pose_ref = [traj.x_ref(k); traj.y_ref(k); traj.theta_ref(k)];
    vel_ref = [traj.vx_ref(k); traj.vy_ref(k); traj.wz_ref(k)];
    vel_cmd = position_controller(pose_ref, pose_est, vel_ref, params);

    omega_ref = (1/r)*[vel_cmd(1)-vel_cmd(2)-L*vel_cmd(3);
                       vel_cmd(1)+vel_cmd(2)+L*vel_cmd(3);
                       vel_cmd(1)+vel_cmd(2)-L*vel_cmd(3);
                       vel_cmd(1)-vel_cmd(2)+L*vel_cmd(3)];

    [tau_cmd, adrc_state] = adrc_controller(omega_ref, omega_est, imu_data, adrc_state, params);
    pwm_signal = pwm_output(tau_cmd, params);
    tau = pwm_capture(pwm_signal, params);
    [tau_up, ~] = spi_interface('uplink', tau, params);
    x_cur = sm.x;
    x_new = plant_step(x_cur, tau_up, params, dt);
    [as, gs, imu_state] = imu_model(x_new, x_cur, dt, imu_state, params);
    sm = state_manager('update', sm, x_new);
    [~, sh] = spi_interface('downlink', x_new, params);
    enc_counts = encoder_pulse_gen(sh(7:10), dt, params);
    imu_packet = imu_packet_enc(as, gs, params);

    pos_err(k) = sqrt((x_new(1)-traj.x_ref(k))^2 + (x_new(2)-traj.y_ref(k))^2);
end

rms_ss = rms(pos_err(round(N_test/2):end)) * 1000;
if rms_ss < 100
    fprintf('PASS (SS RMS=%.1f mm)\n', rms_ss); pass_count = pass_count + 1;
else
    fprintf('FAIL (SS RMS=%.1f mm, expected <100mm)\n', rms_ss); fail_count = fail_count + 1;
end

%% ===== TEST 8: No NaN/Inf across all trajectories =====
fprintf('Test 8: All trajectories produce valid data ... ');
all_valid = true;
for ti = 1:length({'line','circle','square','figure8'})
    traj_types = {'line','circle','square','figure8'};
    clear encoder_pulse_gen encoder_reader imu_reader position_controller;
    traj = trajectory_generator(traj_types{ti}, 3, dt, params);  % 3s short run
    N3 = round(3/dt);

    x0 = zeros(10,1);
    x0(1) = traj.x_ref(1); x0(2) = traj.y_ref(1); x0(3) = traj.theta_ref(1);
    sm = state_manager('init', [], x0, params);
    adrc_s = []; imu_s = []; pe_s = [];
    pe_s.x = x0(1); pe_s.y = x0(2); pe_s.theta = x0(3);

    enc_c = encoder_pulse_gen(x0(7:10), dt, params);
    [ai, gi, imu_s] = imu_model(x0, x0, dt, imu_s, params);
    imu_p = imu_packet_enc(ai, gi, params);

    for k = 1:N3
        oe = encoder_reader(enc_c, dt, params);
        [am2, gm2, ~] = imu_reader(imu_p, params);
        id.accel = am2; id.gyro = gm2;
        [pe2, pe_s] = pose_estimator(oe, gm2, pe_s, params);
        pr = [traj.x_ref(k); traj.y_ref(k); traj.theta_ref(k)];
        vr = [traj.vx_ref(k); traj.vy_ref(k); traj.wz_ref(k)];
        vc = position_controller(pr, pe2, vr, params);
        or2 = (1/r)*[vc(1)-vc(2)-L*vc(3); vc(1)+vc(2)+L*vc(3);
                     vc(1)+vc(2)-L*vc(3); vc(1)-vc(2)+L*vc(3)];
        [tc, adrc_s] = adrc_controller(or2, oe, id, adrc_s, params);
        ps2 = pwm_output(tc, params);
        t2 = pwm_capture(ps2, params);
        [tu, ~] = spi_interface('uplink', t2, params);
        xc = sm.x;
        xn = plant_step(xc, tu, params, dt);
        [as2, gs2, imu_s] = imu_model(xn, xc, dt, imu_s, params);
        sm = state_manager('update', sm, xn);
        [~, sh2] = spi_interface('downlink', xn, params);
        enc_c = encoder_pulse_gen(sh2(7:10), dt, params);
        imu_p = imu_packet_enc(as2, gs2, params);

        if any(isnan(xn)) || any(isinf(xn))
            fprintf('FAIL (NaN/Inf in %s at step %d)\n', traj_types{ti}, k);
            all_valid = false;
            break;
        end
    end
end
if all_valid
    fprintf('PASS (4 trajectories, no NaN/Inf)\n'); pass_count = pass_count + 1;
else
    fail_count = fail_count + 1;
end

%% ===== SUMMARY =====
fprintf('\n===== M5 Test Results: %d/%d PASS =====\n', pass_count, pass_count + fail_count);
if fail_count == 0
    fprintf('All tests passed.\n');
else
    fprintf('%d test(s) FAILED.\n', fail_count);
end
