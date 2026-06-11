%% DIAGNOSE_REMAINING_ERROR  Identify plant vs controller limits for figure-8 and line.
%
% Tests:
% 1. Plant capability check: can the plant physically follow figure-8?
%    Run with "perfect controller" = direct feedforward torque from inverse dynamics.
%
% 2. ADRC noise sensitivity: compare ADRC with clean vs noisy sensors on line.
%
% 3. Saturation analysis: when and how much does saturation occur on figure-8?
%
% 4. Figure-8 with ideal pose (ground truth) to isolate odometry vs controller.

clear; clc; close all;

params = params_mecanum();
dt = params.dt;
T_sim = params.T_sim;
N = round(T_sim / dt);
r = params.r;
L = params.lx + params.ly;

%% ===== TEST 1: Plant Physical Capability on Figure-8 =====
fprintf('==============================================\n');
fprintf('  TEST 1: Plant Capability (Inverse Dynamics)\n');
fprintf('==============================================\n');
% Perfect controller: compute required torque from trajectory analytically,
% apply directly. If error > 0, it is pure plant limitation.

traj = trajectory_generator('figure8', T_sim, dt, params);

% Compute required wheel speeds from trajectory
omega_req = zeros(4, length(traj.t));
for i = 1:length(traj.t)
    omega_req(:,i) = (1/r) * [traj.vx_ref(i) - traj.vy_ref(i) - L*traj.wz_ref(i);
                               traj.vx_ref(i) + traj.vy_ref(i) + L*traj.wz_ref(i);
                               traj.vx_ref(i) + traj.vy_ref(i) - L*traj.wz_ref(i);
                               traj.vx_ref(i) - traj.vy_ref(i) + L*traj.wz_ref(i)];
end

% Required acceleration (numerical)
domega_req = diff(omega_req, 1, 2) / dt;
domega_req = [domega_req, domega_req(:,end)];  % pad

% Required torque from inverse dynamics: tau = M_eff * domega + b_w * omega
tau_req = zeros(4, length(traj.t));
for i = 1:length(traj.t)
    tau_req(:,i) = params.M_eff * domega_req(:,i) + params.b_w * omega_req(:,i);
end

% Check: what fraction of required torque exceeds tau_max?
tau_max = params.tau_max;
exceed_count = sum(abs(tau_req(:)) > tau_max);
total_count = numel(tau_req);
exceed_pct = exceed_count / total_count * 100;
max_tau_needed = max(abs(tau_req(:)));

fprintf('Figure-8 inverse dynamics:\n');
fprintf('  Max torque required: %.3f N·m (τ_max = %.1f N·m)\n', max_tau_needed, tau_max);
fprintf('  Torque exceeds τ_max: %d / %d samples (%.1f%%)\n', exceed_count, total_count, exceed_pct);
fprintf('  Max wheel speed required: %.1f rad/s (ω_max = %.1f rad/s)\n', ...
    max(abs(omega_req(:))), params.omega_max);

% Now run plant with perfect feedforward (clamped to tau_max)
x0 = zeros(10, 1);
x0(1) = traj.x_ref(1); x0(2) = traj.y_ref(1); x0(3) = traj.theta_ref(1);
sm = state_manager('init', [], x0, params);

pos_err_ff = zeros(1, N);
sat_ff = zeros(1, N);
for k = 1:N
    tau_ff = tau_req(:, k);
    tau_clamped = max(-tau_max, min(tau_max, tau_ff));
    sat_ff(k) = any(abs(tau_ff) > tau_max * 0.99);

    x_cur = sm.x;
    x_new = plant_step(x_cur, tau_clamped, params, dt);
    sm = state_manager('update', sm, x_new);

    pos_err_ff(k) = sqrt((x_new(1)-traj.x_ref(k))^2 + (x_new(2)-traj.y_ref(k))^2);
end

ss = round(N/2);
fprintf('\n  Feedforward-only tracking:\n');
fprintf('    RMS (full): %.1f mm\n', rms(pos_err_ff)*1000);
fprintf('    RMS (SS):   %.1f mm\n', rms(pos_err_ff(ss:end))*1000);
fprintf('    Max:        %.1f mm\n', max(pos_err_ff)*1000);
fprintf('    Saturation: %.1f%%\n\n', mean(sat_ff)*100);

if max_tau_needed <= tau_max * 1.05
    fprintf('  → Plant CAN follow figure-8 without saturation.\n');
    fprintf('  → Remaining error is CONTROLLER limitation.\n');
else
    fprintf('  → Plant CANNOT fully follow figure-8 (torque limited).\n');
    fprintf('  → %.1f%% of trajectory requires more torque than available.\n', exceed_pct);
    fprintf('  → Feedforward SS error = %.1f mm is the PLANT FLOOR.\n', rms(pos_err_ff(ss:end))*1000);
end

%% ===== TEST 2: ADRC Noise Sensitivity on Line =====
fprintf('\n==============================================\n');
fprintf('  TEST 2: ADRC Noise Sensitivity (Line)\n');
fprintf('==============================================\n');
% Compare ADRC with ideal sensors vs real sensors on line trajectory.
% If ideal ADRC >> ideal PID → ADRC tuning issue.
% If ideal ADRC ≈ ideal PID but real ADRC >> real PID → noise amplification.

conditions = {'ideal', 'real'};
controllers = {'pid', 'adrc'};

for ci = 1:2
    for si = 1:2
        ctrl = controllers{ci};
        sensor = conditions{si};

        clear encoder_pulse_gen encoder_reader imu_reader position_controller;

        traj = trajectory_generator('line', T_sim, dt, params);
        x0 = zeros(10,1);
        x0(1) = traj.x_ref(1); x0(2) = traj.y_ref(1); x0(3) = traj.theta_ref(1);
        sm = state_manager('init', [], x0, params);

        pid_s = []; adrc_s = []; imu_s = [];
        pe_s = []; pe_s.x = x0(1); pe_s.y = x0(2); pe_s.theta = x0(3);

        enc = encoder_pulse_gen(x0(7:10), dt, params);
        [ai, gi, imu_s] = imu_model(x0, x0, dt, imu_s, params);
        pkt = imu_packet_enc(ai, gi, params);

        pos_err = zeros(1, N);
        for k = 1:N
            x_cur = sm.x;

            if strcmp(sensor, 'ideal')
                omega_est = x_cur(7:10);
                gyro_meas = [0; 0; x_cur(6)];
                pose_est = [x_cur(1); x_cur(2); x_cur(3)];
            else
                omega_est = encoder_reader(enc, dt, params);
                [am, gm, ~] = imu_reader(pkt, params);
                gyro_meas = gm;
                [pose_est, pe_s] = pose_estimator(omega_est, gm, pe_s, params);
            end

            pr = [traj.x_ref(k); traj.y_ref(k); traj.theta_ref(k)];
            vr = [traj.vx_ref(k); traj.vy_ref(k); traj.wz_ref(k)];
            vc = position_controller(pr, pose_est, vr, params);
            or_ = (1/r)*[vc(1)-vc(2)-L*vc(3); vc(1)+vc(2)+L*vc(3);
                         vc(1)+vc(2)-L*vc(3); vc(1)-vc(2)+L*vc(3)];

            id.accel = zeros(3,1); id.gyro = gyro_meas;
            switch ctrl
                case 'pid'
                    [tc, pid_s] = pid_controller(or_, omega_est, pid_s, params);
                case 'adrc'
                    [tc, adrc_s] = adrc_controller(or_, omega_est, id, adrc_s, params);
            end

            ps = pwm_output(tc, params);
            t_ = pwm_capture(ps, params);
            [tu, ~] = spi_interface('uplink', t_, params);
            x_new = plant_step(x_cur, tu, params, dt);
            [as, gs, imu_s] = imu_model(x_new, x_cur, dt, imu_s, params);
            sm = state_manager('update', sm, x_new);

            if strcmp(sensor, 'real')
                [~, sh] = spi_interface('downlink', x_new, params);
                enc = encoder_pulse_gen(sh(7:10), dt, params);
                pkt = imu_packet_enc(as, gs, params);
            end

            pos_err(k) = sqrt((x_new(1)-traj.x_ref(k))^2 + (x_new(2)-traj.y_ref(k))^2);
        end

        rms_ss = rms(pos_err(ss:end)) * 1000;
        fprintf('  %s / %s sensors: SS RMS = %.2f mm\n', upper(ctrl), sensor, rms_ss);
    end
end

fprintf('\n  If ADRC ideal >> PID ideal → ADRC structure/tuning problem\n');
fprintf('  If ADRC ideal ≈ PID ideal but ADRC real >> PID real → noise amplification\n');

%% ===== TEST 3: Figure-8 Saturation Timeline =====
fprintf('\n==============================================\n');
fprintf('  TEST 3: Figure-8 Saturation Analysis\n');
fprintf('==============================================\n');

for ci = 1:2
    ctrl = controllers{ci};
    clear encoder_pulse_gen encoder_reader imu_reader position_controller;

    traj = trajectory_generator('figure8', T_sim, dt, params);
    x0 = zeros(10,1);
    x0(1) = traj.x_ref(1); x0(2) = traj.y_ref(1); x0(3) = traj.theta_ref(1);
    sm = state_manager('init', [], x0, params);

    pid_s = []; adrc_s = []; imu_s = [];
    pe_s = []; pe_s.x = x0(1); pe_s.y = x0(2); pe_s.theta = x0(3);

    enc = encoder_pulse_gen(x0(7:10), dt, params);
    [ai, gi, imu_s] = imu_model(x0, x0, dt, imu_s, params);
    pkt = imu_packet_enc(ai, gi, params);

    tau_log = zeros(4, N);
    sat_log = false(4, N);
    heading_err = zeros(1, N);

    for k = 1:N
        oe = encoder_reader(enc, dt, params);
        [am, gm, ~] = imu_reader(pkt, params);
        id.accel = am; id.gyro = gm;
        [pe, pe_s] = pose_estimator(oe, gm, pe_s, params);
        pr = [traj.x_ref(k); traj.y_ref(k); traj.theta_ref(k)];
        vr = [traj.vx_ref(k); traj.vy_ref(k); traj.wz_ref(k)];
        vc = position_controller(pr, pe, vr, params);
        or_ = (1/r)*[vc(1)-vc(2)-L*vc(3); vc(1)+vc(2)+L*vc(3);
                     vc(1)+vc(2)-L*vc(3); vc(1)-vc(2)+L*vc(3)];

        switch ctrl
            case 'pid'
                [tc, pid_s] = pid_controller(or_, oe, pid_s, params);
            case 'adrc'
                [tc, adrc_s] = adrc_controller(or_, oe, id, adrc_s, params);
        end

        tau_log(:,k) = tc;
        sat_log(:,k) = abs(tc) >= tau_max * 0.99;

        ps = pwm_output(tc, params);
        t_ = pwm_capture(ps, params);
        [tu, ~] = spi_interface('uplink', t_, params);
        xc = sm.x;
        xn = plant_step(xc, tu, params, dt);
        [as, gs, imu_s] = imu_model(xn, xc, dt, imu_s, params);
        sm = state_manager('update', sm, xn);
        [~, sh] = spi_interface('downlink', xn, params);
        enc = encoder_pulse_gen(sh(7:10), dt, params);
        pkt = imu_packet_enc(as, gs, params);

        e_th = xn(3) - traj.theta_ref(k);
        heading_err(k) = mod(e_th + pi, 2*pi) - pi;
    end

    % Analysis
    sat_pct_per_wheel = mean(sat_log, 2) * 100;
    sat_pct_any = mean(any(sat_log, 1)) * 100;

    % Find saturation bursts
    sat_any = any(sat_log, 1);
    t_vec = (1:N) * dt;

    fprintf('\n%s figure-8:\n', upper(ctrl));
    fprintf('  Overall saturation: %.1f%% of timesteps\n', sat_pct_any);
    fprintf('  Per wheel: [%.1f%%, %.1f%%, %.1f%%, %.1f%%]\n', sat_pct_per_wheel);
    fprintf('  Max |τ_cmd|: %.2f N·m\n', max(abs(tau_log(:))));
    fprintf('  Heading error RMS: %.2f deg\n', rms(heading_err)*180/pi);

    % When does saturation occur?
    fprintf('  Saturation timeline (% in each 1s window):\n    ');
    for s = 0:9
        idx = (s*1000+1):min((s+1)*1000, N);
        fprintf('%.0fs:%.0f%% ', s, mean(sat_any(idx))*100);
    end
    fprintf('\n');
end

%% ===== SUMMARY =====
fprintf('\n==============================================\n');
fprintf('  DIAGNOSIS SUMMARY\n');
fprintf('==============================================\n');
fprintf('1. Plant can/cannot follow figure-8? → Check TEST 1 above\n');
fprintf('2. ADRC worse on line due to noise? → Check TEST 2 above\n');
fprintf('3. Figure-8 error from saturation? → Check TEST 3 above\n');
fprintf('\nIf plant floor (feedforward SS) ≈ 0:\n');
fprintf('  → All error is controller. Tune more or accept as limit.\n');
fprintf('If plant floor > 0 and close to controller SS:\n');
fprintf('  → Plant is the bottleneck. Cannot improve with any controller.\n');
fprintf('If ADRC ideal > PID ideal:\n');
fprintf('  → ADRC inner loop tuning needs work (kp or ESO gains).\n');
