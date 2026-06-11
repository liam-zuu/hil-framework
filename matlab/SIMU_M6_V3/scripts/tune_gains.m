%% TUNE_GAINS  Systematic gain sweep to optimize controller baseline.
%
% Sweeps outer loop gains (Kp_pos, Ki_pos) and inner PID gain (Kp)
% on circle trajectory (hardest case). Finds combination that minimizes
% steady-state error while maintaining stability.
%
% Stability checks: no NaN/Inf, saturation < 30%, error decreasing trend.

clear; clc; close all;

params_base = params_mecanum();
dt = params_base.dt;
T_sim = params_base.T_sim;
N = round(T_sim / dt);
r = params_base.r;
L = params_base.lx + params_base.ly;

%% ===== SWEEP 1: Outer Loop Gains (with ADRC inner, fixed) =====
fprintf('==============================================\n');
fprintf('  SWEEP 1: Outer Loop Gains (ADRC inner)\n');
fprintf('==============================================\n');
fprintf('%-8s | %-8s | %-8s | %-8s | %10s | %10s | %6s | %s\n', ...
    'Kp_pos', 'Ki_pos', 'Kp_th', 'Ki_th', 'SS_RMS(mm)', 'Max(mm)', 'Sat%', 'Status');
fprintf('--------+----------+----------+----------+------------+------------+--------+-------\n');

Kp_pos_vals = [3.0, 4.0, 5.0, 6.0, 8.0];
Ki_pos_vals = [0.5, 1.0, 1.5, 2.0, 3.0];

best_ss = inf;
best_outer = [];

for kp_i = 1:length(Kp_pos_vals)
    for ki_i = 1:length(Ki_pos_vals)
        Kp_pos = Kp_pos_vals(kp_i);
        Ki_pos = Ki_pos_vals(ki_i);
        Kp_theta = Kp_pos * 4/3;  % scale heading proportionally
        Ki_theta = Ki_pos * 2;     % heading integral faster

        params = params_base;
        params.pos_ctrl.Kp_pos = Kp_pos;
        params.pos_ctrl.Ki_pos = Ki_pos;
        params.pos_ctrl.Kp_theta = Kp_theta;
        params.pos_ctrl.Ki_theta = Ki_theta;

        [rms_ss, max_err, sat_pct, stable] = run_circle_test(params, 'adrc', dt, N, r, L);

        status = 'OK';
        if ~stable, status = 'UNSTABLE'; end

        fprintf('%-8.1f | %-8.1f | %-8.1f | %-8.1f | %10.1f | %10.1f | %5.1f | %s\n', ...
            Kp_pos, Ki_pos, Kp_theta, Ki_theta, rms_ss, max_err, sat_pct, status);

        if stable && rms_ss < best_ss
            best_ss = rms_ss;
            best_outer = [Kp_pos, Ki_pos, Kp_theta, Ki_theta];
        end
    end
end

fprintf('\nBest outer gains (ADRC): Kp_pos=%.1f, Ki_pos=%.1f, Kp_θ=%.1f, Ki_θ=%.1f → SS=%.1f mm\n', ...
    best_outer, best_ss);

%% ===== SWEEP 2: Inner PID Kp (with best outer) =====
fprintf('\n==============================================\n');
fprintf('  SWEEP 2: Inner PID Kp (with best outer)\n');
fprintf('==============================================\n');
fprintf('%-8s | %-8s | %-8s | %10s | %10s | %6s | %s\n', ...
    'PID_Kp', 'PID_Ki', 'PID_Kd', 'SS_RMS(mm)', 'Max(mm)', 'Sat%', 'Status');
fprintf('--------+----------+----------+------------+------------+--------+-------\n');

Kp_pid_vals = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08];
Ki_pid_vals = [0.5, 1.0, 1.5];

best_pid_ss = inf;
best_pid = [];

for kp_i = 1:length(Kp_pid_vals)
    for ki_i = 1:length(Ki_pid_vals)
        Kp_pid = Kp_pid_vals(kp_i);
        Ki_pid = Ki_pid_vals(ki_i);
        Kd_pid = Kp_pid * 0.01;  % Kd proportional to Kp

        params = params_base;
        params.pos_ctrl.Kp_pos = best_outer(1);
        params.pos_ctrl.Ki_pos = best_outer(2);
        params.pos_ctrl.Kp_theta = best_outer(3);
        params.pos_ctrl.Ki_theta = best_outer(4);
        params.pid.Kp = Kp_pid;
        params.pid.Ki = Ki_pid;
        params.pid.Kd = Kd_pid;

        [rms_ss, max_err, sat_pct, stable] = run_circle_test(params, 'pid', dt, N, r, L);

        status = 'OK';
        if ~stable, status = 'UNSTABLE'; end

        fprintf('%-8.3f | %-8.1f | %-8.4f | %10.1f | %10.1f | %5.1f | %s\n', ...
            Kp_pid, Ki_pid, Kd_pid, rms_ss, max_err, sat_pct, status);

        if stable && rms_ss < best_pid_ss
            best_pid_ss = rms_ss;
            best_pid = [Kp_pid, Ki_pid, Kd_pid];
        end
    end
end

fprintf('\nBest PID inner: Kp=%.3f, Ki=%.1f, Kd=%.4f → SS=%.1f mm\n', best_pid, best_pid_ss);

%% ===== SWEEP 3: ADRC kp (with best outer) =====
fprintf('\n==============================================\n');
fprintf('  SWEEP 3: ADRC kp (with best outer)\n');
fprintf('==============================================\n');
fprintf('%-8s | %-8s | %10s | %10s | %6s | %s\n', ...
    'ADRC_kp', 'ESO_wo', 'SS_RMS(mm)', 'Max(mm)', 'Sat%', 'Status');
fprintf('--------+----------+------------+------------+--------+-------\n');

kp_adrc_vals = [20, 30, 40, 50, 60];
wo_adrc_vals = [80, 100, 150, 200];

best_adrc_ss = inf;
best_adrc = [];

for kp_i = 1:length(kp_adrc_vals)
    for wo_i = 1:length(wo_adrc_vals)
        kp_a = kp_adrc_vals(kp_i);
        wo_a = wo_adrc_vals(wo_i);

        % ESO bandwidth must be > controller bandwidth
        if wo_a < 2 * kp_a, continue; end

        params = params_base;
        params.pos_ctrl.Kp_pos = best_outer(1);
        params.pos_ctrl.Ki_pos = best_outer(2);
        params.pos_ctrl.Kp_theta = best_outer(3);
        params.pos_ctrl.Ki_theta = best_outer(4);
        params.adrc.kp = kp_a;
        params.adrc.eso_beta1 = 2 * wo_a;
        params.adrc.eso_beta2 = wo_a^2;

        [rms_ss, max_err, sat_pct, stable] = run_circle_test(params, 'adrc', dt, N, r, L);

        status = 'OK';
        if ~stable, status = 'UNSTABLE'; end

        fprintf('%-8.0f | %-8.0f | %10.1f | %10.1f | %5.1f | %s\n', ...
            kp_a, wo_a, rms_ss, max_err, sat_pct, status);

        if stable && rms_ss < best_adrc_ss
            best_adrc_ss = rms_ss;
            best_adrc = [kp_a, wo_a];
        end
    end
end

fprintf('\nBest ADRC inner: kp=%.0f, ω_o=%.0f → SS=%.1f mm\n', best_adrc, best_adrc_ss);

%% ===== FINAL SUMMARY =====
fprintf('\n==============================================\n');
fprintf('  RECOMMENDED GAINS\n');
fprintf('==============================================\n');
fprintf('Outer loop:\n');
fprintf('  Kp_pos = %.1f, Ki_pos = %.1f\n', best_outer(1), best_outer(2));
fprintf('  Kp_theta = %.1f, Ki_theta = %.1f\n', best_outer(3), best_outer(4));
fprintf('\nPID inner loop:\n');
fprintf('  Kp = %.3f, Ki = %.1f, Kd = %.4f\n', best_pid);
fprintf('  → Circle SS RMS = %.1f mm\n', best_pid_ss);
fprintf('\nADRC inner loop:\n');
fprintf('  kp = %.0f, ω_o = %.0f (β1=%.0f, β2=%.0f)\n', ...
    best_adrc(1), best_adrc(2), 2*best_adrc(2), best_adrc(2)^2);
fprintf('  → Circle SS RMS = %.1f mm\n', best_adrc_ss);


%% ===== HELPER FUNCTION =====
function [rms_ss, max_err, sat_pct, stable] = run_circle_test(params, ctrl_type, dt, N, r, L)
    clear encoder_pulse_gen encoder_reader imu_reader position_controller;

    traj = trajectory_generator('circle', params.T_sim, dt, params);

    x0 = zeros(params.n_states, 1);
    x0(1) = traj.x_ref(1); x0(2) = traj.y_ref(1); x0(3) = traj.theta_ref(1);
    sm = state_manager('init', [], x0, params);

    pid_s = []; adrc_s = []; imu_s = [];
    pe_s = []; pe_s.x = x0(1); pe_s.y = x0(2); pe_s.theta = x0(3);

    enc = encoder_pulse_gen(x0(7:10), dt, params);
    [ai, gi, imu_s] = imu_model(x0, x0, dt, imu_s, params);
    pkt = imu_packet_enc(ai, gi, params);

    pos_err = zeros(1, N);
    sat_count = 0;

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

        switch ctrl_type
            case 'pid'
                [tc, pid_s] = pid_controller(or_, oe, pid_s, params);
            case 'adrc'
                [tc, adrc_s] = adrc_controller(or_, oe, id, adrc_s, params);
        end

        sat_count = sat_count + sum(abs(tc) >= params.tau_max * 0.99);
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

        pos_err(k) = sqrt((xn(1)-traj.x_ref(k))^2 + (xn(2)-traj.y_ref(k))^2);

        if any(isnan(xn)) || any(isinf(xn))
            rms_ss = 9999; max_err = 9999; sat_pct = 100; stable = false;
            return;
        end
    end

    ss_start = round(N/2);
    rms_ss = rms(pos_err(ss_start:end)) * 1000;
    max_err = max(pos_err) * 1000;
    sat_pct = sat_count / (4*N) * 100;

    % Stability: error should be decreasing or bounded in last 25%
    last_quarter = pos_err(round(3*N/4):end);
    stable = rms_ss < 500 && ~any(isnan(pos_err)) && ...
             max(last_quarter) < 2*mean(last_quarter);
end
