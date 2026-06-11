function validate_integrator()
% VALIDATE_INTEGRATOR  Verify integration accuracy and convergence.
%
% Tầng 3 validation — numerical integration.
% Tests:
%   3a. Convergence order: error vs dt on log-log, verify slope ≈ 1 (first-order)
%   3b. Circular motion: constant (vx, wz) → radius R = vx/wz
%   3c. Return-to-origin: forward T + backward T → measure drift
%   3d. Heading integration: constant wz → theta grows linearly
%
% CRITICAL: All torques must satisfy tau/b_w < omega_max (34.56 rad/s)
% to avoid speed clamping. Max linear torque = 34.56 * 0.002 = 0.069 N·m.
%
% The plant uses semi-implicit Euler with midpoint rotation for pose.
% Expected convergence order: 1 for velocities, 1-2 for pose (midpoint helps).
%
% Usage: validate_integrator()
%   Requires: params_mecanum.m, plant_step.m on path

    fprintf('=== VALIDATE INTEGRATOR (Tầng 3) ===\n\n');

    params = params_mecanum();
    params.slip.enabled = false;

    r  = params.r;
    L  = params.lx + params.ly;
    b_w = params.b_w;
    tau_max_linear = params.omega_max * b_w;  % 0.0691 N·m

    fprintf('  Linear torque limit: tau < %.4f N·m (omega_max * b_w)\n\n', tau_max_linear);

    n_pass = 0;
    n_fail = 0;

    %% --- Test 3a: Convergence order ---
    % Run same scenario at multiple dt values, compare against fine reference.
    % Use short duration (transient phase) so differences are visible.
    % Collect FULL trajectory and compare RMS, not just endpoint.

    fprintf('--- Test 3a: Convergence order (Richardson) ---\n');

    dt_values = [0.004, 0.002, 0.001, 0.0005];
    dt_ref = 0.00005;  % very fine reference (80× finer than coarsest test)
    T_conv = 0.5;      % short — stay in transient where dt matters

    % Mixed mode torque, within linear range
    tau_conv = [0.03; 0.05; 0.03; 0.05];  % max component 0.05 < 0.069

    % Reference solution: collect trajectory at fine dt
    N_ref = round(T_conv / dt_ref);
    x_ref_traj = zeros(10, N_ref+1);
    x = zeros(10, 1);
    x_ref_traj(:, 1) = x;
    for k = 1:N_ref
        x = plant_step(x, tau_conv, params, dt_ref);
        x_ref_traj(:, k+1) = x;
    end
    t_ref = (0:N_ref) * dt_ref;

    % Solutions at each dt — compare at matching time points
    errors_omega = zeros(size(dt_values));
    errors_pos   = zeros(size(dt_values));
    errors_theta = zeros(size(dt_values));

    for i = 1:length(dt_values)
        dt_i = dt_values(i);
        N_i = round(T_conv / dt_i);

        x = zeros(10, 1);
        omega_err_sq = 0;
        pos_err_sq = 0;
        theta_err_sq = 0;
        n_compare = 0;

        for k = 1:N_i
            x = plant_step(x, tau_conv, params, dt_i);
            t_now = k * dt_i;

            % Find closest reference time point
            [~, idx_ref] = min(abs(t_ref - t_now));
            x_ref_now = x_ref_traj(:, idx_ref);

            omega_err_sq = omega_err_sq + norm(x(7:10) - x_ref_now(7:10))^2;
            pos_err_sq = pos_err_sq + norm(x(1:2) - x_ref_now(1:2))^2;
            theta_err_sq = theta_err_sq + (angle_diff(x(3), x_ref_now(3)))^2;
            n_compare = n_compare + 1;
        end

        errors_omega(i) = sqrt(omega_err_sq / n_compare);
        errors_pos(i) = sqrt(pos_err_sq / n_compare);
        errors_theta(i) = sqrt(theta_err_sq / n_compare);
    end

    fprintf('  dt        | omega_err  | pos_err    | theta_err\n');
    fprintf('  ----------|------------|------------|----------\n');
    for i = 1:length(dt_values)
        fprintf('  %.5f | %.3e  | %.3e  | %.3e\n', ...
            dt_values(i), errors_omega(i), errors_pos(i), errors_theta(i));
    end

    % Fit log-log slope
    valid = errors_omega > 0;
    if sum(valid) >= 2
        p_omega = polyfit(log10(dt_values(valid)), log10(errors_omega(valid)), 1);
        order_omega = p_omega(1);
    else
        order_omega = NaN;
    end

    valid = errors_pos > 0;
    if sum(valid) >= 2
        p_pos = polyfit(log10(dt_values(valid)), log10(errors_pos(valid)), 1);
        order_pos = p_pos(1);
    else
        order_pos = NaN;
    end

    fprintf('\n  Convergence orders (log-log slope):\n');
    fprintf('    Omega: %.2f (expected ~1 for semi-implicit Euler)\n', order_omega);
    fprintf('    Position: %.2f (expected ~1-2 for midpoint rotation)\n', order_pos);

    [n_pass, n_fail] = check(~isnan(order_omega) && order_omega > 0.8 && order_omega < 2.5, ...
        sprintf('Omega convergence order = %.2f (expect 0.8-2.5)', order_omega), n_pass, n_fail);
    [n_pass, n_fail] = check(~isnan(order_pos) && order_pos > 0.8 && order_pos < 3.0, ...
        sprintf('Position convergence order = %.2f (expect 0.8-3.0)', order_pos), n_pass, n_fail);

    % Verify dt=0.001 has acceptable error
    idx_001 = find(abs(dt_values - 0.001) < 1e-6);
    if ~isempty(idx_001)
        [n_pass, n_fail] = check(errors_pos(idx_001) < 0.001, ...
            sprintf('At project dt=0.001: pos_err=%.3e m (< 1mm vs ref)', ...
            errors_pos(idx_001)), n_pass, n_fail);
    end

    fprintf('\n');

    %% --- Test 3b: Circular motion geometry ---
    % Apply torques that produce constant vx and wz at steady state.
    % Let settle to SS, then measure one revolution.
    % All torques within linear range.

    fprintf('--- Test 3b: Circular motion geometry ---\n');

    dt = params.dt;

    % Torque decomposition: forward + rotation
    tau_fwd_val = 0.02;   % omega_ss = 10 rad/s, well within limit
    tau_rot_val = 0.005;  % omega_ss = 2.5 rad/s

    tau_circle = tau_fwd_val * [1;1;1;1] + tau_rot_val * [-1;1;-1;1];

    % Expected SS
    omega_fwd_ss = tau_fwd_val / b_w;
    omega_rot_ss = tau_rot_val / b_w;
    vx_ss = r * omega_fwd_ss;
    wz_ss = (r / L) * omega_rot_ss;
    R_expected = vx_ss / wz_ss;

    fprintf('  Expected: vx_ss=%.4f m/s, wz_ss=%.4f rad/s, R=%.4f m\n', ...
        vx_ss, wz_ss, R_expected);

    % Settle phase (10 time constants)
    J_fwd = (r^2 * params.M / 4) + params.J_w;
    tau_c_max = J_fwd / b_w;
    T_settle = 10 * tau_c_max;
    T_revolution = 2 * pi / wz_ss;
    N_settle = round(T_settle / dt);
    N_rev = round(T_revolution / dt);

    fprintf('  tau_c=%.2fs, T_settle=%.1fs, T_revolution=%.1fs\n', ...
        tau_c_max, T_settle, T_revolution);

    x = zeros(10, 1);

    % Settle
    for k = 1:N_settle
        x = plant_step(x, tau_circle, params, dt);
    end

    fprintf('  At SS: vx=%.4f (exp %.4f), wz=%.4f (exp %.4f)\n', ...
        x(4), vx_ss, x(6), wz_ss);

    % Record one revolution
    traj = zeros(2, N_rev+1);
    traj(:, 1) = x(1:2);
    for k = 1:N_rev
        x = plant_step(x, tau_circle, params, dt);
        traj(:, k+1) = x(1:2);
    end

    % Compute radius: distance from center of arc
    center_x = mean(traj(1,:));
    center_y = mean(traj(2,:));
    radii = sqrt((traj(1,:) - center_x).^2 + (traj(2,:) - center_y).^2);
    R_mean = mean(radii);
    R_std = std(radii);
    R_err_pct = abs(R_mean - R_expected) / R_expected * 100;

    [n_pass, n_fail] = check(R_err_pct < 2.0, ...
        sprintf('Circle radius: R=%.4f m (exp %.4f), err=%.2f%%', ...
        R_mean, R_expected, R_err_pct), n_pass, n_fail);

    [n_pass, n_fail] = check(R_std / R_mean < 0.05, ...
        sprintf('Circle roundness: R_std/R_mean = %.2f%% (< 5%%)', ...
        R_std/R_mean*100), n_pass, n_fail);

    fprintf('\n');

    %% --- Test 3c: Return-to-origin ---
    % With friction b_w > 0, deceleration is faster than acceleration
    % (friction assists), creating REAL physical asymmetry. To isolate
    % integrator error, test with b_w = 0 (frictionless).
    %
    % Profile: tau for T, -tau for 2T, tau for T
    % Omega: 0→peak→0→-peak→0 (symmetric triangle wave)
    % Position: net displacement should be exactly 0 by symmetry.
    % Any drift = integrator error.

    fprintf('--- Test 3c: Return-to-origin (frictionless, symmetric profile) ---\n');

    params_nf = params;
    params_nf.b_w = 0;  % frictionless — isolate integrator error
    % Recompute M_eff_inv is NOT needed — b_w is used directly in plant_step

    tau_go = 0.03 * [1;1;1;1];
    T_phase = 0.5;  % short to keep omega well below omega_max
    N_phase = round(T_phase / dt);

    x = zeros(10, 1);

    % Phase 1: +tau for T (accelerate forward)
    for k = 1:N_phase
        x = plant_step(x, tau_go, params_nf, dt);
    end
    omega_peak = x(7);  % should be tau*T/J_eff
    pos_phase1 = x(1);
    fprintf('  After phase 1 (+tau, %.1fs): omega=%.4f, x=%.6f m\n', T_phase, omega_peak, pos_phase1);

    % Phase 2: -tau for 2T (decelerate, reverse, decelerate)
    for k = 1:(2*N_phase)
        x = plant_step(x, -tau_go, params_nf, dt);
    end
    pos_phase2 = x(1);
    fprintf('  After phase 2 (-tau, %.1fs): omega=%.4f, x=%.6f m\n', 2*T_phase, x(7), pos_phase2);

    % Phase 3: +tau for T (decelerate back to zero)
    for k = 1:N_phase
        x = plant_step(x, tau_go, params_nf, dt);
    end
    pos_final = x(1:2);
    omega_final = x(7);
    fprintf('  After phase 3 (+tau, %.1fs): omega=%.6f, x=%.6f m, y=%.6f m\n', ...
        T_phase, omega_final, pos_final(1), pos_final(2));

    drift = norm(pos_final);
    travel = abs(pos_phase1) + abs(pos_phase2 - pos_phase1) + abs(pos_final(1) - pos_phase2);
    fprintf('  Drift from origin: %.6f m (%.4f mm)\n', drift, drift*1000);
    fprintf('  Omega return error: %.2e rad/s (should be ~0)\n', abs(omega_final));

    [n_pass, n_fail] = check(drift < 0.001, ...
        sprintf('Position drift = %.4f mm (< 1mm)', drift*1000), n_pass, n_fail);
    [n_pass, n_fail] = check(abs(omega_final) < 0.01, ...
        sprintf('Omega returns to zero: |omega| = %.2e rad/s', abs(omega_final)), n_pass, n_fail);

    fprintf('\n');

    %% --- Test 3d: Heading integration accuracy ---
    % Reach SS rotation, then check heading grows linearly.

    fprintf('--- Test 3d: Heading integration (constant wz) ---\n');

    tau_rot = 0.01 * [-1;1;-1;1];  % omega_ss = 5 rad/s, wz_ss small
    J_rot = (r^2 * params.Iz / (4 * L^2)) + params.J_w;
    T_settle_rot = 10 * J_rot / b_w;

    x = zeros(10, 1);

    % Settle
    for k = 1:round(T_settle_rot / dt)
        x = plant_step(x, tau_rot, params, dt);
    end

    wz_settled = x(6);
    wz_expected = (r / L) * (0.01 / b_w);
    fprintf('  wz at SS: %.6f rad/s (expected %.6f)\n', wz_settled, wz_expected);

    % Measure heading for 5s
    T_measure = 5.0;
    N_measure = round(T_measure / dt);

    theta_values = zeros(1, N_measure+1);
    theta_values(1) = x(3);

    for k = 1:N_measure
        x = plant_step(x, tau_rot, params, dt);
        theta_values(k+1) = x(3);
    end

    % Unwrap theta
    theta_uw = unwrap_theta(theta_values);
    theta_uw = theta_uw - theta_uw(1);  % relative to start
    t_meas = (0:N_measure) * dt;

    % Fit linear: theta = slope * t + offset
    p_fit = polyfit(t_meas, theta_uw, 1);
    slope_sim = p_fit(1);
    slope_err_pct = abs(slope_sim - wz_settled) / abs(wz_settled) * 100;

    % RMS residual from linear fit (measures nonlinearity)
    theta_linear_fit = polyval(p_fit, t_meas);
    rms_nonlinearity = sqrt(mean((theta_uw - theta_linear_fit).^2));

    fprintf('  Heading slope: %.6f rad/s (expected %.6f)\n', slope_sim, wz_settled);
    fprintf('  Slope error: %.4f%%\n', slope_err_pct);
    fprintf('  RMS nonlinearity: %.4e rad\n', rms_nonlinearity);

    [n_pass, n_fail] = check(slope_err_pct < 0.1, ...
        sprintf('Heading slope matches wz_ss: err=%.4f%%', slope_err_pct), n_pass, n_fail);

    [n_pass, n_fail] = check(rms_nonlinearity < 0.01, ...
        sprintf('Heading linearity: RMS residual=%.4e rad (< 0.01)', rms_nonlinearity), ...
        n_pass, n_fail);

    fprintf('\n');

    %% --- Summary ---
    fprintf('=== INTEGRATOR VALIDATION SUMMARY ===\n');
    fprintf('Passed: %d / %d\n', n_pass, n_pass + n_fail);
    if n_fail == 0
        fprintf('>>> ALL TESTS PASSED <<<\n');
    else
        fprintf('>>> %d TESTS FAILED <<<\n', n_fail);
    end
end


%% --- Helper functions ---

function d = angle_diff(a, b)
    d = mod(a - b + pi, 2*pi) - pi;
end

function theta_uw = unwrap_theta(theta)
    theta_uw = zeros(size(theta));
    theta_uw(1) = theta(1);
    for i = 2:length(theta)
        d = theta(i) - theta(i-1);
        if d > pi
            d = d - 2*pi;
        elseif d < -pi
            d = d + 2*pi;
        end
        theta_uw(i) = theta_uw(i-1) + d;
    end
end

function [n_pass, n_fail] = check(condition, msg, n_pass, n_fail)
    if condition
        fprintf('  [PASS] %s\n', msg);
        n_pass = n_pass + 1;
    else
        fprintf('  [FAIL] %s\n', msg);
        n_fail = n_fail + 1;
    end
end
