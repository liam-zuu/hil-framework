function validate_cross_ode45()
% VALIDATE_CROSS_ODE45  Cross-validate plant_step.m against MATLAB ode45.
%
% Cross-validation Tầng 1 — same equations, different solver.
% Approach:
%   1. Write dynamics as ODE function: dz/dt = f(t, z, tau, params)
%      where z = [x, y, theta, w1, w2, w3, w4] (7 independent states)
%   2. Solve with ode45 (adaptive Runge-Kutta 4/5, high accuracy)
%   3. Solve with plant_step.m (semi-implicit Euler, dt=0.001)
%   4. Compare all states at same time points
%
% If results match within O(dt) tolerance → plant_step implementation correct.
% If results diverge → bug in Euler implementation or state coupling.
%
% Tests:
%   4a. Forward motion (simple, decoupled)
%   4b. Rotation (different effective inertia)
%   4c. Mixed mode (forward + rotation, coupled dynamics active)
%   4d. Multi-phase (torque changes mid-simulation)
%
% Usage: validate_cross_ode45()
%   Requires: params_mecanum.m, plant_step.m on path

    fprintf('=== VALIDATE CROSS ODE45 (Cross-validation Tầng 1) ===\n\n');

    params = params_mecanum();
    params.slip.enabled = false;

    dt = params.dt;

    n_pass = 0;
    n_fail = 0;

    %% --- Test 4a: Forward motion ---
    fprintf('--- Test 4a: Forward motion ---\n');
    tau_fwd = 0.05 * [1;1;1;1];  % omega_ss = 25 < omega_max
    [n_pass, n_fail] = run_comparison('Forward', tau_fwd, 3.0, params, dt, n_pass, n_fail);

    %% --- Test 4b: Pure rotation ---
    fprintf('--- Test 4b: Pure rotation ---\n');
    tau_rot = 0.02 * [-1;1;-1;1];  % omega_ss = 10 < omega_max
    [n_pass, n_fail] = run_comparison('Rotation', tau_rot, 3.0, params, dt, n_pass, n_fail);

    %% --- Test 4c: Mixed mode (forward + rotation) ---
    fprintf('--- Test 4c: Mixed mode ---\n');
    tau_mix = [0.03; 0.05; 0.03; 0.05];  % max omega_ss ≈ 25 < omega_max
    [n_pass, n_fail] = run_comparison('Mixed', tau_mix, 3.0, params, dt, n_pass, n_fail);

    %% --- Test 4d: Multi-phase (torque switch at t=1.5s) ---
    fprintf('--- Test 4d: Multi-phase (torque switch) ---\n');
    T_phase = 3.0;
    N_phase = round(T_phase / dt);
    t_switch = 1.5;

    tau_phase1 = 0.05 * [1;1;1;1];           % omega_ss = 25
    tau_phase2 = 0.03 * [-1;1;1;-1];        % strafe, omega_ss = 15

    % --- ode45 solution ---
    % Use event or piecewise definition
    opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);

    % Phase 1: 0 to t_switch
    z0 = zeros(7, 1);
    [t_ode1, z_ode1] = ode45(@(t,z) dynamics_ode(t, z, tau_phase1, params), ...
                              [0, t_switch], z0, opts);
    % Phase 2: t_switch to T_phase
    [t_ode2, z_ode2] = ode45(@(t,z) dynamics_ode(t, z, tau_phase2, params), ...
                              [t_switch, T_phase], z_ode1(end,:)', opts);

    % Combine
    t_ode = [t_ode1; t_ode2(2:end)];
    z_ode = [z_ode1; z_ode2(2:end,:)];

    % Reconstruct 10-state from 7-state (add vx, vy, wz)
    x_ode_final = reconstruct_full_state(z_ode(end,:)', params);

    % --- plant_step solution ---
    x_euler = zeros(10, 1);
    N_switch = round(t_switch / dt);

    for k = 1:N_switch
        x_euler = plant_step(x_euler, tau_phase1, params, dt);
    end
    for k = N_switch+1:N_phase
        x_euler = plant_step(x_euler, tau_phase2, params, dt);
    end

    % Compare
    err_omega = norm(x_euler(7:10) - x_ode_final(7:10));
    err_pos   = norm(x_euler(1:2) - x_ode_final(1:2));
    err_theta = abs(angle_diff(x_euler(3), x_ode_final(3)));

    fprintf('  Multi-phase final state comparison:\n');
    fprintf('    Omega error: %.4e rad/s\n', err_omega);
    fprintf('    Position error: %.4e m (%.3f mm)\n', err_pos, err_pos*1000);
    fprintf('    Heading error: %.4e rad (%.4f deg)\n', err_theta, err_theta*180/pi);

    [n_pass, n_fail] = check(err_omega < 0.01, ...
        sprintf('Multi-phase omega err=%.3e (< 0.01 rad/s)', err_omega), n_pass, n_fail);
    [n_pass, n_fail] = check(err_pos < 0.005, ...
        sprintf('Multi-phase pos err=%.3e m (< 5mm)', err_pos), n_pass, n_fail);

    fprintf('\n');

    %% --- Summary ---
    fprintf('=== ODE45 CROSS-VALIDATION SUMMARY ===\n');
    fprintf('Passed: %d / %d\n', n_pass, n_pass + n_fail);
    if n_fail == 0
        fprintf('>>> ALL TESTS PASSED <<<\n');
    else
        fprintf('>>> %d TESTS FAILED <<<\n', n_fail);
    end
end


%% ===================================================================
%  Helper functions
%  ===================================================================

function dz = dynamics_ode(~, z, tau, params)
% DYNAMICS_ODE  ODE function for ode45.
%
% State z = [x; y; theta; w1; w2; w3; w4]  (7 states)
% This is the INDEPENDENT re-implementation of the same physics,
% written from the equations directly — NOT calling plant_step.
%
% Equations:
%   M_eff * dw/dt = tau - b_w * w    (wheel dynamics, no-slip Lagrangian)
%   v_body = H_fwd * w               (forward kinematics)
%   dx/dt = vx*cos(theta) - vy*sin(theta)
%   dy/dt = vx*sin(theta) + vy*cos(theta)
%   dtheta/dt = wz

    theta = z(3);
    omega = z(4:7);

    % Wheel dynamics
    domega = params.M_eff_inv * (tau - params.b_w * omega);

    % Body velocities from forward kinematics
    v_body = params.H_fwd * omega;  % [vx; vy; wz]
    vx = v_body(1);
    vy = v_body(2);
    wz = v_body(3);

    % Pose derivatives (world frame)
    cos_t = cos(theta);
    sin_t = sin(theta);
    dx = vx * cos_t - vy * sin_t;
    dy = vx * sin_t + vy * cos_t;
    dtheta = wz;

    dz = [dx; dy; dtheta; domega];
end


function [n_pass, n_fail] = run_comparison(name, tau, T, params, dt, n_pass, n_fail)
% Run ode45 and plant_step with same inputs, compare results.

    N = round(T / dt);

    % --- ode45 solution ---
    z0 = zeros(7, 1);
    opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
    [t_ode, z_ode] = ode45(@(t,z) dynamics_ode(t, z, tau, params), [0, T], z0, opts);

    % Get final state
    x_ode_final = reconstruct_full_state(z_ode(end,:)', params);

    % --- plant_step solution ---
    x_euler = zeros(10, 1);
    for k = 1:N
        x_euler = plant_step(x_euler, tau, params, dt);
    end

    % --- Compare at final time ---
    err_omega = norm(x_euler(7:10) - x_ode_final(7:10));
    err_vel   = norm(x_euler(4:6) - x_ode_final(4:6));
    err_pos   = norm(x_euler(1:2) - x_ode_final(1:2));
    err_theta = abs(angle_diff(x_euler(3), x_ode_final(3)));

    fprintf('  %s: T=%.1fs\n', name, T);
    fprintf('    Euler omega: [%.4f, %.4f, %.4f, %.4f]\n', x_euler(7:10));
    fprintf('    ode45 omega: [%.4f, %.4f, %.4f, %.4f]\n', x_ode_final(7:10));
    fprintf('    Omega error: %.4e rad/s\n', err_omega);
    fprintf('    Velocity error: %.4e m/s\n', err_vel);
    fprintf('    Position error: %.4e m (%.3f mm)\n', err_pos, err_pos*1000);
    fprintf('    Heading error: %.4e rad (%.4f deg)\n', err_theta, err_theta*180/pi);

    % Tolerance: Euler is O(dt), so at dt=0.001, T=3s:
    % omega error should be ~O(dt) = ~0.001 level (relative to omega_max)
    [n_pass, n_fail] = check(err_omega < 0.05, ...
        sprintf('%s omega error %.3e < 0.05 rad/s', name, err_omega), n_pass, n_fail);
    [n_pass, n_fail] = check(err_pos < 0.01, ...
        sprintf('%s position error %.3e < 10mm', name, err_pos), n_pass, n_fail);

    fprintf('\n');
end


function x10 = reconstruct_full_state(z7, params)
% Reconstruct 10-state vector from 7 independent states.
% z7 = [x, y, theta, w1, w2, w3, w4]
% x10 = [x, y, theta, vx, vy, wz, w1, w2, w3, w4]
    omega = z7(4:7);
    v_body = params.H_fwd * omega;
    x10 = [z7(1:3); v_body; omega];
end


function d = angle_diff(a, b)
% Signed angle difference, wrapped to [-pi, pi].
    d = mod(a - b + pi, 2*pi) - pi;
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
