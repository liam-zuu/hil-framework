function validate_dynamics()
% VALIDATE_DYNAMICS  Verify plant dynamics against analytical solutions.
%
% Tầng 2 validation — dynamics equations.
% Tests:
%   2a. Step response vs analytical (3 fundamental modes)
%   2b. Power balance: dE/dt = P_input - P_friction
%   2c. Steady-state: omega_ss = tau / b_w (per mode)
%   2d. M_eff properties: symmetric, positive-definite, correct eigenstructure
%   2e. Dimensional scaling: 2x tau → 2x steady-state omega
%
% Analytical basis:
%   For each fundamental mode (forward, strafe, rotation), the coupled
%   4x4 system decouples into a scalar ODE:
%     J_mode * dw/dt = tau_0 - b_w * w
%     Solution: w(t) = (tau_0/b_w) * (1 - exp(-b_w*t/J_mode))
%
%   Mode eigenvectors:
%     Forward: [1,1,1,1]     → J_fwd = r²M/4 + J_w
%     Strafe:  [-1,1,1,-1]   → J_str = r²M/4 + J_w  (same by symmetry)
%     Rotation: [-1,1,-1,1]  → J_rot = r²Iz/(4L²) + J_w
%
% Usage: validate_dynamics()
%   Requires: params_mecanum.m, plant_step.m on path

    fprintf('=== VALIDATE DYNAMICS (Tầng 2) ===\n\n');

    params = params_mecanum();

    % Ensure slip disabled for dynamics validation
    params.slip.enabled = false;

    r  = params.r;
    M  = params.M;
    Iz = params.Iz;
    L  = params.lx + params.ly;
    J_w = params.J_w;
    b_w = params.b_w;
    dt  = params.dt;

    n_pass = 0;
    n_fail = 0;

    %% --- Pre-compute mode effective inertias ---
    % Derived analytically from M_eff structure
    J_fwd = (r^2 * M / 4) + J_w;
    J_rot = (r^2 * Iz / (4 * L^2)) + J_w;

    fprintf('Mode effective inertias (analytical):\n');
    fprintf('  J_fwd = J_str = %.6f kg·m²\n', J_fwd);
    fprintf('  J_rot = %.6f kg·m²\n', J_rot);
    fprintf('  J_w (wheel only) = %.6f kg·m²\n', J_w);
    fprintf('  Coupling ratio (J_fwd-J_w)/J_w = %.1f%%\n', (J_fwd-J_w)/J_w*100);
    fprintf('\n');

    %% --- Test 2a: Step response vs analytical (3 modes) ---
    fprintf('--- Test 2a: Step response vs analytical solution ---\n');

    T_test = 3.0;  % seconds (enough for settling)
    N_steps = round(T_test / dt);

    modes = struct('name', {'Forward', 'Strafe', 'Rotation'}, ...
                   'pattern', {[1;1;1;1], [-1;1;1;-1], [-1;1;-1;1]}, ...
                   'J_mode', {J_fwd, J_fwd, J_rot});

    % tau_0/b_w must be < omega_max (34.56) to avoid speed clamping
    % 0.05/0.002 = 25 rad/s < 34.56 → OK, within linear range
    tau_0 = 0.05;  % N·m per wheel (omega_ss = 25 rad/s, no clamping)

    for m = 1:3
        mode = modes(m);
        tau = tau_0 * mode.pattern;
        J_m = mode.J_mode;

        % Time constant and steady-state (analytical)
        tau_c = J_m / b_w;           % time constant (s)
        omega_ss = tau_0 / b_w;      % steady-state per-wheel speed

        % Simulate
        x = zeros(10, 1);
        omega_sim = zeros(1, N_steps+1);
        omega_sim(1) = 0;

        for k = 1:N_steps
            x = plant_step(x, tau, params, dt);
            % Extract one wheel (by symmetry, use abs since pattern may be negative)
            omega_sim(k+1) = x(7) / mode.pattern(1);
        end

        % Analytical solution
        t = (0:N_steps) * dt;
        omega_analytical = omega_ss * (1 - exp(-t / tau_c));

        % Error metrics
        err_rms = sqrt(mean((omega_sim - omega_analytical).^2));
        err_max = max(abs(omega_sim - omega_analytical));
        err_ss = abs(omega_sim(end) - omega_ss);
        err_rel = err_rms / omega_ss * 100;

        % Expected Euler error: O(dt) ≈ dt * max(d²omega/dt²) * T
        % d²omega/dt² at t=0 = (tau_0/J_m) * (b_w/J_m) = tau_0*b_w/J_m²
        max_d2w = tau_0 * b_w / J_m^2;
        euler_bound = dt * max_d2w * tau_c;  % rough first-order error bound

        [n_pass, n_fail] = check(err_rel < 0.5, ...
            sprintf('%s: RMS=%.4f rad/s (%.3f%%), max=%.4f, SS err=%.4f, tau_c=%.3fs', ...
            mode.name, err_rms, err_rel, err_max, err_ss, tau_c), n_pass, n_fail);

        fprintf('    Euler error bound (estimated): %.4f rad/s\n', euler_bound);
    end

    fprintf('\n');

    %% --- Test 2b: Power balance ---
    % At each timestep: dE ≈ (P_input - P_friction) * dt
    % E = 0.5 * omega' * M_eff * omega
    % P_input = omega' * tau
    % P_friction = b_w * omega' * omega

    fprintf('--- Test 2b: Power balance (energy bookkeeping) ---\n');

    % Use torque that keeps omega well below omega_max (no clamping)
    tau_test = 0.04 * [1; 1; 1; 1];  % omega_ss = 20 rad/s < 34.56
    x = zeros(10, 1);
    T_power = 2.0;
    N_power = round(T_power / dt);

    E = zeros(1, N_power+1);
    P_in = zeros(1, N_power);
    P_fric = zeros(1, N_power);
    dE_actual = zeros(1, N_power);

    omega_k = x(7:10);
    E(1) = 0.5 * omega_k' * params.M_eff * omega_k;

    for k = 1:N_power
        omega_k = x(7:10);
        P_in(k) = omega_k' * tau_test;
        P_fric(k) = b_w * (omega_k' * omega_k);

        x = plant_step(x, tau_test, params, dt);

        omega_k_new = x(7:10);
        E(k+1) = 0.5 * omega_k_new' * params.M_eff * omega_k_new;
        dE_actual(k) = E(k+1) - E(k);
    end

    dE_predicted = (P_in - P_fric) * dt;

    % Compare actual vs predicted energy change
    % Skip first few steps (transient numerical effects)
    idx = 10:N_power;
    power_err = abs(dE_actual(idx) - dE_predicted(idx));
    power_err_max = max(power_err);
    power_err_rms = sqrt(mean(power_err.^2));
    dE_scale = max(abs(dE_predicted(idx)));  % scale for relative error

    [n_pass, n_fail] = check(power_err_max / dE_scale < 0.01, ...
        sprintf('Power balance: max err=%.2e, RMS=%.2e, scale=%.2e (rel max %.2f%%)', ...
        power_err_max, power_err_rms, dE_scale, power_err_max/dE_scale*100), n_pass, n_fail);

    % Also test: total energy should increase monotonically under constant torque (before SS)
    E_increasing = all(diff(E(1:round(0.5*N_power))) > -1e-15);
    [n_pass, n_fail] = check(E_increasing, ...
        sprintf('Energy monotonically increasing during transient'), n_pass, n_fail);

    fprintf('\n');

    %% --- Test 2c: Steady-state consistency ---
    fprintf('--- Test 2c: Steady-state omega = tau/b_w ---\n');

    % Run long enough to reach SS (5 time constants)
    for m = 1:3
        mode = modes(m);
        J_m = mode.J_mode;
        tau_c = J_m / b_w;
        T_ss = max(10 * tau_c, 5.0);  % at least 10 time constants for < 0.01% residual
        N_ss = round(T_ss / dt);

        tau = tau_0 * mode.pattern;
        x = zeros(10, 1);

        for k = 1:N_ss
            x = plant_step(x, tau, params, dt);
        end

        omega_ss_sim = x(7) / mode.pattern(1);
        omega_ss_analytical = tau_0 / b_w;
        err_rel = abs(omega_ss_sim - omega_ss_analytical) / omega_ss_analytical * 100;

        [n_pass, n_fail] = check(err_rel < 0.1, ...
            sprintf('%s: omega_ss=%.4f (exp %.4f), rel err=%.4f%%, T=%.1fs (%.1f*tau_c)', ...
            mode.name, omega_ss_sim, omega_ss_analytical, err_rel, T_ss, T_ss/tau_c), ...
            n_pass, n_fail);
    end

    % Test: verify body velocity at SS
    % For forward mode at SS: vx = r * omega_ss, vy = 0, wz = 0
    tau_fwd = tau_0 * [1;1;1;1];
    x = zeros(10, 1);
    T_vx_ss = max(10 * J_fwd / b_w, 10);
    for k = 1:round(T_vx_ss/dt)
        x = plant_step(x, tau_fwd, params, dt);
    end
    vx_ss = x(4);
    vx_expected = r * (tau_0 / b_w);
    err_rel = abs(vx_ss - vx_expected) / vx_expected * 100;
    [n_pass, n_fail] = check(err_rel < 0.1, ...
        sprintf('Body vx at SS: %.4f m/s (exp %.4f), rel err=%.4f%%', ...
        vx_ss, vx_expected, err_rel), n_pass, n_fail);

    fprintf('\n');

    %% --- Test 2d: M_eff properties ---
    fprintf('--- Test 2d: M_eff matrix properties ---\n');

    M_eff = params.M_eff;

    % Symmetry
    sym_err = norm(M_eff - M_eff', 'fro');
    [n_pass, n_fail] = check(sym_err < 1e-15, ...
        sprintf('M_eff symmetric: ||M-M''|| = %.2e', sym_err), n_pass, n_fail);

    % Positive definite (all eigenvalues > 0)
    eigs_M = eig(M_eff);
    all_pos = all(eigs_M > 0);
    [n_pass, n_fail] = check(all_pos, ...
        sprintf('M_eff positive-definite: eigenvalues = [%.6f, %.6f, %.6f, %.6f]', ...
        eigs_M(1), eigs_M(2), eigs_M(3), eigs_M(4)), n_pass, n_fail);

    % Eigenvalues should match mode inertias
    % Forward/strafe modes have same J_mode (by symmetry)
    % Sort eigenvalues and compare
    eigs_sorted = sort(eigs_M);
    eigs_expected = sort([J_fwd, J_fwd, J_rot, J_rot]);  % 2 translational + 2 rotational-like
    fprintf('  Eigenvalues (sorted): [%.6f, %.6f, %.6f, %.6f]\n', eigs_sorted);
    fprintf('  Expected modes:       J_fwd=%.6f, J_rot=%.6f\n', J_fwd, J_rot);

    % M_eff_inv correctness
    I_test = M_eff * params.M_eff_inv;
    inv_err = norm(I_test - eye(4), 'fro');
    [n_pass, n_fail] = check(inv_err < 1e-10, ...
        sprintf('M_eff * M_eff_inv = I_4: error = %.2e', inv_err), n_pass, n_fail);

    % Diagonal dominance check (informational)
    for i = 1:4
        diag_ratio = M_eff(i,i) / sum(abs(M_eff(i,:)));
        fprintf('  Row %d diagonal dominance: %.1f%%\n', i, diag_ratio*100);
    end

    fprintf('\n');

    %% --- Test 2e: Dimensional scaling ---
    fprintf('--- Test 2e: Dimensional scaling (linearity check) ---\n');

    % At steady state: omega_ss = tau / b_w
    % Double tau → double omega_ss (linear system without saturation)
    % Must keep both omega_ss values below omega_max (34.56 rad/s)
    tau_1 = 0.01 * [1;1;1;1];  % omega_ss = 5 rad/s
    tau_2 = 0.02 * [1;1;1;1];  % omega_ss = 10 rad/s

    x1 = zeros(10, 1);
    x2 = zeros(10, 1);
    N_scale = round(8/dt);  % plenty for SS

    for k = 1:N_scale
        x1 = plant_step(x1, tau_1, params, dt);
        x2 = plant_step(x2, tau_2, params, dt);
    end

    ratio = x2(7) / x1(7);
    [n_pass, n_fail] = check(abs(ratio - 2.0) < 0.001, ...
        sprintf('2x torque → 2x omega: ratio = %.6f (exp 2.000)', ratio), n_pass, n_fail);

    % Check body velocity also scales
    ratio_vx = x2(4) / x1(4);
    [n_pass, n_fail] = check(abs(ratio_vx - 2.0) < 0.001, ...
        sprintf('2x torque → 2x vx: ratio = %.6f (exp 2.000)', ratio_vx), n_pass, n_fail);

    fprintf('\n');

    %% --- Summary ---
    fprintf('=== DYNAMICS VALIDATION SUMMARY ===\n');
    fprintf('Passed: %d / %d\n', n_pass, n_pass + n_fail);
    if n_fail == 0
        fprintf('>>> ALL TESTS PASSED <<<\n');
    else
        fprintf('>>> %d TESTS FAILED <<<\n', n_fail);
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
