function validate_kinematics()
% VALIDATE_KINEMATICS  Verify H_fwd, H_inv against analytical formulas.
%
% Tầng 1 validation — kinematics matrices.
% Tests:
%   1a. Motion primitive spot-checks (hand-calculated)
%   1b. H_inv matches textbook formula element-by-element
%   1c. H_fwd = pseudo-inverse of H_inv
%   1d. Forward(Inverse(v)) = v for arbitrary v
%   1e. Dimensional consistency: units check
%
% Reference: Standard X-config mecanum kinematics
%   Taheri et al. 2015, Muir & Neuman 1987
%   Wheel numbering: 1=FL, 2=FR, 3=RL, 4=RR
%   Roller axis at 45° to wheel axis
%
% Usage: validate_kinematics()
%   Requires: params_mecanum.m on path

    fprintf('=== VALIDATE KINEMATICS (Tầng 1) ===\n\n');

    params = params_mecanum();
    r  = params.r;
    lx = params.lx;
    ly = params.ly;
    L  = lx + ly;

    H_fwd = params.H_fwd;  % 3x4
    H_inv = params.H_inv;  % 4x3

    n_pass = 0;
    n_fail = 0;
    tol = 1e-12;

    %% --- Test 1a: Motion primitive spot-checks ---
    % Each test: given omega pattern, compute v_body = H_fwd * omega
    % Compare with hand-calculated expected values

    fprintf('--- Test 1a: Motion primitive spot-checks ---\n');

    omega_0 = 10.0;  % rad/s, arbitrary test speed

    % Test 1a.1: Pure forward — all wheels same direction, same speed
    % Expected: vx = r*omega_0, vy = 0, wz = 0
    omega_fwd = omega_0 * [1; 1; 1; 1];
    v_fwd = H_fwd * omega_fwd;
    v_fwd_expected = [r * omega_0; 0; 0];
    err = norm(v_fwd - v_fwd_expected);
    [n_pass, n_fail] = check(err < tol, ...
        sprintf('Forward: vx=%.4f (exp %.4f), vy=%.2e, wz=%.2e, err=%.2e', ...
        v_fwd(1), v_fwd_expected(1), v_fwd(2), v_fwd(3), err), n_pass, n_fail);

    % Test 1a.2: Pure left strafe — pattern [-1, +1, +1, -1]
    % Expected: vx = 0, vy = r*omega_0, wz = 0
    omega_strafe = omega_0 * [-1; 1; 1; -1];
    v_strafe = H_fwd * omega_strafe;
    v_strafe_expected = [0; r * omega_0; 0];
    err = norm(v_strafe - v_strafe_expected);
    [n_pass, n_fail] = check(err < tol, ...
        sprintf('Strafe left: vx=%.2e, vy=%.4f (exp %.4f), wz=%.2e, err=%.2e', ...
        v_strafe(1), v_strafe(2), v_strafe_expected(2), v_strafe(3), err), n_pass, n_fail);

    % Test 1a.3: Pure CCW rotation — pattern [-1, +1, -1, +1]
    % Expected: vx = 0, vy = 0, wz = r*omega_0/L
    omega_rot = omega_0 * [-1; 1; -1; 1];
    v_rot = H_fwd * omega_rot;
    v_rot_expected = [0; 0; r * omega_0 / L];
    err = norm(v_rot - v_rot_expected);
    [n_pass, n_fail] = check(err < tol, ...
        sprintf('CCW rotation: vx=%.2e, vy=%.2e, wz=%.4f (exp %.4f), err=%.2e', ...
        v_rot(1), v_rot(2), v_rot(3), v_rot_expected(3), err), n_pass, n_fail);

    % Test 1a.4: Diagonal (forward + left strafe) — pattern [0, +1, +1, 0]
    % omega = omega_0*[0;1;1;0] = 0.5*(fwd + strafe)
    % Expected: vx = r*omega_0/2, vy = r*omega_0/2, wz = 0
    omega_diag = omega_0 * [0; 1; 1; 0];
    v_diag = H_fwd * omega_diag;
    v_diag_expected = [r * omega_0 / 2; r * omega_0 / 2; 0];
    err = norm(v_diag - v_diag_expected);
    [n_pass, n_fail] = check(err < tol, ...
        sprintf('Diagonal: vx=%.4f (exp %.4f), vy=%.4f (exp %.4f), err=%.2e', ...
        v_diag(1), v_diag_expected(1), v_diag(2), v_diag_expected(2), err), n_pass, n_fail);

    % Test 1a.5: Forward + CCW rotation — pattern [0, +1, 0, +1]
    % omega = omega_0*[0;1;0;1] = 0.5*(fwd + rot)
    % Expected: vx = r*omega_0/2, vy = 0, wz = r*omega_0/(2L)
    omega_fwd_rot = omega_0 * [0; 1; 0; 1];
    v_fwd_rot = H_fwd * omega_fwd_rot;
    v_fwd_rot_expected = [r * omega_0 / 2; 0; r * omega_0 / (2*L)];
    err = norm(v_fwd_rot - v_fwd_rot_expected);
    [n_pass, n_fail] = check(err < tol, ...
        sprintf('Forward+CCW: vx=%.4f (exp %.4f), wz=%.4f (exp %.4f), err=%.2e', ...
        v_fwd_rot(1), v_fwd_rot_expected(1), v_fwd_rot(3), v_fwd_rot_expected(3), err), n_pass, n_fail);

    fprintf('\n');

    %% --- Test 1b: H_inv matches textbook formula ---
    % Standard X-config mecanum inverse kinematics (Taheri et al.):
    %   w1 = (1/r) * (vx - vy - L*wz)   [FL]
    %   w2 = (1/r) * (vx + vy + L*wz)   [FR]
    %   w3 = (1/r) * (vx + vy - L*wz)   [RL]
    %   w4 = (1/r) * (vx - vy + L*wz)   [RR]

    fprintf('--- Test 1b: H_inv vs textbook formula ---\n');

    H_inv_textbook = (1/r) * [1, -1, -L;
                               1,  1,  L;
                               1,  1, -L;
                               1, -1,  L];

    err = norm(H_inv - H_inv_textbook, 'fro');
    [n_pass, n_fail] = check(err < tol, ...
        sprintf('H_inv matches textbook: Frobenius error = %.2e', err), n_pass, n_fail);

    % Also verify H_fwd matches textbook forward kinematics:
    %   vx = (r/4) * (w1 + w2 + w3 + w4)
    %   vy = (r/4) * (-w1 + w2 + w3 - w4)
    %   wz = (r/(4L)) * (-w1 + w2 - w3 + w4)
    H_fwd_textbook = (r/4) * [1,  1,  1,  1;
                              -1,  1,  1, -1;
                              -1/L, 1/L, -1/L, 1/L];

    err = norm(H_fwd - H_fwd_textbook, 'fro');
    [n_pass, n_fail] = check(err < tol, ...
        sprintf('H_fwd matches textbook: Frobenius error = %.2e', err), n_pass, n_fail);

    fprintf('\n');

    %% --- Test 1c: H_fwd is left pseudo-inverse of H_inv ---
    % H_fwd * H_inv should = I_3 (3x3 identity)
    % Because 4 wheels, 3 DOF → H_fwd is the pseudo-inverse

    fprintf('--- Test 1c: Pseudo-inverse relationship ---\n');

    I3_test = H_fwd * H_inv;
    err = norm(I3_test - eye(3), 'fro');
    [n_pass, n_fail] = check(err < tol, ...
        sprintf('H_fwd * H_inv = I_3: error = %.2e', err), n_pass, n_fail);

    % H_inv * H_fwd should NOT be I_4 (system is overdetermined)
    % But it should be a symmetric projection matrix (H_inv * H_fwd)^2 = H_inv * H_fwd
    P = H_inv * H_fwd;
    err_proj = norm(P*P - P, 'fro');
    [n_pass, n_fail] = check(err_proj < tol, ...
        sprintf('H_inv * H_fwd is projection: ||P²-P|| = %.2e', err_proj), n_pass, n_fail);

    fprintf('\n');

    %% --- Test 1d: Round-trip for arbitrary body velocities ---
    fprintf('--- Test 1d: Round-trip Forward(Inverse(v)) = v ---\n');

    test_velocities = [0.5,  0,    0;      % forward
                       0,    0.3,  0;      % strafe
                       0,    0,    1.5;    % rotation
                       0.3,  0.2,  0.5;    % combo 1
                      -0.4,  0.6, -1.0;    % combo 2
                       1.0, -1.0,  2.0]';  % extreme combo

    all_pass = true;
    max_err = 0;
    for i = 1:size(test_velocities, 2)
        v_in = test_velocities(:, i);
        v_out = H_fwd * (H_inv * v_in);
        err = norm(v_out - v_in);
        max_err = max(max_err, err);
        if err > tol
            all_pass = false;
        end
    end
    [n_pass, n_fail] = check(all_pass, ...
        sprintf('Round-trip 6 velocities: max error = %.2e', max_err), n_pass, n_fail);

    fprintf('\n');

    %% --- Test 1e: Dimensional consistency ---
    fprintf('--- Test 1e: Dimensional consistency ---\n');

    % H_inv has units (1/m) * [m/s ... rad/s*m] → rad/s. Correct for omega.
    % H_fwd has units (m) * [rad/s] → m/s (rows 1-2) or rad/s (row 3). Check:

    % Test: scale vx by factor k → omega should scale by k
    v1 = [1; 0; 0];
    v2 = [2; 0; 0];
    omega1 = H_inv * v1;
    omega2 = H_inv * v2;
    err = norm(omega2 - 2*omega1);
    [n_pass, n_fail] = check(err < tol, ...
        sprintf('H_inv linearity (2*vx → 2*omega): error = %.2e', err), n_pass, n_fail);

    % Test: at omega_max, what is max body speed?
    % All 4 wheels at omega_max → vx_max = r * omega_max
    vx_max = r * params.omega_max;
    fprintf('  Info: At omega_max=%.2f rad/s → vx_max=%.3f m/s (%.1f mm/s)\n', ...
        params.omega_max, vx_max, vx_max*1000);
    fprintf('  Info: At omega_max → wz_max=%.3f rad/s (%.1f deg/s)\n', ...
        r * params.omega_max / L, r * params.omega_max / L * 180/pi);

    fprintf('\n');

    %% --- Summary ---
    fprintf('=== KINEMATICS VALIDATION SUMMARY ===\n');
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
