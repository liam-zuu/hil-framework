function x_new = plant_step(x, tau, params, dt)
% PLANT_STEP  Mecanum AGV dynamics — kinematics + dynamics, NO wheel slip.
%
% Model (Lagrangian, no-slip constraint):
%   M_eff * d(omega)/dt = tau - b_w * omega
%   v_body = H_fwd * omega   (forward kinematics)
%   d(pose)/dt = rotation(theta) * v_body
%
% Wheel slip is added in M6. Under no-slip, body velocities are
% kinematic functions of wheel speeds — not independent states.
%
% Integration: semi-implicit Euler (velocity-first) with midpoint
% rotation for pose. Stable at dt=0.001 given time constants ~3s.
%
% Input:
%   x      [10x1] state: [x,y,theta, vx,vy,wz, w1,w2,w3,w4]
%   tau    [4x1]  applied torques (N.m)
%   params [struct] from params_mecanum
%   dt     [scalar] timestep (s)
% Output:
%   x_new  [10x1] next state

    % Extract current state
    theta = x(3);
    omega = x(7:10);  % wheel speeds [4x1]

    % --- Step 1: Wheel dynamics ---
    % M_eff * d(omega)/dt = tau - b_w * omega
    % Semi-implicit Euler: solve for omega_new
    domega = params.M_eff_inv * (tau - params.b_w * omega);
    omega_new = omega + dt * domega;

    % Clamp to max wheel speed
    omega_new = max(-params.omega_max, min(params.omega_max, omega_new));

    % --- Step 2: Forward kinematics ---
    % Body velocities from wheel speeds (no-slip constraint)
    v_body = params.H_fwd * omega_new;  % [vx; vy; wz] in body frame
    vx_new = v_body(1);
    vy_new = v_body(2);
    wz_new = v_body(3);

    % --- Step 3: Pose integration (midpoint rotation) ---
    theta_mid = theta + wz_new * dt / 2;
    cos_t = cos(theta_mid);
    sin_t = sin(theta_mid);

    x_pos_new = x(1) + (vx_new * cos_t - vy_new * sin_t) * dt;
    y_pos_new = x(2) + (vx_new * sin_t + vy_new * cos_t) * dt;
    theta_new = theta + wz_new * dt;

    % Normalize theta to [-pi, pi]
    theta_new = mod(theta_new + pi, 2*pi) - pi;

    % --- Assemble new state ---
    x_new = [x_pos_new;
             y_pos_new;
             theta_new;
             vx_new;
             vy_new;
             wz_new;
             omega_new];

end
