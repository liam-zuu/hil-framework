function x_new = plant_step(x, tau, params, dt)
% PLANT_STEP  Mecanum AGV dynamics — kinematics + dynamics + optional wheel slip.
%
% Model (Lagrangian, no-slip baseline):
%   M_eff * d(omega)/dt = tau_eff - b_w * omega
%   v_body = H_fwd * omega   (forward kinematics)
%   d(pose)/dt = rotation(theta) * v_body
%
% Wheel slip model (M6, when params.slip.enabled = true):
%   For each wheel:
%     F_N = M*g/4 (normal force, equal weight distribution)
%     tau_friction_max = mu_static * F_N * r (max torque before slip)
%     If |tau_i| > tau_friction_max  OR  spontaneous event:
%       tau_eff_i = sign(tau_i) * mu_kinetic/mu_static * tau_friction_max
%                   + stochastic noise
%       → wheel spins faster (less ground resistance)
%       → body receives less force (reduced traction)
%     Else:
%       tau_eff_i = tau_i (full traction, no slip)
%
% This models the key physical effect: during slip, the wheel-ground
% friction drops from static to kinetic, reducing force transmission.
% ADRC ESO should estimate this force loss as part of z2 (disturbance).
%
% Integration: semi-implicit Euler with midpoint rotation for pose.
%
% Input:
%   x      [10x1] state: [x,y,theta, vx,vy,wz, w1,w2,w3,w4]
%   tau    [4x1]  applied torques (N.m), may include external disturbance
%   params [struct] from params_mecanum (with optional .slip struct)
%   dt     [scalar] timestep (s)
% Output:
%   x_new  [10x1] next state

    % Extract current state
    theta = x(3);
    omega = x(7:10);  % wheel speeds [4x1]

    % --- Wheel slip model (M6) ---
    tau_eff = tau;  % default: full traction

    if isfield(params, 'slip') && params.slip.enabled
        F_N = params.M * params.g / 4;  % normal force per wheel (equal distribution)
        tau_static_max = params.slip.mu_static * F_N * params.r;  % max static friction torque
        mu_ratio = params.slip.mu_kinetic / params.slip.mu_static;  % kinetic/static ratio

        for i = 1:4
            is_slipping = false;

            % Condition 1: torque exceeds static friction limit
            if abs(tau(i)) > tau_static_max
                is_slipping = true;
            end

            % Condition 2: spontaneous slip (surface imperfections, random)
            if rand < params.slip.prob_spontaneous
                is_slipping = true;
            end

            if is_slipping
                % During slip: effective torque drops to kinetic friction level
                % with stochastic variation
                noise = 1 + params.slip.noise_sigma * randn;
                noise = max(0.5, min(1.5, noise));  % bound noise factor

                tau_kinetic = tau_static_max * mu_ratio * noise;
                tau_eff(i) = sign(tau(i)) * min(abs(tau(i)), tau_kinetic);

                % If tau was below kinetic level (spontaneous slip only),
                % reduce by a random fraction
                if abs(tau(i)) <= tau_kinetic
                    slip_severity = 0.3 + 0.4 * rand;  % lose 30-70% of torque
                    tau_eff(i) = tau(i) * (1 - slip_severity);
                end
            end
        end
    end

    % --- Step 1: Wheel dynamics ---
    % M_eff * d(omega)/dt = tau_eff - b_w * omega
    domega = params.M_eff_inv * (tau_eff - params.b_w * omega);
    omega_new = omega + dt * domega;

    % Clamp to max wheel speed
    omega_new = max(-params.omega_max, min(params.omega_max, omega_new));

    % --- Step 2: Forward kinematics ---
    % Body velocities from wheel speeds (kinematic constraint)
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
