function [tau_cmd, adrc_state] = adrc_controller(omega_ref, omega_est, imu_data, adrc_state, params)
% ADRC_CONTROLLER  Per-wheel Active Disturbance Rejection Control.
%
% 2nd-order ESO for 1st-order velocity plant:
%   Plant: J * dw/dt = tau + d
%   Rewrite: dw/dt = b0*tau + f,  b0 = 1/J (known),  f = d/J (unknown)
%
% ESO estimates:
%   z1 — plant output (omega)
%   z2 — total disturbance f (friction + coupling + load + model mismatch)
%
% ESO equations (forward Euler):
%   e_eso = z1 - omega_meas
%   z1(k+1) = z1(k) + dt * (b0*u(k-1) + z2(k) - beta1*e_eso)
%   z2(k+1) = z2(k) + dt * (-beta2 * e_eso)
%
% Control law (proportional + disturbance rejection):
%   u0  = kp * (omega_ref - z1)       ← desired acceleration
%   tau = (u0 - z2) / b0              ← cancel disturbance, apply control
%
% ESO bandwidth w_o from beta gains: beta1 = 2*w_o, beta2 = w_o^2
% Controller bandwidth w_c = kp
%
% imu_data is available for future use (slip detection M6) but not
% used in the basic ADRC — ESO estimates disturbance from output only.
%
% Input:
%   omega_ref   [4x1] reference wheel velocities (rad/s)
%   omega_est   [4x1] estimated wheel velocities (rad/s) from encoder_reader
%   imu_data    [struct] .accel [3x1], .gyro [3x1] — reserved for M6
%   adrc_state  [struct or []] ESO states (.z1, .z2, .u_prev)
%   params      [struct] with .adrc.b0, .eso_beta1, .eso_beta2, .kp, .dt
% Output:
%   tau_cmd     [4x1] torque command (N·m), NOT clamped (pwm_output handles)
%   adrc_state  [struct] updated ESO states

    %% Initialize state on first call
    if isempty(adrc_state)
        adrc_state.z1     = zeros(4, 1);  % ESO: estimated omega
        adrc_state.z2     = zeros(4, 1);  % ESO: estimated total disturbance
        adrc_state.u_prev = zeros(4, 1);  % Previous control input for ESO
    end

    dt    = params.dt;
    b0    = params.adrc.b0;
    beta1 = params.adrc.eso_beta1;
    beta2 = params.adrc.eso_beta2;
    kp    = params.adrc.kp;

    %% ESO update (forward Euler, per wheel)
    % Observation error: how far ESO estimate is from measurement
    e_eso = adrc_state.z1 - omega_est;

    % 2nd-order ESO
    z1_new = adrc_state.z1 + dt * (b0 * adrc_state.u_prev + adrc_state.z2 - beta1 * e_eso);
    z2_new = adrc_state.z2 + dt * (-beta2 * e_eso);

    adrc_state.z1 = z1_new;
    adrc_state.z2 = z2_new;

    %% Control law (proportional + disturbance rejection)
    u0 = kp * (omega_ref - adrc_state.z1);

    % Disturbance compensation: subtract estimated disturbance
    tau_cmd = (u0 - adrc_state.z2) / b0;

    % Store control input for next ESO update
    adrc_state.u_prev = tau_cmd;

end
