function [pose_est, pe_state] = pose_estimator(omega_est, gyro_meas, pe_state, params)
% POSE_ESTIMATOR  Dead reckoning odometry from encoder + IMU.
%
% Estimates robot pose [x, y, theta] in world frame by integrating
% body velocities derived from wheel encoder measurements.
%
% Pipeline:
%   1. Forward kinematics: omega_est → body velocities (vx, vy, wz)
%   2. Heading: use gyro wz for heading rate (less drift than encoder-only)
%   3. Integrate body velocities in world frame (midpoint rotation)
%
% This is the ESP32's best estimate of its own position, used by
% position_controller.m for the outer loop. Errors accumulate over
% time (drift), which is realistic for encoder+IMU dead reckoning.
%
% Uses struct state instead of persistent to allow clean reset.
%
% Input:
%   omega_est  [4x1] filtered wheel velocities (rad/s) from encoder_reader
%   gyro_meas  [3x1] filtered gyroscope readings (rad/s) from imu_reader
%   pe_state   [struct or []] estimator state (.x, .y, .theta)
%   params     [struct] with .H_fwd, .dt
% Output:
%   pose_est   [3x1] estimated pose [x; y; theta] in world frame
%   pe_state   [struct] updated estimator state

    dt = params.dt;

    %% Initialize on first call
    if isempty(pe_state)
        pe_state.x     = 0;
        pe_state.y     = 0;
        pe_state.theta = 0;
    end

    %% Step 1: Forward kinematics — wheel speeds to body velocities
    v_body = params.H_fwd * omega_est;   % [vx; vy; wz] body frame
    vx = v_body(1);
    vy = v_body(2);

    %% Step 2: Heading rate — prefer gyro (less quantization noise)
    % gyro_meas(3) = wz from IMU, already filtered by imu_reader
    wz = gyro_meas(3);

    %% Step 3: Integrate pose (midpoint rotation, same as plant_step)
    theta_mid = pe_state.theta + wz * dt / 2;
    cos_t = cos(theta_mid);
    sin_t = sin(theta_mid);

    pe_state.x     = pe_state.x + (vx * cos_t - vy * sin_t) * dt;
    pe_state.y     = pe_state.y + (vx * sin_t + vy * cos_t) * dt;
    pe_state.theta = pe_state.theta + wz * dt;

    % Normalize theta to [-pi, pi]
    pe_state.theta = mod(pe_state.theta + pi, 2*pi) - pi;

    %% Output
    pose_est = [pe_state.x; pe_state.y; pe_state.theta];

end
