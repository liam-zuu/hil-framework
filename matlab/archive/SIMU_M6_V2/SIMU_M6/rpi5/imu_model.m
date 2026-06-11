function [accel, gyro, imu_state] = imu_model(x, x_prev, dt, imu_state, params)
% IMU_MODEL  Simulated IMU: accelerometer + gyroscope with noise & bias drift.
%
% Physics (body frame, planar robot):
%   True accel: ax = dvx/dt - vy*wz  (centripetal correction)
%               ay = dvy/dt + vx*wz
%               az = +g              (gravity reaction when flat)
%   True gyro:  gx = 0, gy = 0, gz = wz
%
% Noise model:
%   bias(k) = bias(k-1) + sqrt(dt) * drift_coeff * randn  (random walk)
%   measurement = true_value + bias + noise_sigma * randn
%
% Input:
%   x         [10x1] current state
%   x_prev    [10x1] previous state
%   dt        [scalar] timestep
%   imu_state [struct] persistent bias states ([] to initialize)
%   params    [struct] noise/bias parameters
% Output:
%   accel     [3x1] accelerometer readings (m/s^2) — body frame
%   gyro      [3x1] gyroscope readings (rad/s) — body frame
%   imu_state [struct] updated bias states

    % Initialize bias states on first call
    if isempty(imu_state)
        imu_state.accel_bias = params.imu_accel_bias0 * randn(3,1);
        imu_state.gyro_bias  = params.imu_gyro_bias0  * randn(3,1);
    end

    % --- True body-frame acceleration ---
    vx     = x(4);      vy     = x(5);      wz     = x(6);
    vx_prev = x_prev(4); vy_prev = x_prev(5);

    % Finite difference for dv/dt + Coriolis/centripetal terms
    dvx_dt = (vx - vx_prev) / dt;
    dvy_dt = (vy - vy_prev) / dt;

    true_accel = [dvx_dt - vy * wz;      % body-x: includes centripetal
                  dvy_dt + vx * wz;      % body-y: includes centripetal
                  params.g];              % body-z: gravity reaction (flat surface)

    % --- True body-frame gyroscope ---
    true_gyro = [0;       % roll rate (zero for planar)
                 0;       % pitch rate (zero for planar)
                 wz];     % yaw rate

    % --- Bias drift (random walk) ---
    imu_state.accel_bias = imu_state.accel_bias + ...
        sqrt(dt) * params.imu_bias_drift * randn(3,1);
    imu_state.gyro_bias  = imu_state.gyro_bias + ...
        sqrt(dt) * params.imu_bias_drift * randn(3,1);

    % --- Measurement = true + bias + white noise ---
    accel = true_accel + imu_state.accel_bias + ...
            params.imu_accel_noise * randn(3,1);
    gyro  = true_gyro  + imu_state.gyro_bias + ...
            params.imu_gyro_noise  * randn(3,1);

end
