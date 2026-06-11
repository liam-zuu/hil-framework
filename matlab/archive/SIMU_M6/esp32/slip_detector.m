function [slip_flag, slip_ratio] = slip_detector(omega_est, accel, gyro, params)
% SLIP_DETECTOR  Detect wheel slip from encoder + IMU sensor redundancy.
%
% Detection strategy — two complementary methods:
%
% Method 1: Kinematic consistency (overdetermined system)
%   Mecanum has 4 wheels but only 3 DOF (vx, vy, wz). With no slip,
%   all 4 wheels are kinematically consistent: H_fwd * omega = v_body.
%   For each wheel i, compute predicted omega_i from the body velocity
%   estimated by ALL wheels. If actual omega_i deviates significantly
%   from prediction, wheel i may be slipping.
%   Slip ratio = (omega_actual - omega_predicted) / max(|omega_actual|, eps)
%
% Method 2: IMU cross-check (yaw rate)
%   Compare encoder-derived yaw rate (from H_fwd) with IMU gyro wz.
%   Large discrepancy suggests at least one wheel is slipping.
%   This is a global flag, not per-wheel.
%
% Combined decision: slip_flag(i) = true if |slip_ratio(i)| > threshold
%   OR if IMU cross-check fails AND wheel i has highest residual.
%
% Input:
%   omega_est  [4x1] estimated wheel speeds (rad/s) from encoder_reader
%   accel      [3x1] accelerometer readings (m/s²) — body frame
%   gyro       [3x1] gyroscope readings (rad/s) — body frame
%   params     [struct] with .H_fwd, .H_inv, .slip.detect_threshold, etc.
% Output:
%   slip_flag  [4x1 logical] true if wheel i is detected as slipping
%   slip_ratio [4x1] per-wheel slip ratio (deviation from kinematic prediction)

    %% Default threshold (can be overridden in params)
    if isfield(params, 'slip') && isfield(params.slip, 'detect_threshold')
        threshold = params.slip.detect_threshold;
    else
        threshold = 0.15;  % 15% deviation → flag as slip
    end

    if isfield(params, 'slip') && isfield(params.slip, 'imu_wz_threshold')
        imu_wz_thresh = params.slip.imu_wz_threshold;
    else
        imu_wz_thresh = 0.5;  % rad/s discrepancy → IMU cross-check fails
    end

    %% Method 1: Kinematic consistency
    % Compute body velocity from all 4 wheels (least-squares)
    v_body = params.H_fwd * omega_est;   % [vx; vy; wz]

    % Predict what each wheel speed SHOULD be, given v_body
    omega_predicted = params.H_inv * v_body;  % [4x1]

    % Slip ratio: how much each wheel deviates from prediction
    % Normalize by max(|omega_actual|, small_value) to get relative error
    omega_scale = max(abs(omega_est), 1.0);  % avoid division by zero; 1.0 rad/s floor
    slip_ratio = (omega_est - omega_predicted) ./ omega_scale;

    %% Method 2: IMU yaw rate cross-check
    wz_encoder = v_body(3);       % yaw rate from encoders
    wz_imu = gyro(3);             % yaw rate from IMU
    imu_mismatch = abs(wz_encoder - wz_imu) > imu_wz_thresh;

    %% Combined decision
    slip_flag = abs(slip_ratio) > threshold;

    % If IMU cross-check fails, also flag the wheel with highest residual
    if imu_mismatch && ~any(slip_flag)
        [~, worst_wheel] = max(abs(slip_ratio));
        slip_flag(worst_wheel) = true;
    end

end
