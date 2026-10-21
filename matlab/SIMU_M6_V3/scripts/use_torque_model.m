function params = use_torque_model(params)
% USE_TORQUE_MODEL  Switch params to the legacy M6 torque-input plant.
%   params = use_torque_model(params_mecanum());
% Used by the M3-M6 scripts in scripts/ and validate/, which pass torques
% (not PWM duty) to plant_step. The DC motor model is the default since
% 21/10/2026 (spec S01). Polarity and firmware sign are set to +1 so these
% scripts behave exactly as before (also the pre-21/10 IMU noise model).
    params.motor_model    = 'torque';
    params.motor_polarity = ones(4, 1);
    params.motor_dir      = ones(4, 1);
    % IMU noise model before 21/10/2026 (white noise every step, no 20 Hz hold)
    params.imu_accel_noise = 0.05;
    params.imu_gyro_noise  = 0.005;
    params.imu_rate_hz     = 0;
end
