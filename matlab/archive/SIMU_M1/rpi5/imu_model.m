function [accel, gyro, imu_state] = imu_model(x, x_prev, dt, imu_state, params)
% IMU_MODEL  Simulated IMU with noise and bias drift (STUB).

    accel = zeros(3,1);   % STUB: zero acceleration
    gyro  = [0; 0; x(6)]; % STUB: pass through yaw rate only

    if isempty(imu_state)
        imu_state.accel_bias = zeros(3,1);
        imu_state.gyro_bias  = zeros(3,1);
    end

end
