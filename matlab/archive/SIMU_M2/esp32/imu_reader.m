function [accel, gyro, valid] = imu_reader(packet, params)
% IMU_READER  Unpack UART IMU packet (STUB).

    accel = packet.accel;  % STUB: passthrough
    gyro  = packet.gyro;
    valid = true;          % STUB: always valid

end
