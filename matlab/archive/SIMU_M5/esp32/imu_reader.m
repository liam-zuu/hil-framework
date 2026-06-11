function [accel, gyro, valid] = imu_reader(packet, params)
% IMU_READER  Unpack UART IMU packet, reject outliers, and low-pass filter.
%
% Three-stage pipeline:
%   1. Decode: header check → checksum verify → ADC codes to float
%   2. Outlier rejection: if |new - prev| > threshold, hold previous value
%   3. First-order IIR low-pass filter (same structure as encoder_reader)
%
% Uses MATLAB persistent variables for filter and outlier state.
% Call "clear imu_reader" between simulation runs to reset.
%
% Input:
%   packet [struct] IMU packet from imu_packet_enc
%                   .header, .accel_raw, .gyro_raw, .accel_scale, .gyro_scale, .checksum
%   params [struct] with .imu_filter_tau, .imu_outlier_accel, .imu_outlier_gyro
% Output:
%   accel  [3x1] filtered accelerometer readings (m/s²)
%   gyro   [3x1] filtered gyroscope readings (rad/s)
%   valid  [logical] true if packet passed header + checksum

    persistent accel_prev gyro_prev accel_filt gyro_filt initialized;

    % Initialize on first call
    if isempty(initialized)
        accel_prev = zeros(3,1);
        gyro_prev  = zeros(3,1);
        accel_filt = zeros(3,1);
        gyro_filt  = zeros(3,1);
        initialized = true;
    end

    %% Stage 1: Decode with integrity check
    % Header check
    if packet.header ~= uint8(hex2dec('AA'))
        accel = accel_filt;  % Return last filtered value on error
        gyro  = gyro_filt;
        valid = false;
        return;
    end

    % Checksum verification
    all_codes = [packet.accel_raw; packet.gyro_raw];
    chk = int32(0);
    for i = 1:6
        chk = bitxor(chk, all_codes(i));
    end
    if chk ~= packet.checksum
        accel = accel_filt;  % Return last filtered value on error
        gyro  = gyro_filt;
        valid = false;
        return;
    end

    % Decode ADC codes to float
    accel_raw = double(packet.accel_raw) * packet.accel_scale;
    gyro_raw  = double(packet.gyro_raw)  * packet.gyro_scale;

    valid = true;

    %% Stage 2: Outlier rejection
    % If jump from previous raw value exceeds threshold, hold previous
    accel_thr = params.imu_outlier_accel;
    gyro_thr  = params.imu_outlier_gyro;

    for i = 1:3
        if abs(accel_raw(i) - accel_prev(i)) > accel_thr
            accel_raw(i) = accel_prev(i);
        end
        if abs(gyro_raw(i) - gyro_prev(i)) > gyro_thr
            gyro_raw(i) = gyro_prev(i);
        end
    end

    accel_prev = accel_raw;
    gyro_prev  = gyro_raw;

    %% Stage 3: First-order IIR low-pass filter
    dt    = params.dt;
    tau_f = params.imu_filter_tau;
    alpha = dt / (tau_f + dt);

    accel_filt = alpha * accel_raw + (1 - alpha) * accel_filt;
    gyro_filt  = alpha * gyro_raw  + (1 - alpha) * gyro_filt;

    accel = accel_filt;
    gyro  = gyro_filt;

end
