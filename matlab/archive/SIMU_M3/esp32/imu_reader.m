function [accel, gyro, valid] = imu_reader(packet, params)
% IMU_READER  Unpack UART IMU packet (STUB — updated for M3 packet format).
%
% Decodes accel_raw/gyro_raw from imu_packet_enc using scale factors.
% Verifies header and checksum for packet integrity.
% M4 will add: filtering, outlier rejection, error handling.

    % --- Header check ---
    if packet.header ~= uint8(hex2dec('AA'))
        accel = zeros(3,1);
        gyro  = zeros(3,1);
        valid = false;
        return;
    end

    % --- Checksum verification ---
    all_codes = [packet.accel_raw; packet.gyro_raw];
    chk = int32(0);
    for i = 1:6
        chk = bitxor(chk, all_codes(i));
    end
    if chk ~= packet.checksum
        accel = zeros(3,1);
        gyro  = zeros(3,1);
        valid = false;
        return;
    end

    % --- Decode ADC codes back to float ---
    accel = double(packet.accel_raw) * packet.accel_scale;
    gyro  = double(packet.gyro_raw)  * packet.gyro_scale;

    valid = true;

end
