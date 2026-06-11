function packet = imu_packet_enc(accel, gyro, params)
% IMU_PACKET_ENC  Pack IMU data into a UART-style packet.
%
% Simulates what the Nucleo H7 does when forwarding IMU data to ESP32:
%   1. Quantize accel/gyro through ADC (N-bit, full-scale range)
%   2. Pack into integer codes (int16)
%   3. Assemble UART packet: [header, accel(3), gyro(3), checksum]
%   4. Checksum = XOR of all payload bytes (each code treated as 2 bytes)
%
% The ESP32's imu_reader.m will unpack this packet by:
%   - Verifying header and checksum
%   - Converting integer codes back to float using the same scale factors
%
% Input:
%   accel  [3x1] accelerometer data (m/s^2) from imu_model
%   gyro   [3x1] gyroscope data (rad/s) from imu_model
%   params [struct] with .imu_adc_bits, .imu_accel_range, .imu_gyro_range
% Output:
%   packet [struct] UART-style packet:
%     .header     [uint8]  0xAA start byte
%     .accel_raw  [3x1 int] ADC codes for accelerometer
%     .gyro_raw   [3x1 int] ADC codes for gyroscope
%     .accel_scale [scalar] m/s^2 per LSB (for decoder)
%     .gyro_scale  [scalar] rad/s per LSB (for decoder)
%     .checksum   [int32]  XOR checksum of all payload codes

    adc_bits   = params.imu_adc_bits;
    accel_fs   = params.imu_accel_range;   % full-scale ± (m/s^2)
    gyro_fs    = params.imu_gyro_range;    % full-scale ± (rad/s)
    n_levels   = 2^adc_bits;

    % --- ADC quantization ---
    % Scale factor: physical units per LSB
    accel_lsb = (2 * accel_fs) / n_levels;   % m/s^2 per code
    gyro_lsb  = (2 * gyro_fs)  / n_levels;   % rad/s per code

    % Clamp + quantize accelerometer
    accel_clamped = max(-accel_fs, min(accel_fs, accel));
    accel_code    = round(accel_clamped / accel_lsb);
    accel_code    = int32(max(-n_levels/2, min(n_levels/2 - 1, accel_code)));

    % Clamp + quantize gyroscope
    gyro_clamped = max(-gyro_fs, min(gyro_fs, gyro));
    gyro_code    = round(gyro_clamped / gyro_lsb);
    gyro_code    = int32(max(-n_levels/2, min(n_levels/2 - 1, gyro_code)));

    % --- Checksum (XOR of all 6 payload codes) ---
    all_codes = [accel_code; gyro_code];
    chk = int32(0);
    for i = 1:6
        chk = bitxor(chk, all_codes(i));
    end

    % --- Assemble packet ---
    packet.header      = uint8(hex2dec('AA'));
    packet.accel_raw   = accel_code;
    packet.gyro_raw    = gyro_code;
    packet.accel_scale = accel_lsb;
    packet.gyro_scale  = gyro_lsb;
    packet.checksum    = chk;

end
