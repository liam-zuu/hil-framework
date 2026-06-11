function packet = imu_packet_enc(accel, gyro, params)
% IMU_PACKET_ENC  Pack IMU data into UART-style packet (STUB).

    packet.header   = hex2dec('AA');
    packet.accel    = accel;   % STUB: no encoding
    packet.gyro     = gyro;
    packet.checksum = 0;       % STUB: dummy checksum

end
