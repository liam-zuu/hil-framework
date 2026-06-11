function omega_est = encoder_reader(enc_counts, dt, params)
% ENCODER_READER  Decode encoder counts to velocity (STUB).

    counts_per_rad = params.enc_ppr / (2*pi);
    omega_est = enc_counts / (dt * counts_per_rad);

end
