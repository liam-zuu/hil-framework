function enc_counts = encoder_pulse_gen(omega, dt, params)
% ENCODER_PULSE_GEN  Convert wheel speeds to encoder counts (STUB).

    % STUB: ideal conversion, no quantization or noise
    counts_per_rad = params.enc_ppr / (2*pi);
    enc_counts = round(omega * dt * counts_per_rad);

end
