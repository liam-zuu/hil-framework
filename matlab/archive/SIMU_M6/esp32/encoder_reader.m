function omega_est = encoder_reader(enc_counts, dt, params)
% ENCODER_READER  Decode encoder pulse counts to estimated wheel velocity.
%
% Two-stage pipeline:
%   1. Raw decode: enc_counts → omega_raw (rad/s)
%   2. First-order IIR low-pass filter to suppress quantization noise
%
% Filter: omega_filt(k) = alpha * omega_raw(k) + (1-alpha) * omega_filt(k-1)
%   where alpha = dt / (tau_f + dt), tau_f = params.enc_filter_tau
%
% At low speed (omega=5 rad/s), encoder SNR ≈ 6.4 dB without filtering.
% With tau_f = 5ms (alpha ≈ 0.167), effective ~5-sample smoothing,
% reduces noise while keeping phase lag acceptable for 1kHz control loop.
%
% Uses MATLAB persistent variable for filter state.
% Call "clear encoder_reader" between simulation runs to reset.
%
% Input:
%   enc_counts [4x1] encoder pulse counts (integer, signed) from encoder_pulse_gen
%   dt         [scalar] timestep (s)
%   params     [struct] with .enc_ppr, .enc_filter_tau
% Output:
%   omega_est  [4x1] filtered wheel angular velocity estimates (rad/s)

    persistent omega_prev;

    % Initialize on first call
    if isempty(omega_prev)
        omega_prev = zeros(4, 1);
    end

    %% Stage 1: Raw decode — counts to rad/s
    counts_per_rad = params.enc_ppr / (2*pi);
    omega_raw = enc_counts / (dt * counts_per_rad);

    %% Stage 2: First-order IIR low-pass filter
    tau_f = params.enc_filter_tau;
    alpha = dt / (tau_f + dt);

    omega_est = alpha * omega_raw + (1 - alpha) * omega_prev;

    % Store for next timestep
    omega_prev = omega_est;

end
