function enc_counts = encoder_pulse_gen(omega, dt, params)
% ENCODER_PULSE_GEN  Convert wheel angular velocity to encoder pulse counts.
%
% Models a real incremental encoder:
%   1. Angle traveled per timestep: delta_angle = omega * dt
%   2. Convert to fractional counts: frac = delta_angle * PPR / (2*pi)
%   3. Accumulate fractional counts (carry-over between timesteps)
%   4. Output = integer part of accumulator (floor, preserving sign)
%   5. Add Gaussian noise scaled by enc_noise_sigma
%
% The fractional accumulator is critical at low speeds where
% omega*dt < 1 pulse — without it, many timesteps would read 0
% and the encoder_reader would estimate zero velocity.
%
% Uses MATLAB persistent variables for accumulator state.
% Call "clear encoder_pulse_gen" between simulation runs to reset.
%
% Input:
%   omega  [4x1] wheel angular velocities (rad/s) from plant state
%   dt     [scalar] timestep (s)
%   params [struct] with .enc_ppr, .enc_noise_sigma
% Output:
%   enc_counts [4x1] encoder pulse counts (integer, signed)

    persistent accum;

    % Initialize accumulator on first call
    if isempty(accum)
        accum = zeros(4, 1);
    end

    ppr   = params.enc_ppr;
    sigma = params.enc_noise_sigma;

    % Fractional counts this timestep
    counts_per_rad = ppr / (2*pi);
    frac_counts = omega * dt * counts_per_rad;

    % Accumulate
    accum = accum + frac_counts;

    % Extract integer part (fix = truncate toward zero)
    int_counts = fix(accum);

    % Keep fractional remainder in accumulator
    accum = accum - int_counts;

    % Add measurement noise (Gaussian, sigma in units of counts)
    noise = sigma * randn(4, 1);

    % Output: integer counts + noise, then round to integer
    enc_counts = round(int_counts + noise);

end
