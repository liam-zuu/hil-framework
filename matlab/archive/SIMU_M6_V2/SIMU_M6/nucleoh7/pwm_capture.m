function tau = pwm_capture(pwm_signal, params)
% PWM_CAPTURE  Convert signed PWM duty cycle to torque command.
%
% Models the H7 timer-capture peripheral reading PWM from ESP32:
%   1. Add timing jitter (capture resolution uncertainty)
%   2. Quantize to PWM timer resolution (params.pwm_res levels)
%   3. Apply deadband: |pwm| < deadband -> 0 (motor does not move)
%   4. Linear map: pwm -> tau = pwm * tau_max
%
% Deadband models real H-bridge behavior where small duty cycles
% produce no meaningful current through the motor.
%
% Input:
%   pwm_signal [4x1] signed PWM duty cycle [-1, +1] from ESP32
%   params     [struct] with .tau_max, .deadband, .pwm_res, .pwm_jitter_sigma
% Output:
%   tau        [4x1] reconstructed torque command (N·m), signed

    tau_max   = params.tau_max;
    deadband  = params.deadband;
    pwm_res   = params.pwm_res;
    jitter_sigma = params.pwm_jitter_sigma;

    % --- Step 1: Capture timing jitter ---
    % Small Gaussian noise on the measured duty cycle
    pwm = pwm_signal + jitter_sigma * randn(4, 1);

    % Re-clamp after jitter (can't exceed ±1)
    pwm = max(-1, min(1, pwm));

    % --- Step 2: Quantize to PWM timer resolution ---
    % Timer counts: 0 to pwm_res. Duty cycle = count / pwm_res
    % For signed: map [-1,+1] to [-pwm_res, +pwm_res] integer codes
    pwm_code = round(pwm * pwm_res);
    pwm = pwm_code / pwm_res;

    % --- Step 3: Apply deadband ---
    % Motor controller ignores duty cycles below deadband threshold
    for i = 1:4
        if abs(pwm(i)) < deadband
            pwm(i) = 0;
        end
    end

    % --- Step 4: Linear map to torque ---
    tau = pwm * tau_max;

end
