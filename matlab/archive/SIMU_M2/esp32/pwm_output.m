function pwm_signal = pwm_output(tau_cmd, params)
% PWM_OUTPUT  Convert torque command to PWM duty cycle with saturation (STUB).

    % Normalize to [0, 1]
    pwm_signal = tau_cmd / params.tau_max;

    % Saturate
    pwm_signal = max(-1, min(1, pwm_signal));

end
