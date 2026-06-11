function tau = pwm_capture(pwm_signal, params)
% PWM_CAPTURE  Convert signed PWM duty cycle [-1,+1] to torque (STUB).

    % STUB: linear mapping, no deadband
    tau = pwm_signal * params.tau_max;  % signed: [-tau_max, +tau_max]

end
