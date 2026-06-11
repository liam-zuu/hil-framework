function pwm_signal = pwm_output(tau_cmd, params)
% PWM_OUTPUT  Convert torque command to PWM duty cycle.
%
% Two-stage pipeline:
%   1. Normalize: tau_cmd → raw PWM [-1, +1]
%   2. Deadband compensation: remap nonzero commands to [deadband, 1]
%      so that after pwm_capture applies its deadband, the motor
%      receives the intended torque
%   3. Saturate to [-1, +1]
%
% Compensation logic:
%   pwm_capture kills |pwm| < deadband (0.02). Without compensation,
%   controller commands below deadband*tau_max produce zero torque.
%   Compensation maps the controller's [0, 1] range to [deadband, 1],
%   ensuring any nonzero command reaches the motor.
%
% Input:
%   tau_cmd    [4x1] torque command from controller (N·m)
%   params     [struct] with .tau_max, .deadband
% Output:
%   pwm_signal [4x1] signed PWM duty cycle [-1, +1]

    tau_max  = params.tau_max;
    deadband = params.deadband;

    %% Stage 1: Normalize to [-1, +1]
    pwm_raw = tau_cmd / tau_max;

    %% Stage 2: Deadband compensation
    % Map [0, 1] → [deadband, 1] for nonzero commands
    % pwm_comp = sign(pwm) * (|pwm| * (1 - deadband) + deadband)
    pwm_signal = zeros(4, 1);
    for i = 1:4
        if abs(pwm_raw(i)) > 1e-6
            pwm_signal(i) = sign(pwm_raw(i)) * ...
                (abs(pwm_raw(i)) * (1 - deadband) + deadband);
        else
            pwm_signal(i) = 0;
        end
    end

    %% Stage 3: Saturate
    pwm_signal = max(-1, min(1, pwm_signal));

end
