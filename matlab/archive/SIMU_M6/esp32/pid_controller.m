function [tau_cmd, pid_state] = pid_controller(omega_ref, omega_est, pid_state, params)
% PID_CONTROLLER  Per-wheel velocity PID with conditional integration anti-windup.
%
% Standard PID: tau = Kp*e + Ki*integral(e) + Kd*de/dt
%
% Anti-windup: conditional integration.
%   - When output is NOT saturated: normal integration
%   - When output IS saturated: only integrate if error would REDUCE
%     the integral (i.e., error sign opposes integral sign)
%   - This prevents integral from growing during saturation while
%     allowing it to unwind, so the controller can exit saturation
%     and enter the linear region as soon as the error decreases.
%
% Derivative uses backward difference on error. Encoder reader's
% low-pass filter reduces noise before it reaches here.
%
% Input:
%   omega_ref  [4x1] reference wheel velocities (rad/s)
%   omega_est  [4x1] estimated wheel velocities (rad/s) from encoder_reader
%   pid_state  [struct or []] controller state (.integral, .prev_error)
%   params     [struct] with .pid.Kp, .pid.Ki, .pid.Kd, .tau_max, .dt
% Output:
%   tau_cmd    [4x1] torque command (N·m), NOT clamped (pwm_output handles)
%   pid_state  [struct] updated controller state

    %% Initialize state on first call
    if isempty(pid_state)
        pid_state.integral   = zeros(4, 1);
        pid_state.prev_error = zeros(4, 1);
    end

    dt  = params.dt;
    Kp  = params.pid.Kp;
    Ki  = params.pid.Ki;
    Kd  = params.pid.Kd;
    tau_max = params.tau_max;

    %% Error
    e = omega_ref - omega_est;

    %% Proportional
    P = Kp * e;

    %% Derivative (backward difference on error)
    D = Kd * (e - pid_state.prev_error) / dt;
    pid_state.prev_error = e;

    %% Tentative integral update
    int_new = pid_state.integral + e * dt;

    %% Tentative output (to check saturation)
    tau_tent = P + Ki * int_new + D;

    %% Conditional integration anti-windup
    for i = 1:4
        saturated  = abs(tau_tent(i)) >= tau_max;
        would_grow = sign(e(i)) == sign(pid_state.integral(i));

        if ~saturated || ~would_grow
            % Not saturated, OR error would reduce integral → allow integration
            pid_state.integral(i) = int_new(i);
        end
        % else: keep old integral (freeze)
    end

    %% Safety clamp (backup, should rarely activate)
    int_max = tau_max / Ki;
    pid_state.integral = max(-int_max, min(int_max, pid_state.integral));

    %% Final output
    I = Ki * pid_state.integral;
    tau_cmd = P + I + D;

end
