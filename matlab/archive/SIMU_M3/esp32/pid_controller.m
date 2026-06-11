function [tau_cmd, pid_state] = pid_controller(omega_ref, omega_est, pid_state, params)
% PID_CONTROLLER  Velocity PID for each wheel (STUB).

    if isempty(pid_state)
        pid_state.integral   = zeros(4,1);
        pid_state.prev_error = zeros(4,1);
    end

    % STUB: proportional only
    e = omega_ref - omega_est;
    tau_cmd = params.pid.Kp * e;

    pid_state.prev_error = e;

end
