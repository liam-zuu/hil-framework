function [tau_cmd, adrc_state] = adrc_controller(omega_ref, omega_est, imu_data, adrc_state, params)
% ADRC_CONTROLLER  ESO-based ADRC for each wheel (STUB).

    if isempty(adrc_state)
        adrc_state.z1 = zeros(4,1);  % ESO state 1 (est. output)
        adrc_state.z2 = zeros(4,1);  % ESO state 2 (est. derivative)
        adrc_state.z3 = zeros(4,1);  % ESO state 3 (est. disturbance)
    end

    % STUB: same as P-controller
    e = omega_ref - omega_est;
    tau_cmd = params.pid.Kp * e;  % placeholder, uses PID gains

end
