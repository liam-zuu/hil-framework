function s11_wheel_jam()
% S11 — Wheel jam fault (stuck bearing / wrapped debris)
% Purpose: Simulate mechanical stiction on wheel 1 — sudden viscous
%          friction increase at t=3.5s.
%
% Friction strength: b_extra = 50 × b_w = 0.1 N·m·s/rad
%   At ω=5 rad/s: jam torque = 0.5 N·m = 100% of tau_max
%   → PID integral cannot compensate fast enough
%   → ADRC ESO should detect velocity drop and increase z2 rapidly
%
% Previous version used 5×b_w = 0.01 → jam torque only 0.05 N·m (10%)
% → both controllers handled it trivially (+1.5% difference).
%
% Expected: ADRC significantly outperforms PID (>20% improvement).

    %scenario_setup_paths();
    params = params_mecanum();

    spec = struct();
    spec.type   = 'circle';
    spec.R      = 1.0;
    spec.period = 30;
    spec.t_hold = 0.5;
    spec.t_ramp = 2.5;

    % Fault: wheel 1 friction jumps ×50 at t=3.5s
    params.fault.wheel_jam = struct( ...
        'enabled', true, ...
        'wheel',   1, ...
        'b_extra',50 * params.b_w, ...   % 0.1 N·m·s/rad (was 5×)
        't_start', 3.5);

    T_sim = 30;
    seed  = 42;
    run_scenario(mfilename, spec, params, T_sim, seed);
end