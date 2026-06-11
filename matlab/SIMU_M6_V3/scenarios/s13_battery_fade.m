function s13_battery_fade()
% S13 — Battery fade / actuator degradation
% Purpose: Simulate battery voltage drop causing tau_max to shrink linearly
%          from 0.5 N·m to 0.3 N·m between t=3s and t=10s. Tests anti-windup
%          behavior when actuator saturation limit decreases over time.
% Expected:
%   - Without anti-windup: PID integral keeps growing → position jump when
%     saturation releases (integral "dump")
%   - With conditional integration anti-windup (M4): integral freezes when
%     saturated with same-sign error → graceful degradation
%   - ADRC: ESO z2 absorbs the saturation as disturbance → smoother

    %scenario_setup_paths();
    params = params_mecanum();

    % Use figure-8 — high demand, likely to saturate at reduced tau_max
    spec = struct();
    spec.type   = 'figure8';
    spec.size   = 1.0;
    spec.period = 8.0;
    spec.t_hold = 0.5;
    spec.t_ramp = 2.5;

    % Fault: linear tau_max fade 0.5 → 0.3 N·m from t=3s to t=10s
    params.fault.battery_fade = struct( ...
        'enabled',         true, ...
        't_start',         3.0, ...
        't_end',           10.0, ...
        'tau_max_nominal', 0.5, ...
        'tau_max_final',   0.3);

    T_sim = 18;
    seed  = 42;
    run_scenario(mfilename, spec, params, T_sim, seed);
end
