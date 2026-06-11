function s01_line_baseline()
% S01 — Line tracking baseline
% Purpose: Simplest tracking scenario. Establishes noise floor for both controllers.
% Expected: Both PID and ADRC SS < 10 mm. Nearly tied (no disturbance).

    % scenario_setup_paths();
    params = params_mecanum();

    spec = struct();
    spec.type   = 'line';
    spec.v      = 0.3;       % m/s
    spec.t_hold = 0.5;
    spec.t_ramp = 2.5;

    T_sim = 10;
    seed  = 42;
    run_scenario(mfilename, spec, params, T_sim, seed);
end
