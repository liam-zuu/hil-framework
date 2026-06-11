function s04_circle_R20()
% S04 — Circle tracking, R=2.0m, period=60s
% Purpose: Very low curvature, long simulation. Check for drift over time.
% Expected: Both SS < 10 mm, but heading integration drift may become visible.

    %scenario_setup_paths();
    params = params_mecanum();

    spec = struct();
    spec.type   = 'circle';
    spec.R      = 2.0;
    spec.period = 60;
    spec.t_hold = 0.5;
    spec.t_ramp = 2.5;

    T_sim = 65;
    seed  = 42;
    run_scenario(mfilename, spec, params, T_sim, seed);
end
