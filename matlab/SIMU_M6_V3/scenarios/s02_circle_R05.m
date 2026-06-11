function s02_circle_R05()
% S02 — Circle tracking, R=0.5m, period=20s
% Purpose: Low-curvature baseline for circular tracking.
% Expected: Both SS < 10 mm, smooth steady-state.

    %scenario_setup_paths();
    params = params_mecanum();

    spec = struct();
    spec.type   = 'circle';
    spec.R      = 0.5;
    spec.period = 20;
    spec.t_hold = 0.5;
    spec.t_ramp = 2.5;

    T_sim = 25;
    seed  = 42;
    run_scenario(mfilename, spec, params, T_sim, seed);
end
