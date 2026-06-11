function s03_circle_R10()
% S03 — Circle tracking, R=1.0m, period=30s
% Purpose: Baseline circular scenario used across M5-M6 tuning.
% Expected: Matches M5.2 baseline (~7mm SS).

    %scenario_setup_paths();
    params = params_mecanum();

    spec = struct();
    spec.type   = 'circle';
    spec.R      = 1.0;
    spec.period = 30;
    spec.t_hold = 0.5;
    spec.t_ramp = 2.5;

    T_sim = 35;
    seed  = 42;
    run_scenario(mfilename, spec, params, T_sim, seed);
end
