function s10_racetrack()
% S10 — Racetrack (oval loop with 2 straights + 2 semicircles)
% Purpose: Mixed trajectory — straight-line stretches + constant curvature arcs.
%          Tests transition between motion modes.
% Expected: Small transient at straight→arc transitions.

    %scenario_setup_paths();
    params = params_mecanum();

    spec = struct();
    spec.type   = 'racetrack';
    spec.length = 3.0;         % total length (includes semicircles at ends)
    spec.width  = 1.0;         % width = diameter of semicircles
    spec.v      = 0.3;
    spec.t_hold = 0.5;
    spec.t_ramp = 2.5;

    T_sim = 30;    % ~1 lap after ramp
    seed  = 42;
    run_scenario(mfilename, spec, params, T_sim, seed);
end
