function s06_rounded_square()
% S06 — Rounded-corner square
% Purpose: Same shape as S05 but corners smoothed. Comparison shows impact
%          of trajectory smoothness on tracking error.
% Expected: Much lower error at corners vs S05. Both controllers benefit.

   % scenario_setup_paths();
    params = params_mecanum();

    spec = struct();
    spec.type     = 'rounded_square';
    spec.side     = 1.0;
    spec.R_corner = 0.2;
    spec.v        = 0.3;
    spec.t_hold   = 0.5;
    spec.t_ramp   = 2.5;

    T_sim = 20;
    seed  = 42;
    run_scenario(mfilename, spec, params, T_sim, seed);
end
