function s07_figure8()
% S07 — Figure-8 (M5.2 benchmark)
% Purpose: High-demand trajectory with continuous curvature change.
% Expected: ADRC tốt hơn PID do ESO estimate disturbance từ centripetal coupling.

    %scenario_setup_paths();
    params = params_mecanum();

    spec = struct();
    spec.type   = 'figure8';
    spec.size   = 1.0;    % amplitude (m)
    spec.period = 8.0;    % period (s) — after M5.2 fix from 5s to 8s
    spec.t_hold = 0.5;
    spec.t_ramp = 2.5;

    T_sim = 18;    % ~2 full figure-8 cycles after ramp
    seed  = 42;
    run_scenario(mfilename, spec, params, T_sim, seed);
end
