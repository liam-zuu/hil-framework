function s09_sinusoidal()
% S09 — Sinusoidal path (smooth wavy motion)
% Purpose: Continuous curvature variation, smoother than zigzag.
% Expected: Both controllers should handle well. ADRC may have slight edge.

    %scenario_setup_paths();
    params = params_mecanum();

    spec = struct();
    spec.type      = 'sinusoidal';
    spec.amplitude = 0.5;      % ±0.5m in y
    spec.frequency = 0.2;      % Hz (spatial)
    spec.v         = 0.3;      % forward speed
    spec.t_hold    = 0.5;
    spec.t_ramp    = 2.5;

    T_sim = 15;
    seed  = 42;
    run_scenario(mfilename, spec, params, T_sim, seed);
end
