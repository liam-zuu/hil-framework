function s05_square()
% S05 — Square trajectory (sharp corners)
% Purpose: Corner handling. Sharp 90° heading change = hard for controller.
% Expected: Large transient at each corner. ADRC should recover faster.

    %scenario_setup_paths();
    params = params_mecanum();

    spec = struct();
    spec.type   = 'square';
    spec.side   = 1.0;
    spec.v      = 0.3;
    spec.t_hold = 0.5;
    spec.t_ramp = 2.5;

    % Time to complete one lap = 4*side/v = 4*1/0.3 ≈ 13.3s
    T_sim = 20;
    seed  = 42;
    run_scenario(mfilename, spec, params, T_sim, seed);
end
