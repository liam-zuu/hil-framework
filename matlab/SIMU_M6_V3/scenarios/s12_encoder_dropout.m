function s12_encoder_dropout()
% S12 — Encoder signal dropout
% Purpose: Simulate loose encoder connector on wheel 2. Signal reads 0 for
%          500ms despite wheel actually rotating.
% Expected:
%   - PID: sees omega_2 = 0 while ref != 0 → large error → huge tau spike
%          → when signal returns, wheel over-speeds → oscillation
%   - ADRC: ESO z1 estimate smooths glitch somewhat, but large z2 injection
%          also possible → depends on ESO bandwidth
%   - slip_detector: should flag (kinematic inconsistency: 3 wheels moving,
%          1 wheel at 0 doesn't match any valid body motion)

    %scenario_setup_paths();
    params = params_mecanum();

    spec = struct();
    spec.type   = 'circle';
    spec.R      = 1.0;
    spec.period = 30;
    spec.t_hold = 0.5;
    spec.t_ramp = 2.5;

    % Fault: wheel 2 encoder dropout from t=4s to t=4.5s
    params.fault.enc_dropout = struct( ...
        'enabled', true, ...
        'wheel',   2, ...
        't_start', 4.0, ...
        't_end',   4.5);

    T_sim = 33;
    seed  = 42;
    run_scenario(mfilename, spec, params, T_sim, seed);
end
