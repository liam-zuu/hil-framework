function s08_zigzag()
% S08 — Zigzag (triangle wave path)
% Purpose: Frequent heading reversal, tests controller response to
%          direction changes.
%
% Params tuned so heading change per vertex ≈ 17° (was 50° — too aggressive,
% both controllers saturated 20%+ and hit 100° heading error).
%   atan(4*A/λ) = atan(4*0.15/2.0) = atan(0.3) ≈ 17°
%
% Expected: ADRC handles direction changes better via ESO disturbance
%           estimation. Both SS < 30mm.
 
    % scenario_setup_paths();
    params = params_mecanum();
 
    spec = struct();
    spec.type       = 'zigzag';
    spec.amplitude  = 0.15;      % ±0.15m in y (was 0.3 — halved)
    spec.wavelength = 2.0;       % 2m per cycle in x (was 1.0 — doubled)
    spec.v          = 0.3;
    spec.t_hold     = 0.5;
    spec.t_ramp     = 2.5;
 
    T_sim = 20;    % longer to get more zigzag cycles in SS
    seed  = 42;
    run_scenario(mfilename, spec, params, T_sim, seed);
end