function cfg = nightmare_config()
% NIGHTMARE_CONFIG  Parameter configuration for s14_industrial_nightmare
%
% Defines the compound fault timeline for the ultimate stress test.
% Separated from scenario file for easy parameter tuning.
%
% HONEST LIMITATIONS (sequential onset is approximate, not exact):
%   - Battery fade + wheel jam + mass bias + load disturbance honor t_start
%     (they're gated by time inside run_single_scenario_v2 fault hooks)
%   - Wheel slip is enabled THROUGHOUT (plant_step.m cannot be time-gated
%     without modifying plant code). Spontaneous slip probability is kept
%     low so early-phase impact is minimal. Main slip contribution comes
%     from high-torque events (which naturally happen later under compound
%     load).
%   - Encoder PPR reduction is applied FROM t=0 (cannot change mid-sim
%     without phased runner). Documented in scenario comment.
%
% Timeline (approximate):
%   t = 0     : idle hold (ramp phase)
%   t = 2.5   : cruise begins
%   t = 4     : mass bias activates (payload drag)
%   t = 6     : (slip probabilistic — always-on, but torques now higher)
%   t = 8     : battery fade begins (tau_max shrinks linearly)
%   t = 12    : combined load disturbance adds on top
%   t = 18    : end
%
% Despite these approximations, the scenario demonstrates:
%   - PID degrades heavily under compound faults (target: > 50mm SS)
%   - ADRC maintains tracking (target: < 15mm SS)

    cfg = struct();

    %% Global
    cfg.T_sim = 18;
    cfg.seed  = 42;

    %% Trajectory
    cfg.spec = struct( ...
        'type',   'circle', ...
        'R',      1.0, ...
        'period', 20, ...
        't_hold', 0.5, ...
        't_ramp', 2.0);

    %% Fault config — honor t_start for faults that support time-gating

    % Mass bias (constant drag torque after t_start)
    cfg.mass_bias = struct( ...
        'enabled',    true, ...
        't_start',    4.0, ...
        'tau_bias',   0.04);    % N·m per wheel (~8% tau_max)

    % Wheel slip — applied throughout; see HONEST LIMITATIONS above.
    cfg.slip = struct( ...
        'enabled',          true, ...
        'mu_static',        0.6, ...     % reduced from 0.8 (wet/slippery)
        'mu_kinetic',       0.35, ...
        'prob_spontaneous', 0.002, ...   % moderate, not aggressive
        'noise_sigma',      0.20, ...
        'detect_threshold', 0.15, ...    % keep default thresholds
        'imu_wz_threshold', 0.5);

    % Battery fade
    cfg.battery_fade = struct( ...
        'enabled',         true, ...
        't_start',         8.0, ...
        't_end',           18.0, ...
        'tau_max_nominal', 0.5, ...
        'tau_max_final',   0.35);

    % Combined load disturbance (M6 style — uses params.disturbance.*)
    cfg.disturbance = struct( ...
        'enabled',      true, ...
        'type',         'combined', ...
        'start_time',   12.0, ...
        'magnitude',    0.06, ...
        'ramp_rate',    0.01, ...
        'random_sigma', 0.04);

    % PPR degradation (applied from t=0, see limitation)
    cfg.enc_ppr_override = 512;   % drop from 1024
end
