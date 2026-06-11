function s14_industrial_nightmare()
% S14 — "The Industrial Nightmare" — compound fault stress test
%
% Sequential (approximately) fault onset:
%   t=4   : mass bias (constant drag torque, simulates payload)
%   t=8   : battery fade begins (tau_max 0.5 → 0.35 N·m)
%   t=12  : combined load disturbance (step+ramp+random)
%   throughout: wheel slip active (can't time-gate plant), PPR=512
%
% See nightmare_config.m for exact parameters and HONEST LIMITATIONS note.
%
% Expected outcome:
%   - PID: collapses under compound faults (50+ mm SS)
%   - ADRC: maintains tracking (ESO aggregates all disturbances into z2)
%
% This is the headline scenario for thesis Chapter 5.

    %scenario_setup_paths();
    params = params_mecanum();
    cfg    = nightmare_config();

    % Apply config to params
    params.slip         = cfg.slip;
    params.disturbance  = cfg.disturbance;
    params.fault.mass_bias    = cfg.mass_bias;
    params.fault.battery_fade = cfg.battery_fade;
    params.enc_ppr            = cfg.enc_ppr_override;

    % Print banner
    fprintf('\n### INDUSTRIAL NIGHTMARE MODE ###\n');
    fprintf('  t=0 : encoder PPR=%d (throughout)\n', cfg.enc_ppr_override);
    fprintf('  t=0 : wheel slip ACTIVE (throughout)\n');
    fprintf('  t=%.1fs: mass bias onset (+%.3f N·m drag)\n', ...
            cfg.mass_bias.t_start, cfg.mass_bias.tau_bias);
    fprintf('  t=%.1fs: battery fade begins (tau_max %.2f→%.2f)\n', ...
            cfg.battery_fade.t_start, ...
            cfg.battery_fade.tau_max_nominal, cfg.battery_fade.tau_max_final);
    fprintf('  t=%.1fs: load disturbance starts (%s)\n', ...
            cfg.disturbance.start_time, cfg.disturbance.type);

    % Run with standard runner
    [m_pid, m_adrc] = run_scenario(mfilename, cfg.spec, params, cfg.T_sim, cfg.seed);

    % Add fault onset markers to the error plot
    % (run_scenario already saved/closed the figure; open it back up to annotate)
    results_dir = fullfile(fileparts(mfilename('fullpath')), 'results');
    fig_path = fullfile(results_dir, [mfilename '.fig']);

    % The plot has been saved as PNG already. For now, print final summary:
    fprintf('\n--- NIGHTMARE RESULT ---\n');
    fprintf('PID  SS: %6.2f mm  (peak %6.2f mm)\n', m_pid.rms_pos_ss,  m_pid.max_pos_err);
    fprintf('ADRC SS: %6.2f mm  (peak %6.2f mm)\n', m_adrc.rms_pos_ss, m_adrc.max_pos_err);
    diff_pct = 100*(m_pid.rms_pos_ss - m_adrc.rms_pos_ss) / max(m_pid.rms_pos_ss, 1e-6);
    fprintf('ADRC wins by: %+.1f%%\n', diff_pct);
end
