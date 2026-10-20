function T = postprocess_v2(in_dir, out_dir)
% POSTPROCESS_V2  Recompute v2 metrics and figures from saved scenario .mat files.
%   T = postprocess_v2(in_dir, out_dir)
%   in_dir  : folder with sXX_*.mat written by run_scenario.m (not modified)
%   out_dir : new folder for summary_v2.csv, sXX_*_v2.png, summary_v2.png
% Does not re-run the simulation.

    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    files = dir(fullfile(in_dir, 's*_*.mat'));
    names = sort(erase({files.name}, '.mat'));

    vars = {'rms_w_err','rms_w_err_est','rms_tau','tv_tau','sat_pct', ...
            'rms_e_track','rms_e_ctrl','rms_e_drift','e_drift_end','drift_share'};
    rows = {};
    for i = 1:numel(names)
        S  = load(fullfile(in_dir, [names{i} '.mat']), 'log_pid', 'log_adrc', 'traj', 'params');
        mp = metrics_compute_v2(S.log_pid,  S.traj, S.params);
        ma = metrics_compute_v2(S.log_adrc, S.traj, S.params);
        fig = plot_scenario_v2(names{i}, S, mp, ma);
        exportgraphics(fig, fullfile(out_dir, [names{i} '_v2.png']), 'Resolution', 150);
        close(fig);
        rows(end+1, :) = [names(i), cellfun(@(v) mp.(v), vars, 'UniformOutput', false), ...
                                    cellfun(@(v) ma.(v), vars, 'UniformOutput', false)]; %#ok<AGROW>
        fprintf('[%2d/%d] %s\n', i, numel(names), names{i});
    end
    T = cell2table(rows, 'VariableNames', ['Scenario', strcat('PID_', vars), strcat('ADRC_', vars)]);
    writetable(T, fullfile(out_dir, 'summary_v2.csv'));

    fig = plot_summary_v2(T);
    exportgraphics(fig, fullfile(out_dir, 'summary_v2.png'), 'Resolution', 150);
    close(fig);
end
