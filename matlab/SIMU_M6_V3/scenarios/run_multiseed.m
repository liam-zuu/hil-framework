function T = run_multiseed(seeds, out_dir, scenarios)
% RUN_MULTISEED  Run every scenario for several RNG seeds, metrics only.
%   T = run_multiseed(1:30, 'D:\...\run04')
%   T = run_multiseed(1:30, out_dir, {'s01_line_baseline','s14_industrial_nightmare'})
%
% Each scenario file keeps its own spec/params; only the seed is replaced
% (see the override note in run_scenario.m). PID and ADRC share the seed
% inside one run, so results are paired. Writes:
%   out_dir/raw/<scenario>_seedNNN.mat   per-run metrics (small)
%   out_dir/multiseed_long.csv           one row per scenario x seed x controller
% Does not write to results/ and does not save full logs.

    scenario_setup_paths();
    if nargin < 3 || isempty(scenarios)
        d = dir(fullfile(fileparts(mfilename('fullpath')), 's[0-9][0-9]_*.m'));
        scenarios = sort(erase({d.name}, '.m'));
    end
    raw_dir = fullfile(out_dir, 'raw');
    if ~exist(raw_dir, 'dir'), mkdir(raw_dir); end

    cleaner = onCleanup(@() rmappdata_safe());
    t0 = tic;
    for i = 1:numel(scenarios)
        for s = seeds
            f = fullfile(raw_dir, sprintf('%s_seed%03d.mat', scenarios{i}, s));
            if exist(f, 'file'), continue; end      % resumable
            setappdata(0, 'M6_RUN_OVERRIDE', struct('seed', s, 'out_dir', raw_dir));
            evalc('feval(scenarios{i})');            % silence per-run console output
            close all;
        end
        fprintf('[%2d/%d] %-26s done  (%.1f min elapsed)\n', i, numel(scenarios), scenarios{i}, toc(t0)/60);
    end

    %% Collect into a long table
    v1 = {'rms_pos_ss', 'max_pos_err', 'max_torque'};
    v2 = {'rms_w_err','rms_w_err_est','rms_tau','tv_tau','sat_pct', ...
          'rms_e_track','rms_e_ctrl','rms_e_drift','e_drift_end','drift_share'};
    rows = {};
    for i = 1:numel(scenarios)
        for s = seeds
            R = load(fullfile(raw_dir, sprintf('%s_seed%03d.mat', scenarios{i}, s)));
            for c = {'pid', 'adrc'}
                m1 = R.(['m_' c{1}]);  m2 = R.(['v2_' c{1}]);
                rows(end+1, :) = [scenarios(i), {s}, {upper(c{1})}, ...
                    cellfun(@(v) m1.(v), v1, 'UniformOutput', false), ...
                    cellfun(@(v) m2.(v), v2, 'UniformOutput', false)]; %#ok<AGROW>
            end
        end
    end
    T = cell2table(rows, 'VariableNames', [{'Scenario', 'Seed', 'Controller'}, v1, v2]);
    writetable(T, fullfile(out_dir, 'multiseed_long.csv'));
end

function rmappdata_safe()
    if isappdata(0, 'M6_RUN_OVERRIDE'), rmappdata(0, 'M6_RUN_OVERRIDE'); end
end
