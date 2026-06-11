function results = run_all_scenarios(varargin)
% RUN_ALL_SCENARIOS Execute all scenario files sequentially, collect results
%
% Usage:
%   run_all_scenarios()              % run everything
%   run_all_scenarios('skip_slow')   % skip long scenarios (s04, s10)
%   run_all_scenarios('tracking')    % only s01-s10 (no faults)
%   run_all_scenarios('faults')      % only s11-s14 (fault scenarios)
%
% Returns struct with metrics for every scenario. Also triggers
% generate_summary_report() at the end.

    mode = 'all';
    if nargin >= 1, mode = varargin{1}; end

    scenario_setup_paths();

    % Scenario list
    all_scenarios = { ...
        's01_line_baseline', ...
        's02_circle_R05', ...
        's03_circle_R10', ...
        's04_circle_R20', ...
        's05_square', ...
        's06_rounded_square', ...
        's07_figure8', ...
        's08_zigzag', ...
        's09_sinusoidal', ...
        's10_racetrack', ...
        's11_wheel_jam', ...
        's12_encoder_dropout', ...
        's13_battery_fade', ...
        's14_industrial_nightmare'...
        's15_oily_floor'};

    slow_scenarios = {'s04_circle_R20', 's10_racetrack'};
    tracking_only  = all_scenarios(1:10);
    faults_only    = all_scenarios(11:14);

    switch lower(mode)
        case 'all',       list = all_scenarios;
        case 'skip_slow', list = setdiff(all_scenarios, slow_scenarios, 'stable');
        case 'tracking',  list = tracking_only;
        case 'faults',    list = faults_only;
        otherwise,        list = all_scenarios;
    end

    results = struct();
    t_total_start = tic;

    for i = 1:length(list)
        name = list{i};
        fprintf('\n\n############ [%d/%d] %s ############\n', ...
                i, length(list), name);
        t_scenario_start = tic;

        try
            % Each scenario file is self-contained and calls run_scenario(),
            % which saves outputs. We capture metrics from the saved .mat file.
            feval(name);
            close all;

            % Load metrics for summary
            mat_path = fullfile(fileparts(mfilename('fullpath')), 'results', [name '.mat']);
            if exist(mat_path, 'file')
                S = load(mat_path, 'm_pid', 'm_adrc');
                results.(name).m_pid  = S.m_pid;
                results.(name).m_adrc = S.m_adrc;
                results.(name).elapsed_s = toc(t_scenario_start);
                results.(name).status = 'ok';
            else
                results.(name).status = 'no_output';
            end
        catch ME
            fprintf(2, 'ERROR in %s: %s\n', name, ME.message);
            results.(name).status = 'error';
            results.(name).error_msg = ME.message;
        end
    end

    t_total = toc(t_total_start);
    fprintf('\n\n==============================================\n');
    fprintf('All scenarios complete. Total time: %.1f min\n', t_total/60);
    fprintf('==============================================\n');

    % Save collated results
    results_dir = fullfile(fileparts(mfilename('fullpath')), 'results');
    save(fullfile(results_dir, 'all_results.mat'), 'results');

    % Generate summary report
    try
        generate_summary_report(results);
    catch ME
        fprintf(2, 'Summary report generation failed: %s\n', ME.message);
    end
end
