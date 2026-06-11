function [m_pid, m_adrc] = run_scenario(scenario_name, spec, params, T_sim, seed)
% RUN_SCENARIO Common runner for all scenario files
%
% Handles the boilerplate: setup paths, generate trajectory, run both
% controllers, plot, save outputs. Each scenario file just configures
% spec + fault flags and calls this.
%
% Inputs:
%   scenario_name : string (e.g. 's02_circle_R05')
%   spec          : trajectory spec struct
%   params        : parameter struct (already includes any fault.* fields)
%   T_sim         : simulation duration (s)
%   seed          : RNG seed for reproducibility
%
% Outputs:
%   m_pid, m_adrc : metrics structs (also saved to results/)

    fprintf('\n========================================\n');
    fprintf('Scenario: %s\n', scenario_name);
    fprintf('Trajectory: %s\n', spec.type);
    fprintf('T_sim = %.1fs, seed = %d\n', T_sim, seed);
    active = {};
    if isfield(params,'fault')
        faults = fieldnames(params.fault);
        for i = 1:length(faults)
            F = params.fault.(faults{i});
            if isstruct(F) && isfield(F,'enabled') && F.enabled
                active{end+1} = ['fault.' faults{i}]; %#ok<AGROW>
            end
        end
    end
    if isfield(params,'disturbance') && isstruct(params.disturbance) && ...
       isfield(params.disturbance,'enabled') && params.disturbance.enabled
        active{end+1} = ['disturbance(' params.disturbance.type ')'];
    end
    if isfield(params,'slip') && isstruct(params.slip) && ...
       isfield(params.slip,'enabled') && params.slip.enabled
        active{end+1} = 'slip';
    end
    if ~isempty(active)
        fprintf('Active faults: %s\n', strjoin(active, ', '));
    end
    fprintf('========================================\n');

    %% Generate trajectory
    traj = trajectory_generator_v2(spec, params.dt, T_sim);

    %% Run PID
    fprintf('Running PID ... ');
    t_start = tic;
    [m_pid, log_pid] = run_single_scenario_v2('pid', traj, params, seed);
    fprintf('done (%.1fs)\n', toc(t_start));

    %% Run ADRC (same seed for fair comparison)
    fprintf('Running ADRC... ');
    t_start = tic;
    [m_adrc, log_adrc] = run_single_scenario_v2('adrc', traj, params, seed);
    fprintf('done (%.1fs)\n', toc(t_start));

    %% Plot
    fig = plot_comparison(log_pid, log_adrc, traj, m_pid, m_adrc, scenario_name);

    %% Save outputs
    results_dir = fullfile(fileparts(mfilename('fullpath')), 'results');
    if ~exist(results_dir,'dir'), mkdir(results_dir); end

    % PNG
    png_path = fullfile(results_dir, [scenario_name '.png']);
    saveas(fig, png_path);
    fprintf('Saved: %s\n', png_path);

    % MAT
    mat_path = fullfile(results_dir, [scenario_name '.mat']);
    save(mat_path, 'm_pid', 'm_adrc', 'log_pid', 'log_adrc', 'traj', 'spec', 'params');
    fprintf('Saved: %s\n', mat_path);

    % TXT report
    txt_path = fullfile(results_dir, [scenario_name '.txt']);
    write_text_report(txt_path, scenario_name, spec, params, m_pid, m_adrc);
    fprintf('Saved: %s\n', txt_path);

    %% Console summary
    fprintf('\n--- Summary ---\n');
    fprintf('%-20s %10s %10s  %+s\n', 'Metric', 'PID', 'ADRC', 'Δ');
    fprintf('%-20s %10.2f %10.2f  %+.1f%%\n', 'RMS SS (mm)', ...
            m_pid.rms_pos_ss, m_adrc.rms_pos_ss, ...
            100*(m_pid.rms_pos_ss-m_adrc.rms_pos_ss)/max(m_pid.rms_pos_ss,1e-6));
    fprintf('%-20s %10.2f %10.2f\n', 'Peak err (mm)', ...
            m_pid.max_pos_err, m_adrc.max_pos_err);
    fprintf('%-20s %10.2f %10.2f\n', 'Max torque (N·m)', ...
            m_pid.max_torque, m_adrc.max_torque);
end

function write_text_report(path, name, spec, params, m_pid, m_adrc)
    fid = fopen(path, 'w');
    fprintf(fid, '================================================================\n');
    fprintf(fid, 'HIL SIMULATION — SCENARIO REPORT\n');
    fprintf(fid, '================================================================\n');
    fprintf(fid, 'Scenario: %s\n', name);
    fprintf(fid, 'Date: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, '\n');

    % Trajectory info
    fprintf(fid, '--- TRAJECTORY ---\n');
    fprintf(fid, 'Type: %s\n', spec.type);
    fprintf(fid, 'Hold: %.2fs, Ramp: %.2fs\n', spec.t_hold, spec.t_ramp);
    fns = fieldnames(spec);
    for i = 1:length(fns)
        if ~ismember(fns{i}, {'type','t_hold','t_ramp'})
            v = spec.(fns{i});
            if isnumeric(v) && isscalar(v)
                fprintf(fid, '%s: %g\n', fns{i}, v);
            end
        end
    end
    fprintf(fid, '\n');

    % Fault info
    has_fault = false;
    if isfield(params,'fault')
        faults = fieldnames(params.fault);
        for i = 1:length(faults)
            F = params.fault.(faults{i});
            if isstruct(F) && isfield(F,'enabled') && F.enabled
                if ~has_fault
                    fprintf(fid, '--- ACTIVE FAULTS ---\n');
                    has_fault = true;
                end
                fprintf(fid, 'fault.%s:\n', faults{i});
                sub = fieldnames(F);
                for j = 1:length(sub)
                    v = F.(sub{j});
                    if isnumeric(v) && isscalar(v)
                        fprintf(fid, '  %s: %g\n', sub{j}, v);
                    elseif ischar(v)
                        fprintf(fid, '  %s: %s\n', sub{j}, v);
                    end
                end
            end
        end
    end
    if isfield(params,'disturbance') && isstruct(params.disturbance) && ...
       isfield(params.disturbance,'enabled') && params.disturbance.enabled
        if ~has_fault
            fprintf(fid, '--- ACTIVE FAULTS ---\n');
            has_fault = true;
        end
        D = params.disturbance;
        fprintf(fid, 'disturbance (M6-style):\n');
        fprintf(fid, '  type: %s\n', D.type);
        fprintf(fid, '  magnitude: %g\n', D.magnitude);
        fprintf(fid, '  start_time: %g\n', D.start_time);
    end
    if isfield(params,'slip') && isstruct(params.slip) && ...
       isfield(params.slip,'enabled') && params.slip.enabled
        if ~has_fault
            fprintf(fid, '--- ACTIVE FAULTS ---\n');
            has_fault = true;
        end
        fprintf(fid, 'slip: mu_static=%.2f, mu_kinetic=%.2f, p_spont=%.3f\n', ...
                params.slip.mu_static, params.slip.mu_kinetic, params.slip.prob_spontaneous);
    end
    if ~has_fault
        fprintf(fid, '--- ACTIVE FAULTS ---\n(none)\n');
    end
    fprintf(fid, '\n');

    % Metrics
    fprintf(fid, '--- METRICS ---\n');
    fprintf(fid, '%-20s %12s %12s %12s\n', 'Metric', 'PID', 'ADRC', 'Delta');
    fprintf(fid, '%s\n', repmat('-', 1, 60));
    fn = {'rms_pos_full','rms_pos_ss','max_pos_err','t_peak',...
          'rms_theta','max_theta_err','max_torque','sat_pct', ...
          'slip_events','control_effort'};
    units = {'mm','mm','mm','s','deg','deg','N·m','%','count','N·m·s'};
    for i = 1:length(fn)
        vp = m_pid.(fn{i}); va = m_adrc.(fn{i});
        fprintf(fid, '%-20s %12.3f %12.3f %12.3f  [%s]\n', ...
                fn{i}, vp, va, vp-va, units{i});
    end
    ss_improvement = 100*(m_pid.rms_pos_ss - m_adrc.rms_pos_ss)/max(m_pid.rms_pos_ss,1e-6);
    fprintf(fid, '\nADRC SS improvement over PID: %+.1f%%\n', ss_improvement);
    fprintf(fid, '\n================================================================\n');
    fclose(fid);
end
