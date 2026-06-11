function generate_summary_report(results)
% GENERATE_SUMMARY_REPORT Create summary outputs from all_scenarios results
%
% Produces:
%   results/summary_table.txt   — plain text ranked table
%   results/summary_table.csv   — CSV for Excel
%   results/summary_report.pdf  — composite PDF with summary table + all scenario plots
%
% Input: results struct from run_all_scenarios()

    if nargin < 1
        % Try loading from disk
        mat_path = fullfile(fileparts(mfilename('fullpath')), 'results', 'all_results.mat');
        if exist(mat_path,'file')
            S = load(mat_path);
            results = S.results;
        else
            error('No results provided and no all_results.mat found.');
        end
    end

    results_dir = fullfile(fileparts(mfilename('fullpath')), 'results');
    if ~exist(results_dir,'dir'), mkdir(results_dir); end

    %% Collate into arrays
    names = fieldnames(results);
    ok_mask = false(size(names));
    for i = 1:length(names)
        if isfield(results.(names{i}),'status') && ...
           strcmp(results.(names{i}).status, 'ok')
            ok_mask(i) = true;
        end
    end
    names = names(ok_mask);
    n = length(names);

    if n == 0
        warning('No successful scenarios to summarize.');
        return;
    end

    pid_rms_ss   = zeros(n,1);
    adrc_rms_ss  = zeros(n,1);
    pid_rms_full = zeros(n,1);
    adrc_rms_full= zeros(n,1);
    pid_peak     = zeros(n,1);
    adrc_peak    = zeros(n,1);
    pid_tau      = zeros(n,1);
    adrc_tau     = zeros(n,1);
    improvement  = zeros(n,1);
    for i = 1:n
        r = results.(names{i});
        pid_rms_ss(i)    = r.m_pid.rms_pos_ss;
        adrc_rms_ss(i)   = r.m_adrc.rms_pos_ss;
        pid_rms_full(i)  = r.m_pid.rms_pos_full;
        adrc_rms_full(i) = r.m_adrc.rms_pos_full;
        pid_peak(i)      = r.m_pid.max_pos_err;
        adrc_peak(i)     = r.m_adrc.max_pos_err;
        pid_tau(i)       = r.m_pid.max_torque;
        adrc_tau(i)      = r.m_adrc.max_torque;
        improvement(i)   = 100*(pid_rms_ss(i)-adrc_rms_ss(i))/max(pid_rms_ss(i),1e-6);
    end

    %% Text summary table
    txt_path = fullfile(results_dir, 'summary_table.txt');
    fid = fopen(txt_path, 'w');
    fprintf(fid, 'HIL SIMULATION — SCENARIO SUMMARY\n');
    fprintf(fid, 'Generated: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, '%s\n\n', repmat('=', 1, 110));

    fprintf(fid, '%-30s %10s %10s %10s %10s %10s %10s %+10s\n', ...
            'Scenario', 'PID_SS', 'ADRC_SS', 'PID_full', 'ADRC_full', ...
            'PID_peak', 'ADRC_peak', 'ADRC_vs_PID');
    fprintf(fid, '%-30s %10s %10s %10s %10s %10s %10s %10s\n', ...
            '', '(mm)', '(mm)', '(mm)', '(mm)', '(mm)', '(mm)', '(%)');
    fprintf(fid, '%s\n', repmat('-', 1, 110));

    for i = 1:n
        fprintf(fid, '%-30s %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %+10.1f\n', ...
                names{i}, pid_rms_ss(i), adrc_rms_ss(i), ...
                pid_rms_full(i), adrc_rms_full(i), ...
                pid_peak(i), adrc_peak(i), improvement(i));
    end

    fprintf(fid, '%s\n', repmat('-', 1, 110));
    adrc_wins = sum(improvement > 5);
    pid_wins  = sum(improvement < -5);
    ties      = sum(abs(improvement) <= 5);
    fprintf(fid, '\nOverall: ADRC wins=%d, PID wins=%d, ties=%d (out of %d)\n', ...
            adrc_wins, pid_wins, ties, n);
    fprintf(fid, 'Mean improvement (ADRC vs PID): %+.1f%%\n', mean(improvement));
    fprintf(fid, 'Max improvement: %+.1f%% (%s)\n', max(improvement), ...
            names{improvement == max(improvement)});
    fclose(fid);
    fprintf('Saved: %s\n', txt_path);

    %% CSV export
    csv_path = fullfile(results_dir, 'summary_table.csv');
    fid = fopen(csv_path, 'w');
    fprintf(fid, 'Scenario,PID_SS_mm,ADRC_SS_mm,PID_full_mm,ADRC_full_mm,PID_peak_mm,ADRC_peak_mm,PID_tau_max,ADRC_tau_max,ADRC_improvement_pct\n');
    for i = 1:n
        fprintf(fid, '%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f\n', ...
                names{i}, pid_rms_ss(i), adrc_rms_ss(i), ...
                pid_rms_full(i), adrc_rms_full(i), ...
                pid_peak(i), adrc_peak(i), pid_tau(i), adrc_tau(i), improvement(i));
    end
    fclose(fid);
    fprintf('Saved: %s\n', csv_path);

    %% Summary bar chart
    fig_summary = figure('Position', [100 100 1200 700], 'Color','w');
    subplot(2,1,1);
    bar_data = [pid_rms_ss, adrc_rms_ss];
    bar(bar_data);
    set(gca, 'XTickLabel', strrep(names, '_', '\_'), 'XTickLabelRotation', 45);
    ylabel('RMS SS position error (mm)');
    title('PID vs ADRC — Steady-State Tracking Error by Scenario');
    legend('PID','ADRC','Location','best'); grid on;

    subplot(2,1,2);
    colors = repmat([0 0.6 0.2], n, 1);
    colors(improvement < 0, :) = repmat([0.8 0 0], sum(improvement<0), 1);
    b = bar(improvement, 'FaceColor','flat');
    b.CData = colors;
    set(gca, 'XTickLabel', strrep(names, '_', '\_'), 'XTickLabelRotation', 45);
    ylabel('ADRC improvement over PID (%)');
    title('ADRC SS improvement (green) / degradation (red)');
    grid on; yline(0, 'k-');

    png_path = fullfile(results_dir, 'summary_bars.png');
    saveas(fig_summary, png_path);
    fprintf('Saved: %s\n', png_path);

    %% PDF report (composite of all per-scenario PNGs + summary bars)
    pdf_path = fullfile(results_dir, 'summary_report.pdf');
    build_pdf_report(pdf_path, names, results_dir, fig_summary);
    fprintf('Saved: %s\n', pdf_path);

    fprintf('\nSummary report generation complete.\n');
end


function build_pdf_report(pdf_path, names, results_dir, fig_summary)
% Build multi-page PDF by exporting each figure then concatenating.
% MATLAB: use exportgraphics with 'Append' (R2020a+)
% Octave: use print with PDF; concatenation requires pdftk/ghostscript.

    try
        % Try MATLAB-style append
        exportgraphics(fig_summary, pdf_path, 'ContentType','vector');
        for i = 1:length(names)
            png_file = fullfile(results_dir, [names{i} '.png']);
            if exist(png_file, 'file')
                f = figure('Visible','off');
                img = imread(png_file);
                imshow(img); axis off;
                exportgraphics(f, pdf_path, 'Append', true, 'ContentType','image');
                close(f);
            end
        end
    catch ME_mat
        % Fallback: just save summary as standalone PDF
        warning('Multi-page PDF append not supported (%s).', ME_mat.message);
        try
            saveas(fig_summary, pdf_path);
        catch
            fprintf(2, 'Could not save PDF. Use PNG files in results/ directly.\n');
        end
    end
end
