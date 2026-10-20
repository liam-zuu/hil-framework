function S = analyze_multiseed(out_dir)
% ANALYZE_MULTISEED  Paired PID-vs-ADRC statistics from run_multiseed output.
%   S = analyze_multiseed(out_dir)   reads out_dir/multiseed_long.csv
%
% For each scenario and metric, PID and ADRC share a seed, so the
% comparison is paired:  d_i = (ADRC_i - PID_i) / mean(PID) * 100   [%]
% Reported: mean of d, 95% t-interval of the mean, and in how many seeds
% ADRC was lower (better: every metric here is "lower is better").
% Writes multiseed_stats.csv and multiseed_forest.png.

    C_PID = [0 114 178]/255;   C_ADRC = [213 94 0]/255;   INK = [0.25 0.25 0.25];
    T = readtable(fullfile(out_dir, 'multiseed_long.csv'));
    scen = unique(T.Scenario, 'stable');
    mets = {'rms_w_err', 'Wheel-speed RMS error';
            'tv_tau',    'Torque total variation';
            'rms_e_track', 'Tracking RMS |true-ref|'};

    rows = {};
    for i = 1:numel(scen)
        P = sortrows(T(strcmp(T.Scenario, scen{i}) & strcmp(T.Controller, 'PID'),  :), 'Seed');
        A = sortrows(T(strcmp(T.Scenario, scen{i}) & strcmp(T.Controller, 'ADRC'), :), 'Seed');
        assert(isequal(P.Seed, A.Seed), 'seed mismatch in %s', scen{i});
        n = height(P);
        for k = 1:size(mets, 1)
            p = P.(mets{k,1});  a = A.(mets{k,1});
            d = 100 * (a - p) / mean(p);
            se = std(d) / sqrt(n);
            h  = tinv(0.975, n - 1) * se;
            rows(end+1, :) = {scen{i}, mets{k,1}, n, mean(p), std(p), mean(a), std(a), ...
                              mean(d), mean(d) - h, mean(d) + h, sum(a < p)}; %#ok<AGROW>
        end
    end
    S = cell2table(rows, 'VariableNames', {'Scenario','Metric','n_seeds','PID_mean','PID_sd', ...
        'ADRC_mean','ADRC_sd','diff_pct_mean','diff_pct_ci_lo','diff_pct_ci_hi','ADRC_better_count'});
    writetable(S, fullfile(out_dir, 'multiseed_stats.csv'));

    %% Forest plot: one panel per metric, scenarios on y
    fig = figure('Color', 'w', 'Position', [50 50 1500 700], 'Visible', 'off');
    tl = tiledlayout(fig, 1, size(mets, 1), 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, sprintf('ADRC relative to PID, paired over %d seeds (mean and 95%% CI)', S.n_seeds(1)), ...
        'FontWeight', 'bold', 'Color', INK);
    subtitle(tl, {'left of 0 = ADRC lower (better); right of 0 = PID better', ...
        'orange: 95% CI entirely below 0 (ADRC better)   blue: entirely above 0 (PID better)   gray: CI contains 0 (no clear difference)'}, 'Color', INK);
    y = (1:numel(scen))';
    isFault = cellfun(@(s) sscanf(s, 's%d') >= 11, scen);
    y(isFault) = y(isFault) + 0.8;
    for k = 1:size(mets, 1)
        ax = nexttile(tl); hold(ax, 'on');
        R = S(strcmp(S.Metric, mets{k,1}), :);
        xline(ax, 0, '-', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
        for i = 1:numel(scen)
            better = R.diff_pct_ci_hi(i) < 0;  worse = R.diff_pct_ci_lo(i) > 0;
            c = [0.6 0.6 0.6];
            if better, c = C_ADRC; elseif worse, c = C_PID; end
            plot(ax, [R.diff_pct_ci_lo(i) R.diff_pct_ci_hi(i)], [y(i) y(i)], '-', 'Color', c, 'LineWidth', 2);
            scatter(ax, R.diff_pct_mean(i), y(i), 49, c, 'filled', 'MarkerEdgeColor', 'w');
        end
        ax.YDir = 'reverse';  ax.YLim = [0.3, max(y) + 0.7];  ax.YTick = y;
        if k == 1, ax.YTickLabel = strrep(scen, '_', '\_'); else, ax.YTickLabel = {}; end
        yline(ax, min(y(isFault)) - 0.9, '-', 'Color', [0.7 0.7 0.7]);
        ax.Box = 'off'; ax.XGrid = 'on'; ax.GridColor = [0.88 0.88 0.88]; ax.GridAlpha = 1;
        ax.XColor = INK; ax.YColor = INK; ax.FontSize = 9;
        xlabel(ax, '(ADRC - PID) / PID  [%]');
        title(ax, mets{k,2}, 'FontWeight', 'normal', 'Color', INK);
        xl = xlim(ax);                                   % direct labels at the right edge
        for i = 1:numel(scen)
            text(ax, xl(2), y(i), sprintf(' %+.1f%%', R.diff_pct_mean(i)), 'Color', INK, ...
                'FontSize', 8, 'HorizontalAlignment', 'left', 'Clipping', 'off');
        end
    end
    exportgraphics(fig, fullfile(out_dir, 'multiseed_forest.png'), 'Resolution', 150);
    close(fig);
end
