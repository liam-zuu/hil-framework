function fig = plot_summary_v2(T)
% PLOT_SUMMARY_V2  Dot-plot summary of v2 metrics across scenarios.
%   Four panels share one scenario axis (s01 at top). Each metric gets its own
%   axis (no dual axes). Log scale where scenarios span orders of magnitude.
%   Tracking scenarios (s01-s10) and fault scenarios (s11+) are separated.

    C_PID = [0 114 178]/255;   C_ADRC = [213 94 0]/255;   % Okabe-Ito, validated
    INK = [0.25 0.25 0.25];   GRID = [0.88 0.88 0.88];

    names = T.Scenario;  n = numel(names);
    y = (1:n)';
    isFault = cellfun(@(s) sscanf(s, 's%d') >= 11, names);
    y(isFault) = y(isFault) + 0.8;                       % visual gap between groups
    labels = strrep(names, '_', '\_');

    panels = { ...
        'rms_w_err',   'Wheel-speed RMS error (rad/s)',   'log', 'Inner loop: controller under test';
        'tv_tau',      'Torque total variation (N\cdotm/s)', 'log', 'Control effort: smoothness (lower = less chatter)';
        'rms_e_track', 'Tracking RMS |true-ref| (mm)',    'log', 'Position: what v1 reported';
        'drift_share', 'Estimator drift share (%)',       'linear', 'Position: share caused by estimator drift'};

    fig = figure('Color', 'w', 'Position', [50 50 1600 700], 'Visible', 'off');
    tl = tiledlayout(fig, 1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, 'M6 scenarios: PID (blue) vs ADRC (orange), steady-state window', ...
        'FontWeight', 'bold', 'Color', INK);

    for k = 1:size(panels, 1)
        ax = nexttile(tl); hold(ax, 'on');
        p = T.(['PID_'  panels{k,1}]);   a = T.(['ADRC_' panels{k,1}]);
        for i = 1:n                                     % connector shows the gap
            plot(ax, [p(i) a(i)], [y(i) y(i)], '-', 'Color', [0.75 0.75 0.75], 'LineWidth', 2, ...
                'HandleVisibility', 'off');
        end
        % ADRC drawn first and larger, PID on top and smaller: both stay visible when equal
        scatter(ax, a, y, 80, C_ADRC, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1);
        scatter(ax, p, y, 36, C_PID,  'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1);
        ax.YDir = 'reverse';  ax.YLim = [0.3, max(y) + 0.7];
        ax.XScale = panels{k,3};
        if strcmp(panels{k,3}, 'linear')
            ax.XLim = [0 105];
        else
            v = [p; a];  ax.XLim = 10.^[floor(log10(min(v))) - 0.1, ceil(log10(max(v))) + 0.1];
        end
        yline(ax, min(y(isFault)) - 0.9, '-', 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
        ax.YTick = y;
        if k == 1, ax.YTickLabel = labels; else, ax.YTickLabel = {}; end
        ax.Box = 'off'; ax.XGrid = 'on'; ax.YGrid = 'on'; ax.GridColor = GRID; ax.GridAlpha = 1;
        ax.XColor = INK; ax.YColor = INK; ax.FontSize = 9;
        xlabel(ax, panels{k,2});
        title(ax, panels{k,4}, 'FontWeight', 'normal', 'FontSize', 9, 'Color', INK);
        if k == 1
            legend(ax, {'ADRC', 'PID'}, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off');
            text(ax, ax.XLim(1), min(y(isFault)) - 0.55, ' fault scenarios', 'Color', [0.45 0.45 0.45], ...
                'FontSize', 8, 'VerticalAlignment', 'middle');
        end
    end
end
