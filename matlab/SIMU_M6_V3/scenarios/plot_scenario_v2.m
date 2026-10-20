function fig = plot_scenario_v2(name, S, mp, ma)
% PLOT_SCENARIO_V2  One figure per scenario, organised by controller layer.
%   S  : struct loaded from <name>.mat (log_pid, log_adrc, traj, params)
%   mp, ma : metrics_compute_v2 output for PID / ADRC
%
% Panels: (a) XY true + estimated pose, (b) position error split into
% control error and estimator drift, (c) wheel-speed tracking error,
% (d) torque as 20 ms band (min-max) + mean instead of raw 1 kHz chatter,
% (e) metric table.

    C_PID = [0 114 178]/255;   C_ADRC = [213 94 0]/255;   % Okabe-Ito, validated
    INK = [0.25 0.25 0.25];   GRID = [0.85 0.85 0.85];
    lp = S.log_pid;  la = S.log_adrc;  tr = S.traj;  t = lp.t_log;  N = numel(t);

    fig = figure('Color', 'w', 'Position', [50 50 1500 820], 'Visible', 'off');
    tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, strrep(name, '_', '\_'), 'FontWeight', 'bold', 'FontSize', 13, 'Color', INK);
    subtitle(tl, sprintf('PID (blue) vs ADRC (orange)  |  steady-state window %.1f-%.1f s', ...
        mp.ss_window(1), mp.ss_window(2)), 'Color', INK);

    %% (a) XY
    ax = nexttile(tl); hold(ax, 'on');
    plot(ax, tr.x_ref(1:N), tr.y_ref(1:N), 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5, 'LineStyle', '--');
    plot(ax, lp.x_log, lp.y_log, 'Color', C_PID,  'LineWidth', 1.5);
    plot(ax, la.x_log, la.y_log, 'Color', C_ADRC, 'LineWidth', 1.5);
    plot(ax, lp.pos_est_log(1,:), lp.pos_est_log(2,:), ':', 'Color', C_PID,  'LineWidth', 1.2);
    plot(ax, la.pos_est_log(1,:), la.pos_est_log(2,:), ':', 'Color', C_ADRC, 'LineWidth', 1.2);
    axis(ax, 'equal'); style(ax);
    legend(ax, {'Reference', 'PID true', 'ADRC true', 'PID estimated', 'ADRC estimated'}, ...
        'Location', 'best', 'Box', 'off');
    title(ax, '(a) Path: true (solid) vs estimated (dotted)'); xlabel(ax, 'X (m)'); ylabel(ax, 'Y (m)');

    %% (b) Position error decomposition
    ax = nexttile(tl); hold(ax, 'on');
    plot(ax, t, 1000*mp.ts.e_ctrl,  'Color', C_PID,  'LineWidth', 1.5);
    plot(ax, t, 1000*ma.ts.e_ctrl,  'Color', C_ADRC, 'LineWidth', 1.5);
    plot(ax, t, 1000*mp.ts.e_drift, '--', 'Color', C_PID,  'LineWidth', 1.5);
    plot(ax, t, 1000*ma.ts.e_drift, '--', 'Color', C_ADRC, 'LineWidth', 1.5);
    xline(ax, mp.ss_window(1), ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    style(ax);
    legend(ax, {'PID control |est-ref|', 'ADRC control |est-ref|', ...
        'PID drift |true-est|', 'ADRC drift |true-est|'}, 'Location', 'northwest', 'Box', 'off');
    title(ax, '(b) Position error: control vs estimator drift');
    xlabel(ax, 'Time (s)'); ylabel(ax, 'Error (mm)');

    %% (c) Wheel-speed tracking error (controller under test)
    ax = nexttile(tl); hold(ax, 'on');
    w = max(1, round(0.05 / (t(2)-t(1))));             % 50 ms moving RMS
    plot(ax, t, sqrt(movmean(mp.ts.e_w_rms.^2, w)), 'Color', C_PID,  'LineWidth', 1.5);
    plot(ax, t, sqrt(movmean(ma.ts.e_w_rms.^2, w)), 'Color', C_ADRC, 'LineWidth', 1.5);
    xline(ax, mp.ss_window(1), ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    style(ax);
    legend(ax, {sprintf('PID  (SS RMS %.3f)', mp.rms_w_err), sprintf('ADRC (SS RMS %.3f)', ma.rms_w_err)}, ...
        'Location', 'northeast', 'Box', 'off');
    title(ax, '(c) Wheel-speed error |\omega_{ref}-\omega| (50 ms RMS)');
    xlabel(ax, 'Time (s)'); ylabel(ax, 'rad/s');

    %% (d) Torque band, wheel 1
    ax = nexttile(tl); hold(ax, 'on');
    band(ax, t, lp.tau_log(1,:), C_PID);
    band(ax, t, la.tau_log(1,:), C_ADRC);
    yline(ax, [-1 1]*S.params.tau_max, ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    style(ax);
    legend(ax, {'PID min-max', 'PID mean', 'ADRC min-max', 'ADRC mean'}, 'Location', 'southeast', 'Box', 'off');
    title(ax, '(d) Torque wheel 1 (20 ms windows; dotted = \tau_{max})');
    xlabel(ax, 'Time (s)'); ylabel(ax, '\tau (N\cdotm)');

    %% (e) Metric table (spans 2 tiles)
    ax = nexttile(tl, [1 2]); axis(ax, 'off');
    rows = { ...
        'Inner loop (controller under test)', '', '';
        '  Wheel-speed RMS error, true (rad/s)', mp.rms_w_err,     ma.rms_w_err;
        '  Wheel-speed RMS error, as measured (rad/s)', mp.rms_w_err_est, ma.rms_w_err_est;
        'Control effort', '', '';
        '  RMS torque (N.m)',                 mp.rms_tau,  ma.rms_tau;
        '  Torque total variation (N.m/s)',   mp.tv_tau,   ma.tv_tau;
        '  Saturated samples (%)',            mp.sat_pct,  ma.sat_pct;
        'Position (outer loop + estimator)', '', '';
        '  Tracking RMS |true-ref| (mm)',     mp.rms_e_track, ma.rms_e_track;
        '  Control RMS |est-ref| (mm)',       mp.rms_e_ctrl,  ma.rms_e_ctrl;
        '  Drift RMS |true-est| (mm)',        mp.rms_e_drift, ma.rms_e_drift;
        '  Drift share of tracking error (%)', mp.drift_share, ma.drift_share};
    y = 0.95;  dy = 0.075;
    text(ax, 0.00, y, 'Metric (steady-state window)', 'FontWeight', 'bold', 'Color', INK, 'FontName', 'Consolas');
    text(ax, 0.62, y, 'PID',  'FontWeight', 'bold', 'Color', INK, 'FontName', 'Consolas', 'HorizontalAlignment', 'right');
    text(ax, 0.78, y, 'ADRC', 'FontWeight', 'bold', 'Color', INK, 'FontName', 'Consolas', 'HorizontalAlignment', 'right');
    for i = 1:size(rows, 1)
        y = y - dy;
        isHead = ischar(rows{i,2});
        text(ax, 0.00, y, rows{i,1}, 'Color', INK, 'FontName', 'Consolas', ...
            'FontWeight', ternary(isHead, 'bold', 'normal'));
        if ~isHead
            text(ax, 0.62, y, sprintf('%.3f', rows{i,2}), 'Color', INK, 'FontName', 'Consolas', 'HorizontalAlignment', 'right');
            text(ax, 0.78, y, sprintf('%.3f', rows{i,3}), 'Color', INK, 'FontName', 'Consolas', 'HorizontalAlignment', 'right');
        end
    end

    function style(a)
        a.Box = 'off'; a.XGrid = 'on'; a.YGrid = 'on'; a.GridColor = GRID; a.GridAlpha = 1;
        a.XColor = INK; a.YColor = INK; a.FontSize = 9; a.TitleFontWeight = 'normal';
        a.Title.Color = INK;
    end
end

function band(ax, t, x, c)
    w  = max(1, round(0.02 / (t(2)-t(1))));
    n  = floor(numel(x) / w);
    X  = reshape(x(1:n*w), w, n);
    tc = mean(reshape(t(1:n*w), w, n), 1);
    lo = min(X, [], 1);  hi = max(X, [], 1);  mu = mean(X, 1);
    fill(ax, [tc fliplr(tc)], [lo fliplr(hi)], c, 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    plot(ax, tc, mu, 'Color', c, 'LineWidth', 1.5);
end

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
