%% PLOT_M6_RESULTS  Visualize M6 disturbance & robustness results.
%
% Run AFTER run_m6_disturbance.m — requires mean_ss, std_ss, cond,
% controllers, trajectories in workspace.
%
% Generates 4 figure windows:
%   Fig 1 — Per-condition grouped bar (PID vs ADRC, 4 trajectories each)
%   Fig 2 — Summary: average SS error across all trajectories per condition
%   Fig 3 — Encoder PPR degradation curve
%   Fig 4 — Worst-case breakdown (all 4 trajectories, side by side)

%% ── Sanity check ────────────────────────────────────────────────────────────
if ~exist('mean_ss','var') || ~exist('cond','var')
    error('Run run_m6_disturbance.m first to populate mean_ss / std_ss.');
end

n_cond = size(mean_ss, 1);   % 12
n_ctrl = size(mean_ss, 2);   % 2  (1=PID, 2=ADRC)
n_traj = size(mean_ss, 3);   % 4

cond_names = {cond.name};
traj_labels = {'Line','Circle','Square','Figure-8'};

C_PID  = [0.84 0.15 0.15];   % red
C_ADRC = [0.09 0.64 0.29];   % green
C_bg   = [0.97 0.97 0.97];

%% ═══════════════════════════════════════════════════════════════════════════
%%  FIG 1 — Per-condition: grouped bar PID vs ADRC (one subplot per condition)
%% ═══════════════════════════════════════════════════════════════════════════
figure('Name','M6 — Per-condition PID vs ADRC','NumberTitle','off', ...
       'Color','w','Position',[50 50 1400 900]);

n_cols = 4; n_rows = 3;

for ci = 1:n_cond
    ax = subplot(n_rows, n_cols, ci);

    % data: [4 traj × 2 ctrl]
    data = squeeze(mean_ss(ci, :, :))';   % [4×2]
    err  = squeeze(std_ss(ci, :, :))';    % [4×2]

    hb = bar(data, 'grouped');
    hb(1).FaceColor = C_PID;
    hb(2).FaceColor = C_ADRC;
    hb(1).EdgeColor = 'none';
    hb(2).EdgeColor = 'none';

    % Error bars
    hold on;
    ngroups = size(data,1);  % 4
    nbars   = size(data,2);  % 2
    groupwidth = min(0.8, nbars/(nbars+1.5));
    for b = 1:nbars
        x = (1:ngroups) - groupwidth/2 + (2*b-1)*groupwidth/(2*nbars);
        errorbar(x, data(:,b), err(:,b), 'k.', 'LineWidth', 1.0, 'CapSize', 4);
    end
    hold off;

    % Formatting
    ax.Color = C_bg;
    ax.Box = 'off';
    ax.XTickLabel = traj_labels;
    ax.FontSize = 8;
    ax.YLabel.String = 'SS Error (mm)';
    ax.GridColor = [0.8 0.8 0.8];
    ax.YGrid = 'on';

    % Title with color coding for severity
    pid_avg  = mean(mean_ss(ci, 1, :));
    adrc_avg = mean(mean_ss(ci, 2, :));
    impr = (pid_avg - adrc_avg) / max(pid_avg, 0.01) * 100;

    if impr > 50
        tc = [0.7 0 0];       % dark red = big ADRC advantage
    elseif impr > 20
        tc = [0.8 0.4 0];     % orange
    else
        tc = [0.2 0.2 0.2];   % neutral
    end
    title(cond_names{ci}, 'FontSize', 9, 'FontWeight', 'bold', 'Color', tc);

    % Improvement annotation
    annotation_str = sprintf('+%.0f%%', impr);
    text(0.97, 0.95, annotation_str, 'Units','normalized', ...
         'HorizontalAlignment','right', 'VerticalAlignment','top', ...
         'FontSize', 8, 'FontWeight','bold', 'Color', C_ADRC, 'Parent', ax);
end

% Shared legend
hL = legend(hb, {'PID','ADRC'}, 'Orientation','horizontal', ...
            'FontSize', 9, 'Box','off');
hL.Position = [0.42 0.01 0.15 0.03];

sgtitle('M6 — SS Position Error per Condition (mean ± std, N=5 trials)', ...
        'FontSize', 13, 'FontWeight', 'bold');

%% ═══════════════════════════════════════════════════════════════════════════
%%  FIG 2 — Summary: average across trajectories, all conditions
%% ═══════════════════════════════════════════════════════════════════════════
figure('Name','M6 — Condition Summary','NumberTitle','off', ...
       'Color','w','Position',[100 100 1100 480]);

pid_avg_all  = squeeze(mean(mean_ss(:, 1, :), 3));   % [12×1]
adrc_avg_all = squeeze(mean(mean_ss(:, 2, :), 3));   % [12×1]

ax2 = axes(); hold on;

x = 1:n_cond;
w = 0.35;

b1 = bar(x - w/2, pid_avg_all,  w, 'FaceColor', C_PID,  'EdgeColor','none');
b2 = bar(x + w/2, adrc_avg_all, w, 'FaceColor', C_ADRC, 'EdgeColor','none');

% Error bars (avg std across trajectories)
pid_std_avg  = squeeze(mean(std_ss(:, 1, :), 3));
adrc_std_avg = squeeze(mean(std_ss(:, 2, :), 3));
errorbar(x - w/2, pid_avg_all,  pid_std_avg,  'k.', 'LineWidth',1, 'CapSize',5);
errorbar(x + w/2, adrc_avg_all, adrc_std_avg, 'k.', 'LineWidth',1, 'CapSize',5);

% Improvement % label on top of each pair
for ci = 1:n_cond
    impr = (pid_avg_all(ci) - adrc_avg_all(ci)) / max(pid_avg_all(ci), 0.01) * 100;
    y_top = max(pid_avg_all(ci), adrc_avg_all(ci)) + max(pid_std_avg(ci), adrc_std_avg(ci)) + 1;
    if impr > 5
        text(ci, y_top, sprintf('+%.0f%%', impr), ...
             'HorizontalAlignment','center', 'FontSize', 7.5, ...
             'FontWeight','bold', 'Color', C_ADRC);
    end
end

ax2.XTick = x;
ax2.XTickLabel = cond_names;
ax2.XTickLabelRotation = 30;
ax2.FontSize = 10;
ax2.YLabel.String = 'Avg SS Error across 4 trajectories (mm)';
ax2.YGrid = 'on';
ax2.Box = 'off';
ax2.Color = C_bg;

% Highlight PPR and worst case zones
highlight_idx = [4 5 12];   % PPR 256, PPR 128, Worst case
for idx = highlight_idx
    fill([idx-0.5 idx+0.5 idx+0.5 idx-0.5], ...
         [0 0 ax2.YLim(2) ax2.YLim(2)], ...
         [1 0.9 0.9], 'EdgeColor','none', 'FaceAlpha',0.25);
end

legend([b1 b2], {'PID','ADRC'}, 'Location','northwest', 'FontSize',10, 'Box','off');
title('M6 — Average SS Error per Condition (all 4 trajectories)', ...
      'FontSize', 13, 'FontWeight','bold');

hold off;

%% ═══════════════════════════════════════════════════════════════════════════
%%  FIG 3 — Encoder PPR degradation curve
%% ═══════════════════════════════════════════════════════════════════════════
% Conditions: 1=Nominal(PPR1024), 3=PPR512, 4=PPR256, 5=PPR128
ppr_idx    = [1, 3, 4, 5];
ppr_values = [1024, 512, 256, 128];

figure('Name','M6 — Encoder PPR Degradation','NumberTitle','off', ...
       'Color','w','Position',[150 150 700 450]);

ax3 = axes(); hold on;

for ti = 1:n_traj
    pid_vals  = squeeze(mean_ss(ppr_idx, 1, ti));   % [4×1]
    adrc_vals = squeeze(mean_ss(ppr_idx, 2, ti));   % [4×1]

    ls_pid  = {'-o','-s','-^','-d'};
    ls_adrc = {'--o','--s','--^','--d'};
    traj_colors = [0.2 0.5 0.9;  0.9 0.5 0.1;  0.5 0.2 0.8;  0.1 0.7 0.5];

    plot(ppr_values, pid_vals,  ls_pid{ti},  'Color', traj_colors(ti,:), ...
         'LineWidth', 1.5, 'MarkerFaceColor', traj_colors(ti,:), 'MarkerSize', 7);
    plot(ppr_values, adrc_vals, ls_adrc{ti}, 'Color', traj_colors(ti,:)*0.7, ...
         'LineWidth', 1.5, 'MarkerFaceColor', traj_colors(ti,:)*0.7, 'MarkerSize', 7);
end

ax3.XDir = 'reverse';   % high PPR on left = better hardware on left
ax3.XScale = 'log';
ax3.XTick = ppr_values;
ax3.XTickLabel = {'1024 (nom)','512','256','128'};
ax3.FontSize = 10;
ax3.XLabel.String = 'Encoder PPR';
ax3.YLabel.String = 'SS Position Error (mm)';
ax3.YGrid = 'on';
ax3.Box = 'off';
ax3.Color = C_bg;

% Annotation
text(180, ax3.YLim(2)*0.9, 'PID — solid lines', 'FontSize', 9, 'Color', [0.4 0.1 0.1]);
text(180, ax3.YLim(2)*0.82,'ADRC — dashed (flat)', 'FontSize', 9, 'Color', [0.1 0.5 0.2]);

legend_entries = {};
for ti = 1:n_traj
    legend_entries{end+1} = ['PID-'  traj_labels{ti}];
    legend_entries{end+1} = ['ADRC-' traj_labels{ti}];
end

title('Encoder PPR Degradation — PID collapses, ADRC immune', ...
      'FontSize', 12, 'FontWeight','bold');
subtitle('PID error grows exponentially as PPR decreases; ADRC stays flat');

hold off;

%% ═══════════════════════════════════════════════════════════════════════════
%%  FIG 4 — Worst case vs Nominal: side-by-side per trajectory
%% ═══════════════════════════════════════════════════════════════════════════
figure('Name','M6 — Worst Case Breakdown','NumberTitle','off', ...
       'Color','w','Position',[200 200 850 480]);

% Conditions: 1=Nominal, 12=Worst case
nom_pid   = squeeze(mean_ss(1,    1, :));   % [4×1]
nom_adrc  = squeeze(mean_ss(1,    2, :));
wc_pid    = squeeze(mean_ss(end,  1, :));
wc_adrc   = squeeze(mean_ss(end,  2, :));

x = 1:n_traj;
w = 0.2;
offsets = [-1.5 -0.5 0.5 1.5] * w;

ax4 = axes(); hold on;

b_nom_pid  = bar(x + offsets(1), nom_pid,  w, 'FaceColor', C_PID,  'EdgeColor','none', 'FaceAlpha', 0.4);
b_nom_adrc = bar(x + offsets(2), nom_adrc, w, 'FaceColor', C_ADRC, 'EdgeColor','none', 'FaceAlpha', 0.4);
b_wc_pid   = bar(x + offsets(3), wc_pid,   w, 'FaceColor', C_PID,  'EdgeColor','none');
b_wc_adrc  = bar(x + offsets(4), wc_adrc,  w, 'FaceColor', C_ADRC, 'EdgeColor','none');

% Degradation factor labels above worst-case bars
for ti = 1:n_traj
    deg = wc_pid(ti) / max(nom_pid(ti), 0.1);
    text(x(ti) + offsets(3), wc_pid(ti) + 1.5, sprintf('%.1f×', deg), ...
         'HorizontalAlignment','center', 'FontSize', 8, ...
         'FontWeight','bold', 'Color', C_PID);
end

ax4.XTick = x;
ax4.XTickLabel = traj_labels;
ax4.FontSize = 11;
ax4.YLabel.String = 'SS Position Error (mm)';
ax4.YGrid = 'on';
ax4.Box = 'off';
ax4.Color = C_bg;

legend([b_nom_pid b_nom_adrc b_wc_pid b_wc_adrc], ...
       {'PID nominal','ADRC nominal','PID worst case','ADRC worst case'}, ...
       'Location','northwest', 'FontSize', 9, 'Box','off');

title('Worst Case: Slip + PPR 256 + Combined Load', ...
      'FontSize', 12, 'FontWeight','bold');
subtitle('Numbers above bars = degradation factor vs nominal');

hold off;

fprintf('\nPlotting complete. 4 figures generated.\n');
fprintf('  Fig 1 — Per-condition grouped bar (12 subplots)\n');
fprintf('  Fig 2 — Summary: avg across trajectories\n');
fprintf('  Fig 3 — PPR degradation curve\n');
fprintf('  Fig 4 — Worst case vs nominal breakdown\n');
