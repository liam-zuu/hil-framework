function R = s01_identify(data_dir, out_dir, R_motor_ohm)
% S01_IDENTIFY  Identify wheel-motor parameters from spec S01 lifted-wheel runs.
%   R = s01_identify(data_dir)                  % data_dir has runs.csv + raw/
%   R = s01_identify(data_dir, out_dir, [Rfl Rfr Rrl Rrr])
%
% Input: output of tools/s01_openloop_runner.py. Uses runs with aborted == 0
% and block_ok == 1 only. All quantities refer to the wheel (gearbox output) shaft.
% u = pwm/1000 (duty, -1..1), V = battery voltage (mean of block start/end).
%
% Answers the spec questions:
%  Q1  steady-state omega vs u per wheel/direction: linear fit on |pwm| >= 200
%      (slope s, intercept, R^2), dead band = x-intercept of that fit, and the
%      smallest tested level with |omega_ss| > 0.5 rad/s.
%  Q2  step time constant tau_step: fit w(t) = w_ss (1 - exp(-(t-t0-d)/tau))
%      on the first 1.5 s of the hold (fminsearch on tau and delay d).
%  Q3  test of "PWM is a torque source" (pwm_capture.m) against a DC motor
%      tau = (K/R)(V u - K w):
%        torque-source model  -> tau_step ~ tau_coast (both J/b)
%        DC motor (back-EMF)  -> tau_step ~ tau_brake  <  tau_coast
%      coast = EN 0 (open circuit, J/b plus Coulomb friction), brake = PWM 0, EN 1.
%  Q4  with the DC-motor model, per repetition r:
%        K = (1 - tau_step/tau_coast) * V / s           [V s/rad = N m/A]
%        J = K * tau_step * V / (R_motor * s)          (needs R_motor)
%        b = J / tau_coast,  Tc from the coast fit,  stall torque K V / R_motor
%  Q5  per-wheel means with 95% t-intervals over repetitions, one-way ANOVA
%      across wheels (anova1) for tau_step and s.
% Outputs in out_dir: s01_steady_state.csv, s01_time_constants.csv,
% s01_params_per_wheel.csv, s01_summary.txt and three PNG figures.

    if nargin < 2 || isempty(out_dir), out_dir = fullfile(data_dir, 'identification'); end
    if nargin < 3, R_motor_ohm = []; end
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    WHEELS = {'FL','FR','RL','RR'};

    runs = readtable(fullfile(data_dir, 'runs.csv'), 'TextType', 'string', 'Delimiter', ',');
    keep = runs.aborted == 0 & runs.block_ok == 1;
    fprintf('%d runs, %d used (aborted == 0 and block_ok == 1)\n', height(runs), sum(keep));
    runs = runs(keep, :);
    runs.V = (runs.v_batt_start + runs.v_batt_end) / 2;
    runs.sgn = 1 - 2 * (runs.direction == "neg");
    runs.u = runs.sgn .* runs.pwm_level / 1000;

    %% Per-run fits
    n = height(runs);
    runs.w_ss = nan(n,1); runs.tau = nan(n,1); runs.delay = nan(n,1); runs.Tc_over_J = nan(n,1);
    for i = 1:n
        D = readtable(fullfile(data_dir, runs.file(i)));
        t = D.t_us * 1e-6;  w = movmean(D.omega_rad_s, 5);
        switch runs.scenario(i)
            case "step"
                hold = t >= 0.3 & t < 3.3;
                runs.w_ss(i) = mean(w(t >= 2.8 & t < 3.3));
                k = hold & t < 1.8;
                [runs.tau(i), runs.delay(i)] = fit_rise(t(k) - 0.3, w(k), runs.w_ss(i));
            case {"coast", "brake"}
                k = t >= 1.5;
                [runs.tau(i), runs.Tc_over_J(i)] = fit_decay(t(k) - 1.5, w(k));
        end
    end

    %% Q1: steady state
    ss = runs(runs.scenario == "step", :);
    G = groupsummary(ss, {'wheel','direction','pwm_level'}, {'mean','std'}, 'w_ss');
    writetable(G, fullfile(out_dir, 's01_steady_state.csv'));

    %% Q2/Q3: time constants
    TCt = groupsummary(runs(ismember(runs.scenario, ["step","coast","brake"]), :), ...
        {'wheel','scenario','direction','pwm_level'}, {'mean','std'}, 'tau');
    writetable(TCt, fullfile(out_dir, 's01_time_constants.csv'));

    %% Q4/Q5: per wheel, per repetition
    rows = {};
    for wi = 1:numel(WHEELS)
        W = runs(runs.wheel == WHEELS{wi}, :);
        if isempty(W), continue; end
        for dirn = ["pos", "neg"]
            reps = unique(W.rep)';
            P = nan(numel(reps), 9);    % s, R2, deadband, tau_step, tau_coast, tau_brake, K, J, b
            for r = 1:numel(reps)
                S = W(W.scenario == "step" & W.direction == dirn & W.rep == reps(r), :);
                lin = S(S.pwm_level >= 200, :);
                if height(lin) >= 2
                    c = polyfit(abs(lin.u), abs(lin.w_ss), 1);
                    yhat = polyval(c, abs(lin.u));
                    R2 = 1 - sum((abs(lin.w_ss) - yhat).^2) / sum((abs(lin.w_ss) - mean(abs(lin.w_ss))).^2);
                    P(r, 1:3) = [c(1), R2, -c(2)/c(1)*1000];
                end
                P(r, 4) = mean(S.tau(S.pwm_level >= 200), 'omitnan');
                C = W(W.scenario == "coast" & W.rep == reps(r), :);   % coast/brake run in + only
                B = W(W.scenario == "brake" & W.rep == reps(r), :);
                P(r, 5) = mean(C.tau, 'omitnan');
                P(r, 6) = mean(B.tau, 'omitnan');
                V = mean(S.V);
                K = (1 - P(r,4) / P(r,5)) * V / P(r,1);
                P(r, 7) = K;
                if numel(R_motor_ohm) == 4
                    J = K * P(r,4) * V / (R_motor_ohm(wi) * P(r,1));
                    P(r, 8:9) = [J, J / P(r,5)];
                end
            end
            names = {'slope_rad_s_per_duty','R2_linear','deadband_pwm','tau_step_s','tau_coast_s', ...
                     'tau_brake_s','K_Vs_per_rad','J_kgm2','b_Nms_per_rad'};
            first_turn = min(S_levels_turning(W(W.scenario == "step" & W.direction == dirn, :)));
            for q = 1:numel(names)
                x = P(:, q);  x = x(~isnan(x));
                [m, lo, hi] = tci(x);
                rows(end+1, :) = {WHEELS{wi}, char(dirn), names{q}, numel(x), m, lo, hi}; %#ok<AGROW>
            end
            rows(end+1, :) = {WHEELS{wi}, char(dirn), 'first_level_turning_pwm', 1, first_turn, NaN, NaN}; %#ok<AGROW>
        end
    end
    T = cell2table(rows, 'VariableNames', {'wheel','direction','quantity','n','mean','ci95_lo','ci95_hi'});
    writetable(T, fullfile(out_dir, 's01_params_per_wheel.csv'));

    %% Q5: across wheels (pos direction, levels >= 200)
    st = runs(runs.scenario == "step" & runs.direction == "pos" & runs.pwm_level >= 200, :);
    p_tau = NaN;
    if numel(unique(st.wheel)) > 1
        p_tau = anova1(st.tau, st.wheel, 'off');
    end

    %% Q3 verdict
    rs = T(strcmp(T.quantity, 'tau_step_s') & strcmp(T.direction, 'pos'), :);
    rc = T(strcmp(T.quantity, 'tau_coast_s') & strcmp(T.direction, 'pos'), :);
    rb = T(strcmp(T.quantity, 'tau_brake_s') & strcmp(T.direction, 'pos'), :);
    fid = fopen(fullfile(out_dir, 's01_summary.txt'), 'w');
    fprintf(fid, 'S01 identification summary (%s)\nData: %s\nRuns used: %d\n\n', datestr(now, 'yyyy-mm-dd HH:MM'), data_dir, n);
    fprintf(fid, 'Q3 time constants, + direction, mean [95%% CI] (s):\n');
    for i = 1:height(rs)
        fprintf(fid, '  %s  step %.4f [%.4f, %.4f]  coast %.4f [%.4f, %.4f]  brake %.4f [%.4f, %.4f]  step/coast = %.2f\n', ...
            rs.wheel{i}, rs.mean(i), rs.ci95_lo(i), rs.ci95_hi(i), rc.mean(i), rc.ci95_lo(i), rc.ci95_hi(i), ...
            rb.mean(i), rb.ci95_lo(i), rb.ci95_hi(i), rs.mean(i) / rc.mean(i));
    end
    fprintf(fid, ['  Reading: step/coast near 1 supports "PWM = torque source";\n' ...
                  '  step clearly below coast and close to brake supports the DC-motor (back-EMF) model.\n']);
    fprintf(fid, '\nQ5 one-way ANOVA of tau_step across wheels (+, |pwm| >= 200): p = %.4g\n', p_tau);
    if numel(R_motor_ohm) ~= 4
        fprintf(fid, '\nR_motor not given: J and b not computed (only K and time-constant ratios).\n');
    end
    fprintf(fid, '\nFull tables: s01_params_per_wheel.csv, s01_steady_state.csv, s01_time_constants.csv\n');
    fclose(fid);
    type(fullfile(out_dir, 's01_summary.txt'));

    plot_figures(runs, data_dir, out_dir, WHEELS);
    R = struct('runs', runs, 'params', T, 'steady', G, 'tau', TCt, 'p_anova_tau', p_tau);
end

% ── helpers ──────────────────────────────────────────────────────────────────

function lv = S_levels_turning(S)
    lv = S.pwm_level(abs(S.w_ss) > 0.5);
    if isempty(lv), lv = NaN; end
end

function [m, lo, hi] = tci(x)
    m = mean(x); lo = NaN; hi = NaN;
    if numel(x) >= 2
        h = tinv(0.975, numel(x) - 1) * std(x) / sqrt(numel(x));
        lo = m - h; hi = m + h;
    end
end

function [tau, d] = fit_rise(t, w, w_ss)
    tau = NaN; d = NaN;
    if abs(w_ss) < 0.5, return; end
    f = @(p) sum((w - w_ss * (1 - exp(-max(t - abs(p(2)), 0) / abs(p(1))))).^2);
    p = fminsearch(f, [0.05, 0.005], optimset('Display', 'off'));
    tau = abs(p(1)); d = abs(p(2));
end

function [tau, a] = fit_decay(t, w)
% J dw/dt = -b w - Tc sgn(w)  ->  w(t) = (w0 + a tau) exp(-t/tau) - a tau,  a = Tc/J
    tau = NaN; a = NaN;
    w0 = mean(w(1:min(5, end)));
    if abs(w0) < 0.5, return; end
    sg = sign(w0);  w = sg * w;  w0 = abs(w0);
    k = find(w < 0.05 * w0, 1);  if isempty(k), k = numel(w); end
    t = t(1:k);  w = w(1:k);
    f = @(p) sum((w - ((w0 + abs(p(2)) * abs(p(1))) .* exp(-t / abs(p(1))) - abs(p(2)) * abs(p(1)))).^2);
    p = fminsearch(f, [0.1, 1], optimset('Display', 'off'));
    tau = abs(p(1)); a = abs(p(2));
end

function plot_figures(runs, data_dir, out_dir, WHEELS)
    C = [0 114 178; 213 94 0; 0 158 115; 204 121 167] / 255;   % Okabe-Ito, fixed order per wheel
    INK = [0.25 0.25 0.25];

    % Q1: omega_ss vs duty
    fig = figure('Color', 'w', 'Position', [50 50 800 500], 'Visible', 'off'); ax = axes(fig); hold(ax, 'on');
    for wi = 1:numel(WHEELS)
        S = runs(runs.scenario == "step" & runs.wheel == WHEELS{wi}, :);
        if isempty(S), continue; end
        scatter(ax, S.u, S.w_ss, 36, C(wi,:), 'filled', 'MarkerEdgeColor', 'w', 'DisplayName', WHEELS{wi});
    end
    grid(ax, 'on'); ax.GridColor = [0.85 0.85 0.85]; ax.GridAlpha = 1; ax.XColor = INK; ax.YColor = INK;
    xlabel(ax, 'duty u = pwm/1000'); ylabel(ax, 'steady-state \omega (rad/s)');
    title(ax, 'Q1: steady-state wheel speed vs PWM duty', 'FontWeight', 'normal');
    legend(ax, 'Location', 'northwest', 'Box', 'off');
    exportgraphics(fig, fullfile(out_dir, 'fig_q1_omega_ss.png'), 'Resolution', 150); close(fig);

    % Q3: time constants by run type
    fig = figure('Color', 'w', 'Position', [50 50 800 450], 'Visible', 'off'); ax = axes(fig); hold(ax, 'on');
    types = ["step", "brake", "coast"];
    for wi = 1:numel(WHEELS)
        for k = 1:3
            x = runs.tau(runs.wheel == WHEELS{wi} & runs.scenario == types(k) & runs.direction == "pos" & runs.pwm_level >= 200);
            x = x(~isnan(x));
            if isempty(x), continue; end
            xpos = k + (wi - 2.5) * 0.15;
            scatter(ax, xpos * ones(size(x)), x, 25, C(wi,:), 'filled', 'MarkerFaceAlpha', 0.5, 'HandleVisibility', 'off');
            plot(ax, xpos + [-0.05 0.05], mean(x) * [1 1], '-', 'Color', C(wi,:), 'LineWidth', 2, ...
                'DisplayName', WHEELS{wi}, 'HandleVisibility', ternary(k == 1, 'on', 'off'));
        end
    end
    ax.XTick = 1:3; ax.XTickLabel = {'step (driven)', 'brake (PWM 0, EN 1)', 'coast (EN 0)'};
    grid(ax, 'on'); ax.GridColor = [0.85 0.85 0.85]; ax.GridAlpha = 1; ax.XColor = INK; ax.YColor = INK;
    ylabel(ax, 'time constant (s)');
    title(ax, {'Q3: torque source predicts step = coast;', 'DC motor with back-EMF predicts step = brake < coast'}, 'FontWeight', 'normal');
    legend(ax, 'Location', 'northwest', 'Box', 'off');
    exportgraphics(fig, fullfile(out_dir, 'fig_q3_time_constants.png'), 'Resolution', 150); close(fig);

    % Example fits: one step and one coast run of the first wheel
    fig = figure('Color', 'w', 'Position', [50 50 900 380], 'Visible', 'off');
    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact');
    for k = 1:2
        typ = ["step", "coast"];
        cand = find(runs.scenario == typ(k) & runs.direction == "pos" & runs.pwm_level >= 600, 1);
        ax = nexttile(tl); hold(ax, 'on');
        if isempty(cand), continue; end
        D = readtable(fullfile(data_dir, runs.file(cand)));
        plot(ax, D.t_us * 1e-6, D.omega_rad_s, 'Color', [0.7 0.7 0.7]);
        plot(ax, D.t_us * 1e-6, movmean(D.omega_rad_s, 5), 'Color', C(1,:), 'LineWidth', 1.5);
        grid(ax, 'on'); xlabel(ax, 't (s)'); ylabel(ax, '\omega (rad/s)');
        title(ax, sprintf('%s, tau = %.3f s', strrep(runs.run_id(cand), '_', '\_'), runs.tau(cand)), 'FontWeight', 'normal');
    end
    exportgraphics(fig, fullfile(out_dir, 'fig_example_runs.png'), 'Resolution', 150); close(fig);
end

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
