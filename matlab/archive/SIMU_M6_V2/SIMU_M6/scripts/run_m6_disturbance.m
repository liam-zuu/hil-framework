%% RUN_M6_DISTURBANCE  M6 — Disturbance & Robustness: Multi-trial comparison.
%
% Each scenario runs N_TRIALS times with different random seeds.
% Reports mean ± std to eliminate random variance artifacts.
%
% 12 conditions × 2 controllers × 4 trajectories × N_TRIALS = total runs.
% Estimated time: ~30-40 min on MATLAB (T_sim=10s, N_TRIALS=5).
%
% Steps covered: M6.2, M6.3, M6.4, M6.5, M6.6

clear; clc; close all;

%% ===== CONFIGURATION =====
N_TRIALS     = 5;                          % trials per scenario
SEEDS        = [101, 202, 303, 404, 505];  % fixed seeds for reproducibility
controllers  = {'pid', 'adrc'};
trajectories = {'line', 'circle', 'square', 'figure8'};

n_ctrl = length(controllers);
n_traj = length(trajectories);

params_base = params_mecanum();
% params_base.T_sim = 10;  % uncomment if default is not 10

%% ===== DEFINE CONDITIONS =====
% Each condition: name + function that modifies params_base
cond = struct();
ci = 0;

ci=ci+1; cond(ci).name = 'Nominal';
cond(ci).setup = @(p) setfields(p, 'slip.enabled', false, 'disturbance.enabled', false);

ci=ci+1; cond(ci).name = 'Wheel slip';
cond(ci).setup = @(p) setfields(p, 'slip.enabled', true, 'disturbance.enabled', false);

ci=ci+1; cond(ci).name = 'Enc noise x2.5';
cond(ci).setup = @(p) setfields(p, 'enc_noise_sigma', 0.05, 'slip.enabled', false, 'disturbance.enabled', false);

ci=ci+1; cond(ci).name = 'Enc noise x5';
cond(ci).setup = @(p) setfields(p, 'enc_noise_sigma', 0.10, 'slip.enabled', false, 'disturbance.enabled', false);

ci=ci+1; cond(ci).name = 'Enc noise x10';
cond(ci).setup = @(p) setfields(p, 'enc_noise_sigma', 0.20, 'slip.enabled', false, 'disturbance.enabled', false);

ci=ci+1; cond(ci).name = 'IMU noise x3';
cond(ci).setup = @(p) setfields(p, 'imu_accel_noise', p.imu_accel_noise*3, ...
    'imu_gyro_noise', p.imu_gyro_noise*3, 'imu_bias_drift', p.imu_bias_drift*3, ...
    'slip.enabled', false, 'disturbance.enabled', false);

ci=ci+1; cond(ci).name = 'IMU noise x5';
cond(ci).setup = @(p) setfields(p, 'imu_accel_noise', p.imu_accel_noise*5, ...
    'imu_gyro_noise', p.imu_gyro_noise*5, 'imu_bias_drift', p.imu_bias_drift*5, ...
    'slip.enabled', false, 'disturbance.enabled', false);

ci=ci+1; cond(ci).name = 'Load: step';
cond(ci).setup = @(p) setfields(p, 'slip.enabled', false, ...
    'disturbance.enabled', true, 'disturbance.type', 'step', ...
    'disturbance.magnitude', 0.05, 'disturbance.start_time', 3.0);

ci=ci+1; cond(ci).name = 'Load: ramp';
cond(ci).setup = @(p) setfields(p, 'slip.enabled', false, ...
    'disturbance.enabled', true, 'disturbance.type', 'ramp', ...
    'disturbance.magnitude', 0.1, 'disturbance.ramp_rate', 0.02, ...
    'disturbance.start_time', 3.0);

ci=ci+1; cond(ci).name = 'Load: random';
cond(ci).setup = @(p) setfields(p, 'slip.enabled', false, ...
    'disturbance.enabled', true, 'disturbance.type', 'random', ...
    'disturbance.random_sigma', 0.03, 'disturbance.start_time', 3.0);

ci=ci+1; cond(ci).name = 'Load: combined';
cond(ci).setup = @(p) setfields(p, 'slip.enabled', false, ...
    'disturbance.enabled', true, 'disturbance.type', 'combined', ...
    'disturbance.magnitude', 0.05, 'disturbance.ramp_rate', 0.02, ...
    'disturbance.random_sigma', 0.03, 'disturbance.start_time', 3.0);

ci=ci+1; cond(ci).name = 'Worst case';
cond(ci).setup = @(p) setfields(p, 'slip.enabled', true, ...
    'enc_noise_sigma', 0.10, ...
    'disturbance.enabled', true, 'disturbance.type', 'combined', ...
    'disturbance.magnitude', 0.05, 'disturbance.ramp_rate', 0.02, ...
    'disturbance.random_sigma', 0.03, 'disturbance.start_time', 3.0);

n_cond = length(cond);

%% ===== STORAGE =====
% mean_ss(cond, ctrl, traj), std_ss(cond, ctrl, traj)
mean_ss = zeros(n_cond, n_ctrl, n_traj);
std_ss  = zeros(n_cond, n_ctrl, n_traj);

fprintf('====================================================================\n');
fprintf('  M6 — DISTURBANCE & ROBUSTNESS (MULTI-TRIAL, N=%d)\n', N_TRIALS);
fprintf('====================================================================\n');
fprintf('  T_sim=%.0fs, dt=%.4fs, Seeds=%s\n', ...
        params_base.T_sim, params_base.dt, mat2str(SEEDS));
fprintf('  Total runs: %d conditions × %d ctrl × %d traj × %d trials = %d\n', ...
        n_cond, n_ctrl, n_traj, N_TRIALS, n_cond*n_ctrl*n_traj*N_TRIALS);
fprintf('====================================================================\n\n');

%% ===== RUN ALL =====
total_runs = n_cond * n_ctrl * n_traj * N_TRIALS;
run_count = 0;
tic;

for ci_cond = 1:n_cond
    fprintf('--- Condition %d/%d: %s ---\n', ci_cond, n_cond, cond(ci_cond).name);

    % Apply condition setup
    params = cond(ci_cond).setup(params_base);

    for ci_ctrl = 1:n_ctrl
        for ti = 1:n_traj
            ss_trials = zeros(1, N_TRIALS);

            for trial = 1:N_TRIALS
                res = run_single_scenario(controllers{ci_ctrl}, trajectories{ti}, params, SEEDS(trial));
                ss_trials(trial) = res.rms_pos_ss;
                run_count = run_count + 1;
            end

            mean_ss(ci_cond, ci_ctrl, ti) = mean(ss_trials);
            std_ss(ci_cond, ci_ctrl, ti)  = std(ss_trials);

            fprintf('  %s / %-8s : %.1f ± %.1f mm  [%s]\n', ...
                    upper(controllers{ci_ctrl}), trajectories{ti}, ...
                    mean(ss_trials), std(ss_trials), ...
                    sprintf('%.1f ', ss_trials));
        end
    end
    fprintf('\n');
end

elapsed = toc;
fprintf('Total time: %.1f min (%d runs, %.1f s/run avg)\n\n', ...
        elapsed/60, run_count, elapsed/run_count);

%% =====================================================================
%%  M6.5 — COMPREHENSIVE COMPARISON TABLE (mean ± std)
%% =====================================================================
fprintf('=============================================================================================\n');
fprintf('  M6.5 — COMPARISON TABLE: Mean ± Std (mm), N=%d trials per scenario\n', N_TRIALS);
fprintf('=============================================================================================\n\n');

% Header
fprintf('%-18s |', 'Condition');
for ti = 1:n_traj
    fprintf('  %s_PID   %s_ADRC |', trajectories{ti}(1:min(3,end)), trajectories{ti}(1:min(3,end)));
end
fprintf(' PID_avg  ADRC_avg  Impr\n');
fprintf('%s\n', repmat('-', 1, 140));

for ci_cond = 1:n_cond
    fprintf('%-18s |', cond(ci_cond).name);
    pid_sum = 0; adrc_sum = 0;
    for ti = 1:n_traj
        pm = mean_ss(ci_cond, 1, ti);  ps = std_ss(ci_cond, 1, ti);
        am = mean_ss(ci_cond, 2, ti);  as_ = std_ss(ci_cond, 2, ti);
        fprintf(' %4.1f±%3.1f %4.1f±%3.1f |', pm, ps, am, as_);
        pid_sum  = pid_sum + pm;
        adrc_sum = adrc_sum + am;
    end
    pid_avg  = pid_sum / n_traj;
    adrc_avg = adrc_sum / n_traj;
    imp = 0;
    if pid_avg > 0
        imp = (pid_avg - adrc_avg) / pid_avg * 100;
    end
    fprintf('  %5.1f    %5.1f   %+5.1f%%\n', pid_avg, adrc_avg, imp);
end
fprintf('%s\n', repmat('-', 1, 140));

%% ===== TABLE 2: Degradation from Nominal =====
fprintf('\n--- Degradation from Nominal (×factor of mean SS error) ---\n');
fprintf('%-18s |', 'Condition');
for ti = 1:n_traj
    fprintf(' %s_PID %s_ADR |', trajectories{ti}(1:min(3,end)), trajectories{ti}(1:min(3,end)));
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 110));

for ci_cond = 1:n_cond
    fprintf('%-18s |', cond(ci_cond).name);
    for ti = 1:n_traj
        pid_nom  = mean_ss(1, 1, ti);  % condition 1 = nominal
        adrc_nom = mean_ss(1, 2, ti);
        pid_cur  = mean_ss(ci_cond, 1, ti);
        adrc_cur = mean_ss(ci_cond, 2, ti);
        pd = pid_cur / max(pid_nom, 0.1);
        ad = adrc_cur / max(adrc_nom, 0.1);
        fprintf(' %4.1fx  %4.1fx  |', pd, ad);
    end
    fprintf('\n');
end
fprintf('%s\n', repmat('-', 1, 110));

%% =====================================================================
%%  M6.6 — ANALYSIS
%% =====================================================================
fprintf('\n');
fprintf('=============================================================================================\n');
fprintf('  M6.6 — ANALYSIS: When does ADRC outperform PID?\n');
fprintf('=============================================================================================\n');

fprintf('\n--- Per-condition summary (average across 4 trajectories) ---\n');
fprintf('%-18s | %10s | %10s | %10s | %s\n', 'Condition', 'PID_avg', 'ADRC_avg', 'Impr.', 'Winner');
fprintf('%s\n', repmat('-', 1, 75));

for ci_cond = 1:n_cond
    pid_avg  = mean(mean_ss(ci_cond, 1, :));
    adrc_avg = mean(mean_ss(ci_cond, 2, :));
    imp = (pid_avg - adrc_avg) / max(pid_avg, 0.01) * 100;
    if imp > 5
        winner = 'ADRC';
    elseif imp < -5
        winner = 'PID';
    else
        winner = 'TIE';
    end
    fprintf('%-18s | %7.1f mm | %7.1f mm | %+7.1f%% | %s\n', ...
            cond(ci_cond).name, pid_avg, adrc_avg, imp, winner);
end
fprintf('%s\n', repmat('-', 1, 75));

% --- Win count ---
adrc_wins = 0;
total_pairs = n_cond * n_traj;
for ci_cond = 1:n_cond
    for ti = 1:n_traj
        if mean_ss(ci_cond, 2, ti) < mean_ss(ci_cond, 1, ti)
            adrc_wins = adrc_wins + 1;
        end
    end
end
fprintf('\nADRC wins: %d / %d scenario pairs (%.1f%%)\n', ...
        adrc_wins, total_pairs, adrc_wins/total_pairs*100);

% --- Variance analysis ---
fprintf('\n--- Variance analysis (std / mean ratio) ---\n');
fprintf('%-18s | PID CoV  | ADRC CoV | Note\n', 'Condition');
fprintf('%s\n', repmat('-', 1, 60));
for ci_cond = 1:n_cond
    pid_cov  = mean(std_ss(ci_cond, 1, :)) / max(mean(mean_ss(ci_cond, 1, :)), 0.1) * 100;
    adrc_cov = mean(std_ss(ci_cond, 2, :)) / max(mean(mean_ss(ci_cond, 2, :)), 0.1) * 100;
    note = '';
    if pid_cov > 30 || adrc_cov > 30
        note = '← HIGH VARIANCE';
    end
    fprintf('%-18s | %6.1f%%  | %6.1f%%  | %s\n', cond(ci_cond).name, pid_cov, adrc_cov, note);
end

% --- Robustness ---
fprintf('\n--- Robustness: nominal vs worst case ---\n');
pid_nom   = mean(mean_ss(1, 1, :));
adrc_nom  = mean(mean_ss(1, 2, :));
pid_worst = mean(mean_ss(n_cond, 1, :));
adrc_worst= mean(mean_ss(n_cond, 2, :));
fprintf('  PID:  nominal %.1fmm → worst %.1fmm (%.1f× degradation)\n', pid_nom, pid_worst, pid_worst/max(pid_nom,0.1));
fprintf('  ADRC: nominal %.1fmm → worst %.1fmm (%.1f× degradation)\n', adrc_nom, adrc_worst, adrc_worst/max(adrc_nom,0.1));

% --- Per-trajectory breakdown for slip ---
fprintf('\n--- Wheel slip breakdown (condition 2) —--\n');
fprintf('  %-8s | %8s | %8s | %10s\n', 'Traj', 'PID(mm)', 'ADRC(mm)', 'ADRC impr');
fprintf('  %s\n', repmat('-', 1, 45));
for ti = 1:n_traj
    pm = mean_ss(2, 1, ti);
    am = mean_ss(2, 2, ti);
    imp = (pm - am) / max(pm, 0.01) * 100;
    fprintf('  %-8s | %7.1f  | %7.1f  | %+8.1f%%\n', trajectories{ti}, pm, am, imp);
end

fprintf('\n====================================================================\n');
fprintf('  M6 MULTI-TRIAL COMPARISON COMPLETE\n');
fprintf('====================================================================\n');
