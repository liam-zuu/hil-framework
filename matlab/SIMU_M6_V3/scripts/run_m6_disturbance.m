%% RUN_M6_DISTURBANCE  M6 — Disturbance & Robustness: Multi-trial comparison (v3).
%
% Changes from v2:
%   - Encoder noise sweep → Encoder PPR sweep (1024→512→256→128)
%     Reason: additive noise sigma had no effect due to round() in pulse_gen.
%     Reducing PPR reduces counts/step, creating real quantization degradation.
%   - Load disturbance magnitude increased 3-5×:
%     step 0.05→0.15 N·m (30% tau_max), ramp 0.02→0.05 N·m/s,
%     random σ 0.03→0.10 N·m (20% tau_max)
%     Reason: 10% tau_max was absorbed trivially by outer loop PI integral.
%   - Worst case updated: slip + PPR 256 + large combined load
%
% 12 conditions × 2 controllers × 4 trajectories × 5 trials = 480 runs.
% Estimated time: ~6 min on MATLAB (T_sim=10s).

clear; clc; close all;

%% ===== CONFIGURATION =====
N_TRIALS     = 5;
SEEDS        = [101, 202, 303, 404, 505];
controllers  = {'pid', 'adrc'};
trajectories = {'line', 'circle', 'square', 'figure8'};

n_ctrl = length(controllers);
n_traj = length(trajectories);

params_base = params_mecanum();
% params_base.T_sim = 10;  % uncomment if default is not 10

%% ===== DEFINE CONDITIONS =====
cond = struct();
ci = 0;

% --- 1. Baseline ---
ci=ci+1; cond(ci).name = 'Nominal';
cond(ci).setup = @(p) setfields(p, 'slip.enabled', false, 'disturbance.enabled', false);

% --- 2. Wheel slip ---
ci=ci+1; cond(ci).name = 'Wheel slip';
cond(ci).setup = @(p) setfields(p, 'slip.enabled', true, 'disturbance.enabled', false);

% --- 3-5. Encoder PPR sweep (quantization degradation) ---
% Nominal PPR=1024. At omega=10 rad/s, dt=0.001:
%   PPR 1024: 1.63 counts/step (ok)
%   PPR 512:  0.81 counts/step (borderline)
%   PPR 256:  0.41 counts/step (very coarse)
%   PPR 128:  0.20 counts/step (extreme)
ci=ci+1; cond(ci).name = 'PPR 512';
cond(ci).setup = @(p) setfields(p, 'enc_ppr', 512, 'slip.enabled', false, 'disturbance.enabled', false);

ci=ci+1; cond(ci).name = 'PPR 256';
cond(ci).setup = @(p) setfields(p, 'enc_ppr', 256, 'slip.enabled', false, 'disturbance.enabled', false);

ci=ci+1; cond(ci).name = 'PPR 128';
cond(ci).setup = @(p) setfields(p, 'enc_ppr', 128, 'slip.enabled', false, 'disturbance.enabled', false);

% --- 6-7. IMU noise sweep ---
ci=ci+1; cond(ci).name = 'IMU noise x3';
cond(ci).setup = @(p) setfields(p, 'imu_accel_noise', p.imu_accel_noise*3, ...
    'imu_gyro_noise', p.imu_gyro_noise*3, 'imu_bias_drift', p.imu_bias_drift*3, ...
    'slip.enabled', false, 'disturbance.enabled', false);

ci=ci+1; cond(ci).name = 'IMU noise x5';
cond(ci).setup = @(p) setfields(p, 'imu_accel_noise', p.imu_accel_noise*5, ...
    'imu_gyro_noise', p.imu_gyro_noise*5, 'imu_bias_drift', p.imu_bias_drift*5, ...
    'slip.enabled', false, 'disturbance.enabled', false);

% --- 8-11. Load disturbance (increased magnitudes) ---
% step: 0.15 N·m = 30% of tau_max (was 0.05 = 10%)
ci=ci+1; cond(ci).name = 'Load: step 30%';
cond(ci).setup = @(p) setfields(p, 'slip.enabled', false, ...
    'disturbance.enabled', true, 'disturbance.type', 'step', ...
    'disturbance.magnitude', 0.15, 'disturbance.start_time', 3.0);

% ramp: 0.05 N·m/s, cap at 0.25 N·m = 50% tau_max (was 0.02/s, cap 0.1)
ci=ci+1; cond(ci).name = 'Load: ramp 50%';
cond(ci).setup = @(p) setfields(p, 'slip.enabled', false, ...
    'disturbance.enabled', true, 'disturbance.type', 'ramp', ...
    'disturbance.magnitude', 0.25, 'disturbance.ramp_rate', 0.05, ...
    'disturbance.start_time', 3.0);

% random: σ=0.10 N·m = 20% tau_max (was σ=0.03 = 6%)
ci=ci+1; cond(ci).name = 'Load: random 20%';
cond(ci).setup = @(p) setfields(p, 'slip.enabled', false, ...
    'disturbance.enabled', true, 'disturbance.type', 'random', ...
    'disturbance.random_sigma', 0.10, 'disturbance.start_time', 3.0);

% combined: step 0.15 + ramp 0.05/s + random σ=0.10
ci=ci+1; cond(ci).name = 'Load: combined';
cond(ci).setup = @(p) setfields(p, 'slip.enabled', false, ...
    'disturbance.enabled', true, 'disturbance.type', 'combined', ...
    'disturbance.magnitude', 0.15, 'disturbance.ramp_rate', 0.05, ...
    'disturbance.random_sigma', 0.10, 'disturbance.start_time', 3.0);

% --- 12. Worst case: slip + PPR 256 + combined load ---
ci=ci+1; cond(ci).name = 'Worst case';
cond(ci).setup = @(p) setfields(p, 'slip.enabled', true, ...
    'enc_ppr', 256, ...
    'disturbance.enabled', true, 'disturbance.type', 'combined', ...
    'disturbance.magnitude', 0.15, 'disturbance.ramp_rate', 0.05, ...
    'disturbance.random_sigma', 0.10, 'disturbance.start_time', 3.0);

n_cond = length(cond);

%% ===== STORAGE =====
mean_ss = zeros(n_cond, n_ctrl, n_traj);
std_ss  = zeros(n_cond, n_ctrl, n_traj);

fprintf('====================================================================\n');
fprintf('  M6 — DISTURBANCE & ROBUSTNESS (MULTI-TRIAL v3, N=%d)\n', N_TRIALS);
fprintf('====================================================================\n');
fprintf('  T_sim=%.0fs, dt=%.4fs, Seeds=%s\n', ...
        params_base.T_sim, params_base.dt, mat2str(SEEDS));
fprintf('  Total runs: %d conditions × %d ctrl × %d traj × %d trials = %d\n', ...
        n_cond, n_ctrl, n_traj, N_TRIALS, n_cond*n_ctrl*n_traj*N_TRIALS);
fprintf('  Changes from v2:\n');
fprintf('    - Encoder noise sweep → PPR sweep (1024/512/256/128)\n');
fprintf('    - Load disturbance: step 0.15, ramp 0.05/s, random sigma=0.10\n');
fprintf('    - Worst case: slip + PPR 256 + combined load\n');
fprintf('====================================================================\n\n');

%% ===== RUN ALL =====
run_count = 0;
tic;

for ci_cond = 1:n_cond
    fprintf('--- Condition %d/%d: %s ---\n', ci_cond, n_cond, cond(ci_cond).name);

    % Apply condition setup (always start from fresh params_base)
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
%%  M6.5 — COMPREHENSIVE COMPARISON TABLE
%% =====================================================================
fprintf('=============================================================================================\n');
fprintf('  M6.5 — COMPARISON TABLE: Mean ± Std (mm), N=%d trials per scenario\n', N_TRIALS);
fprintf('=============================================================================================\n\n');

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
fprintf('\n--- Degradation from Nominal (x factor of mean SS error) ---\n');
fprintf('%-18s |', 'Condition');
for ti = 1:n_traj
    fprintf(' %s_PID %s_ADR |', trajectories{ti}(1:min(3,end)), trajectories{ti}(1:min(3,end)));
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 110));

for ci_cond = 1:n_cond
    fprintf('%-18s |', cond(ci_cond).name);
    for ti = 1:n_traj
        pid_nom  = mean_ss(1, 1, ti);
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
        note = '<-- HIGH VARIANCE';
    end
    fprintf('%-18s | %6.1f%%  | %6.1f%%  | %s\n', cond(ci_cond).name, pid_cov, adrc_cov, note);
end

% --- Robustness ---
fprintf('\n--- Robustness: nominal vs worst case ---\n');
pid_nom   = mean(mean_ss(1, 1, :));
adrc_nom  = mean(mean_ss(1, 2, :));
pid_worst = mean(mean_ss(n_cond, 1, :));
adrc_worst= mean(mean_ss(n_cond, 2, :));
fprintf('  PID:  nominal %.1fmm -> worst %.1fmm (%.1fx degradation)\n', pid_nom, pid_worst, pid_worst/max(pid_nom,0.1));
fprintf('  ADRC: nominal %.1fmm -> worst %.1fmm (%.1fx degradation)\n', adrc_nom, adrc_worst, adrc_worst/max(adrc_nom,0.1));

% --- Per-trajectory breakdown for slip ---
fprintf('\n--- Wheel slip breakdown (condition 2) ---\n');
fprintf('  %-8s | %10s | %10s | %10s\n', 'Traj', 'PID(mm)', 'ADRC(mm)', 'ADRC impr');
fprintf('  %s\n', repmat('-', 1, 45));
for ti = 1:n_traj
    pm = mean_ss(2, 1, ti);
    am = mean_ss(2, 2, ti);
    imp = (pm - am) / max(pm, 0.01) * 100;
    fprintf('  %-8s | %8.1f±%.1f | %8.1f±%.1f | %+8.1f%%\n', ...
            trajectories{ti}, pm, std_ss(2,1,ti), am, std_ss(2,2,ti), imp);
end

% --- Per-trajectory breakdown for worst case ---
fprintf('\n--- Worst case breakdown (condition %d) ---\n', n_cond);
fprintf('  %-8s | %10s | %10s | %10s\n', 'Traj', 'PID(mm)', 'ADRC(mm)', 'ADRC impr');
fprintf('  %s\n', repmat('-', 1, 45));
for ti = 1:n_traj
    pm = mean_ss(n_cond, 1, ti);
    am = mean_ss(n_cond, 2, ti);
    imp = (pm - am) / max(pm, 0.01) * 100;
    fprintf('  %-8s | %8.1f±%.1f | %8.1f±%.1f | %+8.1f%%\n', ...
            trajectories{ti}, pm, std_ss(n_cond,1,ti), am, std_ss(n_cond,2,ti), imp);
end

fprintf('\n====================================================================\n');
fprintf('  M6 MULTI-TRIAL COMPARISON COMPLETE (v3)\n');
fprintf('====================================================================\n');
