%% RUN_M6_DISTURBANCE  M6 — Disturbance & Robustness: Full comparison suite.
%
% Runs PID vs ADRC across multiple disturbance conditions:
%   1. Nominal (baseline — should match M5.2 results)
%   2. Wheel slip enabled
%   3. Sensor noise sweep (encoder σ: 0.02→0.2, IMU: 1×→5×)
%   4. Load torque disturbance (step, ramp, random, combined)
%
% For each condition × 4 trajectories × 2 controllers = comprehensive table.
% Final analysis: when ADRC outperforms PID and by how much.
%
% Steps covered: M6.1 (slip in plant), M6.2 (slip test), M6.3 (noise sweep),
%                M6.4 (load disturbance), M6.5 (comparison table), M6.6 (analysis)

clear; clc; close all;

%% ===== CONFIGURATION =====
controllers  = {'pid', 'adrc'};
trajectories = {'line', 'circle', 'square', 'figure8'};

n_ctrl = length(controllers);
n_traj = length(trajectories);

% Store all results: results{condition_idx}.(ctrl).(traj) = res struct
all_results = {};
condition_names = {};
condition_idx = 0;

params_base = params_mecanum();  % baseline M5.2 parameters
params_base.T_sim = 10;  % 5s sufficient for SS convergence (settling ~2s)

fprintf('====================================================================\n');
fprintf('  M6 — DISTURBANCE & ROBUSTNESS: PID vs ADRC COMPARISON\n');
fprintf('====================================================================\n');
fprintf('  Simulation: T=%.0fs, dt=%.4fs, N=%d steps\n', ...
        params_base.T_sim, params_base.dt, round(params_base.T_sim/params_base.dt));
fprintf('====================================================================\n\n');

%% ===== CONDITION 1: NOMINAL (baseline, no disturbance) =====
fprintf('--- Condition 1: NOMINAL (M5.2 baseline) ---\n');
condition_idx = condition_idx + 1;
condition_names{condition_idx} = 'Nominal';
params = params_base;
params.slip.enabled = false;
params.disturbance.enabled = false;

for ci = 1:n_ctrl
    for ti = 1:n_traj
        fprintf('  %s / %s ... ', upper(controllers{ci}), trajectories{ti});
        res = run_single_scenario(controllers{ci}, trajectories{ti}, params);
        all_results{condition_idx}.(controllers{ci}).(trajectories{ti}) = res;
        fprintf('SS=%.1fmm\n', res.rms_pos_ss);
    end
end

%% ===== CONDITION 2: WHEEL SLIP =====
fprintf('\n--- Condition 2: WHEEL SLIP (mu_s=0.8, mu_k=0.5, p_spont=0.002) ---\n');
condition_idx = condition_idx + 1;
condition_names{condition_idx} = 'Wheel slip';
params = params_base;
params.slip.enabled = true;

for ci = 1:n_ctrl
    for ti = 1:n_traj
        fprintf('  %s / %s ... ', upper(controllers{ci}), trajectories{ti});
        res = run_single_scenario(controllers{ci}, trajectories{ti}, params);
        all_results{condition_idx}.(controllers{ci}).(trajectories{ti}) = res;
        fprintf('SS=%.1fmm, slips=%d\n', res.rms_pos_ss, res.slip_events);
    end
end

%% ===== CONDITION 3: ENCODER NOISE ×2.5 (σ=0.05) =====
fprintf('\n--- Condition 3: ENCODER NOISE x2.5 (sigma=0.05) ---\n');
condition_idx = condition_idx + 1;
condition_names{condition_idx} = 'Enc noise x2.5';
params = params_base;
params.enc_noise_sigma = 0.05;

for ci = 1:n_ctrl
    for ti = 1:n_traj
        fprintf('  %s / %s ... ', upper(controllers{ci}), trajectories{ti});
        res = run_single_scenario(controllers{ci}, trajectories{ti}, params);
        all_results{condition_idx}.(controllers{ci}).(trajectories{ti}) = res;
        fprintf('SS=%.1fmm\n', res.rms_pos_ss);
    end
end

%% ===== CONDITION 4: ENCODER NOISE ×5 (σ=0.10) =====
fprintf('\n--- Condition 4: ENCODER NOISE x5 (sigma=0.10) ---\n');
condition_idx = condition_idx + 1;
condition_names{condition_idx} = 'Enc noise x5';
params = params_base;
params.enc_noise_sigma = 0.10;

for ci = 1:n_ctrl
    for ti = 1:n_traj
        fprintf('  %s / %s ... ', upper(controllers{ci}), trajectories{ti});
        res = run_single_scenario(controllers{ci}, trajectories{ti}, params);
        all_results{condition_idx}.(controllers{ci}).(trajectories{ti}) = res;
        fprintf('SS=%.1fmm\n', res.rms_pos_ss);
    end
end

%% ===== CONDITION 5: ENCODER NOISE ×10 (σ=0.20) =====
fprintf('\n--- Condition 5: ENCODER NOISE x10 (sigma=0.20) ---\n');
condition_idx = condition_idx + 1;
condition_names{condition_idx} = 'Enc noise x10';
params = params_base;
params.enc_noise_sigma = 0.20;

for ci = 1:n_ctrl
    for ti = 1:n_traj
        fprintf('  %s / %s ... ', upper(controllers{ci}), trajectories{ti});
        res = run_single_scenario(controllers{ci}, trajectories{ti}, params);
        all_results{condition_idx}.(controllers{ci}).(trajectories{ti}) = res;
        fprintf('SS=%.1fmm\n', res.rms_pos_ss);
    end
end

%% ===== CONDITION 6: IMU NOISE ×3 =====
fprintf('\n--- Condition 6: IMU NOISE x3 ---\n');
condition_idx = condition_idx + 1;
condition_names{condition_idx} = 'IMU noise x3';
params = params_base;
params.imu_accel_noise = params_base.imu_accel_noise * 3;
params.imu_gyro_noise  = params_base.imu_gyro_noise * 3;
params.imu_bias_drift  = params_base.imu_bias_drift * 3;

for ci = 1:n_ctrl
    for ti = 1:n_traj
        fprintf('  %s / %s ... ', upper(controllers{ci}), trajectories{ti});
        res = run_single_scenario(controllers{ci}, trajectories{ti}, params);
        all_results{condition_idx}.(controllers{ci}).(trajectories{ti}) = res;
        fprintf('SS=%.1fmm\n', res.rms_pos_ss);
    end
end

%% ===== CONDITION 7: IMU NOISE ×5 =====
fprintf('\n--- Condition 7: IMU NOISE x5 ---\n');
condition_idx = condition_idx + 1;
condition_names{condition_idx} = 'IMU noise x5';
params = params_base;
params.imu_accel_noise = params_base.imu_accel_noise * 5;
params.imu_gyro_noise  = params_base.imu_gyro_noise * 5;
params.imu_bias_drift  = params_base.imu_bias_drift * 5;

for ci = 1:n_ctrl
    for ti = 1:n_traj
        fprintf('  %s / %s ... ', upper(controllers{ci}), trajectories{ti});
        res = run_single_scenario(controllers{ci}, trajectories{ti}, params);
        all_results{condition_idx}.(controllers{ci}).(trajectories{ti}) = res;
        fprintf('SS=%.1fmm\n', res.rms_pos_ss);
    end
end

%% ===== CONDITION 8: LOAD DISTURBANCE — STEP =====
fprintf('\n--- Condition 8: LOAD DISTURBANCE — STEP (0.05 N·m at t=3s) ---\n');
condition_idx = condition_idx + 1;
condition_names{condition_idx} = 'Load: step';
params = params_base;
params.disturbance.enabled   = true;
params.disturbance.type      = 'step';
params.disturbance.magnitude = 0.05;
params.disturbance.start_time = 3.0;

for ci = 1:n_ctrl
    for ti = 1:n_traj
        fprintf('  %s / %s ... ', upper(controllers{ci}), trajectories{ti});
        res = run_single_scenario(controllers{ci}, trajectories{ti}, params);
        all_results{condition_idx}.(controllers{ci}).(trajectories{ti}) = res;
        fprintf('SS=%.1fmm\n', res.rms_pos_ss);
    end
end

%% ===== CONDITION 9: LOAD DISTURBANCE — RAMP =====
fprintf('\n--- Condition 9: LOAD DISTURBANCE — RAMP (0.02 N·m/s from t=3s) ---\n');
condition_idx = condition_idx + 1;
condition_names{condition_idx} = 'Load: ramp';
params = params_base;
params.disturbance.enabled    = true;
params.disturbance.type       = 'ramp';
params.disturbance.magnitude  = 0.1;   % max ramp value
params.disturbance.ramp_rate  = 0.02;
params.disturbance.start_time = 3.0;

for ci = 1:n_ctrl
    for ti = 1:n_traj
        fprintf('  %s / %s ... ', upper(controllers{ci}), trajectories{ti});
        res = run_single_scenario(controllers{ci}, trajectories{ti}, params);
        all_results{condition_idx}.(controllers{ci}).(trajectories{ti}) = res;
        fprintf('SS=%.1fmm\n', res.rms_pos_ss);
    end
end

%% ===== CONDITION 10: LOAD DISTURBANCE — RANDOM =====
fprintf('\n--- Condition 10: LOAD DISTURBANCE — RANDOM (σ=0.03 N·m) ---\n');
condition_idx = condition_idx + 1;
condition_names{condition_idx} = 'Load: random';
params = params_base;
params.disturbance.enabled      = true;
params.disturbance.type         = 'random';
params.disturbance.random_sigma = 0.03;
params.disturbance.start_time   = 3.0;

for ci = 1:n_ctrl
    for ti = 1:n_traj
        fprintf('  %s / %s ... ', upper(controllers{ci}), trajectories{ti});
        res = run_single_scenario(controllers{ci}, trajectories{ti}, params);
        all_results{condition_idx}.(controllers{ci}).(trajectories{ti}) = res;
        fprintf('SS=%.1fmm\n', res.rms_pos_ss);
    end
end

%% ===== CONDITION 11: LOAD DISTURBANCE — COMBINED =====
fprintf('\n--- Condition 11: LOAD DISTURBANCE — COMBINED (step+ramp+random) ---\n');
condition_idx = condition_idx + 1;
condition_names{condition_idx} = 'Load: combined';
params = params_base;
params.disturbance.enabled      = true;
params.disturbance.type         = 'combined';
params.disturbance.magnitude    = 0.05;
params.disturbance.ramp_rate    = 0.02;
params.disturbance.random_sigma = 0.03;
params.disturbance.start_time   = 3.0;

for ci = 1:n_ctrl
    for ti = 1:n_traj
        fprintf('  %s / %s ... ', upper(controllers{ci}), trajectories{ti});
        res = run_single_scenario(controllers{ci}, trajectories{ti}, params);
        all_results{condition_idx}.(controllers{ci}).(trajectories{ti}) = res;
        fprintf('SS=%.1fmm\n', res.rms_pos_ss);
    end
end

%% ===== CONDITION 12: WORST CASE — SLIP + NOISE + DISTURBANCE =====
fprintf('\n--- Condition 12: WORST CASE (slip + enc_noise x5 + load combined) ---\n');
condition_idx = condition_idx + 1;
condition_names{condition_idx} = 'Worst case';
params = params_base;
params.slip.enabled = true;
params.enc_noise_sigma = 0.10;
params.disturbance.enabled      = true;
params.disturbance.type         = 'combined';
params.disturbance.magnitude    = 0.05;
params.disturbance.ramp_rate    = 0.02;
params.disturbance.random_sigma = 0.03;
params.disturbance.start_time   = 3.0;

for ci = 1:n_ctrl
    for ti = 1:n_traj
        fprintf('  %s / %s ... ', upper(controllers{ci}), trajectories{ti});
        res = run_single_scenario(controllers{ci}, trajectories{ti}, params);
        all_results{condition_idx}.(controllers{ci}).(trajectories{ti}) = res;
        fprintf('SS=%.1fmm\n', res.rms_pos_ss);
    end
end

n_cond = condition_idx;

%% =====================================================================
%%  M6.5 — COMPREHENSIVE COMPARISON TABLE
%% =====================================================================
fprintf('\n\n');
fprintf('=============================================================================================\n');
fprintf('  M6.5 — COMPREHENSIVE COMPARISON: PID vs ADRC × %d Conditions × %d Trajectories\n', n_cond, n_traj);
fprintf('=============================================================================================\n');

% Table 1: Steady-state RMS position error (mm)
fprintf('\n--- TABLE 1: Steady-State RMS Position Error (mm) ---\n');
fprintf('%-18s |', 'Condition');
for ti = 1:n_traj
    fprintf(' %s_PID %s_ADRC |', trajectories{ti}(1:min(3,end)), trajectories{ti}(1:min(3,end)));
end
fprintf(' Avg_PID Avg_ADRC  ADRC%%\n');
fprintf('%s\n', repmat('-', 1, 120));

adrc_wins_total = 0;
adrc_wins_count = 0;
improvement_sum = 0;
improvement_count = 0;

for ci_cond = 1:n_cond
    fprintf('%-18s |', condition_names{ci_cond});
    pid_avg = 0;
    adrc_avg = 0;
    for ti = 1:n_traj
        pid_ss  = all_results{ci_cond}.pid.(trajectories{ti}).rms_pos_ss;
        adrc_ss = all_results{ci_cond}.adrc.(trajectories{ti}).rms_pos_ss;
        fprintf(' %6.1f  %6.1f  |', pid_ss, adrc_ss);
        pid_avg  = pid_avg + pid_ss;
        adrc_avg = adrc_avg + adrc_ss;

        % Track ADRC wins
        if adrc_ss < pid_ss
            adrc_wins_total = adrc_wins_total + 1;
        end
        adrc_wins_count = adrc_wins_count + 1;
        if pid_ss > 0
            improvement_sum = improvement_sum + (pid_ss - adrc_ss) / pid_ss * 100;
            improvement_count = improvement_count + 1;
        end
    end
    pid_avg  = pid_avg / n_traj;
    adrc_avg = adrc_avg / n_traj;
    if pid_avg > 0
        imp = (pid_avg - adrc_avg) / pid_avg * 100;
    else
        imp = 0;
    end
    fprintf(' %6.1f   %6.1f  %+5.1f%%\n', pid_avg, adrc_avg, imp);
end
fprintf('%s\n', repmat('-', 1, 120));

% Table 2: Degradation from nominal
fprintf('\n--- TABLE 2: Degradation from Nominal (×factor of SS error increase) ---\n');
fprintf('%-18s |', 'Condition');
for ti = 1:n_traj
    fprintf(' %s_PID %s_ADRC |', trajectories{ti}(1:min(3,end)), trajectories{ti}(1:min(3,end)));
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 100));

for ci_cond = 1:n_cond
    fprintf('%-18s |', condition_names{ci_cond});
    for ti = 1:n_traj
        pid_nom  = all_results{1}.pid.(trajectories{ti}).rms_pos_ss;
        adrc_nom = all_results{1}.adrc.(trajectories{ti}).rms_pos_ss;
        pid_cur  = all_results{ci_cond}.pid.(trajectories{ti}).rms_pos_ss;
        adrc_cur = all_results{ci_cond}.adrc.(trajectories{ti}).rms_pos_ss;

        if pid_nom > 0
            pid_deg = pid_cur / pid_nom;
        else
            pid_deg = 1;
        end
        if adrc_nom > 0
            adrc_deg = adrc_cur / adrc_nom;
        else
            adrc_deg = 1;
        end
        fprintf(' %5.1fx  %5.1fx  |', pid_deg, adrc_deg);
    end
    fprintf('\n');
end
fprintf('%s\n', repmat('-', 1, 100));

%% =====================================================================
%%  M6.6 — CONDITION ANALYSIS
%% =====================================================================
fprintf('\n\n');
fprintf('=============================================================================================\n');
fprintf('  M6.6 — ANALYSIS: When does ADRC outperform PID?\n');
fprintf('=============================================================================================\n');

fprintf('\n--- Per-condition ADRC advantage (avg across trajectories) ---\n');
fprintf('%-18s | %10s | %10s | %10s | %s\n', 'Condition', 'PID_avg', 'ADRC_avg', 'Impr.', 'Winner');
fprintf('%s\n', repmat('-', 1, 75));

for ci_cond = 1:n_cond
    pid_avg = 0;
    adrc_avg = 0;
    for ti = 1:n_traj
        pid_avg  = pid_avg  + all_results{ci_cond}.pid.(trajectories{ti}).rms_pos_ss;
        adrc_avg = adrc_avg + all_results{ci_cond}.adrc.(trajectories{ti}).rms_pos_ss;
    end
    pid_avg  = pid_avg / n_traj;
    adrc_avg = adrc_avg / n_traj;
    if pid_avg > 0
        imp = (pid_avg - adrc_avg) / pid_avg * 100;
    else
        imp = 0;
    end
    if imp > 2
        winner = 'ADRC';
    elseif imp < -2
        winner = 'PID';
    else
        winner = 'TIE';
    end
    fprintf('%-18s | %8.1f mm | %8.1f mm | %+8.1f%% | %s\n', ...
            condition_names{ci_cond}, pid_avg, adrc_avg, imp, winner);
end
fprintf('%s\n', repmat('-', 1, 75));

fprintf('\n--- Summary statistics ---\n');
fprintf('Total scenario pairs: %d (= %d conditions × %d trajectories)\n', ...
        adrc_wins_count, n_cond, n_traj);
fprintf('ADRC wins: %d / %d (%.1f%%)\n', adrc_wins_total, adrc_wins_count, ...
        adrc_wins_total/adrc_wins_count*100);
if improvement_count > 0
    fprintf('Average ADRC improvement: %+.1f%%\n', improvement_sum / improvement_count);
end

fprintf('\n--- Key findings ---\n');

% Find condition with largest ADRC advantage
best_imp = -inf;
best_cond = '';
worst_imp = inf;
worst_cond = '';
for ci_cond = 1:n_cond
    pid_avg = 0; adrc_avg = 0;
    for ti = 1:n_traj
        pid_avg  = pid_avg  + all_results{ci_cond}.pid.(trajectories{ti}).rms_pos_ss;
        adrc_avg = adrc_avg + all_results{ci_cond}.adrc.(trajectories{ti}).rms_pos_ss;
    end
    pid_avg = pid_avg / n_traj;
    adrc_avg = adrc_avg / n_traj;
    imp = (pid_avg - adrc_avg) / max(pid_avg, 0.01) * 100;
    if imp > best_imp
        best_imp = imp; best_cond = condition_names{ci_cond};
    end
    if imp < worst_imp
        worst_imp = imp; worst_cond = condition_names{ci_cond};
    end
end
fprintf('  ADRC strongest advantage: "%s" (%+.1f%%)\n', best_cond, best_imp);
fprintf('  ADRC weakest / PID better: "%s" (%+.1f%%)\n', worst_cond, worst_imp);

% Robustness: worst-case degradation
fprintf('\n--- Robustness: worst-case error under all disturbances ---\n');
fprintf('%-6s | %10s | %10s\n', 'Ctrl', 'Nominal', 'Worst case');
fprintf('%s\n', repmat('-', 1, 35));
for ci = 1:n_ctrl
    nom_avg = 0; worst_avg = 0;
    for ti = 1:n_traj
        nom_avg   = nom_avg   + all_results{1}.pid.(trajectories{ti}).rms_pos_ss;
        worst_avg = worst_avg + all_results{n_cond}.(controllers{ci}).(trajectories{ti}).rms_pos_ss;
    end
    nom_avg = nom_avg / n_traj;
    worst_avg = worst_avg / n_traj;
    fprintf('%-6s | %8.1f mm | %8.1f mm (%.1fx)\n', upper(controllers{ci}), nom_avg, worst_avg, worst_avg/max(nom_avg,0.01));
end

fprintf('\n====================================================================\n');
fprintf('  M6 COMPARISON COMPLETE\n');
fprintf('====================================================================\n');
