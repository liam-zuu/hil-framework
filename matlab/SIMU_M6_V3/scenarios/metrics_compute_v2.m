function m = metrics_compute_v2(log, traj, params)
% METRICS_COMPUTE_V2  Metrics split by what each controller layer is responsible for.
%
% metrics_compute.m mixes three effects into one "position error":
%   (1) inner wheel-speed loop (PID / ADRC: the controller under test),
%   (2) outer position loop, which only sees the ESTIMATED pose,
%   (3) dead-reckoning drift of pose_estimator (true pose vs estimated pose).
% (3) cannot be corrected by any controller, so it hides PID/ADRC differences.
% This function reports the three separately, plus control effort/smoothness.
%
% Groups (all over the steady-state window t > t_ramp_end + 2 s, like v1,
% unless the name says _full):
%   Inner loop     rms_w_err      RMS(omega_ref - omega_true), 4 wheels pooled [rad/s]
%                  rms_w_err_est  RMS(omega_ref - omega_est), what the controller sees [rad/s]
%   Effort         rms_tau        RMS applied torque [N.m]
%                  tv_tau         total variation sum|d tau| / duration, 4 wheels summed [N.m/s]
%                                 (chatter indicator: smooth signal -> small)
%                  sat_pct        % samples with |tau| > 0.98 tau_max
%   Position       rms_e_track    RMS |true - ref|   (what v1 calls rms_pos_ss) [mm]
%                  rms_e_ctrl     RMS |est  - ref|   (error the outer loop can act on) [mm]
%                  rms_e_drift    RMS |true - est|   (estimator drift, not controllable) [mm]
%                  e_drift_end    |true - est| at the last sample [mm]
%                  drift_share    rms_e_drift / rms_e_track [%]
%
% omega_ref is not logged; it is rebuilt from vel_cmd_log with the same
% inverse kinematics as run_single_scenario_v2.m.

    t  = log.t_log;
    N  = length(t);
    dt = t(2) - t(1);
    t_ramp_end = traj.spec.t_hold + traj.spec.t_ramp;
    idx = t > t_ramp_end + 2.0;
    if ~any(idx), idx = true(1, N); end
    m.ss_window = [t(find(idx, 1, 'first')), t(end)];

    %% Inner loop: wheel-speed tracking
    r = params.r;  L = params.lx + params.ly;
    v = log.vel_cmd_log;                       % [vx; vy; wz] x N
    w_ref = (1/r) * [v(1,:) - v(2,:) - L*v(3,:);
                     v(1,:) + v(2,:) + L*v(3,:);
                     v(1,:) + v(2,:) - L*v(3,:);
                     v(1,:) - v(2,:) + L*v(3,:)];
    e_w     = w_ref - log.omega_log;
    e_w_est = w_ref - log.omega_est_log;
    m.rms_w_err     = sqrt(mean(e_w(:, idx).^2,     'all'));
    m.rms_w_err_est = sqrt(mean(e_w_est(:, idx).^2, 'all'));

    %% Effort / smoothness
    tau = log.tau_log;
    m.rms_tau = sqrt(mean(tau(:, idx).^2, 'all'));
    dur = sum(idx) * dt;
    m.tv_tau  = sum(abs(diff(tau(:, idx), 1, 2)), 'all') / dur;
    m.sat_pct = 100 * mean(abs(tau(:, idx)) > 0.98 * params.tau_max, 'all');

    %% Position: decomposition
    xr = traj.x_ref(1:N);  yr = traj.y_ref(1:N);
    xt = log.x_log;         yt = log.y_log;
    xe = log.pos_est_log(1,:);  ye = log.pos_est_log(2,:);
    e_track = hypot(xt - xr, yt - yr);
    e_ctrl  = hypot(xe - xr, ye - yr);
    e_drift = hypot(xt - xe, yt - ye);
    m.rms_e_track = 1000 * sqrt(mean(e_track(idx).^2));
    m.rms_e_ctrl  = 1000 * sqrt(mean(e_ctrl(idx).^2));
    m.rms_e_drift = 1000 * sqrt(mean(e_drift(idx).^2));
    m.e_drift_end = 1000 * e_drift(end);
    m.drift_share = 100 * m.rms_e_drift / max(m.rms_e_track, eps);

    %% Time series for plotting (kept small: no extra copies of big logs)
    m.ts.e_track = e_track;  m.ts.e_ctrl = e_ctrl;  m.ts.e_drift = e_drift;
    m.ts.e_w_rms = sqrt(mean(e_w.^2, 1));     % RMS over 4 wheels, per sample
end
