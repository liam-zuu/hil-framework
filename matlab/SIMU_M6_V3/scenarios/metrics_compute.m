function m = metrics_compute(log, traj, params)
% METRICS_COMPUTE Compute performance metrics from simulation log
%
% Inputs:
%   log   : simulation log struct (from run_single_scenario_v2)
%   traj  : trajectory reference (from trajectory_generator_v2)
%   params: parameter struct
%
% Output m struct contains:
%   .rms_pos_full      : RMS position error over full simulation (mm)
%   .rms_pos_ss        : RMS position error in steady state (mm)
%   .max_pos_err       : peak position error (mm)
%   .t_peak            : time of peak error (s)
%   .rms_theta         : RMS heading error (deg)
%   .max_theta_err     : peak heading error (deg)
%   .settle_time       : time to reach <threshold (s)
%   .max_torque        : max |tau| across all wheels (N·m)
%   .sat_pct           : % timesteps with |tau| near tau_max
%   .slip_events       : total slip flags (if slip_detector active)
%   .sync_fails        : GPIO sync failure count
%   .control_effort    : sum of |tau| * dt (N·m·s) — rough energy proxy

    t = log.t_log;
    dt = t(2) - t(1);
    N  = length(t);
    t_ramp_end = traj.spec.t_hold + traj.spec.t_ramp;

    % Position errors
    ex = log.x_log - traj.x_ref(1:N);
    ey = log.y_log - traj.y_ref(1:N);
    pos_err = sqrt(ex.^2 + ey.^2);          % meters
    m.rms_pos_full = 1000 * sqrt(mean(pos_err.^2));
    idx_ss = t > t_ramp_end + 2.0;          % 2s buffer after ramp end
    if any(idx_ss)
        m.rms_pos_ss = 1000 * sqrt(mean(pos_err(idx_ss).^2));
    else
        m.rms_pos_ss = m.rms_pos_full;
    end
    [m.max_pos_err, i_peak] = max(pos_err);
    m.max_pos_err = 1000 * m.max_pos_err;
    m.t_peak = t(i_peak);

    % Heading errors
    theta_err = wrap_angle(log.theta_log - traj.theta_ref(1:N));
    m.rms_theta      = rad2deg(sqrt(mean(theta_err.^2)));
    m.max_theta_err  = rad2deg(max(abs(theta_err)));

    % Settle time (pos_err < 10 mm after ramp end)
    threshold_m = 0.010;
    idx_after_ramp = find(t > t_ramp_end);
    settled = find(pos_err(idx_after_ramp) < threshold_m, 1, 'first');
    if ~isempty(settled)
        m.settle_time = t(idx_after_ramp(settled)) - t_ramp_end;
    else
        m.settle_time = NaN;
    end

    % Torque / saturation
    tau_abs = abs(log.tau_log);
    m.max_torque = max(tau_abs(:));
    tau_max = params.tau_max;
    m.sat_pct = 100 * sum(tau_abs(:) > 0.98*tau_max) / numel(tau_abs);

    % Control effort (rough energy proxy)
    m.control_effort = sum(tau_abs(:)) * dt;

    % Slip / sync
    if isfield(log,'slip_log')
        m.slip_events = sum(log.slip_log(:));
    else
        m.slip_events = 0;
    end
    if isfield(log,'sync_fail_count')
        m.sync_fails = log.sync_fail_count;
    else
        m.sync_fails = 0;
    end
end

function a = wrap_angle(a)
    a = mod(a + pi, 2*pi) - pi;
end
