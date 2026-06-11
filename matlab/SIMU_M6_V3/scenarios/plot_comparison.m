function fig = plot_comparison(log_pid, log_adrc, traj, m_pid, m_adrc, scenario_name)
% PLOT_COMPARISON Side-by-side PID vs ADRC dashboard for one scenario
%
% Layout: 4×2 panels
%   Row 1: XY trajectory | Position error(t) | Heading error(t) | Vel command
%   Row 2: Wheel speeds | Torque cmd | Body velocities | Metrics table
%
% Inputs:
%   log_pid, log_adrc  : logs from run_single_scenario_v2
%   traj               : trajectory struct (reference)
%   m_pid, m_adrc      : metrics structs
%   scenario_name      : string, used in figure title

    fig = figure('Name', scenario_name, 'Position', [50 50 1600 900], ...
                 'Color','w');

    t = log_pid.t_log;
    N = length(t);

    %% Row 1, Col 1: XY trajectory
    subplot(2,4,1); hold on; grid on;
    plot(traj.x_ref, traj.y_ref, 'k--', 'LineWidth', 1.5, 'DisplayName','Reference');
    plot(log_pid.x_log, log_pid.y_log,  'b-',  'LineWidth', 1.2, 'DisplayName','PID');
    plot(log_adrc.x_log, log_adrc.y_log, 'r-',  'LineWidth', 1.2, 'DisplayName','ADRC');
    plot(0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor','k', 'DisplayName','Start');
    xlabel('X (m)'); ylabel('Y (m)');
    title('XY Trajectory'); axis equal; legend('Location','best');

    %% Row 1, Col 2: Position error
    subplot(2,4,2); hold on; grid on;
    ex_p = log_pid.x_log  - traj.x_ref(1:N);
    ey_p = log_pid.y_log  - traj.y_ref(1:N);
    err_pid = 1000*sqrt(ex_p.^2 + ey_p.^2);
    ex_a = log_adrc.x_log - traj.x_ref(1:N);
    ey_a = log_adrc.y_log - traj.y_ref(1:N);
    err_adrc = 1000*sqrt(ex_a.^2 + ey_a.^2);
    plot(t, err_pid,  'b-', 'DisplayName','PID');
    plot(t, err_adrc, 'r-', 'DisplayName','ADRC');
    xline(traj.spec.t_hold + traj.spec.t_ramp, 'k:', 'ramp end');
    xlabel('Time (s)'); ylabel('Error (mm)');
    title(sprintf('Position Error | PID SS=%.1f ADRC SS=%.1f mm', ...
                  m_pid.rms_pos_ss, m_adrc.rms_pos_ss));
    legend('Location','best');

    %% Row 1, Col 3: Heading error
    subplot(2,4,3); hold on; grid on;
    th_err_p = rad2deg(wrap_angle(log_pid.theta_log  - traj.theta_ref(1:N)));
    th_err_a = rad2deg(wrap_angle(log_adrc.theta_log - traj.theta_ref(1:N)));
    plot(t, th_err_p, 'b-'); plot(t, th_err_a, 'r-');
    xlabel('Time (s)'); ylabel('Heading err (deg)');
    title(sprintf('Heading | PID RMS=%.2f° ADRC=%.2f°', m_pid.rms_theta, m_adrc.rms_theta));
    legend('PID','ADRC','Location','best');

    %% Row 1, Col 4: Velocity commands (outer loop output)
    subplot(2,4,4); hold on; grid on;
    plot(t, log_pid.vel_cmd_log(1,:),  'b-',  'DisplayName','PID v_x');
    plot(t, log_adrc.vel_cmd_log(1,:), 'r-',  'DisplayName','ADRC v_x');
    plot(t, log_pid.vel_cmd_log(3,:),  'b--', 'DisplayName','PID w_z');
    plot(t, log_adrc.vel_cmd_log(3,:), 'r--', 'DisplayName','ADRC w_z');
    plot(t, traj.vx_ref, 'k:', 'LineWidth', 1, 'DisplayName','v_x ref');
    xlabel('Time (s)'); ylabel('m/s, rad/s');
    title('Outer-loop velocity cmd'); legend('Location','best');

    %% Row 2, Col 1: Wheel speeds (wheel 1 only for clarity)
    subplot(2,4,5); hold on; grid on;
    plot(t, log_pid.omega_log(1,:),  'b-',  'DisplayName','PID \omega_1');
    plot(t, log_adrc.omega_log(1,:), 'r-',  'DisplayName','ADRC \omega_1');
    plot(t, log_pid.omega_log(2,:),  'b--', 'DisplayName','PID \omega_2');
    plot(t, log_adrc.omega_log(2,:), 'r--', 'DisplayName','ADRC \omega_2');
    xlabel('Time (s)'); ylabel('\omega (rad/s)');
    title('Wheel Speeds (W1, W2)'); legend('Location','best');

    %% Row 2, Col 2: Torque commands
    subplot(2,4,6); hold on; grid on;
    plot(t, log_pid.tau_log(1,:),  'b-');
    plot(t, log_adrc.tau_log(1,:), 'r-');
    if any(log_pid.tau_max_log ~= log_pid.tau_max_log(1))
        % Dynamic tau_max (battery fade)
        plot(t,  log_pid.tau_max_log, 'k:', 'LineWidth', 1.2);
        plot(t, -log_pid.tau_max_log, 'k:', 'LineWidth', 1.2);
    end
    xlabel('Time (s)'); ylabel('\tau (N·m)');
    title(sprintf('Torque W1 | max PID=%.3f ADRC=%.3f', m_pid.max_torque, m_adrc.max_torque));
    legend('PID','ADRC','Location','best');

    %% Row 2, Col 3: Body velocities (plant actual)
    subplot(2,4,7); hold on; grid on;
    plot(t, log_pid.vx_log,  'b-');
    plot(t, log_adrc.vx_log, 'r-');
    plot(t, log_pid.wz_log,  'b--');
    plot(t, log_adrc.wz_log, 'r--');
    plot(t, traj.vx_ref, 'k:', 'LineWidth', 1);
    plot(t, traj.wz_ref, 'k-.', 'LineWidth', 1);
    xlabel('Time (s)'); ylabel('m/s, rad/s');
    title('Plant body velocity');
    legend('PID v_x','ADRC v_x','PID w_z','ADRC w_z','ref v_x','ref w_z','Location','best');

    %% Row 2, Col 4: Metrics table
    subplot(2,4,8); axis off;
    text_cell = {
        sprintf('\\bfScenario: %s\\rm', strrep(scenario_name,'_','\_'));
        sprintf('Trajectory: %s', traj.spec.type);
        '';
        sprintf('%-15s %10s %10s', 'Metric', 'PID', 'ADRC');
        repmat('-', 1, 40);
        sprintf('%-15s %10.2f %10.2f', 'RMS full (mm)', m_pid.rms_pos_full, m_adrc.rms_pos_full);
        sprintf('%-15s %10.2f %10.2f', 'RMS SS (mm)',   m_pid.rms_pos_ss,   m_adrc.rms_pos_ss);
        sprintf('%-15s %10.2f %10.2f', 'Peak err (mm)', m_pid.max_pos_err,  m_adrc.max_pos_err);
        sprintf('%-15s %10.2f %10.2f', 't_peak (s)',    m_pid.t_peak,       m_adrc.t_peak);
        sprintf('%-15s %10.2f %10.2f', 'RMS θ (°)',     m_pid.rms_theta,    m_adrc.rms_theta);
        sprintf('%-15s %10s %10s',     'Settle (s)', ...
                format_optional(m_pid.settle_time), format_optional(m_adrc.settle_time));
        sprintf('%-15s %10.3f %10.3f', 'Max τ (N·m)',   m_pid.max_torque,   m_adrc.max_torque);
        sprintf('%-15s %10.1f %10.1f', 'Sat %',         m_pid.sat_pct,      m_adrc.sat_pct);
        sprintf('%-15s %10d %10d',     'Slip events', m_pid.slip_events, m_adrc.slip_events);
        '';
        sprintf('ADRC vs PID SS: %+.1f%%', ...
                100*(m_pid.rms_pos_ss - m_adrc.rms_pos_ss)/max(m_pid.rms_pos_ss, 1e-6));
    };
    text(0.05, 0.95, text_cell, 'FontName','Courier', 'FontSize', 9, ...
         'VerticalAlignment','top', 'Units','normalized');

    %% Overall title
    sgtitle(sprintf('HIL Scenario: %s — PID (blue) vs ADRC (red)', ...
                    strrep(scenario_name,'_','\_')), ...
            'FontSize', 14, 'FontWeight', 'bold');
end

function s = format_optional(val)
    if isnan(val)
        s = 'N/A';
    else
        s = sprintf('%.2f', val);
    end
end

function a = wrap_angle(a)
    a = mod(a + pi, 2*pi) - pi;
end
