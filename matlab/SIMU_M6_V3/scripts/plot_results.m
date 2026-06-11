%% PLOT_RESULTS  Visualize simulation results — M5 Full Integration.
%
% Expects in workspace: x_hist, t_hist, log, traj, controller_type, traj_type,
%                        rms_err, rms_ss, params

figure('Name', sprintf('HIL M5 — %s / %s', upper(controller_type), traj_type), ...
       'Position', [50 50 1400 900]);

N_plot = min(size(log.omega_ref, 2), length(t_hist)-1);
t_log = t_hist(1:N_plot);
colors = {'b','r','g','m'};

%% 1. XY Trajectory (actual vs reference vs estimated)
subplot(2,4,1);
plot(traj.x_ref, traj.y_ref, 'r--', 'LineWidth', 1); hold on;
plot(x_hist(1,:), x_hist(2,:), 'b-', 'LineWidth', 1.5);
plot(log.pose_est(1,1:N_plot), log.pose_est(2,1:N_plot), 'g:', 'LineWidth', 1);
plot(x_hist(1,1), x_hist(2,1), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
xlabel('X (m)'); ylabel('Y (m)');
title('XY Trajectory');
legend('Reference', 'Actual', 'Estimated', 'Start', 'Location', 'best');
axis equal; grid on;

%% 2. Heading
subplot(2,4,2);
plot(traj.t(1:min(end,length(t_hist))), ...
     rad2deg(traj.theta_ref(1:min(end,length(t_hist)))), 'r--', 'LineWidth', 1); hold on;
plot(t_hist, rad2deg(x_hist(3,:)), 'b-', 'LineWidth', 1);
plot(t_log, rad2deg(log.pose_est(3,1:N_plot)), 'g:', 'LineWidth', 1);
xlabel('Time (s)'); ylabel('\theta (deg)');
title('Heading');
legend('Reference', 'Actual', 'Estimated', 'Location', 'best');
grid on;

%% 3. Position Error
subplot(2,4,3);
N_err = min(length(traj.x_ref), length(t_hist));
ex = x_hist(1,1:N_err) - traj.x_ref(1:N_err);
ey = x_hist(2,1:N_err) - traj.y_ref(1:N_err);
e_pos = sqrt(ex.^2 + ey.^2);
plot(t_hist(1:N_err), e_pos * 1000, 'b-', 'LineWidth', 1); hold on;
% Odometry drift: estimated vs actual
e_odom = sqrt((log.pose_est(1,1:N_plot) - x_hist(1,1:N_plot)).^2 + ...
              (log.pose_est(2,1:N_plot) - x_hist(2,1:N_plot)).^2);
plot(t_log, e_odom * 1000, 'g--', 'LineWidth', 0.8);
yline(rms_err, 'r--', sprintf('RMS=%.1fmm', rms_err), 'LineWidth', 0.8);
xlabel('Time (s)'); ylabel('Error (mm)');
title(sprintf('Position Error (RMS: %.1f mm, SS: %.1f mm)', rms_err, rms_ss));
legend('Tracking error', 'Odometry drift', 'Location', 'best');
grid on;

%% 4. Body Velocity Commands (outer loop output)
subplot(2,4,4);
plot(t_log, log.vel_cmd(1,1:N_plot), 'b-', 'LineWidth', 1); hold on;
plot(t_log, log.vel_cmd(2,1:N_plot), 'r-', 'LineWidth', 1);
plot(t_log, log.vel_cmd(3,1:N_plot), 'g-', 'LineWidth', 1);
% Plot feedforward reference for comparison
N_ref = min(length(traj.vx_ref), N_plot);
plot(t_log(1:N_ref), traj.vy_ref(1:N_ref), 'b--', 'LineWidth', 0.5);
plot(t_log(1:N_ref), traj.vx_ref(1:N_ref), 'r--', 'LineWidth', 0.5);
plot(t_log(1:N_ref), traj.wz_ref(1:N_ref), 'g--', 'LineWidth', 0.5);
xlabel('Time (s)'); ylabel('Velocity');
title('Body Vel Commands (solid) vs Ref (dashed)');
legend('vx_{cmd}', 'vy_{cmd}', 'wz_{cmd}', 'Location', 'best');
grid on;

%% 5. Wheel Speeds (estimated vs reference)
subplot(2,4,5);
for w = 1:4
    plot(t_log, log.omega_est(w,1:N_plot), [colors{w} '-'], 'LineWidth', 0.8); hold on;
    plot(t_log, log.omega_ref(w,1:N_plot), [colors{w} '--'], 'LineWidth', 0.5);
end
xlabel('Time (s)'); ylabel('\omega (rad/s)');
title('Wheel Speeds (solid=est, dashed=ref)');
legend('W1','W1_{ref}','W2','W2_{ref}','W3','W3_{ref}','W4','W4_{ref}', ...
       'Location', 'best');
grid on;

%% 6. Torque Commands
subplot(2,4,6);
for w = 1:4
    plot(t_log, log.tau_cmd(w,1:N_plot), [colors{w} '-'], 'LineWidth', 0.8); hold on;
end
yline(params.tau_max, 'k--', '\tau_{max}');
yline(-params.tau_max, 'k--');
xlabel('Time (s)'); ylabel('\tau (N\cdotm)');
title('Torque Commands');
legend('W1','W2','W3','W4', 'Location', 'best');
grid on;

%% 7. Body Velocities (plant ground truth)
subplot(2,4,7);
plot(t_hist, x_hist(4,:), 'b-', 'LineWidth', 1); hold on;
plot(t_hist, x_hist(5,:), 'r-', 'LineWidth', 1);
plot(t_hist, x_hist(6,:), 'g-', 'LineWidth', 1);
xlabel('Time (s)'); ylabel('Velocity');
title('Plant Body Velocities');
legend('v_x (m/s)', 'v_y (m/s)', '\omega_z (rad/s)', 'Location', 'best');
grid on;

%% 8. Heading Error
subplot(2,4,8);
N_th = min(length(traj.theta_ref), length(t_hist));
e_theta = x_hist(3,1:N_th) - traj.theta_ref(1:N_th);
% Wrap heading error to [-pi, pi]
e_theta = mod(e_theta + pi, 2*pi) - pi;
plot(t_hist(1:N_th), rad2deg(e_theta), 'b-', 'LineWidth', 1);
xlabel('Time (s)'); ylabel('Error (deg)');
title(sprintf('Heading Error (RMS: %.2f deg)', rms(e_theta)*180/pi));
grid on;

sgtitle(sprintf('HIL M5 — %s controller, %s trajectory | RMS: %.1f mm (full), %.1f mm (SS)', ...
        upper(controller_type), traj_type, rms_err, rms_ss), 'FontSize', 13);
