%% PLOT_RESULTS  Visualize simulation results.
%
% Expects: x_hist, t_hist, log, traj, omega_ref_all in workspace.

figure('Name', 'HIL Simulation Results', 'Position', [100 100 1200 800]);

%% 1. XY Trajectory
subplot(2,3,1);
plot(x_hist(1,:), x_hist(2,:), 'b-', 'LineWidth', 1.5); hold on;
plot(traj.x_ref, traj.y_ref, 'r--', 'LineWidth', 1);
plot(x_hist(1,1), x_hist(2,1), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
plot(x_hist(1,end), x_hist(2,end), 'rs', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
xlabel('X (m)'); ylabel('Y (m)');
title('XY Trajectory');
legend('Actual', 'Reference', 'Start', 'End');
axis equal; grid on;

%% 2. Heading
subplot(2,3,2);
plot(t_hist, rad2deg(x_hist(3,:)), 'b-', 'LineWidth', 1); hold on;
t_ref_plot = min(length(traj.theta_ref), length(t_hist));
plot(traj.t(1:t_ref_plot), rad2deg(traj.theta_ref(1:t_ref_plot)), 'r--');
xlabel('Time (s)'); ylabel('\theta (deg)');
title('Heading');
legend('Actual', 'Reference');
grid on;

%% 3. Body velocities
subplot(2,3,3);
plot(t_hist, x_hist(4,:), 'b-', 'LineWidth', 1); hold on;
plot(t_hist, x_hist(5,:), 'r-', 'LineWidth', 1);
plot(t_hist, x_hist(6,:), 'g-', 'LineWidth', 1);
xlabel('Time (s)'); ylabel('Velocity');
title('Body Velocities');
legend('v_x (m/s)', 'v_y (m/s)', '\omega_z (rad/s)');
grid on;

%% 4. Wheel speeds
subplot(2,3,4);
N_plot = min(size(log.omega_ref, 2), length(t_hist)-1);
t_log = t_hist(1:N_plot);
colors = {'b','r','g','m'};
for w = 1:4
    plot(t_log, log.omega_est(w,1:N_plot), [colors{w} '-'], 'LineWidth', 0.8); hold on;
    plot(t_log, log.omega_ref(w,1:N_plot), [colors{w} '--'], 'LineWidth', 0.5);
end
xlabel('Time (s)'); ylabel('\omega (rad/s)');
title('Wheel Speeds (solid=est, dashed=ref)');
legend('W1','W1_{ref}','W2','W2_{ref}','W3','W3_{ref}','W4','W4_{ref}');
grid on;

%% 5. Torque commands
subplot(2,3,5);
for w = 1:4
    plot(t_log, log.tau_cmd(w,1:N_plot), [colors{w} '-'], 'LineWidth', 0.8); hold on;
end
xlabel('Time (s)'); ylabel('\tau (N·m)');
title('Torque Commands');
legend('W1','W2','W3','W4');
grid on;

%% 6. Tracking error
subplot(2,3,6);
if length(traj.x_ref) >= length(t_hist)
    ex = x_hist(1,:) - traj.x_ref(1:length(t_hist));
    ey = x_hist(2,:) - traj.y_ref(1:length(t_hist));
    e_pos = sqrt(ex.^2 + ey.^2);
    plot(t_hist, e_pos * 1000, 'b-', 'LineWidth', 1);
    xlabel('Time (s)'); ylabel('Error (mm)');
    title(sprintf('Position Error (RMS: %.2f mm)', rms(e_pos)*1000));
    grid on;
end

sgtitle(sprintf('HIL Sim M3 — %s controller, %s trajectory', ...
        controller_type, traj_type));
