function vel_cmd = position_controller(pose_ref, pose_est, vel_ref, params)
% POSITION_CONTROLLER  Outer loop: PI position/heading control + feedforward.
%
% PI controller with feedforward and anti-windup:
%   vel_cmd = vel_ref + Kp * error_body + Ki * integral(error_body)
%
% Integral eliminates steady-state tracking error on moving references
% (circle, figure-8) that P-only cannot handle. Anti-windup freezes
% integration when output is saturated to prevent overshoot.
%
% Position error is computed in world frame, then rotated to body frame
% so corrections are applied in the robot's local coordinate system.
% Heading error uses angular wrapping for shortest path.
%
% Gain design rationale:
%   Kp_pos=3.0: outer loop BW ~0.5 Hz, well below inner loop ~5 Hz
%   Ki_pos=0.5: integral time Ti = Kp/Ki = 6s, slow enough to avoid
%     oscillation but fast enough to eliminate SS error within ~3s
%   Kp_theta=4.0, Ki_theta=1.0: heading tracks faster than position
%
% Uses MATLAB persistent variables for integral state.
% Call "clear position_controller" between simulation runs to reset.
%
% Input:
%   pose_ref  [3x1] reference pose [x_ref; y_ref; theta_ref] world frame
%   pose_est  [3x1] estimated pose [x_est; y_est; theta_est] world frame
%   vel_ref   [3x1] feedforward body velocities [vx_ref; vy_ref; wz_ref]
%   params    [struct] with .pos_ctrl.* gains and limits, .dt
% Output:
%   vel_cmd   [3x1] corrected body velocity commands [vx_cmd; vy_cmd; wz_cmd]

    persistent int_pos int_theta;

    %% Initialize on first call
    if isempty(int_pos)
        int_pos   = [0; 0];   % integral of [e_x_body; e_y_body]
        int_theta = 0;        % integral of e_theta
    end

    dt       = params.dt;
    Kp_pos   = params.pos_ctrl.Kp_pos;
    Ki_pos   = params.pos_ctrl.Ki_pos;
    Kp_theta = params.pos_ctrl.Kp_theta;
    Ki_theta = params.pos_ctrl.Ki_theta;

    vx_max = params.pos_ctrl.vx_max;
    vy_max = params.pos_ctrl.vy_max;
    wz_max = params.pos_ctrl.wz_max;

    %% Position error in world frame
    e_x_world = pose_ref(1) - pose_est(1);
    e_y_world = pose_ref(2) - pose_est(2);

    %% Rotate position error to body frame
    theta_est = pose_est(3);
    cos_t = cos(theta_est);
    sin_t = sin(theta_est);

    e_x_body =  cos_t * e_x_world + sin_t * e_y_world;
    e_y_body = -sin_t * e_x_world + cos_t * e_y_world;

    %% Heading error (shortest path, wrapped to [-pi, pi])
    e_theta = pose_ref(3) - pose_est(3);
    e_theta = mod(e_theta + pi, 2*pi) - pi;

    %% Tentative integral update
    int_pos_new   = int_pos + [e_x_body; e_y_body] * dt;
    int_theta_new = int_theta + e_theta * dt;

    %% Tentative output (to check saturation for anti-windup)
    vx_tent = vel_ref(1) + Kp_pos * e_x_body + Ki_pos * int_pos_new(1);
    vy_tent = vel_ref(2) + Kp_pos * e_y_body + Ki_pos * int_pos_new(2);
    wz_tent = vel_ref(3) + Kp_theta * e_theta + Ki_theta * int_theta_new;

    %% Anti-windup: freeze integral if output saturated AND error would grow integral
    % X velocity
    if abs(vx_tent) < vx_max || sign(e_x_body) ~= sign(int_pos(1))
        int_pos(1) = int_pos_new(1);
    end
    % Y velocity
    if abs(vy_tent) < vy_max || sign(e_y_body) ~= sign(int_pos(2))
        int_pos(2) = int_pos_new(2);
    end
    % Yaw rate
    if abs(wz_tent) < wz_max || sign(e_theta) ~= sign(int_theta)
        int_theta = int_theta_new;
    end

    %% Safety clamp on integral (backup)
    int_pos_max = vx_max / (Ki_pos + 1e-10);
    int_pos = max(-int_pos_max, min(int_pos_max, int_pos));
    int_theta_max = wz_max / (Ki_theta + 1e-10);
    int_theta = max(-int_theta_max, min(int_theta_max, int_theta));

    %% Final PI output + feedforward
    vx_cmd = vel_ref(1) + Kp_pos * e_x_body   + Ki_pos   * int_pos(1);
    vy_cmd = vel_ref(2) + Kp_pos * e_y_body   + Ki_pos   * int_pos(2);
    wz_cmd = vel_ref(3) + Kp_theta * e_theta  + Ki_theta * int_theta;

    %% Velocity clamping
    vx_cmd = max(-vx_max, min(vx_max, vx_cmd));
    vy_cmd = max(-vy_max, min(vy_max, vy_cmd));
    wz_cmd = max(-wz_max, min(wz_max, wz_cmd));

    vel_cmd = [vx_cmd; vy_cmd; wz_cmd];

end
