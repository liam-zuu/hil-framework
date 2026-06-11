function traj = trajectory_generator(type, T_sim, dt, params)
% TRAJECTORY_GENERATOR  Generate body-level reference trajectories.
%
% Input:
%   type   [string] 'circle' | 'square' | 'line' | 'figure8'
%   T_sim  [scalar] simulation duration (s)
%   dt     [scalar] timestep (s)
%   params [struct] from params_mecanum
%
% Output:
%   traj [struct] with fields:
%     .t        [1×N] time vector (s)
%     .x_ref    [1×N] X position ref (m)
%     .y_ref    [1×N] Y position ref (m)
%     .theta_ref[1×N] heading ref (rad)
%     .vx_ref   [1×N] body vx ref (m/s)
%     .vy_ref   [1×N] body vy ref (m/s)
%     .wz_ref   [1×N] yaw rate ref (rad/s)

    t = 0:dt:T_sim;
    N = length(t);

    switch lower(type)
        case 'circle'
            R_circ = 0.5;         % radius (m)
            omega_circ = 2*pi/5;  % complete circle in 5s

            theta_ref = omega_circ * t;
            x_ref     = R_circ * cos(theta_ref) - R_circ; % start at origin
            y_ref     = R_circ * sin(theta_ref);

            % Body velocities: tangential speed in body frame
            v_total = R_circ * omega_circ;
            vx_ref  = v_total * ones(1,N);  % forward speed in body frame
            vy_ref  = zeros(1,N);            % no lateral (pure forward tracking)
            wz_ref  = omega_circ * ones(1,N);

        case 'line'
            v_line = 0.3; % m/s forward

            x_ref     = v_line * t;
            y_ref     = zeros(1,N);
            theta_ref = zeros(1,N);
            vx_ref    = v_line * ones(1,N);
            vy_ref    = zeros(1,N);
            wz_ref    = zeros(1,N);

        case 'square'
            side  = 0.5; % m
            v_sq  = 0.2; % m/s
            t_side = side / v_sq;       % time per side
            t_turn = (pi/2) / (pi/2);   % 1s per 90° turn
            omega_turn = pi/2 / t_turn;

            x_ref = zeros(1,N); y_ref = zeros(1,N); theta_ref = zeros(1,N);
            vx_ref = zeros(1,N); vy_ref = zeros(1,N); wz_ref = zeros(1,N);

            % Build piecewise: 4 sides with turns
            seg_times = [t_side, t_turn, t_side, t_turn, t_side, t_turn, t_side, t_turn];
            seg_vx    = [v_sq,   0,      v_sq,   0,      v_sq,   0,      v_sq,   0     ];
            seg_wz    = [0,      omega_turn, 0, omega_turn, 0, omega_turn, 0, omega_turn];

            idx = 1;
            cum_theta = 0;
            cx = 0; cy = 0;
            for s = 1:length(seg_times)
                n_seg = round(seg_times(s) / dt);
                for j = 1:n_seg
                    if idx > N, break; end
                    vx_ref(idx) = seg_vx(s);
                    wz_ref(idx) = seg_wz(s);
                    cum_theta   = cum_theta + seg_wz(s)*dt;
                    theta_ref(idx) = cum_theta;
                    cx = cx + seg_vx(s)*cos(cum_theta)*dt;
                    cy = cy + seg_vx(s)*sin(cum_theta)*dt;
                    x_ref(idx) = cx;
                    y_ref(idx) = cy;
                    idx = idx + 1;
                end
            end
            % Fill remaining with hold
            if idx <= N
                x_ref(idx:end)     = x_ref(idx-1);
                y_ref(idx:end)     = y_ref(idx-1);
                theta_ref(idx:end) = theta_ref(idx-1);
            end

        case 'figure8'
            R_f8 = 0.4;
            omega_f8 = 2*pi/5;

            x_ref     = R_f8 * sin(2*omega_f8*t);
            y_ref     = R_f8 * sin(omega_f8*t);
            theta_ref = atan2(gradient(y_ref,dt), gradient(x_ref,dt));

            % Numerical body velocities
            dx = gradient(x_ref, dt);
            dy = gradient(y_ref, dt);
            vx_ref = dx .* cos(theta_ref) + dy .* sin(theta_ref);
            vy_ref = -dx .* sin(theta_ref) + dy .* cos(theta_ref);
            wz_ref = gradient(theta_ref, dt);

        otherwise
            error('Unknown trajectory type: %s', type);
    end

    traj.t         = t;
    traj.x_ref     = x_ref;
    traj.y_ref     = y_ref;
    traj.theta_ref = theta_ref;
    traj.vx_ref    = vx_ref;
    traj.vy_ref    = vy_ref;
    traj.wz_ref    = wz_ref;

end
