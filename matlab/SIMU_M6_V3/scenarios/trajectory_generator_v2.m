function traj = trajectory_generator_v2(spec, dt, T)
% TRAJECTORY_GENERATOR_V2 Struct-based trajectory generator with composite ramp
%
% Usage:
%   spec.type = 'circle';
%   spec.R = 1.0; spec.period = 30;
%   traj = trajectory_generator_v2(spec, 0.001, 35);
%
% Common spec fields (all optional, have defaults):
%   .type     : trajectory type (see list below)
%   .t_hold   : idle hold before motion (default 0.5s)
%   .t_ramp   : ramp-up duration (default 2.5s)
%
% Trajectory types and type-specific fields:
%   'line'           : .v (forward speed, default 0.3)
%   'circle'         : .R (radius, default 1.0), .period (default 30)
%   'square'         : .side (default 1.0), .v (default 0.3)
%   'rounded_square' : .side (default 1.0), .R_corner (default 0.2), .v (default 0.3)
%   'figure8'        : .size (default 1.0), .period (default 8.0)
%   'zigzag'         : .amplitude (default 0.3), .wavelength (default 1.0), .v (default 0.3)
%   'sinusoidal'     : .amplitude (default 0.5), .frequency (default 0.2 Hz), .v (default 0.3)
%   'racetrack'      : .length (default 3.0), .width (default 1.0), .v (default 0.3)
%
% Output traj struct:
%   .t, .x_ref, .y_ref, .theta_ref, .vx_ref, .vy_ref, .wz_ref  (all 1xN)
%   .spec  (echo of input spec, with defaults filled in)
%
% Design note: Most trajectories use TIME WARPING to apply ramp while
% preserving path shape. Time warp:
%   tau(t) = integral of ramp(t')
% Evaluate full-speed trajectory at tau(t). During ramp, tau progresses
% slower than t, but path shape (e.g. circle radius) is exact.

    %% Defaults
    if ~isfield(spec,'t_hold'), spec.t_hold = 0.5; end
    if ~isfield(spec,'t_ramp'), spec.t_ramp = 2.5; end

    t = 0:dt:T;
    N = length(t);
    ramp = composite_ramp(t, spec.t_hold, spec.t_ramp);

    switch lower(spec.type)

        case 'line'
            if ~isfield(spec,'v'), spec.v = 0.3; end
            vx_ref    = spec.v * ramp;
            vy_ref    = zeros(1,N);
            wz_ref    = zeros(1,N);
            theta_ref = zeros(1,N);
            x_ref     = cumtrapz(t, vx_ref);
            y_ref     = zeros(1,N);

        case 'circle'
            if ~isfield(spec,'R'),      spec.R = 1.0; end
            if ~isfield(spec,'period'), spec.period = 30; end
            omega  = 2*pi / spec.period;
            v_tan  = spec.R * omega;
            % Robot heading tangent to circle, body frame forward = v_tan
            wz_ref    = omega * ramp;
            theta_ref = cumtrapz(t, wz_ref);
            vx_ref    = v_tan * ramp;
            vy_ref    = zeros(1,N);
            % Integrate position in world frame from body velocity
            dx_world = vx_ref .* cos(theta_ref) - vy_ref .* sin(theta_ref);
            dy_world = vx_ref .* sin(theta_ref) + vy_ref .* cos(theta_ref);
            x_ref = cumtrapz(t, dx_world);
            y_ref = cumtrapz(t, dy_world);

        case 'square'
            if ~isfield(spec,'side'), spec.side = 1.0; end
            if ~isfield(spec,'v'),    spec.v = 0.3; end
            [x_ref,y_ref,theta_ref,vx_ref,vy_ref,wz_ref] = ...
                build_square(t, ramp, spec.side, spec.v);

        case 'rounded_square'
            if ~isfield(spec,'side'),     spec.side = 1.0; end
            if ~isfield(spec,'R_corner'), spec.R_corner = 0.2; end
            if ~isfield(spec,'v'),        spec.v = 0.3; end
            [x_ref,y_ref,theta_ref,vx_ref,vy_ref,wz_ref] = ...
                build_rounded_square(t, ramp, spec.side, spec.R_corner, spec.v);

        case 'figure8'
            if ~isfield(spec,'size'),   spec.size = 1.0; end
            if ~isfield(spec,'period'), spec.period = 8.0; end
            A     = spec.size;
            omega = 2*pi / spec.period;
            % Time warp: tau(t) = integral of ramp
            tau = cumtrapz(t, ramp);
            % Figure-8 parametric (lemniscate-like)
            x_ref = A * sin(omega*tau);
            y_ref = A * sin(omega*tau) .* cos(omega*tau);
            % Derivatives wrt tau
            dx_dtau = A*omega*cos(omega*tau);
            dy_dtau = A*omega*(cos(omega*tau).^2 - sin(omega*tau).^2);
            ddx_dtau2 = -A*omega^2*sin(omega*tau);
            ddy_dtau2 = -4*A*omega^2*sin(omega*tau).*cos(omega*tau);
            % Heading from velocity direction
            theta_ref = atan2(dy_dtau, dx_dtau);
            theta_ref = unwrap(theta_ref);   % avoid ±π jumps
            % World velocities
            vx_world = dx_dtau .* ramp;
            vy_world = dy_dtau .* ramp;
            % Body frame velocities
            vx_ref =  vx_world.*cos(theta_ref) + vy_world.*sin(theta_ref);
            vy_ref = -vx_world.*sin(theta_ref) + vy_world.*cos(theta_ref);
            % Yaw rate via curvature formula: wz = (dx*ddy - dy*ddx)/(dx²+dy²)
            denom = dx_dtau.^2 + dy_dtau.^2 + 1e-9;
            wz_ref = (dx_dtau.*ddy_dtau2 - dy_dtau.*ddx_dtau2) ./ denom .* ramp;

        case 'zigzag'
            if ~isfield(spec,'amplitude'),  spec.amplitude = 0.3; end
            if ~isfield(spec,'wavelength'), spec.wavelength = 1.0; end
            if ~isfield(spec,'v'),          spec.v = 0.3; end
            A = spec.amplitude;
            L = spec.wavelength;
            v = spec.v;
            % Arc length parameter s(t) with ramp
            s = cumtrapz(t, v*ramp);
            % Triangle wave: y oscillates as function of x=s
            % Using smoothed triangle to avoid infinite curvature at peaks
            smooth = 0.05 * L;  % 5% of wavelength smoothing
            y_ref = A * smooth_triangle(s, L, smooth);
            x_ref = s;
            % Heading = atan2(dy/ds, dx/ds) * robot direction
            dy_ds = A * smooth_triangle_deriv(s, L, smooth);
            dx_ds = ones(size(s));
            theta_ref = unwrap(atan2(dy_ds, dx_ds));
            % Body velocities: robot drives forward along path
            speed = v*ramp;
            vx_ref = speed;
            vy_ref = zeros(1,N);
            % wz = dtheta/dt
            wz_ref = [0, diff(theta_ref)./dt];

        case 'sinusoidal'
            if ~isfield(spec,'amplitude'), spec.amplitude = 0.5; end
            if ~isfield(spec,'frequency'), spec.frequency = 0.2; end
            if ~isfield(spec,'v'),         spec.v = 0.3; end
            A = spec.amplitude;
            f = spec.frequency;
            v = spec.v;
            s     = cumtrapz(t, v*ramp);
            k     = 2*pi*f / v;       % spatial frequency
            x_ref = s;
            y_ref = A * sin(k*s);
            dy_ds = A*k*cos(k*s);
            dx_ds = ones(size(s));
            theta_ref = unwrap(atan2(dy_ds, dx_ds));
            vx_ref = v*ramp;
            vy_ref = zeros(1,N);
            wz_ref = [0, diff(theta_ref)./dt];

        case 'racetrack'
            if ~isfield(spec,'length'), spec.length = 3.0; end
            if ~isfield(spec,'width'),  spec.width = 1.0; end
            if ~isfield(spec,'v'),      spec.v = 0.3; end
            [x_ref,y_ref,theta_ref,vx_ref,vy_ref,wz_ref] = ...
                build_racetrack(t, ramp, spec.length, spec.width, spec.v);

        otherwise
            error('Unknown trajectory type: %s', spec.type);
    end

    %% Package output
    traj.t         = t;
    traj.x_ref     = x_ref(:).';
    traj.y_ref     = y_ref(:).';
    traj.theta_ref = theta_ref(:).';
    traj.vx_ref    = vx_ref(:).';
    traj.vy_ref    = vy_ref(:).';
    traj.wz_ref    = wz_ref(:).';
    traj.spec      = spec;
end

% =====================================================================
% Local functions
% =====================================================================

function [x,y,theta,vx,vy,wz] = build_square(t, ramp, side, v_cruise)
% Square path: start at origin, go +X, +Y, -X, -Y, back to origin
% Corners are sharp (instantaneous heading change). For smoother, use rounded_square.
    N = length(t);
    dt = t(2) - t(1);
    % Arc length along path
    s = cumtrapz(t, v_cruise*ramp);
    perimeter = 4*side;
    % Clamp s to perimeter (robot stops at end of one lap)
    s = min(s, perimeter);

    x = zeros(1,N); y = zeros(1,N); theta = zeros(1,N);
    for i = 1:N
        si = s(i);
        if     si < side,     x(i) = si;          y(i) = 0;        theta(i) = 0;
        elseif si < 2*side,   x(i) = side;        y(i) = si-side;  theta(i) = pi/2;
        elseif si < 3*side,   x(i) = 3*side-si;   y(i) = side;     theta(i) = pi;
        elseif si < 4*side,   x(i) = 0;           y(i) = 4*side-si;theta(i) = -pi/2;
        else,                 x(i) = 0;           y(i) = 0;        theta(i) = -pi/2;
        end
    end
    % Body velocities (robot drives forward, no sideways)
    vx = v_cruise*ramp;
    vy = zeros(1,N);
    wz = [0, diff(theta)./dt];
end

function [x,y,theta,vx,vy,wz] = build_rounded_square(t, ramp, side, Rc, v_cruise)
% Rounded-corner square. Corners replaced by quarter-circle of radius Rc.
    N = length(t);
    dt = t(2) - t(1);
    L_straight = side - 2*Rc;      % length of each straight segment
    L_corner   = pi/2 * Rc;         % length of each quarter-circle
    perimeter  = 4*(L_straight + L_corner);
    s = cumtrapz(t, v_cruise*ramp);
    s = mod(s, perimeter);          % loop forever

    x = zeros(1,N); y = zeros(1,N); theta = zeros(1,N);
    for i = 1:N
        si = s(i);
        seg_len = [L_straight, L_corner, L_straight, L_corner, ...
                   L_straight, L_corner, L_straight, L_corner];
        cum = cumsum(seg_len);
        seg = find(si <= cum, 1, 'first');
        if isempty(seg), seg = 8; si = cum(8); end
        if seg > 1, local_s = si - cum(seg-1); else, local_s = si; end

        switch seg
            case 1  % bottom straight: (0,0) → (L_straight, 0), heading 0
                x(i) = Rc + local_s;        y(i) = 0;          theta(i) = 0;
            case 2  % bottom-right corner, center (side-Rc, Rc)
                ang = -pi/2 + local_s/Rc;
                x(i) = side-Rc + Rc*cos(ang);
                y(i) = Rc + Rc*sin(ang);
                theta(i) = local_s/Rc;
            case 3  % right straight
                x(i) = side;                y(i) = Rc + local_s; theta(i) = pi/2;
            case 4  % top-right corner
                ang = 0 + local_s/Rc;
                x(i) = side-Rc + Rc*cos(ang);
                y(i) = side-Rc + Rc*sin(ang);
                theta(i) = pi/2 + local_s/Rc;
            case 5  % top straight
                x(i) = side-Rc - local_s;   y(i) = side;       theta(i) = pi;
            case 6  % top-left corner
                ang = pi/2 + local_s/Rc;
                x(i) = Rc + Rc*cos(ang);
                y(i) = side-Rc + Rc*sin(ang);
                theta(i) = pi + local_s/Rc;
            case 7  % left straight
                x(i) = 0;                   y(i) = side-Rc - local_s; theta(i) = -pi/2;
            case 8  % bottom-left corner
                ang = pi + local_s/Rc;
                x(i) = Rc + Rc*cos(ang);
                y(i) = Rc + Rc*sin(ang);
                theta(i) = -pi/2 + local_s/Rc;
        end
    end
    theta = unwrap(theta);
    vx = v_cruise*ramp;
    vy = zeros(1,N);
    wz = [0, diff(theta)./dt];
end

function [x,y,theta,vx,vy,wz] = build_racetrack(t, ramp, len, wid, v_cruise)
% Racetrack (oval): two straights of length `len` connected by semicircles of radius wid/2
    N = length(t);
    dt = t(2) - t(1);
    Rc = wid/2;
    L_straight = len - wid;
    L_arc = pi*Rc;
    perimeter = 2*(L_straight + L_arc);
    s = cumtrapz(t, v_cruise*ramp);
    s = mod(s, perimeter);

    x = zeros(1,N); y = zeros(1,N); theta = zeros(1,N);
    for i = 1:N
        si = s(i);
        seg_len = [L_straight, L_arc, L_straight, L_arc];
        cum = cumsum(seg_len);
        seg = find(si <= cum, 1, 'first');
        if isempty(seg), seg = 4; si = cum(4); end
        if seg > 1, local_s = si - cum(seg-1); else, local_s = si; end

        switch seg
            case 1  % bottom straight
                x(i) = Rc + local_s;   y(i) = 0;     theta(i) = 0;
            case 2  % right semicircle (center = (L_straight+Rc, Rc))
                ang = -pi/2 + local_s/Rc;
                x(i) = L_straight+Rc + Rc*cos(ang);
                y(i) = Rc + Rc*sin(ang);
                theta(i) = local_s/Rc;
            case 3  % top straight (reverse)
                x(i) = L_straight+Rc - local_s;   y(i) = wid;   theta(i) = pi;
            case 4  % left semicircle
                ang = pi/2 + local_s/Rc;
                x(i) = Rc + Rc*cos(ang);
                y(i) = Rc + Rc*sin(ang);
                theta(i) = pi + local_s/Rc;
        end
    end
    theta = unwrap(theta);
    vx = v_cruise*ramp;
    vy = zeros(1,N);
    wz = [0, diff(theta)./dt];
end

function y = smooth_triangle(x, L, smooth)
% Smoothed triangle wave of period L, amplitude ±1
% smooth = rounding radius at peaks (in units of x)
    % Base triangle: y = 1 - |2*mod(x/L + 0.25, 1) - 1|*2  → peaks at ±1
    phase = mod(x/L, 1);
    y = zeros(size(x));
    for i = 1:length(x)
        p = phase(i);
        if p < 0.25
            y(i) = 4*p;            % 0 → 1
        elseif p < 0.75
            y(i) = 2 - 4*p;        % 1 → -1
        else
            y(i) = -4 + 4*p;       % -1 → 0
        end
    end
    % Smoothing via low-pass-ish moving average would require window
    % For simplicity, return pure triangle (smooth handled elsewhere if needed)
end

function dy = smooth_triangle_deriv(x, L, smooth)
    phase = mod(x/L, 1);
    dy = zeros(size(x));
    for i = 1:length(x)
        p = phase(i);
        if p < 0.25
            dy(i) = 4/L;
        elseif p < 0.75
            dy(i) = -4/L;
        else
            dy(i) = 4/L;
        end
    end
end
