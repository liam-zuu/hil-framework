function r = composite_ramp(t, t_hold, t_ramp)
% COMPOSITE_RAMP Industrial-standard trajectory ramp profile
%
% Three-phase profile:
%   Phase 1 (0 <= t < t_hold):           r = 0          (idle hold)
%   Phase 2 (t_hold <= t < t_hold+t_ramp): r linear 0 → 1 (accelerate)
%   Phase 3 (t >= t_hold+t_ramp):         r = 1          (cruise)
%
% Purpose: Avoid velocity step at t=0 that forces infinite acceleration
% reference. Hold phase lets outer-loop PI integral settle before motion.
%
% Usage:
%   r = composite_ramp(t, 0.5, 2.5)   % 0.5s hold + 2.5s ramp
%
% Inputs:
%   t      : time vector (1xN)
%   t_hold : hold duration (s)
%   t_ramp : ramp duration (s)
%
% Output:
%   r      : ramp signal in [0, 1], same shape as t

    r = zeros(size(t));
    % Phase 2: linear ramp from t_hold to t_hold+t_ramp
    idx_ramp   = (t >= t_hold) & (t < t_hold + t_ramp);
    r(idx_ramp) = (t(idx_ramp) - t_hold) / t_ramp;
    % Phase 3: cruise (clamp to 1)
    idx_cruise = t >= t_hold + t_ramp;
    r(idx_cruise) = 1.0;
end
