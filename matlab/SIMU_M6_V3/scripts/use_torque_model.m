function params = use_torque_model(params)
% USE_TORQUE_MODEL  Switch params to the legacy M6 torque-input plant.
%   params = use_torque_model(params_mecanum());
% Used by the M3-M6 scripts in scripts/ and validate/, which pass torques
% (not PWM duty) to plant_step. The DC motor model is the default since
% 21/10/2026 (spec S01). Polarity and firmware sign are set to +1 so these
% scripts behave exactly as before.
    params.motor_model    = 'torque';
    params.motor_polarity = ones(4, 1);
    params.motor_dir      = ones(4, 1);
end
