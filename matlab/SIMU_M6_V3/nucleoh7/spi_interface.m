function [tau_up, states_down] = spi_interface(action, data, params)
% SPI_INTERFACE  SPI full-duplex communication simulation.
%
% Simulates fixed-point quantization that occurs when float values are
% packed into N-bit SPI frames and unpacked on the other side.
%
% Quantization model:
%   value -> clamp to [-range, +range] -> round to N-bit levels -> reconstruct
%   N = params.spi.float_bits (default 16)
%   Quantization error ~ range / 2^(N-1)
%
% Uplink  (H7 -> RPi5): torque [4x1]
% Downlink (RPi5 -> H7): state vector [10x1]
%
% Input:
%   action [string] 'uplink' | 'downlink'
%   data   — uplink: tau [4x1] | downlink: x [10x1]
%   params [struct]
% Output:
%   tau_up      [4x1]  torque after SPI quantization (uplink)
%   states_down [10x1] states after SPI quantization (downlink)

    n_bits = params.spi.float_bits;

    tau_up      = zeros(4, 1);
    states_down = zeros(10, 1);

    switch action
        case 'uplink'
            % Torque: all 4 channels share same full-scale range
            range = params.spi.tau_range;
            tau_up = fixed_point_quantize(data, range, n_bits);

        case 'downlink'
            % States: each state has its own full-scale range
            ranges = params.spi.state_ranges;
            for i = 1:10
                states_down(i) = fixed_point_quantize(data(i), ranges(i), n_bits);
            end

        otherwise
            error('spi_interface: unknown action "%s"', action);
    end

end


function y = fixed_point_quantize(x, full_scale, n_bits)
% FIXED_POINT_QUANTIZE  Simulate N-bit signed fixed-point ADC/DAC.
%
%   Maps x in [-full_scale, +full_scale] to 2^n_bits integer codes,
%   then reconstructs back to float. Models real SPI data packing.

    n_levels = 2^n_bits;
    lsb = (2 * full_scale) / n_levels;

    % Clamp to full-scale range
    x_clamped = max(-full_scale, min(full_scale, x));

    % Quantize: float -> integer code -> float
    code = round(x_clamped / lsb);
    code = max(-n_levels/2, min(n_levels/2 - 1, code));

    y = code * lsb;
end
