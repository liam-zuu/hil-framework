function [tau_up, states_down] = spi_interface(action, data, params)
% SPI_INTERFACE  SPI full-duplex communication sim (STUB — passthrough).

    tau_up = zeros(4,1);
    states_down = zeros(10,1);

    switch action
        case 'uplink'
            tau_up = data;        % STUB: passthrough, no quantization
        case 'downlink'
            states_down = data;   % STUB: passthrough
        otherwise
            error('spi_interface: unknown action "%s"', action);
    end

end
