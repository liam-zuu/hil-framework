function sync_ok = gpio_sync(step_k, cluster_done, params)
% GPIO_SYNC  Timing synchronization between clusters.
%
% Models the GPIO handshake that ensures all 3 clusters (ESP32, H7, RPi5)
% complete their work within the timestep budget. In real HIL hardware,
% a sync pulse on GPIO triggers each cluster, and a timeout fires if
% any cluster doesn't assert its "done" signal in time.
%
% Simulation model:
%   1. Check all clusters reported done
%   2. Add random timing jitter (Gaussian, models ISR latency)
%   3. If jitter exceeds timeout threshold -> sync failure
%
% Sync failures indicate the real-time constraint would be violated.
% They are logged but do not halt the simulation (the real HIL system
% would either skip a step or enter a fault state).
%
% Input:
%   step_k       [scalar] current timestep index
%   cluster_done [3x1 logical] flags: [esp32_done, h7_done, rpi5_done]
%   params       [struct] with .sync_jitter_us, .sync_timeout_us
% Output:
%   sync_ok      [logical] true if all clusters done within timeout

    % All clusters must report done
    if ~all(cluster_done)
        sync_ok = false;
        return;
    end

    % Simulate timing jitter for each cluster (microseconds)
    jitter_sigma = params.sync_jitter_us;
    timeout      = params.sync_timeout_us;

    % Each cluster has independent jitter (absolute value = latency)
    jitter = abs(jitter_sigma * randn(3, 1));

    % Sync OK if worst-case cluster finishes within timeout
    max_jitter = max(jitter);
    sync_ok = (max_jitter < timeout);

end
