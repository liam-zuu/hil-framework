function sm = state_manager(action, sm, varargin)
% STATE_MANAGER  Manage state vector storage and retrieval (STUB).

    switch action
        case 'init'
            x0 = varargin{1};
            params = varargin{2};
            N = round(params.T_sim / params.dt) + 1;
            sm.x      = x0;
            sm.x_prev = x0;
            sm.k      = 1;
            sm.history = zeros(length(x0), N);
            sm.history(:,1) = x0;

        case 'update'
            x_new = varargin{1};
            sm.x_prev = sm.x;
            sm.x      = x_new;
            sm.k      = sm.k + 1;
            if sm.k <= size(sm.history,2)
                sm.history(:, sm.k) = x_new;
            end

        case 'get'
            % sm.x already accessible

        case 'get_prev'
            % sm.x_prev already accessible

        otherwise
            error('state_manager: unknown action "%s"', action);
    end

end
