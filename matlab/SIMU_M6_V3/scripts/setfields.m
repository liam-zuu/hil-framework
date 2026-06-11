function p = setfields(p, varargin)
% SETFIELDS  Set fields in struct, supporting dot notation for nested fields.
%   p = setfields(p, 'field1', val1, 'field2.subfield', val2, ...)
%
% Example:
%   p = setfields(params, 'slip.enabled', true, 'enc_noise_sigma', 0.05)

    for i = 1:2:length(varargin)
        field_path = varargin{i};
        value = varargin{i+1};
        parts = strsplit(field_path, '.');
        if length(parts) == 1
            p.(parts{1}) = value;
        elseif length(parts) == 2
            p.(parts{1}).(parts{2}) = value;
        elseif length(parts) == 3
            p.(parts{1}).(parts{2}).(parts{3}) = value;
        end
    end
end
