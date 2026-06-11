function scenario_setup_paths()
% SCENARIO_SETUP_PATHS Add HIL project paths for scenario execution
%
% Call at start of each scenario file. Assumes this file lives in
% SIMU_M6/scenarios/ alongside scenario files.
%
% NOTE: Folder naming convention per M6_v3: 'nucleoh7' (no underscore),
% 'rpi5', 'esp32'.

    script_dir  = fileparts(mfilename('fullpath'));
    project_dir = fileparts(script_dir);   % SIMU_M6/

    addpath(script_dir);                         % scenarios/
    addpath(fullfile(project_dir, 'scripts'));
    addpath(fullfile(project_dir, 'rpi5'));
    addpath(fullfile(project_dir, 'nucleoh7'));  % FIXED: was nucleo_h7
    addpath(fullfile(project_dir, 'esp32'));
end
