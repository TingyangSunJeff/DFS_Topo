utils_path_yalmip_mosek();

% load('./data_res/Alg_Roofnet_CIFAR10_0412.mat');
load('./data_res/Alg_IAB_CIFAR10_0417.mat');

% load Ea_Adj
% netw_name = "Roofnet_";
netw_name = "IAB_";
model_name = "CIFAR10_";
% model_name = "MNIST_";
alg_name = {'BoydGreedy', 'SCA23', 'SDRRhoEw', 'SDRLambda2Ew', 'prim', 'ring', 'clique'};

if strcmp( model_name, "MNIST_" )
    G_overlay.Kappa = 1663370 * 64 / 1e6;
elseif strcmp( model_name, "CIFAR10_" )
    G_overlay.Kappa = 25.6 * 1e6 * 64 / 1e6;
end

G_overlay.Kappa = G_overlay.Kappa / 1e3;
G_overlay.G_u.delays = G_overlay.G_u.delays / 1e3;
G_overlay.delays = G_overlay.delays / 1e3;

N_Vo = length(G_overlay.Vo);
demands = G_overlay.Kappa * ones(N_Vo, 1);

tau_data = zeros(length(alg_name), 2);


for i = 1 : length(alg_name)
    if strcmp(alg_name{i}, 'clique')
        Ea = 1 : 45;
    else
        Ea_Adj_filename = strcat( './data_res/',netw_name, model_name, alg_name{i}, ".mat");
       
        load(Ea_Adj_filename);
    
        len_Ea = sum(Ea_Adj, 'all')/2;
        Ea = zeros(len_Ea, 1);
        [row, col] = find(Ea_Adj);
        i_Ea = 1;
        for i_r = 1 : length(row)
            u = row(i_r);
            v = col(i_r);
            if isKey( G_overlay.st2idx, strjoin(string([u,v])) )
                pair = G_overlay.st2idx( strjoin(string([u,v])) );
            else
                continue;
            end
            Ea(i_Ea) = pair;
            i_Ea = i_Ea + 1;
        end
    end
    [tau_wRouting, tau_wo] = min_tau_w_overlayRouting(G_overlay, Ea, demands);
    tau_data(i, 1) = tau_wRouting;
    tau_data(i, 2) = tau_wo;
end




function utils_path_yalmip_mosek()
% add YALMIP and Mosek
    currentFolder = pwd;
    if currentFolder(1) == 'C'
        path_yalmip = "C:\Users\yxh5389\Downloads\YALMIP-master";
        path_mosek = "C:\Users\yxh5389\Downloads\mosek\10.1\toolbox\R2017aom";
        path_sedumi = "C:\Users\yxh5389\Downloads\cvx-w64\cvx\sedumi";
    elseif currentFolder(1) == 'H'
        path_yalmip = "H:\Matlab2020b\toolbox\YALMIP-master";
        path_mosek = "H:\MOSEK\10.1\toolbox\r2017aom";
    else
    
    end
    % Get the current MATLAB search path
    currentPath = path;
    
    % Check if myPath is in the MATLAB search path
    if ~contains(currentPath, path_yalmip)
        % Add the path if it does not exist
        addpath(genpath(path_yalmip));
    end
    if ~contains(currentPath, path_mosek)
        addpath(path_mosek);
    end
% %     if ~contains(currentPath, path_sedumi)
% %         addpath(path_sedumi);
% %     end
end