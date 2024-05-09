%% Test different E_a generation algorithms


%% add YALMIP and Mosek
currentFolder = pwd;
utils_path_yalmip_mosek()

% % read_graph_file("Abvt.Graph");

N_nodes = 19; %% maximum number of nodes in test
%% G_type = 0:Hex; 1:Roofnet
G_type = 0;
if G_type == 0
    G_name = 'IAB_';
elseif G_type == 1
    G_name = 'Roofnet_';
end
n_node_start = 10;
N_runs = 5;
N_run = N_runs;
N_list_nV = length(n_node_start:N_nodes);

% % T_SDR = zeros(N_runs, N_list_nV);
% % T_MICP = zeros(N_runs, N_list_nV);
% % T_SDR_lambda2_ew = zeros(N_runs, N_list_nV);
% % T_SDR_rho_ew = zeros(N_runs, N_list_nV);
% % T_SDR_joint_rho_y = zeros(N_runs, N_list_nV);
% % T_SCA_joint_rho_y = zeros(N_runs, N_list_nV);
% % T_SCA_rho_ew = zeros(N_runs, N_list_nV);
% % T_boyd_greedy = zeros(N_runs, N_list_nV);
% % T_MKP = zeros(N_runs, N_list_nV);
% % T_ring1 = zeros(N_runs, N_list_nV);
% % T_ring2 = zeros(N_runs, N_list_nV);
% % T_ring3 = zeros(N_runs, N_list_nV);
% % T_allPair = zeros(N_runs, N_list_nV);
% % T_MST1 = zeros(N_runs, N_list_nV);
% % T_MST2 = zeros(N_runs, N_list_nV);

T_SDR = zeros(N_runs, 2);
T_MICP = zeros(N_runs, 2);
T_SDR_lambda2_ew = zeros(N_runs, 2);
T_SDR_rho_ew = zeros(N_runs, 2);
T_SDR_joint_rho_y = zeros(N_runs, 2);
T_SCA_joint_rho_y = zeros(N_runs, 2);
T_SCA_rho_ew = zeros(N_runs, 2);
T_SCA_add_rho_ew = zeros(N_runs, 2);
T_SCA_both_rho_ew = zeros(N_runs, 2);
T_boyd_greedy = zeros(N_runs, 2);
T_MKP = zeros(N_runs, 2);
T_ring1 = zeros(N_runs, 2);
T_ring2 = zeros(N_runs, 2);
T_ring3 = zeros(N_runs, 2);
T_allPair = zeros(N_runs, 2);
T_MST1 = zeros(N_runs, 2);
T_MST2 = zeros(N_runs, 2);
Ea_algName2write = {'boyd_greedy', 'SDR_rho_ew', 'SDR_lambda2_ew', 'SCA_both_23', 'clique'};

i_nV = 1;
modelType = "CIFAR10";
% modelType = "MNIST";
F_calculated = 0;
for n_V = n_node_start : N_nodes
    if n_V > n_node_start
        break;
    end
    N_links = n_V * (n_V - 1);

    G_overlay = gen_overlay_netw(n_V, N_links, G_type, modelType, F_calculated); %% Generate B matrix for the graph: G_overlay.B
    G_overlay.G_type = G_type;
    G_overlay.F_calculated = F_calculated;

    % % write_spr_to_file(G_overlay, "spr_iab_n19.txt");

    N_pairs = size(G_overlay.Bp, 2);
    Ea_SDR_lambda2_ew = zeros(N_pairs, N_runs);
    Ea_SDR_rho_ew = zeros(N_pairs, N_runs);
    Ea_SDR_joint_rho_y = zeros(N_pairs, N_runs);
    Ea_SCA_joint_rho_y = zeros(N_pairs, N_runs);
    Ea_SCA_rho_ew = zeros(N_pairs, N_runs);
    Ea_SCA_add_rho_ew = zeros(N_pairs, N_runs);
    Ea_SCA_both_rho_ew = zeros(N_pairs, N_runs);
    Ea_boyd_greedy = zeros(N_pairs, N_runs);
    Ea_MKP = zeros(N_pairs, N_runs);

    for i_run = 1 : N_runs
        disp(i_run);
        M1 = unifrnd(2, 4,1,1);
        M2 = unifrnd(2, 4,1,1);
        if i_run == 1
            M1 = 2;
            M2 = 2;
        end
        % M1 = randi([0.01, 0.1],1,1);
        % M2 = randi([0.01, 0.1],1,1);
        % % sol_MICP = biLevel_upper(G_overlay, M1, M2, "MICP");
        % % T_MICP(i_run, i_nV) = sol_MICP.T;

        sol_boyd_greedy = biLevel_upper(G_overlay, M1, M2, "boyd_greedy");
        T_boyd_greedy(i_run, 1) = sol_boyd_greedy.T;
        T_boyd_greedy(i_run, 2) = sol_boyd_greedy.runTime;
        Ea_boyd_greedy(sol_boyd_greedy.Ea, i_run) = 1;
        % % sol_MKP = biLevel_upper(G_overlay, M1, M2, "MKP");
        % % T_MKP(i_run, 1) = sol_MKP.T;
        % % T_MKP(i_run, 2) = sol_MKP.runTime;
        % % Ea_MKP(sol_MKP.Ea, i_run) = 1;
        sol_lambda2_ew = biLevel_upper(G_overlay, M1, M2, "SDR_lambda2_equalWeight");
        T_SDR_lambda2_ew(i_run, 1) = sol_lambda2_ew.T;
        T_SDR_lambda2_ew(i_run, 2) = sol_lambda2_ew.runTime;
        Ea_SDR_lambda2_ew(sol_lambda2_ew.Ea, i_run) = 1;
        sol_23 = biLevel_upper(G_overlay, M1, M2, "SDR_rho_equalWeight");
        T_SDR_rho_ew(i_run, 1) = sol_23.T;
        T_SDR_rho_ew(i_run, 2) = sol_23.runTime;
        Ea_SDR_rho_ew(sol_23.Ea, i_run) = 1;
        % % sol_24 = biLevel_upper(G_overlay, M1, M2, "SDR_joint_rho_y");
        % % T_SDR_joint_rho_y(i_run, 1) = sol_24.T;
        % % T_SDR_joint_rho_y(i_run, 2) = sol_24.runTime;
        % % Ea_SDR_joint_rho_y(sol_24.Ea, i_run) = 1;
        % % sol_SCA24 = biLevel_upper(G_overlay, M1, M2, "SCA_joint_rho_y");
        % % T_SCA_joint_rho_y(i_run, 1) = sol_SCA24.T;
        % % T_SCA_joint_rho_y(i_run, 2) = sol_SCA24.runTime;
        % % Ea_SCA_joint_rho_y(sol_SCA24.Ea, i_run) = 1;
        % % sol_SCA23 = biLevel_upper(G_overlay, M1, M2, "SCA_rho_ew");
        % % T_SCA_rho_ew(i_run, 1) = sol_SCA23.T;
        % % T_SCA_rho_ew(i_run, 2) = sol_SCA23.runTime;
        % % Ea_SCA_rho_ew(sol_SCA23.Ea, i_run) = 1;
        % sol_SCA23add = biLevel_upper(G_overlay, M1, M2, "SCA_add_rho_ew");
        % T_SCA_add_rho_ew(i_run, 1) = sol_SCA23add.T;
        % T_SCA_add_rho_ew(i_run, 2) = sol_SCA23add.runTime;
        % Ea_SCA_add_rho_ew(sol_SCA23add.Ea, i_run) = 1;
        sol_SCA23both = biLevel_upper(G_overlay, M1, M2, "SCA_both_rho_ew");
        T_SCA_both_rho_ew(i_run, 1) = sol_SCA23both.T;
        T_SCA_both_rho_ew(i_run, 2) = sol_SCA23both.runTime;
        Ea_SCA_both_rho_ew(sol_SCA23both.Ea, i_run) = 1;
        [T_ring1(i_run, 1), T_ring1(i_run, 2)] = benchmark_random_ring(G_overlay, M1, M2);
        [T_ring2(i_run, 1), T_ring2(i_run, 2)] = benchmark_random_ring(G_overlay, M1, M2);
        [T_ring3(i_run, 1), T_ring3(i_run, 2)] = benchmark_random_ring(G_overlay, M1, M2);
        [T_allPair(i_run, 1), T_allPair(i_run, 2)] = benchmark_allPair(G_overlay, M1, M2);
        [T_MST1(i_run, 1), T_MST1(i_run, 2)] = benchmark_randomMST(G_overlay, M1, M2);
        [T_MST2(i_run, 1), T_MST1(i_run, 2)] = benchmark_randomMST(G_overlay, M1, M2);
        % save("Alg_cmp.mat","T_SDR","T_boyd_greedy","T_MKP","T_ring1","T_ring2","T_ring3","T_allPair","T_MST1","T_MST2");
    end
    i_nV = i_nV + 1;
end

N_run = 6;

figure;
subplot(1, 2, 1); 
plot(1:N_run, T_SDR_lambda2_ew(1:N_run, 1), '--r');
hold on
plot(1:N_run, T_SDR_rho_ew(1:N_run, 1), '-+r');
hold on
plot(1:N_run, T_SDR_joint_rho_y(1:N_run, 1), '-or');
hold on
plot(1:N_run, T_SCA_rho_ew(1:N_run, 1), '--b');
hold on
plot(1:N_run, T_SCA_joint_rho_y(1:N_run, 1), '-ob');
hold on
plot(1:N_run, T_SCA_both_rho_ew(1:N_run, 1), '-+b');
hold on
plot(1:N_run, T_boyd_greedy(1:N_run, 1), '-g');
hold on
plot(1:N_run, T_MKP(1:N_run, 1), '-cyan');
hold on
plot(1:N_run, T_ring1(1:N_run, 1), '-<k');
hold on
plot(1:N_run, T_ring2(1:N_run, 1), '-^k');
hold on
plot(1:N_run, T_ring3(1:N_run, 1), '--k');
hold on
plot(1:N_run, T_MST1(1:N_run, 1), '-m');
hold on
plot(1:N_run, T_MST2(1:N_run, 1), '-pm');
hold on
plot(1:N_run, T_allPair(1:N_run, 1), '--', 'Color', '#D95319');
legend("SDR-(\lambda_2)", "SDR-(23)", "SDR-(24)", "SCA-(23)", "SCA-(24)", "SCA-(23)-both", "boyd-greedy", "MKP", "ring1", "ring2", "ring3", "MST1", "MST2", "AllPair")

subplot(1, 2, 2); 
plot(1:N_run, T_SDR_lambda2_ew(1:N_run, 2), '--r');
hold on
plot(1:N_run, T_SDR_rho_ew(1:N_run, 2), '-+r');
hold on
plot(1:N_run, T_SDR_joint_rho_y(1:N_run, 2), '-or');
hold on
plot(1:N_run, T_SCA_rho_ew(1:N_run, 2), '--b');
hold on
plot(1:N_run, T_SCA_joint_rho_y(1:N_run, 2), '-ob');
hold on
plot(1:N_run, T_SCA_both_rho_ew(1:N_run, 2), '-+b');
hold on
plot(1:N_run, T_boyd_greedy(1:N_run, 2), '-g');
hold on
plot(1:N_run, T_MKP(1:N_run, 2), '-cyan');
hold on
% plot(1:N_run, T_ring1(1:N_run, 2), '-<k');
% hold on
% plot(1:N_run, T_ring2(1:N_run, 2), '-^k');
% hold on
% plot(1:N_run, T_ring3(1:N_run, 2), '--k');
% hold on
% plot(1:N_run, T_MST1(1:N_run, 2), '-m');
% hold on
% plot(1:N_run, T_MST2(1:N_run, 2), '-pm');
% hold on
% plot(1:N_run, T_allPair(1:N_run, 1), '--', 'Color', '#D95319');{'boyd_greedy', 'SDR_rho_ew', 'SDR_lambda2_ew', 'SCA_both_23'}
legend("SDR-(\lambda_2)", "SDR-(23)", "SDR-(24)", "SCA-(23)", "SCA-(24)", "SCA-(23)-both", "boyd-greedy", "MKP", "ring1", "ring2", "ring3", "MST1", "MST2")

load('roofnet_0404.mat'); %% Resnet-50 for CIFAR-10
% load('Alg_Roofnet_MNIST_0405.mat');
if strcmp(G_name, "Roofnet_")
    G_overlay.C_F = 0.001 * ones( length(G_overlay.C_F), 1 ); %% 1Mbps
    G_overlay.G_u.C = 0.001 * ones( length(G_overlay.G_u.C), 1 ); %% 1Mbps
end
if G_overlay.F_calculated > 0
    f_str = "_finf";
else
    f_str = "";
end
for i_alg = 1 : length(Ea_algName2write)
    if strcmp(Ea_algName2write(i_alg), "boyd_greedy")
        write_Ea_to_file(G_overlay, strcat(G_name, modelType, f_str ,'_BoydGreedy.mat'), Ea_boyd_greedy);
    elseif strcmp(Ea_algName2write(i_alg), "SDR_rho_ew")
        write_Ea_to_file(G_overlay, strcat(G_name, modelType, f_str ,'_SDRRhoEw.mat'), Ea_SDR_rho_ew);
    elseif strcmp(Ea_algName2write(i_alg), "SDR_lambda2_ew")
        write_Ea_to_file(G_overlay, strcat(G_name, modelType, f_str ,'_SDRLambda2Ew.mat'), Ea_SDR_lambda2_ew);
    elseif strcmp(Ea_algName2write(i_alg), "SCA_both_23")
        write_Ea_to_file(G_overlay, strcat(G_name, modelType, f_str ,'_SCA23.mat'), Ea_SCA_both_rho_ew);
    elseif strcmp(Ea_algName2write(i_alg), "clique")
        write_Ea_to_file(G_overlay, strcat(G_name, modelType, f_str ,'_clique.mat'), ones(size(Ea_boyd_greedy, 1), 1));
    end
end

netw_name = "Roofnet";
% netw_name = "IAB";
% alg_name = "prim";
% alg_name = "ring";
tau_prim = gen_tau_from_Ea_file(G_overlay, netw_name, "prim");
tau_ring = gen_tau_from_Ea_file(G_overlay, netw_name, "ring");


%% Functions
function tau = gen_tau_from_Ea_file(G_overlay, netw_name, model_name, alg_name)
    % use to verify tau from Tingyang's Ea file
    filename2read = strcat("./data_res/",netw_name, "_", model_name, "_", alg_name, ".mat");
    Ea2st = load(filename2read).Ea2st; %% 1-index
    len_Ea = length(Ea2st(:, 1));
    Ea_idx = zeros(1, len_Ea);
    for k = 1 : len_Ea
        src = Ea2st(k, 1);
        dst = Ea2st(k, 2);
        if isKey(G_overlay.st2idx, strjoin(string([src, dst])))
            Ea_idx(k) = G_overlay.st2idx( strjoin(string([src, dst])) );
        else
            Ea_idx(k) = G_overlay.st2idx( strjoin(string([dst, src])) );
        end
    end
    tau_sol = solve_tau_noRouting(G_overlay, Ea_idx);
    tau = tau_sol.tau;
end

function sol_lower = lower_MICP(G_overlay, M1, M2)
% Ting's 1st idea: 
% (1) iterate beta
% (2) given beta, solve a MICP for the lower level problem to find y and alpha

% obtain lb and ub of beta: 
% (1) lb: randomly generate some spanning tree, solve for tau
% (2) ub: Ea = E, solve for tau
    Ea_ring = gen_random_ring(G_overlay);
    solution_ring = solve_tau_noRouting(G_overlay, Ea_ring);
    tau_lb = solution_ring.tau * 0.5;
    
    solution_E = solve_tau_noRouting(G_overlay, 1:size(G_overlay.Bp, 2));
    tau_ub = solution_E.tau;
    L_beta = linspace(tau_lb,tau_ub,8);
    rho_res = -1*ones(1, length(L_beta));
    y_res = zeros(size(G_overlay.Bp, 2), length(L_beta));
    best_wall_clock_time = inf;
    best_Ea = 0;
    
    for i_beta = 1: length(L_beta())
        beta = L_beta(i_beta);
        % sol_tmp = solve_rho_MICP_given_beta(G_overlay, beta);
        sol_tmp = heuristic_SDR(G_overlay, beta);
        if sol_tmp.success > 0
            rho_res(i_beta) = sol_tmp.rho;
            y_res(:, i_beta) = sol_tmp.y;
            Ea_tmp = find( y_res(:, i_beta) > 1e-5 );
            K_tmp = gen_K_from_rho(rho_res(i_beta), M1, M2);
            tau_sol_tmp = solve_tau_noRouting(G_overlay, Ea_tmp);
            tau_tmp = tau_sol_tmp.tau;
            wall_clock_time = K_tmp * tau_tmp;
            if wall_clock_time < best_wall_clock_time
                best_wall_clock_time = wall_clock_time;
                best_Ea = Ea_tmp;
            end
        end
    end
    sol_lower = struct;
    sol_lower.Ea = best_Ea;
    sol_lower.T = best_wall_clock_time;
end

function sol_lower = biLevel_upper(G_overlay, M1, M2, lowerAlg)
    % upper framework with varying beta: Greedy perturbation from Boyd based on Fiedler vector
    % % if G_overlay.G_type == 0 %% G_type:    0: IAB;     1:Roofnet
    % %     Ea_ring = gen_random_ring(G_overlay);
    % %     solution_ring = solve_tau_noRouting(G_overlay, Ea_ring);
    % %     % tau_lb = solution_ring.tau;
    % %     tau_ub = solution_ring.tau;
    % % 
    % %     Ea_MST = [4, 8, 9, 11, 16, 18, 31, 33, 36];
    % %     solution_MST = solve_tau_noRouting(G_overlay, Ea_MST);
    % %     tau_lb = solution_MST.tau;
    % % 
    % %     L_beta = linspace(tau_lb,tau_ub,4);
    % % elseif G_overlay.G_type == 1 %% G_type:    0: IAB;     1:Roofnet
    % %     Ea_ring = gen_random_ring(G_overlay);
    % %     solution_ring = solve_tau_noRouting(G_overlay, Ea_ring);
    % %     tau_lb = solution_ring.tau;
    % % 
    % %     solution_E = solve_tau_noRouting(G_overlay, 1:size(G_overlay.Bp, 2));
    % %     tau_ub = solution_E.tau;
    % %     L_beta = linspace(tau_lb,tau_ub,6);
    % % end

    Ea_ring = gen_random_ring(G_overlay);
    solution_ring = solve_tau_noRouting(G_overlay, Ea_ring);
    tau_lb = solution_ring.tau * 0.5;
    
    solution_E = solve_tau_noRouting(G_overlay, 1:size(G_overlay.Bp, 2));
    tau_ub = solution_E.tau;
    L_beta = linspace(tau_lb,tau_ub,10);

    rho_res = -1*ones(1, length(L_beta));
    y_res = zeros(size(G_overlay.Bp, 2), length(L_beta));
    best_wall_clock_time = inf;
    best_Ea = 0;
    runTime = 0;

    for i_beta = 1: length(L_beta)
        beta = L_beta(i_beta);
        if strcmp(lowerAlg, "MICP")
            sol_tmp = solve_rho_MICP_given_beta(G_overlay, beta);
        elseif strcmp(lowerAlg, "SDR")
            sol_tmp = heuristic_SDR(G_overlay, beta);
        elseif strcmp(lowerAlg, "SDR_lambda2_equalWeight") 
            sol_tmp = SDR_lambda2_equalWeight(G_overlay, beta);
        elseif strcmp(lowerAlg, "SDR_rho_equalWeight") 
            sol_tmp = SDR_rho_equalWeight(G_overlay, beta);
        elseif strcmp(lowerAlg, "SDR_joint_rho_y") 
            sol_tmp = SDR_joint_rho_y(G_overlay, beta);
        elseif strcmp(lowerAlg, "SCA_joint_rho_y") 
            sol_tmp = SCA_joint_rho_y(G_overlay, beta);
        elseif strcmp(lowerAlg, "SCA_rho_ew") 
            sol_tmp = SCA_rho_ew(G_overlay, beta);
        elseif strcmp(lowerAlg, "SCA_add_rho_ew") 
            sol_tmp = SCA_add_rho_ew(G_overlay, beta);
        elseif strcmp(lowerAlg, "SCA_both_rho_ew") 
            sol_tmp = SCA_both_rho_ew(G_overlay, beta);
        elseif strcmp(lowerAlg, "boyd_greedy")
            sol_tmp = greedy_boyd_lower(G_overlay, beta);
        elseif strcmp(lowerAlg, "MKP")
            sol_tmp = MKP_lower(G_overlay, beta);
        end
        
        runTime = runTime + sol_tmp.elapsed_time;

        if sol_tmp.success > 0
            rho_res(i_beta) = sol_tmp.rho;
            y_res(:, i_beta) = sol_tmp.y;
            Ea_tmp = find( y_res(:, i_beta) > 1e-5 );
            tic;
            K_tmp = gen_K_from_rho(rho_res(i_beta), M1, M2);
            tau_sol_tmp = solve_tau_noRouting(G_overlay, Ea_tmp);
            elapsed_time = toc;
            runTime = runTime + elapsed_time;
            tau_tmp = tau_sol_tmp.tau;
            wall_clock_time = K_tmp * tau_tmp;
            if wall_clock_time < best_wall_clock_time && wall_clock_time > 1e2
                best_wall_clock_time = wall_clock_time;
                best_Ea = Ea_tmp;
            end
        end
    end
    sol_lower = struct;
    sol_lower.Ea = best_Ea;
    sol_lower.T = best_wall_clock_time;
    sol_lower.runTime = runTime;
end

function sol_res = MKP_lower(G_overlay, beta)
    % multi-dimensional knapsack problem
    N_pairs = size(G_overlay.Bp, 2);
    N_nodes = G_overlay.N_nodes;

    J = (1 / N_nodes) * ones(N_nodes, N_nodes);
    [v_star, ~] = eigs(eye(N_nodes)-J, 1, 'largestabs');
    % % d = length(G_overlay.F_all); %% dimension
    d = length(G_overlay.F_calc); %% dimension
    epsilon = 0.95 * (d/(d+1));
    l = min([ceil(d/epsilon)-(d+1), N_pairs]);
    l = 1;
    zA = 0;
    Ea_best = [];
    weights = zeros(N_pairs, d);
    knapsack_size = zeros(d, 1);
    values = zeros(N_pairs, 1);

    tic;

    for i_p = 1 : N_pairs %% weights
        s = G_overlay.idx2st(i_p, 1);
        t = G_overlay.idx2st(i_p, 2);
        s_vo = G_overlay.Vu2Vo(s); % idx of s in Vo
        t_vo = G_overlay.Vu2Vo(t); % idx of t in Vo
        % weights(i_p) = (v_star(s_vo) - v_star(t_vo))^2;
        values(i_p) = (v_star(s_vo) - v_star(t_vo))^2;
    end
    
    F2pairs = zeros(d, N_pairs);
    for i_F = 1:d
        % % F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
        % % if length(F_eu) == 1
        % %     C_F = G_overlay.G_u.C(F_eu(1));
        % % else
        % %     C_F = min( G_overlay.G_u.C(F_eu) );
        % % end
        C_F = G_overlay.C_F(i_F);
        knapsack_size(i_F) = C_F;
        % F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
        F = G_overlay.F_calc{i_F}; %% tunnels in F
        if isempty(F)
            continue;
        end
        v_pairs = G_overlay.link2pair(F);
        F2pairs(i_F, v_pairs) = 1;
        tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
        % values(v_pairs) = values(v_pairs) + tmp_term2;
        weights(v_pairs, i_F) = tmp_term2;
    end
    
    for k = 1 : l-1
        subsets = nchoosek(1:N_pairs, k); %% L \subset N: |L| <= l-1
        for i_S = 1:size(subsets,1)
            S = subsets(i_S, :); % Extract the i-th subset pairs
            w_S = sum( weights(S, :), 1 );
            noConstrViolation = 1;
            for i_F = 1:d
                if w_S(i_F) > knapsack_size(i_F)
                    noConstrViolation = 0;
                    break;
                end
            end
            if noConstrViolation == 1
                p_S = sum( values(S) );
                if p_S > zA
                    zA = p_S;
                    Ea_best = S;
                end
            end
        end
    end
    subsets = nchoosek(1:N_pairs, l); %% L \subset N: |L| <= l
    for i_S = 1:size(subsets,1)
        if mod(i_S, 200) == 0
            disp(i_S);
        end
        S = subsets(i_S, :); % Extract the i-th subset
        w_S = sum( weights(S, :), 1 );
        minValueInS = min(values(S));
        noConstrViolation = 1;
        for i_F = 1:d
            if w_S(i_F) > knapsack_size(i_F)
                noConstrViolation = 0;
                break;
            end
        end
        if noConstrViolation == 1
            % Alg-1/(d+1)
            yalmip('clear')
            % Define variables
            x = sdpvar(N_pairs,1);

            % Define constraints 
            Constraints = [x(S) == 0, x(values <= minValueInS) == 0, 0 <=x, x <= 1];
            for i_F = 1:d
                % % F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
                C_F = knapsack_size(i_F);
                % % F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
                F = G_overlay.F_calc{i_F};
                if isempty(F)
                    continue;
                end
                v_pairs = G_overlay.link2pair(F);
                Constraints = [Constraints, x(v_pairs)' * weights(v_pairs, i_F) <= C_F - w_S(i_F)];
            end
            
            % Define an objective
            Objective = -values' * x;
            
            % Set some options for YALMIP and solver
            options = sdpsettings('verbose',0,'solver','mosek');
        
            % Solve the problem
            sol = optimize(Constraints,Objective,options);
            x_sol = value(x);
            
            FracSet = find((x_sol>1e-4) & (x_sol <= 0.999));
            IntSet = find( x_sol > 0.999 );
            
            [best_frac, best_frac_idx] = max(values(FracSet));
            sum_IntSet = sum(values(IntSet));
            if sum_IntSet >= best_frac
                best_zH = sum_IntSet;
                best_items = IntSet;
            else
                best_zH = best_frac;
                best_items = FracSet(best_frac_idx);
            end
            p_S = sum( values(S) );
            if p_S + best_zH > zA
                zA = p_S + best_zH;
                Ea_best = union(S, best_items);
            end
        end
    end
    
    elapsed_time = toc;
    
    sol_rho = solve_rho(G_overlay, Ea_best);
    y = sol_rho.alpha;
    rho = sol_rho.rho;
    sol_res = struct();
    sol_res.y = y;
    sol_res.rho = rho;
    sol_res.success = 1;
    sol_res.elapsed_time = elapsed_time;
end

function sol_res = greedy_boyd_lower(G_overlay, beta)
    % Greedy perturbation from Boyd based on Fiedler vector
    N_nodes = G_overlay.N_nodes;
    N_pairs = size(G_overlay.Bp, 2);
    Bp = G_overlay.Bp(G_overlay.Vo, :);
    mat_temr2CF = gen_constr_CF_beta(G_overlay, beta);

    y = zeros(N_pairs, 1);
    y_selected = zeros(N_pairs, 1);
    
    tic;

    for i_n = 1 : N_pairs %% loop over #pairs
        Ea = find(y > 1e-5);
        L_tmp = Bp * diag(y) * Bp';
        [V_all, lambda_a_all] = eigs(L_tmp,2,'smallestabs');
        v2 = V_all(:, 2);
        best_gd = 0;
        best_pair = 0;
        pair_added = 0;
        success = 0;
        for i_p = 1 : N_pairs %% loop over each pair
            if y_selected(i_p) > 1e-4 %% already selected this pair
                continue;
            end
            s = G_overlay.idx2st(i_p, 1);
            t = G_overlay.idx2st(i_p, 2);
            s_vo = G_overlay.Vu2Vo(s); % idx of s in Vo
            t_vo = G_overlay.Vu2Vo(t); % idx of t in Vo
            gd = abs(v2(s_vo) - v2(t_vo));
            if gd > best_gd
                y_bin = y_selected;
                y_bin(i_p) = 1;
                constrViolated = 0;
                for i_F = 1:length(G_overlay.F_calc)
                    % % F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
                    % % if length(F_eu) == 1
                    % %     C_F = G_overlay.G_u.C(F_eu(1));
                    % % else
                    % %     C_F = min( G_overlay.G_u.C(F_eu) );
                    % % end
                    % % F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
                    % % if isempty(F)
                    % %     continue;
                    % % end
                    % % v_pairs = G_overlay.link2pair(F);
                    % % tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
                    % % constr_violation = y_bin(v_pairs)' * tmp_term2 - C_F;
                    constr_violation = mat_temr2CF(i_F, :) * y_bin - G_overlay.C_F(i_F);
                    if constr_violation > 0
                        constrViolated = 1;
                        break;
                    end
                end
                % sol_tau = solve_tau_noRouting(G_overlay, [Ea; i_p]);
                % tau_tmp = sol_tau.tau;
                % if tau_tmp <= beta
                if constrViolated == 0
                    best_gd = gd;
                    best_pair = i_p;
                    pair_added = 1;
                    success = 1;
                end
            end
            % % y_tmp = y;
            % % y_tmp(i_p) = 1;
            % % L = B * diag(y_tmp) * B'; %% obtain degree
            % % degrees = diag(L);
            % % for e = 1 : length(y)
            % %     if y_tmp(e) == 0
            % %         continue;
            % %     end
            % %     u = G_overlay.idx2st(e, 1);
            % %     v = G_overlay.idx2st(e, 2);
            % %     max_degree = max([degrees(u), degrees(v)]);
            % %     if max_degree > 0
            % %         y_tmp(e) = 1 / max([degrees(u), degrees(v)]);
            % %     else
            % %         y_tmp(e) = 0;
            % %     end
            % % end
            % % L_tmp = Bp * diag(y_tmp) * Bp';
            % % [V_all, lambda_a_all] = eigs(L_tmp,2,'smallestabs');
            % % v2 = V_all(:, 2);
        end
        if pair_added > 0
            y_selected(best_pair) = 1;
            sol_rho = solve_rho(G_overlay, sort([Ea; best_pair]));
            y = sol_rho.alpha;
            % % rho = sol_rho.rho;
        else
            break;
        end
    end
    elapsed_time = toc;

    L = Bp * diag(y_selected) * Bp';
    lambda_a_all = eigs(L,2,'smallestabs');
    lambda_a = lambda_a_all(2);
    sol_res = struct;
    sol_res.Ea = find(y_selected > 1e-5);
    sol_res.y = y_selected;
    sol_res.elapsed_time = elapsed_time;
    if lambda_a > 0
        sol_rho = solve_rho(G_overlay, sol_res.Ea);
        sol_res.rho = sol_rho.rho;
        sol_res.success = 1;
    else
        sol_res.success = -2; % disconnected graph
    end
end

function solution = solve_rho_MICP_given_beta(G_overlay, beta)

    % % cvx_begin sdp
    % %     variable Z(5,5) hermitian toeplitz
    % %     dual variable Q
    % %     minimize( norm( Z- eye(5), 'fro' ) )
    % %     Z >= 0 : Q;
    % % cvx_end
% Ting's 1st idea, lower level optimization
    N_nodes = G_overlay.N_nodes;
    N_pairs = size(G_overlay.Bp, 2);
    J = (1 / N_nodes) * ones(N_nodes, N_nodes);
    Bp = G_overlay.Bp(G_overlay.Vo, :);

    % % cvx_begin
    % %     % Define solver
    % %     cvx_solver SeDuMi
    % %     % cvx_solver SDPT3
    % % 
    % %     % Define variables
    % %     variable rho(1,1)
    % %     variable a(N_pairs,1)
    % %     variable y(N_pairs,1) binary
    % % 
    % % 
    % %     % Construct L using alpha and Bp
    % %     % % L = 0;
    % %     % % for i = 1 : N_pairs
    % %     % %     L = L + a(i) * Bp(:, i) * Bp(:, i)';
    % %     % % end
    % %     L = Bp*diag(a)*Bp';
    % % 
    % %     % Objective
    % %     minimize( rho )
    % % 
    % %     subject to
    % %         0 <= rho <= 1
    % % 
    % %         % Semidefinite constraints
    % %         L_plus_J = L + J;
    % %         eye(N_nodes) - L_plus_J + rho*eye(N_nodes) == semidefinite(N_nodes);
    % %         rho*eye(N_nodes) - (eye(N_nodes) - L_plus_J) == semidefinite(N_nodes);
    % % 
    % %         % Linear constraints on alpha and y
    % %         0 <= a <= y;
    % % 
    % %         for i_F = 1:length(G_overlay.F_all)
    % %             F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
    % %             if length(F_eu) == 1
    % %                 C_F = G_overlay.G_u.C(F_eu(1));
    % %             else
    % %                 C_F = min( G_overlay.G_u.C(F_eu) );
    % %             end
    % %             F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
    % %             if isempty(F)
    % %                 continue;
    % %             end
    % %             v_pairs = G_overlay.link2pair(F);
    % %             tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
    % %             y(v_pairs)' * tmp_term2 <= C_F;
    % %         end
    % % cvx_end
    % % 
    % % % Process solution
    % % solution = struct;
    % % solution.rho = cvx_optval;
    % % if strcmp(cvx_status, 'Solved')
    % %     solution.alpha = a;
    % %     solution.y = y;
    % %     solution.success = 1;
    % % else
    % %     disp('Hmm, something went wrong!');
    % %     solution.success = 0;
    % %     solution.info = cvx_status;
    % % end


    yalmip('clear')
    
    % Define variables
    y = binvar(N_pairs,1);
    alpha = sdpvar(N_pairs,1);
    rho = sdpvar(1,1);
    

    L = 0;
    for i = 1 : N_pairs
        L = L + alpha(i) * Bp(:, i) * Bp(:, i)';
    end
    Constraints = [-rho*eye(N_nodes) <= eye(N_nodes) - L - J <= rho*eye(N_nodes)];
    
    % % Constraints = [eye(N_nodes) - Bp*diag(alpha)*Bp' - J - rho*eye(N_nodes) <= 0, ...
    % %     eye(N_nodes) - Bp*diag(alpha)*Bp' - J + rho*eye(N_nodes) >= 0];
    
    Constraints = [Constraints, 0 <= alpha <= y];
    
    for i_F = 1:length(G_overlay.F_calc)
        % % F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
        % % if length(F_eu) == 1
        % %     C_F = G_overlay.G_u.C(F_eu(1));
        % % else
        % %     C_F = min( G_overlay.G_u.C(F_eu) );
        % % end
        % % F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F

        F = G_overlay.F_calc{i_F};
        C_F = G_overlay.C_F(i_F);
        if isempty(F)
            continue;
        end
        v_pairs = G_overlay.link2pair(F);
        tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
        Constraints = [Constraints, y(v_pairs)' * tmp_term2 <= C_F];
    end
    
    % Define an objective
    Objective = rho;
    
    % Set some options for YALMIP and solver
    % options = sdpsettings('verbose',1,'solver','mosek');
    options = sdpsettings('solver','bnb','bnb.solver','fmincon');
    % options = sdpsettings('verbose',1);
    
    % Solve the problem
    sol = optimize(Constraints,Objective,options);
    
    % Analyze error flags
    solution = struct;
    if sol.problem == 0
        % Extract and display value
        solution.rho = value(rho);
        solution.alpha = value(alpha);
        solution.y = value(y);
        solution.success = 1;
    else
        disp('Hmm, something went wrong!');
        solution.success = 0;
        solution.info = sol.info;
        % yalmiperror(sol.problem)
    end

end

function solution = SDR_lambda2_equalWeight(G_overlay, beta)
    % convex relaxation to SDP, then directly rounding
    N_nodes = G_overlay.N_nodes;
    N_pairs = size(G_overlay.Bp, 2);
    Bp = G_overlay.Bp(G_overlay.Vo, :);
    alpha = 1 / (2*N_nodes+1);
    mat_temr2CF = gen_constr_CF_beta(G_overlay, beta);

    yalmip('clear')
    
    % Define variables
    y = sdpvar(N_pairs,1);
    s = sdpvar(1,1);
    
    J = (1 / N_nodes) * ones(N_nodes, N_nodes);
    tic;
    
    Constraints = [0 <= y <= 1, s * (eye(N_nodes) - J) <= alpha * Bp * diag(y) * Bp', mat_temr2CF * y <= G_overlay.C_F];

    % % for i_F = 1:length(G_overlay.F_all)
    % %     F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
    % %     if length(F_eu) == 1
    % %         C_F = G_overlay.G_u.C(F_eu(1));
    % %     else
    % %         C_F = min( G_overlay.G_u.C(F_eu) );
    % %     end
    % %     F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
    % %     if isempty(F)
    % %         continue;
    % %     end
    % %     v_pairs = G_overlay.link2pair(F);
    % %     tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
    % %     Constraints = [Constraints, y(v_pairs)' * tmp_term2 <= C_F];
    % % end
    
    % Define an objective
    Objective = -s;
    
    % Set some options for YALMIP and solver
    options = sdpsettings('verbose',0,'solver','mosek');
    
    % Solve the problem
    sol = optimize(Constraints,Objective,options);
    
    % Analyze error flags
    solution = struct;
    if sol.problem == 0
        % Extract and display value
        solution.y = value(y);
        solution.s = value(s);
    else
        disp('Hmm, something went wrong!');
        sol.info
        yalmiperror(sol.problem)
    end

    % % constr_violation = zeros(length(G_overlay.F_all), 1);
    y_bin = 1*(solution.y > 1e-5);
    % % for i_F = 1:length(G_overlay.F_all)
    % %     F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
    % %     if length(F_eu) == 1
    % %         C_F = G_overlay.G_u.C(F_eu(1));
    % %     else
    % %         C_F = min( G_overlay.G_u.C(F_eu) );
    % %     end
    % %     F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
    % %     if isempty(F)
    % %         continue;
    % %     end
    % %     v_pairs = G_overlay.link2pair(F);
    % %     tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
    % %     constr_violation(i_F) = y_bin(v_pairs)' * tmp_term2 - C_F;
    % % end
    constr_violation = mat_temr2CF * y_bin - G_overlay.C_F;


    % rouding to s.t. beta constraints
    [~, idx_sort] = sort(solution.y);
    y_keep = y_bin;
    for k_sort = 1 : length(idx_sort)
        k = idx_sort(k_sort);
        if solution.y(k) <= 1e-4 %% weight is too small
            y_keep(k) = 0;
            continue;
        else
            % Determine whether removing this pair can help reduce the constr violation
            y_keep(k) = 0;
            is_remove_k = 0;
            for i_F = 1:length(G_overlay.F_calc)
                if constr_violation(i_F) <= 0
                    continue;
                end
                new_violation = mat_temr2CF(i_F, :) * y_keep - G_overlay.C_F(i_F);
                % % F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
                % % if length(F_eu) == 1
                % %     C_F = G_overlay.G_u.C(F_eu(1));
                % % else
                % %     C_F = min( G_overlay.G_u.C(F_eu) );
                % % end
                % % F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
                % % if isempty(F)
                % %     continue;
                % % end
                % % v_pairs = G_overlay.link2pair(F);
                % % tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
                % % new_violation = y_keep(v_pairs)' * tmp_term2 - C_F;
                if new_violation < constr_violation(i_F)
                    is_remove_k = 1;
                    constr_violation(i_F) = new_violation;
                end
            end
            if is_remove_k == 0
                y_keep(k) = 1;
            end
            if ~any(constr_violation > 0)
                break;
            end
        end
    end
    elapsed_time = toc;

    % re-calc best rho
    L = Bp * diag(y_keep) * Bp';
    lambda_a_all = eigs(L,2,'smallestabs');
    lambda_a = lambda_a_all(2);
    solution.Ea = find(y_keep > 0);
    solution.y = y_keep;
    solution.elapsed_time = elapsed_time;
    if lambda_a > 0
        sol_rho = solve_rho(G_overlay, solution.Ea);
        solution.rho = sol_rho.rho;
        solution.success = 1;
    else
        solution.success = -2; % disconnected graph
    end
end

function solution = SDR_rho_equalWeight(G_overlay, beta)
    % convex relaxation to SDP for the exact solution (23)
    N_nodes = G_overlay.N_nodes;
    N_pairs = size(G_overlay.Bp, 2);
    Bp = G_overlay.Bp(G_overlay.Vo, :);
    alpha = 1 / (2*N_nodes+1);
    mat_temr2CF = gen_constr_CF_beta(G_overlay, beta);

    yalmip('clear')
    
    % Define variables
    y = sdpvar(N_pairs,1);
    rho = sdpvar(1,1);
    
    J = (1 / N_nodes) * ones(N_nodes, N_nodes);
    tic;
    
    Constraints = [0 <= y <= 1, -rho * eye(N_nodes) <= eye(N_nodes) - J - alpha * Bp * diag(y) * Bp' <= rho * eye(N_nodes), mat_temr2CF * y <= G_overlay.C_F];

    % % for i_F = 1:length(G_overlay.F_all)
    % %     F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
    % %     if length(F_eu) == 1
    % %         C_F = G_overlay.G_u.C(F_eu(1));
    % %     else
    % %         C_F = min( G_overlay.G_u.C(F_eu) );
    % %     end
    % %     F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
    % %     if isempty(F)
    % %         continue;
    % %     end
    % %     v_pairs = G_overlay.link2pair(F);
    % %     tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
    % %     Constraints = [Constraints, y(v_pairs)' * tmp_term2 <= C_F];
    % % end
    
    % Define an objective
    Objective = rho;
    
    % Set some options for YALMIP and solver
    options = sdpsettings('verbose',0,'solver','mosek');
    
    % Solve the problem
    sol = optimize(Constraints,Objective,options);
    
    % Analyze error flags
    solution = struct;
    if sol.problem == 0
        % Extract and display value
        solution.y = value(y);
        solution.rho = value(rho);
    else
        disp('Hmm, something went wrong!');
        sol.info
        yalmiperror(sol.problem)
    end

    % % constr_violation = zeros(length(G_overlay.F_all), 1);
    y_bin = 1*(solution.y > 1e-5);
    % % for i_F = 1:length(G_overlay.F_all)
    % %     F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
    % %     if length(F_eu) == 1
    % %         C_F = G_overlay.G_u.C(F_eu(1));
    % %     else
    % %         C_F = min( G_overlay.G_u.C(F_eu) );
    % %     end
    % %     F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
    % %     if isempty(F)
    % %         continue;
    % %     end
    % %     v_pairs = G_overlay.link2pair(F);
    % %     tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
    % %     constr_violation(i_F) = y_bin(v_pairs)' * tmp_term2 - C_F;
    % % end
    constr_violation = mat_temr2CF * y_bin - G_overlay.C_F;


    % rouding to s.t. beta constraints
    [~, idx_sort] = sort(solution.y);
    y_keep = y_bin;
    for k_sort = 1 : length(idx_sort)
        k = idx_sort(k_sort);
        if solution.y(k) <= 1e-4 %% weight is too small
            y_keep(k) = 0;
            continue;
        else
            % Determine whether removing this pair can help reduce the constr violation
            y_keep(k) = 0;
            is_remove_k = 0;
            for i_F = 1:length(G_overlay.F_calc)
                if constr_violation(i_F) <= 0
                    continue;
                end
                new_violation = mat_temr2CF(i_F, :) * y_keep - G_overlay.C_F(i_F);
                % % F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
                % % if length(F_eu) == 1
                % %     C_F = G_overlay.G_u.C(F_eu(1));
                % % else
                % %     C_F = min( G_overlay.G_u.C(F_eu) );
                % % end
                % % F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
                % % if isempty(F)
                % %     continue;
                % % end
                % % v_pairs = G_overlay.link2pair(F);
                % % tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
                % % new_violation = y_keep(v_pairs)' * tmp_term2 - C_F;
                if new_violation < constr_violation(i_F)
                    is_remove_k = 1;
                    constr_violation(i_F) = new_violation;
                end
            end
            if is_remove_k == 0
                y_keep(k) = 1;
            end
            if ~any(constr_violation > 0)
                break;
            end
        end
    end
    elapsed_time = toc;

    % re-calc best rho
    L = Bp * diag(y_keep) * Bp';
    lambda_a_all = eigs(L,2,'smallestabs');
    lambda_a = lambda_a_all(2);
    solution.Ea = find(y_keep > 0);
    solution.y = y_keep;
    solution.elapsed_time = elapsed_time;
    if lambda_a > 0
        sol_rho = solve_rho(G_overlay, solution.Ea);
        solution.rho = sol_rho.rho;
        solution.success = 1;
    else
        solution.success = -2; % disconnected graph
    end
end

function solution = SCA_rho_ew(G_overlay, beta)
    % successively kicking out pairs based on exact solution (23)
    N_nodes = G_overlay.N_nodes;
    N_pairs = size(G_overlay.Bp, 2);
    Bp = G_overlay.Bp(G_overlay.Vo, :);
    alpha = 1 / (2*N_nodes+1);
    mat_temr2CF = gen_constr_CF_beta(G_overlay, beta);
    
    constrViolated = 1;
    y_kickout = ones(N_pairs,1); %% if a pair has been kicked out, y_kickout(e) = 0.
    tic;
    while constrViolated > 0
        yalmip('clear')
    
        % Define variables
        y = sdpvar(N_pairs,1);
        rho = sdpvar(1,1);
        
        J = (1 / N_nodes) * ones(N_nodes, N_nodes);
        
        Constraints = [y <= y_kickout, 0 <= y <= 1, -rho * eye(N_nodes) <= eye(N_nodes) - J - alpha * Bp * diag(y) * Bp' <= rho * eye(N_nodes), mat_temr2CF * y <= G_overlay.C_F];
    
        % % for i_F = 1:length(G_overlay.F_all)
        % %     F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
        % %     if length(F_eu) == 1
        % %         C_F = G_overlay.G_u.C(F_eu(1));
        % %     else
        % %         C_F = min( G_overlay.G_u.C(F_eu) );
        % %     end
        % %     F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
        % %     if isempty(F)
        % %         continue;
        % %     end
        % %     v_pairs = G_overlay.link2pair(F);
        % %     tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
        % %     Constraints = [Constraints, y(v_pairs)' * tmp_term2 <= C_F];
        % % end
        
        % Define an objective
        Objective = rho;
        
        % Set some options for YALMIP and solver
        options = sdpsettings('verbose',0,'solver','mosek');
        
        % Solve the problem
        sol = optimize(Constraints,Objective,options);
        
        % Analyze error flags
        solution = struct;
        if sol.problem == 0
            % Extract and display value
            solution.y = value(y);
            solution.rho = value(rho);
        else
            disp('Hmm, something went wrong!');
            sol.info
            yalmiperror(sol.problem)
        end

        [~, idx_sort] = sort(solution.y);
        pair2kickout = -1;
        for i_sort = 1 : length(idx_sort)
            idx = idx_sort(i_sort);
            if solution.y(idx) > 1e-7 && y_kickout(idx) ~= 0
                pair2kickout = idx;
                y_kickout(idx) = 0;
                break;
            end
        end
    
        constr_violation = zeros(length(G_overlay.F_calc), 1);
        y_bin = 1*(solution.y > 1e-5);
        y_bin(pair2kickout) = 0;
        % % for i_F = 1:length(G_overlay.F_all)
        % %     F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
        % %     if length(F_eu) == 1
        % %         C_F = G_overlay.G_u.C(F_eu(1));
        % %     else
        % %         C_F = min( G_overlay.G_u.C(F_eu) );
        % %     end
        % %     F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
        % %     if isempty(F)
        % %         continue;
        % %     end
        % %     v_pairs = G_overlay.link2pair(F);
        % %     tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
        % %     constr_violation(i_F) = y_bin(v_pairs)' * tmp_term2 - C_F;
        % % end
        constr_violation = mat_temr2CF * y_bin - G_overlay.C_F;
        if ~any(constr_violation > 0)
            constrViolated = 0;
            break;
        end
    end
    elapsed_time = toc;

    % re-calc best rho
    L = Bp * diag(y_bin) * Bp';
    lambda_a_all = eigs(L,2,'smallestabs');
    lambda_a = lambda_a_all(2);
    solution.Ea = find(y_bin > 0);
    solution.y = y_bin;
    solution.elapsed_time = elapsed_time;
    if lambda_a > 0
        sol_rho = solve_rho(G_overlay, solution.Ea);
        solution.rho = sol_rho.rho;
        solution.success = 1;
    else
        solution.success = -2; % disconnected graph
    end

end

function solution = SCA_add_rho_ew(G_overlay, beta)
    % successively kicking out pairs based on exact solution (23)
    N_nodes = G_overlay.N_nodes;
    N_pairs = size(G_overlay.Bp, 2);
    Bp = G_overlay.Bp(G_overlay.Vo, :);
    alpha = 1 / (2*N_nodes+1);
    mat_temr2CF = gen_constr_CF_beta(G_overlay, beta);
    
    constrViolated = 1;
    % % y_kickout = ones(N_pairs,1); %% if a pair has been kicked out, y_kickout(e) = 0.
    y_keep = zeros(N_pairs,1); %% if a pair has been kicked out, y_kickout(e) = 0.
    tic;
    while constrViolated > 0
        yalmip('clear')
    
        % Define variables
        y = sdpvar(N_pairs,1);
        rho = sdpvar(1,1);
        
        J = (1 / N_nodes) * ones(N_nodes, N_nodes);
        
        Constraints = [y >= y_keep, 0 <= y <= 1, -rho * eye(N_nodes) <= eye(N_nodes) - J - alpha * Bp * diag(y) * Bp' <= rho * eye(N_nodes), mat_temr2CF * y <= G_overlay.C_F];
    
        % % for i_F = 1:length(G_overlay.F_all)
        % %     F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
        % %     if length(F_eu) == 1
        % %         C_F = G_overlay.G_u.C(F_eu(1));
        % %     else
        % %         C_F = min( G_overlay.G_u.C(F_eu) );
        % %     end
        % %     F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
        % %     if isempty(F)
        % %         continue;
        % %     end
        % %     v_pairs = G_overlay.link2pair(F);
        % %     tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
        % %     Constraints = [Constraints, y(v_pairs)' * tmp_term2 <= C_F];
        % % end
        
        % Define an objective
        Objective = rho;
        
        % Set some options for YALMIP and solver
        options = sdpsettings('verbose',0,'solver','mosek');
        
        % Solve the problem
        sol = optimize(Constraints,Objective,options);
        
        % Analyze error flags
        solution = struct;
        if sol.problem == 0
            % Extract and display value
            solution.y = value(y);
            solution.rho = value(rho);
        else
            disp('Hmm, something went wrong!');
            sol.info
            yalmiperror(sol.problem)
        end

        y_bin = 1*(solution.y > 1e-5);
        constr_violation = mat_temr2CF * y_bin - G_overlay.C_F;
        if ~any(constr_violation > 0)
            constrViolated = 0;
            break;
        end

        [~, idx_sort] = sort(solution.y);
        for i_sort = 1 : length(idx_sort)
            idx_keep = idx_sort(end-i_sort+1);
            if y_keep(idx_keep) ~= 1
                y_tmp = y_keep;
                y_tmp(idx_keep) = 1;
                constr_violoation = mat_temr2CF * y_tmp - G_overlay.C_F;
                if ~any(constr_violoation > 1e-5) %% make sure already keeped pairs will not cause violation
                    y_keep(idx_keep) = 1;
                    break;
                end
            end
        end
    end
    elapsed_time = toc;

    % re-calc best rho
    L = Bp * diag(y_bin) * Bp';
    lambda_a_all = eigs(L,2,'smallestabs');
    lambda_a = lambda_a_all(2);
    solution.Ea = find(y_bin > 0);
    solution.y = y_bin;
    solution.elapsed_time = elapsed_time;
    if lambda_a > 0
        sol_rho = solve_rho(G_overlay, solution.Ea);
        solution.rho = sol_rho.rho;
        solution.success = 1;
    else
        solution.success = -2; % disconnected graph
    end

end

function solution = SCA_both_rho_ew(G_overlay, beta)
    % successively kicking out pairs based on exact solution (23)
    N_nodes = G_overlay.N_nodes;
    N_pairs = size(G_overlay.Bp, 2);
    Bp = G_overlay.Bp(G_overlay.Vo, :);
    alpha = 1 / (2*N_nodes+1);
    mat_temr2CF = gen_constr_CF_beta(G_overlay, beta);
    % % Ea_MST = [4, 8, 9, 11, 16, 18, 31, 33, 36];
    
    constrViolated = 1;
    y_kickout = ones(N_pairs,1); %% if a pair has been kicked out, y_kickout(e) = 0.
    y_keep = zeros(N_pairs,1); %% if a pair has been kicked out, y_kickout(e) = 0.
    % % y_keep(Ea_MST) = 1;
    tic;
    while constrViolated > 0
        yalmip('clear')
    
        % Define variables
        y = sdpvar(N_pairs,1);
        rho = sdpvar(1,1);
        
        J = (1 / N_nodes) * ones(N_nodes, N_nodes);
        
        Constraints = [y <= y_kickout, y >= y_keep, 0 <= y <= 1, -rho * eye(N_nodes) <= eye(N_nodes) - J - alpha * Bp * diag(y) * Bp' <= rho * eye(N_nodes), mat_temr2CF * y <= G_overlay.C_F + 1e-4];
    
        % % for i_F = 1:length(G_overlay.F_all)
        % %     F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
        % %     if length(F_eu) == 1
        % %         C_F = G_overlay.G_u.C(F_eu(1));
        % %     else
        % %         C_F = min( G_overlay.G_u.C(F_eu) );
        % %     end
        % %     F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
        % %     if isempty(F)
        % %         continue;
        % %     end
        % %     v_pairs = G_overlay.link2pair(F);
        % %     tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
        % %     Constraints = [Constraints, y(v_pairs)' * tmp_term2 <= C_F];
        % % end
        
        % Define an objective
        Objective = rho;
        
        % Set some options for YALMIP and solver
        options = sdpsettings('verbose',0,'solver','mosek');
        
        % Solve the problem
        sol = optimize(Constraints,Objective,options);
        
        % Analyze error flags
        solution = struct;
        if sol.problem == 0
            % Extract and display value
            solution.y = value(y);
            solution.rho = value(rho);
        else
            disp('Hmm, something went wrong!');
            sol.info
            yalmiperror(sol.problem)
        end

        y_bin = 1*(solution.y > 1e-5);
        constr_violation = mat_temr2CF * y_bin - G_overlay.C_F;
        if ~any(constr_violation > 1e-4)
            constrViolated = 0;
            break;
        end

        [~, idx_sort] = sort(solution.y);
        % [~, idx_sort] = sort(solution.alpha);
        pair2kickout = -1;
        for i_sort = 1 : length(idx_sort)
            idx = idx_sort(i_sort);
            if solution.y(idx) > 1e-7 && y_kickout(idx) ~= 0 && y_keep(idx) ~= 1
                pair2kickout = idx;
                y_kickout(idx) = 0;
                break;
            end
        end
        for i_sort = 1 : length(idx_sort)
            idx_keep = idx_sort(end-i_sort+1);
            if y_kickout(idx_keep) == 1 && y_keep(idx_keep) ~= 1
                y_tmp = y_keep;
                y_tmp(idx_keep) = 1;
                constr_violation = mat_temr2CF * y_tmp - G_overlay.C_F;
                if ~any(constr_violation > 1e-4) %% make sure already keeped pairs will not cause violation
                    y_keep(idx_keep) = 1;
                    break;
                end
            end
        end
    
        % % constr_violation = zeros(length(G_overlay.F_all), 1);
        
        if length(pair2kickout) > 1 || pair2kickout <=0
            disp("debug")
        else
            y_bin(pair2kickout) = 0;
        end
        
    end
    elapsed_time = toc;

    % re-calc best rho
    L = Bp * diag(y_bin) * Bp';
    lambda_a_all = eigs(L,2,'smallestabs');
    lambda_a = lambda_a_all(2);
    solution.Ea = find(y_bin > 0);
    solution.y = y_bin;
    solution.elapsed_time = elapsed_time;
    if lambda_a > 0
        sol_rho = solve_rho(G_overlay, solution.Ea);
        solution.rho = sol_rho.rho;
        solution.success = 1;
    else
        solution.success = -2; % disconnected graph
    end

end

function solution = SCA_joint_rho_y(G_overlay, beta)
    % successively kicking out pairs based on exact solution (24)
    N_nodes = G_overlay.N_nodes;
    N_pairs = size(G_overlay.Bp, 2);
    Bp = G_overlay.Bp(G_overlay.Vo, :);
    J = (1 / N_nodes) * ones(N_nodes, N_nodes);
    
    constrViolated = 1;
    y_kickout = ones(N_pairs,1); %% if a pair has been kicked out, y_kickout(e) = 0.
    y_keep = zeros(N_pairs,1); %% guaranteed to be left
    mat_temr2CF = gen_constr_CF_beta(G_overlay, beta);

    tic;
    while constrViolated > 0
        cvx_begin sdp quiet
            % Define solver
            % cvx_solver SeDuMi
            cvx_solver SDPT3
    
            % Define variables
            variable rho(1,1)
            variable a(N_pairs,1)
            variable y(N_pairs,1)
    
    
            % Construct L using alpha and Bp
            % % L = 0;
            % % for i = 1 : N_pairs
            % %     L = L + a(i) * Bp(:, i) * Bp(:, i)';
            % % end
            L = Bp*diag(a)*Bp';
    
            % Objective
            minimize( rho )
    
            subject to
                0 <= rho <= 1;
                0 <= y <= 1;
                y <= y_kickout;
                y >= y_keep;
    
                % Semidefinite constraints
                L_plus_J = L + J;
                eye(N_nodes) - L_plus_J + rho*eye(N_nodes) == semidefinite(N_nodes);
                rho*eye(N_nodes) - (eye(N_nodes) - L_plus_J) == semidefinite(N_nodes);
    
                % Linear constraints on alpha and y
                0 <= a <= y;
                mat_temr2CF * y <= G_overlay.C_F;
    
% %                 for i_F = 1:length(G_overlay.F_all)
% %                     F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
% %                     C_F = G_overlay.C_F(i_F);
% %                     F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
% %                     if isempty(F)
% %                         continue;
% %                     end
% %                     v_pairs = G_overlay.link2pair(F);
% %                     tmp_term2 = mat_temr2CF(i_F, v_pairs);
% % % %                     tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
% %                     y(v_pairs)' * tmp_term2 <= C_F;
% %                 end
        cvx_end
        solution = struct;
        solution.y = y;
        solution.alpha = a;
        % % yalmip('clear')
        % % 
        % % % Define variables
        % % y = sdpvar(N_pairs,1);
        % % alpha = sdpvar(N_pairs,1);
        % % rho = sdpvar(1,1);
        % % 
        % % J = (1 / N_nodes) * ones(N_nodes, N_nodes);
        % % 
        % % Constraints = [y <= y_kickout, 0 <= y <= 1, 0 <= alpha <= y, -rho * eye(N_nodes) <= eye(N_nodes) - J - Bp * diag(alpha) * Bp' <= rho * eye(N_nodes)];
        % % 
        % % for i_F = 1:length(G_overlay.F_all)
        % %     F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
        % %     if length(F_eu) == 1
        % %         C_F = G_overlay.G_u.C(F_eu(1));
        % %     else
        % %         C_F = min( G_overlay.G_u.C(F_eu) );
        % %     end
        % %     F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
        % %     if isempty(F)
        % %         continue;
        % %     end
        % %     v_pairs = G_overlay.link2pair(F);
        % %     tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
        % %     Constraints = [Constraints, y(v_pairs)' * tmp_term2 <= C_F];
        % % end
        % % 
        % % % Define an objective
        % % Objective = rho;
        % % 
        % % % Set some options for YALMIP and solver
        % % options = sdpsettings('verbose',0,'solver','mosek');
        % % 
        % % % Solve the problem
        % % sol = optimize(Constraints,Objective,options);
        % % 
        % % % Analyze error flags
        % % solution = struct;
        % % if sol.problem == 0
        % %     % Extract and display value
        % %     solution.y = value(y);
        % %     solution.rho = value(rho);
        % % else
        % %     disp('Hmm, something went wrong!');
        % %     sol.info
        % %     yalmiperror(sol.problem)
        % % end

        y_bin = 1*(solution.y > 1e-5);
        constr_violation = mat_temr2CF * y_bin - G_overlay.C_F;
        if ~any(constr_violation > 0)
            constrViolated = 0;
            break;
        end

        [~, idx_sort] = sort(solution.y);
        % [~, idx_sort] = sort(solution.alpha);
        pair2kickout = -1;
        for i_sort = 1 : length(idx_sort)
            idx = idx_sort(i_sort);
            if solution.y(idx) > 1e-7 && y_kickout(idx) ~= 0 && y_keep(idx) ~= 1
                pair2kickout = idx;
                y_kickout(idx) = 0;
                break;
            end
        end
        for i_sort = 1 : length(idx_sort)
            idx_keep = idx_sort(end-i_sort+1);
            if y_kickout(idx_keep) == 1 && y_keep(idx_keep) ~= 1
                y_tmp = y_keep;
                y_tmp(idx_keep) = 1;
                constr_violoation = mat_temr2CF * y_tmp - G_overlay.C_F;
                if ~any(constr_violoation > 1e-5) %% make sure already keeped pairs will not cause violation
                    y_keep(idx_keep) = 1;
                    break;
                end
            end
        end
    
        % % constr_violation = zeros(length(G_overlay.F_all), 1);
        
        % if length(pair2kickout) > 1 || pair2kickout <=0
        %     disp("debug")
        % end
        y_bin(pair2kickout) = 0;
        % % for i_F = 1:length(G_overlay.F_all)
        % %     F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
        % %     if length(F_eu) == 1
        % %         C_F = G_overlay.G_u.C(F_eu(1));
        % %     else
        % %         C_F = min( G_overlay.G_u.C(F_eu) );
        % %     end
        % %     F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
        % %     if isempty(F)
        % %         continue;
        % %     end
        % %     v_pairs = G_overlay.link2pair(F);
        % %     tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
        % %     constr_violation(i_F) = y_bin(v_pairs)' * tmp_term2 - C_F;
        % % end
        
    end
    elapsed_time = toc;

    % re-calc best rho
    L = Bp * diag(y_bin) * Bp';
    lambda_a_all = eigs(L,2,'smallestabs');
    lambda_a = lambda_a_all(2);
    solution.Ea = find(y_bin > 0);
    solution.y = y_bin;
    solution.elapsed_time = elapsed_time;
    if lambda_a > 0
        sol_rho = solve_rho(G_overlay, solution.Ea);
        solution.rho = sol_rho.rho;
        solution.success = 1;
    else
        solution.success = -2; % disconnected graph
    end

end

function solution = SDR_joint_rho_y(G_overlay, beta)
    % convex relaxation to SDP for the exact solution (24)
    N_nodes = G_overlay.N_nodes;
    N_pairs = size(G_overlay.Bp, 2);
    Bp = G_overlay.Bp(G_overlay.Vo, :);
    J = (1 / N_nodes) * ones(N_nodes, N_nodes);
    mat_temr2CF = gen_constr_CF_beta(G_overlay, beta);
    % alpha = 1 / N_nodes;

    % % cvx_begin sdp
    % %     % Define solver
    % %     % cvx_solver SeDuMi
    % %     cvx_solver SDPT3
    % % 
    % %     % Define variables
    % %     variable rho(1,1)
    % %     variable a(N_pairs,1)
    % %     variable y(N_pairs,1)
    % % 
    % % 
    % %     % Construct L using alpha and Bp
    % %     % % L = 0;
    % %     % % for i = 1 : N_pairs
    % %     % %     L = L + a(i) * Bp(:, i) * Bp(:, i)';
    % %     % % end
    % %     L = Bp*diag(a)*Bp';
    % % 
    % %     % Objective
    % %     minimize( rho )
    % % 
    % %     subject to
    % %         0 <= rho <= 1;
    % %         0 <= y <= 1;
    % % 
    % %         % Semidefinite constraints
    % %         L_plus_J = L + J;
    % %         eye(N_nodes) - L_plus_J + rho*eye(N_nodes) == semidefinite(N_nodes);
    % %         rho*eye(N_nodes) - (eye(N_nodes) - L_plus_J) == semidefinite(N_nodes);
    % % 
    % %         % Linear constraints on alpha and y
    % %         0 <= a <= y;
    % % 
    % %         for i_F = 1:length(G_overlay.F_all)
    % %             F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
    % %             if length(F_eu) == 1
    % %                 C_F = G_overlay.G_u.C(F_eu(1));
    % %             else
    % %                 C_F = min( G_overlay.G_u.C(F_eu) );
    % %             end
    % %             F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
    % %             if isempty(F)
    % %                 continue;
    % %             end
    % %             v_pairs = G_overlay.link2pair(F);
    % %             tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
    % %             y(v_pairs)' * tmp_term2 <= C_F;
    % %         end
    % % cvx_end
    % % 
    % % % Process solution
    % % solution = struct;
    % % solution.rho = cvx_optval;
    % % if strcmp(cvx_status, 'Solved')
    % %     solution.alpha = a;
    % %     solution.y = y;
    % %     solution.success = 1;
    % % else
    % %     disp('Hmm, something went wrong!');
    % %     solution.success = 0;
    % %     solution.info = cvx_status;
    % % end

    yalmip('clear')

    tic;
    
    % Define variables
    y = sdpvar(N_pairs,1);
    alpha = sdpvar(N_pairs,1);
    rho = sdpvar(1,1);
    
    Constraints = [0 <= y <= 1, 0 <= alpha <= y, -rho * eye(N_nodes) <= eye(N_nodes) - J - Bp * diag(alpha) * Bp' <= rho * eye(N_nodes), mat_temr2CF * y <= G_overlay.C_F];

    % % for i_F = 1:length(G_overlay.F_all)
    % %     F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
    % %     if length(F_eu) == 1
    % %         C_F = G_overlay.G_u.C(F_eu(1));
    % %     else
    % %         C_F = min( G_overlay.G_u.C(F_eu) );
    % %     end
    % %     F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
    % %     if isempty(F)
    % %         continue;
    % %     end
    % %     v_pairs = G_overlay.link2pair(F);
    % %     tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
    % %     Constraints = [Constraints, y(v_pairs)' * tmp_term2 <= C_F];
    % % end

    % Define an objective
    Objective = rho;
    
    % Set some options for YALMIP and solver
    options = sdpsettings('verbose',0,'solver','mosek');
    
    % Solve the problem
    sol = optimize(Constraints,Objective,options);
    
    % Analyze error flags
    solution = struct;
    if sol.problem == 0
        % Extract and display value
        solution.y = value(y);
        solution.rho = value(rho);
    else
        disp('Hmm, something went wrong!');
        sol.info
        yalmiperror(sol.problem)
    end

    % % constr_violation = zeros(length(G_overlay.F_all), 1);
    y_bin = 1*(solution.y > 1e-5);
    % % for i_F = 1:length(G_overlay.F_all)
    % %     F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
    % %     if length(F_eu) == 1
    % %         C_F = G_overlay.G_u.C(F_eu(1));
    % %     else
    % %         C_F = min( G_overlay.G_u.C(F_eu) );
    % %     end
    % %     F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
    % %     if isempty(F)
    % %         continue;
    % %     end
    % %     v_pairs = G_overlay.link2pair(F);
    % %     tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
    % %     constr_violation(i_F) = y_bin(v_pairs)' * tmp_term2 - C_F;
    % % end
    constr_violation = mat_temr2CF * y_bin - G_overlay.C_F;


    % rouding to s.t. beta constraints
    [~, idx_sort] = sort(solution.y);
    y_keep = y_bin;
    for k_sort = 1 : length(idx_sort)
        k = idx_sort(k_sort);
        if solution.y(k) <= 1e-4 %% weight is too small
            y_keep(k) = 0;
            continue;
        else
            % Determine whether removing this pair can help reduce the constr violation
            y_keep(k) = 0;
            is_remove_k = 0;
            for i_F = 1:length(G_overlay.F_calc)
                if constr_violation(i_F) <= 0
                    continue;
                end
                new_violation = mat_temr2CF(i_F, :) * y_keep - G_overlay.C_F(i_F);
                % % F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
                % % if length(F_eu) == 1
                % %     C_F = G_overlay.G_u.C(F_eu(1));
                % % else
                % %     C_F = min( G_overlay.G_u.C(F_eu) );
                % % end
                % % F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
                % % if isempty(F)
                % %     continue;
                % % end
                % % v_pairs = G_overlay.link2pair(F);
                % % tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
                % % new_violation = y_keep(v_pairs)' * tmp_term2 - C_F;
                if new_violation < constr_violation(i_F)
                    is_remove_k = 1;
                    constr_violation(i_F) = new_violation;
                end
            end
            if is_remove_k == 0
                y_keep(k) = 1;
            end
            if ~any(constr_violation > 0)
                break;
            end
        end
    end

    elapsed_time = toc;

    % re-calc best rho
    L = Bp * diag(y_keep) * Bp';
    lambda_a_all = eigs(L,2,'smallestabs');
    lambda_a = lambda_a_all(2);
    solution.Ea = find(y_keep > 0);
    solution.y = y_keep;
    solution.elapsed_time = elapsed_time;
    if lambda_a > 0
        sol_rho = solve_rho(G_overlay, solution.Ea);
        solution.rho = sol_rho.rho;
        solution.success = 1;
    else
        solution.success = -2; % disconnected graph
    end
end

function solution = heuristic_SDR(G_overlay, beta)
    % convex relaxation to SDP, then directly rounding
    N_nodes = G_overlay.N_nodes;
    N_pairs = size(G_overlay.Bp, 2);
    Bp = G_overlay.Bp(G_overlay.Vo, :);
    yalmip('clear')
    
    % Define variables
    y = sdpvar(N_pairs,1);
    s = sdpvar(1,1);
    
    J = (1 / N_nodes) * ones(N_nodes, N_nodes);
    
    Constraints = [0 <= y <= 1, s * (eye(N_nodes) - J) <= Bp * diag(y) * Bp'];

    for i_F = 1:length(G_overlay.F_calc)
        % % F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
        % % if length(F_eu) == 1
        % %     C_F = G_overlay.G_u.C(F_eu(1));
        % % else
        % %     C_F = min( G_overlay.G_u.C(F_eu) );
        % % end
        % % F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F

        F = G_overlay.F_calc{i_F};
        C_F= G_overlay.C_F(i_F);
        if isempty(F)
            continue;
        end
        v_pairs = G_overlay.link2pair(F);
        tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
        Constraints = [Constraints, y(v_pairs)' * tmp_term2 <= C_F];
    end
    
    % Define an objective
    Objective = -s;
    
    % Set some options for YALMIP and solver
    options = sdpsettings('verbose',0,'solver','mosek');
    
    % Solve the problem
    sol = optimize(Constraints,Objective,options);
    
    % Analyze error flags
    solution = struct;
    if sol.problem == 0
        % Extract and display value
        solution.y = value(y);
        solution.s = value(s);
    else
        disp('Hmm, something went wrong!');
        sol.info
        yalmiperror(sol.problem)
    end

    constr_violation = zeros(length(G_overlay.F_calc), 1);
    y_bin = 1*(solution.y > 1e-5);
    for i_F = 1:length(G_overlay.F_calc)
        % % F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
        % % if length(F_eu) == 1
        % %     C_F = G_overlay.G_u.C(F_eu(1));
        % % else
        % %     C_F = min( G_overlay.G_u.C(F_eu) );
        % % end
        % % F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F

        F = G_overlay.F_calc{i_F};
        C_F= G_overlay.C_F(i_F);
        if isempty(F)
            continue;
        end
        v_pairs = G_overlay.link2pair(F);
        tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
        constr_violation(i_F) = y_bin(v_pairs)' * tmp_term2 - C_F;
    end


    % rouding to s.t. beta constraints
    [~, idx_sort] = sort(solution.y);
    y_keep = y_bin;
    for k_sort = 1 : length(idx_sort)
        k = idx_sort(k_sort);
        if solution.y(k) <= 1e-4 %% weight is too small
            y_keep(k) = 0;
            continue;
        else
            % Determine whether removing this pair can help reduce the constr violation
            y_keep(k) = 0;
            is_remove_k = 0;
            for i_F = 1:length(G_overlay.F_calc)
                if constr_violation(i_F) <= 0
                    continue;
                end
                % % F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
                % % if length(F_eu) == 1
                % %     C_F = G_overlay.G_u.C(F_eu(1));
                % % else
                % %     C_F = min( G_overlay.G_u.C(F_eu) );
                % % end
                % % F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F

                F = G_overlay.F_calc{i_F};
                C_F= G_overlay.C_F(i_F);
                if isempty(F)
                    continue;
                end
                v_pairs = G_overlay.link2pair(F);
                tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
                new_violation = y_keep(v_pairs)' * tmp_term2 - C_F;
                if new_violation < constr_violation(i_F)
                    is_remove_k = 1;
                    constr_violation(i_F) = new_violation;
                end
            end
            if is_remove_k == 0
                y_keep(k) = 1;
            end
            if ~any(constr_violation > 0)
                break;
            end
            % deprecated: solve the routing problem to see whether tau_tmp <= beta.
            % % Ea_tmp = find(y_keep > 0);
            % % sol_tau = solve_tau_noRouting(G_overlay, Ea_tmp);
            % % tau_tmp = sol_tau.tau;
            % % if tau_tmp <= beta
            % %     solution.Ea = Ea_tmp;
            % %     solution.y = y_keep;
            % %     break;
            % % else
            % %     y_keep(k) = 0;
            % % end
        end
    end

    % re-calc best rho
    L = Bp * diag(y_keep) * Bp';
    lambda_a_all = eigs(L,2,'smallestabs');
    lambda_a = lambda_a_all(2);
    solution.Ea = find(y_keep > 0);
    solution.y = y_keep;
    if lambda_a > 0
        sol_rho = solve_rho(G_overlay, solution.Ea);
        solution.rho = sol_rho.rho;
        solution.success = 1;
    else
        solution.success = -2; % disconnected graph
    end
end

function Ea = gen_random_ring(G_overlay)
%% generate a random overlay ring
    shuffle_idx = randperm(G_overlay.N_nodes);
    Vo_shuffle = G_overlay.Vo(shuffle_idx);
    Ea = zeros(1, length(shuffle_idx));
    for i_u = 1 : length(Vo_shuffle)-1
        u = Vo_shuffle(i_u);
        v = Vo_shuffle(i_u + 1);
        keys = strjoin(string([min([u,v]), max([u,v])]));
        Ea(i_u) = G_overlay.st2idx( keys );
    end
    % close the ring: tail to head
    u = Vo_shuffle(end);
    v = Vo_shuffle(1);
    keys = strjoin(string([min([u,v]), max([u,v])]));
    Ea(end) = G_overlay.st2idx( keys );
end

function solution = solve_tau_noRouting(G_overlay, Ea)
%% solve tau in the case that there is no overlay routing. Ea: vector of pair_idx
    yalmip('clear')
    
    % Define variables
    f = sdpvar(G_overlay.N_links,1);
    tau = sdpvar(1,1);
    Eo = unique(G_overlay.pair2link(Ea, :));
    
    Constraints = [f >= 0, tau >= 0, f(setdiff(1:G_overlay.N_links, Eo)) <= 1e-4];
    for i_Ea = 1 : length(Ea)
        pair_idx = Ea(i_Ea);
        % u = find( G_overlay.Bp(:, pair_idx) == 1 );
        % v = find( G_overlay.Bp(:, pair_idx) == -1 );
        % eo_1 = G_overlay.st2eo( strjoin(string([u, v])) );
        % eo_2 = G_overlay.st2eo( strjoin(string([v, u])) );
        eo_1 = G_overlay.pair2link(pair_idx, 1);
        eo_2 = G_overlay.pair2link(pair_idx, 2);
        Constraints = [Constraints, tau >= G_overlay.Kappa / f(eo_1) + G_overlay.delays(eo_1), tau >= G_overlay.Kappa / f(eo_2) + G_overlay.delays(eo_2)]; %% tau-f constraints
    end
    
    % new: estimated F_calc (tunnels)
    for i_F = 1 : length(G_overlay.F_calc)
        F_eo = G_overlay.F_calc{i_F}; %% eo in this F
        Constraints = [Constraints, sum( f(F_eo) ) <= G_overlay.C_F(i_F)];
    end

    % deprecated: Only for true F_all.
    % % for i_F = 1 : length(G_overlay.F_all)
    % %     F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
    % %     % F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
    % %     % s_vec = s_vec .* 0;
    % %     % s_vec( 1, G_overlay.eo2eu(:, F_eu(1)) > 0 ) = 1; %% equivalent to F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); s_vec(F) = 1
    % %     if length(F_eu) == 1
    % %         C_F = G_overlay.G_u.C(F_eu(1));
    % %     else
    % %         C_F = min( G_overlay.G_u.C(F_eu) );
    % %     end
    % %     Constraints = [Constraints, sum(f( G_overlay.eo2eu(:, F_eu(1)) > 0 )) <= C_F];
    % % end
    
    % Define an objective
    Objective = tau;
    
    % Set some options for YALMIP and solver
    options = sdpsettings('verbose',0,'solver','mosek');
    
    % Solve the problem
    sol = optimize(Constraints,Objective,options);
    
    % Analyze error flags
    solution = struct;
    if sol.problem == 0
        % Extract and display value
        solution.f = value(f);
        solution.tau = value(tau);
    else
        disp('Hmm, something went wrong!');
        sol.info
        yalmiperror(sol.problem)
    end

end

function K = gen_K_from_rho(rho, M_1, M_2)
% calculate the number of iterations from equation
p = 1 - rho;
% % zeta = 2.6;
% % epsilon_0 = 0.07;
% % sigma = 2;
% % M_1 = 2.21;
% % M_2 = 2.2;

zeta = 0.5;
epsilon_0 = 0.06;
sigma = 4;
K = sigma^2/(10*epsilon_0^2) + (zeta*sqrt(M_1+1) + sigma*sqrt(p)) ./ (p*epsilon_0^(1.5)) + sqrt((M_1+1) * (M_2+1)) ./ (p*epsilon_0);
end





function mat_temr2CF = gen_constr_CF_beta(G_overlay, beta)
    % generate tmp_term2 vector given beta for all F_all
    % % mat_temr2CF = zeros(length(G_overlay.F_all), size(G_overlay.Bp, 2));
    mat_temr2CF = zeros(length(G_overlay.F_calc), size(G_overlay.Bp, 2));
    for i_F = 1:length(G_overlay.F_calc)
        % % F = find(G_overlay.F2link(i_F, :) > 0); %% tunnels in F
        F = G_overlay.F_calc{i_F};
        v_pairs = G_overlay.link2pair(F); %% pairs in F
        tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
        if length(v_pairs) == length(unique(v_pairs))
            mat_temr2CF(i_F, v_pairs) = tmp_term2;
        else
            for i_T = 1 : length(F)
                mat_temr2CF(i_F, v_pairs(i_T)) = mat_temr2CF(i_F, v_pairs(i_T)) + tmp_term2(i_T);
            end
        end
    end
end

function A = gen_linkSet_connect(G_overlay)
% generate a subset of overlay link A to form a connected subgraph
    N_links = G_overlay.N_links;
    N_nodes = G_overlay.N_nodes;
    shuffle_nodes = randperm(N_nodes); %% shuffle overlay node indices to form a path for connectivity
    vec_A = zeros(1, N_links); %% an indicator vector for the chosen links in A
    for i_u = 1 : N_nodes-1 %% add links forming the path (tree) to vec_A
        s_0 = shuffle_nodes(i_u);
        t_0 = shuffle_nodes(i_u+1);
        s = min([s_0, t_0]);
        t = max([s_0, t_0]);
        link_id = G_overlay.st2link( strjoin(string([s, t])) );
        vec_A(link_id) = 1;
    end
    
    links_not_select = find(vec_A == 0);
    shuffle_links_rest = randperm(length(links_not_select));
    rd_cardinality = randi([1, length(links_not_select)-4],1,1); %% assume that #links not selected is at least 5.
    vec_A( shuffle_links_rest(1:rd_cardinality) ) = 1;
    A = find(vec_A > 0);
end


function [B,x] = gen_B_from_S_with_A(S, A)
% Given superset S and a subset A, randomly choose B with a random cardinality
    S_minus_A = setdiff(S, A);
    shuffle_index = randperm(length(S_minus_A)); %% shuffle index
    B_cardinality = randi([1, length(S_minus_A)-2],1,1); %% randomly gen a |B|
    B = S_minus_A(shuffle_index(1:B_cardinality));
    B = sort( [A, B] );
    S_minus_B = setdiff(S, B);
    shuffle_index = randperm(length(S_minus_B)); %% shuffle index
    x = S_minus_B( shuffle_index(1) );
end

function solution = solve_rho(G_overlay, E_a)
    %%% E_a is list of overlay links.
    % N_links = G_overlay.N_links;
    N_nodes = G_overlay.N_nodes;
    N_pairs = size(G_overlay.Bp, 2);
    Bp = G_overlay.Bp(G_overlay.Vo, :);
    E_a_c = setdiff(1:N_pairs, E_a); %% complement of E_a over E
    
    % Every time you call sdpvar etc, an internal database grows larger
    yalmip('clear')
    
    % Define variables
    rho = sdpvar(1,1);
    alpha = sdpvar(N_pairs,1);
    J = (1 / N_nodes) * ones(N_nodes, N_nodes);
    
    % Define constraints 
    Constraints = [];
    for i = 1 : length(E_a_c)
      Constraints = [Constraints, alpha(E_a_c(i)) == 0];
    end
    % % Constraints = [Constraints, alpha(E_a_c(i)) == 0];
    % % sum_alpha_B = 0;
    % % for i = 1 : size(B, 2)
    % %     sum_alpha_B = sum_alpha_B + alpha(i) * B(:, i) * B(:, i)';
    % % end
    % % Constraints = [Constraints, eye(N_nodes) - sum_alpha_B - rho*eye(N_nodes) - J <= 0, ...
    % %     eye(N_nodes) - sum_alpha_B + rho*eye(N_nodes) - J >= 0];
    
    Constraints = [Constraints, eye(N_nodes) - Bp*diag(alpha)*Bp' - J - rho*eye(N_nodes) <= 0, ...
        eye(N_nodes) - Bp*diag(alpha)*Bp' - J + rho*eye(N_nodes) >= 0];
    
    % Define an objective
    Objective = rho;
    
    % Set some options for YALMIP and solver
    options = sdpsettings('verbose',0,'solver','mosek');

    % Solve the problem
    sol = optimize(Constraints,Objective,options);

    % Analyze error flags
    solution = struct;
    if sol.problem == 0
        % Extract and display value
        solution.rho = value(rho);
        solution.alpha = value(alpha);
    else
        disp('Hmm, something went wrong!');
        sol.info
        yalmiperror(sol.problem)
    end

end

function [T, runTime] = benchmark_random_ring(G_overlay, M1, M2)
    tic;
    Ea_ring = gen_random_ring(G_overlay);
    solution_ring = solve_tau_noRouting(G_overlay, Ea_ring);
    tau_lb = solution_ring.tau;
    sol_rho = solve_rho(G_overlay, Ea_ring);
    y = sol_rho.alpha;
    rho = sol_rho.rho;
    K_tmp = gen_K_from_rho(rho, M1, M2);

    T = K_tmp * tau_lb;
    runTime = toc;
end

function [T, runTime] = benchmark_allPair(G_overlay, M1, M2)
    tic;
    Ea = 1 : size(G_overlay.Bp, 2);
    solution_E = solve_tau_noRouting(G_overlay, Ea);
    tau_lb = solution_E.tau;
    sol_rho = solve_rho(G_overlay, Ea);
    y = sol_rho.alpha;
    rho = sol_rho.rho;
    K_tmp = gen_K_from_rho(rho, M1, M2);

    T = K_tmp * tau_lb;
    runTime = toc;
end

function [T, runTime] = benchmark_randomMST(G_overlay, M1, M2)
    shuffle_idx = randperm(G_overlay.N_nodes);
    tic;
    G = graph(G_overlay.idx2st(:, 1)', G_overlay.idx2st(:, 2)', G_overlay.pairDelay);
    T_root = G_overlay.Vo(shuffle_idx(1));
    MST = minspantree(G,'Root', T_root);
    MST_edges = table2array(MST.Edges(:, 1:end-1));
    Ea = zeros(size(MST_edges, 1), 1);
    for i_st = 1 : size(MST_edges, 1)
        s = min(MST_edges(i_st, :));
        t = max(MST_edges(i_st, :));
        p_idx = G_overlay.st2idx( strjoin(string([s, t])) );
        Ea(i_st) = p_idx;
    end
    solution_E = solve_tau_noRouting(G_overlay, Ea);
    tau_lb = solution_E.tau;
    sol_rho = solve_rho(G_overlay, Ea);
    y = sol_rho.alpha;
    rho = sol_rho.rho;
    K_tmp = gen_K_from_rho(rho, M1, M2);

    T = K_tmp * tau_lb;
    runTime = toc;
end

function prepare_constr_C_given_beta(G_overlay, beta)
 % prepare the capacity constraint given beta
    for i_F = 1:length(G_overlay.F_calc)
        % % F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
        % % if length(F_eu) == 1
        % %     C_F = G_overlay.G_u.C(F_eu(1));
        % % else
        % %     C_F = min( G_overlay.G_u.C(F_eu) );
        % % end
        % % F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F

        F = G_overlay.F_calc{i_F};
        C_F = G_overlay.C_F(i_F);
        if isempty(F)
            continue;
        end
        v_pairs = G_overlay.link2pair(F);
        tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
        constr_violation(i_F) = y_bin(v_pairs)' * tmp_term2 - C_F;
    end

end

function read_graph_file(filename)
    
    % Open the file for reading
    fileID = fopen(filename, 'r');
    
    % Check if the file was successfully opened
    if fileID == -1
        error('File cannot be opened: %s', filename);
    end
    
    % Read and display each line of the file
    lineNumber = 1;
    while ~feof(fileID)
        line = fgets(fileID); % Reads a line, including newline characters
        newStr = split(line, ' ');
        if line(1) == 'M'
            N_nodes = str2double(newStr(2));
        elseif line(1) == 'E'
            N_edges = str2double(newStr(2));
        elseif line(1) == 'e'
            u = str2double(newStr(2)) + 1;
            v = str2double(newStr(3)) + 1;
            delay
        end
        lineNumber = lineNumber + 1;
    end
    
    % Close the file
    fclose(fileID);

end

function write_spr_to_file(G_overlay, filename)
    % write SPR routing to file
    % Open file for writing (overwrites existing content)
    fileID = fopen(filename, 'w'); 
    for i_u = 1 : length(G_overlay.Vo)
        u = G_overlay.Vo(i_u);
        for i_v = 1 : length(G_overlay.Vo)
            v = G_overlay.Vo(i_v);
            if v == u
                continue;
            end
            route = G_overlay.st2spr( strjoin(string([u, v])) );
            route2wr = route - 1; %% To 0-indexed
            string2wr = strcat(string(u-1), " ", string(v-1), ": ", string(length(route)), ": ", strjoin(string(route2wr)), "\n");
            fprintf(fileID, string2wr);
        end
    end
    fclose(fileID); % Close the file
end



function write_roofnet_to_Graph(G_overlay, filename)
    % write roofnet D into .Graph file
    G_u = G_overlay.G_u;
    N_Vu = size(G_u.Adj, 1);
    N_Eu = size(G_u.D, 2);

    fileID = fopen(filename, 'w'); 

    fprintf(fileID, strcat("NODES ", string(N_Vu), '\n'));
    fprintf(fileID, "label x y\n");
    for i_u = 1 : N_Vu
        fprintf(fileID, strcat(string(i_u-1), "_location 1 1\n"));
    end
    fprintf(fileID, "\n");
    fprintf(fileID, strcat("EDGES ", string(N_Eu), "\n"));
    fprintf(fileID, "label src dest weight bw delay\n");
    
    i_edge = 0;
    for i_e = 1 : N_Eu
        src = find(G_u.D(:, i_e) == 1);
        dst = find(G_u.D(:, i_e) == -1);
        fprintf(fileID, strcat('edge_', string(i_edge), " ", string(src-1), " ",string(dst-1), " 1000 1000 0.5 \n"));
        i_edge = i_edge + 1;
    end

    fclose(fileID); % Close the file
end



function write_Ea_to_file(G_overlay, filename, Ea_mat_input)

    % write Ea to Adj form
    N_Vu = size(G_overlay.G_u.Adj, 2);
    N_Ea = size(Ea_mat_input, 2); %% #different Ea 
    Ea_dict = {};
    tau_dict = {};
    
    idx_Ea = 1;
    for i_Ea = 1 : N_Ea
        new_Ea = 1;
        if sum(Ea_mat_input(:, i_Ea)) == 0
            continue;
        end
        for j_Ea = 1 : i_Ea-1
            tmp = Ea_mat_input(:, i_Ea) + Ea_mat_input(:, j_Ea);
            if ~any(tmp == 1) %% repeated Ea
                new_Ea = 0;
                break;
            end
            new_Ea = 1;
        end
        if new_Ea > 0
            Ea_Adj = zeros(N_Vu, N_Vu);
            Ea_local = find(Ea_mat_input(:, i_Ea) > 0);
            for i_e = 1 : length(Ea_local)
                pair = Ea_local(i_e);
                src = G_overlay.idx2st(pair, 1);
                dst = G_overlay.idx2st(pair, 2);
                Ea_Adj(src, dst) = 1;
                Ea_Adj(dst, src) = 1;
            end
            Ea_dict{idx_Ea} = Ea_Adj;
            tau_sol = solve_tau_noRouting(G_overlay, Ea_local);
            tau_dict{idx_Ea} = tau_sol.tau;
            idx_Ea = idx_Ea + 1;
        end
    end
    save(filename,"Ea_dict", "tau_dict");
end

function write_Adj2graph(G_overlay, graph_name)
    % graph_name in the form of 'Roofnet_'
    Adj = G_overlay.G_u.Adj;
    [row, col] = find(Adj);
    filename = strcat(graph_name, 'underlay.Graph');
    fileID = fopen(filename, 'w'); 
    fprintf(fileID, strcat("NODES ", string(size(Adj, 1)), '\n'));
    fprintf(fileID, "label x y\n");
    for i_u = 1 : size(Adj, 1)
        fprintf(fileID, strcat(string(i_u-1), "_location 1 1\n"));
    end
    fprintf(fileID, "\n");
    fprintf(fileID, strcat("EDGES ", string(length(row)), "\n"));
    fprintf(fileID, "label src dest weight bw delay\n");
    
    i_edge = 0;
    for i_row = 1 : length(row)
        fprintf(fileID, strcat('edge_', string(i_row), " ", string(row(i_row)-1), " ",string(col(i_row)-1), " 1000 0.4 0.0005 \n"));
        i_edge = i_edge + 1;
    end

    fclose(fileID); % Close the file
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