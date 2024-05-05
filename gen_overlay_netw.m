function G_overlay = gen_overlay_netw(N_nodes, N_links, G_type, modelType, F_calculated)
if G_type == 0
    G_u = gen_underlay_netw(1, G_type);
    % N_nodes = size(G_u.Adj, 1);
    % N_links = N_nodes * (N_nodes - 1); %% assume directed overlay links
elseif G_type == 1
    G_u = read_RoofNet();
else
    disp("!!!!Wrong G_type!!!!")
end

G_overlay = struct;
G_overlay.G_type = G_type;
G_overlay.C_F_lb = 1 - 1.5 / 100; %% lb of C_F estimation for unifrnd
G_overlay.G_u = G_u;
% randomly select N_nodes underlay nodes from Vu as overlay nodes
N_Vu = size(G_u.Adj, 1);
% read Vo and routing table from file
% Vo = [4 7 8 10 11 13 14 16 17 19];
[Vo, spr_table] = read_ov_routeTable(G_type);
G_overlay.spr_table = spr_table; %%%%%  routing table.
G_overlay.st2spr = spr_table;
G_overlay.N_nodes = length(Vo);
G_overlay.Vo = Vo;
N_links = G_overlay.N_nodes * (G_overlay.N_nodes-1);

% deprecated
% % shuffle_Vu = randperm(N_Vu);
% % G_overlay.N_nodes = N_nodes;
% % G_overlay.Vo = sort( shuffle_Vu(1:N_nodes) ); %% randomly select nodes

G_overlay.Vu2Vo = zeros(N_Vu, 1); %% i-th vu to j-th vo
G_overlay.Vu2Vo(G_overlay.Vo) = (1:length(G_overlay.Vo))';
G_overlay.N_links = N_links;
if strcmp(modelType, "CIFAR10")
    N_param = 25.6 * 1e6;
elseif strcmp(modelType, "MNIST")
    N_param = 1663370;
else
    disp("Non-existing model type")
end
G_overlay.Kappa = N_param * 64 / 1e6; %% data volume in Mbits. Kappa/f = Mbits/Gbps = 1e-3 sec = ms
% Create pair. !!Attention!! In the following, we assign each overlay pair an index. Overlay link eo has no index!
G_overlay.Bp = zeros(N_nodes, nchoosek(N_nodes,2)); %% map from idx to ST pair
G_overlay.st2idx = containers.Map(); %% HashTable mapping ST pair to overlay pair ID: <(s, t), idx>
G_overlay.idx2st = zeros(size(G_overlay.Bp, 2), 2); %% pair_idx to corresponding <s, t>
G_overlay.pair2link = zeros(size(G_overlay.Bp, 2), 2); %% pair_idx to overlay link ID
G_overlay.link2pair = zeros(N_links); %% overlay link to pair_idx
G_overlay.pairDelay = zeros(size(G_overlay.Bp, 2), 1); %% avg delay of each pair
% Create link eo
G_overlay.B = zeros(N_nodes, N_links); %% map from eo idx to eo pair <idx, (s,t)>
G_overlay.st2eo = containers.Map(); %% HashTable mapping ST pair to overlay link ID: <(s, t), eo>
G_overlay.st2spr = containers.Map(); %% HashTable mapping eo to underlay path: <(s, t), vector>
G_overlay.link2st = zeros(N_links, 2); %% eo -> (s, t)
G_overlay.eo2eu = zeros(N_links, size(G_u.D, 2)); %% mat mapping from eo to eu
G_overlay.delays = zeros(N_links, 1); %% overlay link propagation delay
i_pair = 1;
n_pair = size(G_overlay.Bp, 2);
for i_u = 1 : size(G_overlay.Vo, 2)
    u = G_overlay.Vo(i_u);
    for i_v = i_u+1 : size(G_overlay.Vo, 2)
        v = G_overlay.Vo(i_v);
        G_overlay.Bp(u, i_pair) = 1;
        G_overlay.Bp(v, i_pair) = -1;
        G_overlay.st2idx( strjoin(string([u, v])) ) = i_pair;
        G_overlay.idx2st(i_pair, 1) = u;
        G_overlay.idx2st(i_pair, 2) = v;

        i_link = i_pair + n_pair;
        G_overlay.B(u, i_pair) = 1;
        G_overlay.B(v, i_pair) = -1;
        G_overlay.B(u, i_link) = -1;
        G_overlay.B(v, i_link) = 1;
        G_overlay.st2eo( strjoin(string([u, v])) ) = i_pair;
        G_overlay.st2eo( strjoin(string([v, u])) ) = i_link;
        G_overlay.link2st(i_pair, 1) = u;
        G_overlay.link2st(i_pair, 2) = v;
        G_overlay.link2st(i_link, 1) = v;
        G_overlay.link2st(i_link, 2) = u;
        G_overlay.pair2link(i_pair, 1) = i_pair;
        G_overlay.pair2link(i_pair, 2) = i_link;
        G_overlay.link2pair(i_pair) = i_pair;
        G_overlay.link2pair(i_link) = i_pair;

        % create overlay delays from the imported rouing table
        route_uv = G_overlay.spr_table( strjoin(string([u, v])) );
        eo_uv = i_pair;
        eo_vu = i_link; %% reverse link
        for j = 1 : length(route_uv) - 1
            um = route_uv(j);
            vm = route_uv(j+1);
            eu = G_u.ST2idx( strjoin(string([um, vm])) );
            G_overlay.eo2eu(eo_uv, eu) = 1;
            G_overlay.delays(eo_uv) = G_overlay.delays(eo_uv) + G_overlay.G_u.delays(eu);
            eu_vu = G_u.ST2idx( strjoin(string([vm, um])) ); %% reverse link
            G_overlay.eo2eu(eo_vu, eu_vu) = 1; %% reverse link
        end
        G_overlay.delays(eo_vu) = G_overlay.delays(eo_uv); %% reverse link

        i_pair = i_pair + 1;
    end
    % gen SPR table. Enforce symmetrical routing path
    % % SPR_u = shortestpathtree(G_u.DG, u, G_overlay.Vo, 'OutputForm','cell');
    % % for i_v = 1:length(SPR_u)
    % %     if SPR_u{i_v}(end) == u
    % %         continue;
    % %     end
    % %     v = SPR_u{i_v}(end);
    % %     if isKey(G_overlay.st2spr, strjoin(string([v, u])))
    % %         route_vu = G_overlay.st2spr( strjoin(string([v, u])) );
    % %         route = flip(route_vu);
    % %     else
    % %         route = SPR_u{i_v};
    % %     end
    % %
    % %     G_overlay.st2spr( strjoin(string([u, v])) ) = route;
    % %     eo = G_overlay.st2eo( strjoin(string([u, v])) );
    % %     for j = 1 : length(route) - 1
    % %         um = route(j);
    % %         vm = route(j+1);
    % %         eu = G_u.ST2idx( strjoin(string([um, vm])) );
    % %         G_overlay.eo2eu(eo, eu) = 1;
    % %         G_overlay.delays(eo) = G_overlay.delays(eo) + G_overlay.G_u.delays(eu);
    % %     end
    % % end

    % deprecated
    % % SPR_u = shortestpathtree(G_u.DG, u, G_overlay.Vo, 'OutputForm','cell');
    % % for i_v = 1:length(SPR_u)
    % %     if SPR_u{i_v}(end) == u
    % %         continue;
    % %     end
    % %     route = SPR_u{i_v};
    % %     v = route(end);
    % %     G_overlay.st2spr( strjoin(string([u, v])) ) = route;
    % %     eo = G_overlay.st2eo( strjoin(string([u, v])) );
    % %     for j = 1 : length(route) - 1
    % %         um = route(j);
    % %         vm = route(j+1);
    % %         eu = G_u.ST2idx( strjoin(string([um, vm])) );
    % %         G_overlay.eo2eu(eo, eu) = 1;
    % %         G_overlay.delays(eo) = G_overlay.delays(eo) + G_overlay.G_u.delays(eu);
    % %     end
    % % end
end
for i_p = 1 : size(G_overlay.Bp, 2)
    eo1 = G_overlay.pair2link(i_p, 1);
    eo2 = G_overlay.pair2link(i_p, 2);
    G_overlay.pairDelay(i_p) = G_overlay.delays(eo1) + G_overlay.delays(eo2);
end
G_overlay.F_all = gen_category_from_overlay(G_overlay);

if F_calculated <= 0
    N_F = length(G_overlay.F_all);
    G_overlay.F2pair = zeros(N_F, n_pair);
    G_overlay.C_F = zeros(N_F, 1);
    G_overlay.F2link = zeros(N_F, N_links);
    % %     G_overlay.F2term2 = zeros(N_F, n_pair); %% for each F, value of temp_term2 for a given pair
    for i_F = 1:length(G_overlay.F_all)
        F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
        if length(F_eu) == 1
            C_F = G_overlay.G_u.C(F_eu(1));
        else
            C_F = min( G_overlay.G_u.C(F_eu) );
        end
        G_overlay.C_F(i_F) = C_F;
        F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
        if isempty(F)
            continue;
        end
        G_overlay.F2link(i_F, F) = 1;
        v_pairs = G_overlay.link2pair(F);
        G_overlay.F2pair(i_F, v_pairs) = 1;
        % % if length(v_pairs) ~= length(F) || length(v_pairs) ~= length(unique(v_pairs))
        % %     disp("debug")
        % % end
        % %         tmp_term2 = G_overlay.Kappa ./ (beta - G_overlay.delays(F)); %% Kappa_i / (beta - l_e)
        % %         Constraints = [Constraints, y(v_pairs)' * tmp_term2 <= C_F];
    end
    G_overlay.F_calc_eu = G_overlay.F_all;
    G_overlay.F_all = transform_Feu_to_Feo(G_overlay);
    G_overlay.F_calc = G_overlay.F_all;
else
    if G_overlay.G_type == 0 %% 0:IAB; 1:Roofnet
        netw_name = "IAB";
    elseif G_overlay.G_type == 1 %% 0:IAB; 1:Roofnet
        netw_name = "Roofnet";
    end
    filename_F = strcat("./data_res/F_dict2mat_", netw_name, ".mat");
    G_overlay.F_calc = read_F_calc_py2mat(G_overlay, filename_F);
    N_F = length(G_overlay.F_calc);
    G_overlay.F2pair = zeros(N_F, n_pair);
    G_overlay.C_F = zeros(N_F, 1);
    G_overlay.F2link = zeros(N_F, N_links);
    % %     G_overlay.F2term2 = zeros(N_F, n_pair); %% for each F, value of temp_term2 for a given pair
    for i_F = 1:length(G_overlay.F_calc)
        F = G_overlay.F_calc{i_F};
        F_eu = find(G_overlay.eo2eu(F, :) > 0);
        C_F_true = calc_effectiveCapacity(G_overlay, F);
        G_overlay.C_F(i_F) = unifrnd(G_overlay.C_F_lb, 1) * C_F_true;
        if isempty(F)
            error("empty F");
        end
        G_overlay.F2link(i_F, F) = 1;
        v_pairs = G_overlay.link2pair(F);
        G_overlay.F2pair(i_F, v_pairs) = 1;
    end
end
end





%% helper_function
function F_all_eo = transform_Feu_to_Feo(G_overlay)
% transfrom eu-based F_all to eo-based
N_F = length(G_overlay.F_calc_eu);
F_all_eo = cell(1, N_F);

for i_F = 1 : N_F
    F_eu = G_overlay.F_calc_eu{i_F};
    F_eo = find( sum(G_overlay.eo2eu(:, F_eu), 2) > 0 );
    F_all_eo{i_F} = F_eo;
end
end

function C_F = calc_effectiveCapacity(G_overlay, F)
% solve opt to obtain the effective capacity of F
yalmip('clear')

% Define variables
f = sdpvar(G_overlay.N_links,1);
%     Eo = unique(G_overlay.pair2link(Ea, :));

Constraints = [f >= 0];
for eu = 1 : size(G_overlay.G_u.D, 2)
    F_traverse_eu = intersect( find(G_overlay.eo2eu(:, eu) > 0), F );
    if isempty(F_traverse_eu)
        continue;
    end
    Constraints = [Constraints, sum( f(F_traverse_eu) ) <= G_overlay.G_u.C(eu)];
end

% Define an objective
Objective = -sum( f(F) );

% Set some options for YALMIP and solver
options = sdpsettings('verbose',0,'solver','mosek');

% Solve the problem
sol = optimize(Constraints,Objective,options);

% Analyze error flags
%     solution = struct;
if sol.problem == 0
    % Extract and display value
    %         solution.f = value(f);
    %         solution.tau = value(tau);
    C_F = sum( value(f(F)) );
    % %         if C_F - 1e-3 >= 1e-5
    % %             error("wrong C_F");
    % %         end
else
    error("Opt failed in calc_effectiveCapacity");
    %         disp('Hmm, something went wrong!');
    %         sol.info
    %         yalmiperror(sol.problem)
end
end

function F_calc = read_F_calc_py2mat(G_overlay, filename_F)
% read estimated Fs from py file
F_M = load(filename_F).F_M;
len_F = size(F_M, 1);
len_pairs = size(F_M, 2);

F_calc = cell(1, len_F);

for k = 1 : len_F
    tmp = squeeze(F_M(k, :, :));
    F = zeros(len_pairs, 1);
    for i_p = 1 : len_pairs
        src = tmp(i_p, 1) + 1;
        dst = tmp(i_p, 2) + 1;
        if ~(src==1 && dst == 1)
            F(i_p) = G_overlay.st2eo(strjoin(string([src, dst])));
        else
            F = F(1: i_p-1);
            F_calc{k} = F;
            break;
        end
    end
end
end

function G_u = read_RoofNet()
    % Read links

    N_v = 38;
    N_pair = 219;
    N_E = N_pair * 2;
    D = zeros(N_v, N_E);
    Adj = zeros(N_v, N_v);
    ST2idx = containers.Map(); %% HashTable mapping ST pair to underlay link ID: <(s, t), idx>
    % Open the file for reading
    fileID = fopen("Roofnet_AvgSuccessRate_1Mbps.txt", 'r');
    
    % Check if the file was successfully opened
    if fileID == -1
        error('File cannot be opened: %s', filename);
    end
    
    % Read and display each line of the file
    idx_edge = 1;
    while ~feof(fileID)
        line = fgets(fileID); % Reads a line, including newline characters
        newStr = split(line, ' ');
        if line(1) == 'n'
            continue;
        else
            newStr = split(line, ' ');
            src = str2double(newStr(1));
            dst = str2double(newStr(2));
            D(src, idx_edge) = 1;
            D(dst, idx_edge) = -1;
            Adj(src, dst) = 1;
            ST2idx( strjoin(string([src, dst])) ) = idx_edge;

            % reverse link
            idx_edge_reverse = idx_edge + N_pair;
            D(dst, idx_edge_reverse) = 1;
            D(src, idx_edge_reverse) = -1;
            ST2idx( strjoin(string([dst, src])) ) = idx_edge_reverse;
            idx_edge = idx_edge + 1;
        end
    end
    Adj = Adj + Adj';
    
    % Close the file
    fclose(fileID);
    
    G_u = struct;
    G_u.Adj = Adj;
    G_u.D = D;
    % G_u.DG = DG;
    % G_u.delays = 0.5 + (1.5-0.5)*rand(size(D, 2),1); %% r = a + (b-a).*rand(N,1).
    G_u.delays = 5e-4*ones(size(D, 2), 1); %% 5e-4 ms
    G_u.C = 0.001*ones(size(D, 2), 1); %% 1Mbps
    % G_u.C = 0.1 + (0.3-0.1)*rand(size(D, 2),1); %% 0.1Gbps ~ 0.3Gbps
    G_u.ST2idx = ST2idx;
end


function [Vo, spr_table] = read_ov_routeTable(G_type)
    % read overlay nodes and routing table
    if G_type == 0 %% IAB network
%         Vo_filename = "IAB_topo4mat.mat";
        Vo_filename = "IAB_topo4mat.mat";
        route_filename = "IAB_underlay_routing_map.txt";  %% 0-indexed
    elseif G_type == 1 %% Roofnet
%         Vo_filename = "roofnet_topo4mat.mat";
        Vo_filename = "roofnet_overlay_nodes.mat";
        route_filename = "roofnet_underlay_routing_map.txt"; %% 0-indexed
    end
    Vo = sort(load(Vo_filename).overlay_V); %% already 1-indexed
    fileID = fopen(route_filename, 'r'); 
    spr_table = containers.Map();
    
    % Check if the file was successfully opened
    if fileID == -1
        error('File cannot be opened: %s', filename);
    end
    
    % Read and display each line of the file
    while ~feof(fileID)
        line = fgets(fileID); % Reads a line, including newline characters
        newStr = split(line, {' ', ': '});
        N_hop = str2double(newStr(3));
        src = str2double(newStr(1)) + 1;
        dst = str2double(newStr(2)) + 1;
        route = -1 * ones(1, N_hop);
        for k = 1 : N_hop
            route(k) = str2double(newStr(3+k)) + 1; %% pay attention to 1-index
        end
        spr_table( strjoin(string([src, dst])) ) = route;
    end
    
    % Close the file
    fclose(fileID);
end


function G_u = gen_underlay_netw(N_nodes, G_type)
%% G_type = 0:Hex; 1:roofnet
    if G_type == 0
        [ Adj, D, DG, node_xval, node_yval] = G_gen(G_type);
    end
    
    G_u = struct;
    G_u.Adj = Adj;
    G_u.D = D;
    G_u.DG = DG;
    % G_u.delays = 0.5 + (1.5-0.5)*rand(size(D, 2),1); %% r = a + (b-a).*rand(N,1).
    G_u.delays = 5e-4*ones(size(D, 2), 1); %% 5e-4 ms
    if G_type == 0 %% IAB
        G_u.C = 0.4*ones(size(D, 2), 1); %% 0.4Gbps
    elseif G_type == 1 %% roofnet
        G_u.C = 0.001*ones(size(D, 2), 1); %% 1Mbps
    end
    % G_u.C = 0.1 + (0.3-0.1)*rand(size(D, 2),1); %% 0.1Gbps ~ 0.3Gbps
    G_u.ST2idx = containers.Map(); %% HashTable mapping ST pair to underlay link ID: <(s, t), idx>
    for eu = 1 : size(D, 2)
        u = find(D(:, eu) == 1);
        v = find(D(:, eu) == -1);
        G_u.ST2idx( strjoin(string([u, v])) ) = eu;
    end

end