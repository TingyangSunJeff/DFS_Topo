%% Test super-modularity of rho

%% add YALMIP and Mosek
currentFolder = pwd;
utils_path_yalmip_mosek()

N_nodes = 40; %% maximum number of nodes in test

for n_V = 35 : N_nodes
    N_links = n_V * (n_V - 1)/2;

    G_overlay = gen_overlay_netw(n_V, N_links); %% Generate B matrix for the graph: G_overlay.B

    % % test_res = test_superModular(G_overlay);
    test_res = test_modular_main(G_overlay);
end








%% Functions
function G_overlay = gen_overlay_netw(N_nodes, N_links)
G_overlay = struct;
G_overlay.N_nodes = N_nodes;
G_overlay.N_links = N_links;
G_overlay.B = zeros(N_nodes, N_links);
G_overlay.st2link = containers.Map(); %% HashTable mapping ST pair to overlay link ID: <(s, t), idx>
G_overlay.link2st = zeros(N_links, 2);
i_link = 1;
for u = 1 : N_nodes
    for v = u+1 : N_nodes
        G_overlay.B(u, i_link) = 1;
        G_overlay.B(v, i_link) = -1;
        G_overlay.st2link( strjoin(string([u, v])) ) = i_link;
        G_overlay.link2st(i_link, 1) = u;
        G_overlay.link2st(i_link, 2) = v;
        i_link = i_link + 1;
    end
end
end

function test_res = test_modular_main(G_overlay)
%% Main function to test the submodular/supermodular properties of a given objective
% Test by "For any A \subset B \subset S and x \in (B-A), f(A+{x}) - f(A) <= f(B+{x}) - f(B)"
% 1. Generate E_a_0 (set A) and E_a_1 (set B) and x.
% 2. Call functions to generate function values.

N_links = G_overlay.N_links;
N_nodes = G_overlay.N_nodes;
%% number of randomly chosen A and B for each |A|
max_A_size = min([(N_links-1), 3]);
n_A = 30;
n_B = 100;
test_res = zeros(1, max_A_size*n_A*n_B);
i_test_case = 1;
for i_A = 1 : n_A
    disp("N_nodes = " + string(N_nodes) + "; N_links=" + string(N_links) + "; i_A=" + string(i_A));
    A = gen_linkSet_connect(G_overlay); %% randomly generate E_a_0 (set A)
    %% Generate f(A)
    %% solve rho from SDP
    % % res_A = solve_rho(G_overlay, A);
    % % rho_A = res_A.rho;
    %% Boyd's local degree
    [rho_A, alpha_A] = rho_BoydLocalDegree(G_overlay, A);
    % % K_A = gen_K_from_p(1-rho_A^2);
    K_A = gen_K_from_alpha(alpha_A);
    for i_B = 1 : n_B %% randomly generate some B to test
        %% construct B
        [B, x] = gen_B_from_S_with_A(1:N_links, A); %% x is in B but not in A.
        
        %% Generate f(A+{x}), f(B+{x}), f(B)
        %% solve rho from SDP
        % % res_B = solve_rho(G_overlay, B);
        % % res_Ax = solve_rho(G_overlay, [A, x]);
        % % res_Bx = solve_rho(G_overlay, [B, x]);
        % % rho_B = res_B.rho;  rho_Ax = res_Ax.rho;   rho_Bx = res_Bx.rho; 
        %% Boyd's local degree
        [rho_Ax, alpha_Ax] = rho_BoydLocalDegree(G_overlay, [A, x]); 
        [rho_Bx, alpha_Bx] = rho_BoydLocalDegree(G_overlay, [B, x]); 
        [rho_B, alpha_B] = rho_BoydLocalDegree(G_overlay, B);
        K_B = gen_K_from_p(1-rho_B^2); K_Bx = gen_K_from_p(1-rho_Bx^2); K_Ax = gen_K_from_p(1-rho_Ax^2);
        K_B = gen_K_from_p(1-rho_B^2); K_Bx = gen_K_from_p(1-rho_Bx^2); K_Ax = gen_K_from_p(1-rho_Ax^2);

        %% Definition check
        % % if (rho_Ax - rho_A) - (rho_Bx - rho_B) <= 1e-6
        % %     test_res(i_test_case) = 1;
        % % else
        % %     test_res(i_test_case) = 0;
        % % end
        test_res(i_test_case) = (K_Ax - K_A) - (K_Bx - K_B);

        i_test_case = i_test_case + 1;
    end

end
end

function [rho, alpha] = rho_BoydLocalDegree(G_overlay, E_a)
N_links = G_overlay.N_links;
N_nodes = G_overlay.N_nodes;
%% Test the submodular/supermodular property of Boyd's local degree
degrees = sum( abs(G_overlay.B(:, E_a)), 2 );
alpha = zeros(N_links, 1);

for e_a = E_a
    s = G_overlay.link2st(e_a, 1);
    t = G_overlay.link2st(e_a, 2);
    alpha(e_a) = 1 / max([degrees(s), degrees(t)]);
end

L_Ea = G_overlay.B * diag(alpha) * G_overlay.B';
J = (1 / N_nodes) * ones(N_nodes, N_nodes);
rho = norm( eye(N_nodes) - L_Ea - J ); %% spectral norm
end

function test_res = test_superModular(G_overlay)

%% Test by "For any A \subset B \subset S and x \in (B-A), f(A+{x}) - f(A) <= f(B+{x}) - f(B)"
N_links = G_overlay.N_links;
N_nodes = G_overlay.N_nodes;
%% number of randomly chosen A and B for each |A|
max_A_size = min([(N_links-1), 3]);
n_A = 30;
n_B = 100;
test_res = zeros(1, max_A_size*n_A*n_B);
i_test_case = 1;
for i_size_A = 1 : max_A_size %% |A|
    disp("N_nodes = " + string(N_nodes) + "; N_links=" + string(N_links) + "; size_A=" + string(i_size_A));
    % visited_A = containers.Map(); %% each generated A is a vector in visited_A. Used to avoid repeated A.
    % n_A_use = min([n_A, nchoosek(N_links, i_size_A)]); %% it is possible that nchoosek(N_links, i_size_A) < n_A, which will cause unbreakable while loop.
    for i_A = 1 : n_A %% randomly choosen A with |A|
        % % while 1 %% avoid repeated A.
        % %     link_idx_random = randperm(N_links); %% selected links
        % %     % size_A = i_size_A;  %% !!!directly use i_size_A as size_A
        % %     size_A = randi([1, N_links-2],1,1); %% !!! randomly gen a |A| from [1, N_links-1]
        % %     % % new_key = strjoin( string(sort( link_idx_random(1:size_A) )) ); %% if we want to avoid repeated key
        % %     % % if ~isKey(visited_A,new_key) 
        % %     % %     visited_A(new_key) = 1;
        % %     % %     break;
        % %     % % end
        % %     break
        % % end
        % % disp("random size_A=" + string(size_A));
        % % A = link_idx_random(1:size_A);
        A = gen_linkSet_connect(G_overlay);
        %% Generate f(A)
        res_A = solve_rho(G_overlay, A);
        rho_A = res_A.rho;
        for i_B = 1 : n_B %% randomly generate some B to test
            %% construct B
            [B, x] = gen_B_from_S_with_A(1:N_links, A); %% x is in B but not in A.
            
            %% Generate f(A+{x}), f(B+{x}), f(B)
            res_B = solve_rho(G_overlay, B);
            res_Ax = solve_rho(G_overlay, [A, x]);
            res_Bx = solve_rho(G_overlay, [B, x]);
            rho_B = res_B.rho;  rho_Ax = res_Ax.rho;   rho_Bx = res_Bx.rho; 

            %% Definition check
            if (rho_Ax - rho_A) - (rho_Bx - rho_B) <= 1e-6
                test_res(i_test_case) = 1;
            else
                test_res(i_test_case) = 0;
            end
            i_test_case = i_test_case + 1;
        end
    end
end


end

function solution = solve_rho(G_overlay, E_a)
%%% E_a is list of overlay links.
N_links = G_overlay.N_links;
N_nodes = G_overlay.N_nodes;
B = G_overlay.B;
E_a_c = setdiff(1:N_links, E_a); %% complement of E_a over E

% Every time you call sdpvar etc, an internal database grows larger
yalmip('clear')

% Define variables
rho = sdpvar(1,1);
alpha = sdpvar(N_links,1);
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

Constraints = [Constraints, eye(N_nodes) - B*diag(alpha)*B' - J - rho*eye(N_nodes) <= 0, ...
    eye(N_nodes) - B*diag(alpha)*B' - J + rho*eye(N_nodes) >= 0];

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

function A = gen_linkSet_connect(G_overlay)
%% generate a subset of overlay link A to form a connected subgraph
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
%% Given superset S and a subset A, randomly choose B with a random cardinality
S_minus_A = setdiff(S, A);
shuffle_index = randperm(length(S_minus_A)); %% shuffle index
B_cardinality = randi([1, length(S_minus_A)-2],1,1); %% randomly gen a |B|
B = S_minus_A(shuffle_index(1:B_cardinality));
B = sort( [A, B] );
S_minus_B = setdiff(S, B);
shuffle_index = randperm(length(S_minus_B)); %% shuffle index
x = S_minus_B( shuffle_index(1) );
end

function K = gen_K_from_p(p)
%% Given an p, compute the K(p,t)
zeta = 100;
sigma = 1;
M_1 = 50;
M_2 = 100;

K = (zeta*sqrt(M_1+1) + sigma*sqrt(p)) ./ p + sqrt((M_1+1) * (M_2+1)) ./ p;
end

function K = gen_K_from_alpha(G_overlay, alpha)
%% Given an alpha, either from SDP or a random generation, compute the K(p,t)
zeta = 100;
sigma = 1;
M_1 = 50;
M_2 = 100;
W = G_overlay.B*diag(alpha)*G_overlay.B';
N_nodes = G_overlay.N_nodes;
J = (1 / N_nodes) * ones(N_nodes, N_nodes);
p = 1 - norm(W - J); %% spectral norm

K = (zeta*sqrt(M_1+1) + sigma*sqrt(p)) ./ p + sqrt((M_1+1) * (M_2+1)) ./ p;
end

function utils_path_yalmip_mosek()

%% add YALMIP and Mosek
currentFolder = pwd;
if currentFolder(1) == 'C'
    path_yalmip = "C:\Users\yxh5389\Downloads\YALMIP-master";
    path_mosek = "C:\Users\yxh5389\Downloads\mosek\10.1\toolbox\R2017aom";
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

end