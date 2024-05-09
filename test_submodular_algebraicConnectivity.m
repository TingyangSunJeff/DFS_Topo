%% Test sub-modularity of rho_bar, i.e., algebraic connectivity lambda_2

%% Generate Graph
max_N_nodes = 20;
N_try = 100;

for N_nodes = 20 : max_N_nodes
    N_links = (N_nodes - 1)*N_nodes / 2; %% undirected graph
    [link2st, st2link, B] = gen_link2st(N_nodes, N_links); %% mapping between link_idx and (s, t) pair
%     G = graph(link2st(:,1)', link2st(:, 2)');
    res_submodular = zeros(1, N_try);
    res_supermodular = zeros(1, N_try);
    L_concave_ratio = 0.1: 0.1: 0.9;
    res_concave = zeros(length(L_concave_ratio), N_try);
    
    starMST = 1:N_nodes-1; %% a MST as a star originated in the node 1
    links_left = N_nodes:N_links;
    alpha = ones(N_links, 1);
    
    %% Def: for any S, T, f(s)+f(T) >= f(s \cup T) + f(s \cap T)
    for i_try = 1:N_try
        % Ea_S = 1:randi([1, N_links],1);
        Ea_S = gen_random_Ea(starMST, links_left);
        [lambda_S, L_S] = gen_algebraic_connectivity(Ea_S, B, link2st);

        % Ea_T = 1:randi([1, N_links],1);
        Ea_T = gen_random_Ea(starMST, links_left);
        [lambda_T, L_T] = gen_algebraic_connectivity(Ea_T, B, link2st);

        [lambda_ScupT, ~] = gen_algebraic_connectivity(union(Ea_S, Ea_T), B, link2st);
        [lambda_ScapT, ~] = gen_algebraic_connectivity(intersect(Ea_S, Ea_T), B, link2st);

        if lambda_S + lambda_T >= lambda_ScupT + lambda_ScapT - 1e-5
            res_submodular(i_try) = 1;
        else
            res_supermodular(i_try) = 1;
        end

        %% check concavity
        for i_a = 1 : length(L_concave_ratio)
            a = L_concave_ratio(i_a);
            L_a = a * L_S + (1-a)* L_T;
            lambda_a_all = eigs(L_a,2,'smallestabs');
            lambda_a = lambda_a_all(2);
            if lambda_a >= a*lambda_S + (1-a)*lambda_T - 1e-5
                res_concave(i_a, i_try) = 1;
            else
                res_concave(i_a, i_try) = 2;
            end
        end
    end


    %% Def for S \subset T, x \in E\setminus T, f(S+{x}) - f(S) \ge f(T+{x}) - f(T)
    % % for i_try = 1:N_try
    % %     x_set = [];
    % %     while isempty(x_set)
    % %         Ea_S = gen_random_Ea(starMST, links_left);
    % %         [lambda_S, ~] = gen_algebraic_connectivity(Ea_S, B, link2st);
    % % 
    % %         Ea_T = unique( [Ea_S, gen_random_Ea(starMST, links_left)] );
    % %         [lambda_T, ~] = gen_algebraic_connectivity(Ea_T, B, link2st);
    % % 
    % %         x_set = setdiff(1:N_links, Ea_T);
    % %     end
    % %     x = x_set(randi([1, length(x_set)],1));
    % %     [lambda_Sx, ~] = gen_algebraic_connectivity([Ea_S, x], B, link2st);
    % %     [lambda_Tx, ~] = gen_algebraic_connectivity([Ea_T, x], B, link2st);
    % % 
    % %     if lambda_Sx - lambda_S >= lambda_Tx - lambda_T - 1e-5
    % %         res_submodular(i_try) = 1;
    % %     else
    % %         res_supermodular(i_try) = 1;
    % %     end
    % % end
end



%% functions
function [link2st, st2link, B] = gen_link2st(N_nodes, N_links)

link2st = zeros(N_links, 2); %% <link_idx, (S, T)>
st2link = containers.Map();
B = zeros(N_nodes, N_links);
i_link = 1;
for u = 1 : N_nodes-1
    for v = u+1 : N_nodes
        link2st(i_link, 1) = u;
        link2st(i_link, 2) = v;
        str2link( char(join(string([u, v]))) ) = i_link;
        B(u, i_link) = 1;
        B(v, i_link) = -1;
        i_link = i_link + 1;
    end
end

end

function [lambda_2, L] = gen_algebraic_connectivity(Ea, B, link2st)
%% generate the algebraic_connectivity Lambda_2 for Ea
N_links = size(B, 2);
alpha = zeros(N_links, 1);
alpha(Ea) = 1;
L = B * diag(alpha) * B'; %% obtain degree
degrees = diag(L);
for e = Ea
    u = link2st(e, 1);
    v = link2st(e, 2);
    alpha(e) = 1 / max([degrees(u), degrees(v)]);
end
L = B * diag(alpha) * B';

lambda_all = eigs(L,2,'smallestabs');
lambda_2 = lambda_all(2);

end

function Ea = gen_random_Ea(starMST, links_left)
%% randomly generate an Ea based on starMST

n_Ea = randi([1, length(links_left)],1);
shuffle_index = randperm( length(links_left) );
new_links = links_left(shuffle_index(1:n_Ea));
Ea = sort([starMST, new_links]);

end
