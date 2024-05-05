function [tau_wRouting, tau_wo, tau_equalRate] = min_tau_w_overlayRouting(G_overlay, Ea, demands)
%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Ea: list of indices
% demands: list of kappa

M = 2 * max(G_overlay.C_F);
N_nodes = G_overlay.N_nodes;
N_pairs = size(G_overlay.Bp, 2);
N_Eo = size(G_overlay.B, 2);
Vo = G_overlay.Vo;
N_Vo = length(Vo);
N_Vu = size(G_overlay.G_u.Adj, 1);
Adj_Ea = zeros(N_Vu, N_Vu);
for i_pair = 1: length(Ea)
    pair = Ea(i_pair);
    src = G_overlay.idx2st(pair, 1);
    dst = G_overlay.idx2st(pair, 2);
    Adj_Ea(src, dst) = 1;
end
Adj_Ea = Adj_Ea + Adj_Ea';
tol = 1e-5;

% init solution: w/o overlay routing
tau_sol = solve_tau_noRouting(G_overlay, Ea);
f_wo = zeros(N_Vo, N_Eo);
tau_wo = tau_sol.tau;
z_wo = zeros(N_Vo, N_Eo); %% h -- eo. Each h is uniquely defined by its src.
r_wo = zeros(N_Vo, N_Vo, N_Eo); %% for each <src, dst> in h defined by src, whether it passes eo: <h, k, (i,j)>
d_wo = inf * ones(N_Vo,1);
for i_e = 1 : length(Ea)
    pair = Ea(i_e);
    u = G_overlay.idx2st(pair, 1); v = G_overlay.idx2st(pair, 2);
    i_u = G_overlay.Vu2Vo(u); i_v = G_overlay.Vu2Vo(v);
    eo_uv = G_overlay.st2eo(strjoin(string([u, v])));
    eo_vu = G_overlay.st2eo(strjoin(string([v, u])));
    f_wo(i_u, eo_uv) = tau_sol.f(eo_uv);
    f_wo(i_v, eo_vu) = tau_sol.f(eo_vu);
    z_wo(i_u, eo_uv) = 1;
    z_wo(i_v, eo_vu) = 1;
    r_wo(i_u, i_v, eo_uv) = 1;
    r_wo(i_v, i_u, eo_vu) = 1;
    d_wo(i_u) = min([d_wo(i_u), f_wo(i_u, eo_uv)]);
    d_wo(i_v) = min([d_wo(i_v), f_wo(i_v, eo_vu)]);
end
d_inv_wo = 1./ d_wo;    

% % test_feasibility()




% E_a_c = setdiff(1:N_pairs, E_a); %% complement of E_a over E

% Every time you call sdpvar etc, an internal database grows larger
yalmip('clear')

% Define variables
z = binvar(N_Vo, N_Eo); %% h -- eo. Each h is uniquely defined by its src.
r = binvar(N_Vo, N_Vo, N_Eo, 'full'); %% for each <src, dst> in h defined by src, whether it passes eo: <h, k, (i,j)>
d = sdpvar(N_Vo,1);
d_inv = sdpvar(N_Vo,1);
f = sdpvar(N_Vo, N_Eo);
tau = sdpvar(1,1);

Constrs = [];

constr_5a_raw = [];
% filter r: if <src, dst> not in Ea, r(src, dst, :) = 0;
for i_u = 1 : N_Vo
    u = Vo(i_u);
    for i_v = 1 : N_Vo
        v = Vo(i_v);
        if Adj_Ea(u, v) == 0
            constr_5a_raw = [constr_5a_raw, r(i_u, i_v, :) == 0];
        end
    end
end
constr_5a_name = tag(constr_5a_raw, 'constr5a');
Constrs = [Constrs, constr_5a_name];

% (5b)
constr_5b_raw = [];
for i_u = 1 : N_Vo %% for each h
    u = Vo(i_u);
    for i_v = 1 : N_Vo %% for each <u,v> in h, i.e., v\in T_h
        v = Vo(i_v);
        if Adj_Ea(u, v) == 0
            continue;
        end
        constr_5b_raw = [constr_5b_raw, tau >= demands(i_u) * d_inv(i_u) + G_overlay.delays' * squeeze(r(i_u, i_v, :)) - tol];
    end
end
constr_5b_name = tag(constr_5b_raw, 'constr5b');
Constrs = [Constrs, constr_5b_name];

% d and d_inv
for i_h = 1 : N_Vo
    Constrs = [Constrs, d(i_h) * d_inv(i_h) == 1];
end

% (5c)
constr_5c_raw = [];
for i_F = 1 : length(G_overlay.F_all)
    % % F = find( G_overlay.F2link(i_F, :) > 0 );
    constr_5c_raw = [constr_5c_raw, sum(f(:, G_overlay.F2link(i_F, :) > 0),"all") <= G_overlay.C_F(i_F) + tol];
end
constr_5c_name = tag(constr_5c_raw, 'constr5c');
Constrs = [Constrs, constr_5c_name];

% (5d)
constr_5d_raw = [];
for i_h = 1 : N_Vo
    src = Vo(i_h);
    for i_k = 1 : N_Vo %% traverse k in T_h
        dst = Vo(i_k);
        if Adj_Ea(src, dst) == 0 || dst == src
            continue;
        end
        for i_u = 1 : N_Vo %% traverse u \in V
            u = Vo(i_u);
            if u == src
                b = 1;
            elseif u == dst
                b = -1;
            else
                b = 0;
            end
            % % eo_lfs = find(G_overlay.B(u, :) == 1);
            % % eo_rhs = find(G_overlay.B(u, :) == -1);
            constr_5d_raw = [constr_5d_raw, sum(r(i_h, i_k, G_overlay.B(u, :) == 1), "all") == sum(r(i_h, i_k, G_overlay.B(u, :) == -1), "all") + b];
        end
    end
end
% % constr_5d_raw = [constr_5d_raw, sum(r(1, 2, G_overlay.B(1, :) == 1), "all") == sum(r(1, 2, G_overlay.B(1, :) == -1), "all") + 1];
constr_5d_name = tag(constr_5d_raw, 'constr5d');
Constrs = [Constrs, constr_5d_name];

% (5e)
constr_5e_raw = [];
for i_h = 1 : N_Vo
    for i_k = 1 : N_Vo
        constr_5e_raw = [constr_5e_raw, squeeze(r(i_h, i_k, :))  <= z(i_h, :)'];
    end
end
constr_5e_name = tag(constr_5e_raw, 'constr5e');
Constrs = [Constrs, constr_5e_name];

% (5f)
constr_5f_raw = [];
for i_h = 1 : N_Vo
    for i_k = 1 : N_Vo
        constr_5f_raw = [constr_5f_raw, d(i_h) - M * (1-z(i_h, :)) - tol <= f(i_h, :) <= d(i_h) + tol ];
    end
end
constr_5f_name = tag(constr_5f_raw, 'constr5f');
Constrs = [Constrs, constr_5f_name];

% (5g)
for i_h = 1 : N_Vo
    for i_k = 1 : N_Vo
        Constrs = [Constrs, f(i_h, :) <= M * z(i_h, :) + tol ];
    end
end

% (5f)
Constrs = [Constrs, d >= 0, f >= 0 ];

% initial solution.
assign(z, z_wo);
% % for i_src = 1 : 1
% %     for i_kh = 1 : N_Vo
% %         assign(squeeze(r(i_src, i_kh, :)), squeeze(r_wo(i_src, i_kh, :)));
% %         disp([i_src, i_kh, i_src, i_kh])
% %         % % for i_e = 1 : N_Eo
% %         % %     assign(r(i_h, i_k, i_e), r_wo(i_h, i_k, i_e));
% %         % % end
% %     end
% % end
for i_kh = 1 : N_Vo
    assign(squeeze(r(1, i_kh, :)), squeeze(r_wo(1, i_kh, :)));
end
for i_kh = 1 : N_Vo
    assign(squeeze(r(2, i_kh, :)), squeeze(r_wo(2, i_kh, :)));
end
for i_kh = 1 : N_Vo
    assign(squeeze(r(3, i_kh, :)), squeeze(r_wo(3, i_kh, :)));
end
for i_kh = 1 : N_Vo
    assign(squeeze(r(4, i_kh, :)), squeeze(r_wo(4, i_kh, :)));
end
for i_kh = 1 : N_Vo
    assign(squeeze(r(5, i_kh, :)), squeeze(r_wo(5, i_kh, :)));
end
for i_kh = 1 : N_Vo
    assign(squeeze(r(6, i_kh, :)), squeeze(r_wo(6, i_kh, :)));
end
for i_kh = 1 : N_Vo
    assign(squeeze(r(7, i_kh, :)), squeeze(r_wo(7, i_kh, :)));
end
for i_kh = 1 : N_Vo
    assign(squeeze(r(8, i_kh, :)), squeeze(r_wo(8, i_kh, :)));
end
for i_kh = 1 : N_Vo
    assign(squeeze(r(9, i_kh, :)), squeeze(r_wo(9, i_kh, :)));
end
for i_kh = 1 : N_Vo
    assign(squeeze(r(10, i_kh, :)), squeeze(r_wo(10, i_kh, :)));
end
% warmstart(r, r_wo);
% % assign(r(1,2,:), r_wo(1,2,:));
% % assign(r(1,3,:), r_wo(1,3,:));
% % assign(r(1,4,:), r_wo(1,4,:));
% assign(r(1,2,1), 1);
% assign(r(1,2,2:end), 0);
% assign(r(2,1,46), 1);
% assign(r(2,1,[1:45, 47:end]), 0);
assign(d, d_wo);
assign(d_inv, d_inv_wo);
assign(f, f_wo);
assign(tau, tau_wo);

% Define an objective
Objective = tau;

% Set some options for YALMIP and solver
% options = sdpsettings('verbose',0,'solver','gurobi');
options = sdpsettings('verbose', 2, 'debug', 1, 'solver','gurobi', 'usex0',1, 'gurobi.timelimit',400, 'gurobi.FeasibilityTol', tol);

% [Model,recover_model,diagnostic,internalmodel] = export(Constrs,Objective,options);

% Solve the problem
sol = optimize(Constrs,Objective,options);
% gurobi_model = exportmodel(Constrs, Objective, sdpsettings('solver', 'gurobi'), 'gurobi.mps');
% x_sol = value(x);
tau_wRouting = value(tau);
tau_equalRate = test_lemma3_2();




    function tau = test_lemma3_2()
        %%%%%%%%%%% test Lemma 3.2 for optimal rate allocation given z %%%%%%%%%%%
        t_Fs = zeros(length(G_overlay.F_calc), 1);
        z = value(z);
        for i_vo = 1 : N_Vo %%%%% loop over h
            used_eo = find(z(i_vo, :) > 0); %%%% z_{ij}^h=1: z = binvar(N_Vo, N_Eo); %% h -- eo. Each h is uniquely defined by its src.
            for i_F = 1 : length(G_overlay.F_calc)
                F = G_overlay.F_calc{i_F};
                t_Fs(i_F) = t_Fs(i_F) + length( intersect(F, used_eo) );
            end
        end
        tau = G_overlay.Kappa / min(G_overlay.C_F ./ t_Fs);
    end

    function test_feasibility()
        % filter r: if <src, dst> not in Ea, r(src, dst, :) = 0;
        for i_u = 1 : N_Vo
            u = Vo(i_u);
            for i_v = 1 : N_Vo
                v = Vo(i_v);
                if Adj_Ea(u, v) == 0
                    Constrs = r_wo(i_u, i_v, :) - 0;
                    if any(Constrs ~= 0)
                        disp("Debug");
                    end
                end
            end
        end
        
        % (5b)
        for i_u = 1 : N_Vo %% for each h
            u = Vo(i_u);
            for i_v = 1 : N_Vo %% for each <u,v> in h, i.e., v\in T_h
                v = Vo(i_v);
                if Adj_Ea(u, v) == 0
                    continue;
                end
                Constrs = tau_wo - ( demands(i_u) * d_inv_wo(i_u) + G_overlay.delays' * squeeze(r_wo(i_u, i_v, :))-tol );
                if any(Constrs < 0)
                    disp("Debug");
                end
            end
        end
        
        % d and d_inv
        for i_h = 1 : N_Vo
            Constrs = d_wo(i_h) * d_inv_wo(i_h) - 1;
            if any(Constrs ~= 0)
                disp("Debug");
            end
        end
        
        % (5c)
        for i_F = 1 : length(G_overlay.F_all)
            % % F = find( G_overlay.F2link(i_F, :) > 0 );
            Constrs = sum(f_wo(:, G_overlay.F2link(i_F, :) > 0),"all") - G_overlay.C_F(i_F);
            if any(Constrs > tol)
                disp("Debug");
            end
        end
        
        % (5d)
        % constr_5d_raw = [];
        for i_h = 1 : N_Vo
            src = Vo(i_h);
            for i_k = 1 : N_Vo %% traverse k in T_h
                dst = Vo(i_k);
                if Adj_Ea(src, dst) == 0 || dst == src
                    continue;
                end
                for i_u = 1 : N_Vo %% traverse u \in V
                    u = Vo(i_u);
                    if u == src
                        b = 1;
                    elseif u == dst
                        b = -1;
                    else
                        b = 0;
                    end
                    % % eo_lfs = find(G_overlay.B(u, :) == 1);
                    % % eo_rhs = find(G_overlay.B(u, :) == -1);
                    Constrs = sum(r_wo(i_h, i_k, G_overlay.B(u, :) == 1), "all") - ( sum(r_wo(i_h, i_k, G_overlay.B(u, :) == -1), "all") + b );
                    if any(Constrs ~= 0)
                        disp("Debug");
                    end
                end
            end
        end

        
        % (5e)
        for i_h = 1 : N_Vo
            for i_k = 1 : N_Vo
                Constrs = squeeze(r_wo(i_h, i_k, :)) - z_wo(i_h, :)';
                if any(Constrs > 0)
                    disp("Debug");
                end
            end
        end
        
        % (5f)
        for i_h = 1 : N_Vo
            for i_k = 1 : N_Vo
                Constrs = d_wo(i_h) - M * (1-z_wo(i_h, :)) -tol - f_wo(i_h, :);
                if any(Constrs > 0)
                    disp("Debug");
                end
                Constrs = f_wo(i_h, :) - (d_wo(i_h) + tol);
                if any(Constrs > 0)
                    disp("Debug");
                end
            end
        end
        
        % (5g)
        for i_h = 1 : N_Vo
            for i_k = 1 : N_Vo
                Constrs = f_wo(i_h, :) - M * z_wo(i_h, :);
                if any(Constrs > 0)
                    disp("Debug");
                end
            end
        end
        
        % (5f)
        % % Constrs = [Constrs, d >= 0, f >= 0 ];
    end

end 