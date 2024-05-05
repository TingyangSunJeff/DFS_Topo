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
    
    % s_vec = zeros(1, G_overlay.N_links); %% vectors used for summation
    for i_F = 1 : length(G_overlay.F_calc)
        % deprecated
        % % F_eu = G_overlay.F_all{i_F}; %% underlay links in this F
        % % % F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
        % % % s_vec = s_vec .* 0;
        % % % s_vec( 1, G_overlay.eo2eu(:, F_eu(1)) > 0 ) = 1; %% equivalent to F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); s_vec(F) = 1
        % % if length(F_eu) == 1
        % %     C_F = G_overlay.G_u.C(F_eu(1));
        % % else
        % %     C_F = min( G_overlay.G_u.C(F_eu) );
        % % end
        
        F = G_overlay.F_calc{i_F};
        C_F = G_overlay.C_F(i_F);

        Constraints = [Constraints, sum(f( F )) <= C_F];
    end
    
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