load('./data_res/Alg_Roofnet_CIFAR10_0412.mat');

G_overlay.Kappa = G_overlay.Kappa / 1e3;
G_overlay.G_u.delays = G_overlay.G_u.delays / 1e3;
G_overlay.delays = G_overlay.delays / 1e3;
% G_overlay.C_F = G_overlay.C_F*1e3;
% G_overlay.G_u.C = G_overlay.G_u.C * 1e3;

N_Vo = length(G_overlay.Vo);
demands = G_overlay.Kappa * ones(N_Vo, 1);
Ea = find(Ea_SCA_both_rho_ew(:, 1) > 0);
min_tau_w_overlayRouting(G_overlay, Ea, demands)