function [F_all, C_F_all, F2link, F2pair] = update_F_to_true(G_overlay)
% obtain the ground-true categories and the associated C_F

F_all = gen_category_from_overlay(G_overlay);
n_pair = size(G_overlay.Bp, 2);
N_links = G_overlay.N_nodes * (G_overlay.N_nodes-1);

N_F = length(F_all);
F2pair = zeros(N_F, n_pair);
F2link = zeros(N_F, N_links);
C_F_all = zeros(N_F, 1);
% %     G_overlay.F2term2 = zeros(N_F, n_pair); %% for each F, value of temp_term2 for a given pair
for i_F = 1:length(F_all)
    F_eu = F_all{i_F}; %% underlay links in this F
    if length(F_eu) == 1
        C_F = G_overlay.G_u.C(F_eu(1));
    else
        C_F = min( G_overlay.G_u.C(F_eu) );
    end
    C_F_all(i_F) = C_F;
    F = find(G_overlay.eo2eu(:, F_eu(1)) > 0); %% tunnels in F
    if isempty(F)
        continue;
    end
    F2link(i_F, F) = 1;
    v_pairs = G_overlay.link2pair(F);
    F2pair(i_F, v_pairs) = 1;
end



% % G_overlay.F_calc_eu = G_overlay.F_all;
% % G_overlay.F_all = transform_Feu_to_Feo(G_overlay);
% % G_overlay.F_calc = G_overlay.F_all;