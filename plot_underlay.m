function plot_underlay(G_overlay)
%% differentiating overlay nodes, underlay nodes, links on underlay routing paths

N_Vu = size(G_overlay.G_u.Adj, 2);
N_Eu_u = size(G_overlay.G_u.D, 2) / 2; %% N_Eu of undirected links
Adj_used = zeros(N_Vu, N_Vu);
eu2st = zeros(N_Eu_u, 2); %% eu to s-t pair
i_eu_u = 1;
% % indicator_Vo = zeros(N_Vu, 1);
% % indicator_Vo(G_overlay.Vo) = 1;
used_eu = zeros(N_Eu_u, 2); %% used underlay links  s-t pairs
i_used_eu = 1;
for eu = 1 : size(G_overlay.G_u.D, 2)
    st_pair = find(G_overlay.G_u.D(:, eu) ~= 0);
    src = st_pair(1);
    dst = st_pair(2);
    if Adj_used(src, dst) == 1
        continue;
    end
    Adj_used(src, dst) = 1; Adj_used(dst, src) = 1;
    eu2st(i_eu_u, :) = [src, dst];
    if sum( G_overlay.eo2eu(:, eu) ) > 0 
        used_eu(i_used_eu, 1) = src;
        used_eu(i_used_eu, 2) = dst;
        i_used_eu = i_used_eu + 1;
    end
    i_eu_u = i_eu_u + 1;
end

used_eu = used_eu(1:i_used_eu-1, :);

G_u = graph(eu2st(:, 1)', eu2st(:, 2)');

if G_overlay.G_type == 1 %% roofnet
    p1 = plot(G_u, 'NodeColor','b', 'EdgeColor', "#4DBEEE",'LineStyle','--', 'LineWidth', 0.01); %% plot the basic underlay topology
elseif G_overlay.G_type == 0 %% IAB
    load('IAB_Coordinates.mat')
    p1 = plot(G_u, 'XData', node_xval,'YData',node_yval, 'NodeColor','b', 'EdgeColor', "#4DBEEE",'LineStyle','--', 'LineWidth', 0.01); %% plot the basic underlay topology
end

highlight(p1, G_overlay.Vo, 'NodeColor','k', 'MarkerSize', 4) %% highlight overlay nodes

highlight(p1, used_eu(:, 1), used_eu(:, 2), 'EdgeColor', 'r', 'LineWidth', 0.5, 'LineStyle', '-') %% highlight the used eu