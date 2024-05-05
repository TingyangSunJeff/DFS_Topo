function [ Adj, D, DG, node_xval, node_yval] = G_gen(G_type)

%% G_type = 0:Hex; 1:GtsCe
if G_type == 0 %Hex: nodes on axis and 4 symmetric areas
    n_edge = 3; % maximal number of nodes in each row, i.e, number of nodes in each edge
    d = 200; % interval = 200m
    d_1 = d*sqrt(3);
    node_xAxis_xval = -floor((n_edge-1)/2)*d_1: d_1: floor((n_edge-1)/2)*d_1;
    node_xAxis_yval = zeros(1,length(node_xAxis_xval));
    node_yAxis_yval = -(n_edge-1)*d: d:(n_edge-1)*d;
    node_yAxis_xval = zeros(1,2*(n_edge-1));
    node_yAxis_yval(n_edge) = []; % tick out the repeated original node
    
    n_area_nodes = n_edge^2;
    tmp_area_xval = zeros(1, n_area_nodes);
    tmp_area_yval = zeros(1, n_area_nodes);
    idx = 1;
    n_interval = 2*(n_edge-1)-1;
    for i = 1 : n_edge-1
        for j = 1 : 2 : n_interval
            tmp_area_xval(1, idx) = d_1 / 2 * i;
            tmp_area_yval(1, idx) = d/2 * (j+1) - mod(i,2)* d/2;
            idx = idx + 1;
        end
        n_interval = n_interval - 1;
    end
    n_area_nodes = sum(tmp_area_xval ~= 0);
    
    node_areas_xval = zeros(4, n_area_nodes); % quadrant 1 2 3 4
    node_areas_yval = zeros(4, n_area_nodes); % quadrant 1 2 3 4
    % quadrant 1
    node_areas_yval(1, :) = tmp_area_yval(1, 1:n_area_nodes);
    node_areas_xval(1, :) = tmp_area_xval(1, 1:n_area_nodes);
    % quadrant 2
    node_areas_yval(2, :) = node_areas_yval(1, :);
    node_areas_xval(2, :) = -node_areas_xval(1, :);
    % quadrant 3
    node_areas_yval(3, :) = -node_areas_yval(1, :);
    node_areas_xval(3, :) = -node_areas_xval(1, :);
    % quadrant 4
    node_areas_yval(4, :) = -node_areas_yval(1, :);
    node_areas_xval(4, :) = node_areas_xval(1, :);
    node_xval = [node_xAxis_xval, node_yAxis_xval, node_areas_xval(1, :), node_areas_xval(2, :), node_areas_xval(3, :), node_areas_xval(4, :)];
    node_yval = [node_xAxis_yval, node_yAxis_yval, node_areas_yval(1, :), node_areas_yval(2, :), node_areas_yval(3, :), node_areas_yval(4, :)];
    
    Adj = zeros(length(node_xval), length(node_xval));
    D = zeros(length(node_xval), length(node_xval)^2);
    idx = 1;
    %% symmetrical routing
    Adj(2, 5) = 1; Adj(2, 11) = 1; Adj(2, 8) = 1; Adj(2, 17) = 1; Adj(5, 14) = 1; Adj(5, 18) = 1;
    Adj(14, 16) = 1; Adj(14, 1) = 1; Adj(8, 6) = 1; Adj(8, 9) = 1; Adj(17, 19) = 1; Adj(17, 3) = 1;
    Adj(3, 10) = 1; Adj(3, 19) = 1; Adj(5, 15) = 1; Adj(15, 4) = 1; Adj(18, 4) = 1; Adj(18, 19) = 1;
    Adj(15, 16) = 1; Adj(11, 12) = 1; Adj(11, 1) = 1; Adj(12, 13) = 1; Adj(6, 7) = 1;
    Adj(6, 12) = 1; Adj(13, 1) = 1; Adj(7, 9) = 1; Adj(9, 10) = 1; 
    % Adj(11, 14) = 1; 
    Adj = Adj + Adj';
    [row, col] = find(Adj);
    i_edge = 1;
    for i_row = 1 : length(row)
        D(row(i_row), i_edge) = 1;
        D(col(i_row), i_edge) = -1;
        i_edge = i_edge + 1;
    end
    D = D(:, 1:i_edge-1);
    %% asymmetrical routing
    % % for i = 1 : length(node_xval)
    % %     %% upper child
    % %     idx_y_tmp = find(node_yval == node_yval(i) + d);
    % %     for j = 1 : length(idx_y_tmp)
    % %         if node_xval(idx_y_tmp(j)) == node_xval(i)
    % %             Adj(i, idx_y_tmp(j)) = 1;
    % %             D(i, idx) = 1; D(idx_y_tmp(j), idx) = -1; idx = idx + 1;
    % %             break;
    % %         end
    % %     end
    % %     %% lower child (left and right)
    % %     idx_y_tmp = find(node_yval == node_yval(i) - d/2);
    % %     for j = 1 : length(idx_y_tmp)
    % %         if node_xval(idx_y_tmp(j)) == node_xval(i)-d_1/2 || node_xval(idx_y_tmp(j)) == node_xval(i)+d_1/2
    % %             Adj(i, idx_y_tmp(j)) = 1;
    % %             D(i, idx) = 1; D(idx_y_tmp(j), idx) = -1; idx = idx + 1;
    % %         end
    % %     end
    % % end
    % % D = D(:, 1:idx-1);
    
    n_iab = length(Adj);
    Adj_full = [Adj, eye(n_iab); zeros(n_iab, 2*n_iab)];
%     for i = 1 : n_iab
%         Adj_full(i, i+n_iab) = 1;
%     end
    
    DG = digraph(Adj);
    SPR = shortestpathtree(DG,2,'OutputForm','cell');
    sa = 2; sb = 2;
    ta = 1:length(Adj);
    tb = 3:10;
    path_a = SPR; 
    path_b = SPR(tb,1);
    ta = ta + n_iab; tb = tb + n_iab; 
    
    %% Plot topology
% %     sa_idx = 49; sb_idx = 50;
% %     H2 = [Adj_full, zeros(38, 9+3); zeros(9+3, 47+3)]; % 49=S_A, 50 = S_B extra 3 since we need 7+3=10 tB, we add it to [9, 10, 15]->[46,47,48]
% %     H2(4, 39) = 1;
% %     H2(4,[40, 41]) = 1;
% %     H2(17, 42) = 1;
% %     H2(19, [43, 44]) = 1;
% %     H2(16, 45) = 1;
% %     H2([sa_idx, sb_idx], 2) = 1;
% %     H2(5, 46) = 1; H2(17, 47) = 1; H2(15, 48) = 1;
% %     G_T = digraph(H2);
% %     x_UEcoordinates = node_xval + 20;
% %     y_UEcoordinates = node_yval + 50;
% %     tmp = 80;
% %     tmp1 = 60;
% %     x_additional = [node_xval(4)+tmp, node_xval(4), node_xval(4)-tmp, node_xval(17)+tmp, node_xval(19)+tmp, node_xval(19)+tmp, node_xval(16)+tmp, ...
% %         node_xval(5)+tmp, node_xval(17)+tmp, node_xval(15)+tmp, -400, 400];
% %     y_additional = [node_yval(4)-tmp1, node_yval(4)-tmp1, node_yval(4)-tmp1, node_yval(17)+tmp1, node_yval(19)+tmp1, node_yval(19)-tmp1, node_yval(16), ...
% %         node_yval(5), node_yval(17), node_yval(15), -400, -400];
% % %     x_additional = [x_additional, node_xval(9)+tmp, node_xval(10)+tmp, node_xval(15)+tmp];
% % %     y_additional = [y_additional, node_yval(9)+tmp1, node_yval(10)+tmp1, node_yval(15)+tmp1];
% %     % G_gnb = digraph([0]);
% %     p1 = plot(G_T, 'XData',[node_xval, x_UEcoordinates, x_additional],'YData',[node_yval, y_UEcoordinates, y_additional], 'NodeColor','b', 'EdgeColor', "#4DBEEE",'LineStyle',':', 'LineWidth', 0.01);
% %     highlight(p1, 1:19, 'NodeColor','k', 'MarkerSize', 4)
% %     highlight(p1, sa_idx, 'NodeColor','g', 'MarkerSize', 4)
% %     highlight(p1, sb_idx, 'NodeColor','#EDB120', 'MarkerSize', 4)
% %     highlight(p1, [39, 40, 41, 42, 43, 44, 45, 46, 47, 48], 'NodeColor','m', 'MarkerSize', 4)
% %     highlight(p1, [sa_idx, sb_idx], [2, 2], 'EdgeColor', 'r', 'LineWidth', 0.5, 'LineStyle', '-')
% %     highlight(p1, 1:19, 20:38, 'EdgeColor', "#EDB120", 'LineWidth', 0.5, 'LineStyle', '-')
% % %     highlight(p1, 1:19, 20:38, 'EdgeColor', "#EDB120", 'LineStyle', '-')
% % %     highlight(p1, [1,3,3,4, 19, 19, 19, 9, 10, 15], 39:48, 'EdgeColor', "#EDB120", 'LineWidth', 0.5)
% %     highlight(p1, [4,4,4,17, 19, 19, 16, 5, 17, 15], 39:48, 'EdgeColor', "#EDB120", 'LineStyle', '-')
% %     for k = 1 : length(SPR)
% %         route = SPR{k};
% %         for kk = 1 : length(route)-1
% %             highlight(p1, route(kk), route(kk+1), 'LineWidth', 0.5, 'LineStyle', '-')
% %         end
% %     end
% %     disp('G_Gen Complete');
elseif G_type == 1 %% 1:GtsCe
    current_folder = pwd;
    graph_name = "GtsCe";
    if current_folder(1)=='H'
        file_name = strcat("H:\OneDrive - The Pennsylvania State University\NSF-overlay inference and control\Team Ting\Online Overlay Routing\Code\Data\\", graph_name, ".Graph");
    elseif current_folder(1)=='C'
        file_name = strcat("C:\Users\yxh5389\OneDrive - The Pennsylvania State University\NSF-overlay inference and control\Team Ting\Online Overlay Routing\Code\Data\\", graph_name, ".Graph");
    else
        file_name = "/export/home/Yudi_Huang/topology_data/GtsCe.graph";
    end
    n_nodes = 149;
    n_edges = 386; % directed, 198 undirected
    
    Adj = zeros(n_nodes, n_nodes);
    D = zeros(n_nodes, n_edges);
    
    fid = fopen(file_name);
    node_x = zeros(n_nodes, 1);
    node_y = zeros(n_nodes, 1);
    idx_node = 1;
    edges = zeros(2, n_edges);
    idx_edge = 1;
    for i = 1 : 1e4
        line = fgetl(fid);
        if any(line == -1) || (isempty(line) && i >= n_nodes+100)
            break;
        end
        newStr = split(line, ' ');
        if length(line) < 2
            continue;
        elseif line(2) == '_' || line(3) == '_' || line(4) == '_'
            node_x(idx_node) = str2double(newStr(2));
            node_y(idx_node) = str2double(newStr(3));
            idx_node = idx_node + 1;
        elseif line(1) == 'e'
            edges(1, idx_edge) = str2double(newStr(2)) + 1;
            edges(2, idx_edge) = str2double(newStr(3)) + 1;
            D(edges(1, idx_edge), idx_edge) = 1;
            D(edges(2, idx_edge), idx_edge) = -1;
            Adj(edges(1, idx_edge), edges(2, idx_edge)) = 1;
            Adj(edges(2, idx_edge), edges(1, idx_edge)) = 1;
            idx_edge = idx_edge + 1;
        end
    end
    
    Adj_full = 0;
    sa = randi([1 n_nodes],1,1); sb = randi([1 n_nodes],1,1);
    DG = digraph(edges(1,:), edges(2,:));
    path_a = shortestpathtree(DG,sa,'OutputForm','cell');
    [~,~,EE_a] = shortestpathtree(DG,sa);
    path_b = shortestpathtree(DG,sb,'OutputForm','cell');
    [~,~,EE_b] = shortestpathtree(DG,sb);
    SPR = [path_a(:)', path_b(:)'];
    ta = 1:length(Adj);
    tb = 1:length(Adj);
    node_xval = 0; node_yval = 0;
    intersection_mat_ab = zeros(length(path_a), length(path_b));
    vec_a = zeros(n_edges, 1);
    vec_b = zeros(n_edges, 1);
    for ia = 1 : length(path_a)
        vec_a = vec_a * 0;
        edges_a = findedge(DG, path_a{ia}(1:end-1), path_a{ia}(2:end));
        vec_a( edges_a ) = 1;
        for ib = 1 : length(path_b)
            vec_b = vec_b * 0;
            edges_b = findedge(DG, path_b{ib}(1:end-1), path_b{ib}(2:end));
            vec_b( edges_b ) = 1;
            if vec_a' * vec_b > 0
                intersection_mat_ab(ia, ib) = 1;
            end
        end
    end

%     G = graph;
%     for i = 1 : length(edges)%
% %         G = addedge(G, edges(1,i), edges(2,i));
%         if mod(i, 2) == 1
%             G = G.addedge(edges(1,i), edges(2,i));
%         end
%     end
end


%     function SPR = read_route_table(sa, sb)
%         SPR = {};
%         if current_folder(1)=='H'
%             route_file_name = strcat("H:\OneDrive - The Pennsylvania State University\NSF-overlay inference and control\Team Ting\Online Overlay Routing\Code\Data\\", graph_name, "\\");
%         else
%             route_file_name = "/export/home/Yudi_Huang/topology_data/GtsCe.graph";
%         end
%     end

end