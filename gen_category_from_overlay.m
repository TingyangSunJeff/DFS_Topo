function F_all = gen_category_from_overlay(G_overlay)
%% generate all categories into a cell from G_overlay
    F_all = cell(1, size(G_overlay.G_u.D, 2));
    i_F = 1;
    used_eu = zeros(size(G_overlay.G_u.D, 2), 1);
    for eu = 1 : size(G_overlay.G_u.D, 2)
        % F = find(G_overlay.eo2eu(:, eu) > 0); %% F: overlay links
        if sum( G_overlay.eo2eu(:, eu) ) == 0 %% not traversed by any eo
            used_eu(eu) = 1;
            continue;
        end
        if used_eu(eu) == 1
            continue;
        end
        F = [eu]; %% F: underlay links
        % % if length(F) == 1
        % %     F_all{i_F} = F;
        % %     i_F = i_F + 1;
        % %     continue;
        % % end
        % % if isempty(F)
        % %     continue;
        % % end
        for eu_j = eu+1 : size(G_overlay.G_u.D, 2)
            % % if eu == 36 && eu_j == 116
            % %     disp("Debug");
            % % end
            if any( G_overlay.eo2eu(:, eu)-G_overlay.eo2eu(:, eu_j) ~= 0 )
                continue;
            end
            F = [F, eu_j]; %% F: underlay links
        end
        used_eu(F) = 1;
        F_all{i_F} = F;
        i_F = i_F + 1;
    end
    F_all = F_all(1:i_F-1);

end