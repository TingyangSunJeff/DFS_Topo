load('./data_res/Alg_Roofnet_CIFAR10_0412.mat');

Vo = G_overlay.Vo;

eu_cnt = zeros(1, size(G_overlay.G_u.D, 2)); %% cnt the #uses by overlay tunnels induced by Vo
for i_u = 1 : length(Vo)
    u = Vo(i_u);
    for i_v = 1 : length(Vo)
        v = Vo(i_v);
        if u == v
            continue;
        end
        eo = G_overlay.st2eo( strjoin(string([u, v])) );
        eu_B = find( G_overlay.eo2eu(eo, :) > 0 );
        % % if ~isKey(G_overlay.G_u.ST2idx, strjoin(string([u, v])))
        % %     disp("debug")
        % % end
        % % eu = G_overlay.G_u.ST2idx( strjoin(string([u, v])) );
        eu_cnt(eu_B) = eu_cnt(eu_B) + 1;
    end
end

figure
histogram(eu_cnt(eu_cnt > 0))
title("histogram of underlay link usage")
xlabel("number of usage")
ylabel("number of underlay links")