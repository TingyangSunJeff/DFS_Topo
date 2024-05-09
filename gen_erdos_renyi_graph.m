%% generate Erdos-Renyi Graph

function gen_erdos_renyi_graph(N_node, p)

N_total_pair = nchoosek(N_node, 2);
pair2st = zeros(N_total_pair, 3); %% idx-(s,t)-selected
Adj = zeros(N_node, N_node);

e = 1;
for u = 1 : N_node-1
    for v = u+1 : N_node
        pair2st(e, 1:2) = [u, v];
        rng = unifrnd(0,1);
        if rng <= p
            pair2st(e, 3) = 1;
            Adj(u, v) = 1;
        end
        e = e + 1;
    end
end

Adj = Adj + Adj';