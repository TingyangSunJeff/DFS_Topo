import numpy as np
from scipy.special import comb
from collections import defaultdict
import scipy.io
import os
from cvxpy import Variable, Problem, Minimize, sum as cvx_sum

def gen_overlay_netw(N_nodes, N_links, G_type, modelType, F_calculated):
    if G_type == 0:
        G_u = gen_underlay_netw(1, G_type)
    elif G_type == 1:
        G_u = read_RoofNet()
    else:
        raise ValueError("Wrong G_type")

    G_overlay = {}
    G_overlay['G_type'] = G_type
    G_overlay['C_F_lb'] = 1 - 1.5 / 100
    G_overlay['G_u'] = G_u
    N_Vu = G_u['Adj'].shape[0]

    Vo, spr_table = read_ov_routeTable(G_type)
    G_overlay['spr_table'] = spr_table
    G_overlay['st2spr'] = spr_table
    G_overlay['N_nodes'] = len(Vo)
    G_overlay['Vo'] = Vo
    N_links = G_overlay['N_nodes'] * (G_overlay['N_nodes']-1)
    G_overlay['Vu2Vo'] = np.zeros(N_Vu, dtype=int)
    G_overlay['Vu2Vo'][G_overlay['Vo']] = np.arange(1, len(G_overlay['Vo']) + 1)
    G_overlay['N_links'] = N_links

    if modelType == "CIFAR10":
        N_param = 25.6 * 1e6
    elif modelType == "MNIST":
        N_param = 1663370
    else:
        raise ValueError("Non-existing model type")

    G_overlay['Kappa'] = N_param * 64 / 1e6
    G_overlay['Bp'] = np.zeros((N_nodes, int(comb(N_nodes, 2))), dtype=int)
    G_overlay['st2idx'] = {}
    G_overlay['idx2st'] = np.zeros((G_overlay['Bp'].shape[1], 2), dtype=int)
    G_overlay['pair2link'] = np.zeros((G_overlay['Bp'].shape[1], 2), dtype=int)
    G_overlay['link2pair'] = np.zeros(N_links, dtype=int)
    G_overlay['pairDelay'] = np.zeros(G_overlay['Bp'].shape[1])
    G_overlay['B'] = np.zeros((N_nodes, N_links), dtype=int)
    G_overlay['st2eo'] = {}
    G_overlay['st2spr'] = {}
    G_overlay['link2st'] = np.zeros((N_links, 2), dtype=int)
    G_overlay['eo2eu'] = np.zeros((N_links, G_u['D'].shape[1]), dtype=int)
    G_overlay['delays'] = np.zeros(N_links)

    i_pair = 1
    n_pair = G_overlay['Bp'].shape[1]
    for i_u in range(G_overlay['Vo'].shape[0]):
        u = G_overlay['Vo'][i_u]
        for i_v in range(i_u + 1, G_overlay['Vo'].shape[0]):
            v = G_overlay['Vo'][i_v]
            G_overlay['Bp'][u, i_pair - 1] = 1
            G_overlay['Bp'][v, i_pair - 1] = -1
            G_overlay['st2idx'][str((u, v))] = i_pair
            G_overlay['idx2st'][i_pair - 1] = [u, v]

            i_link = i_pair + n_pair
            G_overlay['B'][u, i_pair - 1] = 1
            G_overlay['B'][v, i_pair - 1] = -1
            G_overlay['B'][u, i_link - 1] = -1
            G_overlay['B'][v, i_link - 1] = 1
            G_overlay['st2eo'][str((u, v))] = i_pair
            G_overlay['st2eo'][str((v, u))] = i_link
            G_overlay['link2st'][i_pair - 1] = [u, v]
            G_overlay['link2st'][i_link - 1] = [v, u]
            G_overlay['pair2link'][i_pair - 1] = [i_pair, i_link]
            G_overlay['link2pair'][i_pair - 1] = i_pair
            G_overlay['link2pair'][i_link - 1] = i_pair

            route_uv = G_overlay['spr_table'][str((u, v))]
            eo_uv = i_pair
            eo_vu = i_link
            for j in range(len(route_uv) - 1):
                um = route_uv[j]
                vm = route_uv[j + 1]
                eu = G_u['ST2idx'][str((um, vm))]
                G_overlay['eo2eu'][eo_uv - 1, eu - 1] = 1
                G_overlay['delays'][eo_uv - 1] += G_u['delays'][eu - 1]
                eu_vu = G_u['ST2idx'][str((vm, um))]
                G_overlay['eo2eu'][eo_vu - 1, eu_vu - 1] = 1

            G_overlay['delays'][eo_vu - 1] = G_overlay['delays'][eo_uv - 1]
            i_pair += 1

    for i_p in range(G_overlay['Bp'].shape[1]):
        eo1 = G_overlay['pair2link'][i_p, 0]
        eo2 = G_overlay['pair2link'][i_p, 1]
        G_overlay['pairDelay'][i_p] = G_overlay['delays'][eo1 - 1] + G_overlay['delays'][eo2 - 1]

    G_overlay['F_all'] = gen_category_from_overlay(G_overlay)
    if F_calculated <= 0:
        N_F = len(G_overlay['F_all'])
        G_overlay['F2pair'] = np.zeros((N_F, n_pair), dtype=int)
        G_overlay['C_F'] = np.zeros(N_F)
        G_overlay['F2link'] = np.zeros((N_F, N_links), dtype=int)

        for i_F in range(len(G_overlay['F_all'])):
            F_eu = G_overlay['F_all'][i_F]
            if len(F_eu) == 1:
                C_F = G_overlay['G_u']['C'][F_eu[0] - 1]
            else:
                C_F = min(G_overlay['G_u']['C'][np.array(F_eu) - 1])
            G_overlay['C_F'][i_F] = C_F
            F = np.where(G_overlay['eo2eu'][:, F_eu[0] - 1] > 0)[0] + 1
            if len(F) == 0:
                continue
            G_overlay['F2link'][i_F, F - 1] = 1
            v_pairs = G_overlay['link2pair'][F - 1]
            G_overlay['F2pair'][i_F, v_pairs - 1] = 1

        G_overlay['F_calc_eu'] = G_overlay['F_all']
        G_overlay['F_all'] = transform_Feu_to_Feo(G_overlay)
        G_overlay['F_calc'] = G_overlay['F_all']
    else:
        if G_overlay['G_type'] == 0:
            netw_name = "IAB"
        elif G_overlay['G_type'] == 1:
            netw_name = "Roofnet"
        filename_F = f"./data_res/F_dict2mat_{netw_name}.mat"
        G_overlay['F_calc'] = read_F_calc_py2mat(G_overlay, filename_F)
        N_F = len(G_overlay['F_calc'])
        G_overlay['F2pair'] = np.zeros((N_F, n_pair), dtype=int)
        G_overlay['C_F'] = np.zeros(N_F)
        G_overlay['F2link'] = np.zeros((N_F, N_links), dtype=int)

        for i_F in range(len(G_overlay['F_calc'])):
            F = G_overlay['F_calc'][i_F]
            F_eu = np.where(G_overlay['eo2eu'][F - 1, :] > 0)[1] + 1
            C_F_true = calc_effectiveCapacity(G_overlay, F)
            G_overlay['C_F'][i_F] = np.random.uniform(G_overlay['C_F_lb'], 1) * C_F_true
            if len(F) == 0:
                raise ValueError("empty F")
            G_overlay['F2link'][i_F, F - 1] = 1
            v_pairs = G_overlay['link2pair'][F - 1]
            G_overlay['F2pair'][i_F, v_pairs - 1] = 1

    return G_overlay



def transform_Feu_to_Feo(G_overlay):
    N_F = len(G_overlay['F_calc_eu'])
    F_all_eo = [None] * N_F

    for i_F in range(N_F):
        F_eu = G_overlay['F_calc_eu'][i_F]
        F_eo = np.where(np.sum(G_overlay['eo2eu'][:, F_eu - 1], axis=1) > 0)[0] + 1
        F_all_eo[i_F] = F_eo

    return F_all_eo

def calc_effectiveCapacity(G_overlay, F):
    f = Variable(G_overlay['N_links'])
    constraints = [f >= 0]

    for eu in range(G_overlay['G_u']['D'].shape[1]):
        F_traverse_eu = np.intersect1d(np.where(G_overlay['eo2eu'][:, eu] > 0)[0] + 1, F)
        if len(F_traverse_eu) == 0:
            continue
        constraints.append(cvx_sum(f[F_traverse_eu - 1]) <= G_overlay['G_u']['C'][eu])

    objective = Minimize(-cvx_sum(f[F - 1]))

    problem = Problem(objective, constraints)
    problem.solve(solver='MOSEK', verbose=False)

    if problem.status not in ["infeasible", "unbounded"]:
        return cvx_sum(f[F - 1]).value
    else:
        raise ValueError("Optimization failed in calc_effectiveCapacity")

def read_F_calc_py2mat(G_overlay, filename_F):
    F_M = scipy.io.loadmat(filename_F)['F_M']
    len_F = F_M.shape[0]
    len_pairs = F_M.shape[1]

    F_calc = [None] * len_F

    for k in range(len_F):
        tmp = F_M[k, :, :]
        F = np.zeros(len_pairs, dtype=int)
        for i_p in range(len_pairs):
            src = int(tmp[i_p, 0]) + 1
            dst = int(tmp[i_p, 1]) + 1
            if not (src == 1 and dst == 1):
                F[i_p] = G_overlay['st2eo'][str((src, dst))]
            else:
                F = F[:i_p]
                F_calc[k] = F
                break

    return F_calc

def read_RoofNet():
    N_v = 38
    N_pair = 219
    N_E = N_pair * 2
    D = np.zeros((N_v, N_E))
    Adj = np.zeros((N_v, N_v))
    ST2idx = {}

    with open("Roofnet_AvgSuccessRate_1Mbps.txt", 'r') as fileID:
        idx_edge = 1
        for line in fileID:
            if line.startswith('n'):
                continue
            else:
                newStr = line.split()
                src = int(newStr[0])
                dst = int(newStr[1])
                D[src, idx_edge - 1] = 1
                D[dst, idx_edge - 1] = -1
                Adj[src, dst] = 1
                ST2idx[str((src, dst))] = idx_edge

                idx_edge_reverse = idx_edge + N_pair
                D[dst, idx_edge_reverse - 1] = 1
                D[src, idx_edge_reverse - 1] = -1
                ST2idx[str((dst, src))] = idx_edge_reverse
                idx_edge += 1

    Adj += Adj.T

    G_u = {
        'Adj': Adj,
        'D': D,
        'delays': 5e-4 * np.ones(D.shape[1]),
        'C': 0.001 * np.ones(D.shape[1]),
        'ST2idx': ST2idx
    }

    return G_u

def read_ov_routeTable(G_type):
    if G_type == 0:
        Vo_filename = "IAB_topo4mat.mat"
        route_filename = "IAB_underlay_routing_map.txt"
    elif G_type == 1:
        Vo_filename = "roofnet_overlay_nodes.mat"
        route_filename = "roofnet_underlay_routing_map.txt"

    Vo = np.sort(scipy.io.loadmat(Vo_filename)['overlay_V'])
    spr_table = {}

    with open(route_filename, 'r') as fileID:
        for line in fileID:
            newStr = line.split()
            N_hop = int(newStr[2])
            src = int(newStr[0]) + 1
            dst = int(newStr[1]) + 1
            route = [-1] * N_hop
            for k in range(N_hop):
                route[k] = int(newStr[3 + k]) + 1
            spr_table[str((src, dst))] = route

    return Vo, spr_table

def gen_underlay_netw(N_nodes, G_type):
    if G_type == 0:
        Adj, D, DG, node_xval, node_yval = G_gen(G_type)

    G_u = {
        'Adj': Adj,
        'D': D,
        'DG': DG,
        'delays': 5e-4 * np.ones(D.shape[1]),
        'C': 0.4 * np.ones(D.shape[1]) if G_type == 0 else 0.001 * np.ones(D.shape[1]),
        'ST2idx': {}
    }

    for eu in range(D.shape[1]):
        u = np.where(D[:, eu] == 1)[0][0]
        v = np.where(D[:, eu] == -1)[0][0]
        G_u['ST2idx'][str((u, v))] = eu + 1

    return G_u

def test_lemma3_2(G_overlay, z, tol):
    """
    Test Lemma 3.2 for optimal rate allocation given z
    """
    # Initialize t_Fs as a zero array
    t_Fs = np.zeros(len(G_overlay['F_calc']))
    
    # Extract values from the input variables
    z = np.array(z)  # Assuming z is a NumPy array or can be converted to one
    C_e = G_overlay['C_F']
    
    # Iterate over each virtual overlay
    for i_vo in range(len(z)):
        used_eo = np.where(z[i_vo, :] > 0)[0]  # Find indices where z[i_vo, :] > 0
        
        for i_F in range(len(G_overlay['F_calc'])):
            F = G_overlay['F_calc'][i_F]
            t_Fs[i_F] += len(np.intersect1d(F, used_eo))
    
    tau = G_overlay['Kappa'] / np.min(C_e / t_Fs)
    
    return tau