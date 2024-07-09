import numpy as np
import networkx as nx
import cvxpy as cp
import random
import time

def utils_path_yalmip_mosek():
    print("config completed")

N_nodes = 19  # maximum number of nodes in test
G_type = 0
if G_type == 0:
    G_name = 'IAB_'
elif G_type == 1:
    G_name = 'Roofnet_'

n_node_start = 10
N_runs = 5
N_run = N_runs
N_list_nV = len(range(n_node_start, N_nodes + 1))

T_SDR = np.zeros((N_runs, 2))
T_MICP = np.zeros((N_runs, 2))
T_SDR_lambda2_ew = np.zeros((N_runs, 2))
T_SDR_rho_ew = np.zeros((N_runs, 2))
T_SDR_joint_rho_y = np.zeros((N_runs, 2))
T_SCA_joint_rho_y = np.zeros((N_runs, 2))
T_SCA_rho_ew = np.zeros((N_runs, 2))
T_SCA_add_rho_ew = np.zeros((N_runs, 2))
T_SCA_both_rho_ew = np.zeros((N_runs, 2))
T_boyd_greedy = np.zeros((N_runs, 2))
T_MKP = np.zeros((N_runs, 2))
T_ring1 = np.zeros((N_runs, 2))
T_ring2 = np.zeros((N_runs, 2))
T_ring3 = np.zeros((N_runs, 2))
T_allPair = np.zeros((N_runs, 2))
T_MST1 = np.zeros((N_runs, 2))
T_MST2 = np.zeros((N_runs, 2))
Ea_algName2write = ['boyd_greedy', 'SDR_rho_ew', 'SDR_lambda2_ew', 'SCA_both_23', 'clique']

i_nV = 1
modelType = "CIFAR10"
F_calculated = 1

def gen_overlay_netw(n_V, N_links, G_type, modelType, F_calculated):
    G_overlay = {}
    G_overlay['B'] = np.zeros((n_V, n_V))  # Placeholder for the B matrix
    G_overlay['G_type'] = G_type
    G_overlay['F_calculated'] = F_calculated
    return G_overlay

def biLevel_upper(G_overlay, M1, M2, lowerAlg):
    Ea_ring = gen_random_ring(G_overlay)
    solution_ring = solve_tau_noRouting(G_overlay, Ea_ring)
    tau_lb = solution_ring['tau']

    solution_E = solve_tau_noRouting(G_overlay, list(range(1, G_overlay['N_links'] + 1)))
    tau_ub = solution_E['tau']
    L_beta = np.linspace(tau_lb, tau_ub, 10)

    rho_res = -1 * np.ones(len(L_beta))
    y_res = np.zeros((G_overlay['N_links'], len(L_beta)))
    best_wall_clock_time = float('inf')
    best_Ea = None
    runTime = 0

    for beta in L_beta:
        if lowerAlg == "boyd_greedy":
            sol_tmp = greedy_boyd_lower(G_overlay, beta)
        elif lowerAlg == "MKP":
            sol_tmp = MKP_lower(G_overlay, beta)
        elif lowerAlg == "SDR_lambda2_equalWeight":
            sol_tmp = SDR_lambda2_equalWeight(G_overlay, beta)
        elif lowerAlg == "SDR_rho_equalWeight":
            sol_tmp = SDR_rho_equalWeight(G_overlay, beta)
        elif lowerAlg == "SCA_both_rho_ew":
            sol_tmp = SCA_both_rho_ew(G_overlay, beta)
        else:
            raise ValueError(f"Unknown algorithm: {lowerAlg}")

        runTime += sol_tmp['elapsed_time']

        if sol_tmp['success'] > 0:
            rho_res.append(sol_tmp['rho'])
            y_res[:, i_beta] = sol_tmp['y']
            Ea_tmp = [i for i, y in enumerate(y_res[:, i_beta]) if y > 1e-5]
            K_tmp = gen_K_from_rho(rho_res[i_beta], M1, M2)
            tau_sol_tmp = solve_tau_noRouting(G_overlay, Ea_tmp)
            tau_tmp = tau_sol_tmp['tau']
            wall_clock_time = K_tmp * tau_tmp
            if wall_clock_time < best_wall_clock_time:
                best_wall_clock_time = wall_clock_time
                best_Ea = Ea_tmp

    return {
        'Ea': best_Ea,
        'T': best_wall_clock_time,
        'runTime': runTime
    }

def gen_random_ring(G_overlay):
    shuffle_idx = np.random.permutation(G_overlay['N_nodes'])
    Vo_shuffle = G_overlay['Vo'][shuffle_idx]
    Ea = np.zeros(len(shuffle_idx) - 1)
    for i in range(len(Vo_shuffle) - 1):
        u, v = Vo_shuffle[i], Vo_shuffle[i + 1]
        key = str(min(u, v)) + "_" + str(max(u, v))
        Ea[i] = G_overlay['st2idx'][key]
    u, v = Vo_shuffle[-1], Vo_shuffle[0]
    key = str(min(u, v)) + "_" + str(max(u, v))
    Ea[-1] = G_overlay['st2idx'][key]
    return Ea

def solve_tau_noRouting(G_overlay, Ea):
    f = cp.Variable(G_overlay['N_links'])
    tau = cp.Variable()
    Eo = list(set(G_overlay['pair2link'][Ea, :].flatten()))

    constraints = [f >= 0, tau >= 0]
    constraints += [f[i] <= 1e-4 for i in range(G_overlay['N_links']) if i not in Eo]
    for pair_idx in Ea:
        eo_1, eo_2 = G_overlay['pair2link'][pair_idx]
        constraints += [tau >= G_overlay['Kappa'] / f[eo_1] + G_overlay['delays'][eo_1],
                        tau >= G_overlay['Kappa'] / f[eo_2] + G_overlay['delays'][eo_2]]

    for F in G_overlay['F_calc']:
        constraints.append(cp.sum(f[F]) <= G_overlay['C_F'][F])

    objective = cp.Minimize(tau)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)

    return {
        'f': f.value,
        'tau': tau.value
    }

def gen_K_from_rho(rho, M1, M2):
    p = 1 - rho
    zeta = 0.5
    epsilon_0 = 0.06
    sigma = 4
    K = (sigma**2 / (10 * epsilon_0**2) +
         (zeta * np.sqrt(M1 + 1) + sigma * np.sqrt(p)) / (p * epsilon_0**1.5) +
         np.sqrt((M1 + 1) * (M2 + 1)) / (p * epsilon_0))
    return K

def greedy_boyd_lower(G_overlay, beta):
    N_pairs = G_overlay['N_links']
    Bp = G_overlay['Bp'][G_overlay['Vo'], :]
    y_selected = np.zeros(N_pairs)

    start_time = time.time()

    for _ in range(N_pairs):
        L_tmp = Bp @ np.diag(y_selected) @ Bp.T
        _, v2 = np.linalg.eig(L_tmp)
        v2 = v2[:, 1]
        best_gd = 0
        best_pair = -1
        for i_p in range(N_pairs):
            if y_selected[i_p] > 1e-4:
                continue
            s, t = G_overlay['idx2st'][i_p]
            gd = abs(v2[s] - v2[t])
            if gd > best_gd:
                y_bin = y_selected.copy()
                y_bin[i_p] = 1
                constrViolated = False
                for F in G_overlay['F_calc']:
                    C_F = G_overlay['C_F'][F]
                    constr_violation = np.sum(y_bin[F] * G_overlay['Kappa'] / (beta - G_overlay['delays'][F])) - C_F
                    if constr_violation > 0:
                        constrViolated = True
                        break
                if not constrViolated:
                    best_gd = gd
                    best_pair = i_p
        if best_pair >= 0:
            y_selected[best_pair] = 1
        else:
            break

    elapsed_time = time.time() - start_time

    L = Bp @ np.diag(y_selected) @ Bp.T
    _, lambda_a = np.linalg.eig(L)
    lambda_a = lambda_a[1]
    return {
        'Ea': np.where(y_selected > 1e-