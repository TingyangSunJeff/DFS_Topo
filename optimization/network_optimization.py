from gurobipy import Model, GRB, quicksum
from utils import adjust_matrix, edges_to_delay_map_with_reversed, map_overlay_to_underlay_edges
import cvxpy as cp
import numpy as np

def optimize_network_route_rate_direct(fully_connected_overlay, multicast_demands, undelay_network, link_capacity_map):
    underlay_links = list(undelay_network.edges)
    overlay_nodes = fully_connected_overlay.nodes
    overlay_links = []
    # Populate the undirected_links list
    for link in fully_connected_overlay.edges():
        overlay_links.append(link)  # Add original direction
        overlay_links.append(link[::-1])  # Add reverse direction

    H=[]
    for source, destinations, data_size in multicast_demands:
        for destination in destinations:
            H.append((source, destination, data_size))

    link_delay_map, link_path_map = edges_to_delay_map_with_reversed(fully_connected_overlay.edges(data=True))

    overlay_links_map = map_overlay_to_underlay_edges(undelay_network, overlay_links, link_path_map)
    # Initialize Gurobi model
    m = Model("tau_optimization_direct")
    m.setParam('TimeLimit', 100)
    M = 0.4 * 1.e+9  # unit: bits per second

    # Define decision variables
    f = {}  # Flow variable for the amount of demand on a link
    f_inv = {} 
    # d = {}  # Continuous variable representing the demand satisfaction level
    # d_inv = {}  # Inverse of d, for modeling purposes
    tau = m.addVar(vtype=GRB.CONTINUOUS, name="tau")
    m.setObjective(tau, GRB.MINIMIZE)

    # Populate demand pairs and their data sizes
    H = [(source, destination, data_size) for source, destinations, data_size in multicast_demands for destination in destinations]

    for h in multicast_demands:
        source, T_h, data_size = h
        # Convert destinations to a frozenset for immutability and to ensure uniqueness
        h_index = (source, frozenset(T_h), data_size)
        # d[h_index] = m.addVar(vtype=GRB.CONTINUOUS, name=f"d_{h_index}", lb=0, ub=M)
        # d_inv[h_index] = m.addVar(vtype=GRB.CONTINUOUS, name=f"inv_{h_index}")

        # Create the inverse constraint for d and d_inv
        # m.addConstr(d_inv[h_index] * d[h_index] == 1, name=f"inv_constraint_{h_index}")

        # Iterate through each overlay link and create binary variables z and flow variables f for each demand
        for i, j in overlay_links:
            link_index = (i, j)
            f[(link_index, h_index)] = m.addVar(vtype=GRB.CONTINUOUS, name=f"f_{link_index}_{h_index}", lb=0)
            f_inv[(link_index, h_index)] = m.addVar(vtype=GRB.CONTINUOUS, name=f"f_inv_{link_index}_{h_index}")
            m.addConstr(f_inv[(link_index, h_index)] * f[(link_index, h_index)] == 1, name=f"finv_constraint_{h_index}")


    # Constraints
    # (5b) Tau is an upper bound on the completion time of the slowest flow
    for s_h, T_h, k_h in multicast_demands:
        for k in T_h:
            m.addConstr(tau >= k_h * f_inv[(int(s_h), k), (s_h, frozenset(T_h), k_h)] + quicksum(link_delay_map[s_h, k] for k in T_h))

    # (5c) The total traffic rate imposed by the overlay on any underlay link is within its capacity
    for underlay_link in underlay_links:
        if overlay_links_map[underlay_link]:
            m.addConstr(quicksum(f[ovelaylink, (s_h, frozenset(T_h), k_h)] 
                                 for ovelaylink in overlay_links_map[underlay_link] for s_h, T_h, k_h in multicast_demands) <= link_capacity_map[tuple(sorted(underlay_link))])


    m.optimize()

    # Check and print the optimal solution
    if m.status == GRB.OPTIMAL:
        print('Optimal value of tau:', m.getVarByName("tau").X)
    else:
        print('No optimal solution found')

    return m.getVarByName("tau").X


def optimize_network_route_rate(fully_connected_overlay, multicast_demands, undelay_network, link_capacity_map):
    underlay_links = list(undelay_network.edges)
    overlay_nodes = fully_connected_overlay.nodes
    overlay_links = []
    # Populate the undirected_links list
    for link in fully_connected_overlay.edges():
        overlay_links.append(link)  # Add original direction
        overlay_links.append(link[::-1])  # Add reverse direction

    H=[]
    for source, destinations, data_size in multicast_demands:
        for destination in destinations:
            H.append((source, destination, data_size))

    link_delay_map, link_path_map = edges_to_delay_map_with_reversed(fully_connected_overlay.edges(data=True))

    overlay_links_map = map_overlay_to_underlay_edges(undelay_network, overlay_links, link_path_map)
    # Placeholder for initialization of Gurobi model
    m = Model("tau_optimization")
    m.setParam('TimeLimit', 100)
    M = 1000000  # unit:bits per s

    # Define decision variables
    z = {}  # Binary variable for whether a link is used by a demand
    f = {}  # Flow variable for the amount of demand on a link
    d = {}  # Continuous variable representing the demand satisfaction level
    d_inv = {}  # Inverse of d, for modeling purposes

    for h in multicast_demands:
        source, T_h, data_size = h
        # Convert destinations to a frozenset for immutability and to ensure uniqueness
        h_index = (source, frozenset(T_h), data_size)
        d[h_index] = m.addVar(vtype=GRB.CONTINUOUS, name=f"d_{h_index}", lb=0, ub=M)
        d_inv[h_index] = m.addVar(vtype=GRB.CONTINUOUS, name=f"inv_{h_index}")

        # Create the inverse constraint for d and d_inv
        m.addConstr(d_inv[h_index] * d[h_index] == 1, name=f"inv_constraint_{h_index}")

        # Iterate through each overlay link and create binary variables z and flow variables f for each demand
        for i, j in overlay_links:
            link_index = (i, j)
            z[(link_index, h_index)] = m.addVar(vtype=GRB.BINARY, name=f"z_{link_index}_{h_index}")
            f[(link_index, h_index)] = m.addVar(vtype=GRB.CONTINUOUS, name=f"f_{link_index}_{h_index}", lb=0)

    r = m.addVars(overlay_links, H, vtype=GRB.BINARY, name="r")

    # Objective: Minimize tau 
    tau = m.addVar(vtype=GRB.CONTINUOUS, name="tau")
    m.setObjective(tau, GRB.MINIMIZE)

    # Constraints
    # (5b) Tau is an upper bound on the completion time of the slowest flow
    for s_h, T_h, k_h in multicast_demands:
        for k in T_h:
            m.addConstr(tau >= k_h * d_inv[s_h, frozenset(T_h), k_h] + quicksum(link_delay_map[i, j] * r[i, j, s_h, k, k_h] for i, j in overlay_links))

    # (5c) The total traffic rate imposed by the overlay on any underlay link is within its capacity
    for underlay_link in underlay_links:
        if overlay_links_map[underlay_link]:
            m.addConstr(quicksum(f[ovelaylink, (s_h, frozenset(T_h), k_h)] 
                                 for ovelaylink in overlay_links_map[underlay_link] for s_h, T_h, k_h in multicast_demands) <= link_capacity_map[tuple(sorted(underlay_link))])

    # (5d) Flow conservation for non-source, non-destination nodes
    for s_h, k, k_h  in H:
        for i in overlay_nodes:
            b = 1 if i==s_h else (-1 if i==k else 0)
            m.addConstr(quicksum(r[i, j, s_h, k, k_h] for j in overlay_nodes if (i, j) in overlay_links) ==
                        quicksum(r[j, i, s_h, k, k_h] for j in overlay_nodes if (i, j) in overlay_links) + b)

    # (5e) Flow only happens on active paths
    for s_h, T_h, k_h in multicast_demands:
        for k in T_h:
            for i, j in overlay_links:
                m.addConstr(r[i, j, s_h, k, k_h] <= z[(i, j), (s_h, frozenset(T_h), k_h)])

    # Constraint (5f): If z_ij^h is 0 (link (i,j) is not used by flow h), then f_ij^h must be 0; otherwise, it's bounded by d_h
    for s_h, T_h, k_h in multicast_demands:
        for i, j in overlay_links:
            m.addConstr(d[s_h, frozenset(T_h), k_h] - M * (1 - z[(i, j), (s_h, frozenset(T_h), k_h)]) <= f[(i, j), (s_h, frozenset(T_h), k_h)], name=f"flow_lower_bound_{s_h}_{T_h}_{k_h}_{i}_{j}")
            m.addConstr(f[(i, j), (s_h, frozenset(T_h), k_h)] <= d[s_h, frozenset(T_h), k_h], name=f"flow_upper_bound_{s_h}_{T_h}_{k_h}_{i}_{j}")


    # # (5g) Flow rate f is less than or equal to M times the binary decision on z
    # for h in H:
    #     for i, j in Ea:
    #         m.addConstr(f[h, i, j] <= M * z[h, i, j])


    # Optimize the model
    m.optimize()

    # Check and print the optimal solution
    if m.status == GRB.OPTIMAL:
        print('Optimal value of tau:', m.getVarByName("tau").X)
    else:
        print('No optimal solution found')

    return m.getVarByName("tau").X

# # Additional functions and helpers could also be defined here if needed

def optimize_K_mixing_matrix(fully_connected_overlay, activated_links):
    num_nodes = len(fully_connected_overlay.nodes()) # number of nodes (size of I and J) m in paper
    num_edges = len(fully_connected_overlay.edges()) # number of edges (size of alpha)
    edges = list(fully_connected_overlay.edges())
    # Initialize the incidence matrix B with zeros
    B = np.zeros((num_nodes, num_edges))

    # Unique nodes in the overlay network
    nodes = list(set(node for edge in edges for node in edge))
    nodes.sort()  # Sort the nodes for consistent ordering

    # Create a mapping from node to index in the nodes list
    node_to_index = {node: idx for idx, node in enumerate(nodes)}

    # Populate the incidence matrix B
    for edge_index, (i, j) in enumerate(edges):
        node_i_index = node_to_index[i]
        node_j_index = node_to_index[j]
        B[node_i_index][edge_index] = 1   # 'from' node, start of edge
        B[node_j_index][edge_index] = -1  # 'to' node, end of edge

    J = (1.0 / num_nodes) * np.ones((num_nodes, num_nodes))  # ideal mixing matrix mxm

    # Define E_alpha based on your problem (the edges where alpha is not constrained to zero)
    E_alpha = [(i, j) for i, j in activated_links]

    # Define the variable for alpha (with the size num_edges)
    alpha = cp.Variable(num_edges)

    # Define the SDP constraint matrix
    constraint_matrix = cp.diag(np.ones(num_nodes)) - B @ cp.diag(alpha) @ B.T - J

    # Define the variable for rho
    rho = cp.Variable()

    # Define the constraints
    constraints = [-rho * cp.diag(np.ones(num_nodes)) << constraint_matrix,
                constraint_matrix << rho * cp.diag(np.ones(num_nodes))]
    # Add constraints for alpha_ij = 0 where (i, j) not in E_alpha
    for edge_index, edge in enumerate(edges):
        if edge not in E_alpha:
            constraints.append(alpha[edge_index] == 0)
    # constraints.append(alpha >= 0)
    # Define the objective
    objective = cp.Minimize(rho)

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    # After solving the problem, you can access the optimal value of rho and the optimal alpha
    optimal_rho = rho.value
    optimal_alpha = alpha.value
    # print("=====", optimal_rho)
    optimal_alpha = [x if x >= 1e-5 else 0 for x in optimal_alpha]

    W = np.eye(num_nodes) - np.dot(np.dot(B, np.diag(optimal_alpha)), B.T)
    return optimal_rho, W


def metropolis_weights(overlay_nodes, active_edges, node_degrees):
    """
    Calculate the Metropolis weights for a given time-varying graph.

    Parameters:
    - overlay_nodes: A list of node indices representing the overlay network.
    - active_edges: A set of tuples representing the active edges at time t using underlay indices.
    - node_degrees: A dictionary representing the degree of each node at time t using underlay indices.

    Returns:
    - weights: A 2D list representing the weight matrix W(t) where weights[i][j] is the weight from node i to node j in the overlay.
    """
    num_nodes = len(overlay_nodes)
    # Map underlay indices to overlay indices
    overlay_index_map = {node: idx for idx, node in enumerate(overlay_nodes)}

    # Initialize weight matrix with zeros
    weights = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

    # Fill weights for active edges
    for i, j in active_edges:
        if i in overlay_index_map and j in overlay_index_map:
            overlay_i = overlay_index_map[i]
            overlay_j = overlay_index_map[j]

            if overlay_i != overlay_j:  # Ensure it's not a self-loop
                weight_ij = 1 / (1 + max(node_degrees[i], node_degrees[j]))
                weights[overlay_i][overlay_j] = weight_ij
                weights[overlay_j][overlay_i] = weight_ij  # Since the graph is undirected

    # Fill self-loop weights
    for i in range(num_nodes):
        # Ensure row sum is 1 by subtracting sum of other weights in row
        neighbor_sum = sum(weights[i][j] for j in range(num_nodes) if j != i)
        weights[i][i] = 1 - neighbor_sum

    return weights