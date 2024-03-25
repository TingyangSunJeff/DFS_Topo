from network import create_underlay_network, find_shortest_path_with_delay
from optimization import optimize_network_route_rate, optimize_K_mixing_matrix, load_and_preprocess_data, d_psgd_training
from network import create_fully_connected_overlay_network
from network import minimum_spanning_tree, activate_links_random_spanning_tree, activate_links_ring_topology
from utils import draw_underlay_network_with_mst, draw_fully_connected_overlay_network, Ea_to_demand_model, calculate_path_delays, plot_acc_loss_over_epochs
from config import OVERLAY_NODES, ROOFNET_FILE_PATH
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def main():
    # Load the Roofnet data from the uploaded file
    roofnet_data = pd.read_csv(ROOFNET_FILE_PATH, delim_whitespace=True)

    edges = []
    for index, row in roofnet_data.iterrows():
        u = row['node1'] - 1  # Adjusting node numbering to start from 0
        v = row['node2'] - 1
        success_rate = row['success_rate']
        capacity = 1 # 1 Gbps
        delay = 1 / success_rate  # Example inverse relationship
        edges.append((u, v, capacity, delay))

    # Use a sorted tuple (smaller index first) as the key
    link_capacity_map = {tuple(sorted((edge[0], edge[1]))) : edge[2] for edge in edges}
    # Number of nodes can be determined as the unique nodes in the dataset
    num_nodes = len(set(roofnet_data['node1'].unique()).union(set(roofnet_data['node2'].unique())))

    # Create the underlay network from the given edges
    underlay = create_underlay_network(num_nodes, edges)

    # Create a fully connected overlay network from a subset of nodes
    fully_connected_overlay = create_fully_connected_overlay_network(underlay, OVERLAY_NODES)

    # Example: Activate links using a random spanning tree algorithm
    activated_links = activate_links_random_spanning_tree(fully_connected_overlay)
    # print("Activated Links (Random Spanning Tree):", activated_links)
    # Alternatively, activate links forming a ring topology
    activated_links_ring = activate_links_ring_topology(fully_connected_overlay)
    # print("Activated Links (Ring Topology):", activated_links_ring)

    # Draw the networks and their respective trees/topologies
    # draw_underlay_network_with_mst(underlay, mst)
    draw_fully_connected_overlay_network(fully_connected_overlay, OVERLAY_NODES, activated_links)

    # (11)
    optimal_rho_tilde, mixing_matrix = optimize_K_mixing_matrix(fully_connected_overlay, activated_links)
    # print("Optimal rho_tilde from main:", optimal_rho_tilde)
    # print("Mixing matrix from main:\n", mixing_matrix)
    multicast_demands_clique = Ea_to_demand_model(fully_connected_overlay.edges, 100)
    multicast_demands_ring = Ea_to_demand_model(activated_links_ring, 100)
    multicast_demands_random = Ea_to_demand_model(activated_links, 100)
    # (5)
    tau_random = optimize_network_route_rate(fully_connected_overlay, multicast_demands_random, underlay, link_capacity_map)
    tau_ring = optimize_network_route_rate(fully_connected_overlay, multicast_demands_ring, underlay, link_capacity_map)
    tau_baseline = optimize_network_route_rate(fully_connected_overlay, multicast_demands_clique, underlay, link_capacity_map)
    with open('tau_results.pkl', 'wb') as file:
        pickle.dump((tau_random, tau_ring, tau_baseline), file)
    # # D-PSGD
    # # Load data
    # (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    # # Run training
    # loss_history, val_accuracies = d_psgd_training(x_train, y_train, x_test, y_test, mixing_matrix, epochs=100)
    # # Save to file
    # with open('model_metrics.pkl', 'wb') as file:
    #     pickle.dump((loss_history, val_accuracies), file)
    # print("Data saved to 'model_metrics.pkl'.")
    # # plot_acc_loss_over_epochs(loss_history, val_accuracies)

if __name__ == "__main__":
    main()
