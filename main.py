from network import create_underlay_network, find_shortest_path_with_delay
from optimization import optimize_network_route_rate, optimize_K_mixing_matrix, load_and_preprocess_data, d_psgd_training
from network import create_fully_connected_overlay_network, activate_links_prim_topology
from network import activate_links_random_spanning_tree, activate_links_ring_topology
from utils import plot_degree_distribution, draw_fully_connected_overlay_network, Ea_to_demand_model, load_network_data, plot_acc_loss_over_epochs
import pandas as pd
import pickle
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def process_adjacency_matrix(matrix):
    """
    Transform an adjacency matrix into a list of links (tuples).
    """
    links = []
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 1:
                links.append((i, j))
    return links

def process_files(file_paths):
    """
    Process multiple files containing adjacency matrices and return a list of activated links.
    """
    all_links = []
    for file_path in file_paths:
        # Load the list of adjacency matrices from the pickle file
        with open(file_path, 'rb') as file:
            adjacency_matrices = pickle.load(file)
        # Process each matrix in the list
        for matrix in adjacency_matrices:
            matrix = np.array(matrix)  # Ensure it's a NumPy array for processing
            links = process_adjacency_matrix(matrix)
            all_links.append(links)
    
    return all_links


def main():
    with open('./network_settings.json', 'r') as json_file:
        loaded_network_settings = json.load(json_file)  
    # edges, num_nodes = load_network_data(ROOFNET_FILE_PATH)
    network_type = "Roofnet" # Roofnet/AboveNet
    edges, num_nodes, node_degrees = load_network_data(loaded_network_settings, network_type)
    # print(edges)
    # print(node_degrees)
    plot_degree_distribution(node_degrees)
    # Use a sorted tuple (smaller index first) as the key
    link_capacity_map = {tuple(sorted((edge[0], edge[1]))) : edge[2] for edge in edges}

    # Sort nodes by their degree, resulting in a list of tuples (node, degree)
    sorted_nodes_by_degree = sorted(node_degrees.items(), key=lambda x: x[1])
    # print(sorted_nodes_by_degree)
    # Select the top N nodes with the lowest degrees
    overlay_nodes = [node for node, degree in 
                     sorted_nodes_by_degree[:loaded_network_settings[network_type]["Number_of_overlay_nodes"]]]
    # with open('abovenet_overlay_nodes.pkl', 'wb') as file:
    #     pickle.dump(overlay_nodes, file)

    # Create the underlay network from the given edges
    underlay = create_underlay_network(num_nodes, edges)
    # Create a fully connected overlay network from a subset of nodes
    fully_connected_overlay, underlay_routing_map = create_fully_connected_overlay_network(underlay, overlay_nodes)
    # print(underlay_routing_map)
    # with open('roofnet_underlay_routing_map.pkl', 'wb') as file:
    #     pickle.dump(underlay_routing_map, file)
    # Example: Activate links using a random spanning tree algorithm
    activated_links = activate_links_random_spanning_tree(fully_connected_overlay)
    # print("Activated Links (Random Spanning Tree):", activated_links)
    # Alternatively, activate links forming a ring topology
    activated_links_ring = activate_links_ring_topology(fully_connected_overlay)
    # print("Activated Links (Ring Topology):", activated_links_ring)
    activated_links_prim = activate_links_prim_topology(fully_connected_overlay)
    # Draw the networks and their respective trees/topologies
    # draw_underlay_network_with_mst(underlay, mst)
    draw_fully_connected_overlay_network(fully_connected_overlay, overlay_nodes, activated_links_prim)

    # (11)
    # optimal_rho_tilde, mixing_matrix_random = optimize_K_mixing_matrix(fully_connected_overlay, activated_links)
    # optimal_rho_tilde, mixing_matrix_ring = optimize_K_mixing_matrix(fully_connected_overlay, activated_links_ring)
    # optimal_rho_tilde, mixing_matrix_clique = optimize_K_mixing_matrix(fully_connected_overlay, fully_connected_overlay.edges)
    # optimal_rho_tilde, mixing_matrix_prim = optimize_K_mixing_matrix(fully_connected_overlay, activated_links_prim)
    # mixing_matrices = {
    #     'mixing_matrix_random.pkl': mixing_matrix_random,
    #     'mixing_matrix_ring.pkl': mixing_matrix_ring,
    #     'mixing_matrix_clique.pkl': mixing_matrix_clique,
    #     'mixing_matrix_prim.pkl': mixing_matrix_prim,
    # }
    # for filename, matrix in mixing_matrices.items():
    #     with open(f'./mixing_matrix/{filename}', 'wb') as file:
    #         pickle.dump(matrix, file)
    # Save each mixing matrix in a separate file with a descriptive name
    # file_paths = ['/scratch2/tingyang/DFS_Topo/Ea/Roofnet_SDRRhoEw.pkl', "/scratch2/tingyang/DFS_Topo/Ea/Roofnet_BoydGreedy.pkl"]  # Add your file paths here
    # all_links = process_files(file_paths)
    # for file_path, Ea in zip(file_paths, all_links):
    #     # Extract base name without extension
    #     base_name = os.path.splitext(os.path.basename(file_path))[0]
    #     output_filename = f'mixing_matrix_{base_name}.pkl'
    #     _, mixing_matrix_random = optimize_K_mixing_matrix(fully_connected_overlay, Ea)
    #     output_path = os.path.join('./mixing_matrix', output_filename)
    #     with open(output_path, 'wb') as file:
    #         pickle.dump(mixing_matrix_random, file)


    # mixing_matrices = {
    #     'mixing_matrix_random.pkl': mixing_matrix_random,
    #     'mixing_matrix_ring.pkl': mixing_matrix_ring,
    #     'mixing_matrix_clique.pkl': mixing_matrix_clique,
    #     'mixing_matrix_prim.pkl': mixing_matrix_prim,
    # }
    # for filename, matrix in mixing_matrices.items():
    #     with open(filename, 'wb') as file:
    #         pickle.dump(matrix, file)
    # # print("Optimal rho_tilde from main:", optimal_rho_tilde)
    # # print("Mixing matrix from main:\n", mixing_matrix)
    data_size = loaded_network_settings["resnet"] * 64
    multicast_demands_clique = Ea_to_demand_model(fully_connected_overlay.edges, overlay_nodes, data_size)
    multicast_demands_ring = Ea_to_demand_model(activated_links_ring, overlay_nodes, data_size)
    multicast_demands_random = Ea_to_demand_model(activated_links, overlay_nodes, data_size)
    multicast_demands_prim = Ea_to_demand_model(activated_links_prim, overlay_nodes, data_size)
    # # # (5)
    tau_random = optimize_network_route_rate(fully_connected_overlay, multicast_demands_random, underlay, link_capacity_map)
    tau_ring = optimize_network_route_rate(fully_connected_overlay, multicast_demands_ring, underlay, link_capacity_map)
    tau_baseline = optimize_network_route_rate(fully_connected_overlay, multicast_demands_clique, underlay, link_capacity_map)
    tau_prim= optimize_network_route_rate(fully_connected_overlay, multicast_demands_prim, underlay, link_capacity_map)
    print("======", tau_random, tau_ring, tau_baseline, tau_prim)
    with open('tau_results.pkl', 'wb') as file:
        pickle.dump((tau_random, tau_ring, tau_baseline), file)


if __name__ == "__main__":
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    main()
