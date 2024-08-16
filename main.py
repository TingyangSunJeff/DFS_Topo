from network import create_underlay_network, find_shortest_path_with_delay
from optimization import optimize_network_route_rate, optimize_K_mixing_matrix, load_and_preprocess_data, d_psgd_training, optimize_network_route_rate_direct, metropolis_weights
from network import create_fully_connected_overlay_network, activate_links_prim_topology
from network import activate_links_random_spanning_tree, activate_links_ring_topology
from utils import plot_degree_distribution, draw_fully_connected_overlay_network, Ea_to_demand_model, load_network_data, plot_acc_loss_over_epochs, calculate_node_degrees
import pandas as pd
import pickle
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io

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

def process_files_and_generate_matrices(file_paths, fully_connected_overlay):
    """
    Process multiple files containing adjacency matrices, generate mixing matrices, 
    and save them with unique filenames.
    """
    Ea_diction = {}
    tau_diction = {}
    for file_idx, file_path in enumerate(file_paths):
        # Load the list of adjacency matrices from the pickle file
        with open(file_path, 'rb') as file:
            adjacency_matrices_dic = pickle.load(file)
        # print(adjacency_matrices_dic)
        adjacency_matrices = adjacency_matrices_dic["Ea"]
        tau_value_list = adjacency_matrices_dic["tau_list"]
        # print(adjacency_matrices)
        # Extract base name without extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Process each matrix in the list
        for matrix_idx, matrix in enumerate(adjacency_matrices):
            matrix = np.array(matrix)  # Ensure it's a NumPy array for processing
            links = process_adjacency_matrix(matrix)
            # Assuming optimize_K_mixing_matrix is defined elsewhere and takes 'links' as input
            # _, mixing_matrix_random = optimize_K_mixing_matrix(fully_connected_overlay, links)
            file_key = f"{base_name}_{matrix_idx + 1}"
            Ea_diction[file_key] = links
            if file_key == "IAB_MNIST_clique_1":
                file_key = "IAB_MNIST_clique"
            tau_diction[file_key] = tau_value_list[matrix_idx]
            # output_filename = f'mixing_matrix_{file_key}.pkl'  # Adjust index to start from 1
            # output_path = os.path.join('./mixing_matrix', output_filename)
            # with open(output_path, 'wb') as file:
            #     pickle.dump(mixing_matrix_random, file)

    return Ea_diction, tau_diction


def process_mat_files_and_generate_matrices(file_paths, fully_connected_overlay):
    """
    Process multiple MAT files containing adjacency matrices, generate mixing matrices, 
    and save them with unique filenames.
    """
    Ea_diction = {}
    tau_diction = {}

    for file_idx, file_path in enumerate(file_paths):
        # Load the MAT file
        mat_data = scipy.io.loadmat(file_path)

        # Extract Ea_dict and tau_dict from the loaded data
        Ea_dict = mat_data['Ea_dict']
        tau_dict = mat_data['tau_dict']
        
        # Extract base name without extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Process each matrix in Ea_dict and corresponding tau in tau_dict
        for matrix_idx in range(len(Ea_dict[0])):
            matrix = Ea_dict[0][matrix_idx]
            tau_value = tau_dict[0][matrix_idx]

            # Ensure the matrix is a NumPy array for processing
            matrix = np.array(matrix)

            # Assuming process_adjacency_matrix is defined elsewhere and takes 'matrix' as input
            links = process_adjacency_matrix(matrix)

            # Assuming optimize_K_mixing_matrix is defined elsewhere and takes 'links' as input
            _, mixing_matrix_random = optimize_K_mixing_matrix(fully_connected_overlay, links)

            # Generate a unique key for the dictionary
            file_key = f"{base_name}_{matrix_idx + 1}"

            # Store the links and tau_value in the dictionaries
            Ea_diction[file_key] = links
            tau_diction[file_key] = tau_value

            # Define the output filename and path
            output_filename = f'mixing_matrix_{file_key}.pkl'
            output_path = os.path.join('./mixing_matrix', output_filename)
            print("-------------", output_path)
            # Save the mixing matrix to a PKL file
            with open(output_path, 'wb') as file:
                pickle.dump(mixing_matrix_random, file)

    return Ea_diction, tau_diction

    
def main():
    with open('./network_settings.json', 'r') as json_file:
        loaded_network_settings = json.load(json_file)  
    # edges, num_nodes = load_network_data(ROOFNET_FILE_PATH)
    network_type = "Roofnet" # Roofnet/AboveNet/IAB
    dataset = "CIFAR10"
    edges, num_nodes, node_degrees = load_network_data(loaded_network_settings, network_type)
    # print(edges)
    print(node_degrees)
    plot_degree_distribution(node_degrees)
    # Use a sorted tuple (smaller index first) as the key
    link_capacity_map = {tuple(sorted((edge[0], edge[1]))) : edge[2] for edge in edges}

    # Sort nodes by their degree, resulting in a list of tuples (node, degree)
    sorted_nodes_by_degree = sorted(node_degrees.items(), key=lambda x: x[1])
    # print(sorted_nodes_by_degree)
    # Select the top N nodes with the lowest degrees
    overlay_nodes = []
    if network_type == "IAB":
        overlay_nodes = [3, 6, 7, 9, 10, 12, 13, 15, 16, 18]
        print("!!!!!!!!!!")
    else:
        overlay_nodes = [node for node, degree in 
                        sorted_nodes_by_degree[:loaded_network_settings[network_type]["Number_of_overlay_nodes"]]]
    # with open('abovenet_overlay_nodes.pkl', 'wb') as file:
    #     pickle.dump(overlay_nodes, file)
    print(overlay_nodes)
    # Create the underlay network from the given edges
    underlay = create_underlay_network(num_nodes, edges)
    # print(underlay.edges)
    # Create a fully connected overlay network from a subset of nodes
    fully_connected_overlay = create_fully_connected_overlay_network(underlay, overlay_nodes, network_type)
    print(fully_connected_overlay.edges)
    # with open('roofnet_underlay_routing_map.pkl', 'wb') as file:
    #     pickle.dump(underlay_routing_map, file)
    # Example: Activate links using a random spanning tree algorithm

    # # (11)

    # activated_links = activate_links_random_spanning_tree(fully_connected_overlay)
    activated_links_ring = activate_links_ring_topology(fully_connected_overlay)
    activated_links_prim = activate_links_prim_topology(fully_connected_overlay)
    # print("~~~~~~~~~~~~~", activated_links_prim)
    # with open(f'./Ea/{network_type}_ring.pkl', 'wb') as file:
    #     pickle.dump(activated_links_ring, file)
    # with open(f'./Ea/{network_type}_prim.pkl', 'wb') as file:
    #     pickle.dump(activated_links_prim, file)
    # with open(f'./Ea/{network_type}_clique.pkl', 'wb') as file:
    #     pickle.dump(list(fully_connected_overlay.edges), file)
    # optimal_rho_tilde, mixing_matrix_ring = optimize_K_mixing_matrix(fully_connected_overlay, activated_links_ring)
    # optimal_rho_tilde, mixing_matrix_clique = optimize_K_mixing_matrix(fully_connected_overlay, list(fully_connected_overlay.edges))
    # optimal_rho_tilde, mixing_matrix_prim = optimize_K_mixing_matrix(fully_connected_overlay, activated_links_prim)
    # mixing_matrices = {
    #     # 'random.pkl': mixing_matrix_random,
    #     'ring.pkl': mixing_matrix_ring,
    #     'clique.pkl': mixing_matrix_clique,
    #     'prim.pkl': mixing_matrix_prim,
    # }
    # for filename, matrix in mixing_matrices.items():
    #     with open(f'./mixing_matrix/mixing_matrix_{network_type}_MNIST_{filename}', 'wb') as file:
    #         pickle.dump(matrix, file)
    # # Save each mixing matrix in a separate file with a descriptive name

    # load proposed Ea
    file_paths = [
        "/scratch2/tingyang/DFS_Topo/Ea/Roofnet_CIFAR10_SDRRhoEw.pkl",
        "/scratch2/tingyang/DFS_Topo/Ea/Roofnet_CIFAR10_SDRLambda2Ew.pkl",
        "/scratch2/tingyang/DFS_Topo/Ea/Roofnet_CIFAR10_SCA23.pkl",
        "/scratch2/tingyang/DFS_Topo/Ea/Roofnet_CIFAR10_BoydGreedy.pkl",
    ]

    Ea_diction, tau_diction = process_files_and_generate_matrices(file_paths, fully_connected_overlay)
    # metropolis_weights()


    # Ea_diction, tau_diction = process_mat_files_and_generate_matrices(file_paths, fully_connected_overlay)
    # print(Ea_diction)
    # tau_diction["IAB_MNIST_prim"] = 8192
    # tau_diction["IAB_MNIST_ring"] = 8192
    # print(fully_connected_overlay)
    # Ea_diction ={}
    benchmark_list = [f"{network_type}_{dataset}_ring", f"{network_type}_{dataset}_clique", f"{network_type}_{dataset}_prim"]
    
    for key in benchmark_list:
        if key == f"{network_type}_{dataset}_clique":
            Ea_diction[key] = [(int(u), int(v)) for edge in fully_connected_overlay.edges for (u, v) in (edge, edge[::-1])]
        elif key == f"{network_type}_{dataset}_ring":
            Ea_diction[key] = activated_links_ring
        elif key == f"{network_type}_{dataset}_prim":
            Ea_diction[key] = activated_links_prim
    # print(Ea_diction)

    # Calculate the mixing matrices
    mixing_matrices = {}

    for topo_name, active_edges in Ea_diction.items():
        Ea_node_degrees = calculate_node_degrees(active_edges)
        # print(Ea_node_degrees)
        mixing_matrices[topo_name] = metropolis_weights(overlay_nodes, active_edges, Ea_node_degrees)

    # Print the mixing matrices
    for topo_name, matrix in mixing_matrices.items():
        print(f"Mixing Matrix for {topo_name}:")
        for row in matrix:
            print([round(val, 3) for val in row])
        print()

    # Draw the networks and their respective trees/topologies
    # draw_underlay_network_with_mst(underlay, mst)
    # draw_fully_connected_overlay_network(fully_connected_overlay, overlay_nodes, activated_links_prim)

    for filename, matrix in mixing_matrices.items():
        with open(f"mixing_matrix_{filename}.pkl", 'wb') as file:
            pickle.dump(matrix, file)
    # # print("Optimal rho_tilde from main:", optimal_rho_tilde)
    # # print("Mixing matrix from main:\n", mixing_matrix)

    # # # # (5)
    # output_dic = {} # topo : tau
    # data_size = loaded_network_settings["mnist"] * 32 # unit:bit tensorflow use float32 datatype
    # for key, Ea in Ea_diction.items():
    #     multicast_demands = Ea_to_demand_model(Ea, overlay_nodes, data_size)
    #     # tau = optimize_network_route_rate(fully_connected_overlay, multicast_demands, underlay, link_capacity_map)
    #     tau = optimize_network_route_rate_direct(fully_connected_overlay, multicast_demands, underlay, link_capacity_map)
    #     output_dic[key] = tau # unit:seconds
    # print(output_dic)


if __name__ == "__main__":
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         print(e)
    main()
