from network import create_underlay_network, find_shortest_path_with_delay
from optimization import optimize_network_route_rate, optimize_K_mixing_matrix, load_and_preprocess_data, d_psgd_training
from network import create_fully_connected_overlay_network
from network import minimum_spanning_tree, activate_links_random_spanning_tree, activate_links_ring_topology
from utils import plot_degree_distribution, draw_fully_connected_overlay_network, Ea_to_demand_model, load_network_data, plot_acc_loss_over_epochs
import pandas as pd
import pickle
import json
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    with open('./network_settings.json', 'r') as json_file:
        loaded_network_settings = json.load(json_file)  
    # edges, num_nodes = load_network_data(ROOFNET_FILE_PATH)
    network_type = "Roofnet" # Roofnet/AboveNet
    edges, num_nodes, node_degrees = load_network_data(loaded_network_settings, network_type)

    plot_degree_distribution(node_degrees)
    # Use a sorted tuple (smaller index first) as the key
    link_capacity_map = {tuple(sorted((edge[0], edge[1]))) : edge[2] for edge in edges}

    # Sort nodes by their degree, resulting in a list of tuples (node, degree)
    sorted_nodes_by_degree = sorted(node_degrees.items(), key=lambda x: x[1])
    
    # Select the top N nodes with the lowest degrees
    overlay_nodes = [node for node, degree in 
                     sorted_nodes_by_degree[:loaded_network_settings[network_type]["Number_of_overlay_nodes"]]]
    # Create the underlay network from the given edges
    underlay = create_underlay_network(num_nodes, edges)
    # Create a fully connected overlay network from a subset of nodes
    fully_connected_overlay = create_fully_connected_overlay_network(underlay, overlay_nodes)

    # Example: Activate links using a random spanning tree algorithm
    activated_links = activate_links_random_spanning_tree(fully_connected_overlay)
    # print("Activated Links (Random Spanning Tree):", activated_links)
    # Alternatively, activate links forming a ring topology
    activated_links_ring = activate_links_ring_topology(fully_connected_overlay)
    # print("Activated Links (Ring Topology):", activated_links_ring)

    # Draw the networks and their respective trees/topologies
    # draw_underlay_network_with_mst(underlay, mst)
    draw_fully_connected_overlay_network(fully_connected_overlay, overlay_nodes, activated_links)

    # (11)
    optimal_rho_tilde, mixing_matrix_random = optimize_K_mixing_matrix(fully_connected_overlay, activated_links)
    optimal_rho_tilde, mixing_matrix_ring = optimize_K_mixing_matrix(fully_connected_overlay, activated_links_ring)
    optimal_rho_tilde, mixing_matrix_clique = optimize_K_mixing_matrix(fully_connected_overlay, fully_connected_overlay.edges)

    # print("Optimal rho_tilde from main:", optimal_rho_tilde)
    # print("Mixing matrix from main:\n", mixing_matrix)
    multicast_demands_clique = Ea_to_demand_model(fully_connected_overlay.edges, overlay_nodes)
    multicast_demands_ring = Ea_to_demand_model(activated_links_ring, overlay_nodes)
    multicast_demands_random = Ea_to_demand_model(activated_links, overlay_nodes)
    # # (5)
    # tau_random = optimize_network_route_rate(fully_connected_overlay, multicast_demands_random, underlay, link_capacity_map)
    # tau_ring = optimize_network_route_rate(fully_connected_overlay, multicast_demands_ring, underlay, link_capacity_map)
    # tau_baseline = optimize_network_route_rate(fully_connected_overlay, multicast_demands_clique, underlay, link_capacity_map)
    # print("======", tau_random, tau_ring, tau_baseline)
    # with open('tau_results.pkl', 'wb') as file:
    #     pickle.dump((tau_random, tau_ring, tau_baseline), file)

    # D-PSGD
    # Load data
    dataset_flag = "mnist"
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data(dataset_flag)
    # Run training
    loss_history, val_accuracies = d_psgd_training(x_train, y_train, x_test, y_test,
                                                    mixing_matrix_random, 
                                                    loaded_network_settings[network_type]["Number_of_overlay_nodes"], 
                                                    dataset_flag, epochs=100)
    # Save to file
    with open('model_metrics.pkl', 'wb') as file:
        pickle.dump((loss_history, val_accuracies), file)
    print("Data saved to 'model_metrics.pkl'.")
    # plot_acc_loss_over_epochs(loss_history, val_accuracies)

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
