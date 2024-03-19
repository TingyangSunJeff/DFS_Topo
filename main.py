from network import create_underlay_network
from optimization import optimize_network_route_rate, optimize_K_mixing_matrix, load_and_preprocess_data, d_psgd_training
from network import create_fully_connected_overlay_network
from network import minimum_spanning_tree, activate_links_random_spanning_tree, activate_links_ring_topology
from utils import draw_underlay_network_with_mst, draw_fully_connected_overlay_network, Ea_to_demand_model, calculate_path_delays, map_overlay_to_mst_edges, plot_acc_loss_over_epochs
from config import OVERLAY_NODES, ROOFNET_FILE_PATH
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load the Roofnet data from the uploaded file
    roofnet_data = pd.read_csv(ROOFNET_FILE_PATH, delim_whitespace=True)

    edges = []
    for index, row in roofnet_data.iterrows():
        u = row['node1'] - 1  # Adjusting node numbering to start from 0
        v = row['node2'] - 1
        success_rate = row['success_rate']
        capacity = success_rate * 1000  # Example transformation
        delay = 1 / success_rate  # Example inverse relationship
        edges.append((u, v, capacity, delay))
    # Use a sorted tuple (smaller index first) as the key
    link_capacity_map = {tuple(sorted((edge[0], edge[1]))) : edge[2] for edge in edges}
    # Number of nodes can be determined as the unique nodes in the dataset
    num_nodes = len(set(roofnet_data['node1'].unique()).union(set(roofnet_data['node2'].unique())))

    # Create the underlay network from the given edges
    underlay = create_underlay_network(num_nodes, edges)

    # Compute the minimum spanning tree for the underlay network
    mst = minimum_spanning_tree(underlay)
    link_delay_map = calculate_path_delays(mst)
    # print("Minimum Spanning Tree Delays:", link_delay_map)

    # Create a fully connected overlay network from a subset of nodes
    fully_connected_overlay = create_fully_connected_overlay_network(underlay, OVERLAY_NODES)

    # Example: Activate links using a random spanning tree algorithm
    activated_links = activate_links_random_spanning_tree(fully_connected_overlay)
    # print("Activated Links (Random Spanning Tree):", activated_links)

    # Alternatively, activate links forming a ring topology
    activated_links_ring = activate_links_ring_topology(fully_connected_overlay)
    # print("Activated Links (Ring Topology):", activated_links_ring)

    # Draw the networks and their respective trees/topologies
    draw_underlay_network_with_mst(underlay, mst)
    draw_fully_connected_overlay_network(fully_connected_overlay, OVERLAY_NODES, activated_links)

    # (11)
    optimal_rho_tilde, mixing_matrix = optimize_K_mixing_matrix(fully_connected_overlay, activated_links)

    # print("Optimal rho_tilde from main:", optimal_rho_tilde)
    # print("Mixing matrix from main:\n", mixing_matrix)

    multicast_demands = Ea_to_demand_model(activated_links, 100)

    H=[]
    # Initialize an empty list to hold both original and reversed links
    Ea = []
    overlay_directed_links = []
    mst_edges = list(mst.edges(data=True))
    # Populate the undirected_links list
    for link in fully_connected_overlay.edges():
        overlay_directed_links.append(link)  # Add original direction
        overlay_directed_links.append(link[::-1])  # Add reverse direction

    # Populate the undirected_links list
    for link in list(activated_links):
        Ea.append(link)  # Add original direction
        Ea.append(link[::-1])  # Add reverse direction

    for source, destinations, data_size in multicast_demands:
        for destination in destinations:
            H.append((source, destination, data_size))
    overlay_mst_edges_map = map_overlay_to_mst_edges(mst, overlay_directed_links)
    # (5)
    z,f,d,r,tau = optimize_network_route_rate(H, Ea, mst_edges, overlay_directed_links, multicast_demands, link_delay_map, overlay_mst_edges_map, link_capacity_map)

    # D-PSGD 
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    # Run training
    lost_hitory, val_accuracies = d_psgd_training(x_train, y_train, x_test, y_test, mixing_matrix)
    plot_acc_loss_over_epochs(lost_hitory, val_accuracies)
if __name__ == "__main__":
    main()
