# utils/graph_utils.py
import networkx as nx
from config import OVERLAY_NODES

def calculate_path_delays(network, weight='delay'):
    """Calculate the shortest path delays between all pairs of nodes in a network."""
    path_delay_map = {}
    for source in network.nodes():
        for target in network.nodes():
            if source != target:
                path_length = nx.shortest_path_length(network, source=source, target=target, weight=weight)
                path_delay_map[(source, target)] = path_length
    return path_delay_map

def define_multicast_demands(overlay_network, source_node, destinations, data_size):
    """Define multicast demands for a given source and set of destination nodes."""
    return {'source': source_node, 'destinations': destinations, 'data_size': data_size}

def calculate_path_delays(mst, weight='delay'):
    path_delay_map = {}
    for source in mst.nodes():
        for target in mst.nodes():
            if source != target:
                # Use the weight parameter to consider the delay attribute on the edges
                path_delay = nx.shortest_path_length(mst, source, target, weight=weight)
                path_delay_map[(source, target)] = path_delay
    return path_delay_map

def Ea_to_demand_model(activated_links, data_size):
    multicast_demands = []
    for node in OVERLAY_NODES:
        destinations = {v for u, v in activated_links if u == node} | {u for u, v in activated_links if v == node}
        multicast_demand = (node, destinations, data_size)
        multicast_demands.append(multicast_demand)
    return multicast_demands

def map_overlay_to_mst_edges(mst, overlay_links):
    # Initialize a dictionary to hold the list of demands for each edge in the MST
    # mst network
    # demand set H
    overlay_links_map = {(u, v): [] for u, v in mst.edges()}

    # Iterate over each demand
    for source, destination in overlay_links:
        # Find the path in the MST from source to destination
        path = nx.shortest_path(mst, source=source, target=destination)

        # Go through the path and add the demand to each edge on the path
        for i in range(len(path) - 1):
            edge = tuple(sorted((path[i], path[i+1])))
            # For undirected graphs, we need to consider that the edge might be in either direction
            if edge in overlay_links_map:
                overlay_links_map[edge].append((source, destination))

    return overlay_links_map

