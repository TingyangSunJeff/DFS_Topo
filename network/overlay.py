import networkx as nx
from network import find_shortest_path_with_delay

def create_overlay_network(underlay, overlay_nodes):
    # Create an empty graph
    overlay = nx.Graph()

    # Add nodes to overlay, ensuring they are also in underlay
    for node in overlay_nodes:
        if node in underlay:
            overlay.add_node(node)

    # Add overlay edges
    # Overlay edges should map to paths in the underlay network
    overlay.add_edge(0, 2)  # This represents an overlay link
    # ... more overlay links based on your design

    return overlay

def create_fully_connected_overlay_network(underlay, overlay_nodes):
    # Create an empty graph for the overlay network
    overlay = nx.Graph()

    # Add nodes to overlay, ensuring they are also in underlay
    for node in overlay_nodes:
        if node in underlay:
            overlay.add_node(node)
    underlay_routing_map = {}
    # Initialize as a fully connected network
    for i in range(len(overlay_nodes)):
        for j in range(i + 1, len(overlay_nodes)):
            # find_shortest_path_with_delay(underlay,overlay_nodes[1],overlay_nodes[17])
            path, path_delay = find_shortest_path_with_delay(underlay,overlay_nodes[i],overlay_nodes[j])
            underlay_routing_map[f"{overlay_nodes[i]}_{overlay_nodes[j]}"] = path
            # path_reverse, path_delay_reverse = find_shortest_path_with_delay(underlay,overlay_nodes[j],overlay_nodes[i])
            # print(path, path_delay)
            # print(path_reverse, path_delay_reverse)
            overlay.add_edge(overlay_nodes[i], overlay_nodes[j], delay=path_delay, underlay_path=path)

    # At this stage, all overlay links are potential and need to be activated based on the design algorithm
    return overlay, underlay_routing_map