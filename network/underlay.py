import networkx as nx

def create_underlay_network(num_nodes, edges):
    # Create an empty graph
    underlay = nx.Graph()

    # Add more nodes
    for i in range(num_nodes):  # Increasing the number of nodes to 10
        underlay.add_node(i)

    # Add more edges with capacity and delay as attributes

    for u, v, capacity, delay in edges:
        underlay.add_edge(u, v, capacity=capacity, delay=delay)


    return underlay

def find_shortest_path_with_delay(underlay_graph, source_node, dest_node, network_type, weight='delay'):
    """
    Find the shortest path in a graph with delays on each link using Dijkstra's algorithm.

    Parameters:
    - graph: The networkx graph instance.
    - source: The source node for the path.
    - target: The target node for the path.
    - weight: The edge attribute that holds the delay (weight) of the link.

    Returns:
    A list of nodes that represents the shortest path from the source to the target.
    """
    # Parse the underlay routing map
    source_node = int(source_node)
    dest_node = int(dest_node)
    with open(f"./{network_type}_underlay_routing_map.txt", 'r') as file:
        underlay_routing_map = file.read()
    routing_map = {}
    for line in underlay_routing_map.split('\n'):
        if line.strip():  # Skip empty lines
            parts = line.split(': ')
            source_end, dest_end = map(int, parts[0].split())
            path = list(map(int, parts[2].split()))
            routing_map[(source_end, dest_end)] = path
    # Check if there is a path between the given source and destination nodes
    if (source_node, dest_node) in routing_map:
        shortest_path = routing_map[(source_node, dest_node)]
        
        if underlay_graph:
            # Compute path delay using underlay_graph
            total_delay = 0
            for u, v in zip(shortest_path[:-1], shortest_path[1:]):
                total_delay += underlay_graph[u][v][weight]  # Assuming delay is stored as an edge attribute
            return shortest_path, total_delay
        else:
            return shortest_path, None
    else:
        return None, None  # No path found