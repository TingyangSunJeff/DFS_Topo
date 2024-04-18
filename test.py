import networkx as nx

def read_underlay_routing_map(file_path):
    with open(file_path, 'r') as file:
        underlay_routing_map = file.read()
    return underlay_routing_map

def find_shortest_path_with_delay(source_node, dest_node, underlay_graph=None):
    # Parse the underlay routing map
    with open("/scratch2/tingyang/DFS_Topo/IAB_underlay_routing_map.txt", 'r') as file:
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
                print(u, v)
                total_delay += underlay_graph[u][v]['delay']  # Assuming delay is stored as an edge attribute
            return shortest_path, total_delay
        else:
            return shortest_path, None
    else:
        return None, None  # No path found

# Example usage:
underlay_routing_map_file = "underlay_routing_map.txt"
underlay_graph = nx.Graph()  # Placeholder for NetworkX graph, you can populate this as needed

# Assuming underlay_graph is populated with nodes and edges including their respective delays
underlay_graph.add_edge(15, 13, delay=5)
underlay_graph.add_edge(13, 4, delay=3)
underlay_graph.add_edge(4, 1, delay=2)
underlay_graph.add_edge(1, 7, delay=4)
underlay_graph.add_edge(7, 5, delay=6)
underlay_graph.add_edge(5, 6, delay=7)

source_node = 6
dest_node = 15

path, delay = find_shortest_path_with_delay(source_node, dest_node, underlay_graph)
print("Shortest path:", path)
print("Path delay:", delay)
