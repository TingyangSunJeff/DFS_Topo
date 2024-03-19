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