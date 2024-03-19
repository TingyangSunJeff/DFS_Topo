# network/algorithms.py
import networkx as nx
import random
import heapq

def minimum_spanning_tree(underlay):
    """
    Compute the minimum spanning tree of the underlay network.
    """
    mst = nx.minimum_spanning_tree(underlay, weight='delay')
    return mst

def activate_links_random_spanning_tree(overlay_network):
    """
    Generate a random spanning tree from the fully connected overlay network.
    This represents the activated overlay links.
    """
    activated_links = set()
    nodes = list(overlay_network.nodes())
    random.shuffle(nodes)  # Shuffle to start from a random node

    visited = {nodes[0]}
    while len(visited) < len(nodes):
        new_edge = random.choice([edge for edge in overlay_network.edges() if (edge[0] in visited) ^ (edge[1] in visited)])
        activated_links.add(tuple(sorted(new_edge)))
        visited.update(new_edge)

    return activated_links

def activate_links_ring_topology(overlay_network):
    """
    Generate a ring topology from the fully connected overlay network.
    This forms the activated overlay links in a ring structure.
    """
    activated_links = set()
    nodes = list(overlay_network.nodes())

    for i in range(len(nodes)):
        next_node = nodes[(i + 1) % len(nodes)]  # Loop back to the first node
        activated_links.add((nodes[i], next_node))

    return activated_links

def activate_links_prim_topology(overlay_network):
    """
    Generate a topology based on Prim's algorithm from the overlay network.
    This configuration represents the activated overlay links forming a minimum spanning tree.

    :param overlay_network: A weighted overlay network.
    :return: A set of activated links (edges) in the overlay network forming a minimum spanning tree.
    """
    activated_links = set()

    # Choose an arbitrary starting node
    start_node = next(iter(overlay_network.nodes()))
    visited = set([start_node])

    # Create a priority queue and add all edges from the start node to the queue
    edges = [(weight, start_node, to) for to, weight in overlay_network[start_node].items()]
    heapq.heapify(edges)

    while edges:
        weight, from_node, to_node = heapq.heappop(edges)
        if to_node not in visited:
            visited.add(to_node)
            activated_links.add((from_node, to_node))

            # Add all edges from the new node to the queue
            for next_node, next_weight in overlay_network[to_node].items():
                if next_node not in visited:
                    heapq.heappush(edges, (next_weight, to_node, next_node))

    return activated_links
# Additional algorithms can be added here as needed, for example:
# - Greedy algorithms for link activation based on specific criteria
# - Algorithms for network flow optimization
# - Any custom algorithm relevant to your network design or optimization tasks

