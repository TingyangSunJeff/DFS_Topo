import numpy as np
import pandas as pd

def adjust_matrix(matrix):
    adjusted_matrix = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if abs(matrix[i, j]) >= 0.1 and abs(matrix[i, j]) < 1:
                adjusted_matrix[i, j] = matrix[i, j]
    
    # Normalize each row
    row_sums = np.sum(adjusted_matrix, axis=1)
    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            adjusted_matrix[i, :] /= row_sums[i]
    
    # Normalize each column
    col_sums = np.sum(adjusted_matrix, axis=0)
    for j in range(len(col_sums)):
        if col_sums[j] > 0:
            adjusted_matrix[:, j] /= col_sums[j]
    
    return adjusted_matrix

def load_network_data(loaded_network_settings, network_type):
    """
    Load network data from a file, adjust node numbering, and prepare edge data.
    This function supports different formats such as Roofnet and Abovenet and also returns the number of nodes.

    Parameters:
    - file_path (str): Path to the network data file.
    - file_type (str): Type of the network data ('roofnet' or 'abovenet').

    Returns:
    - tuple:
        - edges (list of tuples): Each tuple represents an edge with the following elements:
            - u (int): Adjusted node1 index.
            - v (int): Adjusted node2 index.
            - capacity (int or float): Edge capacity.
            - delay (float): Edge delay.
        - num_nodes (int): Number of nodes in the network.
    """
    edges = []
    num_nodes = 0
    node_degrees = {}
    file_path = loaded_network_settings[network_type]["file_path"]
    if network_type == 'Roofnet':
        data = pd.read_csv(file_path, sep='\s+')
        # Assuming the highest node index represents the number of nodes
        num_nodes = max(data['node1'].max(), data['node2'].max())
        for index, row in data.iterrows():
            u = row['node1'] - 1
            v = row['node2'] - 1
            capacity = 1000  # Assuming capacity as 0.001 Gbps for Roofnet
            delay = 0.5 # 5.e-04 ms 
            edges.append((u, v, capacity, delay))
            edges.append((v, u, capacity, delay))
            for node in [u, v]:
                node_degrees[node] = node_degrees.get(node, 0) + 1

    elif network_type == "AboveNet":
        with open(file_path, 'r') as file:
            lines = file.readlines()
            edge_section = False
            for line in lines:
                if line.startswith('NODES'):
                    num_nodes = int(line.split()[1])
                    continue
                if line.startswith('EDGES'):
                    edge_section = True
                    continue
                if edge_section and not line.startswith('label'):
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue  # Skip lines that don't have enough data
                    u = int(parts[1])  # src
                    v = int(parts[2])  # dest
                    weight = int(parts[4])  # capacity 1 Gbps
                    delay = float(parts[5])  # prop delay  [0.1, 13.8] ms
                    edges.append((u, v, weight, delay))
                    # For directed graphs, increment out-degree of u and in-degree of v
                    node_degrees[u] = node_degrees.get(u, 0) + 1
                    node_degrees[v] = node_degrees.get(v, 0) + 1

    return edges, num_nodes, node_degrees

