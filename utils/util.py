import numpy as np
import pandas as pd
import os

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
    file_name = loaded_network_settings[network_type]["file_path"]
    file_path = os.path.join(os.getcwd(), file_name)
    if network_type == 'Roofnet':
        data = pd.read_csv(file_path, sep='\s+')
        # Assuming the highest node index represents the number of nodes
        num_nodes = max(data['node1'].max(), data['node2'].max())
        for index, row in data.iterrows():
            u = row['node1'] - 1
            v = row['node2'] - 1
            capacity = 1000000  # capacity as 0.001 Gbps / 1000000 bits per s
            delay = 5.e-07 # unit: s  5.e-04 ms 
            edges.append((u, v, capacity, delay))
            edges.append((v, u, capacity, delay))
            for node in [u, v]:
                node_degrees[node] = node_degrees.get(node, 0) + 1
    elif network_type == 'IAB':
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
                    weight = float(parts[4]) * 1.e+9  # capacity 0.4 Gbps / 1000000
                    delay = float(parts[5]) * 0.001  # prop delay  0.0005 ms
                    edges.append((u, v, weight, delay))
                    # For directed graphs, increment out-degree of u and in-degree of v
                    node_degrees[u] = node_degrees.get(u, 0) + 1
                    node_degrees[v] = node_degrees.get(v, 0) + 1
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
                    weight = int(parts[4])  # capacity 1 Gbps / e+9
                    delay = float(parts[5])  # prop delay  [0.1, 13.8] ms
                    edges.append((u, v, weight, delay))
                    # For directed graphs, increment out-degree of u and in-degree of v
                    node_degrees[u] = node_degrees.get(u, 0) + 1
                    node_degrees[v] = node_degrees.get(v, 0) + 1

    return edges, num_nodes, node_degrees


def calculate_node_degrees(edge_list):
    """
    Calculate the node degrees for a given list of edges.

    Parameters:
    - edge_list: A list of tuples representing the edges in the graph.

    Returns:
    - node_degrees: A dictionary with node indices as keys and their degrees as values.
    """
    node_degrees = {}

    for edge in edge_list:
        node1, node2 = edge

        # Increment degree for both nodes in the edge
        if node1 in node_degrees:
            node_degrees[node1] += 1
        else:
            node_degrees[node1] = 1

        if node2 in node_degrees:
            node_degrees[node2] += 1
        else:
            node_degrees[node2] = 1

    return node_degrees


def check_convergence(accuracies, window_size=3, variance_threshold=0.01, stable_epochs=3):
    """
    Check if the training has converged based on a sliding window approach.

    Parameters:
    - accuracies: List of accuracy values per epoch.
    - window_size: The number of epochs to consider in the sliding window.
    - variance_threshold: The variance threshold to define convergence.
    - stable_epochs: Number of consecutive epochs the variance should remain below the threshold to declare convergence.

    Returns:
    - converged: Boolean indicating if convergence has been achieved.
    - convergence_epoch: The epoch at which convergence was achieved.
    """
    num_epochs = len(accuracies)
    
    for start_epoch in range(num_epochs - window_size + 1):
        window = accuracies[start_epoch:start_epoch + window_size]
        window_variance = np.var(window)

        if window_variance < variance_threshold:
            # Check if this stability holds for the next `stable_epochs` windows
            stable_count = 0
            for i in range(start_epoch, min(start_epoch + stable_epochs, num_epochs - window_size + 1)):
                next_window = accuracies[i:i + window_size]
                next_window_variance = np.var(next_window)
                if next_window_variance < variance_threshold:
                    stable_count += 1
                else:
                    break

            if stable_count >= stable_epochs:
                return True, start_epoch + window_size

    return False, None

# # Example accuracy data
# accuracies = [0.80, 0.85, 0.87, 0.91, 0.91, 0.92, 0.92, 0.93, 0.93, 0.93, 0.94, 0.94, 0.95]

# converged, convergence_epoch = check_convergence(accuracies, window_size=3, variance_threshold=0.0005, stable_epochs=3)

# if converged:
#     print(f"Convergence achieved at epoch {convergence_epoch}.")
# else:
#     print("Convergence not achieved.")
