# config.py

# Path to the Roofnet data file
PROJECT_PATH = "/scratch2/tingyang/DFS_Topo/"
ROOFNET_FILE_PATH = '/scratch2/tingyang/DFS_Topo/Roofnet_AvgSuccessRate_1Mbps.txt'
ABOVENET_FILE_PATH = '/scratch2/tingyang/DFS_Topo/Abvt.graph'

# Network configuration
NUM_NODES = 10  # Total number of nodes in the underlay network
OVERLAY_NODES = [0, 8, 15, 20, 31]  # Example subset of nodes for the overlay network

# Example network edges: (node1, node2, capacity, delay)
EDGES = [
    (0, 1, 10, 50), (1, 2, 15, 30), (2, 3, 10, 5), (3, 4, 20, 15),
    (4, 5, 10, 5), (5, 6, 15, 10), (6, 7, 10, 50), (7, 8, 20, 15),
    (8, 9, 10, 5), (0, 3, 25, 20), (3, 6, 30, 25), (6, 9, 15, 10),
    (1, 7, 40, 30), (2, 8, 35, 25), (0, 2, 20, 15), (1, 3, 30, 20),
    (4, 6, 25, 20), (5, 7, 35, 25), (7, 9, 45, 30), (8, 0, 50, 35)
]

multicast_demands = [(0, {1, 2}, 100), (4, {2}, 100)] # determined by the mixing matrix

# Additional configurations can go here, such as:
# - Visualization settings
# - Algorithm-specific parameters (e.g., weights for calculating edge importance)
# - External service configurations if your project interacts with APIs or databases
