import pickle
import os

# List of file paths
file_paths = [
    "/scratch2/tingyang/DFS_Topo/Ea/Roofnet_CIFAR10_SDRRhoEw.pkl",
    "/scratch2/tingyang/DFS_Topo/Ea/Roofnet_CIFAR10_SDRLambda2Ew.pkl",
    "/scratch2/tingyang/DFS_Topo/Ea/Roofnet_CIFAR10_SCA23.pkl",
    "/scratch2/tingyang/DFS_Topo/Ea/Roofnet_CIFAR10_BoydGreedy.pkl"
]
# {'ring': 4,532,774.55340644, 'RST': 6,043,696.83337543, 'baseline': 15,109,245.464599185, 'prim': 4,532,772.784910211}
# Loop through each file
for file_path in file_paths:
    # Load the contents of the file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)    
        print(data)
        # Extract the tau value
        # tau = data.get('tau')
        
        # Print the file name and tau value
        # print(f"File: {os.path.basename(file_path)} - Tau: {tau}")
