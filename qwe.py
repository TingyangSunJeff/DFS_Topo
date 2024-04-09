import pickle
import os

# List of file paths
file_paths = [
    "/scratch2/tingyang/DFS_Topo/Ea/Roofnet_CIFAR10_SDRRhoEw.pkl",
    "/scratch2/tingyang/DFS_Topo/Ea/Roofnet_CIFAR10_SDRLambda2Ew.pkl",
    "/scratch2/tingyang/DFS_Topo/Ea/Roofnet_CIFAR10_SCA23.pkl",
    "/scratch2/tingyang/DFS_Topo/Ea/Roofnet_CIFAR10_BoydGreedy.pkl"
]

# Loop through each file
for file_path in file_paths:
    # Load the contents of the file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)    
        print(data)
        # Extract the tau value
        tau = data.get('tau')
        
        # Print the file name and tau value
        print(f"File: {os.path.basename(file_path)} - Tau: {tau}")
