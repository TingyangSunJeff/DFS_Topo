import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the file paths
file_paths = [
    "/scratch2/tingyang/DFS_Topo/Ea/Roofnet_CIFAR10_SCA23.pkl"
]

with open(file_paths[0], 'rb') as file:
    metrics_history = pickle.load(file)

print(metrics_history)