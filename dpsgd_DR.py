import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the file paths
file_paths = [(1, 2), (1, 10), (1, 15), (1, 17), (2, 1), (2, 9), (2, 17), (8, 9), (8, 10), (8, 12), (8, 16), (9, 2), (9, 8), (9, 11), (9, 16), (10, 1), (10, 8), (10, 11), (11, 9), (11, 10), (11, 12), (12, 8), (12, 11), (12, 15), (12, 17), (15, 1), (15, 12), (16, 8), (16, 9), (16, 17), (17, 1), (17, 2), (17, 12), (17, 16)]


overlay_links = []
# Populate the undirected_links list
for link in file_paths:
    overlay_links.append(link)  # Add original direction
    overlay_links.append(link[::-1])  # Add reverse direction

print(overlay_links)

# with open(file_paths[0], 'rb') as file:
#     metrics_history = pickle.load(file)

# print(metrics_history)