import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

# with open(f"/scratch2/tingyang/DFS_Topo/tau_results.pkl", 'rb') as file:
#     time_per_epoch_diction = pickle.load(file)
dic_tau = {
    "Greedy" : 4.7253*1e3/ (4.9152* 1e3),
    "SCA" : 1.6063/1.6384,
    'Relaxation-rho': 4.7261/4.9152,
    "Relaxation-lambda": 6.3640/6.5536,
    "Prim": 3.1801/3.2768,
    "Ring": 3.1814/3.2768,
    "Clique": 10.3371/14.746
}
propotion = 4.7253*1e3/ (4.9152* 1e3)
print("=====", dic_tau )