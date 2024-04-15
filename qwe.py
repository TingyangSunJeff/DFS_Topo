import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the file paths
file_paths = [
    "/scratch2/tingyang/DFS_Topo/result_for_cnn_Roofnet_MNIST_prim.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_cnn_Roofnet_MNIST_clique.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_cnn_Roofnet_MNIST_BoydGreedy_1.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_cnn_Roofnet_MNIST_ring.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_cnn_Roofnet_MNIST_SCA23_1.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_cnn_Roofnet_MNIST_SDRLambda2Ew_2.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_cnn_Roofnet_MNIST_SDRRhoEw_1.pkl",
]

categorized_results = {
    'SCA': 'Roofnet_MNIST_SCA23_1',
    'Relaxation-lambda': 'Roofnet_MNIST_SDRLambda2Ew_2',
    'Relaxation-rho': 'Roofnet_MNIST_SDRRhoEw_1',
    'Greedy': 'Roofnet_MNIST_BoydGreedy_1',
    'Ring': 'Roofnet_MNIST_ring',
    'Clique': 'Roofnet_MNIST_clique',
    'Prim': 'Roofnet_MNIST_prim'
}

def read_metrics(file_path):
    with open(file_path, 'rb') as file:
        metrics_history = pickle.load(file)
    return metrics_history

def average_metrics(metrics_list):
    # Assuming the length of lists for all agents and all epochs are the same
     # Calculate the average training loss across all agents for each epoch
    avg_train_loss = np.mean(metrics_list['train_loss'], axis=0)
    
    # Calculate the average test accuracy across all agents for each epoch
    avg_test_accuracy = np.mean(metrics_list['test_accuracy'], axis=0)

    return avg_train_loss, avg_test_accuracy

# Plotting
# line_styles = ['-', '--', '-.', ':']
# markers = ['o', 's', '^', 'D', 'x', '*', 'p', 'v', '>', '<', 'h']
def plot_metrics(metrics_dict, ylabel, network_type):
    plt.figure(figsize=(12, 8))
    # Define a consistent style: solid lines for all plots
    line_style = '-'
    
    # Expected order of plots
    order = ['Clique', 'Ring', 'Prim', 'SCA', 'Relaxation-rho', 'Relaxation-lambda', 'Greedy']
    
    # Sort the dictionary according to the desired plot order
    sorted_metrics = {k: metrics_dict[categorized_results[k]] for k in order if categorized_results[k] in metrics_dict}
    # Plot each metric with labels
    for idx, (label, metric) in enumerate(sorted_metrics.items()):
        plt.plot(metric, label=label, linestyle=line_style)
    
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    
    # Place the legend
    if sorted_metrics:  # Only call legend if there are labeled plots
        plt.legend(loc='best', fontsize=25)
    
    plt.grid(True)

    # Save the plot as an EPS file
    save_path = os.path.join(os.getcwd(), 'graph_result', f'{ylabel}_{network_type}_mnist.eps')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.savefig(save_path)

    plt.savefig(save_path, format='eps', bbox_inches='tight')  # Specify format here to handle EPS


def plot_metrics_time(metrics, ylabel, time_per_epoch, network_type):
    plt.figure(figsize=(12, 8))  # Larger size for better readability

    # Define styles, markers, and colors
    line_style = '-'
    
    # Expected order of plots
    order = ['Clique', 'Ring', 'Prim', 'SCA', 'Relaxation-rho', 'Relaxation-lambda', 'Greedy']
    
    # Sort the dictionary according to the desired plot order
    sorted_metrics = {k: metrics[categorized_results[k]] for k in order if categorized_results[k] in metrics}

    for idx, (key, metric) in enumerate(sorted_metrics.items()):
        epochs = np.arange(1, len(metric) + 1)
        time = epochs * (time_per_epoch[categorized_results[key]] / 60)  # Calculate cumulative time for each epoch
        
        plt.plot(time, metric, label=key, linestyle=line_style)

    plt.xlabel('Time (minutes)', fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.legend(loc='best', fontsize=25)
    plt.grid(True)  # Grid for visual guidance

    # Save the plot
    save_path = os.path.join(os.getcwd(), 'graph_result_time', f'{ylabel}_over_time_{network_type}_mnist.eps')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.savefig(save_path)
    plt.savefig(save_path, format='eps', bbox_inches='tight')   # Higher dpi for better image quality

    # Show the plot (optional, useful if running interactively)

network_type = "Roofnet" #"Roofnet"
with open(f"/scratch2/tingyang/DFS_Topo/tau_results_{network_type}_mnist.pkl", 'rb') as file:
    time_per_epoch_diction = pickle.load(file)

print(time_per_epoch_diction)


# time_per_epoch_diction = {'MNIST_SCA23_1': 4532.774887751348, 'MNIST_SCA23_2': 4532.774883521915, 'MNIST_SCA23_3': 4532.774881939509, 
#         'MNIST_SDRLambda2Ew_1': 6043.699793544735, 'MNIST_SDRLambda2Ew_2': 6043.6998125462205, 
#         'MNIST_SDRLambda2Ew_3': 6043.699812221992, 'MNIST_SDRRhoEw_1': 4532.797657392192, 
#         'MNIST_BoydGreedy_1': 4532.774885300265, 'MNIST_ring': 3021.8499280883507, 'MNIST_random': 3021.850796229334, 'MNIST_clique': 13598.324506617035, 
#         'MNIST_prim': 3021.8499001230703}

# Initialize lists to store averaged metrics
all_avg_train_loss = {}
all_avg_test_accuracy = {}

for file_path in file_paths:
    metrics_history = read_metrics(file_path)
    avg_train_loss, avg_test_accuracy = average_metrics(metrics_history)
    
    # Extract the descriptive name from the file path
    matrix_name = os.path.basename(file_path).split('.')[0].replace('result_for_cnn_', '')
    
    # Store the averaged metrics in the dictionaries with the extracted name as the key
    all_avg_train_loss[matrix_name] = avg_train_loss
    all_avg_test_accuracy[matrix_name] = avg_test_accuracy

# Plot the averaged training loss and test accuracy
# print(all_avg_test_accuracy)
plot_metrics(all_avg_train_loss, 'Loss', network_type)
plot_metrics(all_avg_test_accuracy, 'Accuracy', network_type)
plot_metrics_time(all_avg_train_loss, 'Loss', time_per_epoch_diction, network_type)
plot_metrics_time(all_avg_test_accuracy, 'Accuracy', time_per_epoch_diction, network_type)