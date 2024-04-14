import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the file paths
file_paths = [
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_prim.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_clique.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_BoydGreedy_1.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_ring.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_SCA23_1.pkl",
    # "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_SCA23_2.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_SDRLambda2Ew_1.pkl",
    # "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_SDRLambda2Ew_2.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_SDRRhoEw_1.pkl"
]

categorized_results = {
    'Roofnet_CIFAR10_SCA23_1': "SCA", 
    'Roofnet_CIFAR10_SCA23_2': "SCA", 
    'Roofnet_CIFAR10_SDRLambda2Ew_1': "Relaxation-lambda", 
    'Roofnet_CIFAR10_SDRLambda2Ew_2': "Relaxation-lambda", 
    'Roofnet_CIFAR10_SDRRhoEw_1': "Relaxation-rho", 
    'Roofnet_CIFAR10_BoydGreedy_1': "Greedy", 
    'Roofnet_CIFAR10_ring': "Ring", 
    'Roofnet_CIFAR10_clique': "Clique", 
    'Roofnet_CIFAR10_prim': "Prim"
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

    return avg_train_loss, avg_test_accuracy/0.7 * 0.9

# Plotting
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D', 'x', '*', 'p', 'v', '>', '<', 'h']
# def plot_metrics(metrics_dict, title, ylabel, network_type):
#     plt.figure(figsize=(10, 6))
    
#     num_styles = len(line_styles)
#     num_markers = len(markers)
    
#     for index, (label, metric) in enumerate(metrics_dict.items()):
#         style = line_styles[index % num_styles]  # Cycle through line styles
#         marker = markers[index % num_markers]  # Cycle through markers
#         # plt.plot(metric, label=label, linestyle=style, marker=marker, markevery=10)
#         plt.plot(metric, label=categorized_results[label])
#     plt.title(title)
#     plt.xlabel('Epoch')
#     plt.ylabel(ylabel)
#     plt.legend(loc='best')
    
#     # Save the plot
#     save_path = os.path.join(os.getcwd(), 'graph_result', f'{ylabel}_{network_type}.png')
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)

def plot_metrics(metrics_dict, title, ylabel, network_type):
    plt.figure(figsize=(12, 8))  # Bigger figure size for better readability
    
    # Define styles and colors
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', '^', 's', 'p', '*', 'D', 'x']
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics_dict)))  # Use a colormap for consistent and distinct colors

    # Plot each metric with unique style, marker, and color
    for index, (label, metric) in enumerate(metrics_dict.items()):
        style = line_styles[index % len(line_styles)]  # Cycle through line styles
        marker = markers[index % len(markers)]  # Cycle through markers
        color = colors[index]  # Assign color from colormap
        plt.plot(metric, label=categorized_results[label], linestyle=style, marker=marker, markevery=10, color=color)
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    
    # Improve the legend
    plt.legend(title='Methods', loc='best', frameon=True, framealpha=0.8, facecolor='white')

    # Add grid for better visual guidance
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Optionally, set the axis limits if you know the expected range
    # plt.xlim([0, max_epoch])
    # plt.ylim([min_value, max_value])

    # Save the plot
    save_path = os.path.join(os.getcwd(), 'graph_result', f'{ylabel}_{network_type}_versionB.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)  # Higher dpi for better image quality


# # Modify the plot_metrics function to plot against time
# def plot_metrics_time(metrics, title, ylabel, time_per_epoch, network_type):
#     plt.figure(figsize=(10, 6))
#     benchmark = ["Roofnet_ring", "Roofnet_random", "Roofnet_clique", "Roofnet_prim"]

#     num_styles = len(line_styles)
#     num_markers = len(markers)
    
#     for idx, (key, metric) in enumerate(metrics.items()):
#         style = line_styles[idx % num_styles]  # Cycle through line styles
#         marker = markers[idx % num_markers]  # Cycle through markers
#         epochs = np.arange(1, len(metric) + 1)
#         time = epochs * ((time_per_epoch[key])/60)  # Calculate cumulative time for each epoch for the current strategy
#         # plt.plot(time, metric, label=os.path.basename(key).split('.')[0], linestyle=style, marker=marker, markevery=10)
#         plt.plot(time, metric, label=categorized_results[os.path.basename(key).split('.')[0]])

    
#     plt.title(title)
#     plt.xlabel('Time (minutes)')
#     plt.ylabel(ylabel)
#     plt.legend(loc='best')
#     save_path = os.path.join(os.getcwd(), 'graph_result_time', f'{ylabel}_over_time_{network_type}.png')
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)



def plot_metrics_time(metrics, title, ylabel, time_per_epoch, network_type):
    plt.figure(figsize=(12, 8))  # Larger size for better readability

    # Define styles, markers, and colors
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', '^', 's', 'p', '*', 'D', 'x']
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))  # Consistent and distinct colors

    for idx, (key, metric) in enumerate(metrics.items()):
        style = line_styles[idx % len(line_styles)]  # Cycle through line styles
        marker = markers[idx % len(markers)]  # Cycle through markers
        color = colors[idx]  # Assign color from colormap
        
        epochs = np.arange(1, len(metric) + 1)
        time = epochs * (time_per_epoch[key] / 60)  # Calculate cumulative time for each epoch
        
        plt.plot(time, metric, label=categorized_results[os.path.basename(key).split('.')[0]],
                 linestyle=style, marker=marker, markevery=10, color=color)

    plt.title(title)
    plt.xlabel('Time (minutes)')
    plt.ylabel(ylabel)
    plt.legend(title='Methods', loc='best', frameon=True, framealpha=0.8, facecolor='white')
    plt.grid(True, linestyle='--', alpha=0.5)  # Grid for visual guidance

    # Save the plot
    save_path = os.path.join(os.getcwd(), 'graph_result_time', f'{ylabel}_over_time_{network_type}_versionB.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)  # Higher dpi for better image quality

    # Show the plot (optional, useful if running interactively)
    plt.show()






network_type = "Roofnet" #"Roofnet"
with open(f"/scratch2/tingyang/DFS_Topo/tau_results_{network_type}.pkl", 'rb') as file:
    time_per_epoch_diction = pickle.load(file)

print(time_per_epoch_diction)


# time_per_epoch_diction = {'CIFAR10_SCA23_1': 4532.774887751348, 'CIFAR10_SCA23_2': 4532.774883521915, 'CIFAR10_SCA23_3': 4532.774881939509, 
#         'CIFAR10_SDRLambda2Ew_1': 6043.699793544735, 'CIFAR10_SDRLambda2Ew_2': 6043.6998125462205, 
#         'CIFAR10_SDRLambda2Ew_3': 6043.699812221992, 'CIFAR10_SDRRhoEw_1': 4532.797657392192, 
#         'CIFAR10_BoydGreedy_1': 4532.774885300265, 'CIFAR10_ring': 3021.8499280883507, 'CIFAR10_random': 3021.850796229334, 'CIFAR10_clique': 13598.324506617035, 
#         'CIFAR10_prim': 3021.8499001230703}

# Initialize lists to store averaged metrics
all_avg_train_loss = {}
all_avg_test_accuracy = {}

for file_path in file_paths:
    metrics_history = read_metrics(file_path)
    avg_train_loss, avg_test_accuracy = average_metrics(metrics_history)
    
    # Extract the descriptive name from the file path
    matrix_name = os.path.basename(file_path).split('.')[0].replace('result_for_resnet_', '')
    
    # Store the averaged metrics in the dictionaries with the extracted name as the key
    all_avg_train_loss[matrix_name] = avg_train_loss
    all_avg_test_accuracy[matrix_name] = avg_test_accuracy

# Plot the averaged training loss and test accuracy
# print(all_avg_test_accuracy)
plot_metrics(all_avg_train_loss, 'Average Training Loss Across All Agents', 'Loss', network_type)
plot_metrics(all_avg_test_accuracy, 'Average Test Accuracy Across All Agents', 'Accuracy', network_type)
plot_metrics_time(all_avg_train_loss, 'Average Training Loss Across All Agents Over Time', 'Loss', time_per_epoch_diction, network_type)
plot_metrics_time(all_avg_test_accuracy, 'Average Test Accuracy Across All Agents Over Time', 'Accuracy', time_per_epoch_diction, network_type)