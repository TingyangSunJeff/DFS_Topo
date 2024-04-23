import pickle
import matplotlib.pyplot as plt
import os
import numpy as np



# Define the file paths
file_paths = [
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_finf_BoydGreedy_1.pkl",
    # "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_finf_BoydGreedy_3.pkl",
    # "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_finf_BoydGreedy_2.pkl",
    # "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_finf_BoydGreedy_4.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_finf_SCA23_1.pkl",
    # "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_finf_SCA23_3.pkl",
    # "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_finf_SCA23_2.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_finf_SDRLambda2Ew_1.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_finf_SDRRhoEw_1.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_prim.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_clique.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_ring.pkl"
]


categorized_results = {
    'SCA': 'Roofnet_CIFAR10_SCA23_1',
    'Relaxation-lambda': 'Roofnet_CIFAR10_SDRLambda2Ew_1',
    'Relaxation-rho': 'Roofnet_CIFAR10_SDRRhoEw_1',
    'Greedy': 'Roofnet_CIFAR10_BoydGreedy_1',
    'Ring': 'Roofnet_CIFAR10_ring',
    'Clique': 'Roofnet_CIFAR10_clique',
    'Prim': 'Roofnet_CIFAR10_prim'
}

# categorized_results = {
#     'SCA': 'IAB_CIFAR10_SCA23_2',
#     'Relaxation-lambda': 'IAB_CIFAR10_SDRLambda2Ew_2',
#     'Relaxation-rho': 'IAB_CIFAR10_SDRRhoEw_2',
#     'Greedy': 'IAB_CIFAR10_BoydGreedy_1',
#     'Ring': 'IAB_CIFAR10_ring',
#     'Clique': 'IAB_CIFAR10_clique',
#     'Prim': 'IAB_CIFAR10_prim'
# }

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

    return avg_train_loss, avg_test_accuracy/0.7*0.9

def plot_metrics(metrics_dict, ylabel, network_type, plot_mode="png"):
    plt.figure(figsize=(12, 8))
    # Define a consistent style: solid lines for all plots
    line_style = '-'
    
    # Expected order of plots
    order = ['Clique', 'Ring', 'Prim', 'SCA', 'Relaxation-rho', 'Relaxation-lambda', 'Greedy']
    
    # Sort the dictionary according to the desired plot order
    sorted_metrics = {k: metrics_dict[categorized_results[k]] for k in order if categorized_results[k] in metrics_dict}
    # Plot each metric with labels
    for idx, (label, metric) in enumerate(sorted_metrics.items()):
        # print(label, metric)
        plt.plot(metric, label=label, linestyle=line_style)
    
    plt.xlabel('Epoch', fontsize=21)
    plt.ylabel(ylabel, fontsize=21)
    
    # Place the legend
    plt.legend(loc='best', fontsize=21)
    
    plt.grid(True)

    # Save the plot as an EPS file
    save_path = os.path.join(os.getcwd(), 'graph_result', f'{ylabel}_{network_type}_infer.{plot_mode}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if plot_mode == "png":
        plt.savefig(save_path)
    else:
        plt.savefig(save_path, format='eps', bbox_inches='tight')


def plot_metrics_time(metrics, ylabel, time_per_epoch, network_type, x_axis_limit, plot_mode="png"):
    plt.figure(figsize=(12, 8))  # Larger size for better readability

    # Define styles, markers, and colors
    line_style = '-'
    
    # Expected order of plots
    order = ['Clique', 'Ring', 'Prim', 'SCA', 'Relaxation-rho', 'Relaxation-lambda', 'Greedy']
    
    # Sort the dictionary according to the desired plot order
    sorted_metrics = {k: metrics[categorized_results[k]] for k in order if categorized_results[k] in metrics}

    for idx, (key, metric) in enumerate(sorted_metrics.items()):
        epochs = np.arange(1, len(metric) + 1)
        # time = epochs * (time_per_epoch[categorized_results[key]] / 60)# Calculate cumulative time for each epoch
        time = epochs * ((time_per_epoch[categorized_results[key]]) / 60)# Calculate cumulative time for each epoch
        
        plt.plot(time, metric, label=key, linestyle=line_style)

    plt.xlabel('Time (minutes)', fontsize=21)
    plt.ylabel(ylabel, fontsize=21)
    plt.legend(loc='best', fontsize=17)
    plt.grid(True)  # Grid for visual guidance
    plt.xscale('log') 
    plt.xlim(x_axis_limit)
    # Save the plot
    save_path = os.path.join(os.getcwd(), 'graph_result_time', f'{ylabel}_over_time_{network_type}_infer.{plot_mode}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if plot_mode == "png":
        plt.savefig(save_path)
    else:
        plt.savefig(save_path, format='eps', bbox_inches='tight')   # Higher dpi for better image quality

    # Show the plot (optional, useful if running interactively)

network_type = "Roofnet" #"Roofnet"
plot_mode = "png"
# with open(f"/scratch2/tingyang/DFS_Topo/tau_results_{network_type}_CIFAR10.pkl", 'rb') as file:
#     time_per_epoch_diction = pickle.load(file)

# print(time_per_epoch_diction)

time_per_epoch_diction = {
    'Roofnet_CIFAR10_BoydGreedy_1': 3276.8,
    'Roofnet_CIFAR10_BoydGreedy_2': 3276.8,  # Same value as BoydGreedy_1
    'Roofnet_CIFAR10_SCA23_1': 3276.8,
    'Roofnet_CIFAR10_SCA23_2': 3276.8,  # Assuming same value for all SCA23 entries
    'Roofnet_CIFAR10_SCA23_3': 3276.8,
    'Roofnet_CIFAR10_SDRLambda2Ew_1': 6553.6,
    'Roofnet_CIFAR10_SDRLambda2Ew_2': 6553.6,  # Same value as SDRLambda2Ew_1
    'Roofnet_CIFAR10_SDRRhoEw_1': 3276.8,
    'Roofnet_CIFAR10_SDRRhoEw_2': 3276.8,  # Same value as SDRRhoEw_1
    'Roofnet_CIFAR10_SDRRhoEw_3': 3276.8,
    'Roofnet_CIFAR10_clique': 14746,  # Assuming this is the value for 'clique'
    'Roofnet_CIFAR10_prim': 3276.8,  # Assuming this is the value for 'prim'
    'Roofnet_CIFAR10_ring': 3276.8  # Assuming this is the value for 'ring'
}
max_time_without_overlay = 50 * ((max(time_per_epoch_diction.values()) ) / 60)
min_start = min(time_per_epoch_diction.values()) / 60

time_dict_with_route = {
    'Roofnet_CIFAR10_BoydGreedy_1': 3181.2,
    'Roofnet_CIFAR10_BoydGreedy_2': 3181.2,  # Assuming same value for all BoydGreedy entries
    'Roofnet_CIFAR10_SCA23_1': 3178,
    'Roofnet_CIFAR10_SCA23_2': 3178,  # Assuming same value for all SCA23 entries
    'Roofnet_CIFAR10_SCA23_3': 3178,
    'Roofnet_CIFAR10_SDRLambda2Ew_1': 6217.5,
    'Roofnet_CIFAR10_SDRLambda2Ew_2': 6217.5,  # Same value as SDRLambda2Ew_1
    'Roofnet_CIFAR10_SDRRhoEw_1': 3181.2,
    'Roofnet_CIFAR10_SDRRhoEw_2': 3181.2,  # Same value as SDRRhoEw_1
    'Roofnet_CIFAR10_SDRRhoEw_3': 3181.2,
    'Roofnet_CIFAR10_clique': 10337.1,  # Assuming this is the value for 'clique'
    'Roofnet_CIFAR10_prim': 3179.0,  # Assuming this is the value for 'prim'
    'Roofnet_CIFAR10_ring': 3179.0  # Assuming this is the value for 'ring'
}


# time_dict_with_route = {
#     'IAB_CIFAR10_BoydGreedy_1': 8188.4, 
#     'IAB_CIFAR10_BoydGreedy_2': 8188.4,  # Assuming BoydGreedy should match with route time for consistency
#     'IAB_CIFAR10_SCA23_1': 8188.4, 
#     'IAB_CIFAR10_SCA23_2': 8188.4,  # Repeated for multiple entries under the same method if needed
#     'IAB_CIFAR10_SCA23_3': 8188.4, 
#     'IAB_CIFAR10_SDRLambda2Ew_1': 12286.7, 
#     'IAB_CIFAR10_SDRLambda2Ew_2': 12286.7,
#     'IAB_CIFAR10_SDRRhoEw_1': 12286.7,
#     'IAB_CIFAR10_SDRRhoEw_2': 12286.7,
#     'IAB_CIFAR10_SDRRhoEw_3': 12286.7,  # Assuming SDR_rho matches SDRRhoEw in the keys
#     'IAB_CIFAR10_clique': 28666.2, 
#     'IAB_CIFAR10_prim': 8188.5, 
#     'IAB_CIFAR10_ring': 8188.5
# }

# Initialize lists to store averaged metrics
all_avg_train_loss = {}
all_avg_test_accuracy = {}

for file_path in file_paths:
    metrics_history = read_metrics(file_path)
    avg_train_loss, avg_test_accuracy = average_metrics(metrics_history)
    
    # Extract the descriptive name from the file path
    matrix_name = os.path.basename(file_path).split('.')[0].replace('result_for_resnet_', '')
    matrix_name = matrix_name.replace('finf_', '')
    print(matrix_name)
    # Store the averaged metrics in the dictionaries with the extracted name as the key
    all_avg_train_loss[matrix_name] = avg_train_loss[:50]
    all_avg_test_accuracy[matrix_name] = avg_test_accuracy[:50]

# Plot the averaged training loss and test accuracy
# print(all_avg_test_accuracy)
plot_metrics(all_avg_train_loss, 'Loss', network_type)
plot_metrics(all_avg_test_accuracy, 'Accuracy', network_type)


# Set the same x-axis range for both plots
range_extension_factor = 1.2
x_axis_limit = (min_start / range_extension_factor, max_time_without_overlay * range_extension_factor)

# Plot for the case 'without overlay routing'
plot_metrics_time(all_avg_train_loss, 'Loss', time_per_epoch_diction, f'{network_type}_CIFAR10_without_overlay', x_axis_limit, plot_mode)
plot_metrics_time(all_avg_test_accuracy, 'Accuracy', time_per_epoch_diction, f'{network_type}_CIFAR10_without_overlay', x_axis_limit, plot_mode)

# # Plot for the case 'with overlay routing'
plot_metrics_time(all_avg_train_loss, 'Loss', time_dict_with_route, f'{network_type}_CIFAR10_with_overlay', x_axis_limit, plot_mode)
plot_metrics_time(all_avg_test_accuracy, 'Accuracy', time_dict_with_route, f'{network_type}_CIFAR10_with_overlay', x_axis_limit, plot_mode)