import pickle
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import os
import numpy as np

# Define the file paths
all_file_paths = [
    "result_for_resnet_Ring_0niid_schedule.pkl",
    "result_for_resnet_SCA_0niid_schedule.pkl",
    "result_for_resnet_SMMD_SM_0niid_schedule.pkl",
    "result_for_resnet_Clique_0niid_schedule.pkl",
    # "result_for_resnet_MST_0niid_schedule.pkl",
    # "result_for_resnet_Ring_0.3niid.pkl",
    # "result_for_resnet_Ring_0niid.pkl",
    # "result_for_resnet_SCA_0.3niid.pkl",
    # "result_for_resnet_SCA_0niid.pkl",
    # "result_for_resnet_SMMD_SM_0.3niid.pkl",
    # "result_for_resnet_SMMD_SM_0niid.pkl"
]


# Set the mode to either "SMMD_PM", "SMMD_SM", or "all"
mode = "all"   # Change this value to choose the desired mode

if mode == "SMMD_PM":
    file_paths = [fp for fp in all_file_paths if "SMMD_PM" in fp]
elif mode == "SMMD_SM":
    file_paths = [fp for fp in all_file_paths if "SMMD_SM" in fp]
elif mode == "all":
    file_paths = all_file_paths
else:
    raise ValueError("Invalid mode. Choose 'SMMD_PM', 'SMMD_SM', or 'all'.")
mode += "0niid"
def update_results_and_order(categorized_results, order, all_file_paths):
    """
    For each file in all_file_paths, this function removes the "result_for_resnet_" prefix 
    and the ".pkl" extension to derive a key. It then updates:
      - categorized_results: if the key is not already present, it's added with itself as value.
      - order: if the key is not already in the list, it's appended.
    Returns the updated categorized_results and order.
    """
    prefix = "result_for_resnet_"
    for fp in all_file_paths:
        # Remove the prefix if present.
        file_core = fp[len(prefix):] if fp.startswith(prefix) else fp
        
        # Remove the file extension.
        if file_core.endswith(".pkl"):
            file_core = file_core[:-4]
        
        # Update categorized_results if key not present.
        if file_core not in categorized_results:
            categorized_results[file_core] = file_core
        
        # Update order list if key not already present.
        if file_core not in order:
            order.append(file_core)
            
    return categorized_results, order

def read_metrics(file_path):
    with open(file_path, 'rb') as file:
        metrics_history = pickle.load(file)
    return metrics_history

def average_metrics(metrics_list):
    # Calculate the average training loss and test accuracy across all agents for each epoch
    avg_train_loss = np.mean(metrics_list['train_loss'], axis=0)
    avg_test_accuracy = np.mean(metrics_list['test_accuracy'], axis=0)
    return avg_train_loss, avg_test_accuracy

def plot_metrics(folder_name, data_type, threshold, metrics_dict, ylabel, network_type, withinfer, order, mode, plot_mode="png"):
    plt.figure(figsize=(12, 8))
    line_style = '-'
    # For keys that do not match exactly, you might need to adjust them.
    sorted_metrics = {k: metrics_dict[categorized_results[k]] for k in order if categorized_results[k] in metrics_dict}
    convergence_epochs = {}
    for label, metric in sorted_metrics.items():
        plt.plot(metric, label=label, linestyle=line_style)
        # Record the epoch index where the metric reaches the threshold
        convergence_epoch = np.where(metric <= threshold)[0]
        if convergence_epoch.size > 0:
            convergence_epochs[label] = convergence_epoch[0]
        else:
            convergence_epochs[label] = 'Did not converge'
    plt.xlabel('Epoch', fontsize=21)
    plt.ylabel(ylabel, fontsize=21)
    plt.legend(loc='best', fontsize=21)
    plt.grid(True)
    save_path = os.path.join(os.getcwd(), folder_name, f'{ylabel}_{network_type}_{data_type}{withinfer}_{mode}.{plot_mode}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return convergence_epochs

def plot_metrics_time(folder_name, metrics, ylabel, time_per_epoch, network_type, x_axis_limit, order, mode, plot_mode="png"):
    plt.figure(figsize=(12, 8))
    line_style = '-'
    # Expected order of plots, extend as needed
    sorted_metrics = {k: metrics[categorized_results[k]] for k in order if categorized_results[k] in metrics}
    for key, metric in sorted_metrics.items():
        epochs = np.arange(1, len(metric) + 1)
        time_cumulative = epochs * (time_per_epoch[categorized_results[key]])
        plt.plot(time_cumulative, metric, label=key, linestyle=line_style)
    plt.xlabel('Time (s)', fontsize=21)
    plt.ylabel(ylabel, fontsize=21)
    plt.legend(loc='best', fontsize=17)
    plt.grid(True)
    plt.xscale('log')
    plt.xlim(x_axis_limit)
    save_path = os.path.join(os.getcwd(), folder_name + "_time", f'{ylabel}_over_time_{network_type}_{mode}.{plot_mode}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# Determine network type and data type from one of the file paths
if "IAB" in file_paths[0]:
    network_type = "IAB"
elif "Roofnet" in file_paths[0]:
    network_type = "Roofnet"
else:
    network_type = "Unknown"

if "CIFAR10" in file_paths[0]:
    data_type = "CIFAR10"
elif "MNIST" in file_paths[0]:
    data_type = "MNIST"
else:
    data_type = "Unknown"

plot_mode = "png"
withinfer = ""
data_type = "CIFAR10"
network_type = "Roofnet"
threshold = 0.15 if data_type == "MNIST" else 0.001
epoch = 300 if data_type == "MNIST" else 300
model = "cnn" if data_type == "MNIST" else "resnet"
folder_name = "graph_result_niid_new"
# Update the categorized_results dictionary if needed
categorized_results = {
    # 'SCA': f'{network_type}_{data_type}_SCA23_1',
    # 'Relaxation-lambda': f'{network_type}_{data_type}_SDRLambda2Ew_1',
    # 'Relaxation-rho': f'{network_type}_{data_type}_SDRRhoEw_1',
    # 'Greedy': f'{network_type}_{data_type}_BoydGreedy_1',
    # 'Ring': f'{network_type}_{data_type}_ring',
    # 'Clique': f'{network_type}_{data_type}_clique',
    # 'Prim': f'{network_type}_{data_type}_prim',
}

order = [
    # 'Clique', 'Ring', 'Prim', 
    # 'SCA', 
    # 'Relaxation-rho', 'Relaxation-lambda', 
    # 'Greedy',
]

categorized_results, order = update_results_and_order(categorized_results, order, all_file_paths)
print("Updated categorized_results:")
print(categorized_results)
print("\nUpdated order:")
print(order)

# Initialize dictionaries to store averaged metrics from each file
all_avg_train_loss = {}
all_avg_test_accuracy = {}

for file_path in file_paths:
    metrics_history = read_metrics(file_path)
    avg_train_loss, avg_test_accuracy = average_metrics(metrics_history)
    
    # Extract the descriptive name from the file path
    matrix_name = os.path.basename(file_path).split('.pkl')[0].replace(f'result_for_{model}_', '')
    matrix_name = matrix_name.replace('finf_', '')
    # Store the averaged metrics in the dictionaries with the extracted name as the key
    all_avg_train_loss[matrix_name] = avg_train_loss[:epoch]
    all_avg_test_accuracy[matrix_name] = avg_test_accuracy[:epoch]

# print(all_avg_train_loss)
# Plot the averaged training loss and test accuracy
plot_metrics(folder_name, data_type, threshold, all_avg_train_loss, 'Loss', network_type, withinfer, order, mode, plot_mode)
plot_metrics(folder_name, data_type, threshold, all_avg_test_accuracy, 'Accuracy', network_type, withinfer, order, mode, plot_mode)



time_dict_without_route = {
    # 'Roofnet_CIFAR10_BoydGreedy_1': 4.9152 * 1e3,
    # 'Roofnet_CIFAR10_BoydGreedy_2': 4.9152 * 1e3,
    'SCA_0niid': 1.6384 * 1e3,
    'SCA_0.3niid': 1.6384 * 1e3,
    'SCA_0.6niid': 1.6384 * 1e3,
    'SMMD_PM_penalize_9T_0niid': 1.6384 * 1e3 * 2,
    'SMMD_PM_penalize_9T_0.6niid': 1.6384 * 1e3 * 2,
    'SMMD_PM_penalize_9T_0.9niid': 1.6384 * 1e3 * 2,
    'SCA_0niid_schedule': 1.6384 * 1e3,
    'SCA_0.3niid_schedule': 1.6384 * 1e3,
    'SCA_0.6niid_schedule': 1.6384 * 1e3,
    'SMMD_PM_penalize_9T_0niid_schedule': 1.6384 * 1e3 * 2,
    'SMMD_PM_penalize_9T_0.6niid_schedule': 1.6384 * 1e3 * 2,
    'SMMD_PM_penalize_9T_0.9niid_schedule': 1.6384 * 1e3 * 2,
    # 'Roofnet_CIFAR10_SDRRhoEw_1': 4.9152 * 1e3,
    # 'Roofnet_CIFAR10_SDRRhoEw_2': 4.9152 * 1e3,
    # 'Roofnet_CIFAR10_SDRRhoEw_3': 4.9152 * 1e3,
    # 'Roofnet_CIFAR10_SDRLambda2Ew_1': 6.5536 * 1e3,
    # 'Roofnet_CIFAR10_SDRLambda2Ew_2': 6.5536 * 1e3,
    'Clique_0niid_schedule': 14.746 * 1e3,
    'Clique_0.3niid_schedule': 14.746 * 1e3,
    'Clique_0.6niid_schedule': 14.746 * 1e3,
    # 'Clique_0niid': 14.746 * 1e3,
    # 'Clique_0.3niid': 14.746 * 1e3,
    # 'Clique_0.6niid': 14.746 * 1e3,
    # 'Roofnet_CIFAR10_prim': 3.2768 * 1e3,
    # 'Roofnet_CIFAR10_ring': 3.2768 * 1e3,
    'Ring_0niid_schedule': 3.2768 * 1e3,
    'Ring_0.3niid_schedule': 3.2768 * 1e3,
    'Ring_0.6niid_schedule': 3.2768 * 1e3,
    'Ring_0niid': 3.2768 * 1e3,
    'Ring_0.3niid': 3.2768 * 1e3,
    'Ring_0.6niid': 3.2768 * 1e3,
     # SMMD_PM (without routing)
    'SMMD_SM_0niid_schedule': 1638.385625,
    'SMMD_SM_0.3niid_schedule': 1638.385625,
    'SMMD_SM_0.6niid_schedule': 1638.385625,
    'SMMD_SM_0niid': 1638.385625,
    'SMMD_SM_0.3niid': 1638.385625,
    'SMMD_SM_0.6niid': 1638.385625
}



time_dict_with_route = {
    # 'Roofnet_CIFAR10_BoydGreedy_1': 4.7107 * 1e3,
    # 'Roofnet_CIFAR10_BoydGreedy_2': 4.7107 * 1e3,
    # 'Roofnet_CIFAR10_SCA23_1': 1.6063 * 1e3,
    'SCA_0niid': 1.6063 * 1e3,
    'SCA_0.3niid': 1.6063 * 1e3,
    'SCA_0.6niid': 1.6063 * 1e3,
    'SMMD_PM_penalize_9T_0niid': 1.6063 * 1e3 * 2,
    'SMMD_PM_penalize_9T_0.6niid': 1.6063 * 1e3 * 2,
    'SMMD_PM_penalize_9T_0.9niid': 1.6063 * 1e3 * 2,
    'Ring_0niid': 3.1790 * 1e3,
    'Ring_0.3niid': 3.1790 * 1e3,
    'Ring_0.6niid': 3.1790 * 1e3,
    'SCA_0niid_schedule': 1.6063 * 1e3,
    'SCA_0.3niid_schedule': 1.6063 * 1e3,
    'SCA_0.6niid_schedule': 1.6063 * 1e3,
    'SMMD_PM_penalize_9T_0niid_schedule': 1.6063 * 1e3 * 2,
    'SMMD_PM_penalize_9T_0.6niid_schedule': 1.6063 * 1e3 * 2,
    'SMMD_PM_penalize_9T_0.9niid_schedule': 1.6063 * 1e3 * 2,
    'Ring_0niid_schedule': 3.1790 * 1e3,
    'Ring_0.3niid_schedule': 3.1790 * 1e3,
    'Ring_0.6niid_schedule': 3.1790 * 1e3,
    # 'Roofnet_CIFAR10_SDRRhoEw_1': 4.7161 * 1e3,
    # 'Roofnet_CIFAR10_SDRRhoEw_2': 4.7161 * 1e3,
    # 'Roofnet_CIFAR10_SDRRhoEw_3': 4.7161 * 1e3,
    # 'Roofnet_CIFAR10_SDRLambda2Ew_1': 6.2387 * 1e3,
    # 'Roofnet_CIFAR10_SDRLambda2Ew_2': 6.2387 * 1e3,
    # 'Roofnet_CIFAR10_clique': 10.3371 * 1e3,
    'Clique_0niid_schedule': 10.3371 * 1e3,
    'Clique_0.3niid_schedule': 10.3371 * 1e3,
    'Clique_0.6niid_schedule': 10.3371 * 1e3,
    # 'Roofnet_CIFAR10_prim': 3.1790 * 1e3,
    # 'Roofnet_CIFAR10_ring': 3.1790 * 1e3,
    # SMMD_PM (with routing)
    'SMMD_SM_0niid_schedule': 1606.274512,
    'SMMD_SM_0.3niid_schedule': 1606.274512,
    'SMMD_SM_0.6niid_schedule': 1606.274512,
    'SMMD_SM_0niid': 1606.274512,
    'SMMD_SM_0.3niid': 1606.274512,
    'SMMD_SM_0.6niid': 1606.274512
}

# Set the same x-axis range for both plots
max_time_without_overlay = epoch * (max(time_dict_without_route.values()))
min_start = min(time_dict_without_route.values())
range_extension_factor = 1.2
x_axis_limit = (min_start / range_extension_factor, max_time_without_overlay * range_extension_factor)

# # Plot for the case 'without overlay routing'
plot_metrics_time(folder_name, all_avg_train_loss, 'Loss', time_dict_without_route, f'{network_type}_{data_type}{withinfer}_without_overlay', x_axis_limit, order, mode, plot_mode)
plot_metrics_time(folder_name, all_avg_test_accuracy, 'Accuracy', time_dict_without_route, f'{network_type}_{data_type}{withinfer}_without_overlay', x_axis_limit, order, mode, plot_mode)

# Plot for the case 'with overlay routing'
plot_metrics_time(folder_name, all_avg_train_loss, 'Loss', time_dict_with_route, f'{network_type}_{data_type}{withinfer}_with_overlay', x_axis_limit, order, mode, plot_mode)
plot_metrics_time(folder_name, all_avg_test_accuracy, 'Accuracy', time_dict_with_route, f'{network_type}_{data_type}{withinfer}_with_overlay', x_axis_limit, order, mode, plot_mode)