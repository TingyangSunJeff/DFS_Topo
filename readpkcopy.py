import pickle
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import os
import numpy as np

# Define the file paths
all_file_paths = [
    "result_for_resnet_Roofnet_CIFAR10_clique.pkl",
    "result_for_resnet_Roofnet_CIFAR10_prim.pkl",
    "result_for_resnet_Roofnet_CIFAR10_ring.pkl",
    "result_for_resnet_Roofnet_CIFAR10_SCA23_1.pkl",
    # "result_for_resnet_Roofnet_CIFAR10_SDRLambda2Ew_1.pkl",
    # "result_for_resnet_Roofnet_CIFAR10_SDRRhoEw_1.pkl",
    # "result_for_resnet_SMMD_PM_3T.pkl",
    "result_for_resnet_SMMD_PM_2T.pkl",
    "result_for_resnet_SMMD_PM_3T.pkl",
    "result_for_resnet_SMMD_PM_4T.pkl",
    "result_for_resnet_SMMD_PM_5T.pkl",
    "result_for_resnet_SMMD_SM_20T.pkl",
    "result_for_resnet_SMMD_SM_10T.pkl",
    "result_for_resnet_SMMD_SM_30T.pkl"
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


def read_metrics(file_path):
    with open(file_path, 'rb') as file:
        metrics_history = pickle.load(file)
    return metrics_history

def average_metrics(metrics_list):
    # Calculate the average training loss and test accuracy across all agents for each epoch
    avg_train_loss = np.mean(metrics_list['train_loss'], axis=0)
    avg_test_accuracy = np.mean(metrics_list['test_accuracy'], axis=0)
    return avg_train_loss, avg_test_accuracy

def plot_metrics(data_type, threshold, metrics_dict, ylabel, network_type, withinfer, order, mode, plot_mode="png"):
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
    save_path = os.path.join(os.getcwd(), 'graph_result_new', f'{ylabel}_{network_type}_{data_type}{withinfer}_{mode}.{plot_mode}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return convergence_epochs

def plot_metrics_time(metrics, ylabel, time_per_epoch, network_type, x_axis_limit, order, mode, plot_mode="png"):
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
    save_path = os.path.join(os.getcwd(), 'graph_result_time_new', f'{ylabel}_over_time_{network_type}_{mode}.{plot_mode}')
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
epoch = 100 if data_type == "MNIST" else 80
model = "cnn" if data_type == "MNIST" else "resnet"

# Update the categorized_results dictionary if needed
categorized_results = {
    'SCA': f'{network_type}_{data_type}_SCA23_1',
    'Relaxation-lambda': f'{network_type}_{data_type}_SDRLambda2Ew_1',
    'Relaxation-rho': f'{network_type}_{data_type}_SDRRhoEw_1',
    'Greedy': f'{network_type}_{data_type}_BoydGreedy_1',
    'Ring': f'{network_type}_{data_type}_ring',
    'Clique': f'{network_type}_{data_type}_clique',
    'Prim': f'{network_type}_{data_type}_prim',
    'SMMD_SM_10T': 'SMMD_SM_10T',
    'SMMD_PM_10T': 'SMMD_PM_10T',
    'SMMD_PM_2T': 'SMMD_PM_2T',
    'SMMD_PM_3T': 'SMMD_PM_3T',
    'SMMD_PM_4T': 'SMMD_PM_4T',
    'SMMD_PM_5T': 'SMMD_PM_5T',
    'SMMD_PM_6T': 'SMMD_PM_6T',
    'SMMD_PM_7T': 'SMMD_PM_7T',
    'SMMD_PM_8T': 'SMMD_PM_8T',
    'SMMD_PM_9T': 'SMMD_PM_9T',
    'SMMD_SM_20T': 'SMMD_SM_20T',
    'SMMD_SM_30T': 'SMMD_SM_30T',
    'SMMD_SM_40T': 'SMMD_SM_40T',
    'SMMD_SM_45T': 'SMMD_SM_45T'
}


order = [
    'Clique', 'Ring', 'Prim', 
    'SCA', 
    # 'Relaxation-rho', 'Relaxation-lambda', 'Greedy',
    'SMMD_PM_2T', 'SMMD_PM_3T', 'SMMD_PM_4T', 'SMMD_PM_5T',
    'SMMD_PM_6T', 'SMMD_PM_7T', 'SMMD_PM_8T', 'SMMD_PM_9T', 'SMMD_PM_10T',
    'SMMD_SM_10T', 'SMMD_SM_20T', 'SMMD_SM_30T', 'SMMD_SM_40T', 'SMMD_SM_45T'
]

# Initialize dictionaries to store averaged metrics from each file
all_avg_train_loss = {}
all_avg_test_accuracy = {}

for file_path in file_paths:
    metrics_history = read_metrics(file_path)
    avg_train_loss, avg_test_accuracy = average_metrics(metrics_history)
    
    # Extract the descriptive name from the file path
    matrix_name = os.path.basename(file_path).split('.')[0].replace(f'result_for_{model}_', '')
    matrix_name = matrix_name.replace('finf_', '')
    print(matrix_name)
    # Store the averaged metrics in the dictionaries with the extracted name as the key
    all_avg_train_loss[matrix_name] = avg_train_loss[:epoch]
    all_avg_test_accuracy[matrix_name] = avg_test_accuracy[:epoch]
# Plot the averaged training loss and test accuracy
plot_metrics(data_type, threshold, all_avg_train_loss, 'Loss', network_type, withinfer, order, mode, plot_mode)
plot_metrics(data_type, threshold, all_avg_test_accuracy, 'Accuracy', network_type, withinfer, order, mode, plot_mode)



time_dict_without_route = {
    'Roofnet_CIFAR10_BoydGreedy_1': 4.9152 * 1e3,
    'Roofnet_CIFAR10_BoydGreedy_2': 4.9152 * 1e3,
    'Roofnet_CIFAR10_SCA23_1': 1.6384 * 1e3,
    'Roofnet_CIFAR10_SCA23_2': 1.6384 * 1e3,
    'Roofnet_CIFAR10_SCA23_3': 1.6384 * 1e3,
    'Roofnet_CIFAR10_SDRRhoEw_1': 4.9152 * 1e3,
    'Roofnet_CIFAR10_SDRRhoEw_2': 4.9152 * 1e3,
    'Roofnet_CIFAR10_SDRRhoEw_3': 4.9152 * 1e3,
    'Roofnet_CIFAR10_SDRLambda2Ew_1': 6.5536 * 1e3,
    'Roofnet_CIFAR10_SDRLambda2Ew_2': 6.5536 * 1e3,
    'Roofnet_CIFAR10_clique': 14.746 * 1e3,
    'Roofnet_CIFAR10_prim': 3.2768 * 1e3,
    'Roofnet_CIFAR10_ring': 3.2768 * 1e3,
     # SMMD_PM (without routing)
    'SMMD_PM_10T': 14745.53767,
    'SMMD_PM_9T': 14745.54245,
    'SMMD_PM_8T': 13107.18831,
    'SMMD_PM_7T': 11468.75517,
    'SMMD_PM_6T': 9830.275623,
    'SMMD_PM_5T': 7698.478985,
    'SMMD_PM_4T': 6553.523775,
    'SMMD_PM_3T': 4915.165368,
    'SMMD_PM_2T': 3276.78583,
    # SMMD_SM (without routing)
    'SMMD_SM_20T': 6553.582935,
    'SMMD_SM_15T': 4915.172324,
    'SMMD_SM_25T': 8191.968195,
    'SMMD_SM_30T': 9830.275996,
    'SMMD_SM_10T': 3276.790484
}



time_dict_with_route = {
    'Roofnet_CIFAR10_BoydGreedy_1': 4.7107 * 1e3,
    'Roofnet_CIFAR10_BoydGreedy_2': 4.7107 * 1e3,
    'Roofnet_CIFAR10_SCA23_1': 1.6063 * 1e3,
    'Roofnet_CIFAR10_SCA23_2': 1.6063 * 1e3,
    'Roofnet_CIFAR10_SCA23_3': 1.6063 * 1e3,
    'Roofnet_CIFAR10_SDRRhoEw_1': 4.7161 * 1e3,
    'Roofnet_CIFAR10_SDRRhoEw_2': 4.7161 * 1e3,
    'Roofnet_CIFAR10_SDRRhoEw_3': 4.7161 * 1e3,
    'Roofnet_CIFAR10_SDRLambda2Ew_1': 6.2387 * 1e3,
    'Roofnet_CIFAR10_SDRLambda2Ew_2': 6.2387 * 1e3,
    'Roofnet_CIFAR10_clique': 10.3371 * 1e3,
    'Roofnet_CIFAR10_prim': 3.1790 * 1e3,
    'Roofnet_CIFAR10_ring': 3.1790 * 1e3,
    # SMMD_PM (with routing)
    'SMMD_PM_10T': 13052.30926,
    'SMMD_PM_9T': 13405.07267,
    'SMMD_PM_8T': 12013.79537,
    'SMMD_PM_7T': 10616.58283,
    'SMMD_PM_6T': 9181.554955,
    'SMMD_PM_5T': 7698.478985,
    'SMMD_PM_4T': 6241.41079,
    'SMMD_PM_3T': 4594.954563,
    'SMMD_PM_2T': 3181.357921,
    # SMMD_SM (with routing)
    'SMMD_SM_20T': 6220.816388,
    'SMMD_SM_15T': 4725.247485,
    'SMMD_SM_25T': 7631.366969,
    'SMMD_SM_30T': 9187.104194,
    'SMMD_SM_10T': 3181.359211
}

# Set the same x-axis range for both plots
max_time_without_overlay = epoch * (max(time_dict_without_route.values()))
min_start = min(time_dict_without_route.values())
range_extension_factor = 1.2
x_axis_limit = (min_start / range_extension_factor, max_time_without_overlay * range_extension_factor)

# # Plot for the case 'without overlay routing'
plot_metrics_time(all_avg_train_loss, 'Loss', time_dict_without_route, f'{network_type}_{data_type}{withinfer}_without_overlay', x_axis_limit, order, mode, plot_mode)
plot_metrics_time(all_avg_test_accuracy, 'Accuracy', time_dict_without_route, f'{network_type}_{data_type}{withinfer}_without_overlay', x_axis_limit, order, mode, plot_mode)

# Plot for the case 'with overlay routing'
plot_metrics_time(all_avg_train_loss, 'Loss', time_dict_with_route, f'{network_type}_{data_type}{withinfer}_with_overlay', x_axis_limit, order, mode, plot_mode)
plot_metrics_time(all_avg_test_accuracy, 'Accuracy', time_dict_with_route, f'{network_type}_{data_type}{withinfer}_with_overlay', x_axis_limit, order, mode, plot_mode)