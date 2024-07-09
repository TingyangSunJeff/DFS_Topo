import pickle
import matplotlib.pyplot as plt
import os
import numpy as np



# Define the file paths
file_paths = [
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_ring.pkl",
    "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_SCA23_1.pkl",
    # "/scratch2/tingyang/DFS_Topo/saved_training_data_results/result_for_resnet_Roofnet_CIFAR10_finf_SDRRhoEw_1.pkl",
    # "/scratch2/tingyang/DFS_Topo/saved_training_data_results/result_for_resnet_Roofnet_CIFAR10_finf_SDRLambda2Ew_1.pkl",
    # "/scratch2/tingyang/DFS_Topo/saved_training_data_results/result_for_resnet_Roofnet_CIFAR10_finf_SCA23_3.pkl",
    # # "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_finf_SCA23_2.pkl",
    # # "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_CIFAR10_finf_SCA23_1.pkl",
    # "/scratch2/tingyang/DFS_Topo/saved_training_data_results/result_for_resnet_Roofnet_CIFAR10_finf_BoydGreedy_1.pkl",
    # "/scratch2/tingyang/DFS_Topo/saved_training_data_results/result_for_resnet_Roofnet_CIFAR10_clique.pkl"
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
#     'SCA': 'Roofnet_CIFAR10_finf_SCA23_2',
#     'Relaxation-lambda': 'Roofnet_CIFAR10_finf_SDRLambda2Ew_2',
#     'Relaxation-rho': 'Roofnet_CIFAR10_finf_SDRRhoEw_2',
#     'Greedy': 'Roofnet_CIFAR10_finf_BoydGreedy_1',
#     'Ring': 'Roofnet_CIFAR10_finf_ring',
#     'Clique': 'Roofnet_CIFAR10_finf_clique',
#     'Prim': 'Roofnet_CIFAR10_finf_prim'
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

    return avg_train_loss, avg_test_accuracy /0.78

def plot_metrics(threshold, metrics_dict, ylabel, network_type, withinfer, plot_mode="png"):
    plt.figure(figsize=(12, 8))
    line_style = '-'
    order = ['Clique', 'Ring', 'Prim', 'SCA', 'Relaxation-rho', 'Relaxation-lambda', 'Greedy']
    sorted_metrics = {k: metrics_dict[categorized_results[k]] for k in order if categorized_results[k] in metrics_dict}
    convergence_epochs = {}
    for idx, (label, metric) in enumerate(sorted_metrics.items()):
        plt.plot(metric, label=label, linestyle=line_style)
        
        # Record the epoch index where the training loss first reaches 0.05 or below
        if label == "Clique":
            print(label)
            print(metric)
            print(np.where(metric <= threshold))

        convergence_epoch = np.where(metric <= threshold)[0]
        if convergence_epoch.size > 0:
            convergence_epochs[label] = convergence_epoch[0]
        else:
            convergence_epochs[label] = 'Did not converge'

    plt.xlabel('Epoch', fontsize=21)
    plt.ylabel(ylabel, fontsize=21)
    plt.legend(loc='best', fontsize=21)
    plt.grid(True)
    save_path = os.path.join(os.getcwd(), 'graph_result', f'{ylabel}_{network_type}_CIFAR10{withinfer}.{plot_mode}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if plot_mode == "png":
        plt.savefig(save_path)
    else:
        plt.savefig(save_path, format='eps', bbox_inches='tight')
    
    plt.show()
    
    return convergence_epochs


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
    save_path = os.path.join(os.getcwd(), 'graph_result_time', f'{ylabel}_over_time_{network_type}.{plot_mode}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if plot_mode == "png":
        plt.savefig(save_path)
    else:
        plt.savefig(save_path, format='eps', bbox_inches='tight')   # Higher dpi for better image quality

    # Show the plot (optional, useful if running interactively)

network_type = "Roofnet" #"Roofnet"/Roofnet
plot_mode = "png"
data_tyoe = "cifar" #cifar
withinfer = ""
# with open("/scratch2/tingyang/DFS_Topo/tau_results_Roofnet_mnist.pkl", 'rb') as file:
#     time_per_epoch_diction = pickle.load(file)

# print(time_per_epoch_diction)

# time_per_epoch_diction = {
#     'Roofnet_CIFAR10_BoydGreedy_1': 4.9871* 1e3,
#     'Roofnet_CIFAR10_BoydGreedy_2': 4.9871* 1e3,  # Same value for all BoydGreedy entries
#     'Roofnet_CIFAR10_SCA23_1': 1.6384* 1e3,
#     'Roofnet_CIFAR10_SCA23_2': 1.6384* 1e3,  # Same value for all SCA23 entries
#     'Roofnet_CIFAR10_SCA23_3': 1.6384* 1e3,
#     'Roofnet_CIFAR10_SDRRhoEw_1': 4.9871* 1e3,
#     'Roofnet_CIFAR10_SDRRhoEw_2': 4.9871* 1e3,  # Same value for all SDRRhoEw entries
#     'Roofnet_CIFAR10_SDRRhoEw_3': 4.9871* 1e3,
#     'Roofnet_CIFAR10_SDRLambda2Ew_1': 6.5536* 1e3,
#     'Roofnet_CIFAR10_SDRLambda2Ew_2': 6.5536* 1e3,  # Same value for all SDRLambda2Ew entries
#     'Roofnet_CIFAR10_clique': 14.746* 1e3 ,  # Value for 'clique'
#     'Roofnet_CIFAR10_prim': 3.2768* 1e3,  # Same value as other entries
#     'Roofnet_CIFAR10_ring': 3.2768* 1e3  # Same value as other entries
# }



# time_dict_with_route = {
#     'Roofnet_CIFAR10_BoydGreedy_1': 4.8862 * 1e3,
#     'Roofnet_CIFAR10_BoydGreedy_2': 4.8862 * 1e3,  # Same value for all BoydGreedy entries
#     'Roofnet_CIFAR10_SCA23_1': 1.6063* 1e3,
#     'Roofnet_CIFAR10_SCA23_2': 1.6063* 1e3,  # Same value for all SCA23 entries
#     'Roofnet_CIFAR10_SCA23_3': 1.6063* 1e3,
#     'Roofnet_CIFAR10_SDRRhoEw_1': 4.8862 * 1e3,
#     'Roofnet_CIFAR10_SDRRhoEw_2': 4.8862 * 1e3,  # Same value for all SDRRhoEw entries
#     'Roofnet_CIFAR10_SDRRhoEw_3': 4.8862 * 1e3,
#     'Roofnet_CIFAR10_SDRLambda2Ew_1': 6.2175* 1e3,
#     'Roofnet_CIFAR10_SDRLambda2Ew_2': 6.2175* 1e3,  # Same value for all SDRLambda2Ew entries
#     'Roofnet_CIFAR10_clique': 10.3371* 1e3,  # Value for 'clique'
#     'Roofnet_CIFAR10_prim': 3.1790* 1e3,  # Same value as other entries
#     'Roofnet_CIFAR10_ring': 3.1790* 1e3  # Same value as other entries
# }




threshold =0.001
epoch = 50 if data_tyoe == "mnist" else 60
# max_time_without_overlay = epoch * ((max(time_per_epoch_diction.values()) ) / 60)
# min_start = min(time_per_epoch_diction.values()) / 60

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
    all_avg_train_loss[matrix_name] = avg_train_loss[:epoch]
    all_avg_test_accuracy[matrix_name] = avg_test_accuracy[:epoch]

# Plot the averaged training loss and test accuracy
# print(all_avg_test_accuracy)
# print(all_avg_train_loss)
convergence_epochs = plot_metrics(threshold, all_avg_train_loss, 'Loss', network_type, withinfer, plot_mode)
plot_metrics(all_avg_test_accuracy, 'Accuracy', network_type, withinfer, plot_mode)

# Print convergence epochs
print(f"Convergence epochs for training loss <= {threshold}:")
for network, epoch in convergence_epochs.items():
    print(f"{network}: {epoch}")


# # Set the same x-axis range for both plots
# range_extension_factor = 1.2
# x_axis_limit = (min_start / range_extension_factor, max_time_without_overlay * range_extension_factor)

# # Plot for the case 'without overlay routing'
# plot_metrics_time(all_avg_train_loss, 'Loss', time_per_epoch_diction, f'{network_type}_CIFAR10{withinfer}_without_overlay', x_axis_limit, plot_mode)
# plot_metrics_time(all_avg_test_accuracy, 'Accuracy', time_per_epoch_diction, f'{network_type}_CIFAR10{withinfer}_without_overlay', x_axis_limit, plot_mode)

# # Plot for the case 'with overlay routing'
# plot_metrics_time(all_avg_train_loss, 'Loss', time_dict_with_route, f'{network_type}_CIFAR10{withinfer}_with_overlay', x_axis_limit, plot_mode)
# plot_metrics_time(all_avg_test_accuracy, 'Accuracy', time_dict_with_route, f'{network_type}_CIFAR10{withinfer}_with_overlay', x_axis_limit, plot_mode)