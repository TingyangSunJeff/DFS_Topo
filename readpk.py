import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the file paths
file_paths = [
#     '/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_SDRRhoEw_1.pkl',
#     '/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_BoydGreedy_1.pkl',
#     "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_SDRLambda2Ew_1.pkl",
#     "/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_SCA23_1.pkl",
    '/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_ring.pkl',
    # '/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_random.pkl',
    '/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_prim.pkl',
    '/scratch2/tingyang/DFS_Topo/result_for_resnet_Roofnet_clique.pkl'
]

def read_metrics(file_path):
    with open(file_path, 'rb') as file:
        metrics_history = pickle.load(file)
    return metrics_history

def average_metrics(metrics_list):
    # Assuming the length of lists for all agents and all epochs are the same
     # Calculate the average training loss across all agents for each epoch
    avg_train_loss = np.mean(metrics_list['train_loss'], axis=0)
    
    # Calculate the average test accuracy across all agents for each epoch
    avg_test_accuracy = np.mean(metrics_list['train_accuracy'], axis=0)
    return avg_train_loss, avg_test_accuracy

# Plotting
def plot_metrics(metrics_dict, title, ylabel):
    plt.figure(figsize=(10, 6))
    for label, metric in metrics_dict.items():
        plt.plot(metric, label=label)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    save_path = os.path.join(os.getcwd(), 'graph_result', f'{ylabel}_lr0.001.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


# Modify the plot_metrics function to plot against time
def plot_metrics_time(metrics, title, ylabel, time_per_epoch):
    plt.figure(figsize=(10, 6))
    for idx, (key, metric) in enumerate(metrics.items()):
        # Calculate cumulative time for each epoch for the current strategy
        epochs = np.arange(1, len(metric) + 1)
        time = epochs * (time_per_epoch[key]/60)
        plt.plot(time, metric, label=os.path.basename(file_paths[idx]).split('.')[0])
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    save_path = os.path.join(os.getcwd(), 'graph_result_time', f'{ylabel}_over_time_lr0.001.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


with open("/scratch2/tingyang/DFS_Topo/tau_results.pkl", 'rb') as file:
    time_per_epoch_diction = pickle.load(file)

print(time_per_epoch_diction)


afaf = {'Roofnet_CIFAR10_SCA23_1': 4532.774887751348, 'Roofnet_CIFAR10_SCA23_2': 4532.774883521915, 'Roofnet_CIFAR10_SCA23_3': 4532.774881939509, 
        'Roofnet_CIFAR10_SDRLambda2Ew_1': 6043.699793544735, 'Roofnet_CIFAR10_SDRLambda2Ew_2': 6043.6998125462205, 
        'Roofnet_CIFAR10_SDRLambda2Ew_3': 6043.699812221992, 'Roofnet_CIFAR10_SDRRhoEw_1': 4532.797657392192, 
        'Roofnet_CIFAR10_BoydGreedy_1': 4532.774885300265, 'ring': 3021.8499280883507, 'random': 3021.850796229334, 'clique': 13598.324506617035, 
        'prim': 3021.8499001230703}
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
print(all_avg_test_accuracy)
plot_metrics(all_avg_train_loss, 'Average Training Loss Across All Agents', 'Loss')
plot_metrics(all_avg_test_accuracy, 'Average Test Accuracy Across All Agents', 'Accuracy')
plot_metrics_time(all_avg_train_loss, 'Average Training Loss Across All Agents Over Time', 'Loss', time_per_epoch_diction)
plot_metrics_time(all_avg_test_accuracy, 'Average Test Accuracy Across All Agents Over Time', 'Accuracy', time_per_epoch_diction)