import pickle
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import os
import numpy as np

# ---------------------------
# Load the tau dictionaries from pickle files.
tau_without_path = os.path.join("tau_dict", "tau_norouting_dict.pkl")
tau_with_path    = os.path.join("tau_dict", "tau_routing_dict.pkl")

with open(tau_without_path, "rb") as f:
    time_dict_without_route = pickle.load(f)

with open(tau_with_path, "rb") as f:
    time_dict_with_route = pickle.load(f)

# ---------------------------
# Define the desired order and legend names.
# For example, you want the legend to show:
#   "Clique", "Prim", "Ring", "SCA", and "FMMD"
desired_order = ["Clique", "Prim", "Ring", "SCA", "FMMD"]

# Mapping for possible raw (cleaned) names to your desired legend names.
legend_mapping = {
    "MST": "Prim",
    "Prim": "Prim",
    "SMMD_SM": "FMMD",
    "FMMD": "FMMD"
    # Any name not listed here will be used as is.
}

def remap_tau_dict(original_tau_dict, mapping):
    """
    Returns a new dictionary with keys remapped according to the mapping.
    For example, if the original tau dict has key "MST" and mapping maps "MST" to "Prim",
    then the returned dictionary will have key "Prim" with the corresponding value.
    """
    remapped = {}
    for key, value in original_tau_dict.items():
        new_key = mapping.get(key, key)
        remapped[new_key] = value
    return remapped

# Remap the tau dictionaries so that keys reflect the desired legend names.
time_dict_without_route = remap_tau_dict(time_dict_without_route, legend_mapping)
time_dict_with_route = remap_tau_dict(time_dict_with_route, legend_mapping)

# ---------------------------
# Define the file paths for result pickle files.
all_file_paths = [
    "result_for_resnet_Roofnet_mixing_matrix_Clique_0niid_schedule.pkl",
    "result_for_resnet_Roofnet_mixing_matrix_MST_0niid_schedule.pkl",
    "result_for_resnet_Roofnet_mixing_matrix_Ring_0niid_schedule.pkl",
    "result_for_resnet_Roofnet_mixing_matrix_SCA_0niid_schedule.pkl",
    "result_for_resnet_Roofnet_mixing_matrix_SMMD_SM_0niid_schedule.pkl"
]

# ---------------------------
def clean_name(raw_name):
    """
    Given a raw algorithm key (e.g., "Clique_0niid_schedule" or "SMMD_SM_0niid_schedule"),
    remove extra tokens ("0niid", "schedule", etc.) so that only the essential algorithm name remains.
    For example, "Clique_0niid_schedule" becomes "Clique" and "SMMD_SM_0niid_schedule" becomes "SMMD_SM".
    """
    tokens = raw_name.split("_")
    new_tokens = []
    for token in tokens:
        token_lower = token.lower()
        # Stop adding tokens once a known extra token is reached.
        if "niid" in token_lower or "schedule" in token_lower:
            break
        new_tokens.append(token)
    return "_".join(new_tokens)

def map_legend_name(raw_key):
    """
    Clean the raw key and then map it to the desired legend name if defined.
    """
    cleaned = clean_name(raw_key)
    return legend_mapping.get(cleaned, cleaned)

def update_results_and_order(categorized_results, order, file_paths):
    """
    For each file in file_paths, remove the prefix 'result_for_resnet_' and suffix '.pkl'.
    Also remove the "mixing_matrix_" part so that only the raw algorithm key remains.
    Update:
      - categorized_results: a dict mapping raw keys to themselves.
      - order: a list of raw keys in order of appearance.
    """
    prefix = "result_for_resnet_"
    marker = "mixing_matrix_"
    for fp in file_paths:
        file_core = fp[len(prefix):] if fp.startswith(prefix) else fp
        if file_core.endswith(".pkl"):
            file_core = file_core[:-4]
        if marker in file_core:
            file_core = file_core.split(marker)[-1]
        if file_core not in categorized_results:
            categorized_results[file_core] = file_core
        if file_core not in order:
            order.append(file_core)
    return categorized_results, order

def read_metrics(file_path):
    with open(file_path, 'rb') as file:
        metrics_history = pickle.load(file)
    return metrics_history

def average_metrics(metrics_dict):
    """
    Calculate the average training loss and test accuracy across agents.
    Expects metrics_dict to have keys 'train_loss' and 'test_accuracy'.
    """
    avg_train_loss = np.mean(metrics_dict['train_loss'], axis=0)
    avg_test_accuracy = np.mean(metrics_dict['test_accuracy'], axis=0)
    return avg_train_loss, avg_test_accuracy

def plot_metrics(folder_name, data_type, threshold, metrics_dict, ylabel, network_type, extra_label, order_list, mode_str, plot_mode="png"):
    plt.figure(figsize=(12, 8))
    line_style = '-'
    # Build a new ordered dictionary using the desired_order.
    selected_metrics = {}
    for disp_name in desired_order:
        for raw_key in order_list:
            if map_legend_name(raw_key) == disp_name:
                selected_metrics[disp_name] = metrics_dict[raw_key]
                break
    if not selected_metrics:
        print("No matching metrics found for desired order!")
    
    convergence_epochs = {}
    for disp_label, metric in selected_metrics.items():
        plt.plot(metric, label=disp_label, linestyle=line_style)
        conv = np.where(metric <= threshold)[0]
        convergence_epochs[disp_label] = conv[0] if conv.size > 0 else 'Did not converge'
    plt.xlabel('Epoch', fontsize=21)
    plt.ylabel(ylabel, fontsize=21)
    plt.legend(loc='best', fontsize=21)
    plt.grid(True)
    save_path = os.path.join(os.getcwd(), folder_name, f'{ylabel}_{network_type}_{data_type}{extra_label}_{mode_str}.{plot_mode}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return convergence_epochs

def plot_metrics_time(folder_name, metrics, ylabel, tau_dict, network_type, x_axis_limit, order_list, mode_str, plot_mode="png"):
    """
    Plot metric evolution versus cumulative time using the tau dictionary.
    The final legend names and their order are determined by the desired_order list.
    """
    plt.figure(figsize=(12, 8))
    line_style = '-'
    selected_metrics = {}
    for disp_name in desired_order:
        for raw_key in order_list:
            if map_legend_name(raw_key) == disp_name:
                selected_metrics[disp_name] = metrics[raw_key]
                break

    for disp_label, metric in selected_metrics.items():
        # Look up the corresponding tau value using the final legend name.
        matched_tau = tau_dict.get(disp_label, None)
        if matched_tau is None:
            print(f"No matching tau value found for {disp_label} in tau dictionary. Skipping this curve.")
            continue
        epochs = np.arange(1, len(metric) + 1)
        time_cumulative = epochs * matched_tau
        plt.plot(time_cumulative, metric, label=disp_label, linestyle=line_style)
    plt.xlabel('Time (s)', fontsize=21)
    plt.ylabel(ylabel, fontsize=21)
    plt.legend(loc='best', fontsize=17)
    plt.grid(True)
    plt.xscale('log')
    plt.xlim(x_axis_limit)
    save_path = os.path.join(os.getcwd(), folder_name + "_time", f'{ylabel}_over_time_{network_type}_{mode_str}.{plot_mode}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# ---------------------------
# Auto-detect network topology and model/data type from a sample file name.
sample_file = all_file_paths[0]
if "Roofnet" in sample_file:
    network_type = "Roofnet"
elif "IAB" in sample_file:
    network_type = "IAB"
else:
    network_type = "Unknown"

if "resnet" in sample_file.lower():
    model = "resnet"
    data_type = "CIFAR10"
elif "cnn" in sample_file.lower():
    model = "cnn"
    data_type = "MNIST"
else:
    model = "unknown"
    data_type = "unknown"

# Set threshold based on data_type.
threshold = 0.15 if data_type == "MNIST" else 0.001

# ---------------------------
# Define the output folder name based on detected network and data types.
folder_name = f"graph_result_{network_type.lower()}_{data_type.lower()}"

# Mode string for file naming (set empty if not needed).
mode_str = ""

# ---------------------------
# Update the categorized_results dictionary based on file names.
categorized_results = {}
order = []
categorized_results, order = update_results_and_order(categorized_results, order, all_file_paths)
print("Raw categorized_results:")
print(categorized_results)
print("\nRaw order:")
print(order)

# ---------------------------
# Initialize dictionaries for the averaged metrics.
all_avg_train_loss = {}
all_avg_test_accuracy = {}

# Read metrics and compute averages.
for file_path in all_file_paths:
    metrics_history = read_metrics(file_path)
    avg_train_loss, avg_test_accuracy = average_metrics(metrics_history)
    # Extract raw algorithm key from file name.
    base_name = os.path.basename(file_path).split('.pkl')[0]
    if "result_for_resnet_" in base_name:
        base_name = base_name[len("result_for_resnet_"):]
    if "mixing_matrix_" in base_name:
        base_name = base_name.split("mixing_matrix_")[-1]
    key = base_name
    all_avg_train_loss[key] = avg_train_loss
    all_avg_test_accuracy[key] = avg_test_accuracy

# Determine the number of epochs as the minimum length across all results.
epoch = min([len(loss) for loss in all_avg_train_loss.values()])

# Trim each array to this epoch length.
for key in all_avg_train_loss:
    all_avg_train_loss[key] = all_avg_train_loss[key][:epoch]
    all_avg_test_accuracy[key] = all_avg_test_accuracy[key][:epoch]

# ---------------------------
# Plot metrics versus epoch.
plot_metrics(folder_name, data_type, threshold, all_avg_train_loss, 'Loss', network_type, "", order, mode_str)
plot_metrics(folder_name, data_type, threshold, all_avg_test_accuracy, 'Accuracy', network_type, "", order, mode_str)

# ---------------------------
# Compute x-axis limits using the non-routing tau values.
all_tau_without = list(time_dict_without_route.values())
max_time_without_overlay = epoch * max(all_tau_without)
min_start = min(all_tau_without)
range_extension_factor = 1.2
x_axis_limit = (min_start / range_extension_factor, max_time_without_overlay * range_extension_factor)

# Plot evolution versus cumulative time.
# For 'without overlay routing'
plot_metrics_time(folder_name, all_avg_train_loss, 'Loss', time_dict_without_route, f'{network_type}_{data_type}_without_overlay', x_axis_limit, order, mode_str)
plot_metrics_time(folder_name, all_avg_test_accuracy, 'Accuracy', time_dict_without_route, f'{network_type}_{data_type}_without_overlay', x_axis_limit, order, mode_str)

# For 'with overlay routing'
plot_metrics_time(folder_name, all_avg_train_loss, 'Loss', time_dict_with_route, f'{network_type}_{data_type}_with_overlay', x_axis_limit, order, mode_str)
plot_metrics_time(folder_name, all_avg_test_accuracy, 'Accuracy', time_dict_with_route, f'{network_type}_{data_type}_with_overlay', x_axis_limit, order, mode_str)
