import os
import pickle
import scipy.io

# Define a dictionary mapping each algorithm to the expected variable names
# in the corresponding .mat file.
# Each .mat file is assumed to contain:
#   - a mixing matrix variable (e.g., W_clique, W_mst, etc.)
#   - non-routing tau variable (e.g., tau_clique_norouting, tau_mst_norouting, etc.)
#   - routing tau variable (e.g., tau_clique_routing, tau_mst_routing, etc.)
algorithms = {
    "Clique": {
        "mix_var": "W_clique",
        "tau_norouting": "tau_clique_norouting",
        "tau_routing": "tau_clique_routing"
    },
    "MST": {
        "mix_var": "W_mst",
        "tau_norouting": "tau_mst_norouting",
        "tau_routing": "tau_mst_routing"
    },
    "SCA": {
        "mix_var": "W_SCA",
        "tau_norouting": "tau_SCA_norouting",
        "tau_routing": "tau_SCA_routing"
    },
    "Ring": {
        "mix_var": "W_ring",
        "tau_norouting": "tau_ring_norouting",
        "tau_routing": "tau_ring_routing"
    },
    "SMMD_SM": {
        "mix_var": "W_SMMD",
        "tau_norouting": "tau_SMMD_norouting",
        "tau_routing": "tau_SMMD_routing"
    }
}

# Dictionaries to store tau values (non-routing and routing)
tau_norouting_dict = {}
tau_routing_dict = {}

# Folder where your .mat files are stored.
# Adjust this path if needed (e.g., "data").
mat_folder = ""

# Process each algorithm
for algo, var_names in algorithms.items():
    # Construct the file name (for example: Roofnet_mixing_matrix_Clique.mat)
    filename = f"Roofnet_mixing_matrix_{algo}.mat"
    filepath = os.path.join(mat_folder, filename)

    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist. Skipping {algo}.")
        continue

    print(f"Processing {filepath} ...")
    # Load the MAT file
    mat_data = scipy.io.loadmat(filepath)

    # Extract the mixing matrix (e.g., W_clique, W_mst, etc.)
    mix_var = var_names["mix_var"]
    if mix_var not in mat_data:
        print(f"Variable '{mix_var}' not found in {filename}. Skipping {algo}.")
        continue
    mixing_matrix = mat_data[mix_var]

    # Save the mixing matrix in a pickle file with the same base name (.pkl)
    pkl_filename = filename.replace(".mat", ".pkl")
    pkl_filepath = os.path.join(mat_folder, pkl_filename)
    with open(pkl_filepath, "wb") as f:
        pickle.dump(mixing_matrix, f)
    print(f"Saved mixing matrix to {pkl_filepath}")

    # Extract the tau values for non-routing and routing.
    tau_nr_var = var_names["tau_norouting"]
    tau_r_var = var_names["tau_routing"]

    if tau_nr_var not in mat_data:
        print(f"Variable '{tau_nr_var}' not found in {filename}.")
        continue
    if tau_r_var not in mat_data:
        print(f"Variable '{tau_r_var}' not found in {filename}.")
        continue

    # In MATLAB files the tau values might be stored as a NumPy array.
    # If they are a single number, we extract it by using .item() method.
    tau_norouting = mat_data[tau_nr_var]
    tau_routing = mat_data[tau_r_var]
    if tau_norouting.size == 1:
        tau_norouting = tau_norouting.item()
    if tau_routing.size == 1:
        tau_routing = tau_routing.item()

    tau_norouting_dict[algo] = tau_norouting
    tau_routing_dict[algo] = tau_routing

# Output the dictionaries
print("\nNon-routing tau dictionary:")
print(tau_norouting_dict)

print("\nRouting tau dictionary:")
print(tau_routing_dict)

# Optionally, save the tau dictionaries as pickle files
with open(os.path.join(mat_folder, "tau_norouting_dict.pkl"), "wb") as f:
    pickle.dump(tau_norouting_dict, f)
with open(os.path.join(mat_folder, "tau_routing_dict.pkl"), "wb") as f:
    pickle.dump(tau_routing_dict, f)

print("\nDone processing all files.")
