#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT_PATH="dpsgd_mnist.py"

# Define an array with the paths of your mixing matrices
MIXING_MATRICES=(
    # "/scratch2/tingyang/DFS_Topo/mixing_matrix/mixing_matrix_Roofnet_MNIST_finf_BoydGreedy_1.pkl"
    # "/scratch2/tingyang/DFS_Topo/mixing_matrix/mixing_matrix_Roofnet_MNIST_finf_SCA23_1.pkl"
    # "/scratch2/tingyang/DFS_Topo/mixing_matrix/mixing_matrix_Roofnet_MNIST_finf_SCA23_2.pkl"
    # "/scratch2/tingyang/DFS_Topo/mixing_matrix/mixing_matrix_Roofnet_MNIST_finf_SDRLambda2Ew_1.pkl"
    # "/scratch2/tingyang/DFS_Topo/mixing_matrix/mixing_matrix_Roofnet_MNIST_finf_SDRLambda2Ew_2.pkl"
    # "/scratch2/tingyang/DFS_Topo/mixing_matrix/mixing_matrix_Roofnet_MNIST_finf_SDRRhoEw_1.pkl"
    "/scratch2/tingyang/DFS_Topo/mixing_matrix/mixing_matrix_Roofnet_MNIST_finf_SDRRhoEw_2.pkl"
)

# Define an array with the GPU IDs you want to use
GPUS=(0)

# Ensure the GPUs array is not longer than the MIXING_MATRICES array
if [ ${#GPUS[@]} -gt ${#MIXING_MATRICES[@]} ]; then
    echo "Error: More GPUs specified than mixing matrices."
    exit 1
fi

# Loop through the mixing matrices and call the Python script for each, on a different GPU
for i in "${!MIXING_MATRICES[@]}"; do
    MATRIX_PATH=${MIXING_MATRICES[$i]}
    GPU_ID=${GPUS[$i]}
    
    # Extract the base name without the extension (e.g., "mixing_matrix_prim" from "mixing_matrix_prim.pkl")
    BASE_NAME=$(basename -- "$MATRIX_PATH")
    BASE_NAME="${BASE_NAME%.*}"

    # Extract the descriptor part of the name (e.g., "prim" from "mixing_matrix_prim")
    MATRIX_NAME=${BASE_NAME#mixing_matrix_}

    # Construct the output file name based on the matrix name
    OUTPUT_FILE="result_for_cnn_${MATRIX_NAME}.pkl"
    
    # Use CUDA_VISIBLE_DEVICES to assign a GPU to this script run
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT_PATH $MATRIX_PATH $OUTPUT_FILE &
done

# Wait for all background processes to finish
wait
