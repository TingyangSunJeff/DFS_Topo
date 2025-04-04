#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT_PATH="dpsgd_cifar_new.py"

# Define an array with the paths of your mixing matrices
MIXING_MATRICES=(
    "mixing_matrix_MST.mat"
    "mixing_matrix_Clique.mat"
    # "mixing_matrix_Ring.pkl"
    # "mixing_matrix_SCA.pkl"
    # "mixing_matrix_SMMD_SM.pkl"
)

# Define an array with the GPU IDs you want to use
GPUS=(0 1)

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
    OUTPUT_FILE="result_for_resnet_${MATRIX_NAME}"
    
    # Use CUDA_VISIBLE_DEVICES to assign a GPU to this script run
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT_PATH $MATRIX_PATH $OUTPUT_FILE &
done

# Wait for all background processes to finish
wait
