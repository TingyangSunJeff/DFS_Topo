#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT_PATH="test.py"

# Define an array with the paths of your mixing matrices
MIXING_MATRICES=(
    "/scratch2/tingyang/DFS_Topo/mixing_matrix/mixing_matrix_clique.pkl"
    "/scratch2/tingyang/DFS_Topo/mixing_matrix/mixing_matrix_prim.pkl"
    "/scratch2/tingyang/DFS_Topo/mixing_matrix/mixing_matrix_random.pkl"
    "/scratch2/tingyang/DFS_Topo/mixing_matrix/mixing_matrix_ring.pkl"
)

# Loop through the mixing matrices and call the Python script for each
for MATRIX_PATH in "${MIXING_MATRICES[@]}"; do
    # Extract the base name without the extension (e.g., "mixing_matrix_prim" from "mixing_matrix_prim.pkl")
    BASE_NAME=$(basename -- "$MATRIX_PATH")
    BASE_NAME="${BASE_NAME%.*}"

    # Extract the descriptor part of the name (e.g., "prim" from "mixing_matrix_prim")
    # This assumes the naming convention is consistent and uses "_" as a separator
    MATRIX_NAME=${BASE_NAME#mixing_matrix_}

    # Construct the output file name based on the matrix name
    OUTPUT_FILE="result_for_resnet_${MATRIX_NAME}.pkl"
    
    # Call the Python script with the current mixing matrix and output file name
    python $PYTHON_SCRIPT_PATH $MATRIX_PATH $OUTPUT_FILE
done
