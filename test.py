import numpy as np

# Your initial matrix
matrix = np.array([
    [6.66674570e-01, 3.55873952e-10, 4.65960537e-11, -4.73879340e-10, 3.33325430e-01],
    [3.55873952e-10, 4.99995354e-01, 5.00004645e-01, 3.55873952e-10, 1.91274667e-10],
    [4.65960537e-11, 5.00004645e-01, -9.41992221e-06, 4.65960537e-11, 5.00004775e-01],
    [-4.73879340e-10, 3.55873952e-10, 4.65960537e-11, 6.66674570e-01, 3.33325430e-01],
    [3.33325430e-01, 1.91274667e-10, 5.00004775e-01, 3.33325430e-01, -1.66655636e-01]
])

# Define a function to adjust the matrix
def adjust_matrix(matrix):
    adjusted_matrix = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if abs(matrix[i, j]) >= 0.1 and abs(matrix[i, j]) < 1:
                adjusted_matrix[i, j] = matrix[i, j]
    
    # Normalize each row
    row_sums = np.sum(adjusted_matrix, axis=1)
    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            adjusted_matrix[i, :] /= row_sums[i]
    
    # Normalize each column
    col_sums = np.sum(adjusted_matrix, axis=0)
    for j in range(len(col_sums)):
        if col_sums[j] > 0:
            adjusted_matrix[:, j] /= col_sums[j]
    
    return adjusted_matrix

# Adjust the matrix
adjusted_matrix = adjust_matrix(matrix)

# Print adjusted matrix
print("Adjusted Matrix:")
print(adjusted_matrix)

# Validate sums
row_sums = np.sum(adjusted_matrix, axis=1)
col_sums = np.sum(adjusted_matrix, axis=0)

print("Row sums:", row_sums)
print("Column sums:", col_sums)
