import numpy as np

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