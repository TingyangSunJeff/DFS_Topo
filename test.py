import pickle

def read_and_print_pkl(file_path):
    """
    Read a pickle file and print its contents.

    Parameters:
    - file_path: The path to the pickle file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print("Contents of the pickle file:")
            print(data)
    except Exception as e:
        print(f"An error occurred while reading the pickle file: {e}")

# Example usage
file_path = '/scratch2/tingyang/DFS_Topo/mixing_matrix/mixing_matrix_Roofnet_CIFAR10_SCA23_1.pkl'  # Replace with your actual pickle file path
read_and_print_pkl(file_path)
