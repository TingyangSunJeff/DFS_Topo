import pickle

def read_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        content = pickle.load(file)
    return content

file_path = '/scratch2/tingyang/DFS_Topo/mixing_matrix/mixing_matrix_Roofnet_CIFAR10_clique_1.pkl'  # Replace with your file path
content = read_pkl_file(file_path)

print(content)
