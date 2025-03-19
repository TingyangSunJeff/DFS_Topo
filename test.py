#!/usr/bin/env python3
import scipy.io as sio
import pickle
import os

def mat_to_pkl(mat_file, pkl_file, varname='W'):
    # Load the MATLAB .mat file into a dictionary
    data = sio.loadmat(mat_file)
    
    # Extract the variable (default 'W') from the dictionary
    if varname in data:
        W = data[varname]
    else:
        raise KeyError(f"Variable '{varname}' not found in {mat_file}")
    
    # Print the details of W
    print(f"Converting: {mat_file}")
    print(f"  Type: {type(W)}, shape: {W.shape}, dtype: {W.dtype}")
    
    # Save the variable W into a pickle file
    with open(pkl_file, 'wb') as f:
        pickle.dump(W, f)
    
    print(f"  Successfully saved {varname} from {mat_file} to {pkl_file}\n")

def main():
    # Hard-coded list of .mat files you want to convert
    mat_files = [
        "mixing_matrix_SMMD_PM_7T.mat",
        "mixing_matrix_SMMD_PM_8T.mat",
        "mixing_matrix_SMMD_PM_10T.mat",
        # "mixing_matrix_SMMD_SM_20T.mat",
        # "mixing_matrix_SMMD_SM_30T.mat",
        # "mixing_matrix_SMMD_SM_40T.mat"
    ]
    
    for mat_file in mat_files:
        # Derive a .pkl file name from the .mat file
        base_name = os.path.splitext(mat_file)[0]
        pkl_file = f"{base_name}.pkl"
        
        # Convert the .mat file to .pkl
        mat_to_pkl(mat_file, pkl_file, varname='W_SDP')

if __name__ == '__main__':
    main()
