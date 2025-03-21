#!/usr/bin/env python3
import pickle
import argparse
import numpy as np

def print_numpy_arrays(data, indent=0):
    prefix = " " * indent
    # If the data is a NumPy array, print its details
    if isinstance(data, np.ndarray):
        print(f"{prefix}Found NumPy array:")
        print(f"{prefix}  Type: {type(data)}")
        print(f"{prefix}  Shape: {data.shape}")
        print(f"{prefix}  Dtype: {data.dtype}")
        # Optionally print a small preview if the array is not too large
        if data.size <= 100:
            print(f"{prefix}  Contents: {data}")
        else:
            print(f"{prefix}  Contents (first few elements): {data.flatten()[:10]}")
    # If the data is a dictionary, check each key and value
    elif isinstance(data, dict):
        print(f"{prefix}Found dict with {len(data)} keys:")
        for key, value in data.items():
            print(f"{prefix}Key: {key}")
            print_numpy_arrays(value, indent+4)
    # If the data is a list or tuple, iterate over its elements
    elif isinstance(data, (list, tuple)):
        print(f"{prefix}Found {type(data).__name__} with {len(data)} elements:")
        print(data)
    # Otherwise, print the type of the data (if desired)
    else:
        print(f"{prefix}Type: {type(data)} (not a NumPy array)")

def main(pkl_file):
    # Load the pickle file
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    
    print(f"Inspecting pickle file: {pkl_file}\n")
    print_numpy_arrays(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Load a pickle file and print all NumPy arrays contained within it."
    )
    parser.add_argument("pkl_file", help="Path to the pickle (.pkl) file")
    args = parser.parse_args()
    main(args.pkl_file)
