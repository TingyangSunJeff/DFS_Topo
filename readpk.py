#!/usr/bin/env python3
import numpy as np
import pickle

def main():
    m = 10
    # Create a 10x10 identity matrix.
    I = np.eye(m)
    
    # Save the identity matrix to a pickle file.
    with open('mixing_matrix_identity.pkl', 'wb') as f:
        pickle.dump(I, f)
    
    print("Saved a 10x10 identity matrix to mixing_matrix_identity.pkl")

if __name__ == '__main__':
    main()
