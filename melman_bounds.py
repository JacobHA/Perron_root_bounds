# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:18:34 2022
@author: jacob
"""
# Import required packages
from numpy import ndarray, any, identity, diag, inf, fill_diagonal, sqrt, zeros_like
from numpy import random
from numpy.linalg import eigvals
import time

def Melman_bounds(A):
    """
    Calculates the bounds on the Perron root, derived by A. Melman for a given non-negative matrix A.

    Link to paper (DOI):
    https://doi.org/10.1080/03081087.2012.667096

    Parameters
    ----------
    A : numpy.ndarray
        A non-negative matrix.

    Returns
    -------
    numpy.ndarray
        The upper and lower bounds for A's Perron root.
        
    """

    # Check that M is a 2D numpy array
    if not isinstance(A, ndarray):
        raise TypeError("M must be a numpy array.")
    if len(A.shape) != 2:
        raise ValueError("M must be a 2D array.")

    # Check if M is a square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input A must be a square matrix.")

    # Check that M is non-negative
    if any(A < 0):
        raise ValueError("Input A must be non-negative.")        


    # R_i(A) in Melman's notation:
    R = A.sum(axis=1) # Calculate the row sums of M

    # R'_i(A) in Melman's notation:
    Rp = R - diag(A) 

    # R''_i(A) in Melman's notation:
    Rpp = Rp - A

    # Calculate the bounds on the Perron root, based on eq. (5) and (6) in Melman's paper
    bounds = zeros_like(A)
    for i,A_ii in enumerate(diag(A)):
        for j,A_jj in enumerate(diag(A)):
            if i != j:
                bounds[i][j] = A_ii + A_jj + Rpp[i][j] + sqrt( (A_ii - A_jj + Rpp[i][j])**2 + 4*A[i][j]*Rp[j])

    # Mask the diagonal elements (i!=j)
    fill_diagonal(bounds, -inf)
    LB = 0.5*bounds.max(axis=0).min()
    fill_diagonal(bounds, inf)
    UB = 0.5*bounds.min(axis=0).max()
    return (LB, UB)


def main():
    # Test the bounds
    # Generate a random matrix, 1_000x1_000 with range 0-1 (uniform distribution)
    A = random.rand(1_000, 1_000) 
    start = time.process_time()
    LB, UB = Melman_bounds(A)
    print(f"Time taken for bound calculation: {time.process_time() - start}s")
    # Print the bounds wrt to true root:
    print(f"{LB:.2f} <= {abs(eigvals(A)[0]):.2f} <= {UB:.2f}")


if __name__ == "__main__":
    main()
    