import numpy as np
from linalg_functions import Perron_info, col_sums, row_sums

# Write a decorator that will test the functoin for the transpose of B also
# and return the minimum of the two values:


def improve_upper(func):
    def wrapper(A, B):
        return min(func(A, B), func(A, B.T))
    return wrapper


def improve_lower(func):
    def wrapper(A, B):
        return max(func(A, B), func(A, B.T))
    return wrapper


@ improve_lower
def bapat(A, B):
    # From R. Bapat
    rA, uA, vA = Perron_info(A)
    product = np.prod((B/A)**(A * np.outer(uA, vA) / rA))

    return rA*product


@ improve_lower
def karlin_ost(A, B):
    # First iteration is Karlin-Ost Bound:
    rA, uA, vA = Perron_info(A)
    thetaA = -np.log(rA)
    # Grab row sums (exponentiated energies) of B:
    expd_energy_B = col_sums(B)
    expd_energy_A = col_sums(A)
    energy_diff = -(np.log(expd_energy_B) - np.log(expd_energy_A))
    return np.exp(-thetaA - np.dot(uA * vA, energy_diff))


def main():
    dimension = 100
    A = np.random.rand(dimension, dimension)
    B = np.random.rand(dimension, dimension)
    rB, _, _ = Perron_info(B)
    print(f"Lower bound on Perron root of B: {bapat(A,B)}")
    print(f"Lower bound on Perron root of B: {karlin_ost(A,B)}")
    print(f"True Perron root of B: {rB}")


if __name__ == "__main__":
    main()
