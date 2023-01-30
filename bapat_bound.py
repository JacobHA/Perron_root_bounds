import numpy as np
from linalg_functions import Perron_info, col_sums, row_sums

# Write a decorator that will test the functoin for the transpose of B also
# and return the minimum of the two values:


def improve_upper(func, **kwargs):
    def wrapper(A, B, **kwargs):
        return min(func(A, B, **kwargs), func(A, B.T, **kwargs))
    return wrapper


def improve_lower(func, **kwargs):
    def wrapper(A, B, **kwargs):
        return max(func(A, B, **kwargs), func(A, B.T, **kwargs))
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


@ improve_lower
def first_iter(A, B):
    rA, uA, vA = Perron_info(A)
    thetaA = -np.log(rA)
    rB_bound = karlin_ost(A, B)
    thetaB_bound = -np.log(rB_bound)
    # Grab row sums (exponentiated energies) of B:
    expd_energy_B = col_sums(B)
    expd_energy_A = col_sums(A)
    new_energy_diff = -(np.log(expd_energy_B) - np.log(expd_energy_A))
    u1v1 = uA * vA * np.exp(new_energy_diff)
    u1v1 /= u1v1.sum()
    correction = np.log(np.dot(u1v1, np.exp(-new_energy_diff)))

    return np.exp(-thetaB_bound + correction)


@ improve_upper
def conjecture(A, B):
    rA, uA, vA = Perron_info(A)
    thetaA = -np.log(rA)
    rB_bound = karlin_ost(A, B)
    thetaB_bound = -np.log(rB_bound)
    # Grab row sums (exponentiated energies) of B:
    expd_energy_B = col_sums(B)
    expd_energy_A = col_sums(A)
    new_energy_diff = -(np.log(expd_energy_B) - np.log(expd_energy_A))
    u1v1 = uA * vA * np.exp(-new_energy_diff)
    u1v1 /= u1v1.sum()
    correction = np.dot(u1v1, -new_energy_diff)
    return np.exp(-thetaA + correction)


def main():
    dimension = 100
    A = np.random.rand(dimension, dimension)
    B = np.random.rand(dimension, dimension)
    rB, _, _ = Perron_info(B)
    print(f"Lower bound on Perron root of B: {bapat(A,B)}")
    # print(f"Other lower bound on Perron root of B: {bapat(A,B.T)}")

    print(f"Lower bound on Perron root of B: {karlin_ost(A,B)}")
    # print(f"Other lower bound on Perron root of B: {karlin_ost(A, B.T)}")

    print(f"Lower bound on Perron root of B: {first_iter(A,B)}")
    # print(f"Other lower bound on Perron root of B: {first_iter(A, B.T)}")
    print(f"True Perron root of B: {rB}")


def test_conjecture1(dimension=100):
    # Test the conjecture that the first iteration is always an upper bound
    # on the Perron root of B:
    for sample in range(100):
        print(sample)
        A = np.random.rand(dimension, dimension)
        B = np.random.rand(dimension, dimension)
        rB, _, _ = Perron_info(B)
        if conjecture(A, B) < rB:
            print(f"Conjecture failed on sample {sample}")
            print(f"First iteration: {conjecture(A, B)}")
            print(f"True Perron root: {rB}")

            return


def test_conjecture2():
    # Conjecture that the first iteration is tighter than the Karlin-Ost bound
    dimension = 100
    for sample in range(50):
        print(sample)
        A = np.random.rand(dimension, dimension)
        B = np.random.rand(dimension, dimension)
        rB, _, _ = Perron_info(B)
        if first_iter(A, B) < karlin_ost(A, B):
            print(f"Conjecture failed on sample {sample}")
            print(f"First iteration: {first_iter(A, B)}")
            print(f"Karlin-Ost bound: {karlin_ost(A, B)}")
            print(f"True Perron root: {rB}")
            return


def test_tightness():
    # Test the tightness of each bound on 1000 random samples:
    dimension = 100
    bapat_errors = []
    karlin_ost_errors = []
    first_iter_errors = []
    for sample in range(10):
        A = np.random.rand(dimension, dimension)
        B = np.random.rand(dimension, dimension)
        rB, _, _ = Perron_info(B)
        bapat_bound = bapat(A, B)
        karlin_ost_bound = karlin_ost(A, B)
        first_iter_bound = first_iter(A, B)
        bapat_errors.append((rB - bapat_bound)/rB)
        karlin_ost_errors.append((rB - karlin_ost_bound)/rB)
        first_iter_errors.append((rB - first_iter_bound)/rB)
    print(f"Average error of Bapat bound: {np.mean(bapat_errors)}")
    print(f"Average error of Karlin-Ost bound: {np.mean(karlin_ost_errors)}")
    print(f"Average error of first iter bound: {np.mean(first_iter_errors)}")


if __name__ == "__main__":
    main()
    # Failing for dim < 10
    # test_conjecture1(dimension=5)
    # test_tightness()
