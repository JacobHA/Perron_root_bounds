import numpy as np
from numpy import random
from numpy.linalg import eigvals
from linalg_functions import row_sums, col_sums
import time


def two_row_sums(M):
    # defintion provided by doi: 10.3934/math.2020047
    # notation: M_i(A)
    row_vec = row_sums(M)
    return M @ row_vec


def avg_two_row_sums(M):
    # defintion provided by doi: 10.3934/math.2020047
    # notation: m_i(A)
    return two_row_sums(M) / row_sums(M)


def w_i_avg_of_avg(M):
    # defintion provided by doi: 10.3934/math.2020047
    # notation: w_i(A), eqn. 1.1
    m_j = avg_two_row_sums(M)
    numerator = M @ m_j

    return numerator / m_j  # triple check this


def auxiliary_quantities(M):
    # mask out diagonal
    mask = np.ones(M.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    max_nondiag = M[mask].max()
    min_nondiag = M[mask].min()

    diagonal = np.diag(M)
    max_diag, min_diag = diagonal.max(), diagonal.min()

    return max_nondiag, min_nondiag, max_diag, min_diag


def PSI_L(M, verbose=False):
    # THEOREM 4
    m_j = avg_two_row_sums(M)
    w_j = w_i_avg_of_avg(M)
    w_j = sorted(w_j, reverse=True)  # needed descending per eqn. 2.1
    N__, __, M__, __ = auxiliary_quantities(M)
    assert min(m_j) != 0, "Non-applicable when zero min."
    ratios = np.array([m / m_j for m in m_j])  # DOUBLE CHECK THIS
    mask = np.ones(ratios.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    b_m = ratios[mask].max()
    res_array = []
    for l in range(len(w_j)):
        w_l = w_j[l]
        sum_term = sum(w_j[:l-1] - w_l)
        if l == 0:
            sum_term = 0
        DELTA_L = (w_l - M__ + N__ * b_m) ** 2 + 4*N__*b_m * sum_term
        res_array.append(0.5*(w_l + M__ - N__ * b_m + np.sqrt(DELTA_L)))

    if verbose:
        return res_array
    else:
        return np.array(res_array).min()


def psi_n(M):
    m_j = avg_two_row_sums(M)
    w_j = w_i_avg_of_avg(M)
    w_j = sorted(w_j, reverse=True)  # needed descending per eqn. 2.1
    __, T__, __, S__ = auxiliary_quantities(M)
    assert min(m_j) != 0, "Non-applicable when zero min."
    c_m = np.nanmin(m_j) / np.nanmax(m_j)  # DOUBLE CHECK THIS
    DELTA_N = []
    for l in range(len(w_j)):
        w_l = w_j[l]
        sum_term = sum(w_j[:l-1] - w_l)
        if l == 0:
            sum_term = 0
        DELTA_N.append((w_l - S__ + T__ * c_m) ** 2 + 4*T__ * c_m * sum_term)

    return 0.5*(w_j + S__ - T__ * c_m + np.sqrt(DELTA_N))


def PHI_L_HAT(M, verbose=False):
    # THEOREM 6

    r_i = row_sums(M)
    r_i = sorted(r_i, reverse=True)  # needed descending per eqn. 2.1
    N__, __, M__, __ = auxiliary_quantities(M)

    res_array = []
    for n in range(len(r_i)):
        r_n = r_i[n]
        sum_term = sum(r_i[:n-1] - r_n)
        if n == 0:
            sum_term = 0

        res_array.append((r_n + M__ - N__)/2 +
                         np.sqrt(((r_n - M__ + N__)/2)**2 + N__ * sum_term))
    if verbose:
        return res_array
    else:
        return np.array(res_array).min()  # i.e. the best upper bound


def PHI_L_TILDE(M, verbose=False):
    N__, __, M__, __ = auxiliary_quantities(M)

    m_j = avg_two_row_sums(M)
    m_j = sorted(m_j, reverse=True)
    r_j = row_sums(M)

    ratios = np.array([r / r_j for r in r_j])  # DOUBLE CHECK THIS
    mask = np.ones(ratios.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    b = ratios[mask].max()

    res_array = []
    for l in range(len(m_j)):
        m_l = m_j[l]
        sum_term = sum(m_j[:l-1] - m_l)
        if l == 0:
            sum_term = 0
        res_array.append((m_l + M__ - N__*b)/2 +
                         np.sqrt(((m_l - M__ + N__*b)/2)**2 + N__*b*sum_term))

    if verbose:
        return res_array
    else:
        return np.array(res_array).min()  # i.e. the best upper bound


def phi_n_hat(M):
    r_i = row_sums(M)
    r_i = sorted(r_i, reverse=True)
    __, T__, __, S__ = auxiliary_quantities(M)
    for n in range(len(r_i)):
        r_n = r_i[n]
        sum_term = sum(r_i[:n-1] - r_n)
        if n == 0:
            sum_term = 0
    return (r_i + S__ - T__)/2 + np.sqrt(((r_i - S__ + T__)/2) ** 2 + T__ * sum_term)


def phi_n_tilde(M):
    m_j = avg_two_row_sums(M)
    r_m = row_sums(M)
    __, T__, __, S__ = auxiliary_quantities(M)
    m_j = sorted(m_j, reverse=True)
    for n in range(len(m_j)):
        m_n = m_j[n]
        sum_term = sum(m_j[:n-1] - m_n)
        if n == 0:
            sum_term = 0
    c = min(r_m)/max(r_m)
    return (m_j + S__ - T__*c)/2 + np.sqrt(((m_j - S__ + T__*c)/2)**2 + T__*c * sum_term)


def Adam_bounds(M):
    n = M.shape[0] - 1
    ub = min(PSI_L(M), PHI_L_HAT(M), PHI_L_TILDE(M))
    lb = max(psi_n(M)[n], phi_n_hat(M)[n], phi_n_tilde(M)[n])

    return lb, ub


def main():
    # Test the bounds
    # Generate a random matrix, 1_000x1_000 with range 0-1 (uniform distribution)
    A = random.rand(1_000, 1_000)
    start = time.process_time()
    LB, UB = Adam_bounds(A)
    print(f"Time taken for bound calculation: {time.process_time() - start}s")
    # Print the bounds wrt to true root:
    print(f"{LB:.2f} <= {abs(eigvals(A)[0]):.2f} <= {UB:.2f}")


if __name__ == "__main__":
    main()
