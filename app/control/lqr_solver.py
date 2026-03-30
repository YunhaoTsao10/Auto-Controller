import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are


def solve_lqr(A, B, Q, R, discrete: bool = False):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    n = A.shape[0]
    assert A.shape == (n, n)
    assert B.shape[0] == n
    m = B.shape[1]
    assert Q.shape == (n, n)
    assert R.shape == (m, m)

    if discrete:
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    else:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ (B.T @ P)

    return K