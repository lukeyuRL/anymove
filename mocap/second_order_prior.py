"""Exact O(T) pentadiagonal Cholesky solve for a Gaussian acceleration prior.

Computes argmin_x ||x-y||^2 + gamma*||D2*x||^2 with D2=[1,-2,1].
Using a banded solver avoids a separate dense T x T inverse for every bone
axis, which becomes expensive on 92-joint Dino motion sequences.
"""
import numpy as np


def solve_second_order(observed, gamma):
    y = np.asarray(observed, dtype=np.float64)
    if (y.ndim != 1 or y.size < 3 or not np.isfinite(y).all()
            or not np.isfinite(gamma) or gamma < 0):
        raise ValueError("invalid second-order input or regularization")
    if gamma == 0:
        return y.copy()

    t = len(y)
    diagonal = np.ones(t, dtype=np.float64)
    diagonal[:-2] += gamma
    diagonal[1:-1] += 4 * gamma
    diagonal[2:] += gamma
    first = np.zeros(t, dtype=np.float64)
    first[1:] -= 2 * gamma
    first[2:-1] -= 2 * gamma
    second = np.zeros(t, dtype=np.float64)
    second[2:] = gamma

    # L has exactly two lower diagonals; A=L@L.T is SPD for gamma >= 0.
    l0, l1, l2 = np.zeros(t), np.zeros(t), np.zeros(t)
    for i in range(t):
        if i >= 2:
            l2[i] = second[i] / l0[i-2]
        if i >= 1:
            l1[i] = (first[i] - (l2[i]*l1[i-1] if i >= 2 else 0.)) / l0[i-1]
        pivot = diagonal[i] - l1[i]**2 - l2[i]**2
        if pivot <= 0:
            raise ValueError("non-positive Cholesky pivot in motion prior")
        l0[i] = np.sqrt(pivot)

    z = np.zeros(t)
    for i in range(t):
        z[i] = (y[i] -
                (l1[i]*z[i-1] if i else 0.) -
                (l2[i]*z[i-2] if i >= 2 else 0.)) / l0[i]
    result = np.zeros(t)
    for i in range(t-1, -1, -1):
        result[i] = (z[i] -
                     (l1[i+1]*result[i+1] if i+1 < t else 0.) -
                     (l2[i+2]*result[i+2] if i+2 < t else 0.)) / l0[i]
    return result
