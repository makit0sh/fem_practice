# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def solve_gauss(A_array, b_vector):
    N = b_vector.size
    A = A_array.copy()
    b = b_vector.copy()
    w = np.zeros_like(A)
    for k in range(N-1):
        for i in range(k+1, N):
            w[i, k] = A[i, k]/A[k, k]
            b[i] = b[i] - w[i, k]*b[k]
            A[i, k] = 0
            for j in range(k+1, N):
                A[i, j] = A[i, j] - w[i, k]*A[k, j]
    for i in reversed(range(N)):
        b[i] = b[i] / A[i, i]
        for j in range(i):
            b[j] = b[j] - A[j, i]*b[i]
    return b

def solve_jacobi(A_array, b_vector, EPS=1e-5):
    x = np.zeros_like(b_vector)
    x_next = np.zeros_like(x)
    LmU = A_array.copy()
    for i in range(x.size):
        x_next[i] = b_vector[i]/A_array[i, i]
        LmU[i, i] = 0

    n=0
    while np.linalg.norm(x-x_next) > EPS:
        n += 1
        x = x_next
        x_next = np.zeros_like(x)
        x_next = -LmU.dot(x)+b_vector
        for i in range(x.size):
            x_next[i] = x_next[i]/A_array[i, i]
    return x_next, n


def main():
    A_array = np.array(
            [[5., -3, -1, 0, 2, 1],
                [0, -5, -3, 1, 0, 2],
                [-1, 3, -6, -2, 3, 3],
                [-1, 0, 3, 5, -2, -1],
                [0, 3, 3, -1, 5, -4],
                [2, 3, 2, 3, 2, -4]] )
    b_vector = np.array([ 16., 12, 12, 16, 18, 20 ])

    #A_array = np.array([[2, 3], [4, 7]])
    #b_vector = np.array([2, 6])
    print("Solved by numpy.linarg.solve (for comparison)")
    x = np.linalg.solve(A_array, b_vector)
    b_check = A_array.dot(x)
    print(x)
    print(b_check)

    print("ガウスの消去法で直接求解")
    x = solve_gauss(A_array, b_vector)
    b_check = A_array.dot(x)
    print(x)
    print(b_check)

    print("Jacobi法により反復法で求解")
    x,n = solve_jacobi(A_array, b_vector)
    b_check = A_array.dot(x)
    print(x)
    print(n)
    print(b_check)

if __name__ == '__main__':
    main()
