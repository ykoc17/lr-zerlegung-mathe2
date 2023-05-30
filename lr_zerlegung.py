import numpy as np
import matplotlib as plt

def lr_ohne(A):
    n = len(A)
    L = np.zeros((n, n))
    for p in range(n): L[p][p] = 1
    R = A.copy()

    for i in range(n):
        pivot = R[i][i]
        for i_cont in range(i+1, len(R)):
            num = R[i_cont][i]
            if (pivot == 0): continue
            else:
                l_i = -(num / pivot)
                L[i_cont][i] = -l_i
                for j_cont in range(i, len(R)):
                    R[i_cont][j_cont] = R[i][j_cont]*l_i+R[i_cont][j_cont]
    return L, R

def lr_mit(A):
    n = len(A)
    P = np.identity(n) #Permutationsmatrix
    L = np.identity(n)
    R = A.copy()

    for i in range(n):

        absR_row_i = np.abs(R[:,i])
        sliced_absR_row_i = absR_row_i[i:]
        index_maxabsR_row_i = np.argmax(sliced_absR_row_i)
        if index_maxabsR_row_i != 0:
            P[:, [0, index_maxabsR_row_i]] = P[:, [index_maxabsR_row_i, 0]]
            R[[0, index_maxabsR_row_i]] = R[[index_maxabsR_row_i, 0]]

        pivot = R[i][i]
        for i_cont in range(i + 1, len(R)):
            num = R[i_cont][i]
            if (pivot == 0):
                continue
            else:
                l_i = -(num / pivot)
                L[i_cont][i] = -l_i
                for j_cont in range(i, len(R)):
                    R[i_cont][j_cont] += R[i][j_cont] * l_i
    return L, R, P

def linsolve_ohne(A, b):
    n = len(A)
    L, R = lr_ohne(A)
    x,  x_s = np.zeros(n), np.zeros(n)

    #Vorwärtseinsetzen
    x_s[0] = b[0]
    for i in range(1, n):
        sum = 0
        for k in range(i): sum += x_s[k]*L[i][k]
        x_s[i] = b[i] - sum

    #Rückwärtseinsetzen
    x[n-1] = x_s[n-1]/R[n-1][n-1]
    for i in reversed(range(0, n-1)):
        sum = 0
        for k in range(1, n-i): sum += x[n-k]*R[i][n-k]
        x[i] = (x_s[i] - sum)/R[i][i]

    return x

def linsolve_mit(A, b):
    n = len(A)
    L, R, P = lr_mit(A)
    x,  x_s = np.zeros(n), np.zeros(n)
    Pb = np.dot(P, b)

    #Vorwärtseinsetzen
    x_s[0] = Pb[0]
    for i in range(1, n):
        sum = 0
        for k in range(i): sum += x_s[k]*L[i][k]
        x_s[i] = Pb[i] - sum

    #Rückwärtseinsetzen
    x[n-1] = x_s[n-1]/R[n-1][n-1]
    for i in reversed(range(0, n-1)):
        sum = 0
        for k in range(1, n-i): sum += x[n-k]*R[i][n-k]
        x[i] = (x_s[i] - sum)/R[i][i]

    return x

#a)
A = np.array([[1., 1., 3.],
              [1., 2., 2.],
              [2., 1., 5.]])

b = np.array([2., 1., 1.])

L, R, P = lr_mit(A)
print(L)
print(R)
print(P)
print(linsolve_mit(A, b))

#b)
A = np.array([[0., 1., 0., -1.],
              [1., 2., 2., 1. ],
              [1., 1., 1., 1. ],
              [2., 1., -1., 2.]])

b = np.array([2., -1., -2., -11.])

#c)


#d)