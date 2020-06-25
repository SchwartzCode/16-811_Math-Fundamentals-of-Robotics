import numpy as np
from scipy.linalg import lu, solve


A_3a = np.array([ [2, 2, 5],
               [1, 1, 5],
               [3, 2, 5] ])

b_3a = np.array([5, -5, 0]).reshape(3,1)

A_3b = np.array([ [-3, -4, -1],
               [2, 3, 1],
               [3, 5, 2] ])

b_3b = np.array([3, 5, 1]).reshape(3,1)

A_3c = np.array([ [1, -1, 0],
                  [0, 1, -2],
                  [1, 0, -2] ])

b_3c = np.array([2, 3, 5]).reshape(3,1)


def LDU_decomp(A):
# This almost definitely doesn't work quite right
    i = 0
    max = len(A) - 1
    L_old = 1

    while (i < max):
        L = np.identity(len(A))
        for j in range(i+1 , len(A)):
            L[j,i] = - A[j,i] / A[i,i]
        #print("before:", A, "\n", L, "\n")
        A = L.dot(A)
        L_old = L.dot(L_old)
        #print(A, "\n\n")
        i += 1

    D = np.identity(len(A))
    for i in range(len(A)):
        if A[i,i] != 1:
            D[i,i] = A[i,i]
            #A[i,i] = 1

    L_fin = np.linalg.inv(L_old)

    return L_fin, A

L, U = LDU_decomp(A_3a)

print("PROBLEM 1\n================\nOriginal Matrix:")
print(A_3a)
print("L:\n", L)
#print("D:\n", D)
print("U:\n", U)
print("L*U:\n", L.dot(U))

A1 = np.array([ [1, -1, 0], [0, 2, -1], [1, 0, -0.5] ])
A2 = np.array([ [-1, 1, 0, 0], [-1, 0, 1, 0], [0, -4, 1, 0], [0, -1, 0, 1], [0, 0, -2, 1] ])
A3 = np.array([ [2, 2, 5], [1, 1, 5], [3, 2, 5] ])

def decompA_LDU_SVD(A):
    print("A original:\n", A)

    P, L, U = lu(A)
    print("\nP:\n", P, "\nL:\n", L, "\nU:\n", U)
    print("P inverted:\n", np.linalg.inv(P))
    A_recomp = P.dot(L.dot(U))

    print("A Recomposed (PLU):\n", A_recomp)

    U, S, Vh = np.linalg.svd(A)

    S = np.diag(S)

    print("\nU:\n", U, "\nS:\n", S, "\nVh:\n", Vh)

    #print("U:\n", U, "\nSigma:\n", S, "\nV_T:\n", Vh)
    A_recomp_SVD = U[:,:len(S[0])].dot(S.dot(Vh))
    for i in range(len(A_recomp_SVD)):
        for j in range(len(A_recomp_SVD[0])):
            A_recomp_SVD[i,j] = round(A_recomp_SVD[i,j], 2)

    print("A recomposed (SVD):\n", A_recomp_SVD)


    return

print("\n\nPROBLEM 2\n================\n")
decompA_LDU_SVD(A1)
decompA_LDU_SVD(A2)
decompA_LDU_SVD(A3)

decompA_LDU_SVD(A_3a)


# I did part A of question 3 on paper

def SVD_leastSquares_solution(A, b):
     U, S, Vh = np.linalg.svd(A)

     for i in range(len(S)):
         if abs(S[i]) < 1e-5:
             S[i] = 0
         else:
             S[i] = 1 / S[i]

     S = np.diag(S)

     print("\nU:\n", U, "\nS:\n", S, "\nVh:\n", Vh)

     U_T = np.transpose(U)
     V = np.transpose(Vh)

     x = V.dot(S.dot(U_T.dot(b)))

     print("X vector (SVD approx):\n", x)

     return x

SVD_leastSquares_solution(A_3c, b_3c)
