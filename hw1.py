import numpy as np
from scipy.linalg import lu


A = np.array([ [2, 7, 6],
               [9, 5, 1],
               [4, 3, 8] ])



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

L, U = LDU_decomp(A)

print("PROBLEM 1\n================\nOriginal Matrix:")
print(A)
print("L:\n", L)
#print("D:\n", D)
print("U:\n", U)
print("L*D*U:\n", L.dot(U))

A1 = np.array([ [1, -1, 0], [0, 2, -1], [1, 0, -0.5] ])
A2 = np.array([ [-1, 1, 0, 0], [-1, 0, 1, 0], [0, -4, 1, 0], [0, -1, 0, 1], [0, 0, -2, 1] ])
A3 = np.array([ [2, 2, 5], [1, 1, 5], [3, 2, 5] ])

def decompA_LDU_SVD(A):
    print("A original:\n", A)

    P, L, U = lu(A)
    #print("\nP:\n", P, "\nL:\n", L, "\nU:\n", U)
    A_recomp = P.dot(L.dot(U))

    print("A Recomposed (LDU):\n", A_recomp)

    U, S, Vh = np.linalg.svd(A)

    S = np.diag(S)

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
