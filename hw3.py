import numpy as np
from matplotlib import pyplot as plt

x1 = np.arange(-np.pi/2, np.pi/2, 0.01)
y1_true = 0.5 + np.sin(x1)
y1_taylor = 0.5 + x1 - x1**3 / 6

y1_quadratic = 0.5 + 0.761*x1 #part C
y1_LSE = 0.5 + x1*24/np.pi**3   #part D

plt.plot(x1, y1_true, label='Analytical')
plt.plot(x1, y1_taylor, linestyle = '--', label='Taylor Approximation (3rd Order)')

plt.plot(x1, y1_quadratic, color='purple', linestyle='--', label='Best Uniform Approimation')
plt.plot(x1, y1_LSE, color='green', linestyle='--', label='Least Squares Approximation')
#plt.plot(x1, -np.sin(x1), label='Second Deriv')
#plt.plot(x1, 0.5 + 4 * x1**2 / np.pi**2)
#plt.plot(x1, 0.3*(x1+1.5)**2 - 0.5)
plt.legend(loc='upper right')
plt.show()

l2_err_1c = 0.0
max_err_1c = 0.0
l2_err_1d = 0.0
max_err_1d = 0.0
for i in range(len(x1)):
    err_1c = y1_true[i] - y1_quadratic[i]
    if abs(err_1c) > max_err_1c:
        max_err_1c = err_1c
    l2_err_1c += (err_1c)**2

    err_1d = y1_true[i] - y1_LSE[i]
    if abs(err_1d) > max_err_1d:
        max_err_1d = err_1d
    l2_err_1d += (err_1d)**2
print("===== PROBLEM 1 =====\n")
print("Maximum Instantaneous Error for Quadratic Approximation:\t", max_err_1c)
l2_err_1c = np.sqrt(l2_err_1c)
print("Cumulative Error for Quadratic Approximation:\t\t", l2_err_1c)

print("Maximum Instantaneous Error for Least Squares Approximation:\t", max_err_1d)
l2_err_1c = np.sqrt(l2_err_1c)
print("Cumulative Error for Least Squares Approximation:\t\t", l2_err_1d)

print("\n\n===== PROBLEM 2 =====\n")

prob2file = open('problem2.txt', 'r')
p2string = prob2file.read()
p2_y = np.array(p2string.split(" "), dtype='float64').reshape(101,1)

p2_x = np.linspace(0,1,num=101)

p2_ones = np.ones(len(p2_x))
p2_term3 = np.array(np.sin(6*np.pi*p2_x))
p2_term4 = np.array(np.cos(p2_x))
p2_term5 = np.array(p2_x ** 2)
p2_term6 = np.array(p2_x ** 3)
p2_term7 = np.array(p2_x ** 4)
p2_term8 = np.array(p2_x ** 5)

A_p2 = np.array([p2_x, p2_term3]).T

print(p2_y.shape)

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


     x = V.dot(S.dot(U_T.dot(b)[:len(S[1]),:]))

     print("X vector (SVD approx):\n", x)

     return x

p2_coeffs = SVD_leastSquares_solution(A_p2, p2_y)
p2_est = A_p2.dot(p2_coeffs)

plt.plot(p2_x, p2_y, label = 'Original Function')
plt.plot(p2_x, p2_est, linestyle='--', label='Approximation')
plt.legend(loc="lower right")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Problem 2 - Aprroximation Using Basis Functions")
plt.show()

# Problem 3 was all done on paper (with the assistance of a calculator for the large integral)

# I'll do problem 4 if I have time in August
