import numpy as np
from matplotlib import pyplot as plt

x1 = np.arange(-np.pi/2, np.pi/2, 0.01)
y1_true = 0.5 + np.sin(x1)
y1_taylor = 0.5 + x1 - x1**3 / 6
y1_quadratic = 0.5 + 2*x1/np.pi
y1_LSE = 0.5 + x1*24/np.pi**3

plt.plot(x1, y1_true, label='Analytical')
plt.plot(x1, y1_taylor, linestyle = '--', label='Taylor Approximation (3rd Order)')
#plt.plot(x1, -np.sin(x1), label='Second Deriv')
#plt.plot(x1, 0.5 + 4 * x1**2 / np.pi**2)
#plt.plot(x1, 0.3*(x1+1.5)**2 - 0.5)
plt.plot(x1, y1_quadratic, color='purple', linestyle='--', label='Best Quadratic Approimation (it\'s linear tho)')
plt.plot(x1, y1_LSE, color='green', linestyle='--', label='Least Squares Approximation')
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

print("Maximum Instantaneous Error for Quadratic Approximation:\t", max_err_1c)
l2_err_1c = np.sqrt(l2_err_1c)
print("Cumulative Error for Quadratic Approximation:\t\t", l2_err_1c)

print("Maximum Instantaneous Error for Least Squares Approximation:\t", max_err_1d)
l2_err_1c = np.sqrt(l2_err_1c)
print("Cumulative Error for Least Squares Approximation:\t\t", l2_err_1d)
