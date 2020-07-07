import numpy as np
from matplotlib import pyplot as plt

x1 = np.arange(-np.pi/2, np.pi/2, 0.01)
y1_true = 0.5 + np.sin(x1)
y1_taylor = 0.5 + x1 - x1**3 / 6
y1_quadratic = 0.5 + 2*x1/np.pi

plt.plot(x1, y1_true, label='Analytical')
plt.plot(x1, y1_taylor, linestyle = '--', label='Taylor Approximation')
#plt.plot(x1, -np.sin(x1), label='Second Deriv')
#plt.plot(x1, 0.5 + 4 * x1**2 / np.pi**2)
#plt.plot(x1, 0.3*(x1+1.5)**2 - 0.5)
plt.plot(x1, y1_quadratic, color='purple', label='Best Quadratic Approimation (it\'s linear tho)')
plt.legend(loc='upper right')
plt.show()

l2_err_1c = 0.0
max_err_1c = 0.0
for i in range(len(x1)):
    err_1c = y1_true[i] - y1_quadratic[i]
    if abs(err_1c) > max_err_1c:
        max_err_1c = err_1c
    l2_err_1c += (err_1c)**2

print("Maximum Instantaneous Error for Quadratic Approximation:", max_err_1c)
l2_err_1c = np.sqrt(l2_err_1c)
print("Cumulative Error for Quadratic Approximation:", l2_err_1c)
