import numpy as np
from matplotlib import pyplot as plt

print("===== PROBLEM 1 =====\n")

x1 = np.linspace(1,2,num=21)
y1a = np.array( (x1 - 1)**(1/3) )

y1b = np.array([1])

last = 1
for i in reversed(x1[:-1]):
    slope = 1 / (3 * last**2)
    last = last - 0.05 * slope
    y1b = np.concatenate(([last], y1b))

def rk4_step(x, y, h, f):
    k1 = h*f(y)
    k2 = h*f(y - k1/2)
    k3 = h*f(y - k2/2)
    k4 = h*f(y - k3)

    y_new = y - (k1 + 2*k2 + 2*k3 + k4) / 6

    return y_new

def rhs_1c(y):
    return 1 / (3 * y**2)

y1c = np.array([1])

for x in reversed(x1[:-1]):
    new_y = rk4_step(x, y1c[0], 0.05, rhs_1c)
    print(new_y)
    y1c = np.concatenate(([new_y], y1c))

plt.plot(x1, y1a, label='Analytical')
plt.plot(x1, y1b, label='Euler Estimate', linestyle='--')
plt.plot(x1, y1c, label='Runge-Kutta 4 Approx', linestyle='--')
plt.xlim(1,2)
plt.title("Problem 1 - Estimating Differential Equation")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='lower right')
plt.show()
