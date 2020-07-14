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



plt.plot(x1, y1a, label='Analytical')
plt.plot(x1, y1b, label='Euler Estimate', linestyle='--')
plt.xlim(1,2)
plt.title("Problem 1 - Estimating Differential Equation")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='lower right')
plt.show()
