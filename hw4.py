import numpy as np
from matplotlib import pyplot as plt

print("===== PROBLEM 1 =====")
print("  See plot")

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
    y1c = np.concatenate(([new_y], y1c))

def AB_step(h, last_y, f_vals):
    new_y = last_y - (h / 24) * (55*f_vals[0] - 59*f_vals[1] + 37*f_vals[2] - 9*f_vals[3])
    return new_y

y1d = np.array([1, 1.01639635681485, 1.03228011545637, 1.04768955317165])
f_vals_1d = rhs_1c(y1d)
for x in reversed(x1[:-1]):
    new_y = AB_step(0.05, y1d[0], f_vals_1d)
    f_vals_1d = np.concatenate(([rhs_1c(new_y)], f_vals_1d))
    y1d = np.concatenate(([new_y], y1d))

plt.plot(x1, y1a, label='Analytical')
plt.plot(x1, y1b, label='Euler Approx', linestyle='--')
plt.plot(x1, y1c, label='Runge-Kutta 4 Approx', linestyle='--')
plt.plot(x1, y1d[:-3], label='Adams-Bashforth Approx', linestyle='--')
plt.xlim(1,2)
plt.title("Problem 1 - Estimating Differential Equation")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='lower right')
plt.show()

# My IDE isn't great with printing out numbers in nice format so I just skipped
# the 'make a table' part of this question

print("\n\n===== PROBLEM 2 =====")

def p2_dx(x):
    return 3*x**2 - 4*x

def p2_dy(y):
    return 3*y**2 + 6*y

x2 = np.linspace(-5,5,num=1001)
y2 = np.linspace(-5,5,num=1001)



p2_contour = np.zeros((1001,1001))

for i in range(len(x2)):
    p2_contour[:,i] += p2_dx(x2[i])
    p2_contour[i,:] += p2_dy(y2[i])

p2_levels = np.linspace(-10,40,num=11)
p2_levels_f = np.linspace(-10,40,num=11)
plt.contour(x2,y2, p2_contour, p2_levels, colors='white')
plt.xlim(-2,3)
plt.ylim(-3.5,1.5)
plt.contourf(x2,y2,p2_contour, p2_levels_f)
plt.colorbar()
plt.title("Problem 2 - Contour Plot Describing Gradient")
plt.scatter([0,0,(4/3),(4/3)],[0,-2,0,-2], color='red', label='Local Minima')
plt.legend(loc='upper left')
plt.show()




p2b_x = 1
p2b_y = -1
rate = 0.1
counter = 0

while((abs(p2_dx(p2b_x))+ abs(p2_dy(p2b_y))) > 1e-2 ):
    print(counter, p2b_x, p2b_y)
    p2b_x -= rate*p2_dx(p2b_x)
    p2b_y -= rate*p2_dy(p2b_y)
    counter += 1

print("Starting at (1,-1), it takes", counter, " steps to get to a local minimum using a learning rate of", rate, ".")
print("The minimum is at (", p2b_x, ",", p2b_y, ")")
print("\nWhich local minima the function converges at depends on the step size")

"""
# This was wrong but it's a decent example of subplots so I'm gonna leave it here
plt.figure(figsize=[13,4])
plt.subplot(1,3,1)
#plt.plot(x2_zero1, y2, color='blue', label='Zeros of Partial X Deriv')
#plt.plot(x2_zero2, y2, color='blue')
#plt.plot(x2, y2_zero1, color='orange', label='Zeros of Partial Y Deriv')
#plt.plot(x2, y2_zero2, color='orange')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Plotting Zero Contours in X/Y Plane")

plt.subplot(1,3,2)
plt.plot(x2, (3*x2**2 - 4*x2))
plt.title("Partial Derivative of X")
plt.xlabel("X")
plt.ylabel("f_x(x)")


plt.subplot(1,3,3)
plt.title("Partial Derivative of Y")
plt.plot(x2, (3*x2**2 + 6*x2))
plt.xlabel("Y")
plt.ylabel("f_y(y)")
plt.show()
"""
