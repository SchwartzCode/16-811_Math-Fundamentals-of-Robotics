import numpy as np
import math as m
from cmath import sqrt
import matplotlib.pyplot as plt

def log_func(x):
    result = (np.log(x) / np.log(6)) ** 1.5
    return result

def divided_difference_coeff(vals):
    #print("vals:", vals)
    if len(vals) % 2 != 0:
        print("Error! input must be an even length!")
        print(vals)
        return
    else:
        if len(vals) == 2:
            return vals[1]

        elif len(vals) == 4:
            returnVal = (vals[3] - vals[2]) / (vals[1] - vals[0])
            #print("Output:", returnVal)
            return returnVal
        else:
            middle = len(vals) // 2
            x_vals = vals[:middle]
            f_vals = vals[middle:]

            vals_left = np.concatenate([x_vals[:-1], f_vals[:-1]])
            vals_right = np.concatenate([x_vals[1:], f_vals[1:]])

            output = (divided_difference_coeff(vals_right) - divided_difference_coeff(vals_left)) / (x_vals[-1] - x_vals[0])
            #print("output:", output)
            return output

def interpolation(vals, x_interp):
    middle = len(vals) // 2
    x_vals = vals[:middle]
    f_vals = vals[middle:]

    guesstimate = 0

    for i in range(len(x_vals)-1,0,-1):
        guesstimate *= (x_interp - x_vals[i-1])
        guesstimate += divided_difference_coeff(np.concatenate([x_vals[:i], f_vals[:i]]))

    return guesstimate

a = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

b = log_func(a)

arrTest = [1, 2, 3, 1, 2, 3]
arr = np.concatenate([a,b])
#print(arrTest)

coeffs = [ ]
#out = divided_difference_coeff(arrTest)
#print(out)
out = interpolation(arr, 2.25)
print("\tPROBLEM 1a\n========================")
print("x values followed by f(x) values:\n", arr)
print("Using this data to interpolate f(2.25):\t", out)

print("\n\tPROBLEM 1b\n========================")

def genVals_1b(n):
    x_vals = np.array([])
    y_vals = np.array([])
    for i in range(n):
        x_vals = np.concatenate([x_vals, np.array([-1 + 2*i/n]) ])
        y_vals = np.concatenate([y_vals, np.array([ 6 / (1 + 25*x_vals[i]**2)]) ])
    return x_vals, y_vals

xB, yB = genVals_1b(2)
arr_1b1 = np.concatenate([xB, yB])
est_1b1 = interpolation(arr_1b1, 0.05)
print("2 data points estimate:\t", est_1b1)

xB2, yB2 = genVals_1b(4)
arr_1b2 = np.concatenate([xB2, yB2])
est_1b2 = interpolation(arr_1b2, 0.05)
print("4 data points estimate:\t", est_1b2)

xB3, yB3 = genVals_1b(7)
arr_1b3 = np.concatenate([xB3, yB3])
est_1b3 = interpolation(arr_1b3, 0.05)
print("7 data points estimate:\t", est_1b3)

xB3, yB3 = genVals_1b(10)
arr_1b3 = np.concatenate([xB3, yB3])
est_1b3 = interpolation(arr_1b3, 0.05)
print("10 data points estimate:\t", est_1b3)

xB3, yB3 = genVals_1b(20)
arr_1b3 = np.concatenate([xB3, yB3])
#est_1b3 = interpolation(arr_1b3, 0.05)
#print("20 data points estimate:\t", est_1b3)


print("\n\tPROBLEM 1c\n========================")


def getErr(n):
    xC, yC = genVals_1b(n)
    arrC = np.concatenate([xC, yC])
    i_vals = np.linspace(-1,0.5, num=21)

    #print("\nArrC:\n", arrC)

    err_max = 0

    for i in i_vals:
        estC = interpolation(arrC, i)
        actualVal = 6 / (1 + 25*i**2)

        #print("i:", i, "\t\tEst:\t", estC)
        #print("Actual val:\t", actualVal)

        if abs(estC - actualVal) > err_max:
            err_max = abs(estC - actualVal)

    return err_max

n_vals = [2, 4, 6, 8, 10] # 12, 14, 16, 18, 20]

print("Max error in range -1 to 1 for n=_:")


for n in n_vals:
    print("n =", n, ":\t", getErr(n))

print("\nSo it looks like the interpolation improves with more points to a point (around 10)")
print("and then starts becoming less accurate with more points. It seems this happens because")
print("the algorithm starts 'over-fitting' the curve, and you get some crazy, high-order")
print("stuff going on at the edges of the range you're interpolating. See here for info on this phenomenon:")
print("https://www.johndcook.com/blog/2009/04/01/polynomial-interpolation-errors/\n\n")

print("\n\tPROBLEM 2\n========================")
i2s = np.arange(0,2*np.pi,2e-5)
sin_vals = np.sin(i2s)

diff_max = 0
for i in range(1,len(sin_vals)):
    diff = abs(sin_vals[i] - sin_vals[i-1])
    if diff > diff_max:
        diff_max = diff

print("Maximum Difference:\t", diff_max)
print("I believe this is adequate because the answer will always be accurate within 1e-6")
print("or less, which means it should be accurate to 6 decimal places.")

print("\n\n\tPROBLEM 3\n========================")
def newton_method_tan(x_val):
    val = x_val - np.tan(x_val)
    deriv = 1 - (1 / np.cos(x_val))**2

    #print("X:\t", x_val, "f(x):\t", val, "f'(x):\t", deriv)

    next_x_val = x_val - (val / deriv)
    return next_x_val


x_vals = np.arange(7,15,0.1)
for x_val in x_vals:
    counter = 0
    start = x_val
    while counter < 10:
        x_val = newton_method_tan(x_val)
        counter += 1

    if start == x_vals[0]:
        solutions = [x_val]
    else:
        solutions.append(x_val)

print("Left x boundary for tan(x) root near 11:\t", solutions[10])
print("Right x boundary for tan(x) root near 11:\t", solutions[15])

#aaa = np.array([1,2,3,4,5,6])
#b = np.concatenate([aaa[:2], aaa[-2:]])
#print(b)



print("\n\n\tPROBLEM 5\n========================")
def MullerMethod(guesses, f, iter = 5000):
    counter = 0
    res = 0.0
    while counter < iter:
        #print("guesses: ", guesses)
        guess_sols = [ f(guesses[0]) ]
        for g in guesses[1:]:
            guess_sols.append( f(g) )

        f1 = f(guesses[0]); f2 = f(guesses[1]); f3 = f(guesses[2]);
        d1 = f1 - f3;
        d2 = f2 - f3;
        h1 = guesses[0] - guesses[2];
        h2 = guesses[1] - guesses[2];
        a0 = f3;
        a1 = (((d2 * pow(h1, 2)) -
               (d1 * pow(h2, 2))) /
              ((h1 * h2) * (h1 - h2)));
        a2 = (((d1 * h2) - (d2 * h1)) /
              ((h1 * h2) * (h1 - h2)));

        x1 = ((-2 * a0) / (a1 + abs(sqrt(a1 * a1 - 4 * a0 * a2))));
        x2 = ((-2 * a0) / (a1 -  abs(sqrt(a1 * a1 - 4 * a0 * a2))));


        if (x1 >= x2):
            res = x1 + guesses[2]
        else:
            res = x2 + guesses[2]

        if abs(res - guesses[2]) < 1e-4:
            res = round(res, 6)
            print("Root found at x =", res)
            return res
        else:
            guesses[0] = guesses[1]
            guesses[1] = guesses[2]
            guesses[2] = res
            counter+=1
            #print("resid: ", res)

    print("Muller method didn't work; exceeded max iterations :( ")
    return guesses[2]

def f_q5(x):
    return x**3 - 4*x**2 + 6*x - 4

def f_q5_only_complex(x):
    #roots: x = 1 +/- j
    return x**2 - 2*x + 2

def f_q5_no_complex(x):
    # roots: x = [1, 2.5, 10]
    # very easy to converge to 2.5 and 10, 1 is very difficult to get
    return x**3 - 13.5*x**2 + 37.5*x - 25

#sol_q5 = MullerMethod(np.array([0.7, 0.8, 0.9]), f_q5_no_complex)
# had trouble getting the algorithm to recognize complex roots, kinda works
# though so I'm going to leave it for now
sol_q5 = MullerMethod(np.array([0.5+0.5j, 0.75+0.75j, 0.99+0.99j]), f_q5)
print(sol_q5)

print("\n\n\tPROBLEM 6\n========================")
A = np.array([ [1, -9, 26, -24],
               [0, 1, -9, 26],
               [0, 1, 3, -10],
               [0, 0, 1, 3] ])
aDet = np.linalg.det(A)
print(aDet)
print("These equations share a root: ", aDet == 0)

print("\n\n\tPROBLEM 7\n========================")

def p(x):
    print(x)
    print(0.5 - x**2)
    val = abs(np.sqrt(0.5 - x**2))
    print(val)
    return [val, -val]

def q(x, y):
    return x**2 + y**2 + 2*x*y - x + y

x_vals7A = np.arange(-np.sqrt(2),np.sqrt(2),1e-4)
#print(x_vals)
y_vals7A = p(x_vals7A)

x_vals7B = [0.5, 0, -0.125, 0, 0.375, 1, 2.118]
y_vals7B = [-2.118, -1, -0.375, 0, 0.125, 0, -0.5]

print(len(y_vals7A), len(y_vals7A[0]))
print(y_vals7A[1])
#plt.plot(x_vals7A, y_vals7A[0], color='blue', label='1 = x^2 + y^2')
#plt.plot(x_vals7A, y_vals7A[1], color='blue')
plt.plot(x_vals7B, y_vals7B, color='red')
plt.legend(loc='upper right')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Problem 7: Finding shared zeros of two bivariate polynomials")
plt.show()

print("\n\n")
