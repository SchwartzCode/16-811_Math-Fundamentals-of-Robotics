import numpy as np


def log_func(x):
    result = (np.log(x) / np.log(6)) ** 1.5
    return result

def divided_difference_coeff(vals):
    if len(vals) % 2 != 0:
        print("Error! input must be an even length!")
        print(vals)
        return
    else:
        if len(vals) == 2:
            return vals[1]

        elif len(vals) == 4:
            returnVal = (vals[3] - vals[2]) / (vals[1] - vals[0])
            return returnVal
        else:
            middle = len(vals) // 2
            x_vals = vals[:middle]
            f_vals = vals[middle:]

            vals_left = np.concatenate([x_vals[:-1], f_vals[:-1]])
            vals_right = np.concatenate([x_vals[1:], f_vals[1:]])

            #print("left:\n", vals_left)
            #print("right:\n", vals_right)


            return (divided_difference_coeff(vals_right) - divided_difference_coeff(vals_left)) / (x_vals[-1] - x_vals[0])

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
    val = np.tan(x_val)
    deriv = 1 / np.cos(x_val)**2

    next_x_val = x_val + (val / deriv)

    return next_x_val


x_low = 10.5
counter = 0
while counter < 10:
    x_low = newton_method_tan(x_low)
    counter += 1

print("Left x boundary for tan(x) when approaching 11:\t\t", x_low)

x_high = 11.5
counter = 0
while counter < 10:
    x_high = newton_method_tan(x_high)
    counter += 1

print("Right x boundary for tan(x) when approaching 11:\t", x_high)




print("\n\n\tPROBLEM 4\n========================")
print("skipping this for now, will return using pages 20-24 of nonlinear equation roots notes")


print("\n\n\tPROBLEM 5\n========================")


print("\n\n")
