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
out = interpolation(arrTest, 1.5)
print(out)
print(arr)
