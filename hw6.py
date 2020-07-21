import numpy as np
import random
import os
from matplotlib import pyplot as plt

#points = np.array([[0,0], [1,1], [1,0], [0,1], [0.5,1.5], [0.5,0.5]])

points = np.array([[0,0]])
for i in range(49):
    newPoint = np.array([[-1+2*random.random(), -1+2*random.random()]])
    points = np.vstack([points, newPoint])


leftDex = 4
for i in range(len(points[:,0])):
    if i == leftDex:
        print("heeple")
    elif points[i,0] < points[leftDex,0]:
        #if point is further left than current min
        leftDex = i
    elif points[i,0] == points[leftDex, 0]:
        if points[i,1] < points[leftDex,1]:
            #if point is as left and lower than current min
            leftDex = i

def getNextHullPoint(hullDexes, points, firstPoint=False):
    maxAngle = -1 #set starting point lower than any real point would be
    nextDex = -1

    print(hullDexes)

    for i in range(len(points)):
        if i == hullDexes[-1]:
            print("skippy doo")
        else:
            if firstPoint:
                newAngle = getAngle([points[hullDexes[0],0], points[hullDexes[0],1] - 1], points[hullDexes[0]], points[i])
            else:
                newAngle = getAngle(points[hullDexes[-2]], points[hullDexes[-1]], points[i])

            if newAngle > maxAngle:
                maxAngle = newAngle
                nextDex = i

    return nextDex

def getAngle(point1, point2, point3):
    mag1 = np.sqrt( (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 )
    mag2 = np.sqrt( (point3[0] - point2[0])**2 + (point3[1] - point2[1])**2 )

    dotProd = np.dot((point1 - point2), (point3 - point2))

    theta = np.arccos(dotProd / (mag1*mag2))
    return theta


currDex = leftDex
hullPoints = np.array([ leftDex ])

currDex = getNextHullPoint(hullPoints, points, firstPoint=True)
hullPoints = np.append(hullPoints, [currDex])

while (currDex != leftDex):
    currDex = getNextHullPoint(hullPoints, points)
    hullPoints = np.append(hullPoints, [currDex])

x_vals = [ ]
y_vals = [ ]

for i in hullPoints:
    x_vals.append(points[i,0])
    y_vals.append(points[i,1])


plt.title("Convex Hull Demonstration")
plt.scatter(points[:,0], points[:,1], label='Points')
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(points[leftDex,0], points[leftDex,1], color='purple', label='Left Point')
plt.legend(loc='upper right')

plt.plot(x_vals, y_vals, color='green', linestyle='--', label='Convex Hull')
plt.show()
