import numpy as np
import random
import os
from matplotlib import pyplot as plt

#points = np.array([[0,0], [1,1], [1,0], [0,1], [0.5,1.5], [0.5,0.5]])

points1 = np.array([[2+2*random.random(), 1+2*random.random()]])
for i in range(19):
    newPoint = np.array([[2+2*random.random(), 1+2*random.random()]])
    points1 = np.vstack([points1, newPoint])

points2 = np.array([[1+2*random.random(), 4+2*random.random()]])
for i in range(19):
    newPoint = np.array([[1+2*random.random(), 4+2*random.random()]])
    points2 = np.vstack([points2, newPoint])

points3 = np.array([[4+2*random.random(), 4+2*random.random()]])
for i in range(19):
    newPoint = np.array([[4+2*random.random(), 4+2*random.random()]])
    points3 = np.vstack([points3, newPoint])

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

def getConvexPoly(points):
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

    return leftDex, x_vals, y_vals

leftDex1, x1, y1 = getConvexPoly(points1)
leftDex2, x2, y2 = getConvexPoly(points2)
leftDex3, x3, y3 = getConvexPoly(points3)

plt.title("Convex Hull Demonstration")
plt.scatter(points1[:,0], points1[:,1], color='blue', label='Points')
plt.scatter(points2[:,0], points2[:,1], color='blue')
plt.scatter(points3[:,0], points3[:,1], color='blue')
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(points1[leftDex1,0], points1[leftDex1,1], color='purple', label='Left Point')
plt.scatter(points2[leftDex2,0], points2[leftDex2,1], color='purple')
plt.scatter(points3[leftDex3,0], points3[leftDex3,1], color='purple')

plt.plot(x1, y1, color='green', linestyle='--', label='Convex Hull')
plt.plot(x2, y2, color='green', linestyle='--')
plt.plot(x3, y3, color='green', linestyle='--')

plt.xlim(0,8)
plt.ylim(0,8)
plt.legend(loc='upper right')
plt.show()
