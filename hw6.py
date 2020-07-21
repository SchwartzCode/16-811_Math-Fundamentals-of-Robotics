import numpy as np
import random
import os
from matplotlib import pyplot as plt

#points = np.array([[0,0], [1,1], [1,0], [0,1], [0.5,1.5], [0.5,0.5]])

points = np.array([[0,0]])
for i in range(99):
    newPoint = np.array([[-2+4*random.random(), -2+4*random.random()]])
    points = np.vstack([points, newPoint])

#points = points.T
print(points.shape)

minDex = 4
for i in range(len(points[:,0])):
    if i == minDex:
        print("heeple")
    elif points[i,0] < points[minDex,0]:
        #if point is further left than current min
        minDex = i
    elif points[i,0] == points[minDex, 0]:
        if points[i,1] < points[minDex,1]:
            #if point is as left and lower than current min
            minDex = i

def getNextHullPoint(currDex, Points):


    return nextDex

currDex = minDex
hullPoints = np.array([points[minDex]])


while (currDex != minDex):
    currDex = getNextHullPoint(currDex, points)
    hullPoints = np.vstack([hullPoints, points[currDex]])



plt.title("Convex Hull Demonstration")
plt.scatter(points[:,0], points[:,1], label='Points')
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(points[minDex,0], points[minDex,1], color='purple', label='Left Point')
plt.legend(loc='upper right')
plt.show()
