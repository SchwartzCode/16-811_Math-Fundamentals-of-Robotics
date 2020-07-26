import numpy as np
import random
import os
from matplotlib import pyplot as plt

#points = np.array([[0,0], [1,1], [1,0], [0,1], [0.5,1.5], [0.5,0.5]])

class blob(object):
    points = np.array([1])
    leftDex = np.array([-1,-1])
    hullPts = np.array([])

    def __init__(self, x_cent, y_cent, width, point_count=20):

        self.points = np.array([[x_cent+width*random.random(), y_cent+width*random.random()]])
        for i in range(19):
            newPoint = np.array([[x_cent+width*random.random(), y_cent+width*random.random()]])
            self.points = np.vstack([self.points, newPoint])

        leftDex, hullx, hully = self.getConvexPoly(self.points)
        self.hullPts = np.transpose( np.vstack([hullx, hully]) )



    def getAngle(self, point1, point2, point3):
        mag1 = np.sqrt( (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 )
        mag2 = np.sqrt( (point3[0] - point2[0])**2 + (point3[1] - point2[1])**2 )

        dotProd = np.dot((point1 - point2), (point3 - point2))

        theta = np.arccos(dotProd / (mag1*mag2))
        return theta

    def getNextHullPoint(self, hullDexes, points, firstPoint=False):
        maxAngle = -1 #set starting point lower than any real point would be
        nextDex = -1

        for i in range(len(points)):
            if i != hullDexes[-1]:
                if firstPoint:
                    newAngle = self.getAngle([points[hullDexes[0],0], points[hullDexes[0],1] - 1], points[hullDexes[0]], points[i])
                else:
                    newAngle = self.getAngle(points[hullDexes[-2]], points[hullDexes[-1]], points[i])

                if newAngle > maxAngle:
                    maxAngle = newAngle
                    nextDex = i

        return nextDex

    def getConvexPoly(self, points):
        leftDex = 4
        for i in range(len(points[:,0])):
            if i != leftDex:

                if points[i,0] < points[leftDex,0]:
                    #if point is further left than current min
                    leftDex = i
                elif points[i,0] == points[leftDex, 0]:
                    if points[i,1] < points[leftDex,1]:
                        #if point is as left and lower than current min
                        leftDex = i

        currDex = leftDex
        hullPoints = np.array([ leftDex ])

        currDex = self.getNextHullPoint(hullPoints, points, firstPoint=True)
        hullPoints = np.append(hullPoints, [currDex])

        while (currDex != leftDex):
            currDex = self.getNextHullPoint(hullPoints, points)
            hullPoints = np.append(hullPoints, [currDex])

        x_vals = [ ]
        y_vals = [ ]

        for i in hullPoints:
            x_vals.append(points[i,0])
            y_vals.append(points[i,1])

        return leftDex, x_vals, y_vals




points1 = np.array([[2+2*random.random(), 1+2*random.random()]])
points2 = np.array([[1+2*random.random(), 4+2*random.random()]])
points3 = np.array([[4+2*random.random(), 4+2*random.random()]])
for i in range(19):
    newPoint1 = np.array([[2+2*random.random(), 1+2*random.random()]])
    points1 = np.vstack([points1, newPoint1])
    newPoint2 = np.array([[1+2*random.random(), 4+2*random.random()]])
    points2 = np.vstack([points2, newPoint2])
    newPoint3 = np.array([[4+2*random.random(), 4+2*random.random()]])
    points3 = np.vstack([points3, newPoint3])




def genCoM():
    print("aa")



startPt = np.array([0,0])
finishPt = np.array([8,8])

pathPts = np.array([startPt, finishPt])
obstructed = False

"""
if (pathPts): #check for obstruction
    obstructed = True
    iter = 0

while(obstructed and iter < 1e3):
    print("a")
    # add point to fix current obstruction
    # check again to see if obstructed, set obstructed variable accoringly
"""

obj1 = blob(2, 1, 2)
obj2 = blob(5 ,2, 1.5)
obj3 = blob(2, 5, 2)
obj4 = blob(6, 5, 2)


plt.title("Convex Hull Demonstration")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.scatter(obj1.points[:,0], obj1.points[:,1], color='blue', label='Points')
plt.scatter(obj2.points[:,0], obj2.points[:,1], color='blue', label='Points')
plt.scatter(obj3.points[:,0], obj3.points[:,1], color='blue', label='Points')
plt.scatter(obj4.points[:,0], obj4.points[:,1], color='blue', label='Points')

plt.plot(obj1.hullPts[:,0], obj1.hullPts[:,1], color='green', linestyle='--', label='Convex Hull')
plt.plot(obj2.hullPts[:,0], obj2.hullPts[:,1], color='green', linestyle='--', label='Convex Hull')
plt.plot(obj3.hullPts[:,0], obj3.hullPts[:,1], color='green', linestyle='--', label='Convex Hull')
plt.plot(obj4.hullPts[:,0], obj4.hullPts[:,1], color='green', linestyle='--', label='Convex Hull')

plt.scatter(startPt[0],startPt[1], color='red', label='Start')
plt.scatter(finishPt[0], finishPt[1], color='orange', label='Finish')

plt.plot(pathPts[:,0], pathPts[:,1], label='Path')

plt.xlim(-1,12)
plt.ylim(-1,10)
plt.legend(loc='lower right')
plt.show()
